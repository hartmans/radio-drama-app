from __future__ import annotations

import asyncio
import io
import textwrap
import xml.sax
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from .errors import DocumentError, SourceLocation


@dataclass(slots=True)
class DocumentNode:
    """Semantic node produced from the XML input document.

    Document nodes are plain objects rather than injectables. They preserve
    enough source context to produce user-facing document errors and to plan
    into injectable planning objects later.
    """

    location: SourceLocation
    end_location: SourceLocation | None = None
    parent: "ElementNode | None" = None
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.__class__.__name__

    def error(
        self,
        message: str,
        *,
        location: SourceLocation | None = None,
    ) -> DocumentError:
        return DocumentError(message, node=self, location=location)


@dataclass(slots=True)
class TextNode(DocumentNode):
    """Literal text content inside an XML element."""
    text: str = ""

    @property
    def display_name(self) -> str:
        return "text"


@dataclass(slots=True)
class ElementNode(DocumentNode):
    """Base class for semantic XML elements with ordered child nodes."""

    tag_name: ClassVar[str] = ""
    allowed_child_tags: ClassVar[dict[str, type["ElementNode"]]] = {}
    permitted_in_contexts: ClassVar[tuple["ElementContext", ...]] = ()
    accepts_contexts: ClassVar[tuple["ElementContext", ...]] = ()
    allow_text: ClassVar[bool] = False

    children: list[DocumentNode] = field(default_factory=list)

    def __init_subclass__(cls, **kwargs) -> None:
        cls.allowed_child_tags = dict(getattr(cls, "allowed_child_tags", {}))
        cls.permitted_in_contexts = tuple(getattr(cls, "permitted_in_contexts", ()))
        cls.accepts_contexts = tuple(getattr(cls, "accepts_contexts", ()))

        for context in cls.permitted_in_contexts:
            _register_context_class(context.is_this_context, cls)
        for context in cls.permitted_in_contexts:
            for accepting_cls in tuple(context.accepts_this_context):
                _register_allowed_child(accepting_cls, cls)

        for context in cls.accepts_contexts:
            _register_context_class(context.accepts_this_context, cls)
        for context in cls.accepts_contexts:
            for child_cls in tuple(context.is_this_context):
                _register_allowed_child(cls, child_cls)

    @property
    def display_name(self) -> str:
        return f"<{self.tag_name}>"

    def add_child(self, child: DocumentNode) -> None:
        child.parent = self
        self.children.append(child)

    def create_child_element(
        self,
        tag_name: str,
        location: SourceLocation,
        attributes: dict[str, str] | None = None,
    ) -> "ElementNode":
        """Construct a permitted child element or raise a document error."""
        child_type = self.allowed_child_tags.get(tag_name)
        if child_type is None:
            raise DocumentError(
                f"{self.display_name} does not allow child element <{tag_name}>",
                node=self,
                location=location,
        )
        return child_type(
            location=location,
            attributes=dict(attributes or {}),
        )

    def child_elements_named(self, tag_name: str) -> list["ElementNode"]:
        return [child for child in self.element_children if child.tag_name == tag_name]

    def require_one_child(self, tag_name: str) -> "ElementNode":
        children = self.child_elements_named(tag_name)
        if not children:
            raise self.error(f"{self.display_name} requires one <{tag_name}> element")
        if len(children) > 1:
            raise children[1].error(
                f"{self.display_name} may contain only one <{tag_name}> element"
            )
        return children[0]

    def require_children(self, tag_name: str) -> list["ElementNode"]:
        children = self.child_elements_named(tag_name)
        if not children:
            raise self.error(f"{self.display_name} requires at least one <{tag_name}> element")
        return children

    @property
    def element_children(self) -> list["ElementNode"]:
        return [child for child in self.children if isinstance(child, ElementNode)]

    @property
    def text_content(self) -> str:
        return "".join(
            child.text for child in self.children if isinstance(child, TextNode)
        )

    @property
    def normalized_text_content(self) -> str:
        return textwrap.dedent(self.text_content).strip()

    def validate_document(self) -> None:
        for child in self.element_children:
            child.validate_document()


@dataclass(slots=True)
class AttributeOrTextValueNode(ElementNode):
    """Element whose semantic value may be authored as text or one attribute."""

    value_attribute_name: ClassVar[str] = "value"

    @property
    def value_from_attribute_or_text(self) -> str:
        attribute_value = self.attributes.get(self.value_attribute_name)
        normalized_attribute_value = (
            attribute_value.strip() if attribute_value is not None else ""
        )
        normalized_text_value = self.normalized_text_content

        if attribute_value is not None and not normalized_attribute_value:
            raise self.error(
                f"{self.display_name} {self.value_attribute_name} attribute cannot be empty"
            )
        if (
            attribute_value is not None
            and normalized_text_value
            and normalized_text_value != normalized_attribute_value
        ):
            raise self.error(
                f"{self.display_name} text content must match the "
                f"{self.value_attribute_name} attribute when both are present"
            )
        if normalized_attribute_value:
            return normalized_attribute_value
        if normalized_text_value:
            return normalized_text_value
        raise self.error(
            f"{self.display_name} requires either a {self.value_attribute_name} "
            "attribute or text content"
        )

    def validate_document(self) -> None:
        self.value_from_attribute_or_text
        return ElementNode.validate_document(self)


@dataclass(slots=True)
class ElementContext:
    """Registry describing which element classes accept or inhabit a context."""

    accepts_this_context: list[type[ElementNode]] = field(default_factory=list)
    is_this_context: list[type[ElementNode]] = field(default_factory=list)


def _register_allowed_child(
    parent_cls: type[ElementNode],
    child_cls: type[ElementNode],
) -> None:
    if not child_cls.tag_name:
        return
    parent_cls.allowed_child_tags[child_cls.tag_name] = child_cls


def _register_context_class(
    classes: list[type[ElementNode]],
    cls: type[ElementNode],
) -> None:
    for index, existing in enumerate(classes):
        if existing is cls:
            return
        if _same_declared_class(existing, cls):
            classes[index] = cls
            return
    classes.append(cls)


def _same_declared_class(left: type[ElementNode], right: type[ElementNode]) -> bool:
    return (
        left.__module__ == right.__module__
        and left.__qualname__ == right.__qualname__
    )


AudioPlanContext = ElementContext()


@dataclass(slots=True)
class SpeakerMapNode(ElementNode):
    """Document node for the YAML speaker-to-voice mapping."""

    tag_name: ClassVar[str] = "speaker-map"
    allow_text: ClassVar[bool] = True

    async def plan(self, ainjector):
        from .planning import SpeakerMapPlan

        return await ainjector(SpeakerMapPlan, node=self)


@dataclass(slots=True)
class ScriptNode(ElementNode):
    """Document node for one ordered renderable script unit."""

    tag_name: ClassVar[str] = "script"
    allow_text: ClassVar[bool] = True
    permitted_in_contexts: ClassVar[tuple[ElementContext, ...]] = (AudioPlanContext,)
    accepts_contexts: ClassVar[tuple[ElementContext, ...]] = (AudioPlanContext,)

    @property
    def preset(self) -> str | None:
        preset_name = self.attributes.get("preset")
        if preset_name is None:
            return None
        normalized = preset_name.strip()
        return normalized or None

    async def plan(self, ainjector):
        from .planning import ScriptPlan

        return await ScriptPlan.from_node(ainjector, self)


@dataclass(slots=True)
class MarkNode(AttributeOrTextValueNode):
    """Document node for a zero-length named audio mark."""

    tag_name: ClassVar[str] = "mark"
    allow_text: ClassVar[bool] = True
    value_attribute_name: ClassVar[str] = "id"
    permitted_in_contexts: ClassVar[tuple[ElementContext, ...]] = (AudioPlanContext,)

    @property
    def id(self) -> str:
        return self.value_from_attribute_or_text

    async def plan(self, ainjector):
        from .planning import MarkPlan

        return await ainjector(MarkPlan, node=self, id=self.id)


@dataclass(slots=True)
class ProductionNode(ElementNode):
    """Root document node for one production."""

    tag_name: ClassVar[str] = "production"
    allowed_child_tags: ClassVar[dict[str, type[ElementNode]]] = {"speaker-map": SpeakerMapNode}
    accepts_contexts: ClassVar[tuple[ElementContext, ...]] = (AudioPlanContext,)

    def validate_document(self) -> None:
        return ElementNode.validate_document(self)

    @property
    def speaker_map_node(self) -> SpeakerMapNode:
        return self.require_one_child("speaker-map")

    @property
    def script_nodes(self) -> list[ScriptNode]:
        return self.child_elements_named("script")

    async def plan(self, ainjector):
        """Plan the production in a child injector with shared production resources."""
        from .init import radio_drama_injector
        from .planning import PRODUCTION_PLANNING_INJECTOR_KEY, AudioPlan, ProductionPlan
        from .effects import PresetPlan
        from .sound import ProductionDocumentPath

        production_injector = radio_drama_injector(
            ainjector.injector,
            event_loop=asyncio.get_running_loop(),
        )
        if (
            production_injector.injector_containing(ProductionDocumentPath) is None
            and self.location.source is not None
        ):
            production_injector.add_provider(ProductionDocumentPath(Path(self.location.source)))
        production_injector.add_provider(PRODUCTION_PLANNING_INJECTOR_KEY, production_injector)
        production_ainjector = production_injector(type(ainjector))
        planned_children = [await child.plan(production_ainjector) for child in self.element_children]
        audio_plans = [plan for plan in planned_children if isinstance(plan, AudioPlan)]
        production_plan = await production_ainjector(
            ProductionPlan,
            node=self,
            audio_plans=audio_plans,
        )
        return await production_ainjector(
            PresetPlan,
            node=self,
            audio_plan=production_plan,
            preset_name="master",
        )


class _ProductionContentHandler(xml.sax.handler.ContentHandler):
    """SAX handler that builds the semantic document tree with source locations."""

    def __init__(self, source_name: str | None) -> None:
        super().__init__()
        self.source_name = source_name
        self.locator = None
        self.stack: list[ElementNode] = []
        self.root: ProductionNode | None = None

    def setDocumentLocator(self, locator) -> None:  # noqa: N802
        self.locator = locator

    def startElement(self, name: str, attrs) -> None:  # noqa: N802
        location = self._current_location()
        attributes = {str(key): str(value) for key, value in attrs.items()}
        if not self.stack:
            if name != "production":
                raise DocumentError(
                    "The document root must be <production>",
                    location=location,
                )
            root = ProductionNode(location=location, attributes=attributes)
            self.root = root
            self.stack.append(root)
            return

        parent = self.stack[-1]
        child = parent.create_child_element(name, location, attributes)
        parent.add_child(child)
        self.stack.append(child)

    def endElement(self, name: str) -> None:  # noqa: N802
        node = self.stack.pop()
        node.end_location = self._current_location()
        if node.tag_name != name:
            raise DocumentError(
                f"Malformed XML close tag </{name}> for {node.display_name}",
                node=node,
            )

    def characters(self, content: str) -> None:
        if not content or not self.stack:
            return
        parent = self.stack[-1]
        if not parent.allow_text and not content.isspace():
            raise DocumentError(
                f"{parent.display_name} does not allow text content",
                node=parent,
                location=self._current_location(),
            )
        if parent.allow_text:
            parent.add_child(TextNode(location=self._current_location(), text=content))

    def _current_location(self) -> SourceLocation:
        if self.locator is None:
            return SourceLocation(self.source_name, None, None)
        line = self.locator.getLineNumber()
        column = self.locator.getColumnNumber()
        return SourceLocation(
            self.source_name,
            line if line >= 0 else None,
            column + 1 if column >= 0 else None,
        )


from . import sound as _sound  # noqa: F401


def parse_production_string(xml_text: str, *, source_name: str | None = None) -> ProductionNode:
    """Parse XML text into a validated ``ProductionNode`` tree."""
    handler = _ProductionContentHandler(source_name=source_name)
    try:
        xml.sax.parse(io.StringIO(xml_text), handler)
    except xml.sax.SAXParseException as exc:
        raise DocumentError(
            exc.getMessage(),
            location=SourceLocation(
                source_name,
                exc.getLineNumber(),
                exc.getColumnNumber() + 1,
            ),
        ) from None
    if handler.root is None:
        raise DocumentError("The XML document did not produce a <production> root")
    handler.root.validate_document()
    return handler.root


def parse_production_file(path: str | Path) -> ProductionNode:
    """Read and parse a production XML file."""
    xml_path = Path(path)
    with xml_path.open("r", encoding="utf-8") as handle:
        return parse_production_string(handle.read(), source_name=str(xml_path))
