from __future__ import annotations

import asyncio
import io
import textwrap
import xml.sax
from dataclasses import dataclass, field
from pathlib import Path

from .errors import DocumentError, SourceLocation


@dataclass(slots=True)
class DocumentNode:
    location: SourceLocation
    end_location: SourceLocation | None = None
    parent: "ElementNode | None" = None

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
    text: str = ""

    @property
    def display_name(self) -> str:
        return "text"


@dataclass(slots=True)
class ElementNode(DocumentNode):
    tag_name: str = ""
    children: list[DocumentNode] = field(default_factory=list)
    allowed_child_tags: dict[str, type["ElementNode"]] = field(default_factory=dict, init=False)
    allow_text: bool = field(default=False, init=False)

    @property
    def display_name(self) -> str:
        return f"<{self.tag_name}>"

    def add_child(self, child: DocumentNode) -> None:
        child.parent = self
        self.children.append(child)

    def create_child_element(self, tag_name: str, location: SourceLocation) -> "ElementNode":
        child_type = self.allowed_child_tags.get(tag_name)
        if child_type is None:
            raise DocumentError(
                f"{self.display_name} does not allow child element <{tag_name}>",
                node=self,
                location=location,
        )
        return child_type(location=location, tag_name=tag_name)

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


@dataclass(slots=True)
class SpeakerMapNode(ElementNode):
    allow_text: bool = field(default=True, init=False)

    async def plan(self, ainjector):
        from .planning import SpeakerMapPlan

        return await ainjector(SpeakerMapPlan, node=self)


@dataclass(slots=True)
class ScriptNode(ElementNode):
    allow_text: bool = field(default=True, init=False)

    async def plan(self, ainjector):
        from .planning import ScriptPlan

        return await ainjector(ScriptPlan, node=self)


@dataclass(slots=True)
class ProductionNode(ElementNode):
    allowed_child_tags: dict[str, type[ElementNode]] = field(
        default_factory=lambda: {
            "speaker-map": SpeakerMapNode,
            "script": ScriptNode,
        },
        init=False,
    )

    def validate_phase1(self) -> None:
        self.require_one_child("speaker-map")
        self.require_children("script")

    @property
    def speaker_map_node(self) -> SpeakerMapNode:
        return self.require_one_child("speaker-map")

    @property
    def script_nodes(self) -> list[ScriptNode]:
        return self.require_children("script")

    async def plan(self, ainjector):
        from carthage.dependency_injection import InjectionKey, Injector

        from .planning import ProductionPlan, SpeakerMapPlan

        production_injector = Injector(parent_injector=ainjector.injector)
        production_injector.replace_provider(
            InjectionKey(asyncio.AbstractEventLoop),
            asyncio.get_running_loop(),
            close=False,
        )
        production_ainjector = production_injector(type(ainjector))
        speaker_map_plan = await self.speaker_map_node.plan(production_ainjector)
        production_injector.add_provider(
            InjectionKey(SpeakerMapPlan),
            speaker_map_plan,
            close=False,
        )
        script_plans = [await script_node.plan(production_ainjector) for script_node in self.script_nodes]
        return await production_ainjector(
            ProductionPlan,
            node=self,
            script_plans=script_plans,
        )


class _ProductionContentHandler(xml.sax.handler.ContentHandler):
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
        if not self.stack:
            if name != "production":
                raise DocumentError(
                    "The document root must be <production>",
                    location=location,
                )
            root = ProductionNode(location=location, tag_name=name)
            self.root = root
            self.stack.append(root)
            return

        parent = self.stack[-1]
        child = parent.create_child_element(name, location)
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


def parse_production_string(xml_text: str, *, source_name: str | None = None) -> ProductionNode:
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
    handler.root.validate_phase1()
    return handler.root


def parse_production_file(path: str | Path) -> ProductionNode:
    xml_path = Path(path)
    with xml_path.open("r", encoding="utf-8") as handle:
        return parse_production_string(handle.read(), source_name=str(xml_path))
