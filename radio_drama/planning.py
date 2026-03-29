from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping, Sequence, cast

import numpy as np
import yaml
from carthage.dependency_injection import (
    AsyncInjectable,
    ExistingProvider,
    InjectionKey,
    Injector,
    inject,
)

from .config import ProductionConfig
from .debug import write_debug_message
from .errors import DocumentError
from .expressions import coerce_array_exp, coerce_real, eval_expression, validate_expression
from .rendering import ProductionResult, RenderResult
from .audio import SUPPORTED_AUDIO_EXTENSIONS


if TYPE_CHECKING:
    from .document import (
        DocumentNode,
        IgnoreNode,
        MarkNode,
        ProductionNode,
        ScriptNode,
        SpeakerMapNode,
        TextNode,
    )


_SPEAKER_LINE_RE = re.compile(r"^([^:\n]+?)\s*:\s*(.*)$")
PRODUCTION_PLANNING_INJECTOR_KEY = InjectionKey("radio_drama.production_planning_injector")
AudioAttrValue = float | str
AudioAttrs = dict[str, AudioAttrValue]


@dataclass(frozen=True, slots=True)
class SpeakerVoiceReference:
    """Resolved voice reference for one canonical speaker name."""
    authored_name: str
    voice_name: str
    resolved_path: Path


@dataclass(slots=True)
class DialogueContents:
    """One ordered item inside a script, later addressable by aligned time."""

    start_pos: float = field(default=math.nan, kw_only=True)


@dataclass(slots=True)
class DialogueLine(DialogueContents):
    """Normalized dialogue stanza belonging to one resolved speaker."""
    speaker: SpeakerVoiceReference
    spoken_text: str
    handling: Literal["normal", "ignore", "special"] = "normal"
    node: "DocumentNode | None" = None


@dataclass(frozen=True, slots=True)
class ScriptRenderRequest:
    """Semantic render request sent to a speech resource."""
    dialogue_lines: list[DialogueLine]
    first_words: str = ""


@inject(injector=Injector)
class PlanningNode(AsyncInjectable):
    """Base class for injectable planning objects.

    Planning nodes keep the source ``DocumentNode`` that produced them and
    provide a memoized async ``render()`` entry point so downstream callers do
    not need to coordinate duplicate work themselves.
    """

    def __init__(self, node: DocumentNode | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.node = node
        self._render_task: asyncio.Task | None = None

    def document_error(self, message: str, *, node: DocumentNode | None = None) -> DocumentError:
        target = node or self.node
        if target is None:
            return DocumentError(message)
        return target.error(message)

    async def render(self):
        if self._render_task is None:
            self._render_task = asyncio.create_task(self.render_node())
        try:
            return await self._render_task
        except BaseException:
            self._render_task = None
            raise

    async def render_node(self):
        return None


class AudioPlan(PlanningNode):
    """Planning node whose render path produces audio."""

    def __init__(
        self,
        node: DocumentNode | None = None,
        *,
        attrs: Mapping[str, AudioAttrValue] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(node=node, **kwargs)
        self.audio_marks: list[str] = []
        self._audio_mark_counts: dict[str, int] = {}
        self._layout_task: asyncio.Task | None = None
        self.audio_marks_inner: dict[str, float] = {}
        self.audio_marks_render: dict[str, float] = {}
        self.inner_first = 0.0
        self.inner_last = 0.0
        self.start = 0.0
        self.end = 0.0
        self.length = 0.0
        self._raw_inner_first = 0.0
        self._raw_inner_last = 0.0
        self._raw_length = 0.0
        self._content_start = 0.0
        self._content_end = 0.0
        resolved_attrs = type(self).attrs_from_node(node) if attrs is None else dict(attrs)
        self._replace_attrs(resolved_attrs)

    async def async_resolve(self):
        preset_name = cast(str | None, self.attrs.get("preset"))
        if preset_name is None:
            return self
        wrapper_attrs = {key: value for key, value in self.attrs.items() if key != "preset"}
        await self.async_ready()
        self._replace_attrs({})
        from .effects import PresetPlan

        return await self.ainjector(
            PresetPlan,
            node=self.node,
            audio_plan=self,
            preset_name=preset_name,
            attrs=wrapper_attrs,
        )

    @property
    def natural_length(self) -> float:
        return self.inner_last - self.inner_first

    async def layout(self) -> None:
        if self._layout_task is None:
            self._layout_task = asyncio.create_task(self._layout_audio())
        try:
            await self._layout_task
        except BaseException:
            self._layout_task = None
            raise

    async def render(self, incoming_marks: Mapping[str, float] | None = None) -> RenderResult:
        if self._render_task is None:
            self._render_task = asyncio.create_task(self._render_audio(incoming_marks))
        try:
            return cast(RenderResult, await self._render_task)
        except BaseException:
            self._render_task = None
            raise

    async def layout_node(self) -> None:
        raise NotImplementedError

    async def render_node(self) -> RenderResult:
        raise NotImplementedError

    async def post_render(self, result: RenderResult) -> RenderResult:
        updated_result = self._apply_node_render_geometry(result)
        if updated_result.frame_count == 0:
            return updated_result
        if self.gain_expression is not None:
            gain_expression = eval_expression(
                self.gain_expression,
                self.render_time_variables(),
                coerce_array_exp,
            )
            gain_db = gain_expression.to_size(updated_result.frame_count)
            gain_multiplier = np.float32(10.0) ** (
                gain_db.astype(np.float32, copy=False) / np.float32(20.0)
            )
            if updated_result.audio.ndim == 1:
                updated_result.audio *= gain_multiplier
            else:
                updated_result.audio *= gain_multiplier[:, np.newaxis]
        if self.pan_expression is None or updated_result.audio.ndim != 2 or updated_result.audio.shape[1] < 2:
            return updated_result

        pan_expression = eval_expression(
            self.pan_expression,
            self.render_time_variables(),
            coerce_array_exp,
        )
        pan = np.clip(pan_expression.to_size(updated_result.frame_count), -1.0, 1.0)
        far_channel_gain = np.cos(np.abs(pan) * (np.pi / 2.0)).astype(np.float32, copy=False)
        far_channel_gain[np.abs(pan) >= 1.0] = 0.0
        left_gain = np.where(pan <= 0.0, 1.0, far_channel_gain).astype(np.float32, copy=False)
        right_gain = np.where(pan >= 0.0, 1.0, far_channel_gain).astype(np.float32, copy=False)

        updated_result.audio[:, 0] *= left_gain
        updated_result.audio[:, 1] *= right_gain
        return updated_result

    def leaf_audio_plans(self) -> list["AudioPlan"]:
        return [self]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def inner_plans(self, *audio_plans: "AudioPlan") -> None:
        for audio_plan in audio_plans:
            for audio_mark in audio_plan.audio_marks:
                mark_count = self._audio_mark_counts.get(audio_mark, 0) + 1
                self._audio_mark_counts[audio_mark] = mark_count
                if mark_count == 1:
                    self.audio_marks.append(audio_mark)
                elif mark_count == 2:
                    self.audio_marks.remove(audio_mark)

    def cut_before_mark(self, audio_mark: str) -> None:
        if audio_mark not in self.audio_marks:
            raise ValueError(f"Unknown or ambiguous audio mark {audio_mark!r}")

    def cut_after_mark(self, audio_mark: str) -> None:
        if audio_mark not in self.audio_marks:
            raise ValueError(f"Unknown or ambiguous audio mark {audio_mark!r}")

    @classmethod
    def attrs_from_node(cls, node: DocumentNode | None) -> AudioAttrs:
        if node is None:
            return {}
        attrs: AudioAttrs = {}
        start = cls._expression_attribute(node, "start")
        pre_gap = cls._timing_attribute_seconds(
            node,
            "pre_gap",
            allow_negative=True,
            allow_missing=True,
        )
        end = cls._expression_attribute(node, "end")
        post_gap = cls._timing_attribute_seconds(
            node,
            "post_gap",
            allow_negative=True,
            allow_missing=True,
        )
        length = cls._timing_attribute_seconds(
            node,
            "length",
            allow_negative=False,
            allow_missing=True,
        )
        gain = cls._expression_attribute(node, "gain")
        pan = cls._expression_attribute(node, "pan")
        preset = cls._preset_attribute(node)
        if start is not None:
            attrs["start"] = start
        if pre_gap is not None:
            attrs["pre_gap"] = pre_gap
        if end is not None:
            attrs["end"] = end
        if post_gap is not None:
            attrs["post_gap"] = post_gap
        if length is not None:
            attrs["length"] = length
        if gain is not None:
            attrs["gain"] = gain
        if pan is not None:
            attrs["pan"] = pan
        if preset is not None:
            attrs["preset"] = preset
        return attrs

    def process_attrs(self, attrs: Mapping[str, AudioAttrValue]) -> None:
        self.start_expression = None
        self.pre_gap_expression = None
        self.end_expression = None
        self.post_gap_expression = None
        self.length_expression = None
        self.pre_gap = 0.0
        self.post_gap = 0.0
        self.gain_expression = None
        self.pan_expression = None
        if "start" in attrs and "pre_gap" in attrs:
            raise self.document_error(
                f"{self.node.display_name} may not specify both start and pre_gap"
            )
        if "length" in attrs and "post_gap" in attrs:
            raise self.document_error(
                f"{self.node.display_name} may not specify both length and post_gap"
            )
        right_side_attrs = sum(1 for attribute_name in ("end", "length", "post_gap") if attribute_name in attrs)
        if right_side_attrs > 1:
            raise self.document_error(
                f"{self.node.display_name} may not specify more than one of end, length, and post_gap"
            )
        self.start_expression = cast(str | None, attrs.get("start"))
        self.pre_gap_expression = self._attr_expression_text(attrs.get("pre_gap"))
        self.end_expression = cast(str | None, attrs.get("end"))
        self.post_gap_expression = self._attr_expression_text(attrs.get("post_gap"))
        self.length_expression = self._attr_expression_text(attrs.get("length"))
        self.pre_gap = 0.0 if self.pre_gap_expression is None else float(self.pre_gap_expression)
        self.post_gap = 0.0 if self.post_gap_expression is None else float(self.post_gap_expression)
        self.gain_expression = cast(str | None, attrs.get("gain"))
        self.pan_expression = cast(str | None, attrs.get("pan"))

    def _replace_attrs(self, attrs: Mapping[str, AudioAttrValue]) -> None:
        self.attrs = dict(attrs)
        self.process_attrs(self.attrs)

    @staticmethod
    def _node_error(node: DocumentNode, message: str) -> DocumentError:
        return node.error(message)

    @classmethod
    def _timing_attribute_seconds(
        cls,
        node: DocumentNode | None,
        attribute_name: str,
        *,
        allow_negative: bool,
        allow_missing: bool = False,
    ) -> float | None:
        if node is None:
            return None if allow_missing else 0.0
        raw_value = node.attributes.get(attribute_name)
        if raw_value is None:
            return None if allow_missing else 0.0
        normalized = raw_value.strip()
        if not normalized:
            raise cls._node_error(node, f"{node.display_name} {attribute_name} cannot be empty")
        try:
            seconds = float(normalized)
        except ValueError as exc:
            raise cls._node_error(
                node,
                f"{node.display_name} {attribute_name} must be a number of seconds"
            ) from exc
        if not allow_negative and seconds < 0:
            raise cls._node_error(
                node,
                f"{node.display_name} {attribute_name} must be non-negative seconds"
            )
        return seconds

    @classmethod
    def _preset_attribute(cls, node: DocumentNode | None) -> str | None:
        if node is None:
            return None
        raw_value = node.attributes.get("preset")
        if raw_value is None:
            return None
        normalized = raw_value.strip()
        if not normalized:
            raise cls._node_error(node, f"{node.display_name} preset attribute cannot be empty")
        return normalized

    @classmethod
    def _expression_attribute(cls, node: DocumentNode | None, attribute_name: str) -> str | None:
        if node is None:
            return None
        raw_value = node.attributes.get(attribute_name)
        if raw_value is None:
            return None
        normalized = raw_value.strip()
        if not normalized:
            raise cls._node_error(node, f"{node.display_name} {attribute_name} cannot be empty")
        try:
            validate_expression(normalized)
        except (SyntaxError, ValueError) as exc:
            raise cls._node_error(
                node,
                f"{node.display_name} {attribute_name} must be a valid expression: {exc}"
            ) from exc
        return normalized

    @staticmethod
    def _attr_expression_text(value: AudioAttrValue | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, float):
            return str(value)
        return str(value)

    def _rebuild_audio_marks(self, audio_plans: Sequence["AudioPlan"]) -> None:
        self.audio_marks.clear()
        self._audio_mark_counts.clear()
        self.inner_plans(*audio_plans)

    @property
    def explicit_start(self) -> bool:
        return self.start_expression is not None

    def left_side_variables(
        self,
        *,
        outer_marks: Mapping[str, float] | None = None,
        explicit_start: bool,
    ) -> dict[str, float]:
        variables = {"natural_length": self._raw_inner_last - self._raw_inner_first}
        for mark_id, position in self.audio_marks_inner.items():
            variables[f"inner_{mark_id}"] = position
        if explicit_start and outer_marks is not None:
            for mark_id, position in outer_marks.items():
                variables[f"outer_{mark_id}"] = position
        return variables

    def right_side_variables(
        self,
        *,
        outer_marks: Mapping[str, float] | None = None,
        explicit_start: bool,
    ) -> dict[str, float]:
        variables = self.left_side_variables(
            outer_marks=outer_marks,
            explicit_start=explicit_start,
        )
        if explicit_start:
            variables["start"] = self.start
            variables["first"] = self.start + self.inner_first
            variables["last"] = self.start + self.inner_last
        return variables

    def render_time_variables(self) -> dict[str, float]:
        variables = {"natural_length": float(self._seconds_to_frames(self.natural_length))}
        variables.update(self.audio_marks_render)
        return variables

    def evaluate_expression(
        self,
        expression: str,
        variables: Mapping[str, float],
        *,
        attribute_name: str,
    ) -> float:
        try:
            return eval_expression(expression, variables, coerce_real)
        except Exception as exc:
            raise self.document_error(
                f"{self.node.display_name} {attribute_name} expression failed: {exc}"
            ) from exc

    def _finalize_intrinsic_layout(self) -> None:
        self.audio_marks_inner = dict(self._layout_marks_inner)
        self._content_start = self._raw_inner_first
        self._content_end = self._raw_inner_last
        self.inner_first = self._raw_inner_first
        self.inner_last = self._raw_inner_last
        self.length = self._raw_length
        self.end = self.length
        self.start = 0.0

    def _merge_incoming_marks(self, incoming_marks: Mapping[str, float] | None) -> None:
        if incoming_marks is not None:
            for mark_id, position in incoming_marks.items():
                self.audio_marks_inner.setdefault(mark_id, float(position) - self.start)
        self.audio_marks_render = {
            mark_id: float(self._seconds_to_frames(position - self.inner_first))
            for mark_id, position in self.audio_marks_inner.items()
        }

    def _apply_node_render_geometry(self, result: RenderResult) -> RenderResult:
        total_frames = max(0, self._seconds_to_frames(self.natural_length))
        content_start_frame = self._seconds_to_frames(self._content_start - self.inner_first)
        if total_frames == result.frame_count and content_start_frame == 0:
            return result
        audio = self._empty_audio(total_frames)
        if result.frame_count != 0:
            write_start = max(0, content_start_frame)
            write_end = min(total_frames, write_start + result.frame_count)
            source_end = max(0, write_end - write_start)
            audio[write_start:write_end] = result.audio[:source_end]
        return type(result)(audio=audio)

    def _seconds_to_frames(self, seconds: float) -> int:
        return int(round(seconds * self.config.resolved_output_sample_rate))

    def _frames_to_seconds(self, frame_count: int) -> float:
        return float(frame_count) / self.config.resolved_output_sample_rate

    def _empty_audio(self, frame_count: int) -> np.ndarray:
        if self.config.resolved_output_channels == 1:
            return np.zeros(frame_count, dtype=np.float32)
        return np.zeros((frame_count, self.config.resolved_output_channels), dtype=np.float32)

    async def _layout_audio(self) -> None:
        self._layout_marks_inner: dict[str, float] = {}
        self._raw_inner_first = 0.0
        self._raw_inner_last = 0.0
        self._raw_length = 0.0
        self._content_start = 0.0
        self._content_end = 0.0
        await self.layout_node()
        self._finalize_intrinsic_layout()

    async def _render_audio(self, incoming_marks: Mapping[str, float] | None) -> RenderResult:
        await self.layout()
        self._merge_incoming_marks(incoming_marks)
        return await self.post_render(await self.render_node())


@inject(config=ProductionConfig)
class MarkPlan(AudioPlan):
    """Zero-length plan that introduces one named cut mark."""

    def __init__(self, node: "MarkNode", id: str, **kwargs) -> None:
        super().__init__(node=node, **kwargs)
        self.id = id
        self.audio_marks.append(id)
        self._audio_mark_counts[id] = 1

    def __repr__(self) -> str:
        return f"MarkPlan(id={self.id!r})"

    async def layout_node(self) -> None:
        self._layout_marks_inner = {self.id: 0.0}
        self._raw_length = 0.0

    async def render_node(self) -> RenderResult:
        return RenderResult.empty(channels=self.config.resolved_output_channels)


@dataclass(slots=True)
class DialogueAudio(DialogueContents):
    """Inline zero-duration audio insertion point within a script."""

    audio_plan: AudioPlan


@inject(config=ProductionConfig)
class SpeakerMapPlan(PlanningNode):
    """Validated speaker map with canonical lookup into resolved voice files."""

    def __init__(self, node: SpeakerMapNode, **kwargs) -> None:
        super().__init__(node=node, **kwargs)
        self._voices_by_key: dict[str, SpeakerVoiceReference] = {}

    async def async_ready(self):
        """Parse YAML, validate entries, and resolve voice references."""
        loaded = yaml.safe_load(self.node.normalized_text_content)
        if not isinstance(loaded, dict):
            raise self.document_error(
                "The <speaker-map> YAML must be a mapping of speaker names to voice names"
            )
        if not loaded:
            raise self.document_error("The <speaker-map> did not define any speakers")

        voices_by_key: dict[str, SpeakerVoiceReference] = {}
        for speaker_name, voice_name in loaded.items():
            if not isinstance(speaker_name, str) or not isinstance(voice_name, str):
                raise self.document_error(
                    "Speaker names and voice names in <speaker-map> must be strings"
                )
            normalized_speaker = speaker_name.strip()
            normalized_voice = voice_name.strip()
            if not normalized_speaker or not normalized_voice:
                raise self.document_error(
                    "Speaker names and voice names in <speaker-map> cannot be empty"
                )
            key = normalized_speaker.lower()
            if key in voices_by_key:
                raise self.document_error(
                    f"Speaker {speaker_name!r} is defined more than once in <speaker-map>"
                )
            voices_by_key[key] = SpeakerVoiceReference(
                authored_name=normalized_speaker,
                voice_name=normalized_voice,
                resolved_path=self._resolve_voice_path(normalized_speaker, normalized_voice),
            )

        self._voices_by_key = voices_by_key
        production_injector = self._production_injector()
        if production_injector is not None:
            try:
                production_injector.add_provider(self)
            except ExistingProvider as exc:
                raise self.document_error("A <production> may contain only one <speaker-map>") from exc
        return await super().async_ready()

    def _production_injector(self) -> Injector | None:
        provider_injector = self.ainjector.injector.injector_containing(PRODUCTION_PLANNING_INJECTOR_KEY)
        if provider_injector is None:
            return None
        return provider_injector.get_instance(PRODUCTION_PLANNING_INJECTOR_KEY)

    def lookup(self, speaker_name: str) -> SpeakerVoiceReference:
        return self._voices_by_key[speaker_name.strip().lower()]

    @property
    def voices_by_key(self) -> Mapping[str, SpeakerVoiceReference]:
        return self._voices_by_key

    def _resolve_voice_path(self, speaker_name: str, voice_name: str) -> Path:
        direct_candidates = [
            Path(voice_name).expanduser(),
            (self.config.resolved_voice_directory / voice_name).expanduser(),
        ]
        for candidate in direct_candidates:
            if candidate.is_file():
                return candidate

        voice_catalog = self._load_voice_catalog()
        candidate_keys = [
            voice_name,
            Path(voice_name).name,
            Path(voice_name).stem,
            voice_name.lower(),
            Path(voice_name).name.lower(),
            Path(voice_name).stem.lower(),
        ]
        for candidate in candidate_keys:
            resolved = voice_catalog.get(candidate)
            if resolved is not None:
                return resolved

        available = ", ".join(sorted({path.name for path in voice_catalog.values()}))
        raise self.document_error(
            f"Voice {voice_name!r} for speaker {speaker_name!r} was not found in "
            f"{self.config.resolved_voice_directory}. Available voices: {available}"
        )

    def _load_voice_catalog(self) -> dict[str, Path]:
        voice_directory = self.config.resolved_voice_directory
        if not voice_directory.is_dir():
            raise self.document_error(f"Voice directory does not exist: {voice_directory}")

        catalog: dict[str, Path] = {}
        for child in sorted(voice_directory.iterdir()):
            if not child.is_file() or child.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                continue
            catalog.setdefault(child.name, child)
            catalog.setdefault(child.stem, child)
            catalog.setdefault(child.name.lower(), child)
            catalog.setdefault(child.stem.lower(), child)

        if not catalog:
            raise self.document_error(f"No supported voice files were found in {voice_directory}")
        return catalog

    async def render_node(self):
        return None


@inject(config=ProductionConfig)
class ScriptPlan(AudioPlan):
    """Plan for one script element and its eventual speech render request."""

    def __init__(self, node: ScriptNode, **kwargs) -> None:
        super().__init__(node=node, **kwargs)
        self.speaker_map_plan: SpeakerMapPlan | None = None
        self.contents: list[DialogueContents] = []
        self.ordered_speakers: list[SpeakerVoiceReference] = []
        self.render_request: ScriptRenderRequest | None = None
        self._registered_request = None

    def __repr__(self) -> str:
        line = self.node.location.line
        first_words = self._preferred_first_words(self.dialogue_lines)
        if line is None:
            if first_words:
                return f"ScriptPlan(line=unknown, first_words={first_words!r})"
            return "ScriptPlan(line=unknown)"
        if first_words:
            return f"ScriptPlan(line={line}, first_words={first_words!r})"
        return f"ScriptPlan(line={line})"

    async def async_ready(self):
        """Normalize dialogue and register the request with the shared resource."""
        self.speaker_map_plan = self._require_speaker_map_plan()
        self.contents = await self._parse_contents()
        self.ordered_speakers = self._ordered_unique_speakers(self.dialogue_lines)
        if self.dialogue_lines:
            first_words = self._preferred_first_words(self.dialogue_lines)
            self.render_request = ScriptRenderRequest(
                dialogue_lines=list(self.dialogue_lines),
                first_words=first_words,
            )
        resource = await self.ainjector.get_instance_async(self._tts_resource_type())
        self._registered_request = await resource.register_request(self.render_request)
        return await super().async_ready()

    def _tts_resource_type(self):
        if self.node.tts == "qwen":
            from .qwen_tts import QwenTtsResource

            return QwenTtsResource
        from .vibevoice import VibeVoiceResource

        return VibeVoiceResource

    def _require_speaker_map_plan(self) -> SpeakerMapPlan:
        provider_injector = self.ainjector.injector.injector_containing(SpeakerMapPlan)
        if provider_injector is None:
            raise self.document_error(
                "A <script> requires a <speaker-map> to be planned before it"
            )
        return provider_injector.get_instance(SpeakerMapPlan)

    async def render_node(self) -> RenderResult:
        return await self._registered_request.render()

    async def layout_node(self) -> None:
        result = await self._registered_request.render()
        self._raw_inner_last = self._frames_to_seconds(result.frame_count)
        self._raw_length = self._raw_inner_last

    @property
    def dialogue_lines(self) -> list[DialogueLine]:
        return [content for content in self.contents if isinstance(content, DialogueLine)]

    @property
    def dialogue_audios(self) -> list[DialogueAudio]:
        return [content for content in self.contents if isinstance(content, DialogueAudio)]

    @classmethod
    async def from_node(cls, ainjector, node: ScriptNode) -> AudioPlan:
        node.tts
        script_plan = await ainjector(
            cls,
            node=node,
            attrs={} if node.element_children else None,
        )
        audio_plan: AudioPlan = script_plan

        if script_plan.needs_forced_alignment():
            audio_plan = await cls._build_aligned_audio_plan(
                ainjector,
                node,
                script_plan,
                attrs=type(script_plan).attrs_from_node(node),
            )
        return audio_plan

    @classmethod
    async def _build_aligned_audio_plan(
        cls,
        ainjector,
        node: ScriptNode,
        script_plan: "ScriptPlan",
        *,
        attrs: Mapping[str, AudioAttrValue],
    ) -> AudioPlan:
        from .forced_alignment import AlignedScriptSource, ScriptSlice

        aligned_script_source = await ainjector(
            AlignedScriptSource,
            node=node,
            script_plan=script_plan,
        )
        audio_plans: list[AudioPlan] = []
        content_index = 0

        while content_index < len(script_plan.contents):
            content = script_plan.contents[content_index]
            if isinstance(content, DialogueAudio):
                audio_plans.append(content.audio_plan)
                content_index += 1
                continue
            if content.handling == "special":
                audio_plans.append(
                    await ainjector(
                        ScriptSlice,
                        node=content.node,
                        aligned_script_source=aligned_script_source,
                        start_marker=content_index,
                        end_marker=content_index + 1,
                        name=content.spoken_text[:30] or content.speaker.authored_name,
                    )
                )
                content_index += 1
                continue
            if content.handling != "normal":
                content_index += 1
                continue
            end_index = script_plan.advance_normal_dialogue_slice_end(content_index)
            audio_plans.append(
                await ainjector(
                    ScriptSlice,
                    node=node,
                    attrs={},
                    aligned_script_source=aligned_script_source,
                    start_marker=content_index,
                    end_marker=end_index,
                    name=cls._script_slice_name(script_plan.contents, content_index),
                )
            )
            content_index = end_index

        if not audio_plans:
            audio_plans.append(
                await ainjector(
                    ScriptSlice,
                    node=node,
                    attrs={},
                    aligned_script_source=aligned_script_source,
                    start_marker=len(script_plan.contents),
                    end_marker=len(script_plan.contents),
                    name="ignored script",
                )
            )
        return await ainjector(
            ComposeAudioPlan,
            node=node,
            audio_plans=audio_plans,
            attrs=attrs,
        )

    @staticmethod
    def _script_slice_name(
        contents: Sequence[DialogueContents],
        marker_index: int,
    ) -> str:
        if marker_index >= len(contents):
            return "script end"
        for content in contents[marker_index:]:
            if isinstance(content, DialogueLine) and content.handling == "normal":
                return content.spoken_text[:30]
            if isinstance(content, DialogueAudio):
                return repr(content.audio_plan)
        return "ignored script"

    async def _parse_contents(self) -> list[DialogueContents]:
        from .document import IgnoreNode, LineNode, TextNode
        from .forced_alignment import ScriptSlice

        contents: list[DialogueContents] = []
        pending_text: list[str] = []

        def flush_pending_text() -> None:
            if not pending_text:
                return
            contents.extend(self._parse_dialogue_text("".join(pending_text)))
            pending_text.clear()

        for child in self.node.children:
            if isinstance(child, TextNode):
                pending_text.append(child.text)
                continue
            flush_pending_text()
            if isinstance(child, IgnoreNode):
                contents.extend(
                    self._parse_dialogue_text(
                        child.text_content,
                        handling="ignore",
                    )
                )
                continue
            if isinstance(child, LineNode):
                line_attrs = ScriptSlice.attrs_from_node(child)
                handling: Literal["normal", "special"] = "special" if line_attrs else "normal"
                contents.append(
                    DialogueLine(
                        speaker=self.speaker_map_plan.lookup(child.speaker),
                        spoken_text=child.normalized_text_content,
                        handling=handling,
                        node=child if handling == "special" else None,
                    )
                )
                continue
            contents.append(DialogueAudio(audio_plan=await child.plan(self.ainjector)))
        flush_pending_text()
        return contents

    def _parse_dialogue_text(
        self,
        text: str,
        *,
        handling: Literal["normal", "ignore"] = "normal",
    ) -> list[DialogueLine]:
        """Parse dialogue text into speaker-scoped lines.

        A new ``DialogueLine`` starts only when the parser encounters a
        non-empty line of the form ``speaker: ...`` and that speaker resolves
        in the current ``SpeakerMapPlan``. Blank lines end only the current
        paragraph, not the current ``DialogueLine``; continuation lines are
        folded into the current speaker's text. The current ``DialogueLine``
        is therefore finalized only by a new recognized speaker stanza or by
        reaching the end of this text chunk.
        """
        text = re.sub(r"^\s*\n", "", text)
        text = re.sub(r"\n\s*$", "", text)
        if not text:
            return []

        lines: list[DialogueLine] = []
        current_speaker: SpeakerVoiceReference | None = None
        current_paragraph: list[str] = []
        current_paragraphs: list[str] = []

        def flush_paragraph() -> None:
            if current_paragraph:
                current_paragraphs.append(" ".join(current_paragraph).strip())
                current_paragraph.clear()

        def flush_stanza() -> None:
            flush_paragraph()
            if current_speaker is None:
                return
            spoken_text = " ".join(paragraph for paragraph in current_paragraphs if paragraph).strip()
            current_paragraphs.clear()
            if spoken_text:
                lines.append(
                    DialogueLine(
                        speaker=current_speaker,
                        spoken_text=spoken_text,
                        handling=handling,
                    )
                )

        for raw_line in text.splitlines():
            stripped_line = raw_line.strip()
            if not stripped_line:
                flush_paragraph()
                continue
            match = _SPEAKER_LINE_RE.match(stripped_line)
            if match is not None:
                candidate_speaker = match.group(1).strip()
                try:
                    speaker_ref = self.speaker_map_plan.lookup(candidate_speaker)
                except KeyError:
                    speaker_ref = None
                if speaker_ref is not None:
                    flush_stanza()
                    current_speaker = speaker_ref
                    current_paragraph.append(match.group(2).strip())
                    continue
            if current_speaker is None:
                raise self.document_error(
                    "Scripts may begin only with a recognized `speaker:` stanza"
                )
            current_paragraph.append(stripped_line)

        flush_stanza()
        return lines

    def needs_forced_alignment(self) -> bool:
        return any(
            isinstance(content, DialogueAudio)
            or (isinstance(content, DialogueLine) and content.handling != "normal")
            for content in self.contents
        )

    def next_non_audio_index(self, start_index: int) -> int:
        index = start_index
        while index < len(self.contents) and isinstance(self.contents[index], DialogueAudio):
            index += 1
        return index

    def advance_normal_dialogue_slice_end(self, start_index: int) -> int:
        index = start_index
        while index < len(self.contents):
            content = self.contents[index]
            if isinstance(content, DialogueAudio):
                break
            if not isinstance(content, DialogueLine) or content.handling != "normal":
                break
            index += 1
        return index

    def _ordered_unique_speakers(
        self,
        dialogue_lines: Sequence[DialogueLine],
    ) -> list[SpeakerVoiceReference]:
        seen: set[str] = set()
        ordered: list[SpeakerVoiceReference] = []
        for line in dialogue_lines:
            key = line.speaker.authored_name.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(line.speaker)
        return ordered

    @staticmethod
    def _preferred_first_words(
        dialogue_lines: Sequence[DialogueLine],
    ) -> str:
        preferred_line = next(
            (
                line
                for line in dialogue_lines
                if line.handling == "normal" and line.spoken_text.strip()
            ),
            None,
        )
        if preferred_line is None:
            preferred_line = next(
                (line for line in dialogue_lines if line.spoken_text.strip()),
                None,
            )
        if preferred_line is None:
            return "empty-script"
        label = " ".join(preferred_line.spoken_text.split()).strip()
        return label[:40] or "empty-script"


@inject(config=ProductionConfig)
class ComposeAudioPlan(AudioPlan):
    """Audio plan that places child render results in one mixed timeline."""

    def __init__(
        self,
        node: DocumentNode,
        audio_plans: Sequence[AudioPlan],
        **kwargs,
    ) -> None:
        super().__init__(node=node, **kwargs)
        self.audio_plans = list(audio_plans)
        self.inner_plans(*self.audio_plans)

    def __repr__(self) -> str:
        return f"ComposeAudioPlan(children={len(self.audio_plans)})"

    def leaf_audio_plans(self) -> list[AudioPlan]:
        flattened: list[AudioPlan] = []
        for audio_plan in self.audio_plans:
            flattened.extend(audio_plan.leaf_audio_plans())
        return flattened

    def cut_before_mark(self, audio_mark: str) -> None:
        super().cut_before_mark(audio_mark)
        for index, audio_plan in enumerate(self.audio_plans):
            if audio_mark not in audio_plan.audio_marks:
                continue
            self.audio_plans = self.audio_plans[index:]
            self.audio_plans[0].cut_before_mark(audio_mark)
            self._rebuild_audio_marks(self.audio_plans)
            return
        raise ValueError(f"Unknown or ambiguous audio mark {audio_mark!r}")

    def cut_after_mark(self, audio_mark: str) -> None:
        super().cut_after_mark(audio_mark)
        for index, audio_plan in enumerate(self.audio_plans):
            if audio_mark not in audio_plan.audio_marks:
                continue
            self.audio_plans = self.audio_plans[: index + 1]
            self.audio_plans[-1].cut_after_mark(audio_mark)
            self._rebuild_audio_marks(self.audio_plans)
            return
        raise ValueError(f"Unknown or ambiguous audio mark {audio_mark!r}")

    async def layout_node(self) -> None:
        await asyncio.gather(*(audio_plan.layout() for audio_plan in self.audio_plans))
        running_outer_marks: dict[str, float] = {}
        running_outer_mark_counts: dict[str, int] = {}
        cursor = 0.0
        explicit_children: list[AudioPlan] = []

        for audio_plan in self.audio_plans:
            if audio_plan.explicit_start:
                explicit_children.append(audio_plan)
                continue
            cursor = self._place_child_automatically(
                audio_plan,
                cursor=cursor,
                outer_marks=running_outer_marks,
            )
            self._merge_child_outer_marks(
                running_outer_marks,
                running_outer_mark_counts,
                audio_plan,
            )

        for audio_plan in explicit_children:
            audio_plan.start = audio_plan.evaluate_expression(
                audio_plan.start_expression,
                audio_plan.left_side_variables(
                    outer_marks=running_outer_marks,
                    explicit_start=True,
                ),
                attribute_name="start",
            )
            self._resolve_child_right_side(audio_plan, running_outer_marks)
            self._merge_child_outer_marks(
                running_outer_marks,
                running_outer_mark_counts,
                audio_plan,
            )

        if not self.audio_plans:
            self._layout_marks_inner = {}
            self._raw_inner_first = 0.0
            self._raw_inner_last = 0.0
            self._raw_length = 0.0
            return

        self._raw_inner_first = min(
            min(audio_plan.start, audio_plan.start + audio_plan.inner_first)
            for audio_plan in self.audio_plans
        )
        self._raw_inner_last = max(
            max(audio_plan.end, audio_plan.start + audio_plan.inner_last)
            for audio_plan in self.audio_plans
        )
        self._raw_length = max((audio_plan.end for audio_plan in self.audio_plans), default=0.0)
        self._layout_marks_inner = self._visible_child_marks_inner()

    async def render_node(self) -> RenderResult:
        rendered_results = await asyncio.gather(
            *(audio_plan.render(self.audio_marks_inner) for audio_plan in self.audio_plans)
        )
        return self._compose_results(rendered_results)

    def _place_child_automatically(
        self,
        audio_plan: AudioPlan,
        *,
        cursor: float,
        outer_marks: Mapping[str, float],
    ) -> float:
        pre_gap = 0.0
        if audio_plan.pre_gap_expression is not None:
            pre_gap = audio_plan.evaluate_expression(
                audio_plan.pre_gap_expression,
                audio_plan.left_side_variables(explicit_start=False),
                attribute_name="pre_gap",
            )
        audio_plan.start = cursor + pre_gap
        self._resolve_child_right_side(audio_plan, outer_marks)
        return cursor + audio_plan.length

    def _resolve_child_right_side(
        self,
        audio_plan: AudioPlan,
        outer_marks: Mapping[str, float],
    ) -> None:
        intrinsic_length = audio_plan.inner_last
        resolved_length = intrinsic_length
        if audio_plan.length_expression is not None:
            resolved_length = audio_plan.evaluate_expression(
                audio_plan.length_expression,
                audio_plan.right_side_variables(
                    outer_marks=outer_marks,
                    explicit_start=False,
                ),
                attribute_name="length",
            )
        elif audio_plan.post_gap_expression is not None:
            post_gap = audio_plan.evaluate_expression(
                audio_plan.post_gap_expression,
                audio_plan.right_side_variables(
                    outer_marks=outer_marks,
                    explicit_start=False,
                ),
                attribute_name="post_gap",
            )
            resolved_length = intrinsic_length + post_gap
        default_end = audio_plan.start + resolved_length
        if audio_plan.end_expression is not None:
            audio_plan.end = audio_plan.evaluate_expression(
                audio_plan.end_expression,
                audio_plan.right_side_variables(
                    outer_marks=outer_marks,
                    explicit_start=True,
                ),
                attribute_name="end",
            )
        else:
            audio_plan.end = default_end
        audio_plan.length = audio_plan.end - audio_plan.start

    def _merge_child_outer_marks(
        self,
        outer_marks: dict[str, float],
        outer_mark_counts: dict[str, int],
        audio_plan: AudioPlan,
    ) -> None:
        for mark_id, position in audio_plan.audio_marks_inner.items():
            count = outer_mark_counts.get(mark_id, 0) + 1
            outer_mark_counts[mark_id] = count
            if count == 1:
                outer_marks[mark_id] = audio_plan.start + position
            else:
                outer_marks.pop(mark_id, None)

    def _visible_child_marks_inner(self) -> dict[str, float]:
        mark_counts: dict[str, int] = {}
        mark_positions: dict[str, float] = {}
        for audio_plan in self.audio_plans:
            for mark_id, position in audio_plan.audio_marks_inner.items():
                mark_counts[mark_id] = mark_counts.get(mark_id, 0) + 1
                mark_positions[mark_id] = audio_plan.start + position
        return {
            mark_id: mark_positions[mark_id]
            for mark_id, count in mark_counts.items()
            if count == 1
        }

    def _compose_results(self, results: Sequence[RenderResult]) -> RenderResult:
        if not results:
            return RenderResult.empty(channels=self.config.resolved_output_channels)
        placements: list[tuple[int, RenderResult]] = []
        total_frames = max(0, self._seconds_to_frames(self._raw_inner_last - self._raw_inner_first))
        audio = self._empty_audio(total_frames)
        for audio_plan, result in zip(self.audio_plans, results, strict=True):
            start_frame = self._seconds_to_frames(
                audio_plan.start + audio_plan.inner_first - self._raw_inner_first
            )
            end_frame = start_frame + result.frame_count
            write_debug_message(
                self.config,
                "compose_audio",
                (
                    f"{self!r} places {audio_plan!r} from "
                    f"{self._frames_to_seconds(start_frame):.3f}s to "
                    f"{self._frames_to_seconds(end_frame):.3f}s"
                ),
            )
            placements.append((start_frame, result))
            if result.frame_count == 0:
                continue
            write_start = max(0, start_frame)
            write_end = min(total_frames, write_start + result.frame_count)
            source_end = max(0, write_end - write_start)
            audio[write_start:write_end] += result.audio[:source_end]
        return RenderResult(audio=audio)


@inject(config=ProductionConfig)
class SlicePlan(AudioPlan):
    """Audio plan that returns a time slice of an existing render result."""

    def __init__(
        self,
        result: RenderResult,
        *,
        start_time: float,
        end_time: float,
        name: str | None = None,
        node=None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("attrs", {})
        super().__init__(node=node, **kwargs)
        self.result = result
        self.start_time = start_time
        self.end_time = end_time
        self.name = name

    def __repr__(self) -> str:
        if self.name is not None:
            return f"SlicePlan(name={self.name!r})"
        return (
            "SlicePlan("
            f"start_time={self.start_time:.3f}, "
            f"end_time={self.end_time:.3f})"
        )

    async def layout_node(self) -> None:
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        self._raw_inner_last = self.end_time - self.start_time
        self._raw_length = self._raw_inner_last

    async def render_node(self) -> RenderResult:
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        frame_rate = self.config.resolved_output_sample_rate
        start_frame = max(0, int(round(self.start_time * frame_rate)))
        end_frame = max(start_frame, int(round(self.end_time * frame_rate)))
        return self.result.slice_frames(start_frame, end_frame)


@inject(config=ProductionConfig)
class ProductionPlan(ComposeAudioPlan):
    """Top-level production plan that preserves script order."""

    @classmethod
    def attrs_from_node(cls, node: DocumentNode | None) -> AudioAttrs:
        attrs = super().attrs_from_node(node)
        attrs["preset"] = "master"
        return attrs

    async def render_node(self) -> ProductionResult:
        """Render scripts in document order and clip to the production boundary."""
        combined = await super().render_node()
        trimmed = self._trim_to_production_boundary(combined)
        return ProductionResult(audio=trimmed.audio)

    def _apply_node_render_geometry(self, result: RenderResult) -> RenderResult:
        return result

    def _trim_to_production_boundary(self, result: RenderResult) -> RenderResult:
        trim_start_frames = max(0, self._seconds_to_frames(-self.inner_first))
        trim_end_frames = max(
            trim_start_frames,
            self._seconds_to_frames(self.length - self.inner_first),
        )
        audio = result.audio[trim_start_frames:trim_end_frames]
        final_frames = max(0, self._seconds_to_frames(self.length))
        if audio.shape[0] < final_frames:
            padded = self._empty_audio(final_frames)
            padded[:audio.shape[0]] = audio
            audio = padded
        return RenderResult(audio=audio)
