from __future__ import annotations

import asyncio
import math
from math import gcd
from collections.abc import Iterable
from typing import TYPE_CHECKING, Mapping, Sequence, cast

import numpy as np
from carthage.dependency_injection import inject
from scipy.signal import resample_poly

from .config import ProductionConfig
from .debug import write_debug_message
from .expressions import coerce_real, eval_expression, validate_expression
from .planning import AudioAttrValue, AudioAttrs, PlanningNode
from .rendering import RenderResult


if TYPE_CHECKING:
    from .document import DocumentNode, MarkNode


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
}


def normalize_audio_array(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    return np.ascontiguousarray(array, dtype=np.float32)


def resample_audio(
    audio: np.ndarray,
    *,
    input_sample_rate: int,
    output_sample_rate: int,
) -> np.ndarray:
    if input_sample_rate == output_sample_rate:
        return normalize_audio_array(audio)
    factor = gcd(input_sample_rate, output_sample_rate)
    up = output_sample_rate // factor
    down = input_sample_rate // factor
    if audio.ndim == 1:
        return np.ascontiguousarray(resample_poly(audio, up, down), dtype=np.float32)
    return np.ascontiguousarray(resample_poly(audio, up, down, axis=0), dtype=np.float32)


def convert_channel_count(audio: np.ndarray, *, output_channels: int) -> np.ndarray:
    if output_channels < 1:
        raise ValueError("output_channels must be at least 1")
    if output_channels == 1:
        if audio.ndim == 1:
            return normalize_audio_array(audio)
        if audio.shape[1] == 1:
            return normalize_audio_array(audio[:, 0])
        return normalize_audio_array(audio.mean(axis=1))
    if audio.ndim == 1:
        mono = audio[:, np.newaxis]
    elif audio.shape[1] == 1:
        mono = audio
    elif audio.shape[1] == output_channels:
        return normalize_audio_array(audio)
    else:
        mono = audio.mean(axis=1, keepdims=True)
    return normalize_audio_array(np.repeat(mono, output_channels, axis=1))


def convert_audio_format(
    audio: np.ndarray,
    *,
    input_sample_rate: int,
    output_sample_rate: int,
    output_channels: int,
) -> np.ndarray:
    converted = resample_audio(
        normalize_audio_array(audio),
        input_sample_rate=input_sample_rate,
        output_sample_rate=output_sample_rate,
    )
    return convert_channel_count(converted, output_channels=output_channels)


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
        self._layout_complete = False
        self.audio_marks_inner: dict[str, float] = {}
        self.audio_marks_render: dict[str, float] = {}
        self._incoming_marks_inner: dict[str, float] = {}
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
        self._local_audio_marks: list[str] = []
        resolved_attrs = type(self).attrs_from_node(node) if attrs is None else dict(attrs)
        self._replace_attrs(resolved_attrs)

    async def async_resolve(self):
        loop_attr_names = {
            "loop_beg",
            "loop_end",
            "loop_loops",
            "loop_until",
            "loop_silence",
            "loop_outro",
            "loop_whole",
        }
        wrapper_attr_names = type(self).wrapper_attr_names()
        loop_enabled = "loop_until" in self.attrs or "loop_loops" in self.attrs
        if not loop_enabled:
            return self
        loop_attrs = {key: value for key, value in self.attrs.items() if key in loop_attr_names}
        wrapper_attrs = {
            key: value
            for key, value in self.attrs.items()
            if key in wrapper_attr_names
        }
        retained_attrs = {
            key: value
            for key, value in self.attrs.items()
            if key not in loop_attr_names and key not in wrapper_attr_names
        }
        self._replace_attrs(retained_attrs)
        return await self.ainjector(
            LoopPlan,
            node=self.node,
            audio_plan=self,
            attrs={**wrapper_attrs, **loop_attrs},
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

    async def render(self) -> RenderResult:
        if self._render_task is None:
            self._render_task = asyncio.create_task(self._render_audio())
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
        from .effects import gain, pan

        updated_result = self._apply_node_render_geometry(result)
        if updated_result.frame_count == 0:
            return updated_result
        variables = self.render_time_variables()
        post_stage = None
        if self.gain_expression is not None:
            post_stage = gain(self.gain_expression, variables=variables)
        if self.pan_expression is None or updated_result.audio.ndim != 2 or updated_result.audio.shape[1] < 2:
            if post_stage is None:
                return updated_result
        else:
            pan_stage = pan(self.pan_expression, variables=variables)
            post_stage = pan_stage if post_stage is None else post_stage | pan_stage
        if post_stage is None:
            return updated_result
        try:
            post_stage.apply(
                updated_result.audio,
                sample_rate=self.config.resolved_output_sample_rate,
            )
        except Exception as exc:
            raise self.document_error(
                f"{self.node.display_name} render-time effect failed: {exc}"
            ) from exc
        return updated_result

    def leaf_audio_plans(self) -> list[AudioPlan]:
        return [self]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def output_preset_key(self) -> tuple[str, ...]:
        return self.preset_key

    def _add_visible_audio_mark(self, audio_mark: str) -> None:
        mark_count = self._audio_mark_counts.get(audio_mark, 0) + 1
        self._audio_mark_counts[audio_mark] = mark_count
        if mark_count == 1:
            self.audio_marks.append(audio_mark)
        elif mark_count == 2:
            self.audio_marks.remove(audio_mark)

    def inner_plans(self, *audio_plans: AudioPlan) -> None:
        for audio_plan in audio_plans:
            for audio_mark in audio_plan.audio_marks:
                self._add_visible_audio_mark(audio_mark)

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
        pre_gap = cls._expression_attribute(node, "pre_gap")
        end = cls._expression_attribute(node, "end")
        post_gap = cls._timing_attribute_seconds(
            node,
            "post_gap",
            allow_negative=True,
            allow_missing=True,
        )
        length = cls._expression_attribute(node, "length")
        loop_beg = cls._expression_attribute(node, "loop_beg")
        loop_end = cls._expression_attribute(node, "loop_end")
        loop_until = cls._expression_attribute(node, "loop_until")
        loop_loops = cls._number_attribute(
            node,
            "loop_loops",
            allow_negative=False,
            allow_missing=True,
        )
        loop_silence = cls._timing_attribute_seconds(
            node,
            "loop_silence",
            allow_negative=False,
            allow_missing=True,
        )
        loop_outro = cls._boolean_attribute(node, "loop_outro")
        loop_whole = cls._choice_attribute(node, "loop_whole", ("extend", "shorten", "no"))
        first_mark = cls._mark_attribute(node, "first_mark")
        last_mark = cls._mark_attribute(node, "last_mark")
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
        if loop_beg is not None:
            attrs["loop_beg"] = loop_beg
        if loop_end is not None:
            attrs["loop_end"] = loop_end
        if loop_until is not None:
            attrs["loop_until"] = loop_until
        if loop_loops is not None:
            attrs["loop_loops"] = loop_loops
        if loop_silence is not None:
            attrs["loop_silence"] = loop_silence
        if loop_outro is not None:
            attrs["loop_outro"] = loop_outro
        if loop_whole is not None:
            attrs["loop_whole"] = loop_whole
        if first_mark is not None:
            attrs["first_mark"] = first_mark
        if last_mark is not None:
            attrs["last_mark"] = last_mark
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
        self.loop_beg_expression = None
        self.loop_end_expression = None
        self.loop_until_expression = None
        self.loop_loops = None
        self.loop_silence = 0.0
        self.loop_outro = False
        self.loop_whole = "no"
        self.post_gap = 0.0
        self.first_mark = None
        self.last_mark = None
        self.gain_expression = None
        self.pan_expression = None
        self.preset_name = None
        self.preset_key: tuple[str, ...] = ()
        if "start" in attrs and "pre_gap" in attrs:
            raise self.document_error(
                f"{self.node.display_name} may not specify both start and pre_gap"
            )
        if "length" in attrs and "post_gap" in attrs:
            raise self.document_error(
                f"{self.node.display_name} may not specify both length and post_gap"
            )
        if "loop_until" in attrs and "loop_loops" in attrs:
            raise self.document_error(
                f"{self.node.display_name} may not specify both loop_until and loop_loops"
            )
        right_side_attrs = sum(
            1
            for attribute_name in ("end", "length", "post_gap")
            if attribute_name in attrs
        )
        if right_side_attrs > 1:
            raise self.document_error(
                f"{self.node.display_name} may not specify more than one of end, length, and post_gap"
            )
        loop_configured = "loop_until" in attrs or "loop_loops" in attrs
        loop_detail_attrs = {"loop_beg", "loop_end", "loop_silence", "loop_outro", "loop_whole"}
        if not loop_configured and any(attribute_name in attrs for attribute_name in loop_detail_attrs):
            raise self.document_error(
                f"{self.node.display_name} loop_beg, loop_end, loop_silence, loop_outro, and loop_whole require loop_until or loop_loops"
            )
        self.start_expression = cast(str | None, attrs.get("start"))
        self.pre_gap_expression = self._attr_expression_text(attrs.get("pre_gap"))
        self.end_expression = cast(str | None, attrs.get("end"))
        self.post_gap_expression = self._attr_expression_text(attrs.get("post_gap"))
        self.length_expression = self._attr_expression_text(attrs.get("length"))
        self.loop_beg_expression = cast(str | None, attrs.get("loop_beg"))
        self.loop_end_expression = cast(str | None, attrs.get("loop_end"))
        self.loop_until_expression = cast(str | None, attrs.get("loop_until"))
        self.loop_loops = cast(float | None, attrs.get("loop_loops"))
        self.loop_silence = cast(float, attrs.get("loop_silence", 0.0))
        self.loop_outro = cast(bool, attrs.get("loop_outro", False))
        self.loop_whole = cast(str, attrs.get("loop_whole", "no"))
        self.post_gap = 0.0 if self.post_gap_expression is None else float(self.post_gap_expression)
        self.first_mark = cast(str | None, attrs.get("first_mark"))
        self.last_mark = cast(str | None, attrs.get("last_mark"))
        if self.first_mark is not None and self.first_mark == self.last_mark:
            raise self.document_error(
                f"{self.node.display_name} may not specify the same id for first_mark and last_mark"
            )
        self._local_audio_marks = [
            mark_id
            for mark_id in (self.first_mark, self.last_mark)
            if mark_id is not None
        ]
        self.gain_expression = cast(str | None, attrs.get("gain"))
        self.pan_expression = cast(str | None, attrs.get("pan"))
        self.preset_name = cast(str | None, attrs.get("preset"))
        if self.preset_name is not None:
            from .effects import available_effect_chains, normalize_effect_chain_name

            normalized_preset_name = normalize_effect_chain_name(self.preset_name)
            available = set(available_effect_chains())
            if normalized_preset_name not in available:
                formatted = ", ".join(sorted(available))
                raise self.document_error(
                    f"Unknown preset {self.preset_name!r}. Available presets: {formatted}"
                )
            self.preset_name = normalized_preset_name
            self.preset_key = (normalized_preset_name,)

    @classmethod
    def wrapper_attr_names(cls) -> frozenset[str]:
        """Return audio attrs that belong on wrapper plans rather than the inner plan.

        ``AudioPlan.async_resolve()`` moves these attrs onto wrapper plans such
        as ``LoopPlan`` when a node is wrapped so the outermost plan produced
        from one document node continues to own ordinary AudioPlan semantics.

        New general AudioPlan attrs that should still apply after wrapping
        should be added here. Sound-specific attrs such as file trims do not
        belong here, because they must remain on the inner ``SoundPlan``.
        """

        return frozenset({
            "preset",
            "start",
            "pre_gap",
            "end",
            "post_gap",
            "length",
            "first_mark",
            "last_mark",
            "gain",
            "pan",
        })

    def _replace_attrs(self, attrs: Mapping[str, AudioAttrValue]) -> None:
        self.attrs = dict(attrs)
        self.process_attrs(self.attrs)
        self._rebuild_audio_marks(self._mark_children())

    @staticmethod
    def _node_error(node: DocumentNode, message: str):
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
                f"{node.display_name} {attribute_name} must be a number of seconds",
            ) from exc
        if not allow_negative and seconds < 0:
            raise cls._node_error(
                node,
                f"{node.display_name} {attribute_name} must be non-negative seconds",
            )
        return seconds

    @classmethod
    def _number_attribute(
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
            number = float(normalized)
        except ValueError as exc:
            raise cls._node_error(
                node,
                f"{node.display_name} {attribute_name} must be a number",
            ) from exc
        if not allow_negative and number < 0:
            raise cls._node_error(
                node,
                f"{node.display_name} {attribute_name} must be non-negative",
            )
        return number

    @classmethod
    def _preset_attribute(cls, node: DocumentNode | None) -> str | None:
        return cls._mark_attribute(node, "preset")

    @classmethod
    def _mark_attribute(cls, node: DocumentNode | None, attribute_name: str) -> str | None:
        if node is None:
            return None
        raw_value = node.attributes.get(attribute_name)
        if raw_value is None:
            return None
        normalized = raw_value.strip()
        if not normalized:
            raise cls._node_error(node, f"{node.display_name} {attribute_name} attribute cannot be empty")
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
                f"{node.display_name} {attribute_name} must be a valid expression: {exc}",
            ) from exc
        return normalized

    @classmethod
    def _boolean_attribute(cls, node: DocumentNode | None, attribute_name: str) -> bool | None:
        if node is None or attribute_name not in node.attributes:
            return None
        return node.boolean_attribute(attribute_name)

    @classmethod
    def _choice_attribute(
        cls,
        node: DocumentNode | None,
        attribute_name: str,
        choices: tuple[str, ...],
    ) -> str | None:
        if node is None:
            return None
        raw_value = node.attributes.get(attribute_name)
        if raw_value is None:
            return None
        normalized = raw_value.strip().lower()
        if not normalized:
            raise cls._node_error(node, f"{node.display_name} {attribute_name} cannot be empty")
        if normalized not in choices:
            formatted_choices = ", ".join(repr(choice) for choice in choices)
            raise cls._node_error(
                node,
                f"{node.display_name} {attribute_name} must be one of {formatted_choices}",
            )
        return normalized

    @staticmethod
    def _attr_expression_text(value: AudioAttrValue | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, float):
            return str(value)
        return str(value)

    def _rebuild_audio_marks(self, audio_plans: Sequence[AudioPlan]) -> None:
        self.audio_marks.clear()
        self._audio_mark_counts.clear()
        for audio_mark in self._local_audio_marks:
            self._add_visible_audio_mark(audio_mark)
        for audio_plan in audio_plans:
            for audio_mark in audio_plan.audio_marks:
                self._add_visible_audio_mark(audio_mark)

    def _mark_children(self) -> Sequence[AudioPlan]:
        return ()

    @property
    def explicit_start(self) -> bool:
        return self.start_expression is not None

    def left_side_variables(self) -> dict[str, float]:
        variables = {"natural_length": self._raw_inner_last - self._raw_inner_first}
        for mark_id, position in self.audio_marks_inner.items():
            variables[f"inner_{mark_id}"] = position
        return variables

    def start_variables(
        self,
        *,
        outer_marks: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        variables: dict[str, float] = {}
        if outer_marks is not None:
            for mark_id, position in outer_marks.items():
                variables[f"outer_{mark_id}"] = position
        return variables

    def right_side_variables(
        self,
        *,
        outer_marks: Mapping[str, float] | None = None,
        explicit_start: bool,
    ) -> dict[str, float]:
        variables = self.left_side_variables()
        variables["pre_gap"] = self.pre_gap
        if explicit_start:
            if outer_marks is not None:
                for mark_id, position in outer_marks.items():
                    variables[f"outer_{mark_id}"] = position
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
        self._layout_complete = True
        self._content_start = self._raw_inner_first
        self._content_end = self._raw_inner_last
        self.inner_first = self._raw_inner_first
        self.inner_last = self._raw_inner_last
        self.length = self._raw_length
        self.end = self.length
        self.start = 0.0
        self.audio_marks_inner = self._resolved_layout_marks_inner()
        self._rebuild_render_marks()

    def _resolved_layout_marks_inner(self) -> dict[str, float]:
        marks = dict(self._layout_marks_inner)
        if self.first_mark is not None:
            if self.first_mark in marks:
                marks.pop(self.first_mark, None)
            else:
                marks[self.first_mark] = self.inner_first
        if self.last_mark is not None:
            if self.last_mark in marks:
                marks.pop(self.last_mark, None)
            else:
                marks[self.last_mark] = self.inner_last
        return marks

    def incoming_marks(self, incoming_marks: Mapping[str, float] | None = None) -> None:
        self._incoming_marks_inner = {
            mark_id: position - self.start
            for mark_id, position in (incoming_marks or {}).items()
        }
        self._rebuild_render_marks()

    def _rebuild_render_marks(self) -> None:
        render_marks = dict(self.audio_marks_inner)
        for mark_id, position in self._incoming_marks_inner.items():
            render_marks.setdefault(mark_id, position)
        self.audio_marks_render = {
            mark_id: float(self._seconds_to_frames(position - self.inner_first))
            for mark_id, position in render_marks.items()
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
        self._layout_complete = False
        self._layout_marks_inner: dict[str, float] = {}
        self._raw_inner_first = 0.0
        self._raw_inner_last = 0.0
        self._raw_length = 0.0
        self._content_start = 0.0
        self._content_end = 0.0
        await self.layout_node()
        self._finalize_intrinsic_layout()

    async def _render_audio(self) -> RenderResult:
        await self.layout()
        self._rebuild_render_marks()
        return await self.post_render(await self.render_node())


@inject(config=ProductionConfig)
class MarkPlan(AudioPlan):
    """Zero-length plan that introduces one named cut mark."""

    def __init__(self, node: MarkNode, id: str, **kwargs) -> None:
        super().__init__(node=node, **kwargs)
        self.id = id
        self._local_audio_marks = [id]
        self._rebuild_audio_marks(())

    def __repr__(self) -> str:
        return f"MarkPlan(id={self.id!r})"

    async def layout_node(self) -> None:
        self._layout_marks_inner = {self.id: 0.0}
        self._raw_length = 0.0

    async def render_node(self) -> RenderResult:
        return RenderResult.empty(channels=self.config.resolved_output_channels)


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
        self._rebuild_audio_marks(self.audio_plans)

    def __repr__(self) -> str:
        return f"ComposeAudioPlan(children={len(self.audio_plans)})"

    def _mark_children(self) -> Sequence[AudioPlan]:
        return getattr(self, "audio_plans", ())

    def child_plans(self) -> Iterable[PlanningNode]:
        return self.audio_plans

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
        if audio_mark == self.first_mark:
            return
        if audio_mark == self.last_mark:
            self.audio_plans = []
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
        if audio_mark == self.last_mark:
            return
        if audio_mark == self.first_mark:
            self.audio_plans = []
            self._rebuild_audio_marks(self.audio_plans)
            return
        raise ValueError(f"Unknown or ambiguous audio mark {audio_mark!r}")

    async def layout_node(self) -> None:
        running_outer_marks: dict[str, float] = {}
        running_outer_mark_counts: dict[str, int] = {}
        cursor = 0.0
        automatic_children = [audio_plan for audio_plan in self.audio_plans if not audio_plan.explicit_start]
        explicit_children = [audio_plan for audio_plan in self.audio_plans if audio_plan.explicit_start]

        await asyncio.gather(*(audio_plan.layout() for audio_plan in automatic_children))

        for audio_plan in automatic_children:
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
            audio_plan.pre_gap = 0.0
            audio_plan.start = audio_plan.evaluate_expression(
                audio_plan.start_expression,
                audio_plan.start_variables(
                    outer_marks=running_outer_marks,
                ),
                attribute_name="start",
            )
            audio_plan.incoming_marks(running_outer_marks)

        await asyncio.gather(*(audio_plan.layout() for audio_plan in explicit_children))

        for audio_plan in explicit_children:
            audio_plan.start = audio_plan.evaluate_expression(
                audio_plan.start_expression,
                audio_plan.start_variables(
                    outer_marks=running_outer_marks,
                ),
                attribute_name="start",
            )
            audio_plan.incoming_marks(running_outer_marks)
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
        from .effects import EffectMixer

        for audio_plan in self.audio_plans:
            audio_plan.incoming_marks(self.audio_marks_inner)
        rendered_results = await asyncio.gather(
            *(audio_plan.render() for audio_plan in self.audio_plans)
        )
        total_frames = max(0, self._seconds_to_frames(self._raw_inner_last - self._raw_inner_first))
        mixer = EffectMixer(
            total_frames=total_frames,
            channels=self.config.resolved_output_channels,
        )
        for audio_plan, result in zip(self.audio_plans, rendered_results, strict=True):
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
                    f"{self._frames_to_seconds(end_frame):.3f}s "
                    f"on preset bus {audio_plan.preset_key!r}"
                ),
            )
            mixer.add(
                start_frame=start_frame,
                end_frame=end_frame,
                audio=result.audio,
                preset_key=audio_plan.output_preset_key() or self.preset_key,
            )
        return RenderResult(
            audio=await mixer.apply(sample_rate=self.config.resolved_output_sample_rate)
        )

    def output_preset_key(self) -> tuple[str, ...]:
        return ()

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
                audio_plan.left_side_variables(),
                attribute_name="pre_gap",
            )
        audio_plan.pre_gap = pre_gap
        audio_plan.start = cursor + pre_gap
        self._resolve_child_right_side(audio_plan, outer_marks)
        return audio_plan.end

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
            if resolved_length < 0:
                raise audio_plan.document_error(
                    f"{audio_plan.node.display_name} length must be non-negative seconds"
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


@inject(config=ProductionConfig)
class LoopPlan(AudioPlan):
    """Audio plan wrapper that repeats one interval of an inner plan."""

    def __init__(
        self,
        node: DocumentNode,
        audio_plan: AudioPlan,
        **kwargs,
    ) -> None:
        super().__init__(node=node, **kwargs)
        self.audio_plan = audio_plan
        self._rebuild_audio_marks((self.audio_plan,))
        self.resolved_loop_beg = 0.0
        self.resolved_loop_end = 0.0
        self.resolved_loop_stop = 0.0
        self.resolved_loop_silence = 0.0

    def __repr__(self) -> str:
        return f"LoopPlan(audio_plan={self.audio_plan!r})"

    def _mark_children(self) -> Sequence[AudioPlan]:
        return (self.__dict__["audio_plan"],) if "audio_plan" in self.__dict__ else ()

    def child_plans(self) -> Iterable[PlanningNode]:
        return (self.audio_plan,)

    def __getattr__(self, name: str):
        return getattr(self.audio_plan, name)

    async def async_resolve(self):
        return self

    def incoming_marks(self, incoming_marks: Mapping[str, float] | None = None) -> None:
        super().incoming_marks(incoming_marks)
        if not self._layout_complete or self.loop_until_expression is None:
            return
        self.resolved_loop_stop = self._resolve_loop_stop()
        if self.resolved_loop_stop + self._loop_epsilon() < self.resolved_loop_end:
            raise self.document_error(
                f"{self.node.display_name} loop_until must be greater than or equal to loop_end"
            )
        self._rebuild_loop_layout()

    async def layout_node(self) -> None:
        await self.audio_plan.layout()
        self.resolved_loop_beg = self._resolve_wrapped_loop_expression(
            self.loop_beg_expression,
            default=0.0,
            attribute_name="loop_beg",
        )
        self.resolved_loop_end = self._resolve_wrapped_loop_expression(
            self.loop_end_expression,
            default=self.audio_plan.inner_last,
            attribute_name="loop_end",
        )
        if self.resolved_loop_beg >= self.resolved_loop_end:
            raise self.document_error(
                f"{self.node.display_name} loop_beg must be less than loop_end"
            )
        self.resolved_loop_silence = self.loop_silence
        self.resolved_loop_stop = self._resolve_loop_stop()
        if self.resolved_loop_stop + self._loop_epsilon() < self.resolved_loop_end:
            raise self.document_error(
                f"{self.node.display_name} loop_until must be greater than or equal to loop_end"
            )
        self._rebuild_loop_layout()

    async def render_node(self) -> RenderResult:
        base_result = await self.audio_plan.render()
        segments: list[RenderResult] = []
        if self.resolved_loop_beg > self._raw_inner_first:
            segments.append(
                self._render_wrapped_interval(
                    base_result,
                    self._raw_inner_first,
                    self.resolved_loop_beg,
                )
            )
        segments.append(
            self._render_wrapped_interval(
                base_result,
                self.resolved_loop_beg,
                self.resolved_loop_end,
            )
        )
        remaining_repeat_frames = max(
            0,
            self._seconds_to_frames(self.resolved_loop_stop - self.resolved_loop_end),
        )
        silence_frames = self._seconds_to_frames(self.resolved_loop_silence)
        loop_body_frames = self._seconds_to_frames(self.resolved_loop_end - self.resolved_loop_beg)
        while remaining_repeat_frames > 0:
            silence_chunk_frames = min(silence_frames, remaining_repeat_frames)
            if silence_chunk_frames > 0:
                segments.append(self._silent_result(self._frames_to_seconds(silence_chunk_frames)))
                remaining_repeat_frames -= silence_chunk_frames
            segment_frames = min(loop_body_frames, remaining_repeat_frames)
            if segment_frames > 0:
                segments.append(
                    self._render_wrapped_interval(
                        base_result,
                        self.resolved_loop_beg,
                        self.resolved_loop_beg + self._frames_to_seconds(segment_frames),
                    )
                )
                remaining_repeat_frames -= segment_frames
            if silence_chunk_frames == 0 and segment_frames == 0:
                break
        if self.loop_outro and self.audio_plan.inner_last > self.resolved_loop_end:
            segments.append(
                self._render_wrapped_interval(
                    base_result,
                    self.resolved_loop_end,
                    self.audio_plan.inner_last,
                )
            )
        return RenderResult.concatenate(segments)

    def _resolve_wrapped_loop_expression(
        self,
        expression: str | None,
        *,
        default: float,
        attribute_name: str,
    ) -> float:
        if expression is None:
            return default
        return self.audio_plan.evaluate_expression(
            expression,
            self.audio_plan.left_side_variables(),
            attribute_name=attribute_name,
        )

    def _resolve_loop_stop(self) -> float:
        cycle_frames = self._seconds_to_frames(
            (self.resolved_loop_end - self.resolved_loop_beg) + self.resolved_loop_silence
        )
        if cycle_frames <= 0:
            raise self.document_error(
                f"{self.node.display_name} loop cycle must be longer than zero"
            )
        if self.loop_loops is not None:
            repeat_frames = int(round(self.loop_loops * cycle_frames))
            return self.resolved_loop_end + self._frames_to_seconds(repeat_frames)

        base_marks = self._loop_expression_marks()
        variables = {"natural_length": self.audio_plan.natural_length}
        for mark_id, position in base_marks.items():
            variables[f"inner_{mark_id}"] = position
        raw_stop = self.evaluate_expression(
            cast(str, self.loop_until_expression),
            variables,
            attribute_name="loop_until",
        )
        extra_frames = self._seconds_to_frames(max(0.0, raw_stop - self.resolved_loop_end))
        if self.loop_whole == "extend":
            extra_frames = ((extra_frames + cycle_frames - 1) // cycle_frames) * cycle_frames
        elif self.loop_whole == "shorten":
            extra_frames = (extra_frames // cycle_frames) * cycle_frames
        return self.resolved_loop_end + self._frames_to_seconds(extra_frames)

    def _loop_expression_marks(self) -> dict[str, float]:
        marks = dict(self.audio_plan.audio_marks_inner)
        for mark_id, position in self._incoming_marks_inner.items():
            marks.setdefault(mark_id, position)
        return marks

    def _loop_layout_marks_inner(self) -> dict[str, float]:
        marks: dict[str, float] = {}
        epsilon = self._loop_epsilon()
        for mark_id, position in self.audio_plan.audio_marks_inner.items():
            if position < self.resolved_loop_beg - epsilon:
                marks[mark_id] = position
                continue
            if (
                abs(position - self.resolved_loop_beg) <= epsilon
                or abs(position - self.resolved_loop_end) <= epsilon
            ):
                marks.setdefault(mark_id, position)
        if self.loop_outro:
            outro_offset = self.resolved_loop_stop - self.resolved_loop_end
            for mark_id, position in self.audio_plan.audio_marks_inner.items():
                if position > self.resolved_loop_end + epsilon:
                    marks.setdefault(mark_id, outro_offset + position)
        return marks

    def _rebuild_loop_layout(self) -> None:
        self._raw_inner_first = min(self.audio_plan.inner_first, self.resolved_loop_beg)
        outro_duration = (
            max(0.0, self.audio_plan.inner_last - self.resolved_loop_end)
            if self.loop_outro
            else 0.0
        )
        self._raw_inner_last = self.resolved_loop_stop + outro_duration
        self._raw_length = self._raw_inner_last
        self._layout_marks_inner = self._loop_layout_marks_inner()
        self._content_start = self._raw_inner_first
        self._content_end = self._raw_inner_last
        self.inner_first = self._raw_inner_first
        self.inner_last = self._raw_inner_last
        self.length = self._raw_length
        self.audio_marks_inner = self._resolved_layout_marks_inner()
        self._rebuild_render_marks()

    def _render_wrapped_interval(
        self,
        base_result: RenderResult,
        start_time: float,
        end_time: float,
    ) -> RenderResult:
        duration = max(0.0, end_time - start_time)
        frame_count = self._seconds_to_frames(duration)
        if frame_count == 0:
            return RenderResult.empty(channels=self.config.resolved_output_channels)
        audio = self._empty_audio(frame_count)
        overlap_start = max(start_time, self.audio_plan.inner_first)
        overlap_end = min(end_time, self.audio_plan.inner_last)
        if overlap_end <= overlap_start:
            return RenderResult(audio=audio)
        source_start = self._seconds_to_frames(overlap_start - self.audio_plan.inner_first)
        source_end = self._seconds_to_frames(overlap_end - self.audio_plan.inner_first)
        write_start = self._seconds_to_frames(overlap_start - start_time)
        write_end = write_start + max(0, source_end - source_start)
        audio[write_start:write_end] = base_result.audio[source_start:source_end]
        return RenderResult(audio=audio)

    def _silent_result(self, duration: float) -> RenderResult:
        return RenderResult(audio=self._empty_audio(self._seconds_to_frames(duration)))

    def _loop_epsilon(self) -> float:
        return 1e-9


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


__all__ = [
    "AudioPlan",
    "ComposeAudioPlan",
    "LoopPlan",
    "MarkPlan",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "SlicePlan",
    "convert_audio_format",
    "convert_channel_count",
    "normalize_audio_array",
    "resample_audio",
]
