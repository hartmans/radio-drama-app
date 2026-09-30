from __future__ import annotations

import asyncio
from collections.abc import Iterable
import logging
import math
from dataclasses import dataclass
from typing import Sequence

from carthage.dependency_injection import inject

from ..audio import AudioPlan, ComposeAudioPlan
from ..config import ProductionConfig
from ..debug import write_debug_message
from ..dialogue import DialogueLine, ScriptEvent
from ..planning import PlanningNode
from ..rendering import RenderResult, ScriptRenderResult, ScriptTiming


from .base import ForcedAlignmentResource
from .projection import (copy_dialogue_contents, fill_start_positions_from_timing,
    _debug_line_preview, _marker_frames_from_contents, _boundary_info_for_marker)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AlignedScriptResult:
    """Rendered dry script audio plus marker frames for inline insertions."""

    render_result: RenderResult
    marker_frames: tuple[int, ...]
    contents: tuple[ScriptEvent, ...]
    timing: ScriptTiming


@inject(config=ProductionConfig)
class AlignedScriptSource(PlanningNode):
    """One audio provider and its source-local alignment event projection."""

    def __init__(
        self,
        node,
        audio_provider: PlanningNode,
        contents: Sequence[ScriptEvent],
        **kwargs,
    ) -> None:
        super().__init__(node=node, **kwargs)
        self.audio_provider = audio_provider
        self.contents: list[ScriptEvent] = copy_dialogue_contents(contents)

    def child_plans(self) -> Iterable[PlanningNode]:
        return (self.audio_provider,)

    async def render_node(self) -> AlignedScriptResult:
        if isinstance(self.audio_provider, AudioPlan):
            await self.audio_provider.layout()
        base_result = await self.audio_provider.render()
        from ..dialogue import ScriptPlan

        if isinstance(self.audio_provider, ScriptPlan):
            timing = await self.audio_provider.ensure_timing(self.contents, base_result)
        elif isinstance(base_result, ScriptRenderResult) and base_result.timing is not None:
            timing = base_result.timing
        else:
            resource = await self.ainjector.get_instance_async(ForcedAlignmentResource)
            timing = await resource.script_timing(
                self.contents, base_result,
                sample_rate=self.config.resolved_output_sample_rate,
                transcript_kind="partial",
            )
        self.contents = fill_start_positions_from_timing(self.contents, timing)
        for content in self.contents:
            if not isinstance(content, DialogueLine):
                continue
            preview = _debug_line_preview(content.spoken_text)
            write_debug_message(
                self.config,
                "forced_alignment",
                f"{content.start_pos:.3f}s {preview}",
            )
        return AlignedScriptResult(
            render_result=base_result,
            marker_frames=_marker_frames_from_contents(
                self.contents,
                frame_count=base_result.frame_count,
                sample_rate=self.config.resolved_output_sample_rate,
            ),
            contents=tuple(self.contents),
            timing=timing,
        )


@inject(config=ProductionConfig)
class ScriptSlice(AudioPlan):
    """Audio plan that slices an aligned script by marker index."""

    def __init__(
        self,
        aligned_script_source: AlignedScriptSource,
        *,
        start_marker: int,
        end_marker: int,
        name: str | None = None,
        speaker_effect_expression: str | None = None,
        node=None,
        **kwargs,
    ) -> None:
        super().__init__(node=node, **kwargs)
        self.aligned_script_source = aligned_script_source
        self.start_marker = start_marker
        self.end_marker = end_marker
        self.name = name
        self.speaker_effect_expression = speaker_effect_expression

    def child_plans(self) -> Iterable[PlanningNode]:
        return (self.aligned_script_source,)

    def __repr__(self) -> str:
        if self.name is not None:
            return f"ScriptSlice(name={self.name!r})"
        return (
            "ScriptSlice("
            f"start_marker={self.start_marker}, "
            f"end_marker={self.end_marker})"
        )

    async def async_resolve(self):
        """Bypass alignment when this slice selects its provider's whole output."""
        if (
            self.start_marker != 0
            or self.end_marker != len(self.aligned_script_source.contents)
            or not isinstance(self.aligned_script_source.audio_provider, AudioPlan)
        ):
            return await super().async_resolve()
        provider = self.aligned_script_source.audio_provider
        if not self.attrs and self.speaker_effect_expression is None:
            return provider
        if not self.attrs:
            return self
        return await self.ainjector(
            ComposeAudioPlan,
            node=self.node,
            audio_plans=[provider],
            attrs=self.attrs,
        )

    async def layout_node(self) -> None:
        aligned_result = await self.aligned_script_source.render()
        self._log_nan_marker_if_used(aligned_result, self.start_marker, marker_name="start")
        self._log_nan_marker_if_used(aligned_result, self.end_marker, marker_name="end")
        start_frame = aligned_result.marker_frames[self.start_marker]
        end_frame = max(start_frame, aligned_result.marker_frames[self.end_marker])
        self.inner_last = self._frames_to_seconds(end_frame - start_frame)
        self.advance = self.inner_last

    async def render_node(self) -> RenderResult:
        aligned_result = await self.aligned_script_source.render()
        start_frame = aligned_result.marker_frames[self.start_marker]
        end_frame = aligned_result.marker_frames[self.end_marker]
        end_frame = max(start_frame, end_frame)
        result = aligned_result.render_result.slice_frames(
            start_frame,
            end_frame,
        )
        if self.speaker_effect_expression is not None and result.frame_count:
            from ..effects import EffectChainRegistry, effect_chain, effect_chain_variables
            from ..expressions import eval_expression

            effect_chains = self.ainjector.injector.get_instance(EffectChainRegistry)
            stage = eval_expression(
                self.speaker_effect_expression,
                effect_chain_variables(effect_chains.stages()),
                effect_chain,
            )
            await asyncio.to_thread(
                stage.apply,
                result.audio,
                sample_rate=self.config.resolved_output_sample_rate,
            )
        return result

    def _log_nan_marker_if_used(
        self,
        aligned_result: AlignedScriptResult,
        marker_index: int,
        *,
        marker_name: str,
    ) -> None:
        boundary = _boundary_info_for_marker(aligned_result.contents, marker_index)
        if boundary is None:
            return
        boundary_pos, boundary_label = boundary
        if not math.isnan(boundary_pos):
            return
        logger.error(
            "Alignment produced NaN %s marker for %r at marker %d (%s); slice timing will fall back to 0",
            marker_name,
            self,
            marker_index,
            boundary_label,
        )

