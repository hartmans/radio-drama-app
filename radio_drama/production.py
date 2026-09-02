from __future__ import annotations

import asyncio
from pathlib import Path

from carthage.dependency_injection import inject

from .audio import ComposeAudioPlan
from .config import ProductionConfig
from .effects import EffectChainRegistry
from .rendering import ProductionResult, RenderResult


@inject(config=ProductionConfig, effect_chains=EffectChainRegistry)
class ProductionPlan(ComposeAudioPlan):
    """Top-level production plan that preserves script order."""

    async def render_node(self) -> ProductionResult:
        """Render scripts in document order and clip to the production boundary."""

        combined = await super().render_node()
        trimmed = self._trim_to_production_boundary(combined)
        if trimmed.frame_count == 0:
            return ProductionResult(audio=trimmed.audio)
        master_chain = self.effect_chains["master"]
        master_chain.apply(
            trimmed.audio,
            sample_rate=self.config.resolved_output_sample_rate,
        )
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
            padded[: audio.shape[0]] = audio
            audio = padded
        return RenderResult(audio=audio)


async def write_production(
    plan: ProductionPlan,
    result: ProductionResult,
    output_path: str | Path,
) -> None:
    """Write a rendered production in the format selected by its output suffix."""

    from .freesound import credits_for_plan, markdown_credits
    from .frontmatter import write_audio_file

    output = Path(output_path)
    sound_credits = ""
    if output.suffix.lower() != ".wav":
        credits = await credits_for_plan(plan)
        if credits:
            sound_credits = markdown_credits(credits, minimal=True)
    await asyncio.to_thread(
        write_audio_file,
        output,
        result,
        plan.config.resolved_output_sample_rate,
        plan.node.frontmatter,
        sound_credits=sound_credits,
    )


__all__ = ["ProductionPlan", "write_production"]
