from __future__ import annotations

from carthage.dependency_injection import inject

from .audio import ComposeAudioPlan
from .config import ProductionConfig
from .rendering import ProductionResult, RenderResult


@inject(config=ProductionConfig)
class ProductionPlan(ComposeAudioPlan):
    """Top-level production plan that preserves script order."""

    async def render_node(self) -> ProductionResult:
        """Render scripts in document order and clip to the production boundary."""

        from .effects import build_named_effect_chain

        combined = await super().render_node()
        trimmed = self._trim_to_production_boundary(combined)
        if trimmed.frame_count == 0:
            return ProductionResult(audio=trimmed.audio)
        master_chain = build_named_effect_chain("master")
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
            padded[:audio.shape[0]] = audio
            audio = padded
        return RenderResult(audio=audio)


__all__ = ["ProductionPlan"]
