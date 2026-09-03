from __future__ import annotations

import asyncio
import os
from pathlib import Path

import yaml
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
    *,
    podcast: bool = False,
) -> None:
    """Write a rendered production in the format selected by its output suffix."""

    from .freesound import credits_for_plan, markdown_credits
    from .frontmatter import write_audio_file

    output = Path(output_path)
    sound_credits = ""
    if output.suffix.lower() != ".wav" or podcast:
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
    if podcast:
        await asyncio.to_thread(
            write_podcast_sidecar,
            output,
            result,
            plan.config.resolved_output_sample_rate,
            plan.node.frontmatter,
            sound_credits=sound_credits,
        )


def write_podcast_sidecar(
    audio_path: str | Path,
    result: ProductionResult,
    sample_rate: int,
    frontmatter,
    *,
    sound_credits: str = "",
) -> Path:
    """Write the staticsite episode page paired with one rendered audio file."""

    from .frontmatter import credits_comment

    if frontmatter.guid is None:
        raise ValueError("Podcast output requires front matter guid")
    audio_path = Path(audio_path)
    sidecar_path = audio_path.with_suffix(".md")
    notes = credits_comment(frontmatter.credits, sound_credits=sound_credits)
    if frontmatter.description:
        notes = "\n\n".join(part for part in (frontmatter.description, notes) if part)
    metadata = {
        "title": frontmatter.title,
        "series": frontmatter.series,
        "episode": frontmatter.episode,
        "season": frontmatter.season,
        "date": frontmatter.date,
        "description": notes or None,
        "podcast_audio": audio_path.name,
        "podcast_duration": _podcast_duration(result.frame_count, sample_rate),
        "podcast_guid": frontmatter.guid,
        "copyright": frontmatter.copyright,
        "image": (
            Path(os.path.relpath(frontmatter.artwork, sidecar_path.parent)).as_posix()
            if frontmatter.artwork is not None
            else None
        ),
        "syndicated": True,
    }
    metadata = {name: value for name, value in metadata.items() if value is not None}
    yaml_text = yaml.safe_dump(
        metadata,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).rstrip()
    body = notes + "\n" if notes else ""
    sidecar_path.write_text(f"---\n{yaml_text}\n---\n\n{body}", encoding="utf-8")
    return sidecar_path


def _podcast_duration(frame_count: int, sample_rate: int) -> str:
    """Return a whole-second podcast duration as HH:MM:SS."""

    total_seconds = (frame_count + sample_rate - 1) // sample_rate
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


__all__ = ["ProductionPlan", "write_podcast_sidecar", "write_production"]
