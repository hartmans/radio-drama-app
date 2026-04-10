from __future__ import annotations

import argparse
import asyncio
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import soundfile as sf
from carthage.dependency_injection import AsyncInjector

from .config import ProductionConfig
from .dialogue import DialogueAudio, DialogueContents, DialogueLine, ScriptPlan
from .document import parse_production_file
from .errors import DocumentError
from .forced_alignment import AlignedScriptSource
from .init import radio_drama_injector
from .planning import PlanningNode
from .rendering import RenderResult


@dataclass(frozen=True, slots=True)
class TrainingChunk:
    speaker_name: str
    start_marker: int
    end_marker: int


def _sanitize_path_component(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._")
    return sanitized or "speaker"


def _boundary_is_reliable(
    contents: list[DialogueContents] | tuple[DialogueContents, ...],
    marker_index: int,
) -> bool:
    if marker_index <= 0 or marker_index >= len(contents):
        return True
    previous = contents[marker_index - 1]
    current = contents[marker_index]
    boundary_pos = previous.start_pos if isinstance(previous, DialogueAudio) else current.start_pos
    return not math.isnan(boundary_pos)


def chunk_training_intervals(
    contents: list[DialogueContents] | tuple[DialogueContents, ...],
) -> list[TrainingChunk]:
    reliable_markers = [0]
    reliable_markers.extend(
        marker_index
        for marker_index in range(1, len(contents))
        if _boundary_is_reliable(contents, marker_index)
    )
    reliable_markers.append(len(contents))

    chunks: list[TrainingChunk] = []
    for start_marker, end_marker in pairwise(reliable_markers):
        dialogue_lines = [
            content
            for content in contents[start_marker:end_marker]
            if isinstance(content, DialogueLine) and content.spoken_text.strip()
        ]
        if not dialogue_lines:
            continue
        speaker_names = {line.speaker.authored_name for line in dialogue_lines}
        if len(speaker_names) != 1:
            continue
        chunks.append(
            TrainingChunk(
                speaker_name=dialogue_lines[0].speaker.authored_name,
                start_marker=start_marker,
                end_marker=end_marker,
            )
        )
    return chunks


async def export_training_samples(
    production_path: str | Path,
    output_directory: str | Path,
    *,
    config: ProductionConfig | None = None,
) -> int:
    production_path = Path(production_path)
    output_directory = Path(output_directory)
    config = config or ProductionConfig()
    production_node = parse_production_file(production_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    injector = radio_drama_injector(
        config=config,
        event_loop=asyncio.get_running_loop(),
        document_path=production_path,
    )
    try:
        ainjector = injector(AsyncInjector)
        production_plan = await production_node.plan(ainjector)
        return await export_training_samples_from_plan(
            production_plan,
            output_directory,
            sample_rate=config.resolved_output_sample_rate,
        )
    finally:
        injector.close()


async def export_training_samples_from_plan(
    production_plan: PlanningNode,
    output_directory: Path,
    *,
    sample_rate: int,
) -> int:
    aligned_by_script: dict[int, AlignedScriptSource] = {}
    script_plans: list[ScriptPlan] = []
    for plan in production_plan.all_plans():
        if isinstance(plan, AlignedScriptSource):
            aligned_by_script[id(plan.script_plan)] = plan
        elif isinstance(plan, ScriptPlan):
            script_plans.append(plan)

    counters: defaultdict[str, int] = defaultdict(int)
    written = 0
    for script_plan in script_plans:
        aligned_source = aligned_by_script.get(id(script_plan))
        if aligned_source is None:
            aligned_source = await script_plan.ainjector(
                AlignedScriptSource,
                node=script_plan.node,
                script_plan=script_plan,
            )
        aligned_result = await aligned_source.render()
        for chunk in chunk_training_intervals(aligned_result.contents):
            result = _slice_training_chunk(
                aligned_result.render_result,
                chunk,
                marker_frames=aligned_result.marker_frames,
            )
            if result.frame_count == 0:
                continue
            counters[chunk.speaker_name] += 1
            speaker_directory = output_directory / _sanitize_path_component(chunk.speaker_name)
            speaker_directory.mkdir(parents=True, exist_ok=True)
            output_path = speaker_directory / f"chunk_{counters[chunk.speaker_name]:06d}.wav"
            sf.write(output_path, result.audio, sample_rate)
            written += 1
    return written


def _slice_training_chunk(
    render_result: RenderResult,
    chunk: TrainingChunk,
    *,
    marker_frames: tuple[int, ...],
) -> RenderResult:
    return render_result.slice_frames(
        marker_frames[chunk.start_marker],
        marker_frames[chunk.end_marker],
    )


def build_config(args: argparse.Namespace) -> ProductionConfig:
    return ProductionConfig(
        voice_directory=Path(args.voice_dir) if args.voice_dir is not None else None,
        sounds_directory=Path(args.sounds_dir) if args.sounds_dir is not None else None,
        model_name=args.model_file,
        output_sample_rate=args.output_sample_rate,
        output_channels=args.output_channels,
        batch_size=args.batch_size,
        device=args.device,
        cfg_scale=args.cfg_scale,
        disable_prefill=args.disable_prefill,
        ddpm_inference_steps=args.ddpm_inference_steps,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-speaker training WAV chunks from a production XML file."
    )
    parser.add_argument("file", help="Input XML file.")
    parser.add_argument("output_dir", help="Directory that will receive speaker subdirectories.")
    parser.add_argument("--voice-dir", default=None, help="Directory containing reference voice files.")
    parser.add_argument("--sounds-dir", default=None, help="Directory containing sound files for relative <sound> references.")
    parser.add_argument("--model-file", default=None, help="Path to the VibeVoice model directory.")
    parser.add_argument("--output-sample-rate", type=int, default=None, help="Output WAV sample rate override.")
    parser.add_argument("--output-channels", type=int, default=None, help="Output WAV channel count override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Maximum speech backend batch size override.")
    parser.add_argument("--device", default=None, help="Preferred torch device override.")
    parser.add_argument("--cfg-scale", type=float, default=None, help="VibeVoice cfg_scale override.")
    parser.add_argument(
        "--disable-prefill",
        action="store_const",
        const=True,
        default=None,
        help="Disable VibeVoice prefill.",
    )
    parser.add_argument(
        "--ddpm-inference-steps",
        type=int,
        default=None,
        help="VibeVoice DDPM inference steps override.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    try:
        written = asyncio.run(
            export_training_samples(
                args.file,
                args.output_dir,
                config=config,
            )
        )
    except DocumentError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Wrote {written} training sample files to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
