from __future__ import annotations

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

from .audio import resample_audio
from .cli import build_injector_from_namespace, initialize_arg_parser
from .dialogue import DialogueAudio, DialogueLine, ScriptEvent, ScriptPlan
from .document import parse_production_file
from .errors import DocumentError
from .forced_alignment import AlignedScriptSource
from .planning import PlanningNode
from .rendering import RenderResult


@dataclass(frozen=True, slots=True)
class TrainingChunk:
    speaker_name: str
    start_marker: int
    end_marker: int
    slug: str


def _sanitize_path_component(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._")
    return sanitized or "speaker"


def _boundary_is_reliable(
    contents: list[ScriptEvent] | tuple[ScriptEvent, ...],
    marker_index: int,
) -> bool:
    if marker_index <= 0 or marker_index >= len(contents):
        return True
    previous = contents[marker_index - 1]
    current = contents[marker_index]
    boundary_pos = previous.start_pos if isinstance(previous, DialogueAudio) else current.start_pos
    return not math.isnan(boundary_pos)


def chunk_training_intervals(
    contents: list[ScriptEvent] | tuple[ScriptEvent, ...],
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
        slug = " ".join(dialogue_lines[0].spoken_text.split()).strip()
        if slug: slug = _sanitize_path_component(slug[0:40])
        chunks.append(
            TrainingChunk(
                speaker_name=dialogue_lines[0].speaker.authored_name,
                start_marker=start_marker,
                end_marker=end_marker,
                slug=slug
            )
        )
    return chunks


async def export_training_samples(
    production_path: str | Path,
    output_directory: str | Path,
    *,
    injector=None,
    sample_rate: int | None = None,
    output_path: str | Path | None = None,
) -> int:
    production_path = Path(production_path)
    output_directory = Path(output_directory)
    production_node = parse_production_file(production_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    created_injector = injector
    if created_injector is None:
        from .config import ProductionConfig
        from .init import radio_drama_injector
        from carthage.dependency_injection import AsyncInjector

        created_injector = radio_drama_injector(
            config=ProductionConfig(),
            event_loop=asyncio.get_running_loop(),
            document_path=production_path,
            output_path=Path(output_path) if output_path is not None else None,
        )
    try:
        from carthage.dependency_injection import AsyncInjector

        ainjector = created_injector(AsyncInjector)
        production_plan = await production_node.plan(ainjector)
        return await export_training_samples_from_plan(
            production_plan,
            output_directory,
            sample_rate=sample_rate,
        )
    finally:
        if injector is None:
            created_injector.close()


async def export_training_samples_from_plan(
    production_plan: PlanningNode,
    output_directory: Path,
    *,
    sample_rate: int | None,
) -> int:
    aligned_by_script: dict[int, AlignedScriptSource] = {}
    script_plans: list[ScriptPlan] = []
    for plan in production_plan.all_plans():
        if isinstance(plan, AlignedScriptSource):
            if isinstance(plan.audio_provider, ScriptPlan):
                aligned_by_script[id(plan.audio_provider)] = plan
        elif isinstance(plan, ScriptPlan):
            script_plans.append(plan)

    if sample_rate is None:
        sample_rate = _default_sample_rate(script_plans)
        production_plan.config.output_sample_rate = sample_rate
    render_sample_rate = production_plan.config.resolved_output_sample_rate
    if sample_rate is None:
        sample_rate = render_sample_rate

    counters: defaultdict[str, int] = defaultdict(int)
    written = 0
    for script_plan in script_plans:
        aligned_source = aligned_by_script.get(id(script_plan))
        if aligned_source is None:
            aligned_source = await script_plan.ainjector(
                AlignedScriptSource,
                node=script_plan.node,
                audio_provider=script_plan,
                contents=script_plan.script_events,
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
            chunk_audio = result.audio
            if sample_rate != render_sample_rate:
                chunk_audio = resample_audio(
                    chunk_audio,
                    input_sample_rate=render_sample_rate,
                    output_sample_rate=sample_rate,
                )
            counters[chunk.speaker_name] += 1
            speaker_directory = output_directory / _sanitize_path_component(chunk.speaker_name)
            speaker_directory.mkdir(parents=True, exist_ok=True)
            slug = chunk.slug
            if slug: slug += "_"
            output_path = speaker_directory / f"chunk_{slug}{counters[chunk.speaker_name]:06d}.wav"
            sf.write(output_path, chunk_audio, sample_rate)
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


def _default_sample_rate(script_plans: list[ScriptPlan]) -> int | None:
    sample_rates = {
        script_plan._registered_request.resource.sample_rate
        for script_plan in script_plans
        if script_plan._registered_request is not None
    }
    if len(sample_rates) == 1:
        return next(iter(sample_rates))
    return None


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = initialize_arg_parser(
        "Export per-speaker training WAV chunks from a production XML file.",
        output_help=(
            "Output WAV path used to locate the shared speech cache. "
            "Defaults to the input path with a .wav extension."
        ),
    )
    parser.add_argument("output_dir", help="Directory that will receive speaker subdirectories.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        async def runner() -> int:
            injector, config, production_path, output_path = build_injector_from_namespace(
                args,
                event_loop=asyncio.get_running_loop(),
            )
            try:
                return await export_training_samples(
                    production_path,
                    args.output_dir,
                    injector=injector,
                    sample_rate=config.resolved_output_sample_rate,
                    output_path=output_path,
                )
            finally:
                injector.close()

        written = asyncio.run(runner())
    except DocumentError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Wrote {written} training sample files to {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
