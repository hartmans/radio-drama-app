"""Shared line-oriented request assembly for container TTS engines."""

from __future__ import annotations

import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .server import artifact_name


@dataclass(frozen=True, slots=True)
class LineWork:
    """One synthesizable dialogue line and its eventual segment path."""

    request_index: int
    line_index: int
    line: Mapping[str, Any]
    path: Path


def line_is_synthesizable(handling: str, spoken_text: str) -> bool:
    """Whether a line-oriented engine should synthesize one authored line.

    Cross-line engines keep ignored guidance lines; line-oriented engines do
    not speak them, and empty lines have nothing to say. The predicate is
    shared by host backends (for example Qwen) and container engines so the
    kept-line set and its timing fill stay consistent.
    """
    return handling != "ignore" and bool(spoken_text.strip())


def _line_is_synthesizable(content: Mapping[str, Any]) -> bool:
    """Synthesizability for one serialized dialogue content entry."""
    if content.get("type") != "line":
        return False
    return line_is_synthesizable(
        str(content.get("handling", "normal")), str(content.get("spoken_text", ""))
    )


def _line_kept_mask(request: Mapping[str, Any]) -> list[bool]:
    """Parallel to the request's authored lines: True where one is synthesized."""
    return [
        _line_is_synthesizable(content)
        for content in request["dialogue_contents"]
        if content.get("type") == "line"
    ]


def fill_ignored_line_spans(
    kept_spans: Sequence[tuple[float, float]], is_ignored: Sequence[bool]
) -> list[tuple[float, float]]:
    """Expand per-synthesized-line spans back to one span per authored line.

    ``kept_spans`` holds the real spans in synthesized order. Every authored
    line that was not synthesized (ignored guidance, or empty) receives a
    zero-width span at its boundary so downstream timing keeps one span per
    dialogue line. Synthesis is contiguous, so a skipped line's position is
    the end of the preceding kept span, equal to the start of the following
    one; leading skips fall to 0.0.
    """
    full: list[tuple[float, float] | None] = [None] * len(is_ignored)
    kept_index = 0
    for index, ignored in enumerate(is_ignored):
        if not ignored:
            full[index] = tuple(kept_spans[kept_index])
            kept_index += 1
    for index, ignored in enumerate(is_ignored):
        if not ignored:
            continue
        position = 0.0
        found = False
        for previous in range(index - 1, -1, -1):
            if full[previous] is not None:
                position = full[previous][1]
                found = True
                break
        if not found:
            for following in range(index + 1, len(full)):
                if full[following] is not None:
                    position = full[following][0]
                    break
        full[index] = (position, position)
    return [span for span in full if span is not None]


def prepare_line_work(
    requests: Sequence[Mapping[str, Any]],
) -> tuple[list[Path], list[LineWork]]:
    """Flatten synthesizable dialogue lines while retaining request ordering."""

    outputs = [Path(artifact_name(request)) for request in requests]
    work = []
    for request_index, (request, output) in enumerate(zip(requests, outputs, strict=True)):
        line_index = 0
        for content in request["dialogue_contents"]:
            if not _line_is_synthesizable(content):
                continue
            work.append(
                LineWork(
                    request_index=request_index,
                    line_index=line_index,
                    line=content,
                    path=output.with_suffix(f".line-{line_index}.wav"),
                )
            )
            line_index += 1
    return outputs, work


def finish_line_work(
    outputs: Sequence[Path],
    work: Sequence[LineWork],
    requests: Sequence[Mapping[str, Any]],
    *,
    sample_rate: int,
) -> list[Mapping[str, Any]]:
    """Concatenate like-formatted segments, one span per authored line.

    Skipped (ignored or empty) lines receive zero-width spans at their
    boundary so the reported spans stay parallel to the request's authored
    dialogue lines.
    """

    by_request: list[list[LineWork]] = [[] for _ in outputs]
    for item in work:
        by_request[item.request_index].append(item)
    results = []
    for request_index, (output, request_work) in enumerate(
        zip(outputs, by_request, strict=True)
    ):
        starts = []
        frames = 0
        chunks = []
        expected = None
        for item in request_work:
            with wave.open(str(item.path), "rb") as source:
                params = (
                    source.getnchannels(), source.getsampwidth(),
                    source.getframerate(), source.getcomptype(),
                )
                if expected is None:
                    expected = params
                elif params != expected:
                    raise RuntimeError("TTS engine returned incompatible WAV segments")
                if params[2] != sample_rate or params[3] != "NONE":
                    raise RuntimeError("TTS engine returned an unsupported WAV format")
                starts.append(frames / sample_rate)
                frames += source.getnframes()
                chunks.append(source.readframes(source.getnframes()))
        channels, width, rate, _ = expected or (1, 2, sample_rate, "NONE")
        with wave.open(str(output), "wb") as destination:
            destination.setnchannels(channels)
            destination.setsampwidth(width)
            destination.setframerate(rate)
            destination.writeframes(b"".join(chunks))
        if starts:
            ends = [*starts[1:], frames / sample_rate]
            kept_spans = list(zip(starts, ends, strict=True))
        else:
            kept_spans = []
        is_ignored = [not kept for kept in _line_kept_mask(requests[request_index])]
        results.append(
            {
                "wav": output.name,
                "dialogue_line_spans": [
                    list(span)
                    for span in fill_ignored_line_spans(kept_spans, is_ignored)
                ],
            }
        )
    return results


def remove_line_work(work: Sequence[LineWork]) -> None:
    """Remove intermediate segment artifacts after successful assembly."""

    for item in work:
        item.path.unlink(missing_ok=True)
