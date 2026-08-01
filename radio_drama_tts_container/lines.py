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


def prepare_line_work(
    requests: Sequence[Mapping[str, Any]],
) -> tuple[list[Path], list[LineWork]]:
    """Flatten non-empty dialogue lines while retaining request ordering."""

    outputs = [Path(artifact_name(request)) for request in requests]
    work = []
    for request_index, (request, output) in enumerate(zip(requests, outputs, strict=True)):
        line_index = 0
        for content in request["dialogue_contents"]:
            if content.get("type") != "line" or not str(content.get("spoken_text", "")).strip():
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
    outputs: Sequence[Path], work: Sequence[LineWork], *, sample_rate: int
) -> list[Mapping[str, Any]]:
    """Concatenate like-formatted segments and report authored line starts."""

    by_request: list[list[LineWork]] = [[] for _ in outputs]
    for item in work:
        by_request[item.request_index].append(item)
    results = []
    for output, request_work in zip(outputs, by_request, strict=True):
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
        results.append({"wav": output.name, "dialogue_line_start_positions": starts})
    return results


def remove_line_work(work: Sequence[LineWork]) -> None:
    """Remove intermediate segment artifacts after successful assembly."""

    for item in work:
        item.path.unlink(missing_ok=True)
