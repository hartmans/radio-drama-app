from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
import wave
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO


PROTOCOL = "radio-drama-tts"
PROTOCOL_VERSION = 1
RenderBatch = Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]]


def artifact_name(request: Mapping[str, Any], suffix: str = ".wav") -> str:
    """Return a deterministic, cache-relative artifact name for a request."""

    label = re.sub(r"[^A-Za-z0-9]+", "_", str(request.get("first_words", "audio")))
    label = label.strip("_").lower()[:40] or "audio"
    encoded = json.dumps(request, sort_keys=True, ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"proxy_{label}_{digest}{suffix}"


def write_pcm16_wav(
    path: str | Path,
    samples: Iterable[float],
    *,
    sample_rate: int,
    channels: int = 1,
) -> None:
    """Write normalized interleaved floating-point samples using only stdlib."""

    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        frames = bytearray()
        for sample in samples:
            bounded = max(-1.0, min(1.0, float(sample)))
            frames.extend(struct.pack("<h", round(bounded * 32767)))
        output.writeframes(frames)


def run_server(
    render_batch: RenderBatch,
    *,
    capabilities: Iterable[str] = (),
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
    """Serve the proxy JSON-lines protocol until stdin reaches EOF.

    Engine entry points supply only ``render_batch``. It receives the request
    mappings and returns one result mapping per request. Diagnostics must be
    written to stderr because stdout is reserved for protocol responses.
    """

    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    handshake_line = input_stream.readline()
    if not handshake_line:
        return
    handshake = json.loads(handshake_line)
    if handshake.get("protocol") != PROTOCOL or PROTOCOL_VERSION not in handshake.get(
        "versions", ()
    ):
        raise RuntimeError("Host does not support this TTS proxy protocol")
    _write_json(
        output_stream,
        {
            "protocol": PROTOCOL,
            "version": PROTOCOL_VERSION,
            "ready": True,
            "capabilities": sorted(set(capabilities)),
        },
    )
    for line in input_stream:
        if not line.strip():
            continue
        message = json.loads(line)
        response: dict[str, Any] = {"id": message.get("id")}
        try:
            if message.get("protocol") != PROTOCOL:
                raise ValueError("Incorrect protocol name")
            if message.get("version") != PROTOCOL_VERSION:
                raise ValueError("Unsupported protocol version")
            if message.get("method") != "render_batch":
                raise ValueError("Unsupported proxy method")
            requests = message["requests"]
            results = list(render_batch(requests))
            if len(results) != len(requests):
                raise ValueError("Engine returned the wrong number of results")
            response["results"] = results
        except Exception as exc:
            response["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(output_stream, response)


def _write_json(stream: TextIO, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n")
    stream.flush()
