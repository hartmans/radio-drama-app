"""Radio-drama proxy adapter for the Higgs TTS 3 SGLang-Omni API."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from radio_drama_tts_container import artifact_name, run_server


MODEL = os.environ.get("HIGGS_MODEL", "bosonai/higgs-tts-3-4b")
HOST = os.environ.get("HIGGS_HOST", "127.0.0.1")
PORT = int(os.environ.get("HIGGS_PORT", "8000"))
STARTUP_TIMEOUT = float(os.environ.get("HIGGS_STARTUP_TIMEOUT", "900"))
SAMPLE_RATE = 24_000

CONTROL_TAGS = {
    "emotion": frozenset(
        {
            "affection", "amusement", "anger", "arousal", "awe",
            "bitterness", "confusion", "contemplation", "contentment",
            "determination", "disgust", "elation", "enthusiasm", "fear",
            "helplessness", "longing", "pride", "relief", "sadness",
            "shame", "surprise",
        }
    ),
    "prosody": frozenset(
        {
            "speed_very_slow", "speed_slow", "speed_fast", "speed_very_fast",
            "pitch_low", "pitch_high", "expressive_high", "expressive_low",
            "pause", "long_pause",
        }
    ),
    "style": frozenset({"singing", "shouting", "whispering"}),
    "sfx": frozenset(
        {
            "cough", "laughter", "crying", "screaming", "burping",
            "humming", "sigh", "sniff", "sneeze",
        }
    ),
}
CONTROL_EXPRESSION = re.compile(r"\[([a-z]+):([a-z_]+)\]")


def expand_control_expressions(text: str) -> str:
    """Translate recognized radio-drama brackets to Higgs control tokens."""

    def replace(match: re.Match[str]) -> str:
        category, tag = match.groups()
        if tag not in CONTROL_TAGS.get(category, ()):
            return match.group(0)
        return f"<|{category}:{tag}|>"

    return CONTROL_EXPRESSION.sub(replace, text)


class HiggsTtsEngine:
    """Map proxy dialogue requests onto Higgs voice-cloning HTTP requests."""

    def __init__(self, *, base_url: str | None = None) -> None:
        self.base_url = base_url or f"http://{HOST}:{PORT}"
        self._manage_server = base_url is None
        self._server: subprocess.Popen | None = None

    def ensure_server(self) -> None:
        """Start the in-container model server lazily after protocol handshake."""

        if self._manage_server and self._server is None:
            self._server = start_sglang_server()

    def close(self) -> None:
        if self._server is None:
            return
        self._server.terminate()
        try:
            self._server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._server.kill()
        self._server = None

    def render_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        prepared = [self._prepare_request(request) for request in requests]
        lines = [line for item in prepared for line in item["lines"]]
        try:
            if lines:
                self.ensure_server()
            batch_size = int(os.environ.get("HIGGS_BATCH_SIZE", "16"))
            if batch_size < 1:
                raise ValueError("HIGGS_BATCH_SIZE must be at least 1")
            for start in range(0, len(lines), batch_size):
                chunk = lines[start : start + batch_size]
                audio_results = self.synthesize_batch([item["line"] for item in chunk])
                for item, audio in zip(chunk, audio_results, strict=True):
                    item["path"].write_bytes(audio)
            return [self._finish_request(item) for item in prepared]
        finally:
            if not keep_line_wavs():
                for item in lines:
                    item["path"].unlink(missing_ok=True)

    def render_request(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        prepared = self._prepare_request(request)
        try:
            for item in prepared["lines"]:
                self.synthesize_line(item["line"], item["path"])
            return self._finish_request(prepared)
        finally:
            if not keep_line_wavs():
                for item in prepared["lines"]:
                    item["path"].unlink(missing_ok=True)

    def _prepare_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        output_path = Path(artifact_name(request))
        return {
            "output_path": output_path,
            "lines": [
                {
                    "line": line,
                    "path": output_path.with_suffix(f".line-{index}.wav"),
                }
                for index, line in enumerate(self.synthesized_lines(request))
            ],
        }

    def _finish_request(self, prepared: Mapping[str, Any]) -> Mapping[str, Any]:
        output_path = prepared["output_path"]
        segment_paths = [item["path"] for item in prepared["lines"]]
        line_starts: list[float] = []
        frame_count = 0
        for segment_path in segment_paths:
            segment_frames, sample_rate = wav_frame_count(segment_path)
            if sample_rate != SAMPLE_RATE:
                raise RuntimeError(
                    f"Higgs returned {sample_rate} Hz audio; expected {SAMPLE_RATE} Hz"
                )
            line_starts.append(frame_count / SAMPLE_RATE)
            frame_count += segment_frames
        concatenate_wavs(segment_paths, output_path, sample_rate=SAMPLE_RATE)
        return {
            "wav": output_path.name,
            "dialogue_line_start_positions": line_starts,
        }

    @staticmethod
    def synthesized_lines(request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        """Return non-empty dialogue lines in their authored order.

        This mirrors the existing speech-resource contract: source selection
        and slicing happen in host planning, while the base speech render keeps
        one timing entry parallel to each non-empty dialogue line. Gap events
        remain host-side alignment information.
        """

        return [
            content
            for content in request["dialogue_contents"]
            if content.get("type") == "line"
            and str(content.get("spoken_text", "")).strip()
        ]

    def synthesize_line(self, line: Mapping[str, Any], output_path: Path) -> None:
        self.ensure_server()
        payload = self._line_payload(line)
        request = urllib.request.Request(
            f"{self.base_url}/v1/audio/speech",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                output_path.write_bytes(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Higgs synthesis failed ({exc.code}): {detail}") from exc

    def synthesize_batch(
        self, lines: Sequence[Mapping[str, Any]]
    ) -> list[bytes]:
        payload: dict[str, Any] = {
            "model": MODEL,
            "temperature": float(os.environ.get("HIGGS_TEMPERATURE", "0.8")),
            "top_k": int(os.environ.get("HIGGS_TOP_K", "50")),
            "max_new_tokens": int(os.environ.get("HIGGS_MAX_NEW_TOKENS", "2048")),
            "items": [self._line_payload(line, include_defaults=False) for line in lines],
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/audio/speech/batch",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Higgs batch synthesis failed ({exc.code}): {detail}") from exc
        results = result["results"]
        if len(results) != len(lines):
            raise RuntimeError("Higgs batch response has the wrong number of results")
        audio: list[bytes] = []
        for index, item in enumerate(results):
            if item.get("index") != index:
                raise RuntimeError("Higgs batch response is out of order")
            if item.get("status") != "success":
                raise RuntimeError(f"Higgs batch item {index} failed: {item.get('error')}")
            audio.append(base64.b64decode(item["audio_data"], validate=True))
        return audio

    @staticmethod
    def _line_payload(
        line: Mapping[str, Any], *, include_defaults: bool = True
    ) -> dict[str, Any]:
        speaker = line["speaker"]
        payload: dict[str, Any] = {
            "input": expand_control_expressions(line["spoken_text"]),
            "references": [
                {
                    "audio_path": speaker["voice_path"],
                    "text": speaker["transcript"],
                }
            ],
        }
        if include_defaults:
            payload.update(
                model=MODEL,
                temperature=float(os.environ.get("HIGGS_TEMPERATURE", "0.8")),
                top_k=int(os.environ.get("HIGGS_TOP_K", "50")),
                max_new_tokens=int(os.environ.get("HIGGS_MAX_NEW_TOKENS", "2048")),
            )
        return payload


def keep_line_wavs() -> bool:
    return os.environ.get("HIGGS_KEEP_LINE_WAVS", "").lower() in {
        "1", "true", "yes", "on"
    }


def wav_frame_count(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as source:
        return source.getnframes(), source.getframerate()


def concatenate_wavs(
    inputs: Sequence[Path], output: Path, *, sample_rate: int
) -> None:
    """Concatenate like-formatted Higgs WAV responses without extra packages."""

    expected_params: tuple[int, int, int, str] | None = None
    chunks: list[bytes] = []
    for path in inputs:
        with wave.open(str(path), "rb") as source:
            params = (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
                source.getcomptype(),
            )
            if expected_params is None:
                expected_params = params
            elif params != expected_params:
                raise RuntimeError("Higgs returned incompatible WAV segment formats")
            chunks.append(source.readframes(source.getnframes()))
    channels, sample_width, actual_rate, compression = expected_params or (
        1,
        2,
        sample_rate,
        "NONE",
    )
    if actual_rate != sample_rate or compression != "NONE":
        raise RuntimeError("Higgs returned an unsupported WAV format")
    with wave.open(str(output), "wb") as destination:
        destination.setnchannels(channels)
        destination.setsampwidth(sample_width)
        destination.setframerate(actual_rate)
        destination.writeframes(b"".join(chunks))


def start_sglang_server() -> subprocess.Popen:
    command = [
        "sgl-omni",
        "serve",
        "--model-path",
        MODEL,
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]
    process = subprocess.Popen(command, stdout=sys.stderr, stderr=sys.stderr)
    deadline = time.monotonic() + STARTUP_TIMEOUT
    health_url = f"http://{HOST}:{PORT}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang-Omni exited during startup ({process.returncode})")
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status < 500:
                    return process
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                return process
            time.sleep(1)
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    process.terminate()
    raise TimeoutError(f"SGLang-Omni did not become ready within {STARTUP_TIMEOUT:g}s")


def main() -> None:
    engine = HiggsTtsEngine()
    try:
        run_server(
            engine.render_batch,
            capabilities={"needs_transcript"},
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
