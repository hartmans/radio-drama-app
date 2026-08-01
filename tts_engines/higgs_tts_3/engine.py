"""Radio-drama proxy adapter for the Higgs TTS 3 SGLang-Omni API."""

from __future__ import annotations

import json
import os
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


class HiggsTtsEngine:
    """Map proxy dialogue requests onto Higgs voice-cloning HTTP requests."""

    def __init__(self, *, base_url: str | None = None) -> None:
        self.base_url = base_url or f"http://{HOST}:{PORT}"

    def render_batch(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        return [self.render_request(request) for request in requests]

    def render_request(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        output_path = Path(artifact_name(request))
        segment_paths: list[Path] = []
        line_starts: list[float] = []
        frame_count = 0
        try:
            for index, line in enumerate(self.synthesized_lines(request)):
                segment_path = output_path.with_suffix(f".line-{index}.wav")
                segment_paths.append(segment_path)
                self.synthesize_line(line, segment_path)
                segment_frames, sample_rate = wav_frame_count(segment_path)
                if sample_rate != SAMPLE_RATE:
                    raise RuntimeError(
                        f"Higgs returned {sample_rate} Hz audio; expected {SAMPLE_RATE} Hz"
                    )
                line_starts.append(frame_count / SAMPLE_RATE)
                frame_count += segment_frames
            concatenate_wavs(segment_paths, output_path, sample_rate=SAMPLE_RATE)
        finally:
            for segment_path in segment_paths:
                segment_path.unlink(missing_ok=True)
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
        payload: dict[str, Any] = {
            "model": MODEL,
            "input": line["spoken_text"],
            "references": [{"audio_path": line["voice_path"]}],
            "temperature": float(os.environ.get("HIGGS_TEMPERATURE", "0.8")),
            "top_k": int(os.environ.get("HIGGS_TOP_K", "50")),
            "max_new_tokens": int(os.environ.get("HIGGS_MAX_NEW_TOKENS", "2048")),
        }
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
    server = start_sglang_server()
    try:
        run_server(HiggsTtsEngine().render_batch)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
