from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import soundfile as sf
from carthage.dependency_injection import AsyncInjector

from radio_drama.config import ProductionConfig
from radio_drama.init import radio_drama_injector
from radio_drama.planning import DialogueAudio, DialogueLine, SoundPlan, SpeakerVoiceReference
from radio_drama.rendering import RenderResult


REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR = REPO_ROOT / "tests" / "resources"
FORCED_ALIGNMENT_CACHE_DIR = RESOURCE_DIR / "forced_alignment_cache"
ALIGNMENT_TOLERANCE_SECONDS = 0.9


async def _make_async_injector(config: ProductionConfig) -> tuple:
    injector = radio_drama_injector(
        config=config,
        event_loop=asyncio.get_running_loop(),
    )
    return injector, injector(AsyncInjector)


def _load_fixture_segments(name: str) -> list[dict[str, object]]:
    payload = json.loads((RESOURCE_DIR / "whisperx_cli" / f"{name}.json").read_text(encoding="utf-8"))
    return list(payload["segments"])


def _load_fixture_audio(name: str) -> tuple[list[float], int]:
    audio, sample_rate = sf.read(RESOURCE_DIR / f"{name}.wav", dtype="float32")
    return audio.tolist(), sample_rate


def _build_fixture_request() -> tuple[list, RenderResult, list[float], int]:
    girl_audio, girl_sample_rate = _load_fixture_audio("girl1")
    lawyer_audio, lawyer_sample_rate = _load_fixture_audio("lawyer1")
    assert girl_sample_rate == lawyer_sample_rate

    girl_segments = _load_fixture_segments("girl1")
    lawyer_segments = _load_fixture_segments("lawyer1")
    girl_duration = len(girl_audio) / girl_sample_rate

    girl_speaker = SpeakerVoiceReference(
        authored_name="Girl",
        voice_name="girl1.wav",
        resolved_path=RESOURCE_DIR / "girl1.wav",
    )
    lawyer_speaker = SpeakerVoiceReference(
        authored_name="Lawyer",
        voice_name="lawyer1.wav",
        resolved_path=RESOURCE_DIR / "lawyer1.wav",
    )

    contents = [
        *(DialogueLine(speaker=girl_speaker, spoken_text=str(segment["text"]).strip()) for segment in girl_segments),
        DialogueAudio(audio_plan=SoundPlan(node=None)),
        *(DialogueLine(speaker=lawyer_speaker, spoken_text=str(segment["text"]).strip()) for segment in lawyer_segments),
    ]
    expected_starts = [
        *(float(segment["start"]) for segment in girl_segments),
        (
            float(girl_segments[-1]["end"])
            + girl_duration
            + float(lawyer_segments[0]["start"])
        )
        / 2.0,
        *(girl_duration + float(segment["start"]) for segment in lawyer_segments),
    ]
    combined_audio = girl_audio + lawyer_audio
    return contents, RenderResult(audio=combined_audio), expected_starts, girl_sample_rate


async def _run_cached_alignment(cached_whisperx_resource_factory, *, mode: str):
    contents, result, expected_starts, sample_rate = _build_fixture_request()
    config = ProductionConfig(
        output_sample_rate=sample_rate,
        output_channels=1,
        device="cuda",
    )

    injector, ainjector = await _make_async_injector(config)
    try:
        resource = await cached_whisperx_resource_factory(
            ainjector,
            mode=mode,
            cache_dir=FORCED_ALIGNMENT_CACHE_DIR,
        )
        aligned_contents = await resource.fill_start_positions(contents, result)
        return aligned_contents, expected_starts
    finally:
        injector.close()


def _assert_plausible_alignment(aligned_contents, expected_starts: list[float]) -> None:
    actual_starts = [float(content.start_pos) for content in aligned_contents]
    assert len(actual_starts) == len(expected_starts)
    assert actual_starts == sorted(actual_starts)
    for actual, expected in zip(actual_starts, expected_starts, strict=True):
        assert actual == pytest.approx(expected, abs=ALIGNMENT_TOLERANCE_SECONDS)


def test_cached_whisperx_resource_aligns_fixture_dialogue_from_cache(
    cached_whisperx_resource_factory,
):
    aligned_contents, expected_starts = asyncio.run(
        _run_cached_alignment(cached_whisperx_resource_factory, mode="cache")
    )

    _assert_plausible_alignment(aligned_contents, expected_starts)


@pytest.mark.live
def test_cached_whisperx_resource_aligns_fixture_dialogue_live(
    cached_whisperx_resource_factory,
):
    aligned_contents, expected_starts = asyncio.run(
        _run_cached_alignment(cached_whisperx_resource_factory, mode="live")
    )

    _assert_plausible_alignment(aligned_contents, expected_starts)
    assert any(FORCED_ALIGNMENT_CACHE_DIR.glob("*.json"))
