from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from carthage.dependency_injection import AsyncInjector

from radio_drama.config import ProductionConfig
from radio_drama.forced_alignment import copy_dialogue_contents
from radio_drama.init import radio_drama_injector
from radio_drama.planning import DialogueAudio, DialogueLine, ScriptRenderRequest, SpeakerVoiceReference
from radio_drama.qwen_tts import QwenTtsResource
from radio_drama.rendering import RenderResult
from radio_drama.sound import SoundPlan
from radio_drama.testing import CachedQwenTtsResource
from radio_drama.testing import CachedVibeVoiceResource
from radio_drama.testing import CachedWhisperXResource


REPO_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = REPO_ROOT / "voices"


async def _make_async_injector(config: ProductionConfig) -> tuple:
    injector = radio_drama_injector(
        config=config,
        event_loop=asyncio.get_running_loop(),
    )
    return injector, injector(AsyncInjector)


class FakeCachedVibeVoiceResource(CachedVibeVoiceResource):
    def __init__(self, native_frame_count: int = 1200, **kwargs) -> None:
        super().__init__(**kwargs)
        self.native_frame_count = native_frame_count
        self.native_call_count = 0

    def _render_batch_native_sync(self, batch):
        self.native_call_count += 1
        return [
            np.full(self.native_frame_count, fill_value=index + 1, dtype=np.float32)
            for index, _ in enumerate(batch)
        ]


class FakeCachedQwenTtsResource(CachedQwenTtsResource):
    def __init__(self, native_frame_count: int = 1200, **kwargs) -> None:
        super().__init__(**kwargs)
        self.native_frame_count = native_frame_count
        self.native_call_count = 0

    def _render_batch_native_sync(self, batch):
        self.native_call_count += 1
        return [
            np.full(self.native_frame_count, fill_value=index + 1, dtype=np.float32)
            for index, _ in enumerate(batch)
        ]


class FakeCachedWhisperXResource(CachedWhisperXResource):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.live_call_count = 0

    async def _live_fill_start_positions(self, contents, result):
        self.live_call_count += 1
        updated = copy_dialogue_contents(contents)
        for index, content in enumerate(updated):
            content.start_pos = index * 0.25
        return updated


def test_cached_vibevoice_resource_replays_cached_metadata(
    cached_vibevoice_resource_factory,
    tmp_path: Path,
):
    config = ProductionConfig(output_sample_rate=48000, output_channels=2)
    request = ScriptRenderRequest(
        normalized_script="Speaker 1: Hello there.",
        voice_samples=("anna.wav",),
    )
    cache_dir = tmp_path / "cache"

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            live_resource = await cached_vibevoice_resource_factory(
                ainjector,
                mode="live",
                cache_dir=cache_dir,
                resource_type=FakeCachedVibeVoiceResource,
                native_frame_count=1200,
            )
            live_registration = await live_resource.register_request(request)
            live_result = await live_registration.render()

            cache_resource = await cached_vibevoice_resource_factory(
                ainjector,
                mode="cache",
                cache_dir=cache_dir,
                resource_type=FakeCachedVibeVoiceResource,
                native_frame_count=1200,
            )
            cache_registration = await cache_resource.register_request(request)
            cache_result = await cache_registration.render()
            return live_resource, live_result, cache_resource, cache_result
        finally:
            injector.close()

    live_resource, live_result, cache_resource, cache_result = asyncio.run(runner())
    assert live_resource.native_call_count == 1
    assert cache_resource.native_call_count == 0
    assert live_result.audio.shape == (2400, 2)
    assert cache_result.audio.shape == (2400, 2)
    assert np.array_equal(live_result.audio, cache_result.audio)


def test_cached_vibevoice_resource_skips_when_cache_is_missing(
    cached_vibevoice_resource_factory,
    tmp_path: Path,
):
    config = ProductionConfig(output_sample_rate=48000, output_channels=2)
    request = ScriptRenderRequest(
        normalized_script="Speaker 1: Missing cache entry.",
        voice_samples=("anna.wav",),
    )

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await cached_vibevoice_resource_factory(
                ainjector,
                mode="cache",
                cache_dir=tmp_path / "cache",
                resource_type=FakeCachedVibeVoiceResource,
            )
            registration = await resource.register_request(request)
            await registration.render()
        finally:
            injector.close()

    with pytest.raises(pytest.skip.Exception):
        asyncio.run(runner())


def test_cached_qwen_resource_replays_cached_metadata(
    cached_qwen_resource_factory,
    tmp_path: Path,
):
    config = ProductionConfig(output_sample_rate=48000, output_channels=2)
    request = ScriptRenderRequest(
        normalized_script="Speaker 1: Hello there.\nSpeaker 2: General Kenobi.",
        voice_samples=("anna.wav", "ben.wav"),
    )
    cache_dir = tmp_path / "cache"

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            live_resource = await cached_qwen_resource_factory(
                ainjector,
                mode="live",
                cache_dir=cache_dir,
                resource_type=FakeCachedQwenTtsResource,
                native_frame_count=1200,
            )
            live_registration = await live_resource.register_request(request)
            live_result = await live_registration.render()

            cache_resource = await cached_qwen_resource_factory(
                ainjector,
                mode="cache",
                cache_dir=cache_dir,
                resource_type=FakeCachedQwenTtsResource,
                native_frame_count=1200,
            )
            cache_registration = await cache_resource.register_request(request)
            cache_result = await cache_registration.render()
            return live_resource, live_result, cache_resource, cache_result
        finally:
            injector.close()

    live_resource, live_result, cache_resource, cache_result = asyncio.run(runner())
    assert live_resource.native_call_count == 1
    assert cache_resource.native_call_count == 0
    assert live_result.audio.shape == (2400, 2)
    assert cache_result.audio.shape == (2400, 2)
    assert np.array_equal(live_result.audio, cache_result.audio)


def test_cached_qwen_resource_skips_when_cache_is_missing(
    cached_qwen_resource_factory,
    tmp_path: Path,
):
    config = ProductionConfig(output_sample_rate=48000, output_channels=2)
    request = ScriptRenderRequest(
        normalized_script="Speaker 1: Missing cache entry.",
        voice_samples=("anna.wav",),
    )

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await cached_qwen_resource_factory(
                ainjector,
                mode="cache",
                cache_dir=tmp_path / "cache",
                resource_type=FakeCachedQwenTtsResource,
            )
            registration = await resource.register_request(request)
            await registration.render()
        finally:
            injector.close()

    with pytest.raises(pytest.skip.Exception):
        asyncio.run(runner())


def test_cached_whisperx_resource_replays_cached_metadata(
    cached_whisperx_resource_factory,
    tmp_path: Path,
):
    config = ProductionConfig(output_sample_rate=48000, output_channels=2)
    speaker = SpeakerVoiceReference(
        authored_name="Anna",
        voice_name="anna.wav",
        resolved_path=Path("anna.wav"),
    )
    contents = [
        DialogueLine(speaker=speaker, spoken_text="Hello there."),
        DialogueAudio(audio_plan=SoundPlan(node=None)),
        DialogueLine(speaker=speaker, spoken_text="General Kenobi."),
    ]
    result = RenderResult(audio=np.zeros((128, 2), dtype=np.float32))
    cache_dir = tmp_path / "forced-alignment-cache"

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            live_resource = await cached_whisperx_resource_factory(
                ainjector,
                mode="live",
                cache_dir=cache_dir,
                resource_type=FakeCachedWhisperXResource,
            )
            live_contents = await live_resource.fill_start_positions(contents, result)

            cache_resource = await cached_whisperx_resource_factory(
                ainjector,
                mode="cache",
                cache_dir=cache_dir,
                resource_type=FakeCachedWhisperXResource,
            )
            cache_contents = await cache_resource.fill_start_positions(contents, result)
            return live_resource, live_contents, cache_resource, cache_contents
        finally:
            injector.close()

    live_resource, live_contents, cache_resource, cache_contents = asyncio.run(runner())
    assert live_resource.live_call_count == 1
    assert cache_resource.live_call_count == 0
    assert [content.start_pos for content in live_contents] == [0.0, 0.25, 0.5]
    assert [content.start_pos for content in cache_contents] == [0.0, 0.25, 0.5]


def test_cached_whisperx_resource_skips_when_cache_is_missing(
    cached_whisperx_resource_factory,
    tmp_path: Path,
):
    config = ProductionConfig(output_sample_rate=48000, output_channels=2)
    speaker = SpeakerVoiceReference(
        authored_name="Anna",
        voice_name="anna.wav",
        resolved_path=Path("anna.wav"),
    )
    contents = [DialogueLine(speaker=speaker, spoken_text="Hello there.")]
    result = RenderResult(audio=np.zeros((128, 2), dtype=np.float32))

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await cached_whisperx_resource_factory(
                ainjector,
                mode="cache",
                cache_dir=tmp_path / "forced-alignment-cache",
                resource_type=FakeCachedWhisperXResource,
            )
            await resource.fill_start_positions(contents, result)
        finally:
            injector.close()

    with pytest.raises(pytest.skip.Exception):
        asyncio.run(runner())


@pytest.mark.live
def test_cached_vibevoice_resource_supports_live_then_cache_modes(
    cached_vibevoice_resource_factory,
    tmp_path: Path,
):
    config = ProductionConfig(
        voice_directory=VOICE_DIR,
        output_sample_rate=48000,
        output_channels=2,
        device="cuda",
    )
    request = ScriptRenderRequest(
        normalized_script=(
            "Speaker 1: This test should populate metadata in live mode.\n"
            "Speaker 2: Then cache mode should replay the structural result."
        ),
        voice_samples=(
            str(VOICE_DIR / "chandra.wav"),
            str(VOICE_DIR / "david.wav"),
        ),
    )
    cache_dir = tmp_path / "live-cache"

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            live_resource = await cached_vibevoice_resource_factory(
                ainjector,
                mode="live",
                cache_dir=cache_dir,
            )
            live_registration = await live_resource.register_request(request)
            live_result = await live_registration.render()

            cache_resource = await cached_vibevoice_resource_factory(
                ainjector,
                mode="cache",
                cache_dir=cache_dir,
            )
            cache_registration = await cache_resource.register_request(request)
            cache_result = await cache_registration.render()
            return live_result, cache_result
        finally:
            injector.close()

    live_result, cache_result = asyncio.run(runner())
    assert live_result.audio.shape[0] > 0
    assert live_result.audio.shape[1] == 2
    assert cache_result.audio.shape == live_result.audio.shape
    assert np.array_equal(live_result.audio, cache_result.audio)
    assert any(cache_dir.glob("*.json"))


@pytest.mark.live
def test_cached_qwen_resource_supports_live_then_cache_modes(
    cached_qwen_resource_factory,
    tmp_path: Path,
):
    config = ProductionConfig(
        voice_directory=VOICE_DIR,
        output_sample_rate=48000,
        output_channels=2,
        device="cuda",
    )
    request = ScriptRenderRequest(
        normalized_script=(
            "Speaker 1: This test should populate metadata in live mode.\n"
            "Speaker 2: Then cache mode should replay the structural result."
        ),
        voice_samples=(
            str(VOICE_DIR / "chandra.wav"),
            str(VOICE_DIR / "david.wav"),
        ),
    )
    cache_dir = tmp_path / "live-qwen-cache"

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            live_resource = await cached_qwen_resource_factory(
                ainjector,
                mode="live",
                cache_dir=cache_dir,
            )
            live_registration = await live_resource.register_request(request)
            live_result = await live_registration.render()

            cache_resource = await cached_qwen_resource_factory(
                ainjector,
                mode="cache",
                cache_dir=cache_dir,
            )
            cache_registration = await cache_resource.register_request(request)
            cache_result = await cache_registration.render()
            return live_result, cache_result
        finally:
            injector.close()

    live_result, cache_result = asyncio.run(runner())
    assert live_result.audio.shape[0] > 0
    assert live_result.audio.shape[1] == 2
    assert cache_result.audio.shape == live_result.audio.shape
    assert np.array_equal(live_result.audio, cache_result.audio)
    assert any(cache_dir.glob("*.json"))
