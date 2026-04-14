from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
from carthage.dependency_injection import InjectionKey

from radio_drama.cache import CacheCollection
from radio_drama.config import ProductionConfig
from radio_drama.effects import VOICE_PREPROCESS_VERSION
from radio_drama.forced_alignment import WhisperXResource
from radio_drama.qwen_tts import QwenTtsResource
from radio_drama.rendering import ScriptRenderResult
from radio_drama.vibevoice import VibeVoiceResource

from phase1_helpers import make_async_injector, request_from_normalized_script


def test_cache_collection_recreates_invalid_hit(tmp_path: Path):
    request = request_from_normalized_script("Speaker 1: Hello there.", ("anna.wav",))
    collection = CacheCollection("vibevoice", tmp_path)
    key = collection.key_for(request)
    stale_wav_path = collection.path_for_subtype(key, "wav")
    stale_wav_path.parent.mkdir(parents=True, exist_ok=True)
    stale_wav_path.write_bytes(b"stale")

    miss_calls = 0

    def on_miss(miss_key, miss_collection):
        nonlocal miss_calls
        miss_calls += 1
        miss_collection.path_for_subtype(miss_key, "wav").write_bytes(b"fresh-wav")
        miss_collection.path_for_subtype(miss_key, "json").write_text(
            json.dumps({"created": True}),
            encoding="utf-8",
        )

    hit = collection.get_or_create(
        request,
        on_miss,
        validate=request.validate_cache_hit,
    )

    assert miss_calls == 1
    assert hit["wav"].read_bytes() == b"fresh-wav"
    assert json.loads(hit["json"].read_text(encoding="utf-8")) == {"created": True}


def test_vibevoice_resource_uses_shared_cache_manager_and_preserves_stem(
    monkeypatch,
    tmp_path: Path,
):
    config = ProductionConfig(output_sample_rate=48000, output_channels=2)
    output_path = tmp_path / "render.wav"
    request = request_from_normalized_script("Speaker 1: Hello there.", ("anna.wav",))

    class FakeCachedVibeVoiceResource(VibeVoiceResource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.native_call_count = 0
            self._sample_rate = 24000

        def _render_batch_native_sync(self, batch):
            self.native_call_count += 1
            return [np.array([0.25, -0.25], dtype=np.float32) for _ in batch]

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    async def render_once():
        injector, ainjector = await make_async_injector(
            config,
            output_path=output_path,
        )
        try:
            resource = await ainjector(FakeCachedVibeVoiceResource)
            registration = await resource.register_request(request)
            result = await registration.render()
            return resource.native_call_count, result
        finally:
            injector.close()

    live_call_count, live_result = asyncio.run(render_once())
    replay_call_count, replay_result = asyncio.run(render_once())

    cache_dir = Path(f"{output_path}.cache")
    expected_stem = f"vibevoice_hello_there_{request.cache_hash()}"
    assert live_call_count == 1
    assert replay_call_count == 0
    assert live_result.audio.shape == (4, 2)
    assert np.array_equal(live_result.audio, replay_result.audio)
    assert sorted(path.name for path in cache_dir.iterdir()) == [
        f"{expected_stem}.json",
        f"{expected_stem}.wav",
    ]
    payload = json.loads((cache_dir / f"{expected_stem}.json").read_text(encoding="utf-8"))
    assert payload["dialogue_lines"][0]["spoken_text"] == "Hello there."
    assert payload["frame_count"] == 2
    assert payload["sample_rate"] == 24000


def test_qwentts_resource_reuses_cached_native_timing(monkeypatch, tmp_path: Path):
    config = ProductionConfig(output_sample_rate=48000, output_channels=2)
    output_path = tmp_path / "render.wav"
    request = request_from_normalized_script(
        "Speaker 1: Hello there.\nSpeaker 2: General Kenobi.",
        ("anna.wav", "ben.wav"),
    )

    class FakeCachedQwenTtsResource(QwenTtsResource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.native_call_count = 0
            self._sample_rate = 24000

        def _render_batch_native_sync(self, batch):
            self.native_call_count += 1
            return [
                ScriptRenderResult(
                    audio=np.full(6, fill_value=index + 1, dtype=np.float32),
                    dialogue_line_start_positions=(0.0, 0.5),
                )
                for index, _ in enumerate(batch)
            ]

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    async def render_once():
        injector, ainjector = await make_async_injector(
            config,
            output_path=output_path,
        )
        try:
            resource = await ainjector(FakeCachedQwenTtsResource)
            registration = await resource.register_request(request)
            result = await registration.render()
            return resource.native_call_count, result
        finally:
            injector.close()

    live_call_count, live_result = asyncio.run(render_once())
    replay_call_count, replay_result = asyncio.run(render_once())

    cache_dir = Path(f"{output_path}.cache")
    expected_stem = f"qwentts_hello_there_{request.cache_hash()}"
    assert live_call_count == 1
    assert replay_call_count == 0
    assert live_result.audio.shape == (12, 2)
    assert np.array_equal(live_result.audio, replay_result.audio)
    assert live_result.dialogue_line_start_positions == (0.0, 0.5)
    assert replay_result.dialogue_line_start_positions == (0.0, 0.5)
    payload = json.loads((cache_dir / f"{expected_stem}.json").read_text(encoding="utf-8"))
    assert payload["frame_count"] == 6
    assert payload["sample_rate"] == 24000
    assert payload["dialogue_line_start_positions"] == [0.0, 0.5]


def test_qwentts_prompt_cache_path_includes_voice_preprocess_version(tmp_path: Path):
    voice_directory = tmp_path / "voices"
    voice_directory.mkdir()
    voice_path = voice_directory / "anna.wav"

    async def runner():
        injector, ainjector = await make_async_injector(
            ProductionConfig(voice_directory=voice_directory),
        )
        try:
            resource = await ainjector(QwenTtsResource)
            return resource._prompt_cache_path(voice_path)
        finally:
            injector.close()

    cache_path = asyncio.run(runner())
    assert cache_path.parts[-2:] == (VOICE_PREPROCESS_VERSION, "anna.wav.pt")


def test_qwentts_resource_preprocesses_reference_voice_before_prompt_build(
    tmp_path: Path,
):
    voice_directory = tmp_path / "voices"
    voice_directory.mkdir()
    voice_path = voice_directory / "anna.wav"
    voice_path.write_bytes(b"fake")
    seen: dict[str, object] = {}

    class FakeWhisperX:
        def transcribe_audio_sample_sync(self, audio, sample_rate=None):
            seen["transcribe_audio"] = np.array(audio, copy=True)
            seen["transcribe_sample_rate"] = sample_rate
            return "reference transcript"

    class FakeModel:
        def create_voice_clone_prompt(self, *, ref_audio, ref_text, x_vector_only_mode):
            seen["ref_audio"] = ref_audio
            seen["ref_text"] = ref_text
            seen["x_vector_only_mode"] = x_vector_only_mode
            return [object()]

    class FakeQwenTtsResource(QwenTtsResource):
        def _ensure_loaded(self):
            return FakeModel()

        def _preprocessed_voice_reference_sync(self, voice_path: str):
            seen["voice_path"] = voice_path
            return np.array([0.25, -0.25], dtype=np.float32), 12345

    async def runner():
        injector, ainjector = await make_async_injector(
            ProductionConfig(voice_directory=voice_directory),
        )
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        try:
            resource = await ainjector(FakeQwenTtsResource)
            return resource._build_prompt_items_for_voice_sync(str(voice_path))
        finally:
            injector.close()

    prompt_items = asyncio.run(runner())
    assert len(prompt_items) == 1
    assert seen["voice_path"] == str(voice_path)
    np.testing.assert_allclose(seen["transcribe_audio"], np.array([0.25, -0.25], dtype=np.float32))
    assert seen["transcribe_sample_rate"] == 12345
    ref_audio, ref_sample_rate = seen["ref_audio"]
    np.testing.assert_allclose(ref_audio, np.array([0.25, -0.25], dtype=np.float32))
    assert ref_sample_rate == 12345
    assert seen["ref_text"] == "reference transcript"
    assert seen["x_vector_only_mode"] is False


def test_vibevoice_resource_preprocesses_unique_reference_voices_per_request(
    tmp_path: Path,
):
    voice_path = tmp_path / "anna.wav"
    other_voice_path = tmp_path / "ben.wav"
    request = request_from_normalized_script(
        "Speaker 1: Hello there.\nSpeaker 1: Welcome back.\nSpeaker 2: General Kenobi.",
        (str(voice_path), str(other_voice_path)),
    )
    seen_paths: list[tuple[Path, int]] = []

    class FakeVibeVoiceResource(VibeVoiceResource):
        def _preprocessed_voice_sample_sync(self, voice_path: Path, *, output_sample_rate: int):
            seen_paths.append((voice_path, output_sample_rate))
            value = float(len(seen_paths))
            return np.full(3, value, dtype=np.float32)

    async def runner():
        injector, ainjector = await make_async_injector(
            ProductionConfig(voice_directory=tmp_path),
        )
        try:
            resource = await ainjector(FakeVibeVoiceResource)
            return resource._normalized_script_and_voice_samples(
                request,
                voice_sample_rate=16000,
            )
        finally:
            injector.close()

    normalized_script, voice_samples = asyncio.run(runner())
    assert normalized_script == (
        "Speaker 1: Hello there.\n"
        "Speaker 1: Welcome back.\n"
        "Speaker 2: General Kenobi."
    )
    assert seen_paths == [
        (voice_path.expanduser().resolve(), 16000),
        (other_voice_path.expanduser().resolve(), 16000),
    ]
    assert len(voice_samples) == 2
    np.testing.assert_allclose(voice_samples[0], np.full(3, 1.0, dtype=np.float32))
    np.testing.assert_allclose(voice_samples[1], np.full(3, 2.0, dtype=np.float32))
