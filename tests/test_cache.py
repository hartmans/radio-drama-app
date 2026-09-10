from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pytest
import numpy as np
import soundfile as sf
from carthage.dependency_injection import InjectionKey

from radio_drama.cache import CacheCollection
from radio_drama.config import ProductionConfig
from radio_drama.dialogue import DialogueLine, ScriptGap, SpeakerVoiceReference
from radio_drama.effects import VOICE_PREPROCESS_VERSION
from radio_drama.forced_alignment import WhisperXResource
from radio_drama.qwen_tts import QwenTtsResource
from radio_drama.rendering import BackendTtsResult, DialogueLineTiming, ScriptTiming
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
        f"{expected_stem}.meta",
        f"{expected_stem}.wav",
    ]
    payload = json.loads((cache_dir / f"{expected_stem}.meta").read_text(encoding="utf-8"))
    assert payload["frame_count"] == 2
    assert payload["sample_rate"] == 24000


@pytest.mark.parametrize("cache_state", ["disabled", "missing", "valid", "invalid"])
def test_registered_cache_misses_batch_when_rendered_sequentially(tmp_path, cache_state):
    config = ProductionConfig(output_sample_rate=24000, output_channels=1, batch_size=4)
    output_path = tmp_path / "render.wav" if cache_state != "disabled" else None
    requests = [
        request_from_normalized_script(f"Speaker 1: Line {index}.", ("anna.wav",))
        for index in range(3)
    ]
    cached_timing = ScriptTiming((DialogueLineTiming(0.0, 2 / 24000),))
    if cache_state in {"valid", "invalid"}:
        collection = CacheCollection("vibevoice", Path(f"{output_path}.cache"))
        key = collection.key_for(requests[1])
        wav_path = collection.path_for_subtype(key, "wav")
        wav_path.parent.mkdir(parents=True)
        sf.write(wav_path, np.array([0.25, -0.25], dtype=np.float32), 24000)
        collection.path_for_subtype(key, "meta").write_text(
            json.dumps({
                "sample_rate": 24000,
                "alignment_key": "native:test",
                "dialogue_line_spans": [[0.0, 2 / 24000]],
            }) if cache_state == "valid" else "invalid metadata",
            encoding="utf-8",
        )

    class BatchingResource(VibeVoiceResource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.batches = []
            self._sample_rate = 24000

        def _render_batch_native_sync(self, batch):
            self.batches.append([registration.request for registration in batch])
            return [np.array([0.25, -0.25], dtype=np.float32) for _ in batch]

    async def runner():
        injector, ainjector = await make_async_injector(config, output_path=output_path)
        try:
            resource = await ainjector(BatchingResource)
            registrations = [await resource.register_request(request) for request in requests]
            assert resource.batches == []
            first = await registrations[0].render()
            expected = [requests[0], requests[2]] if cache_state == "valid" else requests
            assert resource.batches == [expected]
            results = [first] + [await registration.render() for registration in registrations[1:]]
            assert resource.batches == [expected]
            assert all(result.frame_count == 2 for result in results)
            assert await registrations[0].render() is first
            if cache_state == "valid":
                assert results[1].timing == cached_timing
                assert await registrations[1].ensure_timing(
                    requests[1].dialogue_contents, results[1]
                ) == cached_timing
        finally:
            injector.close()

    asyncio.run(runner())


def test_vibevoice_adopts_legacy_wav_without_rewriting_audio(monkeypatch, tmp_path: Path):
    config = ProductionConfig(output_sample_rate=24000, output_channels=1)
    output_path = tmp_path / "render.wav"
    request = request_from_normalized_script("Speaker 1: Legacy audio.", ("anna.wav",))
    cache_dir = Path(f"{output_path}.cache")
    collection = CacheCollection("vibevoice", cache_dir)
    key = collection.key_for(request)
    wav_path = collection.path_for_subtype(key, "wav")
    json_path = collection.path_for_subtype(key, "json")
    wav_path.parent.mkdir(parents=True)
    sf.write(wav_path, np.array([0.25, -0.25], dtype=np.float32), 24000)
    json_path.write_text(
        json.dumps(request.build_cache_payload(sample_rate=24000, frame_count=2)),
        encoding="utf-8",
    )
    original_wav = wav_path.read_bytes()

    class LegacyOnlyVibeVoiceResource(VibeVoiceResource):
        def _render_batch_native_sync(self, batch):
            raise AssertionError("legacy audio should not be regenerated")

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    async def runner():
        injector, ainjector = await make_async_injector(config, output_path=output_path)
        try:
            resource = await ainjector(LegacyOnlyVibeVoiceResource)
            return await (await resource.register_request(request)).render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    meta_path = collection.path_for_subtype(key, "meta")
    assert result.audio.shape == (2,)
    assert wav_path.read_bytes() == original_wav
    assert json_path.is_file()
    assert meta_path.is_file()
    assert json.loads(meta_path.read_text(encoding="utf-8"))["dialogue_line_spans"] is None


def test_forced_alignment_metadata_reuses_audio_and_invalidates_by_projection(
    monkeypatch,
    tmp_path: Path,
):
    config = ProductionConfig(output_sample_rate=24000, output_channels=1)
    output_path = tmp_path / "render.wav"
    request = request_from_normalized_script("Speaker 1: Align me.", ("anna.wav",))

    class UntimedVibeVoiceResource(VibeVoiceResource):
        native_calls = 0

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._sample_rate = 24000

        def _render_batch_native_sync(self, batch):
            type(self).native_calls += 1
            return [np.array([0.25, -0.25], dtype=np.float32) for _ in batch]

    class FakeWhisperX:
        calls = 0
        missing = False

        async def script_timing(self, contents, result):
            type(self).calls += 1
            if self.missing:
                return ScriptTiming((DialogueLineTiming(math.nan, math.nan),))
            return ScriptTiming((DialogueLineTiming(0.0, result.frame_count / 24000),))

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    async def render_and_align(contents):
        injector, ainjector = await make_async_injector(config, output_path=output_path)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        try:
            resource = await ainjector(UntimedVibeVoiceResource)
            registration = await resource.register_request(request)
            result = await registration.render()
            timing = await registration.ensure_timing(contents, result)
            return timing
        finally:
            injector.close()

    contents = list(request.dialogue_contents)
    first_timing = asyncio.run(render_and_align(contents))
    collection = CacheCollection("vibevoice", Path(f"{output_path}.cache"))
    key = collection.key_for(request)
    wav_path = collection.path_for_subtype(key, "wav")
    meta_path = collection.path_for_subtype(key, "meta")
    original_wav = wav_path.read_bytes()
    first_payload = json.loads(meta_path.read_text(encoding="utf-8"))

    replay_timing = asyncio.run(render_and_align(contents))
    changed_contents = [contents[0], ScriptGap(label="changed projection")]
    changed_timing = asyncio.run(render_and_align(changed_contents))
    changed_payload = json.loads(meta_path.read_text(encoding="utf-8"))

    assert first_timing == replay_timing == changed_timing
    assert UntimedVibeVoiceResource.native_calls == 1
    assert FakeWhisperX.calls == 2
    assert wav_path.read_bytes() == original_wav
    assert first_payload["alignment_key"] != changed_payload["alignment_key"]
    assert changed_payload["dialogue_line_spans"] == [[0.0, 2 / 24000]]

    # Old algorithm results must be replaced without synthesizing audio again,
    # even when the authored projection has not changed.
    old_payload = dict(changed_payload)
    old_payload["alignment_key"] = "script-timing-v1:" + changed_payload["alignment_key"].split(":", 1)[1]
    old_payload["dialogue_line_spans"] = [[0.0, 0.0]]
    meta_path.write_text(json.dumps(old_payload), encoding="utf-8")
    assert asyncio.run(render_and_align(changed_contents)) == changed_timing
    assert FakeWhisperX.calls == 3
    assert UntimedVibeVoiceResource.native_calls == 1
    assert wav_path.read_bytes() == original_wav

    # Missing alignment remains missing after a write/read cycle, without
    # regenerating speech or silently replacing the cached NaNs with zeros.
    FakeWhisperX.missing = True
    missing_contents = [contents[0], ScriptGap(label="unmatched projection")]
    for _ in range(2):
        missing_timing = asyncio.run(render_and_align(missing_contents))
        assert math.isnan(missing_timing.dialogue_lines[0].start)
        assert math.isnan(missing_timing.dialogue_lines[0].end)
    assert FakeWhisperX.calls == 4
    assert UntimedVibeVoiceResource.native_calls == 1
    assert wav_path.read_bytes() == original_wav
    missing_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert math.isnan(missing_payload["dialogue_line_spans"][0][0])


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
                BackendTtsResult(
                    audio=np.full(6, fill_value=index + 1, dtype=np.float32),
                    sample_rate=24000,
                    timing=ScriptTiming(
                        (
                            DialogueLineTiming(0.0, 0.5),
                            DialogueLineTiming(0.5, 0.75),
                        )
                    ),
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
    assert live_result.timing == replay_result.timing
    payload = json.loads((cache_dir / f"{expected_stem}.meta").read_text(encoding="utf-8"))
    assert payload["frame_count"] == 6
    assert payload["sample_rate"] == 24000
    assert payload["dialogue_line_spans"] == [[0.0, 0.5], [0.5, 0.75]]


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
            resource.transcription_resource.cache_path = (
                lambda _path: tmp_path / "reference-transcript.txt"
            )
            reference = SpeakerVoiceReference(
                authored_name="Anna",
                voice_name="anna",
                resolved_path=voice_path,
            )
            prompt_items = resource._build_prompt_items_for_voice_sync(reference)
            return prompt_items, reference
        finally:
            injector.close()

    prompt_items, reference = asyncio.run(runner())
    assert len(prompt_items) == 1
    assert seen["voice_path"] == str(voice_path)
    np.testing.assert_allclose(seen["transcribe_audio"], np.array([0.25, -0.25], dtype=np.float32))
    assert seen["transcribe_sample_rate"] == 12345
    ref_audio, ref_sample_rate = seen["ref_audio"]
    np.testing.assert_allclose(ref_audio, np.array([0.25, -0.25], dtype=np.float32))
    assert ref_sample_rate == 12345
    assert seen["ref_text"] == "reference transcript"
    assert reference.transcript == "reference transcript"
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
    request.dialogue_contents.extend([
        DialogueLine(speaker=SpeakerVoiceReference(
            authored_name="Alias", voice_name="anna", resolved_path=voice_path,
            effect_expression="telephone",
        ), spoken_text="Same reference."),
        DialogueLine(speaker=SpeakerVoiceReference(
            authored_name="Louder", voice_name="anna", resolved_path=voice_path,
            gain=3.0,
        ), spoken_text="Different gain."),
    ])
    seen_paths: list[tuple[Path, int]] = []

    class FakeVibeVoiceResource(VibeVoiceResource):
        def _preprocessed_voice_sample_sync(self, voice_path: Path, *, output_sample_rate: int, gain_db=0.0):
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
        "Speaker 2: General Kenobi.\n"
        "Speaker 1: Same reference.\n"
        "Speaker 3: Different gain."
    )
    assert seen_paths == [
        (voice_path.expanduser().resolve(), 16000),
        (other_voice_path.expanduser().resolve(), 16000),
        (voice_path.expanduser().resolve(), 16000),
    ]
    assert len(voice_samples) == 3
    np.testing.assert_allclose(voice_samples[0], np.full(3, 1.0, dtype=np.float32))
    np.testing.assert_allclose(voice_samples[1], np.full(3, 2.0, dtype=np.float32))
