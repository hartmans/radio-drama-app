from __future__ import annotations

import asyncio
import gc
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from carthage.dependency_injection import AsyncInjector, InjectionKey, Injector

from radio_drama.config import MODEL_NATIVE_SAMPLE_RATE, ProductionConfig
from radio_drama.debug import debug_artifact_directory
from radio_drama.document import parse_production_string
from radio_drama.errors import DocumentError
from radio_drama.audio import ComposeAudioPlan
from radio_drama.dialogue import (
    DialogueAudio,
    DialogueLine,
    ScriptPlan,
    ScriptRenderRequest,
    SpeakerVoiceReference,
    TtsResource,
)
from radio_drama.forced_alignment import (
    AlignedScriptSource,
    ScriptSlice,
    WhisperXResource,
)
from radio_drama.init import radio_drama_injector
from radio_drama.qwen_tts import QwenTtsResource
from radio_drama.rendering import (
    BackendTtsResult,
    DialogueLineTiming,
    ProductionResult,
    RenderResult,
    ScriptTiming,
)
from radio_drama.vibevoice import VibeVoiceResource
from radio_drama.sound import NormalizedSoundCache, SoundPlan
from radio_drama.testing import CachedRenderMetadata

from phase1_helpers import (
    make_async_injector as _make_async_injector,
    normalized_script_from_request as _normalized_script_from_request,
    request_from_normalized_script as _request_from_normalized_script,
)


def test_vibevoice_output_debug_writes_native_wavs(tmp_path: Path):
    config = ProductionConfig(
        voice_directory=tmp_path,
        debug_log_path=tmp_path / "output.wav.log",
        debug_categories=("vibevoice_output",),
    )

    class FakeDebugResource(VibeVoiceResource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._sample_rate = MODEL_NATIVE_SAMPLE_RATE

        def _render_batch_native_sync(self, batch):
            return [np.array([0.25, -0.25], dtype=np.float32)]

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
                resource = await ainjector(FakeDebugResource)
                batch = [
                    SimpleNamespace(
                        request=_request_from_normalized_script(
                            "Speaker 1: First line for debug output.",
                            ("anna.wav",),
                        )
                    )
                ]
                resource._render_batch_sync(batch)
        finally:
            injector.close()

    asyncio.run(runner())
    artifact_directory = debug_artifact_directory(config, "vibevoice_output")
    assert artifact_directory is not None
    artifact_files = sorted(artifact_directory.glob("*.wav"))
    assert [path.name for path in artifact_files] == ["000-first_line_for_debug_output.wav"]
    audio, sample_rate = sf.read(artifact_files[0], dtype="float32")
    assert sample_rate == MODEL_NATIVE_SAMPLE_RATE
    assert audio.shape == (2,)


def test_aligned_script_source_render_keeps_audio_and_records_markers(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=2)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult(audio=np.ones((4, 2), dtype=np.float32))

            return Registered()

    class FakeWhisperX:
        async def script_timing(self, contents, result):
            return ScriptTiming(
                (
                    DialogueLineTiming(0.0, 0.0),
                    DialogueLineTiming(1.0, 1.0),
                )
            )

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == sound_file
            return asyncio.create_task(
                asyncio.sleep(0, result=np.zeros((0, 2), dtype=np.float32))
            )

    async def runner():
        injector, ainjector = await _make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>
                    Anna: anna.wav
                    Ben: anna.wav
                  </speaker-map>
                  <script>
                    Anna: First line.
                    <sound ref="door" />
                    Ben: Response.
                  </script>
                </production>
                """,
                source_name=str(xml_path),
            )
            production_plan = await root.plan(ainjector)
            audio_plan = production_plan.audio_plans[0]
            aligned_source = audio_plan.audio_plans[0].aligned_script_source
            return aligned_source, await aligned_source.render()
        finally:
            injector.close()

    aligned_source, result = asyncio.run(runner())
    assert isinstance(aligned_source, AlignedScriptSource)
    assert result.render_result.audio.shape == (4, 2)
    assert list(result.marker_frames) == [0, 2, 2, 4]
    assert [content.start_pos for content in aligned_source.contents] == [0.0, 0.5, 1.0]


def test_script_slice_and_concat_splice_sound_audio(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult(audio=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

            return Registered()

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            return [
                DialogueAudio(audio_plan=content.audio_plan, start_pos=0.5)
                if isinstance(content, DialogueAudio)
                else type(content)(
                    speaker=content.speaker,
                    spoken_text=content.spoken_text,
                    start_pos=0.0 if index == 0 else 1.0,
                )
                for index, content in enumerate(contents)
            ]

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == sound_file
            return asyncio.create_task(
                asyncio.sleep(0, result=np.array([8.0, 9.0], dtype=np.float32))
            )

    async def runner():
        injector, ainjector = await _make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>
                    Anna: anna.wav
                    Ben: anna.wav
                  </speaker-map>
                  <script>
                    Anna: First line.
                    <sound ref="door" />
                    Ben: Response.
                  </script>
                </production>
                """,
                source_name=str(xml_path),
            )
            production_plan = await root.plan(ainjector)
            audio_plan = production_plan.audio_plans[0]
            return audio_plan, await audio_plan.render()
        finally:
            injector.close()

    audio_plan, result = asyncio.run(runner())
    assert isinstance(audio_plan, ComposeAudioPlan)
    assert result.audio.tolist() == [1.0, 2.0, 8.0, 9.0, 3.0, 4.0]


def test_vibevoice_resource_batches_concurrent_requests(monkeypatch, tmp_path: Path):
    class FakeBatchingResource(VibeVoiceResource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.batch_sizes: list[int] = []
            self._sample_rate = MODEL_NATIVE_SAMPLE_RATE

        def _render_batch_sync(self, batch):
            self.batch_sizes.append(len(batch))
            return [
                BackendTtsResult(
                    np.full(index + 1, fill_value=(index + 1) / 10, dtype=np.float32),
                    sample_rate=MODEL_NATIVE_SAMPLE_RATE,
                )
                for index, _ in enumerate(batch)
            ]

    config = ProductionConfig(
        voice_directory=tmp_path,
        output_sample_rate=MODEL_NATIVE_SAMPLE_RATE,
        output_channels=1,
    )

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await ainjector(FakeBatchingResource)
            requests = [
                await resource.register_request(
                    _request_from_normalized_script(
                        f"Speaker 1: Line {index + 1}",
                        ("voice.wav",),
                    )
                )
                for index in range(2)
            ]
            results = await asyncio.gather(*(request.render() for request in requests))
            return resource.batch_sizes, results
        finally:
            injector.close()

    batch_sizes, results = asyncio.run(runner())
    assert batch_sizes == [2]
    assert [result.audio.shape for result in results] == [(1,), (2,)]
    assert np.allclose(results[0].audio, 0.1, atol=4e-5)
    assert np.allclose(results[1].audio, 0.2, atol=4e-5)


def test_cut_before_mark_allows_dropped_vibevoice_requests_to_collect(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_channels=1)

    class FakeCutBatchingResource(VibeVoiceResource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.rendered_scripts: list[str] = []

        async def register_request(self, request: ScriptRenderRequest | None):
            return await super().register_request(request)

        def _render_batch_sync(self, batch):
            self.rendered_scripts.extend(
                _normalized_script_from_request(registration.request) for registration in batch
            )
            return [
                BackendTtsResult(
                    np.array([1.0], dtype=np.float32),
                    sample_rate=MODEL_NATIVE_SAMPLE_RATE,
                )
                for _ in batch
            ]

    class FakeWhisperX:
        async def script_timing(self, contents, result):
            return ScriptTiming(
                tuple(
                    DialogueLineTiming(0.0, 0.0)
                    for content in contents
                    if isinstance(content, DialogueLine)
                )
            )

        async def fill_start_positions(self, contents, result):
            updated = []
            for content in contents:
                if isinstance(content, DialogueAudio):
                    updated.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=0.0))
                else:
                    updated.append(
                        DialogueLine(
                            speaker=content.speaker,
                            spoken_text=content.spoken_text,
                            start_pos=0.0,
                        )
                    )
            return updated

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await ainjector(FakeCutBatchingResource)
            injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), resource, close=False)
            injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>Anna: Line 1</script>
                  <script>
                    <mark id="cut" />
                    Anna: Line 2
                  </script>
                </production>
                """,
                source_name="cut-collect.xml",
            )
            production_plan = await root.plan(ainjector)
            production_plan.cut_before_mark("cut")
            gc.collect()
            await production_plan.render()
            return resource.rendered_scripts
        finally:
            injector.close()

    rendered_scripts = asyncio.run(runner())
    assert rendered_scripts == ["Speaker 1: Line 2"]


def test_cut_before_mark_drops_vibevoice_request_when_dropped_script_has_sound(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    production_path = tmp_path / "cut-sound.xml"
    production_path.write_text("<production />", encoding="utf-8")
    config = ProductionConfig(voice_directory=tmp_path, output_channels=1)

    class FakeCutBatchingResource(VibeVoiceResource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.rendered_scripts: list[str] = []

        def _render_batch_sync(self, batch):
            self.rendered_scripts.extend(
                _normalized_script_from_request(registration.request) for registration in batch
            )
            return [
                BackendTtsResult(
                    np.array([1.0], dtype=np.float32),
                    sample_rate=MODEL_NATIVE_SAMPLE_RATE,
                )
                for _ in batch
            ]

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == sound_file
            return asyncio.create_task(
                asyncio.sleep(0, result=np.array([1.0], dtype=np.float32))
            )

    class FakeWhisperX:
        async def script_timing(self, contents, result):
            return ScriptTiming(
                tuple(
                    DialogueLineTiming(0.0, 0.0)
                    for content in contents
                    if isinstance(content, DialogueLine)
                )
            )

        async def fill_start_positions(self, contents, result):
            updated = []
            for content in contents:
                if isinstance(content, DialogueAudio):
                    updated.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=0.0))
                else:
                    updated.append(
                        DialogueLine(
                            speaker=content.speaker,
                            spoken_text=content.spoken_text,
                            start_pos=0.0,
                        )
                    )
            return updated

    async def runner():
        injector, ainjector = await _make_async_injector(config, document_path=production_path)
        try:
            resource = await ainjector(FakeCutBatchingResource)
            injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), resource, close=False)
            injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
            injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    Anna: Line 1
                    <sound ref="door" />
                  </script>
                  <script>
                    <mark id="cut" />
                    Anna: Line 2
                  </script>
                </production>
                """,
                source_name=str(production_path),
            )
            production_plan = await root.plan(ainjector)
            production_plan.cut_before_mark("cut")
            gc.collect()
            await production_plan.render()
            return resource.rendered_scripts
        finally:
            injector.close()

    rendered_scripts = asyncio.run(runner())
    assert rendered_scripts == ["Speaker 1: Line 2"]


def test_cut_after_mark_drops_vibevoice_requests_after_mark(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_channels=1)

    class FakeCutBatchingResource(VibeVoiceResource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.rendered_scripts: list[str] = []

        def _render_batch_sync(self, batch):
            self.rendered_scripts.extend(
                _normalized_script_from_request(registration.request) for registration in batch
            )
            return [
                BackendTtsResult(
                    np.array([1.0], dtype=np.float32),
                    sample_rate=MODEL_NATIVE_SAMPLE_RATE,
                )
                for _ in batch
            ]

    class FakeWhisperX:
        async def script_timing(self, contents, result):
            return ScriptTiming(
                tuple(
                    DialogueLineTiming(0.0, 0.0)
                    for content in contents
                    if isinstance(content, DialogueLine)
                )
            )

        async def fill_start_positions(self, contents, result):
            updated = []
            for content in contents:
                if isinstance(content, DialogueAudio):
                    updated.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=0.0))
                else:
                    updated.append(
                        DialogueLine(
                            speaker=content.speaker,
                            spoken_text=content.spoken_text,
                            start_pos=0.0,
                        )
                    )
            return updated

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await ainjector(FakeCutBatchingResource)
            injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), resource, close=False)
            injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    Anna: Line 1
                    <mark id="cut" />
                  </script>
                  <script>Anna: Line 2</script>
                </production>
                """,
                source_name="cut-after-collect.xml",
            )
            production_plan = await root.plan(ainjector)
            production_plan.cut_after_mark("cut")
            gc.collect()
            await production_plan.render()
            return resource.rendered_scripts
        finally:
            injector.close()

    rendered_scripts = asyncio.run(runner())
    assert rendered_scripts == ["Speaker 1: Line 1"]


def test_script_gap_attribute_requires_numeric_seconds(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script post_gap="later">Anna: Line 1</script>
                </production>
                """,
                source_name="bad-gap.xml",
            )
            await root.plan(ainjector)
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="<script> post_gap must be a number of seconds"):
        asyncio.run(runner())


def test_script_length_and_post_gap_are_mutually_exclusive(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script length="1.0" post_gap="0.5">Anna: Line 1</script>
                </production>
                """,
                source_name="length-post-gap.xml",
            )
            await root.plan(ainjector)
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="<script> may not specify both length and post_gap"):
        asyncio.run(runner())


def test_script_length_must_be_non_negative(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult(audio=np.array([1.0, 2.0], dtype=np.float32))

            return Registered()

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script length="-1.0">Anna: Line 1</script>
                </production>
                """,
                source_name="negative-length.xml",
            )
            plan = await root.plan(ainjector)
            await plan.render()
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="<script> length must be non-negative seconds"):
        asyncio.run(runner())


def test_production_plan_installs_shared_vibevoice_resource(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>Anna: Line 1</script>
                  <script>Anna: Line 2</script>
                </production>
                """,
                source_name="shared-resource.xml",
            )
            plan = await root.plan(ainjector)
            resource_ids = {
                id(script_plan._registered_request.resource)
                for script_plan in plan.leaf_audio_plans()
                if isinstance(script_plan, ScriptPlan)
            }
            return resource_ids
        finally:
            injector.close()

    resource_ids = asyncio.run(runner())
    assert len(resource_ids) == 1


def test_production_plan_installs_shared_qwen_resource(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script tts="qwen">Anna: Line 1</script>
                  <script tts="qwen">Anna: Line 2</script>
                </production>
                """,
                source_name="shared-qwen-resource.xml",
            )
            plan = await root.plan(ainjector)
            resource_ids = {
                id(script_plan._registered_request.resource)
                for script_plan in plan.leaf_audio_plans()
                if isinstance(script_plan, ScriptPlan)
            }
            return resource_ids
        finally:
            injector.close()

    resource_ids = asyncio.run(runner())
    assert len(resource_ids) == 1


def test_vibevoice_resource_returns_production_format_audio(monkeypatch, tmp_path: Path):
    class FakeProcessor:
        def __init__(self):
            self.audio_processor = type("AudioProcessor", (), {"sampling_rate": 24000})()
            self.tokenizer = object()

        def __call__(self, **kwargs):
            return {"input_ids": np.array([1])}

    class FakeModel:
        def eval(self):
            return None

        def set_ddpm_inference_steps(self, num_steps: int):
            return None

        def generate(self, **kwargs):
            return type(
                "Outputs",
                (),
                {"speech_outputs": [np.ones(2400, dtype=np.float32)]},
            )()

    class FakeResource(VibeVoiceResource):
        def _ensure_loaded(self):
            self._sample_rate = 24000
            return FakeProcessor(), FakeModel()

        def _normalize_audio_array(self, audio):
            return np.asarray(audio, dtype=np.float32)

        def _preprocessed_voice_sample_sync(self, voice_path: Path, *, output_sample_rate: int):
            return np.zeros(240, dtype=np.float32)

    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=48000, output_channels=2)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await ainjector(FakeResource)
            registration = await resource.register_request(
                _request_from_normalized_script("Speaker 1: Hello", ("voice.wav",))
            )
            return await registration.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    assert result.audio.shape == (4800, 2)
    assert np.allclose(result.audio[:, 0], result.audio[:, 1])


def test_vibevoice_resource_prefixes_each_dialogue_paragraph_with_speaker(
    monkeypatch,
    tmp_path: Path,
):
    speaker = SpeakerVoiceReference(
        authored_name="Anna",
        voice_name="anna.wav",
        resolved_path=Path("anna.wav"),
    )

    class FakeProcessor:
        def __init__(self):
            self.audio_processor = type("AudioProcessor", (), {"sampling_rate": 24000})()
            self.tokenizer = object()
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return {"input_ids": np.array([1])}

    class FakeModel:
        def eval(self):
            return None

        def set_ddpm_inference_steps(self, num_steps: int):
            return None

        def generate(self, **kwargs):
            return type(
                "Outputs",
                (),
                {"speech_outputs": [np.ones(2400, dtype=np.float32)]},
            )()

    class FakeResource(VibeVoiceResource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.fake_processor = FakeProcessor()

        def _ensure_loaded(self):
            self._sample_rate = 24000
            return self.fake_processor, FakeModel()

        def _normalize_audio_array(self, audio):
            return np.asarray(audio, dtype=np.float32)

        def _preprocessed_voice_sample_sync(self, voice_path: Path, *, output_sample_rate: int):
            return np.zeros(240, dtype=np.float32)

    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=48000, output_channels=2)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await ainjector(FakeResource)
            registration = await resource.register_request(
                ScriptRenderRequest(
                    dialogue_lines=[
                        DialogueLine(
                            speaker=speaker,
                            spoken_text=(
                                "First paragraph line one.\n"
                                "Continuation.\n\n"
                                "Second paragraph."
                            ),
                        )
                    ]
                )
            )
            await registration.render()
            return resource.fake_processor.calls[0]["text"]
        finally:
            injector.close()

    text_inputs = asyncio.run(runner())
    assert text_inputs == [
        "Speaker 1: First paragraph line one. Continuation.\n"
        "Speaker 1: Second paragraph."
    ]


def test_qwen_resource_returns_production_format_audio(monkeypatch, tmp_path: Path):
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=48000, output_channels=2)

    class FakeResource(QwenTtsResource):
        def _ensure_loaded(self):
            self._sample_rate = 24000
            return object()

        def _prompt_items_by_voice_sync(self, references):
            return {
                str(reference.resolved_path.expanduser().resolve()): [object()]
                for reference in references
            }

        def _generate_line_batch_native_sync(self, texts, prompt_items):
            self._sample_rate = 24000
            return [np.ones(1200, dtype=np.float32) for _ in texts]

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await ainjector(FakeResource)
            registration = await resource.register_request(
                _request_from_normalized_script(
                    "Speaker 1: Hello\nSpeaker 2: There",
                    ("voice1.wav", "voice2.wav"),
                )
            )
            return await registration.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    assert result.audio.shape == (4800, 2)
    assert np.allclose(result.audio[:, 0], result.audio[:, 1])


def test_cached_vibevoice_double_stores_metadata_and_replays(cached_vibevoice_factory, tmp_path: Path):
    cache = cached_vibevoice_factory(mode="live", cache_dir=tmp_path / "cache", seed=7)
    request = _request_from_normalized_script("Speaker 1: Hello", ("anna.wav",))

    live_result = cache.render(
        request,
        producer=lambda _: CachedRenderMetadata(sample_rate=24000, frame_count=123),
    )
    replay = cached_vibevoice_factory(mode="cache", cache_dir=tmp_path / "cache", seed=7)
    replay_result = replay.render(request)

    assert live_result.audio.shape == (123,)
    assert replay_result.audio.shape == (123,)
    assert np.array_equal(live_result.audio, replay_result.audio)
