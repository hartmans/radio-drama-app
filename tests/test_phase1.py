from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from carthage.dependency_injection import AsyncInjector, InjectionKey, Injector

from radio_drama.audio import convert_audio_format
from radio_drama.config import MODEL_NATIVE_SAMPLE_RATE, ProductionConfig
from radio_drama.document import parse_production_string
from radio_drama.errors import DocumentError
from radio_drama.effects import PresetPlan, available_effect_chains, build_named_effect_chain
from radio_drama.forced_alignment import AlignedScriptSource, ScriptSlice, WhisperXResource
from radio_drama.init import radio_drama_injector
from radio_drama.planning import ComposeAudioPlan, DialogueAudio, ScriptRenderRequest
from radio_drama.rendering import ProductionResult, RenderResult
from radio_drama.resources import VibeVoiceResource
from radio_drama.sound import NormalizedSoundCache
from radio_drama.testing import CachedRenderMetadata


async def _make_async_injector(
    config: ProductionConfig,
    *,
    document_path: Path | None = None,
) -> tuple[Injector, AsyncInjector]:
    injector = radio_drama_injector(
        config=config,
        event_loop=asyncio.get_running_loop(),
        document_path=document_path,
    )
    return injector, injector(AsyncInjector)


def test_speaker_map_plan_resolves_stem_lookup(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>
                    Anna: anna
                  </speaker-map>
                  <script>Anna: Hello.</script>
                </production>
                """,
                source_name="test.xml",
            )
            return await root.speaker_map_node.plan(ainjector)
        finally:
            injector.close()

    plan = asyncio.run(runner())
    assert plan.lookup("ANNA").resolved_path == voice_file
    assert plan.lookup("anna").authored_name == "Anna"


def test_script_plan_allows_stanzas_and_paragraph_fill(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult.empty(channels=2)

            return Registered()

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    Anna: First sentence.
                    Continued line.

                    Another paragraph.
                  </script>
                </production>
                """,
                source_name="stanza.xml",
            )
            speaker_map_plan = await root.speaker_map_node.plan(ainjector)
            injector.add_provider(InjectionKey(type(speaker_map_plan)), speaker_map_plan, close=False)
            return await root.script_nodes[0].plan(ainjector)
        finally:
            injector.close()

    script_plan = asyncio.run(runner())
    assert len(script_plan.dialogue_lines) == 1
    assert script_plan.dialogue_lines[0].spoken_text == (
        "First sentence. Continued line. Another paragraph."
    )


def test_script_plan_allows_empty_script(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult.empty(channels=2)

            return Registered()

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>

                  </script>
                </production>
                """,
                source_name="empty-script.xml",
            )
            speaker_map_plan = await root.speaker_map_node.plan(ainjector)
            injector.add_provider(InjectionKey(type(speaker_map_plan)), speaker_map_plan, close=False)
            script_plan = await root.script_nodes[0].plan(ainjector)
            return script_plan.render_request, await script_plan.render()
        finally:
            injector.close()

    render_request, render_result = asyncio.run(runner())
    assert render_request is None
    assert render_result.frame_count == 0
    assert render_result.channel_count == 2


def test_script_with_sound_builds_script_slices_from_aligned_source(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(voice_directory=tmp_path)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult.empty(channels=2)

            return Registered()

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            return contents

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == sound_file
            return asyncio.create_task(
                asyncio.sleep(0, result=np.zeros((0, 2), dtype=np.float32))
            )

    async def runner():
        injector, ainjector = await _make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
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
            first_slice = audio_plan.audio_plans[0]
            dialogue_audio = next(
                content
                for content in first_slice.aligned_script_source.script_plan.contents
                if isinstance(content, DialogueAudio)
            )
            sound_result = await dialogue_audio.audio_plan.render()
            return audio_plan, sound_result
        finally:
            injector.close()

    audio_plan, sound_result = asyncio.run(runner())
    assert isinstance(audio_plan, ComposeAudioPlan)
    assert [type(child).__name__ for child in audio_plan.audio_plans] == [
        "ScriptSlice",
        "SoundPlan",
        "ScriptSlice",
    ]
    first_slice = audio_plan.audio_plans[0]
    second_slice = audio_plan.audio_plans[2]
    assert isinstance(first_slice, ScriptSlice)
    assert isinstance(second_slice, ScriptSlice)
    assert first_slice.aligned_script_source is second_slice.aligned_script_source
    assert isinstance(first_slice.aligned_script_source, AlignedScriptSource)
    assert [type(content).__name__ for content in first_slice.aligned_script_source.script_plan.contents] == [
        "DialogueLine",
        "DialogueAudio",
        "DialogueLine",
    ]
    assert [line.spoken_text for line in first_slice.aligned_script_source.script_plan.dialogue_lines] == [
        "First line.",
        "Response.",
    ]
    assert first_slice.aligned_script_source.script_plan.render_request.normalized_script == (
        "Speaker 1: First line.\nSpeaker 2: Response."
    )
    assert sound_result.frame_count == 0
    assert sound_result.channel_count == 2


def test_sound_plan_prefers_shallowest_relative_match(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    exact_match = tmp_path / "sounds" / "foley" / "door.wav"
    deeper_match = tmp_path / "sounds" / "archive" / "foley" / "door.wav"
    exact_match.parent.mkdir(parents=True, exist_ok=True)
    deeper_match.parent.mkdir(parents=True, exist_ok=True)
    exact_match.write_bytes(b"exact")
    deeper_match.write_bytes(b"deeper")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            return asyncio.create_task(
                asyncio.sleep(0, result=np.full(2, float(len(sound_path.parts)), dtype=np.float32))
            )

    async def runner():
        injector, ainjector = await _make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    Anna: Open it.
                    <sound ref="foley/door" />
                  </script>
                </production>
                """,
                source_name=str(xml_path),
            )
            sound_plan = await root.script_nodes[0].child_elements_named("sound")[0].plan(ainjector)
            await sound_plan.render()
            return sound_plan.resolved_path
        finally:
            injector.close()

    resolved_path = asyncio.run(runner())
    assert resolved_path == exact_match


def test_sound_plan_rejects_ambiguous_relative_matches(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    left_match = tmp_path / "sounds" / "left" / "door.wav"
    right_match = tmp_path / "sounds" / "right" / "door.wav"
    left_match.parent.mkdir(parents=True, exist_ok=True)
    right_match.parent.mkdir(parents=True, exist_ok=True)
    left_match.write_bytes(b"left")
    right_match.write_bytes(b"right")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            return asyncio.create_task(asyncio.sleep(0, result=np.zeros(1, dtype=np.float32)))

    async def runner():
        injector, ainjector = await _make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    Anna: Open it.
                    <sound ref="door" />
                  </script>
                </production>
                """,
                source_name=str(xml_path),
            )
            await root.script_nodes[0].child_elements_named("sound")[0].plan(ainjector)
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="matched multiple files"):
        asyncio.run(runner())


def test_sound_plan_follows_symlinked_sound_directories(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    external_bank = tmp_path / "external-bank"
    external_file = external_bank / "chime.wav"
    external_file.parent.mkdir(parents=True, exist_ok=True)
    external_file.write_bytes(b"chime")
    (tmp_path / "sounds").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sounds" / "library").symlink_to(external_bank, target_is_directory=True)
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            return asyncio.create_task(asyncio.sleep(0, result=np.ones(1, dtype=np.float32)))

    async def runner():
        injector, ainjector = await _make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    Anna: Listen.
                    <sound ref="library/chime" />
                  </script>
                </production>
                """,
                source_name=str(xml_path),
            )
            sound_plan = await root.script_nodes[0].child_elements_named("sound")[0].plan(ainjector)
            await sound_plan.render()
            return sound_plan.resolved_path
        finally:
            injector.close()

    resolved_path = asyncio.run(runner())
    assert resolved_path.resolve() == external_file.resolve()


def test_normalized_sound_cache_reuses_shared_sound_path(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class CountingSoundCache(NormalizedSoundCache):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.normalize_call_count = 0

        def _normalize_sound_sync(self, sound_path: Path):
            self.normalize_call_count += 1
            assert sound_path == sound_file.resolve()
            return np.array([1.0, 2.0], dtype=np.float32)

    async def runner():
        injector, ainjector = await _make_async_injector(config, document_path=xml_path)
        try:
            cache = await ainjector(CountingSoundCache)
            injector.replace_provider(InjectionKey(NormalizedSoundCache), cache, close=False)
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    Anna: Open it.
                    <sound ref="door" />
                    <sound ref="door" />
                  </script>
                </production>
                """,
                source_name=str(xml_path),
            )
            sound_nodes = root.script_nodes[0].child_elements_named("sound")
            sound_plans = [await sound_node.plan(ainjector) for sound_node in sound_nodes]
            await asyncio.gather(*(sound_plan.render() for sound_plan in sound_plans))
            return cache.normalize_call_count
        finally:
            injector.close()

    normalize_call_count = asyncio.run(runner())
    assert normalize_call_count == 1


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
        async def fill_start_positions(self, contents, result):
            updated = []
            for index, content in enumerate(contents):
                if isinstance(content, DialogueAudio):
                    updated.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=0.5))
                else:
                    updated.append(
                        type(content)(
                            speaker=content.speaker,
                            spoken_text=content.spoken_text,
                            start_pos=float(index),
                        )
                    )
            return updated

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == sound_file
            return asyncio.create_task(
                asyncio.sleep(0, result=np.zeros((0, 2), dtype=np.float32))
            )

    async def runner():
        injector, ainjector = await _make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
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
    assert list(result.marker_frames) == [0, 2, 4]
    assert [content.start_pos for content in aligned_source.contents] == [0.0, 0.5, 2.0]


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
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
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


def test_script_plan_rejects_non_speaker_prefix(tmp_path: Path):
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
                  <script>
                    This should fail.
                  </script>
                </production>
                """,
                source_name="bad-script.xml",
            )
            speaker_map_plan = await root.speaker_map_node.plan(ainjector)
            injector.add_provider(InjectionKey(type(speaker_map_plan)), speaker_map_plan, close=False)
            await root.script_nodes[0].plan(ainjector)
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="Scripts may begin only with a recognized `speaker:` stanza"):
        asyncio.run(runner())


def test_convert_audio_format_resamples_and_upmixes():
    source_audio = np.ones(2400, dtype=np.float32)
    result = convert_audio_format(
        source_audio,
        input_sample_rate=24000,
        output_sample_rate=48000,
        output_channels=2,
    )
    assert result.shape == (4800, 2)
    assert np.allclose(result[:, 0], result[:, 1])


def test_render_result_from_time_returns_shared_slice():
    base_audio = np.arange(20, dtype=np.float32).reshape(10, 2)
    result = RenderResult(audio=base_audio)

    sliced = result.from_time(0.25, sample_rate=8)

    assert sliced.audio.shape == (8, 2)
    assert np.array_equal(sliced.audio, base_audio[2:])
    assert np.shares_memory(result.audio, sliced.audio)


def test_vibevoice_resource_batches_concurrent_requests(monkeypatch, tmp_path: Path):
    class FakeBatchingResource(VibeVoiceResource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.batch_sizes: list[int] = []
            self._sample_rate = MODEL_NATIVE_SAMPLE_RATE

        def _render_batch_sync(self, batch):
            self.batch_sizes.append(len(batch))
            return [
                np.full(index + 1, fill_value=index + 1, dtype=np.float32)
                for index, _ in enumerate(batch)
            ]

    config = ProductionConfig(voice_directory=tmp_path)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await ainjector(FakeBatchingResource)
            requests = [
                await resource.register_request(
                    ScriptRenderRequest(
                    normalized_script=f"Speaker 1: Line {index + 1}",
                    voice_samples=("voice.wav",),
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
    assert [result.audio.tolist() for result in results] == [[1.0], [2.0, 2.0]]


def test_production_plan_renders_scripts_in_order(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=24000, output_channels=1)

    class FakeVibeVoice:
        def empty_result(self) -> RenderResult:
            return RenderResult.empty(channels=1)

        async def register_request(self, request: ScriptRenderRequest):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    value = float(request.normalized_script[-1])
                    return RenderResult(audio=np.array([value], dtype=np.float32))

            return Registered()

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>Anna: Line 1</script>
                  <script>Anna: Line 2</script>
                </production>
                """,
                source_name="ordered.xml",
            )
            plan = await root.plan(ainjector)
            return plan, await plan.audio_plan.render()
        finally:
            injector.close()

    plan, result = asyncio.run(runner())
    assert isinstance(plan, PresetPlan)
    assert plan.preset_name == "master"
    assert result.audio.tolist() == [1.0, 2.0]


def test_production_plan_applies_script_gaps(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    value = float(request.normalized_script[-1])
                    return RenderResult(audio=np.array([value], dtype=np.float32))

            return Registered()

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script post_gap="0.5">Anna: Line 1</script>
                  <script pre_gap="0.25">Anna: Line 2</script>
                </production>
                """,
                source_name="script-gaps.xml",
            )
            plan = await root.plan(ainjector)
            return plan, await plan.audio_plan.render()
        finally:
            injector.close()

    plan, result = asyncio.run(runner())
    assert isinstance(plan, PresetPlan)
    assert plan.preset_name == "master"
    assert result.audio.tolist() == [1.0, 0.0, 0.0, 0.0, 2.0]


def test_production_plan_mixes_overlapping_scripts(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    if request.normalized_script.endswith("1"):
                        return RenderResult(audio=np.array([1.0, 2.0], dtype=np.float32))
                    return RenderResult(audio=np.array([10.0, 20.0], dtype=np.float32))

            return Registered()

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script length="0">Anna: Line 1</script>
                  <script>Anna: Line 2</script>
                </production>
                """,
                source_name="overlap.xml",
            )
            plan = await root.plan(ainjector)
            return plan, await plan.audio_plan.render()
        finally:
            injector.close()

    plan, result = asyncio.run(runner())
    assert isinstance(plan, PresetPlan)
    assert isinstance(plan.audio_plan, ComposeAudioPlan)
    assert result.audio.tolist() == [11.0, 22.0]


def test_production_plan_trims_audio_before_zero(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult(audio=np.array([1.0, 2.0], dtype=np.float32))

            return Registered()

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script pre_gap="-0.25">Anna: Line 1</script>
                </production>
                """,
                source_name="trim-start.xml",
            )
            plan = await root.plan(ainjector)
            return await plan.audio_plan.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    assert result.audio.tolist() == [2.0]


def test_production_plan_trims_audio_after_end(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult(audio=np.array([1.0, 2.0], dtype=np.float32))

            return Registered()

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script post_gap="-0.25">Anna: Line 1</script>
                </production>
                """,
                source_name="trim-end.xml",
            )
            plan = await root.plan(ainjector)
            return await plan.audio_plan.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    assert result.audio.tolist() == [1.0]


def test_preset_plan_preserves_inner_script_gap(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=2)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    value = float(request.normalized_script[-1])
                    return RenderResult(audio=np.full((16, 2), value, dtype=np.float32))

            return Registered()

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script preset="narrator" post_gap="0.25">Anna: Line 1</script>
                  <script>Anna: Line 2</script>
                </production>
                """,
                source_name="preset-gap.xml",
            )
            plan = await root.plan(ainjector)
            return plan, await plan.audio_plan.render()
        finally:
            injector.close()

    plan, result = asyncio.run(runner())
    assert isinstance(plan, PresetPlan)
    assert isinstance(plan.audio_plan, ComposeAudioPlan)
    assert result.audio.shape == (33, 2)
    assert np.allclose(result.audio[16], 0.0)


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

    async def runner():
        injector, ainjector = await _make_async_injector(config)
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
            await root.plan(ainjector)
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="<script> length must be non-negative seconds"):
        asyncio.run(runner())


def test_script_margins_cannot_be_set_on_nodes(tmp_path: Path):
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
                  <script pre_margin="0.5">Anna: Line 1</script>
                </production>
                """,
                source_name="pre-margin.xml",
            )
            await root.plan(ainjector)
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="<script> pre_margin cannot be set on document nodes"):
        asyncio.run(runner())


def test_script_preset_wraps_script_plan_and_applies_effects(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=24000, output_channels=2)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    audio = np.tile(
                        np.linspace(-0.2, 0.2, 512, dtype=np.float32)[:, np.newaxis],
                        (1, 2),
                    )
                    return RenderResult(audio=audio)

            return Registered()

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>
                    Anna: anna.wav
                    Ben: anna.wav
                  </speaker-map>
                  <script preset="narrator">
                    Anna: I can hear the thought.
                    Ben: Then say it out loud.
                  </script>
                </production>
                """,
                source_name="preset.xml",
            )
            production_plan = await root.plan(ainjector)
            audio_plan = production_plan.audio_plans[0]
            rendered = await audio_plan.render()
            return production_plan, audio_plan, rendered
        finally:
            injector.close()

    production_plan, audio_plan, rendered = asyncio.run(runner())
    assert isinstance(audio_plan, PresetPlan)
    assert audio_plan.preset_name == "narrator"
    assert len(production_plan.leaf_audio_plans()) == 1
    assert rendered.audio.shape == (512, 2)
    assert not np.allclose(rendered.audio[:, 0], np.linspace(-0.2, 0.2, 512, dtype=np.float32))


def test_unknown_preset_raises_document_error(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=24000, output_channels=2)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    audio = np.ones((128, 2), dtype=np.float32) * 0.05
                    return RenderResult(audio=audio)

            return Registered()

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script preset="missing-preset">Anna: Hello.</script>
                </production>
                """,
                source_name="missing-preset.xml",
            )
            plan = await root.plan(ainjector)
            await plan.audio_plans[0].render()
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="Unknown preset 'missing-preset'"):
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
                for script_plan in plan.script_plans
            }
            return resource_ids
        finally:
            injector.close()

    resource_ids = asyncio.run(runner())
    assert len(resource_ids) == 1


def test_named_effect_chains_include_demo_presets():
    assert available_effect_chains() == (
        "indoor1",
        "indoor2",
        "master",
        "narrator",
        "outdoor1",
        "outdoor2",
        "thoughts",
    )
    assert build_named_effect_chain("Narrator").name == "narrator"
    assert build_named_effect_chain("Narrator1").name == "narrator"
    assert build_named_effect_chain("Narrator2").name == "thoughts"


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
                    normalized_script="Speaker 1: Hello",
                    voice_samples=("voice.wav",),
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
    request = ScriptRenderRequest(
        normalized_script="Speaker 1: Hello",
        voice_samples=("anna.wav",),
    )

    live_result = cache.render(
        request,
        producer=lambda _: CachedRenderMetadata(sample_rate=24000, frame_count=123),
    )
    replay = cached_vibevoice_factory(mode="cache", cache_dir=tmp_path / "cache", seed=7)
    replay_result = replay.render(request)

    assert live_result.audio.shape == (123,)
    assert replay_result.audio.shape == (123,)
    assert np.array_equal(live_result.audio, replay_result.audio)
