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
from radio_drama.init import radio_drama_injector
from radio_drama.planning import ConcatAudioPlan, ScriptRenderRequest
from radio_drama.rendering import ProductionResult, RenderResult
from radio_drama.resources import VibeVoiceResource
from radio_drama.testing import CachedRenderMetadata


async def _make_async_injector(config: ProductionConfig) -> tuple[Injector, AsyncInjector]:
    injector = radio_drama_injector(
        config=config,
        event_loop=asyncio.get_running_loop(),
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
            return await plan.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
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
            return await plan.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    assert result.audio.tolist() == [1.0, 0.0, 0.0, 0.0, 2.0]


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
            return plan, await plan.render()
        finally:
            injector.close()

    plan, result = asyncio.run(runner())
    assert isinstance(plan, ConcatAudioPlan)
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
