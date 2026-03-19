from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from radio_drama.carthage_support import AsyncInjector, InjectionKey, Injector
from radio_drama.document import parse_production_string
from radio_drama.errors import DocumentError
from radio_drama.phase1 import (
    OutputFormatResource,
    ProductionConfig,
    ScriptPlan,
    ScriptRenderRequest,
    VibeVoiceResource,
)
from radio_drama.rendering import ProductionResult, RenderResult
from radio_drama.testing import CachedRenderMetadata


async def _make_async_injector(config: ProductionConfig) -> tuple[Injector, AsyncInjector]:
    injector = Injector()
    injector.add_provider(config)
    injector.replace_provider(
        InjectionKey(asyncio.AbstractEventLoop),
        asyncio.get_running_loop(),
        close=False,
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


def test_script_plan_rejects_non_speaker_lines(tmp_path: Path):
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
            root.script_nodes[0].plan(speaker_map_plan)
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="Scripts may contain only `speaker: text` lines"):
        asyncio.run(runner())


def test_output_format_resource_resamples_and_upmixes(tmp_path: Path):
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=48000, output_channels=2)

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await ainjector(OutputFormatResource)
            source_audio = np.ones(2400, dtype=np.float32)
            return resource.convert(RenderResult(audio=source_audio, sample_rate=24000))
        finally:
            injector.close()

    result = asyncio.run(runner())
    assert result.sample_rate == 48000
    assert result.audio.shape == (4800, 2)
    assert np.allclose(result.audio[:, 0], result.audio[:, 1])


def test_vibevoice_resource_batches_concurrent_requests(monkeypatch, tmp_path: Path):
    class FakeBatchingResource(VibeVoiceResource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.batch_sizes: list[int] = []
            self._sample_rate = 24000

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
                ScriptPlan(
                    node=root.script_nodes[0],
                    dialogue_lines=[],
                    ordered_speakers=[],
                    render_request=ScriptRenderRequest(
                        normalized_script=f"Speaker 1: Line {index + 1}",
                        voice_samples=("voice.wav",),
                    ),
                )
                for index, root in enumerate(
                    [
                        parse_production_string(
                            """
                            <production>
                              <speaker-map>Anna: anna.wav</speaker-map>
                              <script>Anna: One.</script>
                            </production>
                            """
                        ),
                        parse_production_string(
                            """
                            <production>
                              <speaker-map>Anna: anna.wav</speaker-map>
                              <script>Anna: Two.</script>
                            </production>
                            """
                        ),
                    ]
                )
            ]
            results = await asyncio.gather(*(resource.render_script(request) for request in requests))
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
        async def render_script(self, script_plan: ScriptPlan) -> RenderResult:
            value = float(script_plan.render_request.normalized_script[-1])
            return RenderResult(audio=np.array([value], dtype=np.float32), sample_rate=24000)

    class FakeOutputFormat:
        def convert(self, render_result: RenderResult) -> ProductionResult:
            return ProductionResult(audio=render_result.audio, sample_rate=render_result.sample_rate)

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.add_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        injector.add_provider(InjectionKey(OutputFormatResource), FakeOutputFormat(), close=False)
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

    assert live_result.sample_rate == 24000
    assert live_result.audio.shape == (123,)
    assert replay_result.sample_rate == 24000
    assert replay_result.audio.shape == (123,)
    assert np.array_equal(live_result.audio, replay_result.audio)
