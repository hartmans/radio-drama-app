from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from carthage.dependency_injection import InjectionKey, inject

import radio_drama.effects as effects_module
from radio_drama.audio import AudioPlan, ComposeAudioPlan, LoopPlan, MarkPlan, convert_audio_format
from radio_drama.config import ProductionConfig
from radio_drama.dialogue import DialogueAudio, DialogueLine, ScriptRenderRequest
from radio_drama.document import parse_production_string
from radio_drama.effects import EffectPipeline
from radio_drama.errors import DocumentError
from radio_drama.forced_alignment import WhisperXResource
from radio_drama.rendering import RenderResult
from radio_drama.sound import NormalizedSoundCache, SoundPlan
from radio_drama.vibevoice import VibeVoiceResource

from phase1_helpers import make_async_injector, normalized_script_from_request


def test_production_with_direct_sound_renders_without_speaker_map(tmp_path: Path):
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == sound_file
            return asyncio.create_task(asyncio.sleep(0, result=np.array([1.0, 2.0], dtype=np.float32)))

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <sound ref="door" />
                </production>
                """,
                source_name=str(xml_path),
            )
            plan = await root.plan(ainjector)
            return await plan.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    assert result.audio.tolist() == [1.0, 2.0]


def test_sound_plan_defers_normalization_until_render(tmp_path: Path):
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeSoundCache:
        def __init__(self) -> None:
            self.preloaded_paths: list[Path] = []

        async def preload(self, sound_path: Path):
            self.preloaded_paths.append(sound_path)
            return asyncio.create_task(asyncio.sleep(0, result=np.array([1.0, 2.0], dtype=np.float32)))

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        sound_cache = FakeSoundCache()
        injector.replace_provider(InjectionKey(NormalizedSoundCache), sound_cache, close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <sound ref="door" />
                </production>
                """,
                source_name=str(xml_path),
            )
            plan = await root.plan(ainjector)
            preload_count_before_render = len(sound_cache.preloaded_paths)
            await plan.render()
            return preload_count_before_render, sound_cache.preloaded_paths
        finally:
            injector.close()

    preload_count_before_render, preloaded_paths = asyncio.run(runner())
    assert preload_count_before_render == 0
    assert preloaded_paths == [sound_file]


def test_sound_plan_trims_audio_with_from_and_to(tmp_path: Path):
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == sound_file
            return asyncio.create_task(
                asyncio.sleep(0, result=np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float32))
            )

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <sound ref="door" from="0.25" to="1.0" />
                </production>
                """,
                source_name=str(xml_path),
            )
            plan = await root.child_elements_named("sound")[0].plan(ainjector)
            return plan, await plan.render()
        finally:
            injector.close()

    plan, result = asyncio.run(runner())
    assert isinstance(plan, SoundPlan)
    assert plan.inner_first == 0.0
    assert plan.natural_length == 0.75
    np.testing.assert_array_equal(result.audio, np.array([11.0, 12.0, 13.0], dtype=np.float32))


def test_sound_plan_keeps_trim_attrs_on_inner_sound_when_wrapped(tmp_path: Path):
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == sound_file
            return asyncio.create_task(asyncio.sleep(0, result=np.array([1.0, 2.0], dtype=np.float32)))

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <sound ref="door" from="0.25" to="1.0" preset="narrator" />
                </production>
                """,
                source_name=str(xml_path),
            )
            return await root.child_elements_named("sound")[0].plan(ainjector)
        finally:
            injector.close()

    plan = asyncio.run(runner())
    assert isinstance(plan, SoundPlan)
    assert plan.file_from == 0.25
    assert plan.file_to == 1.0
    assert "from" in plan.attrs
    assert "to" in plan.attrs
    assert "preset" in plan.attrs


def test_sound_plan_rejects_to_before_from(tmp_path: Path):
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            return asyncio.create_task(asyncio.sleep(0, result=np.ones(1, dtype=np.float32)))

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <sound ref="door" from="1.0" to="0.25" />
                </production>
                """,
                source_name=str(xml_path),
            )
            await root.child_elements_named("sound")[0].plan(ainjector)
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="<sound> to must be greater than or equal to from"):
        asyncio.run(runner())


def test_explicit_start_sound_trim_can_use_rebased_inner_marks(tmp_path: Path):
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    leadin_file = tmp_path / "sounds" / "leadin.wav"
    door_file = tmp_path / "sounds" / "door.wav"
    leadin_file.parent.mkdir(parents=True, exist_ok=True)
    leadin_file.write_bytes(b"leadin")
    door_file.write_bytes(b"door")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            if sound_path == leadin_file:
                return asyncio.create_task(
                    asyncio.sleep(0, result=np.array([1.0, 2.0], dtype=np.float32))
                )
            assert sound_path == door_file
            return asyncio.create_task(
                asyncio.sleep(0, result=np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float32))
            )

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <sound ref="leadin" last_mark="cut" />
                  <sound ref="door" start="0" from="inner_cut" to="inner_cut + 0.5" />
                </production>
                """,
                source_name=str(xml_path),
            )
            compose_plan = await root.plan(ainjector)
            target_sound_plan = compose_plan.audio_plans[1]
            await compose_plan.layout()
            return target_sound_plan, await target_sound_plan.render()
        finally:
            injector.close()

    plan, result = asyncio.run(runner())
    assert isinstance(plan, SoundPlan)
    assert plan.file_from == pytest.approx(0.5)
    assert plan.file_to == pytest.approx(1.0)
    assert plan.natural_length == pytest.approx(0.5)
    np.testing.assert_array_equal(result.audio, np.array([12.0, 13.0], dtype=np.float32))


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
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
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
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
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
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
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


def test_sound_plan_prefers_configured_sounds_directory(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    configured_sounds = tmp_path / "example_sounds"
    configured_file = configured_sounds / "court" / "gavel.wav"
    configured_file.parent.mkdir(parents=True, exist_ok=True)
    configured_file.write_bytes(b"gavel")
    config = ProductionConfig(
        voice_directory=tmp_path,
        sounds_directory=configured_sounds,
        output_sample_rate=4,
        output_channels=1,
    )

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == configured_file
            return asyncio.create_task(asyncio.sleep(0, result=np.ones(1, dtype=np.float32)))

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    Anna: Order.
                    <sound ref="court/gavel" />
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
    assert resolved_path == configured_file


def test_mark_plan_emits_zero_length_audio_and_one_mark(tmp_path: Path):
    config = ProductionConfig(output_sample_rate=4, output_channels=1)

    async def runner():
        injector, ainjector = await make_async_injector(config)
        try:
            root = parse_production_string(
                """
                <production>
                  <mark id="cut" />
                </production>
                """,
                source_name="mark.xml",
            )
            mark_plan = await root.child_elements_named("mark")[0].plan(ainjector)
            return mark_plan, await mark_plan.render()
        finally:
            injector.close()

    mark_plan, result = asyncio.run(runner())
    assert isinstance(mark_plan, MarkPlan)
    assert mark_plan.audio_marks == ["cut"]
    assert result.frame_count == 0
    assert mark_plan.audio_marks_render == {"cut": 0.0}


def test_compose_audio_plan_hides_ambiguous_marks(tmp_path: Path):
    config = ProductionConfig(output_sample_rate=4, output_channels=1)

    async def runner():
        injector, ainjector = await make_async_injector(config)
        try:
            root = parse_production_string(
                """
                <production>
                  <mark id="cut" />
                  <mark>cut</mark>
                </production>
                """,
                source_name="mark-ambiguity.xml",
            )
            audio_plans = [await child.plan(ainjector) for child in root.child_elements_named("mark")]
            compose_plan = await ainjector(ComposeAudioPlan, node=root, audio_plans=audio_plans)
            return compose_plan, await compose_plan.render()
        finally:
            injector.close()

    compose_plan, _ = asyncio.run(runner())
    assert compose_plan.audio_marks == []
    assert compose_plan.audio_marks_render == {}
    with pytest.raises(ValueError, match="Unknown or ambiguous audio mark 'cut'"):
        compose_plan.cut_before_mark("cut")
    with pytest.raises(ValueError, match="Unknown or ambiguous audio mark 'cut'"):
        compose_plan.cut_after_mark("cut")


def test_compose_audio_plan_bubbles_render_time_mark_positions(tmp_path: Path):
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "tone.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"tone")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == sound_file
            return asyncio.create_task(asyncio.sleep(0, result=np.array([1.0, 1.0], dtype=np.float32)))

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <mark id="start" />
                  <sound ref="tone" />
                  <mark id="after" />
                </production>
                """,
                source_name=str(xml_path),
            )
            compose_plan = await root.plan(ainjector)
            result = await compose_plan.render()
            return compose_plan, result
        finally:
            injector.close()

    audio_plan, result = asyncio.run(runner())
    assert result.audio.tolist() == [1.0, 1.0]
    assert audio_plan.audio_marks_render == {"start": 0.0, "after": 2.0}


def test_loop_plan_repeats_region_suppresses_loop_marks_and_shifts_outro(tmp_path: Path):
    config = ProductionConfig(output_sample_rate=4, output_channels=1)

    @inject(config=ProductionConfig)
    class FakeAudioPlan(AudioPlan):
        def __init__(self, result: RenderResult, **kwargs) -> None:
            super().__init__(node=None, **kwargs)
            self.result = result

        async def layout_node(self) -> None:
            self._raw_inner_last = self._frames_to_seconds(self.result.frame_count)
            self._raw_length = self._raw_inner_last
            self._layout_marks_inner = {"pre": 0.25, "beg": 0.5, "mid": 0.75, "end": 1.0, "out": 1.25}

        async def render_node(self) -> RenderResult:
            return self.result

    async def runner():
        injector, ainjector = await make_async_injector(config)
        try:
            plan = await ainjector(
                FakeAudioPlan,
                result=RenderResult(audio=np.arange(6, dtype=np.float32)),
                attrs={
                    "loop_beg": "0.5",
                    "loop_end": "1.0",
                    "loop_loops": 1.0,
                    "loop_silence": 0.25,
                    "loop_outro": True,
                },
            )
            assert isinstance(plan, LoopPlan)
            result = await plan.render()
            return plan, result
        finally:
            injector.close()

    plan, result = asyncio.run(runner())
    np.testing.assert_allclose(
        result.audio,
        np.array([0.0, 1.0, 2.0, 3.0, 0.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
    )
    assert plan.audio_marks_inner == {"pre": 0.25, "beg": 0.5, "end": 1.0, "out": 2.0}
    assert plan.audio_marks_render == {"pre": 1.0, "beg": 2.0, "end": 4.0, "out": 8.0}
    assert "mid" not in plan.audio_marks_inner


def test_loop_plan_loop_until_whole_extend_adjusts_loop_stop(tmp_path: Path):
    config = ProductionConfig(output_sample_rate=4, output_channels=1)

    @inject(config=ProductionConfig)
    class FakeAudioPlan(AudioPlan):
        def __init__(self, result: RenderResult, **kwargs) -> None:
            super().__init__(node=None, **kwargs)
            self.result = result

        async def layout_node(self) -> None:
            self._raw_inner_last = self._frames_to_seconds(self.result.frame_count)
            self._raw_length = self._raw_inner_last

        async def render_node(self) -> RenderResult:
            return self.result

    async def runner():
        injector, ainjector = await make_async_injector(config)
        try:
            plan = await ainjector(
                FakeAudioPlan,
                result=RenderResult(audio=np.arange(6, dtype=np.float32)),
                attrs={
                    "loop_beg": "0.5",
                    "loop_end": "1.0",
                    "loop_until": "1.6",
                    "loop_silence": 0.25,
                    "loop_whole": "extend",
                },
            )
            await plan.layout()
            return plan
        finally:
            injector.close()

    plan = asyncio.run(runner())
    assert plan.resolved_loop_stop == pytest.approx(1.75)
    assert plan.inner_last == pytest.approx(1.75)


def test_loop_plan_loop_until_can_target_later_automatic_mark(tmp_path: Path):
    config = ProductionConfig(output_sample_rate=4, output_channels=1)

    @inject(config=ProductionConfig)
    class FakeAudioPlan(AudioPlan):
        def __init__(self, result: RenderResult, **kwargs) -> None:
            super().__init__(node=None, **kwargs)
            self.result = result

        async def layout_node(self) -> None:
            self._raw_inner_last = self._frames_to_seconds(self.result.frame_count)
            self._raw_length = self._raw_inner_last

        async def render_node(self) -> RenderResult:
            return self.result

    async def runner():
        injector, ainjector = await make_async_injector(config)
        try:
            root = parse_production_string("<production />", source_name="loop-until-later-mark.xml")
            loop_plan = await ainjector(
                FakeAudioPlan,
                result=RenderResult(audio=np.arange(6, dtype=np.float32)),
                attrs={"start": "0", "loop_beg": "0.5", "loop_end": "1.0", "loop_until": "inner_later"},
            )
            later_plan = await ainjector(
                FakeAudioPlan,
                result=RenderResult(audio=np.arange(16, dtype=np.float32)),
                attrs={"last_mark": "later"},
            )
            compose_plan = await ainjector(ComposeAudioPlan, node=root, audio_plans=[loop_plan, later_plan])
            await compose_plan.layout()
            return loop_plan, compose_plan
        finally:
            injector.close()

    loop_plan, compose_plan = asyncio.run(runner())
    assert loop_plan.resolved_loop_stop == pytest.approx(4.0)
    assert loop_plan.inner_last == pytest.approx(4.0)
    assert compose_plan.audio_marks == ["later"]


def test_compose_audio_debug_logs_placement_spans(tmp_path: Path):
    config = ProductionConfig(
        output_sample_rate=4,
        output_channels=1,
        debug_log_path=tmp_path / "render.wav.log",
        debug_categories=("compose_audio",),
    )

    @inject(config=ProductionConfig)
    class FakeAudioPlan(AudioPlan):
        def __init__(self, label: str, result: RenderResult, **kwargs) -> None:
            super().__init__(node=None, **kwargs)
            self.label = label
            self.result = result

        def __repr__(self) -> str:
            return f"FakeAudioPlan({self.label!r})"

        async def layout_node(self) -> None:
            self._raw_inner_last = self._frames_to_seconds(self.result.frame_count)
            self._raw_length = self._raw_inner_last

        async def render_node(self) -> RenderResult:
            return self.result

    async def runner():
        injector, ainjector = await make_async_injector(config)
        try:
            first = await ainjector(
                FakeAudioPlan,
                label="first",
                result=RenderResult(audio=np.array([1.0, 2.0], dtype=np.float32)),
            )
            second = await ainjector(
                FakeAudioPlan,
                label="second",
                result=RenderResult(audio=np.array([3.0], dtype=np.float32)),
                attrs={"pre_gap": "0.25"},
            )
            compose_plan = await ainjector(
                ComposeAudioPlan,
                node=parse_production_string("<production />", source_name="compose.xml"),
                audio_plans=[first, second],
            )
            await compose_plan.render()
        finally:
            injector.close()

    asyncio.run(runner())
    log_text = config.debug_log_path.read_text(encoding="utf-8")
    assert "FakeAudioPlan('first')" in log_text
    assert "FakeAudioPlan('second')" in log_text
    assert "0.000s to 0.500s" in log_text
    assert "0.750s to 1.000s" in log_text


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
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
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
