from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from carthage.dependency_injection import InjectionKey, inject

import radio_drama.effects as effects_module
from radio_drama.audio import AudioPlan, ComposeAudioPlan
from radio_drama.config import ProductionConfig
from radio_drama.dialogue import ScriptPlan, ScriptRenderRequest, TtsResource
from radio_drama.document import parse_production_string
from radio_drama.effects import (
    EffectChainRegistry,
    EffectMixer,
    EffectPipeline,
    effect_chain,
    effect_chain_function,
    effect_chain_variables,
    load_preprocessed_voice_reference,
    modulated_delay,
    numpy_stage,
)
from radio_drama.errors import DocumentError
from radio_drama.rendering import RenderResult
from radio_drama.sound import NormalizedSoundCache, SoundPlan
from radio_drama.vibevoice import VibeVoiceResource

from phase1_helpers import make_async_injector, normalized_script_from_request


def test_sound_plan_applies_gain_from_node(tmp_path: Path):
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
                asyncio.sleep(0, result=np.array([0.5, -0.5], dtype=np.float32))
            )

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <sound ref="door" gain="6.0206" />
                </production>
                """,
                source_name=str(xml_path),
            )
            plan = await root.child_elements_named("sound")[0].plan(ainjector)
            return await plan.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    np.testing.assert_allclose(result.audio, np.array([1.0, -1.0], dtype=np.float32), atol=1e-4)


def test_sound_plan_wraps_preset_from_node(tmp_path: Path):
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
                asyncio.sleep(0, result=np.array([0.5, -0.5], dtype=np.float32))
            )

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <sound ref="door" preset="narrator" />
                </production>
                """,
                source_name=str(xml_path),
            )
            return await root.child_elements_named("sound")[0].plan(ainjector)
        finally:
            injector.close()

    plan = asyncio.run(runner())
    assert isinstance(plan, SoundPlan)
    assert plan.preset_name == "narrator"
    assert plan.preset_key == ("narrator",)


def test_audio_plan_pan_expression_uses_render_time_marks(tmp_path: Path):
    config = ProductionConfig(output_sample_rate=4, output_channels=2)

    @inject(config=ProductionConfig)
    class FakeAudioPlan(AudioPlan):
        def __init__(self, result: RenderResult, **kwargs) -> None:
            super().__init__(node=None, **kwargs)
            self.result = result

        async def layout_node(self) -> None:
            self.inner_last = self._frames_to_seconds(self.result.frame_count)
            self.advance = self.inner_last
            self.mark_positions = {"cut": 0.25}

        async def render_node(self) -> RenderResult:
            return self.result

    async def runner():
        injector, ainjector = await make_async_injector(config)
        try:
            plan = await ainjector(
                FakeAudioPlan,
                result=RenderResult(audio=np.ones((4, 2), dtype=np.float32)),
                attrs={"pan": "line(cut, -1, cut + 0.25 * s + 0.25 * seconds, 1)"},
            )
            return await plan.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    np.testing.assert_allclose(result.audio[:, 0], np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(result.audio[:, 1], np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32))


def test_audio_plan_pan_expression_uses_linear_balance(tmp_path: Path):
    config = ProductionConfig(output_sample_rate=4, output_channels=2)

    @inject(config=ProductionConfig)
    class FakeAudioPlan(AudioPlan):
        def __init__(self, result: RenderResult, **kwargs) -> None:
            super().__init__(node=None, **kwargs)
            self.result = result

        async def layout_node(self) -> None:
            self.inner_last = self._frames_to_seconds(self.result.frame_count)
            self.advance = self.inner_last

        async def render_node(self) -> RenderResult:
            return self.result

    async def render_with_pan(pan_expression: str) -> RenderResult:
        injector, ainjector = await make_async_injector(config)
        try:
            plan = await ainjector(
                FakeAudioPlan,
                result=RenderResult(audio=np.ones((2, 2), dtype=np.float32)),
                attrs={"pan": pan_expression},
            )
            return await plan.render()
        finally:
            injector.close()

    right_heavy = asyncio.run(render_with_pan("0.5"))
    left_heavy = asyncio.run(render_with_pan("-0.5"))

    np.testing.assert_allclose(right_heavy.audio[:, 0], np.full(2, 0.5, dtype=np.float32))
    np.testing.assert_allclose(right_heavy.audio[:, 1], np.ones(2, dtype=np.float32))
    np.testing.assert_allclose(left_heavy.audio[:, 0], np.ones(2, dtype=np.float32))
    np.testing.assert_allclose(left_heavy.audio[:, 1], np.full(2, 0.5, dtype=np.float32))


def test_audio_plan_gain_expression_uses_render_time_marks(tmp_path: Path):
    config = ProductionConfig(output_sample_rate=4, output_channels=1)

    @inject(config=ProductionConfig)
    class FakeAudioPlan(AudioPlan):
        def __init__(self, result: RenderResult, **kwargs) -> None:
            super().__init__(node=None, **kwargs)
            self.result = result

        async def layout_node(self) -> None:
            self.inner_last = self._frames_to_seconds(self.result.frame_count)
            self.advance = self.inner_last
            self.mark_positions = {"cut": 0.25}

        async def render_node(self) -> RenderResult:
            return self.result

    async def runner():
        injector, ainjector = await make_async_injector(config)
        try:
            plan = await ainjector(
                FakeAudioPlan,
                result=RenderResult(audio=np.ones(4, dtype=np.float32)),
                attrs={"gain": "line(cut, -6, cut + 2, 0)"},
            )
            return await plan.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    expected = np.array(
        [1.0, 10.0 ** (-6.0 / 20.0), 10.0 ** (-3.0 / 20.0), 1.0],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result.audio, expected, atol=1e-6)


def test_audio_plan_effect_runs_between_gain_and_pan_in_a_thread(tmp_path: Path, monkeypatch):
    config = ProductionConfig(output_sample_rate=4, output_channels=2)
    calls: list[str] = []

    @effect_chain_function
    def test_post_render_effect():
        @numpy_stage
        def stage(audio: np.ndarray, sample_rate: int) -> None:
            del sample_rate
            calls.append("effect")
            audio += 1.0

        return stage

    original_to_thread = asyncio.to_thread

    async def recording_to_thread(func, /, *args, **kwargs):
        calls.append("thread")
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", recording_to_thread)

    @inject(config=ProductionConfig)
    class FakeAudioPlan(AudioPlan):
        def __init__(self, result: RenderResult, **kwargs) -> None:
            super().__init__(node=None, **kwargs)
            self.result = result

        async def layout_node(self) -> None:
            self.inner_last = self._frames_to_seconds(self.result.frame_count)
            self.advance = self.inner_last

        async def render_node(self) -> RenderResult:
            return self.result

    async def runner():
        injector, ainjector = await make_async_injector(config)
        try:
            plan = await ainjector(
                FakeAudioPlan,
                result=RenderResult(audio=np.ones((2, 2), dtype=np.float32)),
                attrs={"gain": "6.0206", "effect": "test_post_render_effect()", "pan": "1"},
            )
            return await plan.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    assert calls == ["thread", "effect"]
    np.testing.assert_allclose(result.audio[:, 0], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(result.audio[:, 1], np.full(2, 3.0, dtype=np.float32), atol=1e-4)


def test_effect_expression_can_use_gain_and_pan_stages(tmp_path: Path):
    config = ProductionConfig(output_sample_rate=4, output_channels=2)

    @inject(config=ProductionConfig)
    class FakeAudioPlan(AudioPlan):
        def __init__(self, result: RenderResult, **kwargs) -> None:
            super().__init__(node=None, **kwargs)
            self.result = result

        async def layout_node(self) -> None:
            self.inner_last = self._frames_to_seconds(self.result.frame_count)
            self.advance = self.inner_last

        async def render_node(self) -> RenderResult:
            return self.result

    async def runner():
        injector, ainjector = await make_async_injector(config)
        try:
            plan = await ainjector(
                FakeAudioPlan,
                result=RenderResult(audio=np.ones((2, 2), dtype=np.float32)),
                attrs={"effect": "gain(line(6.0206)) | pan(line(1))"},
            )
            return await plan.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    np.testing.assert_allclose(result.audio[:, 0], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(result.audio[:, 1], np.full(2, 2.0, dtype=np.float32), atol=1e-4)


def test_sound_plan_applies_pan_expression_from_node(tmp_path: Path):
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=2)

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == sound_file
            return asyncio.create_task(
                asyncio.sleep(0, result=np.ones((2, 2), dtype=np.float32))
            )

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <sound ref="door" pan="1" />
                </production>
                """,
                source_name=str(xml_path),
            )
            sound_plan = await root.child_elements_named("sound")[0].plan(ainjector)
            return await sound_plan.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    np.testing.assert_allclose(result.audio[:, 0], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(result.audio[:, 1], np.ones(2, dtype=np.float32))


def test_preset_bus_preserves_script_timeline_length(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=2)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    value = float(normalized_script_from_request(request)[-1])
                    return RenderResult(audio=np.full((16, 2), value, dtype=np.float32))

            return Registered()

    async def runner():
        injector, ainjector = await make_async_injector(config)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
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
            return await plan.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    assert result.audio.shape == (33, 2)


def test_nested_script_presets_keep_outer_compose_scope(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    value = 1.0 if "Inner line" in normalized_script_from_request(request) else 2.0
                    return RenderResult(audio=np.array([value], dtype=np.float32))

            return Registered()

    async def runner():
        injector, ainjector = await make_async_injector(config)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script preset="indoor1">
                    <script preset="narrator">Anna: Inner line.</script>
                    Anna: Outer line.
                  </script>
                </production>
                """,
                source_name="nested-preset-scope.xml",
            )
            return await root.plan(ainjector)
        finally:
            injector.close()

    production_plan = asyncio.run(runner())
    outer_plan = production_plan.audio_plans[0]
    assert isinstance(outer_plan, ComposeAudioPlan)
    assert outer_plan.preset_key == ("indoor1",)
    assert len(outer_plan.audio_plans) == 2
    assert outer_plan.audio_plans[0].preset_key == ("narrator",)
    assert outer_plan.audio_plans[1].preset_key == ()


def test_same_preset_regions_share_one_compose_bus():
    call_count = 0

    class CountingStage:
        def apply(self, audio: np.ndarray, *, sample_rate: int) -> None:
            nonlocal call_count
            del sample_rate
            call_count += 1
            audio += 1.0

    @effect_chain_function
    def test_counting_stage():
        return EffectPipeline((CountingStage(),))

    async def runner():
        config = ProductionConfig(output_sample_rate=4, output_channels=1)
        injector, ainjector = await make_async_injector(config)
        try:
            registry = EffectChainRegistry()
            registry.add_from_expression("narrator", "test_counting_stage()")
            injector.replace_provider(InjectionKey(EffectChainRegistry), registry, close=False)
            mixer = await ainjector(
                EffectMixer,
                total_frames=2,
                channels=1,
            )
            mixer.add(
                start_frame=0,
                end_frame=1,
                audio=np.array([1.0], dtype=np.float32),
                preset_key=("narrator",),
            )
            mixer.add(
                start_frame=1,
                end_frame=2,
                audio=np.array([2.0], dtype=np.float32),
                preset_key=("narrator",),
            )
            return await mixer.apply(sample_rate=4)
        finally:
            injector.close()

    result = asyncio.run(runner())
    assert call_count == 1
    assert result.tolist() == [2.0, 3.0]


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
        injector, ainjector = await make_async_injector(config)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
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
            rendered = await production_plan.render()
            return audio_plan, rendered
        finally:
            injector.close()

    audio_plan, rendered = asyncio.run(runner())
    assert isinstance(audio_plan, ScriptPlan)
    assert audio_plan.preset_name == "narrator"
    assert audio_plan.preset_key == ("narrator",)
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
        injector, ainjector = await make_async_injector(config)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
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


def test_named_effect_chains_include_demo_presets():
    registry = EffectChainRegistry()
    assert {
        "indoor1",
        "indoor2",
        "master",
        "narrator",
        "narrator_nofocus",
        "outdoor1",
        "outdoor2",
        "phone",
        "thoughts",
    }.issubset(registry.names())
    assert registry["Narrator"] is registry["narrator"]
    assert registry["Narrator1"] is registry["narrator"]
    assert registry["Narrator2"] is registry["thoughts"]


def test_effect_chain_function_decorator_adds_factory_to_expression_variables():
    @effect_chain_function
    def test_stage(amount: float):
        del amount
        return EffectPipeline(())

    assert effect_chain_variables()["test_stage"] is test_stage


def test_modulated_delay_mixes_dry_and_delayed_audio():
    audio = np.zeros(8, dtype=np.float32)
    audio[0] = 1.0
    stage = modulated_delay(
        delay_ms=2.0,
        depth_ms=0.0,
        rate_hz=0.0,
        wet_mix=0.25,
        dry_mix=0.5,
    )

    stage.apply(audio, sample_rate=1000)

    np.testing.assert_allclose(
        audio,
        np.array([0.5, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_modulated_delay_uses_stereo_phase_offset():
    audio = np.column_stack(
        [np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32)]
    )
    stage = modulated_delay(
        delay_ms=2.0,
        depth_ms=1.0,
        rate_hz=0.0,
        wet_mix=1.0,
        dry_mix=0.0,
        stereo_phase_degrees=90.0,
    )

    stage.apply(audio, sample_rate=1000)

    np.testing.assert_allclose(audio[4], np.array([2.0, 1.0], dtype=np.float32))


def test_modulated_delay_rejects_noncausal_depth():
    with pytest.raises(ValueError, match="depth_ms must not exceed delay_ms"):
        modulated_delay(
            delay_ms=2.0,
            depth_ms=3.0,
            rate_hz=0.1,
            wet_mix=0.2,
        )


def test_effect_chain_variables_include_presets_and_aliases():
    registry = EffectChainRegistry()
    variables = effect_chain_variables(registry.stages())

    assert variables["phone"] is registry["phone"]
    assert variables["narrator1"] is registry["narrator"]
    assert "ffmpeg_filter_stage" not in variables
    assert "voice_loudnorm" not in variables


def test_effect_chain_registry_evaluates_replacements_in_order():
    registry = EffectChainRegistry()

    original_phone = registry["phone"]
    registry.add_from_expression("extended", "phone | filter_audio(btype='lowpass', cutoff_hz=2000)")
    registry.add_from_expression("phone", "extended")

    extended = registry["extended"]
    assert isinstance(extended, EffectPipeline)
    assert extended.stages[:-1] == original_phone.stages
    assert registry["phone"] is extended


def test_effect_chain_registry_evaluates_alias_replacements_in_order():
    registry = EffectChainRegistry()

    registry.add_from_expression("narrator1", "phone")
    registry.add_from_expression("custom_narrator", "narrator1")

    assert registry["custom_narrator"] is registry["phone"]
    assert registry["narrator"] is registry["phone"]


def test_effect_chain_registry_rejects_effect_function_names():
    registry = EffectChainRegistry()

    with pytest.raises(ValueError, match="reserved for an effect function"):
        registry.add_from_expression("filter_audio", "phone")


def test_preset_map_is_processed_before_audio_plans(tmp_path: Path):
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        try:
            root = parse_production_string(
                """
                <production>
                  <sound ref="door" preset="custom_phone" />
                  <preset-map>
                    custom_phone: phone | filter_audio(btype="lowpass", cutoff_hz=2000)
                    phone: custom_phone
                    master: custom_phone
                  </preset-map>
                </production>
                """,
                source_name=str(xml_path),
            )
            plan = await root.plan(ainjector)
            registry = plan.effect_chains
            return plan.audio_plans[0], registry
        finally:
            injector.close()

    sound_plan, registry = asyncio.run(runner())
    assert sound_plan.preset_name == "custom_phone"
    assert registry["phone"] is registry["custom_phone"]
    assert registry["master"] is registry["custom_phone"]


def test_production_rejects_more_than_one_preset_map():
    async def runner():
        config = ProductionConfig()
        injector, ainjector = await make_async_injector(config)
        try:
            root = parse_production_string(
                """
                <production>
                  <preset-map>one: phone</preset-map>
                  <preset-map>two: indoor1</preset-map>
                </production>
                """,
                source_name="duplicate-preset-map.xml",
            )
            await root.plan(ainjector)
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="only one <preset-map>"):
        asyncio.run(runner())


def test_effect_chain_rejects_non_stage_expression_results():
    with pytest.raises(TypeError, match="effect chain"):
        effect_chain(1)


def test_load_preprocessed_voice_reference_downmixes_and_resamples(
    monkeypatch,
    tmp_path: Path,
):
    voice_path = tmp_path / "voice.wav"
    sf.write(
        voice_path,
        np.full((4, 2), [0.0, 1.0], dtype=np.float32),
        4,
    )

    class FakeStage:
        def apply(self, audio: np.ndarray, *, sample_rate: int) -> None:
            assert sample_rate == 4
            audio += 0.25

    monkeypatch.setattr(
        effects_module,
        "build_voice_preprocess_chain",
        lambda: FakeStage(),
    )
    monkeypatch.setattr(
        effects_module,
        "resample_audio",
        lambda audio, *, input_sample_rate, output_sample_rate: np.repeat(audio, 2),
    )

    audio, sample_rate = load_preprocessed_voice_reference(
        voice_path,
        output_sample_rate=8,
    )

    assert sample_rate == 8
    assert audio.shape == (8,)
    np.testing.assert_allclose(audio, np.full(8, 0.75, dtype=np.float32), atol=1e-4)


def test_master_effect_chain_preserves_output_format():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    chain = EffectChainRegistry()["master"]
    audio = np.linspace(-0.2, 0.2, 1024, dtype=np.float32)
    stereo_audio = np.column_stack([audio, audio])
    chain.apply(stereo_audio, sample_rate=48000)

    assert stereo_audio.shape == (1024, 2)
    assert stereo_audio.dtype == np.float32
