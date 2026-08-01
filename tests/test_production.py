from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from carthage.dependency_injection import InjectionKey

from radio_drama.config import ProductionConfig
from radio_drama.dialogue import DialogueAudio, ScriptRenderRequest, TtsResource
from radio_drama.document import parse_production_string
from radio_drama.effects import EffectChainRegistry, EffectPipeline, effect_chain_function
from radio_drama.errors import DocumentError
from radio_drama.forced_alignment import WhisperXResource
from radio_drama.production import ProductionPlan
from radio_drama.rendering import RenderResult
from radio_drama.vibevoice import VibeVoiceResource

from phase1_helpers import make_async_injector as _make_async_injector, normalized_script_from_request


@effect_chain_function
def identity_stage_for_tests():
    return EffectPipeline(())


def _noop_effect_chains() -> EffectChainRegistry:
    registry = EffectChainRegistry()
    for preset_name in registry.names():
        registry.add_from_expression(preset_name, "identity_stage_for_tests()")
    return registry


async def make_async_injector(config: ProductionConfig, **kwargs):
    kwargs.setdefault("effect_chains", _noop_effect_chains())
    return await _make_async_injector(config, **kwargs)


@pytest.fixture
def noop_effect_chains() -> EffectChainRegistry:
    return _noop_effect_chains()


def test_cut_before_mark_on_production_can_target_inner_script(tmp_path: Path, noop_effect_chains):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(
        voice_directory=tmp_path,
        output_sample_rate=4,
        output_channels=1,
    )

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult(audio=np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))

            return Registered()

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            updated: list[DialogueAudio | object] = []
            for content in contents:
                if isinstance(content, DialogueAudio):
                    updated.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=0.5))
                else:
                    updated.append(content)
            return updated

    async def runner():
        injector, ainjector = await make_async_injector(config, effect_chains=noop_effect_chains)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script preset="narrator">
                    Anna: First line.
                    <mark id="cut" />
                    Anna: Second line.
                  </script>
                </production>
                """,
                source_name="cut-mark.xml",
            )
            production_plan = await root.plan(ainjector)
            production_plan.cut_before_mark("cut")
            return production_plan.mark_names, await production_plan.render()
        finally:
            injector.close()

    audio_marks, result = asyncio.run(runner())
    assert audio_marks == ["cut"]
    assert result.audio.tolist() == [2.0, 2.0]


def test_cut_before_mark_on_production_can_target_script_first_mark(tmp_path: Path, noop_effect_chains):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(
        voice_directory=tmp_path,
        output_sample_rate=4,
        output_channels=1,
    )

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult(audio=np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))

            return Registered()

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            updated: list[DialogueAudio | object] = []
            for content in contents:
                if isinstance(content, DialogueAudio):
                    updated.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=0.5))
                else:
                    updated.append(content)
            return updated

    async def runner():
        injector, ainjector = await make_async_injector(config, effect_chains=noop_effect_chains)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script first_mark="brennan_office" preset="narrator">
                    Anna: First line.
                    <mark id="later" />
                    Anna: Second line.
                  </script>
                </production>
                """,
                source_name="cut-script-first-mark.xml",
            )
            production_plan = await root.plan(ainjector)
            production_plan.cut_before_mark("brennan_office")
            return production_plan.mark_names, await production_plan.render()
        finally:
            injector.close()

    audio_marks, result = asyncio.run(runner())
    assert audio_marks == ["brennan_office", "later"]
    assert result.audio.tolist() == [1.0, 1.0, 2.0, 2.0]


def test_cut_after_mark_on_production_can_target_inner_script(tmp_path: Path, noop_effect_chains):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(
        voice_directory=tmp_path,
        output_sample_rate=4,
        output_channels=1,
    )

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult(audio=np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))

            return Registered()

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            updated: list[DialogueAudio | object] = []
            for content in contents:
                if isinstance(content, DialogueAudio):
                    updated.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=0.5))
                else:
                    updated.append(content)
            return updated

    async def runner():
        injector, ainjector = await make_async_injector(config, effect_chains=noop_effect_chains)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script preset="narrator">
                    Anna: First line.
                    <mark id="cut" />
                    Anna: Second line.
                  </script>
                </production>
                """,
                source_name="cut-after-mark.xml",
            )
            production_plan = await root.plan(ainjector)
            production_plan.cut_after_mark("cut")
            return production_plan.mark_names, await production_plan.render()
        finally:
            injector.close()

    audio_marks, result = asyncio.run(runner())
    assert audio_marks == ["cut"]
    assert result.audio.tolist() == [1.0, 1.0]


def test_cut_after_mark_on_production_can_target_script_last_mark(tmp_path: Path, noop_effect_chains):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(
        voice_directory=tmp_path,
        output_sample_rate=4,
        output_channels=1,
    )

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult(audio=np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))

            return Registered()

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            updated: list[DialogueAudio | object] = []
            for content in contents:
                if isinstance(content, DialogueAudio):
                    updated.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=0.5))
                else:
                    updated.append(content)
            return updated

    async def runner():
        injector, ainjector = await make_async_injector(config, effect_chains=noop_effect_chains)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script last_mark="brennan_office" preset="narrator">
                    Anna: First line.
                    <mark id="earlier" />
                    Anna: Second line.
                  </script>
                </production>
                """,
                source_name="cut-script-last-mark.xml",
            )
            production_plan = await root.plan(ainjector)
            production_plan.cut_after_mark("brennan_office")
            return production_plan.mark_names, await production_plan.render()
        finally:
            injector.close()

    audio_marks, result = asyncio.run(runner())
    assert audio_marks == ["brennan_office", "earlier"]
    assert result.audio.tolist() == [1.0, 1.0, 2.0, 2.0]


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
                    value = float(normalized_script_from_request(request)[-1])
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
                  <script>Anna: Line 1</script>
                  <script>Anna: Line 2</script>
                </production>
                """,
                source_name="ordered.xml",
            )
            plan = await root.plan(ainjector)
            return plan, await plan.render()
        finally:
            injector.close()

    plan, result = asyncio.run(runner())
    assert isinstance(plan, ProductionPlan)
    assert result.audio.tolist() == [1.0, 2.0]


def test_production_plan_applies_script_gaps(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    value = float(normalized_script_from_request(request)[-1])
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
                  <script post_gap="0.5">Anna: Line 1</script>
                  <script pre_gap="0.25">Anna: Line 2</script>
                </production>
                """,
                source_name="script-gaps.xml",
            )
            plan = await root.plan(ainjector)
            return plan, await plan.render()
        finally:
            injector.close()

    plan, result = asyncio.run(runner())
    assert isinstance(plan, ProductionPlan)
    assert result.audio.tolist() == [1.0, 0.0, 0.0, 0.0, 2.0]


def test_production_plan_mixes_overlapping_scripts(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    if normalized_script_from_request(request).endswith("1"):
                        return RenderResult(audio=np.array([1.0, 2.0], dtype=np.float32))
                    return RenderResult(audio=np.array([10.0, 20.0], dtype=np.float32))

            return Registered()

    async def runner():
        injector, ainjector = await make_async_injector(config)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script length="natural_length - natural_length">Anna: Line 1</script>
                  <script>Anna: Line 2</script>
                </production>
                """,
                source_name="overlap.xml",
            )
            plan = await root.plan(ainjector)
            return plan, await plan.render()
        finally:
            injector.close()

    plan, result = asyncio.run(runner())
    assert isinstance(plan, ProductionPlan)
    assert result.audio.tolist() == [11.0, 22.0]


def test_script_length_expression_must_resolve_non_negative(tmp_path: Path):
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
        injector, ainjector = await make_async_injector(config)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script length="0 - natural_length">Anna: Line 1</script>
                </production>
                """,
                source_name="negative-length-expression.xml",
            )
            plan = await root.plan(ainjector)
            await plan.render()
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="<script> length must be non-negative seconds"):
        asyncio.run(runner())


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
        injector, ainjector = await make_async_injector(config)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
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
            return await plan.render()
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
        injector, ainjector = await make_async_injector(config)
        injector.replace_provider(InjectionKey(TtsResource, tts="vibevoice"), FakeVibeVoice(), close=False)
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
            return await plan.render()
        finally:
            injector.close()

    result = asyncio.run(runner())
    assert result.audio.tolist() == [1.0]
