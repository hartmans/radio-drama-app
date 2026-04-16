from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from carthage.dependency_injection import InjectionKey

from radio_drama.audio import ComposeAudioPlan
from radio_drama.config import ProductionConfig
from radio_drama.dialogue import AudioScriptPlan, DialogueAudio, DialogueLine, ScriptGap, ScriptRenderRequest
from radio_drama.document import parse_production_string
from radio_drama.errors import DocumentError
from radio_drama.forced_alignment import AlignedScriptSource, ScriptSlice, WhisperXResource
from radio_drama.qwen_tts import QwenTtsResource
from radio_drama.rendering import RenderResult
from radio_drama.sound import NormalizedSoundCache
from radio_drama.vibevoice import VibeVoiceResource

from phase1_helpers import make_async_injector, normalized_script_from_request


def test_speaker_map_plan_resolves_stem_lookup(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    async def runner():
        injector, ainjector = await make_async_injector(config)
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
        injector, ainjector = await make_async_injector(config)
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
    dialogue_lines = [
        content for content in script_plan.dialogue_contents if isinstance(content, DialogueLine)
    ]
    assert len(dialogue_lines) == 1
    assert dialogue_lines[0].spoken_text == (
        "First sentence. Continued line. Another paragraph."
    )


def test_script_plan_routes_qwen_scripts_to_qwen_resource(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            raise AssertionError("qwen scripts should not use VibeVoiceResource")

    class FakeQwen:
        def __init__(self):
            self.requests = []

        async def register_request(self, request: ScriptRenderRequest | None):
            self.requests.append(request)

            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult.empty(channels=2)

            return Registered()

    async def runner():
        injector, ainjector = await make_async_injector(config)
        fake_qwen = FakeQwen()
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(QwenTtsResource), fake_qwen, close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script tts="qwen">Anna: Hello.</script>
                </production>
                """,
                source_name="qwen-script.xml",
            )
            speaker_map_plan = await root.speaker_map_node.plan(ainjector)
            injector.add_provider(InjectionKey(type(speaker_map_plan)), speaker_map_plan, close=False)
            script_plan = await root.script_nodes[0].plan(ainjector)
            return fake_qwen, script_plan
        finally:
            injector.close()

    fake_qwen, script_plan = asyncio.run(runner())
    assert len(fake_qwen.requests) == 1
    assert script_plan.node.tts == "qwen"


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
        injector, ainjector = await make_async_injector(config)
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


def test_script_plan_reports_missing_speaker_map(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    async def runner():
        injector, ainjector = await make_async_injector(config)
        try:
            root = parse_production_string(
                """
                <production>
                  <script>Anna: Hello.</script>
                </production>
                """,
                source_name="missing-speaker-map.xml",
            )
            await root.plan(ainjector)
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="requires a <speaker-map> to be planned before it"):
        asyncio.run(runner())


def test_production_plan_rejects_duplicate_speaker_maps(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    async def runner():
        injector, ainjector = await make_async_injector(config)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <speaker-map>Ben: anna.wav</speaker-map>
                </production>
                """,
                source_name="duplicate-speaker-map.xml",
            )
            await root.plan(ainjector)
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="may contain only one <speaker-map>"):
        asyncio.run(runner())


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
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
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
                for content in first_slice.aligned_script_source.script_plan.script_events
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
    assert [type(content).__name__ for content in first_slice.aligned_script_source.script_plan.script_events] == [
        "DialogueLine",
        "DialogueAudio",
        "DialogueLine",
    ]
    assert [
        content.spoken_text
        for content in first_slice.aligned_script_source.script_plan.dialogue_contents
        if isinstance(content, DialogueLine)
    ] == [
        "First line.",
        "Response.",
    ]
    assert normalized_script_from_request(first_slice.aligned_script_source.script_plan.render_request) == (
        "Speaker 1: First line.\nSpeaker 2: Response."
    )
    assert sound_result.frame_count == 0
    assert sound_result.channel_count == 2


def test_sound_script_renders_wrapped_sound_without_tts_registration(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(
        voice_directory=tmp_path,
        output_sample_rate=4,
        output_channels=1,
    )
    base_audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            raise AssertionError("sound-script should not register a speech render request")

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == sound_file
            return asyncio.create_task(asyncio.sleep(0, result=base_audio))

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <sound-script ref="door" from="natural_length / 2">
                    Anna: Transcript only.
                  </sound-script>
                </production>
                """,
                source_name=str(xml_path),
            )
            speaker_map_plan = await root.speaker_map_node.plan(ainjector)
            injector.add_provider(InjectionKey(type(speaker_map_plan)), speaker_map_plan, close=False)
            script_plan = await root.script_nodes[0].plan(ainjector)
            return script_plan, await script_plan.render()
        finally:
            injector.close()

    script_plan, render_result = asyncio.run(runner())
    assert isinstance(script_plan, AudioScriptPlan)
    assert render_result.audio.tolist() == pytest.approx([0.3, 0.4])


def test_sound_script_gap_aligns_against_wrapped_sound_audio(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(
        voice_directory=tmp_path,
        output_sample_rate=4,
        output_channels=1,
    )
    base_audio = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            raise AssertionError("sound-script should not register a speech render request")

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            assert result.audio.tolist() == pytest.approx(base_audio.tolist())
            assert [type(content).__name__ for content in contents] == [
                "DialogueLine",
                "ScriptGap",
                "DialogueLine",
            ]
            updated = []
            for content in contents:
                if isinstance(content, DialogueLine):
                    updated.append(
                        DialogueLine(
                            speaker=content.speaker,
                            spoken_text=content.spoken_text,
                            handling=content.handling,
                            node=content.node,
                            start_pos=0.0 if not updated else 0.5,
                        )
                    )
                else:
                    updated.append(ScriptGap(label=content.label, start_pos=0.25))
            return updated

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == sound_file
            return asyncio.create_task(asyncio.sleep(0, result=base_audio))

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <sound-script ref="door">
                    Anna: Before the missing material.
                    <script-gap />
                    Anna: After the missing material.
                  </sound-script>
                </production>
                """,
                source_name=str(xml_path),
            )
            speaker_map_plan = await root.speaker_map_node.plan(ainjector)
            injector.add_provider(InjectionKey(type(speaker_map_plan)), speaker_map_plan, close=False)
            production_plan = await root.plan(ainjector)
            audio_plan = production_plan.audio_plans[0]
            first_slice = audio_plan.audio_plans[0]
            return audio_plan, await first_slice.aligned_script_source.render(), await audio_plan.render()
        finally:
            injector.close()

    audio_plan, aligned_result, render_result = asyncio.run(runner())
    assert isinstance(audio_plan, ComposeAudioPlan)
    assert [type(child).__name__ for child in audio_plan.audio_plans] == ["ScriptSlice"]
    assert isinstance(audio_plan.audio_plans[0].aligned_script_source.script_plan, AudioScriptPlan)
    assert [type(content).__name__ for content in aligned_result.contents] == [
        "DialogueLine",
        "ScriptGap",
        "DialogueLine",
    ]
    assert [content.start_pos for content in aligned_result.contents] == [0.0, 0.25, 0.5]
    assert render_result.audio.tolist() == pytest.approx(base_audio.tolist())


def test_script_with_ignore_discards_guidance_audio(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(
        voice_directory=tmp_path,
        output_sample_rate=4,
        output_channels=1,
    )

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            assert request is not None
            assert normalized_script_from_request(request) == (
                "Speaker 1: Settle into a calm, reflective tone.\n"
                "Speaker 1: The actual line starts here."
            )

            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult(audio=np.arange(12, dtype=np.float32))

            return Registered()

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            updated: list[DialogueAudio | DialogueLine] = []
            for index, content in enumerate(contents):
                if isinstance(content, DialogueLine):
                    updated.append(
                        DialogueLine(
                            speaker=content.speaker,
                            spoken_text=content.spoken_text,
                            handling=content.handling,
                            start_pos=0.0 if index == 0 else 1.0,
                        )
                    )
                else:
                    updated.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=content.start_pos))
            return updated

    async def runner():
        injector, ainjector = await make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    <ignore>Anna: Settle into a calm, reflective tone.</ignore>
                    Anna: The actual line starts here.
                  </script>
                </production>
                """,
                source_name="ignore-script.xml",
            )
            speaker_map_plan = await root.speaker_map_node.plan(ainjector)
            injector.add_provider(InjectionKey(type(speaker_map_plan)), speaker_map_plan, close=False)
            script_audio_plan = await root.script_nodes[0].plan(ainjector)
            return script_audio_plan, await script_audio_plan.render()
        finally:
            injector.close()

    script_audio_plan, result = asyncio.run(runner())
    assert isinstance(script_audio_plan, ComposeAudioPlan)
    assert [type(child).__name__ for child in script_audio_plan.audio_plans] == ["ScriptSlice"]
    assert result.audio.tolist() == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]


def test_script_line_without_audio_attrs_does_not_create_extra_script_slice(tmp_path: Path):
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
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    Anna: First line.
                    <line speaker="Anna">Second line.</line>
                    <sound ref="door" />
                    Anna: Third line.
                  </script>
                </production>
                """,
                source_name=str(xml_path),
            )
            production_plan = await root.plan(ainjector)
            return production_plan.audio_plans[0]
        finally:
            injector.close()

    audio_plan = asyncio.run(runner())
    assert isinstance(audio_plan, ComposeAudioPlan)
    assert [type(child).__name__ for child in audio_plan.audio_plans] == [
        "ScriptSlice",
        "SoundPlan",
        "ScriptSlice",
    ]


def test_script_line_audio_attrs_create_special_script_slice(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult.empty(channels=2)

            return Registered()

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            updated: list[DialogueAudio | DialogueLine] = []
            for index, content in enumerate(contents):
                if isinstance(content, DialogueLine):
                    updated.append(
                        DialogueLine(
                            speaker=content.speaker,
                            spoken_text=content.spoken_text,
                            handling=content.handling,
                            node=content.node,
                            start_pos=float(index),
                        )
                    )
                else:
                    updated.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=content.start_pos))
            return updated

    async def runner():
        injector, ainjector = await make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    Anna: First line.
                    <line speaker="Anna" gain="6.0206" post_gap="0.25">Second line.</line>
                    Anna: Third line.
                  </script>
                </production>
                """,
                source_name="line-audio-attrs.xml",
            )
            production_plan = await root.plan(ainjector)
            return production_plan.audio_plans[0]
        finally:
            injector.close()

    audio_plan = asyncio.run(runner())
    assert isinstance(audio_plan, ComposeAudioPlan)
    assert [type(child).__name__ for child in audio_plan.audio_plans] == [
        "ScriptSlice",
        "ScriptSlice",
        "ScriptSlice",
    ]
    special_slice = audio_plan.audio_plans[1]
    assert isinstance(special_slice, ScriptSlice)
    assert special_slice.node.display_name == "<line>"
    assert special_slice.gain_expression == "6.0206"


def test_script_line_boundary_marks_create_special_slice_and_bubble(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult(audio=np.arange(12, dtype=np.float32))

            return Registered()

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            updated: list[DialogueAudio | DialogueLine] = []
            line_index = 0
            for content in contents:
                if isinstance(content, DialogueLine):
                    updated.append(
                        DialogueLine(
                            speaker=content.speaker,
                            spoken_text=content.spoken_text,
                            handling=content.handling,
                            node=content.node,
                            start_pos=float(line_index),
                        )
                    )
                    line_index += 1
                else:
                    updated.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=content.start_pos))
            return updated

    async def runner():
        injector, ainjector = await make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    Anna: First line.
                    <line speaker="Anna" first_mark="enter" last_mark="exit">Second line.</line>
                    Anna: Third line.
                  </script>
                </production>
                """,
                source_name="line-boundary-marks.xml",
            )
            production_plan = await root.plan(ainjector)
            special_slice = production_plan.audio_plans[0].audio_plans[1]
            return production_plan, special_slice, await special_slice.render()
        finally:
            injector.close()

    production_plan, special_slice, result = asyncio.run(runner())
    assert isinstance(special_slice, ScriptSlice)
    assert production_plan.audio_marks == ["enter", "exit"]
    assert special_slice.audio_marks == ["enter", "exit"]
    assert special_slice.audio_marks_render == {"enter": 0.0, "exit": 4.0}
    np.testing.assert_allclose(result.audio, np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))


def test_script_group_attrs_create_special_script_slices(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult.empty(channels=2)

            return Registered()

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            updated: list[DialogueAudio | DialogueLine] = []
            for index, content in enumerate(contents):
                if isinstance(content, DialogueLine):
                    updated.append(
                        DialogueLine(
                            speaker=content.speaker,
                            spoken_text=content.spoken_text,
                            handling=content.handling,
                            node=content.node,
                            start_pos=float(index),
                        )
                    )
                else:
                    updated.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=content.start_pos))
            return updated

    async def runner():
        injector, ainjector = await make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    Anna: First line.
                    <group gain="6.0206" preset="thoughts">
                      Anna: Second line.
                      Anna: Third line.
                    </group>
                    Anna: Fourth line.
                  </script>
                </production>
                """,
                source_name="group-audio-attrs.xml",
            )
            production_plan = await root.plan(ainjector)
            return production_plan.audio_plans[0]
        finally:
            injector.close()

    audio_plan = asyncio.run(runner())
    assert isinstance(audio_plan, ComposeAudioPlan)
    assert [type(child).__name__ for child in audio_plan.audio_plans] == [
        "ScriptSlice",
        "ScriptSlice",
        "ScriptSlice",
        "ScriptSlice",
    ]
    first_group_slice = audio_plan.audio_plans[1]
    second_group_slice = audio_plan.audio_plans[2]
    assert isinstance(first_group_slice, ScriptSlice)
    assert isinstance(second_group_slice, ScriptSlice)
    assert first_group_slice.node.display_name == "<group>"
    assert second_group_slice.node.display_name == "<group>"
    assert first_group_slice.gain_expression == "6.0206"
    assert second_group_slice.gain_expression == "6.0206"
    assert first_group_slice.preset_name == "thoughts"
    assert second_group_slice.preset_name == "thoughts"


def test_script_plan_rejects_non_speaker_prefix(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(voice_directory=tmp_path)

    async def runner():
        injector, ainjector = await make_async_injector(config)
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

    with pytest.raises(DocumentError, match="Scripts may begin only with a recognized `speaker:` stanza") as excinfo:
        asyncio.run(runner())
    assert excinfo.value.location is not None
    assert excinfo.value.location.line == 4


def test_sound_script_missing_speaker_after_gap_reports_chunk_start(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    config = ProductionConfig(voice_directory=tmp_path)

    class FakeSoundCache:
        async def preload(self, sound_path: Path):
            assert sound_path == sound_file
            return asyncio.create_task(asyncio.sleep(0, result=np.zeros(4, dtype=np.float32)))

    async def runner():
        injector, ainjector = await make_async_injector(config, document_path=xml_path)
        injector.replace_provider(InjectionKey(NormalizedSoundCache), FakeSoundCache(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <sound-script ref="door">
                    Anna: First line.
                    <script-gap />
                    Missing speaker here.
                  </sound-script>
                </production>
                """,
                source_name="bad-sound-script.xml",
            )
            speaker_map_plan = await root.speaker_map_node.plan(ainjector)
            injector.add_provider(InjectionKey(type(speaker_map_plan)), speaker_map_plan, close=False)
            await root.script_nodes[0].plan(ainjector)
        finally:
            injector.close()

    with pytest.raises(DocumentError, match="Scripts may begin only with a recognized `speaker:` stanza") as excinfo:
        asyncio.run(runner())
    assert excinfo.value.location is not None
    assert excinfo.value.location.source == "bad-sound-script.xml"
    assert excinfo.value.location.line == 6
