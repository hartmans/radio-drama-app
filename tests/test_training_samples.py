from __future__ import annotations

import asyncio
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from carthage.dependency_injection import AsyncInjector, InjectionKey

from radio_drama.config import ProductionConfig
from radio_drama.dialogue import DialogueLine, SpeakerVoiceReference
from radio_drama.document import parse_production_string
from radio_drama.forced_alignment import WhisperXResource
from radio_drama.init import radio_drama_injector
from radio_drama.rendering import RenderResult, ScriptRenderResult
from radio_drama.training_samples import chunk_training_intervals, export_training_samples_from_plan
from radio_drama.vibevoice import VibeVoiceResource

from phase1_helpers import make_async_injector


def test_chunk_training_intervals_combines_same_speaker_and_omits_mixed():
    alice = SpeakerVoiceReference(
        authored_name="Alice",
        voice_name="alice.wav",
        resolved_path=Path("alice.wav"),
    )
    bob = SpeakerVoiceReference(
        authored_name="Bob",
        voice_name="bob.wav",
        resolved_path=Path("bob.wav"),
    )

    combined = chunk_training_intervals(
        [
            DialogueLine(speaker=alice, spoken_text="One", start_pos=0.0),
            DialogueLine(speaker=alice, spoken_text="Two", start_pos=math.nan),
            DialogueLine(speaker=alice, spoken_text="Three", start_pos=1.0),
        ]
    )
    assert [(chunk.speaker_name, chunk.start_marker, chunk.end_marker) for chunk in combined] == [
        ("Alice", 0, 2),
        ("Alice", 2, 3),
    ]

    mixed = chunk_training_intervals(
        [
            DialogueLine(speaker=alice, spoken_text="One", start_pos=0.0),
            DialogueLine(speaker=bob, spoken_text="Two", start_pos=math.nan),
            DialogueLine(speaker=bob, spoken_text="Three", start_pos=1.0),
        ]
    )
    assert [(chunk.speaker_name, chunk.start_marker, chunk.end_marker) for chunk in mixed] == [
        ("Bob", 2, 3),
    ]


def test_production_plan_all_plans_deduplicates_shared_aligned_source(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    sound_file = tmp_path / "sounds" / "door.wav"
    sound_file.parent.mkdir(parents=True, exist_ok=True)
    sound_file.write_bytes(b"door")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")

    class FakeVibeVoice:
        async def register_request(self, request):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult(audio=np.zeros(4, dtype=np.float32))

            return Registered()

    async def runner():
        injector, ainjector = await make_async_injector(
            ProductionConfig(voice_directory=tmp_path),
            document_path=xml_path,
        )
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>
                    Anna: anna.wav
                  </speaker-map>
                  <script>
                    Anna: First line.
                    <sound ref="door" />
                    Anna: Second line.
                  </script>
                </production>
                """,
                source_name=str(xml_path),
            )
            production_plan = await root.plan(ainjector)
            return list(production_plan.all_plans())
        finally:
            injector.close()

    plans = asyncio.run(runner())
    aligned_sources = [plan for plan in plans if type(plan).__name__ == "AlignedScriptSource"]
    script_plans = [plan for plan in plans if type(plan).__name__ == "ScriptPlan"]
    assert len(aligned_sources) == 1
    assert len(script_plans) == 1


def test_export_training_samples_from_plan_writes_per_speaker_chunks(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    output_dir = tmp_path / "samples"

    class FakeTimedVibeVoice:
        async def register_request(self, request):
            dialogue_lines = [line for line in request.dialogue_lines if line.spoken_text.strip()]
            line_count = len(dialogue_lines)
            frame_count = line_count * 4
            positions = tuple(float(index) for index in range(line_count))

            class Registered:
                async def render(self_nonlocal) -> ScriptRenderResult:
                    return ScriptRenderResult(
                        audio=np.arange(frame_count, dtype=np.float32),
                        dialogue_line_start_positions=positions,
                    )

            return Registered()

    async def runner():
        injector, ainjector = await make_async_injector(
            ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1),
            document_path=xml_path,
        )
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeTimedVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>
                    Alice: anna.wav
                    Bob: anna.wav
                  </speaker-map>
                  <script>
                    Alice: First line.
                    Alice: Second line.
                  </script>
                  <script>
                    Bob: Third line.
                  </script>
                </production>
                """,
                source_name=str(xml_path),
            )
            production_plan = await root.plan(ainjector)
            return await export_training_samples_from_plan(
                production_plan,
                output_dir,
                sample_rate=4,
            )
        finally:
            injector.close()

    written = asyncio.run(runner())
    assert written == 3
    alice_files = sorted((output_dir / "Alice").glob("chunk_*.wav"))
    bob_files = sorted((output_dir / "Bob").glob("chunk_*.wav"))
    assert [path.name for path in alice_files] == [
        "chunk_First_line_000001.wav",
        "chunk_Second_line_000002.wav",
    ]
    assert [path.name for path in bob_files] == ["chunk_Third_line_000001.wav"]

    alice_audio, alice_rate = sf.read(alice_files[0], dtype="float32")
    bob_audio, bob_rate = sf.read(bob_files[0], dtype="float32")
    assert alice_rate == 4
    assert bob_rate == 4
    assert alice_audio.shape == (4,)
    assert bob_audio.shape == (4,)


def test_export_training_samples_from_plan_resamples_output(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    output_dir = tmp_path / "samples"

    class FakeTimedVibeVoice:
        async def register_request(self, request):
            class Registered:
                async def render(self_nonlocal) -> ScriptRenderResult:
                    return ScriptRenderResult(
                        audio=np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32),
                        dialogue_line_start_positions=(0.0,),
                    )

            return Registered()

    async def runner():
        injector, ainjector = await make_async_injector(
            ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1),
            document_path=xml_path,
        )
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeTimedVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>
                    Alice: anna.wav
                  </speaker-map>
                  <script>
                    Alice: First line.
                  </script>
                </production>
                """,
                source_name=str(xml_path),
            )
            production_plan = await root.plan(ainjector)
            return await export_training_samples_from_plan(
                production_plan,
                output_dir,
                sample_rate=8,
            )
        finally:
            injector.close()

    written = asyncio.run(runner())
    assert written == 1
    output_path = next((output_dir / "Alice").glob("chunk_*.wav"))
    audio, sample_rate = sf.read(output_path, dtype="float32")
    assert sample_rate == 8
    assert audio.shape == (8,)


def test_export_training_samples_from_plan_defaults_to_backend_sample_rate(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    output_dir = tmp_path / "samples"

    class FakeNativeRateVibeVoice:
        sample_rate = 4

        async def register_request(self, request):
            class Registered:
                def __init__(self, resource):
                    self.resource = resource

                async def render(self_nonlocal) -> ScriptRenderResult:
                    return ScriptRenderResult(
                        audio=np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32),
                        dialogue_line_start_positions=(0.0,),
                    )

            return Registered(self)

    async def runner():
        injector, ainjector = await make_async_injector(
            ProductionConfig(voice_directory=tmp_path, output_channels=1),
            document_path=xml_path,
        )
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeNativeRateVibeVoice(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>
                    Alice: anna.wav
                  </speaker-map>
                  <script>
                    Alice: First line.
                  </script>
                </production>
                """,
                source_name=str(xml_path),
            )
            production_plan = await root.plan(ainjector)
            return await export_training_samples_from_plan(
                production_plan,
                output_dir,
                sample_rate=None,
            )
        finally:
            injector.close()

    written = asyncio.run(runner())
    assert written == 1
    output_path = next((output_dir / "Alice").glob("chunk_*.wav"))
    audio, sample_rate = sf.read(output_path, dtype="float32")
    assert sample_rate == 4
    assert audio.shape == (4,)


def test_training_samples_reuses_vibevoice_cache(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    xml_path = tmp_path / "production.xml"
    xml_path.write_text("<production />", encoding="utf-8")
    output_dir = tmp_path / "samples"
    cache_output = tmp_path / "render.wav"

    class FakeCachedVibeVoiceResource(VibeVoiceResource):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.native_call_count = 0
            self._sample_rate = 4

        def _render_batch_native_sync(self, batch):
            self.native_call_count += 1
            return [np.arange(8, dtype=np.float32) for _ in batch]

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            updated = []
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
                    updated.append(content)
            return updated

    async def export_once():
        injector = radio_drama_injector(
            config=ProductionConfig(voice_directory=tmp_path, output_sample_rate=4, output_channels=1),
            event_loop=asyncio.get_running_loop(),
            document_path=xml_path,
            output_path=cache_output,
        )
        try:
            ainjector = injector(AsyncInjector)
            resource = await ainjector(FakeCachedVibeVoiceResource)
            injector.replace_provider(InjectionKey(VibeVoiceResource), resource, close=False)
            injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
            root = parse_production_string(
                """
                <production>
                  <speaker-map>
                    Alice: anna.wav
                  </speaker-map>
                  <script>
                    Alice: First line.
                    Alice: Second line.
                  </script>
                </production>
                """,
                source_name=str(xml_path),
            )
            production_plan = await root.plan(ainjector)
            written = await export_training_samples_from_plan(
                production_plan,
                output_dir,
                sample_rate=4,
            )
            return resource.native_call_count, written
        finally:
            injector.close()

    first_calls, first_written = asyncio.run(export_once())
    second_calls, second_written = asyncio.run(export_once())

    assert first_calls == 1
    assert second_calls == 0
    assert first_written == 2
    assert second_written == 2
