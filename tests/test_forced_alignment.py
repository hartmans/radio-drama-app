from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from carthage.dependency_injection import AsyncInjector, InjectionKey, Injector

from radio_drama.config import ProductionConfig
from radio_drama.debug import debug_artifact_directory
from radio_drama.document import parse_production_string
from radio_drama.forced_alignment import (
    AlignedScriptSource,
    AlignedClause,
    AlignedWord,
    AlignmentResult,
    ForcedAlignmentRequest,
    WhisperXResponse,
    WhisperXResource,
    _alignment_result_from_whisperx_response,
    fill_start_positions_from_alignment,
    fill_start_positions_from_rendered_script,
)
from radio_drama.init import radio_drama_injector
from radio_drama.dialogue import DialogueAudio, DialogueLine, ScriptRenderRequest, SpeakerVoiceReference
from radio_drama.rendering import RenderResult, ScriptRenderResult
from radio_drama.vibevoice import VibeVoiceResource


RESOURCE_DIR = Path(__file__).resolve().parent / "resources" / "forced_alignment"


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


def _case_paths() -> list[Path]:
    return sorted(RESOURCE_DIR.glob("*.json"))


@pytest.mark.parametrize("case_path", _case_paths(), ids=lambda path: path.stem)
def test_forced_alignment_cases(case_path: Path):
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    response_payload = payload["whisperx_response"]
    response = WhisperXResponse(
        transcription_segments=tuple(response_payload["transcription_segments"]),
        aligned_segments=(
            None
            if response_payload["aligned_segments"] is None
            else tuple(response_payload["aligned_segments"])
        ),
        decision=response_payload["decision"],
    )
    alignment = _alignment_result_from_whisperx_response(
        payload["transcript"],
        response,
        duration_seconds=float(payload["duration_seconds"]),
    )

    speakers: dict[str, SpeakerVoiceReference] = {}
    contents: list[DialogueLine] = []
    for line_payload in payload["dialogue_lines"]:
        speaker_name = line_payload["speaker"]
        speakers.setdefault(
            speaker_name,
            SpeakerVoiceReference(
                authored_name=speaker_name,
                voice_name=f"{speaker_name}.wav",
                resolved_path=Path(f"{speaker_name}.wav"),
            ),
        )
        contents.append(
            DialogueLine(
                speaker=speakers[speaker_name],
                spoken_text=line_payload["spoken_text"],
            )
        )

    filled = fill_start_positions_from_alignment(contents, alignment)
    assert len(filled) == len(payload["dialogue_lines"])
    for content, line_payload in zip(filled, payload["dialogue_lines"], strict=True):
        assert content.spoken_text == line_payload["spoken_text"]
        expected = line_payload["expected_start_seconds"]
        if expected is None:
            assert math.isnan(content.start_pos)
        else:
            assert content.start_pos == pytest.approx(float(expected))


def test_forced_alignment_debug_logs_line_positions(tmp_path: Path):
    voice_file = tmp_path / "anna.wav"
    voice_file.write_bytes(b"fake")
    config = ProductionConfig(
        voice_directory=tmp_path,
        output_sample_rate=4,
        output_channels=1,
        debug_log_path=tmp_path / "render.wav.log",
        debug_categories=("forced_alignment",),
    )

    class FakeVibeVoice:
        async def register_request(self, request: ScriptRenderRequest | None):
            class Registered:
                async def render(self_nonlocal) -> RenderResult:
                    return RenderResult(audio=np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))

            return Registered()

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            updated = []
            next_line_start = 0.0
            for content in contents:
                if isinstance(content, DialogueAudio):
                    updated.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=0.5))
                    continue
                updated.append(
                    type(content)(
                        speaker=content.speaker,
                        spoken_text=content.spoken_text,
                        start_pos=next_line_start,
                    )
                )
                next_line_start += 0.5
            return updated

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.replace_provider(InjectionKey(VibeVoiceResource), FakeVibeVoice(), close=False)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        try:
            root = parse_production_string(
                """
                <production>
                  <speaker-map>Anna: anna.wav</speaker-map>
                  <script>
                    Anna: First line for alignment logging.
                    <mark id="cut" />
                    Anna: Second line for alignment logging.
                  </script>
                </production>
                """,
                source_name="forced-alignment.xml",
            )
            production_plan = await root.plan(ainjector)
            aligned_source = production_plan.audio_plans[0].audio_plans[0].aligned_script_source
            await aligned_source.render()
        finally:
            injector.close()

    asyncio.run(runner())
    log_text = config.debug_log_path.read_text(encoding="utf-8")
    assert "[forced_alignment] 0.000s 'First line for alignment logging.'" in log_text
    assert "[forced_alignment] 0.500s 'Second line for alignment logging.'" in log_text


def test_fill_start_positions_from_rendered_script_uses_native_line_starts():
    speaker = SpeakerVoiceReference(
        authored_name="Anna",
        voice_name="anna.wav",
        resolved_path=Path("anna.wav"),
    )
    contents = [
        DialogueLine(speaker=speaker, spoken_text="First line."),
        DialogueAudio(audio_plan=object()),
        DialogueLine(speaker=speaker, spoken_text="Second line."),
    ]

    filled = fill_start_positions_from_rendered_script(
        contents,
        (0.25, 1.0),
        duration_seconds=2.5,
    )

    assert filled[0].start_pos == 0.25
    assert filled[1].start_pos == 1.0
    assert filled[2].start_pos == 1.0


def test_aligned_script_source_prefers_native_script_timing(tmp_path: Path):
    config = ProductionConfig(
        voice_directory=tmp_path,
        output_sample_rate=4,
        output_channels=1,
    )
    speaker = SpeakerVoiceReference(
        authored_name="Anna",
        voice_name="anna.wav",
        resolved_path=tmp_path / "anna.wav",
    )
    contents = [
        DialogueLine(speaker=speaker, spoken_text="First line."),
        DialogueLine(speaker=speaker, spoken_text="Second line."),
    ]

    class FakeScriptPlan:
        def __init__(self) -> None:
            self.contents = contents

        async def render(self) -> ScriptRenderResult:
            return ScriptRenderResult(
                audio=np.ones(8, dtype=np.float32),
                dialogue_line_start_positions=(0.25, 1.0),
            )

    class FakeWhisperX:
        async def fill_start_positions(self, contents, result):
            raise AssertionError("native script timing should bypass WhisperX")

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        injector.replace_provider(InjectionKey(WhisperXResource), FakeWhisperX(), close=False)
        try:
            aligned_source = await ainjector(
                AlignedScriptSource,
                node=None,
                script_plan=FakeScriptPlan(),
            )
            return await aligned_source.render()
        finally:
            injector.close()

    aligned_result = asyncio.run(runner())

    assert aligned_result.marker_frames == (0, 4, 8)
    assert [content.start_pos for content in aligned_result.contents] == [0.25, 1.0]


def test_forced_alignment_uses_exact_clause_boundaries_without_word_alignment():
    speaker = SpeakerVoiceReference(
        authored_name="Anna",
        voice_name="anna.wav",
        resolved_path=Path("anna.wav"),
    )
    contents = [
        DialogueLine(speaker=speaker, spoken_text="We will have order in this court."),
        DialogueLine(
            speaker=speaker,
            spoken_text="Mr. Brennan, have you proved the elements necessary to invoke involuntary truth finding?",
        ),
    ]
    alignment = AlignmentResult(
        words=(),
        clauses=(
            AlignedClause(
                text="We will have order in this court.",
                start=12.0,
                end=14.0,
            ),
            AlignedClause(
                text="Mr. Brennan, have you proved the elements necessary to invoke involuntary truth finding?",
                start=14.0,
                end=19.5,
            ),
        ),
    )

    filled = fill_start_positions_from_alignment(contents, alignment)

    assert [content.start_pos for content in filled] == [12.0, 14.0]


def test_forced_alignment_prefers_exact_clause_start_when_first_word_is_missing():
    speaker = SpeakerVoiceReference(
        authored_name="Anna",
        voice_name="anna.wav",
        resolved_path=Path("anna.wav"),
    )
    contents = [
        DialogueLine(speaker=speaker, spoken_text="Alpha."),
        DialogueLine(speaker=speaker, spoken_text="Bravo Charlie."),
    ]
    alignment = AlignmentResult(
        words=(
            AlignedWord(text="Alpha", start=1.0, end=2.0),
            AlignedWord(text="Charlie", start=2.5, end=3.0),
        ),
        clauses=(
            AlignedClause(text="Alpha.", start=1.0, end=2.0),
            AlignedClause(text="Bravo Charlie.", start=2.0, end=4.0),
        ),
    )

    filled = fill_start_positions_from_alignment(contents, alignment)

    assert [content.start_pos for content in filled] == [1.0, 2.0]


def test_forced_alignment_does_not_infer_line_start_from_clause_end_boundary():
    speaker = SpeakerVoiceReference(
        authored_name="Anna",
        voice_name="anna.wav",
        resolved_path=Path("anna.wav"),
    )
    contents = [
        DialogueLine(speaker=speaker, spoken_text="Alpha Bravo."),
        DialogueLine(speaker=speaker, spoken_text="Charlie Delta."),
    ]
    alignment = AlignmentResult(
        words=(
            AlignedWord(text="Alpha", start=10.0, end=10.4),
            AlignedWord(text="Bravo", start=10.4, end=10.8),
            AlignedWord(text="Delta", start=11.5, end=12.0),
        ),
        clauses=(
            AlignedClause(text="Alpha.", start=10.0, end=10.4),
            AlignedClause(text="Bravo Charlie Delta.", start=10.4, end=12.0),
        ),
    )

    filled = fill_start_positions_from_alignment(contents, alignment)

    assert filled[0].start_pos == 10.0
    assert np.isnan(filled[1].start_pos)


def test_whisperx_resource_prefers_exact_aligned_segments_over_coarse_transcription(
    tmp_path: Path,
    monkeypatch,
):
    config = ProductionConfig(
        voice_directory=tmp_path,
        output_sample_rate=48000,
        output_channels=2,
    )

    class FakeModel:
        def transcribe(self, audio, batch_size, language):
            return {
                "segments": [
                    {
                        "text": (
                            "That is not true at all. "
                            "I don't see you rushing to give up your soul. "
                            "We will have order in this court."
                        ),
                        "start": 19.425,
                        "end": 35.11,
                    }
                ]
            }

    fake_whisperx = type(
        "FakeWhisperXModule",
        (),
        {
            "load_model": staticmethod(lambda *args, **kwargs: FakeModel()),
            "load_align_model": staticmethod(lambda *args, **kwargs: ("align-model", {"meta": "data"})),
            "align": staticmethod(
                lambda *args, **kwargs: {
                    "segments": [
                        {
                            "text": "That is not true at all.",
                            "start": 19.425,
                            "end": 20.605,
                            "words": [
                                {"word": "That", "start": 19.425, "end": 19.565},
                                {"word": "is", "start": 19.645, "end": 19.745},
                            ],
                        },
                        {
                            "text": "I don't see you rushing to give up your soul.",
                            "start": 31.449,
                            "end": 33.409,
                            "words": [
                                {"word": "I", "start": 31.449, "end": 31.489},
                                {"word": "don't", "start": 31.529, "end": 31.689},
                            ],
                        },
                        {
                            "text": "We will have order in this court.",
                            "start": 33.77,
                            "end": 35.11,
                            "words": [
                                {"word": "We", "start": 33.77, "end": 33.89},
                                {"word": "will", "start": 33.93, "end": 34.07},
                            ],
                        },
                    ]
                }
            ),
        },
    )()
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await ainjector(WhisperXResource)
            return resource._alignment_result_sync(
                np.zeros((48000, 2), dtype=np.float32),
                48000,
                (
                    "That is not true at all.\n"
                    "I don't see you rushing to give up your soul.\n"
                    "We will have order in this court."
                ),
            )
        finally:
            injector.close()

    alignment = asyncio.run(runner())

    assert alignment.words == ()
    assert [(clause.start, clause.end, clause.text) for clause in alignment.clauses] == [
        (19.425, 20.605, "That is not true at all."),
        (31.449, 33.409, "I don't see you rushing to give up your soul."),
        (33.77, 35.11, "We will have order in this court."),
    ]


def test_forced_alignment_word_matcher_can_resynchronize_after_missed_line():
    speaker = SpeakerVoiceReference(
        authored_name="Anna",
        voice_name="anna.wav",
        resolved_path=Path("anna.wav"),
    )
    contents = [
        DialogueLine(speaker=speaker, spoken_text="Missing words here."),
        DialogueLine(speaker=speaker, spoken_text="Charlie Delta."),
        DialogueLine(speaker=speaker, spoken_text="Echo Foxtrot."),
    ]
    alignment = AlignmentResult(
        words=(
            AlignedWord(text="Charlie", start=2.0, end=2.3),
            AlignedWord(text="Delta", start=2.3, end=2.6),
            AlignedWord(text="Echo", start=3.0, end=3.3),
            AlignedWord(text="Foxtrot", start=3.3, end=3.7),
        ),
        clauses=(
            AlignedClause(text="Charlie Delta Echo Foxtrot.", start=2.0, end=3.7),
        ),
    )

    filled = fill_start_positions_from_alignment(contents, alignment)

    assert np.isnan(filled[0].start_pos)
    assert filled[1].start_pos == 2.0
    assert filled[2].start_pos == 3.0


def test_whisperx_debug_writes_segment_payload(tmp_path: Path):
    config = ProductionConfig(
        voice_directory=tmp_path,
        debug_log_path=tmp_path / "output.wav.log",
        debug_categories=("whisperx",),
    )

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await ainjector(WhisperXResource)
            resource._write_whisperx_debug_output(
                "First debug line.\nSecond debug line.",
                WhisperXResponse(
                    transcription_segments=(
                        {"text": "First debug line.", "start": 1.0, "end": 2.0},
                    ),
                    aligned_segments=(
                        {
                            "text": "First debug line.",
                            "start": 1.0,
                            "end": 2.0,
                            "words": [{"word": "First", "start": 1.0, "end": 1.3}],
                        },
                    ),
                    decision="aligned_word_matching",
                ),
            )
        finally:
            injector.close()

    asyncio.run(runner())
    artifact_directory = debug_artifact_directory(config, "whisperx")
    assert artifact_directory is not None
    artifact_files = sorted(artifact_directory.glob("*.json"))
    assert [path.name for path in artifact_files] == ["000-first_debug_line.json"]
    payload = artifact_files[0].read_text(encoding="utf-8")
    assert '"decision": "aligned_word_matching"' in payload
    assert '"transcription_segments"' in payload
    assert '"aligned_segments"' in payload


def test_whisperx_resource_batches_registered_requests_and_skips_align_when_not_needed(
    tmp_path: Path,
):
    config = ProductionConfig(output_sample_rate=48000, output_channels=2)

    class FakeQueuedWhisperXResource(WhisperXResource):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.batch_sizes: list[int] = []
            self.align_model_requests = 0

        def _prepare_batch_sync(self, batch):
            self.batch_sizes.append(len(batch))
            return [
                self._prepare_request_sync(pending.registration.request)
                for pending in batch
            ]

        def _prepare_request_sync(self, request):
            return type(super()._prepare_request_sync(request))(
                request=request,
                mono_audio=np.zeros(16000, dtype=np.float32),
                transcription_segments=(
                    {
                        "text": "First line.",
                        "start": 1.0,
                        "end": 2.0,
                    },
                    {
                        "text": "Second line.",
                        "start": 2.0,
                        "end": 3.0,
                    },
                ),
            )

        def _ensure_align_model(self):
            self.align_model_requests += 1
            raise AssertionError("align model should not be loaded for exact transcription clauses")

    async def runner():
        injector, ainjector = await _make_async_injector(config)
        try:
            resource = await ainjector(FakeQueuedWhisperXResource)
            request1 = await resource.register_request(
                ForcedAlignmentRequest(
                    audio=np.zeros((48000, 2), dtype=np.float32),
                    sample_rate=48000,
                    transcript="First line.\nSecond line.",
                )
            )
            request2 = await resource.register_request(
                ForcedAlignmentRequest(
                    audio=np.zeros((48000, 2), dtype=np.float32),
                    sample_rate=48000,
                    transcript="First line.\nSecond line.",
                )
            )
            responses = await asyncio.gather(request1.align(), request2.align())
            return resource.batch_sizes, resource.align_model_requests, responses
        finally:
            injector.close()

    batch_sizes, align_model_requests, responses = asyncio.run(runner())
    assert batch_sizes == [2]
    assert align_model_requests == 0
    assert [response.decision for response in responses] == [
        "transcription_exact_clause_match",
        "transcription_exact_clause_match",
    ]
