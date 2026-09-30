"""Backend-independent projection/queue tests and Qwen window orchestration."""
import asyncio
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from carthage.dependency_injection import AsyncInjector, InjectionKey, Injector

from radio_drama.config import ProductionConfig
from radio_drama.dialogue import DialogueLine, ScriptGap, ScriptPlan, ScriptRenderRequest, SpeakerVoiceReference
from radio_drama.forced_alignment import (
    AlignedClause, AlignmentResult, ForcedAlignmentRequest, ForcedAlignmentResource, WordTiming,
)
from radio_drama.forced_alignment.projection import script_timing_from_alignment
from radio_drama.forced_alignment.qwen import (
    AlignmentWindowError, QwenAlignmentResource, audio_windows, decoded_words, merge_window_words,
)
from radio_drama.init import radio_drama_injector
from radio_drama.rendering import RenderResult, DialogueMarkTiming
from radio_drama.document import parse_production_string


@pytest.fixture
def speaker():
    return SpeakerVoiceReference("A", "voice", Path("voice.wav"))


@pytest.fixture(params=["words", "preferred_clauses"])
def alignment_evidence(request):
    words = tuple(WordTiming(text, i, i + .5) for i, text in enumerate(
        "Please put the red folder beside the blue folder on the table".split()))
    clauses = (AlignedClause("Please put the red folder beside the blue folder on the table", .1, 12.),)
    return AlignmentResult(words, (), preferred_clauses=clauses if request.param == "preferred_clauses" else ())


def test_marks_preserve_full_line_and_repeated_word_occurrence(speaker, alignment_evidence):
    text = "Please put the red folder beside the blue folder on the table"
    line = DialogueLine(speaker, text)
    baseline = script_timing_from_alignment([line], alignment_evidence).dialogue_lines[0]
    offsets = tuple(i for i, char in enumerate(text) if i == 0 or text[i - 1] == " ")
    marked = script_timing_from_alignment([replace(line, mark_offsets=offsets)], alignment_evidence).dialogue_lines[0]
    assert (marked.start, marked.end) == (baseline.start, baseline.end)
    first = text.index("folder")
    second = text.index("folder", first + 1)
    assert marked.marks[offsets.index(first)].next_start == 4.
    assert marked.marks[offsets.index(second)].next_start == 8.
    sparse = script_timing_from_alignment([replace(line, mark_offsets=(second, first, second))], alignment_evidence)
    assert sparse.dialogue_lines[0].marks == (marked.marks[offsets.index(second)],
                                             marked.marks[offsets.index(first)], marked.marks[offsets.index(second)])


def test_missing_inner_word_preserves_line_and_next_line(speaker):
    text = "Please put the bright red folder beside the blue folder on the table"
    actual = "Please put the red folder beside the blue folder on the table Next line now"
    evidence = AlignmentResult(tuple(WordTiming(w, i, i + .4) for i, w in enumerate(actual.split())), ())
    lines = [DialogueLine(speaker, text), DialogueLine(speaker, "Next line now")]
    baseline = script_timing_from_alignment(lines, evidence)
    start, after = text.index("bright"), text.index("red")
    marked = script_timing_from_alignment([replace(lines[0], mark_offsets=(start, after, text.index("blue"))), lines[1]], evidence)
    assert [(x.start, x.end) for x in marked.dialogue_lines] == [(x.start, x.end) for x in baseline.dialogue_lines]
    assert baseline.dialogue_lines[0].start == 0.
    assert marked.dialogue_lines[0].marks[0].previous_end == 2.4
    assert marked.dialogue_lines[0].marks[0].next_start is None
    assert marked.dialogue_lines[0].marks[1].previous_end is None
    assert marked.dialogue_lines[0].marks[1].next_start == 3.
    assert marked.dialogue_lines[0].marks[2].next_start == 7.


def test_clause_success_survives_unresolved_inner_marks(speaker):
    line = DialogueLine(speaker, "Hello there.", mark_offsets=(0, 6, 12, 2))
    evidence = AlignmentResult((), (), preferred_clauses=(AlignedClause(line.spoken_text, 1., 4.),))
    result = script_timing_from_alignment([line], evidence).dialogue_lines[0]
    assert (result.start, result.end) == (1., 4.)
    assert result.marks == (DialogueMarkTiming(None, 1.), DialogueMarkTiming(None, None),
                            DialogueMarkTiming(4., None), DialogueMarkTiming(None, None))


def test_inner_boundary_has_same_two_sides_as_between_lines(speaker):
    first, second = "Alpha bravo charlie.", "Delta echo foxtrot."
    evidence = AlignmentResult(None, (AlignedClause(first, 1., 2.), AlignedClause(second, 2.6, 4.)))
    separate = script_timing_from_alignment([DialogueLine(speaker, first), DialogueLine(speaker, second)], evidence)
    combined = script_timing_from_alignment([DialogueLine(speaker, first + " " + second,
                                                          mark_offsets=(len(first) + 1,))], evidence)
    assert combined.dialogue_lines[0].marks[0] == DialogueMarkTiming(
        separate.dialogue_lines[0].end, separate.dialogue_lines[1].start)


def test_invalid_offset_and_audio_identity(speaker):
    first = DialogueLine(speaker, "Hello", mark_offsets=(99,))
    with pytest.raises(ValueError, match="offset 99"):
        script_timing_from_alignment([first], AlignmentResult((), ()))
    request = ScriptRenderRequest(dialogue_lines=[first])
    other = ScriptRenderRequest(dialogue_lines=[replace(first, mark_offsets=(0, 5))])
    assert request.serialize_cache_request() == other.serialize_cache_request()


def test_windows_cover_thirty_minutes_without_missing_tail():
    windows = audio_windows(30 * 60 * 16000)
    assert windows[0].start_frame == 0
    assert windows[-1].end_frame == 30 * 60 * 16000
    assert all(w.end_frame - w.start_frame <= 180 * 16000 for w in windows)
    assert all(a.end_frame - b.start_frame == 10 * 16000 for a, b in zip(windows, windows[1:]))
    assert audio_windows(0) == []
    with pytest.raises(ValueError):
        audio_windows(1, overlap_seconds=180)


def test_seam_preserves_true_repeated_words_and_uses_one_measurement():
    left = (WordTiming("again", 10., 11.), WordTiming("again", 171., 172.), WordTiming("now", 175., 176.))
    right = (WordTiming("again", 171.08, 172.08), WordTiming("now", 175.08, 176.08), WordTiming("tail", 182., 183.))
    merged = merge_window_words(left, right, overlap_start=170., overlap_end=180.)
    assert [w.text for w in merged] == ["again", "again", "now", "tail"]
    assert merged[2] == left[2]
    with pytest.raises(AlignmentWindowError):
        merge_window_words(left, (WordTiming("wrong", 171., 172.),), overlap_start=170., overlap_end=180.)


def test_decoding_keeps_unknowns_and_source_coordinates():
    words = decoded_words([
        {"text": "a", "start_time": .5, "end_time": 1.04},
        {"text": "b", "start_time": float("nan"), "end_time": 4.},
    ], 1., offset=170.)
    assert words == (WordTiming("a", 170.5, 171.), WordTiming("b", None, None))


class FixtureAlignment(ForcedAlignmentResource):
    alignment_identity = "fixture"
    transcription_identity = "fixture"
    async def _process_batch(self, requests):
        return [AlignmentResult(None, (AlignedClause(r.transcript, 0., 1.),)) for r in requests]


@pytest.mark.parametrize("backend, expected", [("whisperx", "WhisperXResource"), ("qwen", "QwenAlignmentResource")])
def test_injector_selects_one_backend_without_model_load(backend, expected):
    async def run():
        injector = radio_drama_injector(config=ProductionConfig(alignment_backend=backend),
                                       event_loop=asyncio.get_running_loop())
        try:
            resource = await injector(AsyncInjector).get_instance_async(ForcedAlignmentResource)
            assert type(resource).__name__ == expected
        finally:
            injector.close()
    asyncio.run(run())


def test_prepared_script_plan_keeps_marks_without_speaker_map(speaker, tmp_path):
    async def run():
        from radio_drama.dialogue import TtsResource
        from radio_drama.rendering import BackendTtsResult, ScriptTiming, DialogueLineTiming
        from radio_drama.cache import CacheManager
        from carthage.dependency_injection import inject
        @inject(cache_manager=CacheManager, config=ProductionConfig)
        class Backend(TtsResource):
            cache_collection_name = "fixture"
            async def register_backend_request(self, request):
                class Registered:
                    async def render(self):
                        return BackendTtsResult(np.zeros(4, dtype=np.float32), 4,
                                                ScriptTiming((DialogueLineTiming(0., 1.),)))
                return Registered()
        injector = radio_drama_injector(config=ProductionConfig(output_sample_rate=4, output_channels=1),
                                       event_loop=asyncio.get_running_loop(), output_path=tmp_path / "out.wav")
        injector.replace_provider(InjectionKey(TtsResource, tts="fixture"), Backend)
        injector.replace_provider(InjectionKey(ForcedAlignmentResource), FixtureAlignment)
        try:
            node = parse_production_string("<production><script tts='fixture'>A: unused</script></production>").children[0]
            events = [DialogueLine(speaker, "Hello there", mark_offsets=(0, 6, 11))]
            plan = await injector(AsyncInjector)(ScriptPlan, node=node, script_events=events, tts="fixture", attrs={})
            assert plan.script_events == events
            assert not plan.needs_source_slicing()
            result = await plan.render()
            timing = await plan.ensure_timing(events, result)
            assert (timing.dialogue_lines[0].start, timing.dialogue_lines[0].end) == (0., 1.)
            assert len(timing.dialogue_lines[0].marks) == 3
        finally:
            injector.close()
    asyncio.run(run())


class FixtureQwen(QwenAlignmentResource):
    """Replace acoustic calls, retaining real request/window orchestration."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transcribed = []
        self.aligned = []

    def _transcribe_batch(self, audios, languages):
        from radio_drama.forced_alignment import TranscriptionResult
        self.transcribed.append((tuple(len(a) for a in audios), tuple(languages)))
        return [TranscriptionResult("hello there", lang) for lang in languages]

    def _align_batch(self, audios, transcripts, languages):
        self.aligned.append((tuple(len(a) for a in audios), tuple(transcripts)))
        return [[{"text": word, "start_time": i, "end_time": i + .4}
                 for i, word in enumerate(text.split())]
                for text in transcripts]


def test_qwen_complete_transcripts_batch_without_asr_or_model_loading():
    config = ProductionConfig(batch_size=3, device="cpu")
    resource = FixtureQwen(config=config, injector=Injector())
    try:
        requests = [ForcedAlignmentRequest(np.zeros(2 * 16000), 16000, "hello there", "complete")
                    for _ in range(7)]
        results = resource._process_batch_sync(requests)
        assert resource.transcribed == []
        assert [len(audios) for audios, _ in resource.aligned] == [3, 3, 1]
        assert all(result.source_text == "hello there" for result in results)
        assert all(result.words[1].start == 1. for result in results)
        assert resource._asr is resource._aligner is None
        with pytest.raises(AlignmentWindowError, match="ASR fallback is disabled"):
            resource._process_batch_sync([ForcedAlignmentRequest(
                np.zeros(181 * 16000), 16000, "hello there", "complete")])
        assert resource.transcribed == []
    finally:
        resource.close()


def test_qwen_long_recording_windows_asr_once_and_offsets_alignment():
    class LongFixture(FixtureQwen):
        def _align_batch(self, audios, transcripts, languages):
            self.aligned.append((tuple(len(a) for a in audios), tuple(transcripts)))
            # Audio encodes absolute time so both overlap measurements agree.
            return [[{"text": str(round(float(audio[0]) + t)), "start_time": t, "end_time": t + .2}
                     for t in range(0, len(audio) // 16000)] for audio in audios]
    resource = LongFixture(config=ProductionConfig(batch_size=3, device="cpu"), injector=Injector())
    try:
        audio = np.arange(351 * 16000, dtype=np.float32) / 16000
        result = resource.transcribe_sync(audio, 16000)
        assert [len(audios) for audios, _ in resource.transcribed] == [3]
        assert [len(audios) for audios, _ in resource.aligned] == [3]
        assert result.text.split() == [str(i) for i in range(351)]
        assert resource.transcribed[0][0] == (180 * 16000, 180 * 16000, 11 * 16000)
    finally:
        resource.close()


def test_alignment_queue_shields_cancellation_and_propagates_failure():
    async def run():
        class Resource(FixtureAlignment):
            async def _process_batch(self, requests):
                await asyncio.sleep(.01)
                if requests[0].transcript == "fail":
                    raise ValueError("failed backend")
                return await super()._process_batch(requests)
        resource = Resource(config=ProductionConfig(batch_size=2), injector=Injector())
        try:
            request = ForcedAlignmentRequest(np.zeros(1), 1, "hello")
            registration = await resource.register_request(request)
            waiter = asyncio.create_task(registration.align())
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert (await registration.align()).clauses[0].text == "hello"
            failed = await resource.register_request(replace(request, transcript="fail"))
            with pytest.raises(ValueError, match="failed backend"):
                await failed.align()
            closed = await resource.register_request(request)
            resource.close()
            assert closed.future.cancelled()
        finally:
            resource.close()
    asyncio.run(run())


def test_qwen_native_processor_calls_use_batched_inputs_and_decode_rows():
    """Exercise the adapter API with CPU tensors, without loading weights."""
    import torch
    from types import SimpleNamespace
    from radio_drama.forced_alignment import TranscriptionResult
    calls = []
    class Inputs(dict):
        def to(self, device, dtype):
            calls.append(("to", device, dtype))
            return self
    class Processor:
        def apply_transcription_request(self, **kwargs):
            assert kwargs["language"] == ["en", "en"]
            assert kwargs["processor_kwargs"]["padding"] is True
            assert kwargs["processor_kwargs"]["sampling_rate"] == 16000
            return Inputs(input_ids=torch.ones((2, 3), dtype=torch.long))
        def decode(self, row, *, return_format):
            assert row.ndim == 1 and row.shape[0] == 2
            assert return_format == "parsed"
            return {"transcription": "hello", "language": None}
        def prepare_forced_aligner_inputs(self, **kwargs):
            assert kwargs["transcript"] == ["hello", "there"]
            return Inputs(input_ids=torch.ones((2, 3), dtype=torch.long)), [["hello"], ["there"]]
        def decode_forced_alignment(self, **kwargs):
            assert kwargs["timestamp_token_id"] == 42
            return [[{"text": words[0], "start_time": .1, "end_time": .4}]
                    for words in kwargs["word_lists"]]
    class Model:
        device, dtype = torch.device("cpu"), torch.float32
        generation_config = SimpleNamespace(eos_token_id=2)
        config = SimpleNamespace(timestamp_token_id=42)
        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 1, 1, 7, 2], [1, 1, 1, 8, 2]])
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.zeros(2, 3, 4))
    resource = FixtureQwen(config=ProductionConfig(device="cpu"), injector=Injector())
    resource._asr_processor = resource._aligner_processor = Processor()
    resource._asr = resource._aligner = Model()
    try:
        audios = [np.zeros(16000), np.zeros(16000)]
        results = QwenAlignmentResource._transcribe_batch(resource, audios, ["en", "en"])
        assert results == [TranscriptionResult("hello", "en")] * 2
        aligned = QwenAlignmentResource._align_batch(resource, audios, ["hello", "there"], ["en", "en"])
        assert [items[0]["text"] for items in aligned] == ["hello", "there"]
        assert len(calls) == 2
    finally:
        resource.close()


def test_qwen_seam_repair_is_bounded_and_reports_unresolved_overlap():
    resource = FixtureQwen(config=ProductionConfig(device="cpu"), injector=Injector())
    try:
        with pytest.raises(AlignmentWindowError, match="three bridge attempts"):
            resource._repair_seam((WordTiming("left", 171., 172.),),
                                  (WordTiming("right", 175., 176.),),
                                  np.zeros(351 * 16000), 170., 180., "en")
        assert len(resource.transcribed) == len(resource.aligned) == 3
        assert [sizes[0] for sizes, _ in resource.transcribed] == [60 * 16000, 120 * 16000, 180 * 16000]
    finally:
        resource.close()


@pytest.fixture(params=["whisperx", "qwen_complete", "qwen_recording"])
def saved_alignment_evidence(request):
    import json
    root = Path(__file__).resolve().parent / "resources"
    if request.param == "whisperx":
        from radio_drama.forced_alignment.whisperx import _alignment_result_from_whisperx
        return _alignment_result_from_whisperx(json.loads((root / "whisperx_cli/girl1.json").read_text()))
    suffix = request.param.removeprefix("qwen_")
    payload = json.loads((root / "qwen_alignment" / ("girl1_" + suffix + ".json")).read_text())
    return AlignmentResult(
        tuple(WordTiming(**word) for word in payload["words"]),
        tuple(AlignedClause(**clause) for clause in payload["clauses"]),
        tuple(AlignedClause(**clause) for clause in payload["preferred_clauses"]),
        payload["source_text"], payload["language"], payload["estimated"])


def test_saved_backend_evidence_preserves_line_spans_when_marks_added(speaker, saved_alignment_evidence):
    import json
    payload = json.loads((Path(__file__).resolve().parent / "resources/whisperx_cli/girl1.json").read_text())
    lines = [DialogueLine(speaker, segment["text"].strip()) for segment in payload["segments"]]
    baseline = script_timing_from_alignment(lines, saved_alignment_evidence)
    marked = [replace(line, mark_offsets=tuple(
        i for i in range(len(line.spoken_text)) if i == 0 or line.spoken_text[i - 1] == " ")) for line in lines]
    refined = script_timing_from_alignment(marked, saved_alignment_evidence)
    for original, measured, expected in zip(baseline.dialogue_lines, refined.dialogue_lines, payload["segments"], strict=True):
        assert (measured.start, measured.end) == (original.start, original.end)
        assert abs(measured.start - expected["start"]) < .9
        assert abs(measured.end - expected["end"]) < .9
        assert measured.marks[0].next_start == measured.start
        assert all(mark.next_start is not None for mark in measured.marks)
