"""Native modern Transformers processor contracts; no GPU or model weights."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("transformers.models.qwen3_asr", reason="Native Qwen requires modern Transformers")


@pytest.mark.live
@pytest.mark.parametrize("aligner", [False, True], ids=["asr", "alignment"])
def test_native_qwen_processor_inputs_and_decoding(aligner):
    import torch
    from transformers import AutoConfig, AutoProcessor
    from radio_drama.config import ProductionConfig
    config = ProductionConfig()
    model_id = config.qwen_alignment_model if aligner else config.qwen_asr_model
    cache_dir = Path(__file__).resolve().parents[1] / ".model-cache"
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    audio = [np.zeros(16000, dtype=np.float32), np.zeros(24000, dtype=np.float32)]
    if aligner:
        inputs, words = processor.prepare_forced_aligner_inputs(
            audio=audio, transcript=["Hello there.", "Hello again."],
            language=["en", "en"], processor_kwargs={"sampling_rate": 16000, "padding": True})
        model_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        timestamp = model_config.timestamp_token_id
        # CPU synthetic logits exercise the real processor's timestamp decoder.
        logits = torch.zeros((*inputs["input_ids"].shape, 3))
        logits[..., 1] = 1.
        decoded = processor.decode_forced_alignment(
            logits=logits, input_ids=inputs["input_ids"], word_lists=words,
            timestamp_token_id=timestamp)
        assert len(decoded) == 2
        assert [item["text"] for item in decoded[0]] == words[0]
        assert all(set(item) == {"text", "start_time", "end_time"} for row in decoded for item in row)
    else:
        inputs = processor.apply_transcription_request(
            audio=audio, language=["en", "en"], processor_kwargs={"sampling_rate": 16000, "padding": True})
        for text in ("Hello there.", "language English<asr_text>Hello there."):
            ids = processor.tokenizer.encode(text, add_special_tokens=False)
            parsed = processor.decode(ids, return_format="parsed")
            assert parsed["transcription"] == "Hello there."
    assert inputs["input_ids"].shape[0] == 2
    assert torch.is_tensor(inputs["input_features"])


@pytest.mark.live
def test_qwen_gpu_alignment_asr_and_batched_requests(tmp_path):
    import asyncio
    from dataclasses import asdict
    import json
    import os
    import soundfile as sf
    from carthage.dependency_injection import AsyncInjector
    from radio_drama.config import ProductionConfig
    from radio_drama.dialogue import DialogueLine, SpeakerVoiceReference
    from radio_drama.forced_alignment import ForcedAlignmentRequest, ForcedAlignmentResource
    from radio_drama.forced_alignment.projection import script_timing_from_alignment
    from radio_drama.init import radio_drama_injector
    root = Path(__file__).resolve().parents[1]
    audio, rate = sf.read(root / "tests/resources/girl1.wav", dtype="float32")
    segments = json.loads((root / "tests/resources/whisperx_cli/girl1.json").read_text())["segments"]
    transcript = " ".join(segment["text"].strip() for segment in segments)
    speaker = SpeakerVoiceReference("Girl", "girl1.wav", root / "tests/resources/girl1.wav")
    lines = [DialogueLine(speaker, segment["text"].strip(), mark_offsets=(0,)) for segment in segments]
    async def run():
        injector = radio_drama_injector(
            config=ProductionConfig(alignment_backend="qwen", device="cuda", batch_size=3),
            event_loop=asyncio.get_running_loop(), output_path=tmp_path / "out.wav")
        try:
            resource = await injector(AsyncInjector).get_instance_async(ForcedAlignmentResource)
            complete = ForcedAlignmentRequest(audio, rate, transcript, "complete", True)
            registrations = [await resource.register_request(complete) for _ in range(2)]
            responses = await asyncio.gather(*(r.align() for r in registrations))
            assert resource._asr is None, "Complete transcripts must not load/run ASR"
            assert responses[0].words == responses[1].words
            timing = script_timing_from_alignment(lines, responses[0])
            for span, expected in zip(timing.dialogue_lines, segments, strict=True):
                assert abs(span.start - expected["start"]) < .9
                assert abs(span.end - expected["end"]) < .9
                assert span.marks[0].next_start == span.start
            recording = await (await resource.register_request(ForcedAlignmentRequest(
                audio, rate, "You know you can tell me anything.", "partial", True))).align()
            assert resource._asr is not None
            assert "anything" in recording.source_text.lower()
            assert len(recording.words) > 10
            if os.environ.get("RADIO_DRAMA_UPDATE_ALIGNMENT_FIXTURES") == "1":
                directory = root / "tests/resources/qwen_alignment"
                directory.mkdir(exist_ok=True)
                for name, result in (("girl1_complete", responses[0]), ("girl1_recording", recording)):
                    (directory / (name + ".json")).write_text(json.dumps(asdict(result), indent=2) + "\n")
        finally:
            injector.close()
    asyncio.run(run())


@pytest.mark.live
def test_qwen_gpu_overlapping_recording_windows(monkeypatch, tmp_path):
    """Use short windows on real speech to exercise both seams and tail."""
    import asyncio
    import json
    import soundfile as sf
    from carthage.dependency_injection import AsyncInjector
    from radio_drama.config import ProductionConfig
    from radio_drama.forced_alignment import ForcedAlignmentRequest, ForcedAlignmentResource
    from radio_drama.forced_alignment import qwen
    from radio_drama.init import radio_drama_injector
    root = Path(__file__).resolve().parents[1]
    girl, rate = sf.read(root / "tests/resources/girl1.wav", dtype="float32")
    lawyer, other_rate = sf.read(root / "tests/resources/lawyer1.wav", dtype="float32")
    assert rate == other_rate
    audio = np.concatenate([girl, lawyer])
    original_windows = qwen.audio_windows
    monkeypatch.setattr(qwen, "audio_windows", lambda count: original_windows(
        count, window_seconds=10, overlap_seconds=4))
    async def run():
        injector = radio_drama_injector(
            config=ProductionConfig(alignment_backend="qwen", device="cuda", batch_size=3),
            event_loop=asyncio.get_running_loop(), output_path=tmp_path / "out.wav")
        try:
            resource = await injector(AsyncInjector).get_instance_async(ForcedAlignmentResource)
            result = await (await resource.register_request(ForcedAlignmentRequest(
                audio, rate, "", "partial", True))).align()
            text = result.source_text.lower()
            assert "anything" in text and "witness" in text
            assert result.words[-1].end <= len(audio) / rate
            assert result.words[-1].end > len(girl) / rate
            # Compare words in physical overlap with a whole-recording reference;
            # this catches duplicated phrases, loss of tails and bad offsets.
            monkeypatch.setattr(qwen, "audio_windows", original_windows)
            full = await (await resource.register_request(ForcedAlignmentRequest(
                audio, rate, result.source_text, "complete", True))).align()
            assert len(result.words) == len(full.words)
            for measured, reference in zip(result.words, full.words, strict=True):
                if measured.start is not None and reference.start is not None:
                    assert abs(measured.start - reference.start) < .8
        finally:
            injector.close()
    asyncio.run(run())
