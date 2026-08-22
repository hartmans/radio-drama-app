from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from radio_drama.proxy import load_proxy_tts_configs
from tts_engines.f5_tts.engine import F5TtsEngine, MODEL, _trim_transcript


def _request() -> dict:
    return {
        "first_words": "hello",
        "dialogue_contents": [
            {
                "type": "line",
                "spoken_text": "Hello there.",
                "speaker": {
                    "voice_path": "/voices/alice.wav",
                    "transcript": "Reference words.",
                },
            }
        ],
    }


def _write_wav(path, waveform, sample_rate):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes((np.asarray(waveform) * 32767).astype("<i2").tobytes())


def test_f5_sample_config_uses_offline_shared_cache():
    sample = Path(__file__).resolve().parents[1] / "tts_engines/f5_tts/tts.toml.example"
    config = load_proxy_tts_configs(sample)["f5"]

    assert config.network == "none"
    assert config.devices == ("nvidia.com/gpu=all",)
    assert config.environment == {}
    assert str(config.mounts[0].source) == "/srv/ai/huggingface-cache"


def test_f5_uses_largest_v1_base_checkpoint():
    assert MODEL == "F5TTS_v1_Base"


def test_f5_passes_reference_transcript_and_generation_controls(monkeypatch):
    seen = {}

    class FakeModel:
        def infer(self, **kwargs):
            seen.update(kwargs)
            return np.zeros(24_000), 24_000, object()

    engine = F5TtsEngine()
    engine.model = FakeModel()
    monkeypatch.setattr(engine, "prepare_reference", lambda path, text: (path, text))
    waveform, sample_rate = engine.synthesize_line(_request()["dialogue_contents"][0])

    assert waveform.shape == (24_000,)
    assert sample_rate == 24_000
    assert seen["ref_file"] == "/voices/alice.wav"
    assert seen["ref_text"] == "Reference words."
    assert seen["gen_text"] == "Hello there."
    assert seen["cross_fade_duration"] == 0.0


def test_trim_transcript_leaves_unclipped_reference_unchanged():
    transcript = "One two three four."
    assert _trim_transcript(transcript, 1.0) == transcript


def test_trim_transcript_keeps_matching_word_prefix():
    assert _trim_transcript("one two three four five", 0.4) == "one two"


def test_trim_transcript_prefers_complete_sentence():
    transcript = "One two. Three four five six. Seven eight."
    assert _trim_transcript(transcript, 0.6) == "One two."


def test_f5_render_batch_uses_shared_line_assembly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class FakeModel:
        target_sample_rate = 24_000

    class FakeEngine(F5TtsEngine):
        def load_model(self):
            return FakeModel()

        def synthesize_line(self, _line):
            return np.zeros(2_400), 24_000

        write_audio = staticmethod(_write_wav)

    result = FakeEngine().render_batch([_request()])

    assert result[0]["dialogue_line_start_positions"] == [0.0]
    assert not list(tmp_path.glob("*.line-*.wav"))
