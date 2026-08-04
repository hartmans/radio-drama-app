from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sound_tools import voxcpm2_voice_design


class _Inputs(dict):
    def to(self, device, dtype):
        self.device = device
        self.dtype = dtype
        return self


class _OutputIds:
    def __getitem__(self, key):
        return key


def test_asr_uses_vibevoice_transcription_api(monkeypatch):
    audio_path = Path("speech.wav")
    inputs = _Inputs(input_ids=SimpleNamespace(shape=(1, 7)))
    generated = _OutputIds()
    calls = {}

    class FakeProcessor:
        def apply_transcription_request(self, **kwargs):
            calls["request"] = kwargs
            return inputs

        def decode(self, generated_ids, **kwargs):
            calls["decode"] = (generated_ids, kwargs)
            return ["hello from VibeVoice"]

    class FakeModel:
        device = "cuda"
        dtype = "bfloat16"

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return generated

    monkeypatch.setattr(
        voxcpm2_voice_design,
        "_ensure_asr_model",
        lambda: (FakeProcessor(), FakeModel()),
    )

    assert voxcpm2_voice_design.asr(audio_path) == "hello from VibeVoice"
    assert calls["request"] == {"audio": str(audio_path)}
    assert calls["generate"] == {
        "input_ids": inputs["input_ids"],
        "max_new_tokens": voxcpm2_voice_design.ASR_MAX_NEW_TOKENS,
    }
    assert calls["decode"] == (
        (slice(None, None, None), slice(7, None, None)),
        {"return_format": "transcription_only"},
    )
