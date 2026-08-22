from __future__ import annotations

import types
import wave
from pathlib import Path

import numpy as np

from radio_drama.proxy import load_proxy_tts_configs
from tts_engines.higgs_tts_3.engine import (
    CONTROL_TAGS,
    HiggsTtsEngine,
    expand_control_expressions,
)


def _write_wav(path: str, samples, sample_rate: int) -> None:
    values = np.asarray(samples).reshape(-1)
    with wave.open(path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes((values * 32767).astype("<i2").tobytes())


def _request(words: tuple[str, ...]) -> dict:
    speaker = {
        "authored_name": "Alice",
        "voice_path": "/voices/0.wav",
        "transcript": "Reference.",
    }
    return {
        "first_words": " ".join(words),
        "dialogue_contents": [
            {"type": "line", "speaker": speaker, "spoken_text": word}
            for word in words
        ],
    }


def test_higgs_sample_proxy_config_enables_checkpoint_cache_and_gpu():
    sample = Path(__file__).resolve().parents[1] / "tts_engines/higgs_tts_3/tts.toml.example"
    config = load_proxy_tts_configs(sample)["higgs"]

    assert config.devices == ("nvidia.com/gpu=all",)
    assert config.network == "none"
    assert config.mounts[0].target == "/models/huggingface"
    assert not config.mounts[0].read_only


def test_higgs_containerfile_uses_transformers_and_offline_cache():
    containerfile = (
        Path(__file__).resolve().parents[1] / "tts_engines/higgs_tts_3/Containerfile"
    ).read_text(encoding="utf-8")

    assert "transformers==5.14.1" in containerfile
    assert "lmsysorg/sglang" not in containerfile
    assert "HF_HUB_OFFLINE=1" in containerfile
    assert "multimodalart/higgs-audio-v3-tts-4b-transformers" in containerfile


def test_higgs_control_expression_catalog_and_unknown_brackets():
    assert sum(map(len, CONTROL_TAGS.values())) == 43
    assert expand_control_expressions(
        "[style:whispering][sfx:sigh]Ahh. [unknown:thing] [aside]"
    ) == "<|style:whispering|><|sfx:sigh|>Ahh. [unknown:thing] [aside]"


def test_higgs_synthesize_line_passes_cloning_and_sampling_arguments(monkeypatch):
    seen = {}

    class FakeModel:
        def generate_speech(self, text, tokenizer, **kwargs):
            seen.update(text=text, tokenizer=tokenizer, **kwargs)
            return np.zeros(12)

    engine = HiggsTtsEngine()
    engine.model = FakeModel()
    engine.tokenizer = "tokenizer"
    engine._reference_audio["/voices/0.wav"] = ("waveform", 48_000)

    result = engine.synthesize_line(_request(("[style:whispering]Hello.",))["dialogue_contents"][0])

    assert result.shape == (12,)
    assert seen["text"] == "<|style:whispering|>Hello."
    assert seen["reference_audio"] == "waveform"
    assert seen["reference_sample_rate"] == 48_000
    assert seen["reference_text"] == "Reference."
    assert seen["temperature"] == 0.8
    assert seen["top_p"] == 1.0
    assert seen["top_k"] == 50


def test_higgs_render_batch_bounds_work_and_uses_shared_line_assembly(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIGGS_BATCH_SIZE", "2")
    seen = []

    class FakeEngine(HiggsTtsEngine):
        def load_model(self):
            return object(), object()

        def synthesize_batch(self, lines):
            seen.append([line["spoken_text"] for line in lines])
            return [np.zeros(2_400, dtype=np.float32) for _ in lines]

        @staticmethod
        def write_audio(path, audio):
            _write_wav(str(path), audio, 24_000)

    results = FakeEngine().render_batch([_request(("One", "Two", "Three"))])

    assert seen == [["One", "Two"], ["Three"]]
    assert results[0]["dialogue_line_start_positions"] == [0.0, 0.1, 0.2]
    assert not list(tmp_path.glob("*.line-*.wav"))


def test_higgs_can_retain_line_wavs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIGGS_KEEP_LINE_WAVS", "true")

    class FakeEngine(HiggsTtsEngine):
        def load_model(self):
            return object(), object()

        def synthesize_batch(self, lines):
            return [np.zeros(10, dtype=np.float32) for _ in lines]

        @staticmethod
        def write_audio(path, audio):
            _write_wav(str(path), audio, 24_000)

    FakeEngine().render_batch([_request(("One",))])

    assert len(list(tmp_path.glob("*.line-0.wav"))) == 1


def test_higgs_load_model_forces_audio_codec_float32(monkeypatch):
    import torch

    codec = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)

    class FakeModel:
        config = types.SimpleNamespace(audio_tokenizer_id=None)

        def to(self, *_args, **_kwargs):
            return self

        def eval(self):
            return self

        def get_audio_codec(self):
            return codec

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *_a, **_k: object()),
        AutoModelForCausalLM=types.SimpleNamespace(
            from_pretrained=lambda *_a, **_k: FakeModel()
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    monkeypatch.setenv("HIGGS_DEVICE", "cpu")

    engine = HiggsTtsEngine()
    engine.load_model()

    assert {parameter.dtype for parameter in codec.parameters()} == {torch.float32}
