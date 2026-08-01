from __future__ import annotations

import sys
import types
import wave
from pathlib import Path

import numpy as np

from radio_drama.proxy import load_proxy_tts_configs
from radio_drama_tts_container import finish_line_work, prepare_line_work
from tts_engines.chatterbox.engine import ChatterboxEngine
from tts_engines.zonos.engine import ZonosEngine, _EosTracker


def _request(label: str, words: tuple[str, ...]) -> dict:
    speaker = {"voice_path": "/voices/narrator.wav"}
    return {
        "first_words": label,
        "dialogue_contents": [
            {"type": "line", "spoken_text": word, "speaker": speaker}
            for word in words
        ],
    }


def _save_wav(path: str, samples, sample_rate: int) -> None:
    values = np.asarray(samples).reshape(-1)
    with wave.open(path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes((values * 32767).astype("<i2").tobytes())


def test_shared_line_assembly_preserves_script_boundaries_and_timing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outputs, work = prepare_line_work(
        [_request("first", ("one", "two")), _request("second", ("three",))]
    )
    for index, item in enumerate(work, 1):
        _save_wav(str(item.path), np.zeros(index * 10), 100)

    results = finish_line_work(outputs, work, sample_rate=100)

    assert results[0]["dialogue_line_start_positions"] == [0.0, 0.1]
    assert results[1]["dialogue_line_start_positions"] == [0.0]
    with wave.open(results[0]["wav"], "rb") as source:
        assert source.getnframes() == 30


def test_zonos_batches_lines_across_pending_scripts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZONOS_BATCH_SIZE", "2")
    monkeypatch.setitem(sys.modules, "torchaudio", types.SimpleNamespace(save=_save_wav))
    seen = []

    class FakeEngine(ZonosEngine):
        def load_model(self):
            return types.SimpleNamespace(
                autoencoder=types.SimpleNamespace(sampling_rate=100)
            )

        def synthesize_batch(self, lines):
            seen.append([line["spoken_text"] for line in lines])
            return [np.zeros((1, 10)) for _ in lines]

    results = FakeEngine().render_batch(
        [_request("first", ("one", "two")), _request("second", ("three",))]
    )

    assert seen == [["one", "two"], ["three"]]
    assert [len(result["dialogue_line_start_positions"]) for result in results] == [2, 1]
    assert not list(tmp_path.glob("*.line-*.wav"))


def test_zonos_tracks_each_batched_items_eos_before_padding():
    import torch

    tracker = _EosTracker(batch_size=2, eos_token_id=1024)

    assert tracker(torch.tensor([[[7]], [[8]]]), 1, 20)
    assert tracker(torch.tensor([[[1024]], [[9]]]), 4, 20)
    assert tracker(torch.tensor([[[1024]], [[1024]]]), 9, 20)
    assert tracker.lengths(default=20) == [4, 9]


def test_chatterbox_reuses_each_speaker_conditioning():
    calls = []
    conditions = []

    class FakeModel:
        conds = None

        def prepare_conditionals(self, path):
            calls.append(path)
            self.conds = object()

    engine = ChatterboxEngine()
    engine.model = FakeModel()

    first = engine.speaker_conditionals("/voices/one.wav")
    second = engine.speaker_conditionals("/voices/one.wav")

    assert first is second
    assert calls == ["/voices/one.wav"]


def test_new_engine_examples_enable_gpu_and_persistent_model_cache():
    root = Path(__file__).resolve().parents[1] / "tts_engines"
    zonos = load_proxy_tts_configs(root / "zonos" / "tts.toml.example")["zonos"]
    chatterbox = load_proxy_tts_configs(root / "chatterbox" / "tts.toml.example")[
        "chatterbox"
    ]

    for config in (zonos, chatterbox):
        assert config.devices == ("nvidia.com/gpu=all",)
        assert config.ipc == "host"
        assert config.environment["HF_HOME"] == "/models/huggingface"
        assert config.mounts[0].target == "/models/huggingface"
        assert not config.mounts[0].read_only
    assert zonos.environment["ZONOS_MODEL"] == "Zyphra/Zonos-v0.1-transformer"
