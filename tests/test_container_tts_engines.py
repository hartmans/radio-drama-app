from __future__ import annotations

import sys
import types
import wave
from pathlib import Path

import numpy as np

from radio_drama.proxy import load_proxy_tts_configs
from radio_drama_tts_container import artifact_name, finish_line_work, prepare_line_work
from tts_engines.chatterbox.engine import ChatterboxEngine
from tts_engines.zonos.engine import (
    ZonosEngine,
    _EosTracker,
    _finish_prerolled_audio,
    _generate_nonempty_codes,
)
from tts_engines.voxcpm2.engine import VoxCPM2Engine
from tts_engines.moss_ttsd.engine import MossTtsdEngine, PreparedRequest


def _request(label: str, words: tuple[str, ...]) -> dict:
    speaker = {"voice_path": "/voices/narrator.wav"}
    return {
        "first_words": label,
        "dialogue_contents": [
            {"type": "line", "spoken_text": word, "speaker": speaker} for word in words
        ],
    }


def _save_wav(path: str, samples, sample_rate: int) -> None:
    values = np.asarray(samples).reshape(-1)
    with wave.open(path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes((values * 32767).astype("<i2").tobytes())


def test_shared_line_assembly_preserves_script_boundaries_and_timing(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    outputs, work = prepare_line_work(
        [_request("first", ("one", "two")), _request("second", ("three",))]
    )
    for index, item in enumerate(work, 1):
        _save_wav(str(item.path), np.zeros(index * 10), 100)

    results = finish_line_work(outputs, work, sample_rate=100)

    assert results[0]["dialogue_line_spans"] == [[0.0, 0.1], [0.1, 0.3]]
    assert results[1]["dialogue_line_spans"] == [[0.0, 0.3]]
    with wave.open(results[0]["wav"], "rb") as source:
        assert source.getnframes() == 30


def test_zonos_batches_lines_across_pending_scripts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZONOS_BATCH_SIZE", "2")
    monkeypatch.setitem(
        sys.modules, "torchaudio", types.SimpleNamespace(save=_save_wav)
    )
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
    assert [len(result["dialogue_line_spans"]) for result in results] == [
        2,
        1,
    ]
    assert not list(tmp_path.glob("*.line-*.wav"))


def test_zonos_tracks_each_batched_items_eos_before_padding():
    import torch

    tracker = _EosTracker(batch_size=2, eos_token_id=1024)

    assert tracker(torch.empty((2, 9, 0), dtype=torch.long), 0, 20)
    assert tracker(torch.tensor([[[7]], [[8]]]), 1, 20)
    assert tracker(torch.tensor([[[1024]], [[9]]]), 4, 20)
    assert tracker(torch.tensor([[[1024]], [[1024]]]), 9, 20)
    assert tracker.lengths(default=20) == [4, 9]


def test_zonos_retries_initial_eos_before_decoding():
    import torch

    class FakeModel:
        eos_token_id = 1024

        def __init__(self):
            self.calls = 0

        def generate(self, _conditioning, **_kwargs):
            self.calls += 1
            length = 0 if self.calls == 1 else 5
            return torch.zeros((2, 9, length), dtype=torch.long)

    model = FakeModel()
    codes, tracker = _generate_nonempty_codes(model, object(), batch_size=2, attempts=3)

    assert model.calls == 2
    assert codes.shape == (2, 9, 5)
    assert tracker.lengths(default=codes.shape[-1]) == [5, 5]


def test_zonos_removes_silence_context_and_fades_onset():
    import torch

    audio = torch.ones((1, 1200))
    result = _finish_prerolled_audio(audio, prefix_samples=200, sample_rate=1000)

    assert result.shape == (1, 1000)
    assert result[0, 0] == 0
    assert 0 < result[0, 1] < 1
    assert result[0, 4] == 1


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


def test_voxcpm2_holds_the_controlled_line_as_the_continuation_prompt(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    calls = []

    class FakeModel:
        tts_model = types.SimpleNamespace(sample_rate=48_000)

        def generate(self, **kwargs):
            calls.append(kwargs)
            return np.zeros(48_000, dtype=np.float32)

    engine = VoxCPM2Engine()
    engine.model = FakeModel()
    requests = [
        _request("first", ("one", "(flustered) two", "three", "four"))
    ]
    requests[0]["dialogue_contents"][0]["speaker"]["transcript"] = "Reference."

    results = engine.render_batch(requests)

    assert [call["text"] for call in calls] == [
        "one",
        "(flustered) two",
        "three",
        "four",
    ]
    assert calls[0]["reference_wav_path"] == "/voices/narrator.wav"
    assert calls[0]["prompt_wav_path"] == "/voices/narrator.wav"
    assert calls[0]["prompt_text"] == "Reference."
    assert calls[1]["reference_wav_path"] == "/voices/narrator.wav"
    assert "prompt_wav_path" not in calls[1]
    assert calls[2]["reference_wav_path"] == "/voices/narrator.wav"
    assert calls[2]["prompt_wav_path"].endswith(".line-1.wav")
    assert calls[2]["prompt_text"] == "two"
    assert calls[3]["reference_wav_path"] == "/voices/narrator.wav"
    assert calls[3]["prompt_wav_path"].endswith(".line-1.wav")
    assert calls[3]["prompt_text"] == "two"
    assert results[0]["dialogue_line_spans"] == [
        [0.0, 1.0],
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ]
    assert not list(tmp_path.glob("*.line-*.wav"))

    engine.synthesize_line(
        {"spoken_text": "fallback", "speaker": {"voice_path": "/voices/other.wav"}}
    )
    assert calls[-1]["reference_wav_path"] == "/voices/other.wav"
    assert "prompt_wav_path" not in calls[-1]


def test_moss_ttsd_batches_complete_scripts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOSS_TTSD_BATCH_SIZE", "2")
    seen = []

    class FakeEngine(MossTtsdEngine):
        def prepare_request(self, request):
            return PreparedRequest(
                output_path=tmp_path / f"{request['first_words']}.wav",
                conversation=request["first_words"],
            )

        def synthesize_batch(self, prepared):
            seen.append([item.conversation for item in prepared])
            return [np.zeros(24_000) for _ in prepared]

        def write_audio(self, path, _audio):
            _save_wav(str(path), np.zeros(24_000), 24_000)

    results = FakeEngine().render_batch(
        [
            _request("first", ("one",)),
            _request("second", ("two",)),
            _request("third", ("three",)),
        ]
    )

    assert seen == [["first", "second"], ["third"]]
    assert [result["wav"] for result in results] == [
        "first.wav",
        "second.wav",
        "third.wav",
    ]


def test_moss_ttsd_normalizes_text_and_coalesces_same_speaker_turns(monkeypatch):
    seen = {}

    class FakeProcessor:
        model_config = types.SimpleNamespace(sampling_rate=24_000)

        def encode_audios_from_wav(self, _wavs, *, sampling_rate):
            assert sampling_rate == 24_000
            return ["prompt-audio"]

        def build_user_message(self, **kwargs):
            seen.update(kwargs)
            return kwargs

        def build_assistant_message(self, **kwargs):
            return kwargs

    class FakeEngine(MossTtsdEngine):
        def load_model(self):
            return object(), FakeProcessor()

        def _reference_audio(self, path):
            return np.zeros((1, 1)), f"codes-{path}"

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cat=lambda values, dim: np.concatenate(values, axis=dim)),
    )
    first = {
        "authored_name": "first",
        "voice_path": "/voices/first.wav",
        "transcript": "“Reference”—one…",
    }
    second = {
        "authored_name": "second",
        "voice_path": "/voices/second.wav",
        "transcript": "“Reference two”.",
    }
    FakeEngine().prepare_request(
        {
            "first_words": "test",
            "dialogue_contents": [
                {"type": "line", "speaker": first, "spoken_text": "“One”—here."},
                {"type": "line", "speaker": first, "spoken_text": "Then… there."},
                {"type": "line", "speaker": second, "spoken_text": "“Two”."},
                {"type": "line", "speaker": first, "spoken_text": "Three."},
            ],
        }
    )

    assert seen["text"] == (
        '[S1] "Reference"---one... [S2] "Reference two". '
        '[S1] "One"---here. Then... there. [S2] "Two". [S1] Three.'
    )


def test_moss_ttsd_accepts_empty_scripts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = MossTtsdEngine().render_batch(
        [{"first_words": "empty", "dialogue_contents": []}]
    )

    assert results == [
        {"wav": artifact_name({"first_words": "empty", "dialogue_contents": []})}
    ]
    with wave.open(results[0]["wav"], "rb") as source:
        assert source.getframerate() == 24_000
        assert source.getnframes() == 0


def test_new_engine_examples_enable_gpu_and_persistent_model_cache():
    root = Path(__file__).resolve().parents[1] / "tts_engines"
    zonos = load_proxy_tts_configs(root / "zonos" / "tts.toml.example")["zonos"]
    chatterbox = load_proxy_tts_configs(root / "chatterbox" / "tts.toml.example")[
        "chatterbox"
    ]
    voxcpm2 = load_proxy_tts_configs(root / "voxcpm2" / "tts.toml.example")["voxcpm2"]
    moss_ttsd = load_proxy_tts_configs(root / "moss_ttsd" / "tts.toml.example")[
        "moss-ttsd"
    ]

    for config in (zonos, chatterbox, voxcpm2, moss_ttsd):
        assert config.devices == ("nvidia.com/gpu=all",)
        assert config.ipc == "host"
        assert config.environment["HF_HOME"] == "/models/huggingface"
        assert config.mounts[0].target == "/models/huggingface"
        assert not config.mounts[0].read_only
    assert zonos.environment["ZONOS_MODEL"] == "Zyphra/Zonos-v0.1-transformer"
    assert voxcpm2.environment["VOXCPM_MODEL"] == "openbmb/VoxCPM2"
    assert moss_ttsd.environment["MOSS_TTSD_BATCH_SIZE"] == "10"
