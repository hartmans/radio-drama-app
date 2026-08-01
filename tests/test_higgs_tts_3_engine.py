from __future__ import annotations

import json
import wave
from pathlib import Path

from radio_drama.proxy import load_proxy_tts_configs
from tts_engines.higgs_tts_3.engine import HiggsTtsEngine


def _write_wav(path: Path, frames: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(frames)


def test_higgs_sample_proxy_config_enables_checkpoint_cache_and_gpu():
    sample = (
        Path(__file__).resolve().parents[1]
        / "tts_engines"
        / "higgs_tts_3"
        / "tts.toml.example"
    )

    config = load_proxy_tts_configs(sample)["higgs"]

    assert config.devices == ("nvidia.com/gpu=all",)
    assert config.ipc == "host"
    assert config.shm_size == "32g"
    assert config.mounts[0].target == "/models/huggingface"
    assert not config.mounts[0].read_only


def test_higgs_engine_maps_each_line_to_its_voice_and_concatenates(
    tmp_path: Path, monkeypatch
):
    request = {
        "first_words": "One two",
        "dialogue_contents": [
            {
                "type": "line",
                "speaker": "Alice",
                "voice_path": "/voices/0.wav",
                "spoken_text": "One.",
                "source": "tts",
            },
            {"type": "gap", "mode": "exclude", "label": "gap"},
            {
                "type": "line",
                "speaker": "Bob",
                "voice_path": "/voices/1.wav",
                "spoken_text": "Two.",
                "source": "recording",
            },
        ],
    }
    seen_lines = []

    class FakeEngine(HiggsTtsEngine):
        def synthesize_line(self, line, output_path):
            seen_lines.append(line)
            _write_wav(output_path, b"\x01\x00\x02\x00")

    monkeypatch.chdir(tmp_path)
    result = FakeEngine().render_request(request)

    assert [line["voice_path"] for line in seen_lines] == [
        "/voices/0.wav",
        "/voices/1.wav",
    ]
    assert result["dialogue_line_start_positions"] == [0.0, 2 / 24_000]
    with wave.open(str(tmp_path / result["wav"]), "rb") as output:
        assert output.getnframes() == 4


def test_higgs_synthesis_uses_openai_voice_clone_request(tmp_path: Path, monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"RIFF-test"

    def fake_urlopen(request):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    output = tmp_path / "line.wav"

    HiggsTtsEngine(base_url="http://higgs.test").synthesize_line(
        {
            "spoken_text": "Hello there.",
            "voice_path": "/voices/7.wav",
        },
        output,
    )

    assert captured["url"] == "http://higgs.test/v1/audio/speech"
    assert captured["payload"]["model"] == "bosonai/higgs-tts-3-4b"
    assert captured["payload"]["input"] == "Hello there."
    assert captured["payload"]["references"] == [
        {"audio_path": "/voices/7.wav"}
    ]
    assert output.read_bytes() == b"RIFF-test"
