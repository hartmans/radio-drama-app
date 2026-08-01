from __future__ import annotations

import json
import wave
from pathlib import Path

from radio_drama.proxy import load_proxy_tts_configs
from tts_engines.higgs_tts_3.engine import CONTROL_TAGS, HiggsTtsEngine, expand_control_expressions


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
    assert config.shm_size is None
    assert config.mounts[0].target == "/models/huggingface"
    assert not config.mounts[0].read_only


def test_higgs_containerfile_installs_and_launches_local_sglang():
    containerfile = (
        Path(__file__).resolve().parents[1]
        / "tts_engines"
        / "higgs_tts_3"
        / "Containerfile"
    ).read_text(encoding="utf-8")

    assert "FROM docker.io/lmsysorg/sglang-omni:dev" in containerfile
    assert "uv venv --python 3.12 --system-site-packages" in containerfile
    assert "uv pip install --python .venv/bin/python" in containerfile
    assert "PATH=/opt/sglang-omni/.venv/bin:$PATH" in containerfile
    assert 'VOLUME ["/models/huggingface"]' in containerfile
    assert 'ENTRYPOINT ["python", "/opt/higgs_tts_3_engine.py"]' in containerfile


def test_higgs_engine_maps_each_line_to_its_voice_and_concatenates(
    tmp_path: Path, monkeypatch
):
    request = {
        "first_words": "One two",
        "dialogue_contents": [
            {
                "type": "line",
                "speaker": {
                    "authored_name": "Alice",
                    "voice_path": "/voices/0.wav",
                    "transcript": "Alice reference.",
                },
                "spoken_text": "One.",
                "source": "tts",
            },
            {"type": "gap", "mode": "exclude", "label": "gap"},
            {
                "type": "line",
                "speaker": {
                    "authored_name": "Bob",
                    "voice_path": "/voices/1.wav",
                    "transcript": "Bob reference.",
                },
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

    assert [line["speaker"]["voice_path"] for line in seen_lines] == [
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
            "spoken_text": "[emotion:affection]Hello [prosody:pause] there.",
            "speaker": {
                "voice_path": "/voices/7.wav",
                "transcript": "The reference transcript.",
            },
        },
        output,
    )

    assert captured["url"] == "http://higgs.test/v1/audio/speech"
    assert captured["payload"]["model"] == "bosonai/higgs-tts-3-4b"
    assert captured["payload"]["input"] == (
        "<|emotion:affection|>Hello <|prosody:pause|> there."
    )
    assert captured["payload"]["references"] == [
        {
            "audio_path": "/voices/7.wav",
            "text": "The reference transcript.",
        }
    ]
    assert output.read_bytes() == b"RIFF-test"


def test_higgs_control_expression_catalog_and_unknown_brackets():
    assert sum(map(len, CONTROL_TAGS.values())) == 43
    assert expand_control_expressions(
        "[style:whispering][sfx:sigh]Ahh. [unknown:thing] [aside]"
    ) == (
        "<|style:whispering|><|sfx:sigh|>Ahh. [unknown:thing] [aside]"
    )
