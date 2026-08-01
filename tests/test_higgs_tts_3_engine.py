from __future__ import annotations

import json
import io
import wave
import base64
from pathlib import Path

from radio_drama.proxy import load_proxy_tts_configs
from tts_engines.higgs_tts_3.engine import CONTROL_TAGS, HiggsTtsEngine, expand_control_expressions


def _write_wav(path: Path, frames: bytes) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(frames)


def _wav_bytes(frames: bytes) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(frames)
    return stream.getvalue()


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


def test_higgs_control_expression_catalog_and_unknown_brackets():
    assert sum(map(len, CONTROL_TAGS.values())) == 43
    assert expand_control_expressions(
        "[style:whispering][sfx:sigh]Ahh. [unknown:thing] [aside]"
    ) == (
        "<|style:whispering|><|sfx:sigh|>Ahh. [unknown:thing] [aside]"
    )


def test_higgs_render_batch_batches_lines_and_can_retain_raw_wavs(
    tmp_path: Path, monkeypatch
):
    speaker = {
        "authored_name": "Alice",
        "voice_path": "/voices/0.wav",
        "transcript": "Reference.",
    }
    requests = [
        {
            "first_words": word,
            "dialogue_contents": [
                {"type": "line", "speaker": speaker, "spoken_text": f"{word}."}
            ],
        }
        for word in ("One", "Two")
    ]
    seen = []

    class FakeEngine(HiggsTtsEngine):
        def ensure_server(self):
            pass

        def synthesize_batch(self, lines):
            seen.append([line["spoken_text"] for line in lines])
            return [_wav_bytes(b"\x01\x00") for _ in lines]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIGGS_KEEP_LINE_WAVS", "true")

    results = FakeEngine().render_batch(requests)

    assert seen == [["One.", "Two."]]
    assert len(results) == 2
    assert sorted(path.name for path in tmp_path.glob("*.line-0.wav")) == [
        result["wav"].removesuffix(".wav") + ".line-0.wav" for result in results
    ]


def test_higgs_render_batch_honors_batch_size_and_removes_line_wavs(
    tmp_path: Path, monkeypatch
):
    request = {
        "first_words": "One two three",
        "dialogue_contents": [
            {
                "type": "line",
                "speaker": {
                    "voice_path": "/voices/0.wav",
                    "transcript": "Reference.",
                },
                "spoken_text": word,
            }
            for word in ("One", "Two", "Three")
        ],
    }
    batch_lengths = []

    class FakeEngine(HiggsTtsEngine):
        def ensure_server(self):
            pass

        def synthesize_batch(self, lines):
            batch_lengths.append(len(lines))
            return [_wav_bytes(b"\x01\x00") for _ in lines]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIGGS_BATCH_SIZE", "2")

    FakeEngine().render_batch([request])

    assert batch_lengths == [2, 1]
    assert list(tmp_path.glob("*.line-*.wav")) == []


def test_higgs_synthesis_batch_uses_sglang_batch_endpoint(monkeypatch):
    captured = {}
    wav = _wav_bytes(b"\x01\x00")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "results": [
                        {
                            "index": 0,
                            "status": "success",
                            "audio_data": base64.b64encode(wav).decode("ascii"),
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    line = {
        "spoken_text": "Hello.",
        "speaker": {
            "voice_path": "/voices/0.wav",
            "transcript": "Reference.",
        },
    }

    result = HiggsTtsEngine(base_url="http://higgs.test").synthesize_batch([line])

    assert captured["url"] == "http://higgs.test/v1/audio/speech/batch"
    assert captured["payload"]["items"] == [
        {
            "input": "Hello.",
            "references": [
                {"audio_path": "/voices/0.wav", "text": "Reference."}
            ],
        }
    ]
    assert result == [wav]


def test_higgs_synthesis_batch_can_set_initial_codec_chunk_frames(monkeypatch):
    captured = {}
    wav = _wav_bytes(b"\x01\x00")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "results": [
                        {
                            "index": 0,
                            "status": "success",
                            "audio_data": base64.b64encode(wav).decode("ascii"),
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request):
        captured["payload"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("HIGGS_INITIAL_CODEC_CHUNK_FRAMES", "8")
    line = {
        "spoken_text": "Hello.",
        "speaker": {
            "voice_path": "/voices/0.wav",
            "transcript": "Reference.",
        },
    }

    HiggsTtsEngine(base_url="http://higgs.test").synthesize_batch([line])

    assert captured["payload"]["initial_codec_chunk_frames"] == 8
