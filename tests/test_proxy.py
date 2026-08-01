from __future__ import annotations

import io
import asyncio
import json
import sys
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from carthage.dependency_injection import AsyncInjector, InjectionKey
from radio_drama.proxy import ProxyMount, load_proxy_tts_configs
from radio_drama.proxy import ProxyTtsConfig, ProxyTtsResource
from radio_drama.cache import CACHE_DIRECTORY_KEY
from radio_drama.config import ProductionConfig
from radio_drama.dialogue import DialogueLine, ScriptRenderRequest, SpeakerVoiceReference
from radio_drama.init import radio_drama_injector
from radio_drama.rendering import RenderResult
from radio_drama.voice_reference import VoiceReferenceTranscriptionResource
from radio_drama_tts_container import artifact_name, run_server, write_pcm16_wav


def test_load_proxy_tts_configs(tmp_path: Path):
    config_path = tmp_path / "tts.toml"
    config_path.write_text(
        """
        [tts.demo]
        image = "localhost/demo:latest"
        command = ["python", "/engine.py"]
        devices = ["nvidia.com/gpu=all"]
        ipc = "host"
        shm_size = "32g"

        [tts.demo.environment]
        MODEL = "/models/demo"

        [[tts.demo.mounts]]
        source = "~/models"
        target = "/models"
        mode = "ro"
        """,
        encoding="utf-8",
    )

    config = load_proxy_tts_configs(config_path)["demo"]

    assert config.image == "localhost/demo:latest"
    assert config.command == ("python", "/engine.py")
    assert config.environment == {"MODEL": "/models/demo"}
    assert config.devices == ("nvidia.com/gpu=all",)
    assert config.ipc == "host"
    assert config.shm_size == "32g"
    assert config.mounts == (
        ProxyMount(
            source=Path("~/models").expanduser(),
            target="/models",
            read_only=True,
        ),
    )


def test_proxy_podman_command_omits_shm_size_for_host_ipc(tmp_path: Path):
    resource = object.__new__(ProxyTtsResource)
    resource.proxy_config = ProxyTtsConfig(
        name="demo",
        image="localhost/demo:latest",
        ipc="host",
        shm_size="32g",
    )
    resource._voice_paths = {}
    resource._voice_mounts = {}

    command = resource._podman_command(tmp_path)

    assert "--ipc=host" in command
    assert "--shm-size=32g" not in command


def test_container_server_handshake_and_render():
    request = {
        "first_words": "Hello there",
        "dialogue_contents": [{"type": "line", "spoken_text": "Hello"}],
    }
    input_stream = io.StringIO(
        "\n".join(
            (
                json.dumps({"protocol": "radio-drama-tts", "versions": [1]}),
                json.dumps(
                    {
                        "protocol": "radio-drama-tts",
                        "version": 1,
                        "id": 7,
                        "method": "render_batch",
                        "requests": [request],
                    }
                ),
                "",
            )
        )
    )
    output_stream = io.StringIO()

    def render_with_library_noise(requests):
        print("third-party model status")
        return [{"wav": artifact_name(requests[0])}]

    run_server(
        render_with_library_noise,
        capabilities={"needs_transcript"},
        input_stream=input_stream,
        output_stream=output_stream,
    )

    responses = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert responses[0] == {
        "protocol": "radio-drama-tts",
        "version": 1,
        "ready": True,
        "capabilities": ["needs_transcript"],
    }
    assert responses[1] == {"id": 7, "results": [{"wav": artifact_name(request)}]}


def test_write_pcm16_wav_uses_standard_library(tmp_path: Path):
    output_path = tmp_path / "sample.wav"

    write_pcm16_wav(output_path, (-1.0, 0.0, 1.0), sample_rate=8000)

    with wave.open(str(output_path), "rb") as source:
        assert source.getframerate() == 8000
        assert source.getnchannels() == 1
        assert source.getnframes() == 3


def test_proxy_resource_renders_through_stub_engine(tmp_path: Path):
    voice_path = tmp_path / "voice.wav"
    write_pcm16_wav(
        voice_path,
        0.1 * np.sin(2 * np.pi * 220 * np.arange(16_000) / 8_000),
        sample_rate=8000,
    )
    engine_path = Path(__file__).resolve().parents[1] / "tts_engines" / "stub" / "engine.py"

    class StubProxyResource(ProxyTtsResource):
        proxy_config = ProxyTtsConfig(name="stub", image="unused")

        def _podman_command(self, cache_directory: Path) -> list[str]:
            repo_root = engine_path.parents[2]
            return [
                sys.executable,
                "-c",
                "import runpy, sys; sys.path.insert(0, sys.argv[1]); runpy.run_path(sys.argv[2], run_name='__main__')",
                str(repo_root),
                str(engine_path),
            ]

    async def runner():
        injector = radio_drama_injector(
            config=ProductionConfig(output_sample_rate=8000, output_channels=1),
            event_loop=asyncio.get_running_loop(),
        )
        injector.add_provider(CACHE_DIRECTORY_KEY, tmp_path / "cache")
        try:
            resource = await injector(AsyncInjector)(StubProxyResource)
            request = ScriptRenderRequest(
                dialogue_lines=[
                    DialogueLine(
                        speaker=SpeakerVoiceReference(
                            authored_name="Narrator",
                            voice_name="voice",
                            resolved_path=voice_path,
                        ),
                        spoken_text="Testing the proxy.",
                    )
                ],
                first_words="Testing the proxy",
            )
            registration = await resource.register_request(request)
            result = await registration.render()
            process = resource._process
            resource.close()
            assert process is not None
            await process.wait()
            return result
        finally:
            injector.close()

    result = asyncio.run(runner())

    assert result.audio.shape == (2000,)


def test_proxy_transcribes_shared_reference_only_when_capability_requires_it(
    tmp_path: Path,
):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"fake")
    seen = {"transcriptions": 0, "exchanges": 0}

    class FakeTranscriptionResource:
        async def transcribe(self, reference):
            seen["transcriptions"] += 1
            reference.transcript = "Reference words."
            return reference.transcript

    class FakeProxyResource(ProxyTtsResource):
        proxy_config = ProxyTtsConfig(name="fake", image="unused")

        async def _ensure_process(self, requests):
            self._capabilities = {"needs_transcript"}
            self._voice_paths = {"narrator": "/voices/0.wav"}

        async def _exchange(self, message):
            seen["exchanges"] += 1
            seen["message"] = message
            output = tmp_path / "cache" / "result.wav"
            output.parent.mkdir(parents=True, exist_ok=True)
            write_pcm16_wav(output, (0.0,), sample_rate=8000)
            return {
                "id": message["id"],
                "results": [
                    {
                        "wav": "result.wav",
                        "dialogue_line_start_positions": [0.0, 0.0001],
                    }
                ],
            }

    async def runner():
        injector = radio_drama_injector(
            config=ProductionConfig(output_sample_rate=8000, output_channels=1),
            event_loop=asyncio.get_running_loop(),
        )
        injector.add_provider(CACHE_DIRECTORY_KEY, tmp_path / "cache")
        injector.replace_provider(
            InjectionKey(VoiceReferenceTranscriptionResource),
            FakeTranscriptionResource(),
            close=False,
        )
        try:
            resource = await injector(AsyncInjector)(FakeProxyResource)
            speaker = SpeakerVoiceReference(
                authored_name="Narrator",
                voice_name="voice",
                resolved_path=voice_path,
            )
            request = ScriptRenderRequest(
                dialogue_lines=[
                    DialogueLine(speaker=speaker, spoken_text="One."),
                    DialogueLine(speaker=speaker, spoken_text="Two."),
                ]
            )
            result = await (await resource.register_request(request)).render()
            replay = await (await resource.register_request(request)).render()
            return speaker, result, replay
        finally:
            injector.close()

    speaker, result, replay = asyncio.run(runner())

    assert isinstance(result, RenderResult)
    assert isinstance(replay, RenderResult)
    assert np.array_equal(result.audio, replay.audio)
    assert result.dialogue_line_start_positions == replay.dialogue_line_start_positions
    assert speaker.transcript == "Reference words."
    assert seen["transcriptions"] == 1
    assert seen["exchanges"] == 1
    cache_payload = json.loads(
        next((tmp_path / "cache").glob("fake_*.json")).read_text(encoding="utf-8")
    )
    assert cache_payload["dialogue_line_start_positions"] == [0.0, 0.0001]
    serialized_lines = seen["message"]["requests"][0]["dialogue_contents"]
    assert serialized_lines[0]["speaker"] == {
        "authored_name": "Narrator",
        "voice_name": "voice",
        "voice_path": "/voices/0.wav",
        "transcript": "Reference words.",
        "gain": 0.0,
    }
    assert serialized_lines[0]["speaker"] == serialized_lines[1]["speaker"]


def test_proxy_prepares_and_reuses_normalized_voice_by_speaker_name(
    tmp_path: Path, monkeypatch
):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"source is replaced by the fake loader")
    calls = []

    def fake_load(path, *, gain_db):
        calls.append((Path(path), gain_db))
        return np.array([0.0, 0.25, -0.25], dtype=np.float32), 16_000

    monkeypatch.setattr("radio_drama.proxy.load_preprocessed_voice_reference", fake_load)
    resource = object.__new__(ProxyTtsResource)
    speaker = SpeakerVoiceReference(
        authored_name=" Narrator ",
        voice_name="voice",
        resolved_path=voice_path,
        gain=3.0,
    )
    request = ScriptRenderRequest(
        dialogue_lines=[DialogueLine(speaker=speaker, spoken_text="Hello.")]
    )

    resource._prepare_voice_references([request], tmp_path / "cache")
    first_mounts = dict(resource._voice_mounts)
    resource._prepare_voice_references([request], tmp_path / "cache")

    assert calls == [(voice_path, 3.0)]
    assert resource._voice_paths == {"narrator": "/voices/0.wav"}
    assert resource._voice_mounts == first_mounts
    cached_path = next(iter(resource._voice_mounts))
    assert cached_path.parent == (tmp_path / "cache" / "normalized_voices").resolve()
    audio, sample_rate = sf.read(cached_path, dtype="float32")
    assert sample_rate == 16_000
    assert np.allclose(audio, [0.0, 0.25, -0.25], atol=1 / 32768)
