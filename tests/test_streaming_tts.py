from __future__ import annotations

import asyncio
import sys
import threading
import types
from contextlib import aclosing
from pathlib import Path

import numpy as np
import pytest

from carthage.dependency_injection import AsyncInjector
from radio_drama.audio import convert_audio_format
from radio_drama.cache import CACHE_DIRECTORY_KEY
from radio_drama.config import ProductionConfig
from radio_drama.dialogue import DialogueLine, ScriptRenderRequest, SpeakerVoiceReference
from radio_drama.init import radio_drama_injector
from radio_drama.proxy import ProxyTtsConfig, ProxyTtsResource
from radio_drama.rendering import BackendTtsResult
from radio_drama.stream_audio import StreamingAudioConverter
from radio_drama.tts_cache import CachedTtsRequest
from radio_drama_tts_container import run_in_thread, write_pcm16_wav
from tts_engines.voxcpm2.engine import VoxCPM2Engine


def request(name="Narrator", words="Hello"):
    return ScriptRenderRequest(dialogue_lines=[DialogueLine(
        speaker=SpeakerVoiceReference(authored_name=name, voice_name=name, resolved_path=Path("voice.wav")),
        spoken_text=words,
    )])


def cached(backend, *, rate=8000, channels=1):
    resource = types.SimpleNamespace(
        config=ProductionConfig(output_sample_rate=rate, output_channels=channels),
        cache_manager={"fake": types.SimpleNamespace(enabled=False)},
        cache_collection_name="fake",
    )
    return CachedTtsRequest(resource=resource, request=request(), _backend_registration=backend)


async def collect(registration):
    return [chunk async for chunk in registration.render_stream()]


def test_shared_stream_replays_for_slow_late_and_cancelled_consumers():
    async def run():
        released = asyncio.Event()

        class Backend:
            calls = 0

            async def render_stream(self):
                self.calls += 1
                yield BackendTtsResult(audio=np.full(10, .1), sample_rate=8000)
                await released.wait()
                yield BackendTtsResult(audio=np.full(20, .2), sample_rate=8000)

            async def render(self):
                raise AssertionError("streaming must satisfy render without duplicate inference")

        backend = Backend()
        registration = cached(backend)
        slow = registration.render_stream()
        first = await anext(slow)
        fast = asyncio.create_task(collect(registration))
        cancelled = registration.render_stream()
        await anext(cancelled)
        await cancelled.aclose()
        waiter = asyncio.create_task(registration.render())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        released.set()
        result = await registration.render()
        fast_chunks = await fast
        slow_chunks = [first, *[chunk async for chunk in slow]]
        late_chunks = await collect(registration)
        for chunks in (fast_chunks, slow_chunks, late_chunks):
            assert np.allclose(np.concatenate([c.audio for c in chunks]), result.audio)
        assert result.frame_count == 30
        assert backend.calls == 1

    asyncio.run(run())


def test_failed_attempt_is_shared_and_later_call_retries_from_start():
    async def run():
        fail = asyncio.Event()

        class Backend:
            calls = 0

            async def render_stream(self):
                self.calls += 1
                yield BackendTtsResult(audio=np.array([self.calls / 10]), sample_rate=8000)
                if self.calls == 1:
                    await fail.wait()
                    raise RuntimeError("failed attempt")
                yield BackendTtsResult(audio=np.array([.3]), sample_rate=8000)

        backend = Backend()
        registration = cached(backend)
        first = registration.render_stream()
        second = registration.render_stream()
        await anext(first)
        await anext(second)
        fail.set()
        for iterator in (first, second):
            with pytest.raises(RuntimeError, match="failed attempt"):
                await anext(iterator)
        retry = await collect(registration)
        assert np.allclose(np.concatenate([c.audio for c in retry]), [.2, .3])
        assert np.allclose((await registration.render()).audio, [.2, .3])
        assert backend.calls == 2

    asyncio.run(run())


def test_streaming_fallback_and_batch_started_first_share_result():
    async def run():
        started = asyncio.Event()
        finish = asyncio.Event()

        class Backend:
            calls = 0

            async def render(self):
                self.calls += 1
                started.set()
                await finish.wait()
                return BackendTtsResult(audio=np.full(100, .25), sample_rate=8000)

        backend = Backend()
        registration = cached(backend)
        batch = asyncio.create_task(registration.render())
        await started.wait()
        streaming = asyncio.create_task(collect(registration))
        finish.set()
        result = await batch
        chunks = await streaming
        assert len(chunks) == 1
        assert np.array_equal(chunks[0].audio, result.audio)
        assert backend.calls == 1
        # Starting with streaming also falls back for a non-streaming backend.
        fallback = cached(backend)
        assert len(await collect(fallback)) == 1
        assert backend.calls == 2

    asyncio.run(run())


@pytest.mark.parametrize("input_rate,output_rate", [(48000, 44100), (24000, 8000), (8000, 48000), (8000, 8000)])
@pytest.mark.parametrize("channels", [1, 2])
def test_stream_conversion_matches_whole_audio_across_arbitrary_chunks(input_rate, output_rate, channels):
    audio = np.random.default_rng(8).uniform(-.5, .5, (3001, 2)).astype(np.float32)
    converter = StreamingAudioConverter(input_rate, output_rate, channels)
    pieces = [audio[:1], audio[1:8], audio[8:555], audio[555:777], audio[777:]]
    converted = [converter.feed(piece) for piece in pieces]
    converted.append(converter.feed(np.empty((0, 2)), final=True))
    expected = convert_audio_format(audio, input_sample_rate=input_rate,
                                    output_sample_rate=output_rate, output_channels=channels)
    result = np.concatenate(converted)
    assert result.shape == expected.shape
    assert np.allclose(result, expected, atol=1e-6)


@pytest.fixture
def streaming_proxy(tmp_path):
    """Exercise actual pipes and sockets with a model-free async container server."""
    engine_path = tmp_path / "engine.py"
    engine_path.write_text('''
import asyncio
import struct
import sys
from pathlib import Path
from radio_drama_tts_container import artifact_name, run_server, write_pcm16_wav

async def render_batch(requests):
    # Let the test hold a batch open while the stream command is accepted.
    Path("batch_started").touch()
    while not Path("release_batch").exists():
        await asyncio.sleep(.01)
    results = []
    for request in requests:
        path = artifact_name(request)
        write_pcm16_wav(path, [.1] * 16, sample_rate=8000)
        results.append({"wav": path})
    return results

async def stream_request(request):
    Path("stream_started").touch()
    try:
        # Fragment frames and write enough to require repeated socket reads.
        data = struct.pack("<f", .25) * 40000
        yield data[:3]
        yield data[3:10000]
        while not Path("release_stream").exists():
            await asyncio.sleep(.01)
        yield data[10000:]
        if request["dialogue_contents"][0]["spoken_text"] == "error":
            raise RuntimeError("model error after partial audio")
    finally:
        Path("stream_closed").touch()

run_server(render_batch, stream_request=stream_request, stream_sample_rate=8000,
           socket_directory=Path(sys.argv[1]))
''')
    repo = Path(__file__).resolve().parents[1]

    class Resource(ProxyTtsResource):
        proxy_config = ProxyTtsConfig(name="streamfake", image="unused")

        def _podman_command(self, cache_directory):
            return [sys.executable, "-c",
                    "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_path(sys.argv.pop(1),run_name='__main__')",
                    str(repo), str(engine_path), str(self._stream_directory)]

    return Resource


async def wait_file(path):
    async def wait():
        while not path.exists():
            await asyncio.sleep(.01)
    await asyncio.wait_for(wait(), 5)


async def injected_resource(resource_type, tmp_path):
    voice = tmp_path / "voice.wav"
    write_pcm16_wav(voice, np.zeros(8000), sample_rate=8000)
    injector = radio_drama_injector(
        config=ProductionConfig(output_sample_rate=8000, output_channels=1),
        event_loop=asyncio.get_running_loop(),
    )
    injector.add_provider(CACHE_DIRECTORY_KEY, tmp_path / "cache")
    resource = await injector(AsyncInjector)(resource_type)
    return injector, resource, voice


def test_proxy_multiplexes_stream_during_batch_prepares_new_voice_and_persists_replay(tmp_path, streaming_proxy):
    async def run():
        injector, resource, voice = await injected_resource(streaming_proxy, tmp_path)
        cache_dir = tmp_path / "cache"
        try:
            batch_request = request(words="batch")
            batch_request.dialogue_lines[0].speaker.resolved_path = voice
            batch = await resource.register_request(batch_request)
            batch_task = asyncio.create_task(batch.render())
            await wait_file(cache_dir / "batch_started")
            process = resource._process
            stream_request = request(name="New voice", words="stream")
            stream_request.dialogue_lines[0].speaker.resolved_path = voice
            # A distinct source exercises additional preparation after startup.
            new_voice = tmp_path / "new_voice.wav"
            write_pcm16_wav(new_voice, np.zeros(8000), sample_rate=8000)
            stream_request.dialogue_lines[0].speaker.resolved_path = new_voice
            stream = await resource.register_request(stream_request)
            iterator = stream.render_stream()
            prefix = await asyncio.wait_for(anext(iterator), 5)
            assert prefix.frame_count > 0
            assert not batch_task.done()
            assert resource._process is process
            path = resource._voice_paths["new voice"]
            assert path.startswith("/cache/normalized_voices/")
            assert (cache_dir / path.removeprefix("/cache/")).exists()
            (cache_dir / "release_stream").touch()
            rendered = await asyncio.wait_for(stream.render(), 5)
            # The producer drained everything while this consumer was idle.
            chunks = [prefix, *[c async for c in iterator]]
            assert rendered.frame_count == 40000
            assert np.allclose(np.concatenate([c.audio for c in chunks]), rendered.audio, atol=1/32768)
            replay = await resource.register_request(stream_request)
            replay_chunks = await collect(replay)
            assert len(replay_chunks) == 1
            assert np.allclose(replay_chunks[0].audio, rendered.audio, atol=1/32768)
            (cache_dir / "release_batch").touch()
            assert (await asyncio.wait_for(batch_task, 5)).frame_count == 16
            assert not list(resource._stream_directory.glob("*.sock"))
        finally:
            process = resource._process
            resource.close()
            if process is not None:
                await process.wait()
            injector.close()

    asyncio.run(run())


def test_container_stream_error_completes_partial_audio_and_stays_alive(tmp_path, streaming_proxy):
    async def run():
        injector, resource, voice = await injected_resource(streaming_proxy, tmp_path)
        try:
            req = request(words="error")
            req.dialogue_lines[0].speaker.resolved_path = voice
            stream = await resource.register_request(req)
            (tmp_path / "cache").mkdir(exist_ok=True)
            (tmp_path / "cache" / "release_stream").touch()
            chunks = await collect(stream)
            assert sum(c.frame_count for c in chunks) == 40000
            assert (await stream.render()).frame_count == 40000
            assert resource._process.returncode is None
        finally:
            process = resource._process
            resource.close()
            if process is not None:
                await process.wait()
            injector.close()

    asyncio.run(run())


def test_early_socket_close_cancels_container_stream(tmp_path, streaming_proxy):
    async def run():
        injector, resource, voice = await injected_resource(streaming_proxy, tmp_path)
        try:
            req = request()
            req.dialogue_lines[0].speaker.resolved_path = voice
            registration = await resource.register_backend_request(req)
            iterator = registration.render_stream()
            await asyncio.wait_for(anext(iterator), 5)
            await iterator.aclose()
            await wait_file(tmp_path / "cache" / "stream_closed")
            assert resource._process.returncode is None
        finally:
            process = resource._process
            resource.close()
            if process is not None:
                await process.wait()
            injector.close()

    asyncio.run(run())


def test_voxcpm_stream_gets_model_between_batch_lines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def run():
        first_started = threading.Event()
        finish_first = threading.Event()
        stream_started = asyncio.Event()
        finish_stream = asyncio.Event()
        calls = []

        class Model:
            tts_model = types.SimpleNamespace(sample_rate=48000)

            def generate(self, **kwargs):
                calls.append(kwargs["text"])
                if kwargs["text"] == "first":
                    first_started.set()
                    assert finish_first.wait(5)
                return np.zeros(48, dtype=np.float32)

            def generate_streaming(self, **kwargs):
                calls.append("stream")
                yield np.zeros(48, dtype=np.float32)

        engine = VoxCPM2Engine()
        engine.model = Model()
        line = lambda text: {"type": "line", "spoken_text": text, "speaker": {"voice_path": "voice.wav"}}
        batch = asyncio.create_task(engine.render_batch([{"dialogue_contents": [line("first"), line("second")]}]))
        assert await asyncio.to_thread(first_started.wait, 5)

        async def stream():
            async with aclosing(engine.stream_request({"dialogue_contents": [line("interactive")]})) as chunks:
                await anext(chunks)
                stream_started.set()
                await finish_stream.wait()
                async for _ in chunks:
                    pass

        streaming = asyncio.create_task(stream())
        await asyncio.sleep(0)
        finish_first.set()
        await asyncio.wait_for(stream_started.wait(), 5)
        assert calls == ["first", "stream"]
        assert engine.model_lock.locked()
        finish_stream.set()
        await streaming
        await batch
        assert calls == ["first", "stream", "second"]

    asyncio.run(run())


def test_cancelled_worker_keeps_model_lock_until_thread_finishes():
    async def run():
        lock = asyncio.Lock()
        started = threading.Event()
        finish = threading.Event()

        def worker():
            started.set()
            assert finish.wait(5)

        async def generation():
            async with lock:
                await run_in_thread(worker)

        task = asyncio.create_task(generation())
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert lock.locked()
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not lock.locked()

    asyncio.run(run())


def test_proxy_stream_startup_error_can_retry_same_registration(tmp_path, streaming_proxy):
    async def run():
        injector, resource, voice = await injected_resource(streaming_proxy, tmp_path)
        exchange = resource._exchange
        attempts = 0

        async def fail_once(message):
            nonlocal attempts
            if message["method"] == "render_stream":
                attempts += 1
                if attempts == 1:
                    future = asyncio.get_running_loop().create_future()
                    future.set_exception(RuntimeError("stream startup failed"))
                    return future
            return await exchange(message)

        resource._exchange = fail_once
        try:
            req = request()
            req.dialogue_lines[0].speaker.resolved_path = voice
            registration = await resource.register_request(req)
            with pytest.raises(RuntimeError, match="stream startup failed"):
                await collect(registration)
            (tmp_path / "cache" / "release_stream").touch()
            chunks = await collect(registration)
            assert sum(c.frame_count for c in chunks) == 40000
            assert (await registration.render()).frame_count == 40000
            assert attempts == 2
        finally:
            process = resource._process
            resource.close()
            if process is not None:
                await process.wait()
            injector.close()

    asyncio.run(run())


def test_command_reader_failure_resolves_all_pending_and_rejects_new_commands():
    async def run():
        resource = object.__new__(ProxyTtsResource)
        reader = asyncio.StreamReader()
        resource._responses = {
            1: asyncio.get_running_loop().create_future(),
            2: asyncio.get_running_loop().create_future(),
        }
        futures = list(resource._responses.values())
        resource._reader_error = None
        process = types.SimpleNamespace(stdout=reader, returncode=9)
        reader.feed_eof()
        await resource._read_responses(process)
        for future in futures:
            with pytest.raises(RuntimeError, match="closed its response channel"):
                await future
        assert not resource._responses
        with pytest.raises(RuntimeError, match="closed its response channel"):
            await resource._exchange({"method": "render_batch"})

    asyncio.run(run())


def test_stream_consumers_cannot_mutate_other_callers_replay():
    async def run():
        class Backend:
            async def render_stream(self):
                yield BackendTtsResult(audio=np.array([.1, .2]), sample_rate=8000)

        registration = cached(Backend())
        chunks = await collect(registration)
        chunks[0].audio[:] = 0
        replay = await collect(registration)
        assert np.allclose(replay[0].audio, [.1, .2])
        assert np.allclose((await registration.render()).audio, [.1, .2])

    asyncio.run(run())
