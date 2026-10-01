from __future__ import annotations

import asyncio
import hashlib
import json
import re
import struct
import sys
import traceback
import uuid
import wave
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import aclosing, redirect_stdout
from pathlib import Path
from typing import Any, TextIO


PROTOCOL = "radio-drama-tts"
PROTOCOL_VERSION = 1
RenderBatch = Callable[
    [Sequence[Mapping[str, Any]]],
    Awaitable[Sequence[Mapping[str, Any]]],
]
StreamRequest = Callable[[Mapping[str, Any]], AsyncIterator[bytes]]


async def run_in_thread(function, /, *args, **kwargs):
    """Run blocking engine work, waiting for it to stop before cancellation exits.

    Cancelling an asyncio waiter does not stop its worker thread. Waiting here
    keeps a surrounding model lock held until the worker has finished using
    the model. Engines must set thread-local inference modes inside the call.
    """
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # A second cancellation must not release model access prematurely.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise


def artifact_name(request: Mapping[str, Any], suffix: str = ".wav") -> str:
    """Return a deterministic, cache-relative artifact name for a request."""

    label = re.sub(r"[^A-Za-z0-9]+", "_", str(request.get("first_words", "audio")))
    label = label.strip("_").lower()[:40] or "audio"
    encoded = json.dumps(request, sort_keys=True, ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"proxy_{label}_{digest}{suffix}"


def write_pcm16_wav(
    path: str | Path,
    samples: Iterable[float],
    *,
    sample_rate: int,
    channels: int = 1,
) -> None:
    """Write normalized interleaved floating-point samples using only stdlib."""

    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        frames = bytearray()
        for sample in samples:
            bounded = max(-1.0, min(1.0, float(sample)))
            frames.extend(struct.pack("<h", round(bounded * 32767)))
        output.writeframes(frames)


def run_server(
    render_batch: RenderBatch,
    *,
    capabilities: Iterable[str] = (),
    stream_request: StreamRequest | None = None,
    stream_sample_rate: int | None = None,
    stream_channels: int = 1,
    socket_directory: Path = Path("/streams"),
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
    """Serve version-one batch commands and optional Unix-socket streaming.

    Async batch callbacks offload their blocking inference. Batches are
    serialized, while streaming commands can be accepted during a batch. Engines supporting both paths
    coordinate model access themselves.

    ``stream_request`` yields interleaved little-endian float32 PCM bytes at
    ``stream_sample_rate`` with ``stream_channels`` channels. The command
    response advertises that format and a socket basename in the mounted
    socket directory once the listener is ready. Only one client connects.
    Client EOF cancels generation; server EOF completes audio, including
    partial audio after an engine error (whose traceback is logged).

    Protocol stdout is saved before redirecting all incidental library output
    to stderr for the lifetime of the service, including worker-thread output.
    """
    output_stream = output_stream or sys.stdout
    input_stream = input_stream or sys.stdin
    with redirect_stdout(sys.stderr):
        asyncio.run(_serve(
            render_batch, set(capabilities), stream_request, stream_sample_rate,
            stream_channels, Path(socket_directory), input_stream, output_stream,
        ))


async def _serve(render_batch, capabilities, stream_request, sample_rate,
                 channels, socket_directory, input_stream, output_stream):
    if stream_request is not None:
        if sample_rate is None or sample_rate <= 0 or channels <= 0:
            raise ValueError("Streaming requires a positive sample rate and channel count")
        capabilities.add("streaming")
        socket_directory.mkdir(parents=True, exist_ok=True)
    handshake_line = await asyncio.to_thread(input_stream.readline)
    if not handshake_line:
        return
    handshake = json.loads(handshake_line)
    if handshake.get("protocol") != PROTOCOL or PROTOCOL_VERSION not in handshake.get("versions", ()):
        raise RuntimeError("Host does not support this TTS proxy protocol")
    _write_json(output_stream, {
        "protocol": PROTOCOL, "version": PROTOCOL_VERSION,
        "ready": True, "capabilities": sorted(capabilities),
    })
    batch_lock = asyncio.Lock()
    tasks = set()

    async def dispatch(message):
        response = {"id": message.get("id")}
        try:
            if message.get("protocol") != PROTOCOL:
                raise ValueError("Incorrect protocol name")
            if message.get("version") != PROTOCOL_VERSION:
                raise ValueError("Unsupported protocol version")
            method = message["method"]
            if method == "render_batch":
                requests = message["requests"]
                async with batch_lock:
                    results = list(await render_batch(requests))
                if len(results) != len(requests):
                    raise ValueError("Engine returned the wrong number of results")
                response["results"] = results
                _write_json(output_stream, response)
            elif method == "render_stream" and stream_request is not None:
                await _stream_socket(message, stream_request, sample_rate, channels,
                                     socket_directory, output_stream)
            else:
                raise ValueError("Unsupported proxy method")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response["error"] = {"type": type(exc).__name__, "message": str(exc)}
            _write_json(output_stream, response)

    try:
        while line := await asyncio.to_thread(input_stream.readline):
            if not line.strip():
                continue
            task = asyncio.create_task(dispatch(json.loads(line)))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def _stream_socket(message, stream_request, sample_rate, channels,
                         socket_directory, output_stream):
    path = socket_directory / f"{uuid.uuid4().hex}.sock"
    connection = asyncio.get_running_loop().create_future()

    def connected(reader, writer):
        if connection.done():
            writer.close()
        else:
            connection.set_result((reader, writer))

    server = await asyncio.start_unix_server(connected, path=path)
    writer = None
    producer = None
    disconnected = None
    try:
        _write_json(output_stream, {
            "id": message["id"], "sample_rate": sample_rate, "channels": channels,
            "encoding": "float32le", "socket": path.name,
        })
        try:
            reader, writer = await asyncio.wait_for(connection, timeout=30)
            server.close()

            async def produce():
                async with aclosing(stream_request(message["request"])) as chunks:
                    async for chunk in chunks:
                        writer.write(chunk)
                        await writer.drain()

            producer = asyncio.create_task(produce())
            disconnected = asyncio.create_task(reader.read(1))
            done, _ = await asyncio.wait((producer, disconnected), return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done:
                producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)
            else:
                await producer
        except Exception:
            # Socket readiness already resolved the command. Subsequent engine
            # errors are logged and complete whatever audio has been delivered.
            traceback.print_exc(file=sys.stderr)
    finally:
        for task in (producer, disconnected):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (producer, disconnected) if t is not None),
                             return_exceptions=True)
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
        server.close()
        await server.wait_closed()
        path.unlink(missing_ok=True)


def _write_json(stream: TextIO, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n")
    stream.flush()
