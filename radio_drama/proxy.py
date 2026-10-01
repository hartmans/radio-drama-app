from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
import tomllib
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Mapping, Sequence

import numpy as np
import soundfile as sf
from carthage.dependency_injection import inject_autokwargs

from .cache import CacheManager
from .config import ProductionConfig
from .dialogue import DialogueLine, ScriptGap, ScriptRenderRequest, TtsResource
from .effects import load_preprocessed_voice_reference
from .rendering import BackendTtsResult, DialogueLineTiming, ScriptTiming
from .voice_reference import VoiceReferenceTranscriptionResource


PROXY_PROTOCOL = "radio-drama-tts"
PROXY_PROTOCOL_VERSION = 1


@dataclass(frozen=True, slots=True)
class ProxyMount:
    """One user-configured persistent bind mount for a TTS container."""

    source: Path
    target: str
    read_only: bool = True


@dataclass(frozen=True, slots=True)
class ProxyTtsConfig:
    """Podman launch configuration for one named proxy TTS backend."""

    name: str
    image: str
    command: tuple[str, ...] = ()
    mounts: tuple[ProxyMount, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    devices: tuple[str, ...] = ()
    network: str = "none"
    ipc: str | None = None
    shm_size: str | None = None
    podman: str = "podman"


@dataclass(slots=True, weakref_slot=True)
class RegisteredProxyTtsRequest:
    resource: "ProxyTtsResource"
    request: ScriptRenderRequest
    future: asyncio.Future
    started: bool = False

    async def render(self) -> BackendTtsResult:
        return await self.resource.render_registered_request(self)

    def render_stream(self) -> AsyncIterator[BackendTtsResult]:
        return self.resource.stream_registered_request(self)


@inject_autokwargs(
    config=ProductionConfig,
    cache_manager=CacheManager,
    transcription_resource=VoiceReferenceTranscriptionResource,
)
class ProxyTtsResource(TtsResource):
    """Render registered scripts through a persistent Podman JSON-lines service.

    Concrete configured subclasses set ``proxy_config``. Requests remain local
    until rendering starts. Newly encountered references are prepared in the
    mounted cache directory without restarting the resident container.
    """

    proxy_config: ProxyTtsConfig

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pending: list[weakref.ReferenceType[RegisteredProxyTtsRequest]] = []
        self._pending_lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._startup_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._batch_semaphore = asyncio.Semaphore(1)
        self._responses: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._reader_error: BaseException | None = None
        self._tasks: set[asyncio.Task] = set()
        self._stream_writers: set[asyncio.StreamWriter] = set()
        self._stream_directory: Path | None = None
        self._request_id = 0
        self._voice_paths: dict[str, str] = {}
        self._reference_targets: dict[tuple[Path, float], str] = {}
        self._capabilities: set[str] = set()

    @property
    def cache_collection_name(self) -> str:
        return self.proxy_config.name

    async def register_backend_request(
        self, request: ScriptRenderRequest
    ) -> RegisteredProxyTtsRequest:
        loop = asyncio.get_running_loop()
        registration = RegisteredProxyTtsRequest(
            resource=self,
            request=request,
            future=loop.create_future(),
        )
        async with self._pending_lock:
            self._pending.append(weakref.ref(registration))
        return registration

    def _retry_registration(self, registration: RegisteredProxyTtsRequest) -> None:
        if registration.future.done() and registration.future.exception() is not None:
            registration.future = asyncio.get_running_loop().create_future()
            registration.started = False
            self._pending.append(weakref.ref(registration))

    async def render_registered_request(
        self, registration: RegisteredProxyTtsRequest
    ) -> BackendTtsResult:
        async with self._pending_lock:
            self._retry_registration(registration)
            if not registration.started and not registration.future.done():
                task = asyncio.create_task(self._drain_pending())
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        return await asyncio.shield(registration.future)

    async def stream_registered_request(
        self, registration: RegisteredProxyTtsRequest
    ) -> AsyncIterator[BackendTtsResult]:
        """Claim pending work for streaming, or reuse an already started batch.

        The cache wrapper owns replay and the shared producer. Socket bytes are
        little-endian interleaved float32 PCM; socket EOF completes the result
        unless the command process has exited. Closing this iterator closes the
        connection and asks the container to stop at its next generation boundary.
        """
        async with self._pending_lock:
            self._retry_registration(registration)
            claimed = not registration.started and not registration.future.done()
            if claimed:
                registration.started = True
                self._pending = [ref for ref in self._pending if ref() is not registration]
        if not claimed:
            yield await asyncio.shield(registration.future)
            return
        writer = None
        try:
            await self._prepare_submission([registration.request])
            if "streaming" not in self._capabilities:
                result = (await self._render_batch([registration]))[0]
                registration.future.set_result(result)
                yield result
                return
            response_future = await self._exchange({
                "protocol": PROXY_PROTOCOL, "version": PROXY_PROTOCOL_VERSION,
                "method": "render_stream", "request": self._serialize_request(registration.request),
            })
            response = await response_future
            sample_rate = int(response["sample_rate"])
            channels = int(response["channels"])
            if response["encoding"] != "float32le" or sample_rate <= 0 or channels <= 0:
                raise RuntimeError("TTS proxy returned an unsupported streaming audio format")
            socket_name = Path(response["socket"])
            if socket_name.name != str(socket_name) or socket_name.name in {"", ".", ".."}:
                raise RuntimeError("TTS proxy returned an unsafe socket path")
            assert self._stream_directory is not None
            reader, writer = await asyncio.open_unix_connection(self._stream_directory / socket_name)
            self._stream_writers.add(writer)
            chunks = []
            pending = b""
            frame_bytes = channels * 4
            while data := await reader.read(65536):
                pending += data
                size = len(pending) // frame_bytes * frame_bytes
                if not size:
                    continue
                audio = np.frombuffer(pending[:size], dtype="<f4").copy()
                pending = pending[size:]
                if channels > 1:
                    audio = audio.reshape(-1, channels)
                chunks.append(audio)
                yield BackendTtsResult(audio=audio, sample_rate=sample_rate)
            if pending:
                raise RuntimeError("TTS proxy stream ended with an incomplete PCM frame")
            process = self._process
            assert process is not None
            # Allow the subprocess transport to report an exit accompanying EOF.
            await asyncio.sleep(0)
            if process.returncode is not None:
                raise RuntimeError(f"TTS proxy exited during streaming (status {process.returncode})")
            audio = np.concatenate(chunks) if chunks else np.empty(
                (0,) if channels == 1 else (0, channels), dtype=np.float32
            )
            result = BackendTtsResult(audio=audio, sample_rate=sample_rate)
            registration.future.set_result(result)
            if not chunks:
                yield result
        except BaseException as exc:
            if not registration.future.done():
                registration.future.set_exception(exc)
                registration.future.exception()
            raise
        finally:
            if writer is not None:
                self._stream_writers.discard(writer)
                writer.close()
                try:
                    await writer.wait_closed()
                except ConnectionError:
                    pass

    async def _drain_pending(self) -> None:
        await asyncio.sleep(0)
        async with self._pending_lock:
            batch = [registration for ref in self._pending
                     if (registration := ref()) and not registration.started
                     and not registration.future.done()]
            for registration in batch:
                registration.started = True
            self._pending.clear()
        if not batch:
            return
        try:
            results = await self._render_batch(batch)
        except BaseException as exc:
            for registration in batch:
                if not registration.future.done():
                    registration.future.set_exception(exc)
                    registration.future.exception()
            return
        for registration, result in zip(batch, results, strict=True):
            if not registration.future.done():
                registration.future.set_result(result)

    async def _render_batch(
        self, batch: Sequence[RegisteredProxyTtsRequest]
    ) -> list[BackendTtsResult]:
        async with self._batch_semaphore:
            await self._prepare_submission([registration.request for registration in batch])
            message = {
                "protocol": PROXY_PROTOCOL,
                "version": PROXY_PROTOCOL_VERSION,
                "method": "render_batch",
                "requests": [
                    self._serialize_request(registration.request)
                    for registration in batch
                ],
            }
            response_future = await self._exchange(message)
            response = await response_future
            raw_results = response["results"]
            if len(raw_results) != len(batch):
                raise RuntimeError("TTS proxy returned the wrong number of results")
            return [self._load_native_result(result) for result in raw_results]

    async def _prepare_submission(self, requests: Sequence[ScriptRenderRequest]) -> None:
        await self._ensure_process(requests)
        if "needs_transcript" in self._capabilities:
            references = {id(line.speaker): line.speaker for request in requests
                          for line in request.dialogue_lines}
            await asyncio.gather(*(self.transcription_resource.transcribe(reference)
                                   for reference in references.values()))

    async def _ensure_process(self, requests: Sequence[ScriptRenderRequest]) -> None:
        async with self._startup_lock:
            cache_directory = self.cache_manager.root_directory
            if cache_directory is None:
                raise RuntimeError("Proxy TTS requires an enabled production cache directory")
            cache_directory.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self._prepare_voice_references, requests, cache_directory)
            if self._process is not None and self._process.returncode is None:
                if self._reader_task is not None and self._reader_task.done():
                    self._process.kill()
                    await self._process.wait()
                else:
                    return
            await asyncio.to_thread(self._ensure_mount_directories)
            if self._stream_directory is None:
                self._stream_directory = Path(tempfile.mkdtemp(prefix="rdtts-"))
            self._process = await asyncio.create_subprocess_exec(
                *self._podman_command(cache_directory), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=None, cwd=cache_directory,
            )
            try:
                await self._send_message({"protocol": PROXY_PROTOCOL, "versions": [PROXY_PROTOCOL_VERSION]})
                response = await self._read_response(self._process)
                if (response.get("protocol") != PROXY_PROTOCOL
                    or response.get("version") != PROXY_PROTOCOL_VERSION
                    or response.get("ready") is not True):
                    raise RuntimeError(f"TTS proxy rejected protocol handshake: {response!r}")
                capabilities = response.get("capabilities", [])
                if not isinstance(capabilities, list) or not all(isinstance(c, str) for c in capabilities):
                    raise RuntimeError("TTS proxy capabilities must be a list of strings")
                self._capabilities = set(capabilities)
                self._reader_error = None
                self._reader_task = asyncio.create_task(self._read_responses(self._process))
            except BaseException:
                self._process.kill()
                await self._process.wait()
                raise

    def _ensure_mount_directories(self) -> None:
        """Create configured host mount directories before Podman starts."""
        for mount in self.proxy_config.mounts:
            mount.source.expanduser().mkdir(parents=True, exist_ok=True)

    def _podman_command(self, cache_directory: Path) -> list[str]:
        proxy = self.proxy_config
        args = [
            proxy.podman,
            "run",
            "--rm",
            "-i",
            f"--network={proxy.network}",
            "--workdir=/cache",
            "--volume",
            f"{cache_directory.resolve()}:/cache:rw",
        ]
        for device in proxy.devices:
            args.extend(("--device", device))
        if proxy.ipc is not None:
            args.append(f"--ipc={proxy.ipc}")
        # Podman rejects --shm-size with the host IPC namespace.  In that mode
        # the container already uses the host's /dev/shm, so the size setting
        # has no meaning.
        if proxy.shm_size is not None and proxy.ipc != "host":
            args.append(f"--shm-size={proxy.shm_size}")
        if self._stream_directory is not None:
            args.extend(("--volume", f"{self._stream_directory}:/streams:rw"))
        for mount in proxy.mounts:
            mode = "ro" if mount.read_only else "rw"
            args.extend(
                ("--volume", f"{mount.source.expanduser().resolve()}:{mount.target}:{mode}")
            )
        for key, value in proxy.environment.items():
            args.extend(("--env", f"{key}={value}"))
        args.append(proxy.image)
        args.extend(proxy.command)
        return args

    async def _exchange(self, message: Mapping[str, object]) -> asyncio.Future:
        """Send a command and return its independently resolved response future."""
        if self._reader_error is not None:
            raise self._reader_error
        self._request_id += 1
        request_id = self._request_id
        future = asyncio.get_running_loop().create_future()
        self._responses[request_id] = future
        try:
            await self._send_message({**message, "id": request_id})
        except BaseException:
            self._responses.pop(request_id, None)
            future.cancel()
            raise
        return future

    async def _send_message(self, message: Mapping[str, object]) -> None:
        async with self._write_lock:
            process = self._process
            assert process is not None and process.stdin is not None
            process.stdin.write(json.dumps(message, ensure_ascii=True).encode("utf-8") + b"\n")
            await process.stdin.drain()

    async def _read_response(self, process: asyncio.subprocess.Process) -> dict:
        assert process.stdout is not None
        line = await process.stdout.readline()
        if not line:
            raise RuntimeError(f"TTS proxy closed its response channel (status {process.returncode})")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("TTS proxy response must be a JSON object")
        return response

    async def _read_responses(self, process: asyncio.subprocess.Process) -> None:
        try:
            while True:
                response = await self._read_response(process)
                try:
                    future = self._responses.pop(response["id"])
                except KeyError:
                    raise RuntimeError("TTS proxy returned a response with an unknown request id") from None
                if future.done():
                    continue
                if "error" in response:
                    future.set_exception(RuntimeError(f"TTS proxy error: {response['error']}"))
                else:
                    future.set_result(response)
        except BaseException as exc:
            self._reader_error = exc
            for future in self._responses.values():
                if not future.done():
                    future.set_exception(exc)
            self._responses.clear()

    def _serialize_request(self, request: ScriptRenderRequest) -> dict[str, object]:
        contents: list[dict[str, object]] = []
        for content in request.dialogue_contents:
            if isinstance(content, DialogueLine):
                speaker_key = self._normalized_speaker_name(
                    content.speaker.authored_name
                )
                contents.append(
                    {
                        "type": "line",
                        "speaker": {
                            "authored_name": content.speaker.authored_name,
                            "voice_name": content.speaker.voice_name,
                            "voice_path": self._voice_paths[speaker_key],
                            "transcript": content.speaker.transcript,
                            "gain": content.speaker.gain,
                        },
                        "spoken_text": content.spoken_text,
                        "handling": content.handling,
                        "source": content.source,
                    }
                )
            elif isinstance(content, ScriptGap):
                contents.append(
                    {"type": "gap", "label": content.label, "mode": content.mode}
                )
        return {"dialogue_contents": contents, "first_words": request.first_words}

    def _prepare_voice_references(
        self,
        requests: Sequence[ScriptRenderRequest],
        cache_directory: Path,
    ) -> None:
        references = {
            self._normalized_speaker_name(line.speaker.authored_name): line.speaker
            for request in requests
            for line in request.dialogue_lines
        }
        voice_directory = cache_directory / "normalized_voices"
        voice_directory.mkdir(parents=True, exist_ok=True)
        targets = self._reference_targets
        for speaker_name, reference in sorted(references.items()):
            reference_key = (reference.resolved_path.expanduser().resolve(), reference.gain)
            if reference_key in targets:
                self._voice_paths[speaker_name] = targets[reference_key]
                continue
            cache_path = voice_directory / self._voice_cache_filename(speaker_name)
            if not cache_path.exists():
                audio, sample_rate = load_preprocessed_voice_reference(
                    reference.resolved_path,
                    gain_db=reference.gain,
                )
                sf.write(cache_path, audio, sample_rate, subtype="PCM_16")
            target = f"/cache/normalized_voices/{cache_path.name}"
            targets[reference_key] = target
            self._voice_paths[speaker_name] = target

    @staticmethod
    def _normalized_speaker_name(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def _voice_cache_filename(speaker_name: str) -> str:
        label = re.sub(r"[^A-Za-z0-9]+", "_", speaker_name).strip("_")[:40]
        label = label or "speaker"
        digest = hashlib.sha256(speaker_name.encode("utf-8")).hexdigest()[:16]
        return f"{label}_{digest}.wav"

    def _resolve_result_path(self, result: Mapping[str, object]) -> Path:
        relative_wav = Path(str(result["wav"] or ""))
        if relative_wav.is_absolute() or ".." in relative_wav.parts:
            raise RuntimeError("TTS proxy returned an unsafe cache artifact path")
        cache_directory = self.cache_manager.root_directory
        assert cache_directory is not None
        resolved_cache = cache_directory.resolve()
        wav_path = (resolved_cache / relative_wav).resolve()
        try:
            wav_path.relative_to(resolved_cache)
        except ValueError:
            raise RuntimeError("TTS proxy cache artifact resolves outside the cache") from None
        return wav_path

    def _load_native_result(
        self, result: Mapping[str, object]
    ) -> BackendTtsResult:
        wav_path = self._resolve_result_path(result)
        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
        spans = result.get("dialogue_line_spans")
        timing = None
        if spans is not None:
            timing = ScriptTiming(
                tuple(
                    DialogueLineTiming(start=float(span[0]), end=float(span[1]))
                    for span in spans
                )
            )
        return BackendTtsResult(
            audio=audio,
            sample_rate=int(sample_rate),
            timing=timing,
        )

    def close(self, canceled_futures: bool = True):
        for task in self._tasks:
            task.cancel()
        if self._reader_task is not None:
            self._reader_task.cancel()
        for writer in self._stream_writers:
            writer.close()
        for ref in self._pending:
            if (registration := ref()) and not registration.future.done():
                registration.future.cancel()
        if self._stream_directory is not None:
            shutil.rmtree(self._stream_directory)
            self._stream_directory = None
        if self._process is not None and self._process.returncode is None:
            self._process.kill()
        return super().close(canceled_futures=canceled_futures)


def load_proxy_tts_configs(path: Path | None = None) -> dict[str, ProxyTtsConfig]:
    """Load named proxy definitions from the XDG radio-drama configuration."""

    if path is None:
        config_home = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
        path = config_home / "radio-drama" / "tts.toml"
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    configs: dict[str, ProxyTtsConfig] = {}
    for name, value in document.get("tts", {}).items():
        mounts_list: list[ProxyMount] = []
        for item in value.get("mounts", []):
            mode = item.get("mode", "ro")
            if mode not in {"ro", "rw"}:
                raise ValueError(f"TTS proxy {name!r} mount mode must be 'ro' or 'rw'")
            target = item["target"]
            if not isinstance(target, str) or not target.startswith("/"):
                raise ValueError(f"TTS proxy {name!r} mount target must be absolute")
            mounts_list.append(
                ProxyMount(
                    source=Path(item["source"]).expanduser(),
                    target=target,
                    read_only=mode == "ro",
                )
            )
        configs[name.lower()] = ProxyTtsConfig(
            name=name.lower(),
            image=value["image"],
            command=tuple(value.get("command", ())),
            mounts=tuple(mounts_list),
            environment=dict(value.get("environment", {})),
            devices=tuple(value.get("devices", ())),
            network=value.get("network", "none"),
            ipc=value.get("ipc"),
            shm_size=value.get("shm_size"),
            podman=value.get("podman", "podman"),
        )
    return configs


def configured_proxy_resource(config: ProxyTtsConfig) -> type[ProxyTtsResource]:
    """Create an injectable resource class bound to one proxy definition."""

    return type(
        f"{config.name.title().replace('-', '')}ProxyTtsResource",
        (ProxyTtsResource,),
        {"proxy_config": config, "__module__": __name__},
    )


__all__ = [
    "PROXY_PROTOCOL",
    "PROXY_PROTOCOL_VERSION",
    "ProxyMount",
    "ProxyTtsConfig",
    "ProxyTtsResource",
    "configured_proxy_resource",
    "load_proxy_tts_configs",
]
