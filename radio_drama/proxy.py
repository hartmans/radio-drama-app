from __future__ import annotations

import asyncio
import json
import os
import tomllib
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import soundfile as sf
from carthage.dependency_injection import inject

from .audio import convert_audio_format
from .cache import CacheManager
from .config import ProductionConfig
from .dialogue import DialogueLine, ScriptGap, ScriptRenderRequest, TtsResource
from .rendering import RenderResult, ScriptRenderResult
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

    async def render(self) -> RenderResult:
        return await self.resource.render_registered_request(self)


@inject(
    config=ProductionConfig,
    cache_manager=CacheManager,
    transcription_resource=VoiceReferenceTranscriptionResource,
)
class ProxyTtsResource(TtsResource):
    """Render registered scripts through a persistent Podman JSON-lines service.

    Concrete configured subclasses set ``proxy_config``. Requests remain local
    until rendering starts, allowing the initial container to receive read-only
    mounts for every voice referenced during production planning.
    """

    proxy_config: ProxyTtsConfig

    def __init__(
        self,
        transcription_resource: VoiceReferenceTranscriptionResource,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.transcription_resource = transcription_resource
        self._pending: list[weakref.ReferenceType[RegisteredProxyTtsRequest]] = []
        self._pending_lock = asyncio.Lock()
        self._drain_task: asyncio.Task | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._rpc_lock = asyncio.Lock()
        self._request_id = 0
        self._voice_paths: dict[Path, str] = {}
        self._capabilities: set[str] = set()

    async def register_request(
        self, request: ScriptRenderRequest | None
    ) -> RegisteredProxyTtsRequest:
        loop = asyncio.get_running_loop()
        registration = RegisteredProxyTtsRequest(
            resource=self,
            request=request or ScriptRenderRequest(),
            future=loop.create_future(),
        )
        async with self._pending_lock:
            if request is None:
                registration.future.set_result(
                    ScriptRenderResult.empty(
                        channels=self.config.resolved_output_channels
                    )
                )
            else:
                self._pending.append(weakref.ref(registration))
        return registration

    async def render_registered_request(
        self, registration: RegisteredProxyTtsRequest
    ) -> RenderResult:
        if registration.future.done():
            return await registration.future
        async with self._pending_lock:
            if self._drain_task is None or self._drain_task.done():
                self._drain_task = asyncio.create_task(self._drain_pending())
        return await registration.future

    async def _drain_pending(self) -> None:
        await asyncio.sleep(0)
        async with self._pending_lock:
            batch = [registration for ref in self._pending if (registration := ref())]
            self._pending.clear()
        if not batch:
            return
        try:
            results = await self._render_batch(batch)
        except Exception as exc:
            for registration in batch:
                if not registration.future.done():
                    registration.future.set_exception(exc)
            return
        for registration, result in zip(batch, results, strict=True):
            if not registration.future.done():
                registration.future.set_result(result)

    async def _render_batch(
        self, batch: Sequence[RegisteredProxyTtsRequest]
    ) -> list[ScriptRenderResult]:
        async with self._rpc_lock:
            await self._ensure_process([registration.request for registration in batch])
            if "needs_transcript" in self._capabilities:
                references = {
                    id(line.speaker): line.speaker
                    for registration in batch
                    for line in registration.request.dialogue_lines
                }
                await asyncio.gather(
                    *(
                        self.transcription_resource.transcribe(reference)
                        for reference in references.values()
                    )
                )
            self._request_id += 1
            request_id = self._request_id
            message = {
                "protocol": PROXY_PROTOCOL,
                "version": PROXY_PROTOCOL_VERSION,
                "id": request_id,
                "method": "render_batch",
                "requests": [self._serialize_request(item.request) for item in batch],
            }
            response = await self._exchange(message)
            if response.get("id") != request_id:
                raise RuntimeError("TTS proxy returned a response with the wrong request id")
            if "error" in response:
                raise RuntimeError(f"TTS proxy error: {response['error']}")
            raw_results = response["results"]
            if len(raw_results) != len(batch):
                raise RuntimeError("TTS proxy returned the wrong number of results")
            return [self._load_result(result) for result in raw_results]

    async def _ensure_process(self, requests: Sequence[ScriptRenderRequest]) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        cache_directory = self.cache_manager.root_directory
        if cache_directory is None:
            raise RuntimeError("Proxy TTS requires an enabled production cache directory")
        cache_directory.mkdir(parents=True, exist_ok=True)
        voice_paths = sorted(
            {
                line.speaker.resolved_path.expanduser().resolve()
                for request in requests
                for line in request.dialogue_lines
            },
            key=str,
        )
        self._voice_paths = {
            path: f"/voices/{index}{path.suffix.lower()}"
            for index, path in enumerate(voice_paths)
        }
        args = self._podman_command(cache_directory)
        self._process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            cwd=cache_directory,
        )
        response = await self._exchange(
            {"protocol": PROXY_PROTOCOL, "versions": [PROXY_PROTOCOL_VERSION]}
        )
        if (
            response.get("protocol") != PROXY_PROTOCOL
            or response.get("version") != PROXY_PROTOCOL_VERSION
            or response.get("ready") is not True
        ):
            raise RuntimeError(f"TTS proxy rejected protocol handshake: {response!r}")
        capabilities = response.get("capabilities", [])
        if not isinstance(capabilities, list) or not all(
            isinstance(capability, str) for capability in capabilities
        ):
            raise RuntimeError("TTS proxy capabilities must be a list of strings")
        self._capabilities = set(capabilities)

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
        for path, target in self._voice_paths.items():
            args.extend(("--volume", f"{path}:{target}:ro"))
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

    async def _exchange(self, message: Mapping[str, object]) -> dict[str, object]:
        process = self._process
        assert process is not None and process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(message, ensure_ascii=True).encode("utf-8") + b"\n")
        await process.stdin.drain()
        line = await process.stdout.readline()
        if not line:
            returncode = await process.wait()
            raise RuntimeError(f"TTS proxy exited before responding (status {returncode})")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("TTS proxy response must be a JSON object")
        return response

    def _serialize_request(self, request: ScriptRenderRequest) -> dict[str, object]:
        contents: list[dict[str, object]] = []
        for content in request.dialogue_contents:
            if isinstance(content, DialogueLine):
                voice_path = content.speaker.resolved_path.expanduser().resolve()
                contents.append(
                    {
                        "type": "line",
                        "speaker": {
                            "authored_name": content.speaker.authored_name,
                            "voice_name": content.speaker.voice_name,
                            "voice_path": self._voice_paths[voice_path],
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

    def _load_result(self, result: Mapping[str, object]) -> ScriptRenderResult:
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
        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
        converted = convert_audio_format(
            audio,
            input_sample_rate=sample_rate,
            output_sample_rate=self.config.resolved_output_sample_rate,
            output_channels=self.config.resolved_output_channels,
        )
        starts = result.get("dialogue_line_start_positions")
        return ScriptRenderResult(
            audio=converted,
            dialogue_line_start_positions=(
                tuple(float(value) for value in starts) if starts is not None else None
            ),
        )

    def close(self, canceled_futures: bool = True):
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
