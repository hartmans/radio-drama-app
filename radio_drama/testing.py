from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .audio import convert_audio_format
from .planning import ScriptRenderRequest
from .rendering import RenderResult
from .resources import RegisteredRenderRequest, VibeVoiceResource


@dataclass(frozen=True, slots=True)
class CachedRenderMetadata:
    sample_rate: int
    frame_count: int


class MissingCachedRenderMetadata(RuntimeError):
    pass


class CachedVibeVoiceDouble:
    def __init__(
        self,
        cache_directory: str | Path,
        *,
        mode: str = "cache",
        seed: int = 0,
    ) -> None:
        if mode not in {"cache", "live"}:
            raise ValueError("mode must be 'cache' or 'live'")
        self.cache_directory = Path(cache_directory)
        self.mode = mode
        self.seed = seed

    def render(
        self,
        request: ScriptRenderRequest,
        producer: Callable[[ScriptRenderRequest], CachedRenderMetadata] | None = None,
    ) -> RenderResult:
        cache_path = self.cache_directory / f"{self._cache_key(request)}.json"
        if cache_path.is_file():
            metadata = CachedRenderMetadata(**json.loads(cache_path.read_text(encoding="utf-8")))
        elif self.mode == "cache":
            import pytest

            pytest.skip(f"No cached metadata for request {cache_path.stem}")
        else:
            if producer is None:
                raise ValueError("producer is required in live mode when cache is missing")
            metadata = producer(request)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(asdict(metadata), indent=2, sort_keys=True),
                encoding="utf-8",
            )

        rng = np.random.default_rng(self.seed)
        audio = rng.standard_normal(metadata.frame_count, dtype=np.float32) * 1e-3
        return RenderResult(audio=audio)

    def _cache_key(self, request: ScriptRenderRequest) -> str:
        payload = json.dumps(
            {
                "normalized_script": request.normalized_script,
                "voice_samples": list(request.voice_samples),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedVibeVoiceResource(VibeVoiceResource):
    def __init__(
        self,
        cache_directory: str | Path,
        *,
        mode: str = "cache",
        seed: int = 0,
        **kwargs,
    ) -> None:
        if mode not in {"cache", "live"}:
            raise ValueError("mode must be 'cache' or 'live'")
        super().__init__(**kwargs)
        self.cache_directory = Path(cache_directory)
        self.mode = mode
        self.seed = seed

    async def _drain_pending(self) -> None:
        while True:
            await asyncio.sleep(0)
            async with self._pending_lock:
                if not self._pending:
                    self._drain_task = None
                    return
                batch = self._pending[: self.config.resolved_batch_size]
                del self._pending[: self.config.resolved_batch_size]

            try:
                rendered_audios = await asyncio.to_thread(self._render_batch_sync, batch)
            except MissingCachedRenderMetadata as exc:
                import pytest

                skip_exc = pytest.skip.Exception(str(exc))
                for pending in batch:
                    if not pending.registration.future.done():
                        pending.registration.future.set_exception(skip_exc)
                continue
            except Exception as exc:
                for pending in batch:
                    if not pending.registration.future.done():
                        pending.registration.future.set_exception(exc)
                continue

            for pending, audio in zip(batch, rendered_audios, strict=True):
                if not pending.registration.future.done():
                    pending.registration.future.set_result(RenderResult(audio=audio))

    def _render_batch_sync(self, batch: Sequence) -> list[np.ndarray]:
        metadata_by_index: dict[int, CachedRenderMetadata] = {}
        uncached_batch: list[tuple[int, object]] = []

        for index, pending in enumerate(batch):
            request = pending.registration.request
            metadata = self._load_cached_metadata(request)
            if metadata is not None:
                metadata_by_index[index] = metadata
                continue
            if self.mode == "cache":
                raise MissingCachedRenderMetadata(
                    f"No cached metadata for request {self._cache_key(request)}"
                )
            uncached_batch.append((index, pending))

        if uncached_batch:
            native_audios = self._render_batch_native_sync(
                [pending for _, pending in uncached_batch]
            )
            for (index, pending), audio in zip(uncached_batch, native_audios, strict=True):
                metadata = CachedRenderMetadata(
                    sample_rate=self.sample_rate,
                    frame_count=int(audio.shape[0]),
                )
                self._store_cached_metadata(pending.registration.request, metadata)
                metadata_by_index[index] = metadata

        return [
            self._render_synthetic_audio(batch[index].registration, metadata_by_index[index])
            for index in range(len(batch))
        ]

    def _render_synthetic_audio(
        self,
        registration: RegisteredRenderRequest,
        metadata: CachedRenderMetadata,
    ) -> np.ndarray:
        request = registration.request
        seed_material = self._cache_key(request)[:16]
        seed = self.seed ^ int(seed_material, 16)
        rng = np.random.default_rng(seed)
        native_audio = rng.standard_normal(metadata.frame_count, dtype=np.float32) * 1e-3
        return convert_audio_format(
            native_audio,
            input_sample_rate=metadata.sample_rate,
            output_sample_rate=self.config.resolved_output_sample_rate,
            output_channels=self.config.resolved_output_channels,
        )

    def _load_cached_metadata(
        self,
        request: ScriptRenderRequest,
    ) -> CachedRenderMetadata | None:
        cache_path = self.cache_directory / f"{self._cache_key(request)}.json"
        if not cache_path.is_file():
            return None
        return CachedRenderMetadata(**json.loads(cache_path.read_text(encoding="utf-8")))

    def _store_cached_metadata(
        self,
        request: ScriptRenderRequest,
        metadata: CachedRenderMetadata,
    ) -> None:
        cache_path = self.cache_directory / f"{self._cache_key(request)}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(asdict(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _cache_key(self, request: ScriptRenderRequest) -> str:
        payload = json.dumps(
            {
                "normalized_script": request.normalized_script,
                "voice_samples": list(request.voice_samples),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
