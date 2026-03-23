from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .audio import convert_audio_format
from .forced_alignment import WhisperXResource, copy_dialogue_contents
from .planning import DialogueAudio, DialogueContents, DialogueLine, ScriptRenderRequest
from .rendering import RenderResult
from .resources import RegisteredRenderRequest, VibeVoiceResource


@dataclass(frozen=True, slots=True)
class CachedRenderMetadata:
    """Persisted structural metadata for one render request."""
    sample_rate: int
    frame_count: int


@dataclass(frozen=True, slots=True)
class CachedForcedAlignmentMetadata:
    """Persisted structural metadata for one forced-alignment request."""

    start_positions: tuple[float, ...]


class MissingCachedRenderMetadata(RuntimeError):
    pass


class MissingCachedForcedAlignmentMetadata(RuntimeError):
    pass


class CachedVibeVoiceDouble:
    """Small non-injectable test double for unit tests above the resource layer."""

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
    """Cache-aware ``VibeVoiceResource`` substitute for pytest.

    In ``live`` mode, uncached requests call the real model, persist structural
    metadata, and return synthetic production-format audio. In ``cache`` mode,
    missing metadata causes the current test to skip.
    """

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
                batch = self._pop_live_batch_locked()
                if not batch:
                    self._drain_task = None
                    return

            try:
                rendered_audios = await asyncio.to_thread(self._render_batch_sync, batch)
            except MissingCachedRenderMetadata as exc:
                import pytest

                skip_exc = pytest.skip.Exception(str(exc))
                for registration in batch:
                    if not registration.future.done():
                        registration.future.set_exception(skip_exc)
                continue
            except Exception as exc:
                for registration in batch:
                    if not registration.future.done():
                        registration.future.set_exception(exc)
                continue

            for registration, audio in zip(batch, rendered_audios, strict=True):
                if not registration.future.done():
                    registration.future.set_result(RenderResult(audio=audio))

    def _render_batch_sync(self, batch: Sequence) -> list[np.ndarray]:
        metadata_by_index: dict[int, CachedRenderMetadata] = {}
        uncached_batch: list[tuple[int, object]] = []

        for index, registration in enumerate(batch):
            request = registration.request
            metadata = self._load_cached_metadata(request)
            if metadata is not None:
                metadata_by_index[index] = metadata
                continue
            if self.mode == "cache":
                raise MissingCachedRenderMetadata(
                    f"No cached metadata for request {self._cache_key(request)}"
                )
            uncached_batch.append((index, registration))

        if uncached_batch:
            native_audios = self._render_batch_native_sync(
                [registration for _, registration in uncached_batch]
            )
            for (index, registration), audio in zip(uncached_batch, native_audios, strict=True):
                metadata = CachedRenderMetadata(
                    sample_rate=self.sample_rate,
                    frame_count=int(audio.shape[0]),
                )
                self._store_cached_metadata(registration.request, metadata)
                metadata_by_index[index] = metadata

        return [
            self._render_synthetic_audio(batch[index], metadata_by_index[index])
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


class CachedWhisperXResource(WhisperXResource):
    """Cache-aware ``WhisperXResource`` substitute for pytest."""
    _CACHE_FORMAT_VERSION = 2

    def __init__(
        self,
        cache_directory: str | Path,
        *,
        mode: str = "cache",
        **kwargs,
    ) -> None:
        if mode not in {"cache", "live"}:
            raise ValueError("mode must be 'cache' or 'live'")
        super().__init__(**kwargs)
        self.cache_directory = Path(cache_directory)
        self.mode = mode

    async def fill_start_positions(
        self,
        contents: Sequence[DialogueContents],
        result: RenderResult,
    ) -> list[DialogueContents]:
        metadata = self._load_cached_metadata(contents, result)
        if metadata is None:
            if self.mode == "cache":
                import pytest

                pytest.skip(f"No cached forced-alignment metadata for request {self._cache_key(contents, result)}")
            aligned_contents = await self._live_fill_start_positions(contents, result)
            metadata = CachedForcedAlignmentMetadata(
                start_positions=tuple(float(content.start_pos) for content in aligned_contents),
            )
            self._store_cached_metadata(contents, result, metadata)
            return aligned_contents
        return self._apply_cached_metadata(contents, metadata)

    async def _live_fill_start_positions(
        self,
        contents: Sequence[DialogueContents],
        result: RenderResult,
    ) -> list[DialogueContents]:
        return await super().fill_start_positions(contents, result)

    def _apply_cached_metadata(
        self,
        contents: Sequence[DialogueContents],
        metadata: CachedForcedAlignmentMetadata,
    ) -> list[DialogueContents]:
        copied = copy_dialogue_contents(contents)
        if len(copied) != len(metadata.start_positions):
            raise MissingCachedForcedAlignmentMetadata(
                "Cached forced-alignment metadata length does not match contents"
            )
        for content, start_pos in zip(copied, metadata.start_positions, strict=True):
            content.start_pos = float(start_pos)
        return copied

    def _load_cached_metadata(
        self,
        contents: Sequence[DialogueContents],
        result: RenderResult,
    ) -> CachedForcedAlignmentMetadata | None:
        cache_path = self.cache_directory / f"{self._cache_key(contents, result)}.json"
        if not cache_path.is_file():
            return None
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return CachedForcedAlignmentMetadata(
            start_positions=tuple(payload["start_positions"]),
        )

    def _store_cached_metadata(
        self,
        contents: Sequence[DialogueContents],
        result: RenderResult,
        metadata: CachedForcedAlignmentMetadata,
    ) -> None:
        cache_path = self.cache_directory / f"{self._cache_key(contents, result)}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(asdict(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _cache_key(
        self,
        contents: Sequence[DialogueContents],
        result: RenderResult,
    ) -> str:
        payload = json.dumps(
            {
                "contents": [_serialize_dialogue_content(content) for content in contents],
                "audio_sha256": hashlib.sha256(
                    np.ascontiguousarray(result.audio, dtype=np.float32).tobytes()
                ).hexdigest(),
                "frame_count": int(result.frame_count),
                "channel_count": int(result.channel_count),
                "sample_rate": int(self.config.resolved_output_sample_rate),
                "cache_format_version": self._CACHE_FORMAT_VERSION,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialize_dialogue_content(content: DialogueContents) -> dict[str, object]:
    if isinstance(content, DialogueLine):
        return {
            "type": "line",
            "speaker": content.speaker.authored_name,
            "spoken_text": content.spoken_text,
        }
    audio_node = getattr(content.audio_plan, "node", None)
    return {
        "type": "audio",
        "node": getattr(audio_node, "display_name", None),
        "attributes": getattr(audio_node, "attributes", {}),
    }
