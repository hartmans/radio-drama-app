from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .dialogue import (
    DialogueAudio,
    DialogueContent,
    DialogueLine,
    ScriptEvent,
    ScriptGap,
    ScriptRenderRequest,
)
from .forced_alignment import ForcedAlignmentResource, copy_dialogue_contents, AlignmentResult, WordTiming, AlignedClause
from .qwen_tts import QwenTtsResource
from .rendering import (
    BackendTtsResult,
    DialogueLineTiming,
    RenderResult,
    ScriptRenderResult,
    ScriptTiming,
)
from .vibevoice import RegisteredRenderRequest, VibeVoiceResource


@dataclass(frozen=True, slots=True)
class CachedRenderMetadata:
    """Persisted structural metadata for one render request."""
    sample_rate: int
    frame_count: int
    dialogue_line_spans: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.dialogue_line_spans is None:
            return
        object.__setattr__(
            self,
            "dialogue_line_spans",
            tuple((float(start), float(end)) for start, end in self.dialogue_line_spans),
        )


class MissingCachedRenderMetadata(RuntimeError):
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
        if metadata.dialogue_line_spans is None:
            return RenderResult(audio=audio)
        return ScriptRenderResult(
            audio=audio,
            timing=ScriptTiming(
                tuple(DialogueLineTiming(start, end) for start, end in metadata.dialogue_line_spans)
            ),
        )

    def _cache_key(self, request: ScriptRenderRequest) -> str:
        return request.cache_hash()


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
                rendered_results = await asyncio.to_thread(self._render_batch_sync, batch)
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

            for registration, result in zip(batch, rendered_results, strict=True):
                if not registration.future.done():
                    registration.future.set_result(result)

    def _render_batch_sync(self, batch: Sequence) -> list[BackendTtsResult]:
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
            native_results = self._render_batch_native_sync(
                [registration for _, registration in uncached_batch]
            )
            for (index, registration), generated in zip(uncached_batch, native_results, strict=True):
                if isinstance(generated, BackendTtsResult):
                    native_audio = generated.audio
                    sample_rate = generated.sample_rate
                    dialogue_line_spans = (
                        tuple((span.start, span.end) for span in generated.timing.dialogue_lines)
                        if generated.timing is not None
                        else None
                    )
                else:
                    native_audio = generated
                    sample_rate = self.sample_rate
                    dialogue_line_spans = None
                metadata = CachedRenderMetadata(
                    sample_rate=sample_rate,
                    frame_count=int(native_audio.shape[0]),
                    dialogue_line_spans=dialogue_line_spans,
                )
                self._store_cached_metadata(registration.request, metadata)
                metadata_by_index[index] = metadata

        return [
            self._render_synthetic_result(batch[index], metadata_by_index[index])
            for index in range(len(batch))
        ]

    def _render_synthetic_result(
        self,
        registration: RegisteredRenderRequest,
        metadata: CachedRenderMetadata,
    ) -> BackendTtsResult:
        request = registration.request
        seed_material = self._cache_key(request)[:16]
        seed = self.seed ^ int(seed_material, 16)
        rng = np.random.default_rng(seed)
        native_audio = rng.standard_normal(metadata.frame_count, dtype=np.float32) * 1e-3
        timing = None
        if metadata.dialogue_line_spans is not None:
            timing = ScriptTiming(
                tuple(DialogueLineTiming(start, end) for start, end in metadata.dialogue_line_spans)
            )
        return BackendTtsResult(
            audio=native_audio,
            sample_rate=metadata.sample_rate,
            timing=timing,
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
        return request.cache_hash()


class CachedQwenTtsResource(QwenTtsResource):
    """Cache-aware ``QwenTtsResource`` substitute for pytest."""

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
                rendered_results = await asyncio.to_thread(self._render_batch_sync, batch)
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

            for registration, result in zip(batch, rendered_results, strict=True):
                if not registration.future.done():
                    registration.future.set_result(result)

    def _render_batch_sync(self, batch: Sequence) -> list[BackendTtsResult]:
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
            native_results = self._render_batch_native_sync(
                [registration for _, registration in uncached_batch]
            )
            for (index, registration), generated in zip(uncached_batch, native_results, strict=True):
                if isinstance(generated, BackendTtsResult):
                    native_audio = generated.audio
                    sample_rate = generated.sample_rate
                    dialogue_line_spans = (
                        tuple((span.start, span.end) for span in generated.timing.dialogue_lines)
                        if generated.timing is not None
                        else None
                    )
                else:
                    native_audio = generated
                    sample_rate = self.sample_rate
                    dialogue_line_spans = None
                metadata = CachedRenderMetadata(
                    sample_rate=sample_rate,
                    frame_count=int(native_audio.shape[0]),
                    dialogue_line_spans=dialogue_line_spans,
                )
                self._store_cached_metadata(registration.request, metadata)
                metadata_by_index[index] = metadata

        return [
            self._render_synthetic_result(batch[index], metadata_by_index[index])
            for index in range(len(batch))
        ]

    def _render_synthetic_result(
        self,
        registration: RegisteredRenderRequest,
        metadata: CachedRenderMetadata,
    ) -> BackendTtsResult:
        request = registration.request
        seed_material = self._cache_key(request)[:16]
        seed = self.seed ^ int(seed_material, 16)
        rng = np.random.default_rng(seed)
        native_audio = rng.standard_normal(metadata.frame_count, dtype=np.float32) * 1e-3
        timing = None
        if metadata.dialogue_line_spans is not None:
            timing = ScriptTiming(
                tuple(DialogueLineTiming(start, end) for start, end in metadata.dialogue_line_spans)
            )
        return BackendTtsResult(
            audio=native_audio,
            sample_rate=metadata.sample_rate,
            timing=timing,
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
        return request.cache_hash()


class CachedForcedAlignmentResource(ForcedAlignmentResource):
    """Replay neutral evidence so real line/mark projection runs in either mode."""
    _CACHE_FORMAT_VERSION = 3

    def __init__(self, cache_directory, *, mode="cache", **kwargs):
        if mode not in {"cache", "live"}:
            raise ValueError("mode must be 'cache' or 'live'")
        super().__init__(**kwargs)
        self.cache_directory = Path(cache_directory)
        self.mode = mode
        self._live_backend = None

    @property
    def alignment_identity(self):
        return f"replay:{self.config.alignment_backend}:v3"

    @property
    def transcription_identity(self):
        return self.alignment_identity

    async def _live_align(self, request):
        if self._live_backend is None:
            if self.config.alignment_backend == "qwen":
                from .forced_alignment.qwen import QwenAlignmentResource
                backend = QwenAlignmentResource
            else:
                from .forced_alignment.whisperx import WhisperXResource
                backend = WhisperXResource
            self._live_backend = await self.ainjector(backend)
        return await (await self._live_backend.register_request(request)).align()

    async def _process_batch(self, requests):
        results = []
        for request in requests:
            digest = hashlib.sha256(np.ascontiguousarray(request.audio, dtype=np.float32).tobytes()).hexdigest()
            key = json.dumps({"audio": digest, "sample_rate": request.sample_rate,
                              "transcript": request.transcript, "kind": request.transcript_kind,
                              "words": request.require_word_alignment, "language": request.language,
                              "identity": self.alignment_identity}, sort_keys=True)
            path = self.cache_directory / (hashlib.sha256(key.encode()).hexdigest() + ".json")
            if path.is_file():
                payload = json.loads(path.read_text())
                result = AlignmentResult(
                    None if payload["words"] is None else tuple(WordTiming(**w) for w in payload["words"]),
                    tuple(AlignedClause(**c) for c in payload["clauses"]),
                    tuple(AlignedClause(**c) for c in payload["preferred_clauses"]),
                    payload["source_text"], payload["language"], payload["estimated"],
                )
            else:
                if self.mode == "cache":
                    import pytest
                    pytest.skip(f"No neutral forced-alignment replay for {path.name}")
                result = await self._live_align(request)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(asdict(result), indent=2) + "\n")
            results.append(result)
        return results


def _serialize_dialogue_content(content: ScriptEvent) -> dict[str, object]:
    if isinstance(content, DialogueLine):
        return {
            "type": "line",
            "speaker": content.speaker.authored_name,
            "spoken_text": content.spoken_text,
            "handling": content.handling,
            "node": getattr(content.node, "display_name", None),
            "attributes": getattr(content.node, "attributes", {}),
        }
    if isinstance(content, ScriptGap):
        return {
            "type": "gap",
            "label": content.label,
            "mode": content.mode,
        }
    audio_node = getattr(content.audio_plan, "node", None)
    return {
        "type": "audio",
        "node": getattr(audio_node, "display_name", None),
        "attributes": getattr(audio_node, "attributes", {}),
    }


def _serialize_render_request(request: ScriptRenderRequest) -> dict[str, object]:
    return request.serialize_cache_request()
