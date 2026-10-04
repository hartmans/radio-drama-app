"""Backend-independent persistent cache and timing for TTS requests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import aclosing
from dataclasses import replace
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, TYPE_CHECKING, Sequence

import numpy as np
import soundfile as sf

from .audio import convert_audio_format
from .stream_audio import StreamingAudioConverter
from .rendering import BackendTtsResult, RenderResult, DialogueLineTiming, ScriptRenderResult, ScriptTiming, DialogueMarkTiming


if TYPE_CHECKING:
    from .dialogue import (
        BackendRegisteredTtsRequest, ScriptEvent, ScriptRenderRequest, TtsResource,
    )


_ALIGNMENT_VERSION = "script-timing-v6"


@dataclass(slots=True)
class _RenderAttempt:
    """One producer and its immutable-in-order chunks, shared by all readers."""

    chunks: list[RenderResult] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[ScriptRenderResult] | None = None
    done: bool = False
    error: BaseException | None = None


@dataclass(slots=True)
class CachedTtsRequest:
    """One lazy backend request mediated by the shared TTS cache."""

    resource: "TtsResource"
    request: "ScriptRenderRequest"
    _batch_attempt: _RenderAttempt | None = None
    _stream_attempt: _RenderAttempt | None = None
    _publish_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _published: bool = False
    _completed_result: ScriptRenderResult | None = None
    _backend_registration: "BackendRegisteredTtsRequest | None" = None
    _backend_result: BackendTtsResult | None = None
    _wav_path: Path | None = None
    _meta_path: Path | None = None
    _alignment_key: str | None = None
    _mark_alignment_key: str | None = None
    _mark_offsets: tuple[tuple[int, ...], ...] = ()

    @classmethod
    async def register(
        cls, resource: "TtsResource", request: "ScriptRenderRequest"
    ) -> "CachedTtsRequest":
        """Resolve cached audio/timing and queue misses before any render starts."""
        registration = cls(resource=resource, request=request)
        if request.dialogue_lines:
            registration._backend_result = await asyncio.to_thread(
                registration._load_cached
            )
            registration._published = registration._backend_result is not None
            if registration._backend_result is None:
                registration._backend_registration = await resource.register_backend_request(
                    request
                )
        return registration

    def audio_available(self) -> bool:
        """Check completed audio without starting generation or waiting for it."""
        if self._published or not self.request.dialogue_lines:
            return True
        # Another registration may have filled this cache entry since registration.
        if self._publish_lock.locked():
            return False
        result = self._load_cached()
        if result is not None:
            self._backend_result = result
            self._published = True
        return self._published

    def _start_attempt(self, *, streaming: bool) -> _RenderAttempt:
        # Without backend streaming, the batch attempt is also the fallback stream.
        streaming = streaming and hasattr(self._backend_registration, "render_stream")
        attribute = "_stream_attempt" if streaming else "_batch_attempt"
        attempt = getattr(self, attribute)
        if attempt is None or (attempt.done and attempt.error is not None):
            attempt = _RenderAttempt()
            setattr(self, attribute, attempt)
            attempt.task = asyncio.create_task(self._produce(attempt, streaming=streaming))
            attempt.task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        return attempt

    async def render(self) -> ScriptRenderResult:
        if self.audio_available():
            return self._converted_result()
        # A stream already underway can satisfy a later batch caller.
        attempt = self._stream_attempt
        if attempt is None or (attempt.done and attempt.error is not None):
            attempt = self._start_attempt(streaming=False)
        return await asyncio.shield(attempt.task)

    async def render_stream(self) -> AsyncIterator[RenderResult]:
        """Replay from the beginning and follow the registration's shared producer.

        A backend without streaming or a cache hit supplies one complete chunk.
        An active batch does not prevent a separate streaming attempt. Consumers
        never throttle socket draining.
        Leaving early keeps the producer running to fill the cache. Failed
        attempts can be retried by a subsequent call; existing readers retain
        their attempt and receive its error rather than a replacement stream.
        """
        if self.audio_available():
            yield RenderResult(audio=self._converted_result().audio.copy())
            return
        attempt = self._start_attempt(streaming=True)
        index = 0
        while True:
            attempt.changed.clear()
            while index < len(attempt.chunks):
                chunk = attempt.chunks[index]
                index += 1
                yield RenderResult(audio=chunk.audio.copy())
            if attempt.done:
                if attempt.error is not None:
                    raise attempt.error
                return
            await attempt.changed.wait()

    async def _produce(self, attempt: _RenderAttempt, *, streaming: bool) -> ScriptRenderResult:
        try:
            if self._published or not self.request.dialogue_lines:
                result = self._converted_result()
            else:
                if streaming:
                    backend_result = await self._produce_stream(attempt)
                else:
                    backend_result = await self._backend_registration.render()
                backend_result = await self._publish(backend_result)
                result = self._converted_result(backend_result)
            if not attempt.chunks:
                attempt.chunks.append(RenderResult(audio=result.audio.copy()))
            return result
        except BaseException as exc:
            attempt.error = exc
            raise
        finally:
            attempt.done = True
            attempt.changed.set()

    async def _produce_stream(self, attempt: _RenderAttempt) -> BackendTtsResult:
        native_chunks = []
        converter = None
        sample_rate = None
        channels = None
        timing = None
        async with aclosing(self._backend_registration.render_stream()) as stream:
            async for chunk in stream:
                if converter is None:
                    sample_rate, channels = chunk.sample_rate, chunk.channel_count
                    converter = StreamingAudioConverter(
                        sample_rate, self.resource.config.resolved_output_sample_rate,
                        self.resource.config.resolved_output_channels,
                    )
                elif chunk.sample_rate != sample_rate or chunk.channel_count != channels:
                    raise RuntimeError("TTS backend changed audio format during streaming")
                # Own the retained samples: engines may reuse buffers after yielding.
                native_chunks.append(chunk.audio.copy())
                timing = chunk.timing if len(native_chunks) == 1 else None
                converted = converter.feed(chunk.audio)
                if len(converted):
                    attempt.chunks.append(RenderResult(audio=converted))
                    attempt.changed.set()
        if converter is not None:
            converted = converter.feed(np.empty((0,) if channels == 1 else (0, channels)), final=True)
            if len(converted):
                attempt.chunks.append(RenderResult(audio=converted))
                attempt.changed.set()
            backend_result = BackendTtsResult(
                audio=np.concatenate(native_chunks), sample_rate=sample_rate, timing=timing,
            )
        else:
            backend_result = BackendTtsResult(
                audio=RenderResult.empty(channels=self.resource.config.resolved_output_channels).audio,
                sample_rate=self.resource.config.resolved_output_sample_rate,
            )
        return backend_result

    async def ensure_timing(
        self,
        contents: Sequence["ScriptEvent"],
        result: ScriptRenderResult,
    ) -> ScriptTiming:
        await self.render()
        assert self._backend_result is not None
        from .dialogue import DialogueLine
        from .forced_alignment import ForcedAlignmentResource
        lines = [event for event in contents if isinstance(event, DialogueLine)]
        offsets = tuple(line.mark_offsets for line in lines)
        timing = self._backend_result.timing
        native = self._alignment_key is not None and self._alignment_key.startswith("native:")
        if native and not any(offsets):
            return ScriptTiming(tuple(replace(span, marks=()) for span in timing.dialogue_lines))
        alignment = await self.resource.ainjector.get_instance_async(ForcedAlignmentResource)
        requested_key = self._forced_alignment_key(contents, alignment.alignment_identity)
        if timing is not None:
            if not native and self._alignment_key == requested_key:
                return timing
            if native and self._mark_offsets == offsets and self._mark_alignment_key == requested_key:
                return timing
        aligned = await alignment.script_timing(
            contents, result,
            sample_rate=self.resource.config.resolved_output_sample_rate,
            transcript_kind="complete",
        )
        if native:
            enriched = []
            for line, existing, measured in zip(lines, timing.dialogue_lines, aligned.dialogue_lines, strict=True):
                marks = tuple(DialogueMarkTiming(
                    existing.end if offset == len(line.spoken_text) else mark.previous_end,
                    existing.start if offset == 0 else mark.next_start,
                ) for offset, mark in zip(line.mark_offsets, measured.marks, strict=True))
                enriched.append(replace(existing, marks=marks))
            timing = ScriptTiming(tuple(enriched))
            self._mark_alignment_key = requested_key
        else:
            timing = aligned
            self._alignment_key = requested_key
        self._backend_result.timing = timing
        self._mark_offsets = offsets
        await asyncio.to_thread(self._write_metadata)
        return timing

    async def _publish(self, result: BackendTtsResult) -> BackendTtsResult:
        """First successful publication wins; losing attempts keep local results."""
        async with self._publish_lock:
            if self._published:
                return result
            self._backend_result = result
            try:
                await asyncio.to_thread(self._store_backend_result)
                persisted = await asyncio.to_thread(self._load_cached)
                if persisted is not None:
                    self._backend_result = persisted
            except BaseException:
                self._backend_result = None
                raise
            self._published = True
            return self._backend_result

    def _converted_result(self, backend_result: BackendTtsResult | None = None) -> ScriptRenderResult:
        if not self.request.dialogue_lines:
            return ScriptRenderResult.empty(channels=self.resource.config.resolved_output_channels)
        backend_result = backend_result if backend_result is not None else self._backend_result
        assert backend_result is not None
        completed = self._published and backend_result is self._backend_result
        if completed and self._completed_result is not None:
            return self._completed_result
        result = ScriptRenderResult(
            audio=convert_audio_format(
                backend_result.audio,
                input_sample_rate=backend_result.sample_rate,
                output_sample_rate=self.resource.config.resolved_output_sample_rate,
                output_channels=self.resource.config.resolved_output_channels,
            ),
            timing=backend_result.timing,
        )
        if completed:
            self._completed_result = result
        return result

    def _cache_paths(self) -> tuple[Path, Path] | None:
        collection = self.resource.cache_manager[self.resource.cache_collection_name]
        if not collection.enabled:
            return None
        key = collection.key_for(self.request)
        return (
            collection.path_for_subtype(key, "wav"),
            collection.path_for_subtype(key, "meta"),
        )

    def _load_cached(self) -> BackendTtsResult | None:
        paths = self._cache_paths()
        if paths is None:
            return None
        wav_path, meta_path = paths
        self._wav_path = wav_path
        self._meta_path = meta_path
        if not wav_path.is_file() or not meta_path.is_file():
            return None
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            sample_rate = int(payload["sample_rate"])
            timing = _timing_from_payload(payload, len(self.request.dialogue_lines))
            audio, actual_rate = sf.read(wav_path, dtype="float32", always_2d=False)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        if int(actual_rate) != sample_rate:
            return None
        self._alignment_key = payload.get("alignment_key") if timing is not None else None
        self._mark_alignment_key = payload.get("mark_alignment_key")
        self._mark_offsets = tuple(tuple(values) for values in payload.get("dialogue_mark_offsets", ()))
        return BackendTtsResult(audio=audio, sample_rate=sample_rate, timing=timing)

    def _store_backend_result(self) -> None:
        assert self._backend_result is not None
        if (
            self._backend_result.timing is not None
            and len(self._backend_result.timing.dialogue_lines)
            != len(self.request.dialogue_lines)
        ):
            raise RuntimeError(
                "TTS backend timing must contain one span per dialogue line"
            )
        paths = self._cache_paths()
        if paths is None:
            return
        wav_path, meta_path = paths
        self._wav_path = wav_path
        self._meta_path = meta_path
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        reusable = self._backend_result.cache_wav_path
        if reusable is None or reusable.resolve() != wav_path.resolve():
            sf.write(wav_path, self._backend_result.audio, self._backend_result.sample_rate)
        if self._backend_result.timing is not None:
            self._alignment_key = f"native:{self._audio_identity()}"
        self._write_metadata()

    def _write_metadata(self) -> None:
        if self._meta_path is None or self._backend_result is None:
            return
        payload = {
            "sample_rate": self._backend_result.sample_rate,
            "frame_count": self._backend_result.frame_count,
            "channels": self._backend_result.channel_count,
            "alignment_key": self._alignment_key,
            "timing_format_version": 2,
            "mark_alignment_key": self._mark_alignment_key,
            "dialogue_mark_offsets": self._mark_offsets,
            "dialogue_mark_timings": (
                [[{"previous_end": mark.previous_end, "next_start": mark.next_start}
                  for mark in span.marks] for span in self._backend_result.timing.dialogue_lines]
                if self._backend_result.timing is not None else None
            ),
            "dialogue_line_spans": (
                [
                    [line.start, line.end]
                    for line in self._backend_result.timing.dialogue_lines
                ]
                if self._backend_result.timing is not None
                else None
            ),
        }
        temporary = self._meta_path.with_suffix(".meta.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._meta_path)

    def _audio_identity(self) -> str:
        if self._wav_path is None or not self._wav_path.is_file():
            assert self._backend_result is not None
            return f"memory:{self._backend_result.frame_count}:{self._backend_result.sample_rate}"
        stat = self._wav_path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def _forced_alignment_key(self, contents: Sequence["ScriptEvent"], backend_identity: str = "") -> str:
        from .dialogue import DialogueLine, ScriptGap

        projection = []
        for content in contents:
            if isinstance(content, DialogueLine):
                projection.append(
                    {
                        "type": "line",
                        "text": content.spoken_text,
                        "source": content.source,
                        "handling": content.handling,
                        "mark_offsets": content.mark_offsets,
                    }
                )
            elif isinstance(content, ScriptGap):
                projection.append(
                    {"type": "gap", "label": content.label, "mode": content.mode}
                )
        encoded = json.dumps({"projection": projection, "backend": backend_identity,
                              "language": self.resource.config.alignment_language,
                              "transcript_kind": "complete"}, sort_keys=True, ensure_ascii=True)
        projection_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return f"{_ALIGNMENT_VERSION}:{self._audio_identity()}:{projection_hash}"


def _timing_from_payload(payload: dict, expected_lines: int) -> ScriptTiming | None:
    spans = payload.get("dialogue_line_spans")
    if spans is None or payload.get("alignment_key") is None:
        return None
    if not isinstance(spans, list) or len(spans) != expected_lines:
        return None
    try:
        lines = tuple(
            DialogueLineTiming(start=float(span[0]), end=float(span[1]))
            for span in spans
            if isinstance(span, list) and len(span) == 2
        )
    except (TypeError, ValueError):
        return None
    if len(lines) != expected_lines:
        return None
    marks = payload.get("dialogue_mark_timings")
    if marks is not None:
        if len(marks) != expected_lines:
            return None
        try:
            lines = tuple(replace(span, marks=tuple(DialogueMarkTiming(
                None if mark["previous_end"] is None else float(mark["previous_end"]),
                None if mark["next_start"] is None else float(mark["next_start"]),
            ) for mark in items)) for span, items in zip(lines, marks, strict=True))
        except (TypeError, ValueError, KeyError):
            return None
    return ScriptTiming(dialogue_lines=lines)


__all__ = ["CachedTtsRequest"]
