"""Shared alignment contracts; backend payloads do not cross this boundary."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import soundfile as sf
from carthage.dependency_injection import AsyncInjectable, inject

from ..config import ProductionConfig


@dataclass(frozen=True, slots=True)
class WordTiming:
    """A source word in transcript order; seconds relative to input audio."""
    text: str
    start: float | None
    end: float | None


@dataclass(frozen=True, slots=True)
class AlignedClause:
    text: str
    start: float | None
    end: float | None


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Acoustic evidence, separate from authored lines and requested marks.

    None words means unavailable; an empty tuple means attempted with no words.
    Independently measured preferred clauses take precedence over aligned clauses.
    """
    words: tuple[WordTiming, ...] | None
    clauses: tuple[AlignedClause, ...]
    preferred_clauses: tuple[AlignedClause, ...] = ()
    source_text: str = ""
    language: str | None = None
    estimated: bool = False


@dataclass(frozen=True, slots=True)
class ForcedAlignmentRequest:
    audio: np.ndarray
    sample_rate: int
    transcript: str
    transcript_kind: Literal["complete", "partial"] = "partial"
    require_word_alignment: bool = False
    language: str = "en"


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    language: str | None = None


@dataclass(slots=True)
class RegisteredForcedAlignmentRequest:
    resource: ForcedAlignmentResource
    request: ForcedAlignmentRequest
    future: asyncio.Future

    async def align(self) -> AlignmentResult:
        return await self.resource.align_registered_request(self)


@inject(config=ProductionConfig)
class ForcedAlignmentResource(AsyncInjectable):
    """Lazy queued alignment and shared authored boundary projection.

    Subclasses implement model calls and identities. One waiter's cancellation
    does not cancel shared inference. Closing cancels outstanding registrations.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pending: list[RegisteredForcedAlignmentRequest] = []
        self._registrations: dict[int, RegisteredForcedAlignmentRequest] = {}
        self._pending_lock = asyncio.Lock()
        self._drain_task = None
        self._closed = False

    @property
    def alignment_identity(self) -> str:
        raise NotImplementedError

    @property
    def transcription_identity(self) -> str:
        raise NotImplementedError

    async def register_request(self, request: ForcedAlignmentRequest):
        if self._closed:
            raise RuntimeError("Alignment resource is closed")
        registration = RegisteredForcedAlignmentRequest(
            self, request, asyncio.get_running_loop().create_future(),
        )
        async with self._pending_lock:
            self._pending.append(registration)
            self._registrations[id(registration)] = registration
        return registration

    async def align_registered_request(self, registration):
        async with self._pending_lock:
            if not registration.future.done() and (self._drain_task is None or self._drain_task.done()):
                self._drain_task = asyncio.create_task(self._drain_pending())
        return await asyncio.shield(registration.future)

    async def _drain_pending(self):
        while self._pending:
            await asyncio.sleep(0)
            async with self._pending_lock:
                batch = self._pending[:self.config.resolved_batch_size]
                del self._pending[:len(batch)]
            try:
                results = await self._process_batch([entry.request for entry in batch])
                pairs = list(zip(batch, results, strict=True))
            except BaseException as exc:
                for entry in batch:
                    if not entry.future.done():
                        if isinstance(exc, asyncio.CancelledError):
                            entry.future.cancel()
                        else:
                            entry.future.set_exception(exc)
                if isinstance(exc, asyncio.CancelledError):
                    raise
            else:
                for entry, result in pairs:
                    if not entry.future.done():
                        entry.future.set_result(result)
            finally:
                for entry in batch:
                    self._registrations.pop(id(entry), None)

    async def _process_batch(self, requests: Sequence[ForcedAlignmentRequest]):
        raise NotImplementedError

    async def script_timing(self, contents, result, *, sample_rate=None,
                            transcript_kind="partial", require_word_alignment=False,
                            language=None):
        from ..dialogue import DialogueLine, ScriptGap
        from ..rendering import ScriptTiming
        from .projection import script_timing_from_alignment
        lines = [event for event in contents if isinstance(event, DialogueLine)]
        if not lines:
            return ScriptTiming(())
        registration = await self.register_request(ForcedAlignmentRequest(
            audio=result.audio,
            sample_rate=sample_rate or self.config.resolved_output_sample_rate,
            transcript="\n".join(line.spoken_text for line in lines),
            transcript_kind=transcript_kind,
            require_word_alignment=(require_word_alignment or any(line.mark_offsets for line in lines)
                                    or any(isinstance(event, ScriptGap) for event in contents)),
            language=language or self.config.alignment_language,
        ))
        return script_timing_from_alignment(contents, await registration.align())

    async def transcribe(self, audio, sample_rate, *, language="en"):
        return await asyncio.to_thread(self.transcribe_sync, audio, sample_rate, language=language)

    def transcribe_sync(self, audio, sample_rate, *, language="en"):
        raise NotImplementedError

    def transcribe_audio_sample_sync(self, audio, sample_rate=None):
        if isinstance(audio, (str, Path)):
            audio, sample_rate = sf.read(Path(audio).expanduser(), dtype="float32")
        if sample_rate is None:
            raise ValueError("sample_rate is required with array audio")
        return self.transcribe_sync(audio, sample_rate, language=self.config.alignment_language).text

    async def transcribe_audio_sample(self, audio, sample_rate=None):
        return await asyncio.to_thread(self.transcribe_audio_sample_sync, audio, sample_rate)

    def close(self, canceled_futures=True):
        self._closed = True
        if self._drain_task is not None:
            self._drain_task.cancel()
        for registration in self._registrations.values():
            registration.future.cancel()
        self._registrations.clear()
        self._pending.clear()
        return super().close(canceled_futures=canceled_futures)
