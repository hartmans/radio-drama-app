from __future__ import annotations

from .projection import _clauses_from_segments, _audio_duration

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Sequence

import numpy as np
import soundfile as sf
from carthage.dependency_injection import inject

from ..audio import resample_audio
from ..config import ProductionConfig
from ..debug import write_debug_json, write_debug_message
from ..model_loading import shared_model_load


_TOKEN_RE = re.compile(r"[A-Za-z']+|[0-9]|(?<=[0-9])\.(?=[0-9])")
_NUMBER_TOKENS = dict(zip(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "point"),
    (*"0123456789", "."),
))
_WHISPERX_LANGUAGE = "en"
_WHISPERX_MODEL = "large-v3"
_WHISPERX_SAMPLE_RATE = 16000
_WHISPERX_TRANSCRIBE_BATCH_SIZE = 10
_WHISPERX_ALIGNMENT_THREADS = 4
logger = logging.getLogger(__name__)


from .base import (ForcedAlignmentResource, ForcedAlignmentRequest, AlignmentResult,
    AlignedClause, WordTiming, TranscriptionResult)
from .projection import (_line_spans_from_exact_clauses, _transcript_lines,
    _normalized_tokens, _optional_float, _debug_transcript_label, _sanitize_debug_label)

@dataclass(frozen=True, slots=True)
class WhisperXResponse:
    transcription_segments: tuple[dict[str, Any], ...]
    aligned_segments: tuple[dict[str, Any], ...] | None
    decision: str


@dataclass(frozen=True, slots=True)
class _PreparedForcedAlignment:
    request: ForcedAlignmentRequest
    mono_audio: np.ndarray
    transcription_segments: tuple[dict[str, Any], ...] | None


@inject(config=ProductionConfig)
class WhisperXResource(ForcedAlignmentResource):
    """Forced-alignment resource that prefers WhisperX and falls back heuristically."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._debug_output_index = 0
        self._debug_output_lock = Lock()
        self._load_lock = RLock()
        self._alignment_executor = ThreadPoolExecutor(
            max_workers=_WHISPERX_ALIGNMENT_THREADS,
            thread_name_prefix="whisperx-align",
        )
        self._whisperx_module = None
        self._asr_model = None
        self._align_model = None
        self._align_metadata: dict[str, Any] | None = None

    @property
    def alignment_identity(self):
        return "whisperx:large-v3:en:policy-v1"

    @property
    def transcription_identity(self):
        return "whisperx:large-v3:en"

    def transcribe_sync(self, audio, sample_rate, *, language="en"):
        if language != "en":
            raise ValueError("WhisperX currently supports English requests only")
        mono = _whisperx_mono_audio(audio, sample_rate)
        model = self._ensure_asr_model()
        transcription = model.transcribe(mono, batch_size=_WHISPERX_TRANSCRIBE_BATCH_SIZE, language=language)
        return TranscriptionResult(_transcription_text_from_segments(tuple(transcription["segments"])), language)

    async def _process_batch(self, requests):
        prepared = await asyncio.to_thread(self._prepare_batch_sync, requests)
        responses = await asyncio.gather(*(self._resolve_prepared_alignment(p) for p in prepared))
        return [
            _alignment_result_from_whisperx_response(
                request.transcript, response,
                duration_seconds=_audio_duration(request.audio, request.sample_rate),
            )
            for request, response in zip(requests, responses, strict=True)
        ]

    def _alignment_result_sync(
        self,
        audio: np.ndarray,
        sample_rate: int,
        transcript: str,
    ) -> AlignmentResult:
        request = ForcedAlignmentRequest(
            audio=audio,
            sample_rate=sample_rate,
            transcript=transcript,
        )
        prepared = self._prepare_request_sync(request)
        whisperx_response = self._resolve_prepared_alignment_sync(prepared)
        return _alignment_result_from_whisperx_response(
            transcript,
            whisperx_response,
            duration_seconds=_audio_duration(audio, sample_rate),
        )

    def _prepare_batch_sync(
        self,
        batch: Sequence[ForcedAlignmentRequest],
    ) -> list[_PreparedForcedAlignment]:
        return [
            self._prepare_request_sync(request)
            for request in batch
        ]

    def _prepare_request_sync(
        self,
        request: ForcedAlignmentRequest,
    ) -> _PreparedForcedAlignment:
        if request.language != "en":
            raise ValueError("WhisperX currently supports English requests only")
        mono_audio = _whisperx_mono_audio(request.audio, request.sample_rate)
        try:
            model = self._ensure_asr_model()
        except ImportError:
            return _PreparedForcedAlignment(
                request=request,
                mono_audio=mono_audio,
                transcription_segments=None,
            )

        transcription = model.transcribe(
            mono_audio,
            batch_size=_WHISPERX_TRANSCRIBE_BATCH_SIZE,
            language=_WHISPERX_LANGUAGE,
        )
        return _PreparedForcedAlignment(
            request=request,
            mono_audio=mono_audio,
            transcription_segments=tuple(transcription["segments"]),
        )

    async def _resolve_prepared_alignment(
        self,
        prepared: _PreparedForcedAlignment,
    ) -> WhisperXResponse | None:
        return await asyncio.get_running_loop().run_in_executor(
            self._alignment_executor,
            self._resolve_prepared_alignment_sync,
            prepared,
        )

    def _resolve_prepared_alignment_sync(
        self,
        prepared: _PreparedForcedAlignment,
    ) -> WhisperXResponse | None:
        if prepared.transcription_segments is None:
            return None
        transcript_lines = _transcript_lines(prepared.request.transcript)
        transcription_clauses = _clauses_from_segments(prepared.transcription_segments)
        if (
            not prepared.request.require_word_alignment
            and _line_spans_from_exact_clauses(transcript_lines, transcription_clauses) is not None
        ):
            response = WhisperXResponse(
                transcription_segments=prepared.transcription_segments,
                aligned_segments=None,
                decision="transcription_exact_clause_match",
            )
            self._write_whisperx_debug_output(prepared.request.transcript, response)
            return response

        align_model, metadata = self._ensure_align_model()
        whisperx = self._ensure_whisperx_module()
        device = self.config.resolved_device
        aligned = whisperx.align(
            list(prepared.transcription_segments),
            align_model,
            metadata,
            prepared.mono_audio,
            device,
            return_char_alignments=False,
        )
        aligned_segments = tuple(aligned.get("segments", []))
        aligned_clauses = _clauses_from_segments(aligned_segments)
        if (
            not prepared.request.require_word_alignment
            and _line_spans_from_exact_clauses(transcript_lines, aligned_clauses) is not None
        ):
            response = WhisperXResponse(
                transcription_segments=prepared.transcription_segments,
                aligned_segments=aligned_segments,
                decision="aligned_exact_clause_match",
            )
            self._write_whisperx_debug_output(prepared.request.transcript, response)
            return response

        response = WhisperXResponse(
            transcription_segments=prepared.transcription_segments,
            aligned_segments=aligned_segments,
            decision="aligned_word_matching",
        )
        self._write_whisperx_debug_output(prepared.request.transcript, response)
        return response

    def _write_whisperx_debug_output(
        self,
        transcript: str,
        response: WhisperXResponse,
    ) -> None:
        if not self.config.debug_enabled("whisperx"):
            return
        output_index = self._reserve_debug_output_index()
        filename = (
            f"{output_index:03d}-"
            f"{_sanitize_debug_label(_debug_transcript_label(transcript))}.json"
        )
        artifact_path = write_debug_json(
            self.config,
            "whisperx",
            filename,
            {
                "decision": response.decision,
                "transcript": transcript,
                "transcription_segments": list(response.transcription_segments),
                "aligned_segments": (
                    list(response.aligned_segments)
                    if response.aligned_segments is not None
                    else None
                ),
            },
        )
        if artifact_path is not None:
            write_debug_message(
                self.config,
                "whisperx",
                f"{artifact_path.name} decision={response.decision}",
            )

    def _reserve_debug_output_index(self) -> int:
        with self._debug_output_lock:
            output_index = self._debug_output_index
            self._debug_output_index += 1
        return output_index

    def _ensure_whisperx_module(self):
        with self._load_lock:
            if self._whisperx_module is None:
                import whisperx  # type: ignore[import-not-found]

                self._whisperx_module = whisperx
            return self._whisperx_module

    def _ensure_asr_model(self):
        with self._load_lock:
            if self._asr_model is not None:
                return self._asr_model
            with shared_model_load():
                if self._asr_model is not None:
                    return self._asr_model
                whisperx = self._ensure_whisperx_module()
                self._asr_model = whisperx.load_model(
                    _WHISPERX_MODEL,
                    self.config.resolved_device,
                    compute_type="default",
                    language=_WHISPERX_LANGUAGE,
                )
                return self._asr_model

    def _ensure_align_model(self):
        with self._load_lock:
            if self._align_model is not None and self._align_metadata is not None:
                return self._align_model, self._align_metadata
            with shared_model_load():
                if self._align_model is not None and self._align_metadata is not None:
                    return self._align_model, self._align_metadata
                whisperx = self._ensure_whisperx_module()
                self._align_model, self._align_metadata = whisperx.load_align_model(
                    language_code=_WHISPERX_LANGUAGE,
                    device=self.config.resolved_device,
                )
                return self._align_model, self._align_metadata

    def close(self, canceled_futures: bool = True):
        self._alignment_executor.shutdown(wait=True, cancel_futures=canceled_futures)
        return super().close(canceled_futures=canceled_futures)


def _alignment_result_from_whisperx(
    payload: dict,
    *,
    clauses: Sequence[AlignedClause] | None = None,
) -> AlignmentResult:
    segments = payload.get("segments", [])
    words: list[WordTiming] = []
    for segment in segments:
        for word in segment.get("words", []) or []:
            words.append(
                WordTiming(
                    text=str(word.get("word", "")),
                    start=_optional_float(word.get("start")),
                    end=_optional_float(word.get("end")),
                )
            )
    if clauses is None:
        clauses = _clauses_from_segments(segments)
    return AlignmentResult(words=tuple(words), clauses=tuple(clauses))


def _alignment_result_from_whisperx_response(transcript, response, *, duration_seconds):
    result = _alignment_evidence_from_whisperx_response(
        transcript, response, duration_seconds=duration_seconds)
    source_text = (transcript if response is None else
                   _transcription_text_from_segments(response.transcription_segments))
    return replace(result, source_text=source_text, language="en")


def _alignment_evidence_from_whisperx_response(
    transcript: str,
    response: WhisperXResponse | None,
    *,
    duration_seconds: float,
) -> AlignmentResult:
    if response is None:
        return _fallback_alignment_result(transcript, duration_seconds=duration_seconds)
    if response.decision == "transcription_exact_clause_match":
        clauses = _clauses_from_segments(response.transcription_segments)
        return AlignmentResult(words=None, clauses=tuple(clauses))
    if response.decision == "aligned_exact_clause_match" and response.aligned_segments is not None:
        clauses = _clauses_from_segments(response.aligned_segments)
        return AlignmentResult(
            words=None, clauses=tuple(clauses),
            preferred_clauses=tuple(_clauses_from_segments(response.transcription_segments)),
        )
    if response.aligned_segments is None:
        clauses = _clauses_from_segments(response.transcription_segments)
        return AlignmentResult(words=None, clauses=tuple(clauses))
    aligned = _alignment_result_from_whisperx(
        {"segments": list(response.aligned_segments)},
        clauses=_clauses_from_segments(response.aligned_segments),
    )
    return AlignmentResult(
        words=aligned.words,
        clauses=aligned.clauses,
        preferred_clauses=tuple(_clauses_from_segments(response.transcription_segments)),
    )


def _fallback_alignment_result(transcript: str, *, duration_seconds: float) -> AlignmentResult:
    clauses: list[AlignedClause] = []
    words: list[WordTiming] = []
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    line_tokens = [list(_normalized_tokens(line)) or [""] for line in lines]
    total_tokens = max(sum(len(tokens) for tokens in line_tokens), 1)
    cursor = 0.0

    for line, tokens in zip(lines, line_tokens, strict=True):
        line_duration = duration_seconds * (len(tokens) / total_tokens)
        line_start = cursor
        line_end = cursor + line_duration
        clauses.append(AlignedClause(text=line, start=line_start, end=line_end))
        token_duration = 0.0 if not tokens else line_duration / len(tokens)
        for token_index, token in enumerate(tokens):
            word_start = line_start + token_index * token_duration
            word_end = line_start + (token_index + 1) * token_duration
            words.append(WordTiming(text=token, start=word_start, end=word_end))
        cursor = line_end

    return AlignmentResult(words=None, clauses=tuple(clauses), source_text=transcript, estimated=True)


def _transcription_sample_audio(
    audio: str | Path | np.ndarray,
    sample_rate: int | None,
) -> np.ndarray:
    if isinstance(audio, (str, Path)):
        loaded_audio, loaded_sample_rate = sf.read(
            str(Path(audio).expanduser()),
            dtype="float32",
            always_2d=False,
        )
        return _whisperx_mono_audio(np.asarray(loaded_audio, dtype=np.float32), loaded_sample_rate)
    if sample_rate is None:
        raise ValueError("sample_rate is required when transcribing numpy audio arrays")
    return _whisperx_mono_audio(audio, sample_rate)


def _transcription_text_from_segments(
    segments: Sequence[dict[str, Any]],
) -> str:
    return " ".join(
        segment.get("text", "").strip()
        for segment in segments
        if segment.get("text", "").strip()
    )


def _whisperx_mono_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono_audio = np.asarray(audio, dtype=np.float32)
    if mono_audio.ndim == 2:
        mono_audio = mono_audio.mean(axis=1)
    return resample_audio(
        mono_audio,
        input_sample_rate=sample_rate,
        output_sample_rate=_WHISPERX_SAMPLE_RATE,
    )

