from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import math
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Sequence

import numpy as np
import soundfile as sf
from carthage.dependency_injection import AsyncInjectable, inject

from .audio import resample_audio
from .config import ProductionConfig
from .debug import write_debug_json, write_debug_message
from .planning import (
    AudioPlan,
    DialogueAudio,
    DialogueContents,
    DialogueLine,
    PlanningNode,
    ScriptPlan,
)
from .rendering import RenderResult


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_WHISPERX_LANGUAGE = "en"
_WHISPERX_MODEL = "large-v3"
_WHISPERX_SAMPLE_RATE = 16000
_WHISPERX_REQUEST_BATCH_SIZE = 10
_WHISPERX_TRANSCRIBE_BATCH_SIZE = 10
_WHISPERX_ALIGNMENT_THREADS = 4


@dataclass(frozen=True, slots=True)
class AlignedWord:
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
    words: tuple[AlignedWord, ...]
    clauses: tuple[AlignedClause, ...]


@dataclass(frozen=True, slots=True)
class ForcedAlignmentRequest:
    audio: np.ndarray
    sample_rate: int
    transcript: str


@dataclass(frozen=True, slots=True)
class WhisperXResponse:
    transcription_segments: tuple[dict[str, Any], ...]
    aligned_segments: tuple[dict[str, Any], ...] | None
    decision: str


@dataclass(frozen=True, slots=True)
class AlignedScriptResult:
    """Rendered dry script audio plus marker frames for inline insertions."""

    render_result: RenderResult
    marker_frames: tuple[int, ...]
    contents: tuple[DialogueContents, ...]


@dataclass(slots=True)
class RegisteredForcedAlignmentRequest:
    resource: "WhisperXResource"
    request: ForcedAlignmentRequest
    future: asyncio.Future

    async def align(self) -> WhisperXResponse | None:
        return await self.resource.align_registered_request(self)


@dataclass(slots=True)
class _PendingForcedAlignment:
    registration: RegisteredForcedAlignmentRequest


@dataclass(frozen=True, slots=True)
class _PreparedForcedAlignment:
    request: ForcedAlignmentRequest
    mono_audio: np.ndarray
    transcription_segments: tuple[dict[str, Any], ...] | None


@inject(config=ProductionConfig)
class WhisperXResource(AsyncInjectable):
    """Forced-alignment resource that prefers WhisperX and falls back heuristically."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._debug_output_index = 0
        self._debug_output_lock = Lock()
        self._load_lock = RLock()
        self._pending: list[_PendingForcedAlignment] = []
        self._pending_lock = asyncio.Lock()
        self._drain_task: asyncio.Task | None = None
        self._alignment_executor = ThreadPoolExecutor(
            max_workers=_WHISPERX_ALIGNMENT_THREADS,
            thread_name_prefix="whisperx-align",
        )
        self._whisperx_module = None
        self._asr_model = None
        self._align_model = None
        self._align_metadata: dict[str, Any] | None = None

    async def fill_start_positions(
        self,
        contents: Sequence[DialogueContents],
        result: RenderResult,
    ) -> list[DialogueContents]:
        if not any(isinstance(content, DialogueAudio) for content in contents):
            return copy_dialogue_contents(contents)

        transcript = "\n".join(
            content.spoken_text for content in contents if isinstance(content, DialogueLine)
        )
        registration = await self.register_request(
            ForcedAlignmentRequest(
                audio=result.audio,
                sample_rate=self.config.resolved_output_sample_rate,
                transcript=transcript,
            )
        )
        whisperx_response = await registration.align()
        alignment = _alignment_result_from_whisperx_response(
            transcript,
            whisperx_response,
            duration_seconds=_audio_duration(result.audio, self.config.resolved_output_sample_rate),
        )
        return fill_start_positions_from_alignment(contents, alignment)

    async def transcribe_audio_sample(
        self,
        audio: str | Path | np.ndarray,
        sample_rate: int | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self.transcribe_audio_sample_sync,
            audio,
            sample_rate,
        )

    def transcribe_audio_sample_sync(
        self,
        audio: str | Path | np.ndarray,
        sample_rate: int | None = None,
    ) -> str:
        mono_audio = _transcription_sample_audio(audio, sample_rate)
        model = self._ensure_asr_model()
        transcription = model.transcribe(
            mono_audio,
            batch_size=_WHISPERX_TRANSCRIBE_BATCH_SIZE,
            language=_WHISPERX_LANGUAGE,
        )
        return _transcription_text_from_segments(tuple(transcription["segments"]))

    async def register_request(
        self,
        request: ForcedAlignmentRequest,
    ) -> RegisteredForcedAlignmentRequest:
        loop = asyncio.get_running_loop()
        registration = RegisteredForcedAlignmentRequest(
            resource=self,
            request=request,
            future=loop.create_future(),
        )
        async with self._pending_lock:
            self._pending.append(_PendingForcedAlignment(registration=registration))
        return registration

    async def align_registered_request(
        self,
        registration: RegisteredForcedAlignmentRequest,
    ) -> WhisperXResponse | None:
        if registration.future.done():
            return await registration.future
        async with self._pending_lock:
            if self._drain_task is None or self._drain_task.done():
                self._drain_task = asyncio.create_task(self._drain_pending())
        return await registration.future

    async def _drain_pending(self) -> None:
        while True:
            await asyncio.sleep(0)
            async with self._pending_lock:
                if not self._pending:
                    self._drain_task = None
                    return
                batch = self._pending[:_WHISPERX_REQUEST_BATCH_SIZE]
                del self._pending[:_WHISPERX_REQUEST_BATCH_SIZE]

            try:
                responses = await self._process_batch(batch)
            except Exception as exc:
                for pending in batch:
                    if not pending.registration.future.done():
                        pending.registration.future.set_exception(exc)
                continue

            for pending, response in zip(batch, responses, strict=True):
                if not pending.registration.future.done():
                    pending.registration.future.set_result(response)

    async def _process_batch(
        self,
        batch: Sequence[_PendingForcedAlignment],
    ) -> list[WhisperXResponse | None]:
        prepared_batch = await asyncio.to_thread(self._prepare_batch_sync, batch)
        tasks = [
            self._resolve_prepared_alignment(prepared)
            for prepared in prepared_batch
        ]
        return list(await asyncio.gather(*tasks))

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
        batch: Sequence[_PendingForcedAlignment],
    ) -> list[_PreparedForcedAlignment]:
        return [
            self._prepare_request_sync(pending.registration.request)
            for pending in batch
        ]

    def _prepare_request_sync(
        self,
        request: ForcedAlignmentRequest,
    ) -> _PreparedForcedAlignment:
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
        if _line_spans_from_exact_clauses(transcript_lines, transcription_clauses) is not None:
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
        if _line_spans_from_exact_clauses(transcript_lines, aligned_clauses) is not None:
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
            whisperx = self._ensure_whisperx_module()
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=_WHISPERX_LANGUAGE,
                device=self.config.resolved_device,
            )
            return self._align_model, self._align_metadata

    def close(self, canceled_futures: bool = True):
        self._alignment_executor.shutdown(wait=True, cancel_futures=canceled_futures)
        return super().close(canceled_futures=canceled_futures)


@inject(config=ProductionConfig)
class AlignedScriptSource(PlanningNode):
    """Shared render/alignment source for script slices around inline audio."""

    def __init__(
        self,
        node,
        script_plan: ScriptPlan,
        **kwargs,
    ) -> None:
        super().__init__(node=node, **kwargs)
        self.script_plan = script_plan
        self.contents: list[DialogueContents] = copy_dialogue_contents(script_plan.contents)

    async def render(self) -> AlignedScriptResult:
        return await super().render()

    async def render_node(self) -> AlignedScriptResult:
        base_result = await self.script_plan.render()
        resource = await self.ainjector.get_instance_async(WhisperXResource)
        self.contents = await resource.fill_start_positions(self.script_plan.contents, base_result)
        for content in self.contents:
            if not isinstance(content, DialogueLine):
                continue
            preview = _debug_line_preview(content.spoken_text)
            write_debug_message(
                self.config,
                "forced_alignment",
                f"{content.start_pos:.3f}s {preview}",
            )
        return AlignedScriptResult(
            render_result=base_result,
            marker_frames=_marker_frames_from_contents(
                self.contents,
                frame_count=base_result.frame_count,
                sample_rate=self.config.resolved_output_sample_rate,
            ),
            contents=tuple(self.contents),
        )


@inject(config=ProductionConfig)
class ScriptSlice(AudioPlan):
    """Audio plan that slices an aligned script by marker index."""

    def __init__(
        self,
        aligned_script_source: AlignedScriptSource,
        *,
        start_marker: int,
        end_marker: int,
        name: str | None = None,
        node=None,
        **kwargs,
    ) -> None:
        super().__init__(node=node, set_gap=False, **kwargs)
        self.aligned_script_source = aligned_script_source
        self.start_marker = start_marker
        self.end_marker = end_marker
        self.name = name

    def __repr__(self) -> str:
        if self.name is not None:
            return f"ScriptSlice(name={self.name!r})"
        return (
            "ScriptSlice("
            f"start_marker={self.start_marker}, "
            f"end_marker={self.end_marker})"
        )

    async def render_node(self) -> RenderResult:
        aligned_result = await self.aligned_script_source.render()
        start_frame = aligned_result.marker_frames[self.start_marker]
        end_frame = aligned_result.marker_frames[self.end_marker]
        end_frame = max(start_frame, end_frame)
        return RenderResult(audio=aligned_result.render_result.audio[start_frame:end_frame])


def copy_dialogue_contents(contents: Sequence[DialogueContents]) -> list[DialogueContents]:
    copied: list[DialogueContents] = []
    for content in contents:
        if isinstance(content, DialogueLine):
            copied.append(
                DialogueLine(
                    speaker=content.speaker,
                    spoken_text=content.spoken_text,
                    start_pos=content.start_pos,
                )
            )
        else:
            copied.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=content.start_pos))
    return copied


def fill_start_positions_from_alignment(
    contents: Sequence[DialogueContents],
    alignment: AlignmentResult,
) -> list[DialogueContents]:
    copied_contents = copy_dialogue_contents(contents)
    dialogue_lines = [content for content in copied_contents if isinstance(content, DialogueLine)]
    raw_line_spans = _line_spans_from_alignment(dialogue_lines, alignment)
    stabilized_line_spans = _stabilize_line_spans(raw_line_spans)

    line_index = 0
    for content in copied_contents:
        if isinstance(content, DialogueLine):
            start_pos, _ = raw_line_spans[line_index]
            content.start_pos = cast_float(start_pos) if start_pos is not None else math.nan
            line_index += 1

    for index, content in enumerate(copied_contents):
        if isinstance(content, DialogueAudio):
            content.start_pos = _dialogue_audio_start_pos(copied_contents, stabilized_line_spans, index)

    return copied_contents


def _marker_frames_from_contents(
    contents: Sequence[DialogueContents],
    *,
    frame_count: int,
    sample_rate: int,
) -> tuple[int, ...]:
    total_duration = 0.0 if sample_rate <= 0 else float(frame_count) / sample_rate
    marker_seconds: list[float] = [0.0]

    for content in contents:
        if not isinstance(content, DialogueAudio):
            continue
        marker_seconds.append(min(max(cast_float(content.start_pos), 0.0), total_duration))

    marker_seconds.append(total_duration)
    stabilized_seconds: list[float] = []
    previous = 0.0
    for second in marker_seconds:
        stabilized = min(max(second, previous), total_duration)
        stabilized_seconds.append(stabilized)
        previous = stabilized

    return tuple(
        min(frame_count, max(0, int(round(second * sample_rate))))
        for second in stabilized_seconds
    )


def _line_spans_from_alignment(
    dialogue_lines: Sequence[DialogueLine],
    alignment: AlignmentResult,
) -> list[tuple[float | None, float | None]]:
    if not dialogue_lines:
        return []

    exact_clause_spans = _line_spans_from_exact_clauses(
        [line.spoken_text for line in dialogue_lines],
        alignment.clauses,
    )
    if exact_clause_spans is not None:
        return exact_clause_spans

    clause_starts = _clause_starts_by_token_offset(alignment.clauses)
    clause_endings = _clause_endings_by_token_offset(alignment.clauses)
    aligned_tokens = _aligned_word_tokens(alignment.words)
    word_search_index = 0
    cumulative_line_tokens = 0
    spans: list[tuple[float | None, float | None]] = []

    for line in dialogue_lines:
        line_tokens = _normalized_tokens(line.spoken_text)
        line_start_offset = cumulative_line_tokens
        cumulative_line_tokens += len(line_tokens)
        start_time: float | None = None
        end_time: float | None = None
        matched_word_span = _match_line_in_aligned_tokens(
            line_tokens,
            aligned_tokens,
            start_index=word_search_index,
        )
        if matched_word_span is not None:
            start_time, end_time, word_search_index = matched_word_span

        clause_start = clause_starts.get(line_start_offset)
        if (
            clause_start is not None
            and clause_start.start is not None
            and _line_begins_with_clause(line_tokens, clause_start)
        ):
            start_time = clause_start.start
        clause_match = clause_endings.get(cumulative_line_tokens)
        if clause_match is not None and _line_ends_with_clause(line_tokens, clause_match):
            if clause_match.end is not None:
                end_time = clause_match.end
        spans.append((start_time, end_time))

    return spans


def _stabilize_line_spans(
    spans: Sequence[tuple[float | None, float | None]],
) -> list[tuple[float, float]]:
    stabilized: list[list[float | None]] = [[start, end] for start, end in spans]

    previous_end = 0.0
    for span in stabilized:
        if span[0] is None:
            span[0] = previous_end
        if span[1] is None:
            span[1] = span[0]
        previous_end = cast_float(span[1])

    next_start: float | None = None
    for span in reversed(stabilized):
        if span[1] is None and next_start is not None:
            span[1] = next_start
        if span[0] is None:
            span[0] = span[1] if span[1] is not None else 0.0
        next_start = cast_float(span[0])

    return [(cast_float(start), cast_float(end)) for start, end in stabilized]


def _dialogue_audio_start_pos(
    contents: Sequence[DialogueContents],
    line_spans: Sequence[tuple[float, float]],
    audio_index: int,
) -> float:
    previous_end: float | None = None
    next_start: float | None = None

    line_counter = 0
    for index, content in enumerate(contents):
        if isinstance(content, DialogueLine):
            start, end = line_spans[line_counter]
            if index < audio_index:
                previous_end = end
            elif index > audio_index and next_start is None:
                next_start = start
                break
            line_counter += 1

    if previous_end is not None and next_start is not None:
        return (previous_end + next_start) / 2.0
    if previous_end is not None:
        return previous_end
    return 0.0


def _alignment_result_from_whisperx(
    payload: dict,
    *,
    clauses: Sequence[AlignedClause] | None = None,
) -> AlignmentResult:
    segments = payload.get("segments", [])
    words: list[AlignedWord] = []
    for segment in segments:
        for word in segment.get("words", []) or []:
            words.append(
                AlignedWord(
                    text=str(word.get("word", "")),
                    start=_optional_float(word.get("start")),
                    end=_optional_float(word.get("end")),
                )
            )
    if clauses is None:
        clauses = _clauses_from_segments(segments)
    return AlignmentResult(words=tuple(words), clauses=tuple(clauses))


def _alignment_result_from_whisperx_response(
    transcript: str,
    response: WhisperXResponse | None,
    *,
    duration_seconds: float,
) -> AlignmentResult:
    if response is None:
        return _fallback_alignment_result(transcript, duration_seconds=duration_seconds)
    if response.decision == "transcription_exact_clause_match":
        clauses = _clauses_from_segments(response.transcription_segments)
        return AlignmentResult(words=(), clauses=tuple(clauses))
    if response.decision == "aligned_exact_clause_match" and response.aligned_segments is not None:
        clauses = _clauses_from_segments(response.aligned_segments)
        return AlignmentResult(words=(), clauses=tuple(clauses))
    if response.aligned_segments is None:
        clauses = _clauses_from_segments(response.transcription_segments)
        return AlignmentResult(words=(), clauses=tuple(clauses))
    return _alignment_result_from_whisperx(
        {"segments": list(response.aligned_segments)},
        clauses=_clauses_from_segments(response.aligned_segments),
    )


def _fallback_alignment_result(transcript: str, *, duration_seconds: float) -> AlignmentResult:
    clauses: list[AlignedClause] = []
    words: list[AlignedWord] = []
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    line_tokens = [_normalized_tokens(line) or [""] for line in lines]
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
            words.append(AlignedWord(text=token, start=word_start, end=word_end))
        cursor = line_end

    return AlignmentResult(words=tuple(words), clauses=tuple(clauses))


def _clause_endings_by_token_offset(clauses: Sequence[AlignedClause]) -> dict[int, AlignedClause]:
    clause_endings: dict[int, AlignedClause] = {}
    token_offset = 0
    for clause in clauses:
        token_offset += len(_normalized_tokens(clause.text))
        clause_endings[token_offset] = clause
    return clause_endings


def _clause_starts_by_token_offset(clauses: Sequence[AlignedClause]) -> dict[int, AlignedClause]:
    clause_starts: dict[int, AlignedClause] = {}
    token_offset = 0
    for clause in clauses:
        token_count = len(_normalized_tokens(clause.text))
        if token_count == 0:
            continue
        clause_starts[token_offset] = clause
        token_offset += token_count
    return clause_starts


def _clauses_from_segments(segments: Sequence[dict]) -> list[AlignedClause]:
    return [
        AlignedClause(
            text=str(segment.get("text", "")),
            start=_optional_float(segment.get("start")),
            end=_optional_float(segment.get("end")),
        )
        for segment in segments
    ]


def _line_spans_from_exact_clauses(
    line_texts: Sequence[str],
    clauses: Sequence[AlignedClause],
) -> list[tuple[float | None, float | None]] | None:
    if not line_texts:
        return []

    clause_index = 0
    spans: list[tuple[float | None, float | None]] = []

    for line_text in line_texts:
        target_token_count = len(_normalized_tokens(line_text))
        accumulated_tokens = 0
        line_start: float | None = None
        line_end: float | None = None

        while accumulated_tokens < target_token_count and clause_index < len(clauses):
            clause = clauses[clause_index]
            clause_index += 1
            clause_token_count = len(_normalized_tokens(clause.text))
            if clause_token_count == 0:
                continue
            if line_start is None and clause.start is not None:
                line_start = clause.start
            if clause.end is not None:
                line_end = clause.end
            accumulated_tokens += clause_token_count

        if accumulated_tokens != target_token_count:
            return None
        spans.append((line_start, line_end))

    remaining_clause_tokens = sum(
        len(_normalized_tokens(clause.text))
        for clause in clauses[clause_index:]
    )
    if remaining_clause_tokens != 0:
        return None
    return spans


def _aligned_word_tokens(
    words: Sequence[AlignedWord],
) -> list[tuple[str, float | None, float | None]]:
    aligned_tokens: list[tuple[str, float | None, float | None]] = []
    for word in words:
        normalized_tokens = _normalized_tokens(word.text)
        if not normalized_tokens:
            continue
        for token in normalized_tokens:
            aligned_tokens.append((token, word.start, word.end))
    return aligned_tokens


def _match_line_in_aligned_tokens(
    line_tokens: Sequence[str],
    aligned_tokens: Sequence[tuple[str, float | None, float | None]],
    *,
    start_index: int,
) -> tuple[float | None, float | None, int] | None:
    if not line_tokens:
        return None

    for candidate_index in range(start_index, len(aligned_tokens)):
        if aligned_tokens[candidate_index][0] != line_tokens[0]:
            continue

        matched_end_index = _match_exact_token_run(
            line_tokens,
            aligned_tokens,
            start_index=candidate_index,
        )
        if matched_end_index is None:
            continue

        start_time = aligned_tokens[candidate_index][1]
        end_time = aligned_tokens[matched_end_index][2]
        return start_time, end_time, matched_end_index + 1
    return None


def _match_exact_token_run(
    line_tokens: Sequence[str],
    aligned_tokens: Sequence[tuple[str, float | None, float | None]],
    *,
    start_index: int,
) -> int | None:
    end_index = start_index + len(line_tokens)
    if end_index > len(aligned_tokens):
        return None
    candidate_tokens = [token for token, _, _ in aligned_tokens[start_index:end_index]]
    if list(line_tokens) != candidate_tokens:
        return None
    return end_index - 1


def _line_begins_with_clause(
    line_tokens: Sequence[str],
    clause: AlignedClause,
) -> bool:
    clause_tokens = _normalized_tokens(clause.text)
    if not clause_tokens or len(clause_tokens) > len(line_tokens):
        return False
    return list(line_tokens[: len(clause_tokens)]) == clause_tokens


def _line_ends_with_clause(
    line_tokens: Sequence[str],
    clause: AlignedClause,
) -> bool:
    clause_tokens = _normalized_tokens(clause.text)
    if not clause_tokens or len(clause_tokens) > len(line_tokens):
        return False
    return list(line_tokens[-len(clause_tokens) :]) == clause_tokens


def _normalized_tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _transcript_lines(transcript: str) -> list[str]:
    return [line.strip() for line in transcript.splitlines() if line.strip()]


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(audio.shape[0]) / sample_rate


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


def cast_float(value: float | None) -> float:
    if value is None or math.isnan(value):
        return 0.0
    return float(value)


def _debug_line_preview(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= 60:
        return repr(normalized)
    return f"{normalized[:30]!r} ... {normalized[-30:]!r}"


def _debug_transcript_label(transcript: str) -> str:
    first_line = next(
        (line.strip() for line in transcript.splitlines() if line.strip()),
        "empty-script",
    )
    return first_line[:40] or "empty-script"


def _sanitize_debug_label(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return sanitized or "alignment"
