from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from threading import Lock
from typing import Sequence

import numpy as np
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
class AlignedScriptResult:
    """Rendered dry script audio plus marker frames for inline insertions."""

    render_result: RenderResult
    marker_frames: tuple[int, ...]
    contents: tuple[DialogueContents, ...]


@inject(config=ProductionConfig)
class WhisperXResource(AsyncInjectable):
    """Forced-alignment resource that prefers WhisperX and falls back heuristically."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._debug_output_index = 0
        self._debug_output_lock = Lock()

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
        alignment = await asyncio.to_thread(
            self._alignment_result_sync,
            result.audio,
            self.config.resolved_output_sample_rate,
            transcript,
        )
        return fill_start_positions_from_alignment(contents, alignment)

    def _alignment_result_sync(
        self,
        audio: np.ndarray,
        sample_rate: int,
        transcript: str,
    ) -> AlignmentResult:
        try:
            import whisperx  # type: ignore[import-not-found]
        except ImportError:
            return _fallback_alignment_result(transcript, duration_seconds=_audio_duration(audio, sample_rate))

        mono_audio = np.asarray(audio, dtype=np.float32)
        if mono_audio.ndim == 2:
            mono_audio = mono_audio.mean(axis=1)
        mono_audio = resample_audio(
            mono_audio,
            input_sample_rate=sample_rate,
            output_sample_rate=_WHISPERX_SAMPLE_RATE,
        )

        device = self.config.resolved_device
        model = whisperx.load_model(
            _WHISPERX_MODEL,
            device=device,
            compute_type="default",
            language=_WHISPERX_LANGUAGE,
        )
        transcription = model.transcribe(mono_audio, batch_size=1, language=_WHISPERX_LANGUAGE)
        transcription_clauses = _clauses_from_segments(transcription["segments"])
        transcript_lines = [line.strip() for line in transcript.splitlines() if line.strip()]
        if _line_spans_from_exact_clauses(transcript_lines, transcription_clauses) is not None:
            self._write_whisperx_debug_output(
                transcript=transcript,
                transcription_segments=transcription["segments"],
                aligned_segments=None,
                decision="transcription_exact_clause_match",
            )
            return AlignmentResult(words=(), clauses=tuple(transcription_clauses))
        align_model, metadata = whisperx.load_align_model(
            language_code=_WHISPERX_LANGUAGE,
            device=device,
        )
        aligned = whisperx.align(
            transcription["segments"],
            align_model,
            metadata,
            mono_audio,
            device,
            return_char_alignments=False,
        )
        aligned_clauses = _clauses_from_segments(aligned.get("segments", []))
        if _line_spans_from_exact_clauses(transcript_lines, aligned_clauses) is not None:
            self._write_whisperx_debug_output(
                transcript=transcript,
                transcription_segments=transcription["segments"],
                aligned_segments=aligned.get("segments", []),
                decision="aligned_exact_clause_match",
            )
            return AlignmentResult(words=(), clauses=tuple(aligned_clauses))
        self._write_whisperx_debug_output(
            transcript=transcript,
            transcription_segments=transcription["segments"],
            aligned_segments=aligned.get("segments", []),
            decision="aligned_word_matching",
        )
        return _alignment_result_from_whisperx(
            aligned,
            clauses=aligned_clauses,
        )

    def _write_whisperx_debug_output(
        self,
        *,
        transcript: str,
        transcription_segments: Sequence[dict],
        aligned_segments: Sequence[dict] | None,
        decision: str,
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
                "decision": decision,
                "transcript": transcript,
                "transcription_segments": list(transcription_segments),
                "aligned_segments": list(aligned_segments) if aligned_segments is not None else None,
            },
        )
        if artifact_path is not None:
            write_debug_message(
                self.config,
                "whisperx",
                f"{artifact_path.name} decision={decision}",
            )

    def _reserve_debug_output_index(self) -> int:
        with self._debug_output_lock:
            output_index = self._debug_output_index
            self._debug_output_index += 1
        return output_index


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
    word_index = 0
    cumulative_line_tokens = 0
    spans: list[tuple[float | None, float | None]] = []

    for line in dialogue_lines:
        line_tokens = _normalized_tokens(line.spoken_text)
        line_start_offset = cumulative_line_tokens
        cumulative_line_tokens += len(line_tokens)
        start_time: float | None = None
        end_time: float | None = None
        token_index = 0

        while word_index < len(alignment.words) and token_index < len(line_tokens):
            word = alignment.words[word_index]
            word_index += 1
            normalized_word = _normalized_tokens(word.text)
            if not normalized_word:
                continue
            token = normalized_word[0]
            if token != line_tokens[token_index]:
                continue
            if start_time is None and word.start is not None:
                start_time = word.start
            if word.end is not None:
                end_time = word.end
            token_index += 1

        clause_start = clause_starts.get(line_start_offset)
        if clause_start is not None and clause_start.start is not None:
            start_time = clause_start.start
        clause_match = clause_endings.get(cumulative_line_tokens)
        if clause_match is not None:
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


def _normalized_tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return float(audio.shape[0]) / sample_rate


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
