from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from carthage.dependency_injection import AsyncInjectable, inject

from .config import ProductionConfig
from .planning import AudioPlan, DialogueAudio, DialogueContents, DialogueLine, ScriptPlan
from .rendering import RenderResult


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


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


@inject(config=ProductionConfig)
class WhisperXResource(AsyncInjectable):
    """Forced-alignment resource that prefers WhisperX and falls back heuristically."""

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

        device = self.config.resolved_device
        model = whisperx.load_model("small", device=device, compute_type="float32")
        transcription = model.transcribe(mono_audio, batch_size=1)
        language_code = transcription.get("language", "en")
        align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        aligned = whisperx.align(
            transcription["segments"],
            align_model,
            metadata,
            mono_audio,
            device,
            return_char_alignments=False,
        )
        return _alignment_result_from_whisperx(aligned)


@inject(config=ProductionConfig)
class ForcedAlignmentPlan(AudioPlan):
    """Wrapper that fills start positions for dialogue contents without changing audio."""

    def __init__(
        self,
        node,
        script_plan: ScriptPlan,
        **kwargs,
    ) -> None:
        super().__init__(node=node, set_gap=False, **kwargs)
        self.script_plan = script_plan
        self.contents: list[DialogueContents] = copy_dialogue_contents(script_plan.contents)

    def leaf_audio_plans(self) -> list[AudioPlan]:
        return self.script_plan.leaf_audio_plans()

    async def render_node(self) -> RenderResult:
        base_result = await self.script_plan.render()
        resource = await self.ainjector.get_instance_async(WhisperXResource)
        self.contents = await resource.fill_start_positions(self.script_plan.contents, base_result)
        return base_result


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
    line_spans = _line_spans_from_alignment(dialogue_lines, alignment)

    line_index = 0
    for content in copied_contents:
        if isinstance(content, DialogueLine):
            start_pos, _ = line_spans[line_index]
            content.start_pos = start_pos
            line_index += 1

    for index, content in enumerate(copied_contents):
        if isinstance(content, DialogueAudio):
            content.start_pos = _dialogue_audio_start_pos(copied_contents, line_spans, index)

    return copied_contents


def _line_spans_from_alignment(
    dialogue_lines: Sequence[DialogueLine],
    alignment: AlignmentResult,
) -> list[tuple[float, float]]:
    if not dialogue_lines:
        return []

    clause_endings = _clause_endings_by_token_offset(alignment.clauses)
    word_index = 0
    cumulative_line_tokens = 0
    spans: list[tuple[float | None, float | None]] = []

    for line in dialogue_lines:
        line_tokens = _normalized_tokens(line.spoken_text)
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

        clause_match = clause_endings.get(cumulative_line_tokens)
        if clause_match is not None:
            if start_time is None and clause_match.start is not None:
                start_time = clause_match.start
            if clause_match.end is not None:
                end_time = clause_match.end
        spans.append((start_time, end_time))

    return _stabilize_line_spans(spans)


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


def _alignment_result_from_whisperx(payload: dict) -> AlignmentResult:
    segments = payload.get("segments", [])
    words: list[AlignedWord] = []
    clauses: list[AlignedClause] = []
    for segment in segments:
        clauses.append(
            AlignedClause(
                text=str(segment.get("text", "")),
                start=_optional_float(segment.get("start")),
                end=_optional_float(segment.get("end")),
            )
        )
        for word in segment.get("words", []) or []:
            words.append(
                AlignedWord(
                    text=str(word.get("word", "")),
                    start=_optional_float(word.get("start")),
                    end=_optional_float(word.get("end")),
                )
            )
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
