from __future__ import annotations

from functools import lru_cache
import logging
import math
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..dialogue import DialogueAudio, DialogueContent, DialogueLine, ScriptEvent, ScriptGap
from ..rendering import DialogueLineTiming, ScriptTiming
from ..text import normalize_text_punctuation


_TOKEN_RE = re.compile(r"[A-Za-z']+|[0-9]|(?<=[0-9])\.(?=[0-9])")
_NUMBER_TOKENS = dict(zip(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "point"),
    (*"0123456789", "."),
))
_CARDINALS = dict(zip(
    ('ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
     'sixteen', 'seventeen', 'eighteen', 'nineteen'), range(10, 20)))
_TENS = dict(zip(('twenty', 'thirty', 'forty', 'fifty', 'sixty',
                 'seventy', 'eighty', 'ninety'), range(20, 100, 10)))
logger = logging.getLogger(__name__)


from .base import AlignedClause, AlignmentResult, WordTiming

@dataclass(frozen=True, slots=True)
class _WordMatchCandidate:
    start_time: float | None
    end_time: float | None
    next_search_index: int
    normalized_cost: float
    exact_match_count: int
    anchor_match_count: int
    token_pairs: tuple[tuple[int, int], ...] = ()


def copy_dialogue_contents(contents: Sequence[ScriptEvent]) -> list[ScriptEvent]:
    copied: list[ScriptEvent] = []
    for content in contents:
        if isinstance(content, DialogueLine):
            copied.append(
                DialogueLine(
                    speaker=content.speaker,
                    spoken_text=content.spoken_text,
                    handling=content.handling,
                    source=content.source,
                    node=content.node,
                    mark_offsets=content.mark_offsets,
                    start_pos=content.start_pos,
                )
            )
        elif isinstance(content, ScriptGap):
            copied.append(
                ScriptGap(
                    label=content.label,
                    mode=content.mode,
                    start_pos=content.start_pos,
                )
            )
        else:
            copied.append(DialogueAudio(audio_plan=content.audio_plan, start_pos=content.start_pos))
    return copied


def fill_start_positions_from_alignment(
    contents: Sequence[ScriptEvent],
    alignment: AlignmentResult,
) -> list[ScriptEvent]:
    dialogue_contents = [content for content in contents if isinstance(content, DialogueContent)]
    spans = _dialogue_content_spans_from_alignment(dialogue_contents, alignment)
    return _fill_start_positions_from_spans(contents, spans)


def fill_start_positions_from_timing(
    contents: Sequence[ScriptEvent],
    timing: ScriptTiming,
) -> list[ScriptEvent]:
    """Project cached or native timing without losing unknown boundaries."""
    line_spans = [
        (None if math.isnan(line.start) else line.start,
         None if math.isnan(line.end) else line.end)
        for line in timing.dialogue_lines
    ]
    dialogue_contents = [content for content in contents if isinstance(content, DialogueContent)]
    spans = _merge_dialogue_content_spans(dialogue_contents, line_spans)
    return _fill_start_positions_from_spans(contents, spans)


def _fill_start_positions_from_spans(
    contents: Sequence[ScriptEvent],
    spans: Sequence[tuple[float | None, float | None]],
) -> list[ScriptEvent]:
    """Keep raw dialogue boundaries and use legacy fallback only for inline audio."""
    copied_contents = copy_dialogue_contents(contents)
    stabilized_spans = _stabilize_line_spans(spans)
    content_index = 0
    for index, content in enumerate(copied_contents):
        if isinstance(content, DialogueContent):
            start, _ = spans[content_index]
            content.start_pos = math.nan if start is None else start
            content_index += 1
        elif isinstance(content, DialogueAudio):
            content.start_pos = _dialogue_audio_start_pos(
                copied_contents, stabilized_spans, index,
            )
    return copied_contents


def _marker_frames_from_contents(
    contents: Sequence[ScriptEvent],
    *,
    frame_count: int,
    sample_rate: int,
) -> tuple[int, ...]:
    total_duration = 0.0 if sample_rate <= 0 else float(frame_count) / sample_rate
    marker_seconds: list[float] = [0.0]

    for index in range(1, len(contents)):
        previous = contents[index - 1]
        current = contents[index]
        boundary_pos = previous.start_pos if isinstance(previous, DialogueAudio) else current.start_pos
        marker_seconds.append(min(max(cast_float(boundary_pos), 0.0), total_duration))

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


def _boundary_info_for_marker(
    contents: Sequence[ScriptEvent],
    marker_index: int,
) -> tuple[float, str] | None:
    if marker_index <= 0 or marker_index >= len(contents):
        return None
    previous = contents[marker_index - 1]
    current = contents[marker_index]
    if isinstance(previous, DialogueAudio):
        return previous.start_pos, f"after {_content_debug_label(previous)}"
    return current.start_pos, f"before {_content_debug_label(current)}"


def _line_spans_from_alignment(
    dialogue_lines: Sequence[DialogueLine],
    alignment: AlignmentResult,
    *,
    allow_positional_clauses: bool = True,
    correspondences: list | None = None,
) -> list[tuple[float | None, float | None]]:
    if not dialogue_lines:
        return []

    normalized_line_tokens = [tuple(_normalized_tokens(line.spoken_text)) for line in dialogue_lines]
    clause_starts = _clause_starts_by_token_offset(alignment.clauses)
    clause_endings = _clause_endings_by_token_offset(alignment.clauses)
    aligned_tokens = _aligned_word_tokens(alignment.words or ())
    aligned_token_index = _aligned_token_positions_by_text(aligned_tokens)
    word_search_index = 0
    cumulative_line_tokens = 0
    spans: list[tuple[float | None, float | None]] = []

    earliest = 0.0
    clause_sources = (alignment.preferred_clauses, alignment.clauses)
    clause_cursors = [0, 0]
    for line_tokens in normalized_line_tokens:
        search_start = word_search_index
        selected_pairs = ()
        line_start_offset = cumulative_line_tokens
        cumulative_line_tokens += len(line_tokens)
        clause_match = None
        for source_index, clauses in enumerate(clause_sources):
            clause_match = _find_exact_clause_span(
                line_tokens, clauses, start_index=clause_cursors[source_index], earliest=earliest,
            )
            if clause_match is not None:
                start, end, next_clause = clause_match
                clause_cursors[source_index] = next_clause
                spans.append((start, end))
                if correspondences is not None:
                    match = _match_line_candidate(line_tokens, aligned_tokens, aligned_token_index,
                                                 start_index=search_start)
                    correspondences.append(() if match is None else match.token_pairs)
                earliest = max(earliest, end)
                while word_search_index < len(aligned_tokens):
                    word_start = aligned_tokens[word_search_index][1]
                    if word_start is not None and word_start >= earliest:
                        break
                    word_search_index += 1
                break
        if clause_match is not None:
            continue
        start_time: float | None = None
        end_time: float | None = None
        matched_candidate = _match_line_candidate(
            line_tokens,
            aligned_tokens,
            aligned_token_index,
            start_index=word_search_index,
        )
        if matched_candidate is not None:
            start_time = matched_candidate.start_time
            end_time = matched_candidate.end_time
            word_search_index = matched_candidate.next_search_index
            selected_pairs = matched_candidate.token_pairs
        if correspondences is not None:
            correspondences.append(selected_pairs)

        clause_start = clause_starts.get(line_start_offset)
        if (
            allow_positional_clauses
            and clause_start is not None
            and clause_start.start is not None
            and clause_start.start >= earliest
            and _line_begins_with_clause(line_tokens, clause_start)
        ):
            start_time = clause_start.start
        clause_match = clause_endings.get(cumulative_line_tokens)
        if allow_positional_clauses and clause_match is not None and _line_ends_with_clause(line_tokens, clause_match):
            if clause_match.end is not None and clause_match.end >= max(earliest, start_time or 0.0):
                end_time = clause_match.end
        spans.append((start_time, end_time))
        if end_time is not None:
            earliest = max(earliest, end_time)

    return spans


def _dialogue_content_spans_from_alignment(
    dialogue_contents: Sequence[DialogueContent],
    alignment: AlignmentResult,
) -> list[tuple[float | None, float | None]]:
    dialogue_lines = [content for content in dialogue_contents if isinstance(content, DialogueLine)]
    if not dialogue_lines:
        return [(None, None) for _ in dialogue_contents]
    if any(isinstance(content, ScriptGap) for content in dialogue_contents):
        line_spans = _line_spans_from_alignment_segments(dialogue_contents, alignment)
    else:
        line_spans = _line_spans_from_alignment(dialogue_lines, alignment)
    return _merge_dialogue_content_spans(dialogue_contents, line_spans)


def _merge_dialogue_content_spans(
    dialogue_contents: Sequence[DialogueContent],
    line_spans: Sequence[tuple[float | None, float | None]],
) -> list[tuple[float | None, float | None]]:
    spans: list[tuple[float | None, float | None]] = [(None, None)] * len(dialogue_contents)
    line_index = 0
    for content_index, content in enumerate(dialogue_contents):
        if isinstance(content, DialogueLine):
            spans[content_index] = line_spans[line_index]
            line_index += 1

    if line_index != len(line_spans):
        raise RuntimeError("Dialogue content spans did not consume all dialogue line spans")

    for content_index, content in enumerate(dialogue_contents):
        if not isinstance(content, ScriptGap):
            continue
        previous_end = next(
            (
                spans[index][1]
                for index in range(content_index - 1, -1, -1)
                if isinstance(dialogue_contents[index], DialogueLine) and spans[index][1] is not None
            ),
            None,
        )
        next_start = next(
            (
                spans[index][0]
                for index in range(content_index + 1, len(dialogue_contents))
                if isinstance(dialogue_contents[index], DialogueLine) and spans[index][0] is not None
            ),
            None,
        )
        spans[content_index] = (previous_end, next_start)

    return spans


def _line_spans_from_alignment_segments(
    dialogue_contents: Sequence[DialogueContent],
    alignment: AlignmentResult,
) -> list[tuple[float | None, float | None]]:
    return _line_spans_from_alignment(
        [content for content in dialogue_contents if isinstance(content, DialogueLine)],
        alignment,
        allow_positional_clauses=False,
    )


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
    contents: Sequence[ScriptEvent],
    dialogue_content_spans: Sequence[tuple[float, float]],
    audio_index: int,
) -> float:
    previous_end: float | None = None
    next_start: float | None = None

    content_counter = 0
    for index, content in enumerate(contents):
        if isinstance(content, DialogueContent):
            start, end = dialogue_content_spans[content_counter]
            if index < audio_index:
                previous_end = end
            elif index > audio_index and next_start is None:
                next_start = start
                break
            content_counter += 1

    if previous_end is not None and next_start is not None:
        return (previous_end + next_start) / 2.0
    if previous_end is not None:
        return previous_end
    return 0.0


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


def _find_exact_clause_span(
    line_tokens: Sequence[str],
    clauses: Sequence[AlignedClause],
    *,
    start_index: int,
    earliest: float,
) -> tuple[float, float, int] | None:
    """Find whole consecutive clauses by text, without moving behind prior output.

    Original ASR segments and acoustically aligned clauses are searched
    separately: exact original segments keep their independent boundaries.
    """
    target = tuple(line_tokens)
    if not target:
        return None
    for index in range(start_index, len(clauses)):
        start = clauses[index].start
        if start is None or start < earliest:
            continue
        tokens: list[str] = []
        for end_index in range(index, len(clauses)):
            tokens.extend(_normalized_tokens(clauses[end_index].text))
            if tuple(tokens) != target[:len(tokens)]:
                break
            if tuple(tokens) == target:
                end = clauses[end_index].end
                if end is not None and end >= start:
                    return start, end, end_index + 1
                break
    return None


def _line_spans_from_exact_clauses(
    line_texts: Sequence[str],
    clauses: Sequence[AlignedClause],
) -> list[tuple[float | None, float | None]] | None:
    if not line_texts:
        return []

    clause_index = 0
    spans: list[tuple[float | None, float | None]] = []
    clause_token_counts = [len(_normalized_tokens(clause.text)) for clause in clauses]

    for line_text in line_texts:
        target_token_count = len(_normalized_tokens(line_text))
        accumulated_tokens = 0
        matched_tokens: list[str] = []
        line_start: float | None = None
        line_end: float | None = None

        while accumulated_tokens < target_token_count and clause_index < len(clauses):
            clause = clauses[clause_index]
            clause_token_count = clause_token_counts[clause_index]
            clause_index += 1
            if clause_token_count == 0:
                continue
            if line_start is None and clause.start is not None:
                line_start = clause.start
            if clause.end is not None:
                line_end = clause.end
            accumulated_tokens += clause_token_count
            matched_tokens.extend(_normalized_tokens(clause.text))

        if tuple(matched_tokens) != _normalized_tokens(line_text):
            return None
        spans.append((line_start, line_end))

    remaining_clause_tokens = sum(clause_token_counts[clause_index:])
    if remaining_clause_tokens != 0:
        return None
    return spans


def _aligned_word_tokens(
    words: Sequence[WordTiming],
) -> list[tuple[str, float | None, float | None]]:
    raw_tokens = []
    for word in words:
        for token in _TOKEN_RE.findall(normalize_text_punctuation(word.text)):
            raw_tokens.append((token.lower(), word.start, word.end))
    return [(token, raw_tokens[first][1], raw_tokens[last][2])
            for token, first, last in _number_normalized_tokens([item[0] for item in raw_tokens])]


def _aligned_token_positions_by_text(
    aligned_tokens: Sequence[tuple[str, float | None, float | None]],
) -> dict[str, tuple[int, ...]]:
    positions: dict[str, list[int]] = {}
    for index, (token, _, _) in enumerate(aligned_tokens):
        positions.setdefault(token, []).append(index)
    return {
        token: tuple(token_positions)
        for token, token_positions in positions.items()
    }


def _match_line_in_aligned_tokens(line_tokens, aligned_tokens, aligned_token_index, *, start_index):
    candidate = _match_line_candidate(line_tokens, aligned_tokens, aligned_token_index,
                                     start_index=start_index)
    if candidate is None:
        return None
    return candidate.start_time, candidate.end_time, candidate.next_search_index


def _match_line_candidate(
    line_tokens: Sequence[str],
    aligned_tokens: Sequence[tuple[str, float | None, float | None]],
    aligned_token_index: dict[str, tuple[int, ...]],
    *,
    start_index: int,
) -> _WordMatchCandidate | None:
    if not line_tokens:
        return None

    max_length_slop = max(2, len(line_tokens) // 3)
    min_candidate_length = max(1, len(line_tokens) - max_length_slop)

    best_candidate = _best_word_match_candidate(
        line_tokens,
        aligned_tokens,
        candidate_indexes=_candidate_start_indexes(
            line_tokens,
            aligned_tokens,
            aligned_token_index,
            start_index=start_index,
        ),
        min_candidate_length=min_candidate_length,
        max_length_slop=max_length_slop,
    )
    if best_candidate is None:
        best_candidate = _best_word_match_candidate(
            line_tokens,
            aligned_tokens,
            candidate_indexes=range(start_index, len(aligned_tokens)),
            min_candidate_length=min_candidate_length,
            max_length_slop=max_length_slop,
        )

    if best_candidate is None:
        return None
    if best_candidate.normalized_cost > 0.35:
        return None
    if best_candidate.exact_match_count == 0:
        return None
    if best_candidate.anchor_match_count == 0:
        return None
    return best_candidate


def _best_word_match_candidate(
    line_tokens: Sequence[str],
    aligned_tokens: Sequence[tuple[str, float | None, float | None]],
    *,
    candidate_indexes: Sequence[int] | range,
    min_candidate_length: int,
    max_length_slop: int,
) -> _WordMatchCandidate | None:
    best_candidate: _WordMatchCandidate | None = None

    for candidate_index in candidate_indexes:
        max_candidate_length = min(
            len(aligned_tokens) - candidate_index,
            len(line_tokens) + max_length_slop,
        )
        if max_candidate_length < min_candidate_length:
            continue

        for candidate_length in range(min_candidate_length, max_candidate_length + 1):
            candidate = _word_match_candidate_for_span(
                line_tokens,
                aligned_tokens,
                candidate_index=candidate_index,
                candidate_length=candidate_length,
            )
            if candidate is None:
                continue
            if best_candidate is None or _word_match_candidate_key(candidate) < _word_match_candidate_key(best_candidate):
                best_candidate = candidate

    return best_candidate


def _word_match_candidate_key(candidate: _WordMatchCandidate) -> tuple[float, int, int, int]:
    return (
        candidate.normalized_cost,
        -candidate.anchor_match_count,
        -candidate.exact_match_count,
        candidate.next_search_index,
    )


def _word_match_candidate_for_span(
    line_tokens: Sequence[str],
    aligned_tokens: Sequence[tuple[str, float | None, float | None]],
    *,
    candidate_index: int,
    candidate_length: int,
) -> _WordMatchCandidate | None:
    span = aligned_tokens[candidate_index: candidate_index + candidate_length]
    span_tokens = [token for token, _, _ in span]
    max_normalized_cost = 0.35
    aligned_pairs, cost = _align_token_sequences(
        line_tokens,
        span_tokens,
        max_cost=max_normalized_cost * max(len(line_tokens), 1),
    )
    if not aligned_pairs:
        return None

    exact_match_count = sum(
        1
        for line_index, span_index in aligned_pairs
        if line_tokens[line_index] == span_tokens[span_index]
    )
    first_pair = aligned_pairs[0]
    last_pair = aligned_pairs[-1]
    anchor_match_count = 0
    if (
        first_pair[0] == 0
        and line_tokens[first_pair[0]] == span_tokens[first_pair[1]]
    ):
        anchor_match_count += 1
    if (
        last_pair[0] == len(line_tokens) - 1
        and line_tokens[last_pair[0]] == span_tokens[last_pair[1]]
    ):
        anchor_match_count += 1

    boundary_penalty = 0.0
    if first_pair[0] != 0:
        boundary_penalty += 0.75
    if last_pair[0] != len(line_tokens) - 1:
        boundary_penalty += 0.75
    if anchor_match_count == 0:
        boundary_penalty += 0.5

    return _WordMatchCandidate(
        start_time=span[first_pair[1]][1],
        end_time=span[last_pair[1]][2],
        next_search_index=candidate_index + last_pair[1] + 1,
        normalized_cost=(cost + boundary_penalty) / max(len(line_tokens), 1),
        exact_match_count=exact_match_count,
        anchor_match_count=anchor_match_count,
        token_pairs=tuple((line_index, candidate_index + span_index)
                          for line_index, span_index in aligned_pairs
                          if line_tokens[line_index] == span_tokens[span_index]),
    )


def _align_token_sequences(
    source_tokens: Sequence[str],
    target_tokens: Sequence[str],
    *,
    max_cost: float | None = None,
) -> tuple[list[tuple[int, int]], float]:
    substitution_cost = 1.25
    insertion_cost = 1.0
    deletion_cost = 1.0
    rows = len(source_tokens) + 1
    cols = len(target_tokens) + 1
    previous_row = [0.0] * cols
    current_row = [0.0] * cols
    steps: list[list[str | None]] = [[None] * cols for _ in range(rows)]

    for col in range(1, cols):
        previous_row[col] = col * insertion_cost
        steps[0][col] = "left"

    for row in range(1, rows):
        current_row[0] = row * deletion_cost
        steps[row][0] = "up"
        row_min_cost = current_row[0]
        for col in range(1, cols):
            diagonal_cost = previous_row[col - 1]
            if source_tokens[row - 1] != target_tokens[col - 1]:
                diagonal_cost += substitution_cost
            up_cost = previous_row[col] + deletion_cost
            left_cost = current_row[col - 1] + insertion_cost
            best_cost = diagonal_cost
            best_step = "diag"
            if up_cost < best_cost:
                best_cost = up_cost
                best_step = "up"
            if left_cost < best_cost:
                best_cost = left_cost
                best_step = "left"
            current_row[col] = best_cost
            steps[row][col] = best_step
            if best_cost < row_min_cost:
                row_min_cost = best_cost
        if max_cost is not None and row_min_cost > max_cost:
            return [], float("inf")
        previous_row, current_row = current_row, previous_row

    aligned_pairs: list[tuple[int, int]] = []
    row = len(source_tokens)
    col = len(target_tokens)
    while row > 0 or col > 0:
        step = steps[row][col]
        if step == "diag":
            aligned_pairs.append((row - 1, col - 1))
            row -= 1
            col -= 1
            continue
        if step == "up":
            row -= 1
            continue
        if step == "left":
            col -= 1
            continue
        break
    aligned_pairs.reverse()
    return aligned_pairs, previous_row[-1]


def _candidate_start_indexes(
    line_tokens: Sequence[str],
    aligned_tokens: Sequence[tuple[str, float | None, float | None]],
    aligned_token_index: dict[str, tuple[int, ...]],
    *,
    start_index: int,
) -> list[int]:
    if start_index >= len(aligned_tokens):
        return []

    max_length_slop = max(2, len(line_tokens) // 3)
    candidate_starts: set[int] = {start_index}
    for line_anchor_index, aligned_positions in _line_anchor_candidates(
        line_tokens,
        aligned_token_index,
        start_index=start_index,
    ):
        for aligned_position in aligned_positions:
            candidate_index = aligned_position - line_anchor_index
            if candidate_index < start_index or candidate_index >= len(aligned_tokens):
                continue
            candidate_starts.add(candidate_index)
            for delta in range(1, max_length_slop + 1):
                if candidate_index - delta >= start_index:
                    candidate_starts.add(candidate_index - delta)
                if candidate_index + delta < len(aligned_tokens):
                    candidate_starts.add(candidate_index + delta)
        if len(candidate_starts) > 1:
            break

    return sorted(candidate_starts)


def _line_anchor_candidates(
    line_tokens: Sequence[str],
    aligned_token_index: dict[str, tuple[int, ...]],
    *,
    start_index: int,
) -> list[tuple[int, tuple[int, ...]]]:
    candidates: list[tuple[int, tuple[int, ...]]] = []
    seen_indexes: set[int] = set()
    token_counts: dict[str, int] = {}
    for token in line_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    for line_index, token in enumerate(line_tokens):
        if line_index in seen_indexes:
            continue
        positions = aligned_token_index.get(token)
        if not positions:
            continue
        filtered_positions = tuple(position for position in positions if position >= start_index)
        if not filtered_positions:
            continue
        candidates.append((line_index, filtered_positions))
        seen_indexes.add(line_index)

    def anchor_sort_key(item: tuple[int, tuple[int, ...]]) -> tuple[int, int, int]:
        line_index, positions = item
        token = line_tokens[line_index]
        anchor_distance = min(line_index, len(line_tokens) - 1 - line_index)
        return (len(positions), token_counts[token], anchor_distance)

    return sorted(candidates, key=anchor_sort_key)


def _line_begins_with_clause(
    line_tokens: Sequence[str],
    clause: AlignedClause,
) -> bool:
    clause_tokens = _normalized_tokens(clause.text)
    if not clause_tokens or len(clause_tokens) > len(line_tokens):
        return False
    return tuple(line_tokens[: len(clause_tokens)]) == clause_tokens


def _line_ends_with_clause(
    line_tokens: Sequence[str],
    clause: AlignedClause,
) -> bool:
    clause_tokens = _normalized_tokens(clause.text)
    if not clause_tokens or len(clause_tokens) > len(line_tokens):
        return False
    return tuple(line_tokens[-len(clause_tokens) :]) == clause_tokens


@lru_cache(maxsize=8192)
def _normalized_tokens(text: str) -> tuple[str, ...]:
    """Match spoken digits and ASR numeric tokens without altering authored speech.

    Numeric groups split into digits; decimal points survive only between
    digits, while the spoken word point maps to the same token. Hyphens and
    sentence punctuation separate tokens. Expanded ASR tokens keep their
    source word's timestamps in ``_aligned_word_tokens``.
    """
    return tuple(token for token, _, _ in _number_normalized_tokens(
        [token.lower() for token in _TOKEN_RE.findall(normalize_text_punctuation(text))]))


def _number_normalized_tokens(tokens):
    """Expand cardinal groups into digits, retaining their source token range.

    Callsigns may mix groups (eleven seventy three) and individual digits.
    Tens consume a following unit; scales consume their cardinal prefix.
    Separate individual digit words remain separate, as in two nine zero nine.
    """
    def cardinal(index):
        token = tokens[index]
        if token in _TENS:
            value = _TENS[token]
            end = index + 1
            if end < len(tokens) and tokens[end] in _NUMBER_TOKENS and tokens[end] != 'point':
                value += int(_NUMBER_TOKENS[tokens[end]])
                end += 1
            return value, end
        if token in _CARDINALS:
            return _CARDINALS[token], index + 1
        if token in _NUMBER_TOKENS and token != 'point':
            return int(_NUMBER_TOKENS[token]), index + 1
        return None, index + 1

    index = 0
    while index < len(tokens):
        first = index
        value, end = cardinal(index)
        if value is None:
            yield _NUMBER_TOKENS.get(tokens[index], tokens[index]), index, index
            index += 1
            continue
        if end < len(tokens) and tokens[end] == 'hundred':
            value *= 100
            end += 1
            if end < len(tokens):
                remainder, after = cardinal(end)
                if remainder is not None:
                    value += remainder
                    end = after
        if end < len(tokens) and tokens[end] == 'thousand':
            value *= 1000
            end += 1
        if tokens[first] in _TENS and end == first + 2 and tokens[first + 1] in _NUMBER_TOKENS:
            # Keep the two actual word boundaries available for inner marks.
            yield str(value)[0], first, first
            yield str(value)[1], first + 1, first + 1
        else:
            for digit in str(value):
                yield digit, first, end - 1
        index = end


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


def cast_float(value: float | None) -> float:
    if value is None or math.isnan(value):
        return 0.0
    return float(value)


def _debug_line_preview(text: str) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= 60:
        return repr(normalized)
    return f"{normalized[:30]!r} ... {normalized[-30:]!r}"


def _content_debug_label(content: ScriptEvent) -> str:
    if isinstance(content, DialogueLine):
        return _debug_line_preview(content.spoken_text)
    if isinstance(content, ScriptGap):
        return content.label or "script gap"
    return repr(content.audio_plan)


def _debug_transcript_label(transcript: str) -> str:
    first_line = next(
        (line.strip() for line in transcript.splitlines() if line.strip()),
        "empty-script",
    )
    return first_line[:40] or "empty-script"


def _sanitize_debug_label(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return sanitized or "alignment"


def validate_mark_offsets(line):
    """Offsets refer to original Python-string boundaries, not normalized tokens."""
    for offset in line.mark_offsets:
        if not isinstance(offset, int) or not 0 <= offset <= len(line.spoken_text):
            message = f"Dialogue mark offset {offset!r} is outside spoken text (length {len(line.spoken_text)})"
            if line.node is not None:
                raise line.node.error(message)
            raise ValueError(message)


def _tokens_with_character_spans(text):
    """Preserve original spans while using the existing punctuation/number rules.

    Punctuation normalization is performed on one character at a time, carrying
    its original extent through replacements/expansion. Word-number normalization
    changes token spelling but never its authored offsets.
    """
    normalized = []
    origins = []
    for index, char in enumerate(text):
        replacement = normalize_text_punctuation(char)
        normalized.append(replacement)
        origins.extend([index] * len(replacement))
    joined = "".join(normalized)
    matches = list(_TOKEN_RE.finditer(joined))
    return [(token, origins[matches[first].start()], origins[matches[last].end() - 1] + 1)
            for token, first, last in _number_normalized_tokens(
                [match.group().lower() for match in matches])]


def script_timing_from_alignment(contents, alignment):
    """Match intact lines first; requested marks never rescore or segment a line.

    Correspondence is an extension of the same accepted full-line candidate.
    Missing internal words affect only their adjacent mark sides. Pure mark
    refinement has no access to mutable line search cursors.
    """
    from ..rendering import DialogueMarkTiming
    lines = [event for event in contents if isinstance(event, DialogueLine)]
    mappings = []
    spans = _line_spans_from_alignment(
        lines, alignment,
        allow_positional_clauses=not any(isinstance(event, ScriptGap) for event in contents),
        correspondences=mappings,
    )
    audio_tokens = _aligned_word_tokens(alignment.words or ())
    timings = []
    for line, (start, end), pairs in zip(lines, spans, mappings, strict=True):
        validate_mark_offsets(line)
        tokens = _tokens_with_character_spans(line.spoken_text)
        correspondence = dict(pairs)
        marks = []
        for offset in line.mark_offsets:
            if any(beg < offset < last for _, beg, last in tokens):
                marks.append(DialogueMarkTiming(None, None))
                continue
            before = next((i for i in range(len(tokens) - 1, -1, -1) if tokens[i][2] <= offset), None)
            after = next((i for i, (_, beg, _) in enumerate(tokens) if beg >= offset), None)
            previous_end = audio_tokens[correspondence[before]][2] if before in correspondence else None
            next_start = audio_tokens[correspondence[after]][1] if after in correspondence else None
            if offset == 0:
                next_start = start
            if offset == len(line.spoken_text):
                previous_end = end
            # Exact clauses provide independent boundary evidence, in priority
            # order. Resolve each side separately; failure cannot alter the line.
            prefix = tuple(token for token, _, last in tokens if last <= offset)
            suffix = tuple(token for token, beg, _ in tokens if beg >= offset)
            for side, target in (("previous", prefix), ("next", suffix)):
                if not target:
                    continue
                for clauses in (alignment.preferred_clauses, alignment.clauses):
                    match = _find_exact_clause_span(target, clauses, start_index=0, earliest=start or 0.0)
                    if match is not None:
                        if side == "previous":
                            previous_end = match[1]
                        else:
                            next_start = match[0]
                        break
            marks.append(DialogueMarkTiming(previous_end, next_start))
        timings.append(DialogueLineTiming(
            math.nan if start is None else start,
            math.nan if end is None else end,
            tuple(marks),
        ))
    return ScriptTiming(tuple(timings))
