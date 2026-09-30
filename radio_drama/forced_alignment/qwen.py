"""Native Transformers Qwen ASR and alignment; no legacy qwen_asr dependency."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from threading import RLock
import math

import numpy as np

from ..audio import resample_audio
from ..debug import write_debug_json
from ..model_loading import shared_model_load
from .base import AlignmentResult, ForcedAlignmentResource, TranscriptionResult, WordTiming


_SAMPLE_RATE = 16000
_WINDOW_SECONDS = 180
_OVERLAP_SECONDS = 10
_MAX_NEW_TOKENS = 8192
_SEAM_TOLERANCE = 0.4
_COMPLETE_PREFIX_ATTEMPTS = 4
_COMPLETE_OVERLAPS = (10, 20, 30)


class AlignmentWindowError(RuntimeError):
    """A window seam cannot be reconciled without dropping source speech."""


@dataclass(frozen=True, slots=True)
class AudioWindow:
    start_frame: int
    end_frame: int

    @property
    def offset(self):
        return self.start_frame / _SAMPLE_RATE


def audio_windows(frame_count, *, sample_rate=_SAMPLE_RATE,
                  window_seconds=_WINDOW_SECONDS, overlap_seconds=_OVERLAP_SECONDS):
    """Cover the input with overlapping, bounded windows, including the tail."""
    window = round(window_seconds * sample_rate)
    overlap = round(overlap_seconds * sample_rate)
    if not 0 <= overlap < window:
        raise ValueError("Overlap must be smaller than the positive window size")
    result = []
    start = 0
    while start < frame_count:
        end = min(start + window, frame_count)
        result.append(AudioWindow(start, end))
        if end == frame_count:
            break
        start = end - overlap
    return result


def _mono_audio(audio, sample_rate):
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    return resample_audio(mono, input_sample_rate=sample_rate, output_sample_rate=_SAMPLE_RATE)


def decoded_words(items, duration, *, offset=0.0):
    """Validate model boundaries without sorting the source transcript.

    The native processor already repairs timestamp order. Small final rounding
    overhang (one 80ms timestamp quantum) is clamped; larger/invalid values remain
    unknown. Offset addition happens only after validation in local coordinates.
    """
    def boundary(value):
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value) or value < 0 or value > duration + .08:
            return None
        return min(value, duration) + offset
    words = []
    for item in items:
        start, end = boundary(item["start_time"]), boundary(item["end_time"])
        if start is not None and end is not None and end < start:
            start = end = None
        words.append(WordTiming(item["text"], start, end))
    return tuple(words)


def merge_window_words(left, right, *, overlap_start, overlap_end):
    """Splice in a matching overlap run using text AND absolute timestamps.

    Choose the anchor nearest the physical seam center. Timestamp agreement
    distinguishes genuine repeated phrases; only records inside the overlap
    participate. Keep one window's measurements on each side, never averages.
    """
    from .projection import _normalized_tokens
    def candidates(words):
        return [(i, word) for i, word in enumerate(words)
                if word.start is not None and word.end is not None
                and overlap_start <= word.start <= overlap_end
                and _normalized_tokens(word.text)]
    a, b = candidates(left), candidates(right)
    anchors = []
    for ai, (li, lw) in enumerate(a):
        for bi, (ri, rw) in enumerate(b):
            if (_normalized_tokens(lw.text) != _normalized_tokens(rw.text)
                    or abs(lw.start - rw.start) > _SEAM_TOLERANCE
                    or abs(lw.end - rw.end) > _SEAM_TOLERANCE):
                continue
            # Require neighboring agreement to avoid a lone repeated stop word.
            run = 1
            while ai + run < len(a) and bi + run < len(b):
                x, y = a[ai + run][1], b[bi + run][1]
                if (_normalized_tokens(x.text) != _normalized_tokens(y.text)
                        or abs(x.start - y.start) > _SEAM_TOLERANCE
                        or abs(x.end - y.end) > _SEAM_TOLERANCE):
                    break
                run += 1
            if run >= 2:
                center = (overlap_start + overlap_end) / 2
                anchors.extend((abs((a[ai + j][1].start + a[ai + j][1].end) / 2 - center),
                                a[ai + j][0], b[bi + j][0]) for j in range(run))
    if not anchors:
        # A truly silent overlap needs no lexical anchor, but neither window
        # may contain an unresolved or crossing word in that interval.
        if (all(w.end is not None and w.end <= overlap_start for w in left)
                and all(w.start is not None and w.start >= overlap_end for w in right)):
            return tuple(left) + tuple(right)
        raise AlignmentWindowError(f"No reliable word correspondence at {overlap_start:.3f}–{overlap_end:.3f}s")
    _, li, ri = min(anchors)
    return tuple(left[:li + 1]) + tuple(right[ri + 1:])


class QwenAlignmentResource(ForcedAlignmentResource):
    """One selected Qwen model family, with serialized calls and batched windows.

    Models are independently lazy: complete bounded transcript alignment never
    loads ASR. Recording ASR and alignment windows share source coordinates.
    Heavyweight imports and model downloads occur only on actual inference.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._inference_lock = RLock()
        self._asr = self._asr_processor = None
        self._aligner = self._aligner_processor = None
        self._debug_index = 0

    @property
    def alignment_identity(self):
        return (f"qwen:{self.config.qwen_asr_model}:{self.config.qwen_alignment_model}:"
                f"windows-{_WINDOW_SECONDS}-{_OVERLAP_SECONDS}:tokens-{_MAX_NEW_TOKENS}:complete-greedy-v2")

    @property
    def transcription_identity(self):
        return f"qwen:{self.config.qwen_asr_model}:windows-{_WINDOW_SECONDS}-{_OVERLAP_SECONDS}:v1"

    def _load(self, *, aligner):
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration, Qwen3ASRForTokenClassification
        model_id = self.config.qwen_alignment_model if aligner else self.config.qwen_asr_model
        model_type = Qwen3ASRForTokenClassification if aligner else Qwen3ASRForConditionalGeneration
        device = self.config.resolved_device
        dtype = torch.float32 if device == "cpu" else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        with shared_model_load():
            processor = AutoProcessor.from_pretrained(model_id)
            model = model_type.from_pretrained(model_id, dtype=dtype)
            model = model.to(device).eval()
        return processor, model

    def _ensure_asr(self):
        if self._asr is None:
            self._asr_processor, self._asr = self._load(aligner=False)
        return self._asr_processor, self._asr

    def _ensure_aligner(self):
        if self._aligner is None:
            self._aligner_processor, self._aligner = self._load(aligner=True)
        return self._aligner_processor, self._aligner

    def _transcribe_batch(self, audios, languages):
        import torch
        processor, model = self._ensure_asr()
        inputs = processor.apply_transcription_request(
            audio=audios, language=languages,
            processor_kwargs={"sampling_rate": _SAMPLE_RATE, "padding": True},
        ).to(model.device, model.dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=_MAX_NEW_TOKENS)
        generated = output[:, inputs["input_ids"].shape[1]:]
        eos = model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or ())
        if generated.shape[1] >= _MAX_NEW_TOKENS:
            for row in generated:
                if not any(int(token) in eos_ids for token in row):
                    raise RuntimeError("Qwen ASR token budget exhausted; reduce window duration")
        parsed = [processor.decode(row, return_format="parsed") for row in generated]
        return [TranscriptionResult(item["transcription"], item["language"] or language)
                for item, language in zip(parsed, languages, strict=True)]

    def _align_batch(self, audios, transcripts, languages):
        import torch
        processor, model = self._ensure_aligner()
        inputs, word_lists = processor.prepare_forced_aligner_inputs(
            audio=audios, transcript=transcripts, language=languages,
            processor_kwargs={"sampling_rate": _SAMPLE_RATE, "padding": True},
        )
        inputs = inputs.to(model.device, model.dtype)
        with torch.inference_mode():
            output = model(**inputs)
        return processor.decode_forced_alignment(
            logits=output.logits, input_ids=inputs["input_ids"], word_lists=word_lists,
            timestamp_token_id=model.config.timestamp_token_id,
        )

    async def _process_batch(self, requests):
        return await asyncio.to_thread(self._process_batch_sync, requests)

    def _alignment_units(self, transcript, language):
        """Obtain stable units through the processor's public language handling.

        Preparing one second of silence obtains word lists without a model call.
        This avoids duplicating language-specific segmentation or assuming ISO
        language codes are accepted by split_words_for_alignment itself.
        """
        processor, _ = self._ensure_aligner()
        _, lists = processor.prepare_forced_aligner_inputs(
            audio=[np.zeros(_SAMPLE_RATE, dtype=np.float32)], transcript=[transcript],
            language=[language], processor_kwargs={"sampling_rate": _SAMPLE_RATE, "padding": True})
        return tuple(lists[0])

    def _complete_window(self, audio, units, frame, cursor, language):
        """Align a provisional prefix; a rate estimate only chooses input text."""
        remaining_frames = len(audio) - frame
        final = remaining_frames <= _WINDOW_SECONDS * _SAMPLE_RATE
        clip = audio[frame:frame + _WINDOW_SECONDS * _SAMPLE_RATE]
        duration = len(clip) / _SAMPLE_RATE
        remaining_words = len(units) - cursor
        count = remaining_words if final else min(remaining_words, max(
            20, round(remaining_words / (remaining_frames / _SAMPLE_RATE) * duration * 1.3) + 12))
        for _ in range(_COMPLETE_PREFIX_ATTEMPTS):
            items = self._align_batch([clip], [" ".join(units[cursor:cursor + count])], [language])[0]
            if tuple(item["text"] for item in items) != units[cursor:cursor + count]:
                raise AlignmentWindowError("Qwen changed complete-transcript unit identities during alignment")
            if final:
                return decoded_words(items, duration, offset=frame / _SAMPLE_RATE), True
            limit = duration - _OVERLAP_SECONDS
            good = 0
            for item in items:
                start, end = item["start_time"], item["end_time"]
                if (start is None or end is None or not math.isfinite(start) or not math.isfinite(end)
                        or not 0 <= start <= end <= limit):
                    break
                good += 1
            if good < count or count == remaining_words:
                if not good:
                    break
                return decoded_words(items[:good], duration, offset=frame / _SAMPLE_RATE), False
            count = min(remaining_words, max(count + 20, round(count * 1.5)))
        raise AlignmentWindowError(
            f"Cannot establish complete-transcript progress at {frame / _SAMPLE_RATE:.3f}s, word {cursor}; "
            "ASR fallback is disabled")

    @staticmethod
    def _complete_overlap_agrees(previous, measured, cursor):
        """Require broad agreement and neighboring anchors by source-unit identity.

        A few incorrect supplied words can move locally without invalidating an
        otherwise consistent seam. Require 80 percent of comparable boundaries
        within tolerance plus two adjacent units whose complete spans agree.
        This validates placement, not whether each supplied word was spoken.
        """
        errors = []
        consecutive = longest = 0
        for index, new in enumerate(measured, cursor):
            if index >= len(previous):
                break
            old = previous[index]
            word_errors = []
            for before, after in ((old.start, new.start), (old.end, new.end)):
                if before is not None and after is not None:
                    word_errors.append(abs(before - after))
            errors.extend(word_errors)
            consecutive = (consecutive + 1 if len(word_errors) == 2
                           and max(word_errors) <= _SEAM_TOLERANCE else 0)
            longest = max(longest, consecutive)
        return (len(errors) >= 4 and longest >= 2
                and sum(error <= _SEAM_TOLERANCE for error in errors) / len(errors) >= .8)

    def _align_complete_sync(self, audio, transcript, language):
        """Advance with a provisional tail and validate overlap before committing.

        The contiguous prefix before the last ten seconds advances the cursor.
        Retries back up both source-unit identity and measured audio time, using
        successively wider overlap. The final window retains every remaining
        unit, including unknown boundaries. Plausible predictions are not proof
        of transcript correctness: a forced aligner can timestamp unspoken words.
        """
        units = self._alignment_units(transcript, language)
        if not units:
            return ()
        accepted = []
        frame = cursor = 0
        while True:
            if accepted:
                attempts = []
                last_end = next((word.end for word in reversed(accepted) if word.end is not None), None)
                for overlap in _COMPLETE_OVERLAPS:
                    target = last_end - overlap
                    back = next((i for i, word in enumerate(accepted)
                                 if word.start is not None and word.start >= target), None)
                    if back is not None and back > cursor:
                        next_frame = round(max(0., accepted[back].start - .3) * _SAMPLE_RATE)
                        if next_frame > frame:
                            attempts.append((next_frame, back))
                if not attempts:
                    raise AlignmentWindowError(
                        f"Insufficient complete-transcript progress at {frame / _SAMPLE_RATE:.3f}s, word {cursor}; "
                        "ASR fallback is disabled")
            else:
                attempts = [(0, 0)]
            failure = None
            for next_frame, next_cursor in attempts:
                try:
                    measured, final = self._complete_window(audio, units, next_frame, next_cursor, language)
                except AlignmentWindowError as exc:
                    failure = exc
                    continue
                if accepted and not self._complete_overlap_agrees(accepted, measured, next_cursor):
                    failure = AlignmentWindowError(
                        f"Conflicting complete-transcript overlap at {next_frame / _SAMPLE_RATE:.3f}s, "
                        f"word {next_cursor}; ASR fallback is disabled")
                    continue
                if next_cursor + len(measured) <= len(accepted) and not final:
                    failure = AlignmentWindowError("Complete-transcript window did not advance")
                    continue
                # Keep the previously accepted measurements through the seam.
                accepted.extend(measured[len(accepted) - next_cursor:])
                frame, cursor = next_frame, next_cursor
                if final or len(accepted) == len(units):
                    if len(accepted) != len(units):
                        raise AlignmentWindowError("Final complete-transcript window lost source units")
                    return tuple(accepted)
                break
            else:
                raise failure

    def _process_batch_sync(self, requests):
        with self._inference_lock:
            jobs = []
            complete_results = {}
            by_request = [[] for _ in requests]
            for index, request in enumerate(requests):
                audio = _mono_audio(request.audio, request.sample_rate)
                windows = audio_windows(len(audio))
                if request.transcript_kind == "complete" and len(windows) > 1:
                    complete_results[index] = self._align_complete_sync(audio, request.transcript, request.language)
                    continue
                for window in windows:
                    by_request[index].append(len(jobs))
                    jobs.append((index, window, audio[window.start_frame:window.end_frame]))
            transcripts = [requests[index].transcript for index, _, _ in jobs]
            for begin in range(0, len(jobs), self.config.resolved_batch_size):
                selected = [i for i in range(begin, min(begin + self.config.resolved_batch_size, len(jobs)))
                            if requests[jobs[i][0]].transcript_kind == "partial"]
                if selected:
                    recognized = self._transcribe_batch([jobs[i][2] for i in selected],
                                                       [requests[jobs[i][0]].language for i in selected])
                    for i, result in zip(selected, recognized, strict=True):
                        transcripts[i] = result.text
            results = [() for _ in jobs]
            for begin in range(0, len(jobs), self.config.resolved_batch_size):
                selected = [i for i in range(begin, min(begin + self.config.resolved_batch_size, len(jobs)))
                            if transcripts[i].strip()]
                if selected:
                    decoded = self._align_batch([jobs[i][2] for i in selected], [transcripts[i] for i in selected],
                                                [requests[jobs[i][0]].language for i in selected])
                    for i, items in zip(selected, decoded, strict=True):
                        results[i] = decoded_words(items, len(jobs[i][2]) / _SAMPLE_RATE,
                                                   offset=jobs[i][1].offset)
            output = []
            for request_index, indices in enumerate(by_request):
                merged = complete_results.get(request_index, ())
                previous_window = None
                for i in indices:
                    _, window, _ = jobs[i]
                    if previous_window is None:
                        merged = results[i]
                    else:
                        try:
                            merged = merge_window_words(merged, results[i], overlap_start=window.offset,
                                                        overlap_end=previous_window.end_frame / _SAMPLE_RATE)
                        except AlignmentWindowError:
                            merged = self._repair_seam(
                                merged, results[i], _mono_audio(requests[request_index].audio,
                                                              requests[request_index].sample_rate),
                                window.offset, previous_window.end_frame / _SAMPLE_RATE,
                                requests[request_index].language,
                            )
                    previous_window = window
                source_text = (requests[request_index].transcript
                               if requests[request_index].transcript_kind == "complete"
                               else " ".join(w.text for w in merged))
                result = AlignmentResult(merged, (), source_text=source_text,
                                         language=requests[request_index].language)
                output.append(result)
                write_debug_json(self.config, "qwen_alignment", f"{self._debug_index:03d}.json",
                                 {"identity": self.alignment_identity, "result": asdict(result)})
                self._debug_index += 1
            return output

    def _repair_seam(self, left, right, audio, start, end, language):
        center = (start + end) / 2
        for extent in (30, 60, 90):
            beg = max(0, round((center - extent) * _SAMPLE_RATE))
            last = min(len(audio), round((center + extent) * _SAMPLE_RATE))
            clip = audio[beg:last]
            text = self._transcribe_batch([clip], [language])[0].text
            if not text.strip():
                continue
            items = self._align_batch([clip], [text], [language])[0]
            bridge = decoded_words(items, len(clip) / _SAMPLE_RATE, offset=beg / _SAMPLE_RATE)
            try:
                first = merge_window_words(left, bridge, overlap_start=beg / _SAMPLE_RATE, overlap_end=end)
                return merge_window_words(first, right, overlap_start=start, overlap_end=last / _SAMPLE_RATE)
            except AlignmentWindowError:
                continue
        raise AlignmentWindowError(f"Failed to reconcile seam {start:.3f}–{end:.3f}s after three bridge attempts")

    def transcribe_sync(self, audio, sample_rate, *, language="en"):
        with self._inference_lock:
            mono = _mono_audio(audio, sample_rate)
            windows = audio_windows(len(mono))
            if not windows:
                return TranscriptionResult("", language)
            if len(windows) > 1:
                # Timed overlap evidence is required to deduplicate long ASR.
                # The shared pipeline transcribes each window exactly once.
                from .base import ForcedAlignmentRequest
                result = self._process_batch_sync([ForcedAlignmentRequest(
                    mono, _SAMPLE_RATE, "", "partial", language=language)])[0]
                return TranscriptionResult(result.source_text, result.language)
            texts = []
            for begin in range(0, len(windows), self.config.resolved_batch_size):
                batch = windows[begin:begin + self.config.resolved_batch_size]
                texts.extend(self._transcribe_batch([mono[w.start_frame:w.end_frame] for w in batch],
                                                    [language] * len(batch)))
            return texts[0]
