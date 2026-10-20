"""Compact per-row observations of generation; never run a processor twice.

Only the four allowed control-token scores are retained. The taps surround the
actual token-guidance processor, so its private KV cache advances normally and
the recorded scores include any preceding processors. Audio offsets count decoded
samples, not LM steps: batch rows may pause or finish independently.
"""
from __future__ import annotations

import json
import math
from importlib.metadata import version

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)


class _ScoreTap:
    def __init__(self, trace, name):
        self.trace = trace
        self.name = name

    def __call__(self, input_ids, scores):
        self.trace.scores[self.name] = scores[:, self.trace.token_ids].detach().clone()
        return scores


class VibeVoiceTrace:
    def __init__(self, config, generation_config, processors, batch_size, reset):
        self.tokens = {
            name: getattr(config, name + "_token_id")
            for name in ("audio_bos", "audio_eos", "audio", "eos")
        }
        self.token_ids = list(self.tokens.values())
        self.scores = {}
        self.rows = [[] for _ in range(batch_size)]
        self.frames = [0] * batch_size
        self.metadata = {
            "format_version": 1,
            "versions": {name: version(name) for name in ("torch", "transformers", "diffusers")},
            "tokens": self.tokens,
            "score_token_order": list(self.tokens),
            "guidance_scale": generation_config.guidance_scale,
            "num_diffusion_steps": generation_config.num_diffusion_steps,
            "reset_tokenizer_caches": reset,
            "batch_size": batch_size,
            "logits_processors": [type(p).__name__ for p in processors],
        }
        self.processors = LogitsProcessorList([_ScoreTap(self, "raw_logits")])
        for processor in processors:
            guidance = isinstance(processor, UnbatchedClassifierFreeGuidanceLogitsProcessor)
            if guidance:
                self.processors.append(_ScoreTap(self, "before_token_guidance"))
            self.processors.append(processor)
            if guidance:
                self.processors.append(_ScoreTap(self, "after_token_guidance"))
        self.processors.append(_ScoreTap(self, "final_scores"))

    def record(self, step, active, selected, chunk_frames):
        names = list(self.scores)
        # One transfer for compact scores rather than full-vocabulary histories.
        values = torch.stack([self.scores[name] for name in names], dim=1).float().cpu().tolist()
        for row, (live, token, frames) in enumerate(zip(
            active.tolist(), selected.tolist(), chunk_frames, strict=True
        )):
            if not live:
                continue
            scores = dict(zip(names, values[row], strict=True))
            choices = {name: self.token_ids[max(range(4), key=score.__getitem__)]
                       for name, score in scores.items()}
            self.rows[row].append({
                "step": step,
                "audio_start_sample": self.frames[row],
                "audio_end_sample": self.frames[row] + frames,
                "selected_token_id": token,
                "allowed_argmax": choices,
                "guidance_changed_argmax": (
                    choices["before_token_guidance"] != choices["after_token_guidance"]
                    if "before_token_guidance" in choices else False
                ),
                # JSON has no infinity literals. Preserve masked scores explicitly.
                "scores": {name: [v if math.isfinite(v) else str(v) for v in score]
                           for name, score in scores.items()},
            })
            self.frames[row] += frames

    def jsonl(self, row, *, sample_rate, model_name):
        """Times refer to the sibling native WAV, before production effects."""
        metadata = dict(self.metadata, batch_row=row, sample_rate=sample_rate,
                        model_name=model_name, frame_count=self.frames[row])
        yield json.dumps({"type": "metadata", **metadata}, allow_nan=False) + "\n"
        for record in self.rows[row]:
            yield json.dumps({
                "type": "step", **record,
                "audio_start_seconds": record["audio_start_sample"] / sample_rate,
                "audio_end_seconds": record["audio_end_sample"] / sample_rate,
            }, allow_nan=False) + "\n"
