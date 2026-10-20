import json
from types import SimpleNamespace

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList, UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from radio_drama.vibevoice_diagnostics import VibeVoiceTrace


def make_trace(processors=(), batch_size=2):
    return VibeVoiceTrace(
        SimpleNamespace(audio_bos_token_id=1, audio_eos_token_id=2,
                        audio_token_id=3, eos_token_id=4),
        SimpleNamespace(guidance_scale=1.2, num_diffusion_steps=8),
        processors, batch_size, False,
    )


def test_trace_observes_actual_guidance_without_extra_calls_or_score_changes():
    class Model:
        calls = 0

        def __call__(self, ids, **kwargs):
            self.calls += 1
            logits = torch.full((1, 1, 8), -10.)
            logits[0, 0, 3] = 4.
            logits[0, 0, 4] = 0.
            return CausalLMOutputWithPast(logits=logits)

    model = Model()
    processor = UnbatchedClassifierFreeGuidanceLogitsProcessor(1.2, model)
    trace = make_trace([processor], batch_size=1)
    logits = torch.full((1, 8), -10.)
    logits[0, 3:5] = torch.tensor([2., 1.9])
    ids = torch.tensor([[1]])
    scores = trace.processors(ids, logits.clone())
    expected = LogitsProcessorList([
        UnbatchedClassifierFreeGuidanceLogitsProcessor(1.2, Model())
    ])(ids, logits.clone())
    assert torch.equal(scores, expected)
    assert model.calls == 1
    trace.record(0, torch.tensor([1]), scores.argmax(-1), [0])
    record = trace.rows[0][0]
    assert record['allowed_argmax']['before_token_guidance'] == 3
    assert record['allowed_argmax']['after_token_guidance'] == 4
    assert record['guidance_changed_argmax'] is True
    assert record['selected_token_id'] == 4


def test_trace_offsets_follow_each_rows_audio_and_ignore_finished_rows():
    trace = make_trace()
    scores = torch.zeros(2, 8)
    scores[:, 1] = -float('inf')
    for step, (active, tokens, frames) in enumerate([
        ([1, 1], [3, 3], [3200, 3200]),
        ([1, 1], [2, 3], [0, 3200]),
        ([1, 1], [4, 3], [0, 3200]),
        ([0, 1], [0, 4], [0, 0]),
    ]):
        trace.processors(torch.ones(2, 1, dtype=torch.long), scores)
        trace.record(step, torch.tensor(active), torch.tensor(tokens), frames)
    assert len(trace.rows[0]) == 3
    assert [r['audio_start_sample'] for r in trace.rows[0]] == [0, 3200, 3200]
    assert trace.frames == [3200, 9600]
    data = [json.loads(line) for line in trace.jsonl(1, sample_rate=24000, model_name='test')]
    assert data[0]['batch_row'] == 1
    assert data[0]['frame_count'] == 9600
    assert data[-1]['audio_start_seconds'] == 0.4
    assert data[-1]['scores']['raw_logits'][0] == '-inf'
    assert data[-1]['guidance_changed_argmax'] is False
