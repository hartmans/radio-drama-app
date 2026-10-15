"""Qwen punctuation preprocessing must agree with authored-line projection."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from carthage.dependency_injection import Injector

from radio_drama.config import ProductionConfig
from radio_drama.dialogue import DialogueLine, SpeakerVoiceReference
from radio_drama.forced_alignment import AlignmentResult
from radio_drama.forced_alignment.projection import script_timing_from_alignment
from radio_drama.forced_alignment.qwen import QwenAlignmentResource, decoded_words


@pytest.mark.parametrize('apostrophe', ["'", '‘', '’', '‚', '‛'])
def test_contractions_survive_qwen_tokenization_and_line_projection(apostrophe):
    Qwen3ASRProcessor = pytest.importorskip(
        'transformers.models.qwen3_asr.processing_qwen3_asr').Qwen3ASRProcessor
    text = f'Don{apostrophe}t stop, please don{apostrophe}t stop.'
    expected = ("Don't", 'stop', 'please', "don't", 'stop')

    class Inputs(dict):
        def to(self, *args):
            return self

    class Processor:
        def prepare_forced_aligner_inputs(self, **kwargs):
            # Exercise the installed tokenizer, without loading any models.
            words = [Qwen3ASRProcessor.split_words_for_alignment(None, text, 'English')
                     for text in kwargs['transcript']]
            return Inputs(input_ids=torch.ones((1, 1), dtype=torch.long)), words

        def decode_forced_alignment(self, **kwargs):
            return [[dict(text=word, start_time=float(i), end_time=i + .5)
                     for i, word in enumerate(row)] for row in kwargs['word_lists']]

    class Model:
        device, dtype = torch.device('cpu'), torch.float32
        config = SimpleNamespace(timestamp_token_id=42)

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.zeros(1, 1, 1))

    resource = QwenAlignmentResource(config=ProductionConfig(device='cpu'), injector=Injector())
    resource._aligner_processor, resource._aligner = Processor(), Model()
    try:
        # Long-window unit discovery and direct short-window alignment both use it.
        assert resource._alignment_units(text, 'en') == expected
        items = resource._align_batch([np.zeros(80000)], [text], ['en'])[0]
        assert tuple(item['text'] for item in items) == expected
        speaker = SpeakerVoiceReference('A', 'voice', Path('voice.wav'))
        line = DialogueLine(speaker, text, mark_offsets=(text.index('please'),))
        result = AlignmentResult(decoded_words(items, 5.), (), source_text=text)
        timing = script_timing_from_alignment([line], result).dialogue_lines[0]
        assert (timing.start, timing.end) == (0., 4.5)
        assert timing.marks[0].previous_end == 1.5
        assert timing.marks[0].next_start == 2.
        assert line.spoken_text == result.source_text == text
    finally:
        resource.close()
