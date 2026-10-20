"""Exercise native generation's batch bookkeeping without weights or inference."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('transformers.models.vibevoice', reason='Native VibeVoice requires modern Transformers')
from radio_drama.vibevoice_transformers import VibeVoiceWithBatchAudioFix


class TaggedGeneration:
    """Script token decisions while retaining the real generation loop."""
    def __init__(self):
        # Row zero finishes early. Row two pauses while row one continues.
        self.tokens = [(3, 3, 3), (4, 3, 4), (4, 4, 3), (9, 4, 4), (9, 9, 9)]
        self.step = 0
        self.config = SimpleNamespace(audio_bos_token_id=3, audio_eos_token_id=5,
                                      audio_token_id=4, is_encoder_decoder=False)
        self.model = SimpleNamespace(
            semantic_tokenizer_encoder=self.semantic,
            multi_modal_projector=lambda value: torch.zeros(len(value), 1, 2),
            semantic_connector=lambda value: value)
        self.decoded_masks = []

    def __call__(self, **kwargs):
        logits = torch.full((3, 1, 10), -100.)
        for row, token in enumerate(self.tokens[self.step]):
            logits[row, 0, token] = 100.
        return SimpleNamespace(logits=logits, last_hidden_state=torch.zeros(3, 1, 2))

    def _prefill(self, *args, **kwargs):
        return self()

    def _valid_auto_compile_criteria(self, *args):
        return False

    def _prepare_negative_generation(self, *args, **kwargs):
        return torch.zeros(3, 1, dtype=torch.long), {}

    def _has_unfinished_sequences(self, finished, *args, **kwargs):
        return not finished

    def _update_model_kwargs_for_generation(self, outputs, kwargs, **unused):
        self.step += 1
        return kwargs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}

    def _optimize_model_for_decode(self):
        return nullcontext()

    def get_input_embeddings(self):
        return lambda tokens: torch.zeros(len(tokens), 2)

    def _reset_negative_cache_for_audio_start(self, *args):
        pass

    def _step_negative_branch(self, mask, tokens, embeds, ids, kwargs, forward):
        return torch.zeros(int(mask.sum()), 2), ids, kwargs

    def _sample_audio_latent(self, positive, negative, *args):
        return torch.zeros(len(positive), 1, 2)

    def _decode_audio_latent(self, latent, mask, batch_size, cache):
        self.decoded_masks.append(mask.tolist())
        # Like the real decoder: output has ALL batch rows, not only active rows.
        return SimpleNamespace(audio=torch.tensor([[10.], [20.], [30.]]), padding_cache=None)

    def semantic(self, audio, **kwargs):
        assert audio.shape == (3, 1)  # Semantic feedback also needs original rows.
        return SimpleNamespace(latents=torch.zeros(3, 1, 2), padding_cache=None)


def test_native_generation_preserves_audio_row_when_batch_members_pause_or_finish():
    harness = TaggedGeneration()
    config = SimpleNamespace(_pad_token_tensor=torch.tensor(0), output_attentions=False,
                             output_hidden_states=False, output_scores=False, output_logits=False,
                             return_dict_in_generate=False, do_sample=False, compile_config=None,
                             is_assistant=False, noise_scheduler=None, monitor_progress=False,
                             num_diffusion_steps=1, guidance_scale=1., max_length=20)
    class Stop:
        eos_token_id = 9
        def __iter__(self):
            return iter([self])
        def __call__(self, ids, scores):
            return ids[:, -1] == self.eos_token_id
    clips = VibeVoiceWithBatchAudioFix._sample(
        harness, torch.ones(3, 1, dtype=torch.long), lambda ids, scores: scores, Stop(), config,
        attention_mask=torch.ones(3, 1, dtype=torch.long), use_cache=True)
    assert harness.decoded_masks == [[True, False, True], [True, True, False], [False, True, True]]
    assert [clip.tolist() for clip in clips] == [[10., 10.], [20., 20.], [30., 30.]]


@pytest.mark.parametrize('setting', [None, '0', '1'])
def test_native_generation_resets_only_ended_segment_rows(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv('VIBEVOICE_RESET', raising=False)
    else:
        monkeypatch.setenv('VIBEVOICE_RESET', setting)
    caches = [SimpleNamespace(layers={
        'conv': SimpleNamespace(is_initialized=True, cache=torch.ones(3, 2, 4))
    }) for _ in range(2)]

    class SegmentGeneration(TaggedGeneration):
        def __init__(self):
            super().__init__()
            # Row zero ends its segment while the other rows still emit audio.
            self.tokens[3] = (5, 4, 4)

        def __call__(self, **kwargs):
            if self.step == 4:
                for cache in caches:
                    expected = torch.zeros(2, 4) if setting == '1' else torch.ones(2, 4)
                    assert torch.equal(cache.layers['conv'].cache[0], expected)
                    assert torch.equal(cache.layers['conv'].cache[1:], torch.ones(2, 2, 4))
            return super().__call__(**kwargs)

        def _decode_audio_latent(self, *args):
            result = super()._decode_audio_latent(*args)
            caches[0].layers['conv'].cache.fill_(1)
            result.padding_cache = caches[0]
            return result

        def semantic(self, *args, **kwargs):
            result = super().semantic(*args, **kwargs)
            caches[1].layers['conv'].cache.fill_(1)
            result.padding_cache = caches[1]
            return result

    monkeypatch.setitem(globals(), 'TaggedGeneration', SegmentGeneration)
    test_native_generation_preserves_audio_row_when_batch_members_pause_or_finish()


def test_native_generation_rejects_invalid_reset_setting(monkeypatch):
    monkeypatch.setenv('VIBEVOICE_RESET', 'true')
    with pytest.raises(ValueError, match='VIBEVOICE_RESET must be 0 or 1'):
        test_native_generation_preserves_audio_row_when_batch_members_pause_or_finish()
