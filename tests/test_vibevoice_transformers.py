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
