# Copyright 2026 The Microsoft Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local corrections to Transformers 5.17 VibeVoice generation.

The generation loop is derived from transformers/models/vibevoice/generation_vibevoice.py.
Decoded audio is indexed by the original batch row, rather
than the position in the compact list of active rows. The decoder and semantic
encoder retain their full batch shape and per-row streaming caches. With
VIBEVOICE_RESET=1 (default 0), at audio EOS,
both tokenizer caches are zeroed for the ending rows, matching legacy VibeVoice's
segment reset behavior. Language model context and voice references are retained.
The resets follow decoding because the native decoder updates all batch rows,
including rows that ended a segment while another row produced audio.

Imported lazily by the live VibeVoice loader; no environment files are modified.
"""
from __future__ import annotations

import os
import torch
from transformers import VibeVoiceForConditionalGeneration
from transformers.generation.utils import ALL_CACHE_NAMES
from transformers.models.vibevoice.generation_vibevoice import VibeVoiceGenerateOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


class VibeVoiceWithBatchAudioFix(VibeVoiceForConditionalGeneration):
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> GenerateNonBeamOutput | torch.LongTensor:
        """
        This method overrides [~generation.utils.GenerationMixin._sample].
        To ease maintenance, modifications are marked with the comment "VibeVoice specific".
        """
        reset_setting = os.environ.get("VIBEVOICE_RESET", "0")
        if reset_setting not in ("0", "1"):
            raise ValueError("VIBEVOICE_RESET must be 0 or 1")
        reset_tokenizer_caches = reset_setting == "1"
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # `pad_token_id` is created on `inputs_tensor.device` in `_prepare_special_tokens`. For multimodal models
        # (e.g. BLIP-2, LLaVA) sharded across devices via `device_map="auto"`, `inputs_tensor` (e.g. `pixel_values`
        # on the vision encoder) and `input_ids` (on the language model) can live on different devices, so we need to
        # realign `pad_token_id` with `input_ids` to avoid cross-device ops below.
        if pad_token_id is not None:
            pad_token_id = pad_token_id.to(input_ids.device)

        model_forward = (
            self.get_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )

        prefill_consumed = False
        outputs = self._prefill(
            input_ids,
            generation_config,
            model_kwargs,
            is_first_iteration=not generation_config.is_assistant,
        )

        # *************** VibeVoice specific ***************
        noise_scheduler = generation_config.noise_scheduler
        monitor_progress = generation_config.monitor_progress
        num_diffusion_steps = generation_config.num_diffusion_steps
        if do_sample:
            logger.warning_once(
                "VibeVoice generation does not support sampling-based token selection. "
                "Tokens will be selected using argmax regardless of do_sample=True."
            )

        # State tracking
        acoustic_cache, semantic_cache, inputs_embeds = None, None, None
        audio_chunks = [[] for _ in range(batch_size)]
        cur_len = input_ids.shape[1]

        # Setup negative generation for classifier-free guidance
        negative_input_ids, negative_model_kwargs = self._prepare_negative_generation(
            batch_size, generation_config, device=input_ids.device
        )
        negative_forward = (
            self._get_negative_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )

        # Generation limits for progress tracking
        initial_length = input_ids.shape[-1]
        initial_length_per_sample = model_kwargs["attention_mask"].sum(dim=-1)
        max_step_per_sample = torch.min(
            generation_config.max_length - initial_length_per_sample,
            torch.full_like(initial_length_per_sample, generation_config.max_length - initial_length),
        )
        if monitor_progress:
            progress_bar = logging.tqdm(total=int(max_step_per_sample.max()), desc="Generating audio", unit=" tokens")
        else:
            progress_bar = None
        # ============================================

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # *************** VibeVoice specific ***************
            if progress_bar is not None:
                progress_bar.update(1)
            # ============================================

            if prefill_consumed:
                next_sequence_length = 1 if model_kwargs["use_cache"] else None
                model_inputs = self.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=next_sequence_length, **model_kwargs
                )
                # *************** VibeVoice specific ***************
                # Subsequent steps use embeddings from previous step
                model_inputs.pop("input_values", None)
                # `padding_mask` is used with `input_values` so we don't need it for subsequent steps
                model_inputs.pop("padding_mask", None)
                model_inputs["inputs_embeds"] = inputs_embeds
                # ============================================
                with self._optimize_model_for_decode():
                    outputs = model_forward(**model_inputs, return_dict=True)
            prefill_consumed = True
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            # token selection
            # *************** VibeVoice specific ***************
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            # ============================================

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # *************** VibeVoice specific ***************
            next_inputs_embeds = self.get_input_embeddings()(next_tokens).unsqueeze(1)

            # When audio_bos is predicted, reset the negative branch KV cache so the unconditional
            # CFG pass starts from a clean single-token context for this sequence.
            diffusion_start_mask = unfinished_sequences.bool() & (next_tokens == self.config.audio_bos_token_id)
            self._reset_negative_cache_for_audio_start(diffusion_start_mask, negative_input_ids, negative_model_kwargs)

            # When audio_token is predicted, run the diffusion head to synthesize the next audio chunk
            # and compute the embedding for the next LM step.
            diffusion_mask = unfinished_sequences.bool() & (next_tokens == self.config.audio_token_id)
            if diffusion_mask.any():
                negative_condition, negative_input_ids, negative_model_kwargs = self._step_negative_branch(
                    diffusion_mask,
                    next_tokens,
                    inputs_embeds,
                    negative_input_ids,
                    negative_model_kwargs,
                    negative_forward,
                )
                positive_condition = outputs.last_hidden_state[diffusion_mask, -1, :]
                audio_latent = self._sample_audio_latent(
                    positive_condition,
                    negative_condition,
                    noise_scheduler,
                    num_diffusion_steps,
                    generation_config.guidance_scale,
                )
                audio_output = self._decode_audio_latent(audio_latent, diffusion_mask, batch_size, acoustic_cache)
                acoustic_cache = audio_output.padding_cache
                # Decoder output retains the original batch, including inactive
                # rows. Compact active-row indexes would copy another clip.
                for sample_idx in diffusion_mask.nonzero(as_tuple=False).view(-1):
                    audio_chunks[sample_idx.item()].append(audio_output.audio[sample_idx.item()])

                # prepare inputs for next LM step
                semantic_outputs = self.model.semantic_tokenizer_encoder(
                    audio_output.audio,
                    padding_cache=semantic_cache,
                    use_cache=True,
                )
                semantic_features = semantic_outputs.latents[diffusion_mask.to(semantic_outputs.latents.device)]
                acoustic_embed = self.model.multi_modal_projector(audio_latent)
                semantic_embed = self.model.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed.to(acoustic_embed.device)
                next_inputs_embeds[diffusion_mask] = diffusion_embeds.to(next_inputs_embeds.device)
                semantic_cache = semantic_outputs.padding_cache

            # Restore legacy segment resets. These are audio segment boundaries,
            # not necessarily dialogue line or speaker boundaries. Reset after
            # full-batch decoding so another active row cannot refill ended rows.
            diffusion_end_mask = unfinished_sequences.bool() & (next_tokens == self.config.audio_eos_token_id)
            if reset_tokenizer_caches and diffusion_end_mask.any():
                for cache in (acoustic_cache, semantic_cache):
                    if cache is not None:
                        for layer in cache.layers.values():
                            if layer.is_initialized:
                                layer.cache[diffusion_end_mask.to(layer.cache.device)] = 0

            inputs_embeds = next_inputs_embeds
            cur_len += 1
            # ============================================

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        # *************** VibeVoice specific ***************
        if progress_bar is not None:
            progress_bar.close()
        # ============================================

        if streamer is not None:
            streamer.end()

        # *************** VibeVoice specific ***************
        generated_audio = [torch.cat(chunks, dim=-1) if chunks else None for chunks in audio_chunks]
        # ============================================

        if return_dict_in_generate:
            cache = None
            if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
                cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
                cache = model_kwargs[cache_key]
            # *************** VibeVoice specific ***************
            return VibeVoiceGenerateOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=cache,
                audio=generated_audio,
            )
        else:
            # NOTE (ebezzam): new tokens in input_ids are simply audio tokens (mainly `audio_token_id` to
            # trigger generation) so returning `input_ids` is insufficient for generating audio
            return generated_audio
