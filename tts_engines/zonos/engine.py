"""Radio-drama proxy adapter for batched Zonos v0.1 synthesis."""

from __future__ import annotations

import os
import types
from collections.abc import Mapping, Sequence
from typing import Any

from radio_drama_tts_container import (
    finish_line_work, prepare_line_work, remove_line_work, run_server,
)

MODEL = os.environ.get("ZONOS_MODEL", "Zyphra/Zonos-v0.1-transformer")
LANGUAGE = os.environ.get("ZONOS_LANGUAGE", "en-us")


class ZonosEngine:
    """Keep one model resident and use Zonos' native line batching."""

    def __init__(self) -> None:
        self.model = None
        self._speaker_embeddings: dict[str, Any] = {}

    def load_model(self):
        if self.model is None:
            import torch
            from zonos.model import Zonos

            self.model = Zonos.from_pretrained(MODEL, device="cuda")
            self.model._prefill = types.MethodType(_batched_prefill, self.model)
            torch.set_grad_enabled(False)
        return self.model

    def speaker_embedding(self, path: str):
        if path not in self._speaker_embeddings:
            import torchaudio
            from zonos.speaker_cloning import SpeakerEmbeddingLDA

            wav, rate = torchaudio.load(path)
            model = self.load_model()
            if model.spk_clone_model is None:
                # torchaudio constructs its mel filter on CPU.  Constructing it
                # inside Zonos' CUDA device context breaks on newer PyTorch;
                # build first, then move the completed module.
                clone_model = SpeakerEmbeddingLDA(device="cpu").to(model.device)
                clone_model.device = model.device
                clone_model.model.device = model.device
                model.spk_clone_model = clone_model
            self._speaker_embeddings[path] = model.make_speaker_embedding(wav, rate)
        return self._speaker_embeddings[path]

    def synthesize_batch(self, lines: Sequence[Mapping[str, Any]]):
        import torch
        from zonos.conditioning import make_cond_dict, supported_language_codes

        model = self.load_model()
        language_ids = {name: index for index, name in enumerate(supported_language_codes)}
        conds = []
        for line in lines:
            language = os.environ.get("ZONOS_LANGUAGE", LANGUAGE).lower()
            conds.append(make_cond_dict(
                text=line["spoken_text"], language=language,
                speaker=self.speaker_embedding(line["speaker"]["voice_path"]),
                device=model.device,
            ))
        combined = dict(conds[0])
        combined["espeak"] = (
            [line["spoken_text"] for line in lines],
            [os.environ.get("ZONOS_LANGUAGE", LANGUAGE).lower()] * len(lines),
        )
        # Conditioner.forward treats its value as positional arguments. Wrap
        # batched tensors in a one-tuple so their batch dimension is retained.
        combined["speaker"] = (torch.cat([cond["speaker"] for cond in conds]),)
        combined["language_id"] = (torch.tensor(
            [language_ids[language] for language in combined["espeak"][1]],
            device=model.device,
        ).view(-1, 1, 1),)
        conditioning = model.prepare_conditioning(combined)
        codes, eos_tracker = _generate_nonempty_codes(
            model,
            conditioning,
            batch_size=len(lines),
            attempts=int(os.environ.get("ZONOS_GENERATION_ATTEMPTS", "3")),
        )
        lengths = eos_tracker.lengths(default=codes.shape[-1])
        return [
            model.autoencoder.decode(codes[index : index + 1, :, :length]).cpu()[0]
            for index, length in enumerate(lengths)
        ]

    def render_batch(self, requests: Sequence[Mapping[str, Any]]):
        import torchaudio

        outputs, work = prepare_line_work(requests)
        try:
            batch_size = int(os.environ.get("ZONOS_BATCH_SIZE", "8"))
            if batch_size < 1:
                raise ValueError("ZONOS_BATCH_SIZE must be at least 1")
            sample_rate = self.load_model().autoencoder.sampling_rate
            for start in range(0, len(work), batch_size):
                chunk = work[start : start + batch_size]
                audio = self.synthesize_batch([item.line for item in chunk])
                for item, samples in zip(chunk, audio, strict=True):
                    torchaudio.save(str(item.path), samples, sample_rate)
            return finish_line_work(outputs, work, sample_rate=sample_rate)
        finally:
            remove_line_work(work)


def _batched_prefill(model, prefix_hidden_states, input_ids, inference_params, cfg_scale):
    """Zonos prefill with CFG replication that retains a real batch dimension."""

    import torch

    if cfg_scale != 1.0:
        input_ids = torch.cat((input_ids, input_ids), dim=0)
    hidden_states = torch.cat((prefix_hidden_states, model.embed_codes(input_ids)), dim=1)
    return model._compute_logits(hidden_states, inference_params, cfg_scale)


class _EosTracker:
    """Record each batch item's valid codec length before its first EOS."""

    def __init__(self, batch_size: int, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id
        self._lengths: list[int | None] = [None] * batch_size

    def __call__(self, frame, step: int, _max_steps: int) -> bool:
        if frame.shape[-1] == 0:
            # Zonos may invoke the callback once with an empty end slice when
            # the generation cursor reaches the allocated token buffer.
            return True
        ended = (frame[:, 0, 0] == self.eos_token_id).detach().cpu().tolist()
        for index, is_ended in enumerate(ended):
            if is_ended and self._lengths[index] is None:
                # Codebook zero is delayed by one frame. At callback step N,
                # its EOS maps to output code index N, leaving N valid codes.
                self._lengths[index] = step
        return True

    def lengths(self, *, default: int) -> list[int]:
        return [default if length is None else length for length in self._lengths]


def _generate_nonempty_codes(model, conditioning, *, batch_size: int, attempts: int):
    """Retry stochastic initial-EOS generations before invoking the codec."""

    if attempts < 1:
        raise ValueError("ZONOS_GENERATION_ATTEMPTS must be at least 1")
    for _attempt in range(attempts):
        eos_tracker = _EosTracker(batch_size, model.eos_token_id)
        codes = model.generate(
            conditioning,
            batch_size=batch_size,
            max_new_tokens=int(os.environ.get("ZONOS_MAX_NEW_TOKENS", str(86 * 30))),
            cfg_scale=float(os.environ.get("ZONOS_CFG_SCALE", "2.0")),
            progress_bar=False,
            callback=eos_tracker,
        )
        if codes.shape[-1] > 0:
            return codes, eos_tracker
    raise RuntimeError(
        f"Zonos generated no audio tokens in {attempts} consecutive attempts"
    )


def main() -> None:
    run_server(ZonosEngine().render_batch)


if __name__ == "__main__":
    main()
