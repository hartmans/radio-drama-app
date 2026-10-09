"""Original VibeVoice API adapter; compatibility belongs to the upstream package.

Queueing, cache identity, reference preprocessing and debug output are shared
with the native backend. Import the upstream package only on first synthesis:
its AutoModel registrations must not affect native Transformers runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from .config import MODEL_NATIVE_SAMPLE_RATE
from .dialogue import ScriptRenderRequest
from .model_loading import shared_model_load
from .vibevoice import RegisteredRenderRequest, VibeVoiceResource

if TYPE_CHECKING:
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


def _upstream_vibevoice_types():
    try:
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    except ModuleNotFoundError as exc:
        if exc.name != "vibevoice":
            raise
        raise RuntimeError(
            "--upstream-vibevoice requires the upstream package in this Python environment. "
            "Install it with: python -m pip install --no-deps -e ~/ai/vibevoice"
        ) from exc
    return VibeVoiceProcessor, VibeVoiceForConditionalGenerationInference


class UpstreamVibeVoiceResource(VibeVoiceResource):
    """Render through upstream's script/voice_samples interface on Transformers 4 or 5."""

    def _render_batch_native_sync(
        self,
        batch: Sequence[RegisteredRenderRequest],
    ) -> list[np.ndarray]:
        """Return model-native mono audio for one batch before format conversion."""
        requests = [registration.request for registration in batch]
        processor, model = self._ensure_loaded()
        normalized_requests = [
            self._normalized_script_and_voice_samples(
                request,
                voice_sample_rate=int(processor.audio_processor.sampling_rate),
            )
            for request in requests
        ]
        inputs = processor(
            text=[normalized_script for normalized_script, _ in normalized_requests],
            voice_samples=[voice_samples for _, voice_samples in normalized_requests],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=self.config.resolved_cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                is_prefill=True,
            )

        generated = list(outputs.speech_outputs or [])
        if len(generated) != len(batch):
            raise RuntimeError(
                f"Model generation returned {len(generated)} clips for {len(batch)} requests"
            )
        return [self._normalize_audio_array(audio) for audio in generated]

    def _ensure_loaded(
        self,
    ) -> tuple[VibeVoiceProcessor, VibeVoiceForConditionalGenerationInference]:
        """Load and cache the processor/model pair on first use."""
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        with shared_model_load():
            if self._processor is not None and self._model is not None:
                return self._processor, self._model

            VibeVoiceProcessor, _ = _upstream_vibevoice_types()
            processor = VibeVoiceProcessor.from_pretrained(self.config.resolved_model_name)
            self._sample_rate = getattr(
                processor.audio_processor,
                "sampling_rate",
                MODEL_NATIVE_SAMPLE_RATE,
            )

            load_dtype, attn_implementation = self._load_settings_for_device(self.device)
            try:
                model = self._load_model(
                    model_name=self.config.resolved_model_name,
                    device=self.device,
                    load_dtype=load_dtype,
                    attn_implementation=attn_implementation,
                )
            except Exception:
                if attn_implementation != "flash_attention_2":
                    raise
                model = self._load_model(
                    model_name=self.config.resolved_model_name,
                    device=self.device,
                    load_dtype=load_dtype,
                    attn_implementation="sdpa",
                )

            model.eval()
            model.set_ddpm_inference_steps(
                num_steps=self.config.resolved_ddpm_inference_steps
            )

            self._processor = processor
            self._model = model
            return processor, model

    def _load_model(
        self,
        model_name: str,
        device: str,
        load_dtype: torch.dtype,
        attn_implementation: str,
    ) -> VibeVoiceForConditionalGenerationInference:
        _, VibeVoiceForConditionalGenerationInference = _upstream_vibevoice_types()
        if device == "mps":
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_name,
                torch_dtype=load_dtype,
                attn_implementation=attn_implementation,
                device_map=None,
            )
            model.to("mps")
            return model

        device_map = "cuda" if device == "cuda" else "cpu"
        return VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_name,
            torch_dtype=load_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

    def _normalized_script_and_voice_samples(
        self,
        request: ScriptRenderRequest,
        *,
        voice_sample_rate: int,
    ) -> tuple[str, list[np.ndarray]]:
        speaker_numbers: dict[tuple[Path, float], int] = {}
        voice_samples: list[np.ndarray] = []
        normalized_lines: list[str] = []

        for line in request.dialogue_lines:
            speaker_key = (
                Path(line.speaker.resolved_path).expanduser().resolve(),
                line.speaker.gain,
            )
            speaker_number = speaker_numbers.get(speaker_key)
            if speaker_number is None:
                speaker_number = len(voice_samples) + 1
                speaker_numbers[speaker_key] = speaker_number
                resolved_path, gain = speaker_key
                if gain:
                    voice_sample = self._preprocessed_voice_sample_sync(
                        resolved_path,
                        output_sample_rate=voice_sample_rate,
                        gain_db=gain,
                    )
                else:
                    voice_sample = self._preprocessed_voice_sample_sync(
                        resolved_path,
                        output_sample_rate=voice_sample_rate,
                    )
                voice_samples.append(voice_sample)
            for paragraph in self._normalized_script_paragraphs(line.spoken_text):
                normalized_lines.append(f"Speaker {speaker_number}: {paragraph}")

        return "\n".join(normalized_lines).replace("’", "'"), voice_samples
