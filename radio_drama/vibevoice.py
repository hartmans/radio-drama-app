from __future__ import annotations

import asyncio
import inspect
import re
import weakref
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
from carthage.dependency_injection import inject
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

from .cache import CACHE_DIRECTORY_KEY, CacheManager
from .config import MODEL_NATIVE_SAMPLE_RATE, ProductionConfig
from .debug import write_debug_message, write_debug_wav
from .effects import load_preprocessed_voice_reference
from .model_loading import shared_model_load
from .dialogue import ScriptRenderRequest, TtsResource
from .rendering import BackendTtsResult


@dataclass(slots=True, weakref_slot=True)
class RegisteredRenderRequest:
    """A queued render request whose result may be fulfilled by a later batch."""

    resource: "VibeVoiceResource"
    request: ScriptRenderRequest
    future: asyncio.Future

    async def render(self) -> BackendTtsResult:
        return await self.resource.render_registered_request(self)


@dataclass(slots=True)
class _PendingRender:
    registration_ref: weakref.ReferenceType[RegisteredRenderRequest]

    def registration(self) -> RegisteredRenderRequest | None:
        return self.registration_ref()


@inject(config=ProductionConfig, cache_manager=CacheManager)
class VibeVoiceResource(TtsResource):
    """Shared VibeVoice model resource for script-level render requests.

    Scripts register requests during planning. Rendering any registered request
    allows the resource to drain the current queue in batches, load the model
    lazily, and return model-native audio to the shared TTS cache layer.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = self._normalize_device(self.config.resolved_device)
        self._processor: VibeVoiceProcessor | None = None
        self._model: VibeVoiceForConditionalGenerationInference | None = None
        self._sample_rate: int | None = None
        self._pending: list[_PendingRender] = []
        self._pending_lock = asyncio.Lock()
        self._drain_task: asyncio.Task | None = None
        self._debug_output_index = 0
        self._debug_output_lock = Lock()

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            self._ensure_loaded()
        assert self._sample_rate is not None
        return self._sample_rate

    @property
    def cache_collection_name(self) -> str:
        return "vibevoice"

    async def register_backend_request(
        self,
        request: ScriptRenderRequest,
    ) -> RegisteredRenderRequest:
        """Register work for later batched rendering.

        ``None`` is treated as an empty render and resolved immediately so plans
        for empty scripts still follow the same request lifecycle.
        """
        loop = asyncio.get_running_loop()
        registration = RegisteredRenderRequest(
            resource=self,
            request=request,
            future=loop.create_future(),
        )
        async with self._pending_lock:
            self._pending.append(
                _PendingRender(registration_ref=weakref.ref(registration))
            )
        return registration

    async def render_registered_request(
        self,
        registration: RegisteredRenderRequest,
    ) -> BackendTtsResult:
        """Render one registration, potentially flushing additional queued work."""
        if registration.future.done():
            return await registration.future
        async with self._pending_lock:
            if self._drain_task is None or self._drain_task.done():
                self._drain_task = asyncio.create_task(self._drain_pending())
        return await registration.future

    async def _drain_pending(self) -> None:
        while True:
            await asyncio.sleep(0)
            async with self._pending_lock:
                batch = self._pop_live_batch_locked()
                if not batch:
                    self._drain_task = None
                    return

            try:
                rendered_results = await asyncio.to_thread(self._render_batch_sync, batch)
            except Exception as exc:
                for registration in batch:
                    if not registration.future.done():
                        registration.future.set_exception(exc)
                continue

            for registration, result in zip(batch, rendered_results, strict=True):
                if not registration.future.done():
                    registration.future.set_result(result)

    def _render_batch_sync(
        self,
        batch: Sequence[RegisteredRenderRequest],
    ) -> list[BackendTtsResult]:
        generated = self._render_batch_with_legacy_cache_sync(batch)
        debug_audio = [(audio, sample_rate) for audio, sample_rate, _ in generated]
        self._write_vibevoice_debug_outputs(batch, debug_audio)
        return [
            BackendTtsResult(
                audio,
                sample_rate=sample_rate,
                cache_wav_path=cache_wav_path,
            )
            for audio, sample_rate, cache_wav_path in generated
        ]

    def _render_batch_with_legacy_cache_sync(
        self,
        batch: Sequence[RegisteredRenderRequest],
    ) -> list[tuple[np.ndarray, int, Path | None]]:
        """Read pre-generic VibeVoice cache pairs, or generate uncached audio."""
        cache_collection = self.cache_manager["vibevoice"]
        if not cache_collection.enabled:
            return [
                (audio, self.sample_rate, None)
                for audio in self._render_batch_native_sync(batch)
            ]

        cached_outputs: dict[int, tuple[np.ndarray, int, Path | None]] = {}
        uncached_batch: list[tuple[int, RegisteredRenderRequest]] = []

        for index, registration in enumerate(batch):
            hit = cache_collection.find(
                registration.request,
                validate=registration.request.validate_cache_hit,
            )
            if hit is not None:
                cached_outputs[index] = self._load_cached_native_audio(hit)
                continue
            uncached_batch.append((index, registration))

        if uncached_batch:
            rendered = self._render_batch_native_sync(
                [registration for _, registration in uncached_batch]
            )
            for (index, _), audio in zip(uncached_batch, rendered, strict=True):
                cached_outputs[index] = (audio, self.sample_rate, None)

        return [cached_outputs[index] for index in range(len(batch))]

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
                is_prefill=not self.config.resolved_disable_prefill,
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
            self._patch_model_config_api(model)
            self._patch_generation_cache_api(model)

            self._processor = processor
            self._model = model
            return processor, model

    def _patch_model_config_api(
        self,
        model: VibeVoiceForConditionalGenerationInference,
    ) -> None:
        config = model.config
        decoder_config = getattr(config, "decoder_config", None)
        if decoder_config is None:
            return
        for field_name in (
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "hidden_size",
            "head_dim",
            "vocab_size",
        ):
            if hasattr(config, field_name):
                continue
            if hasattr(decoder_config, field_name):
                setattr(config, field_name, getattr(decoder_config, field_name))

    def _patch_generation_cache_api(
        self,
        model: VibeVoiceForConditionalGenerationInference,
    ) -> None:
        self._patch_dynamic_cache_api()
        original = model._prepare_cache_for_generation
        parameter_count = len(inspect.signature(original).parameters)
        if parameter_count != 5:
            return

        def compat_prepare_cache_for_generation(
            generation_config,
            model_kwargs,
            generation_mode,
            batch_size,
            max_cache_length,
            device=None,
        ):
            return original(
                generation_config,
                model_kwargs,
                generation_mode,
                batch_size,
                max_cache_length,
            )

        model._prepare_cache_for_generation = compat_prepare_cache_for_generation

    def _patch_dynamic_cache_api(self) -> None:
        from transformers.cache_utils import DynamicCache

        if not hasattr(DynamicCache, "key_cache"):
            DynamicCache.key_cache = property(
                lambda self: [getattr(layer, "keys", None) for layer in self.layers]
            )
        if not hasattr(DynamicCache, "value_cache"):
            DynamicCache.value_cache = property(
                lambda self: [getattr(layer, "values", None) for layer in self.layers]
            )

    def _load_model(
        self,
        model_name: str,
        device: str,
        load_dtype: torch.dtype,
        attn_implementation: str,
    ) -> VibeVoiceForConditionalGenerationInference:
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

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _normalize_device(self, device: str) -> str:
        normalized = (device or self._detect_device()).lower()
        if normalized == "mpx":
            normalized = "mps"
        if normalized == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        if normalized == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if normalized not in {"cuda", "mps", "cpu"}:
            raise ValueError(f"Unsupported device: {device}")
        return normalized

    def _load_settings_for_device(self, device: str) -> tuple[torch.dtype, str]:
        if device == "cuda":
            return torch.bfloat16, "flash_attention_2"
        return torch.float32, "sdpa"

    def _normalize_audio_array(self, audio: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(audio):
            array = audio.detach().float().cpu().numpy()
        else:
            array = np.asarray(audio, dtype=np.float32)
        array = np.squeeze(array)
        if array.ndim != 1:
            raise ValueError(
                f"Expected mono audio after generation, got {array.shape!r}"
            )
        return np.ascontiguousarray(array, dtype=np.float32)

    def _write_vibevoice_debug_outputs(
        self,
        batch: Sequence[RegisteredRenderRequest],
        generated: Sequence[tuple[np.ndarray, int]],
    ) -> None:
        if not self.config.debug_enabled("vibevoice_output"):
            return

        start_index = self._reserve_debug_output_indexes(len(generated))
        for output_index, pending, (audio, sample_rate) in zip(
            range(start_index, start_index + len(generated)),
            batch,
            generated,
            strict=True,
        ):
            request = pending.request
            filename = (
                f"{output_index:03d}-"
                f"{self._sanitize_debug_label(self._debug_request_label(request))}.wav"
            )
            artifact_path = write_debug_wav(
                self.config,
                "vibevoice_output",
                filename,
                audio,
                sample_rate=sample_rate,
            )
            if artifact_path is not None:
                write_debug_message(
                    self.config,
                    "vibevoice_output",
                    f"{artifact_path.name} sample_rate={sample_rate} frames={audio.shape[0]}",
                )

    def _reserve_debug_output_indexes(self, count: int) -> int:
        with self._debug_output_lock:
            start_index = self._debug_output_index
            self._debug_output_index += count
        return start_index

    def _pop_live_batch_locked(self) -> list[RegisteredRenderRequest]:
        live_batch: list[RegisteredRenderRequest] = []
        remaining_pending: list[_PendingRender] = []

        for pending in self._pending:
            registration = pending.registration()
            if registration is None:
                continue
            if len(live_batch) < self.config.resolved_batch_size:
                live_batch.append(registration)
            else:
                remaining_pending.append(pending)

        self._pending = remaining_pending
        return live_batch

    def _debug_request_label(self, request: ScriptRenderRequest) -> str:
        return request.cache_first_words()

    def _sanitize_debug_label(self, text: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
        return sanitized or "audio"

    def _load_cached_native_audio(
        self,
        hit: dict[str, Path],
    ) -> tuple[np.ndarray, int, Path]:
        wav_path = hit["wav"]
        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
        normalized = self._normalize_audio_array(audio)
        return normalized, int(sample_rate), wav_path

    def _normalized_script_and_voice_samples(
        self,
        request: ScriptRenderRequest,
        *,
        voice_sample_rate: int,
    ) -> tuple[str, list[np.ndarray]]:
        speaker_numbers: dict[str, int] = {}
        loaded_voice_samples: dict[tuple[Path, float], np.ndarray] = {}
        voice_samples: list[np.ndarray] = []
        normalized_lines: list[str] = []

        for line in request.dialogue_lines:
            speaker_key = line.speaker.authored_name.lower()
            speaker_number = speaker_numbers.get(speaker_key)
            if speaker_number is None:
                speaker_number = len(voice_samples) + 1
                speaker_numbers[speaker_key] = speaker_number
                resolved_path = Path(line.speaker.resolved_path).expanduser().resolve()
                voice_reference = (resolved_path, line.speaker.gain)
                voice_sample = loaded_voice_samples.get(voice_reference)
                if voice_sample is None:
                    if line.speaker.gain:
                        voice_sample = self._preprocessed_voice_sample_sync(
                            resolved_path,
                            output_sample_rate=voice_sample_rate,
                            gain_db=line.speaker.gain,
                        )
                    else:
                        voice_sample = self._preprocessed_voice_sample_sync(
                            resolved_path,
                            output_sample_rate=voice_sample_rate,
                        )
                    loaded_voice_samples[voice_reference] = voice_sample
                voice_samples.append(voice_sample)
            for paragraph in self._normalized_script_paragraphs(line.spoken_text):
                normalized_lines.append(f"Speaker {speaker_number}: {paragraph}")

        return "\n".join(normalized_lines).replace("’", "'"), voice_samples

    def _preprocessed_voice_sample_sync(
        self,
        voice_path: Path,
        *,
        output_sample_rate: int,
        gain_db: float = 0.0,
    ) -> np.ndarray:
        voice_sample, _ = load_preprocessed_voice_reference(
            voice_path,
            output_sample_rate=output_sample_rate,
            gain_db=gain_db,
        )
        return voice_sample

    def _normalized_script_paragraphs(self, spoken_text: str) -> list[str]:
        paragraphs: list[str] = []
        current_paragraph: list[str] = []

        for raw_line in spoken_text.splitlines():
            stripped_line = raw_line.strip()
            if not stripped_line:
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph.clear()
                continue
            current_paragraph.append(stripped_line)

        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))
        return paragraphs or ([" ".join(spoken_text.split()).strip()] if spoken_text.strip() else [])

__all__ = ["CACHE_DIRECTORY_KEY", "RegisteredRenderRequest", "VibeVoiceResource"]
