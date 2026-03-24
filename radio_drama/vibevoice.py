from __future__ import annotations

import asyncio
import inspect
import re
import weakref
from dataclasses import dataclass
from threading import Lock
from typing import Sequence

import numpy as np
import torch
from carthage.dependency_injection import AsyncInjectable, inject
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

from .audio import convert_audio_format
from .config import MODEL_NATIVE_SAMPLE_RATE, ProductionConfig
from .debug import write_debug_message, write_debug_wav
from .planning import ScriptRenderRequest
from .rendering import RenderResult


@dataclass(slots=True, weakref_slot=True)
class RegisteredRenderRequest:
    """A queued render request whose result may be fulfilled by a later batch."""

    resource: "VibeVoiceResource"
    request: ScriptRenderRequest
    future: asyncio.Future

    async def render(self) -> RenderResult:
        return await self.resource.render_registered_request(self)


@dataclass(slots=True)
class _PendingRender:
    registration_ref: weakref.ReferenceType[RegisteredRenderRequest]

    def registration(self) -> RegisteredRenderRequest | None:
        return self.registration_ref()


@inject(config=ProductionConfig)
class VibeVoiceResource(AsyncInjectable):
    """Shared VibeVoice model resource for script-level render requests.

    Scripts register requests during planning. Rendering any registered request
    allows the resource to drain the current queue in batches, load the model
    lazily, and return production-format audio to each waiting plan.
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

    def empty_result(self) -> RenderResult:
        return RenderResult.empty(channels=self.config.resolved_output_channels)

    async def register_request(
        self,
        request: ScriptRenderRequest | None,
    ) -> RegisteredRenderRequest:
        """Register work for later batched rendering.

        ``None`` is treated as an empty render and resolved immediately so plans
        for empty scripts still follow the same request lifecycle.
        """
        loop = asyncio.get_running_loop()
        registration = RegisteredRenderRequest(
            resource=self,
            request=request or ScriptRenderRequest(normalized_script="", voice_samples=()),
            future=loop.create_future(),
        )
        async with self._pending_lock:
            if request is None:
                registration.future.set_result(self.empty_result())
            else:
                self._pending.append(
                    _PendingRender(registration_ref=weakref.ref(registration))
                )
        return registration

    async def render_registered_request(
        self,
        registration: RegisteredRenderRequest,
    ) -> RenderResult:
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
                rendered_audios = await asyncio.to_thread(self._render_batch_sync, batch)
            except Exception as exc:
                for registration in batch:
                    if not registration.future.done():
                        registration.future.set_exception(exc)
                continue

            for registration, audio in zip(batch, rendered_audios, strict=True):
                if not registration.future.done():
                    registration.future.set_result(RenderResult(audio=audio))

    def _render_batch_sync(
        self,
        batch: Sequence[RegisteredRenderRequest],
    ) -> list[np.ndarray]:
        generated = self._render_batch_native_sync(batch)
        self._write_vibevoice_debug_outputs(batch, generated)
        return [
            convert_audio_format(
                audio,
                input_sample_rate=self.sample_rate,
                output_sample_rate=self.config.resolved_output_sample_rate,
                output_channels=self.config.resolved_output_channels,
            )
            for audio in generated
        ]

    def _render_batch_native_sync(
        self,
        batch: Sequence[RegisteredRenderRequest],
    ) -> list[np.ndarray]:
        """Return model-native mono audio for one batch before format conversion."""
        requests = [registration.request for registration in batch]
        processor, model = self._ensure_loaded()
        inputs = processor(
            text=[request.normalized_script for request in requests],
            voice_samples=[list(request.voice_samples) for request in requests],
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
        generated: Sequence[np.ndarray],
    ) -> None:
        if not self.config.debug_enabled("vibevoice_output"):
            return

        start_index = self._reserve_debug_output_indexes(len(generated))
        for output_index, pending, audio in zip(
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
                sample_rate=self.sample_rate,
            )
            if artifact_path is not None:
                write_debug_message(
                    self.config,
                    "vibevoice_output",
                    f"{artifact_path.name} sample_rate={self.sample_rate} frames={audio.shape[0]}",
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
        first_line = next(
            (
                line.strip()
                for line in request.normalized_script.splitlines()
                if line.strip()
            ),
            "empty-script",
        )
        if ":" in first_line:
            _, first_line = first_line.split(":", 1)
        label = " ".join(first_line.split()).strip()
        return label[:40] or "empty-script"

    def _sanitize_debug_label(self, text: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
        return sanitized or "audio"


__all__ = ["RegisteredRenderRequest", "VibeVoiceResource"]
