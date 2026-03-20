from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
from .planning import ScriptRenderRequest
from .rendering import RenderResult


@dataclass(slots=True)
class RegisteredRenderRequest:
    resource: "VibeVoiceResource"
    request: ScriptRenderRequest
    future: asyncio.Future

    async def render(self) -> RenderResult:
        return await self.resource.render_registered_request(self)


@dataclass(slots=True)
class _PendingRender:
    registration: RegisteredRenderRequest


@inject(config=ProductionConfig)
class VibeVoiceResource(AsyncInjectable):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = self._normalize_device(self.config.resolved_device)
        self._processor: VibeVoiceProcessor | None = None
        self._model: VibeVoiceForConditionalGenerationInference | None = None
        self._sample_rate: int | None = None
        self._pending: list[_PendingRender] = []
        self._pending_lock = asyncio.Lock()
        self._drain_task: asyncio.Task | None = None

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
                self._pending.append(_PendingRender(registration=registration))
        return registration

    async def render_registered_request(
        self,
        registration: RegisteredRenderRequest,
    ) -> RenderResult:
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
                if not self._pending:
                    self._drain_task = None
                    return
                batch = self._pending[: self.config.resolved_batch_size]
                del self._pending[: self.config.resolved_batch_size]

            try:
                rendered_audios = await asyncio.to_thread(self._render_batch_sync, batch)
            except Exception as exc:
                for pending in batch:
                    if not pending.registration.future.done():
                        pending.registration.future.set_exception(exc)
                continue

            for pending, audio in zip(batch, rendered_audios, strict=True):
                if not pending.registration.future.done():
                    pending.registration.future.set_result(RenderResult(audio=audio))

    def _render_batch_sync(self, batch: Sequence[_PendingRender]) -> list[np.ndarray]:
        generated = self._render_batch_native_sync(batch)
        return [
            convert_audio_format(
                audio,
                input_sample_rate=self.sample_rate,
                output_sample_rate=self.config.resolved_output_sample_rate,
                output_channels=self.config.resolved_output_channels,
            )
            for audio in generated
        ]

    def _render_batch_native_sync(self, batch: Sequence[_PendingRender]) -> list[np.ndarray]:
        requests = [pending.registration.request for pending in batch]
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
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        processor = VibeVoiceProcessor.from_pretrained(self.config.resolved_model_name)
        self._sample_rate = getattr(processor.audio_processor, "sampling_rate", MODEL_NATIVE_SAMPLE_RATE)

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
        model.set_ddpm_inference_steps(num_steps=self.config.resolved_ddpm_inference_steps)

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
            raise ValueError(f"Expected mono audio after generation, got {array.shape!r}")
        return np.ascontiguousarray(array, dtype=np.float32)
