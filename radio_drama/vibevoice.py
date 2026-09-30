from __future__ import annotations

import asyncio
import re
import weakref
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Sequence

import numpy as np
import soundfile as sf
import torch
from carthage.dependency_injection import inject

if TYPE_CHECKING:
    from transformers import VibeVoiceForConditionalGeneration, VibeVoiceProcessor


def _vibevoice_types():
    """Import Transformers model classes only when live synthesis needs them."""
    from transformers import VibeVoiceForConditionalGeneration, VibeVoiceProcessor
    return VibeVoiceProcessor, VibeVoiceForConditionalGeneration


from .cache import CACHE_DIRECTORY_KEY, CacheManager
from .config import ProductionConfig
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
        self._model: VibeVoiceForConditionalGeneration | None = None
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
        conversations = [
            self._conversation(request, voice_sample_rate=int(processor.feature_extractor.sampling_rate))
            for request in requests
        ]
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            processor_kwargs={"padding": True, "return_tensors": "pt"},
        )

        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(
                    device=self.device,
                    dtype=model.dtype if value.is_floating_point() else value.dtype,
                )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                # This checkpoint has no generation_config.json; avoid the
                # generic short default while allowing EOS to end each script.
                max_new_tokens=None,
                max_length=model.config.text_config.max_position_embeddings,
                guidance_scale=self.config.resolved_cfg_scale,
                num_diffusion_steps=self.config.resolved_ddpm_inference_steps,
                do_sample=False,
            )

        generated = outputs
        if len(generated) != len(batch):
            raise RuntimeError(
                f"Model generation returned {len(generated)} clips for {len(batch)} requests"
            )
        return [self._normalize_audio_array(audio) for audio in generated]

    def _ensure_loaded(
        self,
    ) -> tuple[VibeVoiceProcessor, VibeVoiceForConditionalGeneration]:
        """Load and cache the processor/model pair on first use."""
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        with shared_model_load():
            if self._processor is not None and self._model is not None:
                return self._processor, self._model

            VibeVoiceProcessor, _ = _vibevoice_types()
            processor = VibeVoiceProcessor.from_pretrained(self.config.resolved_model_name)
            self._sample_rate = int(processor.feature_extractor.sampling_rate)

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

            self._processor = processor
            self._model = model
            return processor, model

    def _load_model(
        self,
        model_name: str,
        device: str,
        load_dtype: torch.dtype,
        attn_implementation: str,
    ) -> VibeVoiceForConditionalGeneration:
        """Use optimized text attention while audio tokenizers stay on eager.

        A single attention setting propagates into all nested Transformers
        configs, but the acoustic tokenizer does not implement SDPA.
        """
        _, VibeVoiceForConditionalGeneration = _vibevoice_types()
        if device == "mps":
            model = VibeVoiceForConditionalGeneration.from_pretrained(
                model_name,
                dtype=load_dtype,
                attn_implementation={"": "eager", "text_config": attn_implementation},
                device_map=None,
            )
            model.to("mps")
            return model

        device_map = "cuda" if device == "cuda" else "cpu"
        return VibeVoiceForConditionalGeneration.from_pretrained(
            model_name,
            dtype=load_dtype,
            device_map=device_map,
            attn_implementation={"": "eager", "text_config": attn_implementation},
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

    def _conversation(
        self,
        request: ScriptRenderRequest,
        *,
        voice_sample_rate: int,
    ) -> list[dict]:
        """Attach a preprocessed reference only at each voice's first paragraph.

        Role IDs are zero-based and shared by resolved path and reference gain,
        independently of authored names and output effects. Keeping each paragraph
        as a turn preserves the previous script normalization.
        """
        speaker_numbers: dict[tuple[Path, float], int] = {}
        conversation: list[dict] = []
        for line in request.dialogue_lines:
            paragraphs = self._normalized_script_paragraphs(line.spoken_text)
            if not paragraphs:
                continue
            speaker_key = (
                Path(line.speaker.resolved_path).expanduser().resolve(),
                line.speaker.gain,
            )
            speaker_number = speaker_numbers.get(speaker_key)
            voice_sample = None
            if speaker_number is None:
                speaker_number = len(speaker_numbers)
                speaker_numbers[speaker_key] = speaker_number
                resolved_path, gain = speaker_key
                kwargs = {"gain_db": gain} if gain else {}
                voice_sample = self._preprocessed_voice_sample_sync(
                    resolved_path, output_sample_rate=voice_sample_rate, **kwargs,
                )
            for paragraph in paragraphs:
                content = [{"type": "text", "text": paragraph.replace("’", "'")}]
                if voice_sample is not None:
                    content.append({"type": "audio", "audio": voice_sample})
                    voice_sample = None
                conversation.append({"role": str(speaker_number), "content": content})
        return conversation

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
