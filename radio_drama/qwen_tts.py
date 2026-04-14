from __future__ import annotations

import asyncio
import json
import os
import weakref
from pathlib import Path
from threading import Lock, RLock
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import soundfile as sf
import torch
from carthage.dependency_injection import AsyncInjectable, inject

from .audio import convert_audio_format
from .cache import CacheCollection, CacheKey, CacheManager
from .config import ProductionConfig
from .effects import VOICE_PREPROCESS_VERSION, load_preprocessed_voice_reference
from .forced_alignment import WhisperXResource
from .model_loading import shared_model_load
from .dialogue import DialogueLine, ScriptRenderRequest
from .rendering import RenderResult, ScriptRenderResult
from .vibevoice import RegisteredRenderRequest


if TYPE_CHECKING:
    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem


_QWEN_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
_QWEN_LANGUAGE = "English"
_QWEN_PROMPT_CACHE_DIRECTORY = Path("~/.cache/radio_drama/qwen_prompts").expanduser()


class _PendingRender:
    __slots__ = ("registration_ref",)

    def __init__(self, registration: RegisteredRenderRequest) -> None:
        self.registration_ref = weakref.ref(registration)

    def registration(self) -> RegisteredRenderRequest | None:
        return self.registration_ref()


@inject(
    config=ProductionConfig,
    whisperx_resource=WhisperXResource,
    cache_manager=CacheManager,
)
class QwenTtsResource(AsyncInjectable):
    """Shared Qwen voice-clone resource for script-level render requests."""

    def __init__(self, whisperx_resource: WhisperXResource, **kwargs) -> None:
        super().__init__(**kwargs)
        self.whisperx_resource = whisperx_resource
        self.device = self._normalize_device(self.config.resolved_device)
        self._model: Qwen3TTSModel | None = None
        self._sample_rate: int | None = None
        self._pending: list[_PendingRender] = []
        self._pending_lock = asyncio.Lock()
        self._drain_task: asyncio.Task | None = None
        self._load_lock = RLock()
        self._prompt_cache_lock = Lock()
        self._voice_prompt_cache: dict[str, list[VoiceClonePromptItem]] = {}
        self._voice_clone_prompt_item_type = None

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            self._ensure_loaded()
        assert self._sample_rate is not None
        return self._sample_rate

    def empty_result(self) -> RenderResult:
        return ScriptRenderResult.empty(channels=self.config.resolved_output_channels)

    async def register_request(
        self,
        request: ScriptRenderRequest | None,
    ) -> RegisteredRenderRequest:
        loop = asyncio.get_running_loop()
        registration = RegisteredRenderRequest(
            resource=self,
            request=request or ScriptRenderRequest(dialogue_lines=[]),
            future=loop.create_future(),
        )
        async with self._pending_lock:
            if request is None:
                registration.future.set_result(self.empty_result())
            else:
                self._pending.append(_PendingRender(registration))
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
    ) -> list[RenderResult]:
        generated = self._render_batch_with_cache_sync(batch)
        rendered_results: list[RenderResult] = []
        for native_result, sample_rate in generated:
            rendered_results.append(
                ScriptRenderResult(
                    audio=convert_audio_format(
                        native_result.audio,
                        input_sample_rate=sample_rate,
                        output_sample_rate=self.config.resolved_output_sample_rate,
                        output_channels=self.config.resolved_output_channels,
                    ),
                    dialogue_line_start_positions=native_result.dialogue_line_start_positions,
                )
            )
        return rendered_results

    def _render_batch_with_cache_sync(
        self,
        batch: Sequence[RegisteredRenderRequest],
    ) -> list[tuple[ScriptRenderResult, int]]:
        cache_collection = self.cache_manager["qwentts"]
        if not cache_collection.enabled:
            return [
                (result, self.sample_rate)
                for result in self._render_batch_native_sync(batch)
            ]

        cached_outputs: dict[int, tuple[ScriptRenderResult, int]] = {}
        uncached_batch: list[tuple[int, RegisteredRenderRequest]] = []

        for index, registration in enumerate(batch):
            hit = cache_collection.find(
                registration.request,
                validate=lambda hit, request=registration.request: (
                    self._validate_cached_native_result(request, hit)
                ),
            )
            if hit is not None:
                cached_outputs[index] = self._load_cached_native_result(hit)
                continue
            uncached_batch.append((index, registration))

        if uncached_batch:
            rendered = self._render_batch_native_sync(
                [registration for _, registration in uncached_batch]
            )
            for (index, registration), result in zip(uncached_batch, rendered, strict=True):
                hit = cache_collection.get_or_create(
                    registration.request,
                    lambda key, collection, request=registration.request, cached_result=result: (
                        self._store_cached_native_result(
                            key,
                            collection,
                            request,
                            cached_result,
                        )
                    ),
                    validate=lambda hit, request=registration.request: (
                        self._validate_cached_native_result(request, hit)
                    ),
                )
                cached_outputs[index] = self._load_cached_native_result(hit)

        return [cached_outputs[index] for index in range(len(batch))]

    def _render_batch_native_sync(
        self,
        batch: Sequence[RegisteredRenderRequest],
    ) -> list[ScriptRenderResult]:
        parsed_scripts = [self._script_lines(registration.request) for registration in batch]
        if not any(parsed_scripts):
            return [
                ScriptRenderResult(audio=np.zeros(0, dtype=np.float32), dialogue_line_start_positions=())
                for _ in batch
            ]

        voice_paths = {
            str(Path(line.speaker.resolved_path).expanduser().resolve())
            for registration in batch
            for line in self._script_lines(registration.request)
        }
        prompt_items_by_voice = self._prompt_items_by_voice_sync(sorted(voice_paths))

        line_texts: list[str] = []
        line_prompts: list[VoiceClonePromptItem] = []
        line_targets: list[int] = []
        for script_index, (registration, script_lines) in enumerate(
            zip(batch, parsed_scripts, strict=True)
        ):
            for line in script_lines:
                voice_path = str(Path(line.speaker.resolved_path).expanduser().resolve())
                line_texts.append(line.spoken_text)
                line_prompts.append(prompt_items_by_voice[voice_path][0])
                line_targets.append(script_index)

        if not line_texts:
            return [
                ScriptRenderResult(audio=np.zeros(0, dtype=np.float32), dialogue_line_start_positions=())
                for _ in batch
            ]

        native_lines = self._generate_line_batch_native_sync(line_texts, line_prompts)
        rendered_by_script: list[list[np.ndarray]] = [[] for _ in batch]
        line_starts_by_script: list[list[float]] = [[] for _ in batch]
        native_sample_rate = self.sample_rate
        for script_index, audio in zip(line_targets, native_lines, strict=True):
            script_frames = sum(clip.shape[0] for clip in rendered_by_script[script_index])
            line_starts_by_script[script_index].append(float(script_frames) / native_sample_rate)
            rendered_by_script[script_index].append(audio)

        return [
            ScriptRenderResult(
                audio=self._concatenate_script_audio(clips),
                dialogue_line_start_positions=tuple(line_starts),
            )
            for clips, line_starts in zip(rendered_by_script, line_starts_by_script, strict=True)
        ]

    def _validate_cached_native_result(
        self,
        request: ScriptRenderRequest,
        hit: dict[str, Path],
    ) -> bool:
        if not request.validate_cache_hit(hit):
            return False
        try:
            payload = json.loads(hit["json"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        positions = payload.get("dialogue_line_start_positions")
        if positions is None:
            return False
        return len(positions) == len(self._script_lines(request))

    def _load_cached_native_result(
        self,
        hit: dict[str, Path],
    ) -> tuple[ScriptRenderResult, int]:
        payload = json.loads(hit["json"].read_text(encoding="utf-8"))
        audio, sample_rate = sf.read(hit["wav"], dtype="float32", always_2d=False)
        positions = tuple(float(position) for position in payload["dialogue_line_start_positions"])
        return (
            ScriptRenderResult(
                audio=self._normalize_audio_array(audio),
                dialogue_line_start_positions=positions,
            ),
            int(sample_rate),
        )

    def _store_cached_native_result(
        self,
        key: CacheKey,
        collection: CacheCollection,
        request: ScriptRenderRequest,
        result: ScriptRenderResult,
    ) -> None:
        wav_path = collection.path_for_subtype(key, "wav")
        json_path = collection.path_for_subtype(key, "json")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, result.audio, self.sample_rate)
        json_path.write_text(
            json.dumps(
                request.build_cache_payload(
                    frame_count=int(result.audio.shape[0]),
                    sample_rate=self.sample_rate,
                    dialogue_line_start_positions=tuple(result.dialogue_line_start_positions),
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _generate_line_batch_native_sync(
        self,
        texts: Sequence[str],
        prompt_items: Sequence[VoiceClonePromptItem],
    ) -> list[np.ndarray]:
        model = self._ensure_loaded()
        generated: list[np.ndarray] = []
        for start in range(0, len(texts), self.config.resolved_batch_size):
            end = start + self.config.resolved_batch_size
            batch_texts = list(texts[start:end])
            wavs, sample_rate = model.generate_voice_clone(
                text=batch_texts,
                language=[_QWEN_LANGUAGE] * len(batch_texts),
                voice_clone_prompt=list(prompt_items[start:end]),
                non_streaming_mode=True,
            )
            if self._sample_rate is None:
                self._sample_rate = int(sample_rate)
            elif int(sample_rate) != self._sample_rate:
                raise RuntimeError(
                    f"Qwen returned sample rate {sample_rate}, expected {self._sample_rate}"
                )
            generated.extend(self._normalize_audio_array(wav) for wav in wavs)
        return generated

    def _script_lines(
        self,
        request: ScriptRenderRequest,
    ) -> list[DialogueLine]:
        return [
            line
            for line in request.dialogue_lines
            if line.spoken_text.strip()
        ]

    def _prompt_items_by_voice_sync(
        self,
        voice_paths: Sequence[str],
    ) -> dict[str, list[VoiceClonePromptItem]]:
        return {
            voice_path: self._prompt_items_for_voice_sync(voice_path)
            for voice_path in voice_paths
        }

    def _prompt_items_for_voice_sync(
        self,
        voice_path: str,
    ) -> list[VoiceClonePromptItem]:
        with self._prompt_cache_lock:
            cached = self._voice_prompt_cache.get(voice_path)
            if cached is not None:
                return cached

        cache_path = self._prompt_cache_path(Path(voice_path))
        if cache_path.is_file():
            prompt_items = self._deserialize_prompt_items(
                torch.load(cache_path, map_location="cpu")
            )
        else:
            prompt_items = self._build_prompt_items_for_voice_sync(voice_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._serialize_prompt_items(prompt_items), cache_path)

        with self._prompt_cache_lock:
            self._voice_prompt_cache[voice_path] = prompt_items
        return prompt_items

    def _build_prompt_items_for_voice_sync(
        self,
        voice_path: str,
    ) -> list[VoiceClonePromptItem]:
        model = self._ensure_loaded()
        reference_audio, sample_rate = self._preprocessed_voice_reference_sync(voice_path)
        transcript = self.whisperx_resource.transcribe_audio_sample_sync(
            reference_audio,
            sample_rate,
        )
        return list(
            model.create_voice_clone_prompt(
                ref_audio=(reference_audio, sample_rate),
                ref_text=transcript,
                x_vector_only_mode=False,
            )
        )

    def _preprocessed_voice_reference_sync(
        self,
        voice_path: str,
    ) -> tuple[np.ndarray, int]:
        return load_preprocessed_voice_reference(voice_path)

    def _serialize_prompt_items(
        self,
        prompt_items: Sequence[VoiceClonePromptItem],
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for item in prompt_items:
            payload.append(
                {
                    "ref_code": (
                        item.ref_code.detach().cpu()
                        if torch.is_tensor(item.ref_code)
                        else None
                    ),
                    "ref_spk_embedding": item.ref_spk_embedding.detach().cpu(),
                    "x_vector_only_mode": bool(item.x_vector_only_mode),
                    "icl_mode": bool(item.icl_mode),
                    "ref_text": item.ref_text,
                }
            )
        return payload

    def _deserialize_prompt_items(
        self,
        payload: Sequence[dict[str, Any]],
    ) -> list[VoiceClonePromptItem]:
        prompt_type = self._voice_clone_prompt_item_cls()
        return [
            prompt_type(
                ref_code=entry["ref_code"],
                ref_spk_embedding=entry["ref_spk_embedding"],
                x_vector_only_mode=bool(entry["x_vector_only_mode"]),
                icl_mode=bool(entry["icl_mode"]),
                ref_text=entry.get("ref_text"),
            )
            for entry in payload
        ]

    def _voice_clone_prompt_item_cls(self):
        with self._load_lock:
            if self._voice_clone_prompt_item_type is None:
                os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
                from qwen_tts import VoiceClonePromptItem

                self._voice_clone_prompt_item_type = VoiceClonePromptItem
            return self._voice_clone_prompt_item_type

    def _ensure_loaded(self) -> Qwen3TTSModel:
        with self._load_lock:
            if self._model is not None:
                return self._model

            with shared_model_load():
                if self._model is not None:
                    return self._model

                os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
                from qwen_tts import Qwen3TTSModel

                load_dtype, attn_implementation = self._load_settings_for_device(self.device)
                try:
                    model = Qwen3TTSModel.from_pretrained(
                        _QWEN_MODEL_NAME,
                        device_map=self._device_map_for_device(self.device),
                        dtype=load_dtype,
                        attn_implementation=attn_implementation,
                    )
                except Exception:
                    if attn_implementation != "flash_attention_2":
                        raise
                    model = Qwen3TTSModel.from_pretrained(
                        _QWEN_MODEL_NAME,
                        device_map=self._device_map_for_device(self.device),
                        dtype=load_dtype,
                        attn_implementation=None,
                    )
                self._sample_rate = int(getattr(model.model.config, "sample_rate", 24000))
                self._model = model
                return model

    def _prompt_cache_path(self, voice_path: Path) -> Path:
        voice_path = voice_path.expanduser().resolve()
        try:
            relative = voice_path.relative_to(self.config.resolved_voice_directory.resolve())
        except ValueError:
            relative = Path("external") / voice_path.relative_to(voice_path.anchor)
        versioned_relative = Path(VOICE_PREPROCESS_VERSION) / relative
        return _QWEN_PROMPT_CACHE_DIRECTORY / versioned_relative.with_suffix(
            f"{relative.suffix}.pt"
        )

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

    def _concatenate_script_audio(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        if not clips:
            return np.zeros(0, dtype=np.float32)
        if len(clips) == 1:
            return np.ascontiguousarray(clips[0], dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(clips), dtype=np.float32)

    def _normalize_audio_array(self, audio: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(audio):
            array = audio.detach().float().cpu().numpy()
        else:
            array = np.asarray(audio, dtype=np.float32)
        array = np.squeeze(array)
        if array.ndim != 1:
            raise ValueError(f"Expected mono audio after generation, got {array.shape!r}")
        return np.ascontiguousarray(array, dtype=np.float32)

    def _normalize_device(self, device: str) -> str:
        normalized = (device or "cpu").lower()
        if normalized == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if normalized == "mps":
            return "cpu"
        if normalized not in {"cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {device}")
        return normalized

    def _load_settings_for_device(self, device: str) -> tuple[torch.dtype, str | None]:
        if device == "cuda":
            return torch.bfloat16, "flash_attention_2"
        return torch.float32, None

    def _device_map_for_device(self, device: str) -> str:
        if device == "cuda":
            return "cuda:0"
        return "cpu"


__all__ = ["QwenTtsResource"]
