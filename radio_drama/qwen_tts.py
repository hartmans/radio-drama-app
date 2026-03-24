from __future__ import annotations

import asyncio
import os
import re
import weakref
from pathlib import Path
from threading import Lock, RLock
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from carthage.dependency_injection import AsyncInjectable, inject

from .audio import convert_audio_format
from .config import ProductionConfig
from .forced_alignment import WhisperXResource
from .planning import ScriptRenderRequest
from .rendering import RenderResult
from .vibevoice import RegisteredRenderRequest


if TYPE_CHECKING:
    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem


_QWEN_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
_QWEN_LANGUAGE = "English"
_QWEN_PROMPT_CACHE_DIRECTORY = Path("~/.cache/radio_drama/qwen_prompts").expanduser()
_SPEAKER_LINE_RE = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.+)$")


class _PendingRender:
    __slots__ = ("registration_ref",)

    def __init__(self, registration: RegisteredRenderRequest) -> None:
        self.registration_ref = weakref.ref(registration)

    def registration(self) -> RegisteredRenderRequest | None:
        return self.registration_ref()


@inject(config=ProductionConfig, whisperx_resource=WhisperXResource)
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
        parsed_scripts = [self._parse_script_lines(registration.request) for registration in batch]
        if not any(parsed_scripts):
            return [np.zeros(0, dtype=np.float32) for _ in batch]

        voice_paths = {
            str(Path(voice_sample).expanduser().resolve())
            for registration in batch
            for voice_sample in registration.request.voice_samples
        }
        prompt_items_by_voice = self._prompt_items_by_voice_sync(sorted(voice_paths))

        line_texts: list[str] = []
        line_prompts: list[VoiceClonePromptItem] = []
        line_targets: list[int] = []
        for script_index, (registration, script_lines) in enumerate(
            zip(batch, parsed_scripts, strict=True)
        ):
            voice_samples = registration.request.voice_samples
            for speaker_index, line_text in script_lines:
                voice_path = str(Path(voice_samples[speaker_index]).expanduser().resolve())
                line_texts.append(line_text)
                line_prompts.append(prompt_items_by_voice[voice_path][0])
                line_targets.append(script_index)

        native_lines = self._generate_line_batch_native_sync(line_texts, line_prompts)
        rendered_by_script: list[list[np.ndarray]] = [[] for _ in batch]
        for script_index, audio in zip(line_targets, native_lines, strict=True):
            rendered_by_script[script_index].append(audio)

        return [
            self._concatenate_script_audio(clips)
            for clips in rendered_by_script
        ]

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

    def _parse_script_lines(
        self,
        request: ScriptRenderRequest,
    ) -> list[tuple[int, str]]:
        parsed: list[tuple[int, str]] = []
        for raw_line in request.normalized_script.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = _SPEAKER_LINE_RE.match(line)
            if match is None:
                raise ValueError(f"Unsupported normalized script line for Qwen TTS: {line!r}")
            speaker_number = int(match.group(1))
            if speaker_number < 1 or speaker_number > len(request.voice_samples):
                raise ValueError(
                    f"Speaker index {speaker_number} is outside the available voice sample range"
                )
            parsed.append((speaker_number - 1, match.group(2).strip()))
        return parsed

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
        transcript = self.whisperx_resource.transcribe_audio_sample_sync(voice_path)
        return list(
            model.create_voice_clone_prompt(
                ref_audio=voice_path,
                ref_text=transcript,
                x_vector_only_mode=False,
            )
        )

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
        return _QWEN_PROMPT_CACHE_DIRECTORY / relative.with_suffix(
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
