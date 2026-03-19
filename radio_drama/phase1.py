from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly

from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

from .carthage_support import AsyncInjectable, AsyncInjector, InjectionKey, Injector, inject
from .document import ProductionNode, ScriptNode, SpeakerMapNode
from .errors import DocumentError
from .rendering import ProductionResult, RenderResult
from sandbox_asyncio_workaround import await_with_codex_asyncio_workaround


_DEFAULT_MODEL_PATH = "/srv/ai/models/vibevoice/vibevoice-large"
_SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
}
_SPEAKER_LINE_RE = re.compile(r"^([^:\n]+?)\s*:\s*(.*)$")


@dataclass(slots=True)
class ProductionConfig:
    voice_directory: Path
    model_name: str = _DEFAULT_MODEL_PATH
    output_sample_rate: int = 48000
    output_channels: int = 2
    batch_size: int = 10
    device: str | None = "cuda"
    cfg_scale: float = 1.3
    disable_prefill: bool = False
    ddpm_inference_steps: int = 5

    def __post_init__(self) -> None:
        self.voice_directory = Path(self.voice_directory).expanduser()


@dataclass(frozen=True, slots=True)
class SpeakerVoiceReference:
    authored_name: str
    voice_name: str
    resolved_path: Path


@dataclass(frozen=True, slots=True)
class DialogueLine:
    authored_name: str
    spoken_text: str


@dataclass(frozen=True, slots=True)
class ScriptRenderRequest:
    normalized_script: str
    voice_samples: tuple[str, ...]


@inject(config=ProductionConfig)
class SpeakerMapPlan(AsyncInjectable):
    def __init__(self, node: SpeakerMapNode, **kwargs) -> None:
        super().__init__(**kwargs)
        self.node = node
        self._voices_by_key: dict[str, SpeakerVoiceReference] = {}

    async def async_ready(self):
        loaded = yaml.safe_load(self.node.normalized_text_content)
        if not isinstance(loaded, dict):
            raise DocumentError(
                "The <speaker-map> YAML must be a mapping of speaker names to voice names",
                node=self.node,
            )
        if not loaded:
            raise DocumentError("The <speaker-map> did not define any speakers", node=self.node)

        voices_by_key: dict[str, SpeakerVoiceReference] = {}
        for speaker_name, voice_name in loaded.items():
            if not isinstance(speaker_name, str) or not isinstance(voice_name, str):
                raise DocumentError(
                    "Speaker names and voice names in <speaker-map> must be strings",
                    node=self.node,
                )
            normalized_speaker = speaker_name.strip()
            normalized_voice = voice_name.strip()
            if not normalized_speaker or not normalized_voice:
                raise DocumentError(
                    "Speaker names and voice names in <speaker-map> cannot be empty",
                    node=self.node,
                )
            key = normalized_speaker.lower()
            if key in voices_by_key:
                raise DocumentError(
                    f"Speaker {speaker_name!r} is defined more than once in <speaker-map>",
                    node=self.node,
                )
            voices_by_key[key] = SpeakerVoiceReference(
                authored_name=normalized_speaker,
                voice_name=normalized_voice,
                resolved_path=self._resolve_voice_path(normalized_speaker, normalized_voice),
            )

        self._voices_by_key = voices_by_key
        return await super().async_ready()

    def lookup(self, speaker_name: str) -> SpeakerVoiceReference:
        return self._voices_by_key[speaker_name.strip().lower()]

    @property
    def voices_by_key(self) -> Mapping[str, SpeakerVoiceReference]:
        return self._voices_by_key

    def _resolve_voice_path(self, speaker_name: str, voice_name: str) -> Path:
        direct_candidates = [
            Path(voice_name).expanduser(),
            (self.config.voice_directory / voice_name).expanduser(),
        ]
        for candidate in direct_candidates:
            if candidate.is_file():
                return candidate

        voice_catalog = self._load_voice_catalog()
        candidate_keys = [
            voice_name,
            Path(voice_name).name,
            Path(voice_name).stem,
            voice_name.lower(),
            Path(voice_name).name.lower(),
            Path(voice_name).stem.lower(),
        ]
        for candidate in candidate_keys:
            resolved = voice_catalog.get(candidate)
            if resolved is not None:
                return resolved

        available = ", ".join(sorted({path.name for path in voice_catalog.values()}))
        raise DocumentError(
            f"Voice {voice_name!r} for speaker {speaker_name!r} was not found in "
            f"{self.config.voice_directory}. Available voices: {available}",
            node=self.node,
        )

    def _load_voice_catalog(self) -> dict[str, Path]:
        if not self.config.voice_directory.is_dir():
            raise DocumentError(
                f"Voice directory does not exist: {self.config.voice_directory}",
                node=self.node,
            )

        catalog: dict[str, Path] = {}
        for child in sorted(self.config.voice_directory.iterdir()):
            if not child.is_file() or child.suffix.lower() not in _SUPPORTED_AUDIO_EXTENSIONS:
                continue
            catalog.setdefault(child.name, child)
            catalog.setdefault(child.stem, child)
            catalog.setdefault(child.name.lower(), child)
            catalog.setdefault(child.stem.lower(), child)

        if not catalog:
            raise DocumentError(
                f"No supported voice files were found in {self.config.voice_directory}",
                node=self.node,
            )
        return catalog


@dataclass(slots=True)
class ScriptPlan:
    node: ScriptNode
    dialogue_lines: list[DialogueLine]
    ordered_speakers: list[SpeakerVoiceReference]
    render_request: ScriptRenderRequest

    @classmethod
    def from_node(cls, node: ScriptNode, speaker_map_plan: SpeakerMapPlan) -> "ScriptPlan":
        dialogue_lines: list[DialogueLine] = []
        ordered_speakers_by_key: dict[str, SpeakerVoiceReference] = {}

        for raw_line in node.normalized_text_content.splitlines():
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            match = _SPEAKER_LINE_RE.match(stripped_line)
            if match is None:
                raise DocumentError(
                    "Scripts may contain only `speaker: text` lines",
                    node=node,
                )
            script_speaker = match.group(1).strip()
            spoken_text = match.group(2).strip()
            if not spoken_text:
                raise DocumentError(
                    "Speaker lines in <script> must include non-empty text",
                    node=node,
                )
            try:
                speaker_ref = speaker_map_plan.lookup(script_speaker)
            except KeyError:
                known_speakers = ", ".join(
                    ref.authored_name for ref in speaker_map_plan.voices_by_key.values()
                )
                raise DocumentError(
                    f"Speaker {script_speaker!r} is used in a script but missing from <speaker-map>. "
                    f"Known speakers: {known_speakers}",
                    node=node,
                ) from None
            ordered_speakers_by_key.setdefault(script_speaker.lower(), speaker_ref)
            dialogue_lines.append(
                DialogueLine(
                    authored_name=speaker_ref.authored_name,
                    spoken_text=spoken_text,
                )
            )

        if not dialogue_lines:
            raise DocumentError(
                "A <script> must contain at least one `speaker: text` line",
                node=node,
            )

        ordered_speakers = list(ordered_speakers_by_key.values())
        if len(ordered_speakers) > 4:
            raise DocumentError(
                f"A <script> uses {len(ordered_speakers)} speakers, but VibeVoice supports at most 4",
                node=node,
            )

        local_speaker_ids = {
            speaker.authored_name.lower(): index + 1
            for index, speaker in enumerate(ordered_speakers)
        }
        normalized_script = "\n".join(
            f"Speaker {local_speaker_ids[line.authored_name.lower()]}: {line.spoken_text}"
            for line in dialogue_lines
        ).replace("’", "'")
        render_request = ScriptRenderRequest(
            normalized_script=normalized_script,
            voice_samples=tuple(str(ref.resolved_path) for ref in ordered_speakers),
        )
        return cls(
            node=node,
            dialogue_lines=dialogue_lines,
            ordered_speakers=ordered_speakers,
            render_request=render_request,
        )


@dataclass(slots=True)
class _PendingRender:
    script_plan: ScriptPlan
    future: asyncio.Future


@inject(config=ProductionConfig)
class VibeVoiceResource(AsyncInjectable):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = self._normalize_device(self.config.device or self._detect_device())
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

    async def render_script(self, script_plan: ScriptPlan) -> RenderResult:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        async with self._pending_lock:
            self._pending.append(_PendingRender(script_plan=script_plan, future=future))
            if self._drain_task is None or self._drain_task.done():
                self._drain_task = asyncio.create_task(self._drain_pending())
        return await future

    async def _drain_pending(self) -> None:
        while True:
            await asyncio.sleep(0)
            async with self._pending_lock:
                if not self._pending:
                    self._drain_task = None
                    return
                batch = self._pending[: self.config.batch_size]
                del self._pending[: self.config.batch_size]

            try:
                thread_awaitable = asyncio.to_thread(self._render_batch_sync, batch)
                if os.environ.get("CODEX_CI") == "1":
                    rendered_audios = await await_with_codex_asyncio_workaround(
                        thread_awaitable,
                        period=0.05,
                    )
                else:
                    rendered_audios = await thread_awaitable
            except Exception as exc:
                for pending in batch:
                    if not pending.future.done():
                        pending.future.set_exception(exc)
                continue

            for pending, audio in zip(batch, rendered_audios, strict=True):
                if not pending.future.done():
                    pending.future.set_result(
                        RenderResult(audio=audio, sample_rate=self.sample_rate)
                    )

    def _render_batch_sync(self, batch: Sequence[_PendingRender]) -> list[np.ndarray]:
        requests = [pending.script_plan.render_request for pending in batch]
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
                cfg_scale=self.config.cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                is_prefill=not self.config.disable_prefill,
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

        processor = VibeVoiceProcessor.from_pretrained(self.config.model_name)
        self._sample_rate = getattr(processor.audio_processor, "sampling_rate", 24000)

        load_dtype, attn_implementation = self._load_settings_for_device(self.device)
        try:
            model = self._load_model(
                model_name=self.config.model_name,
                device=self.device,
                load_dtype=load_dtype,
                attn_implementation=attn_implementation,
            )
        except Exception:
            if attn_implementation != "flash_attention_2":
                raise
            model = self._load_model(
                model_name=self.config.model_name,
                device=self.device,
                load_dtype=load_dtype,
                attn_implementation="sdpa",
            )

        model.eval()
        model.set_ddpm_inference_steps(num_steps=self.config.ddpm_inference_steps)

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
        normalized = device.lower()
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


@inject(config=ProductionConfig)
class OutputFormatResource(AsyncInjectable):
    def convert(self, render_result: RenderResult) -> ProductionResult:
        audio = np.asarray(render_result.audio, dtype=np.float32)
        if audio.ndim == 0:
            audio = audio.reshape(1)
        if render_result.sample_rate != self.config.output_sample_rate:
            audio = self._resample(audio, render_result.sample_rate, self.config.output_sample_rate)
        audio = self._convert_channels(audio, self.config.output_channels)
        return ProductionResult(
            audio=audio,
            sample_rate=self.config.output_sample_rate,
            pre_margin=render_result.pre_margin,
            post_margin=render_result.post_margin,
            pre_gap=render_result.pre_gap,
            post_gap=render_result.post_gap,
        )

    def _resample(
        self,
        audio: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        factor = gcd(source_rate, target_rate)
        up = target_rate // factor
        down = source_rate // factor
        if audio.ndim == 1:
            return np.ascontiguousarray(resample_poly(audio, up, down), dtype=np.float32)
        return np.ascontiguousarray(
            resample_poly(audio, up, down, axis=0),
            dtype=np.float32,
        )

    def _convert_channels(self, audio: np.ndarray, channels: int) -> np.ndarray:
        if channels < 1:
            raise ValueError("output_channels must be at least 1")
        if channels == 1:
            if audio.ndim == 1:
                return np.ascontiguousarray(audio, dtype=np.float32)
            if audio.shape[1] == 1:
                return np.ascontiguousarray(audio[:, 0], dtype=np.float32)
            return np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32)
        if audio.ndim == 1:
            mono = audio[:, np.newaxis]
        elif audio.shape[1] == 1:
            mono = audio
        elif audio.shape[1] == channels:
            return np.ascontiguousarray(audio, dtype=np.float32)
        else:
            mono = audio.mean(axis=1, keepdims=True)
        return np.ascontiguousarray(np.repeat(mono, channels, axis=1), dtype=np.float32)


@inject(vibevoice_resource=VibeVoiceResource, output_format_resource=OutputFormatResource)
class ProductionPlan(AsyncInjectable):
    def __init__(
        self,
        node: ProductionNode,
        speaker_map_plan: SpeakerMapPlan,
        script_plans: Sequence[ScriptPlan],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.node = node
        self.speaker_map_plan = speaker_map_plan
        self.script_plans = list(script_plans)

    async def render(self) -> ProductionResult:
        render_results = await asyncio.gather(
            *(self.vibevoice_resource.render_script(script_plan) for script_plan in self.script_plans)
        )
        combined_audio = np.concatenate(
            [result.audio for result in render_results],
            axis=0,
        )
        combined = RenderResult(audio=combined_audio, sample_rate=render_results[0].sample_rate)
        return self.output_format_resource.convert(combined)


async def plan_production(
    production_node: ProductionNode,
    ainjector: AsyncInjector,
) -> ProductionPlan:
    speaker_map_plan = await ainjector(SpeakerMapPlan, node=production_node.speaker_map_node)
    script_plans = [
        ScriptPlan.from_node(script_node, speaker_map_plan)
        for script_node in production_node.script_nodes
    ]
    return await ainjector(
        ProductionPlan,
        node=production_node,
        speaker_map_plan=speaker_map_plan,
        script_plans=script_plans,
    )


async def render_production(
    production_node: ProductionNode,
    config: ProductionConfig,
) -> ProductionResult:
    injector = Injector()
    injector.add_provider(config)
    injector.replace_provider(
        InjectionKey(asyncio.AbstractEventLoop),
        asyncio.get_running_loop(),
        close=False,
    )
    try:
        ainjector = injector(AsyncInjector)
        production_plan = await plan_production(production_node, ainjector)
        return await production_plan.render()
    finally:
        injector.close()


async def render_production_file(
    xml_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    config: ProductionConfig,
) -> ProductionResult:
    from .document import parse_production_file

    production_node = parse_production_file(xml_path)
    result = await render_production(production_node, config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, result.audio, result.sample_rate)
    return result
