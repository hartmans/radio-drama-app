from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_PATH = "/srv/ai/models/vibevoice/vibevoice-large"
DEFAULT_VOICE_DIRECTORY = Path("./voices")
DEFAULT_SOUNDS_DIRECTORY = Path("./sounds")
DEFAULT_OUTPUT_SAMPLE_RATE = 48000
DEFAULT_OUTPUT_CHANNELS = 2
DEFAULT_BATCH_SIZE = 10
DEFAULT_DEVICE = "cuda"
DEFAULT_CFG_SCALE = 1.2
DEFAULT_DISABLE_PREFILL = False
DEFAULT_DDPM_INFERENCE_STEPS = 8
MODEL_NATIVE_SAMPLE_RATE = 24000


@dataclass(slots=True)
class ProductionConfig:
    voice_directory: Path | None = None
    sounds_directory: Path | None = None
    model_name: str | None = None
    output_sample_rate: int | None = None
    output_channels: int | None = None
    batch_size: int | None = None
    device: str | None = None
    cfg_scale: float | None = None
    disable_prefill: bool | None = None
    ddpm_inference_steps: int | None = None

    def __post_init__(self) -> None:
        if self.voice_directory is not None:
            self.voice_directory = Path(self.voice_directory).expanduser()
        if self.sounds_directory is not None:
            self.sounds_directory = Path(self.sounds_directory).expanduser()

    @property
    def resolved_voice_directory(self) -> Path:
        return (self.voice_directory or DEFAULT_VOICE_DIRECTORY).expanduser()

    @property
    def resolved_sounds_directory(self) -> Path:
        return (self.sounds_directory or DEFAULT_SOUNDS_DIRECTORY).expanduser()

    @property
    def resolved_model_name(self) -> str:
        return self.model_name or DEFAULT_MODEL_PATH

    @property
    def resolved_output_sample_rate(self) -> int:
        return self.output_sample_rate or DEFAULT_OUTPUT_SAMPLE_RATE

    @property
    def resolved_output_channels(self) -> int:
        return self.output_channels or DEFAULT_OUTPUT_CHANNELS

    @property
    def resolved_batch_size(self) -> int:
        return self.batch_size or DEFAULT_BATCH_SIZE

    @property
    def resolved_device(self) -> str:
        return self.device or DEFAULT_DEVICE

    @property
    def resolved_cfg_scale(self) -> float:
        return self.cfg_scale if self.cfg_scale is not None else DEFAULT_CFG_SCALE

    @property
    def resolved_disable_prefill(self) -> bool:
        return (
            self.disable_prefill
            if self.disable_prefill is not None
            else DEFAULT_DISABLE_PREFILL
        )

    @property
    def resolved_ddpm_inference_steps(self) -> int:
        return (
            self.ddpm_inference_steps
            if self.ddpm_inference_steps is not None
            else DEFAULT_DDPM_INFERENCE_STEPS
        )
