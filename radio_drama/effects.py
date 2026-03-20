from __future__ import annotations

import asyncio
import math
import subprocess
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np
from carthage.dependency_injection import inject
from scipy.signal import butter, sosfiltfilt

from .audio import normalize_audio_array
from .config import ProductionConfig
from .planning import AudioPlan
from .rendering import RenderResult


class EffectStage(Protocol):
    """One audio transformation within an effect chain."""

    name: str
    backend: str

    def apply(self, audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
        """Return transformed stereo audio in production format."""


@dataclass(frozen=True, slots=True)
class CallableEffectStage:
    """Simple Python-callable-backed effect stage."""

    name: str
    processor: Callable[[np.ndarray, int], np.ndarray]
    backend: str = "python"

    def apply(self, audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
        return normalize_audio_array(self.processor(audio, sample_rate))


@dataclass(frozen=True, slots=True)
class PedalboardEffectStage:
    """Pedalboard-backed stage loaded only when actually used."""

    name: str
    board_factory: Callable[[], object]
    backend: str = "pedalboard"

    def apply(self, audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
        try:
            board = self.board_factory()
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Pedalboard is required for this effect stage") from exc
        processed = board(np.asarray(audio, dtype=np.float32).T, sample_rate, reset=True)
        return normalize_audio_array(np.asarray(processed, dtype=np.float32).T)


@dataclass(frozen=True, slots=True)
class FFmpegFilterEffectStage:
    """FFmpeg-backed stage for effects that are easiest to express as filters."""

    name: str
    filter_graph: str
    backend: str = "ffmpeg"

    def apply(self, audio: np.ndarray, *, sample_rate: int) -> np.ndarray:
        from scipy.io import wavfile

        normalized = normalize_audio_array(audio)
        with tempfile.TemporaryDirectory(prefix="radio-drama-ffmpeg-") as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            output_path = temp_path / "output.wav"
            wavfile.write(input_path, sample_rate, normalized)
            command = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-af",
                self.filter_graph,
                "-c:a",
                "pcm_f32le",
                str(output_path),
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("ffmpeg is required for this effect stage") from exc
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.strip() or exc.stdout.strip()
                raise RuntimeError(f"ffmpeg effect stage failed: {stderr}") from exc
            rendered_sample_rate, rendered = wavfile.read(output_path)
        if rendered_sample_rate != sample_rate:
            raise RuntimeError(
                f"ffmpeg effect stage changed sample rate from {sample_rate} to {rendered_sample_rate}"
            )
        return normalize_audio_array(rendered)


@dataclass(frozen=True, slots=True)
class EffectChain:
    """Named, reusable chain of stages that transforms one rendered clip."""

    name: str
    stages: tuple[EffectStage, ...]

    def apply(self, result: RenderResult, *, sample_rate: int) -> RenderResult:
        audio = normalize_audio_array(result.audio)
        for stage in self.stages:
            audio = stage.apply(audio, sample_rate=sample_rate)
        return type(result)(
            audio=audio,
            pre_margin=result.pre_margin,
            post_margin=result.post_margin,
            pre_gap=result.pre_gap,
            post_gap=result.post_gap,
        )

    async def render(self, result: RenderResult, *, sample_rate: int) -> RenderResult:
        return await asyncio.to_thread(self.apply, result, sample_rate=sample_rate)


@inject(config=ProductionConfig)
class PresetPlan(AudioPlan):
    """Audio plan wrapper that applies a named preset at render time."""

    def __init__(
        self,
        node,
        audio_plan: AudioPlan,
        preset_name: str,
        **kwargs,
    ) -> None:
        super().__init__(node=node, set_gap=False, **kwargs)
        self.audio_plan = audio_plan
        self.preset_name = preset_name

    def leaf_audio_plans(self) -> list[AudioPlan]:
        return self.audio_plan.leaf_audio_plans()

    def __getattr__(self, name: str):
        return getattr(self.audio_plan, name)

    async def render_node(self) -> RenderResult:
        base_result = await self.audio_plan.render()
        try:
            chain = build_named_effect_chain(self.preset_name)
        except KeyError as exc:
            available = ", ".join(sorted(available_effect_chains()))
            raise self.document_error(
                f"Unknown preset {self.preset_name!r}. Available presets: {available}"
            ) from exc
        return await chain.render(
            base_result,
            sample_rate=self.config.resolved_output_sample_rate,
        )


def callable_stage(
    name: str,
    processor: Callable[[np.ndarray, int], np.ndarray],
    *,
    backend: str,
) -> CallableEffectStage:
    return CallableEffectStage(name=name, processor=processor, backend=backend)


def numpy_stage(
    name: str,
    processor: Callable[[np.ndarray, int], np.ndarray],
) -> CallableEffectStage:
    return callable_stage(name, processor, backend="numpy")


def scipy_signal_stage(
    name: str,
    processor: Callable[[np.ndarray, int], np.ndarray],
) -> CallableEffectStage:
    return callable_stage(name, processor, backend="scipy.signal")


def available_effect_chains() -> tuple[str, ...]:
    return tuple(sorted(_PRESET_CHAINS))


def normalize_effect_chain_name(name: str) -> str:
    normalized_name = name.strip().lower()
    return _PRESET_ALIASES.get(normalized_name, normalized_name)


def build_named_effect_chain(name: str) -> EffectChain:
    return _PRESET_CHAINS[normalize_effect_chain_name(name)]


def _db_to_gain(decibels: float) -> float:
    return float(10.0 ** (decibels / 20.0))


def _filter_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    btype: str,
    cutoff_hz: float,
    order: int = 2,
) -> np.ndarray:
    normalized = normalize_audio_array(audio)
    nyquist = sample_rate / 2.0
    normalized_cutoff = min(max(cutoff_hz / nyquist, 1e-5), 0.999)
    sos = butter(order, normalized_cutoff, btype=btype, output="sos")
    return normalize_audio_array(sosfiltfilt(sos, normalized, axis=0))


def _tilt_tone(
    audio: np.ndarray,
    sample_rate: int,
    *,
    low_band_db: float = 0.0,
    high_band_db: float = 0.0,
) -> np.ndarray:
    normalized = normalize_audio_array(audio)
    low_band = _filter_audio(normalized, sample_rate, btype="lowpass", cutoff_hz=220.0)
    high_band = _filter_audio(normalized, sample_rate, btype="highpass", cutoff_hz=3200.0)
    mid_band = normalized - low_band - high_band
    return normalize_audio_array(
        low_band * _db_to_gain(low_band_db)
        + mid_band
        + high_band * _db_to_gain(high_band_db)
    )


def _compress_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float = 0.0,
) -> np.ndarray:
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    normalized = normalize_audio_array(audio)
    envelope = np.max(np.abs(normalized), axis=1)
    attack_coeff = math.exp(-1.0 / max(sample_rate * attack_ms / 1000.0, 1.0))
    release_coeff = math.exp(-1.0 / max(sample_rate * release_ms / 1000.0, 1.0))
    smoothed = np.zeros_like(envelope)
    level = 0.0
    for index, sample in enumerate(envelope):
        coeff = attack_coeff if sample > level else release_coeff
        level = coeff * level + (1.0 - coeff) * sample
        smoothed[index] = level

    envelope_db = 20.0 * np.log10(np.maximum(smoothed, 1e-6))
    gain_reduction_db = np.where(
        envelope_db > threshold_db,
        (threshold_db + (envelope_db - threshold_db) / ratio) - envelope_db,
        0.0,
    )
    gain = np.power(10.0, (gain_reduction_db + makeup_db) / 20.0).astype(np.float32)
    return normalize_audio_array(normalized * gain[:, np.newaxis])


def _mid_side_mix(
    audio: np.ndarray,
    sample_rate: int,
    *,
    mid_gain: float,
    side_gain: float,
) -> np.ndarray:
    del sample_rate
    normalized = normalize_audio_array(audio)
    left = normalized[:, 0]
    right = normalized[:, 1]
    mid = 0.5 * (left + right) * mid_gain
    side = 0.5 * (left - right) * side_gain
    mixed = normalized.copy()
    mixed[:, 0] = mid + side
    mixed[:, 1] = mid - side
    return normalize_audio_array(mixed)


def _early_reflections(
    audio: np.ndarray,
    sample_rate: int,
    *,
    taps: tuple[tuple[float, float, float], ...],
    dry_mix: float = 1.0,
) -> np.ndarray:
    normalized = normalize_audio_array(audio)
    rendered = normalized * dry_mix
    mono_source = normalized.mean(axis=1)
    for delay_ms, left_gain, right_gain in taps:
        delay_frames = max(1, int(round(sample_rate * delay_ms / 1000.0)))
        delayed = np.pad(mono_source, (delay_frames, 0))[: mono_source.shape[0]]
        rendered[:, 0] += delayed * left_gain
        rendered[:, 1] += delayed * right_gain
    return normalize_audio_array(rendered)


def _feedback_reverb(
    audio: np.ndarray,
    sample_rate: int,
    *,
    delay_ms: float,
    stereo_offset_ms: float,
    feedback: float,
    repeats: int,
    wet_gain: float,
    dry_mix: float,
) -> np.ndarray:
    normalized = normalize_audio_array(audio)
    rendered = normalized * dry_mix
    mono_source = normalized.mean(axis=1)
    for repeat in range(1, repeats + 1):
        base_delay_frames = max(1, int(round(sample_rate * delay_ms * repeat / 1000.0)))
        stereo_delay_frames = max(
            1,
            int(round(sample_rate * (delay_ms + stereo_offset_ms) * repeat / 1000.0)),
        )
        gain = wet_gain * (feedback ** (repeat - 1))
        left_delayed = np.pad(mono_source, (base_delay_frames, 0))[: mono_source.shape[0]]
        right_delayed = np.pad(mono_source, (stereo_delay_frames, 0))[: mono_source.shape[0]]
        rendered[:, 0] += left_delayed * gain
        rendered[:, 1] += right_delayed * (gain * 0.92)
    return normalize_audio_array(rendered)


def _mix_white_noise(
    audio: np.ndarray,
    sample_rate: int,
    *,
    relative_db: float,
    seed: int = 20260320,
) -> np.ndarray:
    del sample_rate
    normalized = normalize_audio_array(audio)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(normalized.shape).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    signal_rms = float(np.sqrt(np.mean(np.square(normalized), dtype=np.float64)))
    target_rms = max(signal_rms * _db_to_gain(relative_db), 1e-4)
    scaled_noise = noise * (target_rms / max(noise_rms, 1e-6))
    return normalize_audio_array(normalized + scaled_noise)


_PRESET_CHAINS: Mapping[str, EffectChain] = {
    "master": EffectChain(
        name="master",
        stages=(
            FFmpegFilterEffectStage(
                name="master_loudnorm",
                filter_graph="loudnorm",
            ),
        ),
    ),
    "narrator": EffectChain(
        name="narrator",
        stages=(
            scipy_signal_stage(
                "highpass_cleanup",
                partial(_filter_audio, btype="highpass", cutoff_hz=85.0),
            ),
            numpy_stage(
                "leveling_compressor",
                partial(
                    _compress_audio,
                    threshold_db=-28.0,
                    ratio=2.8,
                    attack_ms=5.0,
                    release_ms=240.0,
                    makeup_db=2.2,
                ),
            ),
            numpy_stage(
                "center_focus",
                partial(_mid_side_mix, mid_gain=1.18, side_gain=0.62),
            ),
            numpy_stage(
                "presence_tilt",
                partial(_tilt_tone, low_band_db=-1.4, high_band_db=1.6),
            ),
            numpy_stage(
                "cognitive_halo",
                partial(
                    _early_reflections,
                    taps=((9.0, 0.09, 0.12), (18.0, 0.07, 0.05), (31.0, 0.04, 0.06)),
                    dry_mix=0.96,
                ),
            ),
        ),
    ),
    "thoughts": EffectChain(
        name="thoughts",
        stages=(
            scipy_signal_stage(
                "highpass_cleanup",
                partial(_filter_audio, btype="highpass", cutoff_hz=90.0),
            ),
            numpy_stage(
                "produced_compressor",
                partial(
                    _compress_audio,
                    threshold_db=-30.0,
                    ratio=3.2,
                    attack_ms=4.0,
                    release_ms=260.0,
                    makeup_db=2.4,
                ),
            ),
            numpy_stage(
                "centered_stereo",
                partial(_mid_side_mix, mid_gain=1.14, side_gain=0.72),
            ),
            numpy_stage(
                "air_tilt",
                partial(_tilt_tone, low_band_db=-1.3, high_band_db=1.8),
            ),
            numpy_stage(
                "wide_halo",
                partial(
                    _feedback_reverb,
                    delay_ms=44.0,
                    stereo_offset_ms=7.0,
                    feedback=0.58,
                    repeats=4,
                    wet_gain=0.08,
                    dry_mix=0.96,
                ),
            ),
        ),
    ),
    "outdoor1": EffectChain(
        name="outdoor1",
        stages=(
            scipy_signal_stage(
                "highpass_cleanup",
                partial(_filter_audio, btype="highpass", cutoff_hz=100.0),
            ),
            numpy_stage(
                "light_presence",
                partial(_tilt_tone, low_band_db=-0.6, high_band_db=1.0),
            ),
            numpy_stage(
                "open_width",
                partial(_mid_side_mix, mid_gain=0.98, side_gain=1.18),
            ),
            numpy_stage(
                "air_bed",
                partial(_mix_white_noise, relative_db=-28.0),
            ),
            numpy_stage(
                "air_reflection",
                partial(
                    _early_reflections,
                    taps=((24.0, 0.04, 0.05), (46.0, 0.03, 0.025)),
                    dry_mix=0.99,
                ),
            ),
        ),
    ),
    "outdoor2": EffectChain(
        name="outdoor2",
        stages=(
            scipy_signal_stage(
                "highpass_cleanup",
                partial(_filter_audio, btype="highpass", cutoff_hz=115.0),
            ),
            numpy_stage(
                "bright_open_width",
                partial(_mid_side_mix, mid_gain=0.97, side_gain=1.12),
            ),
            numpy_stage(
                "weather_noise",
                partial(_mix_white_noise, relative_db=-24.0),
            ),
            numpy_stage(
                "outdoor_echo_tail",
                partial(
                    _feedback_reverb,
                    delay_ms=66.0,
                    stereo_offset_ms=10.0,
                    feedback=0.6,
                    repeats=5,
                    wet_gain=0.1,
                    dry_mix=0.94,
                ),
            ),
            numpy_stage(
                "upper_air",
                partial(_tilt_tone, low_band_db=-0.8, high_band_db=1.2),
            ),
        ),
    ),
    "indoor1": EffectChain(
        name="indoor1",
        stages=(
            scipy_signal_stage(
                "highpass_cleanup",
                partial(_filter_audio, btype="highpass", cutoff_hz=80.0),
            ),
            numpy_stage(
                "small_room",
                partial(
                    _early_reflections,
                    taps=((12.0, 0.14, 0.09), (21.0, 0.09, 0.14), (33.0, 0.06, 0.06), (48.0, 0.04, 0.04)),
                    dry_mix=0.93,
                ),
            ),
            numpy_stage(
                "slight_narrowing",
                partial(_mid_side_mix, mid_gain=1.08, side_gain=0.74),
            ),
            numpy_stage(
                "warm_room_tilt",
                partial(_tilt_tone, low_band_db=0.8, high_band_db=-0.6),
            ),
            scipy_signal_stage(
                "ceiling_soften",
                partial(_filter_audio, btype="lowpass", cutoff_hz=8200.0),
            ),
        ),
    ),
    "indoor2": EffectChain(
        name="indoor2",
        stages=(
            scipy_signal_stage(
                "highpass_cleanup",
                partial(_filter_audio, btype="highpass", cutoff_hz=85.0),
            ),
            numpy_stage(
                "room_leveling",
                partial(
                    _compress_audio,
                    threshold_db=-27.0,
                    ratio=2.2,
                    attack_ms=7.0,
                    release_ms=200.0,
                    makeup_db=1.2,
                ),
            ),
            numpy_stage(
                "tighter_room",
                partial(
                    _early_reflections,
                    taps=((15.0, 0.16, 0.1), (28.0, 0.1, 0.16), (42.0, 0.07, 0.08), (63.0, 0.05, 0.05)),
                    dry_mix=0.9,
                ),
            ),
            numpy_stage(
                "narrow_focus",
                partial(_mid_side_mix, mid_gain=1.1, side_gain=0.66),
            ),
            scipy_signal_stage(
                "ceiling_soften",
                partial(_filter_audio, btype="lowpass", cutoff_hz=6500.0),
            ),
        ),
    ),
}

_PRESET_ALIASES = {
    "narrator1": "narrator",
    "narrator2": "thoughts",
}
