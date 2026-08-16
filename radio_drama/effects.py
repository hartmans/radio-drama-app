from __future__ import annotations

import asyncio
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence, TypeVar, runtime_checkable

import numpy as np
import soundfile as sf
import yaml
from carthage.dependency_injection import AsyncInjectable, inject, inject_autokwargs
from scipy.signal import butter, sosfiltfilt

from .audio import normalize_audio_array, resample_audio
from .expressions import ArrayExpression, eval_expression
from .planning import PlanningNode


VOICE_PREPROCESS_VERSION = "loudnorm-v1"


@runtime_checkable
class EffectStage(Protocol):
    """Audio transformation that mutates one production-format buffer in place."""

    def apply(self, audio: np.ndarray, *, sample_rate: int) -> None:
        """Mutate ``audio`` in place."""

    def __or__(self, other: "EffectStage") -> "EffectStage":
        """Return one stage that applies ``self`` followed by ``other``."""


class _ComposableEffectStage:
    """Shared composition behavior for concrete effect stage objects."""

    def __or__(self, other: EffectStage) -> EffectStage:
        return _compose_effect_stages(self, other)


@dataclass(frozen=True, slots=True)
class CallableEffectStage(_ComposableEffectStage):
    """Simple Python-callable-backed effect stage."""

    processor: Callable[[np.ndarray, int], None]

    def apply(self, audio: np.ndarray, *, sample_rate: int) -> None:
        if audio.shape[0] == 0:
            return
        working = normalize_audio_array(audio)
        self.processor(working, sample_rate)
        _copy_back(audio, working)


@dataclass(frozen=True, slots=True)
class PedalboardEffectStage(_ComposableEffectStage):
    """Pedalboard-backed stage loaded only when actually used."""

    board_factory: Callable[[], object]

    def apply(self, audio: np.ndarray, *, sample_rate: int) -> None:
        if audio.shape[0] == 0:
            return
        working = normalize_audio_array(audio)
        try:
            board = self.board_factory()
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Pedalboard is required for this effect stage") from exc
        processed = board(working.T, sample_rate, reset=True)
        working[...] = normalize_audio_array(np.asarray(processed).T)
        _copy_back(audio, working)


@dataclass(frozen=True, slots=True)
class FFmpegFilterEffectStage(_ComposableEffectStage):
    """FFmpeg-backed stage for effects that are easiest to express as filters."""

    filter_graph_factory: Callable[[], str]

    def apply(self, audio: np.ndarray, *, sample_rate: int) -> None:
        from scipy.io import wavfile

        if audio.shape[0] == 0:
            return
        working = normalize_audio_array(audio)
        output_channels = 1 if working.ndim == 1 else working.shape[1]
        with tempfile.TemporaryDirectory(prefix="radio-drama-ffmpeg-") as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.wav"
            output_path = temp_path / "output.wav"
            wavfile.write(input_path, sample_rate, working)
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
                self.filter_graph_factory(),
                "-ar",
                str(sample_rate),
                "-ac",
                str(output_channels),
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
        working[...] = normalize_audio_array(rendered)
        _copy_back(audio, working)


@dataclass(frozen=True, slots=True)
class EffectPipeline(_ComposableEffectStage):
    """Immutable sequence of stages that itself behaves like one stage."""

    stages: tuple[EffectStage, ...]

    def apply(self, audio: np.ndarray, *, sample_rate: int) -> None:
        if audio.shape[0] == 0:
            return
        working = normalize_audio_array(audio)
        for stage in self.stages:
            stage.apply(working, sample_rate=sample_rate)
        _copy_back(audio, working)


effect_stages: dict[str, Callable[..., EffectStage]] = {}
_effect_chain_functions: dict[str, Callable[..., EffectStage]] = {}

_EffectFactory = TypeVar("_EffectFactory", bound=Callable[..., EffectStage])


def register_effect_stage(factory: _EffectFactory) -> _EffectFactory:
    effect_stages[factory.__name__] = factory
    return factory


def effect_chain_function(factory: _EffectFactory) -> _EffectFactory:
    """Expose one explicitly approved stage factory to effect-chain expressions."""

    _effect_chain_functions[factory.__name__] = factory
    return factory


def numpy_stage(processor: Callable[[np.ndarray, int], None]) -> CallableEffectStage:
    return CallableEffectStage(processor)


def scipy_signal_stage(processor: Callable[[np.ndarray, int], None]) -> CallableEffectStage:
    return CallableEffectStage(processor)


def pedalboard_stage(board_factory: Callable[[], object]) -> PedalboardEffectStage:
    return PedalboardEffectStage(board_factory)


def ffmpeg_filter_stage(
    filter_graph_factory: Callable[[], str],
) -> FFmpegFilterEffectStage:
    return FFmpegFilterEffectStage(filter_graph_factory)


def normalize_effect_chain_name(name: str) -> str:
    normalized_name = name.strip().lower()
    return _PRESET_ALIASES.get(normalized_name, normalized_name)


def build_voice_preprocess_chain() -> EffectStage:
    """Return the internal effect chain for reference voice preprocessing.

    This chain is intentionally separate from document-authored presets. It is
    part of backend implementation rather than a user-facing production preset.
    """

    return _VOICE_PREPROCESS


def preprocess_voice_reference(
    audio: np.ndarray,
    *,
    sample_rate: int,
) -> np.ndarray:
    """Return one mono reference-voice clip after backend preprocessing.

    Reference voices stay mono throughout preprocessing. Backends that require
    a specific prompt sample rate can resample after this function returns.
    """

    processed = np.asarray(audio, dtype=np.float32)
    if processed.ndim == 2:
        processed = processed.mean(axis=1)
    if processed.ndim != 1:
        raise ValueError(f"Expected mono or stereo voice audio, got {processed.shape!r}")
    processed = normalize_audio_array(processed)
    build_voice_preprocess_chain().apply(processed, sample_rate=sample_rate)
    return normalize_audio_array(processed)


def load_preprocessed_voice_reference(
    voice_path: str | Path,
    *,
    output_sample_rate: int | None = None,
    gain_db: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Load, preprocess, and optionally resample one reference voice file."""

    loaded_audio, loaded_sample_rate = sf.read(
        str(Path(voice_path).expanduser()),
        dtype="float32",
        always_2d=False,
    )
    processed_audio = preprocess_voice_reference(
        np.asarray(loaded_audio, dtype=np.float32),
        sample_rate=int(loaded_sample_rate),
    )
    sample_rate = int(loaded_sample_rate)
    if output_sample_rate is not None and output_sample_rate != sample_rate:
        processed_audio = resample_audio(
            processed_audio,
            input_sample_rate=sample_rate,
            output_sample_rate=output_sample_rate,
        )
        sample_rate = output_sample_rate
    if gain_db:
        processed_audio *= np.float32(_db_to_gain(gain_db))
    return processed_audio, sample_rate


def _copy_back(target: np.ndarray, source: np.ndarray) -> None:
    if source is not target:
        target[...] = source


def _compose_effect_stages(*stages: EffectStage) -> EffectStage:
    flattened: list[EffectStage] = []
    for stage in stages:
        if isinstance(stage, EffectPipeline):
            flattened.extend(stage.stages)
        else:
            flattened.append(stage)
    if len(flattened) == 1:
        return flattened[0]
    return EffectPipeline(tuple(flattened))


def _db_to_gain(decibels: float) -> float:
    return float(10.0 ** (decibels / 20.0))


def _filtered_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    btype: str,
    cutoff_hz: float,
    order: int = 2,
) -> np.ndarray:
    nyquist = sample_rate / 2.0
    normalized_cutoff = min(max(cutoff_hz / nyquist, 1e-5), 0.999)
    sos = butter(order, normalized_cutoff, btype=btype, output="sos")
    return normalize_audio_array(sosfiltfilt(sos, audio, axis=0))


@effect_chain_function
@register_effect_stage
def filter_audio(
    *,
    btype: str,
    cutoff_hz: float,
    order: int = 2,
) -> EffectStage:
    @scipy_signal_stage
    def stage(audio: np.ndarray, sample_rate: int) -> None:
        audio[...] = _filtered_audio(
            audio,
            sample_rate,
            btype=btype,
            cutoff_hz=cutoff_hz,
            order=order,
        )

    return stage


@effect_chain_function
@register_effect_stage
def tilt_tone(
    *,
    low_band_db: float = 0.0,
    high_band_db: float = 0.0,
) -> EffectStage:
    @numpy_stage
    def stage(audio: np.ndarray, sample_rate: int) -> None:
        low_band = _filtered_audio(audio, sample_rate, btype="lowpass", cutoff_hz=220.0)
        high_band = _filtered_audio(audio, sample_rate, btype="highpass", cutoff_hz=3200.0)
        mid_band = audio - low_band - high_band
        audio[...] = normalize_audio_array(
            low_band * _db_to_gain(low_band_db)
            + mid_band
            + high_band * _db_to_gain(high_band_db)
        )

    return stage


@effect_chain_function
@register_effect_stage
def compress_audio(
    *,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float = 0.0,
) -> EffectStage:
    if ratio <= 0:
        raise ValueError("ratio must be positive")

    def board_factory() -> object:
        from pedalboard import Compressor

        return Compressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )

    compressor = pedalboard_stage(board_factory)
    if not makeup_db:
        return compressor

    @numpy_stage
    def makeup_stage(audio: np.ndarray, sample_rate: int) -> None:
        del sample_rate
        audio *= np.float32(_db_to_gain(makeup_db))

    return compressor | makeup_stage


@effect_chain_function
@register_effect_stage
def mid_side_mix(
    *,
    mid_gain: float,
    side_gain: float,
) -> EffectStage:
    @numpy_stage
    def stage(audio: np.ndarray, sample_rate: int) -> None:
        del sample_rate
        left = audio[:, 0]
        right = audio[:, 1]
        mid = 0.5 * (left + right) * mid_gain
        side = 0.5 * (left - right) * side_gain
        audio[:, 0] = mid + side
        audio[:, 1] = mid - side

    return stage


@effect_chain_function
@register_effect_stage
def early_reflections(
    *,
    taps: tuple[tuple[float, float, float], ...],
    dry_mix: float = 1.0,
) -> EffectStage:
    @numpy_stage
    def stage(audio: np.ndarray, sample_rate: int) -> None:
        rendered = audio * dry_mix
        mono_source = audio.mean(axis=1)
        for delay_ms, left_gain, right_gain in taps:
            delay_frames = max(1, int(round(sample_rate * delay_ms / 1000.0)))
            delayed = np.pad(mono_source, (delay_frames, 0))[: mono_source.shape[0]]
            rendered[:, 0] += delayed * left_gain
            rendered[:, 1] += delayed * right_gain
        audio[...] = normalize_audio_array(rendered)

    return stage


@effect_chain_function
@register_effect_stage
def feedback_reverb(
    *,
    delay_ms: float,
    stereo_offset_ms: float,
    feedback: float,
    repeats: int,
    wet_mix: float,
    dry_mix: float,
) -> EffectStage:
    @numpy_stage
    def stage(audio: np.ndarray, sample_rate: int) -> None:
        rendered = audio * dry_mix
        mono_source = audio.mean(axis=1)
        for repeat in range(1, repeats + 1):
            base_delay_frames = max(1, int(round(sample_rate * delay_ms * repeat / 1000.0)))
            stereo_delay_frames = max(
                1,
                int(round(sample_rate * (delay_ms + stereo_offset_ms) * repeat / 1000.0)),
            )
            gain = wet_mix * (feedback ** (repeat - 1))
            left_delayed = np.pad(mono_source, (base_delay_frames, 0))[: mono_source.shape[0]]
            right_delayed = np.pad(mono_source, (stereo_delay_frames, 0))[: mono_source.shape[0]]
            rendered[:, 0] += left_delayed * gain
            rendered[:, 1] += right_delayed * (gain * 0.92)
        audio[...] = normalize_audio_array(rendered)

    return stage


@effect_chain_function
@register_effect_stage
def modulated_delay(
    *,
    delay_ms: float,
    depth_ms: float,
    rate_hz: float,
    wet_mix: float,
    dry_mix: float = 1.0,
    stereo_phase_degrees: float = 90.0,
    phase_degrees: float = 0.0,
) -> EffectStage:
    """Mix the input with a sinusoidally moving, fractional-delay copy."""

    if delay_ms < 0.0:
        raise ValueError("delay_ms must be non-negative")
    if depth_ms < 0.0:
        raise ValueError("depth_ms must be non-negative")
    if depth_ms > delay_ms:
        raise ValueError("depth_ms must not exceed delay_ms")
    if rate_hz < 0.0:
        raise ValueError("rate_hz must be non-negative")

    @numpy_stage
    def stage(audio: np.ndarray, sample_rate: int) -> None:
        frames = audio if audio.ndim == 2 else audio[:, np.newaxis]
        frame_positions = np.arange(frames.shape[0], dtype=np.float64)
        time_seconds = frame_positions / sample_rate
        rendered = frames * dry_mix
        base_phase = math.radians(phase_degrees)
        stereo_phase = math.radians(stereo_phase_degrees)
        delay_scale = sample_rate / 1000.0

        for channel in range(frames.shape[1]):
            channel_phase = base_phase + (stereo_phase if channel % 2 else 0.0)
            delay = delay_ms + depth_ms * np.sin(
                math.tau * rate_hz * time_seconds + channel_phase
            )
            source_positions = frame_positions - delay * delay_scale
            delayed = np.interp(
                source_positions,
                frame_positions,
                frames[:, channel],
                left=0.0,
                right=0.0,
            )
            rendered[:, channel] += delayed * wet_mix

        if audio.ndim == 1:
            audio[...] = rendered[:, 0]
        else:
            audio[...] = rendered

    return stage


@effect_chain_function
@register_effect_stage
def mix_white_noise(
    *,
    relative_db: float,
    seed: int = 20260320,
) -> EffectStage:
    @numpy_stage
    def stage(audio: np.ndarray, sample_rate: int) -> None:
        del sample_rate
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(audio.shape).astype(np.float32)
        noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
        signal_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
        target_rms = max(signal_rms * _db_to_gain(relative_db), 1e-4)
        scaled_noise = noise * (target_rms / max(noise_rms, 1e-6))
        audio += scaled_noise

    return stage


@effect_chain_function
@register_effect_stage
def gain(gain_expression: ArrayExpression) -> EffectStage:
    @numpy_stage
    def stage(audio: np.ndarray, sample_rate: int) -> None:
        del sample_rate
        gain_db = gain_expression.to_size(audio.shape[0])
        gain_multiplier = np.float32(10.0) ** (
            gain_db.astype(np.float32, copy=False) / np.float32(20.0)
        )
        if audio.ndim == 1:
            audio *= gain_multiplier
        else:
            audio *= gain_multiplier[:, np.newaxis]

    return stage


@effect_chain_function
@register_effect_stage
def pan(pan_expression: ArrayExpression) -> EffectStage:
    @numpy_stage
    def stage(audio: np.ndarray, sample_rate: int) -> None:
        del sample_rate
        if audio.ndim != 2 or audio.shape[1] < 2:
            return
        pan_values = np.clip(pan_expression.to_size(audio.shape[0]), -1.0, 1.0)
        far_channel_gain = (1.0 - np.abs(pan_values)).astype(np.float32, copy=False)
        left_gain = np.where(pan_values <= 0.0, 1.0, far_channel_gain).astype(
            np.float32,
            copy=False,
        )
        right_gain = np.where(pan_values >= 0.0, 1.0, far_channel_gain).astype(
            np.float32,
            copy=False,
        )
        power_normalizer = np.sqrt(
            np.float32(2.0) / (np.square(left_gain) + np.square(right_gain))
        ).astype(np.float32, copy=False)
        left_gain *= power_normalizer
        right_gain *= power_normalizer
        audio[:, 0] *= left_gain
        audio[:, 1] *= right_gain

    return stage


@effect_chain_function
def master_loudnorm() -> EffectStage:
    """Return the fixed production mastering stage exposed to expressions."""

    return ffmpeg_filter_stage(lambda: "loudnorm=I=-16:TP=-1.5:LRA=11")


def voice_loudnorm() -> EffectStage:
    """Return the internal fixed reference-voice normalization stage."""

    return ffmpeg_filter_stage(lambda: "loudnorm")


_PRESET_EXPRESSIONS: Mapping[str, str] = {
    "master": 'master_loudnorm()',
    "narrator": (
        'filter_audio(btype="highpass", cutoff_hz=85.0) | '
        'compress_audio(threshold_db=-28.0, ratio=2.8, attack_ms=5.0, release_ms=240.0, makeup_db=2.2) | '
        'mid_side_mix(mid_gain=1.18, side_gain=0.62) | '
        'tilt_tone(low_band_db=-1.4, high_band_db=1.6) | '
        'early_reflections(taps=((9.0, 0.09, 0.12), (18.0, 0.07, 0.05), (31.0, 0.04, 0.06)), dry_mix=0.96)'
    ),
    "narrator_nofocus": (
        'filter_audio(btype="highpass", cutoff_hz=85.0) | '
        'compress_audio(threshold_db=-28.0, ratio=2.8, attack_ms=5.0, release_ms=240.0, makeup_db=2.2) | '
        'tilt_tone(low_band_db=-1.4, high_band_db=1.6) | '
        'early_reflections(taps=((9.0, 0.09, 0.12), (18.0, 0.07, 0.05), (31.0, 0.04, 0.06)), dry_mix=0.96)'
    ),
    "thoughts": (
        'filter_audio(btype="highpass", cutoff_hz=90.0) | '
        'compress_audio(threshold_db=-30.0, ratio=3.2, attack_ms=4.0, release_ms=260.0, makeup_db=2.4) | '
        'mid_side_mix(mid_gain=1.14, side_gain=0.72) | '
        'tilt_tone(low_band_db=-1.3, high_band_db=1.8) | '
        'feedback_reverb(delay_ms=44.0, stereo_offset_ms=7.0, feedback=0.58, repeats=4, wet_mix=0.08, dry_mix=0.96)'
    ),
    "outdoor1": (
        'filter_audio(btype="highpass", cutoff_hz=100.0) | '
        'tilt_tone(low_band_db=-0.6, high_band_db=1.0) | '
        'mid_side_mix(mid_gain=0.98, side_gain=1.18) | '
        'mix_white_noise(relative_db=-28.0) | '
        'early_reflections(taps=((24.0, 0.04, 0.05), (46.0, 0.03, 0.025)), dry_mix=0.99)'
    ),
    "outdoor2": (
        'filter_audio(btype="highpass", cutoff_hz=115.0) | '
        'mid_side_mix(mid_gain=0.97, side_gain=1.12) | '
        'mix_white_noise(relative_db=-24.0) | '
        'feedback_reverb(delay_ms=66.0, stereo_offset_ms=10.0, feedback=0.6, repeats=5, wet_mix=0.1, dry_mix=0.94) | '
        'tilt_tone(low_band_db=-0.8, high_band_db=1.2)'
    ),
    "indoor1": (
        'filter_audio(btype="highpass", cutoff_hz=80.0) | '
        'early_reflections(taps=((12.0, 0.14, 0.09), (21.0, 0.09, 0.14), (33.0, 0.06, 0.06), (48.0, 0.04, 0.04)), dry_mix=0.93) | '
        'mid_side_mix(mid_gain=1.08, side_gain=0.74) | '
        'tilt_tone(low_band_db=0.8, high_band_db=-0.6) | '
        'filter_audio(btype="lowpass", cutoff_hz=8200.0)'
    ),
    "indoor1_nofocus": (
        'filter_audio(btype="highpass", cutoff_hz=80.0) | '
        'early_reflections(taps=((12.0, 0.14, 0.09), (21.0, 0.09, 0.14), (33.0, 0.06, 0.06), (48.0, 0.04, 0.04)), dry_mix=0.93) | '
        'tilt_tone(low_band_db=0.8, high_band_db=-0.6) | '
        'filter_audio(btype="lowpass", cutoff_hz=8200.0)'
    ),
    "indoor2": (
        'filter_audio(btype="highpass", cutoff_hz=85.0) | '
        'compress_audio(threshold_db=-27.0, ratio=1.7, attack_ms=7.0, release_ms=200.0, makeup_db=1.2) | '
        'early_reflections(taps=((15.0, 0.16, 0.1), (28.0, 0.1, 0.16), (42.0, 0.07, 0.08), (63.0, 0.05, 0.05)), dry_mix=0.9) | '
        'mid_side_mix(mid_gain=1.1, side_gain=0.66) | '
        'filter_audio(btype="lowpass", cutoff_hz=16500.0)'
    ),
    "indoor2_nofocus": (
        'filter_audio(btype="highpass", cutoff_hz=85.0) | '
        'compress_audio(threshold_db=-27.0, ratio=1.7, attack_ms=7.0, release_ms=200.0, makeup_db=1.2) | '
        'early_reflections(taps=((15.0, 0.16, 0.1), (28.0, 0.1, 0.16), (42.0, 0.07, 0.08), (63.0, 0.05, 0.05)), dry_mix=0.9) | '
        'filter_audio(btype="lowpass", cutoff_hz=16500.0)'
    ),
    "background": (
        'filter_audio(btype="highpass", cutoff_hz=85.0) | '
        'compress_audio(threshold_db=-27.0, ratio=2.2, attack_ms=7.0, release_ms=200.0, makeup_db=1.2) | '
        'early_reflections(taps=((15.0, 0.16, 0.1), (28.0, 0.1, 0.16), (42.0, 0.07, 0.08), (63.0, 0.05, 0.05)), dry_mix=0.9) | '
        'mid_side_mix(mid_gain=0.4, side_gain=1.8) | '
        'filter_audio(btype="lowpass", cutoff_hz=4500.0)'
    ),
    "phone": (
        'filter_audio(btype="highpass", cutoff_hz=320.0) | '
        'filter_audio(btype="lowpass", cutoff_hz=3200.0) | '
        'compress_audio(threshold_db=-30.0, ratio=3.6, attack_ms=3.0, release_ms=160.0, makeup_db=3.0) | '
        'mix_white_noise(relative_db=-34.0)'
    ),
}
_PRESET_ALIASES = {
    "narrator1": "narrator",
    "narrator2": "thoughts",
}


def effect_chain(value) -> EffectStage:
    """Coerce an expression result to the effect-chain interface."""

    if isinstance(value, EffectStage):
        return value
    raise TypeError(f"Expected an effect chain, got {type(value).__name__}")


def effect_chain_variables(
    presets: Mapping[str, EffectStage] | None = None,
) -> dict[str, object]:
    """Return approved functions and current presets for chain evaluation."""
    
    if presets is None:
        # Build preset stages from _PRESET_EXPRESSIONS strings  
        builtins = {}
        temp_vars = dict(_effect_chain_functions)
        for name in sorted(_PRESET_EXPRESSIONS.keys()):
            expr_str = _PRESET_EXPRESSIONS[name]
            chain = eval_expression(expr_str, {**temp_vars, **builtins}, effect_chain)
            builtins[name] = chain
            temp_vars.update(builtins)
        current_presets = builtins
    else:
        current_presets = presets
    
    variables: dict[str, object] = dict(_effect_chain_functions)
    variables.update(current_presets)
    variables.update(
        {
            alias: current_presets[target]
            for alias, target in _PRESET_ALIASES.items()
            if target in current_presets
        }
    )
    return variables


@dataclass(frozen=True, slots=True)
class PresetEntry:
    """Holds both the expression string and compiled EffectStage for a preset."""
    expression: str
    stage: EffectStage


class EffectChainRegistry:
    """Production-scoped named effect chains initialized from built-in presets."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, PresetEntry] = {}
        
        # Build builtin presets from expression strings
        preset_vars: dict[str, EffectStage] = {}
        for name in sorted(_PRESET_EXPRESSIONS.keys()):
            expr = _PRESET_EXPRESSIONS[name]
            chain = eval_expression(expr, effect_chain_variables(preset_vars), effect_chain)
            self._entries[name] = PresetEntry(expression=expr, stage=chain)
            preset_vars[name] = chain

    def __getitem__(self, name: str) -> EffectStage:
        normalized_name = normalize_effect_chain_name(name)
        return self._entries[normalized_name].stage  # type: ignore[union-attr]

    def __contains__(self, name: str) -> bool:
        normalized_name = normalize_effect_chain_name(name)
        return normalized_name in self._entries

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries.keys()))

    def get_expression(self, name: str) -> str | None:
        """Return the expression string for a preset, or None if not found."""
        normalized_name = normalize_effect_chain_name(name)
        entry = self._entries.get(normalized_name)
        return entry.expression if entry is not None else None  # type: ignore[union-attr]

    def stages(self) -> dict[str, EffectStage]:
        """Return the current named stages for one document expression scope."""

        return {name: entry.stage for name, entry in self._entries.items()}

    def copy(self) -> "EffectChainRegistry":
        """Return an independent registry with this registry's current entries."""

        copied = EffectChainRegistry()
        copied._entries = dict(self._entries)
        return copied

    def add_from_expression(self, name: str, expression: str) -> EffectStage:
        """Add a new preset from an expression string. Stores both expression and stage."""
        normalized_name = normalize_effect_chain_name(name)
        if normalized_name in _effect_chain_functions:
            raise ValueError(f"Preset name {name!r} is reserved for an effect function")
        
        chain = eval_expression(
            expression,
            effect_chain_variables({k: v.stage for k, v in self._entries.items()}),
            effect_chain,
        )
        self._entries[normalized_name] = PresetEntry(expression=expression, stage=chain)
        return chain


@inject(effect_chains=EffectChainRegistry)
class PresetMapPlan(PlanningNode):
    """Evaluate one production's ordered YAML preset definitions."""

    async def async_ready(self):
        loaded = yaml.safe_load(self.node.normalized_text_content)
        if not isinstance(loaded, dict):
            raise self.document_error(
                "The <preset-map> YAML must be a mapping of preset names to expressions"
            )
        for preset_name, expression in loaded.items():
            if not isinstance(preset_name, str) or not preset_name.strip().isidentifier():
                raise self.document_error(
                    "Preset names in <preset-map> must be valid expression identifiers"
                )
            if not isinstance(expression, str) or not expression.strip():
                raise self.document_error(
                    f"Expression for preset {preset_name!r} must be a non-empty string"
                )
            try:
                self.effect_chains.add_from_expression(preset_name.strip(), expression.strip())
            except Exception as exc:
                raise self.document_error(
                    f"Invalid expression for preset {preset_name!r}: {exc}"
                ) from exc
        return await super().async_ready()


_VOICE_PREPROCESS = voice_loudnorm()


@dataclass(slots=True)
class _EffectRegion:
    start_frame: int
    end_frame: int
    audio: np.ndarray
    preset_key: tuple[str, ...]


@inject_autokwargs(effect_chains=EffectChainRegistry)
class EffectMixer(AsyncInjectable):
    """Compose-local preset bus mixer.

    Child renders are added in compose-frame coordinates along with a preset bus
    key. All regions for one preset key are mixed into one full-timeline bus,
    that bus is processed once, and then all buses are summed together.
    """

    def __init__(
        self,
        *,
        total_frames: int,
        channels: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.total_frames = max(0, int(total_frames))
        self.channels = max(1, int(channels))
        self._regions: list[_EffectRegion] = []

    def add(
        self,
        *,
        start_frame: int,
        end_frame: int,
        audio: np.ndarray,
        preset_key: Sequence[str] = (),
    ) -> None:
        self._regions.append(
            _EffectRegion(
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                audio=audio,
                preset_key=tuple(preset_key),
            )
        )

    async def apply(self, *, sample_rate: int) -> np.ndarray:
        buses: dict[tuple[str, ...], np.ndarray] = {}
        for region in self._regions:
            bus = buses.get(region.preset_key)
            if bus is None:
                bus = self._empty_audio()
                buses[region.preset_key] = bus
            self._mix_region_into_bus(bus, region)
        await asyncio.gather(
            *(
                self._apply_bus_preset(preset_key, bus, sample_rate=sample_rate)
                for preset_key, bus in buses.items()
                if preset_key
            )
        )
        mixed = self._empty_audio()
        for bus in buses.values():
            mixed += bus
        return mixed

    async def _apply_bus_preset(
        self,
        preset_key: tuple[str, ...],
        audio: np.ndarray,
        *,
        sample_rate: int,
    ) -> None:
        def apply_preset_stack() -> None:
            for preset_name in preset_key:
                stage = self.effect_chains[preset_name]
                stage.apply(audio, sample_rate=sample_rate)

        await asyncio.to_thread(apply_preset_stack)

    def _mix_region_into_bus(self, bus: np.ndarray, region: _EffectRegion) -> None:
        if region.audio.shape[0] == 0:
            return
        write_start = max(0, region.start_frame)
        write_end = min(self.total_frames, region.end_frame)
        if write_end <= write_start:
            return
        source_start = max(0, -region.start_frame)
        source_end = source_start + (write_end - write_start)
        bus[write_start:write_end] += region.audio[source_start:source_end]

    def _empty_audio(self) -> np.ndarray:
        if self.channels == 1:
            return np.zeros(self.total_frames, dtype=np.float32)
        return np.zeros((self.total_frames, self.channels), dtype=np.float32)
