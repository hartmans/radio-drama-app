from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
}


def normalize_audio_array(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    return np.ascontiguousarray(array, dtype=np.float32)


def resample_audio(
    audio: np.ndarray,
    *,
    input_sample_rate: int,
    output_sample_rate: int,
) -> np.ndarray:
    if input_sample_rate == output_sample_rate:
        return normalize_audio_array(audio)
    factor = gcd(input_sample_rate, output_sample_rate)
    up = output_sample_rate // factor
    down = input_sample_rate // factor
    if audio.ndim == 1:
        return np.ascontiguousarray(resample_poly(audio, up, down), dtype=np.float32)
    return np.ascontiguousarray(resample_poly(audio, up, down, axis=0), dtype=np.float32)


def convert_channel_count(audio: np.ndarray, *, output_channels: int) -> np.ndarray:
    if output_channels < 1:
        raise ValueError("output_channels must be at least 1")
    if output_channels == 1:
        if audio.ndim == 1:
            return normalize_audio_array(audio)
        if audio.shape[1] == 1:
            return normalize_audio_array(audio[:, 0])
        return normalize_audio_array(audio.mean(axis=1))
    if audio.ndim == 1:
        mono = audio[:, np.newaxis]
    elif audio.shape[1] == 1:
        mono = audio
    elif audio.shape[1] == output_channels:
        return normalize_audio_array(audio)
    else:
        mono = audio.mean(axis=1, keepdims=True)
    return normalize_audio_array(np.repeat(mono, output_channels, axis=1))


def convert_audio_format(
    audio: np.ndarray,
    *,
    input_sample_rate: int,
    output_sample_rate: int,
    output_channels: int,
) -> np.ndarray:
    converted = resample_audio(
        normalize_audio_array(audio),
        input_sample_rate=input_sample_rate,
        output_sample_rate=output_sample_rate,
    )
    return convert_channel_count(converted, output_channels=output_channels)
