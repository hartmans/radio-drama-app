"""Bounded-buffer conversion of native streaming audio to production format."""

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .audio import convert_channel_count


class StreamingAudioConverter:
    """Preserve resampling phase and FIR context across arbitrary input chunks.

    The default scipy polyphase filter needs ten times the larger rate factor
    on either side in the upsampled domain. Retain that context and align buffer
    starts to the downsampling factor so chunk boundaries do not reset phase.
    Only the final flush uses end-of-audio padding.
    """

    def __init__(self, input_sample_rate: int, output_sample_rate: int, channels: int):
        factor = gcd(input_sample_rate, output_sample_rate)
        self.up = output_sample_rate // factor
        self.down = input_sample_rate // factor
        self.channels = channels
        self.margin = (10 * max(self.up, self.down) + self.up - 1) // self.up + 2
        self.buffer = np.empty((0,) if channels == 1 else (0, channels), dtype=np.float32)
        self.start = 0
        self.total = 0
        self.emitted = 0

    def feed(self, audio: np.ndarray, *, final: bool = False) -> np.ndarray:
        audio = convert_channel_count(audio, output_channels=self.channels)
        if self.up == self.down:
            return audio
        self.buffer = np.concatenate((self.buffer, audio))
        self.total += len(audio)
        end = ((self.total * self.up + self.down - 1) // self.down if final else
               max(0, (self.total - self.margin) * self.up // self.down))
        if end <= self.emitted:
            return self.buffer[:0].copy()
        converted = resample_poly(self.buffer, self.up, self.down, axis=0)
        offset = self.start * self.up // self.down
        result = np.ascontiguousarray(converted[self.emitted - offset:end - offset], dtype=np.float32)
        self.emitted = end
        retain = max(0, (self.emitted * self.down // self.up - self.margin) // self.down * self.down)
        self.buffer = self.buffer[retain - self.start:].copy()
        self.start = retain
        return result
