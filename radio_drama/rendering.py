from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RenderResult:
    """Audio produced by rendering a plan.

    ``audio`` is normalized to a contiguous ``float32`` numpy array. Current
    plans produce audio in the configured production format, while the gap and
    margin fields remain available for later composition features.
    """
    audio: np.ndarray
    pre_margin: float = 0.0
    post_margin: float = 0.0
    pre_gap: float = 0.0
    post_gap: float = 0.0

    def __post_init__(self) -> None:
        self.audio = np.ascontiguousarray(self.audio, dtype=np.float32)

    @property
    def frame_count(self) -> int:
        return int(self.audio.shape[0]) if self.audio.ndim else 0

    @property
    def channel_count(self) -> int:
        if self.audio.ndim == 1:
            return 1
        return int(self.audio.shape[1])

    def from_time(self, time_seconds: float, *, sample_rate: int) -> "RenderResult":
        start_frame = max(0, int(time_seconds * sample_rate))
        return type(self)(
            audio=self.audio[start_frame:],
            pre_margin=self.pre_margin,
            post_margin=self.post_margin,
            pre_gap=self.pre_gap,
            post_gap=self.post_gap,
        )

    @classmethod
    def empty(cls, *, channels: int = 1) -> "RenderResult":
        if channels == 1:
            audio = np.zeros(0, dtype=np.float32)
        else:
            audio = np.zeros((0, channels), dtype=np.float32)
        return cls(audio=audio)

    @classmethod
    def concatenate(cls, results: list["RenderResult"]) -> "RenderResult":
        if not results:
            return cls.empty()
        audio = np.concatenate([result.audio for result in results], axis=0)
        return cls(audio=audio)


class ProductionResult(RenderResult):
    """Top-level rendered production audio."""
