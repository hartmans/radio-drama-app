from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RenderResult:
    audio: np.ndarray
    sample_rate: int
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

    @classmethod
    def empty(cls, sample_rate: int, *, channels: int = 1) -> "RenderResult":
        if channels == 1:
            audio = np.zeros(0, dtype=np.float32)
        else:
            audio = np.zeros((0, channels), dtype=np.float32)
        return cls(audio=audio, sample_rate=sample_rate)

    @classmethod
    def concatenate(
        cls,
        results: list["RenderResult"],
        *,
        sample_rate: int | None = None,
    ) -> "RenderResult":
        if not results:
            if sample_rate is None:
                raise ValueError("sample_rate is required when concatenating no results")
            return cls.empty(sample_rate=sample_rate)
        target_rate = sample_rate or results[0].sample_rate
        if any(result.sample_rate != target_rate for result in results):
            raise ValueError("Cannot concatenate render results with different sample rates")
        audio = np.concatenate([result.audio for result in results], axis=0)
        return cls(audio=audio, sample_rate=target_rate)


class ProductionResult(RenderResult):
    pass
