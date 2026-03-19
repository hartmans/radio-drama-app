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


class ProductionResult(RenderResult):
    pass
