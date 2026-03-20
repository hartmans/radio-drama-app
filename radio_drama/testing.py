from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .planning import ScriptRenderRequest
from .rendering import RenderResult


@dataclass(frozen=True, slots=True)
class CachedRenderMetadata:
    sample_rate: int
    frame_count: int


class CachedVibeVoiceDouble:
    def __init__(
        self,
        cache_directory: str | Path,
        *,
        mode: str = "cache",
        seed: int = 0,
    ) -> None:
        if mode not in {"cache", "live"}:
            raise ValueError("mode must be 'cache' or 'live'")
        self.cache_directory = Path(cache_directory)
        self.mode = mode
        self.seed = seed

    def render(
        self,
        request: ScriptRenderRequest,
        producer: Callable[[ScriptRenderRequest], CachedRenderMetadata] | None = None,
    ) -> RenderResult:
        cache_path = self.cache_directory / f"{self._cache_key(request)}.json"
        if cache_path.is_file():
            metadata = CachedRenderMetadata(**json.loads(cache_path.read_text(encoding="utf-8")))
        elif self.mode == "cache":
            import pytest

            pytest.skip(f"No cached metadata for request {cache_path.stem}")
        else:
            if producer is None:
                raise ValueError("producer is required in live mode when cache is missing")
            metadata = producer(request)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(asdict(metadata), indent=2, sort_keys=True),
                encoding="utf-8",
            )

        rng = np.random.default_rng(self.seed)
        audio = rng.standard_normal(metadata.frame_count, dtype=np.float32) * 1e-3
        return RenderResult(audio=audio)

    def _cache_key(self, request: ScriptRenderRequest) -> str:
        payload = json.dumps(
            {
                "normalized_script": request.normalized_script,
                "voice_samples": list(request.voice_samples),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
