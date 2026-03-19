from __future__ import annotations

from pathlib import Path

import pytest

from radio_drama.testing import CachedVibeVoiceDouble


@pytest.fixture
def cached_vibevoice_factory(tmp_path: Path):
    def factory(*, mode: str = "cache", cache_dir: Path | None = None, seed: int = 0):
        directory = cache_dir or (tmp_path / "vibevoice-cache")
        return CachedVibeVoiceDouble(directory, mode=mode, seed=seed)

    return factory
