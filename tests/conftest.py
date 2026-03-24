from __future__ import annotations

from pathlib import Path

import pytest

from radio_drama.testing import (
    CachedQwenTtsResource,
    CachedVibeVoiceDouble,
    CachedVibeVoiceResource,
    CachedWhisperXResource,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests marked live",
    )
    parser.addoption(
        "--vibevoice-mode",
        action="store",
        default="cache",
        choices=("cache", "live"),
        help="mode for cache-backed VibeVoice test resources",
    )
    parser.addoption(
        "--vibevoice-cache-dir",
        action="store",
        default=None,
        help="directory for cached VibeVoice render metadata",
    )
    parser.addoption(
        "--qwen-mode",
        action="store",
        default="cache",
        choices=("cache", "live"),
        help="mode for cache-backed Qwen TTS test resources",
    )
    parser.addoption(
        "--qwen-cache-dir",
        action="store",
        default=None,
        help="directory for cached Qwen TTS render metadata",
    )
    parser.addoption(
        "--forced-alignment-mode",
        action="store",
        default="cache",
        choices=("cache", "live"),
        help="mode for cache-backed forced-alignment test resources",
    )
    parser.addoption(
        "--forced-alignment-cache-dir",
        action="store",
        default=None,
        help="directory for cached forced-alignment metadata",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires --run-live")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="need --run-live option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def vibevoice_mode(pytestconfig):
    return pytestconfig.getoption("--vibevoice-mode")


@pytest.fixture
def vibevoice_cache_dir(pytestconfig, tmp_path: Path):
    configured = pytestconfig.getoption("--vibevoice-cache-dir")
    if configured:
        return Path(configured)
    return tmp_path / "vibevoice-cache"


@pytest.fixture
def cached_vibevoice_factory(vibevoice_mode: str, vibevoice_cache_dir: Path):
    def factory(*, mode: str | None = None, cache_dir: Path | None = None, seed: int = 0):
        directory = cache_dir or vibevoice_cache_dir
        return CachedVibeVoiceDouble(directory, mode=mode or vibevoice_mode, seed=seed)

    return factory


@pytest.fixture
def cached_vibevoice_resource_factory(vibevoice_mode: str, vibevoice_cache_dir: Path):
    def factory(
        ainjector,
        *,
        mode: str | None = None,
        cache_dir: Path | None = None,
        seed: int = 0,
        resource_type=CachedVibeVoiceResource,
        **kwargs,
    ):
        directory = cache_dir or vibevoice_cache_dir
        return ainjector(
            resource_type,
            cache_directory=directory,
            mode=mode or vibevoice_mode,
            seed=seed,
            **kwargs,
        )

    return factory


@pytest.fixture
def qwen_mode(pytestconfig):
    return pytestconfig.getoption("--qwen-mode")


@pytest.fixture
def qwen_cache_dir(pytestconfig, tmp_path: Path):
    configured = pytestconfig.getoption("--qwen-cache-dir")
    if configured:
        return Path(configured)
    return tmp_path / "qwen-cache"


@pytest.fixture
def cached_qwen_resource_factory(qwen_mode: str, qwen_cache_dir: Path):
    def factory(
        ainjector,
        *,
        mode: str | None = None,
        cache_dir: Path | None = None,
        seed: int = 0,
        resource_type=CachedQwenTtsResource,
        **kwargs,
    ):
        directory = cache_dir or qwen_cache_dir
        return ainjector(
            resource_type,
            cache_directory=directory,
            mode=mode or qwen_mode,
            seed=seed,
            **kwargs,
        )

    return factory


@pytest.fixture
def forced_alignment_mode(pytestconfig):
    return pytestconfig.getoption("--forced-alignment-mode")


@pytest.fixture
def forced_alignment_cache_dir(pytestconfig):
    configured = pytestconfig.getoption("--forced-alignment-cache-dir")
    if configured:
        return Path(configured)
    return REPO_ROOT / "tests" / "resources" / "forced_alignment_cache"


@pytest.fixture
def cached_whisperx_resource_factory(
    forced_alignment_mode: str,
    forced_alignment_cache_dir: Path,
):
    def factory(
        ainjector,
        *,
        mode: str | None = None,
        cache_dir: Path | None = None,
        resource_type=CachedWhisperXResource,
        **kwargs,
    ):
        directory = cache_dir or forced_alignment_cache_dir
        return ainjector(
            resource_type,
            cache_directory=directory,
            mode=mode or forced_alignment_mode,
            **kwargs,
        )

    return factory
