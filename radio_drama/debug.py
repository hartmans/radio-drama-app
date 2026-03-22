from __future__ import annotations

import json
import shutil
from pathlib import Path
from threading import Lock

from carthage.dependency_injection import inject
import soundfile as sf

from .config import ProductionConfig, SUPPORTED_DEBUG_CATEGORIES


_DEBUG_WRITE_LOCK = Lock()


def write_debug_message(
    production_config: ProductionConfig,
    category: str,
    message: str,
) -> None:
    if category not in SUPPORTED_DEBUG_CATEGORIES:
        raise ValueError(f"Unknown debug category {category!r}")
    if not production_config.debug_enabled(category):
        return
    if production_config.debug_log_path is None:
        return

    log_path = Path(production_config.debug_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{category}] {message}\n"
    with _DEBUG_WRITE_LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def debug_artifact_directory(
    production_config: ProductionConfig,
    category: str,
) -> Path | None:
    if category not in SUPPORTED_DEBUG_CATEGORIES:
        raise ValueError(f"Unknown debug category {category!r}")
    if not production_config.debug_enabled(category):
        return None
    if production_config.debug_log_path is None:
        return None
    return Path(str(production_config.debug_log_path)[:-4] + f".{category}")


def reset_debug_outputs(production_config: ProductionConfig) -> None:
    if production_config.debug_log_path is not None:
        log_path = Path(production_config.debug_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
    for category in SUPPORTED_DEBUG_CATEGORIES:
        artifact_directory = debug_artifact_directory(production_config, category)
        if artifact_directory is None:
            continue
        shutil.rmtree(artifact_directory, ignore_errors=True)


def write_debug_json(
    production_config: ProductionConfig,
    category: str,
    filename: str,
    payload: object,
) -> Path | None:
    artifact_directory = debug_artifact_directory(production_config, category)
    if artifact_directory is None:
        return None
    artifact_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_directory / filename
    with _DEBUG_WRITE_LOCK:
        with artifact_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return artifact_path


def write_debug_wav(
    production_config: ProductionConfig,
    category: str,
    filename: str,
    audio,
    *,
    sample_rate: int,
) -> Path | None:
    artifact_directory = debug_artifact_directory(production_config, category)
    if artifact_directory is None:
        return None
    artifact_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_directory / filename
    with _DEBUG_WRITE_LOCK:
        sf.write(artifact_path, audio, sample_rate)
    return artifact_path


@inject(production_config=ProductionConfig)
def debug(
    category: str,
    message: str,
    *,
    production_config: ProductionConfig,
) -> None:
    write_debug_message(production_config, category, message)
