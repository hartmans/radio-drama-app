from __future__ import annotations

from pathlib import Path
from threading import Lock

from carthage.dependency_injection import inject

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


@inject(production_config=ProductionConfig)
def debug(
    category: str,
    message: str,
    *,
    production_config: ProductionConfig,
) -> None:
    write_debug_message(production_config, category, message)
