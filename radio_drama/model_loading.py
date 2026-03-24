from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from typing import Iterator


_MODEL_LOAD_LOCK = Lock()


@contextmanager
def shared_model_load() -> Iterator[None]:
    """Serialize heavyweight lazy model loads within one process.

    Resources may still render concurrently once loaded. The lock only guards
    startup paths such as ``from_pretrained`` and equivalent model
    initialization calls.
    """
    with _MODEL_LOAD_LOCK:
        yield
