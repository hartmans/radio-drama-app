from __future__ import annotations

import asyncio
import importlib
import os
from contextlib import contextmanager, suppress


async def _keep_loop_awake(period: float) -> None:
    while True:
        await asyncio.sleep(period)


@contextmanager
def _codex_asyncio_import_workaround(period: float = 0.05):
    if os.environ.get("CODEX_CI") != "1":
        yield
        return

    original_run = asyncio.runners.Runner.run

    def patched_run(self, coro, *, context=None):
        async def wrapped():
            sleeper = asyncio.create_task(_keep_loop_awake(period))
            try:
                return await coro
            finally:
                sleeper.cancel()
                with suppress(asyncio.CancelledError):
                    await sleeper

        return original_run(self, wrapped(), context=context)

    asyncio.runners.Runner.run = patched_run
    try:
        yield
    finally:
        asyncio.runners.Runner.run = original_run


with _codex_asyncio_import_workaround():
    _dependency_injection = importlib.import_module("carthage.dependency_injection")

AsyncInjectable = _dependency_injection.AsyncInjectable
AsyncInjector = _dependency_injection.AsyncInjector
InjectionKey = _dependency_injection.InjectionKey
Injector = _dependency_injection.Injector
inject = _dependency_injection.inject

__all__ = [
    "AsyncInjectable",
    "AsyncInjector",
    "InjectionKey",
    "Injector",
    "inject",
]
