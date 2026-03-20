from __future__ import annotations

import asyncio
from contextlib import contextmanager, suppress
from typing import Iterator


async def _keep_loop_awake(period: float) -> None:
    while True:
        await asyncio.sleep(period)


@contextmanager
def codex_asyncio_runner_workaround(period: float = 0.05) -> Iterator[None]:
    """Patch asyncio.run/Runner.run so thread wakeups work in Codex sandbox."""

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
