from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import AsyncIterator, Awaitable, TypeVar


T = TypeVar("T")


async def _keep_loop_awake(period: float) -> None:
    while True:
        await asyncio.sleep(period)


@asynccontextmanager
async def codex_asyncio_wakeup_workaround(
    period: float = 0.2,
) -> AsyncIterator[None]:
    """Keep the event loop awake in Codex sandbox runs.

    This exists only as a workaround for a Codex sandbox bug where
    loop.call_soon_threadsafe(...) may not wake an otherwise-idle event loop.
    Do not treat this as application architecture.
    """

    task = asyncio.create_task(_keep_loop_awake(period))
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def await_with_codex_asyncio_workaround(
    awaitable: Awaitable[T],
    *,
    period: float = 0.2,
) -> T:
    """Await an object while keeping the loop awake in Codex sandbox runs."""

    async with codex_asyncio_wakeup_workaround(period=period):
        return await awaitable
