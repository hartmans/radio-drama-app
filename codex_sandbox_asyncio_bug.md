# Codex Sandbox Bug: `asyncio` Cross-Thread Wakeups Do Not Wake An Idle Event Loop

## Summary

Inside the Codex sandbox, a standard Python `asyncio` pattern does not work: a background thread calling `loop.call_soon_threadsafe(...)` does not wake an otherwise-idle event loop.

The same code works correctly outside the sandbox.

This appears to violate a core `asyncio` contract and can break common libraries that bridge threads into an event loop.

## Impact

This breaks a very common pattern used by Python libraries that:

* do work in background threads
* notify an `asyncio` task via `loop.call_soon_threadsafe(...)`
* rely on the event loop waking promptly to run the scheduled callback

In this case, it caused `sh` async command completion to hang inside the sandbox, which initially looked like a library bug. The issue reduces to a pure-stdlib `asyncio` repro that fails only in the sandbox.

## Minimal repro

```python
import asyncio
import threading
import time

async def main():
    loop = asyncio.get_running_loop()
    event = asyncio.Event()

    def worker():
        time.sleep(0.2)
        loop.call_soon_threadsafe(event.set)

    threading.Thread(target=worker, daemon=True).start()
    await event.wait()
    print("OK")

asyncio.run(main())
```

## Expected behavior

The program should print:

```text
OK
```

## Observed behavior in Codex sandbox

The program hangs indefinitely unless some unrelated scheduled task is also waking the event loop.

Running under a hard timeout in the sandbox:

```bash
timeout 5 ~/ai/vibevoice/.venv/bin/python repro.py
```

Observed result:

* exit code `124`
* no output

## Observed behavior outside sandbox

Running the exact same command outside the sandbox succeeds immediately:

* exit code `0`
* output `OK`

## Strong evidence

1. In the sandbox, the pure-`asyncio` repro above hangs and times out.
2. Outside the sandbox, the exact same code works.
3. Adding an unrelated periodic sleeper task inside the sandbox makes the repro succeed.

That suggests:

* the callback is being queued
* but the event loop is not being woken by the cross-thread notification
* the callback only runs once another timer/task wakes the loop

## Variant that succeeds in sandbox

```python
import asyncio
import threading
import time

async def waiter(event):
    await event.wait()
    print("WAITER OK")

async def sleeper():
    for _ in range(10):
        await asyncio.sleep(0.2)

async def main():
    loop = asyncio.get_running_loop()
    event = asyncio.Event()

    def worker():
        time.sleep(0.2)
        loop.call_soon_threadsafe(event.set)

    threading.Thread(target=worker, daemon=True).start()
    await asyncio.gather(waiter(event), sleeper())

asyncio.run(main())
```

This strongly suggests a broken event-loop wakeup path in the sandbox rather than a bug in application code.

## Why this matters

A lot of Python code assumes this works:

* libraries that bridge threads and coroutines
* subprocess wrappers
* async adapters around blocking code
* background worker threads reporting completion into an event loop

If `loop.call_soon_threadsafe(...)` does not wake the loop in the sandbox, many normal async programs will hang in ways that are difficult to diagnose and may be misattributed to third-party libraries.

## Notes

This report intentionally does not claim a specific source location inside Codex internals.
The repro is:

* very small
* pure stdlib `asyncio`
* deterministic in sandbox vs non-sandbox comparison
* sufficient to justify investigation even without deeper sandbox implementation details
