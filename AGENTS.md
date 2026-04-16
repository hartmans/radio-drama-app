* architecture.md should be a living document describing high-level interfaces and their interactions
* doc strings should be used to describe implementation detail that goes beyond the interface.
* Reason carefully about what should be long-term interface guarantees and what is a result of the current implementation
* Allowing python to throw AttributeErrors and KeyErrors when interfaces are misused is better than a lot of checking for programmer error. It's definitely better than turning invalid input into continues/hidden ignored conditions
* Producing better errors for things in the incoming document is valuable
* ~/ai/vibevoice/.venv is the venv to use
* Commit after changes
* Codex sandbox note: in this environment, `asyncio` cross-thread wakeups can fail unless the loop already has some scheduled timer/task activity. In practice, `loop.call_soon_threadsafe(...)` may not wake an otherwise-idle loop inside sandboxed Python commands, even though the same code works outside the sandbox.
* Codex sandbox workaround: do not put Codex-specific wakeup hacks in main code or tests. If a command needs the workaround, run it through `codex_python_runner.py`, which applies the patch from `sandbox_asyncio_workaround.py` before executing a Python module or script.
* Do not modify code or tests to work around the broken sandbox. Add work arounds in sandbox_asyncio_workaround.py. If there is not a clean way to do so without modifying test or other code, ask or escalate; working around a broken sandbox in main code is not appropriate.
* Pytest hanging is probably the broken sandbox.
* Escalating test runs is preferred to running tests in the sandbox; the entire test suite does not take that long, so running a full escalated test suite without the live tests is a reasonable thing to do. Run live tests when requested or when making changes to have live llm resources are used.
