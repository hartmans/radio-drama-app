from __future__ import annotations

import runpy
import sys

from sandbox_asyncio_workaround import codex_asyncio_runner_workaround


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit("usage: codex_python_runner.py [-m module | script.py] [args...]")

    with codex_asyncio_runner_workaround():
        if argv[0] == "-m":
            if len(argv) < 2:
                raise SystemExit("usage: codex_python_runner.py -m module [args...]")
            module_name = argv[1]
            sys.argv = [module_name, *argv[2:]]
            runpy.run_module(module_name, run_name="__main__", alter_sys=True)
            return 0

        script_path = argv[0]
        sys.argv = [script_path, *argv[1:]]
        runpy.run_path(script_path, run_name="__main__")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
