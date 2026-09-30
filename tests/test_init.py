from __future__ import annotations

import asyncio

from radio_drama.init import radio_drama_injector


def test_radio_drama_injector_sizes_default_executor_once(monkeypatch):
    monkeypatch.setattr("radio_drama.init.os.process_cpu_count", lambda: 7)
    event_loop = asyncio.new_event_loop()
    first_injector = radio_drama_injector(event_loop=event_loop)
    first_executor = event_loop._default_executor
    second_injector = radio_drama_injector(event_loop=event_loop)

    try:
        assert first_executor is not None
        assert first_executor._max_workers == 7
        assert event_loop._default_executor is first_executor
    finally:
        second_injector.close()
        first_injector.close()
        event_loop.run_until_complete(event_loop.shutdown_default_executor())
        event_loop.close()


def test_optional_backends_are_not_imported_by_package_or_injector():
    import subprocess
    import sys
    script = '''
import importlib.abc
import sys
class BlockBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"vibevoice", "qwen_tts", "whisperx"}:
            raise AssertionError("Unexpected optional backend import: " + fullname)
sys.meta_path.insert(0, BlockBackends())
import radio_drama
from radio_drama.config import ProductionConfig
from radio_drama.init import radio_drama_injector
for backend in ("qwen", "whisperx"):
    injector = radio_drama_injector(config=ProductionConfig(alignment_backend=backend, device="cpu"))
    injector.close()
'''
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)
