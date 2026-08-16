from __future__ import annotations

import asyncio
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from carthage.dependency_injection import InjectionKey, Injector

from .cache import CACHE_OUTPUT_PATH_KEY, CacheManager
from .config import ProductionConfig
from .forced_alignment import WhisperXResource
from .qwen_tts import QwenTtsResource
from .dialogue import TtsResource
from .proxy import configured_proxy_resource, load_proxy_tts_configs
from .vibevoice import VibeVoiceResource
from .sound import NormalizedSoundCache, ProductionDocumentPath
from .voice_reference import VoiceReferenceTranscriptionResource


_executor_configured_loops: weakref.WeakSet[asyncio.AbstractEventLoop] = weakref.WeakSet()


def _configure_default_executor(event_loop: asyncio.AbstractEventLoop) -> None:
    """Give one application loop a worker for each CPU available to this process."""

    if event_loop in _executor_configured_loops:
        return
    process_cpu_count = getattr(os, "process_cpu_count", os.cpu_count)
    event_loop.set_default_executor(
        ThreadPoolExecutor(
            max_workers=process_cpu_count() or 1,
            thread_name_prefix="radio-drama",
        )
    )
    _executor_configured_loops.add(event_loop)


def radio_drama_injector(
    base_injector: Injector | None = None,
    *,
    config: ProductionConfig | None = None,
    event_loop: asyncio.AbstractEventLoop | None = None,
    document_path: Path | None = None,
    output_path: Path | None = None,
) -> Injector:
    """Build a radio-drama injector with shared app-level resources.

    The returned injector preserves caller-provided providers from
    ``base_injector`` and installs the production config, event loop, and a
    default speech-resource providers when they are not already present.
    Library entry points and the CLI use the same helper so resource wiring is
    consistent across direct and subprocess-driven execution.
    """
    injector = Injector(parent_injector=base_injector)
    if config is not None:
        injector.add_provider(config)
    if document_path is not None:
        injector.replace_provider(
            InjectionKey(ProductionDocumentPath),
            ProductionDocumentPath(Path(document_path)),
        )
    if injector.injector_containing(CACHE_OUTPUT_PATH_KEY) is None:
        resolved_output_path = None
        if output_path is not None:
            resolved_output_path = Path(output_path)
        elif document_path is not None:
            resolved_output_path = Path(document_path).with_suffix(".wav")
        if resolved_output_path is not None:
            injector.add_provider(CACHE_OUTPUT_PATH_KEY, resolved_output_path)
    if event_loop is not None:
        _configure_default_executor(event_loop)
        injector.replace_provider(
            InjectionKey(asyncio.AbstractEventLoop),
            event_loop,
            close=False,
        )
    if injector.injector_containing(CacheManager) is None:
        injector.add_provider(CacheManager)
    proxy_configs = load_proxy_tts_configs()
    builtin_tts_resources = {
        "vibevoice": VibeVoiceResource,
        "qwen": QwenTtsResource,
    }
    for name, resource_type in builtin_tts_resources.items():
        if name in proxy_configs:
            continue
        resource_key = InjectionKey(TtsResource, tts=name)
        if injector.injector_containing(resource_key) is None:
            injector.add_provider(resource_key, resource_type)
    for name, proxy_config in proxy_configs.items():
        proxy_key = InjectionKey(TtsResource, tts=name)
        if injector.injector_containing(proxy_key) is None:
            injector.add_provider(proxy_key, configured_proxy_resource(proxy_config))
    if injector.injector_containing(WhisperXResource) is None:
        injector.add_provider(WhisperXResource)
    if injector.injector_containing(VoiceReferenceTranscriptionResource) is None:
        injector.add_provider(VoiceReferenceTranscriptionResource)
    if injector.injector_containing(NormalizedSoundCache) is None:
        injector.add_provider(NormalizedSoundCache)
    return injector
