from __future__ import annotations

import asyncio
from pathlib import Path

from carthage.dependency_injection import InjectionKey, Injector

from .config import ProductionConfig
from .forced_alignment import WhisperXResource
from .qwen_tts import QwenTtsResource
from .vibevoice import VibeVoiceResource
from .sound import NormalizedSoundCache, ProductionDocumentPath


def radio_drama_injector(
    base_injector: Injector | None = None,
    *,
    config: ProductionConfig | None = None,
    event_loop: asyncio.AbstractEventLoop | None = None,
    document_path: Path | None = None,
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
    if event_loop is not None:
        injector.replace_provider(
            InjectionKey(asyncio.AbstractEventLoop),
            event_loop,
            close=False,
        )
    if injector.injector_containing(VibeVoiceResource) is None:
        injector.add_provider(VibeVoiceResource)
    if injector.injector_containing(QwenTtsResource) is None:
        injector.add_provider(QwenTtsResource)
    if injector.injector_containing(WhisperXResource) is None:
        injector.add_provider(WhisperXResource)
    if injector.injector_containing(NormalizedSoundCache) is None:
        injector.add_provider(NormalizedSoundCache)
    return injector
