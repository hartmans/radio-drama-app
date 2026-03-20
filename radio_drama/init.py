from __future__ import annotations

import asyncio

from carthage.dependency_injection import InjectionKey, Injector

from .config import ProductionConfig
from .resources import VibeVoiceResource


def radio_drama_injector(
    base_injector: Injector | None = None,
    *,
    config: ProductionConfig | None = None,
    event_loop: asyncio.AbstractEventLoop | None = None,
) -> Injector:
    injector = Injector(parent_injector=base_injector)
    if config is not None:
        injector.add_provider(config)
    if event_loop is not None:
        injector.replace_provider(
            InjectionKey(asyncio.AbstractEventLoop),
            event_loop,
            close=False,
        )
    if injector.injector_containing(VibeVoiceResource) is None:
        injector.add_provider(VibeVoiceResource)
    return injector
