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
    """Build a radio-drama injector with shared app-level resources.

    The returned injector preserves caller-provided providers from
    ``base_injector`` and installs the production config, event loop, and a
    default ``VibeVoiceResource`` provider when one is not already present.
    Library entry points and the CLI use the same helper so resource wiring is
    consistent across direct and subprocess-driven execution.
    """
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
