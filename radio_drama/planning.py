from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import TYPE_CHECKING

from carthage.dependency_injection import AsyncInjectable, InjectionKey

from .errors import DocumentError


if TYPE_CHECKING:
    from .document import DocumentNode


PRODUCTION_PLANNING_INJECTOR_KEY = InjectionKey("radio_drama.production_planning_injector")
AudioAttrValue = float | str | bool
AudioAttrs = dict[str, AudioAttrValue]


class PlanningNode(AsyncInjectable):
    """Base class for injectable planning objects.

    Planning nodes keep the source ``DocumentNode`` that produced them and
    provide a memoized async ``render()`` entry point so downstream callers do
    not need to coordinate duplicate work themselves.
    """

    def __init__(self, node: DocumentNode | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.node = node
        self._render_task: asyncio.Task | None = None

    def document_error(self, message: str, *, node: DocumentNode | None = None) -> DocumentError:
        target = node or self.node
        if target is None:
            return DocumentError(message)
        return target.error(message)

    async def render(self):
        if self._render_task is None:
            self._render_task = asyncio.create_task(self.render_node())
        try:
            return await self._render_task
        except BaseException:
            self._render_task = None
            raise

    async def render_node(self):
        return None

    def child_plans(self) -> Iterable["PlanningNode"]:
        return ()

    def all_plans(self) -> Iterable["PlanningNode"]:
        seen: set[int] = set()
        yield from self._all_plans_seen(seen)

    def _all_plans_seen(self, seen: set[int]) -> Iterable["PlanningNode"]:
        identity = id(self)
        if identity in seen:
            return
        seen.add(identity)
        yield self
        for child in self.child_plans():
            yield from child._all_plans_seen(seen)


def __getattr__(name: str):
    if name in {"AudioPlan", "ComposeAudioPlan", "LoopPlan", "MarkPlan", "SlicePlan"}:
        from . import audio

        return getattr(audio, name)
    if name in {
        "DialogueContent",
        "DialogueAudio",
        "DialogueLine",
        "ScriptEvent",
        "ScriptGap",
        "ScriptPlan",
        "ScriptRenderRequest",
        "SpeakerMapPlan",
        "SpeakerVoiceReference",
    }:
        from . import dialogue

        return getattr(dialogue, name)
    if name == "ProductionPlan":
        from .production import ProductionPlan

        return ProductionPlan
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AudioAttrs",
    "AudioAttrValue",
    "AudioPlan",
    "ComposeAudioPlan",
    "DialogueContent",
    "DialogueAudio",
    "DialogueLine",
    "LoopPlan",
    "MarkPlan",
    "PlanningNode",
    "PRODUCTION_PLANNING_INJECTOR_KEY",
    "ProductionPlan",
    "ScriptEvent",
    "ScriptGap",
    "ScriptPlan",
    "ScriptRenderRequest",
    "SlicePlan",
    "SpeakerMapPlan",
    "SpeakerVoiceReference",
]
