"""Lazy, reusable audio-plan values for the radio-drama REPL."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, TypeVar

from ..audio import AudioPlan
from ..effects import EffectStage, dry
from ..rendering import RenderResult


_T = TypeVar("_T")
Submit = Callable[
    [Coroutine[Any, Any, _T]],
    concurrent.futures.Future[_T],
]


@dataclass(frozen=True, slots=True)
class AudioPlanWrapper:
    """An immutable plan reference plus a REPL-local effect chain.

    The wrapped plan remains responsible for its document-authored effects.
    ``effect_chain`` is applied afterwards to a copy of the plan's memoized
    render, allowing any number of wrappers to reuse the plan independently.
    Dataclass subclasses may add fields; composition uses ``replace()`` so
    those fields are retained in the returned subclass instance.
    """

    plan: AudioPlan
    sample_rate: int
    submit: Submit = field(repr=False, compare=False)
    effect_chain: EffectStage = field(default_factory=dry)
    _layout_root: AudioPlan | None = field(default=None, repr=False, compare=False)

    def __or__(self, other: object):
        if isinstance(other, EffectStage):
            return replace(self, effect_chain=self.effect_chain | other)
        return NotImplemented

    def __getitem__(self, index: int) -> "AudioPlanWrapper":
        """Return a dry wrapper around one direct child of the wrapped plan."""

        child = tuple(self.plan.child_plans())[index]
        if not isinstance(child, AudioPlan):
            raise TypeError(f"Child {index} is not an AudioPlan")
        return replace(
            self,
            plan=child,
            effect_chain=dry(),
            _layout_root=self._layout_target,
        )

    @property
    def _layout_target(self) -> AudioPlan:
        return self._layout_root or self.plan

    def layout(self) -> "AudioPlanWrapper":
        """Complete layout on the injector loop and return this wrapper."""

        self.submit(self._layout_target.layout()).result()
        return self

    @property
    def marks(self) -> "MarkNamespace":
        """Return a namespace providing lazy access to laid-out marks."""

        return MarkNamespace(self)

    async def render(self) -> RenderResult:
        """Render the plan lazily and apply this wrapper's chain to a copy."""

        await self._layout_target.layout()
        plan_result = await self.plan.render()
        result = RenderResult(audio=plan_result.audio.copy())
        await asyncio.to_thread(
            self.effect_chain.apply,
            result.audio,
            sample_rate=self.sample_rate,
        )
        return result


SubmitCoroutine = Submit

class AudioOutput(Protocol):
    """Single-winner output controlled by an opaque playback token."""

    def begin(self) -> object: ...

    def play(self, audio: object, sample_rate: int, token: object) -> None: ...

    def stop(self) -> None: ...


class MarkNamespace:
    """Attribute and item lookup for one wrapper's laid-out mark positions."""

    __slots__ = ("wrapper",)

    def __init__(self, wrapper: AudioPlanWrapper) -> None:
        self.wrapper = wrapper

    def __getitem__(self, name: str) -> float:
        self.wrapper.layout()
        return self.wrapper.plan.mark_positions[name]

    def __getattr__(self, name: str) -> float:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class AudioPlayer:
    """Callable and pipe-terminal interface for asynchronous REPL playback."""

    def __init__(
        self,
        submit: SubmitCoroutine,
        output: AudioOutput,
    ) -> None:
        self.submit = submit
        self.output = output
        self._lock = threading.Lock()
        self._future: concurrent.futures.Future[None] | None = None

    def __call__(
        self, wrapper: AudioPlanWrapper | None = None
    ) -> "AudioPlayer | concurrent.futures.Future[None]":
        if wrapper is None:
            return self
        self.stop()
        token = self.output.begin()
        future = self.submit(self._play(wrapper, token))
        with self._lock:
            self._future = future
        future.add_done_callback(self._finished)
        return future

    def __ror__(self, wrapper: object):
        if not isinstance(wrapper, AudioPlanWrapper):
            return NotImplemented
        return self(wrapper)

    async def _play(self, wrapper: AudioPlanWrapper, token: object) -> None:
        result = await wrapper.render()
        await asyncio.to_thread(
            self.output.play,
            result.audio,
            wrapper.sample_rate,
            token,
        )

    def stop(self) -> None:
        """Cancel pending rendering and terminate active output."""

        with self._lock:
            future = self._future
            self._future = None
        if future is not None:
            future.cancel()
        self.output.stop()

    def _finished(self, future: concurrent.futures.Future[None]) -> None:
        with self._lock:
            if self._future is future:
                self._future = None


__all__ = ["AudioPlanWrapper", "AudioPlayer", "MarkNamespace"]
