"""Lazy, reusable audio-plan values for the radio-drama REPL."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections import Counter
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, TypeVar

import soundfile as sf
from carthage.dependency_injection import inject

from ..audio import AudioPlan
from ..config import ProductionConfig
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
    _children: tuple["AudioPlanWrapper", ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __or__(self, other: object):
        if isinstance(other, EffectStage):
            return replace(self, effect_chain=self.effect_chain | other)
        return NotImplemented

    def __add__(self, other: object):
        if not isinstance(other, AudioPlanWrapper):
            return NotImplemented
        return _compose_wrappers((self, other), overlap=False, template=self)

    def __getitem__(self, index: int | slice) -> "AudioPlanWrapper":
        """Select one child or crop the timeline spanning a child slice."""

        children = self._child_wrappers()
        if isinstance(index, slice):
            if index.step not in (None, 1):
                raise ValueError("AudioPlanWrapper slices do not support a step")
            selected_indexes = tuple(range(len(children))[index])
            return _crop_wrapper(
                self,
                selected_indexes=selected_indexes,
                children=tuple(children[child_index] for child_index in selected_indexes),
            )
        return children[index]

    def _child_wrappers(self) -> tuple["AudioPlanWrapper", ...]:
        if self._children is not None:
            return self._children

        wrappers = []
        for index, child in enumerate(self.plan.child_plans()):
            if not isinstance(child, AudioPlan):
                raise TypeError(f"Child {index} is not an AudioPlan")
            wrappers.append(
                replace(
                    self,
                    plan=child,
                    effect_chain=dry(),
                    _layout_root=self._layout_target,
                    _children=None,
                )
            )
        return tuple(wrappers)

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


@inject(config=ProductionConfig)
class ReplComposeAudioPlan(AudioPlan):
    """REPL-local composition of independently reusable wrapped plans."""

    def __init__(
        self,
        *,
        wrappers: tuple[AudioPlanWrapper, ...],
        overlap: bool,
        **kwargs,
    ) -> None:
        kwargs.setdefault("node", None)
        kwargs.setdefault("attrs", {})
        super().__init__(**kwargs)
        self.wrappers = wrappers
        self.overlap = overlap
        self._wrapper_bounds: tuple[tuple[float, float], ...] = ()

    async def layout_node(self) -> None:
        await asyncio.gather(
            *(wrapper._layout_target.layout() for wrapper in self.wrappers)
        )
        lengths = [wrapper.plan.natural_length for wrapper in self.wrappers]
        self._wrapper_bounds = self._calculate_wrapper_bounds(lengths)
        self.inner_last = max(lengths, default=0.0) if self.overlap else sum(lengths)
        self.advance = self.inner_last
        self.mark_positions = self._wrapper_mark_positions(lengths)

    async def render_node(self) -> RenderResult:
        results = await asyncio.gather(*(wrapper.render() for wrapper in self.wrappers))
        if not results:
            return RenderResult.empty(channels=self.config.resolved_output_channels)
        if not self.overlap:
            return RenderResult.concatenate(results)
        frame_count = max(result.frame_count for result in results)
        audio = self._empty_audio(frame_count)
        for result in results:
            audio[: result.frame_count] += result.audio
        return RenderResult(audio=audio)

    def child_plans(self):
        return tuple(wrapper.plan for wrapper in self.wrappers)

    def wrapper_bounds(self, index: int) -> tuple[float, float]:
        return self._wrapper_bounds[index]

    def _calculate_wrapper_bounds(
        self,
        lengths: list[float],
    ) -> tuple[tuple[float, float], ...]:
        offset = 0.0
        bounds = []
        for length in lengths:
            first = 0.0 if self.overlap else offset
            bounds.append((first, first + length))
            if not self.overlap:
                offset += length
        return tuple(bounds)

    def _wrapper_mark_positions(self, lengths: list[float]) -> dict[str, float]:
        occurrences: list[tuple[str, float]] = []
        offset = 0.0
        for wrapper, length in zip(self.wrappers, lengths, strict=True):
            occurrences.extend(
                (name, position + (0.0 if self.overlap else offset))
                for name, position in wrapper.plan.mark_positions.items()
            )
            if not self.overlap:
                offset += length
        counts = Counter(name for name, _ in occurrences)
        return {name: position for name, position in occurrences if counts[name] == 1}


@inject(config=ProductionConfig)
class ReplCropAudioPlan(AudioPlan):
    """Lazy timeline crop of a fully processed source wrapper."""

    def __init__(
        self,
        *,
        source: AudioPlanWrapper,
        selected_indexes: tuple[int, ...],
        **kwargs,
    ) -> None:
        kwargs.setdefault("node", None)
        kwargs.setdefault("attrs", {})
        super().__init__(**kwargs)
        self.source = source
        self.selected_indexes = selected_indexes
        self.crop_first = 0.0
        self.crop_last = 0.0
        self._selected_bounds: tuple[tuple[float, float], ...] = ()

    async def layout_node(self) -> None:
        await self.source._layout_target.layout()
        bounds = tuple(
            self._source_child_bounds(index) for index in self.selected_indexes
        )
        self._selected_bounds = bounds
        if bounds:
            self.crop_first = min(first for first, _ in bounds)
            self.crop_last = max(last for _, last in bounds)
        self.inner_last = max(0.0, self.crop_last - self.crop_first)
        self.advance = self.inner_last
        self.mark_positions = (
            {
                name: position - self.crop_first
                for name, position in self.source.plan.mark_positions.items()
                if self.crop_first <= position <= self.crop_last
            }
            if bounds
            else {}
        )

    async def render_node(self) -> RenderResult:
        if not self.selected_indexes:
            return RenderResult.empty(channels=self.config.resolved_output_channels)
        source_result = await self.source.render()
        start_frame = self._seconds_to_frames(
            self.crop_first - self.source.plan.inner_first
        )
        end_frame = self._seconds_to_frames(
            self.crop_last - self.source.plan.inner_first
        )
        return source_result.slice_frames(start_frame, end_frame)

    def child_plans(self):
        children = tuple(self.source._child_wrappers())
        return tuple(children[index].plan for index in self.selected_indexes)

    def wrapper_bounds(self, index: int) -> tuple[float, float]:
        first, last = self._selected_bounds[index]
        return first - self.crop_first, last - self.crop_first

    def _source_child_bounds(self, index: int) -> tuple[float, float]:
        source_plan = self.source.plan
        if isinstance(source_plan, (ReplComposeAudioPlan, ReplCropAudioPlan)):
            return source_plan.wrapper_bounds(index)
        child = tuple(source_plan.child_plans())[index]
        return child.start + child.inner_first, child.start + child.inner_last


def concatenate(*wrappers: AudioPlanWrapper) -> AudioPlanWrapper:
    """Return a lazy REPL plan that renders wrappers consecutively."""

    if not wrappers:
        raise TypeError("concatenate requires at least one AudioPlanWrapper")
    return _compose_wrappers(wrappers, overlap=False, template=wrappers[0])


def mix(*wrappers: AudioPlanWrapper) -> AudioPlanWrapper:
    """Return a lazy REPL plan that overlaps wrappers at time zero."""

    if not wrappers:
        raise TypeError("mix requires at least one AudioPlanWrapper")
    return _compose_wrappers(wrappers, overlap=True, template=wrappers[0])


def _compose_wrappers(
    wrappers: tuple[AudioPlanWrapper, ...],
    *,
    overlap: bool,
    template: AudioPlanWrapper,
) -> AudioPlanWrapper:
    for wrapper in wrappers:
        if wrapper.sample_rate != template.sample_rate:
            raise ValueError("Cannot compose wrappers with different sample rates")
        if wrapper.submit != template.submit:
            raise ValueError("Cannot compose wrappers from different REPL runtimes")

    async def build_plan() -> ReplComposeAudioPlan:
        return await template.plan.ainjector(
            ReplComposeAudioPlan,
            wrappers=wrappers,
            overlap=overlap,
        )

    plan = template.submit(build_plan()).result()
    return replace(
        template,
        plan=plan,
        effect_chain=dry(),
        _layout_root=None,
        _children=wrappers,
    )


def _crop_wrapper(
    source: AudioPlanWrapper,
    *,
    selected_indexes: tuple[int, ...],
    children: tuple[AudioPlanWrapper, ...],
) -> AudioPlanWrapper:
    async def build_plan() -> ReplCropAudioPlan:
        return await source.plan.ainjector(
            ReplCropAudioPlan,
            source=source,
            selected_indexes=selected_indexes,
        )

    plan = source.submit(build_plan()).result()
    return replace(
        source,
        plan=plan,
        effect_chain=dry(),
        _layout_root=None,
        _children=children,
    )


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
        result = await render_for_output(wrapper)
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


async def render_for_output(wrapper: AudioPlanWrapper) -> RenderResult:
    """Render audio shared by file and device output operations."""

    return await wrapper.render()


class AudioFileWriter:
    """Write rendered wrappers to files through direct and pipe forms."""

    def __init__(self, submit: SubmitCoroutine) -> None:
        self.submit = submit

    def __call__(
        self,
        wrapper: AudioPlanWrapper,
        path: str | Path,
    ) -> concurrent.futures.Future[None]:
        return self.submit(self._write(wrapper, Path(path).expanduser()))

    def terminal(self, path: str | Path) -> "AudioFileOutputTerminal":
        return AudioFileOutputTerminal(writer=self, path=Path(path).expanduser())

    async def _write(self, wrapper: AudioPlanWrapper, path: Path) -> None:
        result = await render_for_output(wrapper)
        await asyncio.to_thread(sf.write, path, result.audio, wrapper.sample_rate)


@dataclass(frozen=True, slots=True)
class AudioFileOutputTerminal:
    """Pipe terminal returned by ``output(path)``."""

    writer: AudioFileWriter
    path: Path

    def __ror__(self, wrapper: object):
        if not isinstance(wrapper, AudioPlanWrapper):
            return NotImplemented
        return self.writer(wrapper, self.path)


__all__ = [
    "AudioPlanWrapper",
    "AudioPlayer",
    "AudioFileOutputTerminal",
    "AudioFileWriter",
    "MarkNamespace",
    "ReplComposeAudioPlan",
    "ReplCropAudioPlan",
    "concatenate",
    "mix",
    "render_for_output",
]
