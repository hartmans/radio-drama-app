from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass

import numpy as np
from carthage.dependency_injection import AsyncInjector

from radio_drama.audio import AudioPlan
from radio_drama.cache import CacheManager
from radio_drama.effects import gain
from radio_drama.expressions import line
from radio_drama.rendering import RenderResult
from radio_drama.repl.audio_wrapper import AudioPlanWrapper, AudioPlayer
from radio_drama.repl.console import ReplSession


class FakeAudioPlan(AudioPlan):
    def __init__(self, *, children=()) -> None:
        self.result = RenderResult(audio=np.full(4, 0.25, dtype=np.float32))
        self.children = children
        self.layout_count = 0
        self.mark_positions: dict[str, float] = {}

    async def layout(self) -> None:
        self.layout_count += 1

    async def render(self) -> RenderResult:
        return self.result

    def child_plans(self):
        return self.children


def submit(coroutine):
    future = concurrent.futures.Future()
    future.set_result(asyncio.run(coroutine))
    return future


@dataclass(frozen=True, slots=True)
class NamedAudioPlanWrapper(AudioPlanWrapper):
    name: str = "sound"


def test_effect_composition_preserves_wrapper_and_plan() -> None:
    plan = FakeAudioPlan()
    original = NamedAudioPlanWrapper(
        plan=plan,
        sample_rate=48_000,
        submit=submit,
        name="phone",
    )

    louder = original | gain(line(6.0206))
    quieter = original | gain(line(-6.0206))

    assert isinstance(louder, NamedAudioPlanWrapper)
    assert louder.name == "phone"
    assert louder.plan is plan
    assert quieter.plan is plan
    assert original.effect_chain is not louder.effect_chain

    original_result, louder_result, quieter_result = asyncio.run(
        _render_all(original, louder, quieter)
    )
    np.testing.assert_allclose(original_result.audio, 0.25, atol=1e-5)
    np.testing.assert_allclose(louder_result.audio, 0.5, atol=1e-4)
    np.testing.assert_allclose(quieter_result.audio, 0.125, atol=1e-4)
    np.testing.assert_allclose(plan.result.audio, 0.25, atol=1e-5)


async def _render_all(*wrappers: AudioPlanWrapper):
    return await asyncio.gather(*(wrapper.render() for wrapper in wrappers))


def test_player_supports_call_and_pipe_forms() -> None:
    wrapper = AudioPlanWrapper(
        plan=FakeAudioPlan(),
        sample_rate=24_000,
        submit=submit,
    )
    played: list[tuple[np.ndarray, int]] = []

    class Output:
        generation = 0

        def begin(self):
            self.generation += 1
            return self.generation

        def play(self, audio, sample_rate, token):
            assert token == self.generation
            played.append((audio.copy(), sample_rate))

        def stop(self):
            self.generation += 1

    player = AudioPlayer(submit, Output())
    player(wrapper).result(timeout=5)
    (wrapper | player()).result(timeout=5)

    assert [sample_rate for _, sample_rate in played] == [24_000, 24_000]


def test_children_preserve_layout_root_and_marks_trigger_layout() -> None:
    child = FakeAudioPlan()
    child.mark_positions = {"door": 1.25}
    parent = FakeAudioPlan(children=(child,))
    wrapper = AudioPlanWrapper(plan=parent, sample_rate=48_000, submit=submit)

    child_wrapper = wrapper[0]

    assert child_wrapper.plan is child
    assert parent.layout_count == 0
    assert child_wrapper.marks["door"] == 1.25
    assert child_wrapper.marks.door == 1.25
    assert parent.layout_count == 2
    assert child.layout_count == 0


def test_explicit_layout_returns_wrapper() -> None:
    plan = FakeAudioPlan()
    wrapper = AudioPlanWrapper(plan=plan, sample_rate=48_000, submit=submit)

    assert wrapper.layout() is wrapper
    assert plan.layout_count == 1


def test_load_returns_unlaid_production_wrapper(tmp_path) -> None:
    production_path = tmp_path / "production.xml"
    production_path.write_text(
        '<production><mark id="opening" /></production>',
        encoding="utf-8",
    )
    session = ReplSession()
    try:
        production = session.load(production_path)

        assert isinstance(production, AudioPlanWrapper)
        assert production.plan._layout_task is None
        assert production.marks.opening == 0.0
        assert production.plan._layout_task is not None
    finally:
        session.stop()
        session.event_loop.close()


def test_load_uses_each_productions_natural_cache(tmp_path) -> None:
    first_path = tmp_path / "first.xml"
    second_path = tmp_path / "second.xml"
    first_path.write_text("<production />", encoding="utf-8")
    second_path.write_text("<production />", encoding="utf-8")
    session = ReplSession()
    try:
        first = session.load(first_path)
        first_cache = _current_cache_root(session)
        second = session.load(second_path)
        second_cache = _current_cache_root(session)

        assert first_cache == tmp_path / "first.wav.cache"
        assert second_cache == tmp_path / "second.wav.cache"
        assert first.plan.ainjector.injector is not second.plan.ainjector.injector
    finally:
        session.stop()
        session.event_loop.close()


def _current_cache_root(session: ReplSession):
    async def get_root():
        manager = await session.event_loop.injector(AsyncInjector).get_instance_async(
            CacheManager
        )
        return manager.root_directory

    return session.event_loop.submit(get_root()).result()
