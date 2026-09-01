"""Interactive workspace for auditioning radio-drama effects.

The REPL deliberately operates on source audio rather than production plans.  A
loaded production contributes its speaker definitions and its production-scoped
effect registry, while ``sound(path) | effect | play()`` provides a small lazy
pipeline for auditioning those effects.
"""

from __future__ import annotations

import argparse
import asyncio
import code
import concurrent.futures
import readline
import rlcompleter
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Coroutine

import numpy as np
import soundfile as sf
import yaml

from .document import ProductionNode, parse_production_file
from .effects import EffectChainRegistry, EffectStage, effect_chain_variables
from .init import radio_drama_injector


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """The document data retained by a :class:`ReplSession`."""

    path: Path
    document: ProductionNode
    speakers: Mapping[str, object]
    effects: Mapping[str, EffectStage]


@dataclass(frozen=True, slots=True)
class ReplSound:
    """A lazily loaded sound and the effect stages to apply before playback."""

    path: Path
    stages: tuple[EffectStage, ...] = ()

    def __or__(self, other: object):
        if isinstance(other, EffectStage):
            return ReplSound(self.path, (*self.stages, other))
        return NotImplemented


class _Play:
    """Terminal pipeline object returned by :func:`play`."""

    def __init__(self, runner: "ReplEventLoop") -> None:
        self.runner = runner

    def __ror__(self, source: object):
        if not isinstance(source, ReplSound):
            return NotImplemented
        return self.runner.submit(_play(source))


class ReplEventLoop:
    """Host one radio-drama injector and its event loop beside the console."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.injector = None
        self.started = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="radio-drama-repl",
            daemon=True,
        )
        self.thread.start()
        self.started.wait()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        self.injector = radio_drama_injector(event_loop=loop)
        self.started.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()

    def submit(
        self, coroutine: Coroutine[Any, Any, None]
    ) -> concurrent.futures.Future[None]:
        """Schedule a coroutine without blocking the interactive console."""

        loop = self.loop
        if loop is None:
            raise RuntimeError("The radio-drama REPL event loop has not started")
        return asyncio.run_coroutine_threadsafe(coroutine, loop)

    def set_document_path(self, document_path: Path) -> None:
        """Rebuild the app injector on its loop with production path context."""

        self.submit(self._set_document_path(document_path)).result()

    async def _set_document_path(self, document_path: Path) -> None:
        old_injector = self.injector
        self.injector = radio_drama_injector(
            event_loop=asyncio.get_running_loop(),
            document_path=document_path,
        )
        if old_injector is not None:
            old_injector.close()

    def close(self) -> None:
        """Stop the companion loop and wait briefly for its thread to exit."""

        loop = self.loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._stop)
            self.thread.join(timeout=2)

    def _stop(self) -> None:
        if self.injector is not None:
            self.injector.close()
            self.injector = None
        if self.loop is not None:
            self.loop.stop()


def sound(path: str | Path) -> ReplSound:
    """Start a lazy audition pipeline for an audio file."""

    return ReplSound(Path(path).expanduser())


def play() -> _Play:
    """Return the terminal stage which renders and plays a sound pipeline."""

    return _Play(_default_event_loop())


async def _play(source: ReplSound) -> None:
    audio, sample_rate = await asyncio.to_thread(_read_sound, source.path)
    for stage in source.stages:
        await asyncio.to_thread(stage.apply, audio, sample_rate=sample_rate)
    await asyncio.to_thread(_play_sound, audio, sample_rate)


def _read_sound(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def _play_sound(audio: np.ndarray, sample_rate: int) -> None:
    try:
        import sounddevice
    except ImportError as exc:  # pragma: no cover - an optional runtime facility
        raise RuntimeError("sounddevice is required for REPL playback") from exc
    sounddevice.play(audio, sample_rate, blocking=True)


_DEFAULT_EVENT_LOOP: ReplEventLoop | None = None


def _default_event_loop() -> ReplEventLoop:
    global _DEFAULT_EVENT_LOOP
    if _DEFAULT_EVENT_LOOP is None:
        _DEFAULT_EVENT_LOOP = ReplEventLoop()
    return _DEFAULT_EVENT_LOOP


class _ReplCompleter:
    """Combine Python-name completion with generic filesystem completion."""

    def __init__(self, namespace: dict[str, object], roots) -> None:
        self.python = rlcompleter.Completer(namespace)
        self.roots = roots
        self.matches: list[str] = []

    def complete(self, text: str, state: int) -> str | None:
        if state == 0:
            self.matches = self._matches(text)
        if state >= len(self.matches):
            return None
        return self.matches[state]

    def _matches(self, text: str) -> list[str]:
        matches: list[str] = []
        state = 0
        while (match := self.python.complete(text, state)) is not None:
            matches.append(match)
            state += 1
        matches.extend(self._path_matches(text))
        return list(dict.fromkeys(matches))

    def _path_matches(self, text: str) -> list[str]:
        typed = Path(text).expanduser()
        if typed.is_absolute():
            searches = [(Path(typed.anchor), Path(*typed.parts[1:]))]
        else:
            searches = [(root, typed) for root in self.roots()]

        matches: list[str] = []
        for root, relative in searches:
            parent = root / relative.parent
            if not parent.is_dir():
                continue
            prefix = relative.name
            for child in sorted(parent.iterdir()):
                if not child.name.startswith(prefix):
                    continue
                completion = relative.parent / child.name
                rendered = str(completion)
                if child.is_dir():
                    rendered += "/"
                matches.append(rendered)
        return matches


class ReplSession:
    """Own the mutable locals and document state used by one interactive REPL."""

    def __init__(self) -> None:
        self.event_loop = ReplEventLoop()
        self.sound_roots = [Path.cwd()]
        effect_variables = effect_chain_variables()
        self._effect_variable_names = set(effect_variables)
        self.locals: dict[str, object] = effect_variables
        self.locals.update(
            {
                "load": self.load,
                "play": self.play,
                "sound": self.sound,
                "speaker_presets": {},
            }
        )
        self.loaded_document: LoadedDocument | None = None

    def sound(self, path: str | Path) -> ReplSound:
        """Start a pipeline, resolving a bare sound name under known sound roots."""

        requested = Path(path).expanduser()
        if not requested.is_absolute() and not requested.is_file():
            for root in self.sound_roots:
                candidate = root / requested
                if candidate.is_file():
                    requested = candidate
                    break
        return ReplSound(requested)

    def play(self) -> _Play:
        """Return a terminal stage backed by this session's companion loop."""

        return _Play(self.event_loop)

    def load(self, path: str | Path) -> LoadedDocument:
        """Load speaker definitions and effect presets from a production XML file."""

        document_path = Path(path).expanduser()
        document = parse_production_file(document_path)
        speakers = _load_speakers(document)
        registry = _load_effects(document)
        effects = registry.stages()
        self.event_loop.set_document_path(document_path)

        loaded = LoadedDocument(
            path=document_path,
            document=document,
            speakers=speakers,
            effects=effects,
        )
        self.loaded_document = loaded
        sounds_root = document_path.parent / "sounds"
        self.sound_roots = [Path.cwd()]
        if sounds_root.is_dir() and sounds_root not in self.sound_roots:
            self.sound_roots.append(sounds_root)
        for name in self._effect_variable_names:
            self.locals.pop(name, None)
        effect_variables = effect_chain_variables(effects)
        self._effect_variable_names = set(effect_variables)
        self.locals.update(effect_variables)
        self.locals["speaker_presets"] = speakers
        self.locals["document"] = loaded
        return loaded

    def interact(self, banner: str | None = None) -> None:
        """Run a standard Python interactive console over this session's locals."""

        if banner is None:
            banner = (
                "radio_drama REPL\n"
                "Use load('production.xml'), then "
                "sound('sound.wav') | preset | play()."
            )
        self._install_completion()
        try:
            code.interact(banner=banner, local=self.locals)
        finally:
            self.event_loop.close()

    def _install_completion(self) -> None:
        completer = _ReplCompleter(self.locals, lambda: self.sound_roots)
        readline.set_completer(completer.complete)
        delimiters = readline.get_completer_delims().replace("/", "").replace("-", "")
        readline.set_completer_delims(delimiters)
        readline.parse_and_bind("tab: complete")


def _load_effects(document: ProductionNode) -> EffectChainRegistry:
    registry = EffectChainRegistry()
    preset_maps = document.child_elements_named("preset-map")
    if not preset_maps:
        return registry
    node = preset_maps[0]
    loaded = yaml.safe_load(node.normalized_text_content)
    if not isinstance(loaded, dict):
        raise node.error("The <preset-map> YAML must be a mapping of preset names to expressions")
    for preset_name, expression in loaded.items():
        if not isinstance(preset_name, str) or not preset_name.strip().isidentifier():
            raise node.error("Preset names in <preset-map> must be valid expression identifiers")
        if not isinstance(expression, str) or not expression.strip():
            raise node.error(
                f"Expression for preset {preset_name!r} must be a non-empty string"
            )
        try:
            registry.add_from_expression(preset_name.strip(), expression.strip())
        except Exception as exc:
            raise node.error(f"Invalid expression for preset {preset_name!r}: {exc}") from exc
    return registry


def _load_speakers(document: ProductionNode) -> dict[str, object]:
    speaker_maps = document.child_elements_named("speaker-map")
    if not speaker_maps:
        return {}
    node = speaker_maps[0]
    loaded = yaml.safe_load(node.normalized_text_content)
    if not isinstance(loaded, dict):
        raise node.error("The <speaker-map> YAML must be a mapping of speaker names to voice names")
    return dict(loaded)


def main(argv: list[str] | None = None) -> None:
    """Run the radio-drama interactive console."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", nargs="?", help="production XML to load before starting")
    args = parser.parse_args(argv)
    session = ReplSession()
    if args.document:
        session.load(args.document)
    session.interact()


def repl(document: str | Path | None = None) -> None:
    """Start an interactive session, optionally with one document preloaded."""

    session = ReplSession()
    if document is not None:
        session.load(document)
    session.interact()


if __name__ == "__main__":
    main()
