"""Interactive workspace for auditioning radio-drama effects.

The REPL deliberately operates on source audio rather than production plans.  A
loaded production contributes its speaker definitions and its production-scoped
effect registry, while ``sound(path) | effect | play()`` provides a small lazy
pipeline for auditioning those effects.
"""

from __future__ import annotations

import asyncio
import code
import concurrent.futures
import readline
import rlcompleter
import threading
from xml.sax.saxutils import quoteattr
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Coroutine, TypeVar

import yaml
from carthage.dependency_injection import AsyncInjector

from ..config import ProductionConfig
from ..document import ProductionNode, parse_production_file, parse_production_string
from ..effects import EffectStage, effect_chain_variables
from ..init import radio_drama_injector
from ..sound import ProductionDocumentPath
from .audio_wrapper import AudioPlanWrapper, AudioPlayer
from .pulse_player import PulseAudioPlayer


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """The document data retained by a :class:`ReplSession`."""

    path: Path
    document: ProductionNode
    speakers: Mapping[str, object]
    effects: Mapping[str, EffectStage]
    production: AudioPlanWrapper


class ReplEventLoop:
    """Host one radio-drama injector and its event loop beside the console."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.injector = None
        self._retired_injectors = []
        self.config = ProductionConfig()
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
        self.injector = radio_drama_injector(
            config=self.config,
            event_loop=loop,
        )
        self.started.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()

    def submit(self, coroutine: Coroutine[Any, Any, _T]) -> concurrent.futures.Future[_T]:
        """Schedule a coroutine without blocking the interactive console."""

        loop = self.loop
        if loop is None:
            raise RuntimeError("The radio-drama REPL event loop has not started")
        return asyncio.run_coroutine_threadsafe(coroutine, loop)

    def set_document_path(self, document_path: Path) -> None:
        """Rebuild the app injector on its loop with production path context."""

        self.submit(self._set_document_path(document_path)).result()

    async def _set_document_path(self, document_path: Path) -> None:
        self._retired_injectors.append(self.injector)
        self.injector = radio_drama_injector(
            config=self.config,
            event_loop=asyncio.get_running_loop(),
            document_path=document_path,
        )

    def sound_plan(self, reference: str) -> AudioPlanWrapper:
        """Construct a sound plan on the injector's event loop without rendering it."""

        return self.submit(self._sound_plan(reference)).result()

    async def _sound_plan(self, reference: str) -> AudioPlanWrapper:
        source_name = None
        provider = self.injector.injector_containing(ProductionDocumentPath)
        if provider is not None:
            source_name = str(provider.get_instance(ProductionDocumentPath).path)
        document = parse_production_string(
            f"<production><sound ref={quoteattr(reference)} /></production>",
            source_name=source_name,
        )
        sound_node = document.element_children[0]
        plan = await sound_node.plan(self.injector(AsyncInjector))
        config = self.injector.get_instance(ProductionConfig)
        return AudioPlanWrapper(
            plan=plan,
            sample_rate=config.resolved_output_sample_rate,
            submit=self.submit,
        )

    def production_plan(self, document: ProductionNode) -> AudioPlanWrapper:
        """Plan a production without laying it out or rendering it."""

        return self.submit(self._production_plan(document)).result()

    async def _production_plan(self, document: ProductionNode) -> AudioPlanWrapper:
        plan = await document.plan(self.injector(AsyncInjector))
        config = self.injector.get_instance(ProductionConfig)
        return AudioPlanWrapper(
            plan=plan,
            sample_rate=config.resolved_output_sample_rate,
            submit=self.submit,
        )

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
        for injector in reversed(self._retired_injectors):
            injector.close()
        self._retired_injectors.clear()
        if self.loop is not None:
            self.loop.stop()


_T = TypeVar("_T")


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
        self.pulse_player = PulseAudioPlayer()
        self.player = AudioPlayer(
            self.event_loop.submit,
            self.pulse_player,
        )
        self.sound_roots = [Path.cwd()]
        effect_variables = effect_chain_variables()
        self._effect_variable_names = set(effect_variables)
        self.locals: dict[str, object] = effect_variables
        self.locals.update(
            {
                "load": self.load,
                "play": self.play,
                "sound": self.sound,
                "stop": self.stop,
                "speaker_presets": {},
            }
        )
        self.loaded_document: LoadedDocument | None = None

    def sound(self, path: str | Path) -> AudioPlanWrapper:
        """Wrap a lazily rendered ``SoundPlan`` resolved by the app injector."""

        requested = Path(path).expanduser()
        if requested.is_file():
            requested = requested.resolve()
        elif not requested.is_absolute():
            for root in self.sound_roots:
                candidate = root / requested
                if candidate.is_file():
                    requested = candidate
                    break
        return self.event_loop.sound_plan(str(requested))

    def play(self, wrapper: AudioPlanWrapper | None = None):
        """Play a wrapper directly, or return the playback pipe terminal."""

        return self.player(wrapper)

    def stop(self) -> None:
        """Stop pending rendering and any active PulseAudio playback."""

        self.player.stop()

    def load(self, path: str | Path) -> AudioPlanWrapper:
        """Plan and wrap a production without laying it out or rendering it."""

        document_path = Path(path).expanduser()
        document = parse_production_file(document_path)
        speakers = _load_speakers(document)
        self.event_loop.set_document_path(document_path)
        production = self.event_loop.production_plan(document)
        effects = production.plan.effect_chains.stages()

        loaded = LoadedDocument(
            path=document_path,
            document=document,
            speakers=speakers,
            effects=effects,
            production=production,
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
        self.locals["production"] = production
        return production

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
            self.stop()
            self.event_loop.close()

    def _install_completion(self) -> None:
        completer = _ReplCompleter(self.locals, lambda: self.sound_roots)
        readline.set_completer(completer.complete)
        delimiters = readline.get_completer_delims().replace("/", "").replace("-", "")
        readline.set_completer_delims(delimiters)
        readline.parse_and_bind("tab: complete")


def _load_speakers(document: ProductionNode) -> dict[str, object]:
    speaker_maps = document.child_elements_named("speaker-map")
    if not speaker_maps:
        return {}
    node = speaker_maps[0]
    loaded = yaml.safe_load(node.normalized_text_content)
    if not isinstance(loaded, dict):
        raise node.error("The <speaker-map> YAML must be a mapping of speaker names to voice names")
    return dict(loaded)


def repl(document: str | Path | None = None) -> None:
    """Start an interactive session, optionally with one document preloaded."""

    session = ReplSession()
    if document is not None:
        session.load(document)
    session.interact()
