"""Full Python REPL for interactively exploring radio-drama plans and effects."""

from __future__ import annotations

from pathlib import Path

from .audio_wrapper import (
    AudioPlanWrapper,
    AudioPlayer,
    AudioFileOutputTerminal,
    AudioFileWriter,
    MarkNamespace,
    ReplComposeAudioPlan,
    ReplCropAudioPlan,
    concatenate,
    mix,
)
from .console import LoadedDocument, ReplEventLoop, ReplSession, repl


_default_session: ReplSession | None = None


def _session() -> ReplSession:
    global _default_session
    if _default_session is None:
        _default_session = ReplSession()
    return _default_session


def sound(path: str | Path) -> AudioPlanWrapper:
    """Create a lazy sound wrapper using the default programmatic session."""

    return _session().sound(path)


def play(wrapper: AudioPlanWrapper | None = None):
    """Play a wrapper, or return the default session's playback terminal."""

    return _session().play(wrapper)


def output(
    wrapper_or_path: AudioPlanWrapper | str | Path,
    path: str | Path | None = None,
):
    """Write ``output(plan, file)`` or return the ``output(file)`` terminal."""

    return _session().output(wrapper_or_path, path)


def stop() -> None:
    """Stop pending or active playback in the default programmatic session."""

    _session().stop()


__all__ = [
    "AudioPlanWrapper",
    "AudioPlayer",
    "AudioFileOutputTerminal",
    "AudioFileWriter",
    "LoadedDocument",
    "MarkNamespace",
    "ReplEventLoop",
    "ReplComposeAudioPlan",
    "ReplCropAudioPlan",
    "ReplSession",
    "concatenate",
    "mix",
    "output",
    "play",
    "repl",
    "sound",
    "stop",
]
