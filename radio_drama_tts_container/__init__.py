"""Low-dependency helpers for radio-drama TTS container entry points."""

from .server import (
    PROTOCOL,
    PROTOCOL_VERSION,
    artifact_name,
    run_server,
    run_in_thread,
    write_pcm16_wav,
)
from .lines import (
    LineWork,
    fill_ignored_line_spans,
    finish_line_work,
    line_is_synthesizable,
    prepare_line_work,
    remove_line_work,
)
from .speakers import SpeakerSlots

__all__ = [
    "SpeakerSlots",
    "PROTOCOL",
    "PROTOCOL_VERSION",
    "artifact_name",
    "run_server",
    "run_in_thread",
    "write_pcm16_wav",
    "LineWork",
    "fill_ignored_line_spans",
    "finish_line_work",
    "line_is_synthesizable",
    "prepare_line_work",
    "remove_line_work",
]
