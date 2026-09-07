"""Reference-based speaker slot assignment for container TTS engines."""

from collections.abc import Mapping
from typing import Any


class SpeakerSlots:
    """Assign zero-based slots by voice path and gain, in first-use order.

    Create one instance per model request. ``references`` retains the first
    speaker mapping for each slot, including its reference transcript. Names
    and output effects do not distinguish slots. Paths are supplied by the
    host, which shares a mounted path for identical source references and gain.
    Gain is identity metadata here; the mounted audio already includes it.
    """

    def __init__(self) -> None:
        self.references: list[Mapping[str, Any]] = []
        self._slots: dict[tuple[str, float], int] = {}

    def assign(self, speaker: Mapping[str, Any]) -> int:
        """Return the existing slot or append this reference to a new slot."""
        key = (str(speaker["voice_path"]), float(speaker.get("gain", 0.0)))
        if key not in self._slots:
            self._slots[key] = len(self.references)
            self.references.append(speaker)
        return self._slots[key]
