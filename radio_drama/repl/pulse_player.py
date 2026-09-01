"""PulseAudio playback process management for the radio-drama REPL."""

from __future__ import annotations

import subprocess
import sys
import threading

import numpy as np

from ..audio import normalize_audio_array


class PulseAudioPlayer:
    """Play one buffer through an isolated libpulse-simple process.

    libpulse honors environment settings such as ``PULSE_SERVER``. Only one
    owned process is active at a time; starting or stopping playback terminates
    the previous process.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._generation = 0

    def begin(self) -> object:
        """Reserve the newest playback slot and terminate its predecessor."""

        with self._lock:
            self._generation += 1
            self._terminate_locked()
            return self._generation

    def play(self, audio: object, sample_rate: int, token: object) -> None:
        array = normalize_audio_array(np.asarray(audio))
        channels = 1 if array.ndim == 1 else array.shape[1]
        command = [
            sys.executable,
            "-m",
            "radio_drama.repl.pulse_output",
            str(sample_rate),
            str(channels),
        ]
        with self._lock:
            if token != self._generation:
                return
            self._terminate_locked()
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._process = process
        _, stderr = process.communicate(array.tobytes())
        with self._lock:
            was_current = self._process is process
            if was_current:
                self._process = None
        if process.returncode and was_current:
            message = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"PulseAudio playback failed: {message}")

    def stop(self) -> None:
        """Terminate the currently owned PulseAudio playback process, if any."""

        with self._lock:
            self._generation += 1
            self._terminate_locked()

    def _terminate_locked(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


__all__ = ["PulseAudioPlayer"]
