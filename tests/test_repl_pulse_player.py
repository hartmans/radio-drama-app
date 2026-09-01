from __future__ import annotations

import threading

import numpy as np

from radio_drama.repl import pulse_player


class FakeProcess:
    def __init__(self, command) -> None:
        self.command = command
        self.done = threading.Event()
        self.returncode = None
        self.terminated = False
        self.input = None

    def communicate(self, input_bytes):
        self.input = input_bytes
        self.done.wait(timeout=5)
        return b"", b""

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.done.set()

    def wait(self, timeout=None):
        assert self.done.wait(timeout)
        return self.returncode

    def kill(self):
        self.returncode = -9
        self.done.set()

    def finish(self):
        self.returncode = 0
        self.done.set()


def test_latest_pulse_playback_wins_and_stop_terminates(monkeypatch) -> None:
    processes: list[FakeProcess] = []

    def popen(command, **kwargs):
        process = FakeProcess(command)
        processes.append(process)
        return process

    monkeypatch.setattr(pulse_player.subprocess, "Popen", popen)
    output = pulse_player.PulseAudioPlayer()
    audio = np.zeros(8, dtype=np.float32)

    first_token = output.begin()
    first_thread = threading.Thread(target=output.play, args=(audio, 48_000, first_token))
    first_thread.start()
    assert _wait_for(lambda: len(processes) == 1)

    second_token = output.begin()
    first_thread.join(timeout=5)
    assert processes[0].terminated

    second_thread = threading.Thread(target=output.play, args=(audio, 48_000, second_token))
    second_thread.start()
    assert _wait_for(lambda: len(processes) == 2)
    output.stop()
    second_thread.join(timeout=5)

    assert processes[1].terminated
    assert "radio_drama.repl.pulse_output" in processes[1].command
    assert len(processes[1].input) == 8 * np.dtype(np.float32).itemsize

    output.play(audio, 48_000, first_token)
    assert len(processes) == 2


def _wait_for(predicate) -> bool:
    for _ in range(1_000):
        if predicate():
            return True
        threading.Event().wait(0.001)
    return False
