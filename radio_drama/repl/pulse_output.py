"""Minimal libpulse-simple output helper used by the REPL playback process."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import sys


PA_STREAM_PLAYBACK = 1
PA_SAMPLE_FLOAT32LE = 5
PA_SAMPLE_FLOAT32BE = 6


class PulseSampleSpec(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_int),
        ("rate", ctypes.c_uint32),
        ("channels", ctypes.c_uint8),
    ]


def play(data: bytes, *, sample_rate: int, channels: int) -> None:
    """Write one float32 buffer to PulseAudio and wait until it has played."""

    simple = ctypes.CDLL(ctypes.util.find_library("pulse-simple") or "libpulse-simple.so.0")
    pulse = ctypes.CDLL(ctypes.util.find_library("pulse") or "libpulse.so.0")
    simple.pa_simple_new.restype = ctypes.c_void_p
    simple.pa_simple_new.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(PulseSampleSpec),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    simple.pa_simple_write.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int),
    ]
    simple.pa_simple_write.restype = ctypes.c_int
    simple.pa_simple_drain.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    simple.pa_simple_drain.restype = ctypes.c_int
    simple.pa_simple_free.argtypes = [ctypes.c_void_p]
    pulse.pa_strerror.argtypes = [ctypes.c_int]
    pulse.pa_strerror.restype = ctypes.c_char_p

    sample_format = PA_SAMPLE_FLOAT32LE if sys.byteorder == "little" else PA_SAMPLE_FLOAT32BE
    sample_spec = PulseSampleSpec(sample_format, sample_rate, channels)
    error = ctypes.c_int()
    connection = simple.pa_simple_new(
        None,
        b"radio-drama-repl",
        PA_STREAM_PLAYBACK,
        None,
        b"REPL playback",
        ctypes.byref(sample_spec),
        None,
        None,
        ctypes.byref(error),
    )
    if not connection:
        raise RuntimeError(_pulse_error(pulse, error.value))
    try:
        buffer = ctypes.create_string_buffer(data)
        if simple.pa_simple_write(connection, buffer, len(data), ctypes.byref(error)) < 0:
            raise RuntimeError(_pulse_error(pulse, error.value))
        if simple.pa_simple_drain(connection, ctypes.byref(error)) < 0:
            raise RuntimeError(_pulse_error(pulse, error.value))
    finally:
        simple.pa_simple_free(connection)


def _pulse_error(pulse, error: int) -> str:
    return pulse.pa_strerror(error).decode(errors="replace")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_rate", type=int)
    parser.add_argument("channels", type=int)
    args = parser.parse_args(argv)
    play(sys.stdin.buffer.read(), sample_rate=args.sample_rate, channels=args.channels)


if __name__ == "__main__":
    main()
