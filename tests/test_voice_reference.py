from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from carthage.dependency_injection import AsyncInjector, InjectionKey

from radio_drama.config import ProductionConfig
from radio_drama.dialogue import SpeakerVoiceReference
from radio_drama.forced_alignment import WhisperXResource
from radio_drama.init import radio_drama_injector
from radio_drama.voice_reference import VoiceReferenceTranscriptionResource


def test_voice_reference_transcription_is_cached_and_enriches_in_place(tmp_path: Path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"fake")
    cache_path = tmp_path / "transcript.txt"
    calls = []

    class FakeWhisperX:
        def transcribe_audio_sample_sync(self, audio, sample_rate):
            calls.append((np.array(audio), sample_rate))
            return "Reference words."

    async def runner():
        injector = radio_drama_injector(
            config=ProductionConfig(voice_directory=tmp_path),
            event_loop=asyncio.get_running_loop(),
        )
        injector.replace_provider(
            InjectionKey(WhisperXResource), FakeWhisperX(), close=False
        )
        try:
            resource = await injector(AsyncInjector)(VoiceReferenceTranscriptionResource)
            resource.cache_path = lambda _path: cache_path
            first = SpeakerVoiceReference("A", "voice", voice_path)
            assert resource.transcribe_sync(
                first, np.array([0.25], dtype=np.float32), 16000
            ) == "Reference words."
            second = SpeakerVoiceReference("B", "voice", voice_path)
            assert resource.transcribe_sync(second) == "Reference words."
            return first, second
        finally:
            injector.close()

    first, second = asyncio.run(runner())

    assert first.transcript == "Reference words."
    assert second.transcript == "Reference words."
    assert len(calls) == 1
    assert cache_path.read_text(encoding="utf-8") == "Reference words.\n"
