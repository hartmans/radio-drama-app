from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Lock

import numpy as np

from carthage.dependency_injection import AsyncInjectable, inject

from .config import ProductionConfig
from .dialogue import SpeakerVoiceReference
from .effects import VOICE_PREPROCESS_VERSION, load_preprocessed_voice_reference
from .forced_alignment import WhisperXResource


_TRANSCRIPT_CACHE_DIRECTORY = Path(
    "~/.cache/radio_drama/voice_transcripts"
).expanduser()


@inject(config=ProductionConfig, whisperx_resource=WhisperXResource)
class VoiceReferenceTranscriptionResource(AsyncInjectable):
    """Transcribe and cache reusable speaker voice references.

    Speaker references are shared by every dialogue line for that speaker, so
    successful resolution enriches the reference in place for all consumers.
    """

    def __init__(self, whisperx_resource: WhisperXResource, **kwargs) -> None:
        super().__init__(**kwargs)
        self.whisperx_resource = whisperx_resource
        self._lock = Lock()

    async def transcribe(self, reference: SpeakerVoiceReference) -> str:
        if reference.transcript is None:
            await asyncio.to_thread(self.transcribe_sync, reference)
        assert reference.transcript is not None
        return reference.transcript

    def transcribe_sync(
        self,
        reference: SpeakerVoiceReference,
        audio: np.ndarray | None = None,
        sample_rate: int | None = None,
    ) -> str:
        if reference.transcript is not None:
            return reference.transcript
        path = reference.resolved_path.expanduser().resolve()
        with self._lock:
            if reference.transcript is not None:
                return reference.transcript
            cache_path = self.cache_path(path)
            try:
                transcript = cache_path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                if audio is None:
                    audio, sample_rate = load_preprocessed_voice_reference(path)
                if sample_rate is None:
                    raise ValueError("sample_rate is required with prepared reference audio")
                transcript = self.whisperx_resource.transcribe_audio_sample_sync(
                    audio, sample_rate
                ).strip()
                if not transcript:
                    raise RuntimeError(f"ASR returned no transcript for voice reference {path}")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(transcript + "\n", encoding="utf-8")
            if not transcript:
                raise RuntimeError(f"Cached transcript is empty for voice reference {path}")
            reference.transcript = transcript
            return transcript

    def cache_path(self, voice_path: Path) -> Path:
        voice_path = voice_path.expanduser().resolve()
        try:
            relative = voice_path.relative_to(
                self.config.resolved_voice_directory.resolve()
            )
        except ValueError:
            relative = Path("external") / voice_path.relative_to(voice_path.anchor)
        versioned_relative = Path(VOICE_PREPROCESS_VERSION) / relative
        return _TRANSCRIPT_CACHE_DIRECTORY / versioned_relative.with_suffix(
            f"{relative.suffix}.txt"
        )


__all__ = ["VoiceReferenceTranscriptionResource"]
