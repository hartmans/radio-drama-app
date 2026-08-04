"""Interactive VoxCPM2 voice design and cloning helper."""

# VoxCPM2 needs a much newer transformers release than the rest of this project.
from pathlib import Path

import sh
import soundfile as sf
import torch
from voxcpm import VoxCPM


OUTPUT_PATH = "audio.wav"
ASR_MODEL_ID = "microsoft/VibeVoice-ASR-HF"
ASR_MAX_NEW_TOKENS = 1024

_asr_processor = None
_asr_model = None
model = None


def _ensure_tts_model():
    """Load and cache the VoxCPM2 TTS model."""
    global model

    if model is None:
        model = VoxCPM.from_pretrained(
            "openbmb/VoxCPM2",
            load_denoiser=False,
            device="cuda",
        )
    return model


def play(text, control=None, **kwargs):
    """Generate speech, write it to ``audio.wav``, and play it.

    ``control`` is an optional VoxCPM2 voice-design instruction. Keyword
    arguments are passed through to :meth:`VoxCPM.generate`, allowing cloning
    calls such as ``play(text, reference_wav_path="voice.wav")`` as well as
    tuning parameters such as ``cfg_value`` and ``inference_timesteps``.
    """
    if control:
        text = f"({control}){text}"
    tts_model = _ensure_tts_model()
    audio = tts_model.generate(text=text, **kwargs)
    sf.write(OUTPUT_PATH, audio, tts_model.tts_model.sample_rate)
    sh.play(OUTPUT_PATH)
    return audio


def _ensure_asr_model():
    """Load and cache the VibeVoice ASR processor and model."""
    global _asr_model, _asr_processor

    if _asr_model is None:
        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        _asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
        _asr_model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            ASR_MODEL_ID,
            torch_dtype=torch_dtype,
        ).to(device)
    return _asr_processor, _asr_model


def asr(wav_path: str | Path) -> str:
    """Transcribe a WAV file with Microsoft's VibeVoice ASR model.

    The processor and model are loaded on the first call and reused by later
    calls. The model follows the same CPU/GPU and dtype selection used by the
    VibeVoice transcription example: CUDA uses bfloat16 and CPU uses float32.

    Args:
        wav_path: Path to the WAV file to transcribe.

    Returns:
        The decoded transcription text.
    """
    processor, model = _ensure_asr_model()
    inputs = processor.apply_transcription_request(audio=str(wav_path)).to(
        model.device, model.dtype
    )
    output_ids = model.generate(
        **inputs,
        max_new_tokens=ASR_MAX_NEW_TOKENS,
    )
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.decode(generated_ids, return_format="transcription_only")[0]
