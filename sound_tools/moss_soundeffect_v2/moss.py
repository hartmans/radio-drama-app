"""Interactive MOSS-SoundEffect v2 soundscape generator."""

from pathlib import Path

import sounddevice as sd
import soundfile as sf
import torch
from moss_soundeffect_v2 import MossSoundEffectPipeline


MODEL_ID = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
OUTPUT_DIR = Path("/audio")
_pipeline = None


def _ensure_pipeline():
    """Load the resident MOSS pipeline on its first generation request."""
    global _pipeline
    if _pipeline is None:
        _pipeline = MossSoundEffectPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device="cuda"
        )
    return _pipeline


def _play(audio, sample_rate):
    """Play generated audio when a host PulseAudio-compatible server is available."""
    try:
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as error:  # Host audio is optional for this authoring helper.
        print(f"Playback unavailable: {error}")


def generate(prompt, output="moss_soundeffect.wav", *, seconds=10, num_inference_steps=100,
             cfg_scale=4.0, seed=0, play=True):
    """Generate a sound effect, save it below ``/audio``, and optionally play it.

    ``seconds``, ``num_inference_steps``, and ``cfg_scale`` are passed to the
    upstream MOSS pipeline.  ``output`` is relative to `/audio` unless absolute.
    """
    pipeline = _ensure_pipeline()
    audio = pipeline(
        prompt=prompt,
        seconds=seconds,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = OUTPUT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Upstream ``save_audio`` delegates to TorchCodec through torchaudio.  Some
    # CUDA/FFmpeg combinations load the model but cannot load TorchCodec's
    # native encoder; soundfile writes this already-decoded WAV without that
    # optional encoder boundary.
    waveform = audio[0].transpose(0, 1).float().cpu().numpy()
    sf.write(output_path, waveform, pipeline.sample_rate)
    if play:
        _play(waveform, pipeline.sample_rate)
    return output_path
