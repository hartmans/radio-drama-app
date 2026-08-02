"""Interactive VoxCPM2 voice design and cloning helper."""

# VoxCPM2 needs a much newer transformers release than the rest of this project.
import sh
import soundfile as sf
from voxcpm import VoxCPM


OUTPUT_PATH = "audio.wav"

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    load_denoiser=False,
    device="cuda",
)


def play(text, control=None, **kwargs):
    """Generate speech, write it to ``audio.wav``, and play it.

    ``control`` is an optional VoxCPM2 voice-design instruction. Keyword
    arguments are passed through to :meth:`VoxCPM.generate`, allowing cloning
    calls such as ``play(text, reference_wav_path="voice.wav")`` as well as
    tuning parameters such as ``cfg_value`` and ``inference_timesteps``.
    """
    if control:
        text = f"({control}){text}"
    audio = model.generate(text=text, **kwargs)
    sf.write(OUTPUT_PATH, audio, model.tts_model.sample_rate)
    sh.play(OUTPUT_PATH)
    return audio
