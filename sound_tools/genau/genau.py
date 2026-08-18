"""Interactive GenAU ambient-sound generator."""

from pathlib import Path
import shutil

import sounddevice as sd
import torch
from pytorch_lightning import seed_everything
from src.tools.configuration import Configuration
from src.tools.download_manager import get_checkpoint_path
from src.utilities.model.model_util import instantiate_from_config


OUTPUT_DIR = Path("/audio")
_models = {}


def _play(path):
    """Play a saved WAV if a PulseAudio-compatible host sound server is present."""
    try:
        import soundfile as sf
        audio, sample_rate = sf.read(path, always_2d=True)
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as error:  # Playback must not make authored output fail.
        print(f"Playback unavailable: {error}")


def _ensure_model(name):
    """Download, construct, and retain one upstream GenAU model."""
    if name not in _models:
        config_path = get_checkpoint_path(f"{name}_config")
        checkpoint_path = get_checkpoint_path(name)
        config = Configuration(config_path).get_config()
        config["reload_from_ckpt"] = checkpoint_path
        config["model"]["params"]["ckpt_path"] = checkpoint_path
        model = instantiate_from_config(config["model"])
        model.eval().cuda()
        _models[name] = (model, config)
    return _models[name]


def generate(prompt, output="genau.wav", *, model="genau-l-full-hq-data", seed=0,
             cfg_weight=4.0, ddim_steps=100, n_candidates=1, play=True):
    """Generate an ambient sound, write it below `/audio`, and optionally play it.

    GenAU is trained for ambient sounds; it is not intended for speech or music.
    The upstream model's sampling controls are exposed as ``cfg_weight``,
    ``ddim_steps``, and ``n_candidates``.
    """
    seed_everything(seed)
    generator, config = _ensure_model(model)
    source = generator.text_to_audio(
        prompt=prompt,
        ddim_steps=ddim_steps,
        unconditional_guidance_scale=cfg_weight,
        n_gen=n_candidates,
        use_ema=config.get("force_use_ema", False),
    )
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = OUTPUT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)
    if play:
        _play(output_path)
    return output_path
