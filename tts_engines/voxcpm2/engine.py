"""Radio-drama proxy adapter for sequential VoxCPM2 voice cloning."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import soundfile as sf

from radio_drama_tts_container import (
    finish_line_work,
    prepare_line_work,
    remove_line_work,
    run_server,
)


MODEL = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")


def _environment_flag(name: str, default: bool) -> bool:
    """Read a conventional true/false environment setting."""
    return os.environ.get(name, str(default)).lower() in {"1", "true", "yes", "on"}


class VoxCPM2Engine:
    """Keep VoxCPM2 resident and synthesize cloned lines one at a time.

    VoxCPM2 does not currently have a suitable batched interface for the
    prompt-conditioned mode used here. Serializing generation also preserves a
    clean seam for future continuation-based speaker conditioning.
    """

    def __init__(self) -> None:
        self.model = None

    def load_model(self):
        if self.model is None:
            import torch
            from voxcpm import VoxCPM

            self.model = VoxCPM.from_pretrained(
                os.environ.get("VOXCPM_MODEL", MODEL),
                load_denoiser=False,
                optimize=_environment_flag("VOXCPM_OPTIMIZE", True),
                device=os.environ.get("VOXCPM_DEVICE", "cuda"),
            )
            torch.set_grad_enabled(False)
        return self.model

    def synthesize_line(self, line: Mapping[str, object]):
        """Generate one line with prompt cloning when a transcript is present."""
        speaker = line["speaker"]
        reference_wav_path = speaker["voice_path"]
        kwargs = {
            "text": line["spoken_text"],
            "reference_wav_path": reference_wav_path,
            "cfg_value": float(os.environ.get("VOXCPM_CFG_VALUE", "2.0")),
            "inference_timesteps": int(
                os.environ.get("VOXCPM_INFERENCE_TIMESTEPS", "10")
            ),
            "normalize": _environment_flag("VOXCPM_NORMALIZE", True),
        }
        transcript = speaker.get("transcript")
        if transcript:
            kwargs["prompt_wav_path"] = reference_wav_path
            kwargs["prompt_text"] = transcript
        return self.load_model().generate(**kwargs)

    def render_batch(self, requests: Sequence[Mapping[str, object]]):
        outputs, work = prepare_line_work(requests)
        try:
            model = self.load_model()
            for item in work:
                sf.write(
                    item.path,
                    self.synthesize_line(item.line),
                    model.tts_model.sample_rate,
                    subtype="PCM_16",
                )
            return finish_line_work(
                outputs, work, sample_rate=model.tts_model.sample_rate
            )
        finally:
            remove_line_work(work)


def main() -> None:
    run_server(VoxCPM2Engine().render_batch, capabilities={"needs_transcript"})


if __name__ == "__main__":
    main()
