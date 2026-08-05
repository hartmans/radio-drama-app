"""Radio-drama proxy adapter for sequential VoxCPM2 voice cloning."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence

import soundfile as sf

from radio_drama_tts_container import (
    finish_line_work,
    prepare_line_work,
    remove_line_work,
    run_server,
)


MODEL = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")
_LEADING_INSTRUCTION_RE = re.compile(
    r"^\s*\((?P<instruction>[^()]*)\)\s*(?P<text>.*)$", re.DOTALL
)


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

    @staticmethod
    def split_leading_instruction(text: str) -> tuple[bool, str]:
        """Identify a VoxCPM2 control prefix and return its audible text.

        VoxCPM2 supports leading parentheticals as controls only for
        reference-only cloning.  The returned text is the transcript for a
        generated line when it later becomes a continuation prompt.
        """
        match = _LEADING_INSTRUCTION_RE.match(text)
        if match is None or not match.group("instruction").strip():
            return False, text
        return True, match.group("text").strip()

    def synthesize_line(
        self,
        line: Mapping[str, object],
        *,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
    ):
        """Generate one line, optionally continuing from an aligned prompt."""
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
        if prompt_wav_path is not None and prompt_text is not None:
            kwargs["prompt_wav_path"] = prompt_wav_path
            kwargs["prompt_text"] = prompt_text
        return self.load_model().generate(**kwargs)

    def render_batch(self, requests: Sequence[Mapping[str, object]]):
        outputs, work = prepare_line_work(requests)
        prompts_by_request: dict[int, dict[str, tuple[str, str]]] = {}
        try:
            model = self.load_model()
            for item in work:
                speaker = item.line["speaker"]
                speaker_key = str(speaker.get("authored_name", speaker["voice_path"]))
                has_instruction, audible_text = self.split_leading_instruction(
                    str(item.line["spoken_text"])
                )
                prompts = prompts_by_request.setdefault(item.request_index, {})
                prompt = None if has_instruction else prompts.get(speaker_key)
                if prompt is None and not has_instruction:
                    transcript = speaker.get("transcript")
                    if transcript:
                        prompt = (str(speaker["voice_path"]), str(transcript))
                sf.write(
                    item.path,
                    self.synthesize_line(
                        item.line,
                        prompt_wav_path=prompt[0] if prompt else None,
                        prompt_text=prompt[1] if prompt else None,
                    ),
                    model.tts_model.sample_rate,
                    subtype="PCM_16",
                )
                prompts[speaker_key] = (str(item.path), audible_text)
            return finish_line_work(
                outputs, work, sample_rate=model.tts_model.sample_rate
            )
        finally:
            remove_line_work(work)


def main() -> None:
    run_server(VoxCPM2Engine().render_batch, capabilities={"needs_transcript"})


if __name__ == "__main__":
    main()
