"""Radio-drama proxy adapter for Higgs TTS 3 through Transformers."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from radio_drama_tts_container import (
    finish_line_work,
    prepare_line_work,
    remove_line_work,
    run_server,
)


MODEL = os.environ.get(
    "HIGGS_MODEL", "multimodalart/higgs-audio-v3-tts-4b-transformers"
)
AUDIO_TOKENIZER = os.environ.get(
    "HIGGS_AUDIO_TOKENIZER", "bosonai/higgs-audio-v2-tokenizer"
)
SAMPLE_RATE = 24_000

CONTROL_TAGS = {
    "emotion": frozenset(
        {
            "affection", "amusement", "anger", "arousal", "awe",
            "bitterness", "confusion", "contemplation", "contentment",
            "determination", "disgust", "elation", "enthusiasm", "fear",
            "helplessness", "longing", "pride", "relief", "sadness",
            "shame", "surprise",
        }
    ),
    "prosody": frozenset(
        {
            "speed_very_slow", "speed_slow", "speed_fast", "speed_very_fast",
            "pitch_low", "pitch_high", "expressive_high", "expressive_low",
            "pause", "long_pause",
        }
    ),
    "style": frozenset({"singing", "shouting", "whispering"}),
    "sfx": frozenset(
        {
            "cough", "laughter", "crying", "screaming", "burping",
            "humming", "sigh", "sniff", "sneeze",
        }
    ),
}
CONTROL_EXPRESSION = re.compile(r"\[([a-z]+):([a-z_]+)\]")


def _environment_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in {"1", "true", "yes", "on"}


def expand_control_expressions(text: str) -> str:
    """Translate recognized radio-drama brackets to Higgs control tokens."""

    def replace(match: re.Match[str]) -> str:
        category, tag = match.groups()
        if tag not in CONTROL_TAGS.get(category, ()):
            return match.group(0)
        return f"<|{category}:{tag}|>"

    return CONTROL_EXPRESSION.sub(replace, text)


class HiggsTtsEngine:
    """Keep one Transformers Higgs model resident for line-oriented cloning.

    The upstream Transformers port exposes single-item autoregressive
    generation. ``render_batch`` still drains all pending scripts in bounded
    groups, but generation within a group is serial so one model and its KV
    cache do not contend for GPU memory.
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self._reference_audio: dict[str, tuple[Any, int]] = {}

    def load_model(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = os.environ.get("HIGGS_DEVICE", "cuda")
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            local_files_only = _environment_flag("HIGGS_LOCAL_FILES_ONLY", True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL, trust_remote_code=True, local_files_only=local_files_only
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                trust_remote_code=True,
                dtype=dtype,
                local_files_only=local_files_only,
            ).to(device).eval()
            self.model.config.audio_tokenizer_id = AUDIO_TOKENIZER
            torch.set_grad_enabled(False)

            # The Higgs waveform encoder/decoder is numerically unstable in
            # bf16. Load it eagerly and enforce fp32 even if remote-code
            # defaults or the surrounding model dtype change later.
            codec = self.model.get_audio_codec().to(device=device, dtype=torch.float32)
            codec.eval()
            if any(parameter.dtype != torch.float32 for parameter in codec.parameters()):
                raise RuntimeError("Higgs audio tokenizer did not remain in float32")
        return self.model, self.tokenizer

    def reference_audio(self, path: str):
        """Load and cache a mounted speaker reference at its native rate."""
        if path not in self._reference_audio:
            import soundfile
            import torch

            samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
            self._reference_audio[path] = (waveform, sample_rate)
        return self._reference_audio[path]

    def synthesize_line(self, line: Mapping[str, Any]):
        model, tokenizer = self.load_model()
        speaker = line["speaker"]
        reference, reference_rate = self.reference_audio(str(speaker["voice_path"]))
        return model.generate_speech(
            expand_control_expressions(str(line["spoken_text"])),
            tokenizer,
            reference_audio=reference,
            reference_sample_rate=reference_rate,
            reference_text=str(speaker["transcript"]),
            max_new_tokens=int(os.environ.get("HIGGS_MAX_NEW_TOKENS", "2048")),
            temperature=float(os.environ.get("HIGGS_TEMPERATURE", "0.8")),
            top_p=float(os.environ.get("HIGGS_TOP_P", "1.0")),
            top_k=int(os.environ.get("HIGGS_TOP_K", "50")),
        )

    def synthesize_batch(self, lines: Sequence[Mapping[str, Any]]) -> list[Any]:
        """Synthesize a protocol batch using the port's single-item decoder."""
        return [self.synthesize_line(line) for line in lines]

    @staticmethod
    def write_audio(path, audio) -> None:
        import soundfile

        soundfile.write(str(path), audio.numpy(), SAMPLE_RATE, subtype="PCM_16")

    def render_batch(self, requests: Sequence[Mapping[str, Any]]):
        outputs, work = prepare_line_work(requests)
        try:
            if work:
                self.load_model()
            batch_size = int(os.environ.get("HIGGS_BATCH_SIZE", "1"))
            if batch_size < 1:
                raise ValueError("HIGGS_BATCH_SIZE must be at least 1")
            for start in range(0, len(work), batch_size):
                chunk = work[start : start + batch_size]
                audio = self.synthesize_batch([item.line for item in chunk])
                for item, waveform in zip(chunk, audio, strict=True):
                    self.write_audio(item.path, waveform)
            return finish_line_work(outputs, work, sample_rate=SAMPLE_RATE)
        finally:
            if not _environment_flag("HIGGS_KEEP_LINE_WAVS", False):
                remove_line_work(work)


def download_models() -> None:
    """Populate the persistent Hugging Face cache without loading the model."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    snapshot_download(AUDIO_TOKENIZER)


def main() -> None:
    run_server(HiggsTtsEngine().render_batch, capabilities={"needs_transcript"})


if __name__ == "__main__":
    main()
