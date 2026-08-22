"""Radio-drama proxy adapter for the official F5-TTS Python API."""

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


MODEL = "F5TTS_v1_Base"
REPOSITORY = "SWivid/F5-TTS"
CHECKPOINT_FILE = "F5TTS_v1_Base/model_1250000.safetensors"
VOCAB_FILE = "F5TTS_v1_Base/vocab.txt"
VOCODER_REPOSITORY = "charactr/vocos-mel-24khz"


def _trim_transcript(transcript: str, retained_fraction: float) -> str:
    """Keep the word prefix corresponding to F5's retained reference audio."""
    if retained_fraction >= 0.995:
        return transcript
    words = re.findall(r"\S+", transcript)
    retained_words = max(1, round(len(words) * retained_fraction))
    prefix = " ".join(words[:retained_words])
    sentence_end = max(prefix.rfind(mark) for mark in ".!?")
    return prefix[: sentence_end + 1] if sentence_end >= 0 else prefix


def _environment_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in {"1", "true", "yes", "on"}


class F5TtsEngine:
    """Keep the largest official F5-TTS model resident for voice cloning.

    F5-TTS's public high-level API generates one reference-conditioned item at
    a time. Pending scripts are therefore flattened through the shared line
    work contract and synthesized serially on one persistent model.
    """

    def __init__(self) -> None:
        self.model = None

    def load_model(self):
        if self.model is None:
            import soundfile
            import torch
            import torchaudio
            from f5_tts.api import F5TTS
            from huggingface_hub import hf_hub_download

            # Torchaudio 2.9 delegates file loading to optional TorchCodec,
            # whose CUDA wheel is not needed for the PCM WAV references used
            # by this container. Keep F5's expected channels-first contract
            # while using the already-required libsndfile backend directly.
            def load_reference(path: str):
                samples, sample_rate = soundfile.read(
                    path, dtype="float32", always_2d=True
                )
                return torch.from_numpy(samples.T.copy()), sample_rate

            torchaudio.load = load_reference

            local_files_only = _environment_flag("F5_TTS_LOCAL_FILES_ONLY", True)
            checkpoint = hf_hub_download(
                REPOSITORY,
                CHECKPOINT_FILE,
                local_files_only=local_files_only,
            )
            vocab = hf_hub_download(
                REPOSITORY,
                VOCAB_FILE,
                local_files_only=local_files_only,
            )
            self.model = F5TTS(
                model=MODEL,
                ckpt_file=checkpoint,
                vocab_file=vocab,
                device=os.environ.get("F5_TTS_DEVICE", "cuda"),
                hf_cache_dir=os.path.join(
                    os.environ.get("HF_HOME", "/models/huggingface"), "hub"
                ),
            )
            torch.set_grad_enabled(False)
        return self.model

    def synthesize_line(self, line: Mapping[str, Any]):
        speaker = line["speaker"]
        ref_file, ref_text = self.prepare_reference(
            str(speaker["voice_path"]), str(speaker["transcript"])
        )
        waveform, sample_rate, _spectrogram = self.load_model().infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=str(line["spoken_text"]),
            target_rms=float(os.environ.get("F5_TTS_TARGET_RMS", "0.1")),
            cross_fade_duration=float(
                os.environ.get("F5_TTS_CROSS_FADE_DURATION", "0.0")
            ),
            sway_sampling_coef=float(
                os.environ.get("F5_TTS_SWAY_SAMPLING_COEF", "-1.0")
            ),
            cfg_strength=float(os.environ.get("F5_TTS_CFG_STRENGTH", "2.0")),
            nfe_step=int(os.environ.get("F5_TTS_NFE_STEP", "32")),
            speed=float(os.environ.get("F5_TTS_SPEED", "1.0")),
        )
        return waveform, sample_rate

    @staticmethod
    def prepare_reference(ref_file: str, transcript: str) -> tuple[str, str]:
        """Apply F5's reference clipping and keep its transcript aligned.

        Upstream clips audio longer than twelve seconds but otherwise retains
        the complete caller-supplied transcript. That mismatched conditioning
        can collapse generated duration, so trim the transcript in proportion
        to the audio that upstream retained.
        """
        import soundfile
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        original_duration = soundfile.info(ref_file).duration
        processed_file, _ = preprocess_ref_audio_text(
            ref_file, transcript, show_info=lambda _message: None
        )
        processed_duration = soundfile.info(processed_file).duration
        retained_fraction = min(1.0, processed_duration / original_duration)
        return processed_file, _trim_transcript(transcript, retained_fraction)

    @staticmethod
    def write_audio(path, waveform, sample_rate: int) -> None:
        import soundfile

        soundfile.write(str(path), waveform, sample_rate, subtype="PCM_16")

    def render_batch(self, requests: Sequence[Mapping[str, Any]]):
        outputs, work = prepare_line_work(requests)
        try:
            model = self.load_model() if work else None
            sample_rate = int(model.target_sample_rate) if model is not None else 24_000
            for item in work:
                waveform, actual_rate = self.synthesize_line(item.line)
                if actual_rate != sample_rate:
                    raise RuntimeError(
                        f"F5-TTS returned {actual_rate} Hz audio; expected {sample_rate} Hz"
                    )
                self.write_audio(item.path, waveform, actual_rate)
            return finish_line_work(outputs, work, sample_rate=sample_rate)
        finally:
            if not _environment_flag("F5_TTS_KEEP_LINE_WAVS", False):
                remove_line_work(work)


def download_models() -> None:
    """Populate the persistent cache with the F5-TTS model and Vocos vocoder."""
    from huggingface_hub import hf_hub_download, snapshot_download

    hf_hub_download(REPOSITORY, CHECKPOINT_FILE)
    hf_hub_download(REPOSITORY, VOCAB_FILE)
    snapshot_download(VOCODER_REPOSITORY)


def main() -> None:
    run_server(F5TtsEngine().render_batch, capabilities={"needs_transcript"})


if __name__ == "__main__":
    main()
