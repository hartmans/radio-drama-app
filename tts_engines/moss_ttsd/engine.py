"""Radio-drama proxy adapter for batched MOSS-TTSD dialogue synthesis."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radio_drama_tts_container import artifact_name, run_server, write_pcm16_wav


MODEL = os.environ.get("MOSS_TTSD_MODEL", "OpenMOSS-Team/MOSS-TTSD-v1.0")


def _environment_flag(name: str, default: bool) -> bool:
    """Read a conventional true/false environment setting."""
    return os.environ.get(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """One script-level MOSS conversation and its cache-relative WAV path."""

    output_path: Path
    conversation: Any | None


class MossTtsdEngine:
    """Keep MOSS-TTSD resident and batch complete dialogue scripts.

    MOSS-TTSD conditions on every speaker reference and its transcript across
    a complete tagged dialogue. A batch item is consequently one complete
    script rather than one dialogue line.
    """

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self._reference_wavs: dict[str, Any] = {}
        self._reference_codes: dict[str, Any] = {}

    def load_model(self):
        if self.model is None:
            import torch
            from transformers import AutoModel, AutoProcessor

            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            device = os.environ.get("MOSS_TTSD_DEVICE", "cuda")
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            attention = os.environ.get(
                "MOSS_TTSD_ATTN_IMPLEMENTATION",
                "sdpa" if device.startswith("cuda") else "eager",
            )
            local_files_only = _environment_flag("MOSS_TTSD_LOCAL_FILES_ONLY", True)
            if local_files_only:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            self.processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
            self.processor.audio_tokenizer = self.processor.audio_tokenizer.to(device)
            self.model = AutoModel.from_pretrained(
                MODEL,
                trust_remote_code=True,
                attn_implementation=attention,
                dtype=dtype,
            ).to(device)
            self.model.eval()
        return self.model, self.processor

    def _reference_audio(self, path: str):
        """Return cached normalized waveform and MOSS audio codes for one voice."""
        if path not in self._reference_wavs:
            import torchaudio

            _model, processor = self.load_model()
            wav, sample_rate = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            target_rate = int(processor.model_config.sampling_rate)
            if sample_rate != target_rate:
                wav = torchaudio.functional.resample(wav, sample_rate, target_rate)
            self._reference_wavs[path] = wav
            self._reference_codes[path] = processor.encode_audios_from_wav(
                [wav], sampling_rate=target_rate
            )[0]
        return self._reference_wavs[path], self._reference_codes[path]

    def prepare_request(self, request: Mapping[str, Any]) -> PreparedRequest:
        """Translate one proxy script into a MOSS continuation conversation."""
        speakers: dict[str, tuple[str, Mapping[str, Any]]] = {}
        generated_lines: list[str] = []
        for content in request["dialogue_contents"]:
            if content.get("type") != "line":
                continue
            speaker = content["speaker"]
            speaker_key = str(speaker["authored_name"])
            if speaker_key not in speakers:
                speakers[speaker_key] = (f"S{len(speakers) + 1}", speaker)
            speaker_tag, _speaker = speakers[speaker_key]
            generated_lines.append(f"[{speaker_tag}] {content['spoken_text']}")

        output_path = Path(artifact_name(request))
        if not speakers:
            return PreparedRequest(output_path, None)

        import torch

        _model, processor = self.load_model()

        prompt_texts: list[str] = []
        prompt_wavs: list[Any] = []
        reference_codes: list[Any] = []
        for speaker_tag, speaker in speakers.values():
            wav, codes = self._reference_audio(str(speaker["voice_path"]))
            prompt_wavs.append(wav)
            reference_codes.append(codes)
            prompt_texts.append(f"[{speaker_tag}] {speaker['transcript']}")

        prompt_audio = processor.encode_audios_from_wav(
            [torch.cat(prompt_wavs, dim=-1)],
            sampling_rate=int(processor.model_config.sampling_rate),
        )[0]
        message = processor.build_user_message(
            text=" ".join((*prompt_texts, *generated_lines)),
            reference=reference_codes,
        )
        conversation = [
            message,
            processor.build_assistant_message(audio_codes_list=[prompt_audio]),
        ]
        return PreparedRequest(output_path, conversation)

    def synthesize_batch(self, prepared: Sequence[PreparedRequest]) -> Sequence[Any]:
        """Generate and decode a batch of complete MOSS continuation scripts."""
        import torch

        model, processor = self.load_model()
        batch = processor([item.conversation for item in prepared], mode="continuation")
        device = os.environ.get("MOSS_TTSD_DEVICE", "cuda")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=int(os.environ.get("MOSS_TTSD_MAX_NEW_TOKENS", "4096")),
                audio_temperature=float(
                    os.environ.get("MOSS_TTSD_TEMPERATURE", "1.1")
                ),
                audio_top_p=float(os.environ.get("MOSS_TTSD_TOP_P", "0.9")),
                audio_top_k=int(os.environ.get("MOSS_TTSD_TOP_K", "50")),
                audio_repetition_penalty=float(
                    os.environ.get("MOSS_TTSD_REPETITION_PENALTY", "1.1")
                ),
            )
        return [message.audio_codes_list[0] for message in processor.decode(output_ids)]

    def write_audio(self, path: Path, audio: Any) -> None:
        """Write one decoded, model-native 24 kHz mono dialogue WAV."""
        import torchaudio

        _model, processor = self.load_model()
        torchaudio.save(
            str(path),
            audio.detach().cpu().unsqueeze(0),
            processor.model_config.sampling_rate,
        )

    def render_batch(self, requests: Sequence[Mapping[str, Any]]):
        prepared = [self.prepare_request(request) for request in requests]
        batch_size = int(os.environ.get("MOSS_TTSD_BATCH_SIZE", "10"))
        if batch_size < 1:
            raise ValueError("MOSS_TTSD_BATCH_SIZE must be at least 1")
        active = [item for item in prepared if item.conversation is not None]
        for start in range(0, len(active), batch_size):
            chunk = active[start : start + batch_size]
            for item, audio in zip(chunk, self.synthesize_batch(chunk), strict=True):
                self.write_audio(item.output_path, audio)
        for item in prepared:
            if item.conversation is None:
                write_pcm16_wav(item.output_path, (), sample_rate=24_000)
        return [{"wav": item.output_path.name} for item in prepared]


def main() -> None:
    run_server(MossTtsdEngine().render_batch, capabilities={"needs_transcript"})


if __name__ == "__main__":
    main()
