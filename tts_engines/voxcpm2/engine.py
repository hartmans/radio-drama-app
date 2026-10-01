"""Radio-drama proxy adapter for sequential VoxCPM2 voice cloning."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from radio_drama_tts_container import (
    finish_line_work,
    prepare_line_work,
    remove_line_work,
    run_server,
    run_in_thread,
)


SAMPLE_RATE = 48_000
_END = object()

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
        self.model_lock = asyncio.Lock()

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
        if self.model.tts_model.sample_rate != SAMPLE_RATE:
            raise RuntimeError("VoxCPM2 model sample rate differs from the streaming format")
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

    def generation_kwargs(
        self,
        line: Mapping[str, object],
        *,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
    ):
        """Build matching conditioning for normal and streaming generation."""
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
        return kwargs

    def synthesize_line(self, line, *, prompt_wav_path=None, prompt_text=None):
        """Generate one line with inference mode set in the calling thread."""
        import torch

        with torch.no_grad():
            return self.load_model().generate(**self.generation_kwargs(
                line, prompt_wav_path=prompt_wav_path, prompt_text=prompt_text,
            ))

    @staticmethod
    def _next_chunk(generator):
        import torch

        with torch.no_grad():
            return next(generator, _END)

    @classmethod
    def _line_prompt(cls, line, prompts):
        speaker = line["speaker"]
        speaker_key = str(speaker.get("authored_name", speaker["voice_path"]))
        has_instruction, audible_text = cls.split_leading_instruction(str(line["spoken_text"]))
        prompt = None if has_instruction else prompts.get(speaker_key)
        if prompt is None and not has_instruction and speaker.get("transcript"):
            prompt = (str(speaker["voice_path"]), str(speaker["transcript"]))
        return speaker_key, has_instruction, audible_text, prompt

    async def render_batch(self, requests: Sequence[Mapping[str, object]]):
        outputs, work = prepare_line_work(requests)
        prompts_by_request: dict[int, dict[str, tuple[str, str]]] = {}
        try:
            async with self.model_lock:
                model = await run_in_thread(self.load_model)
            for item in work:
                prompts = prompts_by_request.setdefault(item.request_index, {})
                speaker_key, has_instruction, audible_text, prompt = self._line_prompt(item.line, prompts)
                async with self.model_lock:
                    audio = await run_in_thread(
                        self.synthesize_line, item.line,
                        prompt_wav_path=prompt[0] if prompt else None,
                        prompt_text=prompt[1] if prompt else None,
                    )
                await run_in_thread(sf.write,
                    item.path,
                    audio,
                    model.tts_model.sample_rate,
                    subtype="PCM_16",
                )
                if has_instruction:
                    prompts[speaker_key] = (str(item.path), audible_text)
            return await run_in_thread(finish_line_work,
                outputs, work, sample_rate=model.tts_model.sample_rate
            )
        finally:
            await run_in_thread(remove_line_work, work)

    async def stream_request(self, request) -> AsyncIterator[bytes]:
        """Stream one request while retaining exclusive access to model caches.

        Holding the lock across yields prevents another generation from changing
        VoxCPM2's model-owned state. Worker calls complete before cancellation
        releases the lock. Continuation prompts remain local to this request.
        """
        async with self.model_lock:
            model = await run_in_thread(self.load_model)
            prompts = {}
            with tempfile.TemporaryDirectory(prefix="voxcpm-stream-", dir=".") as directory:
                for index, line in enumerate(request["dialogue_contents"]):
                    if line["type"] != "line" or not str(line["spoken_text"]).strip():
                        continue
                    speaker_key, has_instruction, audible_text, prompt = self._line_prompt(line, prompts)
                    generator = model.generate_streaming(**self.generation_kwargs(
                        line, prompt_wav_path=prompt[0] if prompt else None,
                        prompt_text=prompt[1] if prompt else None,
                    ))
                    controlled_chunks = []
                    try:
                        while True:
                            chunk = await run_in_thread(self._next_chunk, generator)
                            if chunk is _END:
                                break
                            audio = np.asarray(chunk, dtype="<f4").reshape(-1)
                            if has_instruction:
                                controlled_chunks.append(audio.copy())
                            yield audio.tobytes()
                    finally:
                        await run_in_thread(generator.close)
                    if has_instruction:
                        path = Path(directory) / f"{index}.wav"
                        audio = np.concatenate(controlled_chunks) if controlled_chunks else np.empty(0)
                        await run_in_thread(sf.write, path, audio, SAMPLE_RATE, subtype="PCM_16")
                        prompts[speaker_key] = (str(path), audible_text)


def main() -> None:
    engine = VoxCPM2Engine()
    run_server(engine.render_batch, capabilities={"needs_transcript"},
               stream_request=engine.stream_request, stream_sample_rate=SAMPLE_RATE)


if __name__ == "__main__":
    main()
