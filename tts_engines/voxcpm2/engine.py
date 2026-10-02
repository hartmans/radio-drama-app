"""Concurrent VoxCPM2 voice cloning through vLLM-Omni."""

from __future__ import annotations

import asyncio
import copy
import os
import re
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import aclosing
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

MODEL = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")
_LEADING_INSTRUCTION_RE = re.compile(
    r"^\s*\((?P<instruction>[^()]*)\)\s*(?P<text>.*)$", re.DOTALL
)


class VoxCPM2Engine:
    """Share one async scheduler across background scripts and live streams."""

    def __init__(self) -> None:
        self.model = None
        self.load_lock = asyncio.Lock()
        self.batch_slots = asyncio.Semaphore(7)

    def load_model(self):
        from transformers import AutoTokenizer
        from vllm_omni.entrypoints.async_omni import AsyncOmni
        from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import build_cjk_split_map

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.environ.get("VOXCPM_MODEL", MODEL), trust_remote_code=True,
        )
        self.split_map = build_cjk_split_map(self.tokenizer)
        self.model = AsyncOmni(
            model=os.environ.get("VOXCPM_MODEL", MODEL),
            deploy_config=os.environ.get("VOXCPM_DEPLOY_CONFIG", "/opt/voxcpm2.yaml"),
        )
        return self.model

    async def ready(self):
        async with self.load_lock:
            if self.model is None:
                await run_in_thread(self.load_model)
        return self.model

    def build_prompt(self, line, *, prompt_wav_path=None, prompt_text=None):
        """Combine independent identity and continuation prefills.

        The upstream helper supports either reference or continuation audio.
        Build both through it, combining their metadata and prefill lengths
        while counting the target text/audio-start only once. This retains the
        original identity reference even after a controlled line sets a prompt.
        """
        from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import build_voxcpm2_prompt

        def build(path=None, transcript=None):
            audio, rate = (None, None)
            if path is not None:
                audio, rate = sf.read(path, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=-1)
                audio = audio.tolist()
            return build_voxcpm2_prompt(
                hf_config=self.model.engine.stage_vllm_configs[0].model_config.hf_config,
                tokenizer=self.tokenizer, split_map=self.split_map,
                text=str(line["spoken_text"]), ref_audio=audio,
                ref_sr=rate, ref_text=transcript,
            )

        reference = build(line["speaker"]["voice_path"])
        if prompt_wav_path is not None:
            continuation = build(prompt_wav_path, prompt_text)
            base = build()
            length = (len(reference["prompt_token_ids"])
                      + len(continuation["prompt_token_ids"])
                      - len(base["prompt_token_ids"]))
            reference["prompt_token_ids"] = [1] * length
            reference["additional_information"].update(continuation["additional_information"])
        return reference

    async def line_chunks(self, line, *, prompt_wav_path=None, prompt_text=None):
        """Consume DELTA audio and close the generator to abort on disconnect."""
        from vllm.sampling_params import RequestOutputKind
        import torch

        model = await self.ready()
        prompt = await run_in_thread(
            self.build_prompt, line,
            prompt_wav_path=prompt_wav_path, prompt_text=prompt_text,
        )
        params = copy.deepcopy(model.default_sampling_params_list)
        for param in params:
            param.output_kind = RequestOutputKind.DELTA
        async with aclosing(model.generate(
            prompt=prompt, request_id=uuid.uuid4().hex,
            sampling_params_list=params, output_modalities=["audio"],
        )) as outputs:
            async for output in outputs:
                mm = output.multimodal_output
                if not mm:
                    continue
                if "model_outputs" in mm:
                    values = mm["model_outputs"]
                elif "audio" in mm:
                    values = mm["audio"]
                else:
                    # Metadata-only events (for example sample rate) carry no PCM.
                    continue
                for value in values if isinstance(values, list) else [values]:
                    if value is not None:
                        audio = torch.as_tensor(value).detach().float().cpu().numpy()
                        yield np.asarray(audio, dtype="<f4").reshape(-1)

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
        by_request = [[] for _ in requests]
        for item in work:
            by_request[item.request_index].append(item)

        async def render_script(items):
            async with self.batch_slots:
                prompts = {}
                for item in items:
                    speaker_key, controlled, audible_text, prompt = self._line_prompt(item.line, prompts)
                    chunks = []
                    async with aclosing(self.line_chunks(
                        item.line, prompt_wav_path=prompt[0] if prompt else None,
                        prompt_text=prompt[1] if prompt else None,
                    )) as audio_chunks:
                        async for chunk in audio_chunks:
                            chunks.append(chunk)
                    audio = np.concatenate(chunks) if chunks else np.empty(0)
                    await run_in_thread(sf.write, item.path, audio, SAMPLE_RATE, subtype="PCM_16")
                    if controlled:
                        prompts[speaker_key] = (str(item.path), audible_text)

        tasks = [asyncio.create_task(render_script(items)) for items in by_request]
        try:
            await asyncio.gather(*tasks)
            return await run_in_thread(finish_line_work, outputs, work, sample_rate=SAMPLE_RATE)
        finally:
            # Stop siblings before deleting paths that may still be in use.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await run_in_thread(remove_line_work, work)

    async def stream_request(self, request) -> AsyncIterator[bytes]:
        """Submit live lines without acquiring background admission slots."""
        prompts = {}
        with tempfile.TemporaryDirectory(prefix="voxcpm-stream-", dir=".") as directory:
            for index, line in enumerate(request["dialogue_contents"]):
                if line["type"] != "line" or not str(line["spoken_text"]).strip():
                    continue
                speaker_key, controlled, audible_text, prompt = self._line_prompt(line, prompts)
                controlled_chunks = []
                async with aclosing(self.line_chunks(
                    line, prompt_wav_path=prompt[0] if prompt else None,
                    prompt_text=prompt[1] if prompt else None,
                )) as chunks:
                    async for audio in chunks:
                        if controlled:
                            controlled_chunks.append(audio.copy())
                        yield audio.tobytes()
                if controlled:
                    path = Path(directory) / f"{index}.wav"
                    audio = np.concatenate(controlled_chunks) if controlled_chunks else np.empty(0)
                    await run_in_thread(sf.write, path, audio, SAMPLE_RATE, subtype="PCM_16")
                    prompts[speaker_key] = (str(path), audible_text)


def main() -> None:
    """Reserve the original stdout pipe for protocol messages only.

    vLLM workers inherit file descriptors, so Python's redirect_stdout alone
    cannot keep their native/subprocess output out of the JSON-lines pipe.
    The duplicate is non-inheritable; workers inherit stderr as stdout instead.
    """
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as protocol_output:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        engine = VoxCPM2Engine()
        try:
            run_server(engine.render_batch, capabilities={"needs_transcript"},
                       stream_request=engine.stream_request, stream_sample_rate=SAMPLE_RATE,
                       output_stream=protocol_output)
        finally:
            if engine.model is not None:
                engine.model.shutdown()


if __name__ == "__main__":
    main()
