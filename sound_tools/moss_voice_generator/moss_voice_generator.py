"""Interactive MOSS-VoiceGenerator voice-design helper."""

from pathlib import Path

import sounddevice as sd
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor


MODEL_ID = "OpenMOSS-Team/MOSS-VoiceGenerator"
OUTPUT_DIR = Path("/audio")
_backend = None


def _ensure_backend():
    """Load MOSS's voice-design processor and model once on the CUDA device."""
    global _backend
    if _backend is None:
        torch.backends.cuda.enable_cudnn_sdp(False)
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            normalize_inputs=True,
        )
        processor.audio_tokenizer = processor.audio_tokenizer.to("cuda")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        model.eval()
        _backend = model, processor
    return _backend


def _play(audio, sample_rate):
    """Play audio if the host's PulseAudio-compatible server is available."""
    try:
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as error:  # Saving a designed voice remains successful without DSP.
        print(f"Playback unavailable: {error}")


def generate(text, instruction, output="moss_voice.wav", *, audio_temperature=1.5,
             audio_top_p=0.6, audio_top_k=50, audio_repetition_penalty=1.1,
             max_new_tokens=4096, play=True):
    """Design a voice from ``instruction``, synthesize ``text``, save, and play it.

    The decoding defaults are the upstream MOSS-VoiceGenerator recommendations.
    ``output`` is relative to `/audio` unless an absolute path is supplied.
    """
    model, processor = _ensure_backend()
    conversation = [[processor.build_user_message(text=text, instruction=instruction)]]
    batch = processor(conversation, mode="generation")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch["input_ids"].to("cuda"),
            attention_mask=batch["attention_mask"].to("cuda"),
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
        )
    audio = processor.decode(outputs)[0].audio_codes_list[0].float().cpu().numpy()
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = OUTPUT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = processor.model_config.sampling_rate
    sf.write(output_path, audio, sample_rate)
    if play:
        _play(audio, sample_rate)
    return output_path
