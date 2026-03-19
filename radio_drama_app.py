from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from radio_drama.phase1 import ProductionConfig, render_production_file


def _default_output_path(input_path: str) -> str:
    return str(Path(input_path).with_suffix(".wav"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Phase 1 radio-drama XML document to WAV.")
    parser.add_argument("file", help="Input XML file.")
    parser.add_argument("--voice-dir", default="./voices", help="Directory containing reference voice files.")
    parser.add_argument("--model-file", default="/srv/ai/models/vibevoice/vibevoice-large", help="Path to the VibeVoice model directory.")
    parser.add_argument("--output", default=None, help="Output WAV path. Defaults to the input path with a .wav extension.")
    parser.add_argument("--output-sample-rate", type=int, default=48000, help="Output WAV sample rate.")
    parser.add_argument("--output-channels", type=int, default=2, help="Output WAV channel count.")
    parser.add_argument("--batch-size", type=int, default=10, help="Maximum VibeVoice batch size.")
    parser.add_argument("--device", default="cuda", help="Preferred torch device.")
    parser.add_argument("--cfg-scale", type=float, default=1.3, help="VibeVoice cfg_scale.")
    parser.add_argument("--disable-prefill", action="store_true", help="Disable VibeVoice prefill.")
    parser.add_argument("--ddpm-inference-steps", type=int, default=5, help="VibeVoice DDPM inference steps.")
    args = parser.parse_args()

    output_path = args.output or _default_output_path(args.file)
    config = ProductionConfig(
        voice_directory=Path(args.voice_dir),
        model_name=args.model_file,
        output_sample_rate=args.output_sample_rate,
        output_channels=args.output_channels,
        batch_size=args.batch_size,
        device=args.device,
        cfg_scale=args.cfg_scale,
        disable_prefill=args.disable_prefill,
        ddpm_inference_steps=args.ddpm_inference_steps,
    )
    asyncio.run(render_production_file(args.file, output_path, config))


if __name__ == "__main__":
    main()
