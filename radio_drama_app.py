from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import soundfile as sf
from carthage.dependency_injection import AsyncInjector

from radio_drama.config import ProductionConfig, SUPPORTED_DEBUG_CATEGORIES
from radio_drama.document import parse_production_file
from radio_drama.init import radio_drama_injector


def _default_output_path(input_path: str) -> str:
    return str(Path(input_path).with_suffix(".wav"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Phase 1 radio-drama XML document to WAV.")
    parser.add_argument("file", help="Input XML file.")
    parser.add_argument("--voice-dir", default=None, help="Directory containing reference voice files.")
    parser.add_argument("--sounds-dir", default=None, help="Directory containing sound files for relative <sound> references.")
    parser.add_argument("--model-file", default=None, help="Path to the VibeVoice model directory.")
    parser.add_argument("--output", default=None, help="Output WAV path. Defaults to the input path with a .wav extension.")
    parser.add_argument("--output-sample-rate", type=int, default=None, help="Output WAV sample rate override.")
    parser.add_argument("--output-channels", type=int, default=None, help="Output WAV channel count override.")
    parser.add_argument("--cut-before", default=None, help="Drop all production audio before the named <mark>.")
    parser.add_argument(
        "--debug",
        action="append",
        choices=SUPPORTED_DEBUG_CATEGORIES,
        default=[],
        help="Enable one debug log category. May be supplied more than once.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Maximum VibeVoice batch size override.")
    parser.add_argument("--device", default=None, help="Preferred torch device override.")
    parser.add_argument("--cfg-scale", type=float, default=None, help="VibeVoice cfg_scale override.")
    parser.add_argument(
        "--disable-prefill",
        action="store_const",
        const=True,
        default=None,
        help="Disable VibeVoice prefill.",
    )
    parser.add_argument(
        "--ddpm-inference-steps",
        type=int,
        default=None,
        help="VibeVoice DDPM inference steps override.",
    )
    args = parser.parse_args()

    output_path = args.output or _default_output_path(args.file)
    debug_categories = tuple(args.debug)
    debug_log_path = Path(f"{output_path}.log") if debug_categories else None
    config = ProductionConfig(
        voice_directory=Path(args.voice_dir) if args.voice_dir is not None else None,
        sounds_directory=Path(args.sounds_dir) if args.sounds_dir is not None else None,
        debug_log_path=debug_log_path,
        debug_categories=debug_categories,
        model_name=args.model_file,
        output_sample_rate=args.output_sample_rate,
        output_channels=args.output_channels,
        batch_size=args.batch_size,
        device=args.device,
        cfg_scale=args.cfg_scale,
        disable_prefill=args.disable_prefill,
        ddpm_inference_steps=args.ddpm_inference_steps,
    )

    async def runner() -> None:
        production_node = parse_production_file(args.file)
        if config.debug_log_path is not None:
            config.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            config.debug_log_path.write_text("", encoding="utf-8")
        injector = radio_drama_injector(
            config=config,
            event_loop=asyncio.get_running_loop(),
            document_path=Path(args.file),
        )
        try:
            ainjector = injector(AsyncInjector)
            production_plan = await production_node.plan(ainjector)
            if args.cut_before is not None:
                production_plan.cut_before_mark(args.cut_before)
            production_result = await production_plan.render()
        finally:
            injector.close()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, production_result.audio, config.resolved_output_sample_rate)

    asyncio.run(runner())


if __name__ == "__main__":
    main()
