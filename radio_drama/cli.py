from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from carthage.dependency_injection import Injector

from .config import ProductionConfig, SUPPORTED_DEBUG_CATEGORIES
from .init import radio_drama_injector
from .frontmatter import SUPPORTED_OUTPUT_TYPES


def default_output_path(production_path: str | Path, output_type: str = "wav") -> Path:
    return Path(production_path).with_suffix(f".{output_type}")


def recognized_output_path(value: str) -> Path:
    """Parse a CLI output path whose suffix selects a supported encoding."""

    path = Path(value)
    if path.suffix.lower().lstrip(".") not in SUPPORTED_OUTPUT_TYPES:
        choices = ", ".join(f".{name}" for name in SUPPORTED_OUTPUT_TYPES)
        raise argparse.ArgumentTypeError(f"output must end in one of: {choices}")
    return path


def initialize_arg_parser(
    description: str,
    *,
    production_argument: str = "production_xml",
    output_help: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(production_argument, help="Input production XML file.")
    parser.add_argument(
        "--voice-dir", default=None, help="Directory containing reference voice files."
    )
    parser.add_argument(
        "--sounds-dir",
        default=None,
        help="Directory containing sound files for relative <sound> references.",
    )
    parser.add_argument(
        "--model-file", default=None, help="Path to the VibeVoice model directory."
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output",
        default=None,
        type=recognized_output_path,
        help=output_help
        or "Output audio path; the .wav, .flac, .mp3, or .ogg suffix selects its format.",
    )
    output_group.add_argument(
        "--output-type",
        choices=SUPPORTED_OUTPUT_TYPES,
        default=None,
        help="Output format using the input filename stem; mutually exclusive with --output.",
    )
    parser.add_argument(
        "-r",
        "--output-sample-rate",
        type=int,
        default=None,
        help="Output WAV sample rate override.",
    )
    parser.add_argument(
        "--output-channels",
        type=int,
        default=None,
        help="Output WAV channel count override.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Maximum speech backend batch size override.",
    )
    parser.add_argument(
        "--device", default=None, help="Preferred torch device override."
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=None, help="VibeVoice cfg_scale override."
    )
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
    parser.add_argument(
        "--debug",
        action="append",
        choices=SUPPORTED_DEBUG_CATEGORIES,
        default=[],
        help="Enable one debug log category. May be supplied more than once.",
    )
    return parser


def resolved_output_path(
    args: argparse.Namespace,
    *,
    production_argument: str = "production_xml",
) -> Path:
    configured = getattr(args, "output", None)
    if configured is not None:
        return Path(configured)
    output_type = getattr(args, "output_type", None) or "wav"
    return default_output_path(getattr(args, production_argument), output_type)


def build_config_from_namespace(
    args: argparse.Namespace,
    *,
    production_argument: str = "production_xml",
) -> ProductionConfig:
    output_path = resolved_output_path(args, production_argument=production_argument)
    debug_categories = tuple(getattr(args, "debug", ()))
    debug_log_path = Path(f"{output_path}.log") if debug_categories else None
    return ProductionConfig(
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


def build_injector_from_namespace(
    args: argparse.Namespace,
    *,
    event_loop: asyncio.AbstractEventLoop,
    production_argument: str = "production_xml",
) -> tuple[Injector, ProductionConfig, Path, Path]:
    production_path = Path(getattr(args, production_argument))
    output_path = resolved_output_path(args, production_argument=production_argument)
    config = build_config_from_namespace(args, production_argument=production_argument)
    injector = radio_drama_injector(
        config=config,
        event_loop=event_loop,
        document_path=production_path,
        output_path=output_path,
    )
    return injector, config, production_path, output_path


__all__ = [
    "build_config_from_namespace",
    "build_injector_from_namespace",
    "default_output_path",
    "initialize_arg_parser",
    "recognized_output_path",
    "resolved_output_path",
]
