#!/usr/bin/env python3
"""Run Seed-VC inference in its isolated Podman container."""

import argparse
import os
from pathlib import Path
import subprocess


DEFAULT_IMAGE = "localhost/seed-vc:latest"
DEFAULT_CACHE = Path("/srv/ai/models/seed-vc")


def _existing_file(value):
    """Return an absolute input path or report a useful argparse error."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"audio file does not exist: {value}")
    return path


def parser():
    """Describe the stable host wrapper and the upstream inference controls."""
    result = argparse.ArgumentParser(
        description="Convert a source recording to a reference voice with Seed-VC."
    )
    result.add_argument("--source", required=True, type=_existing_file)
    result.add_argument("--target", required=True, type=_existing_file,
                        help="reference recording whose voice should be used")
    result.add_argument("--output", required=True, type=Path,
                        help="directory in which Seed-VC writes the converted WAV")
    result.add_argument("--diffusion-steps", type=int, default=30)
    result.add_argument("--length-adjust", type=float, default=1.0)
    result.add_argument("--inference-cfg-rate", type=float, default=0.7)
    result.add_argument("--f0-condition", choices=("True", "False"), default="False")
    result.add_argument("--auto-f0-adjust", choices=("True", "False"), default="False")
    result.add_argument("--semi-tone-shift", type=int, default=0)
    result.add_argument("--fp16", choices=("True", "False"), default="True")
    result.add_argument("--checkpoint", type=_existing_file)
    result.add_argument("--config", type=_existing_file)
    result.add_argument("--image", default=os.environ.get("SEED_VC_IMAGE", DEFAULT_IMAGE))
    result.add_argument("--cache", type=Path,
                        default=Path(os.environ.get("SEED_VC_CACHE", DEFAULT_CACHE)))
    return result


def podman_command(args):
    """Translate host paths and inference options into the container contract."""
    output = args.output.expanduser().resolve()
    cache = args.cache.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    source_container = f"/input/source{args.source.suffix}"
    target_container = f"/input/target{args.target.suffix}"
    command = [
        "podman", "run", "--rm", "--device", "nvidia.com/gpu=all",
        "--network", "none",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "-v", f"{args.source}:{source_container}:ro,Z",
        "-v", f"{args.target}:{target_container}:ro,Z",
        "-v", f"{output}:/output:Z",
        "-v", f"{cache}:/models/seed-vc:Z",
    ]
    inference_args = [
        "--source", source_container,
        "--target", target_container,
        "--output", "/output",
        "--diffusion-steps", str(args.diffusion_steps),
        "--length-adjust", str(args.length_adjust),
        "--inference-cfg-rate", str(args.inference_cfg_rate),
        "--f0-condition", args.f0_condition,
        "--auto-f0-adjust", args.auto_f0_adjust,
        "--semi-tone-shift", str(args.semi_tone_shift),
        "--fp16", args.fp16,
    ]
    if args.checkpoint is not None:
        command.extend(("-v", f"{args.checkpoint}:/input/checkpoint.pth:ro,Z"))
        inference_args.extend(("--checkpoint", "/input/checkpoint.pth"))
    if args.config is not None:
        command.extend(("-v", f"{args.config}:/input/config.yml:ro,Z"))
        inference_args.extend(("--config", "/input/config.yml"))
    return command + [args.image] + inference_args


def main():
    """Parse the host CLI and return Seed-VC's process status."""
    args = parser().parse_args()
    if (args.checkpoint is None) != (args.config is None):
        parser().error("--checkpoint and --config must be supplied together")
    return subprocess.run(podman_command(args), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
