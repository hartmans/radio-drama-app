#!/usr/bin/env python3
"""Run Seed-VC v2 inference in its isolated Podman container."""

import argparse
import os
from pathlib import Path
import subprocess


DEFAULT_IMAGE = "localhost/seed-vc:latest"


def _existing_file(value):
    """Return an absolute input path or report a useful argparse error."""
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {value}")
    return path


def parser():
    """Describe the host wrapper and upstream v2 inference controls."""
    result = argparse.ArgumentParser(
        description="Convert a source recording with the Seed-VC v2 model."
    )
    result.add_argument("--source", required=True, type=_existing_file)
    result.add_argument("--target", required=True, type=_existing_file,
                        help="reference recording whose voice should be used")
    result.add_argument("--output", required=True, type=Path,
                        help="directory in which Seed-VC writes the converted WAV")
    result.add_argument("--diffusion-steps", type=int, default=30)
    result.add_argument("--length-adjust", type=float, default=1.0)
    result.add_argument("--compile", action="store_true")
    result.add_argument("--intelligibility-cfg-rate", type=float, default=0.7)
    result.add_argument("--similarity-cfg-rate", type=float, default=0.7)
    result.add_argument("--top-p", type=float, default=0.9)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--repetition-penalty", type=float, default=1.0)
    result.add_argument("--convert-style", choices=("True", "False"), default="False")
    result.add_argument("--anonymization-only", choices=("True", "False"),
                        default="False")
    result.add_argument("--ar-checkpoint-path", type=_existing_file)
    result.add_argument("--cfm-checkpoint-path", type=_existing_file)
    result.add_argument("--image", default=os.environ.get("SEED_VC_IMAGE", DEFAULT_IMAGE))
    return result


def podman_command(args):
    """Translate host paths and v2 options into the container contract."""
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_container = f"/input/source{args.source.suffix}"
    target_container = f"/input/target{args.target.suffix}"
    command = [
        "podman", "run", "--rm", "--device", "nvidia.com/gpu=all",
        "--network", "none",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        "-v", f"{args.source}:{source_container}:ro,Z",
        "-v", f"{args.target}:{target_container}:ro,Z",
        "-v", f"{output}:/output:Z",
    ]
    inference_args = [
        "/opt/seed-vc/inference_v2.py",
        "--source", source_container,
        "--target", target_container,
        "--output", "/output",
        "--diffusion-steps", str(args.diffusion_steps),
        "--length-adjust", str(args.length_adjust),
        "--intelligibility-cfg-rate", str(args.intelligibility_cfg_rate),
        "--similarity-cfg-rate", str(args.similarity_cfg_rate),
        "--top-p", str(args.top_p),
        "--temperature", str(args.temperature),
        "--repetition-penalty", str(args.repetition_penalty),
        "--convert-style", args.convert_style,
        "--anonymization-only", args.anonymization_only,
    ]
    if args.compile:
        inference_args.extend(("--compile", "True"))
    if args.ar_checkpoint_path is not None:
        command.extend(("-v", f"{args.ar_checkpoint_path}:/input/ar.pth:ro,Z"))
        inference_args.extend(("--ar-checkpoint-path", "/input/ar.pth"))
    if args.cfm_checkpoint_path is not None:
        command.extend(("-v", f"{args.cfm_checkpoint_path}:/input/cfm.pth:ro,Z"))
        inference_args.extend(("--cfm-checkpoint-path", "/input/cfm.pth"))
    return command + ["--entrypoint", "python3", args.image] + inference_args


def main():
    """Parse the host CLI and return Seed-VC's process status."""
    args = parser().parse_args()
    return subprocess.run(podman_command(args), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
