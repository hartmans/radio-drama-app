"""Patch generation settings before Omni workers import or capture graphs."""

from importlib.util import find_spec
from pathlib import Path


def patch_source(source: str) -> str:
    """Fail the image build if the pinned upstream initialization changes."""
    for attribute, old, converter, variable, default in (
        ("_inference_timesteps", "10", "int", "VOXCPM_INFERENCE_TIMESTEPS", "20"),
        ("_cfg_value", "2.0", "float", "VOXCPM_CFG_VALUE", "2.0"),
    ):
        original = f"        self.{attribute} = {old}\n"
        if source.count(original) != 1:
            raise ValueError(f"Expected exactly one VoxCPM2 {attribute} initialization")
        source = source.replace(
            original,
            f'        self.{attribute} = {converter}(__import__("os").environ.get('
            f'"{variable}", "{default}"))\n',
        )
    return source


if __name__ == "__main__":
    package = Path(find_spec("vllm_omni").origin).parent
    talker = package / "model_executor/models/voxcpm2/voxcpm2_talker.py"
    talker.write_text(patch_source(talker.read_text()))
