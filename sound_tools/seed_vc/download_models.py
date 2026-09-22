"""Populate the persistent cache with Seed-VC's speech-conversion models."""

from argparse import Namespace
import gc
import sys

import torch

sys.path.insert(0, "/opt/seed-vc")
import inference  # noqa: E402


def download(f0_condition):
    """Exercise upstream model loading so every lazy Hugging Face asset is cached."""
    models = inference.load_models(
        Namespace(
            fp16=True,
            f0_condition=f0_condition,
            checkpoint=None,
            config=None,
        )
    )
    del models
    gc.collect()
    torch.cuda.empty_cache()


download(False)
download(True)
