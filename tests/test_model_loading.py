"""Check cold runtime initialization in a fresh interpreter."""
import subprocess
import sys


def test_model_load_initializes_dynamo_before_inference():
    # Other tests import Transformers and hide the cold-start import race.
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
from radio_drama.model_loading import shared_model_load

assert 'torch._dynamo' not in sys.modules
with shared_model_load():
    import torch
    assert 'torch._dynamo' in sys.modules
    assert not torch._dynamo.__spec__._initializing
    assert callable(torch._dynamo.disable)
"""],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
