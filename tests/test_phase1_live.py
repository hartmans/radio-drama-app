from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "radio_drama_app.py"
VOICE_DIR = REPO_ROOT / "voices"
VIBEVOICE_ROOT = Path("/home/hartmans/ai/vibevoice")
VENV_SITE_PACKAGES = Path(
    "/home/hartmans/ai/vibevoice/.venv/lib/python3.13/site-packages"
)


def _pythonpath_for_subprocess() -> str:
    entries = [str(REPO_ROOT), str(VIBEVOICE_ROOT), str(VENV_SITE_PACKAGES)]
    inherited = os.environ.get("PYTHONPATH")
    if inherited:
        entries.append(inherited)
    return ":".join(entries)


@pytest.mark.live
def test_live_end_to_end_two_scripts(tmp_path: Path):
    xml_path = tmp_path / "live-production.xml"
    wav_path = tmp_path / "live-production.wav"
    xml_path.write_text(
        """
        <production>
          <speaker-map>
            Guide: chandra.wav
            Builder: david.wav
          </speaker-map>
          <script>
            Guide: We need a working live render.
            This first scene should establish the pipeline.

            Builder: Then the second voice answers clearly.
          </script>
          <script>
            Builder: The next script should still join the same batch.

            Guide: And the file should come out as stereo at forty eight kilohertz.
          </script>
        </production>
        """,
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_for_subprocess()

    completed = subprocess.run(
        [
            sys.executable,
            str(APP_PATH),
            str(xml_path),
            "--voice-dir",
            str(VOICE_DIR),
            "--output",
            str(wav_path),
            "--device",
            "cuda",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )
    assert wav_path.is_file(), f"Expected output file {wav_path} to exist"

    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    assert sample_rate == 48000
    assert audio.ndim == 2
    assert audio.shape[1] == 2
    assert audio.shape[0] > 0
