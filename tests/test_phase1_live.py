from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "radio_drama_app.py"
VOICE_DIR = REPO_ROOT / "voices"


def _pythonpath_for_subprocess() -> str:
    entries = [str(REPO_ROOT)]
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


@pytest.mark.live
def test_live_end_to_end_qwen_script(tmp_path: Path):
    xml_path = tmp_path / "live-qwen-production.xml"
    wav_path = tmp_path / "live-qwen-production.wav"
    xml_path.write_text(
        """
        <production>
          <speaker-map>
            Guide: chandra.wav
            Builder: david.wav
          </speaker-map>
          <script tts="qwen">
            Guide: We need a working Qwen render.

            Builder: This should use voice cloning and still produce stereo output.
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


@pytest.mark.live
def test_live_end_to_end_mixed_tts_scripts(tmp_path: Path):
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    shutil.copy2(VOICE_DIR / "chandra.wav", voice_dir / "guide-fresh.wav")
    shutil.copy2(VOICE_DIR / "david.wav", voice_dir / "builder-fresh.wav")

    xml_path = tmp_path / "live-mixed-production.xml"
    wav_path = tmp_path / "live-mixed-production.wav"
    xml_path.write_text(
        """
        <production>
          <speaker-map>
            Guide: guide-fresh.wav
            Builder: builder-fresh.wav
          </speaker-map>
          <script tts="qwen">
            Guide: The prompt path should load Qwen and WhisperX together.
          </script>
          <script>
            Builder: VibeVoice should still load successfully in the same production render.
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
            str(voice_dir),
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


@pytest.mark.live
def test_live_vibevoice_native_batch(tmp_path: Path):
    """Render scripts with different speaker counts through native Transformers."""
    import asyncio
    import numpy as np

    from phase1_helpers import make_async_injector
    from radio_drama.config import ProductionConfig
    from radio_drama.dialogue import DialogueLine, ScriptRenderRequest, SpeakerVoiceReference
    from radio_drama.vibevoice import VibeVoiceResource

    async def render():
        injector, ainjector = await make_async_injector(
            ProductionConfig(device="cuda", batch_size=2),
        )
        try:
            resource = await ainjector(VibeVoiceResource)
            voices = [
                SpeakerVoiceReference(
                    authored_name=name,
                    voice_name=name,
                    resolved_path=REPO_ROOT / "example_voices" / name,
                )
                for name in ("lawyer1.wav", "lawyer2.wav")
            ]
            requests = [
                ScriptRenderRequest(dialogue_lines=[
                    DialogueLine(speaker=voices[0], spoken_text="Good morning. Are we ready to begin?"),
                    DialogueLine(speaker=voices[1], spoken_text="Yes, let us begin."),
                ]),
                ScriptRenderRequest(dialogue_lines=[
                    DialogueLine(speaker=voices[1], spoken_text="This is a second script in the same batch."),
                ]),
            ]
            registrations = [await resource.register_backend_request(request) for request in requests]
            return await asyncio.gather(*(registration.render() for registration in registrations))
        finally:
            injector.close()

    results = asyncio.run(render())
    assert len(results) == 2
    for index, result in enumerate(results):
        assert result.sample_rate == 24000
        assert result.audio.ndim == 1
        assert result.audio.size > result.sample_rate
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) > 0.01
        sf.write(tmp_path / f"script-{index}.wav", result.audio, result.sample_rate)
