from __future__ import annotations

import asyncio
import io

import numpy as np
from fastapi.testclient import TestClient
from scipy.io import wavfile

from radio_drama.backend import PresetAudioStore, create_app
from radio_drama.rendering import RenderResult


def _base_result() -> RenderResult:
    frames = 4096
    ramp = np.linspace(-0.3, 0.3, frames, dtype=np.float32)
    audio = np.column_stack((ramp, ramp[::-1]))
    return RenderResult(audio=audio)


def test_preset_audio_store_prepares_and_slices():
    store = PresetAudioStore(base_result=_base_result(), sample_rate=48000)

    prepared_names = asyncio.run(
        store.prepare_presets([" narrator1 ", "indoor1", "narrator1"])
    )
    sliced = store.slice_preset("narrator1", from_time=0.01)

    assert prepared_names == ("narrator1", "indoor1")
    assert sliced.audio.shape == (3616, 2)
    assert np.shares_memory(sliced.audio, store.prepared_results["narrator1"].audio)


def test_backend_audio_slice_requires_prepared_preset():
    app = create_app(PresetAudioStore(base_result=_base_result(), sample_rate=48000))

    with TestClient(app) as client:
        response = client.post(
            "/api/audio-slice",
            json={"preset_name": "narrator1", "from_time": 0.0},
        )

    assert response.status_code == 409
    assert "has not been prepared" in response.json()["detail"]


def test_backend_prepare_and_slice_endpoints():
    app = create_app(PresetAudioStore(base_result=_base_result(), sample_rate=48000))

    with TestClient(app) as client:
        prepare_response = client.post(
            "/api/presets/prepare",
            json={"preset_names": ["narrator1", "outdoor2"]},
        )
        slice_response = client.post(
            "/api/audio-slice",
            json={"preset_name": "outdoor2", "from_time": 0.02},
        )

    assert prepare_response.status_code == 200
    assert prepare_response.json()["preset_names"] == ["narrator1", "outdoor2"]
    assert slice_response.status_code == 200
    assert slice_response.headers["content-type"] == "audio/wav"

    sample_rate, audio = wavfile.read(io.BytesIO(slice_response.content))
    assert sample_rate == 48000
    assert audio.shape == (3136, 2)
