from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile

from radio_drama.backend import ExpressionCacheStore, create_app
from radio_drama.effects import EffectChainRegistry
from radio_drama.rendering import RenderResult


@pytest.fixture
def audio_store(tmp_path):
    ramp = np.linspace(-0.3, 0.3, 4096, dtype=np.float32)
    registry = EffectChainRegistry()
    registry.add_from_expression("preview", "gain(line(6.0206))")
    return ExpressionCacheStore(
        base_result=RenderResult(audio=np.column_stack((ramp, ramp[::-1]))),
        sample_rate=48000,
        cache_dir=tmp_path,
        effect_chains=registry,
    )


def test_backend_status_and_base_audio(audio_store):
    with TestClient(create_app(audio_store)) as client:
        response = client.get("/api/status")
        assert response.status_code == 200
        status = response.json()
        assert status["preset_expressions"]["preview"] == "gain(line(6.0206))"
        assert status["total_duration_seconds"] == pytest.approx(4096 / 48000)
        assert status["sample_rate"] == 48000
        audio_response = client.get(f"/api/cache/{status['base_audio_file']}")

    assert audio_response.status_code == 200
    sample_rate, audio = wavfile.read(io.BytesIO(audio_response.content))
    assert sample_rate == 48000
    np.testing.assert_array_equal(audio, audio_store.base_result.audio)


def test_backend_applies_expression_and_reuses_cached_audio(audio_store, monkeypatch):
    original = audio_store.base_result.audio.copy()
    with TestClient(create_app(audio_store)) as client:
        response = client.post("/api/apply-expression", json={"expression": "preview"})
        assert response.status_code == 200
        result = response.json()
        assert result["duration_seconds"] == pytest.approx(4096 / 48000)
        assert result["sample_rate"] == 48000
        url = f"/api/cache/{result['filename']}.wav"
        audio_response = client.get(url)
        assert audio_response.status_code == 200
        sample_rate, audio = wavfile.read(io.BytesIO(audio_response.content))
        assert sample_rate == 48000
        np.testing.assert_allclose(audio, original * 2, atol=1e-6)
        np.testing.assert_array_equal(audio_store.base_result.audio, original)

        def unexpected_evaluation(*args, **kwargs):
            raise AssertionError("cached expressions must not be evaluated again")

        monkeypatch.setattr("radio_drama.backend.app.eval_expression", unexpected_evaluation)
        replay = client.post("/api/apply-expression", json={"expression": "preview"})
        assert replay.status_code == 200
        assert replay.json() == result
        assert client.get(url).content == audio_response.content

        partial = client.get(url, headers={"Range": "bytes=0-43"})
        assert partial.status_code == 206
        assert partial.content == audio_response.content[:44]


@pytest.mark.parametrize("expression", ["unknown_preset", "gain(", "123"])
def test_backend_rejects_invalid_expressions(audio_store, expression):
    with TestClient(create_app(audio_store)) as client:
        response = client.post("/api/apply-expression", json={"expression": expression})

    assert response.status_code == 422
    assert "Failed to apply expression" in response.json()["detail"]
    assert sorted(path.name for path in audio_store.cache_dir.iterdir()) == ["_base.wav"]
