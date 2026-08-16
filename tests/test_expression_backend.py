from __future__ import annotations

import asyncio

import numpy as np
from scipy.io import wavfile

from radio_drama.backend import ExpressionCacheStore
from radio_drama.backend.app import render_production_result
from radio_drama.config import ProductionConfig
from radio_drama.effects import EffectChainRegistry
from radio_drama.rendering import RenderResult


def test_render_production_returns_planned_effect_registry(tmp_path):
    production_path = tmp_path / "production.xml"
    production_path.write_text(
        """
        <production>
          <preset-map>
            production_preset: gain(line(3))
          </preset-map>
        </production>
        """,
        encoding="utf-8",
    )

    _, registry = asyncio.run(
        render_production_result(
            production_path,
            config=ProductionConfig(output_sample_rate=48000, output_channels=2),
        )
    )

    assert registry.get_expression("production_preset") == "gain(line(3))"


def test_expression_store_uses_production_effect_registry(tmp_path):
    registry = EffectChainRegistry()
    registry.add_from_expression("production_preset", "gain(line(6.0206))")
    store = ExpressionCacheStore(
        base_result=RenderResult(audio=np.full((4, 2), 0.25, dtype=np.float32)),
        sample_rate=4,
        cache_dir=tmp_path,
        effect_chains=registry,
    )

    assert store.preset_expressions["production_preset"] == "gain(line(6.0206))"

    filename, _, _ = asyncio.run(
        store.apply_expression_and_cache("production_preset")
    )
    _, rendered = wavfile.read(tmp_path / f"{filename}.wav")

    np.testing.assert_allclose(rendered, np.full((4, 2), 0.5, dtype=np.float32), atol=1e-4)


def test_expression_cache_key_includes_production_presets(tmp_path):
    base_result = RenderResult(audio=np.full((4, 2), 0.25, dtype=np.float32))
    first_registry = EffectChainRegistry()
    first_registry.add_from_expression("production_preset", "gain(line(1))")
    second_registry = EffectChainRegistry()
    second_registry.add_from_expression("production_preset", "gain(line(2))")

    first_store = ExpressionCacheStore(
        base_result=base_result,
        sample_rate=4,
        cache_dir=tmp_path,
        effect_chains=first_registry,
    )
    second_store = ExpressionCacheStore(
        base_result=base_result,
        sample_rate=4,
        cache_dir=tmp_path,
        effect_chains=second_registry,
    )

    assert first_store._expression_sha256("production_preset") != (
        second_store._expression_sha256("production_preset")
    )
