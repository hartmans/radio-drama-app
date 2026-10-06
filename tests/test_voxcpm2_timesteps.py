import pytest

from tts_engines.voxcpm2_longform.patch_timesteps import patch_source


@pytest.mark.parametrize("override, expected, cfg, guidance", [(None, 20, None, 2.0), ("30", 30, "3.5", 3.5)])
def test_longform_patch_sets_generation_before_model_initialization(monkeypatch, override, expected, cfg, guidance):
    monkeypatch.delenv("VOXCPM_INFERENCE_TIMESTEPS", raising=False)
    monkeypatch.delenv("VOXCPM_CFG_VALUE", raising=False)
    if override is not None:
        monkeypatch.setenv("VOXCPM_INFERENCE_TIMESTEPS", override)
    if cfg is not None:
        monkeypatch.setenv("VOXCPM_CFG_VALUE", cfg)
    source = "class Talker:\n    def __init__(self):\n        self._inference_timesteps = 10\n        self._cfg_value = 2.0\n"
    namespace = {}
    exec(patch_source(source), namespace)
    assert namespace["Talker"]()._inference_timesteps == expected
    assert namespace["Talker"]()._cfg_value == guidance


@pytest.mark.parametrize("source", ["", "        self._inference_timesteps = 10\n" * 2, "        self._inference_timesteps = 10\n"])
def test_longform_patch_rejects_changed_upstream(source):
    with pytest.raises(ValueError, match="exactly one"):
        patch_source(source)
