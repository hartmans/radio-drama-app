from types import SimpleNamespace

import numpy as np
import torch

from carthage.dependency_injection import InjectionKey
from radio_drama.cli_utils import initialize_arg_parser, build_config_from_namespace
from radio_drama.config import DEFAULT_MODEL_PATH, DEFAULT_UPSTREAM_MODEL_PATH, ProductionConfig
from radio_drama.dialogue import TtsResource
from radio_drama.init import radio_drama_injector
from radio_drama.vibevoice import VibeVoiceResource
from radio_drama.vibevoice_upstream import UpstreamVibeVoiceResource
from phase1_helpers import request_from_normalized_script


def test_cli_selects_upstream_checkpoint_and_independent_alignment():
    parser = initialize_arg_parser('test')
    config = build_config_from_namespace(parser.parse_args([
        'script.xml', '--upstream-vibevoice', '--alignment-backend', 'whisperx',
    ]))
    assert config.upstream_vibevoice
    assert config.resolved_model_name == DEFAULT_UPSTREAM_MODEL_PATH
    assert config.alignment_backend == 'whisperx'
    config.model_name = 'custom-model'
    assert config.resolved_model_name == 'custom-model'
    assert ProductionConfig().resolved_model_name == DEFAULT_MODEL_PATH
    for upstream, expected in [(True, UpstreamVibeVoiceResource), (False, VibeVoiceResource)]:
        injector = radio_drama_injector(config=ProductionConfig(upstream_vibevoice=upstream, device='cpu'))
        try:
            resource = injector.get_instance(InjectionKey(TtsResource, tts='vibevoice'))
            assert type(resource) is expected
        finally:
            injector.close()


def test_upstream_render_uses_original_api_and_normalizes_speaker_slots():
    calls = {}
    processor = SimpleNamespace(audio_processor=SimpleNamespace(sampling_rate=24000), tokenizer=object())

    class Processor:
        audio_processor = processor.audio_processor
        tokenizer = processor.tokenizer

        def __call__(self, **kwargs):
            calls['processor'] = kwargs
            return {'input_ids': torch.tensor([[1]]), 'speech_tensors': torch.ones(1, 8)}

    class Model:
        def generate(self, **kwargs):
            calls['generate'] = kwargs
            return SimpleNamespace(speech_outputs=[torch.ones(1, 24)])

    class Resource(UpstreamVibeVoiceResource):
        def _ensure_loaded(self):
            self._sample_rate = 24000
            return Processor(), Model()

        def _preprocessed_voice_sample_sync(self, path, **kwargs):
            return np.zeros(8, dtype=np.float32)

    injector = radio_drama_injector(config=ProductionConfig(
        upstream_vibevoice=True, device='cpu', cfg_scale=1.3, ddpm_inference_steps=5,
    ))
    try:
        resource = injector(Resource)
        request = request_from_normalized_script(
            'Speaker 1: Hello.\nSpeaker 2: I’m here.\nSpeaker 1: Welcome.',
            ('anna.wav', 'bob.wav'),
        )
        result = resource._render_batch_native_sync([SimpleNamespace(request=request)])
        assert result[0].shape == (24,)
        assert calls['processor']['text'] == ["Speaker 1: Hello.\nSpeaker 2: I'm here.\nSpeaker 1: Welcome."]
        assert len(calls['processor']['voice_samples'][0]) == 2
        assert calls['generate']['cfg_scale'] == 1.3
        assert calls['generate']['generation_config'] == {'do_sample': False}
        assert calls['generate']['tokenizer'] is processor.tokenizer
        assert calls['generate']['is_prefill'] is True
        assert 'guidance_scale' not in calls['generate']
    finally:
        injector.close()
