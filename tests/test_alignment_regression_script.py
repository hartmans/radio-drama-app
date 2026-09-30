"""Exercise the regression runner through the real injector and timing projection."""
import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from carthage.dependency_injection import InjectionKey

from scripts import check_alignment_regression as regression
from radio_drama.forced_alignment import AlignmentResult, ForcedAlignmentResource, WordTiming


@pytest.mark.parametrize('mode', ['complete', 'asr'])
def test_regression_runner_projects_injected_results_and_flags_missing_boundary(tmp_path, monkeypatch, mode):
    sf.write(tmp_path / 'case.wav', np.zeros(32000), 16000)
    (tmp_path / 'case.json').write_text(json.dumps({'dialogue_lines': [
        {'spoken_text': 'Hello there'}, {'spoken_text': 'Goodbye now'}]}))
    (tmp_path / 'case.meta').write_text(json.dumps({'dialogue_line_spans': [[0., .9], [1., 1.9]]}))
    requests = []
    class Fixture(ForcedAlignmentResource):
        async def _process_batch(self, batch):
            requests.extend(batch)
            return [AlignmentResult((WordTiming('Hello', 0., .4), WordTiming('there', .5, .9),
                                     WordTiming('Goodbye', 1., 1.4), WordTiming('now', 1.5, None)), ())
                    for _ in batch]
    original = regression.radio_drama_injector
    def injector(**kwargs):
        result = original(**kwargs)
        result.replace_provider(InjectionKey(ForcedAlignmentResource), Fixture)
        return result
    monkeypatch.setattr(regression, 'radio_drama_injector', injector)
    args = SimpleNamespace(cache_dir=tmp_path, pattern='*.wav', backend='qwen', device='cpu',
                           batch_size=3, language='en', mode=mode, threshold=.4,
                           output=tmp_path / 'report.json')
    assert asyncio.run(regression.run(args)) == 1
    report = json.loads(args.output.read_text())
    assert report['failed_cases'] == 1
    assert report['results'][0]['failures'] == [
        {'line': 1, 'boundary': 'end', 'reference_s': 1.9, 'actual_s': None, 'reason': 'missing boundary'}]
    assert requests[0].transcript_kind == ('complete' if mode == 'complete' else 'partial')
    assert requests[0].sample_rate == 16000
    assert requests[0].require_word_alignment


def test_regression_threshold_and_unknown_reference():
    from radio_drama.rendering import ScriptTiming, DialogueLineTiming
    timing = ScriptTiming((DialogueLineTiming(.5, float('nan')),))
    report = regression.compare_spans(timing, [(0., None)], .4)
    assert report['failures'][0]['error_s'] == .5
    assert report['unknown_reference_boundaries'] == 1
    assert not regression.compare_spans(timing, [(0., None)], .5)['failures']
