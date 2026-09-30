"""Compare an injected aligner's projected line boundaries with saved ground truth.

Supports request .json + .meta caches and standalone segment .json files beside
.wav files. Reads inputs without updating their cache; reports contain no text.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import sys

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carthage.dependency_injection import AsyncInjector
from radio_drama.config import ProductionConfig
from radio_drama.dialogue import DialogueLine, SpeakerVoiceReference
from radio_drama.forced_alignment import ForcedAlignmentResource
from radio_drama.init import radio_drama_injector
from radio_drama.rendering import RenderResult


def load_case(path):
    """Read authored lines and reference spans, preserving unknown boundaries."""
    payload = json.loads(path.with_suffix('.json').read_text())
    if 'dialogue_lines' in payload:
        entries = payload['dialogue_lines']
        metadata = json.loads(path.with_suffix('.meta').read_text())
        spans = metadata['dialogue_line_spans']
    else:
        entries = [{'spoken_text': segment['text']} for segment in payload['segments']]
        spans = [(segment['start'], segment['end']) for segment in payload['segments']]
    if len(entries) != len(spans):
        raise ValueError(f'{path.name}: line count differs from reference span count')
    speaker = SpeakerVoiceReference('regression', 'unused', path)
    lines = [DialogueLine(speaker, entry['spoken_text'], mark_offsets=tuple(entry.get('mark_offsets', ())))
             for entry in entries]
    return lines, spans


def compare_spans(timing, reference, threshold):
    """Known references require known predictions; unknown references are unscored."""
    failures, errors = [], []
    unknown_reference = 0
    for index, (actual, expected) in enumerate(zip(timing.dialogue_lines, reference, strict=True)):
        for side, measured, truth in zip(('start', 'end'), (actual.start, actual.end), expected, strict=True):
            if truth is None or not math.isfinite(truth):
                unknown_reference += 1
                continue
            if measured is None or not math.isfinite(measured):
                failures.append({'line': index, 'boundary': side, 'reference_s': truth,
                                 'actual_s': None, 'reason': 'missing boundary'})
                continue
            error = abs(measured - truth)
            errors.append(error)
            if error > threshold:
                failures.append({'line': index, 'boundary': side, 'reference_s': truth,
                                 'actual_s': measured, 'error_s': error, 'reason': 'threshold exceeded'})
    return {'compared_boundaries': len(errors), 'unknown_reference_boundaries': unknown_reference,
            'max_error_s': max(errors, default=None), 'failures': failures}


async def run(args):
    paths = sorted(args.cache_dir.rglob(args.pattern))
    if not paths:
        raise ValueError(f'No audio matches {args.pattern!r} in {args.cache_dir}')
    config = ProductionConfig(alignment_backend=args.backend, device=args.device,
                              batch_size=args.batch_size, alignment_language=args.language)
    injector = radio_drama_injector(config=config, event_loop=asyncio.get_running_loop())
    reports = []
    try:
        resource = await injector(AsyncInjector).get_instance_async(ForcedAlignmentResource)
        for path in paths:
            name = str(path.relative_to(args.cache_dir))
            try:
                lines, reference = load_case(path)
                if not any(value is not None and math.isfinite(value) for span in reference for value in span):
                    raise ValueError('No known ground-truth boundaries')
                audio, rate = sf.read(path, dtype='float32')
                timing = await resource.script_timing(
                    lines, RenderResult(audio=audio), sample_rate=rate,
                    transcript_kind='complete' if args.mode == 'complete' else 'partial',
                    require_word_alignment=True, language=args.language)
                report = {'audio': name, **compare_spans(timing, reference, args.threshold)}
            except Exception as exc:
                report = {'audio': name, 'error': f'{type(exc).__name__}: {exc}'}
            reports.append(report)
            print(json.dumps(report, allow_nan=False), flush=True)
        summary = {'backend': args.backend, 'mode': args.mode, 'threshold_s': args.threshold,
                   'cases': len(reports), 'failed_cases': sum(bool(r.get('error') or r.get('failures')) for r in reports),
                   'results': reports}
        if args.output:
            args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print(json.dumps({key: value for key, value in summary.items() if key != 'results'}), flush=True)
        return 1 if summary['failed_cases'] else 0
    finally:
        injector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache_dir', type=Path)
    parser.add_argument('--backend', choices=('qwen', 'whisperx'), default='qwen')
    parser.add_argument('--mode', choices=('complete', 'asr'), default='complete')
    parser.add_argument('--threshold', type=float, default=.4, help='Maximum absolute boundary error in seconds')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--language', default='en')
    parser.add_argument('--pattern', default='*.wav', help='Recursive audio glob relative to cache directory')
    parser.add_argument('--output', type=Path, help='Optional JSON report path; never updates ground truth')
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or args.threshold < 0:
        parser.error('--threshold must be finite and nonnegative')
    return asyncio.run(run(args))


if __name__ == '__main__':
    sys.exit(main())
