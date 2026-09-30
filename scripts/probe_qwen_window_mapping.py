"""Probe public Qwen timestamps using private, existing TTS cache recordings.

The manifest is a JSON list of audio paths with adjacent .json dialogue requests
and .meta line spans. Reports contain counts and timestamp errors, never source
text. Detailed model output stays in the explicitly selected output directory.
This is an experiment, not a validation of a production window scheduler.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carthage.dependency_injection import Injector
from radio_drama.config import ProductionConfig
from radio_drama.forced_alignment.qwen import QwenAlignmentResource, _mono_audio
from radio_drama.forced_alignment.base import AlignmentResult, WordTiming
from radio_drama.forced_alignment.projection import script_timing_from_alignment
from radio_drama.dialogue import DialogueLine, SpeakerVoiceReference


@dataclass
class Probe:
    audio: np.ndarray
    words: list[str]
    prefix_count: int
    duration: float
    case_index: int


def load_probe(path, index, target):
    request = json.loads(path.with_suffix('.json').read_text())
    metadata = json.loads(path.with_suffix('.meta').read_text())
    lines = request['dialogue_lines']
    spans = metadata['dialogue_line_spans']
    assert len(lines) == len(spans)
    # Cut inside the following pause, after a known authored line end.
    possible = [(i, float(end)) for i, (_, end) in enumerate(spans[:-1])
                if end is not None and 30 <= float(end) <= target]
    line_index, end = possible[-1]
    next_start = float(spans[line_index + 1][0])
    cutoff = end + max(0., min(.3, (next_start - end) / 2))
    audio, rate = sf.read(path, dtype='float32')
    audio = _mono_audio(audio, rate)[:round(cutoff * 16000)]
    words = [word for line in lines for word in line['spoken_text'].split()]
    count = sum(len(line['spoken_text'].split()) for line in lines[:line_index + 1])
    return Probe(audio, words, count, len(audio) / 16000, index)


def compare(reference, candidate):
    # Match public word order; the tokenizer can split an authored whitespace word.
    size = min(len(reference), len(candidate))
    assert [w['text'] for w in reference[:size]] == [w['text'] for w in candidate[:size]]
    errors = [abs(a[side] - b[side]) for a,b in zip(reference[:size],candidate[:size])
              for side in ('start_time','end_time')]
    return {'compared_words': size, 'median_error_s': round(float(np.median(errors)),3),
            'p95_error_s': round(float(np.quantile(errors,.95)),3),
            'changed_over_400ms': sum(value > .4 for value in errors)}


def run(args):
    paths = [Path(p) for p in json.loads(args.manifest.read_text())]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    injector = Injector()
    resource = QwenAlignmentResource(injector=injector, config=ProductionConfig(device='cuda'))
    reports = []
    try:
        for index,path in enumerate(paths):
            probe = load_probe(path,index,args.target_seconds)
            extras = [0, 10, 40, 100]
            transcripts = [' '.join(probe.words[:probe.prefix_count + extra]) for extra in extras]
            started = time.monotonic()
            # Batch just two long windows at once during exploration.
            results = []
            for beg in range(0,len(transcripts),2):
                batch = transcripts[beg:beg+2]
                results.extend(resource._align_batch([probe.audio] * len(batch),batch,['en'] * len(batch)))
            (output / f'case-{index}-decoded.json').write_text(json.dumps(results,indent=2))
            reference = results[0]
            record = {'case': index, 'duration_s': round(probe.duration,3),
                      'authored_prefix_words': probe.prefix_count, 'reference_words':len(reference),
                      'elapsed_s': round(time.monotonic()-started,2), 'variants':[]}
            for extra,words in zip(extras,results):
                tail = words[len(reference):]
                record['variants'].append({
                    'extra_authored_words':extra, **compare(reference,words),
                    'tail_words':len(tail),
                    'tail_inside_audio':sum(0 <= w['start_time'] <= w['end_time'] <= probe.duration for w in tail),
                    'tail_after_audio':sum(w['end_time'] > probe.duration for w in tail),
                    'tail_collapsed':sum(w['start_time'] == w['end_time'] for w in tail),
                    'tail_first_start_s':tail[0]['start_time'] if tail else None,
                    'tail_last_end_s':tail[-1]['end_time'] if tail else None})
            reports.append(record)
            print(json.dumps(record),flush=True)
            (output / 'summary.json').write_text(json.dumps(reports,indent=2))
    finally:
        resource.close()
        injector.close()


def run_greedy(args):
    """Explore prefix advancement using public word timestamps and overlap."""
    root=args.output
    root.mkdir(parents=True,exist_ok=True)
    manifest=args.manifest
    paths=[Path(p) for p in json.loads(manifest.read_text())]
    injector=Injector()
    r=QwenAlignmentResource(injector=injector,config=ProductionConfig(device='cuda'))
    processor,_=r._ensure_aligner()
    reports=[]
    try:
        for case,path in enumerate(paths):
            data=json.loads(path.with_suffix('.json').read_text());lines=data['dialogue_lines']
            meta_path=path.with_suffix('.meta')
            meta=json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            full=' '.join(line['spoken_text'] for line in lines)
            units=processor.split_words_for_alignment(full,'English')
            audio,rate=sf.read(path,dtype='float32');audio=_mono_audio(audio,rate)
            duration=len(audio)/16000
            word_cursor=0;audio_cursor=0.;accepted={};rounds=[];started=time.monotonic()
            for attempt in range(100):
                remaining_audio=duration-audio_cursor
                seconds=min(180.,remaining_audio)
                final=remaining_audio<=180.
                rate_estimate=(len(units)-word_cursor)/remaining_audio
                count=len(units)-word_cursor if final else min(len(units)-word_cursor,max(20,round(rate_estimate*seconds*1.3)+12))
                clip=audio[round(audio_cursor*16000):round((audio_cursor+seconds)*16000)]
                while True:
                    items=r._align_batch([clip],[' '.join(units[word_cursor:word_cursor+count])],['en'])[0]
                    assert len(items)==count
                    limit=seconds if final else seconds-10
                    good=0
                    for word in items:
                        if not 0<=word['start_time']<=word['end_time']<=limit:break
                        good+=1
                    if not final and good==count and word_cursor+count<len(units):
                        count=min(len(units)-word_cursor,max(count+20,round(count*1.5)));continue
                    break
                if not good:raise RuntimeError(f'no progress case {case} attempt {attempt}')
                errors=[]
                for j,word in enumerate(items[:good]):
                    index=word_cursor+j
                    new=WordTiming(word['text'],word['start_time']+audio_cursor,word['end_time']+audio_cursor)
                    if index in accepted:errors.extend([abs(new.start-accepted[index].start),abs(new.end-accepted[index].end)])
                    else:accepted[index]=new
                rounds.append({'audio_start':round(audio_cursor,3),'audio_seconds':round(seconds,3),
                               'word_start':word_cursor,'supplied':count,'accepted':good,
                               'overlap_boundaries':len(errors),'overlap_p95_s':round(float(np.quantile(errors,.95)),3) if errors else None if errors else None,
                               'overlap_max_s':round(max(errors),3) if errors else None,'final':final})
                if final:
                    if good!=len(units)-word_cursor:raise RuntimeError(f'final transcript coverage failed case {case}: {good}/{len(units)-word_cursor}')
                    break
                target=items[good-1]['end_time']-10.
                back=next(j for j,word in enumerate(items[:good]) if word['start_time']>=target)
                next_audio=audio_cursor+max(0.,items[back]['start_time']-.3)
                if next_audio<=audio_cursor or back<=0:raise RuntimeError('insufficient forward progress')
                word_cursor+=back;audio_cursor=next_audio
            else:raise RuntimeError('too many windows')
            assert len(accepted)==len(units)
            evidence=AlignmentResult(tuple(accepted[i] for i in range(len(units))),(),source_text=full)
            speaker=SpeakerVoiceReference('fixture','unused',Path('unused'))
            timing=script_timing_from_alignment([DialogueLine(speaker,line['spoken_text']) for line in lines],evidence)
            errors=[abs(actual-reference) for span,expected in zip(timing.dialogue_lines,(meta.get('dialogue_line_spans') or []))
                    for actual,reference in zip((span.start,span.end),expected) if np.isfinite(actual) and np.isfinite(reference)]
            report={'case':case,'duration_s':round(duration,2),'words':len(units),'windows':rounds,
                    'unknown_lines':sum(not np.isfinite(span.start) or not np.isfinite(span.end) for span in timing.dialogue_lines),
                    'reference_line_error_median_s':round(float(np.median(errors)),3) if errors else None,
                    'reference_line_error_p95_s':round(float(np.quantile(errors,.95)),3) if errors else None,
                    'elapsed_s':round(time.monotonic()-started,2)}
            reports.append(report)
            (root/f'greedy-case-{case}.json').write_text(json.dumps([{'text':w.text,'start':w.start,'end':w.end} for w in evidence.words],indent=2))
            (root/'greedy-summary.json').write_text(json.dumps(reports,indent=2))
            print(json.dumps(report),flush=True)
    finally:r.close();injector.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-seconds',type=float,default=170.)
    parser.add_argument('--greedy',action='store_true',help='Probe full-recording greedy advancement with overlap')
    args=parser.parse_args()
    (run_greedy if args.greedy else run)(args)
