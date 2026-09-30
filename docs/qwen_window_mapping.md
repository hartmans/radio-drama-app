# Qwen complete-transcript window mapping experiments

These experiments use public processor word timestamps, complete supplied
transcripts and overlapping audio windows. They do not use ASR, logits or the
processor's unrepaired timestamp predictions.

## Reproducing the probes

`scripts/probe_qwen_window_mapping.py` takes a JSON manifest of audio paths.
Each path must have adjacent cache request `.json` metadata with dialogue lines.
The prefix-excess probe also needs `.meta` metadata with existing line spans;
the full-recording `--greedy` mode can run without reference spans.

```sh
~/venv/bin/python scripts/probe_qwen_window_mapping.py \
  --manifest PRIVATE_MANIFEST.json --output PRIVATE_OUTPUT_DIRECTORY
~/venv/bin/python scripts/probe_qwen_window_mapping.py \
  --manifest PRIVATE_MANIFEST.json --output PRIVATE_OUTPUT_DIRECTORY --greedy
```

Use an ignored directory such as `.alignment-experiments` for private manifests
and results. Detailed decoded word records contain source text and must not be
committed. The source recordings used here are authorized for experimentation,
not for checked-in test fixtures. Model loading honors the default Hugging Face
cache and its environment configuration.

## Observations

Three excerpts were cut at existing aligned line ends, each approximately
168–170 seconds long. Supplying 10, 40 or 100 additional authored words beyond
the known cutoff produced the following behavior:

* most extra words received timestamps beyond the actual audio duration;
* zero to two extra words received timestamps entirely within the audio;
* two excerpts had identical preceding word timestamps across all variants;
* the third changed at most five preceding timestamp boundaries by over 400 ms,
  while its median and 95th-percentile preceding-boundary changes remained zero.

Consequently, accepting every returned word whose timestamp fits inside the
window can accept unspoken text. Leaving a provisional tail and checking audio
and transcript overlap is necessary. Stability under added text alone does not
prove acoustic correctness.

A greedy prototype subsequently processed complete recordings:

| Duration | Words | Windows | Maximum overlap boundary disagreement |
| --- | --- | --- | --- |
| 331.47 s | 836 | 2 | 0.10 s |
| 271.60 s | 666 | 2 | 0.06 s |
| 266.13 s | 435 | 2 | 0.26 s |
| 641.47 s | 1,728 | 4 | 0.22 s |

Every supplied processor word unit received a record. This establishes transcript
coverage, not proof that the audio spoke every supplied word correctly. Comparing
shared line projection with existing cached alignment yielded median boundary
differences of approximately 0.07–0.13 seconds. One recording had a 95th-percentile
difference of 1.69 seconds; three recordings were below 0.22 seconds. Some line
boundaries remained unknown under the shared authored-text matcher. Existing
cached alignments are comparison evidence, not absolute ground truth.

## Production procedure

The scheduler chooses a transcript prefix from the remaining transcript word
count divided by the remaining audio duration, with 30 percent excess and twelve
additional words. These constants are experimental. When all supplied words fit
within the usable window, it increases the prefix and retries.

It aligns 180 seconds of audio and provisionally accepts the contiguous prefix
ending before 170 seconds. It then backs up approximately ten seconds in both
returned word indexes and measured audio time, adding 300 ms of leading context.
Processor word identities come from splitting the complete transcript once;
shared overlap is compared by those identities, including repeated passages.
The final window receives all remaining transcript words. The model's public
outputs determine boundaries; the average speaking rate only chooses input text.

Production requires at least four comparable boundaries, at least 80 percent
within 400 ms, and two adjacent units whose starts and ends both agree. A few
local outliers therefore do not invalidate otherwise consistent surrounding
anchors. On failure it retries with 20- and 30-second overlap. Prefix expansion
has four attempts; ambiguity or lack of progress raises explicitly without ASR.
Previously accepted measurements remain authoritative at the seam. Final windows
retain all remaining units, including unknown boundaries. Completion also handles
long trailing silence once all transcript units have been accepted.

Backend-independent tests cover 30-minute inputs, repeated passages, bounded
conflict retries, unknown final boundaries and no-ASR behavior. Native GPU tests
exercise real overlapping complete windows and incorrect supplied text. Long
requests advance sequentially; independent short requests remain batched.
Broader evaluation of pauses, speaking-rate changes and throughput remains useful.

## Incorrect supplied words

Native Qwen experiments inserted three incorrect words inside a line, at the
start of a line, after the final line, and substituted one word. Interior incorrect
words often received plausible positive intervals taken from real speech. They
cannot reliably be discarded using public timestamps alone. On the publishable
short fixture, all neighboring line boundaries stayed identical; appending wrong
words beyond the audio made the final line end unknown while preserving earlier
lines. Checked-in decoded evidence comes only from the existing test audio.

Production experiments inserted incorrect words near the beginning, a window
seam and the end of each of four private recordings. No neighboring line boundary
was lost. Three recordings preserved neighboring times within 400 ms. On one,
some neighboring timestamps moved by more than 400 ms. Comparing the projection's
word correspondences confirmed that it selected the same neighboring source words
(after accounting for the inserted indexes): those movements came from Qwen's
predictions, rather than mismatches introduced by projection.

These observations support preserving lines and isolating invalid boundaries,
but do not establish that bad words will be missed or that other timestamps will
always remain unchanged. Overlap agreement validates window placement, not the
acoustic truth of every supplied word. Marks placed at an incorrect word can
therefore receive a plausible but false time.

## Running a ground-truth regression

`scripts/check_alignment_regression.py` runs the selected injected
`ForcedAlignmentResource` through request registration, backend inference and
shared `script_timing` projection. It compares every known line start/end with
reference times, flags missing predictions and errors exceeding the threshold,
and exits 1 if any case fails (0 if all pass). Cases with no known ground-truth
boundaries fail explicitly. Unknown reference boundaries are counted but unscored.
Line indexes in reports are zero-based; reports omit transcript text.

```sh
~/venv/bin/python scripts/check_alignment_regression.py /path/to/ground-truth-cache \
  --backend qwen --mode complete --threshold 0.4 --output /tmp/alignment-report.json
~/venv/bin/python scripts/check_alignment_regression.py /path/to/ground-truth-cache \
  --backend qwen --mode asr --threshold 0.4
~/ai/vibevoice/.venv/bin/python scripts/check_alignment_regression.py /path/to/ground-truth-cache \
  --backend whisperx --mode asr --threshold 0.9
```

The directory is searched recursively for `*.wav` (change with `--pattern`). Each
WAV needs a same-stem `.json` containing either:

* a cache request with `dialogue_lines`, each with `spoken_text` and optional
  `mark_offsets`, plus same-stem `.meta` with `dialogue_line_spans` pairs; or
* `segments` with `text`, `start`, and `end` in the `.json` itself.

Reference times are seconds from the WAV's beginning. `complete` declares that
all supplied lines together cover its spoken content; `asr` treats them as a
partial authored transcript and invokes the backend's ASR path. WhisperX retains
its ASR-first implementation in either mode. Word alignment is requested in both
modes. This runner compares line boundaries, not individual mark times or final
production mixing. It never rewrites ground truth and continues reporting other
cases after a case fails. Run reports outside the input directory, especially if
using a broad audio glob. Source data remains external; no private audio is needed
in the repository to retain this regression procedure.
