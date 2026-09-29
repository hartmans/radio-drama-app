# Forced alignment backends and shared timing

Implementation plan, researched 2026-09-29. This document specifies future work;
the implementation has not been changed. Follow the repository's AGENTS.md.

## 1. Decisions and scope

Introduce `ForcedAlignmentResource` as the injectable interface. Its registered
requests return a model-independent `AlignmentResult`. Raw WhisperX dictionaries,
Qwen processor results, tensors, and backend decisions stay inside the concrete
backend. Do not generalize `WhisperXResponse` into the application interface.

Both recorded audio and generated speech use `script_timing()` and then the same
`fill_start_positions_from_timing()` projection. Preserve existing WhisperX
boundary selection, gap inclusion/exclusion, source projection, and inline audio
placement. Different models can produce different timestamps; the refactor itself
must not change which timestamps the application selects from a given result.

Add `DialogueLine.mark_offsets` and corresponding `DialogueMarkTiming` results
inside `DialogueLineTiming`. Keep `spoken_text` intact for TTS. Alignment owns
matching these authored character boundaries to audio, including missing words;
consumers must not implement a second matcher over backend word records.
`ScriptTiming` remains a collection of dialogue-line timings in `rendering.py`.
Do not add source words or source text to it.

The editor will use radio-drama's TTS, caching, and timing through a programmatically
constructed `ScriptPlan`, with a `ScriptNode` (a diagnostic dummy is acceptable).
Provide a prepared-events initialization path and share the existing composition
setup with document input. Making the planning API ScriptNode-free is not an
objective. The internal backend request still contains audio and text, but it is
not the proposed editor integration API. Actual editor migration, named mark
plans, and within-line effect application remain subsequent work.

Use native Transformers Qwen classes and the `-hf` checkpoints. Do not install,
import, vendor, or call the `qwen_asr` package. Retain WhisperX as an explicitly
selectable backend. Choose exactly one alignment backend per injector/run;
do not auto-select according to installed packages or switch backends on failure.

Final application default: `qwen`. Existing WhisperX users select
`--alignment-backend whisperx`. Make this default switch only after the Qwen
live acceptance checks below pass. Intermediate refactoring commits retain the
WhisperX default so output changes and structural changes can be checked separately.

## 2. Evidence and environment

### Installed versions and checks

| Environment | Python | Transformers | torch | torchaudio | WhisperX |
| --- | --- | --- | --- | --- | --- |
| `~/venv` | 3.14 | 5.14.1 | 2.13.0 | 2.11.0 | not needed for Qwen |
| `~/ai/vibevoice/.venv` | 3.13 | 4.57.3 | 2.8.0 | 2.8.0 | 3.8.6 |

In `~/venv`, imports of `Qwen3ASRForConditionalGeneration`,
`Qwen3ASRForTokenClassification`, and `Qwen3ASRProcessor` succeeded. Import of
torchaudio also succeeded. **No Transformers upgrade is needed.** These checks
did not download weights or establish successful GPU inference. The torch and
torchaudio version mismatch is worth recording, but it did not prevent these
imports; Qwen audio preparation can use the repository's scipy resampler.
An unpatched whole-application import in the sandbox stalled in Carthage's
import-time asyncio call (`carthage/sh.py`), consistent with the documented
sandbox wakeup problem. This is separate from the successful Qwen class imports;
repeat whole-application checks outside the sandbox during implementation.
The qwen-tts in ~/ai/Qwen3-TTS is patched to work with modern transformers and is installed in ~/venv

### Actual backend outputs

WhisperX 3.8.6 was inspected in
`~/ai/vibevoice/.venv/lib/python3.13/site-packages/whisperx/`:

- `transcribe()` returns `{"segments": [...], "language": ...}`; segments have
  `text`, `start`, and `end` (and may have additional fields).
- `align()` returns `{"segments": [...], "word_segments": [...]}`. Aligned
  segments have text/boundaries and nested `words`; word records use `word`,
  `start`, `end`, and optionally confidence fields. Runtime records can omit
  timestamps despite the TypedDict annotations. Keep treating absent boundaries
  as unknown.
- Our existing `WhisperXResponse` additionally records original transcription
  segments, aligned segments, and the shortcut decision. This remains a private
  WhisperX diagnostic/intermediate representation.

Native Qwen APIs were checked against the installed Transformers 5.14.1 source,
especially `models/qwen3_asr/processing_qwen3_asr.py`:

- ASR `generate()` returns token IDs. Decode only the generated suffix with
  `processor.decode(..., return_format="parsed")`; each batch item has
  `language` and `transcription`. It does not supply independently timed clauses.
- `prepare_forced_aligner_inputs(audio=..., transcript=..., language=...)`
  returns `(BatchFeature, word_lists)`.
- `Qwen3ASRForTokenClassification(**inputs)` returns logits.
  `decode_forced_alignment(logits=..., input_ids=..., word_lists=...,
  timestamp_token_id=...)` returns one list per batch item, containing
  `{"text": str, "start_time": float, "end_time": float}` in seconds.
- The processor applies timestamp repair and a configurable time-class scale
  (default 80 ms). These are model estimates, not sample-accurate ground truth.
- Use the processor's returned words as the source-word records; do not assume
  whitespace tokenization, punctuation preservation, or authored character offsets.

References: [native Qwen documentation](https://huggingface.co/docs/transformers/v5.14.0/model_doc/qwen3_asr),
[ASR checkpoint](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf),
[aligner checkpoint](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf).
Installed source is authoritative for 5.14.1 call signatures; newer online
examples already differ in prompt construction. Do not implement against the
legacy wrapper's `ASRTranscription` or `ForcedAlignResult` classes.

The upstream wrapper explicitly chunks audio and adds chunk offsets. Its current
utilities use 180 seconds for alignment and 1200 seconds for ASR. Native model
calls do not inherit that wrapper orchestration. Use our explicit 180-second
window policy below. [Upstream limits](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py),
[upstream orchestration](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py).

## 3. Files and dependency direction

Replace `radio_drama/forced_alignment.py` with this package:

```text
radio_drama/forced_alignment/
    __init__.py       # public re-exports, no heavyweight backend imports
    base.py           # neutral request/result types, injectable resource, queue
    projection.py     # matching, normalization, line/gap/inline marker projection
    planning.py       # AlignedScriptSource, AlignedScriptResult, ScriptSlice
    whisperx.py       # WhisperX loading, ASR, alignment, raw responses, conversion
    qwen.py           # native Transformers ASR/alignment, windows, conversion
```

`rendering.py` continues to own `DialogueLineTiming` and `ScriptTiming`, and owns
the new `DialogueMarkTiming`. Native TTS produces these types independently of
forced alignment, so they do not move into the alignment package. `base.py` owns
backend-neutral `WordTiming` evidence and imports rendering types; rendering must
not import the alignment package. `dialogue.py` owns `DialogueLine.mark_offsets`.
Use TYPE_CHECKING and local imports for ScriptEvent/DialogueLine/projection in
base convenience methods to avoid the existing dialogue/alignment import cycle.
Projection may import the lightweight base types. Planning depends on base and
projection; it never imports a concrete backend.

Preserve public imports of planning and projection functions from
`radio_drama.forced_alignment`. Do not re-export `WhisperXResource`; update callers.
Importing the package or
`radio_drama` must not import `whisperx`, `pyannote`, `qwen_asr`, or Qwen model
classes. Backend modules themselves keep third-party imports inside load methods.
Update private-helper tests to import their owning submodule rather than
maintaining a large private compatibility facade.

## 4. Exact neutral interfaces

Use frozen slotted dataclasses for value objects. The following is the target
shape, with imports omitted. Do not keep `AlignedWord = WordTiming` as a compatibility
alias for existing fixtures and callers; update those fixtures and callers.
Do not combine ABC/abstractmethod with Injectable. Methods that subclasses must
implement raise `NotImplementedError`; base orchestration methods have concrete
implementations. Ellipses below denote those implementations or protocol bodies.

```python
# dialogue.py: append this field to the existing DialogueLine dataclass
mark_offsets: tuple[int, ...] = ()

# rendering.py
@dataclass(frozen=True, slots=True)
class DialogueMarkTiming:
    previous_end: float | None
    next_start: float | None

@dataclass(frozen=True, slots=True)
class DialogueLineTiming:
    start: float                 # retain existing NaN = unknown convention
    end: float
    marks: tuple[DialogueMarkTiming, ...] = ()

@dataclass(frozen=True, slots=True)
class ScriptTiming:
    dialogue_lines: tuple[DialogueLineTiming, ...]

# forced_alignment/base.py
@dataclass(frozen=True, slots=True)
class WordTiming:
    text: str
    start: float | None
    end: float | None

@dataclass(frozen=True, slots=True)
class AlignedClause:
    text: str
    start: float | None
    end: float | None

@dataclass(frozen=True, slots=True)
class AlignmentResult:
    words: tuple[WordTiming, ...] | None
    clauses: tuple[AlignedClause, ...]
    preferred_clauses: tuple[AlignedClause, ...] = ()
    source_text: str = ""
    language: str | None = None
    estimated: bool = False

@dataclass(frozen=True, slots=True)
class ForcedAlignmentRequest:
    audio: np.ndarray
    sample_rate: int
    transcript: str
    transcript_kind: Literal["complete", "partial"]
    require_word_alignment: bool = False
    language: str = "en"

@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    language: str | None = None

class RegisteredForcedAlignmentRequest(Protocol):
    async def align(self) -> AlignmentResult: ...

class ForcedAlignmentResource(AsyncInjectable):
    @property
    def alignment_identity(self) -> str:
        raise NotImplementedError

    @property
    def transcription_identity(self) -> str:
        raise NotImplementedError

    async def register_request(
        self, request: ForcedAlignmentRequest,
    ) -> RegisteredForcedAlignmentRequest: ...

    async def script_timing(
        self, contents: Sequence[ScriptEvent], result: RenderResult, *,
        sample_rate: int,
        transcript_kind: Literal["complete", "partial"],
        require_word_alignment: bool = False,
        language: str = "en",
    ) -> ScriptTiming: ...

    async def transcribe(
        self, audio: np.ndarray, sample_rate: int, *, language: str = "en",
    ) -> TranscriptionResult: ...

    def transcribe_sync(
        self, audio: np.ndarray, sample_rate: int, *, language: str = "en",
    ) -> TranscriptionResult:
        raise NotImplementedError

    async def _process_batch(
        self, requests: Sequence[ForcedAlignmentRequest],
    ) -> list[AlignmentResult]:
        raise NotImplementedError
```

### Contracts that must be documented

- Audio is float PCM, shape `(frames,)` or `(frames, channels)` in the existing
  repository convention. Backend preparation averages channels and resamples a
  copy. Every time is seconds relative to the supplied audio's start, including
  after window merging. Never expose model frames or chunk-local times.
- `transcript_kind="complete"` means the supplied text covers the spoken content
  of this exact waveform, in order. It permits avoiding ASR; it does not guarantee
  a TTS model actually pronounced every word. `partial` means authored excerpts
  may omit speech; the backend must recover the spoken transcript with ASR before
  alignment. Never force the partial text across the entire recording.
- The current TTS render request includes all `DialogueLine`s, including ignored
  and recording-sourced lines: inspect `ScriptRenderRequest.dialogue_lines` and
  existing backend behavior before changing this. The TTS timing projection is
  also the whole script. Preserve it, and pass `complete` for this waveform.
  Recording uses `recording_projection()` and always passes `partial`, even when
  it happens to contain all spoken text. Do not infer completeness from gaps.
- `preferred_clauses` replaces `transcription_clauses` in neutral types. It means
  independently measured clause boundaries to try before `clauses`. For WhisperX
  these are original ASR segments. `clauses` are its aligned segments. Qwen emits
  both as empty tuples. Do not turn arbitrary Qwen windows, full-audio envelopes,
  or punctuation-split text into authoritative clauses.
- Preserve the existing matcher order: validated exact preferred clauses,
  validated exact other clauses, then word matching and the existing permitted
  positional boundary rules. Existing validation and fallback semantics remain
  unchanged during the refactor. In particular, do not replace valid clauses with
  word-derived ends merely because words are present.
- `words is None` means no word alignment was performed/available. `words == ()`
  means alignment was attempted and returned no words. Missing individual word
  boundaries stay `None`; no interpolation in neutral conversion. Synthetic
  heuristic results have `estimated=True` and must not expose invented measured
  words to downstream consumers (`words=None`).
- `require_word_alignment` means attempt word alignment and retain its records,
  bypassing clause-only shortcuts. It cannot promise a valid timestamp for every
  word. Rename current `require_word_timings` to this name everywhere in this repo.
  `script_timing` computes `explicit_requirement OR any(ScriptGap) OR any(line.mark_offsets)`;
  gap mode does not change the requirement. Requiring words for all requested
  marks initially is deliberate: clause validation may fail even when a mark
  appears to coincide with a clause boundary. Valid clauses still win. Keep the
  explicit resource argument for future uses.
- `ScriptTiming.dialogue_lines` remains parallel to the supplied dialogue lines,
  including unknown lines. Each line's `marks` is parallel to that input line's
  `mark_offsets` after timing is ensured. `DialogueLineTiming(start, end)` and
  `ScriptTiming(lines)` remain valid constructors. Native TTS may initially leave
  marks empty; `ensure_timing()` is responsible for satisfying the requested
  offsets. Each mark side uses None for unknown or absent evidence. Existing line
  start/end NaNs retain their established meaning.
- Word text is backend tokenized text, not an authored substring or character
  range. Words and `source_text` stay on `AlignmentResult` as evidence for the
  common matcher and diagnostics. Associate authored fragments with source words
  by text matching, never by containment in clause time spans. Radio-drama and
  editor consumers receive mark timings and perform no acoustic text matching.
  The editor maps original-text marks forward through its filtering to offsets
  in the final spoken text, and retains the inverse application association.
- Initially retain the application's English matching semantics. A language
  parameter permits backend reuse; do not claim multilingual script projection
  until `_TOKEN_RE` and numeric normalization have been addressed separately.
- Models and backend payload shapes are implementation details. No generic
  consumer switches on backend name, raw response decision, or model class.

### Resource lifecycle

Move the current registration queue into base, storing neutral requests and
futures of `AlignmentResult`. Preserve lazy execution, request/result ordering,
and reuse of the result when `.align()` is awaited again. `_process_batch` returns
one result per request (`zip(..., strict=True)` at the boundary). Propagate errors
to waiting registrations; do not convert a failed model call into an empty result.
Shield the shared future from cancellation of an individual waiter and explicitly
cancel/resolve queued work on close; add focused lifecycle tests.

Backend batching/concurrency stays internal. Keep WhisperX's bounded executor and
lazy ASR/alignment loading initially. Serialize Qwen inference per resource with
a threading lock shared by ASR, alignment, and synchronous voice-reference calls;
run it off the event loop. Start Qwen model batch size at one, while preserving
the neutral queue's ability to accept several requests. This bounds activation
memory and does not require both model families in one process. Continue using
`shared_model_load` for process-wide model-load serialization.

Base `script_timing()` returns `ScriptTiming(())` immediately when there are no
dialogue lines. Otherwise it constructs the request, awaits alignment, projects
line spans and requested mark sides using shared pure helpers. Base async
`transcribe()` calls `transcribe_sync()` via `asyncio.to_thread`.

Keep existing `transcribe_audio_sample[_sync]` as thin file/array convenience
wrappers returning `.text`, implemented in base with soundfile and explicit sample
rate for arrays. They must delegate to the selected backend. Remove
`fill_start_positions()` and update its callers/tests to `script_timing()` plus
pure timing projection; do not retain a deprecated resource wrapper or legacy
ScriptPlan fallback.

## 5. WhisperX adapter

Move load/preparation/inference/debug methods into `whisperx.py`. Change only the
outgoing boundary: `_process_batch()` converts each private `WhisperXResponse`
to `AlignmentResult` before fulfilling the registration.
Set `source_text` to the joined recognized segment text (not the potentially
partial authored request); carry the ASR language where available. The heuristic
fallback uses the request transcript as source text and marks it estimated.

Preserve all three decisions and their current boundary behavior:

1. `transcription_exact_clause_match`: original transcription clauses in
   `clauses`, `preferred_clauses=()`, `words=None`.
2. `aligned_exact_clause_match`: aligned clauses in `clauses`, original segments
   in `preferred_clauses`, `words=None`.
3. `aligned_word_matching`: same two clause sources plus nested aligned words.

When words are required, skip both shortcuts even if text matches. A text match
at the shortcut is weaker than successful timestamp validation downstream;
retaining words recovers boundaries when those clause timestamps are rejected.
Add a regression case proving this observable effect, not just an align-call
count. Keep the test proving valid original clause times win over word times.

Keep WhisperX's current ASR-first path even for `complete` requests in this
migration. Its independent clause boundaries are valuable, and skipping ASR
would intentionally change their quality. The neutral completeness contract
permits skipping ASR; it does not require every backend to do so. Direct
transcript-only WhisperX alignment is possible but is a separate behavior change.

Preserve the existing heuristic fallback on the same narrow ImportError path,
mark it `estimated=True`, and log the fallback clearly. Do not broaden fallback
to CUDA errors, invalid payloads, or arbitrary exceptions. Qwen does not inherit
this fallback. Keep backend raw debug artifacts under the existing `whisperx`
category and directory.

## 6. Qwen adapter

Default models:

```text
ASR:     Qwen/Qwen3-ASR-1.7B-hf
Aligner: Qwen/Qwen3-ForcedAligner-0.6B-hf
```

Use `AutoProcessor`, `Qwen3ASRForConditionalGeneration`, and
`Qwen3ASRForTokenClassification` from Transformers. Load each processor/model
only when needed. Use the configured device explicitly, model `.eval()`, and
`torch.inference_mode()`. Use float32 on CPU and bfloat16 on capable CUDA devices
(float16 otherwise). Do not require FlashAttention, torch.compile, vLLM, remote
code, or the legacy Qwen package.

For arrays, downmix/resample to 16 kHz with the shared scipy helper; pass sampling
rate through the installed processor API. Never pass a 48 kHz numpy array and
rely on the processor to guess its rate. Input IDs must remain integer tensors;
use BatchFeature's supported device/dtype transfer rather than casting all values.

For a complete transcript of at most 180 seconds, do not load/run ASR:

```python
inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
    audio=mono_16k, transcript=request.transcript, language=request.language,
    sampling_rate=16000,
)
inputs = inputs.to(aligner.device, aligner.dtype)
with torch.inference_mode():
    output = aligner(**inputs)
items = aligner_processor.decode_forced_alignment(
    logits=output.logits,
    input_ids=inputs["input_ids"],
    word_lists=word_lists,
    timestamp_token_id=aligner.config.timestamp_token_id,
)[0]
```

Convert `text/start_time/end_time` directly to `WordTiming` in seconds.
`AlignmentResult` has no clauses, `source_text=request.transcript`, and
`estimated=False`. Preserve raw decoded output in Qwen diagnostics, including
the fact that the processor repairs timestamps. Reject nonfinite/out-of-audio
model boundaries as unknown rather than presenting them as usable cuts; allow
one timestamp quantum of rounding at the audio end and clamp only that overhang.
Record any rejected boundaries in debug output. Do not globally sort words by
timestamp, which would destroy transcript order.

For partial transcripts, first transcribe the complete audio/window:

```python
inputs = asr_processor.apply_transcription_request(
    audio=mono_16k, language=request.language, sampling_rate=16000,
).to(asr.device, asr.dtype)
with torch.inference_mode():
    output_ids = asr.generate(**inputs, do_sample=False, max_new_tokens=8192)
generated = output_ids[:, inputs["input_ids"].shape[1]:]
parsed = asr_processor.decode(generated, return_format="parsed")[0]
spoken_text = parsed["transcription"]
```

Then align `spoken_text`, not the partial authored transcript. Match authored
lines against the resulting whole-source word stream in the common projection.
ASR language output is metadata; use the explicitly requested alignment language
and let unsupported-language errors be clear. An empty recognized transcript
returns an empty word sequence without calling the aligner. Detect token-budget
exhaustion without EOS and raise an actionable inference error; never silently
align a truncated transcript. The 8192 token budget is a policy constant included
in cache identity, not a limit copied from a short documentation example.

### Long audio policy (implement, do not silently truncate)

Use a private Qwen audio-window helper for ASR/alignment. Each window is at most
180 seconds. For longer input choose the cut at the lowest-RMS 20 ms frame in
the final five seconds of the proposed window (ties choose the latest frame).
Windows are contiguous, nonoverlapping, and cover the original sample range
exactly. For the final short window use its actual length. This is a deterministic
initial policy, not a claim that a quiet cut is always a sentence boundary.

For partial input: transcribe and align each window independently, add its start
offset to every returned time, concatenate words in window order, and join
recognized texts with a newline. Do not fabricate clause spans at window edges.
Standalone ASR uses the same window helper but does not load the aligner.

For complete input longer than 180 seconds: use the same ASR-then-alignment window
path and finally match against the authored full transcript. Log that ASR was
needed to associate text with windows. Do not split the supplied text in proportion
to audio duration or send the entire transcript to each window. Thus bypassing ASR
is an optimization for bounded complete requests, not an unconditional promise.
The editor can create bounded ScriptPlans to obtain complete TTS chunks; it does
not need to supply its own synthesized audio or call alignment directly. If later
native TTS chunk boundaries are exposed, use those in a
separate optimization.

Chunk-edge recognition can change/miss a word; this is an explicit Qwen limitation
to test with speech crossing a window. Do not hide it by substituting WhisperX
or uniform estimated timing. If acceptance tests show unacceptable cuts, improve
the Qwen window algorithm before making it the default.

## 7. Common script projection and preserved output

In `AlignedScriptSource.render_node()`:

- ScriptPlan provider: call its registered request's `ensure_timing()`.
- Other provider with native `ScriptRenderResult.timing`: use that timing when
  it covers the requested offsets. Otherwise obtain alignment and merge requested
  marks while preserving native line spans, using the same enrichment helper as
  the TTS cache. Current SoundPlan recordings have no native timing.
- Otherwise: resolve `ForcedAlignmentResource`, call `script_timing` with the
  actual rendered sample rate and `transcript_kind="partial"`.
- All branches call `fill_start_positions_from_timing` exactly once. Keep debug
  logging, marker calculation, and slicing after this common step.

Remove the legacy getattr-based fallback and reconstructed end-time calculation
from `ScriptPlan.ensure_timing()`. The RegisteredTtsRequest protocol is mandatory;
update external-style test doubles accordingly. Native timings and aligned
timings already carry ends; never reconstruct ends from next starts.

The pure projection functions are concrete backend-independent functions, not
abstract methods to be overridden. Move them with their matching helpers largely
unchanged. Adapt field names (`preferred_clauses`, optional `words`) at the edges.
Retain `fill_start_positions_from_alignment()` for direct pure-helper callers;
both entry points must share line-span and marker logic.

Keep these observable rules and prove them with deterministic fixtures:

- Excluding a middle recording gap cuts at the preceding known speech end and
  resumes at the next selected line start. Including it extends a normal recorded
  run to the next recorded line boundary/end according to `slice_end_marker_index`.
- Inclusion is recording-specific; do not apply that extension to TTS. Preserve
  existing special/ignored line handling and source-transition projections.
- Without a gap, run endings generally follow the next selected marker, not the
  previous speech end; pauses can remain in selected audio.
- A plain `DialogueAudio` between lines is inserted at the midpoint of the
  preceding stabilized end and following stabilized start, preserving the source
  audio on both sides. Gap adjacency continues using existing merged gap spans.
- Unknown dialogue boundaries remain NaN until the existing marker fallback;
  inline audio continues using stabilized spans. Source marker zero and final
  duration, frame rounding/clamping, leading/trailing material, and the whole-
  source optimization remain unchanged.

For a fixed WhisperX result, the old direct path and the new timing path must
produce identical marker frames and selected audio samples, not merely roughly
similar line starts.

### Requested character boundaries within a line

`mark_offsets` contains Python string boundary positions in the exact final
`spoken_text`: `0 <= offset <= len(spoken_text)`. Positions count Unicode code
points, not UTF-8 bytes, UTF-16 units, or normalized-token indexes. Preserve caller
order and duplicates in returned marks. Internally sort/deduplicate offsets for
matching, then restore the requested order. There are no names in these value
types; a future mark plan owns naming. Validate invalid offsets at the planning
boundary with useful document errors (or ValueError for pure projection callers).

Resolve a boundary to two sides. For `spoken_text="Hello there"` and offset 6,
`previous_end` is the end of the matched "Hello" and `next_start` is the start
of the matched "there". A pause can make these different. At offset 0 the
previous side is absent; at len(text) the next side is absent. Leading/trailing
whitespace and punctuation have no independently measurable duration. Boundaries
within the same inter-token whitespace/punctuation region use the same adjacent
lexical sides. An offset inside an indivisible normalized lexical token has no
supported subword timing: return both sides None, not interpolation or snapping.
This allows offsets from filters without silently moving their meaning; finer
subword evidence can be added later without changing the result type.

Refactor matching inside `projection.py` to resolve authored boundaries, including
within-line marks, with the existing clause/word matching machinery:

**The semantic unit of matching remains the complete DialogueLine.** An internal
mark has the same two-sided meaning as a boundary between dialogue lines: the
preceding authored speech's end and the following authored speech's start, with
the same clause preference, word fallback, and unknown-boundary rules. Sharing
those semantics and algorithms does not mean running the line matcher separately
on every mark-delimited fragment. Short fragments lose the context, anchors, and
scoring evidence that allow the complete line to match successfully.

Implement two stages with a one-way dependency: first select and retain each
whole-line match and its start/end; then resolve internal marks using evidence
associated with that match. Mark failure is local to that mark or side. It must
never invalidate, narrow, rescore, or replace a successful whole-line match, or
alter the search position used to match subsequent lines.

1. Preserve original character spans while performing punctuation normalization,
   numeric token expansion, and tokenization. Expanded tokens retain their source
   character span. Do not normalize text and then treat normalized offsets as
   offsets in spoken_text. Retain source word indexes for normalized audio tokens.
2. Establish line matches in their existing order with their existing rules.
   Keep a correspondence between authored tokens and source tokens as part of
   that match. Extend the existing matching helpers to return correspondence,
   including unmatched tokens, rather than build a separate consumer matcher.
   Retain the selected source-token range, score/tie-break outcome, and line
   boundaries before inspecting marks. Preserve the current candidate selection,
   minimum evidence, and acceptance thresholds. Adding a correspondence/backtrace
   to a successful word match must not change which candidate wins. Preserve
   current line start/end results for both marked and unmarked requests given
   the same alignment evidence.
3. Conceptually expand each line at requested boundaries into authored fragments
   for boundary lookup only. Do not create new DialogueLines, split a TTS request,
   or independently search each tiny fragment across the whole recording. Use
   the enclosing line match and monotone token correspondence to disambiguate
   repeated words. When a clause supplied line geometry, obtain correspondence
   using the complete line and its selected textual occurrence. Restrict this
   refinement by source text/token correspondence, not strict containment in
   clause timestamps: valid clause times and word times can disagree. If the
   refinement fails, keep the clause match and mark the unresolved internal sides
   unknown. Never change the main matcher's cursors/earliest time during refinement.
4. For each side of a requested boundary, use a validated exact clause boundary
   when it corresponds to that authored boundary; otherwise use the adjacent
   matched word's boundary. Reuse the existing preferred-clause ordering and
   timestamp validation. Valid original clause boundaries remain authoritative.
   Do not take the containing whole clause's end for a boundary inside it.
5. When an immediately adjacent authored token has no match or usable boundary,
   that side is None. Do not jump across omitted authored words to a distant
   matched token or uniformly distribute time. The other side can still resolve.
   No match for the line yields unknown internal marks. Backend estimates (for
   example a Qwen timestamp for a supplied but unspoken word) are not proof the
   word was spoken; preserve this limitation in documentation.
6. Return one `DialogueMarkTiming` per requested offset, even if both sides are
   unknown. Keep raw mark sides independent of the legacy marker stabilization
   used for slicing. Do not clamp them to independently measured line spans.

The enclosing-line match and clause precedence must not change just because a
consumer adds many marks. With fixed AlignmentResult evidence, require exact
equality of line starts/ends (including unknown status), subsequent line matches,
existing event marker frames, and selected source samples for every choice of
mark offsets. Successful marks are also independent of other requested offsets:
adding another mark must not change a previously resolved mark. Do not require a
short fragment to independently pass the whole-line scoring threshold to read a
boundary from an accepted full-line correspondence.

A forced-alignment shortcut may previously have omitted
words; requesting marks can recover previously unknown line boundaries, but
must not replace valid clause boundaries. Tests must distinguish this recovery
from accidental segmentation changes by separating fixed-evidence projection
tests from backend shortcut/enrichment tests. Internal mark failure must never
erase an already successful line boundary in either path.

Consumers choose a side according to their operation: start highlighting/effects
at next_start, end an effect at previous_end, or use their midpoint for insertion
when both exist. The consumer decides what to do when its needed side is unknown;
it does not run text matching. Dense editor marks use exactly the same resolver
as sparse radio-drama transitions. Naming and final composition-time remapping
belong to consuming plans, not DialogueLine or the aligner.

### Programmatic ScriptPlan and shared composition

Retain ScriptNode as document/diagnostic context. Add the following preparation
path to `ScriptPlan`; do not make the editor reconstruct an XML document or call
TtsResource and ForcedAlignmentResource independently:

```python
class ScriptPlan(AudioPlan):
    def __init__(
        self, node: ScriptNode, *,
        script_events: Sequence[ScriptEvent] | None = None,
        tts: str | None = None,
        **kwargs,
    ): ...

    async def compose(self, *, attrs: Mapping[str, AudioAttrValue]) -> AudioPlan:
        ...

    async def ensure_timing(self, contents, result) -> ScriptTiming:
        ...
```

`script_events=None` requests the existing speaker-map lookup and node parsing;
an explicitly supplied sequence (including empty) supplies prepared events with
resolved SpeakerVoiceReferences. Copy the sequence, validate marks, and do not
overwrite it in async_ready. Skip speaker-map lookup for prepared events. Both
paths run the same speaker collection, semantic request creation, and registration.
Store TTS selection on the plan (`tts` argument if supplied, otherwise node.tts),
and use that value in registration/errors. A prepared TTS-only plan needs the
node's normal diagnostic interface; it need not fabricate element children for
text parsing. Prepared recording events may still use the existing recording
declaration on node; removing that dependency is outside this change.

Move the needs_source_slicing/build-aligned-plan decision into `compose(attrs=)`.
Make `from_node()` handle document parsing/attribute ownership and delegate to
that shared method. The prepared-events caller constructs the plan with `attrs={}`
and calls compose with desired outer attributes if it needs composition. Preserve
the existing once-only application of outer attributes and declaration behavior.
Special-line attributes and recording declarations still use their nodes.

For ordinary editor TTS, construct the ScriptPlan via the injector, render it,
and call `ensure_timing(plan.script_events, result)`. Marks alone must not make
needs_source_slicing true. The returned times are relative to that script's dry
source audio. A later effect/composition plan that changes duration is responsible
for translating source times to final-output times; do not promise that dry-source
marks automatically follow arbitrary effects.

The editor maps positions forward through its text filter and puts offsets on
the prepared DialogueLines. Radio-drama supplies TTS backend selection, reference
voices, audio caching, alignment, and missing-word handling. Its integration tests
should exercise this path without an editor dependency or actual editor migration.

## 8. Mark requests, existing consumers, and persistence

Keep the existing signatures of `RegisteredTtsRequest.ensure_timing`,
`CachedTtsRequest.ensure_timing`, and `ScriptPlan.ensure_timing`; requested marks
are carried by the supplied contents. Do not add a consumer-facing word-request
flag. The resource's explicit require_word_alignment flag remains available for
backend orchestration. A gap forces words when forced alignment is needed;
native line spans remain sufficient for existing gap cuts without marks.

Native line-only timing does not satisfy requested mark offsets. Obtain alignment
when necessary and attach resolved marks while preserving native line starts and
ends exactly. Line endpoints at offsets 0/len(text) use the authoritative native
start/end for their present side. Internal marks still use alignment evidence.
Native marks, if supplied in future, must be parallel to the request offsets;
current local/proxy producers leave them empty. A tuple of unresolved mark results
is a completed attempt, so unknown boundaries do not cause endless recomputation.

| Existing producer/consumer | Required adaptation |
| --- | --- |
| `rendering.py` types | Add DialogueMarkTiming and defaulted DialogueLineTiming.marks; ScriptTiming and its dialogue_lines field stay in place. |
| `DialogueLine` and its copies | Append mark_offsets; preserve it through copy_dialogue_contents, recording_projection, and any reconstruction. Validate against spoken_text. |
| Local Qwen/VibeVoice TTS | Continue receiving the same complete spoken_text; existing line timing constructors remain valid. No marks in prompts or synthesized text. |
| `ScriptRenderRequest` audio serialization/hash | Exclude mark_offsets explicitly; verify existing stems/hashes remain identical, including after serialization changes. |
| Proxy TTS and container protocol | Continue sending intact text and receiving dialogue_line_spans. Keep marks on host events; no protocol change required. |
| `ScriptPlan` | Support prepared events and shared compose setup; delegate timing through the mandatory registration interface. |
| `CachedTtsRequest` | Check mark-offset identity and result coverage before returning native/forced timing; enrich native spans without replacing them. |
| `AlignedScriptSource` | Use shared script_timing/projection; retain ScriptTiming on AlignedScriptResult so mark plans can read results instead of losing them after start_pos projection. |
| `fill_start_positions_from_timing`, ScriptSlice | Continue reading line starts/ends for current gap/inline slicing. Internal marks do not add event markers or change slicing automatically. |
| `radio_drama/testing.py` TTS doubles | Default missing marks to empty when replaying old native fixtures; extend serializers for fixtures that explicitly supply marks. Do not discard marks in a new-format round trip. |
| Forced-alignment replay and helpers | Replay AlignmentResult evidence; execute real mark projection. Replace fill_start_positions resource calls. |
| Editor / future named mark plans | Supply offsets and read corresponding mark sides. Retain application names/positions externally; no second matcher. |

Add `timing: ScriptTiming` to `AlignedScriptResult` alongside its existing
render_result, marker_frames, and contents. This describes the source-local
projection and carries marks; it does not change ScriptSlice's sample selection.
Existing consumers of its other fields retain their behavior; update all result
constructors and test doubles to supply timing.

Keep audio cache filenames and WAV files unchanged. Add these optional metadata
fields alongside existing `dialogue_line_spans` and `alignment_key`:

```json
{
  "timing_format_version": 2,
  "dialogue_mark_offsets": [[6], []],
  "dialogue_mark_timings": [[{"previous_end": 0.48, "next_start": 0.62}], []],
  "mark_alignment_key": "..."
}
```

The two outer lists are parallel to dialogue lines; each inner timing list is
parallel to that line's offsets. Encode unknown mark sides as JSON null. Missing
mark metadata means marks have not been resolved; present matching metadata with
null sides is a completed attempt. Retain compatibility reading old line-only
metadata; do not infer missing ends from legacy start-position fixture caches.
Use shared ScriptTiming serialization helpers for production/native-TTS test
metadata, and a separate AlignmentResult serializer for raw alignment replay.
Existing line NaNs must still round-trip; avoid coupling this change to a global
rewrite of old JSON NaN handling.

Forced line and mark keys include: projection/version hash; actual audio identity;
backend/model/revision identity; language; transcript kind; word requirement;
normalization/projection version; backend window/decoding policy version. Add
these by structured hashing, not ad hoc substring checks. Identity properties
must be available without loading models. Native line keys remain independent of
alignment backend. Include each line's exact ordered mark_offsets in mark identity
and in any forced-line key whose word requirement changes because of marks.
Mark identity is separate because native line timing and forced marks can coexist.
A backend switch invalidates forced timing/marks only; adding, removing, reordering,
or densifying marks never synthesizes or rewrites audio. Mark changes can rerun
alignment initially; do not add a new production raw-evidence cache in this work.
Remove stale marks when satisfying a different offset request, while preserving
native spans. A no-mark request returns empty marks, not unlabelled stale results.
Ensure returning from `ensure_timing()` supplies the current immutable timing;
do not expect a previously rendered result's timing reference to update itself.

Update voice-reference transcription cache namespaces to include
`transcription_identity` as well as VOICE_PREPROCESS_VERSION. Preserve explicit
speaker transcripts as authoritative. Legacy unnamespaced transcripts may be
reused for WhisperX only; Qwen must not silently inherit WhisperX output. Audit
Qwen prompt-feature caching so its key reflects changed reference transcripts
or includes this transcription identity; otherwise the new ASR could appear to
have no effect on voice prompts.

Production recorded alignment remains uncached. Test replay caches are separate
from production caching.

## 9. Selection, packaging, and diagnostics

Add `ProductionConfig.alignment_backend: str = "qwen"` (switch at final gate),
`alignment_language: str = "en"`, `qwen_asr_model`, and `qwen_alignment_model`
with the defaults above. Add matching CLI options in `cli_utils.py` so CLI and
REPL initialization share them. Initial model revision is the configured
checkpoint's default revision; include configured revision values if revision
options are exposed, and bump backend policy identity when bundled behavior
changes. Model identity is not the TTS backend name.

`radio_drama_injector()` first preserves an existing `ForcedAlignmentResource`
provider. Otherwise choose only the configured implementation and register it
under that base key. Avoid duplicate base/concrete instances. Unknown backend
names are configuration errors. Update every production consumer, including
`VoiceReferenceTranscriptionResource`, to inject the base key. Rename its
`whisperx_resource` member to `alignment_resource`; synchronous voice-reference
calls use the base synchronous transcription API directly, with no nested event
loop. Test provider override preservation.

In `pyproject.toml`, move `whisperx` and `qwen-tts` out of mandatory dependencies:

```toml
[project.optional-dependencies]
alignment-whisperx = ["whisperx"]
alignment-qwen = ["transformers>=5.14.1,<6", "torch", "accelerate"]
tts-qwen = ["qwen-tts"]
```

Declare directly used common numerical/audio dependencies explicitly rather than
depending on their accidental installation via WhisperX. Keep exact compatible
torch/torchaudio stacks in documented environment setup, not one contradictory
global constraint. Use the user's patched `~/ai/Qwen3-TTS`, already installed in
`~/venv`, for modern local Qwen TTS; document that source in environment setup.
Do not replace it with the older distribution that pins Transformers 4.57.3 or
claim local Qwen TTS requires a proxy. Check the patched project's metadata when
documenting a reproducible install of the two extras together. Proxy TTS remains
an available choice. Do not disable dependency checks or upgrade the working
WhisperX environment as part of this migration.

Keep third-party imports lazy across test collection, public re-exports, and
injector creation. Missing selected backend dependencies should give a clear
installation message naming the selected extra; do not eagerly check unused
engines. Specifically test importing the common package without either optional
alignment library and running cache replay in either environment.

Retain the neutral `forced_alignment` debug category. Add `qwen_alignment` for
raw Qwen transcript, window offsets, decoded words, model IDs, and rejected
boundaries, with sidecar cleanup in `debug.py`. Preserve `whisperx` diagnostics.
Never write tensors or complete logits into normal debug JSON.

## 10. Test migration and acceptance

### Test boundaries

Replace `CachedWhisperXResource` with a neutral `CachedForcedAlignmentResource`
that composes an optional live backend. Cache/replay `AlignmentResult` at the
registration boundary so the real shared `script_timing` and projection execute
in both modes. Cache keys include audio digest, sample rate, transcript, kind,
word requirement, language, backend identity, and fixture format version. Keep
mark offsets out of this raw-result key: they affect projection, not backend
evidence, except through the derived word requirement. The production timing key
does include offsets. Keep
live/cache mode semantics: missing live entries are generated; missing optional
cache entries skip with a precise message. Required regression fixtures must be
checked in and fail if missing, so migration cannot turn coverage into skips.

The old fixture format contains only event starts and cannot recover line ends
or word data. Keep those files as legacy expectations where useful, but record
new neutral results or derive them from saved raw WhisperX responses. Do not
invent end times from starts. Store new entries under backend/version namespaces.
Use raw saved Qwen outputs for adapter conversion tests and neutral saved results
for shared integration tests; neither needs a live model during collection.

Provide a parameterized `alignment_backend` fixture:

- Offline contract/replay tests use both `whisperx` and `qwen` fixture data without
  importing either third-party implementation.
- Add `--alignment-backend={whisperx,qwen}` to pytest for live runs, default
  WhisperX in the existing test environment. A live fixture loads only this one
  selected backend. Do not parameterize live inference across both environments
  within one process. Use `--run-live` plus existing forced-alignment mode flags.
- Replace concrete-key overrides in dialogue, production, training, phase1,
  voice-reference, and cache tests with `ForcedAlignmentResource` doubles. Update
  TTS registrations to implement the protocol rather than keep legacy fallbacks.
- Move WhisperX-specific tests/private raw response tests into a dedicated test
  module; keep neutral matcher/projection tests independent of installed models.
  Update `scripts/build_forced_alignment_case.py` and public exports as well.

### Required cases

1. WhisperX behavior preservation: saved fighter conversation, numeric expansion,
   repetitions, absent lines/NaNs, original-clause preference, exact shortcuts,
   and queue ordering/batching. Compare old/new marker frames and sliced waveform
   samples with the same saved response before removing the old method body.
2. Failed clause validation: text matches at the shortcut, but a clause is missing
   an end (or has invalid chronological order); aligned words provide a valid
   end. With a gap, assert the recovered end changes the actual excluded interval
   and inline marker as expected. A companion case proves valid clauses win even
   when word times differ substantially. Cover both shortcut decisions.
3. Source semantics: recording/TTS transitions, ignored/special lines, leading,
   middle, trailing and consecutive gaps, both gap modes, inline audio before/after
   gaps, and source beginning/end. Verify selected samples using a ramp waveform
   and distinctive inserted audio, not only start_pos values.
4. Qwen complete path never calls ASR for a bounded request. Partial path aligns
   the recognized full transcript, including an unauthored aside. Words are kept
   even when the caller does not require them because Qwen naturally produces them.
5. Native Qwen payload conversion: generated suffix decoding, parsed language,
   nested batch output, start/end units, processor word segmentation, unsupported
   language, missing/invalid times, empty recognition, and token-limit failure.
6. Windows: complete audio coverage, deterministic cuts, no duplicated word
   records, offset addition, a line spanning two windows, >180-second complete
   request using ASR fallback, empty final window prevention, and short audio.
7. Native TTS spans survive mark enrichment unchanged. Existing no-mark line-only
   cache hits still work; marked requests cannot return a line-only cache hit.
   Completed unknown marks do not trigger repeated alignment. Backend/offset
   changes rerun timing without rerendering audio; mark sides/None/NaNs round-trip.
8. Voice-reference transcription uses the selected resource for sync and async
   paths, cache identities differ, explicit transcripts bypass ASR, and a modern
   environment does not import WhisperX/pyannote through this dependency.
9. Import/DI isolation, shared registration cancellation, shutdown, error
   propagation, and no model loads on timing/audio cache hits.
10. Sparse and dense offsets: both mark sides around a pause, line endpoints,
    duplicate/unsorted offsets, Unicode character positions, punctuation and
    whitespace, numeric expansion, repeated words, missing words on either side,
    unknown whole lines, and an offset inside a token. Clause preference applies
    at real clause boundaries and never supplies a whole-clause end for an
    internal boundary. Mark density does not change valid line geometry.
11. Prepared ScriptPlan: explicit empty/nonempty events bypass parsing and speaker
    map lookup, preserve offsets, select the requested TTS backend, and yield
    render/timing through the shared registration. Dummy ScriptNode errors remain
    useful. Document and prepared paths share composition with outer attributes
    applied once; marks alone do not trigger slicing.
12. Adding/reordering/removing offsets preserves exact TTS prompts, proxy payloads,
    semantic audio hashes and WAV paths. Changed offsets do not reuse stale marks.
    Old native test fixtures load with empty marks; new fixture serializers and
    AlignedScriptResult retain them. Simulate editor filtering by supplying mapped
    offsets and verify timing without any matcher in the simulated consumer.

### Specific regressions for whole-line matching versus internal marks

Implement these as deterministic pure-projection tests with saved or constructed
AlignmentResult evidence. Run the same fixtures with no marks, sparse marks,
and a boundary at every word; compare line geometry exactly, treating NaNs as
equal unknowns. These tests must not require a live backend.

1. **Full-line context survives short fragments.** Use an authored line such as
   "Please put the red folder beside the blue folder on the table", with repeated
   "the"/"folder" in earlier unrelated source speech. Place marks around a single
   "the" and the two "folder" occurrences. Establish that the unmarked full line
   selects the intended later occurrence. Assert marked variants select the same
   line start/end and token occurrence; the two folder marks refer to their
   respective words, never an earlier standalone match. Include a fixture where
   a single-word fragment cannot pass the existing matcher independently, while
   the whole line does; assert marks can still use its accepted correspondence.
2. **Missing inner word does not destroy a successful line.** Start from a long
   word-matching fixture that passes the current score with one interior authored
   word omitted in source speech. Assert that baseline success as a prerequisite
   of the fixture. Put marks immediately before and after the missing word. The
   side adjacent to the missing word is None; the other side uses its matched
   neighbor. Assert the whole line's original start/end survive unchanged. Add
   a mark at a well-matched boundary elsewhere and assert it still resolves.
3. **Clause success with unusable inner word data.** Give the complete line an
   exact preferred clause span, and supply empty word evidence or words with
   missing internal timestamps. Assert the full line retains the clause start/end
   although inner marks are unknown. In a companion case provide usable words
   whose times disagree with that clause: only the internal sides use word data;
   the preferred line boundaries stay intact and words are not rejected solely
   for lying outside the clause's measured interval.
4. **A failed mark cannot disturb subsequent lines.** Follow each of the preceding
   fixtures with another line containing repeated words from the first. Compare
   both line spans, gap boundaries, and an adjacent DialogueAudio midpoint against
   the unmarked run. Compare marker frames and ramp-waveform slices exactly.
   This catches refinement that accidentally advances shared search cursors or
   changes earliest-time constraints.
5. **Boundary semantics agree where splitting is unambiguous.** Construct two
   sufficiently long, uniquely matching clauses separated by a known pause:
   first ends at 2.0 seconds, second starts at 2.6. In one request they are two
   lines; in another they are one line with a mark at their textual boundary.
   Assert the inner mark is `(previous_end=2.0, next_start=2.6)`, equal to the
   two-line end/start pair. Repeat with word-only evidence, and with one boundary
   unknown. This is a boundary-evidence comparison, not a claim that arbitrary
   short-line segmentation has equivalent matching scores or slice geometry.
6. **Adding marks is observationally independent.** For a fixed matched line,
   compare a single mark with the same mark among dense, reordered, and duplicated
   offsets. Its sides are identical and output order follows input order. Include
   an unresolvable inside-token offset beside valid marks; it must not degrade
   those marks or the line match. Repeat through mark-cache serialization/replay.
7. **More backend evidence does not replace valid clause geometry.** Exercise
   the real WhisperX adapter with fake model payloads: unmarked text takes each
   clause-only shortcut; marked text forces words. Choose payloads with identical
   valid clause boundaries and conflicting word times, then verify line spans
   remain identical. Separately use a rejected/missing clause boundary where
   forcing words recovers a previously unknown side. That recovery is permitted;
   a successful line becoming unmatched because an inner mark fails is not.

Use existing scoring rules to select fixtures and assert their baseline result;
do not weaken the scoring thresholds to make short-fragment tests pass. Assert
observable boundaries and selected audio, not just method-call counts. Preserve
existing unmarked saved-response regressions as an independent baseline.

### Execution environments

Run the full non-live suite in the existing environment first, escalated as
AGENTS.md recommends:

```sh
~/ai/vibevoice/.venv/bin/python -m pytest -q
```

Run common/offline alignment and cache tests under `~/venv` as well. If sandbox
asyncio hangs, escalate or use `codex_python_runner.py`; do not modify production
code/tests to compensate for the sandbox. A clean modern-environment import is
an acceptance gate, including unrelated eager imports reached through
`radio_drama.__init__`.

For implementation, run separate opt-in live alignment suites with checked-in
speech fixtures:

```sh
~/ai/vibevoice/.venv/bin/python -m pytest tests/test_forced_alignment_live.py \
    --run-live --alignment-backend whisperx --forced-alignment-mode live -q
~/venv/bin/python -m pytest tests/test_forced_alignment_live.py \
    --run-live --alignment-backend qwen --forced-alignment-mode live -q
```

Record Qwen replay fixtures only after inspecting transcripts/times and listening
to the resulting excluded-gap and inline-audio cuts. Check finite ordered
boundaries within duration, nonempty expected words, the complete path's ASR
bypass, and a recording containing omitted speech. Do not require Qwen to
numerically match WhisperX or use the existing 0.9-second tolerance as proof that
a cut does not clip a word. Use hand-audited boundary ranges for the cut cases.
Include a >180-second case with speech near a window boundary. If model weights
are unavailable, explicitly report the live gate as pending; do not claim the
migration complete or switch the default on mocked evidence alone.

## 11. Implementation order and completion checklist

1. Extract the package and neutral result boundary; retain WhisperX behavior and
   default. Move helpers with minimal edits, update imports/fixtures, and verify
   saved-response marker/sample equivalence.
2. Route all production users through the base resource and `script_timing`;
   remove ScriptPlan's legacy fallback. Migrate fixture replay to neutral results.
3. Add mark_offsets, DialogueMarkTiming, character-preserving normalization and
   shared boundary matching. Add the prepared ScriptPlan path and common compose
   setup. Adapt consumers as listed above; keep native line spans and TTS text.
   Add mark-aware persistence and voice-reference cache separation; verify audio
   identity preservation and old timing metadata migration.
4. Add Qwen native Transformers adapter, explicit full/partial transcript policy,
   window handling, mock adapter tests, and model/dependency isolation.
5. Add configuration/CLI options, optional dependency groups, backend replay
   fixtures, and separate live execution selection. Run the two environment gates.
6. Inspect live cuts and replay results; switch the default to Qwen only on passing
   evidence. Update installation and CLI documentation and commit the work.

Update `architecture.md` alongside each implementation stage: neutral requests
and results, authored mark offsets versus source evidence, native/forced mark
enrichment, prepared ScriptPlan construction, shared composition, cache
identity, lazy selected backend, ASR completeness policy, and replay boundaries.
Put queue details, shortcut validation rationale, window policy, and adapter
decoding mechanics in docstrings. Effects are not modified by this work; if a
later change adds within-line preset behavior, update `docs/effects.md` then.

Completion requires one selected backend serving both alignment and reference
ASR; unchanged WhisperX cuts for saved responses; usable Qwen full and partial
paths; sparse/dense authored marks resolved by the common matcher and cached
without regenerating speech; prepared ScriptPlan use with ScriptNode context;
no optional-backend import during offline tests; and passing independent live
gates. Migrating the editor to this TTS/timing interface, its forward filter-offset
mapping, named mark plans, and effect automation are subsequent work. Leave this
planning document untracked and uncommitted as requested; the implementation's
commit instructions above apply when that work is undertaken.
