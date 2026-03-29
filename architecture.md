# Goal

Create an app that turns a human-edited XML production document into a radio drama, while keeping the core interfaces reusable as additional models, scene structure, effects, and alignment features are added.

# Architectural assumptions

* Python
* async orchestration with thread or process offload where model APIs or performance require it
* numpy arrays as the in-process audio representation
* Carthage dependency injection for shared resources and non-local state
* model-facing resources should stay narrow enough that they can eventually move into separate processes or interpreters
* document-facing APIs should prefer strong user-facing errors for invalid input, while programmer misuse can continue to surface as ordinary Python exceptions

# Current architecture

## Document layer

The XML document is parsed into a semantic tree of `DocumentNode` objects. These nodes are plain objects, not injectables.

Current document contract:

* the root is `<production>`
* document nodes retain XML attributes in `DocumentNode.attributes`
* concrete element classes declare a class-level `tag_name`
* child placement is driven by class-level contexts rather than by per-instance child-tag dictionaries
* `<production>` may contain zero or one `<speaker-map>` elements plus any element permitted in the audio-plan context
* `<speaker-map>` content is YAML mapping speaker names to voice references
* `<script>` content is speaker-authored dialogue text
* `<line speaker="...">...</line>` may appear inside a script to author one explicit dialogue line without reparsing its text content as `speaker: ...`
* any audio-producing element may set audio attributes such as `pre_gap`, `post_gap`, `length`, `gain`, `pan`, and `preset`
* `<script>` accepts any element permitted in the audio-plan context, including nested `<script>` and `<sound>`
* `<sound>` identifies a named sound either as `<sound ref="door" />` or `<sound>door</sound>`
* `<mark>` identifies a named zero-duration cut point either as `<mark id="chapter2" />` or `<mark>chapter2</mark>`
* relative `<sound>` references are resolved under a configured sounds directory when one is supplied, otherwise under a `sounds/` tree next to the source XML document
* a script may contain stanza continuation lines and paragraph breaks
* an empty script is valid

The document layer is responsible for:

* preserving source locations for user-facing errors
* enforcing document structure
* exposing semantic nodes that know how to plan themselves into planning objects
* normalizing document-authored sound references even before sound planning exists
* normalizing element values that may be authored either in text content or in one distinguished attribute
* maintaining the context registry that turns “this node is permitted in audio-plan contexts” and “this node accepts audio-plan contexts” into `allowed_child_tags`

The document layer is not responsible for model loading, batching, or resource ownership.

## Planning layer

Planning turns semantic document nodes into injectable `PlanningNode` objects.

Current planning contract:

* every semantic node that participates in production planning exposes `plan(ainjector)`
* all concrete plans are `AsyncInjectable`
* every plan has a `render()` path, even if rendering is a no-op
* plans retain the source document node that produced them
* `render()` is memoized per plan instance so duplicate callers share work
* `AudioPlan` is the central base type for plans whose `render()` returns `RenderResult`
* document-authored audio attributes are currently `pre_gap`, `post_gap`, `length`, `gain`, `pan`, and `preset`
* any element that plans into an `AudioPlan` may author those audio attributes
* `AudioPlan.attrs_from_node(node)` is a class method that parses one node's audio attributes into typed values, stores them in `self.attrs`, and `process_attrs()` then applies those typed attrs to the instance
* `process_attrs()` is responsible for cross-attribute validation such as rejecting `length` and `post_gap` on the same node
* `pan` is validated as an expression at planning time but evaluated only at render time, because it may depend on render-time audio mark positions
* `preset` is handled through `AudioPlan.async_resolve()`, which may replace one plan with a wrapping `PresetPlan`
* every document node's audio attributes must be consumed by the outermost `AudioPlan` produced from that node; inner plans produced from the same node therefore receive `attrs={}`
* every `AudioPlan` may apply node-level `gain` and `pan` during `post_render()`
* every `AudioPlan` carries plan-level timing fields: `pre_margin`, `post_margin`, `pre_gap`, `post_gap`, and optional `length`
* `pre_margin` and `post_margin` remain render-time plan fields rather than document-authored attributes
* productions are planned by walking element children in document order; speaker maps participate as ordinary planning nodes and audio-producing children are collected into the top-level `ProductionPlan`

Current plan types:

* `SpeakerMapPlan`: validates and resolves speaker names to voice references
  once ready, it registers itself in the production injector so later scripts can find it
* `ScriptPlan`: parses dialogue stanzas, normalizes a script-level render request, and registers that request with the shared speech resource during `async_ready()`
  `ScriptPlan.contents` is an ordered list of `DialogueContents` objects
  `DialogueLine` holds spoken text plus a handling mode such as `normal`, `ignore`, or `special`
  `DialogueLine.node` may point back to the originating document node when later planning needs a stable node boundary
  `DialogueAudio` wraps an inner `AudioPlan` such as `SoundPlan`
* `SoundPlan`: resolves one sound asset during planning and lazily starts cached normalization work during render so cut-away plans do not launch unused background sound work
* `MarkPlan`: renders zero frames of silence and introduces one named audio mark into plan composition
* `AlignedScriptSource`: a non-`AudioPlan` planning node that renders the dry `ScriptPlan`, runs forced alignment, and returns an `AlignedScriptResult` containing the dry `RenderResult`, aligned `DialogueContents`, and content-boundary marker frames used by script-local slicing
* `ScriptSlice`: an `AudioPlan` that slices an `AlignedScriptSource` result between two marker indexes
* `SlicePlan`: renders a time slice of an already-rendered `RenderResult`
* `ComposeAudioPlan`: renders child `AudioPlan`s concurrently into one shared timeline, mixing overlaps and advancing by either explicit `length` or natural rendered span
* `PresetPlan`: wraps another `AudioPlan`, resolves a named `EffectChain` at render time, and applies it to that plan's `RenderResult`
  when it wraps a plain `ComposeAudioPlan`, it may use `async_resolve()` to rewrite itself into a replacement `ComposeAudioPlan` before readiness so preset bubbling stays a planning concern rather than a render-time special case
* `ProductionPlan`: the top-level `ComposeAudioPlan`, preserving child order across all production-level audio nodes
  `ProductionPlan.attrs_from_node()` also injects the implicit outer `master` preset so mastering remains part of ordinary audio-attribute resolution

Current mark/cut contract:

* every `AudioPlan` exposes `audio_marks`, the set of unambiguous named cut points bubbled up from its immediate inner plans
* `MarkPlan` is the only leaf plan that introduces a new mark
* container plans bubble marks upward while suppressing any mark that becomes ambiguous among sibling plans
* `cut_before_mark(mark_id)` mutates a plan in place so later rendering begins at that mark when the mark remains unambiguous through the container path
* `cut_after_mark(mark_id)` mutates a plan in place so later rendering stops at that mark under the same ambiguity rules
* `PresetPlan` passes mark bubbling and cutting through to the wrapped audio plan, so top-level production cuts can target marks inside nested script composition

Planning rule for presets:

* `ScriptNode.plan()` remains the public entry point, but `ScriptPlan.from_node()` performs most script-specific plan construction
* a plain script produces a `ScriptPlan`
* if a script contains `DialogueAudio` or any non-`normal` `DialogueLine`, planning constructs one shared `AlignedScriptSource` plus a `ComposeAudioPlan` of `ScriptSlice` plans and any inline audio plans that survive script-local filtering
* marker indexes are assigned during `ScriptPlan.from_node()` and refer to boundaries in the original `ScriptPlan.contents`, not to absolute times
* `<ignore>` content is rendered as part of the dry script request but omitted from the composed output by slicing around its content boundaries
* a `<line>` with no audio attrs is just another `normal` `DialogueLine` and merges into adjacent normal dialogue slices
* a `<line>` whose `ScriptSlice.attrs_from_node(line_node)` result is non-empty becomes a `special` `DialogueLine`; aligned planning then emits a dedicated `ScriptSlice` for only that line, using the line node as the slice node so its audio attrs are consumed by that outermost slice
* if the same script also has audio attributes, those attrs are attached to the outermost audio plan for that script: either the plain `ScriptPlan`, or the composed aligned plan when alignment is needed
* if that outermost plan carries a `preset`, `AudioPlan.async_resolve()` wraps it in a `PresetPlan`
* if that wrapped plan already contains inner `PresetPlan` children from nested scripts, `PresetPlan.async_resolve()` splits the outer preset around only the inner presets whose document node does not set `stack_preset=true`
* in that bubbling case, the replacement outer `ComposeAudioPlan` keeps the non-`preset` audio attrs from the bubbled preset node, while each surviving outer preset segment is rebuilt with `attrs={}` so those attrs still belong to the returned outermost plan
* each replacement outer preset segment covers the largest contiguous slice available on its side of those non-stacking inner presets so DSP state is preserved across ordinary gaps and stacked presets
* `stack_preset` is currently a document-authored boolean attribute interpreted on the inner preset node; missing means false
* higher-level production planning therefore deals in `AudioPlan` rather than bare `ScriptPlan`
* a script resolves its `SpeakerMapPlan` from the production injector at planning time and raises a document error if no speaker map has been planned
* a script may select its speech backend with `tts="vibevoice"` or `tts="qwen"`; the default is `vibevoice`
* the top-level production render is also treated as an `AudioPlan` and is mastered through the named `master` preset, which stacks outside any inner script presets

`radio_drama_injector()` is the standard way to create an injector for radio-drama planning and rendering. It installs shared production-scoped resources while preserving caller overrides from a parent injector. When callers supply an `output_path` and do not override `InjectionKey("cache_dir")`, it also provides a production-scoped VibeVoice cache directory derived from that output path.

## Resource layer

Resources own model lifecycle, batching, and other shared external state.

Current resource contract:

* `VibeVoiceResource` accepts script-level `ScriptRenderRequest` objects
* the VibeVoice-specific resource implementation lives in `radio_drama.vibevoice`
* `ScriptRenderRequest` carries ordered `DialogueLine` objects plus a short leading-text label for cache/debug artifacts
* `VibeVoiceResource` derives its speaker-numbered normalized script and ordered voice-sample list internally from those dialogue lines
* `QwenTtsResource` accepts the same `ScriptRenderRequest` objects and renders scripts by cloning each `DialogueLine` speaker voice line-by-line before concatenating one script result
* requests are registered during planning and may remain pending until some caller renders one of them
* rendering any registered request may drain additional queued requests in the same batch
* resource output is returned in the configured production sample rate and channel layout
* when `InjectionKey("cache_dir")` is present, `VibeVoiceResource` persists model-native WAV output plus adjacent JSON metadata keyed by the semantic render request, and touches cache mtimes whenever cached output is reused
* `WhisperXResource` accepts forced-alignment requests at render time and drains them through one shared ASR model plus a bounded alignment executor
* WhisperX ASR and alignment models are loaded lazily and only when a request path actually needs them
* heavyweight lazy model loads are serialized process-wide across resource types; generation and alignment work may still run concurrently after startup
* `WhisperXResource` also exposes direct single-sample ASR so other resources can derive metadata such as Qwen voice-clone prompt transcripts from reference voice files
* WhisperX-specific raw responses stay at the resource boundary; conversion into model-independent `AlignmentResult` objects happens in pure helper logic outside the resource
* `NormalizedSoundCache` owns production-scoped sound normalization tasks so multiple `SoundPlan`s can share one normalized numpy buffer per resolved asset path
* `ProductionConfig` may override both the voice directory and the sounds directory used for document-authored relative asset references
* `ProductionConfig` also carries optional debug categories and a debug log path for render-time instrumentation

The important boundary is that plans create semantic requests and resources fulfill them. Higher-level planning code should not embed model-specific batching or loading mechanics.

## Rendering layer

`RenderResult` is the common audio result type for renderable plans.

Current rendering contract:

* `RenderResult.audio` is a contiguous `float32` numpy array
* current internal render results are already in production format
* `RenderResult` retains gap and margin fields for later composition work, but not explicit `length`
* `ProductionResult` is the top-level rendered output type
* effect processing consumes and returns `RenderResult`, preserving those fields while replacing the audio buffer
* inline sounds currently splice into dialogue at forced-alignment cut points by slicing the rendered speech and composing the inserted sounds into the same timeline

Current production behavior is timeline composition of rendered script results.
Script-level `pre_gap` and `post_gap` values are measured in seconds, may be negative, and affect either placement or trimming depending on where the composed result is consumed.
`length` overrides the natural occupied span of one `AudioPlan` in its parent's composition timeline.
`RenderResult.audio_marks` holds surviving unambiguous render-time mark positions in sample frames.
`MarkPlan` introduces its mark at frame `0`, and `ComposeAudioPlan` rebases child mark positions into the composed render result while dropping duplicates the same way ambiguous marks stop bubbling at plan time.
`pan` expressions are evaluated against those render-time mark locals, coerced to an array-valued expression, clipped to `[-1, 1]`, and then applied as stereo attenuation where the near side stays at full scale and the far side follows a smooth cosine falloff to silence.
The final production result is then passed through the `master` preset.

## Expression layer

Expression support is intentionally narrow.

Current expression contract:

* `eval_expression(text, locals, return_type)` parses one Python expression with `ast.parse(..., mode="eval")`, validates the allowed syntax, evaluates it with no builtins, and then applies `return_type`
* the currently allowed surface is numeric constants, names, unary `+`/`-`, binary operators, list or tuple literals, and direct function calls
* the only current global helper is `line(...)`
* `ArrayExpression` is the abstract base for expressions that expand to one float32 array for a requested frame count
* `LineExpression` builds arrays from piecewise-linear frame/value control points plus an optional virtual end point at the requested output size
* `coerce_array_exp` preserves `ArrayExpression` values and wraps plain numbers as constant `line(number)` expressions

Current debug hooks:

* `compose_audio` logs the time span where each child plan's samples are placed during composition
* `forced_alignment` logs each aligned `DialogueLine` start position plus a short text preview
* `vibevoice_output` writes model-native WAV artifacts for each rendered script to `OUTPUT.wav.vibevoice/`
* `whisperx` logs the alignment decision and writes the raw transcription/alignment segment payloads to `OUTPUT.wav.whisperx/`

## Effects and presets

Preset support is intentionally narrow at the interface boundary and flexible in implementation.

Current effects contract:

* `EffectChain` is a named ordered sequence of stages
* each stage receives stereo production-format numpy audio plus the output sample rate
* stages may be backed by plain Python/numpy, `scipy.signal`, Pedalboard, or FFmpeg
* preset names are resolved at render time, not baked into `ScriptPlan`
* unknown preset names are document errors attached to the originating audio node

Current built-in presets:

* `master`: the production-level mastering pass, currently just FFmpeg `loudnorm`
* `narrator`, `thoughts`: inner-monologue or produced narration variants with center-focused stereo, stronger leveling, and abstract ambience
* `outdoor1`: a lighter open-air variant with extra width and sparse reflections
* `outdoor2`: a deliberately obvious outdoor diagnostic variant with wider stereo, audible noise bed, and a strong echo tail
* `indoor1`, `indoor2`: room-bound variants with stronger early reflections and a slightly more centered image

## Backend preview service

The preset-preview backend is a thin diagnostic layer above the existing planning and effects interfaces.

Current backend contract:

* `python -m radio_drama.backend <production_xml>` renders the production once at startup into an in-memory `RenderResult`
* the backend keeps that base rendered output and prepares named preset variants on demand from the same base render
* the preview backend also exposes a dry `none` option that returns slices from the unprocessed base render
* preset preparation runs concurrently and reuses the same `EffectChain` interface as document-driven render-time presets
* audio slice requests address a prepared preset plus a playback time, and the backend responds with a WAV stream starting at that point in the production

The backend exists to make preset evaluation easier. It should stay narrow and should not grow a second planning or rendering path separate from the main Python interfaces.

## Testing architecture

Default `pytest` runs should stay fast and should not require the live model.

Current testing contract:

* live tests are marked `live` and run only with `pytest --run-live`
* default test runs skip live tests
* cache-aware testing sits at model/resource boundaries rather than inside plans
* each live resource may have a cache-aware pytest substitute that preserves the same public contract
* cache metadata is structural rather than waveform-based
* plans and higher-level composition code should be testable against either the real resource or its cache-backed substitute without changing plan logic

Current cache/live modes:

* `live`: if metadata is missing, run the real resource, persist structural metadata, and return either a synthetic structural replay or the real structural result, depending on the resource contract under test
* `cache`: if metadata is missing, skip the test

Current cache-backed resources follow the same broad pattern:

* `CachedVibeVoiceResource` sits at the `VibeVoiceResource` boundary
  it persists enough metadata to replay the production-facing render contract without rerunning the speech model
* `CachedQwenTtsResource` sits at the `QwenTtsResource` boundary
  it persists enough metadata to replay the production-facing render contract without rerunning the Qwen speech model
* `CachedWhisperXResource` sits at the `WhisperXResource` boundary
  it persists enough metadata to replay filled `DialogueContents.start_pos` values without rerunning forced alignment

For the current implementation, cached metadata is resource-specific:

* VibeVoice cache metadata includes model-native sample rate and frame count
* Qwen TTS cache metadata includes model-native sample rate and frame count
* WhisperX cache metadata includes the ordered `start_pos` values written onto `DialogueContents`

This keeps tests focused on structural behavior such as batching, ordering, output-format conversion, and alignment cut points rather than exact waveform reproduction.

# Future plans

## Sound layout and automation

The current render-time mark model is intentionally local. It is useful for
cutting and for simple post-render automation on one already-bounded
`RenderResult`, but it is not sufficient for whole-production staging where
automation should be able to refer to marks later in the production and where
long-running sounds may continue across several scripts.

The intended direction is to keep the existing `AudioPlan` tree, keep
automation expression-based, and add an explicit layout pass inside that tree.
The layout pass is responsible for computing timing and mark positions before
final DSP/effects are applied.

### Terminology

The layout proposal uses these terms:

* `length`: how much a plan advances the composition cursor
* `start`: the point at which an item is placed in composition cursor space,
  just after its `pre_gap`
* `first`: where the first sample of an item is placed
* `last`: where the last sample of an item is placed
* `end`: the last position in composition cursor space that the item occupies

The intended invariant is:

* `end - start == length`

`post_gap` therefore affects `length`, not just rendered samples. In the common
case, `first == start`. If a plan carries trailing silence through `post_gap`,
then `last < end`. If a plan later gains render-time overhang concepts such as
negative pre-roll or tail spill, those should affect `first` and `last` without
changing `start` or `end`.

### One layout pass, not a separate whole-graph discovery pass

This proposal does not require a separate "discover everything in the whole
graph first" phase.

Instead:

* layout is one bottom-up pass over the existing plan tree
* leaf plans may materialize their primitive render artifacts during layout
* container plans may lay out children in parallel
* the result of layout is enough timing information to drive final rendering

So the useful separation is:

* layout determines timing, local marks, and clip placement
* render consumes the saved layout result and produces final audio

But leaf work such as TTS, forced alignment, and sound sizing may happen during
layout itself rather than in a distinct pre-pass.

### Proposed layout contract

Every `AudioPlan.layout()` call should set two intrinsic layout facts on the
node itself, both expressed in the node's inner geometry:

* `natural_length`
* `audio_marks_inner`

`natural_length` is the total unclipped natural render/control span of the
node.
`audio_marks_inner` is a dict of visible marks rebased into that same inner
geometry.

This is the core contract that every `AudioPlan` must satisfy, including
leaves.

`ComposeAudioPlan` does more than that. As part of placing children into the
parent's geometry, it also sets these concrete outer-geometry values on each
child:

* `start`
* `end`
* `length`

Those child values are not intrinsic facts about the child. They are placement
facts computed by the parent.

`ComposeAudioPlan` then satisfies the ordinary `AudioPlan` contract for itself
by:

* computing its own `natural_length` as the span of its inner render/control
  extent, effectively `inner_last - inner_first`
* bubbling and rebasing child marks into its own `audio_marks_inner`

This means child layout can run in parallel without knowing production-absolute
coordinates. A child only computes intrinsic inner-geometry facts. The parent
is responsible for turning those into outer-geometry placement.

### Leaf materialization during layout

Leaf plans should be allowed to do the expensive work they need in order to
produce stable layout facts.

That likely means:

* `SoundPlan` may resolve and size its source audio during layout
* `ScriptPlan` may materialize the speech primitive during layout
* script-local forced alignment may also happen during layout, because it is
  required to place script-local marks and slices

If useful for implementation clarity, leaves may expose a helper such as
`render_primitive()` or `layout_primitive()`, but the architectural point is
that this is leaf work in support of layout, not a separate whole-production
discovery stage.

### Container layout

`ComposeAudioPlan` remains the fundamental hierarchical composition structure.
Its layout path would:

1. lay out children concurrently
2. compute each child's `start`, `end`, and `length` in outer geometry
3. advance the parent cursor by each child's `length`
4. rebase child marks into parent-local inner geometry
5. suppress duplicate marks the same way mark bubbling already does
6. compute the parent's own `inner_first` and `inner_last`
7. compute the parent's own inner render/control extent and therefore its
   `natural_length`
8. compute the parent's `audio_marks_inner`

For a composed parent, those inner bounds are derived from all children after
placement, not only from children that produce nonzero samples:

* `child_inner_first_in_parent = child.start + child.inner_first`
* `child_inner_last_in_parent = child.start + child.inner_last`
* `parent.inner_first = min(child_inner_first_in_parent over all children)`
* `parent.inner_last = max(child_inner_last_in_parent over all children)`

This is intentional. Silence introduced by child placement, child `post_gap`,
or internal marks still counts toward the parent's natural render/control
extent.

This keeps the existing tree semantics, but turns timing into an explicit stored
result instead of something that is only implicit in render-time mixing.

### Layout scope and expression evaluation

Expressions remain the preferred automation surface.

The important change is that expressions such as `pan` should evaluate against
marks from a stable layout scope rather than against only the marks already
present in one rendered child `RenderResult`.

The default long-term layout scope is the full production, still subject to the
existing bubbling rule:

* a mark is referable only if it remains unambiguous through the relevant
  container path
* duplicate marks do not bubble

Nothing in this proposal requires inline dialogue anchors. The working authoring
surface remains `script + mark + line`.

### Positioning expressions

The first version of layout expressions should stay narrow and should use
explicit mark namespaces.

Positioning attributes are:

* `start`
* `end`
* `length`
* `pre_gap`
* `post_gap`

The main constraints are:

* at most one of `start` or `pre_gap` may be authored
* at most one of `end`, `length`, or `post_gap` may be authored
* `end` and `start` are both interpreted in outer geometry
* `natural_length` is not an authored attribute

Mark references in positioning expressions must always be prefixed:

* `inner_<mark>` refers to a visible mark produced within the node after the
  node's contents have been laid out, expressed in inner geometry
* `outer_<mark>` refers to a visible mark in outer geometry

Unprefixed mark names are simply not populated as expression variables. They
should therefore fall out as ordinary expression-evaluation `NameError`s rather
than receiving special-case validation.

For the first version, only marks are exposed from the node's inner geometry.
The following inner values are intentionally hidden:

* `inner_length`
* `inner_first`
* `inner_last`

Those values may become useful later, but in the initial design they mostly add
alternate spellings for relationships that are already expressible in outer
geometry and make the dependency model harder to explain.

The evaluation model is intentionally phased:

1. layout the node's contents and determine visible `inner_<mark>` values
2. evaluate the node's left-side positioning (`start` or `pre_gap`) in outer
   geometry
3. determine the node's render/control extent in outer geometry
4. evaluate the node's right-side positioning (`end`, `length`, or `post_gap`)

This phased model is what makes expressions such as `end=last` well-defined
without turning right-side positioning into a recursive dependency on itself.

After layout, the canonical placement fields are:

* `start`
* `end`
* `length`

`pre_gap` and `post_gap` remain authored inputs, but are no longer used for
composition once layout has resolved concrete placement.

### Rendering from layout

Once layout has run, render becomes a top-down application of the saved layout.

At render time:

* `ProductionPlan.render()` is the only render entry point that may omit
  `incoming_marks`
* all other render paths conceptually require `incoming_marks` in outer
  geometry, even if the implementation keeps a public zero-argument `render()`
  wrapper for memoization convenience
* leaves reuse or finalize the primitive artifacts prepared during layout
* containers mix child audio at the already-decided child `start` positions
* effects, presets, gain, pan, and future automation run after placement is
  known
* automation expressions such as `pan` and future gain automation operate in
  natural sample geometry, where `0 == inner_first`
* clipping or slicing during render is an implementation detail layered on top
  of that natural sample geometry
* `render(incoming_marks=...)` receives marks in outer geometry
* each node rebases those marks into inner geometry and merges them into
  `audio_marks_inner`, filling only names that are not already present locally
* each node then derives `audio_marks_render` by rebasing the resulting
  `audio_marks_inner` through `inner_first`, so render/control geometry has
  `inner_first == 0`
* expression locals for automation come from `audio_marks_render`

The key invariant is:

* once a node's layout result has been computed, later rendering does not move
  it

Without that invariant, later mark references and long-running sounds become
recursive.

For automation expressions, inner geometry means:

* `natural_length` is the node's total unclipped natural render/control span
* any rebased mark position may be less than `0`
* any rebased mark position may be greater than `natural_length`

That is expected. Automation expressions should be able to refer to marks that
fall outside the node's natural render/control span, and the array-building
helpers used by those expressions should clip or truncate accordingly rather
than rejecting such positions.

Under the current one-shot render contract, mutating `audio_marks_inner` during
render is acceptable. `render()` is memoized per plan instance, so the node is
not expected to support multiple distinct render contexts. If that ever changes,
the render contract would need to become explicit about multi-render behavior
rather than relying on node-local mutation.

### Loops

Looping sounds are feasible within this model only under explicit convergence
constraints.

The working assumptions are:

* loop structure is internal to the looping sound
  A loop may have an intro segment, a looping middle segment, and an outro
  segment, but those segment boundaries are not determined by external marks.
* a looping node must have a computable `length`
  That `length` determines cursor advance and cannot depend on later layout
  results.
* the rendered extent of a loop may continue until a later visible mark
* that later stop condition must not move sibling placement
* external expressions may refer to authored marks that happen to bound the
  loop, but not to loop iteration count or other expanded loop internals
* samples produced beyond the resolved bounds of the enclosing container may be
  discarded instead of forcing the container to grow

These constraints intentionally separate:

* composition occupancy, described by `start`, `end`, and `length`
* rendered sample support, described by `first` and `last`

That separation is what keeps expression-based automation compatible with loops
that may render beyond their nominal cursor advance.

### Why this is still simpler than a separate constraint graph

This proposal does not require a separate global graph or a general constraint
solver.

The existing plan tree remains the composition structure. The layout pass makes
that structure explicit, parallelizable, and stable enough for whole-production
automation, while still letting local planning rules such as script slicing and
preset wrapping remain ordinary tree-local concerns.

## Document model growth

The current document schema is intentionally small. Future work may add richer structure above scripts, such as scenes, processors, effects, or asset references. Those additions should extend the semantic node tree rather than introducing a separate global planner.

## Resource growth

The current resource layer is centered on VibeVoice. Future model integrations should follow the same broad shape:

* semantic request objects created by plans
* shared resources that own model lifecycle and batching
* rendered results returned in production format

## Rendering growth

The current renderer already composes clips on a shared timeline using `RenderResult` gaps and applies presets through `AudioPlan` resolution. Future rendering work may add:

* non-zero gap and margin handling
* overlapping or mixed clips
* scene transitions
* production-level effects and mastering passes
* alignment-aware composition

## Testing growth

The cache-backed resource tests are the basis for longer-term model-backed testing. Future resources should follow the same shape: keep the live implementation narrow, add a cache-aware substitute at the same boundary, and persist only the structural outputs that higher layers depend on. Future cache metadata will likely grow to include structural outputs such as:

* margins and gaps
* alignment points
* other model-derived timing metadata

As those features appear, tests should continue to prefer structural metadata over waveform snapshots.
