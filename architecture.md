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
* `<group ...>...</group>` may appear inside a script to parse its text as ordinary dialogue stanzas while assigning one shared script-slice node and one shared set of audio attrs to the resulting dialogue lines
* any audio-producing element may set audio attributes such as `pre_gap`, `post_gap`, `length`, `gain`, `pan`, `preset`, `first_mark`, `last_mark`, and looping attrs such as `loop_beg`, `loop_end`, `loop_loops`, `loop_until`, `loop_silence`, `loop_outro`, and `loop_whole`
* `<script>` accepts any element permitted in the audio-plan context, including nested `<script>` and `<sound>`
* `<sound>` identifies a named sound either as `<sound ref="door" />` or `<sound>door</sound>`
  `from` and `to` are sound-specific trim attributes in source-file time, not general outer audio attrs
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
* every plan may expose `child_plans()` for traversal, and `all_plans()` walks the reachable plan graph once by identity so shared subplans such as one `AlignedScriptSource` are not revisited through multiple `ScriptSlice`s
* `radio_drama.planning` now holds the shared planning substrate:
  * `PlanningNode`
  * typed audio-attribute aliases such as `AudioAttrs`
  * the production-planning injector key used to publish production-scoped plans while planning is still in progress
* `radio_drama.audio` now holds reusable audio-plan primitives and audio-format helpers
* `radio_drama.dialogue` now holds speaker-map resolution, normalized dialogue data types, and script planning
* `radio_drama.production` now holds the top-level production compose plan
* `AudioPlan` is the central base type for plans whose `render()` returns `RenderResult`
* `AudioPlan` also owns layout state and exposes a memoized `layout()` pass before render-time mixing or DSP
* document-authored audio attributes are currently `start`, `end`, `pre_gap`, `post_gap`, `length`, `gain`, `pan`, `preset`, `first_mark`, `last_mark`, `loop_beg`, `loop_end`, `loop_loops`, `loop_until`, `loop_silence`, `loop_outro`, and `loop_whole`
* any element that plans into an `AudioPlan` may author those audio attributes
* `AudioPlan.attrs_from_node(node)` is a class method that parses one node's audio attributes into typed values, stores them in `self.attrs`, and `process_attrs()` then applies those typed attrs to the instance
* `process_attrs()` is responsible for cross-attribute validation such as rejecting `start` with `pre_gap`, and rejecting `length` with `post_gap`
* `pan` is validated as an expression at planning time but evaluated only at render time, because it may depend on render-time audio mark positions
* `preset` is parsed into `AudioPlan.preset_key` metadata and is consumed by the nearest enclosing `ComposeAudioPlan` when that compose builds preset buses during render
* looping is handled through `AudioPlan.async_resolve()`, which may replace one plan with a wrapping `LoopPlan`
* every document node's audio attributes must be consumed by the outermost `AudioPlan` produced from that node; inner plans produced from the same node therefore receive `attrs={}`
* every `AudioPlan.layout()` computes intrinsic layout facts for that node:
  * `inner_first`
  * `inner_last`
  * `natural_length`
  * `audio_marks_inner`
* container plans, especially `ComposeAudioPlan`, additionally compute child placement in outer geometry:
  * `start`
  * `end`
  * `length`
* every `AudioPlan` may receive parent-scope incoming marks after placement and before render; those inherited marks are available to render-time automation and, for plans such as `LoopPlan`, may also influence layout-time expression evaluation
* explicit `start` is special: it establishes the mapping from parent outer time into the child's inner geometry, so it only sees outer-scope values and may not depend on child-local `natural_length` or `inner_*` names
* productions are planned by walking element children in document order; speaker maps participate as ordinary planning nodes and audio-producing children are collected into the top-level `ProductionPlan`

Current plan types:

* `SpeakerMapPlan` in `radio_drama.dialogue`: validates and resolves speaker names to voice references
  once ready, it registers itself in the production injector so later scripts can find it
* `ScriptPlan` in `radio_drama.dialogue`: parses dialogue stanzas, normalizes a script-level render request, and registers that request with the shared speech resource during `async_ready()`
  `ScriptPlan.contents` is an ordered list of `DialogueContents` objects
  `DialogueLine` holds spoken text plus a handling mode such as `normal`, `ignore`, or `special`
  `DialogueLine.node` may point back to the originating document node when later planning needs a stable node boundary
  `DialogueAudio` wraps an inner `AudioPlan` such as `SoundPlan`
* `SoundPlan` in `radio_drama.sound`: resolves one sound asset during planning, optionally trims it with `from` / `to` in source-file time, sizes it during layout, and renders one normalized clip
* `MarkPlan` in `radio_drama.audio`: renders zero frames of silence and introduces one named audio mark into plan composition
* `AlignedScriptSource`: a non-`AudioPlan` planning node that renders the dry `ScriptPlan`, runs forced alignment, and returns an `AlignedScriptResult` containing the dry `RenderResult`, aligned `DialogueContents`, and content-boundary marker frames used by script-local slicing
* `ScriptSlice` in `radio_drama.forced_alignment`: an `AudioPlan` that slices an `AlignedScriptSource` result between two marker indexes
* `SlicePlan` in `radio_drama.audio`: renders a time slice of an already-rendered `RenderResult`
* `ComposeAudioPlan` in `radio_drama.audio`: lays out child `AudioPlan`s concurrently, computes child placement in outer geometry, bubbles and suppresses marks, renders child audio into compose-local preset buses, applies each preset bus once, and then sums the buses into one shared timeline
* `LoopPlan` in `radio_drama.audio`: wraps another `AudioPlan`, evaluates `loop_beg` and `loop_end` in the wrapped plan's inner geometry, evaluates `loop_until` in the loop plan's own inner geometry, may rebase parent-scope incoming marks into that same inner geometry for explicit-start loops, repeats the chosen interval with optional inter-loop silence and optional outro rendering, and suppresses wrapped marks from the repeated region while preserving pre-loop and boundary marks
* `ProductionPlan` in `radio_drama.production`: the top-level `ComposeAudioPlan`, preserving child order across all production-level audio nodes and then applying the special `master` preset to the final trimmed production audio

Current mark/cut contract:

* every `AudioPlan` exposes `audio_marks`, the set of unambiguous named cut points bubbled up from its immediate inner plans plus any local `first_mark` / `last_mark`
* `MarkPlan` introduces one explicit authored mark, while any audio-producing node may also introduce boundary marks through `first_mark` and `last_mark`
* `LoopPlan` suppresses bubbled wrapped marks from the repeated region, preserving only the first instance of boundary marks and any surviving pre-loop or outro marks
* container plans bubble marks upward while suppressing any mark that becomes ambiguous among sibling plans
* `cut_before_mark(mark_id)` mutates a plan in place so later rendering begins at that mark when the mark remains unambiguous through the container path
* `cut_after_mark(mark_id)` mutates a plan in place so later rendering stops at that mark under the same ambiguity rules
Planning rule for presets:

* `ScriptNode.plan()` remains the public entry point, but `ScriptPlan.from_node()` performs most script-specific plan construction
* a plain script produces a `ScriptPlan`
* if a script contains `DialogueAudio` or any non-`normal` `DialogueLine`, planning constructs one shared `AlignedScriptSource` plus a `ComposeAudioPlan` of `ScriptSlice` plans and any inline audio plans that survive script-local filtering
* marker indexes are assigned during `ScriptPlan.from_node()` and refer to boundaries in the original `ScriptPlan.contents`, not to absolute times
* `<ignore>` content is rendered as part of the dry script request but omitted from the composed output by slicing around its content boundaries
* a `<line>` with no audio attrs is just another `normal` `DialogueLine` and merges into adjacent normal dialogue slices
* a `<line>` whose `ScriptSlice.attrs_from_node(line_node)` result is non-empty becomes a `special` `DialogueLine`; aligned planning then emits a dedicated `ScriptSlice` for only that line, using the line node as the slice node so its audio attrs are consumed by that outermost slice
* dialogue text parsed from `<group>` becomes `special` `DialogueLine`s whose `node` is the group node; aligned planning therefore emits one dedicated `ScriptSlice` per resulting grouped line, with the group's audio attrs consumed by each slice
* if the same script also has audio attributes, those attrs are attached to the outermost audio plan for that script: either the plain `ScriptPlan`, or the composed aligned plan when alignment is needed
* `preset` stays on that outermost audio plan as metadata rather than introducing another wrapper plan
* when a `ComposeAudioPlan` renders, each child contributes its rendered audio to either the child's own preset bus or, if the child is otherwise dry, the compose node's own preset bus
* a `ComposeAudioPlan` therefore consumes its own preset locally and returns dry audio upward to its parent compose
* sibling children that land on the same preset bus share DSP state across adjacency and across silence within that compose timeline
* higher-level production planning therefore deals in `AudioPlan` rather than bare `ScriptPlan`
* a script resolves its `SpeakerMapPlan` from the production injector at planning time and raises a document error if no speaker map has been planned
* a script may select its speech backend with `tts="vibevoice"` or `tts="qwen"`; the default is `vibevoice`
* the top-level production render is mastered through the named `master` preset after production trimming
* `ComposeAudioPlan` lays out automatic-start children first, gathers their parent-scope marks, resolves each explicit child's `start`, and then lays out explicit-start children with those marks available as incoming scope

`radio_drama_injector()` is the standard way to create an injector for radio-drama planning and rendering. It installs shared production-scoped resources while preserving caller overrides from a parent injector. When callers supply an `output_path`, it also installs the production-scoped `CacheManager`, which derives the shared speech-cache root from that output path unless the caller overrides `InjectionKey("cache_dir")` directly.

## Resource layer

Resources own model lifecycle, batching, and other shared external state.

Current resource contract:

* `VibeVoiceResource` accepts script-level `ScriptRenderRequest` objects
* the VibeVoice-specific resource implementation lives in `radio_drama.vibevoice`
* `ScriptRenderRequest` carries ordered `DialogueLine` objects plus a short leading-text label for cache/debug artifacts
* `ScriptRenderRequest` is also the shared speech-cache identity: it owns the stable semantic hash, the human-readable first-words label used in filenames, the common hit validator requiring adjacent `wav` and `json` artifacts, and the shared JSON payload builder used by both speech backends
* that script-render cache is intentionally production-facing rather than purely implementation-facing: once the director accepts how a production sounds, the cached `ScriptRenderRequest` artifacts may effectively become part of the accepted production assets, so cache-key stability is allowed to be higher there than in purely derived global caches
* `VibeVoiceResource` derives its speaker-numbered normalized script and ordered voice-sample list internally from those dialogue lines
* `QwenTtsResource` accepts the same `ScriptRenderRequest` objects and renders scripts by cloning each `DialogueLine` speaker voice line-by-line before concatenating one script result
* requests are registered during planning and may remain pending until some caller renders one of them
* rendering any registered request may drain additional queued requests in the same batch
* resource output is returned in the configured production sample rate and channel layout
* `CacheManager` is the production-scoped mapping from cache type names such as `vibevoice` and `qwentts` to `CacheCollection` objects
* `CacheManager` derives the shared cache root from the production `output_path`, while still allowing a direct `InjectionKey("cache_dir")` override for callers that need to place the cache elsewhere
* `CacheCollection` keeps the existing filename scheme abstractly: each artifact stem is `{collection_name}_{sanitized_first_words}_{semantic_hash}`, so VibeVoice cache filenames stay stable while Qwen uses the same contract with a different collection prefix
* both speech resources persist model-native WAV output plus adjacent JSON metadata keyed by the semantic render request, and cache reuse touches artifact mtimes
* cache lookup may run a validator over the discovered subtype-to-path mapping; validation failures delete the stale files before the miss path repopulates the cache
* Qwen also keeps a separate global prompt-feature cache for reference voices; unlike the production script-render cache, that prompt cache is implementation-facing and therefore includes the current reference-voice preprocessing version in its cache key
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
* `ProductionResult` is the top-level rendered output type
* effect processing mutates the `RenderResult.audio` buffer in place and does not carry separate timing metadata in the result object
* inline sounds currently splice into dialogue at forced-alignment cut points by slicing the rendered speech and composing the inserted sounds into the same timeline

Current production behavior is layout-driven timeline composition of rendered script results.

Current layout and render contract:

* `AudioPlan.layout()` computes one node's intrinsic inner-geometry facts:
  * `inner_first`
  * `inner_last`
  * `natural_length`
  * `audio_marks_inner`
* `ComposeAudioPlan.layout()` additionally computes child placement in outer geometry:
  * `start`
  * `end`
  * `length`
* `length` is how much a child advances its parent's composition cursor
* `start` and `end` are in the parent's geometry; `natural_length` is in the node's own natural sample geometry
* `pre_gap` and `post_gap` are authored inputs that contribute to layout, but composition after layout uses resolved `start`, `end`, and `length`
* `incoming_marks(...)` is the interface for parent-scope inherited marks
* each node rebases incoming marks into render-time geometry, preserving local marks over inherited ones
* each node derives `audio_marks_render` by rebasing `audio_marks_inner` through `inner_first`, so render-time automation sees natural sample geometry where `0 == inner_first`
* `pan` expressions are evaluated against those render-time locals, coerced to an array-valued expression, clipped to `[-1, 1]`, and then applied as stereo attenuation where the near side stays at full scale and the far side follows the current linear falloff to silence
* mark bubbling for cutting still lives on `AudioPlan.audio_marks`, the set of surviving unambiguous mark ids visible through the plan tree
* the final production result is then passed through the `master` preset

## Expression layer

Expression support is intentionally narrow.

Current expression contract:

* `eval_expression(text, locals, return_type)` parses one Python expression with `ast.parse(..., mode="eval")`, validates the allowed syntax, evaluates it with no builtins, and then applies `return_type`
* the currently allowed surface is numeric constants, names, unary `+`/`-`, binary operators, and direct function calls
* the only current global helper is `line(...)`
* `ArrayExpression` is the abstract base for expressions that expand to one float32 array for a requested frame count
* `LineExpression` builds arrays from variadic piecewise-linear frame/value control points plus an optional virtual end point at the requested output size
* `LineExpression` accepts control points outside the requested output interval and clips or truncates them when expanding to the requested size
* `coerce_array_exp` preserves `ArrayExpression` values and wraps plain numbers as constant `line(number)` expressions
* `coerce_real` is the scalar companion used by layout-time expression evaluation

Current expression scopes:

* render-time automation expressions such as `gain` and `pan` evaluate against `audio_marks_render`
* render-time marks are exposed in natural sample geometry where `0 == inner_first`
* `natural_length` is available in that same geometry and is expressed in sample frames
* a render-time mark may be negative or greater than `natural_length`
* layout helpers use two mark namespaces:
  * `inner_<mark>` for marks already visible in the node's own inner geometry
  * `outer_<mark>` for marks visible in the containing scope
* loop expressions use three scopes:
  * `loop_beg` and `loop_end` evaluate in the wrapped plan's inner geometry
  * `loop_until` evaluates in the loop plan's own inner geometry
  * for explicit-start loops, parent-scope marks from automatic siblings may be rebased into that same inner geometry and exposed under `inner_<mark>`
  * render-time controls on the loop plan use the loop plan's render geometry like any other plan
* `start` is the exception to the usual left-side scope rules because it defines that inner/outer coordinate transform
  it may therefore use `outer_<mark>` names but not child-local `natural_length` or `inner_*` names
* unprefixed mark names are not populated specially and therefore fail as ordinary `NameError`s if referenced

Current debug hooks:

* `compose_audio` logs the time span where each child plan's samples are placed during composition
* `forced_alignment` logs each aligned `DialogueLine` start position plus a short text preview
* `vibevoice_output` writes model-native WAV artifacts for each rendered script to `OUTPUT.wav.vibevoice/`
* `whisperx` logs the alignment decision and writes the raw transcription/alignment segment payloads to `OUTPUT.wav.whisperx/`

## Effects and presets

Preset support is intentionally narrow at the interface boundary and flexible in implementation.

Current effects contract:

* `EffectStage` is the stable DSP interface; it mutates one production-format numpy buffer in place when given the output sample rate
* stage composition uses `|`, and the result is itself an `EffectStage`
* presets are named `EffectStage` singletons rather than a separate chain wrapper type
* most reusable stage factories are registered in `radio_drama.effects.effect_stages` so later expression-driven effect evaluation can use them as eval locals
* stages may be backed by plain Python/numpy, `scipy.signal`, Pedalboard, or FFmpeg
* render-time automation such as `gain` and `pan` is implemented as ordinary effect stages in `radio_drama.effects`
* preset names are validated while processing audio attrs and are consumed later by `ComposeAudioPlan` render-time bus mixing
* unknown preset names are document errors attached to the originating audio node

Current built-in presets:

* `master`: the production-level mastering pass, currently just FFmpeg `loudnorm`
* `narrator`, `thoughts`: inner-monologue or produced narration variants with center-focused stereo, stronger leveling, and abstract ambience
* `narrator_nofocus`: the `narrator` voicing without the center-focusing mid/side stage, useful when later automation such as `pan` should control image placement
* `outdoor1`: a lighter open-air variant with extra width and sparse reflections
* `outdoor2`: a deliberately obvious outdoor diagnostic variant with wider stereo, audible noise bed, and a strong echo tail
* `indoor1`, `indoor2`: room-bound variants with stronger early reflections and a slightly more centered image
* `phone`: a narrow-band, mid-forward telephone/comm variant with heavier leveling, very narrow stereo, and a small hiss bed

## Backend preview service

The preset-preview backend is a thin diagnostic layer above the existing planning and effects interfaces.

Current backend contract:

* `python -m radio_drama.backend <production_xml>` renders the production once at startup into an in-memory `RenderResult`
* the backend keeps that base rendered output and prepares named preset variants on demand from the same base render
* the preview backend also exposes a dry `none` option that returns slices from the unprocessed base render
* preset preparation runs concurrently and reuses the same `EffectStage` interface as document-driven render-time presets
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

## Document model growth

The current document schema is intentionally small. Future work may add richer structure above scripts, such as scenes, processors, effects, or asset references. Those additions should extend the semantic node tree rather than introducing a separate global planner.

## Resource growth

The current resource layer is centered on VibeVoice. Future model integrations should follow the same broad shape:

* semantic request objects created by plans
* shared resources that own model lifecycle and batching
* rendered results returned in production format

## Rendering growth

The current renderer already composes clips from stored layout state, mixes non-`master` presets through compose-local preset buses, and applies `master` at the production boundary. Future rendering work may add:

* non-zero gap and margin handling
* overlapping or mixed clips
* scene transitions
* production-level effects and mastering passes
* alignment-aware composition
* preset stacks as first-class bus keys rather than one-name preset assignments
* preset continuity across compose boundaries when adjacent composed children share the same preset stack
* more sophisticated bus routing or sidechain-style interactions between sibling presets

## Testing growth

The cache-backed resource tests are the basis for longer-term model-backed testing. Future resources should follow the same shape: keep the live implementation narrow, add a cache-aware substitute at the same boundary, and persist only the structural outputs that higher layers depend on. Future cache metadata will likely grow to include structural outputs such as:

* margins and gaps
* alignment points
* other model-derived timing metadata

As those features appear, tests should continue to prefer structural metadata over waveform snapshots.
