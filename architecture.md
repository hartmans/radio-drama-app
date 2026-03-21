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
* `<production>` contains exactly one `<speaker-map>`
* `<production>` contains one or more `<script>` elements in document order
* document nodes retain XML attributes in `DocumentNode.attributes`
* `<speaker-map>` content is YAML mapping speaker names to voice references
* `<script>` content is speaker-authored dialogue text
* `<script preset="...">` selects a named render-time effect preset
* `<script>` may also contain inline `<sound>` elements
* `<sound>` identifies a named sound either as `<sound ref="door" />` or `<sound>door</sound>`
* relative `<sound>` references are resolved under a `sounds/` tree next to the source XML document
* a script may contain stanza continuation lines and paragraph breaks
* an empty script is valid

The document layer is responsible for:

* preserving source locations for user-facing errors
* enforcing document structure
* exposing semantic nodes that know how to plan themselves into planning objects
* normalizing document-authored sound references even before sound planning exists

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
* every `AudioPlan` carries plan-level timing fields: `pre_margin`, `post_margin`, `pre_gap`, `post_gap`, and optional `length`
* document nodes may set `pre_gap`, `post_gap`, or `length`, but not `pre_margin` or `post_margin`
* `length` and `post_gap` are mutually exclusive on one document node
* `set_gap=False` is used for wrapper plans whose timing must remain owned by an inner audio plan

Current plan types:

* `SpeakerMapPlan`: validates and resolves speaker names to voice references
* `ScriptPlan`: parses dialogue stanzas, normalizes a script-level render request, and registers that request with the shared speech resource during `async_ready()`
  `ScriptPlan.contents` is an ordered list of `DialogueContents` objects
  `DialogueLine` holds spoken text
  `DialogueAudio` wraps an inner `AudioPlan` such as `SoundPlan`
* `SoundPlan`: resolves one sound asset, starts cached normalization work during `async_ready()`, and renders the normalized production-format sound
* `AlignedScriptSource`: a non-`AudioPlan` planning node that renders the dry `ScriptPlan`, runs forced alignment, and returns an `AlignedScriptResult` containing the dry `RenderResult`, aligned `DialogueContents`, and marker frames for inline insertions
* `ScriptSlice`: an `AudioPlan` that slices an `AlignedScriptSource` result between two marker indexes
* `SlicePlan`: renders a time slice of an already-rendered `RenderResult`
* `ComposeAudioPlan`: renders child `AudioPlan`s into one shared timeline, mixing overlaps and advancing by either explicit `length` or natural rendered span
* `PresetPlan`: wraps another `AudioPlan`, resolves a named `EffectChain` at render time, and applies it to that plan's `RenderResult`
* `ProductionPlan`: the top-level `ComposeAudioPlan`, preserving script order after the shared speaker map is ready
  the final production render is then wrapped in a top-level `PresetPlan("master")`

Planning rule for presets:

* `ScriptNode.plan()` remains the public entry point, but `ScriptPlan.from_node()` performs most script-specific plan construction
* a plain script produces a `ScriptPlan`
* if a script contains `DialogueAudio`, planning constructs one shared `AlignedScriptSource` plus a `ComposeAudioPlan` of alternating `ScriptSlice` plans and inline audio plans
* marker indexes are assigned during `ScriptPlan.from_node()` and refer to insertion fenceposts in the original script contents rather than to absolute times
* if the same script also has a preset, `PresetPlan` wraps outside that composed audio plan so the preset still covers the full rendered result
* higher-level production planning therefore deals in `AudioPlan` rather than bare `ScriptPlan`
* the top-level production render is also treated as an `AudioPlan` and is mastered through the named `master` preset

`radio_drama_injector()` is the standard way to create an injector for radio-drama planning and rendering. It installs shared production-scoped resources while preserving caller overrides from a parent injector.

## Resource layer

Resources own model lifecycle, batching, and other shared external state.

Current resource contract:

* `VibeVoiceResource` accepts script-level `ScriptRenderRequest` objects
* requests are registered during planning and may remain pending until some caller renders one of them
* rendering any registered request may drain additional queued requests in the same batch
* resource output is returned in the configured production sample rate and channel layout
* `NormalizedSoundCache` owns production-scoped sound normalization tasks so multiple `SoundPlan`s can share one normalized numpy buffer per resolved asset path

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
The final production result is then passed through the `master` preset.

## Effects and presets

Preset support is intentionally narrow at the interface boundary and flexible in implementation.

Current effects contract:

* `EffectChain` is a named ordered sequence of stages
* each stage receives stereo production-format numpy audio plus the output sample rate
* stages may be backed by plain Python/numpy, `scipy.signal`, Pedalboard, or FFmpeg
* preset names are resolved at render time, not baked into `ScriptPlan`
* unknown preset names are document errors attached to the originating `<script>`

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
* `CachedWhisperXResource` sits at the `WhisperXResource` boundary
  it persists enough metadata to replay filled `DialogueContents.start_pos` values without rerunning forced alignment

For the current implementation, cached metadata is resource-specific:

* VibeVoice cache metadata includes model-native sample rate and frame count
* WhisperX cache metadata includes the ordered `start_pos` values written onto `DialogueContents`

This keeps tests focused on structural behavior such as batching, ordering, output-format conversion, and alignment cut points rather than exact waveform reproduction.

# Future plans

## Document model growth

The current document schema is intentionally small. Future work may add richer structure above scripts, such as scenes, processors, effects, or asset references. Those additions should extend the semantic node tree rather than introducing a separate global planner.

## Resource growth

The current resource layer is centered on VibeVoice. Future model integrations should follow the same broad shape:

* semantic request objects created by plans
* shared resources that own model lifecycle and batching
* rendered results returned in production format

## Rendering growth

The current renderer concatenates clips in order and applies any per-script presets before concatenation. Future rendering work is expected to make fuller use of `RenderResult` metadata and may add:

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
