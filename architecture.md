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
* a script may contain stanza continuation lines and paragraph breaks
* an empty script is valid

The document layer is responsible for:

* preserving source locations for user-facing errors
* enforcing document structure
* exposing semantic nodes that know how to plan themselves into planning objects

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

Current plan types:

* `SpeakerMapPlan`: validates and resolves speaker names to voice references
* `ScriptPlan`: parses dialogue stanzas, normalizes a script-level render request, and registers that request with the shared speech resource during `async_ready()`
* `PresetPlan`: wraps another `AudioPlan`, resolves a named `EffectChain` at render time, and applies it to that plan's `RenderResult`
* `ProductionPlan`: owns the ordered production `audio_plans` and concatenates their rendered audio

Planning rule for presets:

* `ScriptNode.plan()` always creates a `ScriptPlan`
* when `ScriptNode.preset` is present, planning wraps that script plan in a `PresetPlan`
* higher-level production planning therefore deals in `AudioPlan` rather than bare `ScriptPlan`

`radio_drama_injector()` is the standard way to create an injector for radio-drama planning and rendering. It installs shared production-scoped resources while preserving caller overrides from a parent injector.

## Resource layer

Resources own model lifecycle, batching, and other shared external state.

Current resource contract:

* `VibeVoiceResource` accepts script-level `ScriptRenderRequest` objects
* requests are registered during planning and may remain pending until some caller renders one of them
* rendering any registered request may drain additional queued requests in the same batch
* resource output is returned in the configured production sample rate and channel layout

The important boundary is that plans create semantic requests and resources fulfill them. Higher-level planning code should not embed model-specific batching or loading mechanics.

## Rendering layer

`RenderResult` is the common audio result type for renderable plans.

Current rendering contract:

* `RenderResult.audio` is a contiguous `float32` numpy array
* current internal render results are already in production format
* `RenderResult` retains gap and margin fields for later composition work
* `ProductionResult` is the top-level rendered output type
* effect processing consumes and returns `RenderResult`, preserving those fields while replacing the audio buffer

Current production behavior is ordered concatenation of rendered script results.

## Effects and presets

Preset support is intentionally narrow at the interface boundary and flexible in implementation.

Current effects contract:

* `EffectChain` is a named ordered sequence of stages
* each stage receives stereo production-format numpy audio plus the output sample rate
* stages may be backed by plain Python/numpy, `scipy.signal`, Pedalboard, or FFmpeg
* preset names are resolved at render time, not baked into `ScriptPlan`
* unknown preset names are document errors attached to the originating `<script>`

Current built-in presets:

* `narrator1`, `narrator2`: inner-monologue or produced narration variants with center-focused stereo, light leveling, and subtle abstract ambience
* `outdoor1`, `outdoor2`: mostly dry open-air variants with light width and sparse reflections
* `indoor1`, `indoor2`: room-bound variants with stronger early reflections and a slightly more centered image

## Testing architecture

Default `pytest` runs should stay fast and should not require the live model.

Current testing contract:

* live tests are marked `live` and run only with `pytest --run-live`
* default test runs skip live tests
* cache-aware testing sits at the `VibeVoiceResource` boundary
* cache metadata is structural rather than waveform-based

Current cache/live modes:

* `live`: if metadata is missing, run the real model, persist structural metadata, and return synthetic audio with matching shape
* `cache`: if metadata is missing, skip the test

For the current implementation, cached metadata consists of:

* model-native sample rate
* model-native frame count

This keeps tests focused on structural behavior such as batching, ordering, concatenation, and output-format conversion rather than exact waveform reproduction.

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

The cache-backed resource tests are the basis for longer-term model-backed testing. Future cache metadata will likely grow to include structural outputs such as:

* margins and gaps
* alignment points
* other model-derived timing metadata

As those features appear, tests should continue to prefer structural metadata over waveform snapshots.
