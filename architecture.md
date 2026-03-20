# Goal

To create an app that can take a human-edited xml script and turn it into a high quality radio drama using leading TTS AI models, sound effects, and a digital effects chain.

# Architectural assumptions

* Python
* async app calling out to threads or multiprocessing where APIs  or performance are enhanced
* Vibevoice for speech synthesis
* Numpy for audio samples
* Most of the app runs two-channel 48000 default sample rate (although sample rate should be configurable)
* VibeVoice is 24k mono
* Longterm numpy, scipy.signal,  ffmpeg, and pedalboard  for effects processing/mastering
* wrappers around process invocation (no direct asyncio called process)
* Model calls/interaction in isolated classes: vibevoice, qwen-tts and some other code bases in this stack have strict dependencies and may need to get thrown into their own processes
* Interfaces to isolated model classes should be pickleable for eventual multiprocess/separate interpreter
* Beyond what is implied below, semantic content and representational content are mixed together. models are smart, not manipulated by processor classes or functions. New functionality should be added by adding new classes without needing to update some global planner or renderer.
* use carthage.Injector and Injectable for managing resources and non-local state
  * Most nodes, plans and resources should be AsyncInjectables
  * Find VibeVoiceResource and SpeakerVoicePlan through injectors
  * Most of Carthage above the dependency injection layer are not useful to this project
* We are using XML, but the document is for humans. Other formats within XML is fine. As an example, the speaker map is a yaml document in an xml element.
# Low Level

## DocumentNode

A DocumentNode represents a part of the document, often an element and everything within the element.

DocumentNodes maintain enough state to print errors.

DocumentNodes are constructed either from a sax parser or by being given a dom element (implementation choice, pick one, not both)

A DocumentNode representing an element has an inner list of document nodes representing text and other elements (in document order).
DocumentNodes do have semantic awareness of what their part of the document means.
So as part of converting the document to DocumentNodes, an outer DocumentNode is responsible for having a map of what inner elements are permitted and what DocumentNodes are used for those.
We assume a DocumentNode has one single set of allowed inner nodes; we can catch illegal combinations like a node being repeated that should not be  during scheduling or some sort of validation function on DocumentNode
DocumentNode should grow a small set of reusable helpers rather than pushing every error-path into individual subclasses.
Examples include:

* constructing document-context errors from the node
* reporting child-element errors consistently
* collecting or normalizing inner text
* common cardinality checks such as exactly-one, at-most-one, or one-or-more child lookups

DocumentNode remains semantic rather than generic, but the repeated mechanics of location-aware validation should live in shared helpers.

## PlanningNode

DocumentNodes are scheduled/planned into PlanningNodes. A PlanningNode represents the composition of processing (such as text to speech, ASR, effects chain, etc) that will be applied  to inputs in order to form the final production.

Inputs include:

* ScriptNodes (DocumentNodes representing tts scripts)
* sound files
* voices

Planning allows us to get a model of the pipeline. It allows us to understand what resources can be batched together (for example batches in calls to an llm that can process multiple batches at once) and to set up a structure to be filled in during rendering.

Work can happen in planning if it does not disrupt later parallelism.
Examples include:

* Downloading samples
* Loading/setting up models

Often work like that may involve a cache to make sure that work is not repeated if multiple parts of the plan reference the same resource.

PlanningNode should be a real base class.
It should be the consistent entry point for planning-layer behavior, including:

* holding the source DocumentNode that produced the plan
* providing a common async construction pattern
* providing a common render entry point
* caching or memoizing render work where appropriate
* shared helpers for location-aware errors raised during planning or rendering

All plan classes should be AsyncInjectables unless there is a strong architectural reason not to.
Convenience in a particular phase is not a sufficient reason to mix constructor patterns.

## RenderResults

All plans can be rendered (an async method).
Most plans render into RenderResult.
A RenderResult contains audio .
It also contains elements  to help combine render results together. These are loosly  modeled on bounding boxes used in page composition:

 pre_margin: Silence or near silence included at the beginning of the audio clip
post_margin: Silence or near silence included at the end of the audio clip
pre_gap: intended silence between the previous clip and the beginning of the margin for this sample.
post_gap: Intended silence between the end of this clip and the beginning  of the next one.

If  pre_gap or post_gap are negative then clips can run together and be mixed.

Margins are included in the audio sample and are typically calculated with DSP.
Gaps  are added and typically under user control.

Two UI/planning level values that should disappear by the time RenderResults are created:

total_pre_gap: If set, calculate the pre_gap such that pre_gap = total_pre_gap-pre_margin
total_post_gap: If set calculate post_gap as total_post_gap-post_margin

Rendering a PlanningNode  may start more work than is required just for that planning node. As an example if a PlanningScript is being batched together with other PlanningScripts , rendering any one of those should render the entire batch.

RenderResult should also grow reusable helpers rather than forcing each plan to rebuild basic audio operations.
Examples include:

* an empty result at a known sample rate
* concatenation helpers
* normalization of array shape and dtype
* frame-count helpers

Not every PlanningNode necessarily renders to audio.
Some plans, such as a validated speaker map, may have a render method that is a no-op and returns `None`.
The important interface guarantee is that every plan has a render path, not that every plan produces sound.

# Phase 1

Phase 1 is the smallest architecture that can replace `vibevoice_app.py` while moving to the XML document format.
It should preserve current VibeVoice behavior and current speaker-to-voice lookup behavior while using `<script>` elements as the renderable unit.
It should include output-format conversion to the configured production format.
It should not yet add sound effects, scene composition, or a general audio graph beyond ordered concatenation plus required output formatting.

## Phase 1 success criteria

* An XML input document can express the same information currently expressed by the plain text file consumed by `vibevoice_app.py`.
* The rendered output for an equivalent document preserves the same core VibeVoice constraints as `vibevoice_app.py`:
  * YAML speaker map
  * script text made entirely of `speaker: text` lines
  * at most 4 distinct speakers per script
  * scripts rendered independently and concatenated in order
* Phase 1 returns production-format audio. VibeVoice still renders 24k mono internally, but Phase 1 converts to the configured output sample rate and channel layout, defaulting to 48k stereo.
* Errors in the incoming document identify the relevant XML node and, when practical, line/column information.

## Phase 1 document format

The Phase 1 document should use `<script>` as the renderable VibeVoice unit.
The plain text `----` separator convention from `vibevoice_app.py` should disappear rather than being recreated in XML.

Proposed shape:

```xml
<production>
  <speaker-map>
anna: anna.wav
ben: ben.wav
  </speaker-map>
  <script>
Anna: First line.
Ben: Response.
  </script>
  <script>
Anna: Another round.
  </script>
</production>
```

Phase 1 guarantees for this format:

* Exactly one `<production>` root.
* Exactly one `<speaker-map>` per production.
* One or more `<script>` elements per production.
* `<speaker-map>` content is YAML text.
* `<script>` content is plain text interpreted as speaker stanzas.
  * a stanza begins with `speaker:`
  * following non-speaker lines belong to the current speaker until another recognized `speaker:` line
  * blank lines may appear within a stanza
  * an entirely empty script is valid and renders to zero samples

Phase 1 does not guarantee that `<script>` is the long-term unit of dramatic structure.
Later phases may add scenes, assets, effects, and transitions around or above scripts.
The stable guarantee is that a production contains ordered renderable text units, and in Phase 1 those units are the `<script>` elements.

## Phase 1 objects to create

* `DocumentNode`: base document node carrying source location and enough state to produce good validation errors.
* `ElementNode`: base element container with ordered child nodes and allowed-child metadata.
* `TextNode`: wrapper around literal text in the XML document.
* `ProductionNode`: semantic root node for a production document.
* `SpeakerMapNode`: parsed representation of speakers to voice files. Input is YAML inside `<speaker-map>`.
* `ScriptNode`: text for one VibeVoice generation request.

* `PlanningNode`: base class for all plans.
* `ProductionPlan`: top-level plan for one production.
* `ScriptPlan`: one VibeVoice request using one script and the shared speaker map.
* `SpeakerMapPlan`: validated map from canonical speaker names to resolved voice references.
* `VibeVoiceResource`: resource owning model lifecycle and fulfilling script render requests.
* `OutputFormatResource`: resamples and upmixes concatenated production audio into the configured output format.
* `ProductionResult`: wraps the rendered production audio and its sample rate.

## Phase 1 parsing and validation

For Phase 1, the document parser should build a semantic tree in one pass from XML and preserve source locations.
That favors a SAX-style parser feeding `DocumentNode` construction rather than building a generic DOM first and then translating it.
The long-term guarantee is the semantic node tree and its error locations, not the XML parsing library used underneath.

Validation should happen in two layers:

* Structural validation during parsing:
  * unknown child elements
  * required child elements missing
  * duplicate elements where the schema does not permit repetition
* Content validation during node-specific validation/planning:
  * `<speaker-map>` YAML is not a mapping
  * speaker names or voice names are empty or non-strings
  * non-speaker content appears before any speaker stanza begins
  * a script references a speaker missing from the speaker map
  * a script uses more than four distinct speakers

Programmer misuse should still surface as normal Python exceptions.
Incoming document problems should be wrapped with document-context errors that include node identity and location.

## Phase 1 planning model

Planning is where the semantic document is converted into renderable work units.
In Phase 1, planning should stay close to current `vibevoice_app.py` behavior rather than prematurely introducing a general audio graph.

Recommended flow:

1. `ProductionNode.plan()` creates a `ProductionPlan`.
2. `SpeakerMapNode` produces a `SpeakerMapPlan` that normalizes speaker names and resolves voice references.
3. Each `ScriptNode` produces a `ScriptPlan`.
4. `ProductionPlan` preserves the source document order of its scripts.
5. Each plan has a `render()` method, even if rendering is a no-op for that plan.
6. `ProductionPlan.render()` asks its child plans to render rather than embedding the plan-to-render transition in separate procedural helpers.
7. `ProductionPlan.render()` concatenates model-native script outputs and then passes the production audio through `OutputFormatResource`.

Document nodes should remain plain semantic objects.
Dependency injection should begin at the planning/resource layer, where shared model state and configuration actually exist.
The planning/resource layer should be organized around reusable abstractions rather than around a one-off "phase1" module.

Stable Phase 1 interfaces:

* `SpeakerMapPlan` exposes a canonical lookup from script speaker name to resolved voice reference.
* `ScriptPlan` exposes the normalized ordered dialogue stanzas for one VibeVoice call.
* `VibeVoiceResource` accepts a script-level request and returns one rendered mono clip at the model-native sample rate.
* `ProductionPlan.render()` concatenates its script results in order with no inserted gaps before output-format conversion.
* `OutputFormatResource` converts concatenated production audio to the configured sample rate and channel layout.

Implementation details (not interface guarantees, but decisions for phase 1):
* `VibeVoiceResource` should do real batching in phase 1. It will be a significant performance win. Some selection of batch size is needed. Default to 10.
* for now, torch device is cuda and dtype is bfloat16
* Voice resolution should happen during planning to catch errors there.



## Phase 1 render results

Phase 1 should use the existing `RenderResult` abstraction for intermediate and final audio.

Phase 1 guarantees:

* `ScriptPlan.render()` returns model-native audio as a contiguous mono numpy array at the model-native sample rate.
* An empty script renders successfully and returns zero samples at the model-native sample rate.
* `ProductionPlan.render()` returns a contiguous numpy array in the configured production output format.
* Phase 1 production output defaults to stereo at 48000 Hz.
* `pre_margin`, `post_margin`, `pre_gap`, and `post_gap` are all zero for Phase 1 output unless a later Phase 1 bug fix proves a non-zero margin is required to describe generated silence already present in the clip.

That keeps the render contract compatible with later DSP work without pretending that Phase 1 already has scene-level composition semantics.

## Phase 1 resource boundaries

`VibeVoiceResource` is the only Phase 1 component that knows about:

* `VibeVoiceProcessor`
* `VibeVoiceForConditionalGenerationInference`
* model loading details
* device placement
* conversion from `ScriptPlan` data into the normalized script text expected by VibeVoice

`ScriptPlan` should not know torch or model details.
It should know only semantic dialogue content and resolved voice references.

`OutputFormatResource` is the Phase 1 component responsible for:

* resampling from model-native audio to the configured production sample rate
* converting mono model output to the configured channel layout
* keeping the conversion policy simple and deterministic so later phases can wrap it in richer mastering or mixing steps

`SpeakerMapPlan` should normalize the speaker map in the same way `vibevoice_app.py` does today:

* preserve user-authored speaker names for user-facing errors
* support case-insensitive lookup
* resolve voice files by direct path, by file name within the configured voice directory, and by stem within the configured voice directory

## Runtime defaults

Runtime defaults should live in configuration objects or resource logic, not in the argparse frontend.
The frontend should primarily pass explicit overrides.
This keeps library use and CLI use from drifting apart.

It is acceptable for many config fields to be `None` until a resource resolves them to defaults.

## Codex sandbox workaround

The Codex asyncio workaround is not part of the application architecture.
If needed for development in this environment, it should live in an external runner wrapper that applies the patch before executing a Python module or script.
Main code and tests should not embed Codex-specific wakeup hacks.

## Phase 1 implementation order

1. Build the XML parser and semantic nodes with source locations.
2. Implement validation and document-context errors.
3. Implement `SpeakerMapPlan` and `ScriptPlan`.
4. Wrap existing VibeVoice generation logic behind `VibeVoiceResource`.
5. Implement `OutputFormatResource`.
6. Implement `ProductionPlan.render()` and WAV output.
7. Add a compatibility test corpus that compares XML-driven output structure against the current `vibevoice_app.py` behavior for equivalent inputs.

## Testing strategy

Use `pytest`.

Most tests should not depend on a live VibeVoice model.
Parser, validation, planning, batching, and output-format logic should be testable with ordinary unit tests and fixture-provided fake render results.

Model-backed tests should run through a cache-aware pytest fixture around `VibeVoiceResource`.
That fixture should support two modes:

* live mode: if a request is missing from cache, run the real model call, record the cached metadata, and return a synthetic audio array for the rest of the test
* cache mode: if a request is missing from cache, call `pytest.skip`

For Phase 1, the cache does not need to store rendered waveform data.
It should store only the model output metadata needed by higher-level tests.
At minimum that means:

* model-native sample rate
* model-native frame count

Later phases can extend cached metadata with values such as margins, gap-related metadata, forced-alignment points, or similar structural render outputs.

The fixture can then return white noise of the cached length at the cached sample rate.
That is enough for tests that only care about ordering, concatenation, batching boundaries, output-format conversion, and other structural properties above the actual model output.

This means `VibeVoiceResource` will often be mocked in tests, and that is acceptable.
Architecturally, it is a reason to keep model invocation behind a narrow request/response boundary so tests can substitute:

* the real model-backed implementation
* a cache-backed implementation
* a pure fake implementation for unit tests

Cache keys should be derived from the semantic model request, not raw incidental input text.
For Phase 1, `VibeVoiceResource` only needs one model configuration, so the key can stay simple.
At minimum it should include the normalized script content and resolved voice references.
It should not try to encode generation parameters in order to chase exact output length, because the cache is only meant to provide plausible structural test data rather than reproduce a specific generation exactly.

## Explicit Phase 1 non-goals

* effects chains
* background sound or music
* scene-level mixing
* general-purpose batching across unrelated model resources
* a document schema for everything needed by later full productions
