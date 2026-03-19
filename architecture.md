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
* Beyond what is implied below, semantic content and representational content are mixed together. models are smart, not manipulated by processor classes
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

# Phase 1

Phase 1 is the smallest architecture that can replace `vibevoice_app.py` while moving to the XML document format.
It should preserve current VibeVoice behavior, current speaker-to-voice lookup behavior, and current round-by-round concatenation behavior.
It should not yet add sound effects, mixing, mastering, scene composition, or any audio graph beyond simple concatenation of VibeVoice output.

## Phase 1 success criteria

* An XML input document can express the same information currently expressed by the plain text file consumed by `vibevoice_app.py`.
* The rendered output for an equivalent document follows the same parsing rules as `vibevoice_app.py`:
  * YAML speaker map
  * speaker lines in `speaker: text` form
  * leading non-speaker lines passed through as prompt text before dialogue
  * at most 4 distinct speakers per VibeVoice round
  * rounds rendered independently and concatenated in order
* Phase 1 returns model-native audio. In practice that means 24k mono output from VibeVoice, not yet 48k stereo production audio.
* Errors in the incoming document identify the relevant XML node and, when practical, line/column information.

## Phase 1 document format

The Phase 1 document should make VibeVoice round boundaries explicit in XML rather than preserving the plain text `----` separator convention inside script text.
That keeps the human-facing format structured without forcing later phases to reverse engineer semantic boundaries from raw text.

Proposed shape:

```xml
<production>
  <speaker-map>
anna: anna.wav
ben: ben.wav
  </speaker-map>
  <script>
    <round>
Intro text that is not speaker-labelled.
Anna: First line.
Ben: Response.
    </round>
    <round>
Anna: Another round.
    </round>
  </script>
</production>
```

Phase 1 guarantees for this format:

* Exactly one `<production>` root.
* Exactly one `<speaker-map>` per production.
* One or more `<script>` elements per production.
* One or more `<round>` elements per `<script>`.
* `<speaker-map>` content is YAML text.
* `<round>` content is plain text interpreted with the same speaker-line rules as `vibevoice_app.py`.

Phase 1 does not guarantee that `<script>` is the long-term unit of dramatic structure.
Later phases may add scenes, assets, effects, and transitions around or above scripts.
The stable guarantee is that a production contains ordered renderable text units, and in Phase 1 those units are reached through `<script>/<round>`.

## Phase 1 objects to create

* `DocumentNode`: base document node carrying source location and enough state to produce good validation errors.
* `ElementNode`: base element container with ordered child nodes and allowed-child metadata.
* `TextNode`: wrapper around literal text in the XML document.
* `ProductionNode`: semantic root node for a production document.
* `SpeakerMapNode`: parsed representation of speakers to voice files. Input is YAML inside `<speaker-map>`.
* `ScriptNode`: ordered container for one or more rounds.
* `RoundNode`: text for one VibeVoice generation round.

* `ProductionPlan`: top-level plan for one production.
* `ScriptPlan`: ordered concatenation of round plans.
* `RoundPlan`: one VibeVoice request using one round of text and the shared speaker map.
* `SpeakerMapPlan`: validated map from canonical speaker names to resolved voice references.
* `VibeVoiceResource`: resource owning model lifecycle and fulfilling round render requests.
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
  * a round has no valid `speaker: text` lines
  * a round references a speaker missing from the speaker map
  * a round uses more than four distinct speakers

Programmer misuse should still surface as normal Python exceptions.
Incoming document problems should be wrapped with document-context errors that include node identity and location.

## Phase 1 planning model

Planning is where the semantic document is converted into renderable work units.
In Phase 1, planning should stay close to current `vibevoice_app.py` behavior rather than prematurely introducing a general audio graph.

Recommended flow:

1. `ProductionNode.plan()` creates a `ProductionPlan`.
2. `SpeakerMapNode` produces a `SpeakerMapPlan` that normalizes speaker names and resolves voice references.
3. Each `RoundNode` produces a `RoundPlan`.
4. Each `ScriptPlan` preserves the source document order of its rounds.
5. `ProductionPlan` preserves the source document order of its scripts.

Document nodes should remain plain semantic objects.
Dependency injection should begin at the planning/resource layer, where shared model state and configuration actually exist.

Stable Phase 1 interfaces:

* `SpeakerMapPlan` exposes a canonical lookup from script speaker name to resolved voice reference.
* `RoundPlan` exposes the normalized prompt text plus normalized ordered dialogue lines for one VibeVoice call.
* `VibeVoiceResource` accepts a round-level request and returns one rendered mono clip.
* `ScriptPlan.render()` concatenates its round results in order with no inserted gaps.
* `ProductionPlan.render()` concatenates its script results in order with no inserted gaps.

Implementation details that are not interface guarantees:

* whether voice resolution happens fully during planning or lazily during first render
* whether `VibeVoiceResource` batches requests immediately in Phase 1 or just renders them one-by-one behind a batch-capable interface
* torch device selection, dtype selection, or flash-attention fallback behavior

## Phase 1 render results

Phase 1 should use the existing `RenderResult` abstraction, but only the subset needed for simple concatenation.

Phase 1 guarantees:

* `audio` is a contiguous mono numpy array
* `sample_rate` is the model-native output sample rate for the rendered clip
* `pre_margin`, `post_margin`, `pre_gap`, and `post_gap` are all zero for Phase 1 VibeVoice output unless a later Phase 1 bug fix proves a non-zero margin is required to describe generated silence already present in the clip

That keeps the render contract compatible with later DSP work without pretending that Phase 1 already has meaningful margin/gap composition semantics.

## Phase 1 resource boundaries

`VibeVoiceResource` is the only Phase 1 component that knows about:

* `VibeVoiceProcessor`
* `VibeVoiceForConditionalGenerationInference`
* model loading details
* device placement
* conversion from `RoundPlan` data into the normalized script text expected by VibeVoice

`RoundPlan` should not know torch or model details.
It should know only semantic text content and resolved voice references.

`SpeakerMapPlan` should normalize the speaker map in the same way `vibevoice_app.py` does today:

* preserve user-authored speaker names for user-facing errors
* support case-insensitive lookup
* resolve voice files by direct path, by file name within the configured voice directory, and by stem within the configured voice directory

## Phase 1 implementation order

1. Build the XML parser and semantic nodes with source locations.
2. Implement validation and document-context errors.
3. Implement `SpeakerMapPlan`, `RoundPlan`, and `ScriptPlan`.
4. Wrap existing VibeVoice generation logic behind `VibeVoiceResource`.
5. Implement `ProductionPlan.render()` and WAV output.
6. Add a compatibility test corpus that compares XML-driven output structure against the current `vibevoice_app.py` behavior for equivalent inputs.

## Explicit Phase 1 non-goals

* effects chains
* background sound or music
* scene-level mixing
* stereo rendering
* sample-rate conversion to project output rate
* general-purpose batching across unrelated model resources
* a document schema for everything needed by later full productions
