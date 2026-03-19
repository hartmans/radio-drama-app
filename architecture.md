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

# Phase 1 Objects to create

* DocumentNode: base document nnode
* ElementNode: base element container
* TextNode: wrapper around text in a document
* SpeakerMapNode: Parsed representation of speakers to voice files. Input is a yaml map inside a <speaker-map> element.
* ScriptNode: A text script of dialogue (speaker: text) <script> element

* ScriptPlan: Connects a ScriptNode to a VibeVoiceResource and produces a RenderedScript on render
* SpeakerMapPlan: Maps speaker names to  actual loaded voices
* VibeVoiceResource: a vibevoice model that collects items to batch and hands out results to ScriptPlans at render time.

* ProductionNode/ProductionPlan/ProductionResult: wraps up the whole thing for an entire production.
