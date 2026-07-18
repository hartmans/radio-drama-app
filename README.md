# Radio Drama App

This app renders a production XML document into a radio drama WAV file. It can also launch a FastAPI backend plus a small React preview frontend for comparing render-time presets.

## Quick Start

```console
python -mvenv .venv
.venv/bin/python install -r requirements.txt
```
You will need a copy of the VibeVoice-Large model, which has been removed by microsoft from Huggingface. See [community pages](https://github.com/vibevoice-community/vibevoice) for download instructions. The license of the model is clearly open weight; if you can obtain a copy its legality is clear.

Render the included demo:

```bash
.venv/bin/python radio_drama_app.py \
  demo.xml \
  --voice-dir example_voices \
  --sounds-dir example_sounds \
  --output demo.wav
```

You probably don't want the preset preview backend; it is mostly there for debugging effects chains. But you could launch it:

```bash
.venv/bin/python -m radio_drama.backend \
  demo.xml \
  --voice-dir example_voices \
  --sounds-dir example_sounds
```

Start the frontend:

```bash
cd frontend
npm run dev
```

The demo frontend is keyboard driven:

* `p` restarts playback from the beginning
* `s` stops playback
* `0` selects dry output
* `1` through `6` select the preview presets

## Example Assets

The repository includes a small self-contained demo set:

* voices in `example_voices/`
* sounds in `example_sounds/`
* a production document in `demo.xml`

`demo.xml` uses relative voice names from `example_voices/` and relative sound names from `example_sounds/`, so the render commands above work directly.

## Command-Line Rendering

The main renderer is `radio_drama_app.py`.

```bash
~/ai/vibevoice/.venv/bin/python radio_drama_app.py INPUT.xml [options]
```

Useful options:

* `--voice-dir PATH`: directory containing reference voice files
* `--sounds-dir PATH`: directory searched recursively for relative `<sound>` references
* `--output PATH`: output WAV path; defaults to `INPUT.wav`
* `--output-sample-rate N`: override the production sample rate
* `--output-channels N`: override the output channel count
* `--model-file PATH`: override the VibeVoice model path
* `--batch-size N`, `--device NAME`, `--cfg-scale X`, `--disable-prefill`, `--ddpm-inference-steps N`: VibeVoice overrides

If `--sounds-dir` is not supplied, relative sound references are resolved under a `sounds/` directory next to the XML file.

## Current XML Schema

The current schema is intentionally small.

### `<production>`

`<production>` is the root element.

Current children:

* zero or one `<speaker-map>`
* any number of audio-producing child elements or `<mark>` elements
* today, those audio-producing elements are `<script>` and `<sound>`

Example:

```xml
<production>
  <speaker-map>
    judge: judge2
    prosecutor: lawyer1
    defense: lawyer2
  </speaker-map>

  <script preset="narrator" post_gap="0.75">
    judge: The courtroom was already tense before the first objection.
  </script>

  <sound ref="gavel" />
</production>
```

### `<speaker-map>`

`<speaker-map>` contains YAML mapping authored speaker names to voice references.

Example:

```xml
<speaker-map>
  judge: judge2
  prosecutor: lawyer1
  defense: lawyer2
</speaker-map>
```

Voice references are resolved from `--voice-dir` when provided, or from the default `./voices` directory otherwise. File stems such as `judge2` are accepted.

### `<script>`

`<script>` is a renderable dialogue block.

Supported attributes:

* `start="EXPR"`: explicit outer-geometry placement in the parent composition
* `end="EXPR"`: explicit outer-geometry end position in the parent composition
* `preset="NAME"`: routes this node's rendered audio into the named preset bus at the nearest enclosing compose
* `gain="EXPR"`: post-render gain automation in decibels
* `pre_gap="EXPR"`: time before the audio occupies space in its parent composition
* `post_gap="SECONDS"`: time after the audio occupies space in its parent composition
* `length="EXPR"`: explicit occupied length in the parent composition
* `pan="EXPR"`: stereo automation evaluated against render-time mark positions
* `first_mark="NAME"`: introduces a mark at the node's natural first boundary
* `last_mark="NAME"`: introduces a mark at the node's natural last boundary
* `loop_beg="EXPR"` / `loop_end="EXPR"`: choose the inner-time slice that should repeat
* `loop_loops="NUMBER"`: repeat count after the first pass through the loop body
* `loop_until="EXPR"`: keep repeating until this inner-time position is reached
* `loop_silence="SECONDS"`: silence inserted between loop iterations
* `loop_outro="BOOL"`: append wrapped audio after `loop_end`
* `loop_whole="extend|shorten|no"`: when `loop_until` lands mid-cycle, either extend to the next whole cycle, shorten to the previous whole cycle, or leave the partial cycle in place

Current rules:

* `start` and `pre_gap` are mutually exclusive
* `length` and `post_gap` are mutually exclusive
* `loop_until` and `loop_loops` are mutually exclusive
* `length` must be non-negative
* `pre_gap` is a seconds-valued expression and may be negative; `post_gap` is measured in seconds and may be negative
* `loop_silence` must be non-negative
* `start` is special because it defines the mapping from parent time into the node's inner time, so `start` expressions may use `outer_<mark>` names but not child-local `natural_length` or `inner_<mark>` names
* dialogue lines use `Speaker: text`
* a leading `<recording ref="..." />` declares the script's default recorded source; prefix a stanza with `~` (for example `~Speaker: text`) or use `<line speaker="Speaker" source="recording">text</line>` to select it
* when a script mixes sources, the complete dialogue remains in the TTS request for context, while only TTS-selected lines use synthesized audio; forced alignment for the recording receives only recording-selected dialogue
* continuation lines are folded into the previous dialogue line
* blank lines become paragraph breaks within the same speaker turn
* a script may be empty
* a script may contain nested `<sound>`, `<script>`, `<mark>`, and `<ignore>` elements in document order
* a script may also contain `<line speaker="...">...</line>` elements in document order
* `<ignore>` dialogue is included in the speech-model render request and then sliced back out of the final script audio
* nested preset-bearing audio nodes stay in the same compose scope; the nearest enclosing compose applies one preset bus per preset name before the final mix
* `loop_until` is usually most useful on an explicit-start node that should keep looping under later automatic dialogue
* in `loop_until`, use `inner_<mark>` names; when an explicit-start loop is laid out after automatic siblings, later parent-scope marks are rebased into the loop's own inner time under that `inner_` prefix

Example:

```xml
<script preset="indoor2" post_gap="0.5">
  judge: Be seated.
  <sound ref="gavel" post_gap="-0.3" />
  prosecutor: The state is ready, your honor.
</script>
```

Mixed recorded and synthesized dialogue example:

```xml
<script>
  <recording ref="alice-take.wav" from="12s" gain="-2db" />
  narrator: Alice waited for the verdict.
  ~alice: I already know what it will be.
  narrator: Nobody in the room believed her.
</script>
```

`<recording>` must be the first element child of its script. `<sound-script>` is no longer supported; ordinary nested `<sound>` elements continue to insert inline audio.

Practical loop example:

```xml
<production>
  <speaker-map>
    narrator: judge2
  </speaker-map>

  <sound
    ref="office_roomtone"
    start="0"
    loop_beg="0"
    loop_end="3.2"
    loop_until="inner_brennan_office"
    loop_whole="extend"
    preset="indoor2"
  />

  <script>
    narrator: Brennan opened the folder and stopped talking for a moment.
    <mark id="brennan_office" />
    narrator: Then he finally looked up.
  </script>
</production>
```

That pattern means:

* start the looping bed immediately
* keep repeating the `0` to `3.2` second region
* stop when the later `brennan_office` mark is reached in the parent composition
* because `loop_until` is written in the loop's inner time, that later parent mark appears as `inner_brennan_office`

### `<ignore>`

`<ignore>` contains dialogue guidance that should influence speech generation but not appear in the final rendered script audio.

Example:

```xml
<script preset="thoughts">
  <ignore>
    narrator: Keep this intimate and inward.
  </ignore>
  narrator: I knew the hallway was empty before I opened the door.
</script>
```

### `<line>`

`<line>` may appear inside a `<script>` to author one explicit dialogue line
without reparsing the text as `Speaker: ...`.

Example:

```xml
<script>
  narrator: The room was empty when I arrived.
  <line speaker="narrator" gain="-3" pan="line(door, -1, natural_length, 0)">
    But by then I could already hear footsteps in the hall.
  </line>
</script>
```

Current rules:

* `speaker` is required
* the text inside `<line>` is used literally and may itself begin with `Name:`
* a `<line>` with no audio attrs merges into surrounding normal dialogue
* a `<line>` with audio attrs becomes its own aligned `ScriptSlice`, so its
  audio attrs apply to that one line while the line still contributes to the
  speech-model context of surrounding dialogue
* `first_mark` and `last_mark` are ordinary audio attrs here, so they also
  force a dedicated `ScriptSlice`
* loop attrs are also ordinary audio attrs here, so they also force a
  dedicated `ScriptSlice`

### `<sound>`

`<sound>` inserts an audio asset into production composition.

Equivalent forms:

```xml
<sound ref="gavel" />
<sound>gavel</sound>
```

Supported attributes:

* `ref="NAME_OR_PATH"`: optional if the text content supplies the same value
* `from="SECONDS"`: optional trim start in source-file time
* `to="SECONDS"`: optional trim end in source-file time
* `start="EXPR"`
* `end="EXPR"`
* `gain="DB"`
* `pre_gap="EXPR"`
* `post_gap="SECONDS"`
* `length="EXPR"`
* `pan="EXPR"`
* `preset="NAME"`

Current sound resolution rules:

* absolute paths are used directly
* relative refs are searched recursively under `--sounds-dir` when provided
* otherwise, relative refs are searched recursively under `sounds/` next to the XML document
* supported extensions are `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, and `.aac`
* the search follows symlinks
* the shallowest matching relative path wins
* if multiple matches tie for shallowest, that is a document error
* references may include path separators, for example `court/gavel`

At render time, sounds are normalized with FFmpeg `loudnorm` and converted into the production sample rate and channel layout.

### `<mark>`

`<mark>` inserts a zero-length named cut point into audio composition.

Equivalent forms:

```xml
<mark id="verdict" />
<mark>verdict</mark>
```

Marks bubble upward through enclosing audio plans when unambiguous, so `--cut-before verdict` or `--cut-after verdict` can target a mark inside a nested script.

## Expression Attributes

Current expression-driven audio attributes:

* `gain`
* `pan`
* `start`
* `end`

`gain` and `pan` evaluate at render time against visible marks in natural
sample geometry. `start` and `end` are layout-time expressions. Unprefixed mark
names are not populated specially; they simply fail as ordinary undefined names
if used.

## Current Presets

Built-in render-time presets:

* `master`
* `narrator`
* `thoughts`
* `outdoor1`
* `outdoor2`
* `indoor1`
* `indoor2`
* `phone`: for people on the other side of a phone call
* `background`: Very side-heavy mix for conversations in the background, especially under narration

`master` is reserved for the final production render. The preview frontend/backend expose `none` plus the other seven presets.

## Demo Production

`demo.xml` demonstrates:

* a `<speaker-map>`
* a `narrator` preset block
* an `indoor2` courtroom scene
* inline `<sound>` usage
* `post_gap` timing

Render it with:

```bash
~/ai/vibevoice/.venv/bin/python radio_drama_app.py \
  demo.xml \
  --voice-dir example_voices \
  --sounds-dir example_sounds \
  --output demo.wav
```

## Voices

Voices were produced using the VoiceDesign model from [Qwen TTS](https://github.com/QwenLLM/Qwen3-TTS)
The example sounds are from freesound.org.
* [gavel](https://freesound.org/people/Science_Witch/sounds/762733/)

## Development Style

This app was mostly vibe coded with Codex and GPT 5.4. There was initial architecture work as input to Codex, and significant code inspection and refactoring instruction to produce an extensible code base.
