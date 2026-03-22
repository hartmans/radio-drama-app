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
* any number of audio-producing child elements
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

* `preset="NAME"`: applies a named effect chain after the script is rendered
* `pre_gap="SECONDS"`: time before the audio occupies space in its parent composition
* `post_gap="SECONDS"`: time after the audio occupies space in its parent composition
* `length="SECONDS"`: explicit occupied length in the parent composition

Current rules:

* `length` and `post_gap` are mutually exclusive
* `length` must be non-negative
* `pre_gap` and `post_gap` are measured in seconds and may be negative
* dialogue lines use `Speaker: text`
* continuation lines are folded into the previous dialogue line
* blank lines become paragraph breaks within the same speaker turn
* a script may be empty
* a script may contain nested `<sound>` and `<script>` elements in document order

Example:

```xml
<script preset="indoor2" post_gap="0.5">
  judge: Be seated.
  <sound ref="gavel" post_gap="-0.3" />
  prosecutor: The state is ready, your honor.
</script>
```

### `<sound>`

`<sound>` inserts an audio asset into production composition.

Equivalent forms:

```xml
<sound ref="gavel" />
<sound>gavel</sound>
```

Supported attributes:

* `ref="NAME_OR_PATH"`: optional if the text content supplies the same value
* `pre_gap="SECONDS"`
* `post_gap="SECONDS"`
* `length="SECONDS"`

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

## Current Presets

Built-in render-time presets:

* `master`
* `narrator`
* `thoughts`
* `outdoor1`
* `outdoor2`
* `indoor1`
* `indoor2`

`master` is reserved for the final production render. The preview frontend/backend expose `none` plus the other six presets.

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
