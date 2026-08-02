# Higgs TTS 3 proxy engine

This engine maps radio-drama proxy requests to the OpenAI-compatible speech
API provided by SGLang-Omni for `bosonai/higgs-tts-3-4b`. Each dialogue line is
synthesized with its resolved speaker reference and the returned WAV segments
are concatenated into one script artifact in `/cache`.

The engine advertises the proxy `needs_transcript` capability. The host then
uses its shared, cached reference ASR resource to enrich each speaker reference
before rendering. The adapter supplies both the mounted reference audio path
and that transcript to Higgs voice cloning.

Dialogue lines remain independent synthesis items. For each bounded group, the
adapter submits concurrent requests to SGLang-Omni's standard
`/v1/audio/speech` endpoint, allowing its continuous scheduler to batch model
execution through the officially supported API. Set `HIGGS_BATCH_SIZE` to
control the maximum concurrency; the default is `16`.

Higgs can occasionally return an utterance with a pathological high-amplitude
onset. The adapter detects an initial 100 ms whose RMS is both at least `0.25`
of full scale and at least three times the following 400 ms. It retries only
the affected lines, concurrently within each retry round, up to two times. The
first clean result is used; if all attempts are affected, the least severe
result is retained. `HIGGS_ONSET_RETRIES` changes the retry count (use `0` to
disable retries), while `HIGGS_ONSET_RMS_THRESHOLD` and
`HIGGS_ONSET_RATIO_THRESHOLD` tune the two detection gates. Rejected attempts
are reported on standard error so the behavior can be monitored.

Set `HIGGS_INITIAL_CODEC_CHUNK_FRAMES` to pass SGLang-Omni's
`initial_codec_chunk_frames` option to Higgs. When it is unset, the adapter
leaves the option out and SGLang-Omni applies its current default. This is an
engine tuning and diagnostic option: changing the first codec decode chunk can
affect time to first audio and the quality of the beginning of an utterance.
With the current Higgs pipeline, unset or `0` uses the full first codec chunk;
positive values below the steady chunk size request an earlier, smaller first
decode. Values at or above the steady size are clamped to the steady size.

For boundary debugging, set `HIGGS_KEEP_LINE_WAVS=1`. The adapter will retain
the exact WAV returned for every line alongside the concatenated cache artifact
as `<artifact>.line-0.wav`, `<artifact>.line-1.wav`, and so on. These files are
normally removed after concatenation. Batch operation does not alter their
meaning: each retained file is still one batch item's unmodified response.

The image is self-contained except for the model checkpoint. It is a thin
derivative of a digest-pinned official `lmsysorg/sglang-omni` image and uses
that image's preinstalled `/opt/omni` runtime without resolving or replacing
any Python, CUDA, or kernel packages. At container startup, the engine
entrypoint launches `sgl-omni serve` inside the same
container on the first render request and waits for that local server to become
ready. The server permits local media reads only under `/voices`, where the
host mounts its read-only prepared speaker references. The radio-drama
JSON-lines handshake happens first, allowing the host to
finish advertised setup such as reference ASR before model loading begins. The
engine does not require or connect to an externally managed inference server.

The checkpoint is downloaded by the in-container Hugging Face client on first
use and stored under `/models/huggingface`, which is declared as a Containerfile
volume. The sample configuration bind-mounts a persistent host directory at
that location so subsequent containers reuse the downloaded checkpoint.

Build from the repository root:

```console
TMPDIR=/tmp podman build -f tts_engines/higgs_tts_3/Containerfile \
  -t localhost/radio-drama-higgs-tts-3 .
```

The base image and resolved CUDA dependencies are large. Set `TMPDIR` to a
filesystem with substantial free space; `/tmp` is only an example and may need
to be replaced on hosts where it is small. Podman's image graph root also needs
enough room for the unpacked layers.

Copy `tts.toml.example` to `$XDG_CONFIG_HOME/radio-drama/tts.toml`, then select
the engine with `<script tts="higgs">`. The sample uses Podman's NVIDIA CDI
device name, host IPC, host networking for the initial checkpoint download,
and a host Hugging Face cache bind mount. Create that host cache directory
before the first run.
Adjust the device name if the host's Podman/NVIDIA setup exposes a different
CDI device. Once the checkpoint is populated, the network policy can be made
more restrictive if the local runtime does not need network access.

## Speech controls

The in-container adapter translates recognized bracketed expressions in spoken
text to Higgs control tokens. This keeps Higgs's `<|category:tag|>` syntax out
of production XML. For example:

```text
obsidia: [emotion:affection][style:whispering]Come closer.
obsidia: Wait for it [prosody:long_pause] now.
obsidia: [sfx:laughter]Haha, I knew you would agree.
```

Sentence-level controls belong at the start of a sentence: emotions, styles,
speed, pitch, and expressiveness. Pauses and sound effects belong at the point
where they occur. Sound effects should be followed immediately by matching
onomatopoeia, with no space: `[sfx:cough]Ahem`, for example. Controls may be
stacked. Unknown expressions and ordinary bracketed text are passed through
unchanged rather than being treated as model controls.

The complete catalog currently documented by the model authors contains 43
controls:

* `emotion` (21): `affection`, `amusement`, `anger`, `arousal`, `awe`,
  `bitterness`, `confusion`, `contemplation`, `contentment`, `determination`,
  `disgust`, `elation`, `enthusiasm`, `fear`, `helplessness`, `longing`,
  `pride`, `relief`, `sadness`, `shame`, `surprise`
* `prosody` sentence controls (8): `speed_very_slow`, `speed_slow`,
  `speed_fast`, `speed_very_fast`, `pitch_low`, `pitch_high`,
  `expressive_high`, `expressive_low`
* `prosody` inline controls (2): `pause`, `long_pause`
* `style` (3): `singing`, `shouting`, `whispering`
* `sfx` (9): `cough`, `laughter`, `crying`, `screaming`, `burping`,
  `humming`, `sigh`, `sniff`, `sneeze`

Use `[category:tag]`, such as `[prosody:speed_slow]`. The upstream prompting
guide notes that `speed_very_slow` has limited slowing range; use inline
`long_pause` controls when a delivery needs more space. The catalog and
placement rules come from the model's
[PROMPTING.md](https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/PROMPTING.md).

The checkpoint is covered by the Boson Higgs TTS 3 Research and Non-Commercial
License, including its Creator Use terms and attribution requirement. Review
the current model license before using generated audio:
https://huggingface.co/bosonai/higgs-tts-3-4b
