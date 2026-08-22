# Higgs TTS 3 proxy engine

This engine runs `multimodalart/higgs-audio-v3-tts-4b-transformers` directly
through Transformers. Each dialogue line is synthesized with its resolved
speaker reference and the returned WAV segments are concatenated into one
script artifact in `/cache`. There is no auxiliary inference server.

The engine advertises the proxy `needs_transcript` capability. The host then
uses its shared, cached reference ASR resource to enrich each speaker reference
before rendering. The adapter supplies both the mounted reference audio path
and that transcript to Higgs voice cloning.

Dialogue lines remain independent synthesis items. The Transformers port's
autoregressive decoder currently accepts one item at a time, so a bounded
protocol batch is generated serially on one resident model. `HIGGS_BATCH_SIZE`
(default `1`) controls how much pending line work the adapter stages at once;
it is a forward-compatible boundary rather than GPU-continuous batching.

The language model runs in bfloat16 on CUDA. The separate Higgs audio tokenizer
used for reference encoding and waveform decoding is eagerly loaded and forced
to float32; startup fails if any of its parameters use another dtype. This is
intentional because the codec's decode is unstable in bfloat16.

For boundary debugging, set `HIGGS_KEEP_LINE_WAVS=1`. The adapter will retain
the exact WAV returned for every line alongside the concatenated cache artifact
as `<artifact>.line-0.wav`, `<artifact>.line-1.wav`, and so on. These files are
normally removed after concatenation. Batch operation does not alter their
meaning: each retained file is still one batch item's unmodified response.

The image is built directly on NVIDIA's CUDA 12.8 base and installs pinned
PyTorch, Torchaudio, SoundFile, and Transformers releases. SoundFile handles
WAV I/O without making it depend on Torchaudio's optional TorchCodec backend.
The image contains no SGLang runtime.
The radio-drama JSON-lines handshake happens before lazy model loading,
allowing the host to finish advertised setup such as reference ASR before GPU
allocation begins.

Both checkpoints live under `/models/huggingface`, which is declared as a
Containerfile volume. Runtime is offline by default. The sample configuration
bind-mounts a persistent host directory at that location.

Build and populate the cache from the repository root:

```console
just -f tts_engines/higgs_tts_3/justfile build
just -f tts_engines/higgs_tts_3/justfile download
```

The base image and resolved CUDA dependencies are large. Set `TMPDIR` to a
filesystem with substantial free space; `/tmp` is only an example and may need
to be replaced on hosts where it is small. Podman's image graph root also needs
enough room for the unpacked layers.

Copy `tts.toml.example` to `$XDG_CONFIG_HOME/radio-drama/tts.toml`, then select
the engine with `<script tts="higgs">`. The sample uses Podman's NVIDIA CDI
device name, host IPC, no runtime network, and a host Hugging Face cache bind
mount. Create that host cache directory before the first download.
Adjust the device name if the host's Podman/NVIDIA setup exposes a different
CDI device. The actual application launches the image from `tts.toml`; it does
not invoke the justfile.

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
