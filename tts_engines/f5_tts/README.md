# F5-TTS proxy engine

This line-oriented proxy uses the official `F5TTS` Python API and the largest
official model class, `F5TTS_v1_Base`. It clones each dialogue speaker from the
host-prepared reference WAV and transcript, keeps one model resident, and
serializes inference because the high-level API accepts one item at a time.
The project’s multi-speaker syntax is deliberately not used; the radio-drama
host already owns speaker selection and exact line ordering.

The engine advertises `needs_transcript`, uses the shared container helpers to
assemble generated line WAVs and line-start timings, and runs offline after the
model and Vocos vocoder have been downloaded. Cross-fading defaults to zero at
this layer because each call produces exactly one authored line and the host
owns later composition.

F5-TTS limits reference audio to 12 seconds. For longer voices, the adapter
uses F5's processed reference and trims the supplied transcript by the same
word-duration proportion so the conditioning audio and text remain aligned.

Build and download from the repository root:

```console
just -f tts_engines/f5_tts/justfile build
just -f tts_engines/f5_tts/justfile download
```

The application launches the image from `tts.toml`; it does not invoke the
justfile. F5-TTS v1 Base is released under CC-BY-NC-4.0. Review the official
model repository before using generated audio commercially.
