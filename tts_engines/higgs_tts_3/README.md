# Higgs TTS 3 proxy engine

This engine maps radio-drama proxy requests to the OpenAI-compatible speech
API provided by SGLang-Omni for `bosonai/higgs-tts-3-4b`. Each dialogue line is
synthesized with its resolved speaker reference and the returned WAV segments
are concatenated into one script artifact in `/cache`.

Build from the repository root:

```console
podman build -f tts_engines/higgs_tts_3/Containerfile \
  -t localhost/radio-drama-higgs-tts-3 .
```

Copy `tts.toml.example` to `$XDG_CONFIG_HOME/radio-drama/tts.toml`, then select
the engine with `<script tts="higgs">`. The sample uses Podman's NVIDIA CDI
device name and a host Hugging Face cache bind mount. Adjust the device name if
the host's Podman/NVIDIA setup exposes a different CDI device.

The checkpoint is covered by the Boson Higgs TTS 3 Research and Non-Commercial
License, including its Creator Use terms and attribution requirement. Review
the current model license before using generated audio:
https://huggingface.co/bosonai/higgs-tts-3-4b
