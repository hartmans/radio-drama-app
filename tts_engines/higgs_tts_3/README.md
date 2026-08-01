# Higgs TTS 3 proxy engine

This engine maps radio-drama proxy requests to the OpenAI-compatible speech
API provided by SGLang-Omni for `bosonai/higgs-tts-3-4b`. Each dialogue line is
synthesized with its resolved speaker reference and the returned WAV segments
are concatenated into one script artifact in `/cache`.

The image is self-contained except for the model checkpoint. Its build installs
SGLang-Omni and the Higgs inference runtime into the image. At container
startup, the engine entrypoint launches `sgl-omni serve` inside the same
container, waits for that local server to become ready, and then starts the
radio-drama JSON-lines protocol on stdin/stdout. It does not require or connect
to an externally managed inference server.

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
device name, host IPC, a 32 GiB shared-memory allocation, host networking for
the initial checkpoint download, and a host Hugging Face cache bind mount.
Adjust the device name if the host's Podman/NVIDIA setup exposes a different
CDI device. Once the checkpoint is populated, the network policy can be made
more restrictive if the local runtime does not need network access.

The checkpoint is covered by the Boson Higgs TTS 3 Research and Non-Commercial
License, including its Creator Use terms and attribution requirement. Review
the current model license before using generated audio:
https://huggingface.co/bosonai/higgs-tts-3-4b
