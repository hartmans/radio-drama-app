# Zonos

This proxy engine uses the full Zonos v0.1 transformer checkpoint with voice
cloning from radio-drama's preprocessed references. It batches dialogue lines
across all pending scripts through Zonos' native batched generator, then
returns exact segment-derived line starts. The model stays resident and speaker
embeddings are reused for the container lifetime.

Build from the repository root and copy the example configuration into your
TTS configuration. `ZONOS_LANGUAGE` is an eSpeak language code (default
`en-us`); `ZONOS_BATCH_SIZE` bounds GPU memory use. Audio-prefix and expressive
conditioning controls are intentionally not exposed yet.

The checkpoint is Apache-2.0 licensed. Review the upstream model card before
distribution or production use: https://huggingface.co/Zyphra/Zonos-v0.1-transformer
