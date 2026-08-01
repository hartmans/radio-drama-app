# Chatterbox

This proxy engine uses Chatterbox Multilingual V3, the largest current
Chatterbox family size, with voice cloning from radio-drama's preprocessed
references. The public decoder supports one sample at a time, so line inference
is serialized on one persistent GPU model; expensive speaker conditionals are
cached and reused for the container lifetime. Exact line starts are derived
from the generated segments.

Build from the repository root and copy the example configuration into your
TTS configuration. `CHATTERBOX_LANGUAGE` defaults to `en` and accepts one of
the multilingual model's supported language identifiers.

Chatterbox is MIT licensed and generated audio contains its built-in PerTh
watermark. See https://huggingface.co/ResembleAI/chatterbox for current details.
