"""Download every model used by Seed-VC v1 inference into a reusable cache."""

from pathlib import Path
import os
import sys

from huggingface_hub import hf_hub_download, snapshot_download


checkpoint_dir = Path(sys.argv[1]).resolve()
hub_cache = checkpoint_dir / "hf_cache"
os.environ["HF_HUB_CACHE"] = str(hub_cache)

for filename in (
    "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
    "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
    "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
):
    hf_hub_download("Plachta/Seed-VC", filename=filename, cache_dir=checkpoint_dir)

hf_hub_download(
    "funasr/campplus", filename="campplus_cn_common.bin", cache_dir=checkpoint_dir
)
hf_hub_download(
    "lj1995/VoiceConversionWebUI", filename="rmvpe.pt", cache_dir=checkpoint_dir
)
snapshot_download(
    "openai/whisper-small",
    cache_dir=hub_cache,
    allow_patterns=("*.json", "*.txt", "*.model", "model.safetensors"),
)
snapshot_download("nvidia/bigvgan_v2_22khz_80band_256x", cache_dir=hub_cache)
snapshot_download("nvidia/bigvgan_v2_44khz_128band_512x", cache_dir=hub_cache)

# Transformers otherwise mistakes this populated Hub cache for its pre-v4.22
# layout and emits an offline migration warning on every invocation.
(hub_cache / "version.txt").write_text("1\n")
