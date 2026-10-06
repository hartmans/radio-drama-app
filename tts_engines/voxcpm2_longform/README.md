# VoxCPM2 engine

The TTS engine uses a resident vLLM-Omni `AsyncOmni` scheduler. It always
supplies the original speaker sample as identity reference audio. The separate
VoxCPM2 voice-design sound tool is unaffected.

For a line that begins with a non-empty parenthetical, such as
`(flustered, quiet)I understand.`, the parenthetical is a VoxCPM2 control
instruction.  The line uses reference-only cloning: it does not supply a
prompt WAV or prompt text, because VoxCPM2's continuation/ultimate-cloning
mode does not support style controls.

For every other line, the engine uses continuation cloning.  The first such
line for a speaker is prompted with that speaker's original reference WAV and
its reference transcript. A controlled line becomes the continuation prompt
for later un-controlled lines from that speaker; its parenthetical control is
excluded from the stored prompt text. The engine keeps that prompt fixed rather
than chaining each generated line into the next, avoiding progressive voice
degradation.

Prompt state is scoped to one render request, so concurrently submitted script
requests do not condition one another.

The engine also advertises `streaming` and serves a single render request over
a Unix socket, using the same resident model and cloning rules as batch renders.
Up to seven non-streaming script requests run concurrently, guarded by a
container-side semaphore. Each script renders its lines in order. Streaming
requests bypass that semaphore and are always submitted to the scheduler;
there is no application-level streaming admission limit. The scheduler runs
at most eight sequences at once, so additional streams can queue there.
Client disconnect closes the async generator and aborts its engine request.
Engine failures are logged and close the socket with audio already produced.

`deploy.yaml` caps the KV cache at **6 GiB**, independently of GPU capacity,
with eight active sequences and a 4096-token context limit. Model weights,
diffusion/AudioVAE buffers, and graph pools use additional memory; 6 GiB is
not a total VRAM cap. The startup utilization fraction is 0.2, so a shared
96 GiB card does not need 90% of its memory free. Prefix caching is disabled.
This follows the upstream
[VoxCPM2 deployment configuration](https://github.com/vllm-project/vllm-omni/blob/v0.30.0/vllm_omni/deploy/voxcpm2.yaml).

To tune runtime settings, mount a custom deployment YAML and set
`VOXCPM_DEPLOY_CONFIG` to its container path. Adjust `kv_cache_memory_bytes`,
`max_model_len`, and `engine_extras.hf_overrides.voxcpm2_runtime_config`
there. The image patches version 0.30.0's hardcoded generation settings
to default to 20 steps, avoiding the echo observed on longer texts at 10 steps.
Set `VOXCPM_INFERENCE_TIMESTEPS` to override this default, as with the native
engine. `VOXCPM_CFG_VALUE` overrides guidance strength, which defaults to 2.0.
The patch runs during the image build and fails if the upstream
initializations change; rebuild the image to apply it. Both settings are read
when each worker initializes its model, before CUDA graph capture; restart
the backend after changing them. The native engine's
`VOXCPM_DEVICE`, `VOXCPM_OPTIMIZE`, and `VOXCPM_NORMALIZE`
environment settings do not apply. Select GPUs
through container device visibility; the deployment uses logical GPU 0.

Rebuild the image to switch existing installations to vLLM-Omni. The image
pins matching vLLM and vLLM-Omni 0.30.0 releases; Omni installs from PyPI.

Mount a persistent read-write cache at `/voxcpm2_longform_cache`, as in
`tts.toml.example`, to reuse compile artifacts after the proxy removes its
container. The example uses the host directory
`~/.cache/radio-drama/voxcpm2_longform_cache`, created automatically by the
proxy. The image and example route PyTorch Inductor, Triton, vLLM (including
its compile cache), CUDA driver JIT, and XDG-based dependency caches into
separate subdirectories of this mount. The Hugging Face model cache remains
at `/models/huggingface`.

Existing installations can add the example's cache environment settings and
mount to their `tts.toml` without rebuilding. Restart the resident backend to
apply them. The first run populates the cache; later compatible runs reuse
artifacts. Changes to models, GPU hardware, or compiler versions can require
recompilation. Model loading and process-local CUDA graph capture still run
at startup, so this does not eliminate all startup work. The cache is
disposable and can be cleared while the backend is stopped.

The image defaults to `HF_HUB_OFFLINE=1`, matching the proxy's default
`network="none"`. Runtime uses the mounted model cache without DNS or remote
repository metadata queries. Existing native VoxCPM2 checkpoint files are
reused by vLLM-Omni.

Prepare the image and cache with `just build` and `just download` from
`tts_engines/voxcpm2_longform`. The download target enables networking only for the
cache-population container and defaults to `/srv/ai/models/voxcpm2`.
Set `VOXCPM_HF_CACHE` to your configured host cache directory if different,
for example `VOXCPM_HF_CACHE=~/.cache/radio-drama/voxcpm2-huggingface just download`
when using `tts.toml.example`. Existing blobs are reused; the target fetches
any missing files and updates the cached snapshot metadata.
