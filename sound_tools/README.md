# Interactive sound tools

These tools run independently in Podman containers, so their model packages
cannot affect the radio-drama application or each other.  Each starts a Python
REPL with a resident-model `generate(...)` helper.  It writes an explicit WAV
under the `/audio` mount and then tries to play it through PulseAudio; playback
errors leave the generated file intact.

Run each recipe from its tool directory.  `just build` builds the image, then
`just download` performs the one networked download of its weights.  `just run`
starts the REPL with `--network=none`, a GPU, the current tool directory as
`/audio`, and the current user's PulseAudio socket.

* `moss_soundeffect_v2` uses MOSS-SoundEffect v2 for effects and soundscapes.
  Its Hugging Face cache is `/srv/ai/huggingface-cache`, mounted at
  `/models/huggingface`.
* `genau` uses GenAU for ambient sounds.  Its project files and checkpoints are
  `/srv/ai/models/genau`, including its Hugging Face and Torch Hub caches.  The
  supplied checkpoint generates native 16 kHz audio.
* `moss_voice_generator` uses MOSS-VoiceGenerator for text plus a free-form
  voice-description instruction.  It uses `/srv/ai/huggingface-cache`.
* `seed_vc` converts the performance in a source recording to the voice in a
  reference recording. Both v1 and v2 models are staged into the image from a
  persistent Podman build cache.

For example, after `just run`:

```python
generate("Rain tapping on a tent roof", "rain.wav", seconds=12)
```

The GenAU model is intended for ambient sounds rather than speech or music.

Seed-VC is a command-line tool rather than an interactive generator. Its build
downloads both speech and F0-conditioned models, reusing the downloads across
later image rebuilds. Pass the upstream inference controls through its wrapper:

```console
cd sound_tools/seed_vc
just build
just run --source source.wav --target reference.wav --output converted
just run-v2 --source source.wav --target reference.wav --output converted-v2
```

The source and reference are mounted read-only, the output directory is mounted
writable, and inference runs without network access. Use `python3 seed_vc.py
--help` for pitch conditioning, diffusion, and length controls.
Use `python3 seed_vc_v2.py --help` for v2's intelligibility, similarity,
sampling, style-conversion, and anonymization controls.
