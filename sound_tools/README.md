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

For example, after `just run`:

```python
generate("Rain tapping on a tent roof", "rain.wav", seconds=12)
```

The GenAU model is intended for ambient sounds rather than speech or music.
