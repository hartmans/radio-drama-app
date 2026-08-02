# MOSS-TTSD engine

This engine renders each radio-drama script as one MOSS-TTSD continuation
conversation.  It batches complete scripts, not individual lines, so that the
model retains dialogue context and speaker identity throughout a script.

## Prompting

MOSS-TTSD v1.0's documented conditioning surface is deliberately small:

* A reference recording for every speaker.
* The exact transcript of each reference recording, prefixed with the speaker
  tag that identifies it.
* The tagged dialogue to generate.

The engine supplies the `[S1]`, `[S2]`, and subsequent tags itself, maps them
to the speakers in a script, and obtains the reference transcripts from the
radio-drama voice configuration.  Accurate transcripts are important: MOSS
uses them as part of the continuation prefix for cloning, rather than as style
instructions.  Write normal dialogue and punctuation in the script; they are
passed to MOSS as the generated text.

The published TTSD v1.0 documentation does **not** describe inline controls
for emotion, speaking rate, duration, pauses, pronunciation, or a per-line
language argument.  In particular, do not rely on `[pause 3.2s]`, language
arguments, Pinyin/IPA pronunciation controls, or token-duration controls here.
Those are documented for the separate single-speaker MOSS-TTS v1.5 model, not
for MOSS-TTSD.  Literal use of such markers in a radio-drama script is therefore
ordinary text, not a supported control protocol.

MOSS-TTSD v1.0 supports 20 languages and cross-lingual cloning, but this engine
does not add a language hint: the authored dialogue text is the model input.
The upstream documentation describes one to five speakers.  This adapter does
not impose its own speaker limit; if a model version rejects a larger cast, its
error is returned to the caller.

## Engine settings

These environment variables tune inference rather than add prompt syntax:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MOSS_TTSD_BATCH_SIZE` | `10` | Complete scripts generated in one model batch. |
| `MOSS_TTSD_MAX_NEW_TOKENS` | `4096` | Maximum audio-token output per script (roughly 12.5 tokens per second). |
| `MOSS_TTSD_TEMPERATURE` | `1.1` | Audio sampling temperature. |
| `MOSS_TTSD_TOP_P` | `0.9` | Audio nucleus-sampling threshold. |
| `MOSS_TTSD_TOP_K` | `50` | Audio top-k sampling limit. |
| `MOSS_TTSD_REPETITION_PENALTY` | `1.1` | Audio repetition penalty. |

The four sampling defaults are MOSS's published recommendations for TTSD.

## Sources

* [MOSS-TTSD v1.0 documentation](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_ttsd_model_card.md)
* [MOSS-TTS v1.5 documentation](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_model_card.md) — controls listed there are intentionally not claimed for this engine.
