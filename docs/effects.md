# Effect Functions Documentation

This document describes the effect chain functions available in the radio-drama-app backend. Each effect can be composed into chains using the pipeline operator (`|`).

Effect stages also support parallel algebra. `stage * control` and
`control * stage` process the current buffer and scale the result by a
linear-amplitude number or array expression. `stage_a + stage_b` gives each
stage its own copy of the same original input and sums their results. Thus:

```python
# Equal dry/effected parallel branches.
dry() * 0.5 + indoor1 * 0.5

# Fade a processed branch in over two seconds.
dry() + indoor1 * line(0, 0, 2 * s, 1)

# Complementary wet and dry ramps using one expression-local binding.
phone * (ramp := line(0, 0, 2 * s, 1)) + dry() * (1 - ramp)
```

Addition is not sequential processing; use `|` when the output of one stage
should feed the next. Parallel addition is linear and is not automatically
normalized, so leave headroom when branch controls sum above 1.

Audio-producing XML elements may also set `effect="..."` to apply a restricted
effect-chain expression to that element's rendered audio. The node applies its
`gain` automation first, then `effect`, then `pan`; this processing is separate
from any compose-local `preset` bus. The expression may use the production's
built-in and document-defined preset names plus documented effect functions.
`gain(...)` and `pan(...)` accept either numbers or array expressions, so both
`gain(3)` and automated forms such as `gain(line(...))` are legal.

---

## early_reflections

Adds discrete echoes that simulate the first sound bounces from room surfaces (walls, ceiling, floor) before the diffuse reverb tail kicks in.

### What it sounds like
- Adds spatial context without making it sound "echoey" or "in a cave"
- Gives the illusion of physical space around a voice
- Each tap is a distinct slapback delay, not a continuous wash
- Used for subtle ambience on internal monologue or to place a character in a specific environment

### Tap Configuration

Each tap is a tuple: `(delay_ms, left_gain, right_gain)`

```python
early_reflections(
    taps=((9.0, 0.09, 0.12), (18.0, 0.07, 0.05), (31.0, 0.04, 0.06)), 
    dry_mix=0.96
)
```

**Tap parameters:**

| Value | Meaning |
|-------|---------|
| `delay_ms` | Delay in milliseconds before the reflection arrives |
| `left_gain` | Volume multiplier for left channel (0 = silent, 1 = full volume) |
| `right_gain` | Volume multiplier for right channel |

**How it works:**
1. Takes mono source from center of stereo signal
2. Creates delayed copies at each tap's delay time
3. Pans reflections by applying different gains to left/right channels
4. Mixes with original via `dry_mix` (0.96 = 96% dry, 4% wet)

### Examples

Small room/close space:
```python
early_reflections(taps=((9.0, 0.09, 0.12), (18.0, 0.07, 0.05), (31.0, 0.04, 0.06)), dry_mix=0.96)
```

Medium room:
```python
early_reflections(taps=((15.0, 0.16, 0.1), (28.0, 0.1, 0.16), (42.0, 0.07, 0.08), (63.0, 0.05, 0.05)), dry_mix=0.9)
```

Very small space/internal monologue:
```python
early_reflections(taps=((24.0, 0.04, 0.05), (46.0, 0.03, 0.025)), dry_mix=0.99)
```

### Configuration tips

1. **Delay spacing**: Keep initial reflections under 50ms for intimate spaces, go longer (60-100ms+) for larger rooms
2. **Gain structure**: Later reflections should be quieter than earlier ones (-3 to -6 dB per tap is natural)
3. **Stereo panning**: Use different left/right gains to simulate off-center bounces (e.g., stronger on one side)
4. **dry_mix**: Keep dry signal dominant (0.9-0.99). Early reflections should supplement, not dominate

The effect takes the first 5+ milliseconds of silence and fills it with spatial information that tricks the ear into perceiving a physical space.

---

## feedback_reverb

Adds a finite series of progressively quieter delayed copies. The left and
right copies use slightly different delay times, giving the repeats stereo
width. This is a compact echo-like ambience rather than a physical room model.

```python
feedback_reverb(
    delay_ms=73.0,
    stereo_offset_ms=13.0,
    feedback=0.55,
    repeats=5,
    wet_mix=0.10,
    dry_mix=0.97,
)
```

`wet_mix` sets the linear amplitude of the first delayed copy; each later copy
is multiplied by `feedback` again. `dry_mix` independently sets the linear
amplitude of the original. They are mixing coefficients and do not need to sum
to one. The effect does not extend the audio buffer, so repeats that would land
after its end are discarded.

---

## modulated_delay

Mixes the input with a short delayed copy whose delay time moves sinusoidally.
Fractional delay positions are linearly interpolated. On stereo audio, the
right-channel oscillator is phase-offset from the left, creating slow movement
in both pitch and apparent position without moving the dry signal.

```python
modulated_delay(
    delay_ms=21.0,
    depth_ms=4.5,
    rate_hz=0.17,
    wet_mix=0.20,
    dry_mix=0.90,
    stereo_phase_degrees=110.0,
)
```

| Parameter | Meaning |
|-----------|---------|
| `delay_ms` | Center delay in milliseconds |
| `depth_ms` | Maximum movement on either side of the center delay; it may not exceed `delay_ms` |
| `rate_hz` | Oscillator cycles per second |
| `wet_mix` | Linear amplitude of the moving delayed copy |
| `dry_mix` | Linear amplitude of the original signal |
| `stereo_phase_degrees` | Right oscillator's phase offset from the left; defaults to 90 degrees |
| `phase_degrees` | Deterministic starting phase of the left oscillator; defaults to 0 degrees |

`wet_mix` and `dry_mix` are independent mixing coefficients rather than a
crossfade, so they do not need to sum to one. Values around 15–25 ms of delay,
3–6 ms of depth, and 0.1–0.25 Hz produce a slow dreamlike drift. Faster rates
and shorter delays sound more like a conventional chorus or vibrato.

Like the other effect stages, `modulated_delay` preserves the input length. Its
initial delayed samples therefore contain only the scaled dry signal.

---

## pan

Applies an automated stereo balance to rendered stereo audio. `pan` takes an
number or array expression whose values range from `-1` (hard left), through `0`
(center), to `1` (hard right). Values outside that range are clamped.

The favored channel remains present while the opposite channel is attenuated.
The resulting pair of gains is normalized so that its summed squared gain
(stereo power) stays equal to the centered signal. This keeps the perceived
loudness approximately stable as the balance moves.

At center, both channels retain a gain of `1.0`. At a hard pan, the remaining
channel has a gain of `sqrt(2)` (approximately `1.414`, or +3 dB), while the
opposite channel is silent. Account for that additional headroom when setting
gain or mastering a production.

### Examples

Keep a sound centered:

```python
pan(line(0))
```

Move from left to right over one second:

```python
pan(line(0 * s, -1, 1 * s, 1))
```

---

## gain

Applies decibel gain controlled by a number or array expression. Positive
values boost and negative values attenuate.

```python
gain(3)
gain(line(0, -12, 2 * s, 0))
```

---

## compress_audio

Applies Pedalboard's dynamic-range compressor to the complete audio buffer,
then applies optional makeup gain. Stereo input is processed as one plugin
stream; the compressor's channel-linking and envelope behavior are defined by
Pedalboard and may differ from a sample-by-sample peak compressor.

```python
compress_audio(
    threshold_db=-28.0,
    ratio=2.8,
    attack_ms=5.0,
    release_ms=240.0,
    makeup_db=2.2,
)
```

| Parameter | Meaning |
|-----------|---------|
| `threshold_db` | Level in dB above which gain reduction begins |
| `ratio` | Compression ratio; it must be positive |
| `attack_ms` | Time for compression to engage after the signal crosses the threshold |
| `release_ms` | Time for compression to disengage after the signal falls below the threshold |
| `makeup_db` | Gain in dB applied after compression; defaults to 0 dB |

Lower thresholds and higher ratios produce stronger compression. Very short
attack times control peaks more aggressively, while longer attacks retain more
of a sound's initial transient. Makeup gain raises both the compressed signal
and its noise floor, so leave headroom for later mixing and mastering.

---

## dry

Returns the input unchanged. `dry()` is primarily useful as the unprocessed
branch of `crossfade()`; including it in an ordinary sequential pipeline has no
effect.

```python
crossfade(dreams, dry(), line(0, -1, 4 * s, 1))
```

---

## crossfade

Processes two branches from the same original input and combines their results
with a frame-varying linear crossfade:

```python
crossfade(
    constrained_dream,
    dry(),
    line(0, -1, 4 * s, 1),
    a_mix=0.8,
    b_mix=1.0,
)
```

| Parameter | Meaning |
|-----------|---------|
| `stage_a` | Effect stage or preset selected at position `-1` |
| `stage_b` | Effect stage or preset selected at position `1` |
| `position` | Number or array expression; values are clipped to `[-1, 1]` |
| `a_mix` | Optional linear-amplitude control for branch A; defaults to 1 |
| `b_mix` | Optional linear-amplitude control for branch B; defaults to 1 |

At position `0`, each branch has a weight of `0.5` before its mix control is
applied. The crossfade is linear rather than equal-power, so two identical
branches reconstruct the original level throughout the transition. Both
branches always receive the same unmodified input, regardless of their order.
`position`, `a_mix`, and `b_mix` may all be constants or array expressions.

---

## equalizer

Splits audio into complementary frequency bands at ordered crossover
frequencies, applies a decibel control to each band, and recombines them. Its
arguments alternate between a cutoff frequency and the gain for the band below
that cutoff, followed by one final gain for the highest band.

```python
# Suppress content below 180 Hz and above 3200 Hz.
equalizer(180, -99, 3200, 0, -99)
```

The example has three bands: below 180 Hz at -99 dB, 180–3200 Hz at 0 dB,
and above 3200 Hz at -99 dB. Cutoffs must be positive, strictly increasing,
and below the output sample rate's Nyquist frequency. The optional `order`
keyword controls the Butterworth crossover order and defaults to 2.

Band gains accept constants or array expressions. This permits a fixed
crossover layout to open or close over time without changing filter
coefficients:

```python
equalizer(
    180, line(0, -30, 4 * s, 0),
    2800, 0,
    line(0, -30, 4 * s, 0),
)
```

With every band at 0 dB, the complementary bands reconstruct the original
signal. Very low gains such as -99 dB are effectively silent but remain finite.

---

## Other Effects

TBD: Document additional effect functions as they are developed.
