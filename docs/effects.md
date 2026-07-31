# Effect Functions Documentation

This document describes the effect chain functions available in the radio-drama-app backend. Each effect can be composed into chains using the pipeline operator (`|`).

Audio-producing XML elements may also set `effect="..."` to apply a restricted
effect-chain expression to that element's rendered audio. The node applies its
`gain` automation first, then `effect`, then `pan`; this processing is separate
from any compose-local `preset` bus. The expression may use the production's
built-in and document-defined preset names plus documented effect functions.
`gain(line(...))` and `pan(line(...))` are available within these expressions;
both take an array expression, so automation stays explicit and composable.

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

## Other Effects

TBD: Document additional effect functions as they are developed.
