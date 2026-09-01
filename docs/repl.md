# Radio-drama REPL

Run the full Python REPL with an optional production document:

```console
python -m radio_drama.repl [production.xml]
```

The namespace contains ordinary Python builtins and locals, the functions and
presets available in effect expressions, and the REPL helpers described below.
In particular, `line`, `min`, and `max` are available alongside effect presets.

## Loading and selecting plans

`load()` plans a production without laying it out or rendering it and returns an
`AudioPlanWrapper`:

```python
p = load("show.xml")
```

Integer indexing selects one direct audio-plan child without rendering its
container:

```python
scene = p[2]
cue = scene[1]
```

A slice has deliberately different semantics. It is a faithful timeline crop
of the source wrapper spanning the selected direct children:

```python
opening = p[0:3]
```

The crop:

* lays out the source to find the earliest and latest selected child samples;
* renders the complete source with its preset buses, gain, pan, effects, and
  wrapper-local effect chain; and
* returns the portion between those timeline bounds.

Consequently, an unselected sibling that overlaps the selected time interval is
audible in the crop. The first crop can also be expensive because faithfully
rendering the source may render children outside the slice. The source plan's
memoized render makes later crops cheaper.

Use integer selection followed by `+` or `mix()` when only particular children
should render.

## Concatenation and mixing

`+` constructs a new lazy timeline that renders the two operand wrappers
consecutively:

```python
edited = p[0] + p[3]
```

`mix()` renders every operand independently and overlaps them at time zero:

```python
layered = mix(p[0], p[3], sound("rain"))
```

Neither operation reuses core `ComposeAudioPlan` placement. They use a
REPL-local plan so the original production plans remain reusable and their
layout state is not mutated. Indexing one of these results returns its original
operand wrapper:

```python
(a + b)[0] is a
mix(a, b)[1] is b
```

Mixing is a linear sum and does not normalize automatically.

## Effects

Pipe an effect or preset into a wrapper to create a new wrapper while retaining
the original:

```python
phone_cue = cue | phone
room_cue = cue | indoor1
```

Effects use `|` for sequential processing. `*` scales a stage's output with a
number or `line(...)` control, and `+` sums independently copied branches:

```python
parallel_room = dry() * 0.4 + indoor1 * line(0, 0, 2 * s, 0.6)
processed = cue | parallel_room
```

Effect multiplication uses linear amplitude. `gain(...)` remains decibel gain;
both constants such as `gain(3)` and automation such as `gain(line(...))` are
accepted.

## Layout and marks

Layout remains lazy. Run it explicitly when desired:

```python
cue.layout()
```

Mark access performs layout automatically and supports item or attribute form:

```python
cue.marks["door_open"]
cue.marks.door_open
```

## Playback

Both playback forms return a `Future` immediately:

```python
future = processed | play()
future = play(processed)
```

Call `future.result()` to wait for completion or surface an output error. New
playback supersedes pending or active playback. `stop()` cancels pending
rendering for playback and terminates active output.

Playback uses PulseAudio directly, honors `PULSE_SERVER`, and drains the Pulse
stream before reporting completion.
