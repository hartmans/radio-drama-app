# REPL design and implementation notes

User-facing syntax and behavior are documented in `docs/repl.md`.

The radio-drama REPL is a full Python REPL. It is intentionally a consumer of
the normal document, planning, rendering, effect, and dependency-injection
interfaces rather than an alternative effect evaluator or planning path.

## Runtime ownership

The interactive console runs on the caller's thread. A companion thread owns an
asyncio event loop and the `radio_drama_injector()` associated with that loop.
Synchronous REPL entry points submit coroutines with
`asyncio.run_coroutine_threadsafe()`. Plans and resources are constructed and
used on their injector's loop; blocking DSP and output work can use the loop's
normal executor boundaries.

Each loaded document gets a production injector on the same companion loop.
Passing its document path through `radio_drama_injector()` also selects the
normal `<production>.wav.cache` resource-cache root. Older injectors remain
alive until the session closes so wrappers from earlier loads retain the
resources against which they were constructed.

## Wrapped plans

`AudioPlanWrapper` is an immutable interactive value containing:

* a normal `AudioPlan`;
* the production sample rate;
* the submission boundary for the injector loop;
* a wrapper-local `EffectStage`, initially `dry()`; and
* an optional layout root shared by child wrappers.

The wrapper-local chain is separate from document-authored effects on the plan.
`wrapper | stage` uses `dataclasses.replace()` to return the same wrapper class
with the additional stage composed onto its local chain. Added fields on
dataclass subclasses are consequently preserved. Rendering copies the plan's
memoized result before applying the wrapper-local chain, so derived wrappers can
reuse one plan without modifying one another's input.

Rendering is lazy. `load()` performs document parsing and planning and returns a
wrapper around the resulting `ProductionPlan`, but does not lay out or render
it. `sound()` similarly constructs and wraps a normal `SoundPlan` without
rendering it.

## Layout, children, and marks

`wrapper.layout()` explicitly runs layout on the injector loop, waits for it,
and returns the wrapper. Wrapper rendering always ensures layout first.

`wrapper[index]` selects a direct `AudioPlan` child and returns a dry wrapper for
that child. The child wrapper retains the original layout root, because parent
layout establishes placement and incoming mark scope needed by descendants.

`wrapper[start:stop]` constructs a lazy crop of the fully rendered source
timeline spanning the selected child bounds. Rendering the source preserves its
preset buses and outer processing and intentionally includes other siblings
that intersect the crop interval.

`wrapper_a + wrapper_b` constructs a REPL-local concatenation plan, while
`mix(wrapper_a, wrapper_b, ...)` overlaps all wrappers at time zero. These plans
render their input wrappers independently rather than passing shared underlying
plans through core `ComposeAudioPlan` layout again. This keeps REPL composition
from mutating placement on reusable production plans.

`wrapper.marks[name]` and `wrapper.marks.name` expose the wrapped plan's local
`mark_positions`. The first lookup ensures layout is complete. Missing item
lookups raise `KeyError`; missing attribute lookups raise `AttributeError`.

## Playback

`play(wrapper)` and `wrapper | play()` render lazily and send float audio to an
isolated libpulse-simple helper process. The helper honors settings such as
`PULSE_SERVER` and calls `pa_simple_drain()` before exiting so Pulse plays the
complete buffer. Playback uses generation tokens: the newest request cancels
any pending predecessor, terminates its output process, and prevents an older
render from starting output later. `stop()` cancels pending rendering and
terminates active output.

## Interactive namespace

The namespace contains ordinary Python locals plus all functions and presets
valid in effect expressions, including `line`, `min`, and `max`. `load`,
`sound`, `mix`, `play`, and `stop` are normal Python callables. Completion
combines Python name and attribute completion with generic filesystem
completion rooted at the working directory and the loaded document's adjacent
`sounds/` directory.
