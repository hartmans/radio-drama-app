# Expressions

This document describes the expression syntax available to people authoring
radio-drama documents.

The short version:

* `gain` and `pan` accept Python-style expressions
* expressions can refer to marks and layout values made available by the node
  they are written on
* invalid syntax or unknown names fails as an ordinary error instead of being
  silently ignored

## Syntax

Expressions are written as a single Python expression. They support:

* numeric literals such as `1`, `-3.5`, and `0.25`
* names such as `natural_length`, `intro`, or `outer_cue`
* unary `+` and `-`
* binary `+`, `-`, `*`, `/`, `**`, and `%`
* function calls, currently only `line(...)`

Expressions do not support:

* attribute access
* indexing
* comprehensions
* lambdas
* keyword arguments
* chained call tricks such as `factory()(1)`

If you use a name that is not available in the current scope, evaluation fails
with the ordinary Python `NameError`.

## `line(...)`

`line(...)` is the main helper for gain ramps and pan sweeps.

Common forms:

* `line(value)`
  A constant value across the whole span.
* `line(frame_1, value_1, ..., frame_n, value_n)`
  A piecewise-linear ramp through the given control points.
* `line(frame_1, value_1, ..., frame_n, value_n, end_value)`
  Like the previous form, but the final value is held to the end of the span.

Use it when you want audio to change gradually instead of jumping.

Examples:

```python
line(0)
line(0, -12, natural_length, -3)
line(0, -1, door_open, 0)
```

The frame positions are sample-frame positions in the current expression
context. They may be negative or extend past the current node span when that is
useful for a ramp that starts before the visible section or continues after it.

## Gain

`gain` is evaluated after the audio for a node has been rendered.

It is typically written in decibels. Positive values boost, negative values
attenuate.

Useful names in gain expressions include:

* `natural_length`
  The node's natural span in sample frames.
* visible marks
  Marks exposed by the node's render-time scope, such as `intro`, `verse_end`,
  or `door_open`.

Examples:

```python
gain="line(0, -6, natural_length, 0)"
gain="-3"
```

The expression engine does not provide conditional syntax, so for more complex
shapes use piecewise ramps with `line(...)`.

## Pan

`pan` is evaluated with the same expression syntax as `gain`.

Pan values are expected to fall in the range `-1` to `1`:

* `-1` means full left
* `0` means center
* `1` means full right

Values outside that range are clipped when used.

Typical uses:

* keep a voice centered with `pan="0"`
* drift a source from left to right with `pan="line(-1, 1)"`
* pan a cue around an event mark with `pan="line(0, -1, cue, 1)"`

## Available names

The exact names available depend on where the expression is written.

Common names for authored expressions include:

* `natural_length`
  The node's span in sample frames.
* `start`
  The node's resolved start position when placement is explicit.
* `first`
  The first rendered sample position after left-side placement.
* `last`
  The last rendered sample position after left-side placement.
* `inner_<mark>`
  A mark already known in the node's own inner geometry.
* `outer_<mark>`
  A mark visible in the containing scope.

For render-time automation, mark names are exposed directly as names in the
current scope.

## Practical notes

* document authors should think in sample frames, not seconds, when writing
  expressions
* use marks when you want a control point to follow a specific event in the
  script or audio timeline
* if you need a simple constant, just write a number
* if you need a ramp, use `line(...)`
