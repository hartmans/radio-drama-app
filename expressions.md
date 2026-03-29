# Expressions

This document describes the expression language used by radio-drama audio
attributes and the variable scopes available to those expressions.

## Language

Expressions are parsed with `ast.parse(..., mode="eval")`, validated against a
small whitelist, and then evaluated with:

* no Python builtins
* one global helper: `line(...)`
* a caller-supplied locals dictionary

Allowed syntax:

* numeric constants
* names
* unary `+` and `-`
* binary `+`, `-`, `*`, `/`, `**`, `%`
* list and tuple literals
* direct function calls such as `line(...)`

Disallowed syntax includes:

* attribute access
* comprehensions
* lambdas
* keyword arguments
* indirect calls such as `factory()(1)`

If a name is not present in the locals dictionary, evaluation fails in the
ordinary Python way with `NameError`. The evaluator does not populate special
placeholder variables just to improve that error.

## `line(...)`

`line(...)` builds an `ArrayExpression` from piecewise-linear control points.

Supported forms:

* `line(number)`
  Returns a constant expression.
* `line([frame_1, value_1, ..., frame_n, value_n])`
  Returns a piecewise-linear expression.
* `line([frame_1, value_1, ..., frame_n, value_n], end_value)`
  Uses `end_value` as the virtual point at the requested output size.

Rules:

* frame indexes must be integers
* frame indexes must be strictly increasing
* frame indexes may be negative
* frame indexes may be greater than the requested output size
* out-of-range control points are clipped or truncated when expanded

`to_size(frame_count)` returns one contiguous `float32` numpy array of length
`frame_count`.

## Return coercion

The evaluator is parameterized by a return-type coercion function.

Current coercions:

* `coerce_array_exp`
  Accepts an `ArrayExpression` directly or wraps a plain number as
  `line(number)`.
* `coerce_real`
  Accepts one real scalar and returns it as `float`.

## Where expressions are used

Current authored expression attributes:

* `pan`
* `start`
* `end`

Current numeric timing attributes:

* `pre_gap`
* `post_gap`
* `length`

Those numeric timing attributes are still parsed as numbers at the document
boundary today, but the layout code uses the same expression-helper machinery
internally when resolving left-side and right-side placement.

## Render-time scope

`pan` is evaluated during `AudioPlan.post_render()`.

Locals available there:

* `natural_length`
  The node's natural render/control span in sample frames.
* one variable per visible render-time mark
  These come from `audio_marks_render`, also in sample-frame coordinates.

Render-time mark coordinates use natural sample geometry:

* `0 == inner_first`
* a mark may be negative
* a mark may be greater than `natural_length`

This is intentional. Automation should be able to refer to marks that lie
before or after the node's natural render span.

Example:

```python
line([door_open, -1, door_open + 24000, 0])
```

## Layout helper scopes

The layout code exposes two helper variable environments. These are part of the
layout contract even where the current document syntax does not yet use every
one of them directly.

### Left-side scope

Used when resolving the node's left-side placement, meaning `start` or
`pre_gap`.

Locals available:

* `natural_length`
  The node's intrinsic natural span before outer placement.
* `inner_<mark>`
  A visible mark already known in the node's inner geometry.
* `outer_<mark>`
  A visible mark in the containing scope when explicit outer placement is being
  resolved.

### Right-side scope

Used when resolving the node's right-side placement, meaning `end`, `length`,
or `post_gap`.

Locals available:

* everything from the left-side scope
* `start`
  The node's resolved outer-geometry start when explicit placement is active.
* `first`
  The node's first rendered sample in outer geometry after left-side placement.
* `last`
  The node's last rendered sample in outer geometry after left-side placement.

The key design point is that right-side placement can refer to `last` without
becoming recursive, because left-side placement is resolved first.

## Mark namespaces

Positioning expressions always use explicit mark namespaces:

* `inner_<mark>`
* `outer_<mark>`

Unprefixed mark names are not populated automatically.

## Current limitations

Deliberate current limitations:

* the only array helper is `line(...)`
* timing expressions are not yet fully generalized at the document boundary
* inner scalar values such as `inner_first`, `inner_last`, and `inner_length`
  are intentionally not exposed as authored names
