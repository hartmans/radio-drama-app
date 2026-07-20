from __future__ import annotations

import numpy as np
import pytest

from radio_drama.expressions import (
    ArrayExpression,
    LineExpression,
    coerce_array_exp,
    eval_expression,
    line,
    validate_expression,
)


def test_line_expression_interpolates_from_implicit_zero() -> None:
    expression = line(2, 1.0, 4, 0.0)

    np.testing.assert_allclose(
        expression.to_size(6),
        np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.0], dtype=np.float32),
    )


def test_line_expression_uses_end_value_when_present() -> None:
    expression = line(2, 1.0, 0.0)

    np.testing.assert_allclose(
        expression.to_size(5),
        np.array([0.0, 0.5, 1.0, 2.0 / 3.0, 1.0 / 3.0], dtype=np.float32),
        atol=1e-6,
    )


def test_line_expression_truncates_before_end_value_takes_effect() -> None:
    expression = line(2, 1.0, 6, 3.0, -1.0)

    np.testing.assert_allclose(
        expression.to_size(4),
        np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
    )


def test_line_expression_constant_expands_to_requested_size() -> None:
    expression = line(0.25)

    np.testing.assert_allclose(
        expression.to_size(4),
        np.full(4, 0.25, dtype=np.float32),
    )


def test_line_expression_allows_out_of_range_frames_and_clips() -> None:
    expression = line(-1, 0.0, 1, 1.0, 6, 0.0)

    np.testing.assert_allclose(
        expression.to_size(4),
        np.array([0.5, 1.0, 0.8, 0.6], dtype=np.float32),
        atol=1e-6,
    )


def test_line_expression_rejects_invalid_frames() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        line(1, 0.0, 1, 1.0)
    with pytest.raises(ValueError, match="integers"):
        line(1.5, 0.0)
    with pytest.raises(TypeError, match="frame/value arguments"):
        line()


def test_eval_expression_supports_line_names_and_arithmetic() -> None:
    expression = eval_expression(
        "line(mark, -1, mark + 2, amount * 2, amount - 1)",
        {"mark": 1.0, "amount": 0.5},
        lambda value: value,
    )

    assert isinstance(expression, LineExpression)
    np.testing.assert_allclose(
        expression.to_size(5),
        np.array([0.0, -1.0, 0.0, 1.0, 0.25], dtype=np.float32),
        atol=1e-6,
    )


def test_eval_expression_supports_min_and_max() -> None:
    assert eval_expression(
        "min(low, 3, max(2, high))",
        {"low": 1.0, "high": 5.0},
        lambda value: value,
    ) == 1.0
    assert eval_expression(
        "max(low, 3, min(2, high))",
        {"low": 1.0, "high": 5.0},
        lambda value: value,
    ) == 3


def test_eval_expression_supports_effect_chain_syntax() -> None:
    calls = []

    class Chain:
        def __or__(self, other):
            return Chain()

    def factory(*, mode: str, taps: tuple[tuple[float, ...], ...]):
        calls.append((mode, taps))
        return Chain()

    result = eval_expression(
        'left | factory(mode="room", taps=((1.0, 0.5), (2.0, 0.25)))',
        {"left": Chain(), "factory": factory},
        lambda value: value,
    )

    assert isinstance(result, Chain)
    assert calls == [("room", ((1.0, 0.5), (2.0, 0.25)))]


def test_validate_expression_rejects_keyword_unpacking() -> None:
    with pytest.raises(ValueError, match="Keyword argument expansion"):
        validate_expression("factory(**options)")


def test_validate_expression_rejects_unsupported_nodes() -> None:
    with pytest.raises(ValueError, match="Unsupported expression node: List"):
        validate_expression("line([1, 2])")
    with pytest.raises(ValueError, match="Unsupported expression node"):
        validate_expression("[value for value in values]")
    with pytest.raises(ValueError, match="Only direct function calls"):
        validate_expression("factory()(1)")


def test_coerce_array_exp_wraps_numbers() -> None:
    coerced = coerce_array_exp(3.0)
    assert isinstance(coerced, ArrayExpression)
    np.testing.assert_allclose(
        coerced.to_size(3),
        np.full(3, 3.0, dtype=np.float32),
    )
