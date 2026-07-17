from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Mapping

import numpy as np


_ALLOWED_BINARY_OPERATORS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
)
_ALLOWED_UNARY_OPERATORS = (
    ast.UAdd,
    ast.USub,
)


@dataclass(frozen=True, slots=True)
class ArrayExpression(ABC):
    @abstractmethod
    def to_size(self, frame_count: int) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class LineExpression(ArrayExpression):
    points: tuple[tuple[int, float], ...] = ()
    end: float | None = None

    def __post_init__(self) -> None:
        previous_frame: int | None = None
        for frame, _ in self.points:
            if previous_frame is not None and frame <= previous_frame:
                raise ValueError("line frame indexes must be strictly increasing")
            previous_frame = frame

    def to_size(self, frame_count: int) -> np.ndarray:
        if frame_count <= 0:
            return np.zeros(0, dtype=np.float32)

        result = np.zeros(frame_count, dtype=np.float32)
        if not self.points:
            result.fill(np.float32(0.0 if self.end is None else self.end))
            return result

        points = list(self.points)
        if points[0][0] > 0:
            points.insert(0, (0, 0.0))

        for point_index, (start_frame, start_value) in enumerate(points):
            if point_index + 1 < len(points):
                end_frame, end_value = points[point_index + 1]
                if end_frame <= 0:
                    continue
                if start_frame >= frame_count:
                    break
                segment_start = max(start_frame, 0)
                segment_end = min(end_frame, frame_count)
                segment_length = end_frame - start_frame
                if segment_length > 0 and segment_end > segment_start:
                    frames = np.arange(segment_start, segment_end, dtype=np.float32)
                    result[segment_start:segment_end] = np.float32(start_value) + (
                        (frames - start_frame) / segment_length
                    ) * np.float32(end_value - start_value)
                continue

            if start_frame >= frame_count:
                break
            segment_start = max(start_frame, 0)
            result[segment_start:] = np.float32(start_value)
            if self.end is None or segment_start >= frame_count:
                break
            segment_length = frame_count - start_frame
            if segment_length <= 0:
                break
            frames = np.arange(segment_start, frame_count, dtype=np.float32)
            result[segment_start:] = np.float32(start_value) + (
                (frames - start_frame) / segment_length
            ) * np.float32(self.end - start_value)

        return result


def line(*values) -> LineExpression:
    if len(values) == 1 and isinstance(values[0], Real):
        return LineExpression(end=float(values[0]))

    if not values:
        raise TypeError("line requires either a number or frame/value arguments")
    if len(values) < 2:
        raise ValueError("line requires frame/value pairs")

    end: float | None = None
    pair_values = values
    if len(values) % 2 != 0:
        pair_values = values[:-1]
        end = float(values[-1])
    if len(pair_values) % 2 != 0 or not pair_values:
        raise ValueError("line requires frame/value pairs with an optional final end value")

    points = []
    for index in range(0, len(pair_values), 2):
        frame_value = pair_values[index]
        if not isinstance(frame_value, Real) or not float(frame_value).is_integer():
            raise ValueError("line frame indexes must be integers")
        points.append((int(frame_value), float(pair_values[index + 1])))
    return LineExpression(points=tuple(points), end=end)


def validate_expression(text: str) -> ast.Expression:
    parsed = ast.parse(text, mode="eval")
    _validate_node(parsed)
    return parsed


def eval_expression(
    text: str,
    variables: Mapping[str, object],
    return_type,
):
    parsed = validate_expression(text)
    return return_type(
        eval(  # noqa: S307
            compile(parsed, "<expression>", "eval"),
            {"__builtins__": {}, "line": line, "min": min, "max": max},
            dict(variables),
        )
    )


def coerce_array_exp(value) -> ArrayExpression:
    if isinstance(value, ArrayExpression):
        return value
    if isinstance(value, Real):
        return line(float(value))
    raise TypeError(f"Expected an ArrayExpression or number, got {type(value).__name__}")


def coerce_real(value) -> float:
    if isinstance(value, Real):
        return float(value)
    raise TypeError(f"Expected a real number, got {type(value).__name__}")


def _validate_node(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate_node(node.body)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, Real):
            return
        raise ValueError("Only numeric constants are allowed in expressions")
    if isinstance(node, ast.Name):
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINARY_OPERATORS):
            raise ValueError("Unsupported binary operator in expression")
        _validate_node(node.left)
        _validate_node(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY_OPERATORS):
            raise ValueError("Unsupported unary operator in expression")
        _validate_node(node.operand)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed in expressions")
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed in expressions")
        for arg in node.args:
            _validate_node(arg)
        return
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")
