"""Fail-fast validation at scientific and hardware boundaries."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import numpy as np


def finite(
    name: str, value: Any, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        x = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and x < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and x > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return x


def integer(name: str, value: Any, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def numeric_array(name: str, value: Any, ndim: int | None = None) -> np.ndarray:
    x = np.asarray(value)
    if x.dtype.kind not in "biuf" or not x.size or (not np.isfinite(x).all()):
        raise ValueError(f"{name} must be a nonempty finite numeric array")
    if ndim is not None and x.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    return x


def binary(name: str, value: Any) -> np.ndarray:
    x = numeric_array(name, value, 1)
    if not np.isin(x, [0, 1]).all():
        raise ValueError(f"{name} must be binary")
    return x.astype(int)
