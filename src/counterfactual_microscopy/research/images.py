"""Radiometry-preserving features, integer registration, and verification evidence.

No per-image min/max normalization: that would erase brightness shortcuts. Integer
images use their detector dtype scale; float images must declare a common [0,1] scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .validation import finite, numeric_array

FEATURE_NAMES = (
    "mean",
    "std",
    "q10",
    "q50",
    "q90",
    "q99",
    "gradient",
    "laplacian",
    "bright_fraction",
    "saturation",
    "dark_fraction",
    "border_mean",
    "center_mean",
    "gradient_x",
    "gradient_y",
    "maxima",
)
EVIDENCE_NAMES = (
    "signed_prediction_delta",
    "absolute_prediction_delta",
    "feature_displacement",
    "quality_delta",
    "one_minus_correlation",
    "registration_shift",
    "candidate_entropy",
)


def as_float(image: np.ndarray) -> np.ndarray:
    x = numeric_array("image", image, 2)
    if x.dtype.kind in "ui":
        if x.min() < 0:
            raise ValueError("negative detector values")
        return x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    if x.min() < 0 or x.max() > 1 + 1e-06:
        raise ValueError("float image must be on a common [0,1] detector scale")
    return x


def features(images: np.ndarray) -> np.ndarray:
    xs = numeric_array("images", images, 3)
    result = []
    for image in xs:
        x = as_float(image)
        h, w = x.shape
        if min(h, w) < 8:
            raise ValueError("images must be at least 8x8")
        gy, gx = np.gradient(x)
        lap = ndimage.laplace(x)
        q = np.quantile(x, [0.1, 0.5, 0.9, 0.99])
        mask = np.zeros(x.shape, dtype=bool)
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
        peaks = (x == ndimage.maximum_filter(x, size=5)) & (x > x.mean() + x.std())
        result.append(
            [
                x.mean(),
                x.std(),
                *q,
                np.mean(gx * gx + gy * gy),
                np.mean(lap * lap),
                np.mean(x > 0.35),
                np.mean(x >= 0.995),
                np.mean(x <= 0.005),
                x[mask].mean(),
                x[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean(),
                x[:, -w // 4 :].mean() - x[:, : w // 4].mean(),
                x[-h // 4 :, :].mean() - x[: h // 4, :].mean(),
                peaks.mean(),
            ]
        )
    return np.asarray(result, dtype=float)


def entropy(p: float | np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid probability")
    q = np.clip(p, 1e-12, 1 - 1e-12)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def quality(image: np.ndarray) -> float:
    return float(np.log1p(100 * np.mean(ndimage.laplace(as_float(image)) ** 2)))


@dataclass(frozen=True)
class Registration:
    shift_y: int
    shift_x: int
    correlation: float
    overlap: float
    valid: bool
    reason: str


def register(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    max_shift: float = 6.0,
    min_correlation: float = 0.05,
    min_overlap: float = 0.7,
) -> Registration:
    finite("max_shift", max_shift, 0)
    finite("min_correlation", min_correlation, -1, 1)
    finite("min_overlap", min_overlap, 0, 1)
    a, b = (as_float(reference), as_float(moving))
    if a.shape != b.shape:
        return Registration(0, 0, 0, 0, False, "shape_mismatch")
    if min(a.std(), b.std()) < 1e-07:
        return Registration(0, 0, 0, 1, False, "flat_image")
    cross = np.fft.fft2(a - a.mean()) * np.fft.fft2(b - b.mean()).conj()
    cross /= np.maximum(np.abs(cross), 1e-12)
    peak = np.unravel_index(np.argmax(np.fft.ifft2(cross).real), a.shape)
    dy, dx = [int(k if k <= n // 2 else k - n) for k, n in zip(peak, a.shape)]
    h, w = a.shape
    ar = a[max(0, dy) : min(h, h + dy), max(0, dx) : min(w, w + dx)].ravel()
    br = b[max(0, -dy) : min(h, h - dy), max(0, -dx) : min(w, w - dx)].ravel()
    corr = float(np.corrcoef(ar, br)[0, 1]) if min(ar.std(), br.std()) > 1e-07 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    overlap = len(ar) / a.size
    reason = (
        "shift_exceeded"
        if np.hypot(dy, dx) > max_shift
        else "insufficient_overlap"
        if overlap < min_overlap
        else "poor_correlation"
        if corr < min_correlation
        else "ok"
    )
    return Registration(dy, dx, corr, overlap, reason == "ok", reason)


def evidence(
    baseline: np.ndarray,
    candidate: np.ndarray,
    p0: float,
    pa: float,
    scale: np.ndarray,
    registration: Registration,
) -> np.ndarray:
    finite("p0", p0, 0, 1)
    finite("pa", pa, 0, 1)
    scale = numeric_array("scale", scale, 1)
    if scale.shape != (len(FEATURE_NAMES),) or np.any(scale <= 0):
        raise ValueError("invalid feature scale")
    f = features(np.stack([baseline, candidate]))
    sign = 1 if p0 >= 0.5 else -1
    return np.array(
        [
            sign * (pa - p0),
            abs(pa - p0),
            min(50.0, float(np.sqrt(np.mean(((f[1] - f[0]) / scale) ** 2)))),
            quality(candidate) - quality(baseline),
            1 - registration.correlation,
            np.hypot(registration.shift_y, registration.shift_x),
            float(entropy(pa)),
        ]
    )
