"""Summed-area tables (Crow, 1984), as used by Viola & Jones, CVPR 2001.

``integral_image`` returns a table padded by one row and column so that
``ii[y, x]`` is the sum of pixels in ``gray[0:y, 0:x]``. A rectangle sum is
then four lookups.
"""

from __future__ import annotations

import numpy as np


def as_gray_float(image: np.ndarray) -> np.ndarray:
    """HxW float32 in [0, 1]. Accepts gray or BGR uint8/float."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        # Rec. 601 luma, BGR channel order (OpenCV).
        b = arr[:, :, 0].astype(np.float32)
        g = arr[:, :, 1].astype(np.float32)
        r = arr[:, :, 2].astype(np.float32)
        gray = 0.114 * b + 0.587 * g + 0.299 * r
    else:
        gray = arr.astype(np.float32)
    if gray.max() > 1.5:
        gray = gray / 255.0
    return gray


def integral_image(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("integral_image expects a 2-D array")
    ii = np.zeros((g.shape[0] + 1, g.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(g, axis=0), axis=1)
    return ii


def rect_sum(ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Sum of gray[y:y+h, x:x+w]. Coordinates are on the original image."""
    if w <= 0 or h <= 0:
        return 0.0
    x2, y2 = x + w, y + h
    return float(ii[y2, x2] - ii[y, x2] - ii[y2, x] + ii[y, x])


def window_std(ii: np.ndarray, ii2: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Standard deviation of a window via the plain and squared integral images."""
    n = float(w * h)
    if n <= 0:
        return 0.0
    mean = rect_sum(ii, x, y, w, h) / n
    mean_sq = rect_sum(ii2, x, y, w, h) / n
    var = max(0.0, mean_sq - mean * mean)
    return var ** 0.5
