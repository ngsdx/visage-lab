"""Haar-like rectangular features evaluated on an integral image.

This is the feature family from Viola & Jones, not the OpenCV cascade
(that lives in ``visage.detectors``). Features are area-normalized so a
response is a difference of region means.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visage.integral import as_gray_float, integral_image, rect_sum, window_std

# (x, y, w, h, weight) inside a detection window.
Rect = tuple[int, int, int, int, float]


@dataclass(frozen=True)
class HaarFeature:
    rects: tuple[Rect, ...]
    kind: str

    def response(self, ii: np.ndarray, x: int = 0, y: int = 0) -> float:
        total = 0.0
        for rx, ry, rw, rh, weight in self.rects:
            total += weight * rect_sum(ii, x + rx, y + ry, rw, rh)
        return total


def _two(rng: np.random.Generator, win: int, horizontal: bool) -> HaarFeature:
    if horizontal:
        w = int(rng.integers(2, win // 2 + 1)) * 2
        h = int(rng.integers(2, win + 1))
        x = int(rng.integers(0, win - w + 1))
        y = int(rng.integers(0, win - h + 1))
        hw = w // 2
        area = float(hw * h)
        rects = (
            (x, y, hw, h, 1.0 / area),
            (x + hw, y, hw, h, -1.0 / area),
        )
        return HaarFeature(rects, "two-h")
    h = int(rng.integers(2, win // 2 + 1)) * 2
    w = int(rng.integers(2, win + 1))
    x = int(rng.integers(0, win - w + 1))
    y = int(rng.integers(0, win - h + 1))
    hh = h // 2
    area = float(w * hh)
    rects = (
        (x, y, w, hh, 1.0 / area),
        (x, y + hh, w, hh, -1.0 / area),
    )
    return HaarFeature(rects, "two-v")


def _three(rng: np.random.Generator, win: int) -> HaarFeature:
    w = int(rng.integers(3, win // 3 + 1)) * 3
    h = int(rng.integers(2, win + 1))
    x = int(rng.integers(0, win - w + 1))
    y = int(rng.integers(0, win - h + 1))
    tw = w // 3
    area = float(tw * h)
    rects = (
        (x, y, tw, h, -1.0 / area),
        (x + tw, y, tw, h, 2.0 / area),
        (x + 2 * tw, y, tw, h, -1.0 / area),
    )
    return HaarFeature(rects, "three")


def _four(rng: np.random.Generator, win: int) -> HaarFeature:
    w = int(rng.integers(2, win // 2 + 1)) * 2
    h = int(rng.integers(2, win // 2 + 1)) * 2
    x = int(rng.integers(0, win - w + 1))
    y = int(rng.integers(0, win - h + 1))
    hw, hh = w // 2, h // 2
    area = float(hw * hh)
    rects = (
        (x, y, hw, hh, 1.0 / area),
        (x + hw, y, hw, hh, -1.0 / area),
        (x, y + hh, hw, hh, -1.0 / area),
        (x + hw, y + hh, hw, hh, 1.0 / area),
    )
    return HaarFeature(rects, "four")


def make_feature_bank(n: int, win: int = 24, seed: int = 0) -> list[HaarFeature]:
    rng = np.random.default_rng(seed)
    kinds = (_two, _three, _four)
    bank: list[HaarFeature] = []
    for i in range(n):
        if i % 5 == 4:
            bank.append(_four(rng, win))
        elif i % 5 == 3:
            bank.append(_three(rng, win))
        else:
            bank.append(_two(rng, win, horizontal=(i % 2 == 0)))
    return bank


def feature_matrix(
    images: list[np.ndarray] | np.ndarray,
    features: list[HaarFeature],
    normalize: bool = True,
) -> np.ndarray:
    """Rows are images, columns are variance-normalized Haar responses."""
    feats = len(features)
    out = np.zeros((len(images), feats), dtype=np.float64)
    for i, image in enumerate(images):
        gray = as_gray_float(image)
        ii = integral_image(gray)
        sigma = 1.0
        if normalize:
            ii2 = integral_image(gray * gray)
            h, w = gray.shape
            sigma = max(window_std(ii, ii2, 0, 0, w, h), 1e-4)
        for j, feat in enumerate(features):
            out[i, j] = feat.response(ii) / sigma
    return out


def window_responses(
    ii: np.ndarray,
    ii2: np.ndarray,
    features: list[HaarFeature],
    x: int,
    y: int,
    win: int,
) -> np.ndarray:
    sigma = max(window_std(ii, ii2, x, y, win, win), 1e-4)
    vals = np.empty(len(features), dtype=np.float64)
    for j, feat in enumerate(features):
        vals[j] = feat.response(ii, x, y) / sigma
    return vals
