"""Local binary patterns (Ojala, Pietikäinen, Mäenpää, PAMI 2002).

8 neighbors at radius 1, starting east and walking clockwise. Codes with
more than two circular bit transitions collapse into one non-uniform bin
(59-bin uniform LBP). Each face is a chi-square histogram over a 4×4 grid.
"""

from __future__ import annotations

import cv2
import numpy as np

from visage.integral import as_gray_float

_OFFSETS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)


def uniform_lut() -> tuple[np.ndarray, int]:
    mapping: dict[int, int] = {}
    idx = 0
    for code in range(256):
        bits = [(code >> b) & 1 for b in range(8)]
        transitions = sum(bits[k] != bits[(k + 1) % 8] for k in range(8))
        if transitions <= 2:
            mapping[code] = idx
            idx += 1
    lut = np.full(256, idx, dtype=np.int32)
    for code, slot in mapping.items():
        lut[code] = slot
    return lut, idx + 1


_LUT, N_BINS = uniform_lut()


def lbp_image(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray)
    if g.dtype != np.uint8:
        g = np.clip(as_gray_float(g) * 255.0, 0, 255).astype(np.uint8)
    center = g[1:-1, 1:-1]
    code = np.zeros(center.shape, dtype=np.uint8)
    for bit, (dy, dx) in enumerate(_OFFSETS):
        nb = g[1 + dy : g.shape[0] - 1 + dy, 1 + dx : g.shape[1] - 1 + dx]
        code |= (nb >= center).astype(np.uint8) << bit
    return _LUT[code]


def spatial_histogram(gray: np.ndarray, grid: int = 4) -> np.ndarray:
    codes = lbp_image(gray)
    h, w = codes.shape
    hist = np.zeros(grid * grid * N_BINS, dtype=np.float64)
    cell_h = h // grid
    cell_w = w // grid
    if cell_h == 0 or cell_w == 0:
        return hist
    k = 0
    for gy in range(grid):
        for gx in range(grid):
            patch = codes[gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w]
            hist[k : k + N_BINS] = np.bincount(patch.ravel(), minlength=N_BINS)[:N_BINS]
            k += N_BINS
    hist /= hist.sum() + 1e-8
    return hist


def chi2(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(0.5 * np.sum((diff * diff) / (a + b + 1e-10)))


class LBPH:
    def __init__(self, size: int = 64, grid: int = 4):
        self.size = size
        self.grid = grid
        self.hists_: list[np.ndarray] = []
        self.labels_: list = []

    def _hist(self, image: np.ndarray) -> np.ndarray:
        gray = as_gray_float(image)
        u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        if u8.shape != (self.size, self.size):
            u8 = cv2.resize(u8, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return spatial_histogram(u8, self.grid)

    def fit(self, images: list[np.ndarray], labels: list | np.ndarray) -> "LBPH":
        self.hists_ = [self._hist(im) for im in images]
        self.labels_ = list(np.asarray(labels).tolist())
        return self

    def predict(self, image: np.ndarray) -> tuple[object, float]:
        if not self.hists_:
            raise RuntimeError("model is not fit")
        hist = self._hist(image)
        distances = [chi2(hist, stored) for stored in self.hists_]
        i = int(np.argmin(distances))
        return self.labels_[i], float(distances[i])
