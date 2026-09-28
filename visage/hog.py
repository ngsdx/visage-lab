"""Histograms of oriented gradients (Dalal & Triggs, CVPR 2005).

Unsigned orientation (0–180°), 9 bins, 8×8 cells, 2×2 blocks, L2-Hys.
Bin centers are linearly interpolated. Spatial interpolation inside the
block is omitted; the block overlap still couples neighboring cells.
"""

from __future__ import annotations

import cv2
import numpy as np

from visage.integral import as_gray_float

CELL = 8
BINS = 9
BIN_WIDTH = 180.0 / BINS


def gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = as_gray_float(gray)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    return mag.astype(np.float32), ang.astype(np.float32)


def cell_histograms(mag: np.ndarray, ang: np.ndarray, cell: int = CELL) -> np.ndarray:
    h, w = mag.shape
    cells_y, cells_x = h // cell, w // cell
    hist = np.zeros((cells_y, cells_x, BINS), dtype=np.float64)
    if cells_y == 0 or cells_x == 0:
        return hist
    m = mag[: cells_y * cell, : cells_x * cell]
    a = ang[: cells_y * cell, : cells_x * cell]
    b = a / BIN_WIDTH
    b0 = np.floor(b).astype(np.int32) % BINS
    b1 = (b0 + 1) % BINS
    frac = (b - np.floor(b)).astype(np.float64)
    for cy in range(cells_y):
        for cx in range(cells_x):
            ys, xs = cy * cell, cx * cell
            pm = m[ys : ys + cell, xs : xs + cell].astype(np.float64)
            i0 = b0[ys : ys + cell, xs : xs + cell]
            i1 = b1[ys : ys + cell, xs : xs + cell]
            f = frac[ys : ys + cell, xs : xs + cell]
            np.add.at(hist[cy, cx], i0, pm * (1.0 - f))
            np.add.at(hist[cy, cx], i1, pm * f)
    return hist


def hog_descriptor(
    gray: np.ndarray,
    cell: int = CELL,
    clip: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (descriptor, cell histograms). Descriptor may be empty if the image is tiny."""
    mag, ang = gradients(gray)
    hist = cell_histograms(mag, ang, cell)
    ch, cw, _ = hist.shape
    blocks = []
    for y in range(max(ch - 1, 0)):
        for x in range(max(cw - 1, 0)):
            block = hist[y : y + 2, x : x + 2, :].ravel()
            block = block / (np.linalg.norm(block) + 1e-6)
            block = np.clip(block, 0.0, clip)
            block = block / (np.linalg.norm(block) + 1e-6)
            blocks.append(block)
    if not blocks:
        return np.zeros(0, dtype=np.float64), hist
    return np.concatenate(blocks).astype(np.float64), hist


def hog_matrix(images: list[np.ndarray], size: int = 64) -> np.ndarray:
    rows = []
    for image in images:
        g = as_gray_float(image)
        if g.shape != (size, size):
            g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        desc, _ = hog_descriptor(g)
        rows.append(desc)
    return np.stack(rows)
