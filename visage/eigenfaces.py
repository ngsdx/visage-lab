"""Eigenfaces (Turk & Pentland, 1991).

A face is a point in pixel space. After subtracting the mean face, PCA
(via SVD of the centered data matrix) keeps the directions of largest
variance. Classification is nearest neighbor in that subspace.
"""

from __future__ import annotations

import cv2
import numpy as np

from visage.integral import as_gray_float


def _prepare(image: np.ndarray, size: int) -> np.ndarray:
    gray = as_gray_float(image)
    if gray.shape != (size, size):
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float64).ravel()


class Eigenfaces:
    def __init__(self, n_components: int | None = None, size: int = 32):
        self.n_components = n_components
        self.size = size
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.proj_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, images: list[np.ndarray], labels: list | np.ndarray) -> "Eigenfaces":
        x = np.stack([_prepare(im, self.size) for im in images])
        y = np.asarray(labels)
        if len(x) < 2:
            raise ValueError("eigenfaces needs at least two images")
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        max_k = max(1, min(len(x) - 1, vt.shape[0]))
        k = self.n_components or min(max_k, max(len(x) // 2, 1))
        k = max(1, min(k, max_k))
        self.components_ = vt[:k]
        self.proj_ = centered @ self.components_.T
        self.labels_ = y
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("model is not fit")
        vec = _prepare(image, self.size)
        return (vec - self.mean_) @ self.components_.T

    def predict(self, image: np.ndarray) -> tuple[object, float]:
        if self.proj_ is None or self.labels_ is None:
            raise RuntimeError("model is not fit")
        z = self.transform(image)
        dist = np.linalg.norm(self.proj_ - z, axis=1)
        i = int(np.argmin(dist))
        return self.labels_[i], float(dist[i])
