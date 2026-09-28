"""SORT-style multi-face tracking (Bewley et al., ICIP 2016), appearance-free.

Constant-velocity Kalman filter on (cx, cy, w, h) and their velocities.
Association is Hungarian matching on IoU, with a gate. Tracks survive a
few missed frames so a detector blink does not mint a new id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from visage.geometry import Detection, iou


def _kalman_mats() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f = np.eye(8)
    for i in range(4):
        f[i, i + 4] = 1.0
    h = np.zeros((4, 8))
    h[0, 0] = h[1, 1] = h[2, 2] = h[3, 3] = 1.0
    q = np.eye(8)
    q[4:, 4:] *= 0.5
    r = np.eye(4) * 2.0
    return f, h, q, r


@dataclass
class Track:
    id: int
    mean: np.ndarray
    cov: np.ndarray
    hits: int = 1
    missed: int = 0
    age: int = 1
    score: float = 0.0
    source: str = ""
    box: tuple[int, int, int, int] = (0, 0, 0, 0)
    history: list[tuple[int, int]] = field(default_factory=list)

    def center(self) -> tuple[float, float]:
        return float(self.mean[0]), float(self.mean[1])


class SortTracker:
    def __init__(self, iou_gate: float = 0.3, max_missed: int = 8):
        self.iou_gate = iou_gate
        self.max_missed = max_missed
        self.tracks: list[Track] = []
        self._next = 1
        self._F, self._H, self._Q, self._R = _kalman_mats()

    def _box_to_z(self, det: Detection) -> np.ndarray:
        cx = det.x + det.w / 2.0
        cy = det.y + det.h / 2.0
        return np.array([cx, cy, float(det.w), float(det.h)], dtype=np.float64)

    def _mean_to_box(self, mean: np.ndarray) -> tuple[int, int, int, int]:
        w = max(1.0, float(mean[2]))
        h = max(1.0, float(mean[3]))
        x = float(mean[0]) - w / 2.0
        y = float(mean[1]) - h / 2.0
        return int(round(x)), int(round(y)), int(round(w)), int(round(h))

    def _predict(self, track: Track) -> None:
        track.mean = self._F @ track.mean
        track.cov = self._F @ track.cov @ self._F.T + self._Q
        track.age += 1
        track.box = self._mean_to_box(track.mean)

    def _update(self, track: Track, det: Detection) -> None:
        z = self._box_to_z(det)
        z_pred = self._H @ track.mean
        s = self._H @ track.cov @ self._H.T + self._R
        k = track.cov @ self._H.T @ np.linalg.inv(s)
        track.mean = track.mean + k @ (z - z_pred)
        track.cov = (np.eye(8) - k @ self._H) @ track.cov
        track.missed = 0
        track.hits += 1
        track.score = det.score
        track.source = det.source
        track.box = self._mean_to_box(track.mean)
        cx, cy = track.center()
        track.history.append((int(round(cx)), int(round(cy))))
        if len(track.history) > 32:
            track.history = track.history[-32:]

    def _spawn(self, det: Detection) -> Track:
        z = self._box_to_z(det)
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = z
        cov = np.eye(8) * 10.0
        cov[4:, 4:] *= 4.0
        box = self._mean_to_box(mean)
        track = Track(
            id=self._next,
            mean=mean,
            cov=cov,
            score=det.score,
            source=det.source,
            box=box,
            history=[(int(round(z[0])), int(round(z[1])))],
        )
        self._next += 1
        return track

    def update(self, detections: list[Detection]) -> list[Track]:
        for track in self.tracks:
            self._predict(track)
        if not self.tracks:
            self.tracks = [self._spawn(d) for d in detections]
            return list(self.tracks)
        if not detections:
            for track in self.tracks:
                track.missed += 1
            self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
            return list(self.tracks)

        cost = np.ones((len(self.tracks), len(detections)), dtype=np.float64)
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                overlap = iou(track.box, det.box)
                cost[i, j] = 1.0 - overlap if overlap >= self.iou_gate else 1.0
        rows, cols = linear_sum_assignment(cost)
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        for r, c in zip(rows, cols):
            if cost[r, c] >= 1.0:
                continue
            self._update(self.tracks[r], detections[c])
            matched_t.add(r)
            matched_d.add(c)
        for i, track in enumerate(self.tracks):
            if i not in matched_t:
                track.missed += 1
        for j, det in enumerate(detections):
            if j not in matched_d:
                self.tracks.append(self._spawn(det))
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return list(self.tracks)
