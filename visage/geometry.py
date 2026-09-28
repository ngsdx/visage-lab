"""Boxes, overlap, greedy NMS, and weighted-box fusion."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float
    source: str
    skin_ratio: float = 0.0
    landmarks: dict | None = None
    extra: dict = field(default_factory=dict)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.w), int(self.h))

    def clamped(self, width: int, height: int) -> "Detection":
        x = max(0, min(int(self.x), width - 1))
        y = max(0, min(int(self.y), height - 1))
        w = max(1, min(int(self.w), width - x))
        h = max(1, min(int(self.h), height - y))
        self.x, self.y, self.w, self.h = x, y, w, h
        return self

    def to_dict(self) -> dict:
        data = {
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
            "score": round(float(self.score), 4),
            "source": self.source,
            "skin_ratio": round(float(self.skin_ratio), 4),
        }
        if self.landmarks:
            data["landmarks"] = {
                k: [round(float(v[0]), 2), round(float(v[1]), 2)]
                for k, v in self.landmarks.items()
            }
        if self.extra:
            data["extra"] = self.extra
        return data


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def nms(dets: list[Detection], thr: float = 0.4) -> list[Detection]:
    """Greedy non-maximum suppression. Higher score wins."""
    ordered = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for det in ordered:
        if all(iou(det.box, k.box) < thr for k in kept):
            kept.append(det)
    return kept


def weighted_box_fusion(dets: list[Detection], thr: float = 0.45) -> list[Detection]:
    """Cluster overlapping boxes and average them, weighted by score.

    A union-find over pairs with IoU >= thr. The surviving source is the
    member with the highest score; landmarks are taken from that member.
    """
    n = len(dets)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if iou(dets[i].box, dets[j].box) >= thr:
                union(i, j)

    groups: dict[int, list[Detection]] = {}
    for i, det in enumerate(dets):
        groups.setdefault(find(i), []).append(det)

    fused: list[Detection] = []
    for group in groups.values():
        weight = sum(max(d.score, 1e-6) for d in group)
        x = sum(d.x * max(d.score, 1e-6) for d in group) / weight
        y = sum(d.y * max(d.score, 1e-6) for d in group) / weight
        w = sum(d.w * max(d.score, 1e-6) for d in group) / weight
        h = sum(d.h * max(d.score, 1e-6) for d in group) / weight
        best = max(group, key=lambda d: d.score)
        yunet = next((d for d in group if d.extra.get("yunet_row")), None)
        skin = sum(d.skin_ratio for d in group) / len(group)
        extra = {"cluster": len(group), "sources": sorted({d.source for d in group})}
        landmarks = best.landmarks
        if yunet is not None:
            extra["yunet_row"] = list(yunet.extra["yunet_row"])
            landmarks = yunet.landmarks or landmarks
        fused.append(
            Detection(
                x=int(round(x)),
                y=int(round(y)),
                w=max(1, int(round(w))),
                h=max(1, int(round(h))),
                score=float(max(d.score for d in group)),
                source="fusion" if len(group) > 1 else best.source,
                skin_ratio=float(skin),
                landmarks=landmarks,
                extra=extra,
            )
        )
    fused.sort(key=lambda d: d.score, reverse=True)
    return fused
