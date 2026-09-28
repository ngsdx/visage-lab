"""Multi-cue face analysis: detect, filter by skin, fuse, align, score quality."""

from __future__ import annotations

import cv2
import numpy as np

from visage.align import order_eyes, roll_degrees, warp_face
from visage.detectors import detect_eyes, detect_haar, detect_yunet
from visage.geometry import Detection, weighted_box_fusion
from visage.quality import assess
from visage.skin import skin_mask, skin_ratio


def _eyes_from_landmarks(det: Detection) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not det.landmarks:
        return None
    if "left_eye" in det.landmarks and "right_eye" in det.landmarks:
        return order_eyes(tuple(det.landmarks["right_eye"]), tuple(det.landmarks["left_eye"]))
    return None


def _eyes_from_cascade(bgr: np.ndarray, det: Detection) -> tuple[tuple[float, float], tuple[float, float]] | None:
    x, y, w, h = det.box
    # Eyes sit in the upper half of a frontal box.
    roi = bgr[y : y + int(h * 0.6), x : x + w]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    rects = detect_eyes(gray)
    if len(rects) < 2:
        return None
    centers = []
    for ex, ey, ew, eh in rects:
        centers.append((x + ex + ew / 2.0, y + ey + eh / 2.0, ew * eh))
    centers.sort(key=lambda c: c[2], reverse=True)
    a = (centers[0][0], centers[0][1])
    b = (centers[1][0], centers[1][1])
    left, right = order_eyes(a, b)
    # Reject pairs that are stacked (two detections of one eye) or too close.
    if abs(right[0] - left[0]) < 0.15 * w:
        return None
    if abs(right[1] - left[1]) > 0.35 * h:
        return None
    return left, right


def analyze(
    bgr: np.ndarray,
    *,
    use_haar: bool = True,
    use_yunet: bool = True,
    skin_floor: float = 0.12,
    fusion_iou: float = 0.45,
    align_size: int = 112,
) -> dict:
    """Run the fusion pipeline. ``bgr`` is an OpenCV image."""
    if bgr.ndim != 3:
        raise ValueError("analyze expects a BGR image")
    raw: list[Detection] = []
    if use_haar:
        raw.extend(detect_haar(bgr))
    if use_yunet:
        raw.extend(detect_yunet(bgr))
    mask = skin_mask(bgr)
    for det in raw:
        det.clamped(bgr.shape[1], bgr.shape[0])
        det.skin_ratio = skin_ratio(mask, det.box)
    # YuNet is a strong detector; the skin floor only drops classical boxes,
    # which is where color is actually informative as a rejector.
    filtered = [
        det
        for det in raw
        if det.source == "yunet" or det.skin_ratio >= skin_floor
    ]
    fused = weighted_box_fusion(filtered, fusion_iou)
    # A lone classical box with a weak cascade weight is usually a profile-cascade
    # false positive (hair, collars). Clusters and YuNet are kept as they are.
    fused = [
        det
        for det in fused
        if det.source in {"fusion", "yunet"} or det.score >= 0.55
    ]
    faces = []
    for det in fused:
        det.clamped(bgr.shape[1], bgr.shape[0])
        eyes = _eyes_from_landmarks(det) or _eyes_from_cascade(bgr, det)
        roll = None
        aligned = None
        if eyes is not None:
            roll = roll_degrees(eyes[0], eyes[1])
            aligned = warp_face(bgr, eyes[0], eyes[1], align_size)
            det.landmarks = det.landmarks or {}
            det.landmarks["image_left_eye"] = eyes[0]
            det.landmarks["image_right_eye"] = eyes[1]
        x, y, w, h = det.box
        crop = bgr[y : y + h, x : x + w]
        quality = assess(crop if crop.size else bgr, roll)
        record = det.to_dict()
        record["quality"] = quality
        record["aligned"] = aligned
        faces.append(record)
    return {
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
        "raw": [d.to_dict() for d in raw],
        "faces": [{k: v for k, v in f.items() if k != "aligned"} | {"has_alignment": f["aligned"] is not None} for f in faces],
        "aligned": [f["aligned"] for f in faces],
        "skin_mask": mask,
    }


def draw(bgr: np.ndarray, result: dict) -> np.ndarray:
    out = bgr.copy()
    for face in result["faces"]:
        color = (50, 72, 220) if str(face["source"]).startswith("haar") or str(face["source"]).startswith("profile") else (220, 224, 230)
        if face["source"] == "yunet":
            color = (90, 170, 196)
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{face['source']} {face['score']:.2f}"
        cv2.putText(out, label, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        marks = face.get("landmarks") or {}
        for key in ("image_left_eye", "image_right_eye", "nose"):
            if key in marks:
                px, py = marks[key]
                cv2.circle(out, (int(px), int(py)), 3, (40, 40, 220), -1, cv2.LINE_AA)
    return out
