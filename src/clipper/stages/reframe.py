"""Stage 6: 16:9 -> 9:16 with subject tracking.

Deliberate design decision: the crop stays static per shot. A crop that pans
within a shot looks restless on fast-cut material and costs considerably more
compute. A hard jump on a cut boundary, by contrast, is not noticeable at all,
because the whole frame changes there anyway.
"""

from __future__ import annotations

import os
import urllib.request
from bisect import bisect_right
from pathlib import Path

# Must come before the cv2 import. Silences the new DNN backend's message
# ("Targets are not supported by the new graph engine"), which shows up once per
# clip and is purely informational.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np

from ..config import MODELS_DIR
from ..models import CropKeyframe, Shot

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_FILE = "face_detection_yunet_2023mar.onnx"


def ensure_yunet() -> Path:
    """Download the YuNet model (~340 KB) on first run."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / YUNET_FILE
    if not path.exists():
        urllib.request.urlretrieve(YUNET_URL, path)
    return path


class SubjectLocator:
    """Locates the subject's horizontal centre per frame (0..1)."""

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        score_threshold: float = 0.6,
        min_weight: float = 0.001,
    ):
        model = ensure_yunet()
        self.detector = cv2.FaceDetectorYN.create(
            model=str(model),
            config="",
            input_size=(frame_w, frame_h),
            score_threshold=score_threshold,
        )
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.min_weight = min_weight

    def faces(self, frame: np.ndarray) -> list[tuple[float, float]]:
        """Return [(cx_norm, weight)] for every detected face.

        Faces below `min_weight` are discarded. The reason is one concrete bad
        crop: a person far in the background occupied 0.04% of the frame area
        but was the only detection in that shot - the crop followed them to the
        frame edge and missed the actual scene. Measured, real subjects sit at a
        median of 0.5% of frame area.
        """
        _, dets = self.detector.detect(frame)
        if dets is None:
            return []
        out = []
        for det in dets:
            x, y, w, h = det[0], det[1], det[2], det[3]
            score = float(det[14]) if len(det) > 14 else 1.0
            cx = (x + w / 2.0) / self.frame_w
            # Larger faces are closer to the camera and therefore more important.
            weight = float(w * h) / (self.frame_w * self.frame_h) * score
            if weight < self.min_weight:
                continue
            out.append((float(cx), weight))
        return out


def _motion_center(prev: np.ndarray, curr: np.ndarray) -> float | None:
    """Motion centroid as a fallback when no face is visible."""
    diff = cv2.absdiff(prev, curr)
    diff = cv2.GaussianBlur(diff, (21, 21), 0)
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    column_sum = mask.sum(axis=0).astype(np.float64)
    total = column_sum.sum()
    if total < 1e-6:
        return None
    xs = np.arange(len(column_sum))
    return float((xs * column_sum).sum() / total / len(column_sum))


def analyze(
    video_path: Path,
    start: float,
    end: float,
    shots: list[Shot],
    cfg: dict,
) -> list[CropKeyframe]:
    """Determine one crop window per shot for the range [start, end)."""
    rf = cfg["reframe"]
    target_w, target_h = rf["target_width"], rf["target_height"]
    sample_fps = rf["sample_fps"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Largest possible 9:16 window that still fits inside the source frame.
    crop_h = src_h
    crop_w = int(round(crop_h * target_w / target_h))
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(round(crop_w * target_h / target_w))
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    max_x = max(0, src_w - crop_w)
    crop_y = max(0, (src_h - crop_h) // 2)
    crop_y -= crop_y % 2

    locator = SubjectLocator(src_w, src_h, min_weight=rf.get("min_face_weight", 0.001))

    # Clip the shot list down to the clip window.
    local_shots = [s for s in shots if s.end > start and s.start < end]
    if not local_shots:
        local_shots = [Shot(index=0, start=start, end=end)]
    shot_starts = [max(s.start, start) for s in local_shots]

    # One single seek to the clip start, then decode sequentially. An earlier
    # version repositioned for every sampled frame - each of those seeks forces
    # the decoder back to the previous keyframe and cost around 0.6 s. Grabbing
    # sequentially (grab) and only decoding at the sample points (retrieve) is
    # orders of magnitude faster.
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    faces_per_shot: list[list[tuple[float, float]]] = [[] for _ in local_shots]
    motion_per_shot: list[list[float]] = [[] for _ in local_shots]
    prev_gray: np.ndarray | None = None
    prev_shot: int | None = None

    interval = 1.0 / sample_fps
    next_sample = start

    while True:
        if not cap.grab():
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t >= end:
            break
        if t < start or t + 1e-6 < next_sample:
            continue

        ok, frame = cap.retrieve()
        if not ok:
            break
        next_sample = t + interval

        idx = bisect_right(shot_starts, t) - 1
        idx = max(0, min(len(local_shots) - 1, idx))

        faces_per_shot[idx].extend(locator.faces(frame))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Only compare motion within the same shot - across a cut the frame
        # difference is meaningless.
        if prev_gray is not None and prev_shot == idx:
            mc = _motion_center(prev_gray, gray)
            if mc is not None:
                motion_per_shot[idx].append(mc)
        prev_gray, prev_shot = gray, idx

    cap.release()

    keyframes: list[CropKeyframe] = []
    for i, shot in enumerate(local_shots):
        centers = faces_per_shot[i]
        if centers:
            weights = np.array([w for _, w in centers], dtype=np.float64)
            xs = np.array([c for c, _ in centers], dtype=np.float64)
            cx_norm = float((xs * weights).sum() / max(weights.sum(), 1e-9))
        elif motion_per_shot[i]:
            cx_norm = float(np.median(motion_per_shot[i]))
        elif keyframes:
            # No signal: hold the previous shot's position instead of jumping to
            # the frame centre - that avoids a visible lurch.
            cx_norm = (keyframes[-1].x + crop_w / 2.0) / src_w
        else:
            cx_norm = 0.5

        x = int(round(cx_norm * src_w - crop_w / 2.0))
        x = max(0, min(max_x, x))
        x -= x % 2

        keyframes.append(
            CropKeyframe(
                t=max(shot.start, start) - start, x=x, y=crop_y, w=crop_w, h=crop_h
            )
        )

    if not keyframes:
        x = max(0, (src_w - crop_w) // 2)
        keyframes = [CropKeyframe(t=0.0, x=x - x % 2, y=crop_y, w=crop_w, h=crop_h)]

    return _smooth(_dedupe(keyframes), max_jump=int(crop_w * 0.12))


def _dedupe(keyframes: list[CropKeyframe]) -> list[CropKeyframe]:
    """Remove keyframes that share the same timestamp.

    If a shot extends past the clip start, its clip-relative start collapses to
    0 together with the following shot's. In the crop expression the first of
    those would be dead code.
    """
    out: list[CropKeyframe] = []
    for kf in sorted(keyframes, key=lambda k: k.t):
        if out and abs(out[-1].t - kf.t) < 1e-6:
            out[-1] = kf
        else:
            out.append(kf)
    return out


def _smooth(keyframes: list[CropKeyframe], max_jump: int) -> list[CropKeyframe]:
    """Smooth out small jumps between consecutive shots.

    When two neighbouring shots have almost the same subject position, a tiny
    offset reads as a wobble. Below the threshold the previous value is kept.
    """
    for i in range(1, len(keyframes)):
        if abs(keyframes[i].x - keyframes[i - 1].x) < max_jump:
            keyframes[i].x = keyframes[i - 1].x
    return keyframes
