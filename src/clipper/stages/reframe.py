"""Stufe 6: 16:9 -> 9:16 mit Motivverfolgung.

Bewusste Designentscheidung: der Crop bleibt pro Shot statisch. Ein Crop, der
innerhalb eines Shots mitwandert, sieht bei schnell geschnittenem Material
unruhig aus und kostet deutlich mehr Rechenzeit. Ein harter Sprung auf der
Schnittgrenze faellt dagegen gar nicht auf, weil dort ohnehin das ganze Bild
wechselt.
"""

from __future__ import annotations

import os
import urllib.request
from bisect import bisect_right
from pathlib import Path

# Muss vor dem cv2-Import stehen. Unterdrueckt die Meldung des neuen DNN-Backends
# ("Targets are not supported by the new graph engine"), die pro Clip erscheint
# und rein informativ ist.
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
    """Laedt das YuNet-Modell (~340 KB) beim ersten Lauf herunter."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / YUNET_FILE
    if not path.exists():
        urllib.request.urlretrieve(YUNET_URL, path)
    return path


class SubjectLocator:
    """Findet pro Frame die horizontale Bildmitte des Motivs (0..1)."""

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
        """Gibt [(cx_norm, gewicht)] fuer alle erkannten Gesichter zurueck.

        Gesichter unterhalb von `min_weight` werden verworfen. Der Grund ist
        ein konkreter Fehlschnitt: eine Person weit im Hintergrund belegte
        0,04 % der Bildflaeche, war aber die einzige Erkennung im Shot - der
        Crop folgte ihr an den Bildrand und verpasste die eigentliche Szene.
        Gemessen liegen echte Motive im Median bei 0,5 % der Flaeche.
        """
        _, dets = self.detector.detect(frame)
        if dets is None:
            return []
        out = []
        for det in dets:
            x, y, w, h = det[0], det[1], det[2], det[3]
            score = float(det[14]) if len(det) > 14 else 1.0
            cx = (x + w / 2.0) / self.frame_w
            # Groessere Gesichter sind naeher an der Kamera und damit wichtiger.
            weight = float(w * h) / (self.frame_w * self.frame_h) * score
            if weight < self.min_weight:
                continue
            out.append((float(cx), weight))
        return out


def _motion_center(prev: np.ndarray, curr: np.ndarray) -> float | None:
    """Schwerpunkt der Bewegung als Rueckfallebene, wenn kein Gesicht sichtbar ist."""
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
    """Ermittelt fuer den Zeitbereich [start, end) je Shot ein Crop-Fenster."""
    rf = cfg["reframe"]
    target_w, target_h = rf["target_width"], rf["target_height"]
    sample_fps = rf["sample_fps"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Konnte {video_path} nicht oeffnen")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Groesstmoegliches 9:16-Fenster, das noch ins Quellbild passt.
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

    # Shots auf das Clipfenster beschneiden.
    local_shots = [s for s in shots if s.end > start and s.start < end]
    if not local_shots:
        local_shots = [Shot(index=0, start=start, end=end)]
    shot_starts = [max(s.start, start) for s in local_shots]

    # Ein einziger Seek an den Clipanfang, danach sequenziell durchdekodieren.
    # Frueher wurde pro Sample-Frame neu positioniert - jeder dieser Seeks
    # zwingt den Decoder zurueck auf den letzten Keyframe und kostete rund
    # 0,6 s. Sequenziell greifen (grab) und nur an den Sample-Punkten
    # dekodieren (retrieve) ist um Groessenordnungen schneller.
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
        # Bewegung nur innerhalb desselben Shots vergleichen - ueber einen
        # Schnitt hinweg ist das Differenzbild bedeutungslos.
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
            # Kein Signal: die Position des vorigen Shots halten statt auf die
            # Bildmitte zu springen - das vermeidet einen sichtbaren Ruck.
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
    """Entfernt Keyframes mit identischem Zeitpunkt.

    Reicht ein Shot ueber den Clipanfang hinaus, faellt sein auf den Clip
    bezogener Start mit dem des Folge-Shots auf 0 zusammen. Im Crop-Ausdruck
    waere der erste davon toter Code.
    """
    out: list[CropKeyframe] = []
    for kf in sorted(keyframes, key=lambda k: k.t):
        if out and abs(out[-1].t - kf.t) < 1e-6:
            out[-1] = kf
        else:
            out.append(kf)
    return out


def _smooth(keyframes: list[CropKeyframe], max_jump: int) -> list[CropKeyframe]:
    """Kleine Sprünge zwischen aufeinanderfolgenden Shots einebnen.

    Wenn zwei benachbarte Shots fast dieselbe Motivposition haben, wirkt ein
    minimaler Versatz wie ein Wackler. Unterhalb der Schwelle wird der vorherige
    Wert uebernommen.
    """
    for i in range(1, len(keyframes)):
        if abs(keyframes[i].x - keyframes[i - 1].x) < max_jump:
            keyframes[i].x = keyframes[i - 1].x
    return keyframes
