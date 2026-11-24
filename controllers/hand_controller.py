"""Single-hand MediaPipe Tasks features, independent of camera and gesture rules."""

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import contracts
from .base import BaseController, ControllerOutput, _prepare_frame, _draw_hud, _drawing_points


ROOT = Path(__file__).resolve().parent.parent
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8), (5, 9),
    (9, 10), (10, 11), (11, 12), (9, 13),
    (13, 14), (14, 15), (15, 16), (13, 17),
    (17, 18), (18, 19), (19, 20), (0, 17),
)


class HandController(BaseController):
    mode = contracts.ControlMode.HAND

    def __init__(self, config: dict | None = None, model_path: str | None = None,
                 landmarker=None):
        self.config = contracts.load_config() if config is None else config
        self._last_timestamp_ms = -1
        self._closed = False
        self.model_path: Path | None = None
        if landmarker is None:
            path = Path(model_path).expanduser() if model_path is not None else ROOT / "hand_landmarker.task"
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file():
                raise FileNotFoundError(f"HandLandmarker model not found: {path}")
            self.model_path = path.resolve()
        import mediapipe as mp

        self._mp = mp
        if landmarker is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.VIDEO, num_hands=1,
                min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            landmarker = vision.HandLandmarker.create_from_options(options)
        if not callable(getattr(landmarker, "detect_for_video", None)):
            raise TypeError("Landmarker must expose detect_for_video(image, timestamp_ms).")
        self._landmarker = landmarker

    @staticmethod
    def _features(coords: np.ndarray, px: np.ndarray) -> dict[str, Any]:
        # Use a pixel metric: normalized x/y have unequal scales on a VGA image.
        points = px.astype(np.float64)

        def distance(a: int, b: int) -> float:
            return float(np.linalg.norm(points[a] - points[b]))

        fingers = {name: bool(distance(0, tip) > distance(0, pip) * 1.12)
                   for name, tip, pip in (("index", 8, 6), ("middle", 12, 10),
                                          ("ring", 16, 14), ("pinky", 20, 18))}
        thumb = bool(distance(4, 17) > distance(3, 17) * 1.08 and distance(4, 2) > distance(3, 2))
        fingers = {"thumb": thumb, **fingers}
        vector = points[8] - points[5]
        length = float(np.linalg.norm(vector))
        pointing = tuple(map(float, vector / length)) if length > 1e-9 else (0.0, 0.0)
        palm = coords[[0, 5, 9, 13, 17], :2].mean(axis=0)
        return {"finger_states": fingers, "extended_count": int(sum(fingers.values())),
                "palm_center_norm": tuple(map(float, palm)),
                "index_tip_norm": tuple(map(float, coords[8, :2])),
                "pointing_vector": pointing}

    def process_frame(self, frame: np.ndarray, timestamp_ms: int, draw: bool = True,
                      shared_context: dict[str, Any] | None = None) -> ControllerOutput:
        if self._closed:
            raise RuntimeError("Cannot process frames with a closed HandController.")
        annotated = _prepare_frame(frame, draw)
        ts_ms = max(self._last_timestamp_ms + 1, int(timestamp_ms))
        self._last_timestamp_ms = ts_ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        hands = getattr(result, "hand_landmarks", None)
        valid = False
        if hands and len(hands[0]) >= 21:
            try:
                normalized = [(float(point.x), float(point.y), float(point.z)) for point in hands[0][:21]]
                coords = np.asarray(normalized, dtype=np.float64)
                height, width = frame.shape[:2]
                px = (coords[:, :2] * (width, height)).astype(np.float32)
                valid = np.isfinite(coords).all() and np.isfinite(px).all()
            except (ValueError, TypeError, AttributeError, OverflowError):
                valid = False
        if not valid:
            if draw:
                _draw_hud(annotated, ["HAND MODE: NO HAND"], (80, 190, 230))
            return ControllerOutput(self.mode.value, False, annotated, None, {})

        handedness, confidence = "Unknown", 0.0
        categories = getattr(result, "handedness", None)
        if categories and categories[0]:
            category = categories[0][0]
            name = getattr(category, "category_name", "") or getattr(category, "display_name", "")
            handedness = name if name in ("Left", "Right") else "Unknown"
            try:
                score = float(getattr(category, "score", 0.0))
                confidence = max(0.0, min(1.0, score)) if math.isfinite(score) else 0.0
            except (ValueError, TypeError):
                confidence = 0.0
        features = self._features(coords, px)
        features.update(handedness=handedness, landmarks_norm=normalized)
        if draw:
            points = _drawing_points(px, annotated)
            for a, b in HAND_CONNECTIONS:
                cv2.line(annotated, tuple(points[a]), tuple(points[b]), (160, 190, 160), 1, cv2.LINE_AA)
            for point in points:
                cv2.circle(annotated, tuple(point), 2, (90, 220, 120), -1, cv2.LINE_AA)
            cv2.circle(annotated, tuple(points[8]), 4, (0, 190, 255), 1, cv2.LINE_AA)
            _draw_hud(annotated, ["MODE: HAND (21-pt)",
                                  f"{handedness} ({confidence:.2f})  FINGERS: {features['extended_count']}"])
        return ControllerOutput(self.mode.value, True, annotated, px, features, confidence=confidence)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            landmarker, self._landmarker = self._landmarker, None
            close = getattr(landmarker, "close", None)
            if callable(close):
                close()
