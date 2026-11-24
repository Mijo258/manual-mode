"""Face pose features from the SafetyEngine's existing 478-point inference."""

import math
from typing import Any

import cv2
import numpy as np

import contracts
from .base import BaseController, ControllerOutput, _prepare_frame, _draw_hud, _drawing_points


class FaceController(BaseController):
    mode = contracts.ControlMode.FACE

    def __init__(self, config: dict | None = None):
        self.config = config  # No model or inference object is needed in this controller.

    def process_frame(self, frame: np.ndarray, timestamp_ms: int, draw: bool = True,
                      shared_context: dict[str, Any] | None = None) -> ControllerOutput:
        annotated = _prepare_frame(frame, draw)
        context = shared_context or {}
        normalized = context.get("face_landmarks_norm")
        pixels = context.get("face_landmarks_px")
        valid = False
        if normalized is not None and pixels is not None:
            try:
                coords = np.asarray(normalized, dtype=np.float64)
                px = np.asarray(pixels, dtype=np.float32)
                valid = (coords.ndim == 2 and coords.shape[0] >= 478 and coords.shape[1] == 3
                         and px.ndim == 2 and px.shape[0] >= 478 and px.shape[1] == 2
                         and np.isfinite(coords[:478]).all() and np.isfinite(px[:478]).all())
            except (ValueError, TypeError, OverflowError):
                valid = False
        if not valid:
            if draw:
                _draw_hud(annotated, ["FACE MODE: NO FACE"], (80, 190, 230))
            return ControllerOutput(self.mode.value, False, annotated, None, {})

        # Preserve the shared pixel array itself when it already has the exact shape/type.
        px = px if px.shape[0] == 478 else px[:478]
        nose_x, nose_y = map(float, coords[1, :2])
        cx = float((coords[234, 0] + coords[454, 0]) / 2.0)
        cy = float((coords[10, 1] + coords[152, 1]) / 2.0)
        face_width = abs(float(coords[454, 0] - coords[234, 0]))
        face_height = abs(float(coords[152, 1] - coords[10, 1]))
        yaw = (nose_x - cx) / max(face_width, 1e-6)
        pitch = (nose_y - cy) / max(face_height, 1e-6)
        # Pixel coordinates give a physical image-plane angle on non-square frames.
        left_eye, right_eye = px[33].astype(np.float64), px[263].astype(np.float64)
        roll = math.degrees(math.atan2(float(right_eye[1] - left_eye[1]),
                                       float(right_eye[0] - left_eye[0])))
        features = {"nose_tip_norm": (nose_x, nose_y), "face_center_norm": (cx, cy),
                    "yaw_offset": float(yaw), "pitch_offset": float(pitch),
                    "roll_deg": float(roll), "landmarks_norm": normalized}
        if draw:
            points = _drawing_points(px, annotated)
            for point in points:
                cv2.circle(annotated, tuple(point), 1, (130, 180, 150), -1, cv2.LINE_8)
            height, width = frame.shape[:2]
            center = (int(round(cx * width)), int(round(cy * height)))
            nose = tuple(points[1])
            cv2.line(annotated, center, nose, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(annotated, center, 4, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.circle(annotated, nose, 4, (90, 180, 255), -1, cv2.LINE_AA)
            _draw_hud(annotated, ["MODE: FACE (478-pt)",
                                  f"YAW: {yaw:+.3f}  PITCH: {pitch:+.3f}",
                                  f"ROLL: {roll:+.1f} deg"])
        return ControllerOutput(self.mode.value, True, annotated, px, features, confidence=1.0)
