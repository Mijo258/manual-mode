"""Standalone dlib eyelid geometry and masked-ROI dark-centroid pupil features."""

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import contracts
from .base import BaseController, ControllerOutput, _prepare_frame, _draw_hud, _drawing_points


ROOT = Path(__file__).resolve().parent.parent


class EyeController(BaseController):
    mode = contracts.ControlMode.EYE

    def __init__(self, config: dict | None = None, predictor_path: str | None = None,
                 detector=None, predictor=None, detect_scale: float = 0.5):
        self.config = contracts.load_config() if config is None else config
        self.detect_scale = float(detect_scale)
        if not math.isfinite(self.detect_scale) or not 0.0 < self.detect_scale <= 1.0:
            raise ValueError("detect_scale must be finite and within (0, 1].")
        self._closed = False
        self.predictor_path: Path | None = None
        if predictor is None:
            configured = self.config.get("assets", {}).get("dlib_shape_predictor_68")
            if predictor_path is not None:
                candidates = [Path(predictor_path).expanduser()]
            else:
                candidates = ([Path(configured).expanduser()] if configured else [])
                candidates.append(ROOT / "shape_predictor_68_face_landmarks.dat")
            candidates = [path if path.is_absolute() else ROOT / path for path in candidates]
            self.predictor_path = next((path.resolve() for path in candidates if path.is_file()), None)
            if self.predictor_path is None:
                raise FileNotFoundError("68-point dlib predictor not found. Checked: "
                                        + ", ".join(map(str, candidates)))
        import dlib

        self._dlib = dlib
        self._detector = dlib.get_frontal_face_detector() if detector is None else detector
        self._predictor = dlib.shape_predictor(str(self.predictor_path)) if predictor is None else predictor

    @staticmethod
    def _ear(eye: np.ndarray) -> float:
        points = eye.astype(np.float64)
        width = float(np.linalg.norm(points[0] - points[3]))
        if width <= 1e-9:
            return 0.0
        return float((np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4]))
                     / (2.0 * width))

    @staticmethod
    def _pupil(gray: np.ndarray, eye: np.ndarray, ear: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return bbox-relative pupil ratios and the estimated full-frame center.

        Mask/erode the eyelid polygon to reduce boundary/eyelash influence, then
        find the centroid of its dark Otsu-selected pixels. Flat, tiny or closed
        eyes have no reliable pupil evidence and return the neutral ratio.
        """
        center = tuple(map(float, eye.mean(axis=0)))
        fallback = ((0.5, 0.5), center)
        if ear <= 0.12:
            return fallback
        height, width = gray.shape
        x0, y0 = np.floor(eye.min(axis=0)).astype(int)
        x1, y1 = np.ceil(eye.max(axis=0)).astype(int) + 1
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        if x1 - x0 < 4 or y1 - y0 < 3:
            return fallback
        roi = gray[y0:y1, x0:x1]
        mask = np.zeros(roi.shape, dtype=np.uint8)
        polygon = np.rint(eye - (x0, y0)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        if min(roi.shape) >= 5:
            mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        blurred = cv2.GaussianBlur(roi, (3, 3), 0)
        values = blurred[mask != 0]
        if values.size < 6 or int(values.max()) - int(values.min()) < 10:
            return fallback
        threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255,
                                     cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        dark = cv2.bitwise_and(cv2.inRange(blurred, 0, threshold), mask)
        moments = cv2.moments(dark, binaryImage=True)
        if moments["m00"] <= 0:
            return fallback
        local_x, local_y = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        ratios = (max(0.0, min(1.0, local_x / (roi.shape[1] - 1))),
                  max(0.0, min(1.0, local_y / (roi.shape[0] - 1))))
        return ratios, (float(x0 + local_x), float(y0 + local_y))

    @staticmethod
    def _iris_ratio(shared_context: dict[str, Any] | None) -> float | None:
        pixels = (shared_context or {}).get("face_landmarks_px")
        if pixels is None:
            return None
        try:
            points = np.asarray(pixels, dtype=np.float64)
            if points.ndim != 2 or points.shape[0] < 478 or points.shape[1] != 2:
                return None
            ratios = []
            for iris, a, b in ((468, 33, 133), (473, 362, 263)):
                if not np.isfinite(points[[iris, a, b]]).all():
                    return None
                left, right = (points[a], points[b]) if points[a, 0] <= points[b, 0] else (points[b], points[a])
                axis = right - left
                denominator = float(np.dot(axis, axis))
                if denominator <= 1e-9:
                    return None
                ratio = float(np.dot(points[iris] - left, axis)) / denominator
                ratios.append(max(0.0, min(1.0, ratio)))
            return float(sum(ratios) / 2.0)
        except (ValueError, TypeError, OverflowError):
            return None

    @staticmethod
    def _iris_pupil_ratio(shared_context: dict[str, Any] | None) -> tuple[float, float] | None:
        """Reuse shared iris/eyelid geometry in image-left/right and top/bottom axes."""
        pixels = (shared_context or {}).get("face_landmarks_px")
        if pixels is None:
            return None
        try:
            points = np.asarray(pixels, dtype=np.float64)
            if points.ndim != 2 or points.shape[0] < 478 or points.shape[1] != 2:
                return None
            ratios = []
            for iris, a, b, upper, lower in ((468, 33, 133, 159, 145), (473, 362, 263, 386, 374)):
                if not np.isfinite(points[[iris, a, b, upper, lower]]).all():
                    return None
                left, right = (points[a], points[b]) if points[a, 0] <= points[b, 0] else (points[b], points[a])
                horizontal = right - left
                width_squared = float(np.dot(horizontal, horizontal))
                if not math.isfinite(width_squared) or width_squared <= 1e-9:
                    return None
                ix = float(np.dot(points[iris] - left, horizontal)) / width_squared
                top, bottom = ((points[upper], points[lower]) if points[upper, 1] <= points[lower, 1]
                               else (points[lower], points[upper]))
                vertical = bottom - top
                height_squared = float(np.dot(vertical, vertical))
                if not math.isfinite(height_squared):
                    return None
                # A near-closed eyelid axis has no reliable vertical pupil evidence.
                # Neutral Y avoids division by zero/noise; the mapper separately brakes on EAR.
                minimum_gap = max(1.0, 0.02 * math.sqrt(width_squared))
                iy = 0.5 if height_squared <= minimum_gap ** 2 else float(np.dot(points[iris] - top, vertical)) / height_squared
                if not math.isfinite(ix) or not math.isfinite(iy):
                    return None
                ratios.append((max(0.0, min(1.0, ix)), max(0.0, min(1.0, iy))))
            return tuple((left + right) / 2.0 for left, right in zip(*ratios))
        except (ValueError, TypeError, OverflowError):
            return None

    def process_frame(self, frame: np.ndarray, timestamp_ms: int, draw: bool = True,
                      shared_context: dict[str, Any] | None = None) -> ControllerOutput:
        if self._closed:
            raise RuntimeError("Cannot process frames with a closed EyeController.")
        annotated = _prepare_frame(frame, draw)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        small_size = (max(1, round(width * self.detect_scale)), max(1, round(height * self.detect_scale)))
        small = cv2.resize(gray, small_size, interpolation=cv2.INTER_AREA)
        faces = self._detector(small, 0)
        if len(faces) == 0:
            if draw:
                _draw_hud(annotated, ["EYE MODE: NO FACE"], (80, 190, 230))
            return ControllerOutput(self.mode.value, False, annotated, None, {})
        face = max(faces, key=lambda rect: rect.area())
        sx, sy = width / small.shape[1], height / small.shape[0]
        left, top = max(0, math.floor(face.left() * sx)), max(0, math.floor(face.top() * sy))
        right = min(width - 1, math.ceil((face.right() + 1) * sx) - 1)
        bottom = min(height - 1, math.ceil((face.bottom() + 1) * sy) - 1)
        if right < left or bottom < top:
            if draw:
                _draw_hud(annotated, ["EYE MODE: NO FACE"], (80, 190, 230))
            return ControllerOutput(self.mode.value, False, annotated, None, {})
        rect = self._dlib.rectangle(left, top, right, bottom)
        shape = self._predictor(gray, rect)
        if getattr(shape, "num_parts", 0) < 68:
            if draw:
                _draw_hud(annotated, ["EYE MODE: INVALID LANDMARKS"], (80, 190, 230))
            return ControllerOutput(self.mode.value, False, annotated, None, {})
        px = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 48)], dtype=np.float32)
        if not np.isfinite(px).all():
            if draw:
                _draw_hud(annotated, ["EYE MODE: INVALID LANDMARKS"], (80, 190, 230))
            return ControllerOutput(self.mode.value, False, annotated, None, {})
        left_eye, right_eye = px[:6], px[6:]
        left_ear, right_ear = self._ear(left_eye), self._ear(right_eye)
        left_ratio, left_center = self._pupil(gray, left_eye, left_ear)
        right_ratio, right_center = self._pupil(gray, right_eye, right_ear)
        average = tuple((a + b) / 2.0 for a, b in zip(left_ratio, right_ratio))
        features = {"left_ear": left_ear, "right_ear": right_ear,
                    "avg_ear": (left_ear + right_ear) / 2.0,
                    "left_eye_px": left_eye, "right_eye_px": right_eye,
                    "left_pupil_ratio": left_ratio, "right_pupil_ratio": right_ratio,
                    "avg_pupil_ratio": average}
        iris_ratio = self._iris_ratio(shared_context)
        if iris_ratio is not None:
            features["iris_gaze_ratio"] = iris_ratio
        iris_pupil_ratio = self._iris_pupil_ratio(shared_context)
        if iris_pupil_ratio is not None:
            features["iris_pupil_ratio"] = iris_pupil_ratio
        if draw:
            points = _drawing_points(px, annotated)
            for eye in (points[:6], points[6:]):
                cv2.polylines(annotated, [eye.reshape(-1, 1, 2)], True, (110, 220, 150), 1, cv2.LINE_AA)
            centers = _drawing_points(np.array([left_center, right_center]), annotated)
            for center in centers:
                cv2.circle(annotated, tuple(center), 2, (0, 190, 255), -1, cv2.LINE_AA)
            _draw_hud(annotated, ["MODE: EYE (dlib 68-pt)",
                                  f"L-EAR: {left_ear:.3f}  R-EAR: {right_ear:.3f}",
                                  f"PUPIL X/Y: {average[0]:.2f} / {average[1]:.2f}"])
        return ControllerOutput(self.mode.value, True, annotated, px, features, confidence=1.0)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            detector, predictor = self._detector, self._predictor
            self._detector = self._predictor = None
            for resource in (detector, predictor):
                close = getattr(resource, "close", None)
                if callable(close):
                    close()
