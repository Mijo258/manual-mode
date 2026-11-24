"""Frame-in driver monitoring and independently managed, non-blocking alarms."""

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

import contracts


ROOT = Path(__file__).resolve().parent
LANDMARK_COUNT = 478
LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
MOUTH_INDICES = (13, 14, 78, 308)
IRIS_EYE_PAIRS = ((468, 33, 133), (473, 362, 263))
_GEOMETRY_EPSILON = 1e-9


@dataclass
class SafetyFrameResult:
    status: contracts.SafetyStatus
    annotated_frame: np.ndarray
    landmarks_norm: list[tuple[float, float, float]] | None
    landmarks_px: np.ndarray | None


def _points(landmarks_px: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < LANDMARK_COUNT or points.shape[1] != 2:
        raise ValueError("Expected at least 478 pixel landmarks with shape (N, 2).")
    if not np.isfinite(points).all():
        raise ValueError("Landmark coordinates must be finite.")
    return points


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    return math.hypot(float(first[0] - second[0]), float(first[1] - second[1]))


def compute_ear(landmarks_px: np.ndarray) -> float:
    """Mean six-point eyelid EAR; collapsed eye corners raise ValueError."""
    points = _points(landmarks_px)
    ratios = []
    for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
        p1, p2, p3, p4, p5, p6 = points[list(indices)]
        width = _distance(p1, p4)
        if width <= _GEOMETRY_EPSILON:
            raise ValueError("Eye corners overlap; EAR is undefined.")
        ratios.append((_distance(p2, p6) + _distance(p3, p5)) / (2.0 * width))
    return float(sum(ratios) / 2.0)


def compute_mar(landmarks_px: np.ndarray) -> float:
    """Inner-lip opening divided by mouth width, using indices 13/14/78/308."""
    points = _points(landmarks_px)
    upper, lower, left, right = points[list(MOUTH_INDICES)]
    width = _distance(left, right)
    if width <= _GEOMETRY_EPSILON:
        raise ValueError("Mouth corners overlap; MAR is undefined.")
    return _distance(upper, lower) / width


def compute_gaze(landmarks_px: np.ndarray) -> tuple[float, float, float]:
    """Project each iris onto its eye axis, ordered by increasing image x.

    The first ratio uses (468, 33, 133), the second (473, 362, 263), exactly
    matching safety_daemon.py. Both ratios are clamped before averaging.
    """
    points = _points(landmarks_px)
    ratios = []
    for iris_index, corner_a_index, corner_b_index in IRIS_EYE_PAIRS:
        corner_a, corner_b = points[corner_a_index], points[corner_b_index]
        left, right = (corner_a, corner_b) if corner_a[0] <= corner_b[0] else (corner_b, corner_a)
        axis = right - left
        axis_squared = float(np.dot(axis, axis))
        if axis_squared <= _GEOMETRY_EPSILON:
            raise ValueError("Eye corners overlap; gaze is undefined.")
        ratio = float(np.dot(points[iris_index] - left, axis)) / axis_squared
        ratios.append(max(0.0, min(1.0, ratio)))
    return ratios[0], ratios[1], (ratios[0] + ratios[1]) / 2.0


class AlarmManager:
    """Throttle alarm notifications; all filesystem/audio work runs on a worker."""

    _PRIORITY = ("SLEEP", "DISTRACTION", "YAWN")
    _PATTERNS = {
        "SLEEP": ((1400, 200), (1400, 200)),
        "DISTRACTION": ((1100, 150), (1100, 150)),
        "YAWN": ((800, 250),),
    }

    def __init__(self, config: dict | None = None, muted: bool = False):
        cfg = contracts.load_config() if config is None else config
        self.alarm_cooldown_sec = float(cfg.get("safety_dms", {}).get("alarm_cooldown_sec", 2.0))
        if not math.isfinite(self.alarm_cooldown_sec) or self.alarm_cooldown_sec < 0:
            raise ValueError("Alarm cooldown must be nonnegative and finite.")
        self.muted = bool(muted)
        assets = cfg.get("assets", {})
        self._assets = {
            "SLEEP": assets.get("alarm_sound_sleep"),
            "YAWN": assets.get("alarm_sound_yawn"),
            "DISTRACTION": assets.get("alarm_sound_gaze"),
        }
        self._lock = threading.Lock()
        self._last_trigger_sec = float("-inf")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._async_playing = threading.Event()
        self._sound_backend = None

    def trigger(self, active_alarms: list[str], now_sec: float | None = None) -> bool:
        if not active_alarms:
            return False
        now = time.monotonic() if now_sec is None else float(now_sec)
        if not math.isfinite(now):
            raise ValueError("Alarm timestamp must be finite.")
        alarm = next((name for name in self._PRIORITY if name in active_alarms), None)
        with self._lock:
            if self._stop_event.is_set() or now - self._last_trigger_sec < self.alarm_cooldown_sec:
                return False
            self._last_trigger_sec = now
            if not self.muted and alarm is not None and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._play, args=(alarm,),
                                                name="dms-alarm", daemon=True)
                try:
                    self._thread.start()
                except RuntimeError:
                    self._thread = None  # Audio resource exhaustion must not interrupt monitoring.
        return True

    def _cancel_async_sound(self) -> None:
        if self._async_playing.is_set() and self._sound_backend is not None:
            try:
                self._sound_backend.PlaySound(None, 0)
            except (RuntimeError, OSError):
                pass
            self._async_playing.clear()

    def _play(self, alarm: str) -> None:
        if self._stop_event.is_set():
            return
        try:
            import winsound
        except ImportError:
            return
        self._sound_backend = winsound
        asset = self._assets.get(alarm)
        if asset:
            try:
                path = Path(asset).expanduser()
                if not path.is_absolute():
                    path = ROOT / path
                if path.suffix.lower() == ".wav" and path.is_file() and not self._stop_event.is_set():
                    winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                    self._async_playing.set()
                    if self._stop_event.is_set():
                        self._cancel_async_sound()
                    return
            except (RuntimeError, OSError, ValueError, TypeError):
                pass  # Missing/unsupported audio assets fall back to the alarm's beep pattern.
        for frequency, duration_ms in self._PATTERNS[alarm]:
            if self._stop_event.is_set():
                break
            try:
                winsound.Beep(frequency, duration_ms)
            except (RuntimeError, OSError):
                break

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self._cancel_async_sound()


class SafetyEngine:
    """Process frames sequentially, sharing one Tasks inference with controllers.

    Duration timers use the guarded frame timestamp in milliseconds. Invalid
    landmarks reset timers just like a missing face; geometry is never replaced
    with a fabricated closed-eye measurement.
    """

    def __init__(self, config: dict | None = None, model_path: str | None = None,
                 landmarker=None):
        self.config = contracts.load_config() if config is None else config
        settings = self.config.get("safety_dms", {})
        self.ear_threshold = float(settings.get("ear_threshold", 0.20))
        self.ear_duration_sec = float(settings.get("ear_duration_sec", 1.5))
        self.mar_threshold = float(settings.get("mar_threshold", 0.50))
        self.mar_duration_sec = float(settings.get("mar_duration_sec", 1.5))
        self.gaze_left_threshold = float(settings.get("gaze_left_threshold", 0.25))
        self.gaze_right_threshold = float(settings.get("gaze_right_threshold", 0.75))
        self.gaze_duration_sec = float(settings.get("gaze_duration_sec", 2.0))
        values = (self.ear_threshold, self.mar_threshold, self.gaze_left_threshold,
                  self.gaze_right_threshold, self.ear_duration_sec,
                  self.mar_duration_sec, self.gaze_duration_sec)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Safety thresholds and durations must be finite.")
        if self.ear_threshold <= 0 or self.mar_threshold <= 0:
            raise ValueError("EAR and MAR thresholds must be positive.")
        if not 0.0 <= self.gaze_left_threshold < self.gaze_right_threshold <= 1.0:
            raise ValueError("Gaze thresholds must satisfy 0 <= left < right <= 1.")
        if min(self.ear_duration_sec, self.mar_duration_sec, self.gaze_duration_sec) < 0:
            raise ValueError("Safety durations must be nonnegative.")
        self._sleep_start_ms: int | None = None
        self._yawn_start_ms: int | None = None
        self._gaze_start_ms: int | None = None
        self._last_timestamp_ms = -1
        self._closed = False
        self.model_path: Path | None = None
        if landmarker is None:
            self.model_path = self._resolve_model(model_path)
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            landmarker = vision.FaceLandmarker.create_from_options(options)
        if not callable(getattr(landmarker, "detect_for_video", None)):
            raise TypeError("Landmarker must expose detect_for_video(image, timestamp_ms).")
        self._landmarker = landmarker

    @staticmethod
    def _resolve_model(model_path: str | None) -> Path:
        if model_path is not None:
            candidate = Path(model_path).expanduser()
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            candidates = (candidate,)
        else:
            candidates = (ROOT / "face_landmarker.task",
                          ROOT / "face-landmark-project" / "face_landmarker.task")
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError("FaceLandmarker model not found. Checked: "
                                + ", ".join(str(path) for path in candidates))

    @staticmethod
    def _duration_state(condition: bool, start_ms: int | None, ts_ms: int,
                        duration_sec: float) -> tuple[int | None, bool]:
        if not condition:
            return None, False
        if start_ms is None:
            start_ms = ts_ms
        return start_ms, (ts_ms - start_ms) / 1000.0 >= duration_sec

    def reset_timers(self) -> None:
        self._sleep_start_ms = self._yawn_start_ms = self._gaze_start_ms = None

    def process_frame(self, frame: np.ndarray, timestamp_ms: int,
                      draw: bool = True) -> SafetyFrameResult:
        if self._closed:
            raise RuntimeError("Cannot process a frame after SafetyEngine.close().")
        if (not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3
                or frame.shape[2] != 3 or frame.dtype != np.uint8):
            raise ValueError("Expected a nonempty BGR uint8 image with shape (H, W, 3).")
        annotated = frame.copy() if draw else frame
        ts_ms = max(self._last_timestamp_ms + 1, int(timestamp_ms))
        self._last_timestamp_ms = ts_ms
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = self._landmarker.detect_for_video(image, ts_ms)
        except Exception as exc:
            self.reset_timers()
            raise RuntimeError(f"FaceLandmarker inference failed: {exc}") from exc

        status = contracts.SafetyStatus(ts_ms / 1000.0, False, 0.0, 0.0, 0.5,
                                        False, False, False, [])
        landmarks_norm = None
        landmarks_px = None
        faces = getattr(detection, "face_landmarks", None)
        if faces and len(faces[0]) >= LANDMARK_COUNT:
            try:
                normalized = [(float(point.x), float(point.y), float(point.z))
                              for point in faces[0][:LANDMARK_COUNT]]
                coordinates = np.asarray(normalized, dtype=np.float64)
                if not np.isfinite(coordinates).all():
                    raise ValueError("Non-finite face landmarks.")
                height, width = frame.shape[:2]
                pixels = (coordinates[:, :2] * (width, height)).astype(np.float32)
                ear, mar = compute_ear(pixels), compute_mar(pixels)
                left_gaze, right_gaze, gaze_ratio = compute_gaze(pixels)
            except (ValueError, TypeError, AttributeError, IndexError, OverflowError):
                self.reset_timers()
            else:
                sleep_cond = ear < self.ear_threshold
                yawn_cond = mar > self.mar_threshold
                gaze_cond = ((left_gaze < self.gaze_left_threshold and right_gaze < self.gaze_left_threshold)
                             or (left_gaze > self.gaze_right_threshold and right_gaze > self.gaze_right_threshold))
                self._sleep_start_ms, sleepy = self._duration_state(
                    sleep_cond, self._sleep_start_ms, ts_ms, self.ear_duration_sec)
                self._yawn_start_ms, yawning = self._duration_state(
                    yawn_cond, self._yawn_start_ms, ts_ms, self.mar_duration_sec)
                self._gaze_start_ms, distracted = self._duration_state(
                    gaze_cond, self._gaze_start_ms, ts_ms, self.gaze_duration_sec)
                alarms = [name for name, active in (("SLEEP", sleepy), ("YAWN", yawning),
                                                   ("DISTRACTION", distracted)) if active]
                status = contracts.SafetyStatus(ts_ms / 1000.0, True, ear, mar, gaze_ratio,
                                                sleepy, yawning, distracted, alarms)
                landmarks_norm, landmarks_px = normalized, pixels
        else:
            self.reset_timers()
        if draw:
            self._draw_hud(annotated, status, landmarks_px)
        return SafetyFrameResult(status, annotated, landmarks_norm, landmarks_px)

    @staticmethod
    def _draw_hud(frame: np.ndarray, status: contracts.SafetyStatus,
                  landmarks_px: np.ndarray | None) -> None:
        height, width = frame.shape[:2]
        if landmarks_px is not None:
            points = np.rint(np.clip(landmarks_px, (0, 0), (width - 1, height - 1))).astype(np.int32)
            for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
                cv2.polylines(frame, [points[list(indices)].reshape(-1, 1, 2)],
                              True, (120, 190, 140), 1, cv2.LINE_AA)
            cv2.line(frame, tuple(points[13]), tuple(points[14]), (140, 190, 210), 1, cv2.LINE_AA)
            for index in (78, 308):
                cv2.circle(frame, tuple(points[index]), 2, (140, 190, 210), -1, cv2.LINE_AA)
            for index in (468, 473):
                cv2.circle(frame, tuple(points[index]), 3, (210, 210, 160), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (8, 8), (min(width - 1, 510), 76), (24, 24, 24), -1)
        telemetry = f"EAR: {status.ear:.3f}   MAR: {status.mar:.3f}   GAZE: {status.gaze_ratio:.2f}"
        cv2.putText(frame, telemetry, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)
        if status.active_alarms:
            cv2.rectangle(frame, (8, 42), (min(width - 1, 510), 76), (0, 0, 165), -1)
            label, color, thickness = "ALARM: " + " | ".join(status.active_alarms), (255, 255, 255), 2
        elif status.face_detected:
            label, color, thickness = "DMS: OK", (80, 220, 100), 1
        else:
            label, color, thickness = "DMS: NO FACE / INVALID LANDMARKS", (80, 190, 230), 1
        cv2.putText(frame, label, (16, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, thickness, cv2.LINE_AA)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.reset_timers()
            close = getattr(self._landmarker, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "SafetyEngine":
        if self._closed:
            raise RuntimeError("Cannot enter a closed SafetyEngine.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
