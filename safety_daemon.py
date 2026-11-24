"""Always-on driver-monitoring safety daemon for the AV perception engine."""

from __future__ import annotations

import logging
import math
import time
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple, Protocol, Sequence, cast

import cv2
import numpy as np
from numpy.typing import NDArray


Frame = NDArray[np.uint8]
LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "face-landmark-project" / "face_landmarker.task"
)
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


class Landmark(Protocol):
    """Structural type for two-dimensional landmark coordinates."""

    x: float
    y: float


class Point2D(NamedTuple):
    """Pixel coordinates used to measure undistorted face geometry."""

    x: float
    y: float


class FaceLandmarkerProcessor(Protocol):
    """Operations required from a MediaPipe Tasks FaceLandmarker instance."""

    def detect_for_video(self, image: Any, timestamp_ms: int) -> Any:
        """Process an RGB MediaPipe Image with an increasing timestamp."""
        ...

    def close(self) -> None:
        """Release model resources."""
        ...


class SafetyDaemonError(RuntimeError):
    """Raised when the driver-monitoring daemon cannot operate safely."""


class SafetyDaemon:
    """Evaluate driver alertness from MediaPipe face and iris landmarks.

    A condition must remain present for its configured number of consecutive
    frames before an alarm is shown. Missing faces or invalid landmark geometry
    reset every counter so observations from separate detections are not joined.
    """

    LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
    MOUTH_INDICES = (13, 14, 78, 308)
    IRIS_EYE_PAIRS = ((468, 33, 133), (473, 362, 263))

    EAR_THRESHOLD = 0.20
    MAR_THRESHOLD = 0.50
    GAZE_RATIO_MIN = 0.25
    GAZE_RATIO_MAX = 0.75

    SLEEPY_FRAME_LIMIT = 20
    YAWN_FRAME_LIMIT = 15
    DISTRACTED_FRAME_LIMIT = 30
    REQUIRED_LANDMARKS = 478

    _GEOMETRY_EPSILON = 1e-9

    @staticmethod
    def _create_face_landmarker(
        model_path: Path,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> FaceLandmarkerProcessor:
        """Load the same 478-point Tasks model used by detect_facemesh.py."""
        if not model_path.is_file():
            raise SafetyDaemonError(
                f"Model file not found: {model_path}\n"
                "Download the model with:\n"
                f'  curl.exe -fL -o "{model_path}" {MODEL_DOWNLOAD_URL}'
            )

        try:
            mp_python = import_module("mediapipe.tasks.python")
            vision = import_module("mediapipe.tasks.python.vision")
        except ImportError as exc:
            raise SafetyDaemonError(
                "MediaPipe Tasks is unavailable. Install into this interpreter "
                "with: python -m pip install -r requirements-facemesh.txt. "
                f"Underlying error: {exc}"
            ) from exc

        try:
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            return cast(
                FaceLandmarkerProcessor,
                vision.FaceLandmarker.create_from_options(options),
            )
        except Exception as exc:
            raise SafetyDaemonError(
                f"Unable to load the MediaPipe Tasks model at {model_path}: {exc}. "
                "Check that it is a complete face_landmarker.task bundle "
                "(3,758,596 bytes)."
            ) from exc

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | Path | None = None,
    ) -> None:
        """Initialize the 478-point Face Landmarker driver-monitoring model.

        Args:
            min_detection_confidence: Minimum initial face-detection confidence.
            min_tracking_confidence: Minimum face-landmark tracking confidence.
            model_path: Optional .task bundle; defaults to the face project's model.

        Raises:
            ValueError: If either confidence is outside the inclusive range 0-1.
            SafetyDaemonError: If MediaPipe Tasks or its model cannot be loaded.
        """
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0 and 1.")
        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be between 0 and 1.")

        self.sleepy_frames = 0
        self.yawn_frames = 0
        self.distracted_frames = 0
        self.active_alarms: tuple[str, ...] = ()
        self.landmark_count = 0
        self.ear: float | None = None
        self.mar: float | None = None
        self.gaze_off_center: bool | None = None
        self.status = "NO FACE"
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.model_path = self.model_path.resolve()
        self._last_timestamp_ms = -1
        self._closed = False

        try:
            self._mediapipe = import_module("mediapipe")
            self._face_landmarker = self._create_face_landmarker(
                model_path=self.model_path,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except SafetyDaemonError:
            raise
        except Exception as exc:
            raise SafetyDaemonError(
                "Unable to initialize MediaPipe Tasks. Install with: "
                "python -m pip install -r requirements-facemesh.txt. "
                f"Underlying error: {exc}"
            ) from exc
        LOGGER.info("Loaded MediaPipe Tasks FaceLandmarker: %s", self.model_path)

    @staticmethod
    def _distance(first: Landmark, second: Landmark) -> float:
        """Return the two-dimensional Euclidean landmark distance."""
        return math.hypot(first.x - second.x, first.y - second.y)

    @classmethod
    def _single_eye_ear(
        cls,
        landmarks: Sequence[Landmark],
        indices: tuple[int, int, int, int, int, int],
    ) -> float:
        """Calculate EAR for one eye using six ordered eye landmarks."""
        p1, p2, p3, p4, p5, p6 = (landmarks[index] for index in indices)
        horizontal = cls._distance(p1, p4)
        if horizontal <= cls._GEOMETRY_EPSILON:
            raise ValueError("Eye corner landmarks overlap; EAR is undefined.")

        return (
            cls._distance(p2, p6) + cls._distance(p3, p5)
        ) / (2.0 * horizontal)

    @classmethod
    def calculate_ear(cls, landmarks: Sequence[Landmark]) -> float:
        """Calculate mean EAR for both eyes from pixel-coordinate landmarks."""
        left_ear = cls._single_eye_ear(landmarks, cls.LEFT_EYE_INDICES)
        right_ear = cls._single_eye_ear(landmarks, cls.RIGHT_EYE_INDICES)
        return (left_ear + right_ear) / 2.0

    @classmethod
    def calculate_mar(cls, landmarks: Sequence[Landmark]) -> float:
        """Calculate inner-lip MAR from pixel-coordinate landmarks."""
        upper_index, lower_index, left_index, right_index = cls.MOUTH_INDICES
        mouth_width = cls._distance(
            landmarks[left_index], landmarks[right_index]
        )
        if mouth_width <= cls._GEOMETRY_EPSILON:
            raise ValueError("Mouth corner landmarks overlap; MAR is undefined.")

        mouth_opening = cls._distance(
            landmarks[upper_index], landmarks[lower_index]
        )
        return mouth_opening / mouth_width

    @classmethod
    def _iris_horizontal_ratio(
        cls,
        landmarks: Sequence[Landmark],
        iris_index: int,
        corner_a_index: int,
        corner_b_index: int,
    ) -> float:
        """Project an iris center onto its eye-corner axis from left to right."""
        iris = landmarks[iris_index]
        corner_a = landmarks[corner_a_index]
        corner_b = landmarks[corner_b_index]

        left_corner, right_corner = (
            (corner_a, corner_b)
            if corner_a.x <= corner_b.x
            else (corner_b, corner_a)
        )
        axis_x = right_corner.x - left_corner.x
        axis_y = right_corner.y - left_corner.y
        axis_length_squared = (axis_x * axis_x) + (axis_y * axis_y)
        if axis_length_squared <= cls._GEOMETRY_EPSILON:
            raise ValueError("Eye corner landmarks overlap; gaze is undefined.")

        iris_x = iris.x - left_corner.x
        iris_y = iris.y - left_corner.y
        return ((iris_x * axis_x) + (iris_y * axis_y)) / axis_length_squared

    @classmethod
    def calculate_gaze(cls, landmarks: Sequence[Landmark]) -> bool:
        """Return ``True`` when both irises indicate a severe sideways gaze.

        Each iris is projected onto the axis between its eye corners. Requiring
        both eyes to cross the same boundary reduces false alarms from a blink,
        partial occlusion, or a single noisy iris estimate.
        """
        ratios = tuple(
            cls._iris_horizontal_ratio(
                landmarks,
                iris_index,
                corner_a_index,
                corner_b_index,
            )
            for iris_index, corner_a_index, corner_b_index in cls.IRIS_EYE_PAIRS
        )
        looking_left = all(ratio < cls.GAZE_RATIO_MIN for ratio in ratios)
        looking_right = all(ratio > cls.GAZE_RATIO_MAX for ratio in ratios)
        return looking_left or looking_right

    def _reset_counters(self) -> None:
        """Clear all consecutive-condition counters."""
        self.sleepy_frames = 0
        self.yawn_frames = 0
        self.distracted_frames = 0
        self.active_alarms = ()
        self.landmark_count = 0
        self.ear = None
        self.mar = None
        self.gaze_off_center = None

    def _draw_status(self, frame: Frame) -> Frame:
        """Show whether monitoring is working and the current measurements."""
        lines = [self.status]
        if self.ear is not None and self.mar is not None:
            gaze = "AWAY" if self.gaze_off_center else "CENTER"
            lines.append(f"EAR: {self.ear:.3f}  MAR: {self.mar:.3f}  Gaze: {gaze}")
        for index, line in enumerate(lines):
            cv2.putText(
                frame, line, (10, frame.shape[0] - 45 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )
        return frame

    @staticmethod
    def _draw_warnings(frame: Frame, alarms: Sequence[str]) -> None:
        """Draw active alarms in red without obscuring the driver's face."""
        for line_number, alarm in enumerate(alarms):
            cv2.putText(
                frame,
                f"WARNING: {alarm}",
                (20, 65 + (line_number * 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

    def process_frame(self, frame: Frame) -> Frame:
        """Analyze one BGR frame, update state, and draw active alarms.

        Args:
            frame: Non-empty OpenCV BGR image.

        Returns:
            The supplied frame with any active warning text drawn onto it.

        Raises:
            ValueError: If ``frame`` is not a valid BGR image.
            SafetyDaemonError: If closed or model inference fails.
        """
        if self._closed:
            raise SafetyDaemonError("Cannot process a frame after daemon shutdown.")
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("process_frame requires a non-empty NumPy frame.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("process_frame requires a three-channel BGR frame.")
        if frame.dtype != np.uint8:
            raise ValueError("process_frame requires uint8 pixel values.")

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self._mediapipe.Image(
                image_format=self._mediapipe.ImageFormat.SRGB, data=rgb_frame,
            )
            # VIDEO mode rejects equal or decreasing timestamps, including when
            # multiple frames are processed in the same millisecond.
            timestamp_ms = max(
                int(time.monotonic() * 1000), self._last_timestamp_ms + 1,
            )
            self._last_timestamp_ms = timestamp_ms
            results = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            self._reset_counters()
            self.status = "MONITORING ERROR"
            raise SafetyDaemonError(f"Face Landmarker frame processing failed: {exc}") from exc

        if not results.face_landmarks:
            self._reset_counters()
            self.status = "NO FACE"
            return self._draw_status(frame)

        normalized_landmarks = cast(
            Sequence[Landmark], results.face_landmarks[0]
        )
        if len(normalized_landmarks) < self.REQUIRED_LANDMARKS:
            self._reset_counters()
            self.status = "INVALID LANDMARKS"
            LOGGER.warning(
                "Face Landmarker returned %d landmarks; %d are required for iris gaze.",
                len(normalized_landmarks),
                self.REQUIRED_LANDMARKS,
            )
            return self._draw_status(frame)

        try:
            # x and y are normalized by different dimensions. Measuring them
            # directly would stretch eye/mouth ratios on a non-square frame.
            height, width = frame.shape[:2]
            landmarks = tuple(
                Point2D(point.x * width, point.y * height)
                for point in normalized_landmarks
            )
            if not all(math.isfinite(p.x) and math.isfinite(p.y) for p in landmarks):
                raise ValueError("Landmarks contain non-finite coordinates.")
            ear = self.calculate_ear(landmarks)
            mar = self.calculate_mar(landmarks)
            gaze_off_center = self.calculate_gaze(landmarks)
        except (IndexError, ValueError) as exc:
            self._reset_counters()
            self.status = "INVALID LANDMARKS"
            LOGGER.warning("Invalid Face Landmarker geometry: %s", exc)
            return self._draw_status(frame)

        self.landmark_count = len(landmarks)
        self.ear = ear
        self.mar = mar
        self.gaze_off_center = gaze_off_center
        self.status = f"TRACKING: {self.landmark_count} landmarks"

        self.sleepy_frames = (
            self.sleepy_frames + 1 if ear < self.EAR_THRESHOLD else 0
        )
        self.yawn_frames = (
            self.yawn_frames + 1 if mar > self.MAR_THRESHOLD else 0
        )
        self.distracted_frames = (
            self.distracted_frames + 1 if gaze_off_center else 0
        )

        alarms: list[str] = []
        if self.sleepy_frames >= self.SLEEPY_FRAME_LIMIT:
            alarms.append("SLEEPING")
        if self.yawn_frames >= self.YAWN_FRAME_LIMIT:
            alarms.append("YAWNING")
        if self.distracted_frames >= self.DISTRACTED_FRAME_LIMIT:
            alarms.append("DISTRACTED")

        self.active_alarms = tuple(alarms)
        self._draw_warnings(frame, self.active_alarms)
        return self._draw_status(frame)

    def close(self) -> None:
        """Release MediaPipe resources; repeated calls are safe."""
        if not self._closed:
            self._face_landmarker.close()
            self._closed = True
            self._reset_counters()


__all__ = ["SafetyDaemon", "SafetyDaemonError"]
