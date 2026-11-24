"""Offline regression tests for driver monitoring; no camera or sound required."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import master_controller
from safety_daemon import Point2D, SafetyDaemon, SafetyDaemonError


def face_landmarks(
    ear: float = 0.30,
    mar: float = 0.10,
    gaze: tuple[float, float] = (0.50, 0.50),
) -> list[Point2D]:
    """Build normalized points from hand-computable 640x480 pixel geometry.

    Each eye is 100 pixels wide, with opening EAR * 100. The mouth is
    100 pixels wide with opening MAR * 100. Iris positions are fractions
    of the horizontal eye width.
    """
    points = [Point2D(320, 240) for _ in range(478)]
    for indices, offset in (
        ((33, 160, 158, 133, 153, 144), 100),
        ((362, 385, 387, 263, 373, 380), 300),
    ):
        half_opening = ear * 50
        eye = (
            (offset, 100), (offset + 30, 100 - half_opening),
            (offset + 70, 100 - half_opening), (offset + 100, 100),
            (offset + 70, 100 + half_opening), (offset + 30, 100 + half_opening),
        )
        for index, (x, y) in zip(indices, eye):
            points[index] = Point2D(x, y)
    points[13] = Point2D(300, 300 - mar * 50)
    points[14] = Point2D(300, 300 + mar * 50)
    points[78] = Point2D(250, 300)
    points[308] = Point2D(350, 300)
    points[468] = Point2D(100 + gaze[0] * 100, 100)
    points[473] = Point2D(300 + gaze[1] * 100, 100)
    return [Point2D(p.x / 640, p.y / 480) for p in points]


class FakeLandmarker:
    def __init__(self) -> None:
        self.faces = [face_landmarks()]
        self.timestamps: list[int] = []
        self.images: list[np.ndarray] = []
        self.close_count = 0

    def detect_for_video(self, image: np.ndarray, timestamp_ms: int) -> SimpleNamespace:
        self.timestamps.append(timestamp_ms)
        self.images.append(image.copy())
        # Tasks results contain a list of lists, without the old .landmark wrapper.
        return SimpleNamespace(face_landmarks=self.faces)

    def close(self) -> None:
        self.close_count += 1


class SafetyDaemonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = FakeLandmarker()
        fake_mediapipe = SimpleNamespace(
            Image=lambda *, image_format, data: data,
            ImageFormat=SimpleNamespace(SRGB="SRGB"),
        )
        with patch("safety_daemon.import_module", return_value=fake_mediapipe), patch.object(
            SafetyDaemon, "_create_face_landmarker", return_value=self.backend,
        ):
            self.daemon = SafetyDaemon()
        self.addCleanup(self.daemon.close)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def process(self, count: int = 1) -> None:
        for _ in range(count):
            self.daemon.process_frame(self.frame.copy())

    def test_tasks_result_and_pixel_geometry(self) -> None:
        self.process()
        self.assertEqual(self.daemon.landmark_count, 478)
        self.assertAlmostEqual(self.daemon.ear, 0.30)
        self.assertAlmostEqual(self.daemon.mar, 0.10)
        self.assertFalse(self.daemon.gaze_off_center)
        self.assertEqual(self.daemon.active_alarms, ())

    def test_sleep_alarm_after_twenty_frames_and_reset_when_awake(self) -> None:
        # Measuring normalized x/y directly would turn .18 into .24 and miss this.
        self.backend.faces = [face_landmarks(ear=0.18)]
        self.process(19)
        self.assertEqual(self.daemon.active_alarms, ())
        self.process()
        self.assertEqual(self.daemon.active_alarms, ("SLEEPING",))
        self.backend.faces = [face_landmarks()]
        self.process()
        self.assertEqual(self.daemon.sleepy_frames, 0)
        self.assertEqual(self.daemon.active_alarms, ())

    def test_short_blinks_do_not_accumulate(self) -> None:
        for _ in range(3):
            self.backend.faces = [face_landmarks(ear=0.10)]
            self.process(10)
            self.assertEqual(self.daemon.active_alarms, ())
            self.backend.faces = [face_landmarks()]
            self.process()
        self.assertEqual(self.daemon.sleepy_frames, 0)

    def test_yawn_alarm_after_fifteen_frames(self) -> None:
        self.backend.faces = [face_landmarks(mar=0.60)]
        self.process(14)
        self.assertEqual(self.daemon.active_alarms, ())
        self.process()
        self.assertEqual(self.daemon.active_alarms, ("YAWNING",))

    def test_sideways_gaze_requires_both_eyes_and_thirty_frames(self) -> None:
        for gaze, expected in (
            ((0.10, 0.10), ("DISTRACTED",)),
            ((0.90, 0.90), ("DISTRACTED",)),
            ((0.10, 0.90), ()),
            ((0.10, 0.50), ()),
        ):
            with self.subTest(gaze=gaze):
                self.backend.faces = [face_landmarks()]
                self.process()
                self.backend.faces = [face_landmarks(gaze=gaze)]
                self.process(29)
                self.assertEqual(self.daemon.active_alarms, ())
                self.process()
                self.assertEqual(self.daemon.active_alarms, expected)

    def test_simultaneous_alarms(self) -> None:
        self.backend.faces = [face_landmarks(ear=0.10, mar=0.60, gaze=(0.10, 0.10))]
        self.process(30)
        self.assertEqual(self.daemon.active_alarms, ("SLEEPING", "YAWNING", "DISTRACTED"))

    def test_missing_face_clears_alarms_and_does_not_join_detections(self) -> None:
        self.backend.faces = [face_landmarks(ear=0.10)]
        self.process(20)
        self.backend.faces = []
        self.process()
        self.assertEqual(self.daemon.status, "NO FACE")
        self.assertEqual(self.daemon.active_alarms, ())
        self.assertEqual(self.daemon.landmark_count, 0)
        self.backend.faces = [face_landmarks(ear=0.10)]
        self.process(19)
        self.assertEqual(self.daemon.active_alarms, ())

    def test_incomplete_or_invalid_geometry_resets_counters(self) -> None:
        collapsed = face_landmarks()
        collapsed[33] = collapsed[133]
        non_finite = face_landmarks()
        non_finite[33] = Point2D(float("nan"), 0.5)
        for invalid in (face_landmarks()[:468], collapsed, non_finite):
            with self.subTest(kind=len(invalid), point=invalid[33]):
                self.backend.faces = [face_landmarks(ear=0.10)]
                self.process(20)
                self.backend.faces = [invalid]
                with self.assertLogs("safety_daemon", level="WARNING"):
                    self.process()
                self.assertEqual(self.daemon.status, "INVALID LANDMARKS")
                self.assertEqual(self.daemon.sleepy_frames, 0)
                self.assertEqual(self.daemon.active_alarms, ())

    def test_video_timestamps_strictly_increase(self) -> None:
        with patch("safety_daemon.time.monotonic", side_effect=[1.0, 1.0, 0.99]):
            self.process(3)
        self.assertEqual(self.backend.timestamps, [1000, 1001, 1002])

    def test_bgr_is_converted_to_rgb_before_inference(self) -> None:
        self.frame[:] = (10, 20, 30)
        self.process()
        np.testing.assert_array_equal(self.backend.images[0][0, 0], [30, 20, 10])

    def test_inference_failure_is_reported_and_clears_alarms(self) -> None:
        self.backend.faces = [face_landmarks(ear=0.10)]
        self.process(20)
        with patch.object(self.backend, "detect_for_video", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(SafetyDaemonError, "model failed"):
                self.process()
        self.assertEqual(self.daemon.active_alarms, ())
        self.assertEqual(self.daemon.sleepy_frames, 0)

    def test_shutdown_is_idempotent_and_prevents_inference(self) -> None:
        self.daemon.close()
        self.daemon.close()
        self.assertEqual(self.backend.close_count, 1)
        with self.assertRaisesRegex(SafetyDaemonError, "shutdown"):
            self.process()

    def test_invalid_frame_is_rejected(self) -> None:
        for frame in (None, np.zeros((0, 0, 3), dtype=np.uint8), self.frame[:, :, 0], self.frame.astype(float)):
            with self.subTest(shape=getattr(frame, "shape", None)):
                with self.assertRaises(ValueError):
                    self.daemon.process_frame(frame)

    def test_missing_model_has_download_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "face_landmarker.task"
            with self.assertRaisesRegex(SafetyDaemonError, "curl.exe -fL"):
                SafetyDaemon._create_face_landmarker(missing, 0.5, 0.5)


class ControllerTests(unittest.TestCase):
    def test_each_captured_frame_is_consumed_once_and_copies_are_independent(self) -> None:
        initial = np.zeros((8, 8, 3), dtype=np.uint8)
        capture = Mock()
        capture.read.side_effect = [(True, initial), (True, initial + 1), (False, None)]
        with patch("master_controller.cv2.VideoCapture", return_value=capture):
            streamer = master_controller.CameraStreamer()
        frame_id, frame = streamer.read_next(-1)
        frame[:] = 255
        self.assertFalse(streamer.read().any())
        self.assertIsNone(streamer.read_next(frame_id))
        streamer.update()
        next_id, next_frame = streamer.read_next(frame_id)
        self.assertGreater(next_id, frame_id)
        np.testing.assert_array_equal(next_frame, initial + 1)
        self.assertIsNone(streamer.read_next(next_id))
        streamer.stop()

    def test_startup_check_avoids_camera_and_gui(self) -> None:
        with patch("master_controller.SafetyDaemon") as daemon, patch(
            "master_controller.CameraStreamer",
        ) as camera, patch("master_controller.cv2.imshow") as display:
            self.assertEqual(master_controller.main(["--check"]), 0)
            daemon.return_value.process_frame.assert_called_once()
            daemon.return_value.close.assert_called_once()
            camera.assert_not_called()
            display.assert_not_called()

    def test_startup_failure_returns_nonzero(self) -> None:
        with patch("master_controller.SafetyDaemon", side_effect=SafetyDaemonError("bad model")):
            with self.assertLogs("master_controller", level="ERROR"):
                self.assertEqual(master_controller.main(["--check"]), 1)

    def test_controller_routes_alarms_and_cleans_up(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("master_controller.SafetyDaemon") as daemon_factory, patch(
            "master_controller.CameraStreamer",
        ) as camera_factory, patch("master_controller.AlarmNotifier") as notifier, patch(
            "master_controller.cv2.imshow",
        ), patch("master_controller.cv2.waitKey", return_value=ord("q")), patch(
            "master_controller.cv2.destroyAllWindows",
        ) as cleanup:
            daemon = daemon_factory.return_value
            daemon.process_frame.return_value = frame
            daemon.active_alarms = ("SLEEPING",)
            camera = camera_factory.return_value.start.return_value
            camera.last_error = None
            camera.read_next.return_value = (0, frame)
            self.assertEqual(master_controller.main([]), 0)
            notifier.return_value.notify.assert_called_once_with(("SLEEPING",))
            camera.stop.assert_called_once()
            daemon.close.assert_called_once()
            cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
