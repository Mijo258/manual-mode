"""Silent automated safety verification; --live enables camera, GUI and alarms."""

import argparse
from contextlib import ExitStack
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

import contracts
from safety_engine import AlarmManager, SafetyEngine, compute_ear, compute_mar, compute_gaze


def make_landmarks(ear: float = 0.30, mar: float = 0.10,
                   left_gaze: float = 0.5, right_gaze: float = 0.5) -> list:
    """Known VGA pixel geometry normalized by width/height, including z."""
    points = np.tile((320.0, 240.0, 0.0), (478, 1))
    points[:, 2] = np.linspace(-0.02, 0.02, 478)
    half_opening = ear * 40.0 / 2.0
    left = ((200, 180), (210, 180 - half_opening), (230, 180 - half_opening),
            (240, 180), (230, 180 + half_opening), (210, 180 + half_opening))
    # Reversed x ordering exercises sorting in the proven second-eye gaze pair.
    right = ((400, 180), (390, 180 - half_opening), (370, 180 - half_opening),
             (360, 180), (370, 180 + half_opening), (390, 180 + half_opening))
    for index, point in zip((33, 160, 158, 133, 153, 144), left):
        points[index, :2] = point
    for index, point in zip((362, 385, 387, 263, 373, 380), right):
        points[index, :2] = point
    points[13, :2], points[14, :2] = (320, 320 - mar * 40), (320, 320 + mar * 40)
    points[78, :2], points[308, :2] = (280, 320), (360, 320)
    points[468, :2], points[473, :2] = (200 + left_gaze * 40, 180), (360 + right_gaze * 40, 180)
    points[:, 0] /= 640
    points[:, 1] /= 480
    return [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in points]


class MockLandmarker:
    def __init__(self, landmarks=None):
        self.landmarks = landmarks
        self.timestamps = []
        self.close_count = 0
        self.first_rgb_pixel = None

    def detect_for_video(self, image, timestamp_ms):
        assert not self.timestamps or timestamp_ms > self.timestamps[-1]
        self.timestamps.append(timestamp_ms)
        self.first_rgb_pixel = tuple(image.numpy_view()[0, 0])
        return SimpleNamespace(face_landmarks=[] if self.landmarks is None else [self.landmarks])

    def close(self):
        self.close_count += 1


def _blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_real_model() -> None:
    engine = SafetyEngine()
    try:
        assert engine.model_path.name == "face_landmarker.task" and engine.model_path.is_file()
        for timestamp_ms, color in ((100, 0), (133, 32), (166, 64)):
            frame = np.full((480, 640, 3), color, dtype=np.uint8)
            result = engine.process_frame(frame, timestamp_ms, draw=False)
            assert result.status.face_detected is False
            assert result.status.active_alarms == []
            assert result.landmarks_norm is None and result.landmarks_px is None
            assert result.annotated_frame is frame and np.all(frame == color)
    finally:
        engine.close()
        engine.close()
    print("Test 1 PASSED: real MediaPipe VIDEO model, three frames and clean close.")


def test_sleep_timer() -> None:
    mock = MockLandmarker(make_landmarks(ear=0.08))
    frame = _blank_frame()
    with SafetyEngine(landmarker=mock) as engine:
        for timestamp_ms in (0, 1000, 1499):
            status = engine.process_frame(frame, timestamp_ms, draw=False).status
            assert status.ear < engine.ear_threshold and np.isclose(status.ear, 0.08)
            assert status.is_sleepy is False and "SLEEP" not in status.active_alarms
        for timestamp_ms in (1500, 1600):
            status = engine.process_frame(frame, timestamp_ms, draw=False).status
            assert status.is_sleepy is True and status.active_alarms == ["SLEEP"]
        mock.landmarks = make_landmarks(ear=0.30)
        status = engine.process_frame(frame, 1700, draw=False).status
        assert status.is_sleepy is False and status.active_alarms == []
        mock.landmarks = make_landmarks(ear=0.08)
        assert engine.process_frame(frame, 1800, draw=False).status.is_sleepy is False
    assert mock.close_count == 1
    print("Test 2 PASSED: EAR geometry, elapsed sleep onset and immediate reset.")


def test_yawn_timer() -> None:
    mock = MockLandmarker(make_landmarks(mar=0.80))
    with SafetyEngine(landmarker=mock) as engine:
        frame = _blank_frame()
        for timestamp_ms in (0, 1000, 1499):
            status = engine.process_frame(frame, timestamp_ms, draw=False).status
            assert status.mar > engine.mar_threshold and np.isclose(status.mar, 0.80)
            assert status.is_yawning is False and "YAWN" not in status.active_alarms
        status = engine.process_frame(frame, 1600, draw=False).status
        assert status.is_yawning is True and status.active_alarms == ["YAWN"]
        mock.landmarks = make_landmarks(mar=0.10)
        status = engine.process_frame(frame, 1700, draw=False).status
        assert status.is_yawning is False and status.active_alarms == []
    print("Test 3 PASSED: MAR geometry, elapsed yawn onset and immediate reset.")


def test_gaze_timer() -> None:
    frame = _blank_frame()
    for ratio in (0.10, 0.90):
        mock = MockLandmarker(make_landmarks(left_gaze=ratio, right_gaze=ratio))
        with SafetyEngine(landmarker=mock) as engine:
            for timestamp_ms in (0, 1000, 1999):
                status = engine.process_frame(frame, timestamp_ms, draw=False).status
                assert 0.0 <= status.gaze_ratio <= 1.0 and np.isclose(status.gaze_ratio, ratio)
                assert status.is_distracted is False and "DISTRACTION" not in status.active_alarms
            status = engine.process_frame(frame, 2100, draw=False).status
            assert status.is_distracted is True and status.active_alarms == ["DISTRACTION"]
            mock.landmarks = make_landmarks()
            status = engine.process_frame(frame, 2200, draw=False).status
            assert np.isclose(status.gaze_ratio, 0.5)
            assert status.is_distracted is False and status.active_alarms == []
    # A single off-center eye, a misleading mean, and opposite gaze directions
    # must all remain inactive even after a full duration has elapsed.
    for left_ratio, right_ratio in ((0.1, 0.5), (0.0, 0.4), (0.1, 0.9)):
        mock = MockLandmarker(make_landmarks(left_gaze=left_ratio, right_gaze=right_ratio))
        with SafetyEngine(landmarker=mock) as engine:
            for timestamp_ms in (0, 2500):
                status = engine.process_frame(frame, timestamp_ms, draw=False).status
                assert status.is_distracted is False and status.active_alarms == []
    outside = make_landmarks(left_gaze=-2.0, right_gaze=3.0)
    pixels = np.array([(p.x * 640, p.y * 480) for p in outside], dtype=np.float32)
    assert compute_gaze(pixels) == (0.0, 1.0, 0.5)
    print("Test 4 PASSED: left/right gaze timers, clamped scalar ratio and two-eye agreement.")


def _assert_no_face(result) -> None:
    status = result.status
    assert status.face_detected is False
    assert (status.ear, status.mar, status.gaze_ratio) == (0.0, 0.0, 0.5)
    assert not status.is_sleepy and not status.is_yawning and not status.is_distracted
    assert status.active_alarms == [] and result.landmarks_norm is None and result.landmarks_px is None


def test_passthrough_and_immutability() -> None:
    landmarks = make_landmarks()
    mock = MockLandmarker(landmarks + [landmarks[0]])  # Exactly 478 are exposed, even with extra points.
    frame = np.full((480, 640, 3), (17, 41, 83), dtype=np.uint8)
    original_bytes = frame.tobytes()
    engine = SafetyEngine(landmarker=mock, model_path="unused-injected-model.task")
    try:
        result = engine.process_frame(frame, 100, draw=True)
        assert len(result.landmarks_norm) == 478
        assert result.landmarks_norm == [(p.x, p.y, p.z) for p in landmarks]
        assert result.landmarks_px.shape == (478, 2) and result.landmarks_px.dtype == np.float32
        expected = np.array([(p.x * 640, p.y * 480) for p in landmarks], dtype=np.float32)
        np.testing.assert_array_equal(result.landmarks_px, expected)
        assert frame.tobytes() == original_bytes and result.annotated_frame is not frame
        assert not np.shares_memory(result.annotated_frame, frame)
        assert result.annotated_frame.tobytes() != original_bytes
        assert mock.first_rgb_pixel == (83, 41, 17)
        assert np.isclose(compute_ear(result.landmarks_px), 0.30)
        assert np.isclose(compute_mar(result.landmarks_px), 0.10)
        result = engine.process_frame(frame, 100, draw=False)
        assert result.annotated_frame is frame and frame.tobytes() == original_bytes
        engine.process_frame(frame, 50, draw=False)
        assert mock.timestamps == [100, 101, 102]
    finally:
        engine.close()
        engine.close()
    assert mock.close_count == 1
    try:
        engine.process_frame(frame, 200)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Processing after close must fail.")

    mock = MockLandmarker(make_landmarks(ear=0.08))
    with SafetyEngine(landmarker=mock) as engine:
        engine.process_frame(frame, 0, draw=False)
        mock.landmarks = None
        _assert_no_face(engine.process_frame(frame, 1000, draw=False))
        mock.landmarks = make_landmarks(ear=0.08)
        assert not engine.process_frame(frame, 1600, draw=False).status.is_sleepy
        mock.landmarks = mock.landmarks[:468]
        _assert_no_face(engine.process_frame(frame, 2000, draw=False))
        mock.landmarks = make_landmarks(ear=0.08)
        assert not engine.process_frame(frame, 2300, draw=False).status.is_sleepy
        assert engine.process_frame(frame, 3800, draw=False).status.is_sleepy
        collapsed = make_landmarks()
        collapsed[133] = collapsed[33]
        mock.landmarks = collapsed
        _assert_no_face(engine.process_frame(frame, 3900, draw=False))
        mock.landmarks = make_landmarks(ear=0.08)
        assert not engine.process_frame(frame, 4000, draw=False).status.is_sleepy
    # All pure helpers reject zero-width geometry rather than inventing alarms.
    for helper in (compute_ear, compute_mar, compute_gaze):
        try:
            helper(np.zeros((478, 2), dtype=np.float32))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{helper.__name__} did not guard against division by zero.")
    no_close = SimpleNamespace(detect_for_video=lambda image, ts: SimpleNamespace(face_landmarks=[]))
    with SafetyEngine(landmarker=no_close) as optional_close:
        _assert_no_face(optional_close.process_frame(frame, 0, draw=False))
    print("Test 5 PASSED: landmark reuse, immutable input, timestamp guard and detection-gap reset.")


def test_alarm_cooldown() -> None:
    manager = AlarmManager(muted=True)
    try:
        with patch("safety_engine.threading.Thread", side_effect=AssertionError("Muted alarm spawned audio.")):
            assert manager.trigger([], now_sec=9.0) is False
            for now_sec, expected in ((10.0, True), (11.0, False), (12.5, True)):
                started = time.perf_counter()
                triggered = manager.trigger(["YAWN", "DISTRACTION", "SLEEP"], now_sec=now_sec)
                elapsed = time.perf_counter() - started
                assert triggered is expected
                assert elapsed < 0.05, f"Muted alarm trigger blocked for {elapsed:.3f}s."
            assert manager._thread is None
    finally:
        manager.stop()
        manager.stop()
    assert manager.trigger(["SLEEP"], now_sec=20.0) is False
    print("Test 6 PASSED: muted alarm cooldown; every trigger completed within 50 ms.")


def run_live() -> None:
    from camera_hub import CameraHub

    manager = AlarmManager(muted=False)
    window_opened = False
    try:
        with SafetyEngine() as engine, CameraHub() as camera:
            last_id = -1
            while camera.is_running():
                packet = camera.get_latest(last_id, timeout=0.1)
                if packet is not None:
                    last_id = packet.frame_id
                    result = engine.process_frame(packet.frame, packet.timestamp_ms)
                    manager.trigger(result.status.active_alarms, now_sec=result.status.timestamp)
                    cv2.putText(result.annotated_frame, f"FPS: {camera.actual_fps:.1f}",
                                (10, result.annotated_frame.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 100), 1, cv2.LINE_AA)
                    cv2.imshow("Standalone DMS - q/Esc to quit", result.annotated_frame)
                    window_opened = True
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        manager.stop()
        if window_opened:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Enable live webcam display and audible alarms.")
    args = parser.parse_args()
    if args.live:
        run_live()
        return
    # Guard the automated mode against accidental camera, GUI or audio calls.
    forbidden_audio = SimpleNamespace(Beep=Mock(side_effect=AssertionError("Automated audio playback.")),
                                      PlaySound=Mock(side_effect=AssertionError("Automated audio playback.")))
    with ExitStack() as guards:
        guards.enter_context(patch("cv2.imshow", side_effect=AssertionError("Automated GUI window.")))
        guards.enter_context(patch("cv2.VideoCapture", side_effect=AssertionError("Automated camera opening.")))
        guards.enter_context(patch.dict("sys.modules", {"winsound": forbidden_audio}))
        test_real_model()
        test_sleep_timer()
        test_yawn_timer()
        test_gaze_timer()
        test_passthrough_and_immutability()
        test_alarm_cooldown()
    print("PHASE 2 VERIFIED OK")


if __name__ == "__main__":
    main()
