"""Verify decoupled controllers without hardware/GUI; --live --mode enables preview."""

import argparse
import ast
from contextlib import ExitStack
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import dlib
import numpy as np

import contracts
from controllers import BaseController, ControllerOutput, FaceController, HandController, EyeController, create_controller


def _assert_output(result: ControllerOutput, frame: np.ndarray, original: bytes, mode: str) -> None:
    assert isinstance(result, ControllerOutput) and result.mode == mode
    assert result.command == contracts.DirectionCommand.STOP.value
    assert frame.tobytes() == original, "Controller changed the caller's input frame."
    assert 0.0 <= result.confidence <= 1.0


def _face_context() -> dict:
    pixels = np.tile((320.0, 240.0), (478, 1)).astype(np.float32)
    for index, point in {234: (160, 240), 454: (480, 240), 10: (320, 120), 152: (320, 360),
                         1: (352, 252), 33: (220, 180), 263: (420, 200)}.items():
        pixels[index] = point
    normalized = [(float(x / 640), float(y / 480), 0.0) for x, y in pixels]
    return {"face_landmarks_norm": normalized, "face_landmarks_px": pixels}


def test_face_controller() -> None:
    context = _face_context()
    frame = np.full((480, 640, 3), (17, 41, 83), dtype=np.uint8)
    original = frame.tobytes()
    pixel_bytes = context["face_landmarks_px"].tobytes()
    with FaceController() as controller:
        result = controller.process_frame(frame, 100, shared_context=context)
        _assert_output(result, frame, original, "FACE")
        assert result.detected is True and result.landmarks_px.shape == (478, 2)
        assert result.landmarks_px.dtype == np.float32
        assert result.landmarks_px is context["face_landmarks_px"]
        assert result.features["landmarks_norm"] is context["face_landmarks_norm"]
        assert np.isclose(result.features["yaw_offset"], 0.1)
        assert np.isclose(result.features["pitch_offset"], 0.05)
        assert np.isclose(result.features["roll_deg"], math.degrees(math.atan2(20, 200)))
        assert np.allclose(result.features["face_center_norm"], (0.5, 0.5))
        assert context["face_landmarks_px"].tobytes() == pixel_bytes
        assert not np.shares_memory(result.annotated_frame, frame)
        raw = controller.process_frame(frame, 133, draw=False, shared_context=context)
        assert raw.annotated_frame is frame
        for missing in (None, {"face_landmarks_norm": None},
                        {"face_landmarks_norm": context["face_landmarks_norm"][:477],
                         "face_landmarks_px": context["face_landmarks_px"]},
                        {"face_landmarks_norm": context["face_landmarks_norm"],
                         "face_landmarks_px": context["face_landmarks_px"][:477]}):
            result = controller.process_frame(frame, 166, shared_context=missing)
            _assert_output(result, frame, original, "FACE")
            assert result.detected is False and result.landmarks_px is None
            assert result.features == {} and result.confidence == 0.0
        controller.reset()
        controller.close()
        controller.close()
    print("Test 1 PASSED: shared face landmarks, pose features, immutable input and no-face handling.")


def _hand_points() -> np.ndarray:
    return np.array([(320, 360), (305, 350), (285, 330), (275, 315), (290, 330),
                     (310, 280), (310, 235), (310, 220), (310, 190),
                     (330, 280), (335, 235), (330, 255), (330, 295),
                     (355, 290), (370, 250), (360, 265), (348, 310),
                     (375, 315), (405, 290), (390, 305), (370, 330)], dtype=np.float64)


class MockHandLandmarker:
    def __init__(self, pixels):
        self.set_points(pixels)
        self.timestamps = []
        self.close_count = 0
        self.rgb_pixel = None

    def set_points(self, pixels):
        self.landmarks = [SimpleNamespace(x=float(x / 640), y=float(y / 480), z=0.0) for x, y in pixels]

    def detect_for_video(self, image, timestamp_ms):
        assert not self.timestamps or timestamp_ms > self.timestamps[-1]
        self.timestamps.append(timestamp_ms)
        self.rgb_pixel = tuple(image.numpy_view()[0, 0])
        return SimpleNamespace(hand_landmarks=[self.landmarks],
                               handedness=[[SimpleNamespace(category_name="Right", score=0.96)]])

    def close(self):
        self.close_count += 1


def test_hand_controller() -> None:
    controller = HandController()
    try:
        assert controller.model_path.name == "hand_landmarker.task" and controller.model_path.is_file()
        for timestamp_ms, color in ((100, 0), (133, 32), (166, 64)):
            frame = np.full((480, 640, 3), color, dtype=np.uint8)
            original = frame.tobytes()
            result = controller.process_frame(frame, timestamp_ms)
            _assert_output(result, frame, original, "HAND")
            assert result.detected is False and result.landmarks_px is None and result.features == {}
    finally:
        controller.close()
        controller.close()

    pixels = _hand_points()
    mock = MockHandLandmarker(pixels)
    frame = np.full((480, 640, 3), (17, 41, 83), dtype=np.uint8)
    original = frame.tobytes()
    with HandController(landmarker=mock) as controller:
        for iteration, (angle, scale, mirror) in enumerate(((0.0, 1.0, False),
                                                          (math.pi / 2, 0.7, False),
                                                          (0.7, 0.8, True))):
            relative = pixels - pixels[0]
            if mirror:
                relative = relative * (-1, 1)
            rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle), math.cos(angle)]])
            transformed = relative @ rotation.T * scale + (320, 300)
            mock.set_points(transformed)
            result = controller.process_frame(frame, (100, 100, 50)[iteration])
            _assert_output(result, frame, original, "HAND")
            assert result.detected is True and result.landmarks_px.shape == (21, 2)
            assert result.landmarks_px.dtype == np.float32
            assert result.features["finger_states"] == {"thumb": False, "index": True,
                                                        "middle": False, "ring": False, "pinky": False}
            assert result.features["finger_states"]["index"] is True and result.features["extended_count"] == 1
            vector = result.features["pointing_vector"]
            assert len(vector) == 2 and np.isfinite(vector).all() and np.isclose(np.linalg.norm(vector), 1.0)
            assert result.features["handedness"] == "Right" and np.isclose(result.confidence, 0.96)
            assert len(result.features["landmarks_norm"]) == 21
            assert mock.rgb_pixel == (83, 41, 17)
        assert mock.timestamps == [100, 101, 102]
        raw = controller.process_frame(frame, 133, draw=False)
        assert raw.annotated_frame is frame
        mock.landmarks = mock.landmarks[:20]
        assert controller.process_frame(frame, 166, draw=False).detected is False
        controller.reset()
    controller.close()
    assert mock.close_count == 1
    print("Test 2 PASSED: real hand model, one-finger geometry, rotation/mirroring and timestamp guard.")


class MockDetector:
    def __init__(self):
        self.image_shape = None
        self.close_count = 0

    def __call__(self, image, upsample):
        self.image_shape = image.shape
        assert upsample == 0
        return [dlib.rectangle(10, 10, 20, 20), dlib.rectangle(100, 75, 249, 174)]

    def close(self):
        self.close_count += 1


class MockPredictor:
    def __init__(self):
        self.gray_shape = None
        self.rect = None
        self.close_count = 0
        self.closed_eyes = False

    def __call__(self, gray, rect):
        self.gray_shape, self.rect = gray.shape, rect
        points = np.tile((320, 240), (68, 1))
        points[36:42] = ((200, 200), (210, 192), (230, 192), (240, 200), (230, 208), (210, 208))
        points[42:48] = ((360, 200), (370, 192), (390, 192), (400, 200), (390, 208), (370, 208))
        if self.closed_eyes:
            points[36:48, 1] = 200
        return SimpleNamespace(num_parts=68,
                               part=lambda i: SimpleNamespace(x=int(points[i, 0]), y=int(points[i, 1])))

    def close(self):
        self.close_count += 1


def test_eye_controller() -> None:
    controller = EyeController()
    try:
        assert controller.predictor_path.name == "shape_predictor_68_face_landmarks.dat"
        for timestamp_ms, color in ((100, 0), (133, 32), (166, 64)):
            frame = np.full((480, 640, 3), color, dtype=np.uint8)
            original = frame.tobytes()
            result = controller.process_frame(frame, timestamp_ms)
            _assert_output(result, frame, original, "EYE")
            assert result.detected is False and result.landmarks_px is None and result.features == {}
    finally:
        controller.close()
        controller.close()

    detector, predictor = MockDetector(), MockPredictor()
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    cv2.circle(frame, (212, 200), 3, (20, 20, 20), -1)
    cv2.circle(frame, (388, 200), 3, (20, 20, 20), -1)
    original = frame.tobytes()
    with EyeController(detector=detector, predictor=predictor) as controller:
        result = controller.process_frame(frame, 100)
        _assert_output(result, frame, original, "EYE")
        assert result.detected is True and result.landmarks_px.shape == (12, 2)
        assert result.landmarks_px.dtype == np.float32
        assert detector.image_shape == (240, 320) and predictor.gray_shape == (480, 640)
        rect = predictor.rect
        assert (rect.left(), rect.top(), rect.right(), rect.bottom()) == (200, 150, 499, 349)
        for key in ("left_ear", "right_ear", "avg_ear"):
            assert np.isclose(result.features[key], 0.4)
        for key in ("left_pupil_ratio", "right_pupil_ratio", "avg_pupil_ratio"):
            assert all(0.0 <= value <= 1.0 for value in result.features[key])
        left_ratio, right_ratio = result.features["left_pupil_ratio"], result.features["right_pupil_ratio"]
        assert np.allclose(left_ratio, (0.3, 0.5), atol=0.06)
        assert np.allclose(right_ratio, (0.7, 0.5), atol=0.06)
        assert result.features["left_eye_px"].shape == result.features["right_eye_px"].shape == (6, 2)
        assert "iris_gaze_ratio" not in result.features
        iris = np.zeros((478, 2), dtype=np.float32)
        for index, point in {33: (200, 180), 133: (240, 180), 468: (232, 180),
                             362: (400, 180), 263: (360, 180), 473: (392, 180)}.items():
            iris[index] = point
        shared = controller.process_frame(frame, 133, draw=False, shared_context={"face_landmarks_px": iris})
        _assert_output(shared, frame, original, "EYE")
        assert shared.annotated_frame is frame and np.isclose(shared.features["iris_gaze_ratio"], 0.8)
        assert shared.features["left_pupil_ratio"] == left_ratio
        assert shared.features["right_pupil_ratio"] == right_ratio
        predictor.closed_eyes = True
        closed = controller.process_frame(frame, 166)
        _assert_output(closed, frame, original, "EYE")
        assert closed.features["left_pupil_ratio"] == closed.features["right_pupil_ratio"] == (0.5, 0.5)
        controller.reset()
    controller.close()
    assert detector.close_count == predictor.close_count == 1
    print("Test 3 PASSED: real dlib model, full-resolution eyelids, standalone pupils and optional iris reuse.")


def test_factory_and_lifecycle() -> None:
    for mode, expected in (("FACE", FaceController), ("hand", HandController), ("EyE", EyeController)):
        with create_controller(mode) as controller:
            assert isinstance(controller, expected) and isinstance(controller, BaseController)
            controller.reset()
        controller.close()
        controller.close()
    with create_controller(contracts.ControlMode.FACE) as controller:
        assert isinstance(controller, FaceController)
    for invalid in ("NONE", contracts.ControlMode.NONE, "invalid"):
        try:
            create_controller(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Factory accepted unsupported mode: {invalid!r}")
    # Enforce the architectural boundary for every controller source file.
    for path in (Path(__file__).resolve().parent / "controllers").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            assert not (isinstance(node, ast.While) and isinstance(node.test, ast.Constant)
                        and node.test.value is True), f"Unbounded GUI loop in {path.name}"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ("VideoCapture", "imshow"), f"Controller owns I/O in {path.name}"
    print("Test 4 PASSED: enum/string factory, unsupported modes, lifecycle and controller I/O boundary.")


def run_live(mode: str) -> None:
    from camera_hub import CameraHub

    window_opened = False
    try:
        with ExitStack() as resources:
            controller = resources.enter_context(create_controller(mode))
            safety = None
            if controller.mode == contracts.ControlMode.FACE:
                from safety_engine import SafetyEngine

                safety = resources.enter_context(SafetyEngine())
            camera = resources.enter_context(CameraHub())
            last_id = -1
            while camera.is_running():
                packet = camera.get_latest(last_id, timeout=0.1)
                if packet is not None:
                    last_id = packet.frame_id
                    context = None
                    if safety is not None:
                        face = safety.process_frame(packet.frame, packet.timestamp_ms, draw=False)
                        context = {"face_landmarks_norm": face.landmarks_norm, "face_landmarks_px": face.landmarks_px}
                    result = controller.process_frame(packet.frame, packet.timestamp_ms, shared_context=context)
                    cv2.putText(result.annotated_frame, f"FPS: {camera.actual_fps:.1f}  frame_id: {last_id}",
                                (10, result.annotated_frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (90, 220, 120), 1, cv2.LINE_AA)
                    cv2.imshow(f"Controller {controller.mode.value} - q/Esc to quit", result.annotated_frame)
                    window_opened = True
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        if window_opened:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Enable webcam and GUI preview.")
    parser.add_argument("--mode", choices=("face", "hand", "eye"), default="face")
    args = parser.parse_args()
    if args.live:
        run_live(args.mode)
        return
    with ExitStack() as guards:
        guards.enter_context(patch("cv2.VideoCapture", side_effect=AssertionError("Automated camera opening.")))
        guards.enter_context(patch("cv2.imshow", side_effect=AssertionError("Automated GUI window.")))
        guards.enter_context(patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options",
                                   side_effect=AssertionError("FaceController duplicated inference.")))
        test_face_controller()
        test_hand_controller()
        test_eye_controller()
        test_factory_and_lifecycle()
    print("PHASE 3 VERIFIED OK")


if __name__ == "__main__":
    main()
