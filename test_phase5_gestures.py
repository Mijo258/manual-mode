"""Headless gesture verification; --live enables the optional complete camera demo."""

import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import math
import subprocess
import sys
import time
from unittest.mock import patch

# Verification must not create or replace project/virtual-environment bytecode.
sys.dont_write_bytecode = True

import cv2
import numpy as np

import arrow_game
import contracts
from controllers import EyeController, HandController
from controllers.base import ControllerOutput
from controllers.face_controller import FaceController
from gesture_mapper import (CommandDebouncer, EyeGestureMapper, FaceGestureMapper,
                            GestureEngine, HandGestureMapper)
from ipc_bus import CommandPublisher, CommandSubscriber


Direction = contracts.DirectionCommand
FRAME = np.full((48, 64, 3), 100, dtype=np.uint8)


def _output(mode: str, features: dict, detected: bool = True) -> ControllerOutput:
    return ControllerOutput(mode, detected, FRAME, None, features, confidence=0.95)


def _face(yaw: float = 0.0, pitch: float = 0.0) -> ControllerOutput:
    return _output("FACE", {"yaw_offset": yaw, "pitch_offset": pitch})


def _hand(point: tuple[float, float] = (0.5, 0.5), *, count: int = 1,
          index: bool = True) -> ControllerOutput:
    states = dict.fromkeys(("thumb", "index", "middle", "ring", "pinky"), False)
    for name in ("index", "middle", "ring", "pinky")[:min(count, 4)]:
        states[name] = True
    if count == 5:
        states["thumb"] = True
    states["index"] = index and count > 0
    return _output("HAND", {"finger_states": states, "extended_count": count,
                            "index_tip_norm": point})


def _eye(left: float = 0.28, right: float = 0.28,
         pupil: tuple[float, float] = (0.5, 0.5)) -> ControllerOutput:
    return _output("EYE", {"left_ear": left, "right_ear": right,
                           "avg_ear": (left + right) / 2.0, "avg_pupil_ratio": pupil})


def test_debouncer() -> None:
    debouncer = CommandDebouncer()
    assert debouncer.debounce_frames == contracts.load_config()["control"]["debounce_frames"] == 3
    # Establish a moving command, then challenge it with twelve single-frame glitches.
    assert debouncer.update(Direction.UP, True) == Direction.STOP
    assert debouncer.update(Direction.UP, True) == Direction.STOP
    assert debouncer.update(Direction.UP, True) == Direction.UP
    for index in range(12):
        glitch = Direction.LEFT if index % 2 == 0 else Direction.RIGHT
        assert debouncer.update(glitch, True) == Direction.UP
    assert debouncer.update(Direction.RIGHT, True) == Direction.UP  # Second matching frame.
    assert debouncer.update(Direction.RIGHT, True) == Direction.RIGHT
    assert debouncer.update(Direction.LEFT, False) == Direction.RIGHT
    assert debouncer.update(Direction.LEFT, False) == Direction.RIGHT
    assert debouncer.update(Direction.LEFT, False) == Direction.STOP
    assert debouncer.update(Direction.UP, True) == Direction.STOP
    assert debouncer.update(Direction.UP, False) == Direction.STOP
    # Loss interrupted the candidate; two new detections are insufficient.
    assert debouncer.update(Direction.UP, True) == Direction.STOP
    assert debouncer.update(Direction.UP, True) == Direction.STOP
    assert debouncer.update(Direction.UP, True) == Direction.UP
    debouncer.reset()
    assert debouncer.stable_command == Direction.STOP
    assert CommandDebouncer(debounce_frames=0).update(Direction.LEFT, True) == Direction.LEFT
    print("Test 1 PASSED: glitches rejected, consecutive confirmation and tracking-loss STOP.")


def test_face_mapper() -> None:
    mapper = FaceGestureMapper()
    cases = ((_face(), Direction.STOP, "HEAD NEUTRAL"),
             (_face(-0.25), Direction.LEFT, "HEAD LEFT"),
             (_face(0.25), Direction.RIGHT, "HEAD RIGHT"),
             (_face(pitch=-0.20), Direction.UP, "HEAD UP"),
             (_face(pitch=0.20), Direction.DOWN, "HEAD DOWN"),
             (_face(-0.18, 0.30), Direction.DOWN, "HEAD DOWN"),
             (_face(-0.18, -0.14), Direction.LEFT, "HEAD LEFT"))
    for output, expected, expected_label in cases:
        mapper.reset()
        original = deepcopy(output.features)
        command, confidence, label = mapper.map(output)
        assert command == expected and label == expected_label and confidence == 0.95
        assert output.features == original

    # Use the actual FaceController to check its coordinate signs, not just invented metrics.
    landmarks = [(0.5, 0.5, 0.0)] * 478
    for index, point in {234: (0.3, 0.5, 0.0), 454: (0.7, 0.5, 0.0),
                         10: (0.5, 0.2, 0.0), 152: (0.5, 0.8, 0.0),
                         33: (0.4, 0.4, 0.0), 263: (0.6, 0.4, 0.0)}.items():
        landmarks[index] = point
    frame_before = FRAME.copy()
    with FaceController() as controller:
        for nose, expected in (((0.38, 0.5, 0.0), Direction.LEFT),
                               ((0.62, 0.5, 0.0), Direction.RIGHT),
                               ((0.5, 0.38, 0.0), Direction.UP),
                               ((0.5, 0.62, 0.0), Direction.DOWN)):
            landmarks[1] = nose
            pixels = (np.asarray(landmarks)[:, :2] * (64, 48)).astype(np.float32)
            output = controller.process_frame(FRAME, 0, draw=False,
                                              shared_context={"face_landmarks_norm": landmarks,
                                                              "face_landmarks_px": pixels})
            mapper.reset()
            assert mapper.map(output)[0] == expected
    assert np.array_equal(FRAME, frame_before)

    mapper.reset()
    assert mapper.map(_face())[0] == Direction.STOP
    assert mapper.map(_face(0.18))[0] == Direction.STOP  # First EMA is 0.081.
    assert mapper.map(_face(0.18))[0] == Direction.RIGHT
    assert mapper.calibrate({"yaw_offset": 0.19, "pitch_offset": -0.12})
    assert mapper.map(_face(0.19, -0.12))[0] == Direction.STOP
    mapper.reset()
    assert mapper.map(_face(-0.06, -0.12))[0] == Direction.LEFT
    previous = mapper.baseline_yaw, mapper.baseline_pitch
    assert not mapper.calibrate({})
    assert not mapper.calibrate({"yaw_offset": math.nan, "pitch_offset": 0.0})
    assert (mapper.baseline_yaw, mapper.baseline_pitch) == previous
    mapper.reset(reset_baseline=True)
    assert mapper.baseline_yaw == mapper.baseline_pitch == 0.0
    configured = FaceGestureMapper({"gestures": {"face": {"ema_alpha": 1.0, "yaw_deadzone": 0.20}}})
    assert configured.map(_face(0.15))[0] == Direction.STOP
    assert configured.map(_face(0.25))[0] == Direction.RIGHT
    print("Test 2 PASSED: actual face coordinate signs, EMA, dominant axis and calibration.")


def test_hand_mapper() -> None:
    mapper = HandGestureMapper()
    assert (mapper.ema_alpha, mapper.deadzone_x, mapper.deadzone_y) == (0.45, 0.12, 0.12)
    assert (mapper.baseline_x, mapper.baseline_y) == (0.5, 0.5)
    for point, command in (((0.5, 0.5), Direction.STOP), ((0.5, 0.25), Direction.UP),
                            ((0.5, 0.75), Direction.DOWN), ((0.25, 0.5), Direction.LEFT),
                            ((0.75, 0.5), Direction.RIGHT), ((0.25, 0.25), Direction.LEFT),
                            ((0.75, 0.75), Direction.RIGHT), ((0.7, 0.15), Direction.UP)):
        mapper.reset()
        output = _hand(point)
        original = deepcopy(output.features)
        result, confidence, label = mapper.map(output)
        assert result == command and confidence == (0.9 if command == Direction.STOP else 1.0)
        assert label == ("INDEX CENTER (STOP)" if command == Direction.STOP else "INDEX " + command.value)
        assert output.features == original

    # Exercise the actual feature producer: it preserves image x/y without another flip.
    for point, command in (((0.25, 0.5), Direction.LEFT), ((0.75, 0.5), Direction.RIGHT),
                            ((0.5, 0.25), Direction.UP), ((0.5, 0.75), Direction.DOWN)):
        coords = np.zeros((21, 3), dtype=np.float64)
        coords[8, :2] = point
        features = HandController._features(coords, (coords[:, :2] * (64, 48)).astype(np.float32))
        assert features["index_tip_norm"] == point and features["finger_states"]["index"] is True
        mapper.reset()
        assert mapper.map(_output("HAND", features))[0] == command
    for count in range(1, 6):
        mapper.reset()
        assert mapper.map(_hand((0.75, 0.5), count=count))[0] == Direction.RIGHT
    assert mapper.map(_hand((0.25, 0.5), count=0)) == (Direction.STOP, 0.9, "FIST (STOP)")
    assert mapper.map(_hand((0.25, 0.5), count=3, index=False)) == (Direction.STOP, 0.6, "INDEX CURLED (STOP)")
    for output in (_output("HAND", {}, detected=False), _output("HAND", {}),
                   _hand((math.nan, 0.5)), _hand((False, 0.5))):
        assert mapper.map(output) == (Direction.STOP, 0.0, "NO HAND")

    mapper.reset()
    assert mapper.map(_hand())[0] == Direction.STOP
    assert mapper.map(_hand((0.25, 0.5)))[0] == Direction.STOP  # EMA x = 0.3875.
    assert mapper.map(_hand((0.25, 0.5)))[0] == Direction.LEFT
    smoothed = mapper._smooth_x, mapper._smooth_y
    mapper.map(_hand((0.99, 0.01), count=0))
    mapper.map(_hand((0.99, 0.01), index=False))
    assert (mapper._smooth_x, mapper._smooth_y) == smoothed
    assert mapper.calibrate({"index_tip_norm": (0.7, 0.4)})
    assert mapper.map(_hand((0.7, 0.4)))[0] == Direction.STOP
    mapper.reset()
    assert (mapper.baseline_x, mapper.baseline_y) == (0.7, 0.4)
    assert mapper.map(_hand((0.45, 0.4)))[0] == Direction.LEFT
    for invalid in ({}, {"index_tip_norm": (math.nan, 0.5)}, {"index_tip_norm": (0.5,)}):
        assert not mapper.calibrate(invalid)
        assert (mapper.baseline_x, mapper.baseline_y) == (0.7, 0.4)
    mapper.reset(reset_baseline=True)
    assert (mapper.baseline_x, mapper.baseline_y) == (0.5, 0.5)
    configured = HandGestureMapper({"gestures": {"hand": {"ema_alpha": 1.0, "deadzone_x": 0.2}}})
    assert configured.map(_hand((0.65, 0.5)))[0] == Direction.STOP
    command, confidence, _ = configured.map(_hand((0.75, 0.5)))
    assert command == Direction.RIGHT and np.isclose(confidence, 0.85)
    print("Test 3 PASSED: index coordinate signs, dominant axis/ties, EMA, brakes and neutral calibration.")


def test_eye_mapper() -> None:
    mapper = EyeGestureMapper()
    assert (mapper.ema_alpha, mapper.deadzone_x, mapper.deadzone_y, mapper.closed_ear_threshold) == (0.45, 0.12, 0.11, 0.19)
    for pupil, command in (((0.5, 0.5), Direction.STOP), ((0.5, 0.25), Direction.UP),
                            ((0.5, 0.75), Direction.DOWN), ((0.25, 0.5), Direction.LEFT),
                            ((0.75, 0.5), Direction.RIGHT), ((0.74, 0.26), Direction.UP)):
        mapper.reset()
        output = _eye(pupil=pupil)
        original = deepcopy(output.features)
        result, confidence, label = mapper.map(output)
        assert result == command and confidence == (0.9 if command == Direction.STOP else 1.0)
        assert label == ("PUPIL CENTER (STOP)" if command == Direction.STOP else "PUPIL " + command.value)
        assert output.features == original
    tie = EyeGestureMapper({"gestures": {"eye": {"deadzone_x": 0.125, "deadzone_y": 0.125, "ema_alpha": 1.0}}})
    assert tie.map(_eye(pupil=(0.25, 0.25)))[0] == Direction.LEFT
    assert tie.map(_eye(pupil=(0.75, 0.75)))[0] == Direction.RIGHT

    preferred = _eye(pupil=(0.25, 0.5))
    preferred.features["iris_pupil_ratio"] = (0.75, 0.5)
    mapper.reset()
    assert mapper.map(preferred)[0] == Direction.RIGHT
    for invalid_iris in (None, (math.nan, 0.5), (True, 0.5), (0.5,), (0.5, 0.5, 0.5)):
        preferred.features["iris_pupil_ratio"] = invalid_iris
        mapper.reset()
        assert mapper.map(preferred)[0] == Direction.LEFT
    assert mapper.map(_output("EYE", {}, detected=False)) == (Direction.STOP, 0.0, "NO EYE")
    assert mapper.map(_output("EYE", {})) == (Direction.STOP, 0.0, "NO PUPIL")

    mapper.reset()
    assert mapper.map(_eye())[0] == Direction.STOP
    assert mapper.map(_eye(pupil=(0.25, 0.5)))[0] == Direction.STOP
    assert mapper.map(_eye(pupil=(0.25, 0.5)))[0] == Direction.LEFT
    smoothed = mapper._smooth_x, mapper._smooth_y
    assert mapper.map(_eye(0.14, 0.14, (0.99, 0.01))) == (Direction.STOP, 0.8, "EYES CLOSED (STOP)")
    assert (mapper._smooth_x, mapper._smooth_y) == smoothed
    assert mapper.map(_eye(pupil=(0.25, 0.5)))[0] == Direction.LEFT
    for output in (_eye(0.13, 0.29), _eye(0.29, 0.13), _eye(0.35, 0.35)):
        mapper.reset()
        assert mapper.map(output)[0] == Direction.STOP  # Winks/wide eyes no longer steer.
    mapper.reset()
    for timestamp, output in ((0.0, _eye()), (0.1, _eye(0.12, 0.12)), (0.2, _eye()),
                               (0.4, _eye(0.12, 0.12)), (0.5, _eye())):
        assert mapper.map(output, timestamp)[0] == Direction.STOP
    assert mapper.calibrate({"iris_pupil_ratio": (0.7, 0.4), "avg_pupil_ratio": (0.3, 0.6)})
    assert mapper.map(_eye(pupil=(0.7, 0.4)))[0] == Direction.STOP
    mapper.reset()
    assert (mapper.baseline_x, mapper.baseline_y) == (0.7, 0.4)
    assert mapper.map(_eye(pupil=(0.45, 0.4)))[0] == Direction.LEFT
    assert not mapper.calibrate({"avg_pupil_ratio": (math.nan, 0.5)})
    assert (mapper.baseline_x, mapper.baseline_y) == (0.7, 0.4)
    mapper.reset(reset_baseline=True)
    assert (mapper.baseline_x, mapper.baseline_y) == (0.5, 0.5)
    configured = EyeGestureMapper({"gestures": {"eye": {"ema_alpha": 1.0, "deadzone_x": 0.2}}})
    assert configured.map(_eye(pupil=(0.65, 0.5)))[0] == Direction.STOP
    command, confidence, _ = configured.map(_eye(pupil=(0.75, 0.5)))
    assert command == Direction.RIGHT and np.isclose(confidence, 0.85)

    # Prove the additive shared feature uses both axes while preserving standalone dlib data.
    from test_phase3_controllers import MockDetector, MockPredictor

    def iris_points(pupil):
        points = np.zeros((480, 2), dtype=np.float32)
        for iris, left, right, upper, lower, origin in ((468, 33, 133, 159, 145, 100),
                                                       (473, 263, 362, 386, 374, 300)):
            points[left], points[right] = (origin, 100), (origin + 100, 100)
            points[upper], points[lower] = (origin + 50, 90), (origin + 50, 110)
            points[iris] = (origin + 100 * pupil[0], 90 + 20 * pupil[1])
        return points

    detector, predictor = MockDetector(), MockPredictor()
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    cv2.circle(frame, (212, 200), 3, (20, 20, 20), -1)
    cv2.circle(frame, (388, 200), 3, (20, 20, 20), -1)
    before = frame.tobytes()
    with EyeController(detector=detector, predictor=predictor) as controller:
        standalone = controller.process_frame(frame, 0, draw=False)
        assert "iris_pupil_ratio" not in standalone.features
        for pupil, command in (((0.5, 0.25), Direction.UP), ((0.5, 0.75), Direction.DOWN),
                                ((0.25, 0.5), Direction.LEFT), ((0.75, 0.5), Direction.RIGHT)):
            points = iris_points(pupil)
            unchanged = points.tobytes()
            shared = controller.process_frame(frame, 1, draw=False, shared_context={"face_landmarks_px": points})
            assert np.allclose(shared.features["iris_pupil_ratio"], pupil)
            assert np.isclose(shared.features["iris_gaze_ratio"], pupil[0])
            assert np.array_equal(shared.landmarks_px, standalone.landmarks_px)
            for key in ("left_ear", "right_ear", "avg_ear", "left_pupil_ratio", "right_pupil_ratio", "avg_pupil_ratio"):
                assert shared.features[key] == standalone.features[key]
            assert points.tobytes() == unchanged and frame.tobytes() == before
            mapper.reset()
            assert mapper.map(shared)[0] == command
        assert np.allclose(controller._iris_pupil_ratio({"face_landmarks_px": iris_points((-0.2, 1.2))}), (0.0, 1.0))
        near_closed = iris_points((0.8, 0.2))
        near_closed[[159, 145, 386, 374], 1] = 100
        assert np.allclose(controller._iris_pupil_ratio({"face_landmarks_px": near_closed}), (0.8, 0.5))
        invalid = iris_points((0.5, 0.5))
        invalid[159] = math.nan
        for context in (None, {"face_landmarks_px": np.zeros((477, 2))}, {"face_landmarks_px": invalid}):
            assert controller._iris_pupil_ratio(context) is None
        predictor.closed_eyes = True
        closed = controller.process_frame(frame, 2, draw=False, shared_context={"face_landmarks_px": iris_points((0.1, 0.9))})
        assert mapper.map(closed) == (Direction.STOP, 0.8, "EYES CLOSED (STOP)")
    assert detector.close_count == predictor.close_count == 1
    print("Test 4 PASSED: pure 2D pupils, iris preference/ROI fallback, closed-eye guard, EMA and calibration.")


def _receive(subscriber: CommandSubscriber) -> contracts.ControlPacket:
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        packet = subscriber.poll_latest()
        if packet is not None:
            return packet
        time.sleep(0.001)
    raise AssertionError("No loopback packet arrived within 0.5 seconds.")


def test_engine_udp_arena() -> None:
    engine = GestureEngine({"gestures": {"hand": {"ema_alpha": 1.0}}})
    arena = arrow_game.ArrowArena()
    x, y = arena.x, arena.y
    with CommandSubscriber("127.0.0.1", 0) as subscriber:
        with CommandPublisher("127.0.0.1", subscriber.bound_port) as publisher:
            for point, timestamp, previous, command, alarms, position in (
                    ((0.5, 0.25), 10.0, Direction.STOP, Direction.UP, [], (x, y - 1)),
                    ((0.75, 0.5), 10.2, Direction.UP, Direction.RIGHT, ["YAWN"], (x + 1, y - 1))):
                for index in range(3):
                    packet, label = engine.build_packet(_hand(point), alarms, 30.0,
                                                        timestamp_sec=timestamp + index * 0.02)
                    assert packet.command == (previous if index < 2 else command).value
                assert command.value in label
                assert publisher.publish(packet)
                received = _receive(subscriber)
                assert received == packet
                arena.apply_packet(received, now_sec=timestamp)
                arena.step(now_sec=timestamp + 0.01)
                assert (arena.x, arena.y) == position
                assert arena.mode == "HAND" and arena.command == command.value
                assert arena.active_alarms == alarms and arena.confidence == 1.0 and arena.fps == 30.0
            for index in range(3):
                packet, label = engine.build_packet(_output("HAND", {}, detected=False), ["YAWN"],
                                                    30.0, timestamp_sec=10.4 + index * 0.02)
                assert packet.command == (Direction.RIGHT if index < 2 else Direction.STOP).value
            assert publisher.publish(packet)
            arena.apply_packet(_receive(subscriber), now_sec=10.4)
            arena.step(now_sec=10.41)
            assert (arena.x, arena.y) == (x + 1, y - 1) and arena.active_alarms == ["YAWN"]

    # A new mode cannot inherit a moving command. Calibration also releases movement.
    assert engine.map_output(_face(-0.25), 11.0)[0] == Direction.STOP
    assert engine.map_output(_face(-0.25), 11.1)[0] == Direction.STOP
    assert engine.map_output(_face(-0.25), 11.2)[0] == Direction.LEFT
    assert engine.calibrate_face(_face(-0.25))
    assert engine.map_output(_face(-0.25), 11.3)[0] == Direction.STOP
    assert not engine.calibrate_face(_hand())
    alarms = ["YAWN"]
    packet, _ = engine.build_packet(_face(-0.25), alarms, 27.5, 11.4)
    alarms.append("SLEEP")
    assert packet.active_alarms == ["YAWN"] and packet.timestamp == 11.4
    assert contracts.ControlPacket.from_json(packet.to_json()) == packet

    # Pupil movement and closed-eye STOP both use ordinary consecutive-frame debouncing.
    responsive = GestureEngine({"control": {"debounce_frames": 8}, "gestures": {"eye": {"ema_alpha": 1.0}}})
    for index in range(8):
        result = responsive.map_output(_eye(pupil=(0.5, 0.25)), index * 0.01)
        assert result[0] == (Direction.STOP if index < 7 else Direction.UP)
    for index in range(8):
        result = responsive.map_output(_eye(0.14, 0.14), 0.1 + index * 0.01)
        assert result[0] == (Direction.UP if index < 7 else Direction.STOP)
    responsive.reset()
    for timestamp, output in ((0.0, _eye()), (0.1, _eye(0.12, 0.12)), (0.2, _eye()),
                               (0.4, _eye(0.12, 0.12)), (0.5, _eye())):
        result = responsive.map_output(output, timestamp)
        assert result[0] == Direction.STOP
    responsive.reset()
    assert responsive.map_output(_eye(), 2.0)[0] == Direction.STOP
    assert engine.map_output(_output("NONE", {}), 12.0)[0] == Direction.STOP
    engine.reset()
    _verify_active_mode_calibration()
    print("Test 5 PASSED: tracked/debounced UDP-to-arrow displacement, alarms and universal calibration.")


def _verify_active_mode_calibration() -> None:
    from test_phase6_master import MockController, _fixture, _steps

    class TrackingController(MockController):
        def process_frame(self, *args, **kwargs):
            output = super().process_frame(*args, **kwargs)
            if self.mode == contracts.ControlMode.HAND:
                output.features["index_tip_norm"] = (0.5, 0.25)
            elif self.mode == contracts.ControlMode.EYE:
                output.features["iris_pupil_ratio"] = (0.75, 0.25)
            return output

    master, fixture = _fixture()

    def factory(mode, config):
        assert all(controller.close_calls == 1 for controller in fixture.controllers)
        controller = TrackingController(mode)
        fixture.controllers.append(controller)
        return controller

    fixture.factory.side_effect = factory
    try:
        for mode in contracts.ControlMode.FACE, contracts.ControlMode.HAND, contracts.ControlMode.EYE:
            master.switch_mode(mode)
            assert master._last_control_output is None and master._last_face_output is None
            _steps(master, 3)
            cached = master._last_control_output
            assert cached is not None and cached.mode == mode.value
            assert (master._last_face_output is cached) if mode == contracts.ControlMode.FACE else master._last_face_output is None
            with patch.object(master.gesture_engine, "calibrate", wraps=master.gesture_engine.calibrate) as calibrate:
                confirmation = io.StringIO()
                with redirect_stdout(confirmation):
                    master.enqueue_action(("CALIBRATE_FACE", None))
                    _steps(master, 1)
                assert calibrate.call_count == 1 and calibrate.call_args.args[0] is cached
                assert f"Calibrated neutral center for {mode.value} mode" in confirmation.getvalue()
            assert fixture.publisher.publish.call_args.args[0].command == "STOP"
            assert fixture.publisher.publish.call_args.args[0].mode == mode.value
        assert master.safety_frames_processed == fixture.camera.frames == 12
        assert master.control_frames_processed == 12
        assert not master.gesture_engine.calibrate(_output("NONE", {}))
        assert not master.gesture_engine.calibrate(_output("HAND", {}, detected=False))
        master.switch_mode("NONE")
        assert master._last_control_output is None and master._last_face_output is None
    finally:
        master.stop()
    assert master._last_control_output is None and master._last_face_output is None
    assert fixture.camera.stop_calls == fixture.safety.close_calls == 1


def _terminate_game(game: subprocess.Popen) -> None:
    if game.poll() is None:
        game.terminate()
        try:
            game.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            game.kill()
            game.wait(timeout=1.0)


def run_live(mode: str) -> None:
    # Heavy models, webcam, audio and the new console are exclusively opt-in.
    from camera_hub import CameraHub
    from controllers import create_controller
    from safety_engine import AlarmManager, SafetyEngine

    config = contracts.load_config()
    gestures = GestureEngine(config)
    with ExitStack() as resources:
        safety = resources.enter_context(SafetyEngine(config))
        controller = resources.enter_context(create_controller(mode, config))
        alarms = AlarmManager(config, muted=False)
        resources.callback(alarms.stop)
        publisher = resources.enter_context(CommandPublisher())
        camera = resources.enter_context(CameraHub())
        game = arrow_game.launch_arrow_terminal()
        resources.callback(_terminate_game, game)
        resources.callback(cv2.destroyAllWindows)
        last_frame_id = -1
        last_processed = last_fresh = time.monotonic()
        measured_fps = 0.0
        window = f"Phase 5 - {mode.upper()} gesture control"
        print("Live control: C calibrates FACE neutral, Q/Esc quits. Arrow game runs in its own terminal.")
        try:
            while camera.is_running() and game.poll() is None:
                frame_packet = camera.get_latest(last_frame_id, timeout=0.1)
                if frame_packet is None:
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    if time.monotonic() - last_fresh > 3.0:
                        print("Live control stopped: no fresh camera frames for 3 seconds.")
                        break
                    continue
                last_frame_id = frame_packet.frame_id
                last_fresh = time.monotonic()
                dms = safety.process_frame(frame_packet.frame, frame_packet.timestamp_ms, draw=False)
                context = {"face_landmarks_norm": dms.landmarks_norm, "face_landmarks_px": dms.landmarks_px}
                output = controller.process_frame(frame_packet.frame, frame_packet.timestamp_ms,
                                                  draw=True, shared_context=context)
                now = time.monotonic()
                instant_fps = 1.0 / max(now - last_processed, 1e-6)
                measured_fps = instant_fps if measured_fps == 0.0 else 0.2 * instant_fps + 0.8 * measured_fps
                last_processed = now
                packet, label = gestures.build_packet(output, dms.status.active_alarms, measured_fps,
                                                       timestamp_sec=frame_packet.timestamp_ms / 1000.0)
                publisher.publish(packet)
                alarms.trigger(dms.status.active_alarms, now_sec=now)
                annotated = output.annotated_frame
                height, width = annotated.shape[:2]
                cv2.rectangle(annotated, (0, max(0, height - 82)), (width - 1, height - 1), (24, 24, 24), -1)
                color = (70, 70, 255) if dms.status.active_alarms else (90, 220, 130)
                for offset, text in ((63, f"{label} | CMD: {packet.command} | FPS: {measured_fps:.1f}"),
                                     (38, "DMS: " + (" | ".join(dms.status.active_alarms) or "OK")),
                                     (13, "C: face calibration | Q/Esc: quit")):
                    cv2.putText(annotated, text, (10, max(15, height - offset)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, color, 1, cv2.LINE_AA)
                cv2.imshow(window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("c"), ord("C")):
                    print("Face calibrated." if gestures.calibrate_face(output) else "Face calibration unavailable.")
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            publisher.publish(contracts.ControlPacket(time.monotonic(), mode.upper(), "STOP", 0.0, [],
                                                      measured_fps), force=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Open camera, DMS, controller and arrow terminal.")
    parser.add_argument("--mode", choices=("face", "hand", "eye"), default="face")
    args = parser.parse_args()
    if args.live:
        run_live(args.mode)
        return
    with ExitStack() as guards:
        guards.enter_context(patch("cv2.VideoCapture", side_effect=AssertionError("Headless test opened a camera.")))
        guards.enter_context(patch("cv2.imshow", side_effect=AssertionError("Headless test opened a GUI.")))
        guards.enter_context(patch("subprocess.Popen", side_effect=AssertionError("Headless test spawned a console.")))
        guards.enter_context(patch("builtins.input", side_effect=AssertionError("Headless test waited for input.")))
        guards.enter_context(patch("arrow_game._read_key", side_effect=AssertionError("Headless test read a key.")))
        test_debouncer()
        test_face_mapper()
        test_hand_mapper()
        test_eye_mapper()
        test_engine_udp_arena()
    print("PHASE 5 VERIFIED OK")


if __name__ == "__main__":
    main()
