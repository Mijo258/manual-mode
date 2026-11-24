"""Deterministic headless verification of the production master orchestration."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True

import cv2
import numpy as np

import contracts
from av_master import AVMasterOrchestrator, MENU_TEXT, parse_menu_choice
from camera_hub import FramePacket
from controllers import BaseController, ControllerOutput, create_controller
from gesture_mapper import GestureEngine
from ipc_bus import CommandPublisher, CommandSubscriber


Mode = contracts.ControlMode


class MockCamera:
    def __init__(self):
        self.actual_fps = 30.0
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.frames = 0
        self.start_calls = self.stop_calls = 0
        self.empty = False

    def start(self):
        self.start_calls += 1

    def get_latest(self, last_frame_id=-1, timeout=0.05, copy=True):
        assert timeout == 0.05 and copy is True
        if self.empty:
            return None
        self.frames += 1
        assert self.frames > last_frame_id
        return FramePacket(self.frames, 100 + self.frames * 33, self.frame.copy())

    def stop(self):
        self.stop_calls += 1


class MockSafety:
    def __init__(self, alarms=None):
        self.alarms = [] if alarms is None else list(alarms)
        self.calls = []
        self.close_calls = 0
        self.normalized = [(0.5, 0.5, 0.0)] * 478
        self.pixels = np.full((478, 2), (320.0, 240.0), dtype=np.float32)

    def process_frame(self, frame, timestamp_ms, draw=True):
        self.calls.append((timestamp_ms, draw))
        status = contracts.SafetyStatus(timestamp_ms / 1000.0, True, 0.28, 0.2, 0.5,
                                        "SLEEP" in self.alarms, "YAWN" in self.alarms,
                                        "DISTRACTION" in self.alarms, list(self.alarms))
        return SimpleNamespace(status=status, annotated_frame=frame.copy() if draw else frame,
                               landmarks_norm=self.normalized, landmarks_px=self.pixels)

    def close(self):
        self.close_calls += 1


class MockAlarms:
    def __init__(self):
        self.calls = []
        self.stop_calls = 0

    def trigger(self, active_alarms):
        self.calls.append(list(active_alarms))
        return bool(active_alarms)

    def stop(self):
        self.stop_calls += 1


class MockController(BaseController):
    def __init__(self, mode):
        self.mode = mode
        self.contexts = []
        self.close_calls = 0

    def process_frame(self, frame, timestamp_ms, draw=True, shared_context=None):
        assert not draw, "Headless orchestration requested controller drawing."
        assert self.close_calls == 0
        self.contexts.append(shared_context)
        if self.mode == Mode.FACE:
            features = {"yaw_offset": 0.25, "pitch_offset": 0.0}
        elif self.mode == Mode.HAND:
            features = {"finger_states": {"thumb": False, "index": True, "middle": False,
                                           "ring": False, "pinky": False}, "extended_count": 1}
        else:
            features = {"left_ear": 0.28, "right_ear": 0.28, "avg_ear": 0.28,
                        "avg_pupil_ratio": (0.5, 0.5)}
        return ControllerOutput(self.mode.value, True, frame, None, features, confidence=1.0)

    def close(self):
        self.close_calls += 1


class MockProcess:
    def __init__(self, slow_exit=False):
        self.returncode = None
        self.terminate_calls = self.kill_calls = 0
        self.waits = []
        self.slow_exit = slow_exit

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        if not self.slow_exit:
            self.returncode = 0

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("mock arrow game", timeout)
        return self.returncode

    def kill(self):
        self.kill_calls += 1
        self.returncode = -1


def _fixture(config=None, publisher=None, alarms=None):
    camera, safety, alarm_manager = MockCamera(), MockSafety(alarms), MockAlarms()
    created, processes = [], []

    def factory(mode, cfg):
        assert all(controller.close_calls == 1 for controller in created), "Multiple controllers remained active."
        controller = MockController(mode)
        created.append(controller)
        return controller

    def launch(**kwargs):
        process = MockProcess()
        processes.append(process)
        return process

    factory_mock, launcher_mock = Mock(side_effect=factory), Mock(side_effect=launch)
    publisher = Mock(spec=CommandPublisher) if publisher is None else publisher
    orchestrator = AVMasterOrchestrator(config=config, camera_hub=camera, safety_engine=safety,
                                        alarm_manager=alarm_manager, gesture_engine=GestureEngine(config),
                                        publisher=publisher, controller_factory=factory_mock,
                                        terminal_launcher=launcher_mock, headless=True, enable_menu_thread=False)
    return orchestrator, SimpleNamespace(camera=camera, safety=safety, alarms=alarm_manager,
                                         publisher=publisher, factory=factory_mock, launcher=launcher_mock,
                                         controllers=created, processes=processes)


def _steps(orchestrator, count):
    for _ in range(count):
        assert orchestrator.step_once() is True


def _receive(subscriber):
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        packet = subscriber.poll_latest()
        if packet is not None:
            return packet
        time.sleep(0.001)
    raise AssertionError("No loopback packet arrived within 0.5 seconds.")


def test_startup_safety_first():
    orch, fixture = _fixture(alarms=["SLEEP"])
    try:
        orch.start()
        orch.start()  # Idempotent start must not launch a second camera producer.
        _steps(orch, 5)
        assert orch.safety_frames_processed == fixture.camera.frames == 5
        assert orch.control_frames_processed == 0 and orch.active_controller is None
        assert fixture.alarms.calls == [["SLEEP"]] * 5
        fixture.factory.assert_not_called()
        fixture.launcher.assert_not_called()
        assert fixture.camera.start_calls == 1 and orch._menu_thread is None
        assert all(not draw for _, draw in fixture.safety.calls)
        fixture.camera.empty = True
        assert orch.step_once() is True and orch.safety_frames_processed == 5
        orch.enqueue_action(("QUIT", None))
        assert orch.step_once() is False
    finally:
        orch.stop()
    print("Test 1 PASSED: five safety-only frames, alarms, one camera start and no controller/terminal.")


def test_exclusive_switching(orch, fixture, subscriber):
    orch.enqueue_action(("SET_MODE", "FACE"))
    _steps(orch, 4)
    assert [controller.mode for controller in fixture.controllers] == [Mode.FACE]
    assert orch.active_mode == Mode.FACE and orch.safety_frames_processed == 4
    assert fixture.launcher.call_count == 1 and orch._last_face_output is not None
    face = fixture.controllers[0]
    for context in face.contexts:
        assert len(context["face_landmarks_norm"]) == 478
        assert context["face_landmarks_norm"] is fixture.safety.normalized
        assert context["face_landmarks_px"] is fixture.safety.pixels
    packet = _receive(subscriber)
    assert packet.mode == "FACE" and packet.command == "RIGHT"

    fixture.safety.alarms = ["YAWN"]
    orch.enqueue_action(("CALIBRATE_FACE", None))
    orch.enqueue_action(("SET_MODE", "HAND"))
    _steps(orch, 4)
    assert face.close_calls == 1 and orch.active_mode == Mode.HAND and orch._last_face_output is None
    assert [controller.mode for controller in fixture.controllers] == [Mode.FACE, Mode.HAND]
    assert fixture.launcher.call_count == 1 and orch.safety_frames_processed == 8
    assert fixture.alarms.calls[-4:] == [["YAWN"]] * 4
    packet = _receive(subscriber)
    assert packet.mode == "HAND" and packet.command == "UP" and packet.active_alarms == ["YAWN"]

    orch.enqueue_action(("SET_MODE", "EYE"))
    _steps(orch, 4)
    assert fixture.controllers[1].close_calls == 1 and orch.active_mode == Mode.EYE
    assert [controller.mode for controller in fixture.controllers] == [Mode.FACE, Mode.HAND, Mode.EYE]
    packet = _receive(subscriber)
    assert packet.mode == "EYE" and packet.command == "STOP" and packet.active_alarms == ["YAWN"]
    assert orch.safety_frames_processed == orch.control_frames_processed == fixture.camera.frames == 12
    assert len(fixture.safety.calls) == len(fixture.alarms.calls) == 12
    assert [ts for ts, _ in fixture.safety.calls] == [100 + index * 33 for index in range(1, 13)]
    fixture.launcher.assert_called_once_with(host="127.0.0.1", port=subscriber.bound_port)
    print("Test 2 PASSED: exclusive FACE/HAND/EYE, shared landmark references, continuous safety and real UDP.")


def test_standby_respawn(orch, fixture, subscriber):
    eye = orch.active_controller
    orch.enqueue_action(("SET_MODE", "NONE"))
    _steps(orch, 2)
    assert eye.close_calls == 1 and orch.active_controller is None and orch.active_mode == Mode.NONE
    packet = _receive(subscriber)
    assert packet.mode == "NONE" and packet.command == "STOP" and packet.active_alarms == ["YAWN"]
    assert orch.safety_frames_processed == 14 and orch.control_frames_processed == 12
    assert fixture.launcher.call_count == 1
    fixture.processes[0].returncode = 0  # User closed the original arrow console.
    orch.enqueue_action(("SET_MODE", "HAND"))
    _steps(orch, 1)
    assert orch.active_mode == Mode.HAND and fixture.launcher.call_count == 2
    assert orch.game_process is fixture.processes[1] and orch.safety_frames_processed == 15
    hand = orch.active_controller
    orch.switch_mode("hand")
    assert orch.active_controller is hand and fixture.factory.call_count == 4 and fixture.launcher.call_count == 2
    print("Test 3 PASSED: standby STOP/alarms keep safety running; exited terminal respawns and live terminal is reused.")


def test_parser_shutdown(orch, fixture, subscriber):
    for choices, expected in ((("1", "face", " FACE "), ("SET_MODE", "FACE")),
                               (("2", "hand"), ("SET_MODE", "HAND")),
                               (("3", "eye"), ("SET_MODE", "EYE")),
                               (("0", "standby", "none"), ("SET_MODE", "NONE")),
                               (("c", "calibrate"), ("CALIBRATE_FACE", None)),
                               (("q", "quit", "exit"), ("QUIT", None))):
        for choice in choices:
            assert parse_menu_choice(choice) == expected
    assert parse_menu_choice("xyz") is None and parse_menu_choice(" ") is None
    assert "[SAFETY DMS: ALWAYS ACTIVE IN BACKGROUND]" in MENU_TEXT
    active = orch.active_controller
    process = orch.game_process
    orch.stop()
    publish_calls = fixture.publisher.publish.call_count
    orch.stop()
    assert process.terminate_calls == 1 and process.waits == [1.5] and process.kill_calls == 0
    assert active.close_calls == 1 and all(controller.close_calls == 1 for controller in fixture.controllers)
    assert fixture.safety.close_calls == fixture.alarms.stop_calls == fixture.camera.stop_calls == 1
    fixture.publisher.close.assert_called_once()
    assert fixture.publisher.publish.call_count == publish_calls
    packet = _receive(subscriber)
    assert packet.command == "STOP" and packet.active_alarms == ["YAWN"]
    assert orch.step_once() is False

    # Failure of an optional controller must leave the safety loop in standby.
    failed, f = _fixture()
    try:
        failed.switch_mode("FACE")
        _steps(failed, 1)
        f.factory.side_effect = FileNotFoundError("injected missing hand model")
        with redirect_stdout(io.StringIO()):
            failed.enqueue_action(("SET_MODE", "HAND"))
            _steps(failed, 2)
        assert failed.active_mode == Mode.NONE and failed.active_controller is None
        assert f.controllers[0].close_calls == 1 and failed.safety_frames_processed == 3
    finally:
        failed.stop()

    # One shutdown failure must not prevent later resources, especially the webcam, closing.
    failing, f = _fixture()
    failing.switch_mode("HAND")
    f.processes[-1].slow_exit = True
    failing.active_controller.close = Mock(side_effect=RuntimeError("injected controller close error"))
    f.safety.close = Mock(side_effect=RuntimeError("injected safety close error"))
    f.alarms.stop = Mock(side_effect=RuntimeError("injected alarm stop error"))
    f.publisher.publish.side_effect = OSError("injected UDP send error")
    f.publisher.close.side_effect = OSError("injected socket close error")
    with redirect_stdout(io.StringIO()):
        failing.stop()
        failing.stop()
    assert f.camera.stop_calls == 1 and f.processes[-1].kill_calls == 1
    f.safety.close.assert_called_once()
    f.alarms.stop.assert_called_once()
    f.publisher.close.assert_called_once()

    # Verify the menu daemon only enqueues actions; input is mocked and never waits.
    menu, f = _fixture()
    read_threads = []
    choices = iter(("hand", "c", "q"))

    def fake_input(prompt):
        read_threads.append(threading.current_thread())
        return next(choices)

    with patch("builtins.input", side_effect=fake_input), redirect_stdout(io.StringIO()):
        thread = threading.Thread(target=menu._menu_loop, daemon=True)
        thread.start()
        thread.join(timeout=1.0)
    assert not thread.is_alive() and all(reader is thread for reader in read_threads)
    f.factory.assert_not_called()
    f.launcher.assert_not_called()
    assert list(menu._action_queue.queue) == [("SET_MODE", "HAND"), ("CALIBRATE_FACE", None), ("QUIT", None)]
    menu.stop()

    # Non-headless GUI orchestration refuses worker-thread mode changes before touching a window/model.
    guarded, f = _fixture()
    guarded.headless = False
    errors = []

    def worker_switch():
        try:
            guarded.switch_mode("HAND")
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target=worker_switch, daemon=True)
    worker.start()
    worker.join(timeout=1.0)
    guarded.headless = True
    guarded.stop()
    assert not worker.is_alive() and errors and "main thread" in errors[0]
    f.factory.assert_not_called()
    print("Test 4 PASSED: parser, idempotent/error-isolated shutdown, queued daemon input and main-thread guard.")


def test_production_wiring():
    from safety_engine import AlarmManager, SafetyEngine

    with CommandSubscriber("127.0.0.1", 0) as subscriber:
        config = deepcopy(contracts.load_config())
        config["ipc"]["port"] = subscriber.bound_port
        camera = MockCamera()
        process = MockProcess()
        launcher = Mock(return_value=process)
        safety = SafetyEngine(config)
        alarms = AlarmManager(config, muted=True)
        publisher = CommandPublisher("127.0.0.1", subscriber.bound_port)
        orch = AVMasterOrchestrator(config=config, camera_hub=camera, safety_engine=safety,
                                    alarm_manager=alarms, gesture_engine=GestureEngine(config),
                                    publisher=publisher, controller_factory=create_controller,
                                    terminal_launcher=launcher, headless=True, enable_menu_thread=False, muted=True)
        created = []
        try:
            for mode in (Mode.NONE, Mode.FACE, Mode.HAND, Mode.EYE, Mode.NONE):
                orch.enqueue_action(("SET_MODE", mode.value))
                _steps(orch, 2)
                assert orch.active_mode == mode
                if mode != Mode.NONE:
                    assert isinstance(orch.active_controller, BaseController)
                    created.append(orch.active_controller)
            assert orch.safety_frames_processed == camera.frames == 10 and orch.control_frames_processed == 6
            assert np.count_nonzero(camera.frame) == 0
            assert _receive(subscriber).mode == "NONE"
            launcher.assert_called_once_with(host="127.0.0.1", port=subscriber.bound_port)
        finally:
            orch.stop()
        assert camera.stop_calls == 1 and safety._closed and alarms._stop_event.is_set()
        assert publisher._socket is None and process.terminate_calls == 1
        assert all(getattr(controller, "_closed", True) for controller in created)
    print("Test 5 PASSED: real face/hand MediaPipe, dlib eyes, gesture, muted alarms and UDP across all modes.")


def main():
    original_popen = subprocess.Popen

    def hidden_system_process(*args, **kwargs):
        # MediaPipe's sounddevice import may query Windows via platform.py's
        # cmd /c ver. Permit that stdlib probe without creating any console;
        # the application's interactive launcher remains forbidden below.
        flags = kwargs.get("creationflags", 0)
        if os.name == "nt":
            if flags & subprocess.CREATE_NEW_CONSOLE:
                raise AssertionError("Headless test requested a new console.")
            kwargs["creationflags"] = flags | subprocess.CREATE_NO_WINDOW
        return original_popen(*args, **kwargs)

    with ExitStack() as guards:
        for name in ("VideoCapture", "imshow", "waitKey", "destroyWindow", "destroyAllWindows", "namedWindow"):
            guards.enter_context(patch.object(cv2, name, side_effect=AssertionError(f"Headless test called cv2.{name}.")))
        guards.enter_context(patch("subprocess.Popen", side_effect=hidden_system_process))
        guards.enter_context(patch("arrow_game.launch_arrow_terminal", side_effect=AssertionError("Headless test launched an arrow console.")))
        guards.enter_context(patch("builtins.input", side_effect=AssertionError("Headless test waited for real input.")))
        test_startup_safety_first()
        with CommandSubscriber("127.0.0.1", 0) as subscriber:
            config = deepcopy(contracts.load_config())
            config["ipc"]["port"] = subscriber.bound_port
            publisher = Mock(wraps=CommandPublisher("127.0.0.1", subscriber.bound_port))
            orch, fixture = _fixture(config, publisher)
            try:
                test_exclusive_switching(orch, fixture, subscriber)
                test_standby_respawn(orch, fixture, subscriber)
                test_parser_shutdown(orch, fixture, subscriber)
            finally:
                orch.stop()
        test_production_wiring()
    print("PHASE 6 VERIFIED OK")


if __name__ == "__main__":
    main()
