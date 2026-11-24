"""Safety-first single-camera orchestrator with queued terminal menu actions."""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

# Running this entry point must not create/replace imported project bytecode.
sys.dont_write_bytecode = True

import cv2

import arrow_game
import contracts
from camera_hub import CameraHub
from controllers import BaseController, ControllerOutput, create_controller
from gesture_mapper import GestureEngine
from ipc_bus import CommandPublisher

if TYPE_CHECKING:
    from safety_engine import AlarmManager, SafetyEngine


Mode = contracts.ControlMode
CONTROL_WINDOW = "Active Control View"
SAFETY_WINDOW = "DMS Safety Monitor (Always Active)"
MENU_TEXT = """
============================================================
 AUTONOMOUS VEHICLE - CONTROL MENU
 [SAFETY DMS: ALWAYS ACTIVE IN BACKGROUND]
============================================================
 [1] Face Landmark Control  (478-pt Face Mesh ONLY)
 [2] Hand Landmark Control  (21-pt Hand Model ONLY)
 [3] Eye Landmark Control   (68-pt dlib Eye ONLY)
 [c] Calibrate Neutral Center (Face / Hand / Eye)
 [0] Standby (Close Control View, Keep Safety Active)
 [q] Quit Entire System Safely
============================================================
"""


def parse_menu_choice(raw: str) -> tuple[str, str | None] | None:
    choice = raw.strip().lower()
    modes = {"1": Mode.FACE, "face": Mode.FACE, "2": Mode.HAND, "hand": Mode.HAND,
             "3": Mode.EYE, "eye": Mode.EYE, "0": Mode.NONE, "standby": Mode.NONE, "none": Mode.NONE}
    if choice in modes:
        return "SET_MODE", modes[choice].value
    if choice in ("c", "calibrate"):
        return "CALIBRATE_FACE", None
    if choice in ("q", "quit", "exit"):
        return "QUIT", None
    return None


class AVMasterOrchestrator:
    """The main thread owns inference, mode changes and every OpenCV window.

    The menu daemon only reads text and queues actions. Headless operation
    suppresses input and the default terminal launcher; an injected launcher
    can record lifecycle events for tests. Injected resources are owned and
    closed by this orchestrator just like the production resources.
    """

    def __init__(
        self,
        config: dict | None = None,
        camera_hub: CameraHub | None = None,
        safety_engine: SafetyEngine | None = None,
        alarm_manager: AlarmManager | None = None,
        gesture_engine: GestureEngine | None = None,
        publisher: CommandPublisher | None = None,
        controller_factory=None,
        terminal_launcher=None,
        headless: bool = False,
        enable_menu_thread: bool = True,
        show_safety_window: bool = True,
        muted: bool = False,
    ):
        self.config = contracts.load_config() if config is None else config
        self.headless = bool(headless)
        self.enable_menu_thread = bool(enable_menu_thread) and not self.headless
        self.show_safety_window = bool(show_safety_window)
        self.controller_factory = create_controller if controller_factory is None else controller_factory
        self.terminal_launcher = arrow_game.launch_arrow_terminal if terminal_launcher is None else terminal_launcher
        self._terminal_enabled = not self.headless or terminal_launcher is not None
        self.active_mode: Mode = Mode.NONE
        self.active_controller: BaseController | None = None
        self.game_process: subprocess.Popen | None = None
        self._action_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self._stop_event = threading.Event()
        self._last_frame_id = -1
        self._last_face_output: ControllerOutput | None = None
        self._last_control_output: ControllerOutput | None = None
        self._control_window_open = False
        self.safety_frames_processed = 0
        self.control_frames_processed = 0
        self._active_alarms: list[str] = []
        self._menu_thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._gui_closed = self.headless
        # Retain injections immediately so partial startup also releases them.
        self.camera_hub = camera_hub
        self.safety_engine = safety_engine
        self.alarm_manager = alarm_manager
        self.gesture_engine = gesture_engine
        self.publisher = publisher
        try:
            if self.camera_hub is None:
                self.camera_hub = CameraHub(self.config)
            if self.safety_engine is None or self.alarm_manager is None:
                from safety_engine import AlarmManager, SafetyEngine

                if self.safety_engine is None:
                    self.safety_engine = SafetyEngine(self.config)
                if self.alarm_manager is None:
                    self.alarm_manager = AlarmManager(self.config, muted=muted)
            if self.gesture_engine is None:
                self.gesture_engine = GestureEngine(self.config)
            if self.publisher is None:
                ipc = self.config.get("ipc", {})
                self.publisher = CommandPublisher(ipc.get("host", "127.0.0.1"), ipc.get("port", 5055),
                                                  self.config.get("control", {}).get("command_send_hz", 20))
        except BaseException:
            self.stop()
            raise

    def _require_main_thread(self) -> None:
        if not self.headless and threading.current_thread() is not threading.main_thread():
            raise RuntimeError("OpenCV orchestration must run on the main thread; queue menu actions instead.")

    @staticmethod
    def _close_resource(label: str, resource, method: str) -> None:
        if resource is not None:
            try:
                getattr(resource, method)()
            except Exception as exc:
                print(f"[MASTER] {label} cleanup failed: {exc}")

    def _menu_loop(self) -> None:
        print(MENU_TEXT)
        while not self._stop_event.is_set():
            try:
                raw = input("Select controlling method > ")
            except (EOFError, KeyboardInterrupt):
                return
            if self._stop_event.is_set():
                return
            action = parse_menu_choice(raw)
            if action is None:
                print("[MENU] Choose 1/2/3, 0, c, or q.")
            else:
                self.enqueue_action(action)
                if action[0] == "QUIT":
                    return

    def start(self) -> None:
        self._require_main_thread()
        if self._closed or self._stop_event.is_set():
            raise RuntimeError("Cannot restart a stopped master orchestrator.")
        if self._started:
            return
        self.camera_hub.start()
        self._started = True
        if self.enable_menu_thread:
            self._menu_thread = threading.Thread(target=self._menu_loop, name="av-menu", daemon=True)
            self._menu_thread.start()

    def _terminal_running(self) -> bool:
        if self.game_process is None:
            return False
        try:
            return self.game_process.poll() is None
        except Exception:
            return False

    def _ensure_terminal(self) -> None:
        if not self._terminal_enabled or self._terminal_running():
            return
        ipc = self.config.get("ipc", {})
        try:
            self.game_process = self.terminal_launcher(host=ipc.get("host", "127.0.0.1"),
                                                       port=ipc.get("port", 5055))
        except Exception as exc:
            self.game_process = None
            print(f"[MASTER] Arrow terminal unavailable: {exc}")

    def _send_stop(self, mode: Mode = Mode.NONE, timestamp_sec: float | None = None) -> None:
        try:
            if self.publisher is not None:
                packet = contracts.ControlPacket(time.monotonic() if timestamp_sec is None else timestamp_sec,
                                                 mode.value, contracts.DirectionCommand.STOP.value, 0.0,
                                                 list(self._active_alarms), self.camera_hub.actual_fps)
                self.publisher.publish(packet, force=True)
        except Exception as exc:
            print(f"[MASTER] STOP publication failed: {exc}")

    def _close_control_window(self) -> None:
        if self._control_window_open and not self.headless:
            self._require_main_thread()
            try:
                cv2.destroyWindow(CONTROL_WINDOW)
            except cv2.error:
                pass
        self._control_window_open = False

    def switch_mode(self, mode: Mode | str) -> None:
        self._require_main_thread()
        if self._closed or self._stop_event.is_set():
            return
        selected = mode if isinstance(mode, Mode) else Mode(str(mode).strip().upper())
        if selected == self.active_mode:
            if selected != Mode.NONE:
                self._ensure_terminal()
            return
        old_controller, self.active_controller = self.active_controller, None
        self.active_mode = Mode.NONE
        self._last_face_output = None
        self._last_control_output = None
        # Stop downstream motion before controller shutdown/model construction.
        self._send_stop()
        self._close_resource("controller", old_controller, "close")
        self.gesture_engine.reset()
        if selected == Mode.NONE:
            self._close_control_window()
            print("[MASTER] Standby (Safety DMS running in background)")
            return
        try:
            controller = self.controller_factory(selected, self.config)
            if controller is None:
                raise RuntimeError("Controller factory returned no controller.")
        except Exception as exc:
            self._close_control_window()
            print(f"[MASTER] {selected.value} unavailable; standby keeps Safety DMS active: {exc}")
            return
        self.active_controller = controller
        self.active_mode = selected
        self._ensure_terminal()
        print(f"[MASTER] Active Control Mode: {selected.value} (Safety DMS running in background)")

    def enqueue_action(self, action: tuple[str, str | None]) -> None:
        self._action_queue.put(action)

    def _calibrate_face(self) -> None:
        # Retain the legacy action/method name while calibrating the active mode.
        if self.active_mode != Mode.NONE and self._last_control_output is not None:
            if self.gesture_engine.calibrate(self._last_control_output):
                self._send_stop(self.active_mode)
                print(f"[MASTER] Calibrated neutral center for {self.active_mode.value} mode")
                return
        print("[MASTER] Calibration needs a detected FACE, HAND or EYE in the active mode.")

    def _drain_actions(self) -> None:
        while not self._stop_event.is_set():
            try:
                action, value = self._action_queue.get_nowait()
            except queue.Empty:
                return
            try:
                if action == "SET_MODE":
                    self.switch_mode(value)
                elif action == "CALIBRATE_FACE":
                    self._calibrate_face()
                elif action == "QUIT":
                    self._stop_event.set()
            except (ValueError, TypeError) as exc:
                print(f"[MASTER] Ignoring invalid menu action: {exc}")

    def _poll_hotkeys(self) -> bool:
        if self.headless:
            return not self._stop_event.is_set()
        self._require_main_thread()
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            self._stop_event.set()
        elif key in (ord("1"), ord("2"), ord("3"), ord("0")):
            self.switch_mode({ord("1"): Mode.FACE, ord("2"): Mode.HAND,
                              ord("3"): Mode.EYE, ord("0"): Mode.NONE}[key])
        elif key in (ord("c"), ord("C")):
            self._calibrate_face()
        return not self._stop_event.is_set()

    def _show_control(self, output: ControllerOutput, packet: contracts.ControlPacket, label: str) -> None:
        self._require_main_thread()
        frame = output.annotated_frame
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, max(0, height - 82)), (width - 1, height - 1), (24, 24, 24), -1)
        lines = (f"ACTIVE: {self.active_mode.value} | CMD: {packet.command} | FPS: {packet.fps:.1f}",
                 f"GESTURE: {label}", "[1/2/3] Switch  [C] Calib  [0] Standby  [Q/Esc] Quit")
        for row, text in enumerate(lines):
            cv2.putText(frame, text, (10, max(15, height - 62 + row * 24)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (90, 220, 140), 1, cv2.LINE_AA)
        cv2.imshow(CONTROL_WINDOW, frame)
        self._control_window_open = True

    def step_once(self) -> bool:
        if self._stop_event.is_set():
            return False
        self._require_main_thread()
        self._drain_actions()
        if self._stop_event.is_set():
            return False
        if not self._started:
            self.start()
        pkt = self.camera_hub.get_latest(last_frame_id=self._last_frame_id, timeout=0.05, copy=True)
        if pkt is None:
            return self._poll_hotkeys()
        self._last_frame_id = pkt.frame_id
        safety_res = self.safety_engine.process_frame(pkt.frame, pkt.timestamp_ms,
                                                     draw=not self.headless and self.show_safety_window)
        self.safety_frames_processed += 1
        self._active_alarms = list(safety_res.status.active_alarms)
        self.alarm_manager.trigger(safety_res.status.active_alarms)
        if self.active_controller is not None:
            context = {"face_landmarks_norm": safety_res.landmarks_norm,
                       "face_landmarks_px": safety_res.landmarks_px}
            output = self.active_controller.process_frame(pkt.frame, pkt.timestamp_ms,
                                                         draw=not self.headless, shared_context=context)
            self.control_frames_processed += 1
            self._last_control_output = output
            if self.active_mode == Mode.FACE:
                self._last_face_output = output
            packet, label = self.gesture_engine.build_packet(output, safety_res.status.active_alarms,
                                                             fps=self.camera_hub.actual_fps,
                                                             timestamp_sec=pkt.timestamp_ms / 1000.0)
            self.publisher.publish(packet)
            if not self.headless:
                self._show_control(output, packet, label)
        elif self._terminal_running():
            # Unchanged standby heartbeats retain the publisher's normal rate limit.
            packet = contracts.ControlPacket(pkt.timestamp_ms / 1000.0, Mode.NONE.value,
                                             contracts.DirectionCommand.STOP.value, 0.0,
                                             list(self._active_alarms), self.camera_hub.actual_fps)
            self.publisher.publish(packet)
        if not self.headless and self.show_safety_window:
            frame = safety_res.annotated_frame
            height, width = frame.shape[:2]
            cv2.rectangle(frame, (0, max(0, height - 32)), (width - 1, height - 1), (24, 24, 24), -1)
            cv2.putText(frame, f"CONTROL: {self.active_mode.value} | FPS: {self.camera_hub.actual_fps:.1f}",
                        (10, max(15, height - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 220, 140), 1, cv2.LINE_AA)
            cv2.imshow(SAFETY_WINDOW, frame)
        return self._poll_hotkeys()

    def run(self) -> None:
        try:
            self.start()
            while self.step_once():
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _stop_terminal(self) -> None:
        process, self.game_process = self.game_process, None
        if process is None:
            return
        try:
            if process.poll() is not None:
                return
            process.terminate()
            process.wait(timeout=1.5)
            return
        except Exception as exc:
            print(f"[MASTER] Arrow terminal graceful shutdown failed: {exc}")
        try:
            process.kill()
            process.wait(timeout=0.5)
        except Exception as exc:
            print(f"[MASTER] Arrow terminal cleanup failed: {exc}")

    def _close_windows(self) -> None:
        # A worker may request resource cleanup, but GUI teardown waits for the
        # main thread's stop()/run() finally. Never join a daemon blocked on input.
        if not self._gui_closed and threading.current_thread() is threading.main_thread():
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            self._control_window_open = False
            self._gui_closed = True

    def stop(self) -> None:
        self._stop_event.set()
        if self._closed:
            self._close_windows()
            return
        self._closed = True
        self._send_stop()
        self._stop_terminal()
        controller, self.active_controller = self.active_controller, None
        self._close_resource("controller", controller, "close")
        self._close_resource("safety engine", self.safety_engine, "close")
        self._close_resource("alarm manager", self.alarm_manager, "stop")
        self._close_resource("publisher", self.publisher, "close")
        self._close_resource("camera hub", self.camera_hub, "stop")
        self._last_face_output = None
        self._last_control_output = None
        self.active_mode = Mode.NONE
        self._close_windows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-safety-window", action="store_true", help="Keep safety inference/audio active without its window.")
    parser.add_argument("--mute", action="store_true", help="Mute alarm playback.")
    parser.add_argument("--mode", choices=("face", "hand", "eye"), default=None)
    args = parser.parse_args()
    orchestrator = AVMasterOrchestrator(show_safety_window=not args.no_safety_window, muted=args.mute)
    if args.mode is not None:
        orchestrator.enqueue_action(("SET_MODE", args.mode.upper()))
    orchestrator.run()


if __name__ == "__main__":
    main()
