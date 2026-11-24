"""Phase 1-6 regression, demo preflight and real-model headless stress benchmark."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import traceback
from unittest.mock import Mock, patch

import contracts
from calibrate_demo import (compute_recommended_thresholds, headless_guard,
                            save_thresholds, validate_config, verify_assets)


ROOT = Path(__file__).resolve().parent
SUBPROCESS_TIMEOUT_SEC = 120
FRAME_SCHEDULE = (("NONE", 20), ("FACE", 30), ("HAND", 30), ("EYE", 30), ("NONE", 10))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@dataclass
class SuiteResult:
    name: str
    status: str
    duration_sec: float
    required: bool = True
    detail: str = ""


def verify_calibration_helpers() -> None:
    """Exercise bounds, invalid samples and backup-before-save failure handling."""
    normal = {"alert_ear": [0.28, 0.32], "closed_ear": [0.08, 0.12],
              "neutral_mar": [0.08, 0.12], "yawn_mar": [0.88, 0.92]}
    _require(compute_recommended_thresholds(normal) == {"ear_threshold": 0.2, "mar_threshold": 0.5},
             "Midpoint recommendations are incorrect.")
    _require(compute_recommended_thresholds({}) == {}, "Empty samples should preserve existing thresholds.")
    _require(compute_recommended_thresholds({"alert_ear": [0.3]}) == {}, "An incomplete sample pair must not tune anything.")
    clamped = compute_recommended_thresholds({"closed_ear": 0.2, "alert_ear": 0.8,
                                               "neutral_mar": 0.6, "yawn_mar": 3.0})
    _require(clamped == {"ear_threshold": 0.45, "mar_threshold": 1.2}, "Safety bounds were not applied.")
    for invalid in ({"closed_ear": 0.3, "alert_ear": 0.2}, {"neutral_mar": 0.8, "yawn_mar": 0.5},
                    {"closed_ear": float("nan"), "alert_ear": 0.3}, {"closed_ear": True, "alert_ear": 0.3},
                    {"closed_ear": -0.1, "alert_ear": 0.3}):
        try:
            compute_recommended_thresholds(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe samples were accepted: {invalid!r}")

    config = contracts.load_config()
    _require(not validate_config(config), "; ".join(validate_config(config)))
    assets = verify_assets(config)
    _require(all(assets.values()), f"Missing assets: {[name for name, found in assets.items() if not found]}")
    for section, key, bad in (("safety_dms", "ear_threshold", 0.46), ("safety_dms", "mar_duration_sec", 0),
                              ("safety_dms", "gaze_left_threshold", 0.8), ("ipc", "port", 1023),
                              ("control", "debounce_frames", 0), ("control", "command_send_hz", True)):
        invalid_config = deepcopy(config)
        invalid_config[section][key] = bad
        _require(bool(validate_config(invalid_config)), f"Invalid {section}.{key} passed sanity checks.")

    # All calibration writes happen in the OS temp directory, never the project.
    with tempfile.TemporaryDirectory(prefix="phase7-calibration-") as folder:
        path = Path(folder) / "project_config.json"
        original_bytes = (json.dumps(config, indent=2) + "\n").encode("utf-8")
        path.write_bytes(original_bytes)
        backup = save_thresholds(str(path), {"ear_threshold": 0.21, "mar_threshold": 0.52})
        _require(backup.read_bytes() == original_bytes, "Backup did not preserve the exact original bytes.")
        expected = deepcopy(config)
        expected["safety_dms"].update(ear_threshold=0.21, mar_threshold=0.52)
        _require(contracts.load_config(str(path)) == expected, "Save altered unrelated config keys/values.")
        previous = path.read_bytes()
        save_thresholds(str(path), {"ear_threshold": 0.22})
        _require(backup.read_bytes() == previous, "A repeat save did not back up the immediately previous config.")
        unchanged = path.read_bytes()
        with patch("calibrate_demo.os.replace", side_effect=OSError("Simulated backup write failure")):
            try:
                save_thresholds(str(path), {"ear_threshold": 0.23})
            except OSError:
                pass
            else:
                raise AssertionError("A failed backup still allowed a save.")
        _require(path.read_bytes() == unchanged, "Backup failure changed the live config.")

        real_replace = os.replace

        def fail_config_commit(source, destination):
            if Path(destination) == path:
                raise OSError("Simulated config replacement failure")
            return real_replace(source, destination)

        with patch("calibrate_demo.os.replace", side_effect=fail_config_commit):
            try:
                save_thresholds(str(path), {"ear_threshold": 0.23})
            except OSError:
                pass
            else:
                raise AssertionError("Config replacement failure was not reported.")
        _require(path.read_bytes() == unchanged and backup.read_bytes() == unchanged,
                 "Failed config replacement must leave both original and verified backup intact.")
        _require(set(p.name for p in Path(folder).iterdir()) == {path.name, backup.name},
                 "Calibration left temporary staging files behind.")


def run_subprocess_suite(script_args: list[str], *, legacy: bool = False, verbose: bool = False) -> SuiteResult:
    name = " ".join(script_args)
    command = [sys.executable, *script_args]
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    started = time.perf_counter()
    stdout = stderr = ""
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SEC,
            stdin=subprocess.DEVNULL, cwd=ROOT, env=environment, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        stdout, stderr = completed.stdout, completed.stderr
        passed = completed.returncode == 0
        detail = f"exit={completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        passed, detail = False, f"TIMEOUT after {SUBPROCESS_TIMEOUT_SEC}s"
        stdout, stderr = exc.stdout or "", exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
    except OSError as exc:
        passed, detail = False, f"{type(exc).__name__}: {exc}"
    status = "PASS" if passed else "LEGACY WARNING" if legacy else "FAIL"
    result = SuiteResult(name, status, time.perf_counter() - started, not legacy, detail)
    print(f"[{status}] {name}: {result.duration_sec:.2f}s ({detail})", flush=True)
    if verbose or not passed:
        for label, output in (("stdout", stdout), ("stderr", stderr)):
            if output.strip():
                print(f"  {label}:\n{output.rstrip()}", flush=True)
    else:
        for line in stdout.splitlines():
            if "WEBCAM" in line:
                print("  " + line, flush=True)
    return result


class ProcessMemory:
    """Native Windows working set and OS-maintained lifetime high-water mark."""

    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("This laptop RAM benchmark requires Windows GetProcessMemoryInfo.")

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        self._counters = Counters()
        self._counters.cb = ctypes.sizeof(Counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self._handle = kernel.GetCurrentProcess()
        self._psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self._get_memory = self._psapi.GetProcessMemoryInfo
        self._get_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        self._get_memory.restype = wintypes.BOOL

    def sample(self) -> tuple[float, float]:
        if not self._get_memory(self._handle, ctypes.byref(self._counters), self._counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return self._counters.WorkingSetSize / 1024**2, self._counters.PeakWorkingSetSize / 1024**2


class SyntheticCameraHub:
    """120 unpaced BGR VGA frames; never touches the built-in webcam."""

    def __init__(self):
        import numpy as np

        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.actual_fps = 30.0  # Synthetic timestamp cadence, not measured processing FPS.
        self.frames = 0
        self.started = False
        self.stopped = False
        self.stop_calls = 0

    def start(self):
        _require(not self.stopped, "Cannot restart the stopped synthetic hub.")
        self.started = True

    def get_latest(self, last_frame_id=-1, timeout=0.05, copy=True):
        from camera_hub import FramePacket

        _require(self.started and not self.stopped, "Synthetic camera is not running.")
        if self.frames >= 120:
            return None
        self.frames += 1
        _require(self.frames > last_frame_id, "Frame IDs must increase.")
        return FramePacket(self.frames, round(self.frames * 1000 / 30), self.frame.copy() if copy else self.frame)

    def stop(self):
        if not self.stopped:
            self.stop_calls += 1
            self.stopped, self.started = True, False


class MockTerminalProcess:
    def __init__(self):
        self.returncode = None
        self.terminate_calls = self.kill_calls = self.wait_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout=None):
        self.wait_calls += 1
        _require(self.returncode is not None, "Mock terminal was not terminated before wait().")
        return self.returncode


def _receive(subscriber, timeout: float = 0.5):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        packet = subscriber.poll_latest()
        if packet is not None:
            return packet
        time.sleep(0.001)
    raise AssertionError("No loopback UDP packet arrived before the bounded receive deadline.")


def _latency_metrics(values: list[float]) -> dict[str, float]:
    _require(bool(values) and all(math.isfinite(v) and v >= 0 for v in values), "Invalid latency samples.")
    ordered = sorted(values)

    def percentile(fraction):
        position = fraction * (len(ordered) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)

    return {"mean_ms": statistics.fmean(values), "p50_ms": percentile(0.50),
            "p95_ms": percentile(0.95), "max_ms": ordered[-1]}


def run_sustained_benchmark(config_path: str = "project_config.json") -> dict:
    """Real detector/controller/gesture/UDP pipeline; only camera/terminal mocked."""
    config = contracts.load_config(config_path)
    _require(not validate_config(config), "; ".join(validate_config(config)))
    _require(all(verify_assets(config).values()), "The real-model benchmark requires all runtime model assets.")
    timings = {mode: [] for mode in ("NONE", "FACE", "HAND", "EYE")}
    observed_memory = {mode: 0.0 for mode in timings}
    switches, controllers_seen, udp_modes = [], [], set()
    memory = ProcessMemory()
    baseline_mb, _ = memory.sample()

    with headless_guard():
        import numpy as np

        from av_master import AVMasterOrchestrator
        from controllers import BaseController, create_controller
        from gesture_mapper import GestureEngine
        from ipc_bus import CommandPublisher, CommandSubscriber
        from safety_engine import AlarmManager, SafetyEngine

        with ExitStack() as resources:
            resources.enter_context(patch("arrow_game.launch_arrow_terminal",
                side_effect=AssertionError("Benchmark attempted to launch a real arrow console.")))
            subscriber = resources.enter_context(CommandSubscriber("127.0.0.1", 0))
            runtime_config = deepcopy(config)
            runtime_config["ipc"].update(host="127.0.0.1", port=subscriber.bound_port)
            camera = SyntheticCameraHub()
            resources.callback(camera.stop)
            safety = resources.enter_context(SafetyEngine(runtime_config))
            alarms = AlarmManager(runtime_config, muted=True)
            resources.callback(alarms.stop)
            publisher = resources.enter_context(CommandPublisher("127.0.0.1", subscriber.bound_port,
                runtime_config["control"]["command_send_hz"]))
            terminal = MockTerminalProcess()
            launcher = Mock(return_value=terminal)
            master = AVMasterOrchestrator(config=runtime_config, camera_hub=camera, safety_engine=safety,
                alarm_manager=alarms, gesture_engine=GestureEngine(runtime_config), publisher=publisher,
                controller_factory=create_controller, terminal_launcher=launcher, headless=True,
                enable_menu_thread=False, muted=True, show_safety_window=False)
            resources.callback(master.stop)
            master.start()
            probe = contracts.ControlPacket(0.0, "NONE", "STOP", 0.0, [], 30.0)
            _require(publisher.publish(probe, force=True), "Initial loopback UDP publication failed.")
            _require(_receive(subscriber).command == "STOP", "UDP probe was corrupted.")

            for mode, count in FRAME_SCHEDULE:
                previous_controller = master.active_controller
                began = time.perf_counter()
                master.switch_mode(mode)
                switches.append((mode, (time.perf_counter() - began) * 1000))
                _require(master.active_mode.value == mode, f"{mode} failed to activate; fallback to standby is a release failure.")
                _require(master.safety_engine is safety and not safety._closed, "A mode switch interrupted/replaced safety.")
                if previous_controller is not None:
                    _require(getattr(previous_controller, "_closed", True), "Previous controller remained open after a mode switch.")
                if mode != "NONE":
                    _require(isinstance(master.active_controller, BaseController), "Production controller factory was bypassed.")
                    controllers_seen.append(master.active_controller)
                observed_memory[mode] = max(observed_memory[mode], memory.sample()[0])

                for _ in range(count):
                    before_safety, before_control = master.safety_frames_processed, master.control_frames_processed
                    began = time.perf_counter()
                    stepped = master.step_once()
                    timings[mode].append((time.perf_counter() - began) * 1000)
                    _require(stepped, "Master stopped before the 120-frame schedule completed.")
                    _require(master.safety_frames_processed == before_safety + 1, "Safety missed or duplicated a delivered frame.")
                    _require(master.control_frames_processed == before_control + (mode != "NONE"),
                             "Exclusive control processed an incorrect number of frames.")
                    _require(master.active_mode.value == mode, "Active mode changed unexpectedly during processing.")
                    observed_memory[mode] = max(observed_memory[mode], memory.sample()[0])
                    packet = subscriber.poll_latest()
                    if packet is not None:
                        udp_modes.add(packet.mode)
                        _require(packet.command == "STOP" and packet.active_alarms == [],
                                 "A blank frame generated unexpected movement or an alarm.")

            _require(master.safety_frames_processed == camera.frames == 120,
                     f"Expected 120 safety frames; got {master.safety_frames_processed}/{camera.frames}.")
            _require(master.control_frames_processed == 90, "Expected exactly 90 exclusive-controller frames.")
            _require(all(len(timings[mode]) == 30 for mode in timings), "Per-mode frame distribution is incorrect.")
            _require(np.count_nonzero(camera.frame) == 0, "The pipeline modified the synthetic source frame.")
            _require({"FACE", "HAND", "EYE"}.issubset(udp_modes), f"Missing controller packets on real loopback UDP: {udp_modes}")
            launcher.assert_called_once_with(host="127.0.0.1", port=subscriber.bound_port)
            _require(master._menu_thread is None and alarms._thread is None, "Headless benchmark started menu/audio threads.")

            stop_output = io.StringIO()
            began = time.perf_counter()
            with redirect_stdout(stop_output):
                master.stop()
            stop_ms = (time.perf_counter() - began) * 1000
            _require("failed" not in stop_output.getvalue().lower(), "Master cleanup reported errors: " + stop_output.getvalue())
            final_packet = _receive(subscriber)
            _require(final_packet.mode == "NONE" and final_packet.command == "STOP", "Shutdown did not deliver final NONE/STOP.")
            _require(camera.stopped and camera.stop_calls == 1 and safety._closed, "Camera/safety cleanup is incomplete.")
            _require(alarms._stop_event.is_set() and publisher._socket is None, "Alarm/publisher cleanup is incomplete.")
            _require(all(getattr(controller, "_closed", True) for controller in controllers_seen), "A controller leaked on shutdown.")
            _require(terminal.terminate_calls == terminal.wait_calls == 1 and terminal.kill_calls == 0,
                     "Mock terminal did not shut down gracefully.")
            _require(master.active_controller is None and master.game_process is None and master._closed,
                     "Master retained an active controller or terminal after stop().")
            master.stop()
            _require(not master.step_once() and camera.stop_calls == 1, "Stopped master resumed or repeated resource cleanup.")
            _, peak_mb = memory.sample()
            safety_frames, control_frames = master.safety_frames_processed, master.control_frames_processed
        _require(subscriber._socket is None, "UDP subscriber was not closed.")

    all_timings = [value for values in timings.values() for value in values]
    return {"modes": {mode: {"frames": len(values), **_latency_metrics(values),
                             "peak_sampled_ws_mb": observed_memory[mode]} for mode, values in timings.items()},
            "overall": _latency_metrics(all_timings), "switches_ms": switches, "stop_ms": stop_ms,
            "baseline_ws_mb": baseline_mb, "peak_process_ws_mb": peak_mb,
            "safety_frames": safety_frames, "control_frames": control_frames}


def print_benchmark(report: dict) -> None:
    print("\n120-frame real-model benchmark: 20 NONE -> 30 FACE -> 30 HAND -> 30 EYE -> 10 NONE")
    print(f"{'MODE':<8} {'FRAMES':>6} {'MEAN ms':>10} {'p50 ms':>10} {'p95 ms':>10} {'MAX ms':>10} {'PEAK sampled WS MB':>20}")
    print("-" * 84)
    for mode, metrics in report["modes"].items():
        print(f"{mode:<8} {metrics['frames']:>6} {metrics['mean_ms']:>10.2f} {metrics['p50_ms']:>10.2f} "
              f"{metrics['p95_ms']:>10.2f} {metrics['max_ms']:>10.2f} {metrics['peak_sampled_ws_mb']:>20.2f}")
    total = report["overall"]
    print(f"{'ALL':<8} {120:>6} {total['mean_ms']:>10.2f} {total['p50_ms']:>10.2f} {total['p95_ms']:>10.2f} "
          f"{total['max_ms']:>10.2f} {report['peak_process_ws_mb']:>20.2f}")
    print("Model-switch/startup costs, measured separately: " +
          " -> ".join(f"{mode} {latency:.2f}ms" for mode, latency in report["switches_ms"]))
    print(f"Clean stop: {report['stop_ms']:.2f}ms; safety={report['safety_frames']}/120; control={report['control_frames']}/90")
    print(f"Runner baseline WS={report['baseline_ws_mb']:.2f} MB; OS lifetime peak WS={report['peak_process_ws_mb']:.2f} MB")
    print("RAM: Windows GetProcessMemoryInfo, bytes/1024^2; process peak includes startup/imports, excludes test subprocesses.")
    print("Latency: step_once() wall time; receiver polling/RAM sampling and mode switches are outside the frame timer.")
    print("Workload: unpaced synthetic black VGA frames, real detectors/models; live landmark/gesture, capture, GUI/audio costs excluded.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Print captured output for passing subprocess suites too.")
    args = parser.parse_args(argv)
    results = []
    started = time.perf_counter()
    print(f"PHASE 7 | Python {sys.version.split()[0]} | {sys.executable}", flush=True)
    print("Config/assets and calibration helper verification", flush=True)
    began = time.perf_counter()
    try:
        verify_calibration_helpers()
        helper_result = SuiteResult("Config/assets/calibration helpers", "PASS", time.perf_counter() - began)
    except Exception as exc:
        helper_result = SuiteResult("Config/assets/calibration helpers", "FAIL", time.perf_counter() - began,
                                    detail=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    results.append(helper_result)
    print(f"[{helper_result.status}] {helper_result.name}: {helper_result.duration_sec:.2f}s {helper_result.detail}", flush=True)

    print("\nIsolated regressions (120s timeout each; no input or new console)", flush=True)
    for script in ("test_phase1_infra.py", "test_phase2_safety.py", "test_phase3_controllers.py",
                   "test_phase4_game.py", "test_phase5_gestures.py", "test_phase6_master.py"):
        results.append(run_subprocess_suite([script], verbose=args.verbose))
    results.append(run_subprocess_suite(["calibrate_demo.py", "--preflight"], verbose=args.verbose))
    results.append(run_subprocess_suite(["-m", "unittest", "test_safety_daemon.py"], legacy=True, verbose=args.verbose))

    print("\nSustained headless integration and latency/RAM benchmark", flush=True)
    began = time.perf_counter()
    try:
        report = run_sustained_benchmark()
        benchmark_result = SuiteResult("120-frame real-model/UDP stress", "PASS", time.perf_counter() - began)
        print_benchmark(report)
    except Exception as exc:
        benchmark_result = SuiteResult("120-frame real-model/UDP stress", "FAIL", time.perf_counter() - began,
                                       detail=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    results.append(benchmark_result)

    print(f"\n{'SUITE':<46} {'STATUS':<15} {'SECONDS':>8}")
    print("-" * 71)
    for result in results:
        print(f"{result.name:<46} {result.status:<15} {result.duration_sec:>8.2f}")
        if result.detail and result.status != "PASS":
            print("  " + result.detail)
    print(f"Total elapsed: {time.perf_counter() - started:.2f}s")
    required_failures = [result for result in results if result.required and result.status != "PASS"]
    legacy_warnings = [result for result in results if result.status == "LEGACY WARNING"]
    if legacy_warnings:
        print("LEGACY WARNING: existing legacy unittest failures are reported above and do not block the release.")
    if required_failures:
        print("PHASE 7 FAILED: " + ", ".join(result.name for result in required_failures))
        return 1
    print("PHASE 7 VERIFIED OK  SYSTEM READY FOR DEMO DAY")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Verification interrupted.", file=sys.stderr)
        raise SystemExit(130)
