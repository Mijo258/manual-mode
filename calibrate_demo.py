"""Headless demo preflight, or opt-in webcam lighting calibration."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from collections import deque
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import importlib
import ipaddress
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import textwrap
import time
from unittest.mock import patch

import contracts


ROOT = Path(__file__).resolve().parent
REQUIRED_SECTIONS = ("camera", "safety_dms", "ipc", "control", "assets", "gestures")
GESTURE_KEYS = {
    "face": ("ema_alpha", "yaw_deadzone", "pitch_up_deadzone", "pitch_down_deadzone"),
    "eye": ("ema_alpha", "wink_closed_ear", "wink_open_ear", "wink_ear_diff",
            "wide_ear_threshold", "blink_ear_threshold", "max_blink_duration_sec",
            "double_blink_window_sec", "double_blink_pulse_sec", "gaze_left_x",
            "gaze_right_x", "gaze_up_y", "gaze_down_y"),
}
TUNABLE_SAFETY_KEYS = ("ear_threshold", "mar_threshold", "gaze_left_threshold", "gaze_right_threshold")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             "Expected a JSON number, not a boolean or string.")
    result = float(value)
    _require(math.isfinite(result), "Numbers must be finite.")
    return result


@contextmanager
def headless_guard():
    """Forbid webcam/GUI/input and hide any Windows stdlib platform probe."""
    original_popen = subprocess.Popen

    def hidden_process(*args, **kwargs):
        flags = kwargs.get("creationflags", 0)
        if os.name == "nt":
            _require(not flags & subprocess.CREATE_NEW_CONSOLE,
                     "Headless verification requested a new console.")
            kwargs["creationflags"] = flags | subprocess.CREATE_NO_WINDOW
        return original_popen(*args, **kwargs)

    with ExitStack() as guards:
        guards.enter_context(patch("builtins.input", side_effect=AssertionError("Headless input call.")))
        guards.enter_context(patch.object(subprocess, "Popen", side_effect=hidden_process))
        try:
            cv2 = importlib.import_module("cv2")
        except Exception:
            cv2 = None  # run_preflight reports the actual import failure below.
        if cv2 is not None:
            for name in ("VideoCapture", "imshow", "namedWindow", "waitKey", "waitKeyEx",
                         "destroyWindow", "destroyAllWindows", "getWindowProperty"):
                guards.enter_context(patch.object(cv2, name,
                    side_effect=AssertionError(f"Headless cv2.{name} call.")))
        yield


def _config_checks(config: dict) -> list[tuple[str, bool, str]]:
    rows = []

    def check(label, operation):
        try:
            detail = operation()
            rows.append((label, True, detail))
        except Exception as exc:
            rows.append((label, False, f"{type(exc).__name__}: {exc}"))

    missing = [name for name in REQUIRED_SECTIONS if not isinstance(config.get(name), dict)]
    rows.append(("Required config sections", not missing,
                 "Missing/non-object: " + ", ".join(missing) if missing else ", ".join(REQUIRED_SECTIONS)))
    if missing:
        return rows
    camera, dms, ipc, control = (config[name] for name in REQUIRED_SECTIONS[:4])

    def camera_check():
        for key in ("index", "width", "height"):
            value = camera[key]
            _require(isinstance(value, int) and not isinstance(value, bool), f"camera.{key} must be an integer.")
            _require(value >= (0 if key == "index" else 1), f"Invalid camera.{key}.")
        _require(camera["backend"] in ("DSHOW", "MSMF", "ANY"), "Camera backend must be DSHOW/MSMF/ANY.")
        _require(_number(camera["fps"]) > 0, "camera.fps must be positive.")
        _require(isinstance(camera["flip_horizontal"], bool), "camera.flip_horizontal must be boolean.")
        return (f"index={camera['index']} {camera['backend']} {camera['width']}x{camera['height']} "
                f"@ {camera['fps']} FPS; flip={camera['flip_horizontal']}")

    def threshold_check(prefix, low, high):
        threshold, duration = _number(dms[f"{prefix}_threshold"]), _number(dms[f"{prefix}_duration_sec"])
        _require(low <= threshold <= high and duration > 0,
                 f"{low:.2f} <= {prefix}_threshold <= {high:.2f}; {prefix}_duration_sec > 0 required.")
        return f"threshold={threshold:g}; duration={duration:g}s"

    def gaze_check():
        left, right, duration = (_number(dms[key]) for key in
            ("gaze_left_threshold", "gaze_right_threshold", "gaze_duration_sec"))
        _require(0 < left < right < 1 and duration > 0,
                 "0 < gaze_left_threshold < gaze_right_threshold < 1; gaze_duration_sec > 0 required.")
        return f"left={left:g}; right={right:g}; duration={duration:g}s"

    def ipc_check():
        port, host = ipc["port"], ipc["host"]
        _require(isinstance(port, int) and not isinstance(port, bool) and 1024 <= port <= 65535,
                 "ipc.port must be an integer in [1024, 65535].")
        _require(isinstance(host, str), "ipc.host must be a string.")
        if host != "localhost":
            address = ipaddress.ip_address(host)
            _require(address.version == 4 and address.is_loopback, "IPC requires IPv4 loopback or localhost.")
        return f"{host}:{port} (IPv4 loopback)"

    def control_check():
        debounce = control["debounce_frames"]
        _require(isinstance(debounce, int) and not isinstance(debounce, bool) and debounce >= 1,
                 "control.debounce_frames must be an integer >= 1.")
        _require(_number(control["command_send_hz"]) > 0, "control.command_send_hz must be positive.")
        return f"debounce={debounce} frames; send={control['command_send_hz']} Hz"

    def gesture_check():
        from gesture_mapper import EyeGestureMapper, FaceGestureMapper

        for section, keys in GESTURE_KEYS.items():
            settings = config["gestures"].get(section)
            _require(isinstance(settings, dict), f"gestures.{section} must be an object.")
            _require(set(settings) == set(keys), f"gestures.{section} must contain exactly: {', '.join(keys)}")
            for key in keys:
                _number(settings[key])
        FaceGestureMapper(config)
        EyeGestureMapper(config)
        return "Face: 4 tunables; Eye: 13 tunables; both production mappers accept config"

    check("Camera settings", camera_check)
    check("Safety EAR", lambda: threshold_check("ear", 0.05, 0.45))
    check("Safety MAR", lambda: threshold_check("mar", 0.20, 1.20))
    check("Safety gaze", gaze_check)
    check("Alarm cooldown", lambda: (
        _require(_number(dms["alarm_cooldown_sec"]) >= 0, "Alarm cooldown must be nonnegative.")
        or f"{dms['alarm_cooldown_sec']}s"))
    check("IPC settings", ipc_check)
    check("Control settings", control_check)
    check("Gesture settings", gesture_check)
    return rows


def validate_config(config: dict) -> list[str]:
    """Return actionable sanity errors; use the production gesture validators."""
    if not isinstance(config, dict):
        return ["Project configuration must contain a JSON object."]
    return [f"{label}: {detail}" for label, passed, detail in _config_checks(config) if not passed]


def _asset_paths(config: dict) -> dict[str, Path | None]:
    face_candidates = (ROOT / "face_landmarker.task", ROOT / "face-landmark-project" / "face_landmarker.task")
    face = next((p for p in face_candidates if p.is_file()), face_candidates[0])
    assets = config.get("assets", {})
    configured = assets.get("dlib_shape_predictor_68") if isinstance(assets, dict) else None
    predictor = None
    if isinstance(configured, str) and configured.strip():
        predictor = Path(configured).expanduser()
        if not predictor.is_absolute():
            predictor = ROOT / predictor
    return {"face_landmarker.task": face, "hand_landmarker.task": ROOT / "hand_landmarker.task",
            "shape_predictor_68_face_landmarks.dat": predictor}


def verify_assets(config: dict) -> dict[str, bool]:
    """Check the runtime root paths, including the predictor specified in JSON."""
    return {name: path is not None and path.is_file() for name, path in _asset_paths(config).items()}


def _print_table(rows: list[tuple[str, bool, str]]) -> None:
    width = max(len(label) for label, _, _ in rows)
    print(f"{'CHECK':<{width}} {'STATUS':<6} DETAILS")
    print("-" * (width + 70))
    for label, passed, detail in rows:
        lines = textwrap.wrap(str(detail), width=78) or [""]
        print(f"{label:<{width}} {'PASS' if passed else 'FAIL':<6} {lines[0]}")
        for line in lines[1:]:
            print(" " * (width + 8) + line)


def run_preflight(config_path: str = "project_config.json") -> tuple[bool, list[str]]:
    """Read-only, no-camera checks, including real model construction/cleanup."""
    rows = [("Active Python", sys.version_info >= (3, 11), f"{sys.version.split()[0]} | {sys.executable}")]
    config = None
    with headless_guard():
        for module_name, label in (("cv2", "OpenCV import/version"), ("mediapipe", "MediaPipe import/version"),
                                   ("dlib", "dlib import/version")):
            try:
                module = importlib.import_module(module_name)
                rows.append((label, True, str(getattr(module, "__version__", "version unavailable"))))
            except Exception as exc:
                rows.append((label, False, f"{type(exc).__name__}: {exc}"))
        try:
            config = contracts.load_config(config_path)
            rows.append(("contracts.load_config", True, str(config_path)))
            rows.extend(_config_checks(config))
            for name, path in _asset_paths(config).items():
                rows.append(("Asset " + name, path is not None and path.is_file(),
                             str(path) if path is not None else "assets.dlib_shape_predictor_68 is missing/invalid"))
        except Exception as exc:
            rows.append(("Config/assets", False, f"{type(exc).__name__}: {exc}"))

        for label, module_name in (("SafetyEngine", "safety_engine"), ("HandController", "controllers"),
                                   ("EyeController", "controllers")):
            resource = None
            try:
                _require(config is not None and not validate_config(config), "A sane config is required for model loading.")
                constructor = getattr(importlib.import_module(module_name), label)
                resource = constructor(config=config)
                resource.close()
                resource.close()  # Exercise idempotent cleanup without a camera.
                rows.append((label + " model load/close", True, "Real model loaded; close() completed cleanly"))
            except Exception as exc:
                rows.append((label + " model load/close", False, f"{type(exc).__name__}: {exc}"))
            finally:
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        rows.append((label + " cleanup", False, str(exc)))
                resource = None

        try:
            import socket

            _require(config is not None and not validate_config(config), "A sane config is required for UDP binding.")
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind((config["ipc"]["host"], 0))
                rows.append(("Localhost UDP bind", True, f"{sock.getsockname()} (ephemeral port; socket closed)"))
        except Exception as exc:
            rows.append(("Localhost UDP bind", False, f"{type(exc).__name__}: {exc}"))
    _print_table(rows)
    all_passed = all(passed for _, passed, _ in rows)
    print("PREFLIGHT PASSED" if all_passed else "PREFLIGHT FAILED")
    return all_passed, [f"{'PASS' if passed else 'FAIL'} | {label} | {detail}" for label, passed, detail in rows]


def compute_recommended_thresholds(samples: dict) -> dict:
    """Pure midpoint calculation; absent pairs leave existing thresholds alone."""
    _require(isinstance(samples, dict), "Samples must be a dictionary.")

    def mean(key):
        values = samples.get(key)
        if values is None:
            return None
        if isinstance(values, (int, float)):
            values = [values]
        _require(isinstance(values, (list, tuple)), f"{key} must be a number or list of numbers.")
        if not values:
            return None
        numbers = [_number(value) for value in values]
        _require(all(value >= 0 for value in numbers), f"{key} samples must be nonnegative.")
        return statistics.fmean(numbers)

    result = {}
    for low_key, high_key, setting, bounds in (
            ("closed_ear", "alert_ear", "ear_threshold", (0.05, 0.45)),
            ("neutral_mar", "yawn_mar", "mar_threshold", (0.20, 1.20))):
        low, high = mean(low_key), mean(high_key)
        if low is None or high is None:
            continue
        _require(low < high, f"Sample again: {high_key} must exceed {low_key}.")
        threshold = round(max(bounds[0], min(bounds[1], (low + high) / 2)), 4)
        _require(low < threshold < high, f"{setting} samples are outside usable safety bounds; sample again.")
        result[setting] = threshold
    return result


def save_thresholds(config_path: str, thresholds: dict) -> Path:
    """Atomically save only DMS thresholds after an exact-byte .json.bak backup."""
    _require(isinstance(thresholds, dict) and bool(thresholds), "No thresholds to save.")
    _require(set(thresholds).issubset(TUNABLE_SAFETY_KEYS), "Only EAR/MAR/gaze DMS thresholds may be saved.")
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    backup = path.with_name(path.name + ".bak")
    _require(not path.is_symlink() and not backup.is_symlink(), "Config/backup must be regular files, not symbolic links.")
    original_bytes = path.read_bytes()
    original = json.loads(original_bytes.decode("utf-8"))
    updated = deepcopy(original)
    updated["safety_dms"].update({key: _number(value) for key, value in thresholds.items()})
    errors = validate_config(updated)
    _require(not errors, "; ".join(errors))
    payload = (json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    pending = []

    def stage(data):
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".phase7-", suffix=".tmp", delete=False) as f:
            pending.append(Path(f.name))
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return pending[-1]

    try:
        backup_temp, config_temp = stage(original_bytes), stage(payload)
        _require(path.read_bytes() == original_bytes, "Config changed during calibration; nothing was saved.")
        os.replace(backup_temp, backup)  # The backup is committed before any config replacement.
        _require(backup.read_bytes() == original_bytes, "Backup verification failed; config was not replaced.")
        _require(path.read_bytes() == original_bytes, "Config changed during backup; nothing was saved.")
        os.replace(config_temp, path)
    finally:
        for temporary in pending:
            temporary.unlink(missing_ok=True)
    return backup


class RollingStats:
    """Last 90 valid measurements, with no fabricated zeroes for tracking loss."""

    def __init__(self, window: int = 90):
        self.values = {key: deque(maxlen=window) for key in ("EAR", "MAR", "GAZE", "YAW", "PITCH")}

    def add(self, measurements: dict) -> None:
        for key, values in self.values.items():
            value = measurements.get(key)
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))

    def reset(self, keys=None) -> None:
        for key in self.values if keys is None else keys:
            self.values[key].clear()

    def describe(self, key: str) -> str:
        values = self.values[key]
        if not values:
            return "min/max/avg: -- / -- / -- (0 frames)"
        return f"min/max/avg: {min(values):.3f} / {max(values):.3f} / {statistics.fmean(values):.3f} ({len(values)}f)"


def run_live_calibration(config_path: str = "project_config.json") -> None:
    """Opt-in live HUD: one SafetyEngine inference shared with FaceController."""
    import cv2
    import numpy as np

    from camera_hub import CameraHub
    from controllers import FaceController
    from gesture_mapper import FaceGestureMapper
    from safety_engine import SafetyEngine

    config = contracts.load_config(config_path)
    errors = validate_config(config)
    _require(not errors, "; ".join(errors))
    _require(all(verify_assets(config).values()), "Required model assets are missing; run --preflight.")
    dms = config["safety_dms"]
    stats, samples = RollingStats(), {}
    mapper = FaceGestureMapper(config)
    neutral_ready = False
    recording, recorded = None, []
    message = "Face neutral is set on first detection; [c] recenters. [s] saves DMS only."
    window = "Phase 7 - Demo Lighting Calibration"

    def close_window():
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass

    with ExitStack() as resources:
        safety = SafetyEngine(config)
        resources.callback(safety.close)
        face = FaceController(config)
        resources.callback(face.close)
        camera = CameraHub(config)
        resources.callback(camera.stop)
        camera.start()
        resources.callback(close_window)
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        last_id, last_arrival = -1, time.monotonic()
        preview = np.zeros((480, 640, 3), dtype=np.uint8)
        measurements = {key: None for key in stats.values}
        latest_output = None
        detected = False
        while True:
            packet = camera.get_latest(last_id, timeout=0.1, copy=True)
            if packet is not None:
                last_id, last_arrival = packet.frame_id, time.monotonic()
                result = safety.process_frame(packet.frame, packet.timestamp_ms, draw=False)
                latest_output = face.process_frame(packet.frame, packet.timestamp_ms, draw=False,
                    shared_context={"face_landmarks_norm": result.landmarks_norm,
                                    "face_landmarks_px": result.landmarks_px})
                detected = result.status.face_detected and latest_output.detected
                preview = packet.frame
                measurements = {key: None for key in stats.values}
                if detected:
                    features = latest_output.features
                    if not neutral_ready:
                        neutral_ready = mapper.calibrate(features)
                    raw = {"EAR": result.status.ear, "MAR": result.status.mar, "GAZE": result.status.gaze_ratio,
                           "YAW": features["yaw_offset"], "PITCH": features["pitch_offset"]}
                    measurements = {**raw, "YAW": raw["YAW"] - mapper.baseline_yaw,
                                    "PITCH": raw["PITCH"] - mapper.baseline_pitch}
                    stats.add(measurements)
                    if recording is not None:
                        recorded.append(raw)
                        message = f"Recording {recording}: {len(recorded)}/15 valid frames"
                        if len(recorded) == 15:
                            if recording == "neutral":
                                samples["alert_ear"] = [v["EAR"] for v in recorded]
                                samples["neutral_mar"] = [v["MAR"] for v in recorded]
                                samples["center_gaze"] = [v["GAZE"] for v in recorded]
                                mapper.calibrate({"yaw_offset": statistics.fmean(v["YAW"] for v in recorded),
                                                  "pitch_offset": statistics.fmean(v["PITCH"] for v in recorded)})
                                stats.reset(("YAW", "PITCH"))
                            else:
                                samples[recording] = [v["EAR" if recording == "closed_ear" else "MAR"] for v in recorded]
                            message = f"Captured {recording}; [s] computes sampled midpoints and saves."
                            recording, recorded = None, []
            elif time.monotonic() - last_arrival > 5:
                raise RuntimeError(camera.last_error or "No fresh webcam frames for 5 seconds; check camera access.")

            height, width = preview.shape[:2]
            hud = np.full((max(height, 490), width + 550, 3), 20, dtype=np.uint8)
            hud[:height, :width] = preview
            cv2.putText(hud, "FACE DETECTED" if detected else "NO FACE / WAITING FOR CAMERA",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 230, 150) if detected else (80, 180, 255), 1)
            markers = {"EAR": f"threshold {dms['ear_threshold']:.2f}", "MAR": f"threshold {dms['mar_threshold']:.2f}",
                       "GAZE": f"limits {dms['gaze_left_threshold']:.2f} / {dms['gaze_right_threshold']:.2f}",
                       "YAW": f"relative to neutral; deadzone {mapper.yaw_deadzone:.2f}",
                       "PITCH": f"relative; up/down {mapper.pitch_up_deadzone:.2f}/{mapper.pitch_down_deadzone:.2f}"}
            lines = [("LIVE METRICS - rolling last 90 detected frames", 26)]
            for index, key in enumerate(stats.values):
                value = "--" if measurements[key] is None else f"{measurements[key]:+.3f}"
                lines.extend(((f"{key}: {value} | {markers[key]}", 62 + index * 52),
                              (stats.describe(key), 82 + index * 52)))
            lines.extend((("[1/2] EAR -/+ .01  [3/4] MAR -/+ .02", 336),
                          ("[5/6] Gaze left -/+ .02  [7/8] Right -/+ .02", 357),
                          ("[b] Neutral  [e] Closed eyes  [y] Yawn (15f each)", 378),
                          ("[r] Reset stats/samples  [c] Face center", 399),
                          ("[s] Backup + save  [q/Esc] Quit", 420)))
            lines.extend((line, 450 + index * 19) for index, line in enumerate(textwrap.wrap(message, 65)[:2]))
            for line, y in lines:
                cv2.putText(hud, line, (width + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (210, 225, 215), 1, cv2.LINE_AA)
            cv2.imshow(window, hud)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("r"), ord("R")):
                stats.reset()
                samples.clear()
                recording, recorded = None, []
                message = "Rolling stats and recorded samples reset."
            elif key in (ord("c"), ord("C")):
                if detected and mapper.calibrate(latest_output.features):
                    stats.reset(("YAW", "PITCH"))
                    neutral_ready = True
                    message = "Face neutral recentered for this calibration session."
                else:
                    message = "Face centering requires a detected face."
            elif key in (ord("b"), ord("e"), ord("y")):
                recording = {ord("b"): "neutral", ord("e"): "closed_ear", ord("y"): "yawn_mar"}[key]
                recorded = []
                message = f"Sampling {recording}: hold the pose for 15 detected frames."
            elif key in map(ord, "12345678"):
                setting, delta = {ord("1"): ("ear_threshold", -0.01), ord("2"): ("ear_threshold", 0.01),
                    ord("3"): ("mar_threshold", -0.02), ord("4"): ("mar_threshold", 0.02),
                    ord("5"): ("gaze_left_threshold", -0.02), ord("6"): ("gaze_left_threshold", 0.02),
                    ord("7"): ("gaze_right_threshold", -0.02), ord("8"): ("gaze_right_threshold", 0.02)}[key]
                bounds = {"ear_threshold": (0.05, 0.45), "mar_threshold": (0.20, 1.20),
                          "gaze_left_threshold": (0.01, dms["gaze_right_threshold"] - 0.01),
                          "gaze_right_threshold": (dms["gaze_left_threshold"] + 0.01, 0.99)}[setting]
                dms[setting] = round(max(bounds[0], min(bounds[1], dms[setting] + delta)), 4)
                setattr(safety, setting, dms[setting])
                safety.reset_timers()
                if setting == "ear_threshold":
                    samples.pop("closed_ear", None)
                    if recording == "closed_ear":
                        recording, recorded = None, []
                elif setting == "mar_threshold":
                    samples.pop("yawn_mar", None)
                    if recording == "yawn_mar":
                        recording, recorded = None, []
                message = f"Pending {setting}={dms[setting]:.4f}; [s] persists it."
            elif key in (ord("s"), ord("S")):
                try:
                    tuned = {name: dms[name] for name in TUNABLE_SAFETY_KEYS}
                    tuned.update(compute_recommended_thresholds(samples))
                    backup = save_thresholds(config_path, tuned)
                    dms.update(tuned)
                    for name, value in tuned.items():
                        setattr(safety, name, value)
                    safety.reset_timers()
                    message = f"Saved {config_path}; backup {backup.name} (previous config)."
                    print(message, flush=True)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    message = f"SAVE FAILED: {exc}"
                    print(message, file=sys.stderr, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", "--show", action="store_true", help="Headless imports/config/assets/model/UDP checks (default).")
    modes.add_argument("--calibrate", "--live", action="store_true", help="Opt-in live webcam HUD and backup-first threshold saving.")
    parser.add_argument("--config", default="project_config.json", help="Config path; relative paths resolve beside contracts.py.")
    args = parser.parse_args(argv)
    try:
        if args.calibrate:
            run_live_calibration(args.config)
            return 0
        return 0 if run_preflight(args.config)[0] else 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"CALIBRATION ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
