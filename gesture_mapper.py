"""Gesture rules for existing controller features; owns no camera or GUI resources.

Face offsets and pupil ratios follow the already-flipped image coordinates:
negative yaw / pitch means image LEFT / UP. Eye names follow EyeController:
left_ear is dlib 36..41, right_ear is 42..47; no anatomical relabeling is done.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import contracts

if TYPE_CHECKING:
    from controllers.base import ControllerOutput

Direction = contracts.DirectionCommand


def _section(config: dict | None, name: str) -> dict:
    cfg = contracts.load_config() if config is None else config
    return cfg.get("gestures", {}).get(name, {})


def _setting(section: dict, key: str, default: float, low: float = 0.0,
             high: float = math.inf) -> float:
    value = float(section.get(key, default))
    if not math.isfinite(value) or not low < value <= high:
        raise ValueError(f"gestures setting {key} must be finite and in ({low}, {high}].")
    return value


def _number(value) -> float:
    if isinstance(value, bool):
        raise ValueError("A geometric metric must be numeric, not boolean.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Geometric metrics must be finite.")
    return result


def _confidence(output: ControllerOutput) -> float:
    try:
        return max(0.0, min(1.0, _number(output.confidence)))
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _timestamp(timestamp_sec: float | None) -> float:
    return time.monotonic() if timestamp_sec is None else _number(timestamp_sec)


class CommandDebouncer:
    def __init__(self, debounce_frames: int | None = None, lost_frames_to_stop: int = 3):
        if debounce_frames is None:
            debounce_frames = contracts.load_config().get("control", {}).get("debounce_frames", 3)
        self.debounce_frames = max(1, int(debounce_frames))
        self.lost_frames_to_stop = max(1, int(lost_frames_to_stop))
        self.reset()

    def reset(self) -> None:
        self.stable_command = Direction.STOP
        self._candidate: Direction | None = None
        self._candidate_frames = 0
        self._lost_frames = 0

    def _accept(self, command: Direction) -> Direction:
        """Commit a confirmed command, also used for a completed double blink."""
        self.stable_command = command
        self._candidate = command
        self._candidate_frames = self.debounce_frames
        self._lost_frames = 0
        return command

    def update(self, raw_command: Direction, detected: bool) -> Direction:
        if not detected:
            self._lost_frames += 1
            # A lost frame breaks the required consecutive detections.
            self._candidate = None
            self._candidate_frames = 0
            if self._lost_frames >= self.lost_frames_to_stop:
                self.stable_command = Direction.STOP
            return self.stable_command
        self._lost_frames = 0
        raw_command = Direction(raw_command)
        if raw_command != self._candidate:
            self._candidate = raw_command
            self._candidate_frames = 1
        else:
            self._candidate_frames = min(self.debounce_frames, self._candidate_frames + 1)
        if self._candidate_frames >= self.debounce_frames:
            return self._accept(raw_command)
        return self.stable_command


class FaceGestureMapper:
    def __init__(self, config: dict | None = None):
        settings = _section(config, "face")
        self.ema_alpha = _setting(settings, "ema_alpha", 0.45, high=1.0)
        self.yaw_deadzone = _setting(settings, "yaw_deadzone", 0.09)
        self.pitch_up_deadzone = _setting(settings, "pitch_up_deadzone", 0.07)
        self.pitch_down_deadzone = _setting(settings, "pitch_down_deadzone", 0.08)
        self.baseline_yaw = self.baseline_pitch = 0.0
        self.reset()

    def calibrate(self, features: dict) -> bool:
        try:
            yaw = _number(features["yaw_offset"])
            pitch = _number(features["pitch_offset"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        self.baseline_yaw, self.baseline_pitch = yaw, pitch
        self._ema_yaw, self._ema_pitch = yaw, pitch
        return True

    def reset(self, reset_baseline: bool = False) -> None:
        self._ema_yaw: float | None = None
        self._ema_pitch: float | None = None
        if reset_baseline:
            self.baseline_yaw = self.baseline_pitch = 0.0

    def map(self, output: ControllerOutput) -> tuple[Direction, float, str]:
        if not output.detected:
            self.reset()
            return Direction.STOP, 0.0, "NO FACE"
        try:
            yaw = _number(output.features["yaw_offset"])
            pitch = _number(output.features["pitch_offset"])
        except (KeyError, TypeError, ValueError, OverflowError):
            self.reset()
            return Direction.STOP, 0.0, "INVALID FACE FEATURES"
        alpha = self.ema_alpha
        self._ema_yaw = yaw if self._ema_yaw is None else alpha * yaw + (1.0 - alpha) * self._ema_yaw
        self._ema_pitch = pitch if self._ema_pitch is None else alpha * pitch + (1.0 - alpha) * self._ema_pitch
        dyaw, dpitch = self._ema_yaw - self.baseline_yaw, self._ema_pitch - self.baseline_pitch
        pitch_zone = self.pitch_up_deadzone if dpitch < 0 else self.pitch_down_deadzone
        yaw_ratio, pitch_ratio = abs(dyaw) / self.yaw_deadzone, abs(dpitch) / pitch_zone
        confidence = _confidence(output)
        if max(yaw_ratio, pitch_ratio) <= 1.0:
            return Direction.STOP, confidence, "HEAD NEUTRAL"
        # Exact ties favor yaw consistently rather than alternating axes.
        if yaw_ratio >= pitch_ratio:
            command = Direction.LEFT if dyaw < 0 else Direction.RIGHT
        else:
            command = Direction.UP if dpitch < 0 else Direction.DOWN
        return command, confidence, f"HEAD {command.value}"


class HandGestureMapper:
    def __init__(self, config: dict | None = None):
        settings = _section(config, "hand")
        self.ema_alpha = _setting(settings, "ema_alpha", 0.45, high=1.0)
        self.deadzone_x = _setting(settings, "deadzone_x", 0.12)
        self.deadzone_y = _setting(settings, "deadzone_y", 0.12)
        self.baseline_x = self.baseline_y = 0.5
        self.reset()

    @staticmethod
    def _index_tip(features: dict) -> tuple[float, float] | None:
        try:
            point = features["index_tip_norm"]
            if isinstance(point, (str, bytes)) or len(point) != 2:
                return None
            return tuple(_number(value) for value in point)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    def calibrate(self, features: dict) -> bool:
        point = self._index_tip(features)
        if point is None:
            return False
        self.baseline_x, self.baseline_y = point
        self._smooth_x, self._smooth_y = point
        return True

    def reset(self, reset_baseline: bool = False) -> None:
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None
        if reset_baseline:
            self.baseline_x = self.baseline_y = 0.5

    def map(self, output: ControllerOutput) -> tuple[Direction, float, str]:
        point = self._index_tip(output.features) if output.detected else None
        if point is None:
            self.reset()
            return Direction.STOP, 0.0, "NO HAND"
        try:
            count = output.features.get("extended_count")
            if count is not None and (isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 5):
                raise ValueError("Invalid extended_count.")
            if count == 0:
                return Direction.STOP, 0.9, "FIST (STOP)"
            states = output.features.get("finger_states")
            if states is not None:
                if not isinstance(states, dict):
                    raise ValueError("finger_states must be a dictionary.")
                if states.get("index") is False:
                    return Direction.STOP, 0.6, "INDEX CURLED (STOP)"
        except (TypeError, ValueError, AttributeError):
            self.reset()
            return Direction.STOP, 0.0, "INVALID HAND FEATURES"

        ix, iy = point
        alpha = self.ema_alpha
        self._smooth_x = ix if self._smooth_x is None else alpha * ix + (1.0 - alpha) * self._smooth_x
        self._smooth_y = iy if self._smooth_y is None else alpha * iy + (1.0 - alpha) * self._smooth_y
        dx, dy = self._smooth_x - self.baseline_x, self._smooth_y - self.baseline_y
        norm_x = abs(dx) / max(self.deadzone_x, 1e-6)
        norm_y = abs(dy) / max(self.deadzone_y, 1e-6)
        if norm_x < 1.0 and norm_y < 1.0:
            return Direction.STOP, 0.9, "INDEX CENTER (STOP)"
        # Coordinates are already mirrored: increasing x/y means RIGHT/DOWN.
        if norm_x >= norm_y:
            command, strength = (Direction.LEFT if dx < 0 else Direction.RIGHT), norm_x
        else:
            command, strength = (Direction.UP if dy < 0 else Direction.DOWN), norm_y
        return command, min(1.0, 0.6 + 0.2 * strength), f"INDEX {command.value}"


class EyeGestureMapper:
    def __init__(self, config: dict | None = None):
        settings = _section(config, "eye")
        self.ema_alpha = _setting(settings, "ema_alpha", 0.45, high=1.0)
        self.deadzone_x = _setting(settings, "deadzone_x", 0.12)
        self.deadzone_y = _setting(settings, "deadzone_y", 0.11)
        self.closed_ear_threshold = _setting(settings, "closed_ear_threshold", 0.19)
        self.baseline_x = self.baseline_y = 0.5
        self.reset()

    def _extract_pupil(self, features: dict) -> tuple[float, float] | None:
        if not isinstance(features, dict):
            return None
        for key in ("iris_pupil_ratio", "avg_pupil_ratio"):
            pupil = features.get(key)
            try:
                if isinstance(pupil, (str, bytes)) or len(pupil) != 2:
                    continue
                return tuple(max(0.0, min(1.0, _number(value))) for value in pupil)
            except (TypeError, ValueError, OverflowError):
                continue
        return None

    def calibrate(self, features: dict) -> bool:
        pupil = self._extract_pupil(features)
        if pupil is None:
            return False
        self.baseline_x, self.baseline_y = pupil
        self._smooth_x, self._smooth_y = pupil
        return True

    def reset(self, reset_baseline: bool = False) -> None:
        self._smooth_x: float | None = None
        self._smooth_y: float | None = None
        if reset_baseline:
            self.baseline_x = self.baseline_y = 0.5

    def map(self, output: ControllerOutput, timestamp_sec: float | None = None) -> tuple[Direction, float, str]:
        if not output.detected:
            self.reset()
            return Direction.STOP, 0.0, "NO EYE"
        try:
            avg = output.features.get("avg_ear")
            if avg is not None and _number(avg) < self.closed_ear_threshold:
                # Preserve the open-eye EMA; a closed ROI is not pupil evidence.
                return Direction.STOP, 0.8, "EYES CLOSED (STOP)"
        except (TypeError, ValueError, OverflowError, AttributeError):
            self.reset()
            return Direction.STOP, 0.0, "INVALID EYE FEATURES"
        pupil = self._extract_pupil(output.features)
        if pupil is None:
            self.reset()
            return Direction.STOP, 0.0, "NO PUPIL"
        px, py = pupil
        alpha = self.ema_alpha
        self._smooth_x = px if self._smooth_x is None else alpha * px + (1.0 - alpha) * self._smooth_x
        self._smooth_y = py if self._smooth_y is None else alpha * py + (1.0 - alpha) * self._smooth_y
        dx, dy = self._smooth_x - self.baseline_x, self._smooth_y - self.baseline_y
        norm_x = abs(dx) / max(self.deadzone_x, 1e-6)
        norm_y = abs(dy) / max(self.deadzone_y, 1e-6)
        if norm_x < 1.0 and norm_y < 1.0:
            return Direction.STOP, 0.9, "PUPIL CENTER (STOP)"
        if norm_x >= norm_y:
            command, strength = (Direction.LEFT if dx < 0 else Direction.RIGHT), norm_x
        else:
            command, strength = (Direction.UP if dy < 0 else Direction.DOWN), norm_y
        return command, min(1.0, 0.6 + 0.2 * strength), f"PUPIL {command.value}"


class GestureEngine:
    def __init__(self, config: dict | None = None):
        self.config = contracts.load_config() if config is None else config
        self.face_mapper = FaceGestureMapper(self.config)
        self.hand_mapper = HandGestureMapper(self.config)
        self.eye_mapper = EyeGestureMapper(self.config)
        control = self.config.get("control", {})
        self.debouncer = CommandDebouncer(control.get("debounce_frames", 3),
                                          control.get("lost_frames_to_stop", 3))
        self.reset()

    def reset(self) -> None:
        self.debouncer.reset()
        self.face_mapper.reset()
        self.hand_mapper.reset()
        self.eye_mapper.reset()
        self._mode: str | None = None
        self._stable_confidence = 0.0
        self._pulse_emitted = False

    @staticmethod
    def _mode_value(output: ControllerOutput) -> str:
        return output.mode.value if isinstance(output.mode, contracts.ControlMode) else str(output.mode).strip().upper()

    def map_output(self, output: ControllerOutput,
                   timestamp_sec: float | None = None) -> tuple[Direction, float, str]:
        now = _timestamp(timestamp_sec)
        mode = self._mode_value(output)
        if mode != self._mode:
            self.reset()  # No moving command or smoothing history crosses a mode change.
            self._mode = mode
        if mode == "FACE":
            raw, confidence, label = self.face_mapper.map(output)
        elif mode == "HAND":
            raw, confidence, label = self.hand_mapper.map(output)
        elif mode == "EYE":
            raw, confidence, label = self.eye_mapper.map(output, now)
        else:
            self.reset()
            return Direction.STOP, 0.0, "UNSUPPORTED MODE (STOP)"
        detected = (bool(output.detected) and not label.startswith("INVALID ")
                    and label not in ("NO HAND", "NO EYE", "NO PUPIL"))
        stable = self.debouncer.update(raw, detected)
        if not detected:
            return stable, 0.0, f"TRACKING LOST ({stable.value})"
        if stable == raw:
            self._stable_confidence = confidence
            return stable, confidence, label
        return stable, self._stable_confidence, f"DEBOUNCING {raw.value} (HOLD {stable.value})"

    def build_packet(self, output: ControllerOutput, active_alarms: list[str], fps: float,
                     timestamp_sec: float | None = None) -> tuple[contracts.ControlPacket, str]:
        now = _timestamp(timestamp_sec)
        command, confidence, label = self.map_output(output, now)
        packet = contracts.ControlPacket(now, self._mode_value(output), command.value,
                                         confidence, list(active_alarms), _number(fps))
        return packet, label

    def calibrate(self, output: ControllerOutput) -> bool:
        mode = self._mode_value(output)
        if mode == "FACE":
            return self.calibrate_face(output)
        if not output.detected:
            return False
        mapper = {"HAND": self.hand_mapper, "EYE": self.eye_mapper}.get(mode)
        if mapper is None or not mapper.calibrate(output.features):
            return False
        self.debouncer.reset()
        self._stable_confidence = 0.0
        self._pulse_emitted = False
        return True

    def calibrate_face(self, output: ControllerOutput) -> bool:
        if self._mode_value(output) != "FACE" or not output.detected:
            return False
        if not self.face_mapper.calibrate(output.features):
            return False
        self.debouncer.reset()
        self._stable_confidence = 0.0
        self._pulse_emitted = False
        return True
