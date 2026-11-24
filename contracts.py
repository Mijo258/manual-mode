"""Standard-library interface contracts for vehicle control and driver safety."""

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path


class ControlMode(str, Enum):
    NONE = "NONE"
    FACE = "FACE"
    HAND = "HAND"
    EYE = "EYE"


class DirectionCommand(str, Enum):
    STOP = "STOP"
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass
class SafetyStatus:
    timestamp: float
    face_detected: bool
    ear: float
    mar: float
    gaze_ratio: float
    is_sleepy: bool
    is_yawning: bool
    is_distracted: bool
    active_alarms: list[str]


@dataclass
class ControlPacket:
    timestamp: float
    mode: str
    command: str
    confidence: float
    active_alarms: list[str]
    fps: float

    def to_json(self) -> str:
        """Serialize this packet as a JSON object."""
        return json.dumps(asdict(self), separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, raw: str) -> "ControlPacket":
        """Build a packet from its JSON object representation."""
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("ControlPacket JSON must contain an object.")
        return cls(**data)


def load_config(path: str = "project_config.json") -> dict:
    """Load a JSON object, resolving relative paths beside this module."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Project configuration must contain a JSON object.")
    return config
