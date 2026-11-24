"""Common frame-in/result-out interface for control feature extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

import contracts


@dataclass
class ControllerOutput:
    mode: str
    detected: bool
    annotated_frame: np.ndarray
    landmarks_px: np.ndarray | None
    features: dict[str, Any]
    command: str = contracts.DirectionCommand.STOP.value
    confidence: float = 0.0


class BaseController(ABC):
    mode: contracts.ControlMode

    @abstractmethod
    def process_frame(self, frame: np.ndarray, timestamp_ms: int, draw: bool = True,
                      shared_context: dict[str, Any] | None = None) -> ControllerOutput:
        """Extract features without modifying the caller's frame."""

    def reset(self) -> None:
        """Default: no temporal gesture state to reset."""

    def close(self) -> None:
        """Default: no owned resources to close."""

    def __enter__(self) -> "BaseController":
        if getattr(self, "_closed", False):
            raise RuntimeError("Cannot enter a closed controller.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _prepare_frame(frame: np.ndarray, draw: bool) -> np.ndarray:
    if (not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3
            or frame.shape[2] != 3 or frame.dtype != np.uint8):
        raise ValueError("Expected a nonempty BGR uint8 image with shape (H, W, 3).")
    return frame.copy() if draw else frame


def _draw_hud(frame: np.ndarray, lines: list[str], color=(100, 220, 130)) -> None:
    cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 1, 440), 16 + 24 * len(lines)),
                  (24, 24, 24), -1)
    for row, line in enumerate(lines):
        cv2.putText(frame, line, (16, 32 + row * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)


def _drawing_points(points: np.ndarray, frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return np.rint(np.clip(points, (0, 0), (width - 1, height - 1))).astype(np.int32)
