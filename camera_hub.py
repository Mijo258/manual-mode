"""One camera producer with independent, timestamped frames for many consumers."""

from dataclasses import dataclass
import threading
import time
from typing import Callable

import cv2
import numpy as np

from contracts import load_config


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp_ms: int
    frame: np.ndarray


class CameraHub:
    """Own a single capture and publish only its latest complete BGR frame.

    ``config`` accepts either the full project config or its camera section.
    An injected capture factory has the same call signature as VideoCapture.
    ``copy=False`` borrows a published image; callers must not modify that image.
    """

    def __init__(self, config: dict | None = None, capture_factory=None):
        settings = load_config()["camera"] if config is None else config.get("camera", config)
        self.index = int(settings.get("index", 0))
        self.backend = str(settings.get("backend", "DSHOW")).upper()
        backends = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF, "ANY": cv2.CAP_ANY}
        if self.backend not in backends:
            raise ValueError("Camera backend must be DSHOW, MSMF or ANY.")
        self.backend_flag = backends[self.backend]
        self.width = int(settings.get("width", 640))
        self.height = int(settings.get("height", 480))
        self.fps = float(settings.get("fps", 30))
        self.flip_horizontal = bool(settings.get("flip_horizontal", True))
        if self.width <= 0 or self.height <= 0 or not 0.0 < self.fps < float("inf"):
            raise ValueError("Camera width, height and FPS must be positive and finite.")
        self._capture_factory: Callable = capture_factory or cv2.VideoCapture
        self._condition = threading.Condition()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._capture = None
        self._thread: threading.Thread | None = None
        self._latest: FramePacket | None = None
        self._frame_id = 0
        self._last_timestamp_ms = -1
        self._actual_fps = 0.0
        self._consecutive_failures = 0
        self._last_error: str | None = None

    def start(self) -> None:
        """Open the camera and start one capture thread; repeated starts are safe."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                if self._stop_event.is_set():
                    raise RuntimeError("The previous camera capture thread is still stopping.")
                return
            cap = None
            try:
                cap = self._capture_factory(self.index, self.backend_flag)
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    cap = self._capture_factory(self.index)
                if not cap.isOpened():
                    raise RuntimeError("The requested backend and default backend could not open it.")
                for prop, value in (
                    (cv2.CAP_PROP_FRAME_WIDTH, self.width),
                    (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
                    (cv2.CAP_PROP_FPS, self.fps),
                    (cv2.CAP_PROP_BUFFERSIZE, 1),
                ):
                    try:
                        cap.set(prop, value)
                    except (cv2.error, OSError, AttributeError, NotImplementedError):
                        pass  # Capture properties are hints; some drivers do not support them.
            except Exception as exc:
                if cap is not None:
                    self._safe_release(cap)
                with self._condition:
                    self._last_error = f"Unable to open webcam {self.index} ({self.backend}): {exc}"
                    self._stop_event.set()
                    self._condition.notify_all()
                raise RuntimeError(self._last_error) from exc

            with self._condition:
                self._latest = None
                self._actual_fps = 0.0
                self._consecutive_failures = 0
                self._last_error = None
                self._capture = cap
                self._stop_event.clear()
                self._thread = threading.Thread(
                    target=self._capture_loop, args=(cap,), name="camera-hub", daemon=True
                )
            try:
                self._thread.start()
            except RuntimeError:
                self._stop_event.set()
                self._release_capture(cap)
                with self._condition:
                    self._condition.notify_all()
                raise

    @staticmethod
    def _safe_release(cap) -> None:
        try:
            cap.release()
        except (cv2.error, OSError, RuntimeError):
            pass

    def _release_capture(self, cap) -> None:
        with self._condition:
            if self._capture is not cap:
                return
            self._capture = None
        self._safe_release(cap)

    def _capture_loop(self, cap) -> None:
        previous_ns: int | None = None
        try:
            while not self._stop_event.is_set():
                try:
                    ok, frame = cap.read()
                except (cv2.error, OSError, RuntimeError) as exc:
                    ok, frame = False, None
                    with self._condition:
                        self._last_error = f"Webcam read failed: {exc}"
                if self._stop_event.is_set():
                    break
                if not ok or not isinstance(frame, np.ndarray) or frame.size == 0:
                    with self._condition:
                        self._consecutive_failures += 1
                    self._stop_event.wait(0.005)
                    continue
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    with self._condition:
                        self._consecutive_failures += 1
                        self._last_error = "Webcam returned an image that is not BGR uint8."
                    self._stop_event.wait(0.005)
                    continue

                # Own the buffer even when a capture implementation reuses its read buffer.
                frame = cv2.flip(frame, 1) if self.flip_horizontal else frame.copy()
                now_ns = time.monotonic_ns()
                with self._condition:
                    if self._stop_event.is_set():
                        break
                    self._frame_id += 1
                    self._last_timestamp_ms = max(self._last_timestamp_ms + 1, now_ns // 1_000_000)
                    if previous_ns is not None and now_ns > previous_ns:
                        instant_fps = 1_000_000_000.0 / (now_ns - previous_ns)
                        self._actual_fps = (
                            instant_fps if self._actual_fps == 0.0
                            else 0.9 * self._actual_fps + 0.1 * instant_fps
                        )
                    previous_ns = now_ns
                    self._consecutive_failures = 0
                    self._last_error = None
                    self._latest = FramePacket(self._frame_id, self._last_timestamp_ms, frame)
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._last_error = f"Camera capture stopped: {exc}"
        finally:
            self._stop_event.set()
            self._release_capture(cap)
            with self._condition:
                self._condition.notify_all()

    @property
    def actual_fps(self) -> float:
        with self._condition:
            return self._actual_fps

    @property
    def last_error(self) -> str | None:
        with self._condition:
            return self._last_error

    @property
    def consecutive_read_failures(self) -> int:
        with self._condition:
            return self._consecutive_failures

    def get_latest(self, last_frame_id: int = -1, timeout: float = 0.5,
                   copy: bool = True) -> FramePacket | None:
        """Wait for a newer frame; copy it under the publication lock by default."""
        if not 0.0 <= timeout < float("inf"):
            raise ValueError("Frame wait timeout must be nonnegative and finite.")
        with self._condition:
            self._condition.wait_for(
                lambda: (self._latest is not None and self._latest.frame_id > last_frame_id)
                or self._stop_event.is_set(), timeout=timeout
            )
            packet = self._latest
            if packet is None or packet.frame_id <= last_frame_id:
                return None
            return FramePacket(packet.frame_id, packet.timestamp_ms,
                               packet.frame.copy() if copy else packet.frame)

    def stop(self) -> None:
        """Wake consumers, join for at most two seconds and release the capture."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        with self._lifecycle_lock:
            # start() may have cleared the event while stop() waited for camera open.
            self._stop_event.set()
            with self._condition:
                self._condition.notify_all()
            thread = self._thread
            if thread is not None and thread.ident is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)
            with self._condition:
                cap = self._capture
            if cap is not None:
                self._release_capture(cap)
            if thread is None or not thread.is_alive():
                self._thread = None

    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive() and not self._stop_event.is_set()

    def __enter__(self) -> "CameraHub":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
