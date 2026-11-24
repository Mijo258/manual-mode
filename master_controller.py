"""Threaded camera capture and 478-point driver monitoring with alarms."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from safety_daemon import SafetyDaemon, SafetyDaemonError


Frame = NDArray[np.uint8]
CameraSource = Union[int, str]

LOGGER = logging.getLogger(__name__)


def sound_alarm() -> None:
    """Play a short Windows beep, or a terminal bell on other platforms."""
    try:
        import winsound
    except ImportError:
        print("\a", end="", flush=True)
    else:
        winsound.Beep(1800, 250)
        winsound.Beep(1400, 250)


class AlarmNotifier:
    """Log alarm transitions and play periodic sounds without blocking capture."""

    def __init__(self, silent: bool = False) -> None:
        self._silent = silent
        self._previous_alarms: tuple[str, ...] = ()
        self._last_sound_time = float("-inf")
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _play() -> None:
        try:
            sound_alarm()
        except RuntimeError as exc:
            LOGGER.warning("Unable to play the alarm sound: %s", exc)

    def notify(self, alarms: tuple[str, ...]) -> None:
        if alarms != self._previous_alarms:
            if alarms:
                LOGGER.warning("Driver alarm: %s", ", ".join(alarms))
            self._previous_alarms = alarms
        if not alarms or self._silent:
            return
        now = time.monotonic()
        if now - self._last_sound_time < 2.0:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._last_sound_time = now
        self._thread = threading.Thread(target=self._play, name="driver-alarm", daemon=True)
        self._thread.start()


class CameraError(RuntimeError):
    """Raised when the camera cannot be initialized or stops delivering frames."""


class CameraStreamer:
    """Continuously capture frames for non-blocking access by multiple consumers.

    The capture thread is the sole producer. Consumers receive independent copies
    of the latest complete frame, so they cannot mutate shared frame state.

    Args:
        src: OpenCV camera index or video-stream URI.

    Raises:
        CameraError: If the camera cannot be opened or its first frame cannot be
            captured.
    """

    _JOIN_TIMEOUT_SECONDS = 2.0

    def __init__(self, src: CameraSource = 0) -> None:
        self._source = src
        self._capture = cv2.VideoCapture(src)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._frame: Optional[Frame] = None
        self._frame_id = 0
        self._last_error: Optional[str] = None
        self.stopped: bool = False

        if not self._capture.isOpened():
            self._capture.release()
            raise CameraError(f"Unable to open camera source {src!r}.")

        grabbed, frame = self._capture.read()
        if not grabbed or frame is None:
            self._capture.release()
            raise CameraError(
                f"Camera source {src!r} opened but did not provide an initial frame."
            )

        self._frame = frame

    def start(self) -> "CameraStreamer":
        """Start the background capture thread and return this streamer.

        Returns:
            This instance, allowing ``CameraStreamer().start()`` construction.

        Raises:
            CameraError: If the streamer was already stopped and cannot be
                restarted.
        """
        with self._lock:
            if self.stopped:
                raise CameraError("A stopped CameraStreamer cannot be restarted.")

            if self._thread is not None and self._thread.is_alive():
                return self

            self._thread = threading.Thread(
                target=self.update,
                name="camera-streamer",
                daemon=True,
            )
            thread = self._thread

        try:
            thread.start()
        except RuntimeError as exc:
            with self._lock:
                self.stopped = True
                self._last_error = f"Unable to start camera thread: {exc}"
            self._capture.release()
            raise CameraError(self._last_error) from exc

        return self

    def update(self) -> None:
        """Continuously capture frames until stopped or camera input fails."""
        try:
            while True:
                with self._lock:
                    if self.stopped:
                        break

                grabbed, frame = self._capture.read()
                if not grabbed or frame is None:
                    with self._lock:
                        self._last_error = (
                            f"Camera source {self._source!r} stopped delivering frames."
                        )
                        self.stopped = True
                    break

                with self._lock:
                    if self.stopped:
                        break
                    self._frame = frame
                    self._frame_id += 1
        except cv2.error as exc:
            with self._lock:
                self._last_error = f"OpenCV camera error: {exc}"
                self.stopped = True
        finally:
            self._capture.release()
            with self._lock:
                self.stopped = True

    def read(self) -> Optional[Frame]:
        """Return a copy of the latest frame, or ``None`` if unavailable."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def read_next(self, previous_frame_id: int) -> Optional[tuple[int, Frame]]:
        """Return a new frame and its ID, or None if it was already consumed."""
        with self._lock:
            if self._frame is None or self._frame_id == previous_frame_id:
                return None
            return self._frame_id, self._frame.copy()

    @property
    def last_error(self) -> Optional[str]:
        """Return the capture-thread error message, if capture has failed."""
        with self._lock:
            return self._last_error

    def stop(self) -> None:
        """Request capture shutdown and briefly wait for resource cleanup."""
        with self._lock:
            self.stopped = True
            thread = self._thread

        if thread is None:
            self._capture.release()
            return

        if thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=self._JOIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                LOGGER.warning(
                    "Camera thread did not stop within %.1f seconds.",
                    self._JOIN_TIMEOUT_SECONDS,
                )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the master router with driver monitoring and a live FPS overlay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", type=Path, help="Path to a face_landmarker.task bundle")
    parser.add_argument("--silent", action="store_true", help="Show alarms without sounds")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check", action="store_true", help="Check model inference without a camera")
    checks.add_argument("--test-alarm", action="store_true", help="Play the alarm sound and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    streamer: Optional[CameraStreamer] = None
    safety_daemon: Optional[SafetyDaemon] = None
    window_opened = False

    try:
        if args.test_alarm:
            sound_alarm()
            LOGGER.info("Alarm sound test completed.")
            return 0

        safety_daemon = SafetyDaemon(model_path=args.model)
        if args.check:
            safety_daemon.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
            LOGGER.info("Startup check PASSED: MediaPipe Tasks model loaded and inference works.")
            return 0

        streamer = CameraStreamer(src=args.camera).start()
        notifier = AlarmNotifier(silent=args.silent)
        LOGGER.info("Driver monitoring started on camera %d. Press q or Esc to quit.", args.camera)
        previous_time = time.perf_counter()
        smoothed_fps = 0.0
        previous_frame_id = -1

        while True:
            capture_error = streamer.last_error
            if capture_error is not None:
                raise CameraError(capture_error)

            captured = streamer.read_next(previous_frame_id)
            if captured is not None:
                previous_frame_id, frame = captured
                frame = safety_daemon.process_frame(frame)
                notifier.notify(safety_daemon.active_alarms)
                current_time = time.perf_counter()
                elapsed = current_time - previous_time
                previous_time = current_time

                if elapsed > 0.0:
                    instantaneous_fps = 1.0 / elapsed
                    smoothed_fps = (
                        instantaneous_fps
                        if smoothed_fps == 0.0
                        else (0.90 * smoothed_fps) + (0.10 * instantaneous_fps)
                    )

                cv2.putText(
                    frame,
                    f"FPS: {smoothed_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Master AV Controller", frame)
                window_opened = True

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    except (CameraError, SafetyDaemonError, ValueError, RuntimeError, cv2.error) as exc:
        LOGGER.error("Master controller stopped: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Master controller interrupted by user.")
    finally:
        if streamer is not None:
            streamer.stop()
        if safety_daemon is not None:
            safety_daemon.close()
        if window_opened:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
