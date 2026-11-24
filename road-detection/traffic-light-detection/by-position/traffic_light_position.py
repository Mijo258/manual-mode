"""Detect vertical traffic lights and classify their state using lamp position."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Final, Literal
from xml.etree import ElementTree

import cv2
import numpy as np
from numpy.typing import NDArray


BGRImage = NDArray[np.uint8]
BoundingBox = tuple[int, int, int, int]
LightState = Literal["RED", "YELLOW", "GREEN", "UNKNOWN"]

CASCADE_DOWNLOAD_COMMAND: Final[str] = (
    "curl -fL -o traffic_light_cascade.xml "
    "https://raw.githubusercontent.com/eyeamskd/traffic-light-detection/"
    "master/cascade7.xml"
)
DEPENDENCY_INSTALL_COMMAND: Final[str] = (
    "python -m pip install --upgrade --force-reinstall --only-binary=:all: "
    "-r requirements.txt"
)
DEFAULT_CASCADE_PATH: Final[Path] = Path(__file__).resolve().with_name(
    "traffic_light_cascade.xml"
)
STATE_COLORS: Final[dict[LightState, tuple[int, int, int]]] = {
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "GREEN": (0, 255, 0),
    "UNKNOWN": (180, 180, 180),
}


class TrafficLightPositionDetector:
    """Classify full, upright three-lamp housings by their brightest region.

    The expected lamp order is red, yellow, green from top to bottom. Hue is
    never used for classification; HSV's Value channel supplies brightness.
    """

    def __init__(
        self,
        cascade_path: str | Path = DEFAULT_CASCADE_PATH,
        *,
        brightness_threshold: int = 180,
        relative_threshold: float = 0.85,
        minimum_contrast: float = 20.0,
        minimum_blob_fraction: float = 0.005,
    ) -> None:
        """Load the cascade and configure brightness-based segmentation.

        Args:
            cascade_path: Cascade XML path; defaults to the script directory.
            brightness_threshold: Minimum brightness on the 0-255 scale.
            relative_threshold: Threshold as a fraction of peak brightness.
            minimum_contrast: Required peak brightness above the crop median.
            minimum_blob_fraction: Minimum component area relative to the crop.

        Raises:
            FileNotFoundError: The cascade XML does not exist.
            ValueError: A setting is invalid or the cascade cannot be loaded.
            RuntimeError: The imported OpenCV lacks the Haar cascade API.
        """
        if not 0 <= brightness_threshold < 255:
            raise ValueError("brightness_threshold must be in [0, 255).")
        if not 0.0 < relative_threshold < 1.0:
            raise ValueError("relative_threshold must be in (0, 1).")
        if not 0.0 < minimum_contrast <= 255.0:
            raise ValueError("minimum_contrast must be in (0, 255].")
        if not 0.0 < minimum_blob_fraction < 1.0:
            raise ValueError("minimum_blob_fraction must be in (0, 1).")

        cascade_file = Path(cascade_path).expanduser().resolve()
        if not cascade_file.is_file():
            raise FileNotFoundError(
                f"Traffic light cascade not found: {cascade_file}\n"
                "Open a terminal in the cascade's parent directory and run:\n"
                f"{CASCADE_DOWNLOAD_COMMAND}\n"
                "The downloaded file must be saved at the path shown above."
            )

        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                f"OpenCV {getattr(cv2, '__version__', 'unknown')} imported from "
                f"{getattr(cv2, '__file__', 'an unknown location')} does not "
                "provide CascadeClassifier. This project requires the OpenCV "
                "4.x package pinned in requirements.txt. In the activated "
                "virtual environment, from the project directory, run:\n"
                f"{DEPENDENCY_INSTALL_COMMAND}\n"
                "Also make sure no local cv2.py file or cv2 directory shadows "
                "the installed package."
            )

        # curl without --fail can save a 404 response as the requested XML.
        # Detect that before passing the file to OpenCV's native XML reader.
        try:
            cascade_root = ElementTree.parse(cascade_file).getroot()
        except ElementTree.ParseError as exc:
            raise ValueError(
                f"Invalid cascade XML: {cascade_file}. The file may contain "
                "an HTTP error response such as '404: Not Found'. "
                "From the cascade's parent directory, download it again:\n"
                f"{CASCADE_DOWNLOAD_COMMAND}"
            ) from exc
        if cascade_root.tag != "opencv_storage":
            raise ValueError(
                f"The file is not an OpenCV cascade XML: {cascade_file}. "
                "From the cascade's parent directory, download it again:\n"
                f"{CASCADE_DOWNLOAD_COMMAND}"
            )

        try:
            self.cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(
                str(cascade_file)
            )
        except (cv2.error, SystemError) as exc:
            raise ValueError(
                f"Invalid cascade XML: {cascade_file}. Download it again using:\n"
                f"{CASCADE_DOWNLOAD_COMMAND}"
            ) from exc
        if self.cascade.empty():
            raise ValueError(
                f"OpenCV could not load the cascade: {cascade_file}. "
                "From the cascade's parent directory, download it again:\n"
                f"{CASCADE_DOWNLOAD_COMMAND}"
            )

        self.brightness_threshold: int = brightness_threshold
        self.relative_threshold: float = relative_threshold
        self.minimum_contrast: float = minimum_contrast
        self.minimum_blob_fraction: float = minimum_blob_fraction

    @staticmethod
    def _validate_bgr(image: BGRImage) -> None:
        """Raise ValueError unless image is a nonempty uint8 H x W x 3 array."""
        if (
            image.size == 0
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
        ):
            raise ValueError("Expected a nonempty uint8 BGR image (H x W x 3).")

    def detect_lights(self, frame: BGRImage) -> list[BoundingBox]:
        """Return traffic light boxes as (x, y, width, height) in frame pixels.

        Args:
            frame: Nonempty uint8 BGR camera frame.

        Raises:
            ValueError: The frame is not a valid BGR image.
        """
        self._validate_bgr(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(12, 30),
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]

    def classify_by_position(self, crop: BGRImage) -> LightState:
        """Locate the brightest region and map its centroid to a vertical third.

        Args:
            crop: BGR crop containing an entire upright three-lamp housing.

        Returns:
            RED, YELLOW, or GREEN for the top, middle, or bottom third.
            UNKNOWN for an empty crop or no sufficiently bright, distinct blob.

        Raises:
            ValueError: A nonempty crop has an invalid shape or data type.
        """
        if crop.size == 0:
            return "UNKNOWN"
        self._validate_bgr(crop)

        # Value preserves bright saturated lamps even when their grayscale
        # intensity is low. No hue or color ranges are consulted.
        brightness = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]
        blurred = cv2.GaussianBlur(brightness, (5, 5), 0)
        _, peak_brightness, _, _ = cv2.minMaxLoc(blurred)
        background_brightness = float(np.median(blurred))
        if (
            peak_brightness <= self.brightness_threshold
            or peak_brightness - background_brightness < self.minimum_contrast
        ):
            return "UNKNOWN"

        threshold_value = max(
            float(self.brightness_threshold),
            self.relative_threshold * peak_brightness,
        )
        _, binary = cv2.threshold(
            blurred, threshold_value, 255, cv2.THRESH_BINARY
        )

        # Keep regions separate: averaging all bright pixels could put the
        # centroid between a lamp and an unrelated reflection.
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        minimum_area = max(3.0, brightness.size * self.minimum_blob_fraction)
        best_label: int | None = None
        best_score: tuple[float, int] = (-1.0, 0)
        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            component_mask = (labels == label).astype(np.uint8)
            mean_brightness = float(cv2.mean(blurred, mask=component_mask)[0])
            score = (mean_brightness, area)
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return "UNKNOWN"

        brightest_region = (labels == best_label).astype(np.uint8)
        moments = cv2.moments(brightest_region, binaryImage=True)
        if moments["m00"] == 0.0:
            return "UNKNOWN"
        centroid: tuple[float, float] = (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )
        third_height = crop.shape[0] / 3.0
        if centroid[1] < third_height:
            return "RED"
        if centroid[1] < 2.0 * third_height:
            return "YELLOW"
        return "GREEN"

    def process_frame(self, frame: BGRImage) -> BGRImage:
        """Return a frame copy with detected boxes and 'Pos: STATE' labels.

        Args:
            frame: Nonempty uint8 BGR camera frame; it is not modified.

        Raises:
            ValueError: The frame is not a valid BGR image.
        """
        boxes = self.detect_lights(frame)
        annotated = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        for x, y, width, height in boxes:
            left, top = max(0, x), max(0, y)
            right = min(frame_width, x + width)
            bottom = min(frame_height, y + height)
            if right <= left or bottom <= top:
                continue

            state = self.classify_by_position(frame[top:bottom, left:right])
            color = STATE_COLORS[state]
            cv2.rectangle(annotated, (left, top), (right - 1, bottom - 1), color, 2)
            label_y = top - 8 if top >= 24 else min(frame_height - 5, top + 20)
            cv2.putText(
                annotated,
                f"Pos: {state}",
                (left, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated


def main() -> None:
    """Display annotated webcam frames and smoothed FPS until q is pressed."""
    detector = TrafficLightPositionDetector()
    capture = cv2.VideoCapture(0)
    try:
        if not capture.isOpened():
            raise RuntimeError(
                "Could not open webcam 0. Check the connection and camera "
                "permissions, and close other applications using the camera."
            )

        previous_time = perf_counter()
        smoothed_fps = 0.0
        while True:
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                raise RuntimeError("Webcam 0 stopped delivering valid frames.")

            annotated = detector.process_frame(frame)
            current_time = perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            if elapsed > 0.0:
                instantaneous_fps = 1.0 / elapsed
                smoothed_fps = (
                    instantaneous_fps
                    if smoothed_fps == 0.0
                    else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
                )
            cv2.putText(
                annotated,
                f"FPS: {smoothed_fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Traffic Light Position Detection - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
