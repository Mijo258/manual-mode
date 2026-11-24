"""Locate a traffic-light housing, then identify and outline its active lamp.

Importing this module does not open a camera or create GUI windows. Supply a
trained OpenCV traffic-light cascade beside this file or through the constructor.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import ClassVar, Literal, Sequence, TypeAlias
from xml.etree import ElementTree

import cv2
import numpy as np
from numpy.typing import NDArray


BGRImage: TypeAlias = NDArray[np.uint8]
BoundingBox: TypeAlias = tuple[int, int, int, int]
BGRColor: TypeAlias = tuple[int, int, int]
LightState: TypeAlias = Literal["RED", "YELLOW", "GREEN", "UNKNOWN"]
SignalColor: TypeAlias = Literal["RED", "YELLOW", "GREEN"]
ColorMasks: TypeAlias = dict[SignalColor, NDArray[np.uint8]]


@dataclass(frozen=True, slots=True)
class ColorThresholds:
    """Brightness and color thresholds applied only inside a detected housing.

    ``min_value`` is the activation floor for a lamp, not for every color pixel.
    ``min_support_value`` retains dim colored rims around bright lamp cores.
    ``peak_fraction`` filters brightness within a color region and compares
    lamp strengths after color regions are separated.
    ``min_pixels=1`` preserves small visible fragments; increase it to reject
    isolated sensor noise. These are camera-dependent thresholds, not a
    guarantee that reflected light can always be distinguished from emission.
    """

    min_saturation: int = 8
    min_value: int = 60
    min_chroma: int = 6
    peak_fraction: float = 0.85
    min_pixels: int = 1
    min_support_value: int = 24

    def __post_init__(self) -> None:
        """Reject thresholds outside the supported pixel and parameter ranges."""
        for name in ("min_saturation", "min_value", "min_chroma", "min_support_value"):
            value: int = getattr(self, name)
            if not isinstance(value, int) or not 0 <= value <= 255:
                raise ValueError(f"{name} must be an integer between 0 and 255.")
        if self.min_chroma == 0:
            raise ValueError("min_chroma must be positive to reject gray/white pixels.")
        if not isfinite(self.peak_fraction) or not 0.0 <= self.peak_fraction <= 1.0:
            raise ValueError("peak_fraction must be finite and between 0 and 1.")
        if not isinstance(self.min_pixels, int) or self.min_pixels < 1:
            raise ValueError("min_pixels must be a positive integer.")


@dataclass(frozen=True, slots=True)
class LightDetection:
    """One housing and its optional active lamp, in original-frame coordinates.

    ``box`` surrounds the traffic-light housing. ``lamp_box`` surrounds only
    the selected illuminated region. UNKNOWN has no lamp box or on-state label.
    """

    box: BoundingBox
    lamp_box: BoundingBox | None
    state: LightState


@dataclass(frozen=True, slots=True)
class _LampCandidate:
    """One connected color region with its enclosed core, in crop coordinates."""

    state: SignalColor
    box: BoundingBox
    pixels: int
    strength: float
    peak_value: float
    color_value: float
    saturation: float
    has_core: bool
    footprint: NDArray[np.bool_]


def _validate_opencv_installation() -> None:
    """Reject incompatible or incomplete OpenCV builds before native API calls.

    Raises:
        RuntimeError: The imported cv2 module lacks required OpenCV attributes.
            The message identifies the interpreter and imported module location
            so the caller can repair the correct Python environment.
    """
    required_attributes: tuple[str, ...] = (
        "CascadeClassifier",
        "cvtColor",
        "inRange",
        "bitwise_or",
        "countNonZero",
        "connectedComponentsWithStats",
        "findContours",
        "drawContours",
        "equalizeHist",
        "rectangle",
        "putText",
        "error",
    )
    missing: list[str] = [
        name for name in required_attributes if not hasattr(cv2, name)
    ]
    if not missing:
        return

    module_file: str | None = getattr(cv2, "__file__", None)
    module_paths: list[str] = [
        str(path) for path in getattr(cv2, "__path__", ())
    ]
    module_location: str = (
        module_file or ", ".join(module_paths) or "<unknown location>"
    )
    opencv_version: str = str(getattr(cv2, "__version__", "unknown"))
    reason: str = (
        "This OpenCV 5 build lacks the Haar cascade API required by this "
        "detector. Install the pinned OpenCV 4 release."
        if "CascadeClassifier" in missing and opencv_version.startswith("5.")
        else "This OpenCV build lacks required APIs. Check the module location "
        "above for a project-local cv2.py/cv2 folder shadowing OpenCV, or an "
        "incomplete installation."
    )
    raise RuntimeError(
        "The imported cv2 module is missing required OpenCV APIs: "
        f"{', '.join(missing)}.\n"
        f"OpenCV version: {opencv_version}\n"
        f"Python interpreter: {sys.executable}\n"
        f"cv2 loaded from: {module_location}\n"
        f"{reason}\n"
        "Install this project's requirements.txt: opencv-python==4.14.0.94 "
        "and numpy==2.4.6. In the same Python environment, run:\n"
        "  python -m pip install --only-binary=:all: -r requirements.txt\n"
        "See EXECUTION_GUIDE.md for the Git Bash commands targeting traffic_venv."
    )


class TrafficLightDetector:
    """Detect housings first, then find one active lamp inside each housing.

    Detection and classification are independent of camera access and GUI
    display, allowing the class to be reused in a ROS 2 image callback.
    Bounding boxes use ``(x, y, width, height)`` pixel coordinates.
    Color analysis never runs on pixels outside detected traffic-light boxes.
    """

    _STATE_COLORS: ClassVar[dict[LightState, BGRColor]] = {
        "RED": (0, 0, 255),
        "YELLOW": (0, 255, 255),
        "GREEN": (0, 255, 0),
        "UNKNOWN": (180, 180, 180),
    }
    _HUE_RANGES: ClassVar[dict[SignalColor, tuple[tuple[int, int], ...]]] = {
        "RED": ((0, 9), (165, 179)),
        "YELLOW": ((10, 38),),
        "GREEN": ((39, 95),),
    }
    _RELIABLE_SATURATION: ClassVar[int] = 64
    _CLIPPED_VALUE: ClassVar[int] = 250

    def __init__(
        self,
        cascade_path: str | Path | None = None,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (20, 20),
        thresholds: ColorThresholds | None = None,
        contrast_fallback: bool = True,
    ) -> None:
        """Load a pretrained traffic-light cascade and detection parameters.

        Args:
            cascade_path: XML model path. Defaults to
                ``traffic_light_cascade.xml`` beside this Python file.
            scale_factor: Image-pyramid scale multiplier, strictly above 1.
            min_neighbors: Minimum neighboring detections retained by OpenCV.
                Must be nonnegative; higher values usually reject more noise.
            min_size: Minimum detection width and height in pixels.
            thresholds: Immutable color sensitivity settings. Defaults retain
                small bright regions while rejecting dark lenses and white glare.
            contrast_fallback: Retry Haar on normalized maximum-channel intensity
                when the grayscale scan finds no housing. This avoids the lower
                grayscale weight of red. Disable to limit processing to one scan.

        Raises:
            RuntimeError: The imported cv2 module lacks required OpenCV APIs.
            FileNotFoundError: The requested cascade file does not exist.
            ValueError: Parameters or cascade XML are invalid, or loading fails.
        """
        _validate_opencv_installation()
        if not isfinite(scale_factor) or scale_factor <= 1.0:
            raise ValueError("scale_factor must be finite and greater than 1.0.")
        if min_neighbors < 0:
            raise ValueError("min_neighbors must be nonnegative.")
        if len(min_size) != 2 or any(size <= 0 for size in min_size):
            raise ValueError("min_size must contain two positive pixel sizes.")

        model_path: Path = (
            Path(__file__).resolve().with_name("traffic_light_cascade.xml")
            if cascade_path is None
            else Path(cascade_path).expanduser().resolve()
        )
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Traffic-light cascade XML was not found: {model_path}. "
                "Obtain a pretrained OpenCV Haar cascade specifically trained "
                "for traffic lights. Save it beside traffic_light_detector.py "
                "as traffic_light_cascade.xml, or pass its path to "
                "TrafficLightDetector(cascade_path=...). OpenCV does not "
                "bundle a traffic-light cascade."
            )

        # Validate once at startup so HTTP error pages fail before native parsing.
        try:
            xml_root = ElementTree.parse(model_path).getroot()
        except ElementTree.ParseError as error:
            raise ValueError(
                f"The cascade file is not valid XML: {model_path}. "
                "A file containing '404: Not Found' is a failed download, "
                "not a trained model. Replace it with the raw XML of a "
                "pretrained traffic-light Haar cascade from a working source."
            ) from error
        if xml_root is None or xml_root.tag != "opencv_storage":
            raise ValueError(
                f"The file is not an OpenCV cascade XML: {model_path}. "
                "Expected an <opencv_storage> root. Download the raw pretrained "
                "traffic-light cascade XML instead of a repository web page."
            )

        try:
            self._cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(
                str(model_path)
            )
        except (cv2.error, SystemError) as error:
            raise ValueError(
                f"OpenCV could not parse the traffic-light cascade: {model_path}. "
                "Supply a valid, pretrained OpenCV cascade XML file."
            ) from error
        if self._cascade.empty():
            raise ValueError(
                f"OpenCV loaded an empty classifier from: {model_path}. "
                "The file must contain a valid, pretrained traffic-light cascade."
            )

        self._scale_factor: float = scale_factor
        self._min_neighbors: int = min_neighbors
        self._min_size: tuple[int, int] = min_size
        self.thresholds: ColorThresholds = thresholds or ColorThresholds()
        self._contrast_fallback: bool = contrast_fallback

    @staticmethod
    def _validate_bgr_image(image: BGRImage) -> None:
        """Raise ValueError unless image is a nonempty uint8 H-by-W-by-3 array."""
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.size == 0
        ):
            raise ValueError(
                "Expected a nonempty uint8 BGR image with shape (height, width, 3)."
            )

    def detect_lights(self, frame: BGRImage) -> list[BoundingBox]:
        """Find housings with Haar, retrying on normalized intensity after a miss.

        The retry uses max(B, G, R), so an equally strong red, yellow, or green
        lamp has the same intensity. No hue masks or color classification run
        until Haar has supplied a housing box. Each call uses only this frame;
        boxes and lamp states are never carried forward from an earlier frame.

        Args:
            frame: Nonempty BGR image with uint8 channels.

        Returns:
            Nonduplicate bounding boxes clipped to the input image. An empty
            list indicates no detections. Each box contains Python integers
            and has positive width and height.

        Raises:
            ValueError: The frame is not a valid BGR image.
        """
        self._validate_bgr_image(frame)
        gray: NDArray[np.uint8] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_boxes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )
        if len(raw_boxes) == 0 and self._contrast_fallback:
            intensity: NDArray[np.uint8] = cv2.equalizeHist(frame.max(axis=2))
            raw_boxes = self._cascade.detectMultiScale(
                intensity,
                scaleFactor=self._scale_factor,
                minNeighbors=self._min_neighbors,
                minSize=self._min_size,
            )

        frame_height, frame_width = frame.shape[:2]
        boxes: list[BoundingBox] = []
        for x, y, width, height in raw_boxes:
            left: int = max(0, int(x))
            top: int = max(0, int(y))
            right: int = min(frame_width, int(x) + int(width))
            bottom: int = min(frame_height, int(y) + int(height))
            if right > left and bottom > top:
                boxes.append((left, top, right - left, bottom - top))
        # Suppress duplicate housing boxes so a light receives only one label.
        retained: list[BoundingBox] = []
        for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
            if all(self._box_iou(box, other) < 0.5 for other in retained):
                retained.append(box)
        return retained

    @staticmethod
    def _box_iou(first: BoundingBox, second: BoundingBox) -> float:
        """Return intersection over union for two positive-size bounding boxes."""
        x, y, width, height = first
        sx, sy, swidth, sheight = second
        overlap_width: int = max(0, min(x + width, sx + swidth) - max(x, sx))
        overlap_height: int = max(0, min(y + height, sy + sheight) - max(y, sy))
        intersection: int = overlap_width * overlap_height
        return intersection / (width * height + swidth * sheight - intersection)

    @staticmethod
    def _color_masks(
        image: BGRImage,
        thresholds: ColorThresholds,
    ) -> tuple[ColorMasks, NDArray[np.uint8]]:
        """Retain color evidence, including dim rims; activation is checked later."""
        TrafficLightDetector._validate_bgr_image(image)
        hsv: NDArray[np.uint8] = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        chroma: NDArray[np.uint8] = value - image.min(axis=2)
        eligible: NDArray[np.bool_] = (
            (saturation >= thresholds.min_saturation)
            & (value >= min(thresholds.min_value, thresholds.min_support_value))
            & (chroma >= thresholds.min_chroma)
        )
        allowed: NDArray[np.uint8] = eligible.astype(np.uint8) * 255
        masks: ColorMasks = {}
        for state, ranges in TrafficLightDetector._HUE_RANGES.items():
            hue_mask: NDArray[np.uint8] = np.zeros_like(value)
            for lower, upper in ranges:
                hue_mask = cv2.bitwise_or(hue_mask, cv2.inRange(hue, lower, upper))
            masks[state] = cv2.bitwise_and(hue_mask, allowed)
        return masks, hsv

    @staticmethod
    def _lamp_candidates(
        masks: ColorMasks,
        hsv: NDArray[np.uint8],
        settings: ColorThresholds,
    ) -> list[_LampCandidate]:
        """Measure independent color regions without discarding their dim rims.

        Filled contours associate an enclosed bright core with its colored rim.
        They never grow beyond the original region's outline. Peak brightness
        uses reliable color pixels plus the enclosed area's 90th percentile: a
        single colored fragment survives, while one unrelated white hot pixel
        cannot illuminate a large dark lens. Color masks themselves are neither
        blurred nor eroded.
        """
        saturation: NDArray[np.uint8] = hsv[:, :, 1]
        value: NDArray[np.uint8] = hsv[:, :, 2]
        candidates: list[_LampCandidate] = []
        for state, mask in masks.items():
            if cv2.countNonZero(mask) < settings.min_pixels:
                continue
            _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for component in range(1, len(stats)):
                x, y, width, height, area = (int(item) for item in stats[component])
                if area < settings.min_pixels:
                    continue
                region = (slice(y, y + height), slice(x, x + width))
                member: NDArray[np.bool_] = labels[region] == component
                footprint: NDArray[np.uint8] = member.astype(np.uint8) * 255
                if area < width * height:
                    contours, _ = cv2.findContours(
                        footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(footprint, contours, -1, 255, cv2.FILLED)
                filled: NDArray[np.bool_] = footprint != 0
                lamp_value: NDArray[np.uint8] = value[region]
                saturated: NDArray[np.bool_] = member & (
                    saturation[region] >= TrafficLightDetector._RELIABLE_SATURATION
                )
                evidence: NDArray[np.bool_] = (
                    saturated
                    if np.count_nonzero(saturated) >= settings.min_pixels
                    else member
                )
                source_peak: float = float(lamp_value[evidence].max())
                # Compare brightness within this hue component only. A yellow
                # core must never raise the cutoff applied to its red rim.
                evidence = evidence & (
                    lamp_value >= source_peak * settings.peak_fraction
                )
                pixels: int = int(np.count_nonzero(evidence))
                if pixels < settings.min_pixels:
                    continue
                peak_value: float = max(
                    source_peak, float(np.percentile(lamp_value[filled], 90))
                )
                if peak_value < settings.min_value:
                    continue
                color_saturation: float = float(
                    np.median(saturation[region][evidence])
                )
                confidence: float = min(
                    1.0, color_saturation / TrafficLightDetector._RELIABLE_SATURATION
                )
                candidates.append(
                    _LampCandidate(
                        state=state,
                        box=(x, y, width, height),
                        pixels=pixels,
                        strength=peak_value * confidence,
                        peak_value=peak_value,
                        color_value=float(np.percentile(lamp_value[evidence], 90)),
                        saturation=color_saturation,
                        has_core=int(np.count_nonzero(filled)) > area,
                        footprint=filled,
                    )
                )
        return candidates

    @staticmethod
    def _is_enclosed_core(outer: _LampCandidate, inner: _LampCandidate) -> bool:
        """Check whether a brighter, differently colored core belongs to a rim.

        Camera clipping and compression can make a red core yellow or white.
        Trust a saturated surrounding rim only when it actually encloses a
        substantial brighter core. Bounding-box overlap alone is insufficient.
        This geometry describes one lamp, not the ordering of the three lamps.
        """
        if (
            not outer.has_core
            or outer.state == inner.state
            or outer.saturation < TrafficLightDetector._RELIABLE_SATURATION
        ):
            return False
        ox, oy, ow, oh = outer.box
        ix, iy, iw, ih = inner.box
        if (
            (ow * oh <= iw * ih)
            or ix < ox or iy < oy
            or ix + iw > ox + ow or iy + ih > oy + oh
        ):
            return False
        outer_area: int = int(np.count_nonzero(outer.footprint))
        inner_area: int = int(np.count_nonzero(inner.footprint))
        if inner_area < 0.1 * outer_area:
            return False
        enclosing: NDArray[np.bool_] = outer.footprint[
            iy - oy : iy - oy + ih, ix - ox : ix - ox + iw
        ]
        if np.count_nonzero(enclosing & inner.footprint) < 0.9 * inner_area:
            return False
        return inner.peak_value >= outer.color_value and (
            inner.peak_value >= TrafficLightDetector._CLIPPED_VALUE
            or inner.peak_value >= max(outer.color_value + 12.0, outer.color_value * 1.2)
        )

    @staticmethod
    def find_active_light(
        crop: BGRImage, *, thresholds: ColorThresholds | None = None
    ) -> tuple[LightState, BoundingBox | None]:
        """Return one active lamp's color and box relative to its housing.

        Separate connected regions for each hue before comparing lamp strength.
        A lamp's bright core supplies activation evidence; its colored pixels,
        including a dim rim, supply hue evidence. A brighter core enclosed by a
        saturated rim belongs to that rim, even if clipping changes the core hue.
        This avoids merging a tiny red light into an adjacent dim green region.

        Keep regions within ``peak_fraction`` of the strongest signal and compare
        their individual color pixel counts. Disconnected specks are not added
        together. Ties between colors return UNKNOWN. No lamp-position rule is
        used, and all analysis and box padding stay inside this housing crop.

        Args:
            crop: Nonempty uint8 BGR image of one detected housing.
            thresholds: Optional brightness, color, and minimum-pixel settings.

        Returns:
            RED, YELLOW, GREEN, or UNKNOWN, and at most one lamp box.

        Raises:
            ValueError: The crop is not a valid BGR image.
        """
        settings: ColorThresholds = thresholds or ColorThresholds()
        masks, hsv = TrafficLightDetector._color_masks(crop, settings)
        candidates: list[_LampCandidate] = TrafficLightDetector._lamp_candidates(
            masks, hsv, settings
        )
        rims: list[_LampCandidate] = [item for item in candidates if item.has_core]
        candidates = [
            inner for inner in candidates
            if not any(
                TrafficLightDetector._is_enclosed_core(outer, inner)
                for outer in rims
            )
        ]
        if not candidates:
            return "UNKNOWN", None
        strongest: float = max(item.strength for item in candidates)
        candidates = [
            item for item in candidates
            if item.strength >= strongest * settings.peak_fraction
        ]
        counts: dict[SignalColor, int] = {
            color: max(
                (item.pixels for item in candidates if item.state == color), default=0
            )
            for color in masks
        }
        highest_count: int = max(counts.values())
        winners: list[SignalColor] = [
            color for color, count in counts.items() if count == highest_count
        ]
        if len(winners) != 1:
            return "UNKNOWN", None
        state: SignalColor = winners[0]
        selected: _LampCandidate = max(
            (item for item in candidates if item.state == state),
            key=lambda item: (item.pixels, item.strength),
        )
        x, y, width, height = selected.box
        padding: int = max(2, round(max(width, height) * 0.1))
        left: int = max(0, x - padding)
        top: int = max(0, y - padding)
        right: int = min(crop.shape[1], x + width + padding)
        bottom: int = min(crop.shape[0], y + height + padding)
        return state, (left, top, right - left, bottom - top)

    @staticmethod
    def classify_by_color(
        crop: BGRImage, *, thresholds: ColorThresholds | None = None
    ) -> LightState:
        """Return the active color of a housing crop, or UNKNOWN when unclear.

        Args:
            crop: Nonempty uint8 BGR crop of a candidate traffic light.
            thresholds: Optional sensitivity settings; no detector instance or
                cascade is required to use this mathematical helper.

        Returns:
            RED, YELLOW, GREEN, or UNKNOWN.

        Raises:
            ValueError: The crop is not a valid BGR image.
        """
        return TrafficLightDetector.find_active_light(
            crop, thresholds=thresholds
        )[0]

    def detect(self, frame: BGRImage) -> list[LightDetection]:
        """Find housings, then inspect only their crops for one active lamp each.

        Returned housing and lamp boxes both use full-frame coordinates.
        No housing detections means no color analysis and an empty result.
        """
        detections: list[LightDetection] = []
        for box in self.detect_lights(frame):
            x, y, width, height = box
            state, local_box = self.find_active_light(
                frame[y : y + height, x : x + width], thresholds=self.thresholds
            )
            lamp_box: BoundingBox | None = None
            if local_box is not None:
                lx, ly, lamp_width, lamp_height = local_box
                lamp_box = (x + lx, y + ly, lamp_width, lamp_height)
            detections.append(LightDetection(box, lamp_box, state))
        return detections

    def process_frame(self, frame: BGRImage) -> BGRImage:
        """Draw one outer housing box and at most one labeled inner lamp box.

        Args:
            frame: Nonempty uint8 BGR image. The input is never modified.

        Returns:
            A copy with thin neutral housing boxes and colored active-lamp boxes.
            The only per-light text is RED is on, YELLOW is on, or GREEN is on.
            Unclear states show only the outer box. Analysis uses original pixels.

        Raises:
            ValueError: The frame is not a valid BGR image.
        """
        detections: list[LightDetection] = self.detect(frame)
        annotated: BGRImage = frame.copy()
        for detection in detections:
            x, y, width, height = detection.box
            cv2.rectangle(
                annotated, (x, y), (x + width - 1, y + height - 1),
                (200, 200, 200), 1,
            )
            if detection.lamp_box is None:
                continue
            lx, ly, lamp_width, lamp_height = detection.lamp_box
            color: BGRColor = self._STATE_COLORS[detection.state]
            cv2.rectangle(
                annotated, (lx, ly),
                (lx + lamp_width - 1, ly + lamp_height - 1), color, 2,
            )
            label: str = f"{detection.state} is on"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            text_x: int = max(0, min(x, frame.shape[1] - text_width - 1))
            if y >= text_height + 8:
                text_y: int = y - 8
            elif y + height + text_height + baseline + 8 < frame.shape[0]:
                text_y = y + height + text_height + 4
            else:
                text_y = max(0, min(y + text_height + 4, frame.shape[0] - baseline - 1))
            cv2.putText(
                annotated, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 4, cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated


def _save_diagnostic_snapshot(
    frame: BGRImage, detector: TrafficLightDetector, directory: Path
) -> Path:
    """Save untouched camera pixels and measurements only after the user presses s.

    The JSON separates a Haar miss from missing/dim color evidence. It contains
    housing coordinates, hue ranges, brightness measurements, and selected states.
    No images are transmitted, and the normal preview receives no extra labels.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stem: str = datetime.now().strftime("traffic_light_%Y%m%d_%H%M%S_%f")
    image_path: Path = directory / f"{stem}.png"
    report_path: Path = directory / f"{stem}.json"
    if not cv2.imwrite(str(image_path), frame):
        raise OSError(f"Could not save camera pixels to {image_path}.")

    housings: list[dict[str, object]] = []
    for detection in detector.detect(frame):
        x, y, width, height = detection.box
        masks, hsv = detector._color_masks(
            frame[y : y + height, x : x + width], detector.thresholds
        )
        colors: dict[str, object] = {}
        for state, mask in masks.items():
            pixels: NDArray[np.uint8] = hsv[mask != 0]
            colors[state] = {
                "support_pixels": len(pixels),
                "pixels_at_activation_floor": int(
                    np.count_nonzero(pixels[:, 2] >= detector.thresholds.min_value)
                ),
                "hue_min": int(pixels[:, 0].min()) if len(pixels) else None,
                "hue_max": int(pixels[:, 0].max()) if len(pixels) else None,
                "max_value": int(pixels[:, 2].max()) if len(pixels) else None,
                "median_saturation": float(np.median(pixels[:, 1])) if len(pixels) else None,
            }
        housings.append({
            "box": detection.box,
            "state": detection.state,
            "lamp_box": detection.lamp_box,
            "colors": colors,
        })
    report: dict[str, object] = {
        "image": image_path.name,
        "opencv_version": cv2.__version__,
        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
        "thresholds": asdict(detector.thresholds),
        "housing_count": len(housings),
        "housings": housings,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    """Run the default webcam demo with live FPS; press q to release resources.

    FPS is computed from smoothed loop durations using a monotonic clock.
    After the first frame, each interval includes capture and processing of
    this frame plus drawing, display, and event handling from the previous one.
    Camera and GUI resources are released on normal exit and on exceptions.

    Args:
        argv: Optional command-line arguments, including the snapshot directory.

    Raises:
        FileNotFoundError: The traffic-light cascade is missing.
        ValueError: The cascade cannot be loaded.
        RuntimeError: OpenCV lacks required APIs, or the webcam is unavailable.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path(__file__).resolve().with_name("diagnostics"),
        help="Save a raw PNG and JSON here when s is pressed (default: diagnostics).",
    )
    arguments: argparse.Namespace = parser.parse_args(argv)
    detector: TrafficLightDetector = TrafficLightDetector()
    capture: cv2.VideoCapture = cv2.VideoCapture(0)
    window_name: str = "Traffic Light Detection - press q to quit"

    try:
        if not capture.isOpened():
            raise RuntimeError(
                "Cannot open default webcam (index 0). Check its connection, "
                "camera permissions, and whether another application is using it."
            )

        previous_tick: float = perf_counter()
        smoothed_seconds: float | None = None
        while True:
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                raise RuntimeError(
                    "Cannot read a frame from default webcam (index 0). "
                    "The camera may have disconnected or become unavailable."
                )

            annotated: BGRImage = detector.process_frame(frame)
            current_tick: float = perf_counter()
            elapsed: float = current_tick - previous_tick
            previous_tick = current_tick
            smoothed_seconds = (
                elapsed
                if smoothed_seconds is None
                else 0.9 * smoothed_seconds + 0.1 * elapsed
            )
            fps: float = 1.0 / smoothed_seconds if smoothed_seconds > 0.0 else 0.0
            fps_label: str = f"FPS: {fps:.1f}"
            cv2.putText(
                annotated, fps_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                annotated, fps_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.imshow(window_name, annotated)
            key: int = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                try:
                    report: Path = _save_diagnostic_snapshot(
                        frame, detector, arguments.snapshot_dir
                    )
                    print(f"Saved raw frame and diagnostic report: {report}")
                except (OSError, cv2.error) as error:
                    print(f"Snapshot could not be saved: {error}", file=sys.stderr)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Traffic-light detector error: {error}") from error
    except cv2.error as error:
        raise SystemExit(f"OpenCV error: {error}") from error
