"""Manual synthetic regression checks; these tests never open a camera.

Run with: python -m unittest -v test_traffic_light_detector
The supplied cascade is loaded for pipeline checks. Mocked housing detections
verify that color analysis stays inside those boxes and the display stays simple.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from traffic_light_detector import (
    BGRColor,
    BGRImage,
    BoundingBox,
    ColorThresholds,
    SignalColor,
    TrafficLightDetector,
)


COLORS: dict[SignalColor, BGRColor] = {
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "GREEN": (0, 255, 0),
}
CENTERS: dict[SignalColor, tuple[int, int]] = {
    "RED": (18, 15),
    "YELLOW": (18, 45),
    "GREEN": (18, 75),
}


def make_housing(active: SignalColor | None) -> BGRImage:
    """Create three colored lenses, with only the requested lamp illuminated."""
    crop: BGRImage = np.zeros((90, 36, 3), dtype=np.uint8)
    for state, color in COLORS.items():
        blue, green, red = color
        lens_color: BGRColor = color if state == active else (blue // 5, green // 5, red // 5)
        cv2.circle(crop, CENTERS[state], 6, lens_color, -1)
    return crop


class ColorClassificationTests(unittest.TestCase):
    """Check physical pixel evidence independently of the Haar model."""

    def test_one_colored_pixel_in_large_dark_crop(self) -> None:
        for state, color in COLORS.items():
            with self.subTest(state=state):
                crop: BGRImage = np.zeros((100, 100, 3), dtype=np.uint8)
                crop[50, 50] = color
                self.assertEqual(TrafficLightDetector.classify_by_color(crop), state)

    def test_low_saturation_color_is_retained(self) -> None:
        colors: dict[SignalColor, BGRColor] = {
            "RED": (242, 242, 255),
            "YELLOW": (242, 255, 255),
            "GREEN": (242, 255, 242),
        }
        for state, color in colors.items():
            with self.subTest(state=state):
                crop: BGRImage = np.full((9, 9, 3), 255, dtype=np.uint8)
                crop[4, 4] = color
                self.assertEqual(TrafficLightDetector.classify_by_color(crop), state)

    def test_white_black_gray_and_blue_have_no_target_color(self) -> None:
        for color in ((0, 0, 0), (255, 255, 255), (90, 90, 90), (255, 0, 0)):
            with self.subTest(color=color):
                crop: BGRImage = np.full((10, 10, 3), color, dtype=np.uint8)
                self.assertEqual(TrafficLightDetector.classify_by_color(crop), "UNKNOWN")

    def test_colored_rim_survives_white_clipped_center(self) -> None:
        crop: BGRImage = np.zeros((15, 15, 3), dtype=np.uint8)
        crop[4:11, 4:11] = (0, 0, 200)
        crop[5:10, 5:10] = (255, 255, 255)
        self.assertEqual(TrafficLightDetector.classify_by_color(crop), "RED")

    def test_tiny_bright_light_beats_large_dim_lens(self) -> None:
        crop: BGRImage = np.full((30, 30, 3), (0, 75, 0), dtype=np.uint8)
        crop[15, 15] = (0, 0, 255)
        self.assertEqual(TrafficLightDetector.classify_by_color(crop), "RED")

    def test_equal_color_counts_remain_unknown(self) -> None:
        crop: BGRImage = np.zeros((12, 12, 3), dtype=np.uint8)
        crop[3, 3] = COLORS["RED"]
        crop[8, 8] = COLORS["GREEN"]
        self.assertEqual(TrafficLightDetector.classify_by_color(crop), "UNKNOWN")

    def test_red_hue_wrap_is_supported(self) -> None:
        hsv: BGRImage = np.full((5, 5, 3), (179, 255, 255), dtype=np.uint8)
        crop: BGRImage = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.assertEqual(TrafficLightDetector.classify_by_color(crop), "RED")

    def test_underexposed_red_lamp_survives_the_previous_brightness_floor(self) -> None:
        for brightness in (60, 70, 85):
            with self.subTest(brightness=brightness):
                crop: BGRImage = make_housing(None)
                cv2.circle(crop, CENTERS["RED"], 6, (0, 0, brightness), -1)
                self.assertEqual(TrafficLightDetector.classify_by_color(crop), "RED")

    def test_tiny_active_fragment_survives_inside_its_dim_lens(self) -> None:
        for state, color in COLORS.items():
            with self.subTest(state=state):
                crop: BGRImage = make_housing(None)
                x, y = CENTERS[state]
                crop[y, x] = color
                self.assertEqual(TrafficLightDetector.classify_by_color(crop), state)

    def test_dim_rim_can_identify_a_bright_white_core(self) -> None:
        dim_colors: dict[SignalColor, BGRColor] = {
            "RED": (0, 0, 35),
            "YELLOW": (0, 35, 35),
            "GREEN": (0, 35, 0),
        }
        for state, color in dim_colors.items():
            with self.subTest(state=state):
                crop: BGRImage = np.zeros((25, 25, 3), dtype=np.uint8)
                crop[5:20, 5:20] = color
                self.assertEqual(TrafficLightDetector.classify_by_color(crop), "UNKNOWN")
                crop[7:18, 7:18] = (255, 255, 255)
                self.assertEqual(TrafficLightDetector.classify_by_color(crop), state)

    def test_red_rim_survives_a_compressed_yellow_core_below_clipping(self) -> None:
        for rim_value in (35, 70, 180):
            with self.subTest(rim_value=rim_value):
                crop: BGRImage = np.zeros((25, 25, 3), dtype=np.uint8)
                crop[5:20, 5:20] = (0, 0, rim_value)
                crop[7:18, 7:18] = (0, 210, 230)
                state, box = TrafficLightDetector.find_active_light(crop)
                self.assertEqual(state, "RED")
                self.assertIsNotNone(box)
                assert box is not None
                self.assertTrue(box[0] <= 12 < box[0] + box[2])
                self.assertTrue(box[1] <= 12 < box[1] + box[3])

    def test_a_white_speck_does_not_activate_an_unlit_colored_lens(self) -> None:
        for state in COLORS:
            with self.subTest(state=state):
                crop: BGRImage = make_housing(None)
                x, y = CENTERS[state]
                crop[y, x] = (255, 255, 255)
                self.assertEqual(TrafficLightDetector.classify_by_color(crop), "UNKNOWN")

    def test_disconnected_specks_do_not_pool_their_color_votes(self) -> None:
        crop: BGRImage = np.zeros((30, 30, 3), dtype=np.uint8)
        crop[15, 14:17] = COLORS["GREEN"]
        for x, y in ((3, 3), (25, 3), (3, 25), (25, 25)):
            crop[y, x] = COLORS["RED"]
        self.assertEqual(TrafficLightDetector.classify_by_color(crop), "GREEN")

    def test_overlap_without_enclosure_does_not_replace_the_lamp_color(self) -> None:
        crop: BGRImage = np.zeros((25, 25, 3), dtype=np.uint8)
        # An open U-shaped red region surrounds a yellow lamp's bounding box,
        # but is not a closed rim belonging to that lamp.
        crop[5:20, 5:7] = (0, 0, 80)
        crop[18:20, 5:20] = (0, 0, 80)
        crop[5:20, 18:20] = (0, 0, 80)
        crop[9:14, 9:14] = COLORS["YELLOW"]
        self.assertEqual(TrafficLightDetector.classify_by_color(crop), "YELLOW")

    def test_warm_traffic_amber_is_yellow(self) -> None:
        for hue in (10, 11, 12, 18, 25, 30, 35):
            with self.subTest(hue=hue):
                hsv: BGRImage = np.full((9, 9, 3), (hue, 220, 220), dtype=np.uint8)
                crop: BGRImage = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                self.assertEqual(TrafficLightDetector.classify_by_color(crop), "YELLOW")

    def test_red_and_green_hue_regressions(self) -> None:
        cases: dict[SignalColor, tuple[int, ...]] = {
            "RED": (0, 3, 6, 9, 170, 175, 179),
            "GREEN": (45, 60, 75, 90, 95),
        }
        for state, hues in cases.items():
            for hue in hues:
                with self.subTest(state=state, hue=hue):
                    hsv: BGRImage = np.full((9, 9, 3), (hue, 220, 220), dtype=np.uint8)
                    crop: BGRImage = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    self.assertEqual(TrafficLightDetector.classify_by_color(crop), state)

    def test_red_rim_overrides_a_clipped_yellow_looking_core(self) -> None:
        for core in ((230, 255, 255), (0, 255, 255)):
            with self.subTest(core=core):
                crop: BGRImage = np.zeros((25, 25, 3), dtype=np.uint8)
                crop[5:20, 5:20] = (0, 0, 180)
                crop[7:18, 7:18] = core
                state, box = TrafficLightDetector.find_active_light(crop)
                self.assertEqual(state, "RED")
                self.assertIsNotNone(box)
                assert box is not None
                x, y, width, height = box
                self.assertTrue(x <= 12 < x + width and y <= 12 < y + height)

    def test_yellow_rim_overrides_a_pale_red_looking_core(self) -> None:
        crop: BGRImage = np.zeros((25, 25, 3), dtype=np.uint8)
        crop[5:20, 5:20] = (0, 120, 200)
        crop[7:18, 7:18] = (242, 242, 255)
        self.assertEqual(TrafficLightDetector.classify_by_color(crop), "YELLOW")

    def test_bright_pale_reflection_does_not_hide_a_separate_lamp(self) -> None:
        cases: tuple[tuple[SignalColor, BGRColor, SignalColor, BGRColor], ...] = (
            ("RED", (0, 0, 180), "YELLOW", (242, 255, 255)),
            ("YELLOW", (0, 150, 200), "RED", (242, 242, 255)),
            ("GREEN", (0, 180, 0), "RED", (242, 242, 255)),
        )
        for expected, lamp_color, other, reflection_color in cases:
            with self.subTest(state=expected):
                crop: BGRImage = make_housing(None)
                cv2.circle(crop, CENTERS[expected], 6, lamp_color, -1)
                rx, ry = CENTERS[other]
                crop[ry, rx] = reflection_color
                self.assertEqual(TrafficLightDetector.classify_by_color(crop), expected)

    def test_slightly_brighter_red_speck_does_not_override_green(self) -> None:
        crop: BGRImage = make_housing(None)
        cv2.circle(crop, CENTERS["GREEN"], 6, (0, 170, 0), -1)
        rx, ry = CENTERS["RED"]
        crop[ry, rx] = (0, 0, 180)
        self.assertEqual(TrafficLightDetector.classify_by_color(crop), "GREEN")

    def test_small_red_lamp_beats_a_larger_dim_green_lens(self) -> None:
        crop: BGRImage = np.zeros((50, 50, 3), dtype=np.uint8)
        crop[20:45, 10:40] = (0, 140, 0)
        crop[5, 25] = (0, 0, 170)
        self.assertEqual(TrafficLightDetector.classify_by_color(crop), "RED")

    def test_minimum_pixels_can_reject_sensor_specks(self) -> None:
        crop: BGRImage = np.zeros((10, 10, 3), dtype=np.uint8)
        crop[5, 5] = COLORS["RED"]
        self.assertEqual(
            TrafficLightDetector.classify_by_color(
                crop, thresholds=ColorThresholds(min_pixels=3)
            ),
            "UNKNOWN",
        )

    def test_finds_one_inner_box_for_each_active_lamp(self) -> None:
        for expected in COLORS:
            with self.subTest(state=expected):
                state, box = TrafficLightDetector.find_active_light(make_housing(expected))
                self.assertEqual(state, expected)
                self.assertIsNotNone(box)
                assert box is not None
                x, y, width, height = box
                cx, cy = CENTERS[expected]
                self.assertTrue(x <= cx < x + width and y <= cy < y + height)
                self.assertLess(height, 30)

    def test_all_dark_lenses_have_no_active_lamp(self) -> None:
        self.assertEqual(
            TrafficLightDetector.find_active_light(make_housing(None)),
            ("UNKNOWN", None),
        )

    def test_inner_box_ignores_a_disconnected_same_color_speck(self) -> None:
        crop: BGRImage = make_housing("RED")
        crop[80, 18] = COLORS["RED"]
        state, box = TrafficLightDetector.find_active_light(crop)
        self.assertEqual(state, "RED")
        self.assertIsNotNone(box)
        assert box is not None
        self.assertLess(box[1] + box[3], 40)

    def test_lamp_padding_stays_inside_housing(self) -> None:
        crop: BGRImage = np.zeros((20, 20, 3), dtype=np.uint8)
        crop[0, 0] = COLORS["YELLOW"]
        state, box = TrafficLightDetector.find_active_light(crop)
        self.assertEqual(state, "YELLOW")
        self.assertIsNotNone(box)
        assert box is not None
        self.assertEqual(box[:2], (0, 0))
        self.assertLessEqual(box[2], 20)
        self.assertLessEqual(box[3], 20)


class HousingPipelineTests(unittest.TestCase):
    """Verify housing-first processing, lamp coordinates, and minimal annotation."""

    detector: TrafficLightDetector

    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = TrafficLightDetector()

    def test_no_housing_means_no_color_analysis(self) -> None:
        frame: BGRImage = np.full((120, 150, 3), COLORS["RED"], dtype=np.uint8)
        with (
            patch.object(self.detector, "detect_lights", return_value=[]),
            patch.object(self.detector, "find_active_light") as classify,
        ):
            annotated = self.detector.process_frame(frame)
        classify.assert_not_called()
        self.assertTrue(np.array_equal(annotated, frame))

    def test_external_colors_do_not_change_the_housing_state(self) -> None:
        frame: BGRImage = np.zeros((120, 150, 3), dtype=np.uint8)
        frame[20:110, 30:66] = make_housing("GREEN")
        frame[:, 90:150] = COLORS["RED"]
        housing: BoundingBox = (30, 20, 36, 90)
        with patch.object(self.detector, "detect_lights", return_value=[housing]):
            detections = self.detector.detect(frame)
        self.assertEqual(len(detections), 1)
        result = detections[0]
        self.assertEqual(result.box, housing)
        self.assertEqual(result.state, "GREEN")
        self.assertIsNotNone(result.lamp_box)
        assert result.lamp_box is not None
        x, y, width, height = result.lamp_box
        self.assertTrue(x <= 48 < x + width and y <= 95 < y + height)
        self.assertGreaterEqual(x, 30)
        self.assertGreaterEqual(y, 20)
        self.assertLessEqual(x + width, 66)
        self.assertLessEqual(y + height, 110)

    def test_two_boxes_one_message_and_input_is_unchanged(self) -> None:
        frame: BGRImage = np.zeros((120, 150, 3), dtype=np.uint8)
        frame[20:110, 30:66] = make_housing("RED")
        original: BGRImage = frame.copy()
        with (
            patch.object(self.detector, "detect_lights", return_value=[(30, 20, 36, 90)]),
            patch("traffic_light_detector.cv2.rectangle", wraps=cv2.rectangle) as rectangles,
            patch("traffic_light_detector.cv2.putText", wraps=cv2.putText) as text,
        ):
            annotated: BGRImage = self.detector.process_frame(frame)
        self.assertEqual(rectangles.call_count, 2)
        self.assertEqual({call.args[1] for call in text.call_args_list}, {"RED is on"})
        self.assertTrue(np.array_equal(frame, original))
        self.assertFalse(np.shares_memory(frame, annotated))
        self.assertFalse(np.array_equal(frame, annotated))

    def test_off_housing_has_only_outer_box_and_no_message(self) -> None:
        frame: BGRImage = make_housing(None)
        with (
            patch.object(self.detector, "detect_lights", return_value=[(0, 0, 36, 90)]),
            patch("traffic_light_detector.cv2.rectangle", wraps=cv2.rectangle) as rectangles,
            patch("traffic_light_detector.cv2.putText", wraps=cv2.putText) as text,
        ):
            self.detector.process_frame(frame)
        self.assertEqual(rectangles.call_count, 1)
        text.assert_not_called()

    def test_duplicate_haar_boxes_are_suppressed(self) -> None:
        frame: BGRImage = np.zeros((120, 100, 3), dtype=np.uint8)
        with patch.object(self.detector, "_cascade") as cascade:
            cascade.detectMultiScale.return_value = np.array(
                [(20, 10, 36, 90), (21, 11, 36, 90)], dtype=np.int32
            )
            boxes = self.detector.detect_lights(frame)
            self.assertEqual(cascade.detectMultiScale.call_count, 1)
        self.assertEqual(len(boxes), 1)

    def test_haar_retry_gives_equal_intensity_to_red_yellow_and_green(self) -> None:
        frame: BGRImage = np.zeros((120, 100, 3), dtype=np.uint8)
        frame[10:20, 10:20] = (0, 0, 180)
        frame[40:50, 10:20] = (0, 180, 0)
        frame[70:80, 10:20] = (0, 180, 180)
        with patch.object(self.detector, "_cascade") as cascade:
            cascade.detectMultiScale.side_effect = [
                (), np.array([(10, 0, 36, 90)], dtype=np.int32)
            ]
            boxes = self.detector.detect_lights(frame)
            gray = cascade.detectMultiScale.call_args_list[0].args[0]
            normalized = cascade.detectMultiScale.call_args_list[1].args[0]
        self.assertEqual(boxes, [(10, 0, 36, 90)])
        self.assertLess(int(gray[15, 15]), int(gray[45, 15]))
        self.assertEqual(int(normalized[15, 15]), int(normalized[45, 15]))
        self.assertEqual(int(normalized[15, 15]), int(normalized[75, 15]))
        self.assertGreater(int(normalized[15, 15]), int(normalized[0, 0]))

    def test_no_state_is_carried_forward_after_both_haar_scans_miss(self) -> None:
        frame: BGRImage = make_housing("RED")
        with patch.object(self.detector, "_cascade") as cascade:
            cascade.detectMultiScale.side_effect = [
                np.array([(0, 0, 36, 90)], dtype=np.int32), (), ()
            ]
            first = self.detector.detect(frame)
            second = self.detector.detect(frame)
        self.assertEqual(first[0].state, "RED")
        self.assertEqual(second, [])


if __name__ == "__main__":
    unittest.main()
