import unittest
from unittest.mock import patch

import cv2
import numpy as np

from robo.detection.panel_detector import DetectorConfig, PanelDetector, draw_detections


def draw_panel(frame, center=(300, 300), roll=0, scale=1, color=(0, 0, 255)):
    radians = np.radians(roll)
    across = np.array([np.cos(radians), np.sin(radians)])
    center = np.array(center, dtype=float)
    points = []
    for side in (-1, 1):
        bar_center = center + side * 94 * scale * across
        corners = cv2.boxPoints((tuple(bar_center), (12 * scale, 80 * scale), roll))
        cv2.fillConvexPoly(frame, np.rint(corners).astype(np.int32), color)
        points.extend(corners)
    return np.array(points)


def reference_panel_scene(touching_object=False):
    """Synthetic scene using the attached photo's proportions, not its pixels."""
    frame = np.full((640, 640, 3), 175, dtype=np.uint8)
    body = np.array([(70, 115), (550, 115), (565, 130), (565, 552),
                     (550, 568), (70, 568), (55, 552), (55, 130)], dtype=np.int32)
    cv2.fillConvexPoly(frame, body, (28, 28, 28))
    marking = np.array([(145, 225), (210, 210), (245, 180), (325, 174),
                        (331, 209), (365, 214), (379, 245), (425, 233),
                        (433, 310), (340, 322), (306, 370), (316, 430),
                        (220, 430), (225, 391), (240, 377), (222, 358),
                        (150, 353)], dtype=np.int32)
    cv2.polylines(frame, [marking], True, (215, 215, 215), 12)
    cv2.polylines(frame, [np.array([(204, 460), (392, 460), (439, 519),
                                  (160, 519)], dtype=np.int32)], True, (215, 215, 215), 12)
    for x, y in ((70, 240), (532, 243)):
        points = np.array([(x + 7, y), (x + 21, y), (x + 28, y + 8),
                           (x + 28, y + 189), (x + 21, y + 197),
                           (x + 7, y + 197), (x, y + 189), (x, y + 8)], dtype=np.int32)
        cv2.fillConvexPoly(frame, points, (20, 10, 255))
        cv2.rectangle(frame, (x + 5, y + 10), (x + 23, y + 187), (15, 130, 255), -1)
    if touching_object:
        # A saturated red object touches the upper outside edge of the left
        # light. Its combined contour is no longer a narrow rectangle.
        cv2.rectangle(frame, (0, 238), (78, 292), (20, 20, 240), -1)
    return frame


def camera_feed_scene(touching_hand=False, touching_reflection=False):
    """Synthetic clipped cores/open halos from the camera screenshot's appearance."""
    frame = reference_panel_scene()
    if touching_hand:
        # The red glow on the hand joins the right halo but is not a white core.
        cv2.rectangle(frame, (563, 200), (639, 495), (80, 100, 245), -1)
    for x, y in ((70, 240), (532, 243)):
        cv2.rectangle(frame, (x - 3, y - 3), (x + 31, y + 200), (28, 28, 28), -1)
        # No red caps: two separate colored side strips, and a wide white center.
        cv2.rectangle(frame, (x - 6, y + 15), (x - 2, y + 182), (35, 45, 235), -1)
        cv2.rectangle(frame, (x + 30, y + 15), (x + 34, y + 182), (35, 45, 235), -1)
        white = np.array([(x + 7, y), (x + 21, y), (x + 28, y + 8),
                          (x + 28, y + 189), (x + 21, y + 197),
                          (x + 7, y + 197), (x, y + 189), (x, y + 8)], dtype=np.int32)
        cv2.fillConvexPoly(frame, white, (255, 255, 255))
    # A white reflection next to the upper right light, as in the screenshot.
    reflection_left = 559 if touching_reflection else 572
    cv2.rectangle(frame, (reflection_left, 225), (594, 275), (255, 255, 255), -1)
    return frame


class PanelDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = PanelDetector()

    def test_full_roll_translation_and_scale(self):
        for roll in range(0, 360, 5):
            for center, scale in [((300, 300), 1), ((180, 210), 0.6), ((370, 360), 1.3)]:
                with self.subTest(roll=roll, center=center, scale=scale):
                    frame = np.zeros((600, 600, 3), dtype=np.uint8)
                    expected = draw_panel(frame, center, roll, scale)
                    detections, _ = self.detector.detect(frame)
                    self.assertEqual(len(detections), 1)
                    polygon = detections[0].corners
                    # Allow rasterization/rectangle fit error at the light tips.
                    for point in expected:
                        distance = cv2.pointPolygonTest(polygon, tuple(point), True)
                        self.assertGreaterEqual(distance, -2.0)
                    self.assertAlmostEqual(cv2.contourArea(polygon),
                                           200 * 80 * scale ** 2, delta=1500 * scale)
                    x, y, w, h = detections[0].bbox
                    self.assertGreaterEqual(x, 0)
                    self.assertGreaterEqual(y, 0)
                    self.assertLessEqual(x + w, 600)
                    self.assertLessEqual(y + h, 600)

    def test_red_hue_variations(self):
        for hue in (0, 5, 175, 179):
            with self.subTest(hue=hue):
                color = tuple(int(v) for v in cv2.cvtColor(
                    np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0])
                frame = np.zeros((600, 600, 3), dtype=np.uint8)
                draw_panel(frame, color=color)
                self.assertEqual(len(self.detector.detect(frame)[0]), 1)

    def test_red_and_orange_led_colors(self):
        for color in ((0, 0, 50), (10, 25, 60), (15, 140, 255), (50, 150, 230)):
            with self.subTest(bgr=color):
                frame = np.zeros((600, 600, 3), dtype=np.uint8)
                draw_panel(frame, color=color)
                self.assertEqual(len(self.detector.detect(frame)[0]), 1)

    def test_rgb_mask_rejects_other_channels_without_uint8_wraparound(self):
        frame = np.array([[(250, 0, 40), (0, 250, 40), (250, 250, 255),
                           (0, 0, 49), (0, 0, 50), (10, 30, 60),
                           (100, 120, 170), (15, 150, 255)]], dtype=np.uint8)
        original = frame.copy()
        self.assertEqual(self.detector.red_mask(frame).tolist(), [[0, 0, 0, 0, 255, 255, 0, 255]])
        np.testing.assert_array_equal(frame, original)

    def test_bright_white_cores_with_red_rims_at_multiple_rolls(self):
        for roll in range(0, 180, 15):
            with self.subTest(roll=roll):
                frame = np.zeros((600, 600, 3), dtype=np.uint8)
                for x in (206, 394):
                    cv2.rectangle(frame, (x - 6, 260), (x + 6, 340), (255, 255, 255), -1)
                    cv2.rectangle(frame, (x - 6, 260), (x + 6, 340), (10, 20, 255), 1)
                frame = cv2.warpAffine(frame, cv2.getRotationMatrix2D((300, 300), roll, 1),
                                       (600, 600))
                result = self.detector.analyze(frame)
                self.assertEqual(len(result.panels), 1)
                self.assertEqual(len(result.light_bars), 2)
                self.assertGreater(result.diagnostics["light_pixels"],
                                   result.diagnostics["red_pixels"])

    def test_camera_white_cores_with_open_red_halos_and_hand_glow(self):
        for touching_hand, touching_reflection in ((False, False), (True, False), (True, True)):
            for roll in (0, 20, 90, 170, 270):
                for scale in (1.0, 0.6):
                    with self.subTest(hand=touching_hand, reflection=touching_reflection,
                                      roll=roll, scale=scale):
                        frame = camera_feed_scene(touching_hand, touching_reflection)
                        transform = cv2.getRotationMatrix2D((320, 320), roll, scale)
                        frame = cv2.warpAffine(frame, transform, (640, 640),
                                               borderValue=(175, 175, 175))
                        result = self.detector.analyze(frame)
                        self.assertEqual(len(result.panels), 1)
                        self.assertEqual(len(result.paired_light_bars), 2)
                        expected_centers = cv2.transform(np.array([[[84, 338.5], [546, 341.5]]],
                                                                  dtype=np.float32), transform)[0]
                        for light in result.panels[0].lights:
                            self.assertLess(min(np.linalg.norm(light.center - point)
                                                for point in expected_centers), 5)
                            self.assertLess(light.width, 35 * scale)

    def test_camera_white_cores_survive_blur_and_jpeg_compression(self):
        frame = cv2.GaussianBlur(camera_feed_scene(True), (3, 3), 0.8)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        self.assertTrue(ok)
        frame = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        result = self.detector.analyze(frame)
        self.assertEqual(len(result.panels), 1)
        self.assertEqual(len(result.paired_light_bars), 2)

    def test_nearby_red_spots_do_not_qualify_uncolored_white_bars(self):
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        draw_panel(frame, color=(255, 255, 255))
        for x in (206, 394):
            for side in (-1, 1):
                cv2.circle(frame, (x + side * 10, 300), 2, (0, 0, 255), -1)
        result = self.detector.analyze(frame)
        self.assertEqual(result.panels, [])
        self.assertEqual(result.light_bars, [])

    def test_small_gaps_are_joined_into_full_length_bars(self):
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        draw_panel(frame)
        for y in (280, 300, 320):
            frame[y:y + 2] = 0
        result = self.detector.analyze(frame)
        self.assertEqual(len(result.panels), 1)
        self.assertEqual(len(result.light_bars), 2)
        for bar in result.light_bars:
            self.assertAlmostEqual(bar.length, 80, delta=1)

    def test_weak_red_edges_are_included_only_next_to_red_seeds(self):
        frame = np.zeros((100, 140, 3), dtype=np.uint8)
        frame[20:80, 30:35] = (10, 10, 80)
        frame[20:80, 35:37] = (50, 50, 60)
        frame[20:80, 100:105] = (50, 50, 60)
        result = self.detector.analyze(frame)
        self.assertEqual(result.red_mask[40, 36], 0)
        self.assertEqual(result.light_mask[40, 36], 255)
        self.assertEqual(cv2.countNonZero(result.light_mask[:, 100:]), 0)

    def test_unseeded_white_objects_stay_out_of_light_mask(self):
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        draw_panel(frame, color=(255, 255, 255))
        cv2.circle(frame, (30, 30), 2, (0, 0, 255), -1)
        result = self.detector.analyze(frame)
        self.assertEqual(result.panels, [])
        self.assertEqual(result.light_bars, [])
        self.assertEqual(cv2.countNonZero(result.light_mask[100:]), 0)

    def test_single_bar_is_counted_but_not_drawn_or_accepted(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(frame, (70, 50), (80, 130), (0, 0, 255), -1)
        original = frame.copy()
        result = self.detector.analyze(frame)
        self.assertEqual(result.panels, [])
        self.assertEqual(result.diagnostics["light_bars"], 1)
        self.assertGreater(result.diagnostics["red_pixels"], 0)
        self.assertEqual(result.paired_light_bars, [])
        annotated = draw_detections(frame, result.panels)
        np.testing.assert_array_equal(annotated, original)
        np.testing.assert_array_equal(frame, original)

    def test_tiny_and_squat_noise_regions_are_rejected(self):
        for sizes in (((2, 6), (2, 6)), ((6, 12), (4, 8))):
            with self.subTest(sizes=sizes):
                frame = np.zeros((200, 200, 3), dtype=np.uint8)
                for center, size in zip(((80, 100), (105, 100)), sizes):
                    corners = cv2.boxPoints((center, size, 0))
                    cv2.fillConvexPoly(frame, np.rint(corners).astype(np.int32), (0, 0, 50))
                self.assertEqual(self.detector.analyze(frame).light_bars, [])

    def test_photo_proportions_and_white_marking_with_attached_red_object(self):
        for touching in (False, True):
            for roll in (0, 15, 45, 90, 135, 180, 270):
                for scale in (1.0, 0.5):
                    with self.subTest(touching=touching, roll=roll, scale=scale):
                        frame = reference_panel_scene(touching)
                        frame = cv2.warpAffine(frame, cv2.getRotationMatrix2D((320, 320), roll, scale),
                                               (640, 640), borderValue=(175, 175, 175))
                        result = self.detector.analyze(frame)
                        self.assertEqual(len(result.panels), 1)
                        self.assertEqual(len(result.light_bars), 2)
                        self.assertGreaterEqual(min(bar.length for bar in result.light_bars),
                                                175 * scale)

    def test_cluttered_scene_keeps_only_reference_lights(self):
        frame = reference_panel_scene(True)
        rng = np.random.default_rng(7)
        # Hundreds of saturated speckles, plus long warm-colored objects.
        for x, y, radius in rng.integers([0, 0, 1], [640, 640, 3], size=(400, 3)):
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), -1)
        cv2.rectangle(frame, (580, 190), (604, 360), (100, 120, 170), -1)
        for roll in (0, 35, 90, 150, 240):
            for scale in (1.0, 0.5):
                with self.subTest(roll=roll, scale=scale):
                    transformed = cv2.warpAffine(frame,
                        cv2.getRotationMatrix2D((320, 320), roll, scale),
                        (640, 640), borderValue=(175, 175, 175))
                    result = self.detector.analyze(transformed)
                    self.assertEqual(len(result.panels), 1)
                    self.assertEqual(len(result.light_bars), 2)

    def test_noise_only_scene_has_no_candidate_bars_or_panels(self):
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        rng = np.random.default_rng(17)
        for x, y, radius in rng.integers([0, 0, 1], [600, 600, 4], size=(500, 3)):
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), -1)
        # Old low-saturation thresholds accepted these full-size false lights.
        draw_panel(frame, color=(100, 120, 170))
        result = self.detector.analyze(frame)
        self.assertEqual(result.panels, [])
        self.assertEqual(result.light_bars, [])

    def test_matching_bars_are_accepted_independent_of_background_brightness(self):
        frame = np.full((600, 600, 3), 200, dtype=np.uint8)
        draw_panel(frame)
        result = self.detector.analyze(frame)
        self.assertEqual(len(result.light_bars), 2)
        self.assertEqual(len(result.panels), 1)

    def test_loose_pair_geometry_keeps_light_detection_unchanged(self):
        strict = PanelDetector(DetectorConfig(max_angle_difference=6, min_length_ratio=0.8,
            min_width_ratio=0.6, max_alignment_offset=0.12, min_separation=1.9,
            max_separation=2.8))
        for second in [((398, 300), (12, 80), 12),  # approximately parallel
                       ((398, 300), (6, 56), 0),   # moderately different size
                       ((398, 320), (12, 80), 0),  # modest vertical offset
                       ((470, 300), (12, 80), 0)]:  # wider spacing
            with self.subTest(second=second):
                frame = np.zeros((600, 600, 3), dtype=np.uint8)
                for rect in [((210, 300), (12, 80), 0), second]:
                    cv2.fillConvexPoly(frame, np.rint(cv2.boxPoints(rect)).astype(np.int32),
                                       (0, 0, 255))
                loose_result = self.detector.analyze(frame)
                strict_result = strict.analyze(frame)
                self.assertEqual(len(loose_result.panels), 1)
                self.assertEqual(strict_result.panels, [])
                self.assertEqual(len(loose_result.light_bars), len(strict_result.light_bars))
                for loose_bar, strict_bar in zip(loose_result.light_bars, strict_result.light_bars):
                    np.testing.assert_array_equal(loose_bar.corners, strict_bar.corners)

    def test_pairing_preserves_original_boxes_and_draws_no_orphan_or_combined_box(self):
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        draw_panel(frame)
        cv2.rectangle(frame, (85, 50), (95, 110), (0, 0, 255), -1)
        bars = self.detector.analyze(frame).light_bars
        self.assertEqual(len(bars), 3)
        snapshots = [bar.corners.copy() for bar in bars]
        for bar in bars:
            for array in (bar.center, bar.axis, bar.corners):
                array.setflags(write=False)
        with patch.object(self.detector, "_light_bars", return_value=bars):
            result = self.detector.analyze(frame)
        self.assertEqual(len(result.panels), 1)
        self.assertEqual(len(result.paired_light_bars), 2)
        self.assertEqual(len({id(bar) for bar in result.paired_light_bars}), 2)
        for accepted in result.paired_light_bars:
            self.assertTrue(any(accepted is bar for bar in bars))
        annotated = draw_detections(frame, result.panels)
        for bar, snapshot in zip(bars, snapshots):
            np.testing.assert_array_equal(bar.corners, snapshot)
        np.testing.assert_array_equal(annotated[35:120, 70:110], frame[35:120, 70:110])
        # No green connecting/enclosing rectangle between the two original boxes.
        np.testing.assert_array_equal(annotated[255:265, 270:330], frame[255:265, 270:330])
        self.assertTrue(np.any(annotated != frame))

    def test_reference_spacing_rejects_unrelated_distant_bars(self):
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        for x in (100, 500):
            cv2.rectangle(frame, (x - 6, 260), (x + 6, 340), (0, 0, 255), -1)
        result = self.detector.analyze(frame)
        self.assertEqual(len(result.light_bars), 2)
        self.assertEqual(result.panels, [])

    def test_missing_light_and_blank_frame_clear_detection(self):
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        draw_panel(frame)
        self.assertEqual(len(self.detector.detect(frame)[0]), 1)
        frame[:, 300:] = 0
        self.assertEqual(self.detector.detect(frame)[0], [])
        frame[:] = 0
        self.assertEqual(self.detector.detect(frame)[0], [])

    def test_multiple_panels_return_and_draw_only_the_best_pair(self):
        frame = np.zeros((600, 900, 3), dtype=np.uint8)
        # A larger, less aligned valid pair is encountered first in extraction.
        weaker_frame = np.zeros_like(frame)
        for rect in [((100, 190), (16, 100), 0), ((335, 210), (16, 100), 0)]:
            cv2.fillConvexPoly(weaker_frame, np.rint(cv2.boxPoints(rect)).astype(np.int32),
                               (0, 0, 255))
        best_frame = np.zeros_like(frame)
        draw_panel(best_frame, (650, 400))
        frame = cv2.bitwise_or(weaker_frame, best_frame)
        weaker = self.detector.analyze(weaker_frame).panels[0]
        expected = self.detector.analyze(best_frame).panels[0]
        self.assertLess(expected.pair_score, weaker.pair_score)
        result = self.detector.analyze(frame)
        self.assertEqual(len(result.light_bars), 4)
        self.assertEqual(len(result.panels), 1)
        self.assertEqual(result.diagnostics["paired_light_bars"], 2)
        self.assertAlmostEqual(result.panels[0].pair_score, expected.pair_score)
        np.testing.assert_array_equal(result.panels[0].corners, expected.corners)
        # Even when handed multiple detections, the renderer selects by score.
        rendered = draw_detections(frame, [weaker, expected])
        np.testing.assert_array_equal(rendered[:280, :430], frame[:280, :430])
        np.testing.assert_array_equal(rendered, draw_detections(frame, result.panels))
        self.assertTrue(np.any(rendered != frame))

    def test_reject_wrong_color_and_round_blobs(self):
        for color in ((255, 0, 0), (0, 255, 0), (255, 255, 255), (0, 0, 10)):
            frame = np.zeros((600, 600, 3), dtype=np.uint8)
            draw_panel(frame, color=color)
            self.assertEqual(self.detector.detect(frame)[0], [])
        frame[:] = 0
        for center in ((220, 300), (380, 300)):
            cv2.circle(frame, center, 20, (0, 0, 255), -1)
        self.assertEqual(self.detector.detect(frame)[0], [])

    def test_reject_bad_pair_geometry(self):
        cases = [((380, 350), (12, 80), 0),  # shifted along the long axis
                 ((380, 300), (12, 40), 0),  # unequal lengths
                 ((380, 300), (12, 80), 30),  # nonparallel
                 ((240, 300), (12, 80), 0)]  # insufficient separation
        for second_rect in cases:
            with self.subTest(second_rect=second_rect):
                frame = np.zeros((600, 600, 3), dtype=np.uint8)
                for rect in [((220, 300), (12, 80), 0), second_rect]:
                    cv2.fillConvexPoly(frame, np.rint(cv2.boxPoints(rect)).astype(np.int32),
                                       (0, 0, 255))
                self.assertEqual(self.detector.detect(frame)[0], [])

    def test_hollow_red_outlines_are_rejected(self):
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        for x in (220, 380):
            cv2.rectangle(frame, (x - 10, 260), (x + 10, 340), (0, 0, 255), 1)
        self.assertEqual(self.detector.detect(frame)[0], [])

    def test_invalid_input_and_config(self):
        for frame in (None, np.zeros((10, 10), dtype=np.uint8),
                      np.zeros((0, 10, 3), dtype=np.uint8),
                      np.zeros((10, 10, 3), dtype=float)):
            with self.assertRaises(ValueError):
                self.detector.detect(frame)
        with self.assertRaises(ValueError):
            DetectorConfig(min_separation=4, max_separation=2)
        with self.assertRaises(ValueError):
            DetectorConfig(min_red_difference=0)


if __name__ == "__main__":
    unittest.main()
