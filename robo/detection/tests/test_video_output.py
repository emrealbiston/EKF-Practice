from datetime import datetime, timezone
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from robo.detection import cli as detection_cli
from robo.detection.panel_detector import PanelDetector, draw_detections
from .test_panel_detector import camera_feed_scene, draw_panel
from robo.detection.video_output import DetectionLog, LiveFeed


def annotated_panel(shape=(300, 400, 3), roll=0):
    frame = np.zeros(shape, dtype=np.uint8)
    draw_panel(frame, (200, 150), roll, 0.8)
    detections, _ = PanelDetector().detect(frame)
    return draw_detections(frame, detections)


def green_pixel_count(frame):
    b, g, r = cv2.split(frame.astype(np.int16))
    return np.count_nonzero((g > 100) & (g > b + 50) & (g > r + 50))


class VideoOutputTests(unittest.TestCase):
    def test_live_http_feed_exports_updated_annotated_frames(self):
        with LiveFeed(port=0) as feed:
            connection = HTTPConnection("127.0.0.1", feed.server.server_port, timeout=3)
            try:
                connection.request("GET", "/")
                page = connection.getresponse()
                self.assertEqual(page.status, 200)
                self.assertIn(b'/stream.mjpg', page.read())
                feed.publish(annotated_panel())
                connection.request("GET", "/stream.mjpg")
                stream = connection.getresponse()
                self.assertEqual(stream.status, 200)
                self.assertIn("multipart/x-mixed-replace", stream.getheader("Content-Type"))
                for expected_boxes in (True, False):
                    self.assertEqual(stream.readline(), b"--frame\r\n")
                    self.assertEqual(stream.readline(), b"Content-Type: image/jpeg\r\n")
                    size = int(stream.readline().split(b":")[1])
                    self.assertEqual(stream.readline(), b"\r\n")
                    jpeg = stream.read(size)
                    self.assertEqual(stream.read(2), b"\r\n")
                    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    self.assertEqual(image.shape, (300, 400, 3))
                    self.assertEqual(green_pixel_count(image) > 100, expected_boxes)
                    if expected_boxes:
                        feed.publish(np.zeros((300, 400, 3), dtype=np.uint8))
            finally:
                connection.close()
        self.assertFalse(feed.thread.is_alive())

    def test_main_shows_live_boxes_and_saves_only_json_on_camera_end_and_ctrl_c(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as directory:
                capture = Mock()
                capture.isOpened.return_value = True
                capture.getBackendName.return_value = "AVFOUNDATION"
                capture.set.return_value = False
                raw = np.zeros((300, 400, 3), dtype=np.uint8)
                draw_panel(raw, (200, 150), 35, 0.8)
                if not interrupted:
                    raw = camera_feed_scene(touching_hand=True, touching_reflection=True)
                capture.read.side_effect = ([(True, raw), KeyboardInterrupt()] if interrupted else
                                            [(True, raw), (True, np.zeros_like(raw)), (False, None)])
                previews = []
                clock = [100] if interrupted else [100, 100 + 1 / 30]
                with patch.object(detection_cli.cv2, "VideoCapture", return_value=capture), \
                        patch.object(detection_cli.cv2, "imshow",
                                     side_effect=lambda _, frame: previews.append(frame.copy())), \
                        patch.object(detection_cli.cv2, "waitKey", return_value=-1), \
                        patch.object(detection_cli.cv2, "getWindowProperty", return_value=1), \
                        patch.object(detection_cli.cv2, "destroyAllWindows") as destroy, \
                        patch.object(detection_cli.cv2, "VideoWriter",
                                     side_effect=AssertionError("Video recording is disabled")), \
                        patch.object(detection_cli, "STREAM_PORT", 0), \
                        patch.object(detection_cli, "OUTPUT_DIR", Path(directory)), \
                        patch.object(detection_cli, "perf_counter", side_effect=clock), \
                        patch("builtins.print"):
                    detection_cli.run()
                capture.release.assert_called_once()
                capture.set.assert_not_called()  # Preserve the current exposure.
                destroy.assert_called_once()
                paths = list(Path(directory).iterdir())
                self.assertEqual(len(paths), 1)
                self.assertEqual(paths[0].suffix, ".jsonl")
                self.assertEqual(len(previews), 1 if interrupted else 2)
                self.assertGreater(green_pixel_count(previews[0]), 100)
                if not interrupted:
                    self.assertEqual(green_pixel_count(previews[1]), 0)
                with paths[0].open() as file:
                    records = [json.loads(line) for line in file]
                self.assertEqual(len(records), len(previews))
                self.assertEqual(records[0]["frame"], 0)
                self.assertEqual(records[0]["detection_count"], 1)
                self.assertEqual(records[0]["diagnostics"]["light_bars"], 2)
                self.assertEqual(records[0]["diagnostics"]["paired_light_bars"], 2)
                self.assertGreater(records[0]["diagnostics"]["red_pixels"], 0)
                if not interrupted:
                    self.assertEqual(records[1]["frame"], 1)
                    self.assertEqual(records[1]["detections"], [])
                    self.assertEqual(records[1]["diagnostics"],
                                     {"red_pixels": 0, "light_pixels": 0, "light_bars": 0,
                                      "paired_light_bars": 0})
                    self.assertAlmostEqual(records[1]["elapsed_seconds"], 1 / 30)

    def test_detection_log_is_timestamped_and_flushed_for_all_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.jsonl"
            timestamp = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            raw = np.zeros((300, 400, 3), dtype=np.uint8)
            draw_panel(raw, (200, 150), 40, 0.8)
            detections, _ = PanelDetector().detect(raw)
            with DetectionLog(path) as log, patch("builtins.print") as console:
                log.write(0, timestamp, 0.0, detections)
                log.write(1, timestamp, 0.15, [])
                # Read before close to prove records are visible during a session.
                records = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(console.call_count, 2)
            self.assertEqual([record["frame"] for record in records], [0, 1])
            self.assertEqual(records[0]["timestamp"], "2026-01-01T12:00:00.000+00:00")
            self.assertEqual(records[0]["detection_count"], 1)
            self.assertEqual(records[0]["detections"][0]["bbox"], list(detections[0].bbox))
            self.assertEqual(records[0]["detections"][0]["corners"], detections[0].corners.tolist())
            self.assertEqual(records[0]["detections"][0]["pair_score"], detections[0].pair_score)
            self.assertEqual(records[0]["detections"][0]["lights"],
                             [{"corners": light.corners.tolist()} for light in detections[0].lights])
            self.assertEqual(records[1]["elapsed_seconds"], 0.15)
            self.assertEqual(records[1]["detection_count"], 0)
            self.assertEqual(records[1]["detections"], [])

    def test_missing_camera_creates_no_exports_and_releases_capture(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with patch.object(detection_cli.cv2, "VideoCapture", return_value=capture), \
                patch.object(detection_cli.cv2, "destroyAllWindows"), \
                patch.object(detection_cli, "LiveFeed") as feed, \
                patch.object(detection_cli, "DetectionLog") as log:
            with self.assertRaisesRegex(RuntimeError, "Cannot open camera"):
                detection_cli.run()
        capture.release.assert_called_once()
        feed.assert_not_called()
        log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
