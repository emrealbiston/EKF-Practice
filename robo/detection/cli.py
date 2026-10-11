"""Run pure detection with live bounding boxes and JSON logs."""

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cv2

from .panel_detector import PanelDetector, draw_detections
from .video_output import DetectionLog, LiveFeed


CAMERA_INDEX = 0
STREAM_PORT = 8080
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def run():
    detector = PanelDetector()
    capture = cv2.VideoCapture(CAMERA_INDEX)
    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot open camera {}".format(CAMERA_INDEX))
        # Preserve the camera's exposure. Forcing the hardware minimum can make
        # usable red LEDs too dim; a frame cannot recover their clipped color.
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Camera opened but did not provide a frame")
        started = perf_counter()
        frame_timestamp = datetime.now(timezone.utc)
        frame_elapsed = 0.0
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
        log_path = OUTPUT_DIR / ("panel-" + stamp + ".jsonl")
        with LiveFeed(STREAM_PORT) as feed, DetectionLog(log_path) as detection_log:
            print("Live feed: " + feed.url)
            print("Detections: " + str(log_path))
            print("Green: the best pair's two original boxes. All other bars are hidden.")
            print("Press m for the light mask; q, Escape, or Ctrl+C to stop.")
            frame_index = 0
            show_mask = False
            while True:
                result = detector.analyze(frame)
                detections = result.panels
                diagnostics = result.diagnostics
                detection_log.write(frame_index, frame_timestamp, frame_elapsed,
                                    detections, diagnostics)
                annotated = draw_detections(frame, detections)
                cv2.putText(annotated, "Frame {} | {:.2f}s | red: {} | paired lights: {} | pairs: {}".format(
                    frame_index, frame_elapsed, diagnostics["red_pixels"],
                    diagnostics["paired_light_bars"], len(detections)), (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                feed.publish(annotated)
                cv2.imshow("Red panel detection", annotated)
                if show_mask:
                    cv2.imshow("Light mask", result.light_mask)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("m"):
                    show_mask = not show_mask
                    if not show_mask:
                        cv2.destroyWindow("Light mask")
                if (key in (27, ord("q"))
                        or cv2.getWindowProperty("Red panel detection", cv2.WND_PROP_VISIBLE) < 1):
                    break
                ok, frame = capture.read()
                if not ok:
                    print("Camera feed ended.")
                    break
                frame_elapsed = perf_counter() - started
                frame_timestamp = datetime.now(timezone.utc)
                frame_index += 1
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main():
    try:
        run()
    except (RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from error

