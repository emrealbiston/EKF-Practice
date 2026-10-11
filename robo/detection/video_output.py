"""Show an annotated live MJPEG feed and save per-frame JSON detection logs."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Condition, Thread

import cv2


class DetectionLog:
    """Flush one JSON record per observed camera frame, including empty lists."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = path.open("x", encoding="utf-8")

    def __enter__(self):
        return self

    def write(self, frame_index, timestamp, elapsed_seconds, detections, diagnostics=None):
        record = {
            "frame": frame_index,
            "timestamp": timestamp.isoformat(timespec="milliseconds"),
            "elapsed_seconds": elapsed_seconds,
            "detection_count": len(detections),
            "detections": [
                {"bbox": list(detection.bbox), "corners": detection.corners.tolist(),
                 "pair_score": detection.pair_score,
                 "lights": [{"corners": light.corners.tolist()} for light in detection.lights]}
                for detection in detections
            ],
        }
        if diagnostics is not None:
            record["diagnostics"] = diagnostics
        self.file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.file.flush()
        details = (" | red_pixels={} | light_pixels={} | light_bars={} | paired_light_bars={}".format(
            diagnostics["red_pixels"], diagnostics["light_pixels"], diagnostics["light_bars"],
            diagnostics["paired_light_bars"])
            if diagnostics is not None else "")
        print("Frame {} | {} | {:.3f}s | detections={} | boxes={}".format(
            frame_index, record["timestamp"], elapsed_seconds, len(detections),
            [detection.bbox for detection in detections]) + details, flush=True)

    def __exit__(self, *_):
        self.file.close()


class LiveFeed:
    """Serve the latest frame, without queuing old frames for slow viewers."""

    def __init__(self, port=8080):
        self.condition = Condition()
        self.jpeg = None
        self.sequence = 0
        self.closed = False
        feed = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.connection.settimeout(2.0)
                if self.path == "/":
                    page = (b'<!doctype html><html><head><title>Panel detections</title>'
                            b'</head><body style="margin:0;background:#111;text-align:center">'
                            b'<img src="/stream.mjpg" alt="Live panel detections" '
                            b'style="max-width:100%;height:auto"></body></html>')
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                elif self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    sequence = -1
                    try:
                        while True:
                            with feed.condition:
                                feed.condition.wait_for(
                                    lambda: feed.closed or (
                                        feed.jpeg is not None and feed.sequence != sequence)
                                )
                                if feed.closed:
                                    return
                                jpeg, sequence = feed.jpeg, feed.sequence
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n"
                            )
                            self.wfile.flush()
                    except (OSError, TimeoutError):
                        return  # Viewer disconnected or stopped consuming frames.
                else:
                    self.send_error(404)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return "http://127.0.0.1:{}".format(self.server.server_port)

    def __enter__(self):
        self.thread.start()
        return self

    def publish(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Cannot encode live-feed frame")
        with self.condition:
            self.jpeg = encoded.tobytes()
            self.sequence += 1
            self.condition.notify_all()

    def __exit__(self, *_):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
