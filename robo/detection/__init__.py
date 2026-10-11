"""OpenCV panel detection API, reusable independently of the live CLI."""

from .panel_detector import DetectorConfig, FrameDetections, PanelDetection, PanelDetector

__all__ = ["DetectorConfig", "FrameDetections", "PanelDetection", "PanelDetector"]
