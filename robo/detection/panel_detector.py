"""Detect a front-facing panel from two red side lights in the current BGR frame."""

from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectorConfig:
    # Compare RGB channel intensities directly; input frames remain OpenCV BGR.
    min_red: int = 50
    min_red_difference: int = 20
    min_area: float = 45.0
    min_length: float = 18.0
    min_width: float = 2.0
    min_aspect_ratio: float = 4.0
    max_aspect_ratio: float = 12.0
    min_fill_ratio: float = 0.75
    max_edge_angle_difference: float = 6.0  # extracting an individual light
    max_angle_difference: float = 15.0  # pairing lights, modulo 180
    min_length_ratio: float = 0.60
    min_width_ratio: float = 0.40
    max_alignment_offset: float = 0.35  # along bars / mean bar length
    # Reference photo: bar aspect about 7, center spacing about 2.35 lengths.
    min_separation: float = 0.8
    max_separation: float = 4.5
    expected_separation: float = 2.35

    def __post_init__(self):
        if not 0 <= self.min_red <= 255 or not 1 <= self.min_red_difference <= 255:
            raise ValueError("Red minimum must be in [0, 255] and difference in [1, 255]")
        if (self.min_area <= 0 or self.min_length <= 0 or self.min_width <= 0
                or not 1 < self.min_aspect_ratio < self.max_aspect_ratio):
            raise ValueError("Area/length/width must be positive; require 1 < min aspect < max aspect")
        if not all(0 < v <= 1 for v in (
            self.min_fill_ratio, self.min_length_ratio, self.min_width_ratio
        )):
            raise ValueError("Fill, length and width ratios must be in (0, 1]")
        if (not 0 < self.max_angle_difference < 90
                or not 0 < self.max_edge_angle_difference < 90 or self.max_alignment_offset <= 0):
            raise ValueError("Angle must be in (0, 90) and alignment tolerance positive")
        if not 0 < self.min_separation < self.max_separation:
            raise ValueError("Require 0 < min_separation < max_separation")
        if not self.min_separation <= self.expected_separation <= self.max_separation:
            raise ValueError("Expected spacing must be inside the separation range")


@dataclass(frozen=True)
class LightBar:
    center: np.ndarray
    axis: np.ndarray  # unit vector along its long side; sign is arbitrary
    length: float
    width: float
    corners: np.ndarray


@dataclass(frozen=True)
class PanelDetection:
    corners: np.ndarray  # four cyclic corners of the rolled box, in image pixels
    bbox: Tuple[int, int, int, int]  # clipped axis-aligned (x, y, width, height)
    lights: Tuple[LightBar, LightBar]  # original individual boxes, unchanged
    pair_score: float  # geometric mismatch; lower is a stronger match


@dataclass(frozen=True)
class FrameDetections:
    panels: List[PanelDetection]
    light_bars: List[LightBar]
    red_mask: np.ndarray
    light_mask: np.ndarray

    @property
    def paired_light_bars(self) -> List[LightBar]:
        return [light for panel in self.panels for light in panel.lights]

    @property
    def diagnostics(self):
        return {
            "red_pixels": cv2.countNonZero(self.red_mask),
            "light_pixels": cv2.countNonZero(self.light_mask),
            "light_bars": len(self.light_bars),
            "paired_light_bars": len(self.paired_light_bars),
        }


class PanelDetector:
    """Stateless color/geometry detector. No tracking, prediction, or pose estimation."""

    def __init__(self, config: DetectorConfig = DetectorConfig()):
        self.config = config

    def red_mask(self, frame: np.ndarray) -> np.ndarray:
        """Select saturated red/orange LEDs, rejecting weak warm color casts."""
        blue, green, red = cv2.split(frame)
        c = self.config
        # Saturating subtraction avoids uint8 wraparound accepting blue/green.
        selected = ((red >= c.min_red)
                    & (cv2.subtract(red, green) >= np.maximum(c.min_red_difference, red * 0.20))
                    & (cv2.subtract(red, blue) >= np.maximum(c.min_red_difference, red * 0.45)))
        return selected.astype(np.uint8) * 255

    def _light_mask(self, frame: np.ndarray, seeds: np.ndarray) -> np.ndarray:
        """Join small gaps and include bright cores only near observed red."""
        blue, green, red = cv2.split(frame)
        strongest_other = cv2.max(blue, green)
        weakest_other = cv2.min(blue, green)
        weak_red = ((red >= self.config.min_red)
                    & (cv2.subtract(red, strongest_other)
                       >= max(3, self.config.min_red_difference // 3)))
        # A clipped LED core may be white, with its color preserved at the rim.
        bright_core = ((cv2.min(red, weakest_other) >= 180)
                       & (cv2.subtract(cv2.max(red, strongest_other),
                                       cv2.min(red, weakest_other)) <= 60)
                       & (cv2.subtract(strongest_other, red) <= 8))
        nearby = cv2.dilate(seeds, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = cv2.bitwise_or(seeds, ((weak_red | bright_core)
                                    & (nearby != 0)).astype(np.uint8) * 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        # Include wider white interiors enclosed by red rims. Leave dark holes
        # alone so hollow red background outlines do not become filled lights.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not np.any(bright_core[y:y + h, x:x + w]):
                continue
            support = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(support, [contour - np.array([[[x, y]]])], -1, 255, cv2.FILLED)
            interior = bright_core[y:y + h, x:x + w] & (support != 0)
            mask[y:y + h, x:x + w][interior] = 255
        # Clean isolated speckles after filling cores, preserving thin red rims.
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    def _bright_light_bars(self, frame: np.ndarray, seeds: np.ndarray):
        """Find clipped white cores with red long-edge halos, even open halos."""
        blue, green, red = cv2.split(frame)
        lowest = cv2.min(red, cv2.min(blue, green))
        highest = cv2.max(red, cv2.max(blue, green))
        bright = ((lowest >= 190) & (highest >= 240)
                  & (cv2.subtract(highest, lowest) <= 65)
                  & (cv2.subtract(cv2.max(blue, green), red) <= 8))
        bright_mask = bright.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        if cv2.countNonZero(bright_mask) < self.config.min_area:
            return [], np.zeros(seeds.shape, dtype=np.uint8)
        bars = []
        support = np.zeros(seeds.shape, dtype=np.uint8)
        for bar in self._light_bars(bright_mask):
            across = np.array([bar.axis[1], -bar.axis[0]], dtype=np.float32)
            coverages = []
            for side in (-1, 1):
                # Test several distances outside each observed white-core edge.
                # Coverage is measured along the bar, so an isolated red spot
                # next to white lettering cannot qualify the whole white bar.
                distances = np.linspace(0.52, 1.10, 12) * bar.width
                distances = np.maximum(distances, bar.width / 2 + 1)
                along_positions = np.linspace(-0.40, 0.40, 31) * bar.length
                points = (bar.center + along_positions[:, None, None] * bar.axis
                          + side * distances[None, :, None] * across)
                points = np.rint(points).astype(int)
                height, width = seeds.shape
                valid = ((points[..., 0] >= 0) & (points[..., 0] < width)
                         & (points[..., 1] >= 0) & (points[..., 1] < height))
                evidence = np.zeros(valid.shape, dtype=bool)
                evidence[valid] = seeds[points[..., 1][valid], points[..., 0][valid]] != 0
                coverages.append(float(np.mean(evidence.any(axis=1))))
            # One clear rim and some red on the opposite rim are sufficient.
            # Neither end needs a red cap, and the halo may join a red hand.
            if max(coverages) < 0.60 or min(coverages) < 0.20:
                continue
            bars.append(bar)
            cv2.fillConvexPoly(support, np.rint(bar.corners).astype(np.int32), 255)
        return bars, cv2.bitwise_and(bright_mask, support)

    @staticmethod
    def _sample(image, center, along, across, length, width, rows=31, columns=9):
        """Sample inside a rolled rectangle, without rotating the whole image."""
        u, v = np.meshgrid(np.linspace(-0.45, 0.45, columns) * width,
                           np.linspace(-0.45, 0.45, rows) * length)
        points = center + u[..., None] * across + v[..., None] * along
        points = np.rint(points).astype(int)
        height, image_width = image.shape[:2]
        valid = ((points[..., 0] >= 0) & (points[..., 0] < image_width)
                 & (points[..., 1] >= 0) & (points[..., 1] < height))
        return image[points[..., 1][valid], points[..., 0][valid]]

    def _bar(self, center, along, length, width, mask):
        c = self.config
        if (length < c.min_length or width < c.min_width or length * width < c.min_area
                or not c.min_aspect_ratio <= length / width <= c.max_aspect_ratio):
            return None
        across = np.array([along[1], -along[0]], dtype=np.float32)
        values = self._sample(mask, center, along, across, length, width)
        if values.size == 0 or np.mean(values != 0) < c.min_fill_ratio:
            return None
        corners = np.array([
            center - along * length / 2 - across * width / 2,
            center - along * length / 2 + across * width / 2,
            center + along * length / 2 + across * width / 2,
            center + along * length / 2 - across * width / 2,
        ], dtype=np.float32)
        return LightBar(center, along, length, width, corners)

    def _bars_from_edges(self, contour, mask):
        """Recover the strip between straight edges of a merged red component."""
        polygon = cv2.approxPolyDP(contour, min(2.0, max(0.8,
                                  cv2.arcLength(contour, True) * 0.003)), True)[:, 0]
        edges = []
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            delta = (end - start).astype(np.float32)
            length = float(np.linalg.norm(delta))
            if length >= max(4.0, self.config.min_length * 0.25):
                edges.append(((start + end).astype(np.float32) / 2, delta / length, length))
        # Bound the work for complicated red background contours.
        edges.sort(key=lambda edge: edge[2], reverse=True)
        edges = self._join_edges(edges[:64])
        edges.sort(key=lambda edge: edge[2], reverse=True)
        bars = []
        for first, second in combinations(edges[:24], 2):
            center_a, axis_a, length_a = first
            center_b, axis_b, length_b = second
            # Opposite contour directions identify opposing sides of a strip.
            if float(np.dot(axis_a, axis_b)) > -np.cos(np.radians(self.config.max_edge_angle_difference)):
                continue
            if min(length_a, length_b) / max(length_a, length_b) < 0.60:
                continue
            along = axis_a - axis_b
            along /= np.linalg.norm(along)
            across = np.array([along[1], -along[0]], dtype=np.float32)
            a, b = float(center_a @ along), float(center_b @ along)
            overlap = min(a + length_a / 2, b + length_b / 2) - max(
                a - length_a / 2, b - length_b / 2)
            if overlap < 0.65 * min(length_a, length_b):
                continue
            width = abs(float((center_a - center_b) @ across))
            # Use the longer observed edge; the other may be interrupted by a
            # touching red object. No extension beyond observed edge endpoints.
            longitudinal_center = a if length_a >= length_b else b
            center = along * longitudinal_center + across * float((center_a + center_b) @ across) / 2
            bar = self._bar(center, along, max(length_a, length_b), width, mask)
            if bar is not None:
                bars.append(bar)
        return bars

    def _join_edges(self, edges):
        """Reconnect nearly collinear boundary fragments around tiny bumps."""
        # Short rasterized fragments have noisier angles than complete bars.
        parallel = np.cos(np.radians(max(10.0, self.config.max_edge_angle_difference)))
        changed = True
        while changed:
            changed = False
            for i, j in combinations(range(len(edges)), 2):
                center_a, axis_a, length_a = edges[i]
                center_b, axis_b, length_b = edges[j]
                if float(axis_a @ axis_b) < parallel:
                    continue
                along = axis_a * length_a + axis_b * length_b
                along /= np.linalg.norm(along)
                across = np.array([along[1], -along[0]], dtype=np.float32)
                tolerance = max(2.0, min(4.0, max(length_a, length_b) * 0.04))
                if abs(float((center_a - center_b) @ across)) > tolerance:
                    continue
                gap = abs(float((center_a - center_b) @ along)) - (length_a + length_b) / 2
                if gap > max(10.0, min(length_a, length_b) * 0.10):
                    continue
                endpoints = np.array([center_a - axis_a * length_a / 2,
                                      center_a + axis_a * length_a / 2,
                                      center_b - axis_b * length_b / 2,
                                      center_b + axis_b * length_b / 2])
                projections = endpoints @ along
                lower, upper = float(projections.min()), float(projections.max())
                midpoint = (center_a * length_a + center_b * length_b) / (length_a + length_b)
                center = along * ((lower + upper) / 2) + across * float(midpoint @ across)
                edges[i] = (center, along, upper - lower)
                edges.pop(j)
                changed = True
                break
        return edges

    def _light_bars(self, mask: np.ndarray) -> List[LightBar]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bars = []
        c = self.config
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < c.min_area:
                continue
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect)
            edges = (corners[1] - corners[0], corners[2] - corners[1])
            sizes = [float(np.linalg.norm(edge)) for edge in edges]
            long_index = int(np.argmax(sizes))
            length, width = sizes[long_index], sizes[1 - long_index]
            if width <= 0 or length < c.min_length:
                continue
            bar = self._bar(np.asarray(rect[0], dtype=np.float32),
                            edges[long_index] / length, length, width, mask)
            if bar is not None:
                bars.append(bar)
            else:
                bars.extend(self._bars_from_edges(contour, mask))
        # A merged component can produce several fits of the same light.
        bars.sort(key=lambda bar: bar.length * bar.width, reverse=True)
        return self._distinct_bars(bars)

    @staticmethod
    def _distinct_bars(bars: List[LightBar]) -> List[LightBar]:
        """Keep the first proposal of each overlapping light, without refitting."""
        distinct = []
        for bar in bars:
            if any(cv2.intersectConvexConvex(bar.corners, other.corners)[0]
                   > 0.60 * min(bar.length * bar.width, other.length * other.width)
                   for other in distinct):
                continue
            distinct.append(bar)
        return distinct

    def _pair(self, first: LightBar, second: LightBar):
        """Score a plausible pair without modifying either observed light box."""
        c = self.config
        dot = float(np.clip(np.dot(first.axis, second.axis), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(abs(dot))))
        length_ratio = min(first.length, second.length) / max(first.length, second.length)
        width_ratio = min(first.width, second.width) / max(first.width, second.width)
        if (angle > c.max_angle_difference or length_ratio < c.min_length_ratio
                or width_ratio < c.min_width_ratio):
            return None

        # Match the signs before averaging axes, including across +/-90 degrees.
        along = first.axis + (second.axis if dot >= 0 else -second.axis)
        along /= np.linalg.norm(along)
        across = np.array([along[1], -along[0]], dtype=np.float32)
        delta = second.center - first.center
        mean_length = (first.length + second.length) / 2.0
        offset = abs(float(np.dot(delta, along))) / mean_length
        separation = abs(float(np.dot(delta, across))) / mean_length
        if offset > c.max_alignment_offset or not c.min_separation <= separation <= c.max_separation:
            return None
        # The side lights must be disjoint in the panel's horizontal direction.
        a = first.corners @ across
        b = second.corners @ across
        if min(float(a.max()), float(b.max())) >= max(float(a.min()), float(b.min())):
            return None

        return (angle / c.max_angle_difference + offset / c.max_alignment_offset
                + (1.0 - length_ratio) + (1.0 - width_ratio)
                + abs(separation - c.expected_separation))

    @staticmethod
    def _panel_extent(first: LightBar, second: LightBar) -> np.ndarray:
        """Combined extent for metadata, separate from the original light boxes."""
        dot = float(first.axis @ second.axis)
        along = first.axis + (second.axis if dot >= 0 else -second.axis)
        along /= np.linalg.norm(along)
        across = np.array([along[1], -along[0]], dtype=np.float32)
        points = np.concatenate((first.corners, second.corners))
        u, v = points @ across, points @ along
        return np.array([
            across * u.min() + along * v.min(),
            across * u.max() + along * v.min(),
            across * u.max() + along * v.max(),
            across * u.min() + along * v.max(),
        ], dtype=np.float32)

    def analyze(self, frame: np.ndarray) -> FrameDetections:
        """Return at most the best-scoring pair, candidates, and current masks.

        Input must be a nonempty uint8 BGR image. Boxes enclose the two lights;
        any unmarked panel material extending beyond them is not observable here.
        """
        if (frame is None or frame.dtype != np.uint8 or frame.ndim != 3
                or frame.shape[2] != 3 or frame.size == 0):
            raise ValueError("Expected a nonempty uint8 BGR frame")
        red_mask = self.red_mask(frame)
        mask = self._light_mask(frame, red_mask)
        core_bars, supported_cores = self._bright_light_bars(frame, red_mask)
        # Prefer a measured white core over its bloomed/skin-connected red halo.
        # Saturated red/orange lights still use the original color-mask path.
        bars = self._distinct_bars(core_bars + self._light_bars(mask))
        mask = cv2.bitwise_or(mask, supported_cores)
        pairs = []
        for i, j in combinations(range(len(bars)), 2):
            score = self._pair(bars[i], bars[j])
            if score is not None:
                pairs.append((score, i, j))

        # Select one global best match; all other pairs and orphan bars are hidden.
        detections = []
        height, width = frame.shape[:2]
        if pairs:
            score, i, j = min(pairs, key=lambda pair: pair[0])
            lights = (bars[i], bars[j])
            corners = self._panel_extent(*lights)
            lower = np.floor(corners.min(axis=0)).astype(int)
            upper = np.ceil(corners.max(axis=0)).astype(int) + 1
            x0, y0 = np.clip(lower, [0, 0], [width, height])
            x1, y1 = np.clip(upper, [0, 0], [width, height])
            detections.append(PanelDetection(corners, (int(x0), int(y0),
                                                      int(x1 - x0), int(y1 - y0)), lights, score))
        return FrameDetections(detections, bars, red_mask, mask)

    def detect(self, frame: np.ndarray) -> Tuple[List[PanelDetection], np.ndarray]:
        """Return current-frame panels and the processed light mask."""
        result = self.analyze(frame)
        return result.panels, result.light_mask


def draw_detections(frame: np.ndarray, detections: List[PanelDetection]) -> np.ndarray:
    """Draw only the best accepted pair, using its original two light boxes."""
    output = frame.copy()
    if not detections:
        return output
    detection = min(detections, key=lambda detection: detection.pair_score)
    for light in detection.lights:
        points = np.rint(light.corners).astype(np.int32)
        cv2.polylines(output, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
        x, y = points.min(axis=0)
        cv2.putText(output, "best pair", (int(x), max(15, int(y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return output
