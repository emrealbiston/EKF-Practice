# Red panel detection

An OpenCV-only, per-frame detector for a front-facing panel with **two red light
bars total**, one on each side. The bars are vertical in the panel's coordinates
and arranged horizontally across a mostly black plate. Defaults are based on
the supplied reference photo: bars approximately seven times as long as they
are wide, with center spacing approximately 2.35 times their length. The white
central marking is allowed but is not matched as a template. The actual camera
reference shows clipped white LED cores with red halos; these are detected using
a separate bright-core path as well as the red/orange color path.
The panel can translate and roll through
360 degrees. Both lights must be visible and separated.

No ML, EKF, tracking, smoothing, motion prediction, or pose/distance estimation
is used. Every box comes from the current frame and disappears when the pair is
no longer detected.

## Run

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.9
or newer, and a desktop OpenCV build for the preview windows.

```sh
uv sync --locked
uv run robo detect
```

The default camera opens automatically. Green outlines mark only the two
original light boxes of the best-scoring accepted pair, labeled `best pair`, in the
preview and in the live browser feed at **http://127.0.0.1:8080**. The live MJPEG
stream is also available at `http://127.0.0.1:8080/stream.mjpg` for other local
software. The feed is available while the detector is running.
Unpaired candidates and other valid pairs have no drawn box. At most one pair
(two original light boxes) is drawn and logged per frame. Candidate bars are
still counted for diagnostics.
Pair checking accepts or rejects existing boxes without resizing, rotating,
moving, or aligning them.

The CLI preserves the camera's existing exposure. It no longer forces the
hardware minimum, which can make useful lights too dim. If the lights turn
completely white, reduce exposure with your camera's controls until red/orange
edges remain visible. White cores are accepted when red halos remain, including
halos that are open at the tips or merge with nearby skin glow. Darkening a
captured frame cannot recover clipped color.

Press `m` in the preview to toggle the processed light mask: white pixels are
included in light detection, and black pixels are excluded.
Press `q` or Escape in the preview, close the preview window, or press Ctrl+C
in the terminal to stop. Video is displayed live with bounding boxes and a
detection count; no video file is recorded.

The only saved output is **`robo/detection/outputs/panel-<UTC timestamp>.jsonl`**, with one
JSON record per processed camera frame, including frames with zero detections.
Records contain a zero-based frame number, UTC timestamp at camera-frame receipt,
elapsed seconds, detection count, and every detected box and its four corners.
The detection list contains zero or one pair. Its `lights` list contains two entries with
the unchanged original light-box `corners`; the top-level `bbox` and `corners`
retain the combined panel extent as metadata.
`pair_score` is a geometric mismatch score: lower means better parallelism,
alignment, size agreement, and spacing. It is a ranking heuristic, not a
calibrated probability. The lowest score among valid pairs wins each frame.
Each record also has `diagnostics` with `red_pixels` (strong red evidence),
`light_pixels` (after including nearby weak red/bright cores and closing small
gaps), `light_bars` (all candidates passing the shape checks), and
`paired_light_bars` (only accepted lights, two per detection).
`bbox` is `[x, y, width, height]` in pixels. Every record is flushed immediately,
so the file can be inspected while the detector runs. The terminal prints per-frame counts and
boxes and diagnostic counts. The preview shows frame number, time, red-pixel
count, accepted-light count, and pair count.

If no panels appear, use the counts to find the failing stage:

- `red_pixels = 0`: no sufficiently red pixels were found in the frame.
- Red pixels but `light_bars = 0`: the colored regions failed the bar shape checks.
- `light_bars = 1`: only one usable side light was found.
- At least two bars but no panels: the bars failed the pair geometry checks.

Run commands from the repository root. Dependencies are declared in the shared
root `pyproject.toml` and pinned in `uv.lock`. uv manages the root `.venv`
environment; no manual activation is needed. This package can be used alongside
future EKF code without starting the live detection CLI.

## Algorithm

1. Split the BGR frame into its red, green, and blue channels. Select saturated
   red/orange pixels with `R >= 50`, `R - G >= max(20, 0.20R)`, and
   `R - B >= max(20, 0.45R)`. Orange LED centers remain valid; weak warm casts
   and nearly white reflections do not seed detection. No HSV hue cutoff is used.
2. Include weak red edges and bright, nearly white cores within two pixels of
   red evidence. Close small gaps with a 3×3 elliptical kernel. Include wider
   bright interiors enclosed by red rims, keeping dark holes unfilled. Remove
   isolated speckles with a 3×3 elliptical opening after filling bright cores.
3. Independently detect bright, nearly white cores: minimum channel intensity
   190, maximum channel intensity at least 240, and channel spread at most 65.
   Find bar-shaped regions and check bands outside both long edges for strong
   red pixels. Require red alongside at least 60% of one edge and 20% of the
   opposite edge, sampled over the middle 80% of the bar's length. The halo
   need not close around the tips. Prefer this measured core box to an
   overlapping red-halo box, so broad bloom or a red hand does not enlarge it.
   Uncolored white objects or a few nearby red spots do not qualify. Supported
   cores are also included in the mask shown with `m`.
4. Find contours and their minimum-area rectangles. Keep long, narrow, filled
   light regions as candidates. If another red object joins a light and ruins
   its contour rectangle, look for opposing straight contour edges enclosing
   a filled strip. Reconnect collinear edge fragments separated by small bumps,
   and discard duplicate fits of the same light.
5. Pair bars with similar lengths/widths and parallel long axes. Project their
   center separation onto the shared long axis and its perpendicular. Require
   roughly aligned centers along the bars and broad spacing across them. These
   checks work in rolled panel coordinates, including at 90 and 180 degrees.
   Panel-center brightness is not an acceptance condition.
6. Select only the global lowest-scoring valid pair each frame. Retain its two
   original light boxes unchanged and hide every other candidate. Compute a
   separate combined extent for downstream metadata after the pair is accepted.

The two green outlines enclose the individual observed lights. The combined
extent in JSON also includes the region between them. The reference plate
extends above and below the lights; these unmarked edges are not extrapolated.

The implementation uses OpenCV channel operations and
[contour/rectangle operations](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html).

## Use in code

```python
from robo.detection import PanelDetector

detector = PanelDetector()
detections, mask = detector.detect(frame)  # uint8 BGR image; processed light mask
for detection in detections:
    for light in detection.lights:
        corners = light.corners  # original (4, 2) float32 rotated light box
    x, y, width, height = detection.bbox  # combined extent, clipped to the frame

# Use this instead of detect() when candidate bars/diagnostics are needed.
result = detector.analyze(frame)
print(result.diagnostics)
```

## Defaults

The default center spacing is between 0.8 and 4.5 times the average light length,
with 2.35 preferred when several possible pairs exist. Bars need at least
45 pixels² of contour area and rectangle area, 18 pixels of length, 2 pixels
of width, a length/width ratio between 4 and 12, and 75% sampled fill.
The shorter bar must be at least 60% of the longer bar's length; the narrower
bar must be at least 40% of the wider bar's width. Long axes must differ by at
most 15 degrees and centers must align along the bars within 35% of their mean
length. The light boxes must be separated across the panel. Extraction of an
individual bar still uses a separate 6-degree edge tolerance; loosening pair
checks does not change that extraction or its boxes. Very small lights are
deliberately rejected to suppress noise.
Color and shape thresholds use reasonable defaults in `DetectorConfig`; there
are no CLI flags. If needed, edit `CAMERA_INDEX` or `STREAM_PORT` at the top of
`robo/detection/cli.py`. Detector thresholds live in `robo/detection/panel_detector.py`.
Overexposure that turns lights completely white,
occlusion, motion blur, merged light contours, and other matching red bars can
cause missed or false detections. There is no temporal state to recover a
missing light, and ambiguous multi-panel scenes can produce wrong pairings.

## Verify

```sh
uv run python -m unittest discover -s robo/detection/tests -t . -v
```

Synthetic tests cover full-turn roll, translation/scale changes, reference-photo
proportions and white markings, touching red objects, speckled clutter,
red/orange LEDs, white cores with closed or open red halos, hand glow,
adjacent white reflections, blur and JPEG compression, small gaps, rejected warm color
casts and tiny regions, loose pair geometry, background-brightness independence,
unchanged original light boxes, hidden orphan lights, best-pair selection with
multiple panels, invalid
geometry, immediate disappearance, live bounding boxes, timestamped JSON logs,
and the live HTTP feed. The attached photo and camera screenshot are visual
references, not saved pixel fixtures; tests do not run on those original images.
