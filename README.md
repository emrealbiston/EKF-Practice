# Robo

All application code lives in the `robo/` Python package. Detection lives in
`robo/detection/`; future EKF code can live alongside it under `robo/ekf/`.
`pyproject.toml`, `uv.lock`, and `.venv` live at the repository root and manage
the whole application.

## Run detection

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python
3.9 or newer. Run from the repository root:

```sh
uv sync --locked
uv run robo detect
```

This opens camera 0, draws bounding boxes live, serves the annotated feed at
http://127.0.0.1:8080, and saves timestamped JSON logs under `robo/detection/outputs/`.
Press `q`, Escape, or Ctrl+C to stop. No video file is recorded.
Green outlines show only the two original light boxes of the best-scoring pair
in each frame. All other bars are hidden; pairing never resizes or aligns boxes.
Press `m` to view the light mask. JSON logs include red-pixel and candidate-bar
counts to help diagnose missed panels.
The detector finds narrow red/orange bars and bright white cores with red halos,
then uses loose parallelism, alignment, size, and spacing checks to accept pairs.
Camera exposure is preserved.

`robo/cli.py` contains the top-level CLI. Detection code, tests, documentation,
and output logs are under `robo/detection/`.
See [robo/detection/README.md](robo/detection/README.md) for the algorithm and log format.
The detector is also available in Python as `from robo.detection import PanelDetector`.

Only `robo detect` is implemented. The planned `uv run robo estimate` command
and EKF estimation have not been implemented.

## Verify

```sh
uv run python -m unittest discover -v
```
