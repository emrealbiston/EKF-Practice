from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import distribution
from io import StringIO
import runpy
import unittest
from unittest.mock import patch

from robo.detection import cli as detection_cli
from robo import cli


class RoboCliTests(unittest.TestCase):
    def test_installed_robo_detect_command_starts_detection(self):
        commands = {entry.name: entry for entry in distribution("robo").entry_points
                    if entry.group == "console_scripts"}
        self.assertIn("robo", commands)
        with patch.object(detection_cli, "run") as run:
            commands["robo"].load()(["detect"])
        run.assert_called_once_with()

    def test_module_command_reports_startup_errors_without_a_traceback(self):
        with patch.object(detection_cli, "run", side_effect=RuntimeError("Cannot open camera 0")), \
                patch("sys.argv", ["robo", "detect"]):
            with self.assertRaises(SystemExit) as error:
                runpy.run_module("robo", run_name="__main__")
        self.assertEqual(str(error.exception), "Cannot open camera 0")

    def test_help_does_not_start_detection(self):
        with patch.object(detection_cli, "run") as run, redirect_stdout(StringIO()) as output:
            with self.assertRaises(SystemExit) as error:
                cli.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("detect", output.getvalue())
        self.assertNotIn("estimate", output.getvalue())
        run.assert_not_called()

    def test_estimate_is_not_implemented_and_does_not_start_detection(self):
        with patch.object(detection_cli, "run") as run, redirect_stderr(StringIO()) as output:
            with self.assertRaises(SystemExit) as error:
                cli.main(["estimate"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("invalid choice", output.getvalue())
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
