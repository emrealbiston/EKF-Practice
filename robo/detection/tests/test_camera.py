import unittest
from unittest.mock import Mock, call, patch

import cv2

from robo.detection.camera import configure_low_exposure


class CameraExposureTests(unittest.TestCase):
    def test_v4l2_requests_normalized_device_minimum_in_manual_mode(self):
        capture = Mock()
        capture.getBackendName.return_value = "V4L2"
        capture.set.return_value = True
        capture.get.return_value = 0.0
        with patch("builtins.print"):
            self.assertTrue(configure_low_exposure(capture))
        self.assertEqual(capture.set.call_args_list, [
            call(cv2.CAP_PROP_MODE, 1), call(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),
            call(cv2.CAP_PROP_EXPOSURE, 0.0),
        ])

    def test_windows_uses_first_accepted_short_exposure(self):
        capture = Mock()
        capture.getBackendName.return_value = "DSHOW"
        capture.set.side_effect = lambda prop, value: prop == cv2.CAP_PROP_AUTO_EXPOSURE or value >= -18
        capture.get.return_value = -18
        with patch("builtins.print"):
            self.assertTrue(configure_low_exposure(capture))
        self.assertEqual(capture.set.call_args_list, [
            call(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0),
            call(cv2.CAP_PROP_EXPOSURE, -20.0), call(cv2.CAP_PROP_EXPOSURE, -19.0),
            call(cv2.CAP_PROP_EXPOSURE, -18.0),
        ])

    def test_unsupported_mac_exposure_is_reported_without_reading_a_fake_value(self):
        capture = Mock()
        capture.getBackendName.return_value = "AVFOUNDATION"
        capture.set.return_value = False
        with patch("builtins.print") as output:
            self.assertFalse(configure_low_exposure(capture))
        self.assertIn("could not be confirmed", output.call_args.args[0])
        capture.get.assert_not_called()

    def test_exposure_request_is_not_confirmed_if_auto_mode_cannot_be_disabled(self):
        capture = Mock()
        capture.getBackendName.return_value = "MSMF"
        capture.set.side_effect = lambda prop, value: prop == cv2.CAP_PROP_EXPOSURE
        with patch("builtins.print"):
            self.assertFalse(configure_low_exposure(capture))
        capture.get.assert_not_called()

    def test_failed_normalization_does_not_apply_raw_zero_exposure(self):
        capture = Mock()
        capture.getBackendName.return_value = "V4L2"
        capture.set.return_value = False
        with patch("builtins.print"):
            self.assertFalse(configure_low_exposure(capture))
        capture.set.assert_called_once_with(cv2.CAP_PROP_MODE, 1)


if __name__ == "__main__":
    unittest.main()
