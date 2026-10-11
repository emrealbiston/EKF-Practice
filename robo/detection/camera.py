"""Best-effort low exposure through OpenCV's backend-specific controls."""

import cv2


def configure_low_exposure(capture):
    """Request low manual exposure; report unsupported controls explicitly.

    OpenCV has no portable exposure-range query. V4L2 provides normalized
    controls, while Windows backends use log2 exposure values. Other backends
    get a single low-setting attempt; this includes unsupported AVFoundation.
    """
    backend = capture.getBackendName()
    if backend == "V4L2":
        # Normalization maps exposure=0 to the camera's reported range minimum.
        normalized = capture.set(cv2.CAP_PROP_MODE, 1)
        manual = normalized and capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        low = manual and capture.set(cv2.CAP_PROP_EXPOSURE, 0.0)
    else:
        manual = capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)
        # Probe Windows' shortest exposure first; drivers can reject or clamp it.
        values = range(-20, 0) if backend in ("DSHOW", "MSMF") else (-13,)
        low = False
        for value in values:
            if capture.set(cv2.CAP_PROP_EXPOSURE, float(value)):
                low = True
                break

    if manual and low:
        reported = capture.get(cv2.CAP_PROP_EXPOSURE)
        print("Low manual exposure requested ({}); reported setting: {}".format(
            backend, reported))
        return True
    print("Low manual exposure could not be confirmed ({}): auto-off={}, low-setting={}. "
          "Use the camera's exposure controls if needed.".format(backend, bool(manual), bool(low)))
    return False
