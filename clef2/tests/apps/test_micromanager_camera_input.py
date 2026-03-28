"""Tests for MicroManagerCameraInput auto-start and strobed acquisition."""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))

# Mock pycromanager before importing the module
sys.modules["pycromanager"] = MagicMock()

from clef2.apps.io.input_device.micromanager_camera_input import MicroManagerCameraInput


def _make_camera(strobed=False, interval_ms=50.0):
    """Create a MicroManagerCameraInput with a mock mmc."""
    config = {
        "device_properties": {},
        "use_strobed_acquisition": strobed,
        "strobe_inter_frame_interval_ms": interval_ms,
    }
    cam = MicroManagerCameraInput(name="test_cam", config=config)
    cam.mmc = MagicMock()
    cam.width = 4
    cam.height = 4
    cam._strobed = strobed
    cam._strobe_interval_s = interval_ms / 1000.0
    return cam


class TestAutoStartAcquisition:
    """_get_input should call start_acquisition on first invocation."""

    def test_auto_starts_continuous(self):
        cam = _make_camera(strobed=False)
        assert cam._acquiring is False

        # First call returns 0 (buffer empty after start), second returns 1
        cam.mmc.getRemainingImageCount.side_effect = [0, 1]
        cam.mmc.popNextImage.return_value = np.zeros((4, 4), dtype=np.uint16)

        result = cam._get_input()

        assert cam._acquiring is True
        cam.mmc.startContinuousSequenceAcquisition.assert_called_once_with(0)
        assert result is not None
        assert result.shape == (4, 4)

    def test_auto_starts_strobed(self):
        cam = _make_camera(strobed=True, interval_ms=1.0)
        assert cam._acquiring is False

        cam.mmc.getImage.return_value = np.zeros((4, 4), dtype=np.uint16)

        result = cam._get_input()

        assert cam._acquiring is True
        # snapImage called once in _get_input (start no longer snaps)
        cam.mmc.snapImage.assert_called_once()
        assert result is not None


class TestStrobedAcquisition:
    """Strobed mode: wait → snapImage → getImage."""

    def test_strobed_snap_before_get(self):
        cam = _make_camera(strobed=True, interval_ms=1.0)
        cam._acquiring = True
        cam._next_snap_time = time.perf_counter()

        cam.mmc.getImage.return_value = np.zeros((4, 4), dtype=np.uint16)

        img = cam._get_input()

        # snapImage must be called before getImage
        assert cam.mmc.snapImage.call_count == 1
        assert cam.mmc.getImage.call_count == 1
        assert img.shape == (4, 4)
        assert img.dtype == np.uint16

    def test_strobed_does_not_use_circular_buffer(self):
        cam = _make_camera(strobed=True, interval_ms=1.0)
        cam._acquiring = True
        cam._next_snap_time = time.perf_counter()

        cam.mmc.getImage.return_value = np.zeros((4, 4), dtype=np.uint16)
        cam._get_input()

        cam.mmc.popNextImage.assert_not_called()
        cam.mmc.getRemainingImageCount.assert_not_called()

    def test_strobed_advances_next_snap_time(self):
        cam = _make_camera(strobed=True, interval_ms=100.0)
        cam._acquiring = True
        t0 = time.perf_counter()
        cam._next_snap_time = t0

        cam.mmc.getImage.return_value = np.zeros((4, 4), dtype=np.uint16)
        cam._get_input()

        assert cam._next_snap_time == pytest.approx(t0 + 0.1, abs=0.001)

    def test_strobed_stop_does_not_call_stop_sequence(self):
        cam = _make_camera(strobed=True)
        cam._acquiring = True

        cam.stop_acquisition()

        cam.mmc.stopSequenceAcquisition.assert_not_called()
        assert cam._acquiring is False

    def test_continuous_stop_calls_stop_sequence(self):
        cam = _make_camera(strobed=False)
        cam._acquiring = True

        cam.stop_acquisition()

        cam.mmc.stopSequenceAcquisition.assert_called_once()


class TestContinuousAcquisition:
    """Continuous mode spin-waits for buffer."""

    def test_spin_waits_for_image(self):
        cam = _make_camera(strobed=False)
        cam._acquiring = True
        # Returns 0 twice then 1 — spin-wait then succeed
        cam.mmc.getRemainingImageCount.side_effect = [0, 0, 1]
        cam.mmc.popNextImage.return_value = np.zeros((4, 4), dtype=np.uint16)

        img = cam._get_input()

        assert img.shape == (4, 4)
        assert cam.mmc.getRemainingImageCount.call_count == 3
        cam.mmc.popNextImage.assert_called_once()

    def test_start_clears_buffer(self):
        cam = _make_camera(strobed=False)
        cam.start_acquisition()

        cam.mmc.stopSequenceAcquisition.assert_called_once()
        cam.mmc.clearCircularBuffer.assert_called_once()
        cam.mmc.startContinuousSequenceAcquisition.assert_called_once_with(0)
