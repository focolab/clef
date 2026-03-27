"""
TIFF Playback Input Device for CLEF2.

Plays back frames from a TIFF file on disk as though they were live camera input.
Supports 3D (TYX) and 4D (TZYX) TIFF stacks with optional looping.
"""

import logging
import time
import numpy as np
from typing import Any, ClassVar, Dict, Optional

from clef2.core.io.input_device.BaseInputDevice import BaseInputDevice

logger = logging.getLogger(__name__)


class TiffPlaybackInputDevice(BaseInputDevice):
    """Input device that plays back frames from a saved TIFF file."""

    device_class: ClassVar[Optional[str]] = "tiff_playback_input"
    device_type: ClassVar[Optional[str]] = "demo"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name, config, io_manager=io_manager)

        self.tiff_path: str = self.config.get("tiff_path", "")
        self.exposure_ms: float = self.config.get("exposure_ms", 10.0)
        self.loop: bool = self.config.get("loop", True)

        self._data: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._ndim: int = 0
        self._total_frames: int = 0

    def connect(self):
        """Load the TIFF file into memory."""
        if not self.tiff_path:
            raise ValueError(f"TiffPlaybackInputDevice '{self.name}': no tiff_path specified in config")

        import tifffile as tf
        self._data = tf.imread(self.tiff_path)
        self._ndim = self._data.ndim

        if self._ndim == 4:  # TZYX
            self._total_frames = self._data.shape[0] * self._data.shape[1]
        elif self._ndim == 3:  # TYX
            self._total_frames = self._data.shape[0]
        elif self._ndim == 2:  # single YX frame
            self._total_frames = 1
        else:
            raise ValueError(
                f"TiffPlaybackInputDevice '{self.name}': unsupported TIFF shape {self._data.shape}. "
                f"Expected 2D (YX), 3D (TYX), or 4D (TZYX)."
            )

        logger.info(
            f"TiffPlaybackInputDevice '{self.name}' loaded {self.tiff_path}, "
            f"shape={self._data.shape}, dtype={self._data.dtype}"
        )

    def get_input(self) -> np.ndarray:
        """Return the next frame from the TIFF stack."""
        if self._data is None:
            raise RuntimeError(f"TiffPlaybackInputDevice '{self.name}': not connected, call connect() first")

        if not self.loop and self._frame_count >= self._total_frames:
            # Return last frame when not looping
            return self._get_frame(self._total_frames - 1)

        frame = self._get_frame(self._frame_count)
        self._frame_count += 1

        time.sleep(self.exposure_ms / 1000.0)
        return frame

    def _get_frame(self, idx: int) -> np.ndarray:
        """Extract a single 2D frame by linear index, with wrap-around."""
        if self._ndim == 4:
            # TZYX: flatten T*Z then index
            nz = self._data.shape[1]
            wrapped = idx % self._total_frames
            t = wrapped // nz
            z = wrapped % nz
            return self._data[t, z, :, :].copy()
        elif self._ndim == 3:
            return self._data[idx % self._total_frames, :, :].copy()
        else:
            return self._data.copy()

    def close(self):
        """Release the loaded data."""
        self._data = None
        self._frame_count = 0
        logger.info(f"TiffPlaybackInputDevice '{self.name}' closed")
