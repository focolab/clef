"""
Recording Playback Input Device for CLEF.

Reads frames from a TIFF file and serves them sequentially as if
acquired from a live camera. Supports TYX and TZYX layouts, looping
when frames are exhausted.
"""

import logging
import time
import numpy as np
from typing import Any, ClassVar, Dict, Optional, Tuple

from core.io.input_device.BaseInputDevice import BaseInputDevice

logger = logging.getLogger(__name__)


class RecordingPlaybackInput(BaseInputDevice):
    """Input device that replays frames from a TIFF recording."""

    device_class: ClassVar[Optional[str]] = "recording_playback_input"
    device_type: ClassVar[Optional[str]] = "playback"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name, config, io_manager=io_manager)

        cfg = self.config
        self.input_file = cfg.get("input_file")
        self.exposure_ms = cfg.get("exposure_ms", 5.0)
        self.loop = cfg.get("loop", True)

        self.input_data = None
        self._frame_count = 0
        self.width = 0
        self.height = 0
        self.num_z_planes = 1

        if self.input_file:
            self._load_tiff(self.input_file)
        else:
            logger.warning(f"RecordingPlaybackInput '{name}': no input_file specified")

    def _load_tiff(self, path: str):
        """Load a TIFF file into memory."""
        try:
            import tifffile as tf
        except ImportError:
            logger.error("tifffile not installed — cannot load recording")
            return

        self.input_data = tf.imread(path)
        shape = self.input_data.shape

        if len(shape) == 4:  # TZYX
            self.height = shape[2]
            self.width = shape[3]
            self.num_z_planes = shape[1]
            self._total_frames = shape[0] * shape[1]
        elif len(shape) == 3:  # TYX
            self.height = shape[1]
            self.width = shape[2]
            self.num_z_planes = 1
            self._total_frames = shape[0]
        else:
            logger.error(f"Unsupported TIFF shape: {shape}")
            self.input_data = None
            return

        logger.info(
            f"RecordingPlaybackInput '{self.name}': loaded {path}, "
            f"shape={shape}, {self._total_frames} frames, {self.width}x{self.height}"
        )

    def _get_input(self) -> np.ndarray:
        """Return next frame from the recording."""
        if self.input_data is not None:
            shape = self.input_data.shape

            if len(shape) == 4:
                t = (self._frame_count // shape[1]) % shape[0]
                z = self._frame_count % shape[1]
                frame = self.input_data[t, z, :, :].copy()
            else:
                idx = self._frame_count % shape[0] if self.loop else min(self._frame_count, shape[0] - 1)
                frame = self.input_data[idx, :, :].copy()
        else:
            frame = np.random.randint(0, 65536, size=(self.height or 200, self.width or 200), dtype=np.uint16)

        self._frame_count += 1
        time.sleep(self.exposure_ms / 1000.0)
        return frame

    def get_roi(self) -> Tuple[int, int, int, int]:
        return (0, 0, self.width, self.height)
