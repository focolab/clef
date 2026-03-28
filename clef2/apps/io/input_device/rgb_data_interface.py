"""
RGB Data Interface for CLEF2.

Stores uint8 RGB image data (height x width x 3) from any input device into a
preallocated numpy buffer and saves as PNG image sequences.
"""

import logging
import numpy as np
from typing import Any, ClassVar, Dict, Optional

from clef2.core.io.input_device.BaseDataInterface import BaseDataInterface

logger = logging.getLogger(__name__)


class RGBDataInterface(BaseDataInterface):
    """Data interface for uint8 RGB image streams from any input device."""

    data_interface_class: ClassVar[Optional[str]] = "rgb_data_interface"

    def __init__(self, input_device: Any = None, config: Dict[str, Any] | None = None):
        super().__init__(input_device, config)
        self._sample_index = 0
        self._samples: Optional[np.ndarray] = None

    def get_sample_shape(self) -> tuple:
        if self.input_device is not None:
            return (self.input_device.height, self.input_device.width, 3)
        return None

    def get_sample_dtype(self) -> np.dtype:
        return np.dtype(np.uint8)

    def configure_sampling(self, num_samples: int = 1, **kwargs):
        """Preallocate a (num_samples, height, width, 3) uint8 buffer."""
        shape = (num_samples,) + self.get_sample_shape()
        self._samples = np.zeros(shape, dtype=self.get_sample_dtype())
        self._sample_index = 0
        logger.info(f"Preallocated RGB sample buffer: shape={shape}")

    def store_input(self, sample: Any):
        self.input_store = sample
        if self._samples is not None and self._sample_index < self._samples.shape[0]:
            self._samples[self._sample_index] = sample
        self._sample_index += 1

    def save_data(self, **kwargs):
        """Save stored RGB images as a PNG sequence."""
        filepath = kwargs.get("filepath")
        if filepath is None:
            logger.warning("No filepath provided to save_data, skipping")
            return

        data = self._samples
        if data is None:
            logger.warning("No sample buffer configured, nothing to save")
            return

        # Trim to actual number of samples stored
        n = min(self._sample_index, data.shape[0])
        data = data[:n]

        from pathlib import Path
        output_dir = Path(filepath)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from PIL import Image
            for i in range(n):
                frame_path = output_dir / f"frame_{i:06d}.png"
                img = Image.fromarray(data[i], mode="RGB")
                img.save(frame_path)
            logger.info(f"Saved {n} RGB frames to {output_dir} (shape={data.shape})")
        except ImportError:
            # Fallback to numpy
            npy_path = str(output_dir / "rgb_data.npy")
            np.save(npy_path, data)
            logger.info(f"PIL not available, saved {n} frames as {npy_path}")

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "data_interface_class": self.data_interface_class,
            "sample_shape": self.get_sample_shape(),
            "sample_dtype": str(self.get_sample_dtype()),
            "samples_stored": self._sample_index,
            "buffer_size": self._samples.shape[0] if self._samples is not None else 0,
        }
