"""
Shared-memory Uint16 Data Interface for CLEF2.

Stores uint16 image data into shared memory segments (one per z-plane)
so that subprocess-based logic algorithms can read frames without IPC overhead.
Optionally preallocates a numpy save buffer for post-session TIFF writing.
"""

import logging
import numpy as np
from multiprocessing import shared_memory
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from clef2.core.io.input_device.BaseDataInterface import BaseDataInterface

logger = logging.getLogger(__name__)


class SharedMemoryUint16DataInterface(BaseDataInterface):
    """Data interface that writes uint16 frames into shared memory for subprocess access."""

    data_interface_class: ClassVar[Optional[str]] = "shm_uint16_data_interface"

    def __init__(self, input_device: Any = None, config: Dict[str, Any] | None = None):
        super().__init__(input_device, config)

        # Shared memory state
        self._shm_list: List[shared_memory.SharedMemory] = []
        self._shm_ndarray_list: List[np.ndarray] = []
        self._image_count_shl: Optional[shared_memory.ShareableList] = None

        # Config
        self._zsize = 1
        self._height = 0
        self._width = 0
        self._shm_name_prefix = self.config.get(
            "shm_name_prefix", "shared_frame_memory"
        )

        # Save buffer (heap-allocated, optional)
        self._save_buffer: Optional[np.ndarray] = None
        self._sample_index = 0
        self._image_count = 0

    @property
    def shm_names(self) -> List[str]:
        """Shared memory segment names, one per z-plane."""
        return [shm.name for shm in self._shm_list]

    @property
    def image_count_shm_name(self) -> str:
        """Name of the ShareableList holding the image count."""
        return "shared_image_count"

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return (self._height, self._width)

    @property
    def zsize(self) -> int:
        return self._zsize

    def get_sample_shape(self) -> tuple:
        return (self._height, self._width)

    def get_sample_dtype(self) -> np.dtype:
        return np.dtype(np.uint16)

    def configure_sampling(
        self,
        num_samples: int = 1,
        save_samples: bool = False,
        **kwargs,
    ):
        """Create shared memory segments and optionally a save buffer.

        Reads height/width from self.input_device and num_z_planes from self.config.

        Args:
            num_samples: Number of frames to preallocate for saving (ignored if save_samples=False).
            save_samples: If True, preallocate a numpy buffer for post-session TIFF saving.
        """
        # Read dimensions from input device
        height = getattr(self.input_device, "height", 0) if self.input_device else 0
        width = getattr(self.input_device, "width", 0) if self.input_device else 0
        zsize = self.config.get("shm_buffer_size", 1)
        shm_name_prefix = self.config.get("shm_name_prefix", "shared_frame_memory")

        self._height = height
        self._width = width
        self._zsize = zsize
        self._shm_name_prefix = shm_name_prefix

        if height == 0 or width == 0:
            logger.warning("height/width not set — skipping shm creation")
            return

        buf_size = height * width * np.dtype(np.uint16).itemsize

        # Create one shm segment per z-plane
        for z in range(zsize):
            name = f"{shm_name_prefix}_{z}"
            shm = self._create_or_attach_shm(name, buf_size)
            ndarray = np.ndarray((height, width), dtype=np.uint16, buffer=shm.buf)
            self._shm_list.append(shm)
            self._shm_ndarray_list.append(ndarray)

        # Shared image count
        count_name = self.image_count_shm_name
        try:
            self._image_count_shl = shared_memory.ShareableList([0], name=count_name)
        except FileExistsError:
            self._image_count_shl = shared_memory.ShareableList(name=count_name)

        logger.info(
            f"SHM data interface: {zsize} segments, "
            f"frame=({height}, {width}), prefix='{shm_name_prefix}'"
        )

        # Optional save buffer
        if save_samples:
            shape = (num_samples, height, width)
            self._save_buffer = np.zeros(shape, dtype=np.uint16)
            logger.info(f"Preallocated save buffer: shape={shape}")

    @staticmethod
    def _create_or_attach_shm(name: str, size: int) -> shared_memory.SharedMemory:
        try:
            return shared_memory.SharedMemory(create=True, size=size, name=name)
        except FileExistsError:
            return shared_memory.SharedMemory(name=name, create=False, size=size)

    def store_input(self, sample: Any):
        if sample is None:
            self.input_store = None
            return

        z = self._image_count % self._zsize

        # Copy 1: frame -> shared memory (subprocess access)
        if self._shm_ndarray_list:
            self._shm_ndarray_list[z][:] = sample

        # Copy 2 (optional): frame -> save buffer
        if (
            self._save_buffer is not None
            and self._sample_index < self._save_buffer.shape[0]
        ):
            self._save_buffer[self._sample_index] = sample

        self._image_count += 1
        self._sample_index += 1

        if self._image_count_shl is not None:
            self._image_count_shl[0] = self._image_count

        self.input_store = (
            self._shm_ndarray_list[z] if self._shm_ndarray_list else sample
        )

    def save_data(self, **kwargs):
        """Save stored images as a TIFF stack."""
        filepath = kwargs.get("filepath")
        if filepath is None:
            logger.warning("No filepath provided to save_data, skipping")
            return

        if self._save_buffer is None:
            logger.warning("No save buffer configured, nothing to save")
            return

        n = min(self._sample_index, self._save_buffer.shape[0])
        data = self._save_buffer[:n]

        try:
            import tifffile as tf
        except ImportError:
            logger.error("tifffile not installed, cannot save TIFF data")
            return

        filepath = str(filepath)
        if not filepath.endswith(".tiff"):
            filepath += ".tiff"

        tf.imwrite(filepath, data)
        logger.info(f"Saved {n} frames to {filepath} (shape={data.shape})")

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "data_interface_class": self.data_interface_class,
            "sample_shape": self.get_sample_shape(),
            "sample_dtype": str(self.get_sample_dtype()),
            "samples_stored": self._sample_index,
            "buffer_size": (
                self._save_buffer.shape[0] if self._save_buffer is not None else 0
            ),
            "shm_names": self.shm_names,
            "image_count_shm_name": self.image_count_shm_name,
            "zsize": self._zsize,
        }

    def close(self):
        """Unlink all shared memory segments."""
        for shm in self._shm_list:
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                logger.info(f"Error closing shm '{shm.name}': {e}")
        self._shm_list.clear()
        self._shm_ndarray_list.clear()

        if self._image_count_shl is not None:
            try:
                self._image_count_shl.shm.close()
                self._image_count_shl.shm.unlink()
            except Exception as e:
                logger.info(f"Error closing image count shm: {e}")
            self._image_count_shl = None
