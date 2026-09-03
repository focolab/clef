"""
Shared-memory Uint16 Data Interface for CLEF.

Stores uint16 image data into a ring of shared memory segments so that
subprocess-based logic algorithms can read frames without IPC overhead.
Optionally preallocates a numpy save buffer for post-session TIFF writing.

Successive frames go into successive segments (segment = image_count % size),
so the ring is over *time*, not over z: it keeps the writer and any reading
subprocess on different segments. With a single segment they share one buffer
and a reader can see a frame while it is still being copied in, so size this by
how far a consumer may lag, not by the number of z-planes. The `zsize` naming
below is historical.
"""

import logging
import time
import numpy as np
from multiprocessing import shared_memory
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from core.io.input_device.BaseDataInterface import BaseDataInterface

logger = logging.getLogger(__name__)

# [count, camera_acquisition_ms, host_perf_counter_s]
IMAGE_COUNT_SLOTS = 3


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

        # Save buffer (heap-allocated, optional) — used for finite-length runs.
        self._save_buffer: Optional[np.ndarray] = None
        self._sample_index = 0
        self._image_count = 0

        # Streaming save state — used when num_samples <= 0 (continuous runs),
        # where preallocating the whole recording in RAM is impossible. Frames
        # are appended to a BigTIFF on disk as they arrive instead.
        self._streaming = False
        self._stream_dir: Optional[str] = None
        self._tiff_writer = None
        self._stream_tmp_path: Optional[str] = None

    @property
    def shm_names(self) -> List[str]:
        """Shared memory segment names, one per ring slot."""
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
        save_dir: Optional[str] = None,
        **kwargs,
    ):
        """Create shared memory segments and optionally a save buffer.

        Reads height/width from self.input_device and num_z_planes from self.config.

        Args:
            num_samples: Number of frames to preallocate for saving. A value <= 0
                means continuous acquisition; saving then streams to disk instead
                of preallocating a fixed buffer (ignored if save_samples=False).
            save_samples: If True, retain frames for post-session TIFF saving.
            save_dir: Directory the streamed TIFF is written into. Kept on the same
                filesystem as the final output so finalizing is an atomic rename.
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

        # Create the ring of frame segments
        for z in range(zsize):
            name = f"{shm_name_prefix}_{z}"
            shm = self._create_or_attach_shm(name, buf_size)
            ndarray = np.ndarray((height, width), dtype=np.uint16, buffer=shm.buf)
            self._shm_list.append(shm)
            self._shm_ndarray_list.append(ndarray)

        # Shared image count, plus the timestamps of the frame it refers to:
        # [count, camera_acquisition_ms, host_perf_counter_s]. Subprocess logic
        # needs per-frame timing to run a control loop, and the camera's own
        # clock is immune to host-loop jitter. Readers that only want the count
        # (e.g. brainalyzer_worker) still just read slot 0.
        count_name = self.image_count_shm_name
        try:
            self._image_count_shl = shared_memory.ShareableList(
                [0, float("nan"), 0.0], name=count_name
            )
        except FileExistsError:
            existing = shared_memory.ShareableList(name=count_name)
            if len(existing) == IMAGE_COUNT_SLOTS:
                self._image_count_shl = existing
            else:
                # Left by an older build with a different layout; the packing
                # format would not accept the timestamp slots.
                logger.warning(
                    f"Image count SHM '{count_name}' has {len(existing)} slots, "
                    f"expected {IMAGE_COUNT_SLOTS}; recreating it."
                )
                existing.shm.close()
                existing.shm.unlink()
                self._image_count_shl = shared_memory.ShareableList(
                    [0, float("nan"), 0.0], name=count_name
                )

        logger.info(
            f"SHM data interface: {zsize} segments, "
            f"frame=({height}, {width}), prefix='{shm_name_prefix}'"
        )

        # Optional saving. Finite runs preallocate a buffer; continuous runs
        # (num_samples <= 0) stream frames to a BigTIFF on disk instead.
        if save_samples:
            if num_samples <= 0:
                self._streaming = True
                self._stream_dir = save_dir or "."
                logger.info(
                    f"Continuous acquisition: streaming frames to disk in "
                    f"'{self._stream_dir}' (no RAM preallocation)"
                )
            else:
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

        # Advance to the next ring slot so a subprocess reading the previous
        # frame is not racing this copy.
        z = self._image_count % self._zsize

        # Copy 1: frame -> shared memory (subprocess access)
        if self._shm_ndarray_list:
            self._shm_ndarray_list[z][:] = sample

        # Copy 2 (optional): frame -> save buffer (finite) or disk (continuous)
        if self._streaming:
            if self._tiff_writer is None:
                self._open_stream_writer()
            if self._tiff_writer is not None:
                self._tiff_writer.write(sample, contiguous=True)
        elif (
            self._save_buffer is not None
            and self._sample_index < self._save_buffer.shape[0]
        ):
            self._save_buffer[self._sample_index] = sample

        self._image_count += 1
        self._sample_index += 1

        if self._image_count_shl is not None:
            # Timestamps first, count last: a reader that observes the new count
            # is then guaranteed to see the timestamps belonging to that frame.
            acq_ms = getattr(self.input_device, "last_acquisition_ms", None)
            self._image_count_shl[1] = (
                float(acq_ms) if acq_ms is not None else float("nan")
            )
            self._image_count_shl[2] = time.perf_counter()
            self._image_count_shl[0] = self._image_count

        self.input_store = (
            self._shm_ndarray_list[z] if self._shm_ndarray_list else sample
        )

    def _open_stream_writer(self):
        """Lazily open a BigTIFF writer on the first frame of a continuous run."""
        import os
        try:
            import tifffile as tf
        except ImportError:
            logger.error("tifffile not installed, cannot stream frames to disk")
            self._streaming = False
            return
        os.makedirs(self._stream_dir, exist_ok=True)
        self._stream_tmp_path = os.path.join(
            self._stream_dir, f"{self._shm_name_prefix}_stream.partial.tiff"
        )
        self._tiff_writer = tf.TiffWriter(self._stream_tmp_path, bigtiff=True)
        logger.info(f"Streaming frames to {self._stream_tmp_path}")

    def save_data(self, **kwargs):
        """Save stored images as a TIFF stack.

        For continuous runs the frames were already streamed to disk; here we
        just close the writer and atomically rename the file into place.
        """
        import os
        filepath = kwargs.get("filepath")
        if filepath is None:
            logger.warning("No filepath provided to save_data, skipping")
            return

        if self._streaming:
            if self._tiff_writer is not None:
                self._tiff_writer.close()
                self._tiff_writer = None
            filepath = str(filepath)
            if not filepath.endswith(".tiff"):
                filepath += ".tiff"
            if self._stream_tmp_path and os.path.exists(self._stream_tmp_path):
                os.replace(self._stream_tmp_path, filepath)
                logger.info(
                    f"Saved {self._sample_index} streamed frames to {filepath}"
                )
            else:
                logger.warning("Streaming enabled but no frames were written")
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
        # Flush the stream writer if save_data never ran (e.g. aborted session)
        # so streamed frames are not lost; the file stays as *.partial.tiff.
        if self._tiff_writer is not None:
            try:
                self._tiff_writer.close()
                logger.info(
                    f"Flushed streamed frames to {self._stream_tmp_path} "
                    "(session ended without save)"
                )
            except Exception as e:
                logger.warning(f"Error closing stream writer: {e}")
            self._tiff_writer = None

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
