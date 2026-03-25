"""
Mightex Polygon (DMD/SLM) Output Device for CLEF2.

Manages a Mightex Polygon SLM via Micro-Manager. Creates a shared memory
buffer at configure-time so that external processes (e.g. BrainalyzerWorker)
can write masks directly. On update_output(), the current buffer contents
are uploaded to the SLM.

Instantiates its own pycromanager Core object.
"""

import logging
import numpy as np
from multiprocessing import shared_memory
from typing import Any, ClassVar, Dict, Optional, Tuple

from clef2.core.io.output_device.BaseOutputDevice import BaseOutputDevice

logger = logging.getLogger(__name__)


class MightexPolygonOutput(BaseOutputDevice):
    """Output device for Mightex Polygon SLM mask upload."""

    device_class: ClassVar[Optional[str]] = "mightex_polygon"
    device_type: ClassVar[Optional[str]] = "hardware"

    SHM_NAME = "polygon_mask_buffer"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name, config, io_manager=io_manager)
        self.mmc = None
        self.slm_device = None
        self.polygon_dims: Optional[Tuple[int, int]] = None
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._mask_array: Optional[np.ndarray] = None

    def connect(self):
        """Instantiate pycromanager Core connection."""
        from pycromanager import Core
        self.mmc = Core(convert_camel_case=False)
        logger.info(f"MightexPolygonOutput '{self.name}' connected to pycromanager Core")

    def configure(self):
        """Query SLM device, get dimensions, create shared memory buffer, blank SLM."""
        cfg = self.config

        # Get SLM device
        slm_device_name = cfg.get("slm_device")
        if slm_device_name:
            self.slm_device = slm_device_name
        else:
            self.slm_device = self.mmc.getSLMDevice()
        self.mmc.setSLMDevice(self.slm_device)

        # Get dimensions
        width = self.mmc.getSLMWidth(self.slm_device)
        height = self.mmc.getSLMHeight(self.slm_device)
        self.polygon_dims = (width, height)

        logger.info(f"Polygon SLM: {self.slm_device}, dimensions: {self.polygon_dims}")

        # Blank the SLM
        self.mmc.setSLMPixelsTo(self.slm_device, 0)

        # Create shared memory buffer for mask (uint8, width * height)
        buf_size = width * height
        try:
            self._shm = shared_memory.SharedMemory(
                create=True, size=buf_size, name=self.SHM_NAME
            )
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(
                name=self.SHM_NAME, create=False, size=buf_size
            )

        self._mask_array = np.ndarray(
            shape=(height, width), dtype=np.uint8, buffer=self._shm.buf
        )
        self._mask_array[:] = 0

        logger.info(
            f"MightexPolygonOutput '{self.name}' configured: "
            f"SLM={self.slm_device}, dims={self.polygon_dims}, "
            f"shm='{self.SHM_NAME}'"
        )

    @property
    def shm_name(self) -> str:
        """Name of the shared memory buffer for external writers."""
        return self.SHM_NAME

    def update_output(self, **kwargs):
        """Read mask from shared memory and upload to SLM.

        Keyword Args:
            action: "upload_mask" (default) reads shm buffer and uploads.
                    "blank" sets all SLM pixels to 0.
        """
        action = kwargs.get("action", "upload_mask")

        if action == "blank":
            self.mmc.setSLMPixelsTo(self.slm_device, 0)
            if self._mask_array is not None:
                self._mask_array[:] = 0
            logger.debug("Blanked polygon SLM")
            return

        # Default: upload current mask from shared memory
        if self._mask_array is not None and self.slm_device is not None:
            self.mmc.setSLMImage(
                self.slm_device, self._mask_array.flatten()
            )
            logger.debug("Uploaded mask from shared memory to SLM")

    def get_dimensions(self) -> Optional[Tuple[int, int]]:
        """Return (width, height) of the SLM."""
        return self.polygon_dims

    def close(self):
        """Blank SLM and release shared memory."""
        if self.mmc is not None and self.slm_device is not None:
            try:
                self.mmc.setSLMPixelsTo(self.slm_device, 0)
            except Exception as e:
                logger.warning(f"Error blanking SLM on close: {e}")

        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception as e:
                logger.warning(f"Error cleaning up polygon shared memory: {e}")

        logger.info(f"MightexPolygonOutput '{self.name}' closed")
