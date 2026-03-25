"""
Micro-Manager Camera Input Device for CLEF2.

Acquires uint16 frames from a Micro-Manager-controlled camera via pycromanager.
Instantiates its own pycromanager Core object.
"""

import logging
import numpy as np
from typing import Any, ClassVar, Dict, Optional

from clef2.core.io.input_device.BaseInputDevice import BaseInputDevice

logger = logging.getLogger(__name__)


class MicroManagerCameraInput(BaseInputDevice):
    """Input device wrapping a Micro-Manager camera via pycromanager."""

    device_class: ClassVar[Optional[str]] = "micromanager_camera"
    device_type: ClassVar[Optional[str]] = "hardware"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name, config, io_manager=io_manager)
        self.mmc = None
        self._roi = None
        self.width = 0
        self.height = 0
        self._acquiring = False

    def connect(self):
        """Instantiate pycromanager Core connection."""
        from pycromanager import Core
        self.mmc = Core(convert_camel_case=False)
        logger.info(f"MicroManagerCameraInput '{self.name}' connected to pycromanager Core")

    def configure(self):
        """Configure camera: ROI, exposure, binning, device properties, buffer."""
        cfg = self.config

        # Set device properties first (e.g. Binning)
        device_properties = cfg.get("device_properties", {})
        cam = self.mmc.getCameraDevice()
        for prop_name, prop_value in device_properties.items():
            try:
                self.mmc.setProperty(cam, prop_name, prop_value)
                logger.debug(f"Set {cam}.{prop_name} = {prop_value}")
            except Exception as e:
                logger.warning(f"Could not set camera property {prop_name}={prop_value}: {e}")

        # Set exposure
        exposure_ms = cfg.get("exposure_ms")
        if exposure_ms is not None:
            self.mmc.setExposure(float(exposure_ms))
            logger.debug(f"Set exposure to {exposure_ms} ms")

        # Set ROI if provided, otherwise query current
        roi = cfg.get("roi")
        if roi is not None:
            self.mmc.setROI(roi[0], roi[1], roi[2], roi[3])
            self._roi = tuple(roi)
        else:
            roi_obj = self.mmc.getROI()
            self._roi = (roi_obj.getX(), roi_obj.getY(),
                         roi_obj.getWidth(), roi_obj.getHeight())

        self.width = self._roi[2]
        self.height = self._roi[3]

        # Set circular buffer
        buffer_mb = cfg.get("buffer_memory_mb", 10000)
        self.mmc.setCircularBufferMemoryFootprint(buffer_mb)

        logger.info(
            f"MicroManagerCameraInput '{self.name}' configured: "
            f"ROI={self._roi}, exposure={self.mmc.getExposure()} ms"
        )

    def start_acquisition(self):
        """Start continuous sequence acquisition."""
        self.mmc.startContinuousSequenceAcquisition(0)
        self._acquiring = True
        logger.debug("Started continuous acquisition")

    def stop_acquisition(self):
        """Stop continuous sequence acquisition."""
        if self._acquiring:
            self.mmc.stopSequenceAcquisition()
            self._acquiring = False
            logger.debug("Stopped acquisition")

    def get_input(self) -> Optional[np.ndarray]:
        """Pop next image from the circular buffer.

        Returns None if buffer is empty.
        """
        if self.mmc.getRemainingImageCount() > 0:
            img = self.mmc.popNextImage().astype(np.uint16)
            img = img.reshape((self.height, self.width))
            return img
        return None

    def get_roi(self) -> tuple:
        """Return current ROI as (x, y, width, height)."""
        return self._roi

    def clear_buffer(self):
        """Clear the circular buffer."""
        self.mmc.clearCircularBuffer()

    def close(self):
        """Stop acquisition and clean up."""
        self.stop_acquisition()
        logger.info(f"MicroManagerCameraInput '{self.name}' closed")
