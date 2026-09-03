"""
89 North LDI (Laser Diode Illuminator) Output Device for CLEF.

Controls the 89 North LDI light source intensity and shutter via Micro-Manager.
This is the excitation light source for the Mightex Polygon DMD.

Instantiates its own pycromanager Core object.
"""

import logging
from typing import Any, ClassVar, Dict, Optional

from core.io.output_device.BaseOutputDevice import BaseOutputDevice

logger = logging.getLogger(__name__)


class LDI89NorthOutput(BaseOutputDevice):
    """Output device for 89 North Laser Diode Illuminator intensity control."""

    device_class: ClassVar[Optional[str]] = "ldi_89north"
    device_type: ClassVar[Optional[str]] = "hardware"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name, config, io_manager=io_manager)
        self.mmc = None
        self._intensity_device = None
        self._intensity_property = None
        self._shutter_device = None

    def connect(self):
        """Instantiate pycromanager Core connection."""
        from pycromanager import Core
        self.mmc = Core(convert_camel_case=False)
        logger.info(f"LDI89NorthOutput '{self.name}' connected to pycromanager Core")

    def configure(self):
        """Set initial intensity to 0 and open shutter."""
        cfg = self.config
        self._intensity_device = cfg.get("intensity_device")
        self._intensity_property = cfg.get("intensity_property")
        self._shutter_device = cfg.get("shutter_device")

        # Initialize intensity to 0
        if self._intensity_device and self._intensity_property:
            self.mmc.setProperty(self._intensity_device, self._intensity_property, 0)
            logger.debug(f"Initialized {self._intensity_device}.{self._intensity_property} to 0")

        # Open shutter
        if self._shutter_device:
            self.mmc.setShutterOpen(self._shutter_device, True)
            logger.debug(f"Opened shutter: {self._shutter_device}")

        logger.info(
            f"LDI89NorthOutput '{self.name}' configured: "
            f"device={self._intensity_device}, property={self._intensity_property}"
        )

    def _update_output(self, **kwargs):
        """Set LDI intensity.

        Keyword Args:
            intensity: int, intensity value to set (0 = off).
        """
        intensity = kwargs.get("intensity", 0)

        if self._intensity_device and self._intensity_property:
            self.mmc.setProperty(
                self._intensity_device, self._intensity_property, int(intensity)
            )
            logger.debug(f"LDI intensity set to {intensity}")

    def close(self):
        """Set intensity to 0."""
        if self.mmc is not None and self._intensity_device and self._intensity_property:
            try:
                self.mmc.setProperty(
                    self._intensity_device, self._intensity_property, 0
                )
            except Exception as e:
                logger.warning(f"Error setting LDI intensity to 0 on close: {e}")

        logger.info(f"LDI89NorthOutput '{self.name}' closed")
