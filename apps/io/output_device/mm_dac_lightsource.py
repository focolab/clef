"""
Micro-Manager DAC voltage light source output device for CLEF.

Drives a light source by writing a float voltage to a Micro-Manager DAC
property (e.g. "DAC488".Volts). Modeled on InvCoreSpinningDisk639 from
wb-live-v4.

Instantiates its own pycromanager Core object.
"""

import logging
from typing import Any, ClassVar, Dict, Optional

from core.io.output_device.BaseOutputDevice import BaseOutputDevice

logger = logging.getLogger(__name__)


class MMDACLightSourceOutput(BaseOutputDevice):
    """Output device that controls a light source via an MM DAC voltage property."""

    device_class: ClassVar[Optional[str]] = "mm_dac_lightsource"
    device_type: ClassVar[Optional[str]] = "hardware"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name, config, io_manager=io_manager)
        self.mmc = None
        self._dac_device = None
        self._dac_property = None
        self._max_volts = 3.5

    def connect(self):
        from pycromanager import Core
        self.mmc = Core(convert_camel_case=False)
        logger.info(f"MMDACLightSourceOutput '{self.name}' connected to pycromanager Core")

    def configure(self):
        cfg = self.config
        self._dac_device = cfg.get("dac_device")
        self._dac_property = cfg.get("dac_property", "Volts")
        self._max_volts = float(cfg.get("max_volts", 3.5))

        if self._dac_device and self._dac_property:
            self.mmc.setProperty(self._dac_device, self._dac_property, 0)
            logger.debug(f"Initialized {self._dac_device}.{self._dac_property} to 0")

        logger.info(
            f"MMDACLightSourceOutput '{self.name}' configured: "
            f"device={self._dac_device}, property={self._dac_property}, "
            f"max_volts={self._max_volts}"
        )

    def _update_output(self, **kwargs):
        """Set DAC voltage.

        Keyword Args:
            intensity: float, voltage to set (clamped to [0, max_volts]).
        """
        intensity = float(kwargs.get("intensity", 0))
        volts = max(0.0, min(intensity, self._max_volts))

        if self._dac_device and self._dac_property:
            self.mmc.setProperty(self._dac_device, self._dac_property, volts)
            logger.debug(f"{self.name} set {self._dac_device}.{self._dac_property} to {volts}")

    def close(self):
        if self.mmc is not None and self._dac_device and self._dac_property:
            try:
                self.mmc.setProperty(self._dac_device, self._dac_property, 0)
            except Exception as e:
                logger.warning(f"Error setting {self._dac_device}.{self._dac_property} to 0 on close: {e}")

        logger.info(f"MMDACLightSourceOutput '{self.name}' closed")
