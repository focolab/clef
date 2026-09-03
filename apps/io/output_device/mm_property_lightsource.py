"""
Micro-Manager property light source output device for CLEF.

Drives a light source by writing an intensity to an arbitrary Micro-Manager
device property (e.g. a light engine's "LightEngineIntensity"). The sibling
mm_dac_lightsource does the same job for sources controlled by a DAC voltage;
this one is for sources that expose intensity as a plain device property.

The device label may be left null, in which case the loaded Micro-Manager
device carrying the named property is found automatically. Labels differ
between rigs and Micro-Manager configurations while property names do not, so
this avoids baking one machine's label into a shared config.

Instantiates its own pycromanager Core object.
"""

import logging
from typing import Any, ClassVar, Dict, Optional

from core.io.output_device.BaseOutputDevice import BaseOutputDevice

logger = logging.getLogger(__name__)


class MMPropertyLightSourceOutput(BaseOutputDevice):
    """Output device that controls a light source via an MM device property."""

    device_class: ClassVar[Optional[str]] = "mm_property_lightsource"
    device_type: ClassVar[Optional[str]] = "hardware"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name, config, io_manager=io_manager)
        self.mmc = None
        self._intensity_device = None
        self._intensity_property = None
        self._max_intensity = 100.0

    def connect(self):
        from pycromanager import Core
        self.mmc = Core(convert_camel_case=False)
        logger.info(
            f"MMPropertyLightSourceOutput '{self.name}' connected to pycromanager Core"
        )

    def configure(self):
        """Resolve the device, then park the source at zero."""
        cfg = self.config
        self._intensity_property = cfg.get("intensity_property")
        self._intensity_device = cfg.get("intensity_device")
        self._max_intensity = float(cfg.get("max_intensity", 100.0))

        if self._intensity_device is None and self._intensity_property:
            self._intensity_device = self._find_device_with_property(
                self._intensity_property
            )
            if self._intensity_device is None:
                logger.warning(
                    f"MMPropertyLightSourceOutput '{self.name}': no loaded "
                    f"Micro-Manager device has property "
                    f"'{self._intensity_property}'; light control is disabled "
                    "for this session."
                )
            else:
                logger.info(
                    f"MMPropertyLightSourceOutput '{self.name}': found "
                    f"'{self._intensity_property}' on device "
                    f"'{self._intensity_device}'"
                )

        # Start dark, so a session never begins by illuminating the sample.
        if self._intensity_device and self._intensity_property:
            self.mmc.setProperty(self._intensity_device, self._intensity_property, 0)

        logger.info(
            f"MMPropertyLightSourceOutput '{self.name}' configured: "
            f"device={self._intensity_device}, "
            f"property={self._intensity_property}, max={self._max_intensity}"
        )

    def _find_device_with_property(self, prop):
        """Return the first loaded device exposing `prop`, or None."""
        try:
            devices = self.mmc.getLoadedDevices()
        except Exception as e:
            logger.warning(f"Could not enumerate Micro-Manager devices: {e}")
            return None
        for device in devices:
            try:
                if self.mmc.hasProperty(device, prop):
                    return device
            except Exception:
                continue
        return None

    def _update_output(self, **kwargs):
        """Set the light source intensity.

        Keyword Args:
            intensity: intensity to set, clamped to [0, max_intensity].
        """
        if not (self._intensity_device and self._intensity_property):
            return

        intensity = float(kwargs.get("intensity", 0))
        # Clamp rather than trust the caller: this drives illumination onto a
        # live sample, and an out-of-range value is a photodamage risk.
        clamped = max(0.0, min(intensity, self._max_intensity))
        if clamped != intensity:
            logger.warning(
                f"{self.name}: intensity {intensity} clamped to {clamped} "
                f"(max_intensity={self._max_intensity})"
            )
        self.mmc.setProperty(
            self._intensity_device, self._intensity_property, clamped
        )
        logger.debug(f"{self.name} intensity set to {clamped}")

    def close(self):
        if self.mmc is not None and self._intensity_device and self._intensity_property:
            try:
                self.mmc.setProperty(
                    self._intensity_device, self._intensity_property, 0
                )
            except Exception as e:
                logger.warning(
                    f"Error setting {self.name} intensity to 0 on close: {e}"
                )
        logger.info(f"MMPropertyLightSourceOutput '{self.name}' closed")
