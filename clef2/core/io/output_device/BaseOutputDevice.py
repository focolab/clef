"""
Base class for all output devices in CLEF.

Subclasses must set `device_class` to a unique string that matches
the `output_device_class` field in the IO config YAML.  Registration
happens automatically via __init_subclass__.

The base class is fully instantiable with no-op implementations,
acting as a dummy device by default.
"""

import logging
from typing import ClassVar, Dict, Optional, Type, Any

from clef2.core.utils.event_log import EventLog

logger = logging.getLogger(__name__)


class BaseOutputDevice:
    """Base class for output devices. All methods are no-ops by default."""

    # --- registry --------------------------------------------------------
    _registry: ClassVar[Dict[str, Type["BaseOutputDevice"]]] = {}

    device_class: ClassVar[Optional[str]] = "base_output_device"
    device_type: ClassVar[Optional[str]] = "base"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.device_class is not None:
            BaseOutputDevice._registry[cls.device_class] = cls
            logger.debug(f"Registered output device: {cls.device_class} -> {cls.__name__}")

    @classmethod
    def get_class(cls, device_class: str) -> Type["BaseOutputDevice"]:
        """Look up a registered output device class by its device_class key."""
        if device_class not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown output device class: '{device_class}'. "
                f"Available: {available}"
            )
        return cls._registry[device_class]

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return all registered device_class keys."""
        return list(cls._registry.keys())

    # --- lifecycle -------------------------------------------------------

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        """
        Args:
            name: Instance name from config (output_device_name).
            config: The output_device_parameters dict from config.
            io_manager: Optional reference to the IOManager that owns this device.
        """
        self.name = name
        self.config = config or {}
        self.io_manager = io_manager
        self.event_log = EventLog()

    def connect(self):
        """Connect to the output device."""
        pass

    def configure(self):
        """Configure the output device."""
        pass

    def update_output(self, **kwargs):
        """Update the output device, then auto-record the event.

        Subclasses should override ``_update_output()`` instead of this method.
        """
        self._update_output(**kwargs)
        self.event_log.record(kwargs)

    def _update_output(self, **kwargs):
        """Override this to handle output. Called by ``update_output()``."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this output device."""
        return {
            "name": self.name,
            "device_class": self.device_class,
            "event_log": self.event_log.to_dict(),
        }

    def close(self):
        """Close the connection to the output device."""
        pass


# __init_subclass__ doesn't fire for the class it's defined on,
# so register the base class manually.
BaseOutputDevice._registry[BaseOutputDevice.device_class] = BaseOutputDevice
