"""
Base class for all input devices in CLEF.

Subclasses must set `device_class` to a unique string that matches
the `input_device_class` field in the IO config YAML.  Registration
happens automatically via __init_subclass__.

The base class is fully instantiable with no-op implementations,
acting as a dummy device by default.
"""

import logging
from typing import ClassVar, Dict, Optional, Type, Any

from clef2.core.utils.event_log import EventLog

logger = logging.getLogger(__name__)


class BaseInputDevice:
    """Base class for input devices. All methods are no-ops by default."""

    # --- registry --------------------------------------------------------
    _registry: ClassVar[Dict[str, Type["BaseInputDevice"]]] = {}

    device_class: ClassVar[Optional[str]] = "base_input_device"
    device_type: ClassVar[Optional[str]] = "base"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.device_class is not None:
            BaseInputDevice._registry[cls.device_class] = cls
            logger.debug(f"Registered input device: {cls.device_class} -> {cls.__name__}")

    @classmethod
    def get_class(cls, device_class: str) -> Type["BaseInputDevice"]:
        """Look up a registered input device class by its device_class key."""
        if device_class not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown input device class: '{device_class}'. "
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
            name: Instance name from config (input_device_name).
            config: The input_device_parameters dict from config.
            io_manager: Optional reference to the IOManager that owns this device.
        """
        self.name = name
        self.config = config or {}
        self.data_interface = None
        self.io_manager = io_manager
        self.event_log = EventLog()

    def connect(self):
        """Connect to the input device."""
        pass

    def configure(self):
        """Configure the input device with necessary settings."""
        pass

    def get_input(self) -> Any:
        """Retrieve input data from the device, then auto-record a timestamp.

        Subclasses should override ``_get_input()`` instead of this method.
        """
        result = self._get_input()
        self.event_log.record()
        return result

    def _get_input(self) -> Any:
        """Override this to supply input data. Called by ``get_input()``."""
        return None

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this input device."""
        return {
            "name": self.name,
            "device_class": self.device_class,
            "event_log": self.event_log.to_dict(),
        }

    def save_data(self, **kwargs):
        """Save data via the data interface."""
        if self.data_interface is not None:
            self.data_interface.save_data(**kwargs)

    def close(self):
        """Close the connection to the input device."""
        pass


# __init_subclass__ doesn't fire for the class it's defined on,
# so register the base class manually.
BaseInputDevice._registry[BaseInputDevice.device_class] = BaseInputDevice
