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

    def __init__(self, name: str, config: Dict[str, Any] | None = None):
        """
        Args:
            name: Instance name from config (input_device_name).
            config: The input_device_parameters dict from config.
        """
        self.name = name
        self.config = config or {}
        self.data_interface = None

    def connect(self):
        """Connect to the input device."""
        pass

    def configure(self):
        """Configure the input device with necessary settings."""
        pass

    def get_input(self) -> Any:
        """Retrieve input data from the device."""
        return None

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
