"""
Base class for data interfaces in CLEF.

A data interface wraps an input device to handle data acquisition,
sampling configuration, and persistence. Each input device has a
corresponding data interface.

Subclasses must set `data_interface_class` to a unique string that matches
the `data_interface_class` field in the IO config YAML. Registration
happens automatically via __init_subclass__.

The base class is fully instantiable with no-op implementations.
"""

import logging
from typing import Any, ClassVar, Dict, Optional, Type
import numpy as np

logger = logging.getLogger(__name__)


class BaseDataInterface:
    """Base class for data interfaces. All methods are no-ops by default."""

    # --- registry --------------------------------------------------------
    _registry: ClassVar[Dict[str, Type["BaseDataInterface"]]] = {}

    data_interface_class: ClassVar[Optional[str]] = "base_data_interface"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.data_interface_class is not None:
            BaseDataInterface._registry[cls.data_interface_class] = cls
            logger.debug(f"Registered data interface: {cls.data_interface_class} -> {cls.__name__}")

    @classmethod
    def get_class(cls, data_interface_class: str) -> Type["BaseDataInterface"]:
        """Look up a registered data interface class by its key."""
        if data_interface_class not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown data interface class: '{data_interface_class}'. "
                f"Available: {available}"
            )
        return cls._registry[data_interface_class]

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return all registered data_interface_class keys."""
        return list(cls._registry.keys())

    # --- lifecycle -------------------------------------------------------

    def __init__(self, input_device: Any = None, config: Dict[str, Any] | None = None):
        """
        Args:
            input_device: The BaseInputDevice instance this interface wraps.
            config: Additional configuration parameters.
        """
        self.input_device = input_device
        self.config = config or {}
        self.input_store: Any = None

    def set_input_device(self, input_device: Any):
        """Set or replace the input device for data acquisition."""
        self.input_device = input_device

    def configure_sampling(self, **kwargs):
        """Configure sampling parameters for the data interface."""
        pass

    def get_input(self) -> Any:
        """Retrieve a single data sample from the input device."""
        if self.input_device is not None:
            return self.input_device.get_input()
        return None

    def store_input(self, sample: Any):
        """Store input sample. Base implementation holds latest sample only.

        Subclasses may use preallocated arrays, growing lists, or other
        storage strategies as needed.
        """
        self.input_store = sample

    def get_sample_shape(self) -> tuple:
        """Get the shape of data samples."""
        return ()

    def get_sample_dtype(self) -> np.dtype:
        """Get the data type of samples."""
        return np.dtype("float64")

    def get_metadata(self) -> Dict[str, Any]:
        """Get data-specific metadata."""
        return {"data_interface_class": self.data_interface_class}

    def save_data(self, **kwargs):
        """Save sampled data to disk or other storage."""
        pass

    def close(self):
        """Clean up any resources used by the data interface."""
        pass


# __init_subclass__ doesn't fire for the class it's defined on,
# so register the base class manually.
BaseDataInterface._registry[BaseDataInterface.data_interface_class] = BaseDataInterface
