"""
Base class for data interfaces in CLEF.

A data interface wraps an input device to handle data acquisition,
sampling configuration, and persistence. Each input device has a
corresponding data interface.

The base class is fully instantiable with no-op implementations.
"""

import logging
from typing import Any, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class BaseDataInterface:
    """Base class for data interfaces. All methods are no-ops by default."""

    def __init__(self, input_device: Any = None, config: Dict[str, Any] | None = None):
        """
        Args:
            input_device: The BaseInputDevice instance this interface wraps.
            config: Additional configuration parameters.
        """
        self.input_device = input_device
        self.config = config or {}

    def set_input_device(self, input_device: Any):
        """Set or replace the input device for data acquisition."""
        self.input_device = input_device

    def configure_sampling(self, **kwargs):
        """Configure sampling parameters for the data interface."""
        pass

    def sample_data(self) -> Any:
        """Retrieve a single data sample from the device."""
        return None

    def get_sample_shape(self) -> tuple:
        """Get the shape of data samples."""
        return ()

    def get_sample_dtype(self) -> np.dtype:
        """Get the data type of samples."""
        return np.dtype("float64")

    def get_metadata(self) -> Dict[str, Any]:
        """Get data-specific metadata."""
        return {}

    def save_data(self, data: Any, path: Optional[str] = None):
        """Save sampled data to disk or other storage."""
        pass

    def close(self):
        """Clean up any resources used by the data interface."""
        pass
