"""
Abstract data interface for hardware abstraction.

Defines the interface that all data backends must implement.
This replaces camera-specific interfaces to support diverse data types
beyond uint16 microscopy images (e.g., RGB screenshots, 1D timeseries, multi-modal streams).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, Union
import numpy as np
from pathlib import Path


class DataInterface(ABC):
    """
    Abstract interface for data sampling operations.
    
    All data backends must implement these methods to provide
    a unified interface for data acquisition. This abstraction
    supports various data types (images, timeseries, screenshots, etc.)
    beyond traditional microscopy cameras.
    """
    
    @abstractmethod
    def sample_data(self) -> np.ndarray:
        """
        Retrieve next data sample.
        
        This is the primary sampling method that replaces camera-specific
        methods like acquire_frame() or pop_next_image().
        
        Returns:
            Data sample as numpy array with shape and dtype specific to data source
        """
        pass
    
    @abstractmethod
    def start_sampling(self, buffer_size: int = 0) -> None:
        """
        Start continuous data sampling.
        
        Args:
            buffer_size: Circular buffer size (0 = unlimited)
        """
        pass
    
    @abstractmethod
    def stop_sampling(self) -> None:
        """Stop continuous data sampling."""
        pass
    
    @abstractmethod
    def get_sample_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of data samples.
        
        Returns:
            Tuple describing data dimensions (e.g., (height, width) for images,
            (n_channels,) for timeseries, (height, width, 3) for RGB)
        """
        pass
    
    @abstractmethod
    def get_sample_dtype(self) -> np.dtype:
        """
        Get the data type of samples.
        
        Returns:
            NumPy dtype (e.g., np.uint16 for microscopy, np.uint8 for RGB,
            np.float32 for normalized data)
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get data-specific metadata.
        
        Returns:
            Dictionary containing metadata relevant to this data type
            (e.g., exposure time for cameras, sampling rate for timeseries)
        """
        pass
    
    @abstractmethod
    def save_data(self, data: np.ndarray, filepath: Union[str, Path], **kwargs) -> None:
        """
        Save data to disk in appropriate format.
        
        Different data types may use different formats:
        - Images: TIFF, HDF5
        - Timeseries: CSV, NPY, HDF5
        - Screenshots: PNG, JPEG
        
        Args:
            data: Data array to save
            filepath: Output file path
            **kwargs: Format-specific save options
        """
        pass
    
    @abstractmethod
    def configure_sampling(self, config: Dict[str, Any]) -> None:
        """
        Configure data sampling parameters.
        
        Args:
            config: Dictionary containing sampling configuration parameters
        """
        pass
    
    @abstractmethod
    def get_remaining_sample_count(self) -> int:
        """
        Get number of samples remaining in buffer.
        
        Returns:
            Number of samples available in buffer
        """
        pass
    
    @abstractmethod
    def clear_buffer(self) -> None:
        """Clear the sample buffer."""
        pass
    
    # Optional methods for backward compatibility with camera interface
    # Subclasses can implement these if needed for legacy code
    
    # def get_sampling_rate(self) -> Optional[float]:
    #     """
    #     Get current sampling rate.
        
    #     Returns:
    #         Sampling rate in Hz, or None if not applicable
    #     """
    #     return None
    
    # def set_sampling_rate(self, rate_hz: float) -> None:
    #     """
    #     Set sampling rate.
        
    #     Args:
    #         rate_hz: Target sampling rate in Hz
    #     """
    #     pass
    
    @property
    def data_type_name(self) -> str:
        """
        Get human-readable name of data type.
        
        Returns:
            String describing data type (e.g., "microscopy_image", "timeseries", "screenshot")
        """
        return self.__class__.__name__
