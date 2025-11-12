"""
Abstract camera interface for hardware abstraction.

Defines the interface that all camera backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np


class CameraInterface(ABC):
    """
    Abstract interface for camera operations.
    
    All camera backends must implement these methods to provide
    a unified interface for image acquisition.
    """
    
    @abstractmethod
    def acquire_frame(self) -> np.ndarray:
        """
        Acquire a single frame from the camera.
        
        Returns:
            Image as numpy array (uint16), shape (height, width)
        """
        pass
    
    @abstractmethod
    def start_acquisition(self, buffer_size: int = 0) -> None:
        """
        Start continuous sequence acquisition.
        
        Args:
            buffer_size: Circular buffer size (0 = unlimited)
        """
        pass
    
    @abstractmethod
    def stop_acquisition(self) -> None:
        """Stop continuous sequence acquisition."""
        pass
    
    @abstractmethod
    def set_exposure(self, exposure_ms: float) -> None:
        """
        Set camera exposure time.
        
        Args:
            exposure_ms: Exposure time in milliseconds
        """
        pass
    
    @abstractmethod
    def get_exposure(self) -> float:
        """
        Get current camera exposure time.
        
        Returns:
            Exposure time in milliseconds
        """
        pass
    
    @abstractmethod
    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """
        Set camera region of interest (ROI).
        
        Args:
            x: Left edge of ROI in pixels
            y: Top edge of ROI in pixels
            width: Width of ROI in pixels
            height: Height of ROI in pixels
        """
        pass
    
    @abstractmethod
    def get_roi(self) -> Tuple[int, int, int, int]:
        """
        Get current camera ROI.
        
        Returns:
            Tuple of (x, y, width, height) in pixels
        """
        pass
    
    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        """
        Get current image dimensions.
        
        Returns:
            Tuple of (width, height) in pixels
        """
        pass

    @abstractmethod
    def configure_camera(self, config: Dict[str, Any]) -> None:
        """
        Configure camera settings.
        
        Args:
            config: Dictionary containing camera configuration parameters
        """
        pass

    ############################################################################
    # We likely don't need these methods for camera interfaces, keeping them for now
    @abstractmethod
    def get_image(self) -> np.ndarray:
        """
        Get the most recently snapped image.
        
        Returns:
            Image as numpy array (uint16), shape (height, width)
        """
        pass

    @abstractmethod
    def get_remaining_image_count(self) -> int:
        """
        Get number of images remaining in buffer.
        
        Returns:
            Number of images available in circular buffer
        """
        pass
    
    @abstractmethod
    def pop_next_image(self) -> np.ndarray:
        """
        Pop next image from circular buffer.
        
        Returns:
            Image as numpy array (uint16), shape (height, width)
        """
        pass
    
    @abstractmethod
    def snap_image(self) -> None:
        """
        Trigger a single image snap (for strobe acquisition).
        Does not return image - use get_image() after.
        """
        pass
        
    @abstractmethod
    def clear_buffer(self) -> None:
        """Clear the circular buffer."""
        pass