"""
Abstract projector/SLM interface for hardware abstraction.

Defines the interface that all projector/SLM backends must implement.
Used for DMD, SLM, and polygon scanning devices.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class ProjectorInterface(ABC):
    """
    Abstract interface for projector/SLM operations.
    
    All projector backends must implement these methods to provide
    a unified interface for spatial light modulators and projectors.
    """
    
    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int]:
        """
        Get projector dimensions.
        
        Returns:
            Tuple of (width, height) in pixels
        """
        pass
    
    @abstractmethod
    def set_image(self, image: np.ndarray) -> None:
        """
        Set projector image/mask.
        
        Args:
            image: 2D numpy array (uint8), shape (height, width)
                  Values 0-255 for grayscale masks
        """
        pass
    
    @abstractmethod
    def set_pixels_to(self, value: int) -> None:
        """
        Set all pixels to a uniform value.
        
        Args:
            value: Pixel value (0-255)
        """
        pass
    
    @abstractmethod
    def get_device_name(self) -> str:
        """
        Get the projector device name.
        
        Returns:
            Device name string
        """
        pass
