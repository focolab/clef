"""
Abstract stage interface for hardware abstraction.

Defines the interface that all stage backends must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union


class StageInterface(ABC):
    """
    Abstract interface for stage operations.
    
    All stage backends must implement these methods to provide
    a unified interface for stage control.
    """
    
    @abstractmethod
    def get_position(self, axis: Optional[str] = None) -> Union[float, Tuple[float, ...]]:
        """
        Get current stage position.
        
        Args:
            axis: Optional axis name ('X', 'Y', 'Z'). If None, returns all axes.
        
        Returns:
            Position(s) in micrometers. If axis specified, returns float.
            If None, returns tuple of (x, y, z) or (z,) for focus-only stages.
        """
        pass
    
    @abstractmethod
    def move_to_position(self, position: Union[float, Tuple[float, ...]], axis: Optional[str] = None) -> None:
        """
        Move stage to specified position.
        
        Args:
            position: Target position in micrometers. Can be single value for Z-axis
                     or tuple for multi-axis stages.
            axis: Optional axis name ('X', 'Y', 'Z'). If None and position is tuple,
                 assumes (x, y, z) or (z,) for focus-only.
        """
        pass
    
    @abstractmethod
    def run_z_stack(self, z_start: float, z_end: float, z_step: float, num_planes: int) -> None:
        """
        Configure and run a Z-stack acquisition sequence.
        
        Args:
            z_start: Starting Z position in micrometers
            z_end: Ending Z position in micrometers
            z_step: Step size in micrometers
            num_planes: Number of Z planes
        """
        pass
    
    @abstractmethod
    def stop_sequence(self) -> None:
        """Stop any running stage sequence."""
        pass
    
    ################################################################################
    # We likely don't need these methods for stage interfaces, keeping for now
    @abstractmethod
    def wait_for_device(self, timeout_ms: Optional[int] = None) -> None:
        """
        Wait for stage to finish current movement.
        
        Args:
            timeout_ms: Optional timeout in milliseconds
        """
        pass
    
    @abstractmethod
    def get_focus_device_name(self) -> str:
        """
        Get the name of the focus device.
        
        Returns:
            Device name string
        """
        pass

