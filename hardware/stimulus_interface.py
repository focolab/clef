"""
Abstract stimulus interface for hardware abstraction.

Defines the interface that all stimulus backends must implement.
Note: This is a hardware-level abstraction. Higher-level stimulus
logic (timing, triggers) is handled by StimBaseClass and subclasses.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class StimulusInterface(ABC):
    """
    Abstract interface for stimulus hardware operations.
    
    This provides low-level hardware control. Higher-level stimulus
    management (trigger detection, timing, cooldowns) is handled
    by the algorithm and StimBaseClass.
    """
    
    @abstractmethod
    def activate_stimulus(self, params: Dict[str, Any]) -> None:
        """
        Activate stimulus with given parameters.
        
        Args:
            params: Dictionary containing stimulus parameters:
                   - position: (x, y) tuple in pixels or coordinates
                   - diameter: diameter in pixels
                   - intensity: intensity value (device-specific units)
                   - duration: duration in frames or milliseconds
                   - channel: optional channel/color specification
        """
        pass
    
    @abstractmethod
    def deactivate_stimulus(self) -> None:
        """Deactivate currently active stimulus."""
        pass
    
    @abstractmethod
    def configure_stimulus(self, config: Dict[str, Any]) -> None:
        """
        Configure stimulus hardware settings.
        
        Args:
            config: Dictionary containing configuration parameters:
                   - roi: optional static ROI
                   - calibration_path: optional calibration file path
                   - device_properties: device-specific properties
        """
        pass
    
    @abstractmethod
    def is_stimulus_active(self) -> bool:
        """
        Check if stimulus is currently active.
        
        Returns:
            True if stimulus is currently active
        """
        pass

