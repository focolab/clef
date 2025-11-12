"""
Abstract base class for hardware backends.

All hardware backends must inherit from this class and implement
the required interfaces.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from hardware.camera_interface import CameraInterface
from hardware.stage_interface import StageInterface
from hardware.stimulus_interface import StimulusInterface
from config.config_manager import HardwareConfig


class BaseHardwareBackend(ABC):
    """
    Abstract base class for all hardware backends.
    
    Provides unified interface that all backends (Micro-Manager, dummy, etc.)
    must implement. Each backend wraps its specific hardware API and exposes
    it through the standard interfaces.
    """
    
    def __init__(self, config: HardwareConfig):
        """
        Initialize backend with hardware configuration.
        
        Args:
            config: HardwareConfig object containing backend settings
        """
        self.config = config
        self._camera: Optional[CameraInterface] = None
        self._stage: Optional[StageInterface] = None
        self._stimulus: Optional[StimulusInterface] = None
        self._initialized = False
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize hardware connections and devices.
        
        Must be called before using any hardware operations.
        Raises exception if initialization fails.
        
        Args:
            **kwargs: Backend-specific initialization parameters
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Close hardware connections and cleanup resources.
        
        Should be called when done with hardware to ensure
        proper cleanup.
        """
        pass
    
    @property
    def camera(self) -> CameraInterface:
        """
        Get camera interface.
        
        Returns:
            CameraInterface implementation
            
        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        if self._camera is None:
            raise RuntimeError("Camera interface not available")
        return self._camera
    
    @property
    def stage(self) -> StageInterface:
        """
        Get stage interface.
        
        Returns:
            StageInterface implementation
            
        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        if self._stage is None:
            raise RuntimeError("Stage interface not available")
        return self._stage
    
    @property
    def stimulus(self) -> StimulusInterface:
        """
        Get stimulus interface.
        
        Returns:
            StimulusInterface implementation
            
        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        if self._stimulus is None:
            raise RuntimeError("Stimulus interface not available")
        return self._stimulus
    
    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._initialized
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get hardware metadata (exposure, ROI, device properties, etc.).
        
        Returns:
            Dictionary containing hardware metadata
        """
        pass

