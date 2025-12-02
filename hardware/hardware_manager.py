"""
Hardware Manager - Unified hardware abstraction layer.

Provides a single interface for all hardware operations, abstracting away
the details of different backends (Micro-Manager, dummy, etc.).
"""

import logging
from typing import Optional, Dict, Any

from hardware.backends.base_backend import BaseHardwareBackend
from hardware.backends.dummy_backend import DummyHardwareBackend
from hardware.backends.micromanager_backend import MicroManagerBackend
from hardware.backends.demo_lorenz_backend import LorenzDemoBackend
from hardware.camera_interface import CameraInterface
from hardware.stage_interface import StageInterface
from hardware.stimulus_interface import StimulusInterface
from config.config_manager import HardwareConfig

logger = logging.getLogger(__name__)


class HardwareManager:
    """
    Unified hardware manager for CLEF.
    
    Provides a single interface for all hardware operations, abstracting away
    backend-specific details. Automatically selects and initializes the
    appropriate backend based on HardwareConfig.
    
    Example:
        >>> from config.config_manager import ConfigManager
        >>> from hardware import HardwareManager
        >>> 
        >>> config_manager = ConfigManager()
        >>> hardware_config = config_manager.load_hardware_config()
        >>> 
        >>> hardware = HardwareManager(hardware_config)
        >>> hardware.initialize()
        >>> 
        >>> # Use hardware interfaces
        >>> img = hardware.camera.acquire_frame()
        >>> hardware.stage.move_to_position(10.0, axis='Z')
        >>> hardware.stimulus.activate_stimulus({'intensity': 50})
        >>> 
        >>> hardware.close()
    """
    
    def __init__(self, config: HardwareConfig):
        """
        Initialize hardware manager with configuration.
        
        Args:
            config: HardwareConfig object containing backend and device settings
        """
        self.config = config
        self._backend: Optional[BaseHardwareBackend] = None
        self._initialized = False
        
        # Select backend based on config
        self._select_backend()
    
    def _select_backend(self) -> None:
        """Select appropriate backend based on configuration."""
        backend_type = self.config.backend.lower()
        
        if backend_type == "dummy" or backend_type == "test":
            self._backend = DummyHardwareBackend(self.config)
            logger.info("Selected DummyHardwareBackend")
        elif backend_type in ["pycromanager", "pymmcore"]:
            self._backend = MicroManagerBackend(self.config)
            logger.info(f"Selected MicroManagerBackend for {backend_type}")
        elif backend_type == "lorenz_demo":
            self._backend = LorenzDemoBackend(self.config)
            logger.info("Selected LorenzDemoBackend")
        else:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Supported: 'dummy', 'test', 'pycromanager', 'pymmcore', 'lorenz_demo'"
            )
    
    def initialize(self, **kwargs) -> None:
        """
        Initialize hardware connections and devices.
        
        Must be called before using any hardware operations.
        Raises exception if initialization fails.
        
        Args:
            **kwargs: Backend-specific initialization parameters
                     (e.g., input_recording for dummy backend)
        """
        if self._initialized:
            logger.warning("Hardware already initialized")
            return
        
        logger.info("Initializing hardware manager...")
        self._backend.initialize(**kwargs)
        self._initialized = True
        logger.info("Hardware manager initialized successfully")
    
    def close(self) -> None:
        """
        Close hardware connections and cleanup resources.
        
        Should be called when done with hardware to ensure proper cleanup.
        """
        if not self._initialized:
            return
        
        logger.info("Closing hardware manager...")
        if self._backend:
            self._backend.close()
        self._initialized = False
        logger.info("Hardware manager closed")
    
    @property
    def camera(self) -> CameraInterface:
        """
        Get camera interface.
        
        Returns:
            CameraInterface for image acquisition operations
            
        Raises:
            RuntimeError: If hardware not initialized
        """
        if not self._initialized:
            raise RuntimeError("Hardware not initialized. Call initialize() first.")
        return self._backend.camera
    
    @property
    def stage(self) -> StageInterface:
        """
        Get stage interface.
        
        Returns:
            StageInterface for stage control operations
            
        Raises:
            RuntimeError: If hardware not initialized
        """
        if not self._initialized:
            raise RuntimeError("Hardware not initialized. Call initialize() first.")
        return self._backend.stage
    
    @property
    def stimulus(self) -> StimulusInterface:
        """
        Get stimulus interface.
        
        Returns:
            StimulusInterface for stimulus control operations
            
        Raises:
            RuntimeError: If hardware not initialized
        """
        if not self._initialized:
            raise RuntimeError("Hardware not initialized. Call initialize() first.")
        return self._backend.stimulus
    
    @property
    def is_initialized(self) -> bool:
        """Check if hardware is initialized."""
        return self._initialized
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get hardware metadata (exposure, ROI, device properties, etc.).
        
        Returns:
            Dictionary containing hardware metadata
        """
        if not self._initialized:
            return {}
        return self._backend.get_metadata()
    
    def get_backend(self) -> BaseHardwareBackend:
        """
        Get underlying backend object.
        
        This method provides access to the backend for components that
        haven't been fully refactored yet. This is temporary and will be
        removed as components are migrated to use the hardware abstraction.
        
        Returns:
            BaseHardwareBackend instance
            
        Raises:
            RuntimeError: If hardware not initialized
        """
        if not self._initialized:
            raise RuntimeError("Hardware not initialized. Call initialize() first.")
        return self._backend
    
    def get_mmc(self):
        """
        Get underlying Micro-Manager Core object (if using Micro-Manager backend).
        
        This method provides access to the raw MMC object for components
        that haven't been refactored yet. This is temporary and will be
        removed as components are migrated to use the hardware abstraction.
        
        Returns:
            Micro-Manager Core object, or None if not using Micro-Manager backend
            
        Raises:
            RuntimeError: If hardware not initialized
        """
        if not self._initialized:
            raise RuntimeError("Hardware not initialized. Call initialize() first.")
        
        if isinstance(self._backend, MicroManagerBackend):
            return self._backend.get_mmc()

        if isinstance(self._backend, DummyHardwareBackend):
            return self._backend.get_mmc()
            
        return None

