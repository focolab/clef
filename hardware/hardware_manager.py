"""
Hardware Manager - Unified hardware abstraction layer with data interface support.

"""

import logging
from typing import Optional, Dict, Any

from hardware.backends.base_backend import BaseHardwareBackend
from hardware.backends.dummy_backend import DummyHardwareBackend
from hardware.backends.micromanager_backend import MicroManagerBackend
from hardware.backends.demo_lorenz_backend import LorenzDemoBackend
from hardware.backends.screenshot_backend import ScreenshotBackend
from hardware.camera_interface import CameraInterface
from hardware.stage_interface import StageInterface
from hardware.stimulus_interface import StimulusInterface
from hardware.data_interface import DataInterface
from hardware.image_data_interface import ImageDataInterface
from config.config_manager import HardwareConfig

logger = logging.getLogger(__name__)


class HardwareManager:
    """
    Unified hardware manager for CLEF with data interface abstraction.
    
    UPDATED: Now exposes a generic DataInterface that can handle diverse
    data types (images, timeseries, screenshots, etc.) while maintaining
    backward compatibility with camera-based microscopy.
    """
    
    def __init__(self, config: HardwareConfig):
        """
        Initialize hardware manager with configuration.
        
        Args:
            config: HardwareConfig object containing backend and device settings
        """
        self.config = config
        self._backend: Optional[BaseHardwareBackend] = None
        self._data_interface: Optional[DataInterface] = None
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
        elif backend_type == "screenshot":
            self._backend = ScreenshotBackend(self.config)
            logger.info("Selected ScreenshotBackend")
        else:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Supported: 'dummy', 'test', 'pycromanager', 'pymmcore', 'lorenz_demo', 'screenshot'"
            )
    
    def initialize(self, **kwargs) -> None:
        """Initialize hardware connections and devices."""
        if self._initialized:
            logger.warning("Hardware already initialized")
            return
        
        logger.info("Initializing hardware manager...")
        self._backend.initialize(**kwargs)
        
        # Create appropriate data interface based on backend type
        if isinstance(self._backend, ScreenshotBackend):
            # NEW: Use RGBDataInterface for screenshot backend
            from hardware.rgb_data_interface import RGBDataInterface
            self._data_interface = RGBDataInterface(self._backend.screenshot_source)
            logger.debug("Created RGBDataInterface for screenshot backend")
        else:
            # Use ImageDataInterface for camera-based backends
            from hardware.image_data_interface import ImageDataInterface
            self._data_interface = ImageDataInterface(self._backend.camera)
            logger.debug("Created ImageDataInterface wrapping camera")
        
        self._initialized = True
        logger.info("Hardware manager initialized successfully")
    
    def close(self) -> None:
        """Close hardware connections and cleanup resources."""
        if not self._initialized:
            return
        
        logger.info("Closing hardware manager...")
        if self._backend:
            self._backend.close()
        self._data_interface = None
        self._initialized = False
        logger.info("Hardware manager closed")
    
    @property
    def data(self) -> DataInterface:
        """
        Get data interface for generic data sampling.
        
        NEW: Primary interface for data acquisition. Replaces direct
        camera access for most use cases.
        
        Returns:
            DataInterface for data sampling operations
            
        Raises:
            RuntimeError: If hardware not initialized
        """
        if not self._initialized:
            raise RuntimeError("Hardware not initialized. Call initialize() first.")
        return self._data_interface
    
    @property
    def camera(self) -> CameraInterface:
        """
        Get camera interface (backward compatibility).
        
        DEPRECATED: Use hardware.data for new code. This property is
        maintained for backward compatibility with existing code.
        
        Returns:
            CameraInterface for image acquisition operations
            
        Raises:
            RuntimeError: If hardware not initialized
        """
        if not self._initialized:
            raise RuntimeError("Hardware not initialized. Call initialize() first.")
        logger.debug("Accessing camera directly (consider using hardware.data instead)")
        return self._backend.camera
    
    @property
    def stage(self) -> StageInterface:
        """Get stage interface."""
        if not self._initialized:
            raise RuntimeError("Hardware not initialized. Call initialize() first.")
        return self._backend.stage
    
    @property
    def stimulus(self) -> StimulusInterface:
        """Get stimulus interface."""
        if not self._initialized:
            raise RuntimeError("Hardware not initialized. Call initialize() first.")
        return self._backend.stimulus
    
    @property
    def is_initialized(self) -> bool:
        """Check if hardware is initialized."""
        return self._initialized
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get hardware metadata including data format information.
        
        Returns:
            Dictionary containing hardware and data metadata
        """
        if not self._initialized:
            return {}
        
        metadata = self._backend.get_metadata()
        
        # Add data interface metadata
        if self._data_interface:
            metadata['data'] = self._data_interface.get_metadata()
        
        return metadata
    
    def get_backend(self) -> BaseHardwareBackend:
        """
        Get underlying backend object.
        
        TEMPORARY: For components not yet refactored.
        """
        if not self._initialized:
            raise RuntimeError("Hardware not initialized. Call initialize() first.")
        return self._backend
    
    def get_mmc(self):
        """
        Get underlying Micro-Manager Core object.
        
        TEMPORARY: For components not yet refactored.
        """
        if not self._initialized:
            raise RuntimeError("Hardware not initialized. Call initialize() first.")
        
        if isinstance(self._backend, MicroManagerBackend):
            return self._backend.get_mmc()
        if isinstance(self._backend, DummyHardwareBackend):
            return self._backend.get_mmc()
        return None
