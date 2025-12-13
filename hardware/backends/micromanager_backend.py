"""
Micro-Manager hardware backend.

Wraps Micro-Manager (MMC) API calls to provide unified hardware interface.
Supports both pycromanager and pymmcore apis.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

from hardware.backends.base_backend import BaseHardwareBackend
from hardware.camera_interface import CameraInterface
from hardware.stage_interface import StageInterface
from hardware.stimulus_interface import StimulusInterface
from config.config_manager import HardwareConfig

logger = logging.getLogger(__name__)

# Import MMSubroutines to use existing initialization logic
from lib import MMSubroutines
        
class MicroManagerCamera(CameraInterface):
    """Camera interface wrapping Micro-Manager Core."""
    
    def __init__(self, mmc, backend: str, roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize Micro-Manager camera interface.
        
        Args:
            mmc: Micro-Manager Core object
            backend: Backend type ('pycromanager' or 'pymmcore')
            roi: Optional initial ROI (x, y, width, height)
        """
        self.mmc = mmc
        self.backend = backend
        
        # Get initial ROI
        if roi:
            self._roi = roi
            self.mmc.setROI(roi[0], roi[1], roi[2], roi[3])
        else:
            roi_obj = self.mmc.getROI()
            if backend == "pycromanager":
                # pycromanager returns Java objects
                self._roi = (roi_obj.getX(), roi_obj.getY(), 
                           roi_obj.getWidth(), roi_obj.getHeight())
            else:
                # pymmcore returns tuple
                self._roi = roi_obj
        
        self._width = self._roi[2]
        self._height = self._roi[3]
    
    def acquire_frame(self) -> np.ndarray:
        """Acquire a single frame."""
        self.mmc.snapImage()
        return self.get_image()
    
    def start_acquisition(self, buffer_size: int = 0) -> None:
        """Start continuous sequence acquisition."""
        self.mmc.setCircularBufferMemoryFootprint(buffer_size if buffer_size > 0 else 10000)
        self.mmc.startContinuousSequenceAcquisition(0)
        logger.debug("Micro-Manager: Started continuous acquisition")
    
    def stop_acquisition(self) -> None:
        """Stop continuous sequence acquisition."""
        self.mmc.stopSequenceAcquisition()
        logger.debug("Micro-Manager: Stopped acquisition")
    
    def get_remaining_image_count(self) -> int:
        """Get remaining images in buffer."""
        return self.mmc.getRemainingImageCount()
    
    def pop_next_image(self) -> np.ndarray:
        """Pop next image from buffer."""
        img = self.mmc.popNextImage().astype(np.uint16)
        if self.backend == "pycromanager":
            img = img.reshape((self._height, self._width))
        return img
    
    def snap_image(self) -> None:
        """Trigger single snap."""
        self.mmc.snapImage()
    
    def get_image(self) -> np.ndarray:
        """Get most recent image."""
        img = self.mmc.getImage().astype(np.uint16)
        if self.backend == "pycromanager":
            img = img.reshape((self._height, self._width))
        return img
    
    def clear_buffer(self) -> None:
        """Clear circular buffer."""
        self.mmc.clearCircularBuffer()
        logger.debug("Micro-Manager: Cleared buffer")
    
    def set_exposure(self, exposure_ms: float) -> None:
        """Set exposure time."""
        self.mmc.setExposure(exposure_ms)
        logger.debug(f"Micro-Manager: Set exposure to {exposure_ms} ms")
    
    def get_exposure(self) -> float:
        """Get exposure time."""
        return self.mmc.getExposure()
    
    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """Set ROI."""
        self.mmc.setROI(x, y, width, height)
        self._roi = (x, y, width, height)
        self._width = width
        self._height = height
        logger.debug(f"Micro-Manager: Set ROI to ({x}, {y}, {width}, {height})")
    
    def get_roi(self) -> Tuple[int, int, int, int]:
        """Get ROI."""
        roi_obj = self.mmc.getROI()
        if self.backend == "pycromanager":
            return (roi_obj.getX(), roi_obj.getY(), 
                   roi_obj.getWidth(), roi_obj.getHeight())
        else:
            return roi_obj
    
    def get_image_size(self) -> Tuple[int, int]:
        """Get image size."""
        return (self._width, self._height)
    
    def configure_camera(self, config: Dict[str, Any]) -> None:
        """
        Configure camera settings.
        
        Args:
            config: Dictionary containing camera configuration parameters
                   (e.g., exposure, binning, properties)
        """
        # Set exposure if provided
        if "exposure" in config:
            self.set_exposure(config["exposure"])
        
        # Set ROI if provided
        if "roi" in config:
            roi = config["roi"]
            if isinstance(roi, (tuple, list)) and len(roi) == 4:
                self.set_roi(roi[0], roi[1], roi[2], roi[3])
        
        # Set device properties if provided
        if "properties" in config:
            cam_device = self.mmc.getCameraDevice()
            for prop_name, prop_value in config["properties"].items():
                try:
                    self.mmc.setProperty(cam_device, prop_name, prop_value)
                except Exception as e:
                    logger.warning(f"Could not set property {prop_name}={prop_value}: {e}")
        
        logger.debug(f"Micro-Manager: Camera configured with {config}")


class MicroManagerStage(StageInterface):
    """Stage interface wrapping Micro-Manager Core."""
    
    def __init__(self, mmc, backend: str):
        """
        Initialize Micro-Manager stage interface.
        
        Args:
            mmc: Micro-Manager Core object
            backend: Backend type ('pycromanager' or 'pymmcore')
        """
        self.mmc = mmc
        self.backend = backend
        self._focus_device = None
    
    def get_position(self, axis: Optional[str] = None) -> Union[float, Tuple[float, ...]]:
        """Get stage position."""
        if axis == 'Z' or axis is None:
            focus_dev = self.get_focus_device_name()
            pos = self.mmc.getPosition(focus_dev)
            if axis == 'Z':
                return pos
            return (pos,)
        else:
            # For X/Y stages, would need to get XYStage device
            # For now, just return Z position
            focus_dev = self.get_focus_device_name()
            return (self.mmc.getPosition(focus_dev),)
    
    def move_to_position(self, position: Union[float, Tuple[float, ...]], axis: Optional[str] = None) -> None:
        """Move stage to position."""
        focus_dev = self.get_focus_device_name()
        if isinstance(position, (int, float)):
            self.mmc.setPosition(focus_dev, float(position))
        elif isinstance(position, (tuple, list)) and len(position) > 0:
            self.mmc.setPosition(focus_dev, float(position[0]))
        logger.debug(f"Micro-Manager: Moved stage to {position}")
    
    def run_z_stack(self, z_start: float, z_end: float, z_step: float, num_planes: int) -> None:
        """
        Configure Z-stack sequence.
        
        Note: This is a simplified implementation. Full Z-stack configuration
        may require backend-specific logic (e.g., ASI stage buffer upload).
        """
        # For now, just log - full implementation would use set_asi_stage_buffer
        # or similar backend-specific methods
        logger.debug(f"Micro-Manager: Z-stack from {z_start} to {z_end}, step {z_step}")
        # TODO: Implement full Z-stack sequence configuration
    
    def stop_sequence(self) -> None:
        """Stop stage sequence."""
        focus_dev = self.get_focus_device_name()
        self.mmc.stopStageSequence(focus_dev)
        logger.debug("Micro-Manager: Stopped stage sequence")
    
    def wait_for_device(self, timeout_ms: Optional[int] = None) -> None:
        """Wait for stage movement."""
        focus_dev = self.get_focus_device_name()
        self.mmc.waitForDevice(focus_dev)
    
    def get_focus_device_name(self) -> str:
        """Get focus device name."""
        if self._focus_device is None:
            self._focus_device = self.mmc.getFocusDevice()
        return self._focus_device


class MicroManagerStimulus(StimulusInterface):
    """
    Stimulus interface wrapping Micro-Manager Core.
    
    Note: This provides low-level hardware control. Most stimulus operations
    in CLEF are handled by StimBaseClass subclasses which may access MMC directly.
    This interface provides a bridge for future refactoring.
    """
    
    def __init__(self, mmc, backend: str):
        """
        Initialize Micro-Manager stimulus interface.
        
        Args:
            mmc: Micro-Manager Core object
            backend: Backend type ('pycromanager' or 'pymmcore')
        """
        self.mmc = mmc
        self.backend = backend
        self._active = False
    
    def activate_stimulus(self, params: Dict[str, Any]) -> None:
        """
        Activate stimulus.
        
        Note: This is a placeholder. Actual stimulus activation is typically
        handled by StimBaseClass subclasses (e.g., InvCoreSpinningDisk639).
        """
        self._active = True
        logger.debug(f"Micro-Manager: Stimulus activated with params {params}")
        # TODO: Implement based on specific stimulus hardware
    
    def deactivate_stimulus(self) -> None:
        """Deactivate stimulus."""
        self._active = False
        logger.debug("Micro-Manager: Stimulus deactivated")
        # TODO: Implement based on specific stimulus hardware
    
    def configure_stimulus(self, config: Dict[str, Any]) -> None:
        """Configure stimulus hardware."""
        logger.debug(f"Micro-Manager: Stimulus configured with {config}")
        # TODO: Implement based on specific stimulus hardware
    
    def is_stimulus_active(self) -> bool:
        """Check if stimulus is active."""
        return self._active


class MicroManagerBackend(BaseHardwareBackend):
    """
    Micro-Manager hardware backend.
    
    Wraps Micro-Manager Core API to provide unified hardware interface.
    Supports both pycromanager and pymmcore backends.
    """
    
    def __init__(self, config: HardwareConfig):
        """Initialize Micro-Manager backend."""
        super().__init__(config)
        self.mmc = None
    
    def initialize(self, input_recording: Optional[str] = None) -> None:
        """
        Initialize Micro-Manager hardware.
        
        Args:
            input_recording: Optional path to input recording for playback mode
        """
        logger.info(f"Initializing Micro-Manager backend: {self.config.backend}")

        # Build args dict for MMSubroutines (temporary bridge)
        # This will be removed when MMSubroutines is fully refactored
        args = {
            "gooey_args": {
                "acquisition_backend": self.config.backend,
                "microscope_name": self.config.microscope_name or "unknown",
                "input_recording": input_recording,
            }
        }
        
        # Use config file from HardwareConfig
        config_file = self.config.mm_config_path
        
        # Initialize MMC using existing function
        self.mmc = MMSubroutines.initialize_mmc(args, config_file=config_file)
        
        # Get ROI
        roi_obj = self.mmc.getROI()
        if self.config.backend == "pycromanager":
            roi = (roi_obj.getX(), roi_obj.getY(), 
                  roi_obj.getWidth(), roi_obj.getHeight())
        else:
            roi = roi_obj
        
        # Create interfaces
        self._camera = MicroManagerCamera(self.mmc, self.config.backend, roi)
        self._stage = MicroManagerStage(self.mmc, self.config.backend)
        self._stimulus = MicroManagerStimulus(self.mmc, self.config.backend)
        
        self._initialized = True
        logger.info("Micro-Manager backend initialized")
    
    def close(self) -> None:
        """Close Micro-Manager hardware."""
        logger.info("Closing Micro-Manager backend...")
        
        if self._camera:
            self._camera.stop_acquisition()
        
        # Use MMSubroutines.close for cleanup
        if self.mmc:
            # Build minimal args for close function
            args = {
                "gooey_args": {
                    "microscope_name": self.config.microscope_name or "unknown",
                }
            }
            MMSubroutines.close(self.mmc, args)
        
        self._initialized = False
        logger.info("Micro-Manager backend closed")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get hardware metadata."""
        if not self._initialized or not self.mmc:
            return {}
        
        # Use MMSubroutines.get_metadata
        args = {
            "gooey_args": {
                "microscope_name": self.config.microscope_name or "unknown",
            }
        }
        metadata = MMSubroutines.get_metadata(args, self.mmc)
        metadata['backend'] = self.config.backend
        return metadata
    
    def get_mmc(self):
        """
        Get underlying Micro-Manager Core object.
        
        This method provides access to the raw MMC object for components
        that haven't been refactored yet. This is temporary and will be
        removed as components are migrated to use the hardware abstraction.
        
        Returns:
            Micro-Manager Core object
        """
        return self.mmc

