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

# Import JavaObject for pycromanager ASI stage buffer
try:
    from pycromanager import JavaObject
except ImportError:
    # Fallback if pycromanager not available
    JavaObject = None
        
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
        
        This method configures the stage for Z-stack acquisition using ASI stage buffer
        for pycromanager backend. For other backends, it may use different methods.
        """
        # Use configure_stage to set up ASI stage buffer
        config = {
            "z_start": z_start,
            "z_end": z_end,
            "z_step": z_step,
            "pad_z": 0,
        }
        self.configure_stage(config)
        logger.debug(f"Micro-Manager: Z-stack from {z_start} to {z_end}, step {z_step}, {num_planes} planes")
    
    def configure_stage(self, config: Dict[str, Any]) -> None:
        """
        Configure stage for acquisition sequences (e.g., ASI stage buffer).
        
        This method sets up ASI stage sequences with Z positions and optional
        TTL property sequences for synchronized acquisition.
        
        Args:
            config: Dictionary containing stage configuration parameters:
                   - z_start: Starting Z position (float, required)
                   - z_end: Ending Z position (float, required)
                   - z_step: Step size (float, required)
                   - pad_z: Number of padding steps at start (int, default 0)
                   - ttl_device: TTL device name for property sequences (str, default "TTL1-8")
                   - ttl_state: TTL state value for each step (str, default "18")
        """
        z_start = config.get("z_start")
        z_end = config.get("z_end")
        z_step = config.get("z_step")
        pad_z = config.get("pad_z", 0)
        ttl_device = config.get("ttl_device", "TTL1-8")
        ttl_state = config.get("ttl_state", "18")
        
        if z_start is None or z_end is None or z_step is None:
            raise ValueError("z_start, z_end, and z_step are required for stage configuration")
        
        # Get focus device
        stage = self.get_focus_device_name()
        
        # Quick semantic check for case of 1Z plane imaging + structural scan
        # spec loop will hang unless zStepSize is set to some value > 0
        if z_step == 0:
            z_step = 1
            logger.warning("z_step was 0, setting to 1 to avoid hanging")
        
        # For pycromanager backend, use JavaObject for stage sequences
        if self.backend == "pycromanager":
            if JavaObject is None:
                raise ImportError("pycromanager.JavaObject required for ASI stage buffer configuration")
            
            # Create Java objects for stage sequence
            dv = JavaObject("mmcorej.DoubleVector")
            sv = JavaObject("mmcorej.StrVector")
            dv_list = []
            
            z = z_start
            
            # Pad zstep array with extra steps at zStart
            for i in range(pad_z):
                dv.add(z)
                dv_list.append(z)
                sv.add("0")
            
            # Ascending z-steps
            while z <= z_end:
                dv.add(z)
                dv_list.append(z)
                sv.add(ttl_state)
                z += z_step
            
            logger.info(
                f"Uploading {len(dv_list)} zPositions to ASI stage: {dv_list}"
            )
            
            # Upload and configure
            self.mmc.setPosition(stage, z_start)
            self.mmc.waitForDevice(stage)
            self.mmc.stopStageSequence(stage)
            self.mmc.loadStageSequence(stage, dv)
            self.mmc.stopPropertySequence(ttl_device, "State")
            self.mmc.loadPropertySequence(ttl_device, "State", sv)
            self.mmc.startStageSequence(stage)
            self.mmc.startPropertySequence(ttl_device, "State")
            
            logger.debug(f"ASI stage buffer configured: {len(dv_list)} positions from {z_start} to {z_end}")
        
        else:
            # For pymmcore backend, ASI stage buffer may not be available
            # Log a warning and use basic Z-stack configuration
            logger.warning(
                f"ASI stage buffer configuration not fully supported for {self.backend} backend. "
                f"Using basic Z-stack configuration."
            )
            # Basic implementation: just set position range
            self.mmc.setPosition(stage, z_start)
            self.mmc.waitForDevice(stage)
    
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
        
        # Apply device properties and system properties from config
        self._apply_device_properties()
        self._apply_device_configs()
        self._apply_system_properties()
        
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
    
    def _apply_device_properties(self) -> None:
        """
        Apply device-specific properties from config.
        
        Sets properties using mmc.setProperty() for each device configured
        in HardwareConfig.devices.
        """
        if not self.config.devices:
            return
        
        for device_key, device_config in self.config.devices.items():
            device_name = device_config.device_name
            properties = device_config.properties
            
            # Convert Pydantic model to dict (handles extra="allow" fields)
            if hasattr(properties, 'model_dump'):
                props_dict = properties.model_dump(exclude_unset=True)
            else:
                props_dict = dict(properties) if hasattr(properties, '__dict__') else {}
            
            for prop_name, prop_value in props_dict.items():
                try:
                    self.mmc.setProperty(device_name, prop_name, prop_value)
                    logger.debug(f"Set {device_name}.{prop_name} = {prop_value}")
                except Exception as e:
                    logger.warning(
                        f"Could not set {device_name}.{prop_name} = {prop_value}: {e}"
                    )
    
    def _apply_device_configs(self) -> None:
        """
        Apply device config group presets from config.
        
        Sets Micro-Manager config group presets using mmc.setConfig() for each
        device configured in HardwareConfig.devices.
        """
        if not self.config.devices:
            return
        
        for device_key, device_config in self.config.devices.items():
            configs = device_config.configs
            
            for config_group, preset_name in configs.items():
                try:
                    self.mmc.setConfig(config_group, preset_name)
                    logger.debug(f"Set config {config_group} = {preset_name}")
                except Exception as e:
                    logger.warning(
                        f"Could not set config {config_group} = {preset_name}: {e}"
                    )
    
    def _apply_system_properties(self) -> None:
        """
        Apply system-level properties from config.
        
        Sets system-level Micro-Manager settings like auto_shutter,
        circular_buffer_memory_footprint, and shutter states.
        """
        if not self.config.system_properties:
            return
        
        sys_props = self.config.system_properties
        
        # Auto shutter
        if sys_props.auto_shutter is not None:
            try:
                self.mmc.setAutoShutter(sys_props.auto_shutter)
                logger.debug(f"Set auto_shutter = {sys_props.auto_shutter}")
            except Exception as e:
                logger.warning(f"Could not set auto_shutter: {e}")
        
        # Circular buffer
        if sys_props.circular_buffer_mb is not None:
            try:
                # Note: setCircularBufferMemoryFootprint unit may vary by Micro-Manager version
                # Existing code uses values like 10000 directly. We pass MB value as-is to match
                # existing behavior. If your Micro-Manager version expects bytes, multiply by 1024*1024.
                self.mmc.setCircularBufferMemoryFootprint(sys_props.circular_buffer_mb)
                logger.debug(f"Set circular_buffer_memory_footprint = {sys_props.circular_buffer_mb}")
            except Exception as e:
                logger.warning(f"Could not set circular_buffer_memory_footprint: {e}")
        
        # Shutters
        for shutter_config in sys_props.shutters:
            try:
                self.mmc.setShutterOpen(shutter_config.device_name, shutter_config.state)
                logger.debug(
                    f"Set shutter {shutter_config.device_name} = {'open' if shutter_config.state else 'closed'}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not set shutter {shutter_config.device_name}: {e}"
                )

