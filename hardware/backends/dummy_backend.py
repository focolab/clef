"""
Dummy hardware backend for testing.

Provides mock implementations of all hardware interfaces that can be used
for testing without real hardware.
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


class DummyCamera(CameraInterface):
    """Dummy camera implementation for testing."""
    
    def __init__(self, width: int = 200, height: int = 200, input_file: Optional[str] = None):
        """
        Initialize dummy camera.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            input_file: Optional path to TIFF file for simulated acquisition
        """
        self.width = width
        self.height = height
        self.exposure_ms = 10.0
        self.roi = (0, 0, width, height)
        self._acquisition_running = False
        self._buffer_count = 0
        self._frame_count = 0
        
        # Load input file if provided
        self.input_data = None
        if input_file:
            try:
                import tifffile as tf
                self.input_data = tf.imread(input_file)
                logger.info(f"Loaded dummy camera data from {input_file}, shape: {self.input_data.shape}")
            except Exception as e:
                logger.warning(f"Could not load input file {input_file}: {e}")
    
    def acquire_frame(self) -> np.ndarray:
        """Acquire a single frame."""
        return self.get_image()
    
    def start_acquisition(self, buffer_size: int = 0) -> None:
        """Start continuous acquisition."""
        self._acquisition_running = True
        self._buffer_count = 1000  # Simulate buffer with images
        logger.debug("Dummy camera: Started continuous acquisition")
    
    def stop_acquisition(self) -> None:
        """Stop continuous acquisition."""
        self._acquisition_running = False
        logger.debug("Dummy camera: Stopped acquisition")
    
    def get_remaining_image_count(self) -> int:
        """Get remaining images in buffer."""
        if not self._acquisition_running:
            return 0
        return max(0, self._buffer_count - self._frame_count)
    
    def pop_next_image(self) -> np.ndarray:
        """Pop next image from buffer."""
        self._frame_count += 1
        return self.get_image()
    
    def snap_image(self) -> None:
        """Trigger single snap."""
        pass
    
    def get_image(self) -> np.ndarray:
        """Get most recent image."""
        if self.input_data is not None and self._frame_count < len(self.input_data):
            return self.input_data[self._frame_count].astype(np.uint16)
        # Generate random noise image
        return np.random.randint(0, 65536, size=(self.height, self.width), dtype=np.uint16)
    
    def clear_buffer(self) -> None:
        """Clear buffer."""
        self._buffer_count = 0
        self._frame_count = 0
        logger.debug("Dummy camera: Cleared buffer")
    
    def set_exposure(self, exposure_ms: float) -> None:
        """Set exposure time."""
        self.exposure_ms = exposure_ms
        logger.debug(f"Dummy camera: Set exposure to {exposure_ms} ms")
    
    def get_exposure(self) -> float:
        """Get exposure time."""
        return self.exposure_ms
    
    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """Set ROI."""
        self.roi = (x, y, width, height)
        self.width = width
        self.height = height
        logger.debug(f"Dummy camera: Set ROI to ({x}, {y}, {width}, {height})")
    
    def get_roi(self) -> Tuple[int, int, int, int]:
        """Get ROI."""
        return self.roi
    
    def get_image_size(self) -> Tuple[int, int]:
        """Get image size."""
        return (self.width, self.height)
    
    def configure_camera(self, config: Dict[str, Any]) -> None:
        """Configure camera settings."""
        # Update exposure if provided
        if "exposure" in config:
            self.set_exposure(config["exposure"])
        # Other config options can be added as needed
        logger.debug(f"Dummy camera: Configured with {config}")


class DummyStage(StageInterface):
    """Dummy stage implementation for testing."""
    
    def __init__(self):
        """Initialize dummy stage."""
        self.x_position = 0.0
        self.y_position = 0.0
        self.z_position = 0.0
        self._sequence_running = False
        self._focus_device = "ZStage"
    
    def get_position(self, axis: Optional[str] = None) -> Union[float, Tuple[float, ...]]:
        """Get stage position."""
        if axis == 'X':
            return self.x_position
        elif axis == 'Y':
            return self.y_position
        elif axis == 'Z':
            return self.z_position
        else:
            return (self.x_position, self.y_position, self.z_position)
    
    def move_to_position(self, position: Union[float, Tuple[float, ...]], axis: Optional[str] = None) -> None:
        """Move stage to position."""
        if axis == 'X' or (axis is None and isinstance(position, (int, float))):
            self.x_position = float(position)
        elif axis == 'Y':
            self.y_position = float(position)
        elif axis == 'Z':
            self.z_position = float(position)
        elif isinstance(position, (tuple, list)):
            if len(position) == 3:
                self.x_position, self.y_position, self.z_position = position
            elif len(position) == 1:
                self.z_position = position[0]
        logger.debug(f"Dummy stage: Moved to {self.get_position()}")
    
    def run_z_stack(self, z_start: float, z_end: float, z_step: float, num_planes: int) -> None:
        """Configure Z-stack sequence."""
        self._sequence_running = True
        logger.debug(f"Dummy stage: Configured Z-stack from {z_start} to {z_end}, step {z_step}")
    
    def stop_sequence(self) -> None:
        """Stop stage sequence."""
        self._sequence_running = False
        logger.debug("Dummy stage: Stopped sequence")
    
    def wait_for_device(self, timeout_ms: Optional[int] = None) -> None:
        """Wait for stage movement."""
        pass
    
    def get_focus_device_name(self) -> str:
        """Get focus device name."""
        return self._focus_device
    
    def configure_stage(self, config: Dict[str, Any]) -> None:
        """Configure stage for acquisition sequences."""
        z_start = config.get("z_start")
        z_end = config.get("z_end")
        z_step = config.get("z_step")
        logger.debug(
            f"Dummy stage: Configured sequence from {z_start} to {z_end}, step {z_step}"
        )


class DummyStimulus(StimulusInterface):
    """Dummy stimulus interface for testing."""
    
    def __init__(self, **kwargs):
        """Initialize dummy stimulus."""
        self._active = False
        self._current_params = None
        self.stimulus_intensity = 0
        self.stimulus_config = {}
        self._configured = False
    
    def activate_stimulus(self, params: Dict[str, Any]) -> None:
        """
        Simulate stimulus activation.
        
        Args:
            params: Dictionary containing stimulus parameters
                   (intensity, position, diameter, etc.)
        """
        self._active = True
        self._current_params = params
        self.stimulus_intensity = params.get('intensity', 10)
        logger.debug(f"Dummy: Activated stimulus with params {params}")
    
    def deactivate_stimulus(self) -> None:
        """Simulate stimulus deactivation."""
        self._active = False
        self._current_params = None
        self.stimulus_intensity = 0
        logger.debug("Dummy: Deactivated stimulus")
    
    def configure_stimulus(self, config: Dict[str, Any]) -> None:
        """
        Store stimulus configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.stimulus_config = config
        self._configured = True
        logger.debug(f"Dummy: Configured stimulus with {config}")
    
    def is_stimulus_active(self) -> bool:
        """
        Check if stimulus is currently active.
        
        Returns:
            True if stimulus is active
        """
        return self._active


class DummyHardwareBackend(BaseHardwareBackend):
    """
    Dummy hardware backend for testing.
    
    Provides mock implementations of all hardware interfaces that can be used
    for testing without real hardware. Can optionally load data from a TIFF file
    for realistic simulation.
    """
    
    def __init__(self, config: HardwareConfig):
        """Initialize dummy backend."""
        super().__init__(config)
        # Note: input_recording_path is in ExperimentConfig, not HardwareConfig
        # This will be passed separately when needed
        self.input_file = None
    
    def initialize(self, input_file: Optional[str] = None) -> None:
        """
        Initialize dummy hardware.
        
        Args:
            input_file: Optional path to TIFF file for simulated acquisition
        """
        logger.info("Initializing dummy hardware backend...")
        
        # Create dummy camera
        self._camera = DummyCamera(input_file=input_file or self.input_file)
        
        # Create dummy stage
        self._stage = DummyStage()
        
        # Create dummy stimulus
        self._stimulus = DummyStimulus()
        
        self._initialized = True
        logger.info("Dummy hardware backend initialized")
    
    def close(self) -> None:
        """Close dummy hardware."""
        logger.info("Closing dummy hardware backend...")
        if self._camera:
            self._camera.stop_acquisition()
        self._initialized = False
        logger.info("Dummy hardware backend closed")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get hardware metadata."""
        if not self._initialized:
            return {}
        
        metadata = {
            'backend': 'dummy',
            'exposure': self._camera.get_exposure(),
            'roi': self._camera.get_roi(),
            'image_size': self._camera.get_image_size(),
            'stage_position': self._stage.get_position(),
        }
        return metadata

