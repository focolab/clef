"""
Screenshot hardware backend for RGB screen capture.

Provides mock hardware interfaces that capture screen regions as RGB data.
"""

import logging
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple

from hardware.backends.base_backend import BaseHardwareBackend
from hardware.camera_interface import CameraInterface
from hardware.stage_interface import StageInterface
from hardware.stimulus_interface import StimulusInterface
from hardware.rgb_data_interface import RGBDataInterface
from config.config_manager import HardwareConfig

logger = logging.getLogger(__name__)


class ScreenshotSource:
    """
    Screen capture source using PIL or mss.
    
    Captures RGB data from a specified screen region.
    """
    
    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 800,
        height: int = 600,
        monitor: int = 1,
        backend: str = "mss"
    ):
        """
        Initialize screenshot source.
        
        Args:
            x: Left edge of capture region
            y: Top edge of capture region
            width: Capture width
            height: Capture height
            monitor: Monitor number (1-indexed)
            backend: Capture backend ("mss" or "pil")
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.monitor = monitor
        self.backend_type = backend.lower()
        
        # Initialize capture backend
        self._init_backend()
        
        logger.info(
            f"ScreenshotSource initialized: region=({x}, {y}, {width}, {height}), "
            f"backend={self.backend_type}"
        )
    
    def _init_backend(self) -> None:
        """Initialize the capture backend."""
        if self.backend_type == "mss":
            try:
                import mss
                self.mss = mss.mss()
                self._capture_func = self._capture_mss
                logger.debug("Using mss backend for screen capture")
            except ImportError:
                logger.warning("mss not available, falling back to PIL")
                self.backend_type = "pil"
                self._init_pil_backend()
        elif self.backend_type == "pil":
            self._init_pil_backend()
        else:
            raise ValueError(f"Unknown backend: {self.backend_type}")
    
    def _init_pil_backend(self) -> None:
        """Initialize PIL backend."""
        try:
            from PIL import ImageGrab
            self.ImageGrab = ImageGrab
            self._capture_func = self._capture_pil
            logger.debug("Using PIL backend for screen capture")
        except ImportError:
            raise ImportError("Neither mss nor PIL available for screen capture")
    
    def _capture_mss(self) -> np.ndarray:
        """Capture using mss."""
        monitor = {
            "left": self.x,
            "top": self.y,
            "width": self.width,
            "height": self.height,
        }
        
        # Capture and convert to numpy RGB
        sct_img = self.mss.grab(monitor)
        # mss returns BGRA, convert to RGB
        img = np.array(sct_img)[:, :, :3]  # Drop alpha
        img = img[:, :, [2, 1, 0]]  # BGR to RGB
        
        return img.astype(np.uint8)
    
    def _capture_pil(self) -> np.ndarray:
        """Capture using PIL."""
        bbox = (self.x, self.y, self.x + self.width, self.y + self.height)
        img = self.ImageGrab.grab(bbox)
        return np.array(img)
    
    def capture_screen(self) -> np.ndarray:
        """
        Capture current screen region.
        
        Returns:
            RGB image as numpy array (uint8), shape (height, width, 3)
        """
        return self._capture_func()
    
    def get_capture_dimensions(self) -> Tuple[int, int]:
        """
        Get capture dimensions.
        
        Returns:
            Tuple of (height, width)
        """
        return (self.height, self.width)
    
    def get_capture_region(self) -> Dict[str, int]:
        """
        Get capture region.
        
        Returns:
            Dictionary with x, y, width, height
        """
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
        }
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'mss'):
            self.mss.close()


class ScreenshotCamera(CameraInterface):
    """
    Camera interface adapter for screenshot source.
    
    Provides camera-like interface for backward compatibility,
    but delegates to ScreenshotSource for actual capture.
    """
    
    def __init__(self, screenshot_source: ScreenshotSource):
        """Initialize with screenshot source."""
        self.source = screenshot_source
        self.exposure_ms = 30  # Approximate frame time
        self._acquisition_running = False
        self.roi = (0, 0, screenshot_source.width, screenshot_source.height)
    
    def acquire_frame(self) -> np.ndarray:
        """Acquire a single frame."""
        img = self.source.capture_screen()
        time.sleep(self.exposure_ms / 1000)
        return img
    
    def start_acquisition(self, buffer_size: int = 0) -> None:
        """Start continuous acquisition."""
        self._acquisition_running = True
        logger.debug("ScreenshotCamera: Started continuous acquisition")
    
    def stop_acquisition(self) -> None:
        """Stop continuous acquisition."""
        self._acquisition_running = False
        logger.debug("ScreenshotCamera: Stopped acquisition")
    
    def get_remaining_image_count(self) -> int:
        """Get remaining images in buffer."""
        return 1  # Always have data available
    
    def pop_next_image(self) -> np.ndarray:
        """Pop next image from buffer."""
        return self.acquire_frame()
    
    def snap_image(self) -> None:
        """Trigger single snap."""
        pass
    
    def get_image(self) -> np.ndarray:
        """Get most recent image."""
        return self.source.capture_screen()
    
    def clear_buffer(self) -> None:
        """Clear buffer."""
        pass
    
    def set_exposure(self, exposure_ms: float) -> None:
        """Set exposure time (frame delay)."""
        self.exposure_ms = exposure_ms
        logger.debug(f"ScreenshotCamera: Set exposure to {exposure_ms} ms")
    
    def get_exposure(self) -> float:
        """Get exposure time."""
        return self.exposure_ms
    
    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """Set ROI (updates capture region)."""
        self.source.x = x
        self.source.y = y
        self.source.width = width
        self.source.height = height
        self.roi = (x, y, width, height)
        logger.debug(f"ScreenshotCamera: Set ROI to ({x}, {y}, {width}, {height})")
    
    def get_roi(self) -> Tuple[int, int, int, int]:
        """Get ROI."""
        return self.roi
    
    def get_image_size(self) -> Tuple[int, int]:
        """Get image size."""
        return (self.source.width, self.source.height)
    
    def configure_camera(self, config: Dict[str, Any]) -> None:
        """Configure camera settings."""
        if "exposure" in config:
            self.set_exposure(config["exposure"])
        logger.debug(f"ScreenshotCamera: Configured with {config}")


class DummyStage(StageInterface):
    """Dummy stage for screenshot backend."""
    
    def __init__(self):
        self.x_position = 0.0
        self.y_position = 0.0
        self.z_position = 0.0
    
    def get_position(self, axis: Optional[str] = None):
        if axis == 'X':
            return self.x_position
        elif axis == 'Y':
            return self.y_position
        elif axis == 'Z':
            return self.z_position
        return (self.x_position, self.y_position, self.z_position)
    
    def move_to_position(self, position, axis: Optional[str] = None) -> None:
        if axis == 'Z':
            self.z_position = float(position)
    
    def run_z_stack(self, z_start: float, z_end: float, z_step: float, num_planes: int) -> None:
        pass
    
    def stop_sequence(self) -> None:
        pass
    
    def wait_for_device(self, timeout_ms: Optional[int] = None) -> None:
        pass
    
    def get_focus_device_name(self) -> str:
        return "DummyZ"
    
    def configure_stage(self, config: Dict[str, Any]) -> None:
        pass


class InputStimulus(StimulusInterface):
    """
    Stimulus interface for keyboard/mouse input.
    
    This is a placeholder - actual input delivery is handled by
    InputStimulusController which uses pynput.
    """
    
    def __init__(self):
        self._active = False
        self._current_params = None
    
    def activate_stimulus(self, params: Dict[str, Any]) -> None:
        """Store activation params (controller handles actual input)."""
        self._active = True
        self._current_params = params
        logger.debug(f"InputStimulus: Activated with params {params}")
    
    def deactivate_stimulus(self) -> None:
        """Deactivate stimulus."""
        self._active = False
        self._current_params = None
        logger.debug("InputStimulus: Deactivated")
    
    def configure_stimulus(self, config: Dict[str, Any]) -> None:
        """Store configuration."""
        logger.debug(f"InputStimulus: Configured with {config}")
    
    def is_stimulus_active(self) -> bool:
        """Check if stimulus is active."""
        return self._active


class ScreenshotBackend(BaseHardwareBackend):
    """
    Screenshot hardware backend for RGB screen capture.
    
    Captures screen regions as RGB data and provides input stimulus capability.
    """
    
    def __init__(self, config: HardwareConfig):
        """Initialize screenshot backend."""
        super().__init__(config)
        self.screenshot_source = None
    
    def initialize(self, **kwargs) -> None:
        """
        Initialize screenshot backend.
        
        Kwargs:
            x: Capture region left edge (default: 0)
            y: Capture region top edge (default: 0)
            width: Capture width (default: 800)
            height: Capture height (default: 600)
            monitor: Monitor number (default: 1)
            backend: Capture backend ("mss" or "pil", default: "mss")
        """
        logger.info("Initializing screenshot hardware backend...")
        
        # Extract capture parameters
        screenshot_args = self.config.screenshot
        x = screenshot_args.get('x', 0)
        y = screenshot_args.get('y', 0)
        width = screenshot_args.get('width', 800)
        height = screenshot_args.get('height', 600)
        monitor = screenshot_args.get('monitor', 1)
        backend = screenshot_args.get('backend', 'mss')
        
        # Create screenshot source
        self.screenshot_source = ScreenshotSource(
            x=x, y=y, width=width, height=height,
            monitor=monitor, backend=backend
        )
        
        # Create camera adapter
        self._camera = ScreenshotCamera(self.screenshot_source)

        # Set ROI
        # self._camera.set_roi(x, y, width, height)
        
        # Create dummy stage
        self._stage = DummyStage()
        
        # Create input stimulus interface
        self._stimulus = InputStimulus()
        
        self._initialized = True
        logger.info("Screenshot hardware backend initialized")
    
    def close(self) -> None:
        """Close screenshot backend."""
        logger.info("Closing screenshot hardware backend...")
        if self.screenshot_source:
            self.screenshot_source.close()
        self._initialized = False
        logger.info("Screenshot hardware backend closed")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get hardware metadata."""
        if not self._initialized:
            return {}
        
        metadata = {
            'backend': 'screenshot',
            'capture_region': self.screenshot_source.get_capture_region(),
            'image_size': self._camera.get_image_size(),
            'color_space': 'RGB',
            'bit_depth': 8,
        }
        return metadata
