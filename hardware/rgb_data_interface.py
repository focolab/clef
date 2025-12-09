"""
RGB Data Interface - Concrete implementation for RGB screen capture data.

Handles 3-channel uint8 RGB data from screenshot backends.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional

from hardware.data_interface import DataInterface
from config.config_manager import ExperimentConfig

logger = logging.getLogger(__name__)


class RGBDataInterface(DataInterface):
    """
    Data interface for RGB screen capture data.
    
    Handles 3-channel uint8 RGB images from screenshot sources.
    Provides generic DataInterface abstraction for RGB data.
    """
    
    def __init__(self, screenshot_source):
        """
        Initialize RGB data interface.
        
        Args:
            screenshot_source: Backend source providing RGB capture capability
        """
        self.source = screenshot_source
        self.metadata_cache: Dict[str, Any] = {}
        logger.debug("RGBDataInterface initialized")
        
        # Data buffer
        self.samples = None
        self.sample_time_list: list[float] = []
        self.sample_dtype = self.get_sample_dtype()
        
        # Get initial dimensions from source
        self.height, self.width = self.source.get_capture_dimensions()
        self.storage_shape = (self.height, self.width, 3)
    
    def sample_data(self) -> np.ndarray:
        """
        Capture RGB screen data.
        
        Returns:
            RGB image as numpy array (uint8), shape (height, width, 3)
        """
        import time
        img = self.source.capture_screen()
        self.sample_time_list.append(time.time())
        return img
    
    def start_sampling(self, buffer_size: int = 0) -> None:
        """
        Start continuous capture (if supported).
        
        Args:
            buffer_size: Unused for screenshot capture
        """
        logger.debug("RGBDataInterface: Continuous capture started")
    
    def stop_sampling(self) -> None:
        """Stop continuous capture."""
        logger.debug("RGBDataInterface: Continuous capture stopped")
    
    def get_sample_shape(self) -> Tuple[int, ...]:
        """
        Get RGB image dimensions.
        
        Returns:
            Tuple of (height, width, 3)
        """
        return self.storage_shape
    
    def get_sample_dtype(self) -> np.dtype:
        """
        Get RGB data type.
        
        Returns:
            NumPy dtype (uint8 for RGB data)
        """
        return np.dtype(np.uint8)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get capture metadata.
        
        Returns:
            Dictionary containing capture settings
        """
        metadata = {
            'data_type': 'rgb_screenshot',
            'color_space': 'RGB',
            'bit_depth': 8,
            'channels': 3,
            'dtype': str(self.get_sample_dtype()),
            'shape': self.get_sample_shape(),
            'capture_region': self.source.get_capture_region(),
        }
        self.metadata_cache = metadata
        return metadata
    
    def save_data(
        self, 
        data: np.ndarray, 
        filepath: Union[str, Path],
        **kwargs
    ) -> None:
        """
        Save RGB data as image sequence or video.
        
        Args:
            data: RGB array to save (supports 3D or 4D stacks)
            filepath: Output file path
            **kwargs: Additional save options
        """
        filepath = Path(filepath)
        
        # For 4D data (T, H, W, C), save as image sequence
        if data.ndim == 4:
            self._save_as_sequence(data, filepath)
        # For 3D data (H, W, C), save single image
        elif data.ndim == 3:
            self._save_single_image(data, filepath)
        else:
            raise ValueError(f"Unexpected data dimensions: {data.shape}")
        
        logger.info(f"Saved RGB data to {filepath} (shape={data.shape}, dtype={data.dtype})")
    
    def _save_single_image(self, data: np.ndarray, filepath: Path) -> None:
        """Save single RGB image."""
        try:
            from PIL import Image
            img = Image.fromarray(data, mode='RGB')
            img.save(filepath)
        except ImportError:
            logger.error("PIL not available for image saving")
            # Fallback to numpy
            np.save(filepath.with_suffix('.npy'), data)
    
    def _save_as_sequence(self, data: np.ndarray, filepath: Path) -> None:
        """Save RGB sequence as numbered images."""
        try:
            from PIL import Image
            
            # Create output directory
            output_dir = filepath.parent / filepath.stem
            output_dir.mkdir(exist_ok=True)
            
            # Save each frame
            num_frames = data.shape[0]
            for i in range(num_frames):
                frame_path = output_dir / f"frame_{i:06d}.png"
                img = Image.fromarray(data[i], mode='RGB')
                img.save(frame_path)
            
            logger.info(f"Saved {num_frames} frames to {output_dir}")
            
        except ImportError:
            logger.error("PIL not available for image saving")
            # Fallback to numpy
            np.save(filepath.with_suffix('.npy'), data)
    
    def configure_sampling(self, config: ExperimentConfig | None) -> None:
        """
        Configure capture settings.
        
        Args:
            config: Experiment configuration with acquisition parameters
        """
        if config is None:
            return
        
        # Initialize sample buffer
        sample_shape = tuple([config.acquisition.num_samples]) + self.get_sample_shape()
        logger.info(f'Initializing RGB sample buffer of shape {sample_shape}')
        self.samples = np.zeros(sample_shape, dtype=self.get_sample_dtype())
        
        logger.debug(f"Configured RGB sampling with {config}")
    
    def get_remaining_sample_count(self) -> int:
        """
        Get number of samples in buffer.
        
        Returns:
            Always returns 1 for screenshot capture (no buffering)
        """
        return 1
    
    def clear_buffer(self) -> None:
        """Clear buffer (no-op for screenshots)."""
        pass
    
    @property
    def data_type_name(self) -> str:
        """Get data type name."""
        return "rgb_screenshot"


def create_rgb_data_interface(screenshot_source) -> RGBDataInterface:
    """
    Factory function to create an RGBDataInterface.
    
    Args:
        screenshot_source: Screenshot capture source
        
    Returns:
        RGBDataInterface wrapping the source
    """
    return RGBDataInterface(screenshot_source)
