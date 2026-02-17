"""
Image Data Interface - Concrete implementation wrapping camera interface.

Maintains backward compatibility with uint16 camera data while providing
the generic DataInterface abstraction.
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional

from hardware.data_interface import DataInterface
from hardware.camera_interface import CameraInterface
import tifffile as tf
from config.config_manager import ExperimentConfig

logger = logging.getLogger(__name__)


class ImageDataInterface(DataInterface):
    """
    Data interface wrapper for camera-based microscopy images.
    
    This class adapts the existing CameraInterface to the generic DataInterface,
    maintaining full backward compatibility while enabling future data type diversity.
    
    Attributes:
        camera: Underlying CameraInterface instance
        metadata_cache: Cached metadata to avoid repeated queries
    """
    
    def __init__(self, camera: CameraInterface):
        """
        Initialize image data interface.
        
        Args:
            camera: CameraInterface instance to wrap
        """
        self.camera = camera
        self.metadata_cache: Dict[str, Any] = {}
        logger.debug("ImageDataInterface initialized wrapping camera interface")

        # Continuous vs discrete data streams
        self.strobe_acquisition = False # TODO untested
        self.next_call: float = 0. # Next discrete sample
        self.false_grab_count: int = 0 # Errors thrown while trying to get images from buffer

        # Pointer to sample index
        self.sample_ndx = 0
        
        # Data buffer for temporary storage
        self.storage_shape = self.get_sample_shape()
        self.sample_dtype = self.get_sample_dtype()
        self.samples = np.zeros(self.get_sample_shape(), dtype=self.get_sample_dtype()) # initialization value... sample shape is for single sample not buffer of samples until configure_sampling
        self.sample_time_list: list[float] = []
        self.sample_grab_t0: float = 0.

        # store some useful local variables for testing
        self.xsize = self.storage_shape[1]
        self.ysize = self.storage_shape[0]
    
    def sample_data(self) -> np.ndarray:
        """
        Sample image data from camera.
        
        Maps to camera's pop_next_image() for buffered acquisition or
        acquire_frame() for single-shot.
        
        Returns:
            Image as numpy array (typically uint16), shape (height, width)
        """

        # Handle strobe timing TODO
        if self.strobe_acquisition:
            nowtime = time.time()
            self.next_call = self.next_call + self.strobe_inter_frame_interval / 1000
            
            if self.next_call - nowtime < 0:
                logger.warning(
                    f"Strobe delay exceeded interval! Frame: {len(self.sample_time_list)}"
                )
            else:
                time.sleep(self.next_call - nowtime)
                
            self.camera.snap_image()
        else:

            # Blocking poll for data if continuous acquisition
            rem = self.camera.get_remaining_image_count()
            while rem == 0:
                rem = self.camera.get_remaining_image_count()

        # got an image
        if rem > 0 or self.strobe_acquisition:

            # Grab image from buffer
            try:
                if self.strobe_acquisition:
                    img = self.camera.get_image()
                else:
                    img = self.camera.pop_next_image()
            except Exception as err:
                false_grab_count += 1
                logger.debug(f"False grab #{false_grab_count}: {err}")

        # append sample to sample vec (only if buffer is configured)
        if self.sample_ndx < self.samples.shape[0] and len(self.samples.shape) == 3:
            self.store_sample(img)
            
        # Fall back to single frame acquisition
        self.sample_time_list.append(time.time())
        return img
    
    def store_sample(self, frame):

        # Note at initialization sample shape is xy, after configure_sampling it becomes tyx
        self.samples[self.sample_ndx,:] = frame
        self.sample_ndx += 1
    
    def start_sampling(self, buffer_size: int = 0) -> None:
        """
        Start continuous image acquisition.
        
        Args:
            buffer_size: Circular buffer size (0 = unlimited)
        """
        # self.camera.start_acquisition(buffer_size=buffer_size)
        self.camera.start_acquisition()
        logger.debug(f"Started continuous image sampling (buffer_size={buffer_size})")
    
    def stop_sampling(self) -> None:
        """Stop continuous image acquisition."""
        self.camera.stop_acquisition()
        logger.debug("Stopped continuous image sampling")
    
    def get_sample_shape(self) -> Tuple[int, ...]:
        """
        Get image dimensions.
        
        Returns:
            Tuple of (height, width)
        """
        width, height = self.camera.get_image_size()
        return (height, width)
    
    def get_sample_dtype(self) -> np.dtype:
        """
        Get image data type.
        
        Returns:
            NumPy dtype (typically np.uint16 for microscopy cameras)
        """
        return np.dtype(np.uint16)  # Standard for scientific cameras
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get camera metadata.
        
        Returns:
            Dictionary containing camera settings (exposure, ROI, etc.)
        """
        metadata = {
            'data_type': 'microscopy_image',
            'exposure_ms': self.camera.get_exposure(),
            'roi': self.camera.get_roi(),
            'image_size': self.camera.get_image_size(),
            'dtype': str(self.get_sample_dtype()),
            'shape': self.get_sample_shape(),
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
        Save image data as TIFF stack.
        
        Args:
            data: Image array to save (supports 2D or 3D stacks)
            filepath: Output file path (should end in .tiff or .tif)
            **kwargs: Additional save options:
                     - compression: TIFF compression type
                     - metadata: Additional metadata dict to embed
        """
        # add .tiff if necessary
        if not filepath.endswith('.tiff'):
            filepath = filepath + '.tiff'

        tf.imwrite(filepath, data)
        logger.info(f"Saved image data to {filepath} (shape={data.shape}, dtype={data.dtype})")
    
    def configure_sampling(self, config: ExperimentConfig | None) -> None:
        """
        Configure camera settings.
        
        Args:
            config: Dictionary with camera configuration:
                   - exposure: exposure time in ms
                   - roi: region of interest as (x, y, width, height)
                   - Other camera-specific settings
        """
        # initiate internal buffer
        sample_shape = tuple([config.acquisition.num_samples]) + self.get_sample_shape()
        logging.info(f'Initializing sample buffer of shape {sample_shape}')
        self.samples = np.zeros(sample_shape, dtype=self.get_sample_dtype())
        
        # Update specific settings if provided
        if 'exposure' in config:
            self.camera.set_exposure(config['exposure'])
        if 'roi' in config:
            x, y, width, height = config['roi']
            self.camera.set_roi(x, y, width, height)
        
        logger.debug(f"Configured image sampling with {config}")
    
    def get_remaining_sample_count(self) -> int:
        """
        Get number of images remaining in camera buffer.
        
        Returns:
            Number of buffered images available
        """
        return self.camera.get_remaining_image_count()
    
    def clear_buffer(self) -> None:
        """Clear camera circular buffer."""
        self.camera.clear_buffer()
        logger.debug("Cleared image buffer")
    
    # def get_sampling_rate(self) -> Optional[float]:
    #     """
    #     Calculate approximate sampling rate from exposure time.
        
    #     Returns:
    #         Approximate frame rate in Hz based on exposure time
    #     """
    #     exposure_ms = self.camera.get_exposure()
    #     if exposure_ms > 0:
    #         # This is approximate - actual rate may be lower due to readout time
    #         return 1000.0 / exposure_ms
    #     return None
    
    # def set_sampling_rate(self, rate_hz: float) -> None:
    #     """
    #     Set frame rate by adjusting exposure time.
        
    #     Note: This sets an upper bound. Actual rate may be limited by
    #     camera readout time and other factors.
        
    #     Args:
    #         rate_hz: Target frame rate in Hz
    #     """
    #     if rate_hz <= 0:
    #         raise ValueError(f"Sampling rate must be positive, got {rate_hz}")
        
    #     exposure_ms = 1000.0 / rate_hz
    #     self.camera.set_exposure(exposure_ms)
    #     logger.debug(f"Set target sampling rate to {rate_hz} Hz (exposure={exposure_ms} ms)")
    
    @property
    def data_type_name(self) -> str:
        """Get data type name."""
        return "microscopy_image"
    
    # Legacy camera access for backward compatibility
    # This allows existing code to access camera methods directly
    
    # def get_camera(self) -> CameraInterface:
    #     """
    #     Get underlying camera interface.
        
    #     This is provided for backward compatibility with code that
    #     directly accesses camera methods.
        
    #     Returns:
    #         CameraInterface instance
    #     """
    #     return self.camera
    
    # def snap_image(self) -> None:
    #     """
    #     Trigger single image snap (for strobe acquisition).
        
    #     Delegates to camera's snap_image() method.
    #     """
    #     self.camera.snap_image()
    
    # def get_image(self) -> np.ndarray:
    #     """
    #     Get most recently snapped image.
        
    #     Delegates to camera's get_image() method.
        
    #     Returns:
    #         Most recent image as numpy array
    #     """
    #     return self.camera.get_image()
    
    # def pop_next_image(self) -> np.ndarray:
    #     """
    #     Pop next image from buffer.
        
    #     Delegates to camera's pop_next_image() method.
        
    #     Returns:
    #         Next buffered image as numpy array
    #     """
    #     return self.camera.pop_next_image()


# Factory function for creating image data interfaces
def create_image_data_interface(camera: CameraInterface) -> ImageDataInterface:
    """
    Factory function to create an ImageDataInterface.
    
    Args:
        camera: CameraInterface instance to wrap
        
    Returns:
        ImageDataInterface wrapping the camera
    """
    return ImageDataInterface(camera)
