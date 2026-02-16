"""
Physical hardware tests for MicroManagerBackend.

These tests validate that the backend can successfully:
1. Initialize physical hardware (camera, stage, stimulus)
2. Configure devices according to HardwareConfig
3. Acquire images from real camera
4. Control stage movements
5. Activate stimulus hardware

Run with: pytest tests/hardware_physical/test_micromanager_backend.py -v --hardware
"""

import pytest
import numpy as np
import logging

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.hardware

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestMicroManagerBackendInitialization:
    """Test backend initialization with physical hardware."""
    
    def test_hardware_manager_initializes(self, hardware_manager_connected):
        """Test that HardwareManager successfully initializes hardware."""
        assert hardware_manager_connected.is_initialized
        logger.info("✓ HardwareManager initialized")
    
    def test_camera_interface_accessible(self, camera_connected):
        """Test that camera interface is accessible."""
        assert camera_connected is not None
        logger.info("✓ Camera interface accessible")
    
    def test_stage_interface_accessible(self, stage_connected):
        """Test that stage interface is accessible."""
        assert stage_connected is not None
        logger.info("✓ Stage interface accessible")
    
    def test_stimulus_interface_accessible(self, stimulus_connected):
        """Test that stimulus interface is accessible."""
        assert stimulus_connected is not None
        logger.info("✓ Stimulus interface accessible")
    
    def test_data_interface_accessible(self, data_interface_connected):
        """Test that data interface is accessible."""
        assert data_interface_connected is not None
        logger.info("✓ Data interface accessible")


class TestCameraOperations:
    """Test camera operations with physical hardware."""
    
    def test_get_exposure(self, camera_connected):
        """Test reading camera exposure time."""
        exposure = camera_connected.get_exposure()
        # assert isinstance(exposure, float)
        assert exposure > 0
        logger.info(f"✓ Camera exposure: {exposure} ms")
    
    def test_set_exposure(self, camera_connected):
        """Test setting camera exposure time."""
        original_exposure = camera_connected.get_exposure()
        
        # Set new exposure
        new_exposure = 20.0
        camera_connected.set_exposure(new_exposure)
        
        # Verify
        actual_exposure = camera_connected.get_exposure()
        assert abs(actual_exposure - new_exposure) < 1.0
        logger.info(f"✓ Set exposure to {actual_exposure} ms")
        
        # Restore
        camera_connected.set_exposure(original_exposure)
    
    def test_get_roi(self, camera_connected):
        """Test reading camera ROI."""
        roi = camera_connected.get_roi()
        assert isinstance(roi, tuple)
        assert len(roi) == 4
        x, y, width, height = roi
        assert all(isinstance(v, int) for v in roi)
        assert width > 0 and height > 0
        logger.info(f"✓ Camera ROI: {roi}")
    
    def test_get_image_size(self, camera_connected):
        """Test reading camera image dimensions."""
        width, height = camera_connected.get_image_size()
        assert isinstance(width, int) and isinstance(height, int)
        assert width > 0 and height > 0
        logger.info(f"✓ Image size: {width}x{height}")
    
    def test_snap_single_frame(self, camera_connected):
        """Test acquiring a single frame from camera."""
        logger.info("Snapping single frame...")
        frame = camera_connected.acquire_frame()
        
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint16
        assert frame.ndim == 2
        assert frame.shape[0] > 0 and frame.shape[1] > 0
        
        logger.info(f"✓ Acquired frame: shape={frame.shape}, dtype={frame.dtype}")
        logger.info(f"  Frame stats: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
    
    def test_continuous_acquisition(self, camera_connected):
        """Test continuous acquisition mode."""
        logger.info("Testing continuous acquisition...")
        
        # Start acquisition
        camera_connected.start_acquisition(buffer_size=100)
        
        # Wait for a few frames to accumulate
        import time
        time.sleep(0.5)
        
        # Check buffer has frames
        remaining = camera_connected.get_remaining_image_count()
        assert remaining > 0
        logger.info(f"✓ Buffer has {remaining} frames")
        
        # Pop a frame
        frame = camera_connected.pop_next_image()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint16
        logger.info(f"✓ Popped frame from buffer: shape={frame.shape}")
        
        # Stop acquisition
        camera_connected.stop_acquisition()
        camera_connected.clear_buffer()
        logger.info("✓ Continuous acquisition test complete")


class TestStageOperations:
    """Test stage operations with physical hardware."""
    
    def test_get_position(self, stage_connected):
        """Test reading stage position."""
        position = stage_connected.get_position()
        assert isinstance(position, (float, tuple))
        logger.info(f"✓ Stage position: {position}")
    
    def test_move_stage_small_distance(self, stage_connected):
        """Test small stage movement."""
        logger.info("Testing small stage movement...")
        
        # Get initial position
        initial_pos = stage_connected.get_position()
        if isinstance(initial_pos, tuple):
            initial_pos = initial_pos[0]
        
        # Move small distance
        move_distance = 5.0  # 5 microns
        target_pos = initial_pos + move_distance
        
        stage_connected.move_to_position(target_pos)
        stage_connected.wait_for_device()
        
        # Verify new position
        new_pos = stage_connected.get_position()
        if isinstance(new_pos, tuple):
            new_pos = new_pos[0]
        
        # Allow 1 micron tolerance
        assert abs(new_pos - target_pos) < 1.0
        logger.info(f"✓ Moved stage: {initial_pos:.2f} → {new_pos:.2f} µm")
        
        # Return to initial position
        stage_connected.move_to_position(initial_pos)
        stage_connected.wait_for_device()
        logger.info("✓ Returned to initial position")
    
    # @pytest.mark.slow
    def test_configure_zstack(self, stage_connected):
        """Test configuring Z-stack sequence."""
        logger.info("Testing Z-stack configuration...")
        
        config = {
            "z_start": -10.5,
            "z_end": 10.5,
            "z_step": 3.0,
            "pad_z": 0
        }
        
        stage_connected.configure_stage(config)
        logger.info("✓ Z-stack configured successfully")


class TestStimulusOperations:
    """Test stimulus operations with physical hardware."""
    
    def test_configure_stimulus(self, stimulus_connected, innovation_core_config):
        """Test configuring stimulus hardware."""
        logger.info("Testing stimulus configuration...")
        
        config = {
            "interface_type": innovation_core_config.stim_interface,
            "calibration_path": innovation_core_config.get_stimulus_device_config().polygon_calibration_path
        }

        stimulus_connected.configure_stimulus(config)
        logger.info("✓ Stimulus configured")

    def test_activate_deactivate_stimulus(self, stimulus_connected, innovation_core_config):
        """Test activating and deactivating stimulus."""
        logger.info("Testing stimulus activation...")

        # Configure first
        config = {
            "interface_type": innovation_core_config.stim_interface,
            "calibration_path": innovation_core_config.get_stimulus_device_config().polygon_calibration_path
        }
        stimulus_connected.configure_stimulus(config)
        
        # Activate with low intensity
        params = {"intensity": 5}  # 5% intensity for safety
        stimulus_connected.activate_stimulus(params)
        
        assert stimulus_connected.is_stimulus_active()
        logger.info("✓ Stimulus activated")
        
        # Brief delay
        import time
        time.sleep(0.1)
        
        # Deactivate
        stimulus_connected.deactivate_stimulus()
        assert not stimulus_connected.is_stimulus_active()
        logger.info("✓ Stimulus deactivated")
    
    def test_polygon_mask_update(self, stimulus_connected, innovation_core_config):
        """Test updating polygon mask."""
        logger.info("Testing polygon mask update...")
        
        # Configure
        config = {
            "interface_type": innovation_core_config.stim_interface,
            "calibration_path": innovation_core_config.get_stimulus_device_config().polygon_calibration_path
        }
        stimulus_connected.configure_stimulus(config)

        # Create stimulus params with ROI event
        stim_params = {
            "event": {
                "event_type": "pulse-rect-roi-list",
                "stim_rect_roi_list": {
                    "x": [600, 700],
                    "y": [400, 500],
                    "width": [50, 50],
                    "height": [50, 50]
                }
            },
            "roi": [0, 0]
        }
        
        stimulus_connected.update_polygon_mask(stim_params)
        logger.info("✓ Polygon mask updated")


class TestDataInterface:
    """Test data interface operations."""
    
    def test_get_sample_shape(self, data_interface_connected):
        """Test getting sample dimensions."""
        shape = data_interface_connected.get_sample_shape()
        assert isinstance(shape, tuple)
        assert len(shape) == 2
        assert all(s > 0 for s in shape)
        logger.info(f"✓ Sample shape: {shape}")
    
    def test_get_sample_dtype(self, data_interface_connected):
        """Test getting sample data type."""
        dtype = data_interface_connected.get_sample_dtype()
        assert dtype == np.uint16
        logger.info(f"✓ Sample dtype: {dtype}")
    
    def test_sample_single_frame(self, data_interface_connected, camera_connected):
        """Test sampling a single frame via data interface."""
        logger.info("Testing single frame sampling...")
        
        # Start acquisition
        camera_connected.start_acquisition(buffer_size=10)
        
        import time
        time.sleep(0.2)
        
        # Sample frame
        frame = data_interface_connected.sample_data()
        
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint16
        assert frame.ndim == 2
        logger.info(f"✓ Sampled frame: shape={frame.shape}")
        
        # Cleanup
        camera_connected.stop_acquisition()
        camera_connected.clear_buffer()


class TestIntegratedAcquisition:
    """Test integrated acquisition with all hardware components."""
    
    # @pytest.mark.slow
    def test_acquire_small_zstack(self, hardware_manager_connected):
        """Test acquiring a small Z-stack with all components."""
        logger.info("Testing integrated Z-stack acquisition...")
        
        camera = hardware_manager_connected.camera
        stage = hardware_manager_connected.stage
        data = hardware_manager_connected.data
        
        # Configure Z-stack
        z_planes = 3
        z_step = 3.0
        
        stage_config = {
            "z_start": -z_step,
            "z_end": z_step,
            "z_step": z_step,
            "pad_z": 0
        }
        stage.configure_stage(stage_config)
        
        # Start acquisition
        camera.start_acquisition(buffer_size=z_planes * 2)
        
        # Acquire frames
        frames = []
        for i in range(z_planes):
            import time
            time.sleep(0.1)
            
            frame = data.sample_data()
            frames.append(frame)
            logger.info(f"  Acquired frame {i+1}/{z_planes}")
        
        # Stop acquisition
        camera.stop_acquisition()
        camera.clear_buffer()
        
        # Verify
        assert len(frames) == z_planes
        for i, frame in enumerate(frames):
            assert isinstance(frame, np.ndarray)
            assert frame.dtype == np.uint16
            logger.info(f"  Frame {i}: shape={frame.shape}, mean={frame.mean():.1f}")
        
        logger.info(f"✓ Successfully acquired {z_planes}-plane Z-stack")
