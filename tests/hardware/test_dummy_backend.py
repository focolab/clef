"""
Unit tests for DummyHardwareBackend.

Tests the dummy backend implementation used for testing without real hardware.
"""

import pytest
import os
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

# Import the engine and supporting modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from hardware.backends.dummy_backend import (
    DummyHardwareBackend,
    DummyCamera,
    DummyStage,
    DummyStimulus,
)
from config.config_manager import HardwareConfig


@pytest.fixture
def dummy_config():
    """Create dummy hardware config."""
    return HardwareConfig(
        backend="dummy",
        stim_interface="dummy"
    )


class TestDummyCamera:
    """Test DummyCamera implementation."""
    
    def test_dummy_camera_initialization_default(self):
        """Test DummyCamera initializes with default parameters."""
        camera = DummyCamera()
        assert camera.width == 200
        assert camera.height == 200
        assert camera.exposure_ms == 5.0
        assert camera.roi == (0, 0, 200, 200)
    
    def test_dummy_camera_initialization_custom(self):
        """Test DummyCamera initializes with custom parameters."""
        camera = DummyCamera(width=100, height=150)
        assert camera.width == 100
        assert camera.height == 150
        assert camera.roi == (0, 0, 100, 150)
    
    def test_dummy_camera_acquire_frame(self):
        """Test acquire_frame returns numpy array."""
        camera = DummyCamera()
        frame = camera.acquire_frame()
        
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint16
        assert frame.shape == (200, 200)
    
    def test_dummy_camera_start_stop_acquisition(self):
        """Test start and stop acquisition."""
        camera = DummyCamera()
        assert camera._acquisition_running is False
        
        camera.start_acquisition()
        assert camera._acquisition_running is True
        
        camera.stop_acquisition()
        assert camera._acquisition_running is False
    
    def test_dummy_camera_get_remaining_image_count(self):
        """Test get_remaining_image_count."""
        camera = DummyCamera()
        camera.start_acquisition()
        
        count = camera.get_remaining_image_count()
        assert isinstance(count, int)
        assert count > 0
    
    def test_dummy_camera_pop_next_image(self):
        """Test pop_next_image returns image."""
        camera = DummyCamera()
        camera.start_acquisition()
        
        img = camera.pop_next_image()
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint16
        assert img.shape == (200, 200)
    
    def test_dummy_camera_snap_and_get_image(self):
        """Test snap_image and get_image."""
        camera = DummyCamera()
        camera.snap_image()
        
        img = camera.get_image()
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint16
        assert img.shape == (200, 200)
    
    def test_dummy_camera_clear_buffer(self):
        """Test clear_buffer resets counters."""
        camera = DummyCamera()
        camera.start_acquisition()
        camera._frame_count = 10
        
        camera.clear_buffer()
        assert camera._frame_count == 0
    
    def test_dummy_camera_set_get_exposure(self):
        """Test set and get exposure."""
        camera = DummyCamera()
        camera.set_exposure(50.0)
        assert camera.get_exposure() == 50.0
    
    def test_dummy_camera_set_get_roi(self):
        """Test set and get ROI."""
        camera = DummyCamera()
        camera.set_roi(10, 20, 100, 150)
        
        roi = camera.get_roi()
        assert roi == (10, 20, 100, 150)
        assert camera.width == 100
        assert camera.height == 150
    
    def test_dummy_camera_get_image_size(self):
        """Test get_image_size."""
        camera = DummyCamera(width=100, height=150)
        size = camera.get_image_size()
        assert size == (100, 150)
    
    def test_dummy_camera_configure_camera(self):
        """Test configure_camera method."""
        camera = DummyCamera()
        config = {"exposure": 30.0, "binning": "2x2"}
        # Should not raise
        camera.configure_camera(config)
    
    def test_dummy_camera_with_input_file(self):
        """Test DummyCamera loads data from input file."""
        pytest.importorskip("tifffile")

        # Create a temporary TIFF file
        with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as f:
            temp_path = f.name

        try:
            # Create test data
            test_data = np.random.randint(0, 65536, size=(10, 200, 200), dtype=np.uint16)
            import tifffile as tf
            tf.imwrite(temp_path, test_data)
            
            camera = DummyCamera(input_file=temp_path)
            assert camera.input_data is not None
            assert camera.input_data.shape == (10, 200, 200)
            
            # Get image should return data from file
            img = camera.get_image()
            assert isinstance(img, np.ndarray)
            assert img.dtype == np.uint16
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDummyStage:
    """Test DummyStage implementation."""
    
    def test_dummy_stage_initialization(self):
        """Test DummyStage initializes."""
        stage = DummyStage()
        assert stage.x_position == 0.0
        assert stage.y_position == 0.0
        assert stage.z_position == 0.0
    
    def test_dummy_stage_get_position_all_axes(self):
        """Test get_position returns all axes."""
        stage = DummyStage()
        pos = stage.get_position()
        assert isinstance(pos, tuple)
        assert len(pos) == 3
        assert pos == (0.0, 0.0, 0.0)
    
    def test_dummy_stage_get_position_single_axis(self):
        """Test get_position returns single axis."""
        stage = DummyStage()
        stage.z_position = 10.5
        
        z_pos = stage.get_position('Z')
        assert z_pos == 10.5
        
        x_pos = stage.get_position('X')
        assert x_pos == 0.0
    
    def test_dummy_stage_move_to_position_single(self):
        """Test move_to_position with single value."""
        stage = DummyStage()
        stage.move_to_position(10.0, axis='Z')
        assert stage.z_position == 10.0
    
    def test_dummy_stage_move_to_position_tuple(self):
        """Test move_to_position with tuple."""
        stage = DummyStage()
        stage.move_to_position((5.0, 10.0, 15.0))
        assert stage.x_position == 5.0
        assert stage.y_position == 10.0
        assert stage.z_position == 15.0
    
    def test_dummy_stage_run_z_stack(self):
        """Test run_z_stack configures sequence."""
        stage = DummyStage()
        stage.run_z_stack(z_start=0.0, z_end=30.0, z_step=3.0, num_planes=11)
        assert stage._sequence_running is True
    
    def test_dummy_stage_stop_sequence(self):
        """Test stop_sequence."""
        stage = DummyStage()
        stage._sequence_running = True
        stage.stop_sequence()
        assert stage._sequence_running is False
    
    def test_dummy_stage_wait_for_device(self):
        """Test wait_for_device (should not raise)."""
        stage = DummyStage()
        stage.wait_for_device()  # Should not raise
    
    def test_dummy_stage_get_device_name(self):
        """Test get_focus_device_name."""
        stage = DummyStage()
        name = stage.get_device_name()
        assert isinstance(name, str)


class TestDummyStimulus:
    """Test DummyStimulus implementation."""
    
    def test_dummy_stimulus_initialization(self):
        """Test DummyStimulus initializes."""
        stimulus = DummyStimulus()
        assert stimulus._active is False
        assert stimulus._current_params is None
    
    def test_dummy_stimulus_activate(self):
        """Test activate_stimulus."""
        stimulus = DummyStimulus()
        params = {"intensity": 50, "position": (100, 100)}
        
        stimulus.activate_stimulus(params)
        assert stimulus._active is True
        assert stimulus._current_params == params
    
    def test_dummy_stimulus_deactivate(self):
        """Test deactivate_stimulus."""
        stimulus = DummyStimulus()
        stimulus._active = True
        stimulus._current_params = {"intensity": 50}
        
        stimulus.deactivate_stimulus()
        assert stimulus._active is False
        assert stimulus._current_params is None
    
    def test_dummy_stimulus_configure(self):
        """Test configure_stimulus."""
        stimulus = DummyStimulus()
        config = {"roi": (0, 0, 100, 100)}
        # Should not raise
        stimulus.configure_stimulus(config)
    
    def test_dummy_stimulus_is_active(self):
        """Test is_stimulus_active."""
        stimulus = DummyStimulus()
        assert stimulus.is_stimulus_active() is False
        
        stimulus.activate_stimulus({"intensity": 50})
        assert stimulus.is_stimulus_active() is True


class TestDummyHardwareBackend:
    """Test DummyHardwareBackend class."""
    
    def test_dummy_backend_initialization(self, dummy_config):
        """Test DummyHardwareBackend initializes."""
        backend = DummyHardwareBackend(dummy_config)
        assert backend.config == dummy_config
        assert backend.is_initialized is False
    
    def test_dummy_backend_initialize(self, dummy_config):
        """Test DummyHardwareBackend.initialize()."""
        backend = DummyHardwareBackend(dummy_config)
        backend.initialize()
        
        assert backend.is_initialized is True
        assert backend._camera is not None
        assert backend._stage is not None
        assert backend._stimulus is not None
    
    def test_dummy_backend_initialize_with_input_file(self, dummy_config):
        """Test initialize with input file."""
        backend = DummyHardwareBackend(dummy_config)
        backend.initialize(input_file="test.tiff")
        assert backend.is_initialized is True
    
    def test_dummy_backend_camera_property(self, dummy_config):
        """Test camera property access."""
        backend = DummyHardwareBackend(dummy_config)
        backend.initialize()
        
        camera = backend.camera
        assert isinstance(camera, DummyCamera)
    
    def test_dummy_backend_camera_before_init(self, dummy_config):
        """Test camera property raises before initialization."""
        backend = DummyHardwareBackend(dummy_config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = backend.camera
    
    def test_dummy_backend_stage_property(self, dummy_config):
        """Test stage property access."""
        backend = DummyHardwareBackend(dummy_config)
        backend.initialize()
        
        stage = backend.stage
        assert isinstance(stage, DummyStage)
    
    def test_dummy_backend_stimulus_property(self, dummy_config):
        """Test stimulus property access."""
        backend = DummyHardwareBackend(dummy_config)
        backend.initialize()
        
        stimulus = backend.stimulus
        assert isinstance(stimulus, DummyStimulus)
    
    def test_dummy_backend_close(self, dummy_config):
        """Test close() method."""
        backend = DummyHardwareBackend(dummy_config)
        backend.initialize()
        assert backend.is_initialized is True
        
        backend.close()
        assert backend.is_initialized is False
    
    def test_dummy_backend_get_metadata(self, dummy_config):
        """Test get_metadata() method."""
        backend = DummyHardwareBackend(dummy_config)
        backend.initialize()
        
        metadata = backend.get_metadata()
        assert isinstance(metadata, dict)
        assert metadata['backend'] == 'dummy'
        assert 'exposure' in metadata
        assert 'roi' in metadata
        assert 'image_size' in metadata
        assert 'stage_position' in metadata
    
    def test_dummy_backend_get_metadata_before_init(self, dummy_config):
        """Test get_metadata returns empty dict before initialization."""
        backend = DummyHardwareBackend(dummy_config)
        metadata = backend.get_metadata()
        assert metadata == {}


class TestDummyBackendIntegration:
    """Integration tests for dummy backend through HardwareManager."""
    
    def test_dummy_backend_camera_operations(self, dummy_config):
        """Test dummy backend camera interface through HardwareManager."""
        from hardware.hardware_manager import HardwareManager
        
        manager = HardwareManager(dummy_config)
        manager.initialize()
        
        manager.camera.start_acquisition()
        frame = manager.camera.acquire_frame()
        
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint16
        assert frame.shape == (200, 200)  # Default ROI
        
        manager.camera.stop_acquisition()
    
    def test_dummy_backend_stage_operations(self, dummy_config):
        """Test dummy backend stage interface."""
        from hardware.hardware_manager import HardwareManager
        
        manager = HardwareManager(dummy_config)
        manager.initialize()
        
        # Move stage
        manager.stage.move_to_position(10.0, axis='Z')
        pos = manager.stage.get_position('Z')
        assert pos == 10.0
    
    def test_dummy_backend_stimulus_operations(self, dummy_config):
        """Test dummy backend stimulus interface."""
        from hardware.hardware_manager import HardwareManager
        
        manager = HardwareManager(dummy_config)
        manager.initialize()
        
        # Activate stimulus
        manager.stimulus.activate_stimulus({"intensity": 50})
        assert manager.stimulus.is_stimulus_active() is True
        
        # Deactivate
        manager.stimulus.deactivate_stimulus()
        assert manager.stimulus.is_stimulus_active() is False

