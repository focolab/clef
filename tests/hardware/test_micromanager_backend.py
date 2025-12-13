"""
Unit tests for MicroManagerBackend (mocked).

Tests the Micro-Manager backend implementation with mocked MMC objects
to avoid requiring real hardware.
"""

import pytest, os
import numpy as np
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from pathlib import Path

# Import the engine and supporting modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hardware.backends.micromanager_backend import (
    MicroManagerBackend,
    MicroManagerCamera,
    MicroManagerStage,
    MicroManagerStimulus,
)
from config.config_manager import HardwareConfig


@pytest.fixture
def mock_mmc():
    """Create a mock Micro-Manager Core object."""
    mmc = MagicMock()
    
    # Mock ROI object (for pycromanager - Java object)
    roi_mock = MagicMock()
    roi_mock.getX.return_value = 0
    roi_mock.getY.return_value = 0
    roi_mock.getWidth.return_value = 200
    roi_mock.getHeight.return_value = 200
    mmc.getROI.return_value = roi_mock
    
    # Mock other common methods
    mmc.getExposure.return_value = 10.0
    mmc.getPosition.return_value = 0.0
    mmc.getFocusDevice.return_value = "ZStage"
    mmc.getRemainingImageCount.return_value = 10
    mmc.getImage.return_value = np.random.randint(0, 65536, size=(200*200,), dtype=np.uint16)
    mmc.popNextImage.return_value = np.random.randint(0, 65536, size=(200*200,), dtype=np.uint16)
    
    return mmc


@pytest.fixture
def pycromanager_config():
    """Create pycromanager hardware config."""
    return HardwareConfig(
        backend="pycromanager",
        stim_interface="dummy",
        mm_config_path="test_config.cfg",
        microscope_name="test_scope"
    )


@pytest.fixture
def pymmcore_config():
    """Create pymmcore hardware config."""
    return HardwareConfig(
        backend="pymmcore",
        stim_interface="dummy",
        mm_config_path="test_config.cfg",
        microscope_name="test_scope"
    )


class TestMicroManagerCamera:
    """Test MicroManagerCamera implementation."""
    
    def test_camera_initialization_pycromanager(self, mock_mmc):
        """Test MicroManagerCamera initializes with pycromanager backend."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        
        assert camera.mmc == mock_mmc
        assert camera.backend == "pycromanager"
        assert camera._width == 200
        assert camera._height == 200
    
    def test_camera_initialization_pymmcore(self, mock_mmc):
        """Test MicroManagerCamera initializes with pymmcore backend."""
        # For pymmcore, getROI returns tuple directly
        mock_mmc.getROI.return_value = (0, 0, 200, 200)
        
        camera = MicroManagerCamera(mock_mmc, "pymmcore")
        assert camera.backend == "pymmcore"
        assert camera._width == 200
        assert camera._height == 200
    
    def test_camera_initialization_with_roi(self, mock_mmc):
        """Test MicroManagerCamera initializes with provided ROI."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager", roi=(10, 20, 100, 150))
        
        assert camera._roi == (10, 20, 100, 150)
        mock_mmc.setROI.assert_called_once_with(10, 20, 100, 150)
    
    def test_camera_acquire_frame(self, mock_mmc):
        """Test acquire_frame."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        frame = camera.acquire_frame()
        
        mock_mmc.snapImage.assert_called_once()
        mock_mmc.getImage.assert_called_once()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint16
    
    def test_camera_start_acquisition(self, mock_mmc):
        """Test start_acquisition."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        camera.start_acquisition(buffer_size=5000)
        
        mock_mmc.setCircularBufferMemoryFootprint.assert_called_once_with(5000)
        mock_mmc.startContinuousSequenceAcquisition.assert_called_once_with(0)
    
    def test_camera_stop_acquisition(self, mock_mmc):
        """Test stop_acquisition."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        camera.stop_acquisition()
        
        mock_mmc.stopSequenceAcquisition.assert_called_once()
    
    def test_camera_get_remaining_image_count(self, mock_mmc):
        """Test get_remaining_image_count."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        count = camera.get_remaining_image_count()
        
        mock_mmc.getRemainingImageCount.assert_called_once()
        assert count == 10
    
    def test_camera_pop_next_image_pycromanager(self, mock_mmc):
        """Test pop_next_image with pycromanager (reshapes)."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        img = camera.pop_next_image()
        
        mock_mmc.popNextImage.assert_called_once()
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint16
        assert img.shape == (200, 200)  # Reshaped
    
    def test_camera_pop_next_image_pymmcore(self, mock_mmc):
        """Test pop_next_image with pymmcore (no reshape)."""
        # pymmcore returns already-shaped array
        mock_mmc.popNextImage.return_value = np.random.randint(0, 65536, size=(200, 200), dtype=np.uint16)
        camera = MicroManagerCamera(mock_mmc, "pymmcore")
        img = camera.pop_next_image()
        
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint16
    
    def test_camera_snap_image(self, mock_mmc):
        """Test snap_image."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        camera.snap_image()
        
        mock_mmc.snapImage.assert_called_once()
    
    def test_camera_get_image_pycromanager(self, mock_mmc):
        """Test get_image with pycromanager (reshapes)."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        img = camera.get_image()
        
        mock_mmc.getImage.assert_called_once()
        assert isinstance(img, np.ndarray)
        assert img.shape == (200, 200)
    
    def test_camera_clear_buffer(self, mock_mmc):
        """Test clear_buffer."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        camera.clear_buffer()
        
        mock_mmc.clearCircularBuffer.assert_called_once()
    
    def test_camera_set_exposure(self, mock_mmc):
        """Test set_exposure."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        camera.set_exposure(50.0)
        
        mock_mmc.setExposure.assert_called_once_with(50.0)
    
    def test_camera_get_exposure(self, mock_mmc):
        """Test get_exposure."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        exposure = camera.get_exposure()
        
        mock_mmc.getExposure.assert_called_once()
        assert exposure == 10.0
    
    def test_camera_set_roi(self, mock_mmc):
        """Test set_roi."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        camera.set_roi(10, 20, 100, 150)
        
        mock_mmc.setROI.assert_called_with(10, 20, 100, 150)
        assert camera._roi == (10, 20, 100, 150)
        assert camera._width == 100
        assert camera._height == 150
    
    def test_camera_get_roi_pycromanager(self, mock_mmc):
        """Test get_roi with pycromanager."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        roi = camera.get_roi()
        
        assert roi == (0, 0, 200, 200)
        mock_mmc.getROI.assert_called()
    
    def test_camera_get_roi_pymmcore(self, mock_mmc):
        """Test get_roi with pymmcore."""
        mock_mmc.getROI.return_value = (5, 10, 100, 150)
        camera = MicroManagerCamera(mock_mmc, "pymmcore")
        roi = camera.get_roi()
        
        assert roi == (5, 10, 100, 150)
    
    def test_camera_get_image_size(self, mock_mmc):
        """Test get_image_size."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        size = camera.get_image_size()
        
        assert size == (200, 200)
    
    def test_camera_configure_camera(self, mock_mmc):
        """Test configure_camera method."""
        camera = MicroManagerCamera(mock_mmc, "pycromanager")
        config = {"exposure": 30.0, "binning": "2x2"}
        # Should not raise (implementation may vary)
        camera.configure_camera(config)


class TestMicroManagerStage:
    """Test MicroManagerStage implementation."""
    
    def test_stage_initialization(self, mock_mmc):
        """Test MicroManagerStage initializes."""
        stage = MicroManagerStage(mock_mmc, "pycromanager")
        
        assert stage.mmc == mock_mmc
        assert stage.backend == "pycromanager"
    
    def test_stage_get_position_z(self, mock_mmc):
        """Test get_position for Z axis."""
        stage = MicroManagerStage(mock_mmc, "pycromanager")
        pos = stage.get_position('Z')
        
        mock_mmc.getFocusDevice.assert_called()
        mock_mmc.getPosition.assert_called()
        assert isinstance(pos, float)
    
    def test_stage_get_position_all(self, mock_mmc):
        """Test get_position for all axes."""
        stage = MicroManagerStage(mock_mmc, "pycromanager")
        pos = stage.get_position()
        
        assert isinstance(pos, tuple)
        assert len(pos) == 1  # Focus-only stage
    
    def test_stage_move_to_position_float(self, mock_mmc):
        """Test move_to_position with float."""
        stage = MicroManagerStage(mock_mmc, "pycromanager")
        stage.move_to_position(10.0)
        
        mock_mmc.getFocusDevice.assert_called()
        mock_mmc.setPosition.assert_called()
    
    def test_stage_move_to_position_tuple(self, mock_mmc):
        """Test move_to_position with tuple."""
        stage = MicroManagerStage(mock_mmc, "pycromanager")
        stage.move_to_position((10.0,))
        
        mock_mmc.setPosition.assert_called()
    
    def test_stage_run_z_stack(self, mock_mmc):
        """Test run_z_stack (simplified implementation)."""
        stage = MicroManagerStage(mock_mmc, "pycromanager")
        # Should not raise (implementation is simplified)
        stage.run_z_stack(z_start=0.0, z_end=30.0, z_step=3.0, num_planes=11)
    
    def test_stage_stop_sequence(self, mock_mmc):
        """Test stop_sequence."""
        stage = MicroManagerStage(mock_mmc, "pycromanager")
        stage.stop_sequence()
        
        mock_mmc.getFocusDevice.assert_called()
        mock_mmc.stopStageSequence.assert_called()
    
    def test_stage_wait_for_device(self, mock_mmc):
        """Test wait_for_device."""
        stage = MicroManagerStage(mock_mmc, "pycromanager")
        stage.wait_for_device()
        
        mock_mmc.getFocusDevice.assert_called()
        mock_mmc.waitForDevice.assert_called()
    
    def test_stage_get_focus_device_name(self, mock_mmc):
        """Test get_focus_device_name."""
        stage = MicroManagerStage(mock_mmc, "pycromanager")
        name = stage.get_focus_device_name()
        
        mock_mmc.getFocusDevice.assert_called_once()
        assert name == "ZStage"
        
        # Second call should use cached value
        name2 = stage.get_focus_device_name()
        assert name2 == "ZStage"
        # getFocusDevice should only be called once
        assert mock_mmc.getFocusDevice.call_count == 1


class TestMicroManagerStimulus:
    """Test MicroManagerStimulus implementation."""
    
    def test_stimulus_initialization(self, mock_mmc):
        """Test MicroManagerStimulus initializes."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager")
        
        assert stimulus.mmc == mock_mmc
        assert stimulus.backend == "pycromanager"
        assert stimulus._active is False
    
    def test_stimulus_activate(self, mock_mmc):
        """Test activate_stimulus."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager")
        params = {"intensity": 50, "position": (100, 100)}
        
        stimulus.activate_stimulus(params)
        assert stimulus._active is True
    
    def test_stimulus_deactivate(self, mock_mmc):
        """Test deactivate_stimulus."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager")
        stimulus._active = True
        
        stimulus.deactivate_stimulus()
        assert stimulus._active is False
    
    def test_stimulus_configure(self, mock_mmc):
        """Test configure_stimulus."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager")
        config = {"roi": (0, 0, 100, 100)}
        
        # Should not raise (implementation is placeholder)
        stimulus.configure_stimulus(config)
    
    def test_stimulus_is_active(self, mock_mmc):
        """Test is_stimulus_active."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager")
        assert stimulus.is_stimulus_active() is False
        
        stimulus.activate_stimulus({"intensity": 50})
        assert stimulus.is_stimulus_active() is True


class TestMicroManagerBackend:
    """Test MicroManagerBackend class."""
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_backend_initialization_pycromanager(self, mock_mm, pycromanager_config, mock_mmc):
        """Test MicroManagerBackend initializes with pycromanager."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(pycromanager_config)
        backend.initialize()
        
        assert backend.is_initialized is True
        assert backend.mmc == mock_mmc
        assert backend._camera is not None
        assert backend._stage is not None
        assert backend._stimulus is not None
        
        # Check MMSubroutines was called correctly
        mock_mm.initialize_mmc.assert_called_once()
        call_args = mock_mm.initialize_mmc.call_args
        assert call_args[0][0]["gooey_args"]["acquisition_backend"] == "pycromanager"
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_backend_initialization_pymmcore(self, mock_mm, pymmcore_config, mock_mmc):
        """Test MicroManagerBackend initializes with pymmcore."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(pymmcore_config)
        backend.initialize()
        
        assert backend.is_initialized is True
        assert backend.mmc == mock_mmc
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_backend_initialization_with_input_recording(self, mock_mm, pycromanager_config, mock_mmc):
        """Test backend initialization with input recording."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(pycromanager_config)
        backend.initialize(input_recording="test.tiff")
        
        call_args = mock_mm.initialize_mmc.call_args
        assert call_args[0][0]["gooey_args"]["input_recording"] == "test.tiff"
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_backend_camera_property(self, mock_mm, pycromanager_config, mock_mmc):
        """Test camera property access."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(pycromanager_config)
        backend.initialize()
        
        camera = backend.camera
        assert isinstance(camera, MicroManagerCamera)
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_backend_stage_property(self, mock_mm, pycromanager_config, mock_mmc):
        """Test stage property access."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(pycromanager_config)
        backend.initialize()
        
        stage = backend.stage
        assert isinstance(stage, MicroManagerStage)
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_backend_stimulus_property(self, mock_mm, pycromanager_config, mock_mmc):
        """Test stimulus property access."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(pycromanager_config)
        backend.initialize()
        
        stimulus = backend.stimulus
        assert isinstance(stimulus, MicroManagerStimulus)
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_backend_close(self, mock_mm, pycromanager_config, mock_mmc):
        """Test close() method."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(pycromanager_config)
        backend.initialize()
        backend.close()
        
        assert backend.is_initialized is False
        mock_mmc.stopSequenceAcquisition.assert_called()
        mock_mm.close.assert_called_once()
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_backend_get_metadata(self, mock_mm, pycromanager_config, mock_mmc):
        """Test get_metadata() method."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        mock_mm.get_metadata.return_value = {"exposure": 10.0, "binning": "1x1"}
        
        backend = MicroManagerBackend(pycromanager_config)
        backend.initialize()
        
        metadata = backend.get_metadata()
        
        assert isinstance(metadata, dict)
        assert metadata['backend'] == 'pycromanager'
        mock_mm.get_metadata.assert_called_once()
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_backend_get_mmc(self, mock_mm, pycromanager_config, mock_mmc):
        """Test get_mmc() method."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(pycromanager_config)
        backend.initialize()
        
        mmc = backend.get_mmc()
        assert mmc == mock_mmc

