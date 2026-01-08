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
from config.config_manager import HardwareConfig, DeviceConfig, DeviceProperties, SystemProperties, ShutterConfig, StimulusDeviceConfig


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
        microscope_name="test_scope",
        stimulus_devices={
            "dummy": StimulusDeviceConfig(type="dummy")
        }
    )


@pytest.fixture
def pymmcore_config():
    """Create pymmcore hardware config."""
    return HardwareConfig(
        backend="pymmcore",
        stim_interface="dummy",
        mm_config_path="test_config.cfg",
        microscope_name="test_scope",
        stimulus_devices={
            "dummy": StimulusDeviceConfig(type="dummy")
        }
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
        camera.start_acquisition(buffer_size=5000) # default val, 20260108 overrides with 10k
        
        mock_mmc.setCircularBufferMemoryFootprint.assert_called_once_with(10000)
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
    
    @patch('hardware.backends.micromanager_backend.JavaObject')
    def test_stage_run_z_stack(self, mock_java_object, mock_mmc):
        """Test run_z_stack (simplified implementation)."""
        # Create mock vector objects that JavaObject will return
        mock_double_vector = MagicMock()
        mock_string_vector = MagicMock()
        
        # Configure JavaObject to return appropriate mocks
        def java_object_factory(class_name):
            if "DoubleVector" in class_name:
                return mock_double_vector
            elif "StrVector" in class_name:
                return mock_string_vector
            return MagicMock()
        
        mock_java_object.side_effect = java_object_factory
        
        stage = MicroManagerStage(mock_mmc, "pycromanager")
        # Should not raise (implementation is simplified)
        stage.run_z_stack(z_start=0.0, z_end=30.0, z_step=3.0, num_planes=11)
        
        # Verify JavaObject was called for stage configuration
        assert mock_java_object.call_count >= 2  # Called for DoubleVector and StrVector

    def test_stage_run_z_stack_pymmcore(self, mock_mmc):
        """Test run_z_stack with pymmcore backend (doesn't need JavaObject)."""
        stage = MicroManagerStage(mock_mmc, "pymmcore")
        # Should not raise
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
    
    def test_stimulus_initialization(self, mock_mmc, pycromanager_config):
        """Test MicroManagerStimulus initializes."""
        # Pass HardwareConfig object, not fixture function
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", pycromanager_config)
        
        assert stimulus.mmc == mock_mmc
        assert stimulus.backend == "pycromanager"
        assert stimulus._active is False
    
    def test_stimulus_activate(self, mock_mmc, pycromanager_config):
        """Test activate_stimulus."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", pycromanager_config)
        
        # Configure first with a dummy stimulus type
        stimulus.configure_stimulus({
            "interface_type": "dummy"
        })
        
        params = {"intensity": 50, "position": (100, 100)}
        stimulus.activate_stimulus(params)
        assert stimulus._active is True
    
    def test_stimulus_deactivate(self, mock_mmc, pycromanager_config):
        """Test deactivate_stimulus."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", pycromanager_config)
        
        # Configure first
        stimulus.configure_stimulus({"interface_type": "dummy"})
        stimulus._active = True
        
        stimulus.deactivate_stimulus()
        assert stimulus._active is False
    
    def test_stimulus_configure(self, mock_mmc, pycromanager_config):
        """Test configure_stimulus."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", pycromanager_config)
        config = {"interface_type": "dummy"}
        
        # Should not raise
        stimulus.configure_stimulus(config)
        assert stimulus._configured is True
    
    def test_stimulus_is_active(self, mock_mmc, pycromanager_config):
        """Test is_stimulus_active."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", pycromanager_config)
        assert stimulus.is_stimulus_active() is False
        
        # Configure first
        stimulus.configure_stimulus({"interface_type": "dummy"})
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


class TestMicroManagerBackendDeviceConfiguration:
    """Test device properties, configs, and system properties initialization."""
    
    @pytest.fixture
    def config_with_devices(self):
        """Create config with device properties and configs."""
        return HardwareConfig(
            backend="pycromanager",
            stim_interface="dummy",
            mm_config_path="test_config.cfg",
            microscope_name="test_scope",
            devices={
                "camera": DeviceConfig(
                    device_name="PrimeBSI",
                    device_type="camera",
                    properties=DeviceProperties(ExposeOutMode="Rolling Shutter", Binning="1")
                ),
                "ttl": DeviceConfig(
                    device_name="TTL1-8",
                    device_type="ttl",
                    properties=DeviceProperties(Blanking="On", Sequence="On"),
                    configs={}
                ),
                "laser": DeviceConfig(
                    device_name="LMM5",
                    device_type="laser",
                    properties=DeviceProperties(),
                    configs={"LMM5": "488+561", "LMM5-561-intensity": "0"}
                )
            }
        )
    
    @pytest.fixture
    def config_with_system_properties(self):
        """Create config with system properties."""
        return HardwareConfig(
            backend="pycromanager",
            stim_interface="dummy",
            mm_config_path="test_config.cfg",
            microscope_name="test_scope",
            system_properties=SystemProperties(
                auto_shutter=False,
                circular_buffer_mb=10000,
                shutters=[
                    ShutterConfig(device_name="LaserShutter", state=True),
                    ShutterConfig(device_name="LMM5-Shutter", state=False)
                ]
            )
        )
    
    @pytest.fixture
    def config_with_all_settings(self):
        """Create config with devices, configs, and system properties."""
        return HardwareConfig(
            backend="pycromanager",
            stim_interface="dummy",
            mm_config_path="test_config.cfg",
            microscope_name="test_scope",
            devices={
                "camera": DeviceConfig(
                    device_name="PrimeBSI",
                    device_type="camera",
                    properties=DeviceProperties(ExposeOutMode="Rolling Shutter")
                ),
                "ttl": DeviceConfig(
                    device_name="TTL1-8",
                    device_type="ttl",
                    properties=DeviceProperties(Blanking="On", Sequence="On")
                ),
                "triggerscope": DeviceConfig(
                    device_name="TriggerScopeMM-Hub",
                    device_type="triggerscope",
                    properties=DeviceProperties(UseActionLEDs="Off")
                )
            },
            system_properties=SystemProperties(
                auto_shutter=False,
                circular_buffer_mb=10000,
                shutters=[ShutterConfig(device_name="LaserShutter", state=True)]
            )
        )
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_apply_device_properties(self, mock_mm, config_with_devices, mock_mmc):
        """Test that device properties are applied via setProperty during initialization."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(config_with_devices)
        backend.initialize()
        
        # Verify setProperty was called for each device property
        set_property_calls = [call for call in mock_mmc.setProperty.call_args_list]
        
        # Check camera properties
        assert any(call[0] == ("PrimeBSI", "ExposeOutMode", "Rolling Shutter") for call in set_property_calls)
        assert any(call[0] == ("PrimeBSI", "Binning", "1") for call in set_property_calls)
        
        # Check TTL properties
        assert any(call[0] == ("TTL1-8", "Blanking", "On") for call in set_property_calls)
        assert any(call[0] == ("TTL1-8", "Sequence", "On") for call in set_property_calls)
        
        # Verify correct number of setProperty calls
        assert mock_mmc.setProperty.call_count == 4
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_apply_device_configs(self, mock_mm, config_with_devices, mock_mmc):
        """Test that device configs are applied via setConfig during initialization."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(config_with_devices)
        backend.initialize()
        
        # Verify setConfig was called for each config preset
        set_config_calls = [call for call in mock_mmc.setConfig.call_args_list]
        
        # Check laser configs
        assert any(call[0] == ("LMM5", "488+561") for call in set_config_calls)
        assert any(call[0] == ("LMM5-561-intensity", "0") for call in set_config_calls)
        
        # Verify correct number of setConfig calls
        assert mock_mmc.setConfig.call_count == 2
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_apply_system_properties_auto_shutter(self, mock_mm, config_with_system_properties, mock_mmc):
        """Test that auto_shutter system property is applied during initialization."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(config_with_system_properties)
        backend.initialize()
        
        # Verify setAutoShutter was called
        mock_mmc.setAutoShutter.assert_called_once_with(False)
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_apply_system_properties_circular_buffer(self, mock_mm, config_with_system_properties, mock_mmc):
        """Test that circular_buffer_mb system property is applied during initialization."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(config_with_system_properties)
        backend.initialize()
        
        # Verify setCircularBufferMemoryFootprint was called
        mock_mmc.setCircularBufferMemoryFootprint.assert_called_once_with(10000)
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_apply_system_properties_shutters(self, mock_mm, config_with_system_properties, mock_mmc):
        """Test that shutter states are applied during initialization."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(config_with_system_properties)
        backend.initialize()
        
        # Verify setShutterOpen was called for each shutter
        shutter_calls = [call for call in mock_mmc.setShutterOpen.call_args_list]
        
        assert any(call[0] == ("LaserShutter", True) for call in shutter_calls)
        assert any(call[0] == ("LMM5-Shutter", False) for call in shutter_calls)
        
        # Verify correct number of calls
        assert mock_mmc.setShutterOpen.call_count == 2
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_apply_all_settings_together(self, mock_mm, config_with_all_settings, mock_mmc):
        """Test that all settings (properties, configs, system) are applied during initialization."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(config_with_all_settings)
        backend.initialize()
        
        # Verify device properties were set
        assert mock_mmc.setProperty.call_count >= 3  # At least 3 properties
        
        # Verify system properties were set
        mock_mmc.setAutoShutter.assert_called_once_with(False)
        mock_mmc.setCircularBufferMemoryFootprint.assert_called_once_with(10000)
        mock_mmc.setShutterOpen.assert_called_once_with("LaserShutter", True)
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_device_properties_error_handling(self, mock_mm, config_with_devices, mock_mmc):
        """Test that errors setting device properties are handled gracefully."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        # Make setProperty raise an exception for one property
        mock_mmc.setProperty.side_effect = [
            None,  # First call succeeds
            Exception("Property not found"),  # Second call fails
            None,  # Third call succeeds
            None,  # Fourth call succeeds
        ]
        
        backend = MicroManagerBackend(config_with_devices)
        # Should not raise - errors should be caught and logged
        backend.initialize()
        
        # Verify initialization still completed
        assert backend.is_initialized is True
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_device_configs_error_handling(self, mock_mm, config_with_devices, mock_mmc):
        """Test that errors setting device configs are handled gracefully."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        # Make setConfig raise an exception for one config
        mock_mmc.setConfig.side_effect = [
            Exception("Config not found"),  # First call fails
            None,  # Second call succeeds
        ]
        
        backend = MicroManagerBackend(config_with_devices)
        # Should not raise - errors should be caught and logged
        backend.initialize()
        
        # Verify initialization still completed
        assert backend.is_initialized is True
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_system_properties_error_handling(self, mock_mm, config_with_system_properties, mock_mmc):
        """Test that errors setting system properties are handled gracefully."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        # Make setAutoShutter raise an exception
        mock_mmc.setAutoShutter.side_effect = Exception("Auto shutter not available")
        
        backend = MicroManagerBackend(config_with_system_properties)
        # Should not raise - errors should be caught and logged
        backend.initialize()
        
        # Verify initialization still completed
        assert backend.is_initialized is True
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_no_devices_config(self, mock_mm, pycromanager_config, mock_mmc):
        """Test initialization with no device configuration (should not raise)."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(pycromanager_config)
        backend.initialize()
        
        # Should initialize successfully even with no devices
        assert backend.is_initialized is True
        # setProperty should not be called if no devices configured
        mock_mmc.setProperty.assert_not_called()
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_no_system_properties_config(self, mock_mm, pycromanager_config, mock_mmc):
        """Test initialization with no system properties (should not raise)."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(pycromanager_config)
        backend.initialize()
        
        # Should initialize successfully even with no system properties
        assert backend.is_initialized is True
        # System property methods should not be called if not configured
        mock_mmc.setAutoShutter.assert_not_called()
        mock_mmc.setCircularBufferMemoryFootprint.assert_not_called()
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_system_properties_partial_config(self, mock_mm, mock_mmc):
        """Test initialization with partial system properties (only some fields set)."""
        config = HardwareConfig(
            backend="pycromanager",
            stim_interface="dummy",
            mm_config_path="test_config.cfg",
            microscope_name="test_scope",
            system_properties=SystemProperties(
                auto_shutter=False,
                # circular_buffer_mb not set (None)
                shutters=[]  # Empty shutters list
            )
        )
        
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(config)
        backend.initialize()
        
        # Only auto_shutter should be called
        mock_mmc.setAutoShutter.assert_called_once_with(False)
        # circular_buffer should not be called (None)
        mock_mmc.setCircularBufferMemoryFootprint.assert_not_called()
        # shutters should not be called (empty list)
        mock_mmc.setShutterOpen.assert_not_called()
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_device_properties_with_extra_fields(self, mock_mm, mock_mmc):
        """Test that extra device property fields (via extra='allow') are applied."""
        # Create properties with extra fields
        props = DeviceProperties(ExposeOutMode="Rolling Shutter")
        # Add extra field dynamically (simulating extra='allow' behavior)
        props_dict = props.model_dump()
        props_dict["CustomProperty"] = "CustomValue"
        
        config = HardwareConfig(
            backend="pycromanager",
            stim_interface="dummy",
            mm_config_path="test_config.cfg",
            microscope_name="test_scope",
            devices={
                "camera": DeviceConfig(
                    device_name="PrimeBSI",
                    device_type="camera",
                    properties=DeviceProperties(**props_dict)
                )
            }
        )
        
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        backend = MicroManagerBackend(config)
        backend.initialize()
        
        # Verify both standard and extra properties were set
        set_property_calls = [call[0] for call in mock_mmc.setProperty.call_args_list]
        assert ("PrimeBSI", "ExposeOutMode", "Rolling Shutter") in set_property_calls
        assert ("PrimeBSI", "CustomProperty", "CustomValue") in set_property_calls

