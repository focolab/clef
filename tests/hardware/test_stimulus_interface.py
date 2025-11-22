"""
Unit tests for stimulus interface and backend implementations.

Tests the stimulus abstraction layer including dummy and Micro-Manager backends.
"""

import pytest
import os
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Import the engine and supporting modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hardware.hardware_manager import HardwareManager
from hardware.backends.dummy_backend import DummyStimulus
from hardware.backends.micromanager_backend import MicroManagerStimulus
from config.config_manager import HardwareConfig, StimulusDeviceConfig


@pytest.fixture
def dummy_config():
    """Create dummy hardware config with stimulus."""
    return HardwareConfig(
        backend="dummy",
        stim_interface="dummy",
        stimulus_devices={
            "dummy": StimulusDeviceConfig(type="dummy")
        }
    )


@pytest.fixture
def widefield_config():
    """Create widefield laser stimulus config."""
    return HardwareConfig(
        backend="pycromanager",
        mm_config_path="test.cfg",
        stim_interface="InvCore-SpinningDisk-639",
        microscope_name="test_scope",
        stimulus_devices={
            "InvCore-SpinningDisk-639": StimulusDeviceConfig(
                type="widefield_laser",
                voltage_device="DAC639",
                voltage_property="Volts",
                ttl_device="TTL1-8",
                ttl_line="TTL-4",
                max_volts=3.5  # Changed from 5.0 to match test expectations
            )
        }
    )


@pytest.fixture
def polygon_config():
    """Create config with polygon stimulus."""
    return HardwareConfig(
        backend="dummy",
        stim_interface="InvCore-LDI-Polygon-640",
        stimulus_devices={
            "InvCore-LDI-Polygon-640": StimulusDeviceConfig(
                type="polygon",
                intensity_device="89 North Laser Diode Illuminator",
                intensity_property="640 Intensity",
                shutter_device="89 North Laser Diode Illuminator",
                slm_device=None
            )
        }
    )


@pytest.fixture
def mock_mmc():
    """Create mock Micro-Manager Core object."""
    mmc = MagicMock()
    
    # Mock ROI
    roi_mock = MagicMock()
    roi_mock.getX.return_value = 0
    roi_mock.getY.return_value = 0
    roi_mock.getWidth.return_value = 200
    roi_mock.getHeight.return_value = 200
    mmc.getROI.return_value = roi_mock
    
    # Mock SLM
    mmc.getSLMDevice.return_value = "Polygon-SLM"
    mmc.getSLMWidth.return_value = 1920
    mmc.getSLMHeight.return_value = 1080
    
    return mmc


class TestDummyStimulusInterface:
    """Test DummyStimulus implementation."""
    
    def test_dummy_stimulus_initialization(self):
        """Test DummyStimulus initializes correctly."""
        stimulus = DummyStimulus()
        
        assert stimulus.is_stimulus_active() is False
        assert stimulus.stimulus_intensity == 0
        assert stimulus.stimulus_config == {}
        assert stimulus._configured is False
    
    def test_dummy_stimulus_configure(self):
        """Test configure_stimulus stores configuration."""
        stimulus = DummyStimulus()
        config = {"intensity": 10, "duration": 48}
        
        stimulus.configure_stimulus(config)
        
        assert stimulus._configured is True
        assert stimulus.stimulus_config == config
    
    def test_dummy_stimulus_activate(self):
        """Test activate_stimulus sets active state."""
        stimulus = DummyStimulus()
        params = {"intensity": 50}
        
        stimulus.activate_stimulus(params)
        
        assert stimulus.is_stimulus_active() is True
        assert stimulus.stimulus_intensity == 50
    
    def test_dummy_stimulus_activate_default_intensity(self):
        """Test activate with default intensity."""
        stimulus = DummyStimulus()
        stimulus.activate_stimulus({})
        
        assert stimulus.is_stimulus_active() is True
        assert stimulus.stimulus_intensity == 10  # Default
    
    def test_dummy_stimulus_deactivate(self):
        """Test deactivate_stimulus resets state."""
        stimulus = DummyStimulus()
        stimulus.activate_stimulus({"intensity": 50})
        
        stimulus.deactivate_stimulus()
        
        assert stimulus.is_stimulus_active() is False
        assert stimulus.stimulus_intensity == 0
    
    def test_dummy_stimulus_is_active(self):
        """Test is_stimulus_active returns correct state."""
        stimulus = DummyStimulus()
        assert stimulus.is_stimulus_active() is False
        
        stimulus.activate_stimulus({"intensity": 50})
        assert stimulus.is_stimulus_active() is True
        
        stimulus.deactivate_stimulus()
        assert stimulus.is_stimulus_active() is False


class TestMicroManagerStimulusWidefieldLaser:
    """Test MicroManagerStimulus with widefield laser configuration."""
    
    def test_widefield_initialization(self, mock_mmc, widefield_config):
        """Test MicroManagerStimulus initializes with widefield config."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", widefield_config)
        
        assert stimulus.mmc == mock_mmc
        assert stimulus.backend == "pycromanager"
        assert stimulus._active is False
        assert stimulus._configured is False
    
    def test_widefield_configure(self, mock_mmc, widefield_config):
        """Test configure widefield laser stimulus."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", widefield_config)
        
        config = {"interface_type": "InvCore-SpinningDisk-639", "intensity": 10}
        stimulus.configure_stimulus(config)
        
        assert stimulus._configured is True
        assert stimulus._stim_type == "InvCore-SpinningDisk-639"
        assert stimulus._device_config.type == "widefield_laser"
        
        # Should set TTL line
        mock_mmc.setProperty.assert_any_call("TTL1-8", "TTL-4", 1)
        # Should initialize voltage to 0
        mock_mmc.setProperty.assert_any_call("DAC639", "Volts", 0)
    
    def test_widefield_activate(self, mock_mmc, widefield_config):
        """Test activate widefield laser."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", widefield_config)
        stimulus.configure_stimulus({"interface_type": "InvCore-SpinningDisk-639"})
        
        stimulus.activate_stimulus({"intensity": 50})
        
        assert stimulus._active is True
        # Should set voltage: 50% * 3.5V = 1.75V
        mock_mmc.setProperty.assert_any_call("DAC639", "Volts", 1.75)
    
    def test_widefield_activate_different_intensities(self, mock_mmc, widefield_config):
        """Test activate with different intensity values."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", widefield_config)
        stimulus.configure_stimulus({"interface_type": "InvCore-SpinningDisk-639"})
        
        # Test 10%
        stimulus.activate_stimulus({"intensity": 10})
        mock_mmc.setProperty.assert_any_call("DAC639", "Volts", 0.35)
        
        # Test 100%
        stimulus.activate_stimulus({"intensity": 100})
        mock_mmc.setProperty.assert_any_call("DAC639", "Volts", 3.5)
    
    def test_widefield_deactivate(self, mock_mmc, widefield_config):
        """Test deactivate widefield laser."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", widefield_config)
        stimulus.configure_stimulus({"interface_type": "InvCore-SpinningDisk-639"})
        stimulus.activate_stimulus({"intensity": 50})
        
        stimulus.deactivate_stimulus()
        
        assert stimulus._active is False
        # Should set voltage to 0
        mock_mmc.setProperty.assert_any_call("DAC639", "Volts", 0)
    
    def test_widefield_configure_without_device_config(self, mock_mmc):
        """Test configure fails gracefully without device config."""
        config = HardwareConfig(
            backend="pycromanager",
            stim_interface="unknown",
            stimulus_devices={}
        )
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", config)
        
        stimulus.configure_stimulus({"interface_type": "unknown"})
        
        assert stimulus._configured is False
        assert stimulus._device_config is None


class TestMicroManagerStimulusPolygon:
    """Test MicroManagerStimulus with polygon configuration."""
    
    def test_polygon_initialization(self, mock_mmc, polygon_config):
        """Test MicroManagerStimulus initializes with polygon config."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_config)
        
        assert stimulus.mmc == mock_mmc
        assert stimulus.slm_device is None
        assert stimulus.polygon_dims is None
    
    def test_polygon_configure(self, mock_mmc, polygon_config):
        """Test configure polygon stimulus."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_config)
        
        config = {"interface_type": "InvCore-LDI-Polygon-640"}
        stimulus.configure_stimulus(config)
        
        assert stimulus._configured is True
        assert stimulus._stim_type == "InvCore-LDI-Polygon-640"
        assert stimulus._device_config.type == "polygon"
        assert stimulus.slm_device == "Polygon-SLM"
        assert stimulus.polygon_dims == (1920, 1080)
        
        # Should initialize intensity to 0
        mock_mmc.setProperty.assert_any_call(
            "89 North Laser Diode Illuminator", "640 Intensity", 0
        )
        # Should blank SLM
        mock_mmc.setSLMPixelsTo.assert_called_with("Polygon-SLM", 0)
        # Should open shutter
        mock_mmc.setShutterOpen.assert_called_with(
            "89 North Laser Diode Illuminator", True
        )
    
    def test_polygon_activate(self, mock_mmc, polygon_config):
        """Test activate polygon stimulus."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_config)
        stimulus.configure_stimulus({"interface_type": "InvCore-LDI-Polygon-640"})
        
        stimulus.activate_stimulus({"intensity": 30})
        
        assert stimulus._active is True
        mock_mmc.setProperty.assert_any_call(
            "89 North Laser Diode Illuminator", "640 Intensity", 30
        )
    
    def test_polygon_deactivate(self, mock_mmc, polygon_config):
        """Test deactivate polygon stimulus."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_config)
        stimulus.configure_stimulus({"interface_type": "InvCore-LDI-Polygon-640"})
        stimulus.activate_stimulus({"intensity": 30})
        
        stimulus.deactivate_stimulus()
        
        assert stimulus._active is False
        mock_mmc.setProperty.assert_any_call(
            "89 North Laser Diode Illuminator", "640 Intensity", 0
        )
    
    def test_polygon_get_dimensions(self, mock_mmc, polygon_config):
        """Test get_polygon_dimensions returns SLM dimensions."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_config)
        stimulus.configure_stimulus({"interface_type": "InvCore-LDI-Polygon-640"})
        
        dims = stimulus.get_polygon_dimensions()
        assert dims == (1920, 1080)
    
    @patch('hardware.backends.micromanager_backend.json.load')
    @patch('builtins.open', create=True)
    def test_polygon_load_calibration(self, mock_open, mock_json_load, mock_mmc, polygon_config):
        """Test loading polygon calibration points."""
        # Mock calibration data
        mock_json_load.return_value = {
            "calibrations": [
                {
                    "objective": "20x",
                    "binning": "1",
                    "datetime": "2024-01-01",
                    "pcx": [1, 2, 3],
                    "pcy": [4, 5, 6],
                    "icx": [7, 8, 9],
                    "icy": [10, 11, 12]
                }
            ]
        }
        
        # Mock MMC methods
        mock_mmc.getProperty.side_effect = lambda dev, prop: {
            ("ObjectiveTurret", "Label"): "20x",
            ("Camera", "Binning"): "1"
        }.get((dev, prop))
        mock_mmc.getCameraDevice.return_value = "Camera"
        
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_config)
        
        config = {
            "interface_type": "InvCore-LDI-Polygon-640",
            "calibration_path": "test_calibration.json"
        }
        stimulus.configure_stimulus(config)
        
        # Should have loaded calibration
        assert stimulus.calibration_points is not None
        calib = stimulus.calibration_points
        assert np.array_equal(calib['pcx'], np.array([1, 2, 3]))
        assert np.array_equal(calib['pcy'], np.array([4, 5, 6]))


class TestStimulusInterfaceHardwareManager:
    """Test stimulus interface through HardwareManager."""
    
    def test_stimulus_interface_access_dummy(self, dummy_config):
        """Test stimulus interface accessible through HardwareManager."""
        hw_manager = HardwareManager(dummy_config)
        hw_manager.initialize()
        
        stimulus = hw_manager.stimulus
        assert stimulus is not None
        assert hasattr(stimulus, 'activate_stimulus')
        assert hasattr(stimulus, 'deactivate_stimulus')
        assert hasattr(stimulus, 'configure_stimulus')
        assert hasattr(stimulus, 'is_stimulus_active')
    
    def test_stimulus_interface_before_initialization(self, dummy_config):
        """Test stimulus raises error before initialization."""
        hw_manager = HardwareManager(dummy_config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = hw_manager.stimulus
    
    def test_stimulus_operations_dummy(self, dummy_config):
        """Test basic stimulus operations with dummy backend."""
        hw_manager = HardwareManager(dummy_config)
        hw_manager.initialize()
        
        # Configure
        hw_manager.stimulus.configure_stimulus({"intensity": 10})
        
        # Activate
        hw_manager.stimulus.activate_stimulus({"intensity": 50})
        assert hw_manager.stimulus.is_stimulus_active() is True
        
        # Deactivate
        hw_manager.stimulus.deactivate_stimulus()
        assert hw_manager.stimulus.is_stimulus_active() is False
    
    @patch('hardware.backends.micromanager_backend.MMSubroutines')
    def test_stimulus_operations_widefield(self, mock_mm, widefield_config, mock_mmc):
        """Test widefield stimulus through HardwareManager."""
        mock_mm.initialize_mmc.return_value = mock_mmc
        
        # Update config to use micromanager backend
        widefield_config.backend = "pycromanager"
        widefield_config.mm_config_path = "test.cfg"
        
        hw_manager = HardwareManager(widefield_config)
        hw_manager.initialize()
        
        # Configure stimulus
        hw_manager.stimulus.configure_stimulus({
            "interface_type": "InvCore-SpinningDisk-639",
            "intensity": 10
        })
        
        # Activate
        hw_manager.stimulus.activate_stimulus({"intensity": 50})
        
        # Should have set voltage
        mock_mmc.setProperty.assert_any_call("DAC639", "Volts", 1.75)


class TestStimulusDeviceConfig:
    """Test StimulusDeviceConfig model."""
    
    def test_stimulus_device_config_widefield(self):
        """Test creating widefield laser config."""
        config = StimulusDeviceConfig(
            type="widefield_laser",
            voltage_device="DAC639",
            voltage_property="Volts",
            ttl_device="TTL1-8",
            ttl_line="TTL-4",
            max_volts=3.5
        )
        
        assert config.type == "widefield_laser"
        assert config.voltage_device == "DAC639"
        assert config.max_volts == 3.5
    
    def test_stimulus_device_config_polygon(self):
        """Test creating polygon config."""
        config = StimulusDeviceConfig(
            type="polygon",
            intensity_device="LDI",
            intensity_property="Intensity",
            shutter_device="LDI-Shutter"
        )
        
        assert config.type == "polygon"
        assert config.intensity_device == "LDI"
        assert config.shutter_device == "LDI-Shutter"
    
    def test_stimulus_device_config_extra_fields(self):
        """Test extra fields are allowed."""
        config = StimulusDeviceConfig(
            type="custom",
            custom_field="custom_value"
        )
        
        assert config.type == "custom"
        # Extra fields should be preserved
        assert config.model_dump().get("custom_field") == "custom_value"


class TestHardwareConfigStimulusDevices:
    """Test HardwareConfig with stimulus device configurations."""
    
    def test_hardware_config_with_stimulus_devices(self):
        """Test HardwareConfig with stimulus_devices."""
        config = HardwareConfig(
            backend="pycromanager",
            stim_interface="InvCore-SpinningDisk-639",
            stimulus_devices={
                "InvCore-SpinningDisk-639": StimulusDeviceConfig(
                    type="widefield_laser",
                    voltage_device="DAC639",
                    max_volts=3.5
                )
            }
        )
        
        assert "InvCore-SpinningDisk-639" in config.stimulus_devices
        stim_config = config.stimulus_devices["InvCore-SpinningDisk-639"]
        assert stim_config.type == "widefield_laser"
    
    def test_get_stimulus_device_config(self):
        """Test get_stimulus_device_config helper method."""
        config = HardwareConfig(
            backend="dummy",
            stim_interface="InvCore-SpinningDisk-639",
            stimulus_devices={
                "InvCore-SpinningDisk-639": StimulusDeviceConfig(
                    type="widefield_laser",
                    max_volts=3.5
                )
            }
        )
        
        # Get using stim_interface
        stim_config = config.get_stimulus_device_config()
        assert stim_config is not None
        assert stim_config.type == "widefield_laser"
        
        # Get using explicit name
        stim_config2 = config.get_stimulus_device_config("InvCore-SpinningDisk-639")
        assert stim_config2 == stim_config
    
    def test_get_stimulus_device_config_missing(self):
        """Test get_stimulus_device_config returns None for missing config."""
        config = HardwareConfig(
            backend="dummy",
            stim_interface="unknown",
            stimulus_devices={}
        )
        
        stim_config = config.get_stimulus_device_config()
        assert stim_config is None
    
    def test_hardware_config_empty_stimulus_devices(self):
        """Test HardwareConfig with empty stimulus_devices."""
        config = HardwareConfig(
            backend="dummy",
            stim_interface="dummy"
        )
        
        assert config.stimulus_devices == {}
        assert config.get_stimulus_device_config() is None
