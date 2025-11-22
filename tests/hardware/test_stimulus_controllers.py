"""
Unit tests for stimulus controllers.

Tests the high-level stimulus controller architecture including
timing, coordination, and metadata tracking.
"""

import pytest
import os
import time
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Import the engine and supporting modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hardware.hardware_manager import HardwareManager
from hardware.stimulus_controllers import create_stimulus_controller
from hardware.stimulus_controllers.base_controller import BaseStimulusController
from hardware.stimulus_controllers.dummy_controller import DummyStimulusController
from hardware.stimulus_controllers.widefield_controller import WidefieldStimulusController
from hardware.stimulus_controllers.polygon_controller import PolygonStimulusController
from config.config_manager import HardwareConfig


@pytest.fixture
def mock_hardware_manager():
    """Create mock HardwareManager."""
    hw_manager = Mock(spec=HardwareManager)
    hw_manager.stimulus = Mock()
    return hw_manager


@pytest.fixture
def minimal_config():
    """Create minimal config dictionary."""
    return {
        "id": "test_rec_001",
        "roi": [0, 0],
        "saveroot": "/tmp/test",
        "t0": time.time(),
        "gooey_args": {
            "zsize": 1,
            "stim_interface": "dummy",
            "acquisition_backend": "dummy",
            "trigger_algorithm": "dummy",
            "microscope_name": "test_scope"
        }
    }


class TestBaseStimulusController:
    """Test BaseStimulusController abstract class."""
    
    def test_base_controller_initialization(self, mock_hardware_manager, minimal_config):
        """Test BaseStimulusController initializes with config."""
        # Create concrete subclass for testing
        class TestController(BaseStimulusController):
            def submit_stim_params(self, stim_params, image_ndx):
                pass
            def _activate_hardware(self, intensity):
                pass
            def _deactivate_hardware(self):
                pass
        
        controller = TestController(mock_hardware_manager, minimal_config)
        
        assert controller.hardware_manager == mock_hardware_manager
        assert controller.config == minimal_config
        assert controller.rec_id == "test_rec_001"
        assert controller.roi == [0, 0]
        assert controller.stim_interface == "dummy"
        assert controller.stim_on_list == []
        assert controller.stim_off_list == []
    
    def test_base_controller_check_stim_activation(self, mock_hardware_manager, minimal_config):
        """Test check_stim triggers activation at correct frame."""
        class TestController(BaseStimulusController):
            def submit_stim_params(self, stim_params, image_ndx):
                pass
            def _activate_hardware(self, intensity):
                self.activated = True
                self.activation_intensity = intensity
            def _deactivate_hardware(self):
                pass
        
        controller = TestController(mock_hardware_manager, minimal_config)
        controller.stim_on_list = [100]
        controller.stim_intensity_list = [50]
        controller.activated = False
        
        # Check at wrong frame - should not activate
        controller.check_stim(99)
        assert controller.activated is False
        
        # Check at correct frame - should activate
        controller.check_stim(100)
        assert controller.activated is True
        assert controller.activation_intensity == 50
        assert len(controller.stim_on_time_list) == 1
    
    def test_base_controller_check_stim_deactivation(self, mock_hardware_manager, minimal_config):
        """Test check_stim triggers deactivation at correct frame."""
        class TestController(BaseStimulusController):
            def submit_stim_params(self, stim_params, image_ndx):
                pass
            def _activate_hardware(self, intensity):
                pass
            def _deactivate_hardware(self):
                self.deactivated = True
        
        controller = TestController(mock_hardware_manager, minimal_config)
        controller.stim_off_list = [148]
        controller.deactivated = False
        
        # Check at wrong frame
        controller.check_stim(147)
        assert controller.deactivated is False
        
        # Check at correct frame
        controller.check_stim(148)
        assert controller.deactivated is True
        assert len(controller.stim_off_time_list) == 1
    
    def test_base_controller_get_metadata(self, mock_hardware_manager, minimal_config):
        """Test get_metadata returns complete metadata."""
        class TestController(BaseStimulusController):
            def submit_stim_params(self, stim_params, image_ndx):
                pass
            def _activate_hardware(self, intensity):
                pass
            def _deactivate_hardware(self):
                pass
        
        controller = TestController(mock_hardware_manager, minimal_config)
        controller.stim_on_list = [100, 500]
        controller.stim_off_list = [148, 548]
        controller.stim_on_time_list = [1000.0, 2000.0]
        controller.stim_param_list = [{"test": "param"}]
        
        metadata = controller.get_metadata()
        
        assert "stim_on_list" in metadata
        assert metadata["stim_on_list"] == [100, 500]
        assert metadata["stim_off_list"] == [148, 548]
        assert "stim_onset_times_simple" in metadata
    
    def test_base_controller_get_stim_time_onsets(self, mock_hardware_manager, minimal_config):
        """Test get_stim_time_onsets with relative timing."""
        class TestController(BaseStimulusController):
            def submit_stim_params(self, stim_params, image_ndx):
                pass
            def _activate_hardware(self, intensity):
                pass
            def _deactivate_hardware(self):
                pass
        
        controller = TestController(mock_hardware_manager, minimal_config)
        t0 = 1000.0
        controller.stim_on_time_list = [1010.0, 1020.0, 1030.0]
        
        # Without t0
        onsets = controller.get_stim_time_onsets()
        assert onsets == [1010.0, 1020.0, 1030.0]
        
        # With t0 (relative timing)
        onsets_rel = controller.get_stim_time_onsets(t0=t0)
        assert onsets_rel == [10.0, 20.0, 30.0]


class TestDummyStimulusController:
    """Test DummyStimulusController implementation."""
    
    def test_dummy_controller_initialization(self, mock_hardware_manager, minimal_config):
        """Test DummyStimulusController initializes."""
        controller = DummyStimulusController(mock_hardware_manager, minimal_config)
        
        assert controller.hardware_manager == mock_hardware_manager
        assert isinstance(controller, BaseStimulusController)
    
    def test_dummy_controller_submit_stim_params_pulsed(self, mock_hardware_manager, minimal_config):
        """Test submit_stim_params with pulsed stimulus."""
        controller = DummyStimulusController(mock_hardware_manager, minimal_config)
        
        stim_params = {
            "stim_on": 100,
            "stim_off": 148,
            "event": {"stim_intensity": 50}
        }
        
        controller.submit_stim_params(stim_params, 50)
        
        assert 100 in controller.stim_on_list
        assert 148 in controller.stim_off_list
        assert 50 in controller.stim_intensity_list
        assert stim_params in controller.stim_param_list
    
    def test_dummy_controller_submit_stim_params_none(self, mock_hardware_manager, minimal_config):
        """Test submit_stim_params handles None gracefully."""
        controller = DummyStimulusController(mock_hardware_manager, minimal_config)
        
        # Should not raise
        controller.submit_stim_params(None, 50)
        
        assert len(controller.stim_on_list) == 0
    
    def test_dummy_controller_activate_hardware(self, mock_hardware_manager, minimal_config):
        """Test _activate_hardware delegates to hardware_manager."""
        controller = DummyStimulusController(mock_hardware_manager, minimal_config)
        
        controller._activate_hardware(50)
        
        mock_hardware_manager.stimulus.activate_stimulus.assert_called_once()
        call_args = mock_hardware_manager.stimulus.activate_stimulus.call_args[0][0]
        assert call_args["intensity"] == 50
    
    def test_dummy_controller_deactivate_hardware(self, mock_hardware_manager, minimal_config):
        """Test _deactivate_hardware delegates to hardware_manager."""
        controller = DummyStimulusController(mock_hardware_manager, minimal_config)
        
        controller._deactivate_hardware()
        
        mock_hardware_manager.stimulus.deactivate_stimulus.assert_called_once()
    
    def test_dummy_controller_full_cycle(self, mock_hardware_manager, minimal_config):
        """Test complete stimulus cycle with dummy controller."""
        controller = DummyStimulusController(mock_hardware_manager, minimal_config)
        
        # Submit params
        stim_params = {
            "stim_on": 100,
            "stim_off": 148,
            "event": {"stim_intensity": 50}
        }
        controller.submit_stim_params(stim_params, 50)
        
        # Check at activation frame
        controller.check_stim(100)
        mock_hardware_manager.stimulus.activate_stimulus.assert_called_once()
        
        # Check at deactivation frame
        controller.check_stim(148)
        mock_hardware_manager.stimulus.deactivate_stimulus.assert_called_once()


class TestWidefieldStimulusController:
    """Test WidefieldStimulusController implementation."""
    
    def test_widefield_controller_initialization(self, mock_hardware_manager, minimal_config):
        """Test WidefieldStimulusController initializes."""
        controller = WidefieldStimulusController(mock_hardware_manager, minimal_config)
        
        assert controller.hardware_manager == mock_hardware_manager
        assert isinstance(controller, BaseStimulusController)
    
    def test_widefield_controller_submit_standard_stim(self, mock_hardware_manager, minimal_config):
        """Test submit standard pulsed stimulus."""
        controller = WidefieldStimulusController(mock_hardware_manager, minimal_config)
        
        stim_params = {
            "stim_on": 100,
            "stim_off": 148,
            "event": {"stim_intensity": 50, "event_type": "pulse"}
        }
        
        controller.submit_stim_params(stim_params, 50)
        
        assert 100 in controller.stim_on_list
        assert 148 in controller.stim_off_list
        assert 50 in controller.stim_intensity_list
    
    def test_widefield_controller_submit_stream_on(self, mock_hardware_manager, minimal_config):
        """Test submit streaming stimulus ON event."""
        controller = WidefieldStimulusController(mock_hardware_manager, minimal_config)
        
        stim_params = {
            "stim_on": 100,
            "event": {"stim_intensity": 50, "event_type": "stream-widefield"}
        }
        
        controller.submit_stim_params(stim_params, 50)
        
        assert 100 in controller.stim_on_list
        assert len(controller.stim_off_list) == 0  # No off yet
        assert stim_params in controller.stim_param_list
    
    def test_widefield_controller_submit_stream_off(self, mock_hardware_manager, minimal_config):
        """Test submit streaming stimulus OFF event."""
        controller = WidefieldStimulusController(mock_hardware_manager, minimal_config)
        
        # Submit ON first
        stim_on_params = {
            "stim_on": 100,
            "event": {"stim_intensity": 50, "event_type": "stream-widefield"}
        }
        controller.submit_stim_params(stim_on_params, 50)
        
        # Now submit OFF
        stim_off_params = {
            "stim_off": 148,
            "event": {"event_type": "stream-widefield"}
        }
        controller.submit_stim_params(stim_off_params, 98)
        
        assert 148 in controller.stim_off_list
        # Should update previous param with stim_off
        assert controller.stim_param_list[0]["stim_off"] == 148
    
    def test_widefield_controller_activate_hardware(self, mock_hardware_manager, minimal_config):
        """Test widefield activation."""
        controller = WidefieldStimulusController(mock_hardware_manager, minimal_config)
        
        controller._activate_hardware(75)
        
        mock_hardware_manager.stimulus.activate_stimulus.assert_called_once()
        call_args = mock_hardware_manager.stimulus.activate_stimulus.call_args[0][0]
        assert call_args["intensity"] == 75


class TestPolygonStimulusController:
    """Test PolygonStimulusController implementation."""
    
    def test_polygon_controller_initialization(self, mock_hardware_manager, minimal_config):
        """Test PolygonStimulusController initializes."""
        controller = PolygonStimulusController(mock_hardware_manager, minimal_config)
        
        assert controller.hardware_manager == mock_hardware_manager
        assert controller.calibration_points == {}
        assert isinstance(controller, BaseStimulusController)
    
    def test_polygon_controller_submit_pulsed_stim(self, mock_hardware_manager, minimal_config):
        """Test submit pulsed stimulus."""
        controller = PolygonStimulusController(mock_hardware_manager, minimal_config)
        
        stim_params = {
            "stim_on": 100,
            "stim_off": 148,
            "event": {
                "stim_intensity": 50,
                "event_type": "circle-click",
                "x": 100,
                "y": 200,
                "stim_diameter": 20
            }
        }
        
        controller.submit_stim_params(stim_params, 50)
        
        assert 100 in controller.stim_on_list
        assert 148 in controller.stim_off_list
        # Should have called update_polygon_mask
        mock_hardware_manager.stimulus.update_polygon_mask.assert_called_once()
    
    def test_polygon_controller_hammer_of_dawn_on(self, mock_hardware_manager, minimal_config):
        """Test Hammer of Dawn stimulus ON event."""
        controller = PolygonStimulusController(mock_hardware_manager, minimal_config)
        
        stim_params = {
            "stim_on": 100,
            "event": {
                "stim_intensity": 50,
                "event_type": "hammer-of-dawn",
                "x": 100,
                "y": 200,
                "stim_diameter": 20
            },
            "stim_intensity": 50
        }
        
        controller.submit_stim_params(stim_params, 50)
        
        assert 100 in controller.stim_on_list
        assert 50 in controller.stim_intensity_list
        
        # Event should be modified to list format
        event = controller.stim_param_list[0]["event"]
        assert isinstance(event["x"], list)
        assert isinstance(event["y"], list)
        assert event["x"] == [100]
        assert event["y"] == [200]
    
    def test_polygon_controller_hammer_of_dawn_update(self, mock_hardware_manager, minimal_config):
        """Test Hammer of Dawn position update."""
        controller = PolygonStimulusController(mock_hardware_manager, minimal_config)
        
        # Submit ON
        stim_on_params = {
            "stim_on": 100,
            "event": {
                "event_type": "hammer-of-dawn",
                "x": 100,
                "y": 200,
                "stim_diameter": 20
            },
            "stim_intensity": 50
        }
        controller.submit_stim_params(stim_on_params, 50)
        
        # Submit position update
        stim_update_params = {
            "event": {
                "event_type": "hammer-of-dawn",
                "x": 110,
                "y": 210,
                "stim_diameter": 20
            }
        }
        controller.submit_stim_params(stim_update_params, 60)
        
        # Should append to existing lists
        event = controller.stim_param_list[0]["event"]
        assert event["x"] == [100, 110]
        assert event["y"] == [200, 210]
    
    def test_polygon_controller_rect_roi_list_stream(self, mock_hardware_manager, minimal_config):
        """Test streaming multi-rectangle ROI list."""
        controller = PolygonStimulusController(mock_hardware_manager, minimal_config)
        
        # Submit ON with ROI list
        stim_params = {
            "stim_on": 100,
            "event": {
                "event_type": "stream-rect-roi-list",
                "stim_intensity": 50,
                "stim_rect_roi_list": {
                    "x": [10, 20, 30],
                    "y": [40, 50, 60],
                    "width": [5, 5, 5],
                    "height": [5, 5, 5]
                }
            }
        }
        controller.submit_stim_params(stim_params, 50)
        
        assert 100 in controller.stim_on_list
        
        # Submit OFF
        stim_off_params = {
            "stim_off": 148,
            "event": {"event_type": "stream-rect-roi-list"}
        }
        controller.submit_stim_params(stim_off_params, 98)
        
        assert 148 in controller.stim_off_list
        assert controller.stim_param_list[0]["stim_off"] == 148
    
    def test_polygon_controller_get_metadata(self, mock_hardware_manager, minimal_config):
        """Test polygon metadata includes calibration points."""
        # Mock calibration points
        mock_hardware_manager.stimulus.get_calibration_points.return_value = {
            "pcx": [1, 2, 3],
            "pcy": [4, 5, 6],
            "icx": [7, 8, 9],
            "icy": [10, 11, 12]
        }
        
        controller = PolygonStimulusController(mock_hardware_manager, minimal_config)
        metadata = controller.get_metadata()
        
        assert "calibration_points" in metadata
        assert metadata["calibration_points"]["pcx"] == [1, 2, 3]


class TestStimulusControllerFactory:
    """Test stimulus controller factory function."""
    
    def test_factory_creates_dummy(self, mock_hardware_manager, minimal_config):
        """Test factory creates DummyStimulusController for dummy."""
        controller = create_stimulus_controller("dummy", mock_hardware_manager, minimal_config)
        assert isinstance(controller, DummyStimulusController)
    
    def test_factory_creates_dummy_variants(self, mock_hardware_manager, minimal_config):
        """Test factory recognizes dummy variants."""
        for name in ["no stim", "test", "dummy"]:
            controller = create_stimulus_controller(name, mock_hardware_manager, minimal_config)
            assert isinstance(controller, DummyStimulusController)
    
    def test_factory_creates_widefield(self, mock_hardware_manager, minimal_config):
        """Test factory creates WidefieldStimulusController."""
        controller = create_stimulus_controller(
            "InvCore-SpinningDisk-639", mock_hardware_manager, minimal_config
        )
        assert isinstance(controller, WidefieldStimulusController)
    
    def test_factory_creates_polygon(self, mock_hardware_manager, minimal_config):
        """Test factory creates PolygonStimulusController."""
        controller = create_stimulus_controller(
            "InvCore-LDI-Polygon-640", mock_hardware_manager, minimal_config
        )
        assert isinstance(controller, PolygonStimulusController)
    
    def test_factory_creates_led(self, mock_hardware_manager, minimal_config):
        """Test factory creates controller for LED (uses widefield)."""
        controller = create_stimulus_controller(
            "InvCore-ThunderscopeLED3", mock_hardware_manager, minimal_config
        )
        assert isinstance(controller, WidefieldStimulusController)
    
    def test_factory_invalid_interface(self, mock_hardware_manager, minimal_config):
        """Test factory raises ValueError for unknown interface."""
        with pytest.raises(ValueError, match="Unknown stimulus interface"):
            create_stimulus_controller(
                "UnknownStimInterface", mock_hardware_manager, minimal_config
            )
    
    def test_factory_case_insensitive(self, mock_hardware_manager, minimal_config):
        """Test factory handles case variations."""
        # Lowercase
        controller1 = create_stimulus_controller(
            "invcore-spinningdisk-639", mock_hardware_manager, minimal_config
        )
        assert isinstance(controller1, WidefieldStimulusController)
        
        # Uppercase
        controller2 = create_stimulus_controller(
            "INVCORE-LDI-POLYGON-640", mock_hardware_manager, minimal_config
        )
        assert isinstance(controller2, PolygonStimulusController)


class TestStimulusControllerIntegration:
    """Integration tests for stimulus controllers with HardwareManager."""
    
    def test_full_stimulus_cycle_with_controller(self):
        """Test complete stimulus cycle through controller."""
        from config.config_manager import HardwareConfig, StimulusDeviceConfig
        
        config = HardwareConfig(
            backend="dummy",
            stim_interface="dummy",
            stimulus_devices={
                "dummy": StimulusDeviceConfig(type="dummy")
            }
        )
        
        hw_manager = HardwareManager(config)
        hw_manager.initialize()
        
        # Create controller
        controller_config = {
            "id": "test",
            "roi": [0, 0],
            "saveroot": "/tmp",
            "t0": time.time(),
            "gooey_args": {
                "stim_interface": "dummy",
                "acquisition_backend": "dummy",
                "trigger_algorithm": "dummy",
                "microscope_name": "test"
            }
        }
        
        controller = create_stimulus_controller("dummy", hw_manager, controller_config)
        
        # Submit stimulus parameters
        stim_params = {
            "stim_on": 100,
            "stim_off": 148,
            "event": {"stim_intensity": 50}
        }
        controller.submit_stim_params(stim_params, 50)
        
        # Check activation
        controller.check_stim(100)
        assert hw_manager.stimulus.is_stimulus_active() is True
        
        # Check deactivation
        controller.check_stim(148)
        assert hw_manager.stimulus.is_stimulus_active() is False
        
        # Get metadata
        metadata = controller.get_metadata()
        assert len(metadata["stim_on_list"]) == 1
        assert len(metadata["stim_off_list"]) == 1
