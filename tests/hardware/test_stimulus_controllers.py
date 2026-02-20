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
import numpy as np

# Import the engine and supporting modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hardware.hardware_manager import HardwareManager
from hardware.stimulus_controllers import create_stimulus_controller
from hardware.stimulus_controllers.base_controller import BaseStimulusController
from hardware.stimulus_controllers.simple_controller import SimpleStimulusController
from hardware.stimulus_controllers.widefield_controller import WidefieldStimulusController
from hardware.stimulus_controllers.polygon_controller import PolygonStimulusController
from config.config_manager import HardwareConfig


@pytest.fixture
def mock_hardware_manager():
    """Create mock HardwareManager."""
    hw_manager = Mock(spec=HardwareManager)
    hw_manager.stimulus = Mock()
    return hw_manager


class TestBaseStimulusController:
    """Test BaseStimulusController abstract class."""
    
    def test_base_controller_initialization(self, mock_hardware_manager):
        """Test BaseStimulusController initializes with config."""
        # Create concrete subclass for testing
        class TestController(BaseStimulusController):
            def submit_stim_params(self, stim_params, image_ndx):
                pass
            def _activate_hardware(self, stim_params):
                pass
            def _deactivate_hardware(self):
                pass
        
        controller = TestController(mock_hardware_manager)
        
        assert controller.hardware_manager == mock_hardware_manager
        assert controller.stim_on_list == []
        assert controller.stim_off_list == []
    
    def test_base_controller_check_stim_activation(self, mock_hardware_manager):
        """Test check_stim triggers activation at correct frame."""
        class TestController(BaseStimulusController):
            def submit_stim_params(self, stim_params, image_ndx):
                pass
            def _activate_hardware(self, stim_params):
                self.activated = True
                self.activated_params = stim_params
            def _deactivate_hardware(self):
                pass

        controller = TestController(mock_hardware_manager)
        controller.last_stim_params = {"stim_on": 100, "event": {"value": 50}}
        controller.activated = False

        # Check at wrong frame - should not activate
        controller.check_stim(99)
        assert controller.activated is False

        # Check at correct frame - should activate
        controller.check_stim(100)
        assert controller.activated is True
        assert controller.activated_params["stim_on"] == 100
        assert len(controller.stim_on_time_list) == 1
    
    def test_base_controller_check_stim_deactivation(self, mock_hardware_manager):
        """Test check_stim triggers deactivation at correct frame."""
        class TestController(BaseStimulusController):
            def submit_stim_params(self, stim_params, image_ndx):
                pass
            def _activate_hardware(self, stim_params):
                pass
            def _deactivate_hardware(self):
                self.deactivated = True
        
        controller = TestController(mock_hardware_manager)
        controller.stim_off_list = [148]
        controller.deactivated = False
        
        # Check at wrong frame
        controller.check_stim(147)
        assert controller.deactivated is False
        
        # Check at correct frame
        controller.check_stim(148)
        assert controller.deactivated is True
        assert len(controller.stim_off_time_list) == 1
    
    def test_base_controller_get_metadata(self, mock_hardware_manager):
        """Test get_metadata returns complete metadata."""
        class TestController(BaseStimulusController):
            def submit_stim_params(self, stim_params, image_ndx):
                pass
            def _activate_hardware(self, stim_params):
                pass
            def _deactivate_hardware(self):
                pass
        
        controller = TestController(mock_hardware_manager)
        controller.stim_on_list = [100, 500]
        controller.stim_off_list = [148, 548]
        controller.stim_on_time_list = [1000.0, 2000.0]
        controller.stim_param_list = [{"test": "param"}]
        
        metadata = controller.get_metadata()
        
        assert "stim_on_list" in metadata
        assert metadata["stim_on_list"] == [100, 500]
        assert metadata["stim_off_list"] == [148, 548]
        assert "stim_onset_times_simple" in metadata
    
    def test_base_controller_get_stim_time_onsets(self, mock_hardware_manager):
        """Test get_stim_time_onsets with relative timing."""
        class TestController(BaseStimulusController):
            def submit_stim_params(self, stim_params, image_ndx):
                pass
            def _activate_hardware(self, stim_params):
                pass
            def _deactivate_hardware(self):
                pass
        
        controller = TestController(mock_hardware_manager)
        t0 = 1000.0
        controller.stim_on_time_list = [1010.0, 1020.0, 1030.0]
        
        # Without t0
        onsets = controller.get_stim_time_onsets()
        assert onsets == [1010.0, 1020.0, 1030.0]
        
        # With t0 (relative timing)
        onsets_rel = controller.get_stim_time_onsets(t0=t0)
        assert onsets_rel == [10.0, 20.0, 30.0]


class TestSimpleStimulusController:
    """Test SimpleStimulusController implementation."""

    def test_simple_controller_initialization(self, mock_hardware_manager):
        """Test SimpleStimulusController initializes."""
        controller = SimpleStimulusController(mock_hardware_manager)

        assert controller.hardware_manager == mock_hardware_manager
        assert isinstance(controller, BaseStimulusController)

    def test_simple_controller_submit_stim_params_pulsed(self, mock_hardware_manager):
        """Test submit_stim_params with pulsed stimulus."""
        controller = SimpleStimulusController(mock_hardware_manager)

        stim_params = {
            "stim_on": 100,
            "stim_off": 148,
            "event": {"value": 50}
        }

        controller.submit_stim_params(stim_params, 50)

        assert 100 in controller.stim_on_list
        assert 148 in controller.stim_off_list
        assert controller.last_stim_params == stim_params
        assert stim_params in controller.stim_param_list

    def test_simple_controller_submit_stim_params_none(self, mock_hardware_manager):
        """Test submit_stim_params handles None gracefully."""
        controller = SimpleStimulusController(mock_hardware_manager)

        # Should not raise
        controller.submit_stim_params(None, 50)

        assert len(controller.stim_on_list) == 0

    def test_simple_controller_activate_hardware(self, mock_hardware_manager):
        """Test _activate_hardware delegates to hardware_manager."""
        controller = SimpleStimulusController(mock_hardware_manager)
        stim_params = {"stim_on": 100, "event": {"value": 50}}

        controller._activate_hardware(stim_params)

        mock_hardware_manager.stimulus.activate_stimulus.assert_called_once_with(stim_params)

    def test_simple_controller_deactivate_hardware(self, mock_hardware_manager):
        """Test _deactivate_hardware delegates to hardware_manager."""
        controller = SimpleStimulusController(mock_hardware_manager)

        controller._deactivate_hardware()

        mock_hardware_manager.stimulus.deactivate_stimulus.assert_called_once()

    def test_simple_controller_full_cycle(self, mock_hardware_manager):
        """Test complete stimulus cycle with simple controller."""
        controller = SimpleStimulusController(mock_hardware_manager)

        stim_params = {
            "stim_on": 100,
            "stim_off": 148,
            "event": {"value": 50}
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
    
    def test_widefield_controller_initialization(self, mock_hardware_manager):
        """Test WidefieldStimulusController initializes."""
        controller = WidefieldStimulusController(mock_hardware_manager)
        
        assert controller.hardware_manager == mock_hardware_manager
        assert isinstance(controller, BaseStimulusController)
    
    def test_widefield_controller_submit_standard_stim(self, mock_hardware_manager):
        """Test submit standard pulsed stimulus."""
        controller = WidefieldStimulusController(mock_hardware_manager)
        
        stim_params = {
            "stim_on": 100,
            "stim_off": 148,
            "event": {"stim_intensity": 50, "event_type": "pulse"}
        }
        
        controller.submit_stim_params(stim_params, 50)
        
        assert 100 in controller.stim_on_list
        assert 148 in controller.stim_off_list

    def test_widefield_controller_submit_stream_on(self, mock_hardware_manager):
        """Test submit streaming stimulus ON event."""
        controller = WidefieldStimulusController(mock_hardware_manager)
        
        stim_params = {
            "stim_on": 100,
            "event": {"stim_intensity": 50, "event_type": "stream-widefield"}
        }
        
        controller.submit_stim_params(stim_params, 50)
        
        assert 100 in controller.stim_on_list
        assert len(controller.stim_off_list) == 0  # No off yet
        assert stim_params in controller.stim_param_list
    
    def test_widefield_controller_submit_stream_off(self, mock_hardware_manager):
        """Test submit streaming stimulus OFF event."""
        controller = WidefieldStimulusController(mock_hardware_manager)
        
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
    
    def test_widefield_controller_activate_hardware(self, mock_hardware_manager):
        """Test widefield activation."""
        controller = WidefieldStimulusController(mock_hardware_manager)
        stim_params = {"stim_on": 100, "event": {"value": 75}}

        controller._activate_hardware(stim_params)

        mock_hardware_manager.stimulus.activate_stimulus.assert_called_once_with(stim_params)


class TestPolygonStimulusController:
    """Test PolygonStimulusController implementation."""
    
    def test_polygon_controller_initialization(self, mock_hardware_manager):
        """Test PolygonStimulusController initializes."""
        controller = PolygonStimulusController(mock_hardware_manager)
        
        assert controller.hardware_manager == mock_hardware_manager
        assert controller.calibration_points == {}
        assert isinstance(controller, BaseStimulusController)
    
    def test_polygon_controller_submit_pulsed_stim(self, mock_hardware_manager):
        """Test submit pulsed stimulus."""
        controller = PolygonStimulusController(mock_hardware_manager)
        
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
    
    def test_polygon_controller_hammer_of_dawn_on(self, mock_hardware_manager):
        """Test Hammer of Dawn stimulus ON event."""
        controller = PolygonStimulusController(mock_hardware_manager)
        
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

        # Event should be modified to list format
        event = controller.stim_param_list[0]["event"]
        assert isinstance(event["x"], list)
        assert isinstance(event["y"], list)
        assert event["x"] == [100]
        assert event["y"] == [200]
    
    def test_polygon_controller_hammer_of_dawn_update(self, mock_hardware_manager):
        """Test Hammer of Dawn position update."""
        controller = PolygonStimulusController(mock_hardware_manager)
        
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
    
    def test_polygon_controller_rect_roi_list_stream(self, mock_hardware_manager):
        """Test streaming multi-rectangle ROI list."""
        controller = PolygonStimulusController(mock_hardware_manager)
        
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
    
    def test_polygon_controller_get_metadata(self, mock_hardware_manager):
        """Test polygon metadata includes calibration points."""
        # Mock calibration points
        mock_hardware_manager.stimulus.get_calibration_points.return_value = {
            "pcx": np.array([1, 2, 3]),
            "pcy": np.array([4, 5, 6]),
            "icx": np.array([7, 8, 9]),
            "icy": np.array([10, 11, 12])
        }
        
        controller = PolygonStimulusController(mock_hardware_manager)
        metadata = controller.get_metadata()
        
        assert "calibration_points" in metadata
        assert metadata["calibration_points"]["pcx"] == [1, 2, 3] # casts back to list


class TestStimulusControllerFactory:
    """Test stimulus controller factory function."""
    
    def test_factory_creates_dummy(self, mock_hardware_manager):
        """Test factory creates SimpleStimulusController for dummy."""
        controller = create_stimulus_controller("dummy", mock_hardware_manager)
        assert isinstance(controller, SimpleStimulusController)
    
    def test_factory_creates_dummy_variants(self, mock_hardware_manager):
        """Test factory recognizes dummy variants."""
        for name in ["no stim", "test", "dummy"]:
            controller = create_stimulus_controller(name, mock_hardware_manager)
            assert isinstance(controller, SimpleStimulusController)
    
    def test_factory_creates_widefield(self, mock_hardware_manager):
        """Test factory creates WidefieldStimulusController."""
        controller = create_stimulus_controller(
            "InvCore-SpinningDisk-639", mock_hardware_manager
        )
        assert isinstance(controller, WidefieldStimulusController)
    
    def test_factory_creates_polygon(self, mock_hardware_manager):
        """Test factory creates PolygonStimulusController."""
        controller = create_stimulus_controller(
            "InvCore-LDI-Polygon-640", mock_hardware_manager
        )
        assert isinstance(controller, PolygonStimulusController)
    
    def test_factory_creates_led(self, mock_hardware_manager):
        """Test factory creates controller for LED (uses widefield)."""
        controller = create_stimulus_controller(
            "InvCore-ThunderscopeLED3", mock_hardware_manager
        )
        assert isinstance(controller, WidefieldStimulusController)
    
    def test_factory_invalid_interface(self, mock_hardware_manager):
        """Test factory raises ValueError for unknown interface."""
        with pytest.raises(ValueError, match="Unknown stimulus interface"):
            create_stimulus_controller(
                "UnknownStimInterface", mock_hardware_manager
            )
    
    def test_factory_case_insensitive(self, mock_hardware_manager):
        """Test factory handles case variations."""
        # Lowercase
        controller1 = create_stimulus_controller(
            "invcore-spinningdisk-639", mock_hardware_manager
        )
        assert isinstance(controller1, WidefieldStimulusController)
        
        # Uppercase
        controller2 = create_stimulus_controller(
            "INVCORE-LDI-POLYGON-640", mock_hardware_manager
        )
        assert isinstance(controller2, PolygonStimulusController)


class TestStimulusControllerIntegration:
    """Integration tests for stimulus controllers with HardwareManager."""
    
    def test_full_stimulus_cycle_with_controller(self):
        """Test complete stimulus cycle through controller."""
        from config.config_manager import HardwareConfig, StimulusDeviceConfig
        
        hw_config = HardwareConfig(
            backend="dummy",
            stim_interface="dummy",
            stimulus_devices={
                "dummy": StimulusDeviceConfig(type="dummy")
            }
        )
        
        hw_manager = HardwareManager(hw_config)
        hw_manager.initialize()
        
        controller = create_stimulus_controller("dummy", hw_manager)
        
        # Submit stimulus parameters
        stim_params = {
            "stim_on": 100,
            "stim_off": 148,
            "event": {"value": 50}
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
