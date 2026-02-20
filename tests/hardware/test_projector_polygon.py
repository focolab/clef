"""
Unit tests for projector interface and polygon stimulus system.

Tests the ProjectorInterface abstraction and polygon-specific functionality
including mask generation, calibration loading, and integration with the
Micro-Manager backend.
"""

import pytest
import os
import json
import numpy as np
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path

# Import the engine and supporting modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hardware.backends.micromanager_backend import (
    MicroManagerBackend,
    MicroManagerProjector,
    MicroManagerStimulus,
)
from hardware.projector_interface import ProjectorInterface
from hardware.stimulus_controllers.polygon_controller import PolygonStimulusController
from config.config_manager import HardwareConfig, StimulusDeviceConfig


@pytest.fixture
def mock_mmc():
    """Create a mock Micro-Manager Core object with SLM support."""
    mmc = MagicMock()
    
    # Mock ROI
    roi_mock = MagicMock()
    roi_mock.getX.return_value = 0
    roi_mock.getY.return_value = 0
    roi_mock.getWidth.return_value = 512
    roi_mock.getHeight.return_value = 512
    mmc.getROI.return_value = roi_mock
    
    # Mock SLM/Projector
    mmc.getSLMDevice.return_value = "Polygon-SLM"
    mmc.getSLMWidth.return_value = 1920
    mmc.getSLMHeight.return_value = 1080
    
    # Mock other methods
    mmc.getCameraDevice.return_value = "Camera"
    mmc.getProperty.return_value = "1x1"
    mmc.getFocusDevice.return_value = "ZStage"
    
    return mmc


@pytest.fixture
def polygon_hardware_config():
    """Create hardware config for polygon stimulus."""
    return HardwareConfig(
        backend="pycromanager",
        mm_config_path="test_config.cfg",
        stim_interface="InvCore-LDI-Polygon-640",
        microscope_name="test_polygon_scope",
        stimulus_devices={
            "InvCore-LDI-Polygon-640": StimulusDeviceConfig(
                type="polygon",
                intensity_device="89 North Laser Diode Illuminator",
                intensity_property="640 Intensity",
                shutter_device="89 North Laser Diode Illuminator",
                slm_device=None,  # Will be queried from MMC
                polygon_calibration_path="./test_calibrations.json"
            )
        }
    )


@pytest.fixture
def mock_calibration_data():
    """Create mock calibration data."""
    return {
        "calibrations": [
            {
                "objective": "20x",
                "binning": "1x1",
                "datetime": "2024-01-15",
                "pcx": [250, 500, 750],
                "pcy": [200, 500, 800],
                "icx": [880, 1722, 2578],
                "icy": [981, 1494, 1998]
            },
            {
                "objective": "20x",
                "binning": "1x1",
                "datetime": "2024-01-01",  # Older calibration
                "pcx": [90, 190, 290],
                "pcy": [90, 190, 290],
                "icx": [931, 1762, 2617],
                "icy": [986, 1490, 1992]
            }
        ]
    }


class TestProjectorInterface:
    """Test ProjectorInterface abstract class."""
    
    def test_projector_interface_is_abstract(self):
        """Test ProjectorInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ProjectorInterface()
    
    def test_projector_interface_methods_exist(self):
        """Test ProjectorInterface defines required methods."""
        required_methods = [
            'get_dimensions',
            'set_image',
            'set_pixels_to',
            'get_device_name'
        ]
        
        for method in required_methods:
            assert hasattr(ProjectorInterface, method)


class TestMicroManagerProjector:
    """Test MicroManagerProjector implementation."""
    
    def test_projector_initialization_with_device_name(self, mock_mmc):
        """Test MicroManagerProjector initializes with explicit device name."""
        projector = MicroManagerProjector(mock_mmc, "pycromanager", device_name="TestSLM")
        
        assert projector.mmc == mock_mmc
        assert projector.backend == "pycromanager"
        assert projector._device_name == "TestSLM"
        
        # Should have called setSLMDevice
        mock_mmc.setSLMDevice.assert_called_once_with("TestSLM")
    
    def test_projector_initialization_query_device(self, mock_mmc):
        """Test MicroManagerProjector queries device name from MMC."""
        projector = MicroManagerProjector(mock_mmc, "pycromanager", device_name=None)
        
        assert projector._device_name == "Polygon-SLM"
        mock_mmc.getSLMDevice.assert_called_once()
    
    def test_projector_get_dimensions(self, mock_mmc):
        """Test get_dimensions returns SLM dimensions."""
        projector = MicroManagerProjector(mock_mmc, "pycromanager", device_name="TestSLM")
        
        dims = projector.get_dimensions()
        
        assert dims == (1920, 1080)
        mock_mmc.getSLMWidth.assert_called_once_with("TestSLM")
        mock_mmc.getSLMHeight.assert_called_once_with("TestSLM")
    
    def test_projector_set_image(self, mock_mmc):
        """Test set_image uploads image to SLM."""
        projector = MicroManagerProjector(mock_mmc, "pycromanager", device_name="TestSLM")
        
        # Create test image
        image = np.ones((1080, 1920), dtype=np.uint8) * 128
        projector.set_image(image)
        
        # Should have called setSLMImage with flattened image
        mock_mmc.setSLMImage.assert_called_once()
        call_args = mock_mmc.setSLMImage.call_args[0]
        assert call_args[0] == "TestSLM"
        assert isinstance(call_args[1], np.ndarray)
        assert call_args[1].shape == (1920 * 1080,)
    
    def test_projector_set_image_wrong_shape(self, mock_mmc):
        """Test set_image raises ValueError for wrong shape."""
        projector = MicroManagerProjector(mock_mmc, "pycromanager", device_name="TestSLM")
        
        # Create image with wrong dimensions
        wrong_image = np.ones((100, 200), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="doesn't match projector dimensions"):
            projector.set_image(wrong_image)
    
    def test_projector_set_pixels_to(self, mock_mmc):
        """Test set_pixels_to sets uniform value."""
        projector = MicroManagerProjector(mock_mmc, "pycromanager", device_name="TestSLM")
        
        projector.set_pixels_to(255)
        
        mock_mmc.setSLMPixelsTo.assert_called_once_with("TestSLM", 255)
    
    def test_projector_set_pixels_to_bounds_check(self, mock_mmc):
        """Test set_pixels_to validates pixel value range."""
        projector = MicroManagerProjector(mock_mmc, "pycromanager", device_name="TestSLM")
        
        # Test below range
        with pytest.raises(ValueError, match="must be 0-255"):
            projector.set_pixels_to(-1)
        
        # Test above range
        with pytest.raises(ValueError, match="must be 0-255"):
            projector.set_pixels_to(256)
    
    def test_projector_get_device_name(self, mock_mmc):
        """Test get_device_name returns device name."""
        projector = MicroManagerProjector(mock_mmc, "pycromanager", device_name="TestSLM")
        
        name = projector.get_device_name()
        assert name == "TestSLM"


class TestMicroManagerStimulusPolygonIntegration:
    """Test polygon-specific functionality in MicroManagerStimulus."""
    
    def test_polygon_configure_creates_slm_interface(self, mock_mmc, polygon_hardware_config, mock_calibration_data):
        """Test configuring polygon stimulus sets up SLM."""
        mock_mmc.getProperty.side_effect = lambda dev, prop: {
            ("ObjectiveTurret", "Label"): "20x",
            ("Camera", "Binning"): "1x1"
        }.get((dev, prop))
        mock_mmc.getCameraDevice.return_value = "Camera"
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)

        config = {"interface_type": "InvCore-LDI-Polygon-640"}
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_calibration_data))):
            stimulus.configure_stimulus(config)

        assert stimulus._configured is True
        assert stimulus.slm_device == "Polygon-SLM"
        assert stimulus.polygon_dims == (1920, 1080)
        
        # Should have initialized polygon hardware
        mock_mmc.setSLMDevice.assert_called_with("Polygon-SLM")
        mock_mmc.setSLMPixelsTo.assert_called_with("Polygon-SLM", 0)
        mock_mmc.setProperty.assert_any_call("89 North Laser Diode Illuminator", "640 Intensity", 0)
        mock_mmc.setShutterOpen.assert_called_with("89 North Laser Diode Illuminator", True)
    
    def test_polygon_get_dimensions(self, mock_mmc, polygon_hardware_config, mock_calibration_data):
        """Test get_polygon_dimensions returns correct dimensions."""
        mock_mmc.getProperty.side_effect = lambda dev, prop: {
            ("ObjectiveTurret", "Label"): "20x",
            ("Camera", "Binning"): "1x1"
        }.get((dev, prop))
        mock_mmc.getCameraDevice.return_value = "Camera"
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_calibration_data))):
            stimulus.configure_stimulus({"interface_type": "InvCore-LDI-Polygon-640"})

        dims = stimulus.get_polygon_dimensions()
        assert dims == (1920, 1080)
    
    def test_polygon_get_dimensions_before_configure(self, mock_mmc, polygon_hardware_config):
        """Test get_polygon_dimensions returns None before configuration."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)
        
        dims = stimulus.get_polygon_dimensions()
        assert dims is None
    
    def test_polygon_load_calibration_success(self, mock_mmc, polygon_hardware_config, mock_calibration_data):
        """Test loading polygon calibration points successfully."""
        # Mock file operations
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_calibration_data))):
            # Mock MMC property queries
            mock_mmc.getProperty.side_effect = lambda dev, prop: {
                ("ObjectiveTurret", "Label"): "20x",
                ("Camera", "Binning"): "1x1"
            }.get((dev, prop))
            mock_mmc.getCameraDevice.return_value = "Camera"
            
            stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)
            
            config = {
                "interface_type": "InvCore-LDI-Polygon-640",
                "calibration_path": "./test_calibrations.json"
            }
            stimulus.configure_stimulus(config)
            
            # Should have loaded calibration
            assert stimulus.calibration_points is not None
            calib = stimulus.calibration_points
            
            # Should use latest calibration (2024-01-15)
            assert np.array_equal(calib['pcx'], np.array([250, 500, 750]))
            assert np.array_equal(calib['pcy'], np.array([200, 500, 800]))
            assert np.array_equal(calib['icx'], np.array([880, 1722, 2578]))
            assert np.array_equal(calib['icy'], np.array([981, 1494, 1998]))
    

    def test_polygon_load_calibration_no_match(self, mock_mmc, polygon_hardware_config):
        """Test loading calibration with no matching objective/binning raises RuntimeError."""
        calib_data = {
            "calibrations": [
                {
                    "objective": "40x",  # Different objective
                    "binning": "1x1",
                    "datetime": "2024-01-15",
                    "pcx": [250, 500, 750],
                    "pcy": [200, 500, 800],
                    "icx": [880, 1722, 2578],
                    "icy": [981, 1494, 1998]
                }
            ]
        }

        with patch('builtins.open', mock_open(read_data=json.dumps(calib_data))):
            mock_mmc.getProperty.side_effect = lambda dev, prop: {
                ("ObjectiveTurret", "Label"): "20x",  # Looking for 20x
                ("Camera", "Binning"): "1x1"
            }.get((dev, prop))
            mock_mmc.getCameraDevice.return_value = "Camera"

            stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)

            config = {
                "interface_type": "InvCore-LDI-Polygon-640",
                "calibration_path": "./test_calibrations.json"
            }
            with pytest.raises(RuntimeError, match="Polygon calibration could not be loaded"):
                stimulus.configure_stimulus(config)
    
    def test_polygon_load_calibration_file_not_found(self, mock_mmc, polygon_hardware_config):
        """Test loading calibration raises RuntimeError when file is missing."""
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)

        config = {
            "interface_type": "InvCore-LDI-Polygon-640",
            "calibration_path": "./nonexistent.json"
        }

        with pytest.raises(RuntimeError, match="Polygon calibration could not be loaded"):
            stimulus.configure_stimulus(config)
    
    def test_polygon_get_calibration_points(self, mock_mmc, polygon_hardware_config, mock_calibration_data):
        """Test get_calibration_points returns loaded calibration."""
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_calibration_data))):
            mock_mmc.getProperty.side_effect = lambda dev, prop: {
                ("ObjectiveTurret", "Label"): "20x",
                ("Camera", "Binning"): "1x1"
            }.get((dev, prop))
            mock_mmc.getCameraDevice.return_value = "Camera"
            
            stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)
            stimulus.configure_stimulus({
                "interface_type": "InvCore-LDI-Polygon-640",
                "calibration_path": "./test_calibrations.json"
            })
            
            calib = stimulus.get_calibration_points()
            assert calib is not None
            assert 'pcx' in calib
            assert 'pcy' in calib
            assert 'icx' in calib
            assert 'icy' in calib
    
    @patch('hardware.backends.micromanager_backend.numba_utils')
    def test_polygon_update_mask_circle_event(self, mock_utils, mock_mmc, polygon_hardware_config, mock_calibration_data):
        """Test update_polygon_mask with circle event."""
        # Mock mask generation
        mock_mask = np.ones((1080, 1920), dtype=np.uint8) * 128
        mock_utils.generate_pg_ellipse_mask.return_value = mock_mask
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_calibration_data))):
            mock_mmc.getProperty.side_effect = lambda dev, prop: {
                ("ObjectiveTurret", "Label"): "20x",
                ("Camera", "Binning"): "1x1"
            }.get((dev, prop))
            mock_mmc.getCameraDevice.return_value = "Camera"
            
            stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)
            stimulus.configure_stimulus({
                "interface_type": "InvCore-LDI-Polygon-640",
                "calibration_path": "./test_calibrations.json"
            })
            
            stim_params = {
                'event': {
                    'event_type': 'circle-click',
                    'x': 200,
                    'y': 300,
                    'stim_diameter': 30
                },
                'roi': [0, 0]
            }
            
            stimulus.update_polygon_mask(stim_params)
            
            # Should have called mask generation
            mock_utils.generate_pg_ellipse_mask.assert_called_once()
            
            # Should have uploaded mask to SLM
            mock_mmc.setSLMImage.assert_called()
            call_args = mock_mmc.setSLMImage.call_args[0]
            assert call_args[0] == "Polygon-SLM"
    
    @patch('hardware.backends.micromanager_backend.numba_utils')
    def test_polygon_update_mask_rectangle_list(self, mock_utils, mock_mmc, polygon_hardware_config, mock_calibration_data):
        """Test update_polygon_mask with rectangle list event."""
        mock_mask = np.ones((1080, 1920), dtype=np.uint8) * 128
        mock_utils.generate_pg_multi_rectangle_mask.return_value = mock_mask
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_calibration_data))):
            mock_mmc.getProperty.side_effect = lambda dev, prop: {
                ("ObjectiveTurret", "Label"): "20x",
                ("Camera", "Binning"): "1x1"
            }.get((dev, prop))
            mock_mmc.getCameraDevice.return_value = "Camera"
            
            stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)
            stimulus.configure_stimulus({
                "interface_type": "InvCore-LDI-Polygon-640",
                "calibration_path": "./test_calibrations.json"
            })
            
            stim_params = {
                'event': {
                    'event_type': 'pulse-rect-roi-list',
                    'stim_rect_roi_list': {
                        'x': [10, 20, 30],
                        'y': [40, 50, 60],
                        'width': [5, 5, 5],
                        'height': [5, 5, 5]
                    }
                },
                'roi': [0, 0]
            }
            
            stimulus.update_polygon_mask(stim_params)
            
            # Should have called multi-rectangle mask generation
            mock_utils.generate_pg_multi_rectangle_mask.assert_called_once()
            mock_mmc.setSLMImage.assert_called()
    
    def test_polygon_update_mask_full_field(self, mock_mmc, polygon_hardware_config, mock_calibration_data):
        """Test update_polygon_mask with full-field event."""
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_calibration_data))):
            mock_mmc.getProperty.side_effect = lambda dev, prop: {
                ("ObjectiveTurret", "Label"): "20x",
                ("Camera", "Binning"): "1x1"
            }.get((dev, prop))
            mock_mmc.getCameraDevice.return_value = "Camera"
            
            stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)
            stimulus.configure_stimulus({
                "interface_type": "InvCore-LDI-Polygon-640",
                "calibration_path": "./test_calibrations.json"
            })
            
            stim_params = {
                'event': {
                    'event_type': 'full-field-button'
                },
                'roi': [0, 0]
            }
            
            stimulus.update_polygon_mask(stim_params)
            
            # Should have set all pixels to 255
            mock_mmc.setSLMPixelsTo.assert_called_with("Polygon-SLM", 255)
    
    def test_polygon_update_mask_no_calibration(self, mock_mmc, polygon_hardware_config, mock_calibration_data):
        """Test update_polygon_mask fails gracefully without calibration."""
        mock_mmc.getProperty.side_effect = lambda dev, prop: {
            ("ObjectiveTurret", "Label"): "20x",
            ("Camera", "Binning"): "1x1"
        }.get((dev, prop))
        mock_mmc.getCameraDevice.return_value = "Camera"
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_calibration_data))):
            stimulus.configure_stimulus({"interface_type": "InvCore-LDI-Polygon-640"})

        # Force calibration to None to test graceful failure in update_polygon_mask
        stimulus.calibration_points = None
        
        stim_params = {
            'event': {
                'event_type': 'circle-click',
                'x': 200,
                'y': 300,
                'stim_diameter': 30
            },
            'roi': [0, 0]
        }
        
        # Should not raise - error should be logged
        stimulus.update_polygon_mask(stim_params)
        
        # Should not have called setSLMImage
        mock_mmc.setSLMImage.assert_not_called()


class TestPolygonStimulusController:
    """Test PolygonStimulusController with new architecture."""
    
    @pytest.fixture
    def mock_hardware_manager(self, mock_mmc, polygon_hardware_config, mock_calibration_data):
        """Create mock hardware manager with polygon stimulus."""
        from hardware.hardware_manager import HardwareManager
        
        hw_manager = Mock(spec=HardwareManager)
        
        # Create real stimulus interface
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_calibration_data))):
            mock_mmc.getProperty.side_effect = lambda dev, prop: {
                ("ObjectiveTurret", "Label"): "20x",
                ("Camera", "Binning"): "1x1"
            }.get((dev, prop))
            mock_mmc.getCameraDevice.return_value = "Camera"
            
            stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)
            stimulus.configure_stimulus({
                "interface_type": "InvCore-LDI-Polygon-640",
                "calibration_path": "./test_calibrations.json"
            })
        
        hw_manager.stimulus = stimulus
        return hw_manager
    
    @pytest.fixture
    def controller_config(self):
        """Create controller configuration."""
        import time
        return {
            "id": "test_polygon_rec",
            "roi": [0, 0],
            "saveroot": "/tmp/test",
            "t0": time.time(),
            # "gooey_args": {
            #     "zsize": 1,
            #     "stim_interface": "InvCore-LDI-Polygon-640",
            #     "acquisition_backend": "pycromanager",
            #     "trigger_algorithm": "Brainalyzer",
            #     "microscope_name": "test_polygon_scope"
            # }
        }
    
    def test_polygon_controller_initialization(self, mock_hardware_manager, controller_config):
        """Test PolygonStimulusController initializes correctly."""
        controller = PolygonStimulusController(mock_hardware_manager)

        assert controller.hardware_manager == mock_hardware_manager
        assert controller.calibration_points == {}
    
    # @patch('hardware.backends.micromanager_backend.numba_utils')
    # # @patch('utils.numba_utils')
    # def test_polygon_controller_spool(self, mock_utils, mock_hardware_manager, controller_config):
    #     """Test spool pre-compiles JIT functions."""
    #     mock_utils.generate_pg_ellipse_mask.return_value = np.zeros((1080, 1920), dtype=np.uint8)
    #     mock_utils.generate_pg_multi_rectangle_mask.return_value = np.zeros((1080, 1920), dtype=np.uint8)
        
    #     controller = PolygonStimulusController(mock_hardware_manager, controller_config)
        
    #     # Should not raise
    #     controller.spool()
        
    #     # Should have called mask generation functions for pre-compilation
    #     # assert mock_utils.generate_pg_ellipse_mask.call_count >= 1
    #     assert mock_utils.generate_pg_multi_rectangle_mask.call_count >= 1 # for brainalyzer
    
    def test_polygon_controller_spool_no_dimensions(self, controller_config):
        """Test spool handles missing polygon dimensions gracefully."""
        from hardware.hardware_manager import HardwareManager

        hw_manager = Mock(spec=HardwareManager)
        hw_manager.stimulus.get_polygon_dimensions.return_value = None

        controller = PolygonStimulusController(hw_manager)

        # Should not raise
        controller.spool()
    
    def test_polygon_controller_submit_pulsed_stim(self, mock_hardware_manager, controller_config):
        """Test submitting pulsed polygon stimulus."""
        controller = PolygonStimulusController(mock_hardware_manager)
        
        stim_params = {
            "stim_on": 100,
            "stim_off": 148,
            "event": {
                "stim_intensity": 50,
                "event_type": "circle-button",
                "x": 200,
                "y": 300,
                "stim_diameter": 30
            }
        }
        
        controller.submit_stim_params(stim_params, 50)
        
        assert 100 in controller.stim_on_list
        assert 148 in controller.stim_off_list
    
    def test_polygon_controller_get_metadata_includes_calibration(self, mock_hardware_manager, controller_config):
        """Test metadata includes calibration points."""
        controller = PolygonStimulusController(mock_hardware_manager)
        
        metadata = controller.get_metadata()
        
        assert "calibration_points" in metadata
        calib = metadata["calibration_points"]
        assert 'pcx' in calib
        assert 'pcy' in calib


class TestMicroManagerBackendProjectorSupport:
    """Test MicroManagerBackend projector initialization."""
    
    def _mock_pycromanager(self, mock_mmc):
        mock_pycro = MagicMock()
        mock_pycro.Core.return_value = mock_mmc
        mock_pycro.JavaObject = MagicMock(return_value=MagicMock())
        return mock_pycro

    def test_backend_creates_projector_for_polygon(self, mock_mmc, polygon_hardware_config):
        """Test backend creates projector interface when polygon configured."""
        import sys
        mock_pycro = self._mock_pycromanager(mock_mmc)
        with patch.dict(sys.modules, {'pycromanager': mock_pycro}):
            backend = MicroManagerBackend(polygon_hardware_config)
            backend.initialize()

            # Should have created projector interface
            assert backend._projector is not None
            assert isinstance(backend._projector, MicroManagerProjector)

    def test_backend_no_projector_for_non_polygon(self, mock_mmc):
        """Test backend doesn't create projector for non-polygon stimulus."""
        config = HardwareConfig(
            backend="pycromanager",
            stim_interface="InvCore-SpinningDisk-639",
            mm_config_path="test.cfg",
            stimulus_devices={
                "InvCore-SpinningDisk-639": StimulusDeviceConfig(
                    type="widefield_laser",
                    max_volts=3.5
                )
            }
        )

        import sys
        mock_pycro = self._mock_pycromanager(mock_mmc)
        with patch.dict(sys.modules, {'pycromanager': mock_pycro}):
            backend = MicroManagerBackend(config)
            backend.initialize()

            # Should not have created projector
            assert backend._projector is None
    
    # @patch('hardware.backends.micromanager_backend.MMSubroutines')
    # @patch('utils.MMSubroutines')
    # def test_backend_projector_property(self, mock_mm, mock_mmc, polygon_hardware_config):
    #     """Test backend projector property."""
    #     mock_mm.initialize_mmc.return_value = mock_mmc
        
    #     backend = MicroManagerBackend(polygon_hardware_config)
    #     backend.initialize()
        
    #     projector = backend.projector
    #     assert projector is not None
    #     assert hasattr(projector, 'get_dimensions')
    #     assert hasattr(projector, 'set_image')


class TestPolygonIntegrationWithHardwareManager:
    """Integration tests for polygon stimulus through HardwareManager."""
    
    def test_polygon_full_workflow(self, mock_mmc, polygon_hardware_config, mock_calibration_data):
        """Test complete polygon stimulus workflow through HardwareManager."""
        import sys
        from hardware.hardware_manager import HardwareManager
        from hardware.stimulus_controllers import create_stimulus_controller

        mock_pycro = MagicMock()
        mock_pycro.Core.return_value = mock_mmc
        mock_pycro.JavaObject = MagicMock(return_value=MagicMock())

        # Mock calibration loading
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_calibration_data))), \
             patch.dict(sys.modules, {'pycromanager': mock_pycro}):
            mock_mmc.getProperty.side_effect = lambda dev, prop: {
                ("ObjectiveTurret", "Label"): "20x",
                ("Camera", "Binning"): "1x1"
            }.get((dev, prop))
            mock_mmc.getCameraDevice.return_value = "Camera"

            # Initialize hardware
            hw_manager = HardwareManager(polygon_hardware_config)
            hw_manager.initialize()

            # Configure stimulus
            hw_manager.stimulus.configure_stimulus({
                "interface_type": "InvCore-LDI-Polygon-640",
                "calibration_path": "./test_calibrations.json"
            })

            # Verify polygon setup
            assert hw_manager.stimulus.get_polygon_dimensions() == (1920, 1080)
            assert hw_manager.stimulus.get_calibration_points() is not None

            controller = create_stimulus_controller(
                "InvCore-LDI-Polygon-640",
                hw_manager,
            )
            
            # Submit stimulus
            stim_params = {
                "stim_on": 100,
                "stim_off": 148,
                "event": {
                    "stim_intensity": 50,
                    "event_type": "circle-button",
                    "x": 200,
                    "y": 300,
                    "stim_diameter": 30
                }
            }
            
            controller.submit_stim_params(stim_params, 50)
            
            # Check activation
            controller.check_stim(100)
            assert hw_manager.stimulus.is_stimulus_active() is True
            
            # Check deactivation
            controller.check_stim(148)
            assert hw_manager.stimulus.is_stimulus_active() is False
            
            # Verify metadata
            metadata = controller.get_metadata()
            assert "stim_on_list" in metadata
            assert "calibration_points" in metadata


class TestPolygonConfigurationValidation:
    """Test polygon configuration validation and error handling."""
    
    def test_polygon_config_missing_intensity_device(self):
        """Test polygon config validation with missing required fields."""
        # This should still create config but warn during use
        config = StimulusDeviceConfig(
            type="polygon",
            # Missing intensity_device
            shutter_device="Shutter"
        )
        
        assert config.type == "polygon"
        assert config.intensity_device is None
    
    def test_polygon_config_from_yaml(self):
        """Test loading polygon config from YAML structure."""
        config_dict = {
            "type": "polygon",
            "intensity_device": "89 North Laser Diode Illuminator",
            "intensity_property": "640 Intensity",
            "shutter_device": "89 North Laser Diode Illuminator",
            "slm_device": None
        }
        
        config = StimulusDeviceConfig(**config_dict)
        
        assert config.type == "polygon"
        assert config.intensity_device == "89 North Laser Diode Illuminator"
        assert config.intensity_property == "640 Intensity"
        assert config.shutter_device == "89 North Laser Diode Illuminator"
        assert config.slm_device is None
    
    def test_hardware_config_polygon_calibration_path(self):
        """Test HardwareConfig with polygon_calibration_path."""
        config = HardwareConfig(
            backend="pycromanager",
            stim_interface="InvCore-LDI-Polygon-640",
            polygon_calibration_path="./calibrations.json"
        )
        
        assert config.polygon_calibration_path == "./calibrations.json"
    
    def test_polygon_config_extra_fields_allowed(self):
        """Test polygon config allows extra fields."""
        config_dict = {
            "type": "polygon",
            "intensity_device": "LDI",
            "custom_field": "custom_value",
            "another_field": 42
        }
        
        config = StimulusDeviceConfig(**config_dict)
        
        assert config.type == "polygon"
        # Extra fields should be preserved
        config_dump = config.model_dump()
        assert config_dump.get("custom_field") == "custom_value"
        assert config_dump.get("another_field") == 42


class TestPolygonErrorHandling:
    """Test error handling in polygon stimulus system."""
    
    def test_polygon_update_mask_unknown_event_type(self, mock_mmc, polygon_hardware_config, mock_calibration_data):
        """Test update_polygon_mask handles unknown event types gracefully."""
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_calibration_data))):
            mock_mmc.getProperty.side_effect = lambda dev, prop: {
                ("ObjectiveTurret", "Label"): "20x",
                ("Camera", "Binning"): "1x1"
            }.get((dev, prop))
            mock_mmc.getCameraDevice.return_value = "Camera"
            
            stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)
            stimulus.configure_stimulus({
                "interface_type": "InvCore-LDI-Polygon-640",
                "calibration_path": "./test_calibrations.json"
            })
            
            stim_params = {
                'event': {
                    'event_type': 'unknown-event-type',
                    'x': 200,
                    'y': 300
                },
                'roi': [0, 0]
            }
            
            # Should not raise - error should be logged
            stimulus.update_polygon_mask(stim_params)
            
            # Should not have uploaded mask
            mock_mmc.setSLMImage.assert_not_called()
    
    def test_polygon_update_mask_on_non_polygon_stimulus(self, mock_mmc):
        """Test update_polygon_mask on non-polygon stimulus does nothing."""
        config = HardwareConfig(
            backend="pycromanager",
            stim_interface="InvCore-SpinningDisk-639",
            mm_config_path="test.cfg",
            stimulus_devices={
                "InvCore-SpinningDisk-639": StimulusDeviceConfig(
                    type="widefield_laser",
                    max_volts=3.5
                )
            }
        )
        
        stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", config)
        stimulus.configure_stimulus({"interface_type": "InvCore-SpinningDisk-639"})
        
        stim_params = {
            'event': {
                'event_type': 'circle-click',
                'x': 200,
                'y': 300,
                'stim_diameter': 30
            },
            'roi': [0, 0]
        }
        
        # Should not raise - should log warning
        stimulus.update_polygon_mask(stim_params)
    
    # def test_polygon_activate_without_configure(self, mock_mmc, polygon_hardware_config):
    #     """Test activating polygon without configuration."""
    #     stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)
        
    #     # Activate without configuring
    #     stimulus.activate_stimulus({"intensity": 50})
        
    #     # Should still set active flag (warning logged)
    #     assert stimulus._active is True
    
    def test_polygon_calibration_load_malformed_json(self, mock_mmc, polygon_hardware_config):
        """Test loading malformed calibration file raises RuntimeError."""
        bad_json = "{invalid json"

        with patch('builtins.open', mock_open(read_data=bad_json)):
            stimulus = MicroManagerStimulus(mock_mmc, "pycromanager", polygon_hardware_config)

            config = {
                "interface_type": "InvCore-LDI-Polygon-640",
                "calibration_path": "./bad_calibrations.json"
            }

            with pytest.raises(RuntimeError, match="Polygon calibration could not be loaded"):
                stimulus.configure_stimulus(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])