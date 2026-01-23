"""
Unit tests for Brainalyzer algorithm.

Tests the refactored Brainalyzer class that uses Config objects
instead of legacy args dict.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config_manager import (
    HardwareConfig,
    ExperimentConfig,
    AlgorithmConfig,
    AcquisitionConfig,
    SubjectMetadata,
    AlgorithmParameters,
    StimulusParameters,
)
from hardware.hardware_manager import HardwareManager


class TestBrainalyzerInitialization:
    """Test Brainalyzer initialization with Config objects."""
    
    @pytest.fixture
    def algorithm_config(self):
        """Create test algorithm config."""
        return AlgorithmConfig(
            algorithm_type="Brainalyzer",
            enable_gui=False,  # No GUI for unit tests
            gui_mode="neural_imaging",
            algorithm_params=AlgorithmParameters(
                stimulus_diameter_pixels=30,
            ),
            stimulus_params=StimulusParameters(
                enabled=False,
            ),
        )
    
    @pytest.fixture
    def experiment_config(self):
        """Create test experiment config."""
        return ExperimentConfig(
            experiment_name="test_experiment",
            output_dir="./test_output",
            acquisition=AcquisitionConfig(
                num_samples=100,
                z_planes=10,
            ),
            subject=SubjectMetadata(
                genotype="test_strain",
            ),
        )
    
    @pytest.fixture
    def hardware_config(self):
        """Create test hardware config."""
        return HardwareConfig(
            backend="dummy",
            stim_interface="dummy",
            microscope_name="test_microscope",
        )
    
    @pytest.fixture
    def hardware_manager(self, hardware_config):
        """Create and initialize test hardware manager."""
        hw_manager = HardwareManager(hardware_config)
        hw_manager.initialize()
        yield hw_manager
        hw_manager.close()
    
    def test_init_with_configs(self, algorithm_config, experiment_config, hardware_manager):
        """Test initialization with Config objects."""
        from algorithms.brainalyzer.Brainalyzer import Brainalyzer
        
        # Create Brainalyzer
        alg = Brainalyzer(
            algorithm_config=algorithm_config,
            experiment_config=experiment_config,
            hardware_manager=hardware_manager,
        )
        
        # Verify core attributes set correctly
        assert alg.rec_id == "test_experiment"
        assert alg.frames_to_grab == 100
        assert alg.zsize == 10
        assert alg.GUI_mode == "neural_imaging"
        assert alg.microscope_name == "test_microscope"
        # assert alg.stim_intensity_ops == [10, 20, 30]
        # assert alg.stim_intensity == 10  # First option
    
    def test_init_without_hardware_manager(self, algorithm_config, experiment_config):
        """Test initialization without hardware manager (optional)."""
        from algorithms.brainalyzer.Brainalyzer import Brainalyzer
        
        alg = Brainalyzer(
            algorithm_config=algorithm_config,
            experiment_config=experiment_config,
        )
        
        # Should use safe defaults
        assert alg.microscope_name == "unknown"
        assert alg.hardware is None
    
    def test_set_roi(self, algorithm_config, experiment_config):
        """Test ROI setting after initialization."""
        from algorithms.brainalyzer.Brainalyzer import Brainalyzer
        
        alg = Brainalyzer(
            algorithm_config=algorithm_config,
            experiment_config=experiment_config,
        )
        
        # Initially has defaults
        assert alg.xsize == 200
        assert alg.ysize == 200
        
        # Set ROI
        roi = (10, 20, 512, 256)
        alg.set_roi(roi)
        
        # Verify updated
        assert alg.roi == roi
        assert alg.xsize == 512
        assert alg.ysize == 256
    
    # def test_backward_compatibility_wrapper(self):
    #     """Test legacy args wrapper for backward compatibility."""
    #     from algorithms.brainalyzer.Brainalyzer import create_brainalyzer_from_legacy_args
        
    #     # Create legacy args dict
    #     legacy_args = {
    #         "gooey_args": {
    #             "total_frames": 200,
    #             "zsize": 5,
    #             "GUI_mode": "behavior",
    #             "microscope_name": "legacy_scope",
    #             "trigger_algorithm": "Brainalyzer",
    #             "stim_intensity_options": [5, 10],
    #             "output_folder": "./legacy_output",
    #         },
    #         "roi": (0, 0, 256, 256),
    #     }
        
    #     # Create using wrapper
    #     alg = create_brainalyzer_from_legacy_args(legacy_args)
        
    #     # Verify attributes
    #     assert alg.frames_to_grab == 200
    #     assert alg.zsize == 5
    #     assert alg.GUI_mode == "behavior"
    #     assert alg.microscope_name == "legacy_scope"
    #     assert alg.roi == (0, 0, 256, 256)


class TestBrainalyzerProcessing:
    """Test Brainalyzer frame processing."""
    
    @pytest.fixture
    def hardware_manager(self):
        """Create and initialize test hardware manager."""
        hardware_config = HardwareConfig(
            backend="dummy",
            stim_interface="dummy",
            microscope_name="test_microscope",
        )
        hw_manager = HardwareManager(hardware_config)
        hw_manager.initialize()
        yield hw_manager
        hw_manager.close()
    
    @pytest.fixture
    def brainalyzer(self, hardware_manager):
        """Create Brainalyzer instance for testing."""
        from algorithms.brainalyzer.Brainalyzer import Brainalyzer
        
        algorithm_config = AlgorithmConfig(
            algorithm_type="Brainalyzer",
            enable_gui=False,
            gui_mode="neural_imaging",
            stimulus_params=StimulusParameters(
                enabled=False,
            ),
        )
        
        experiment_config = ExperimentConfig(
            experiment_name="test",
            output_dir="./test",
            acquisition=AcquisitionConfig(
                num_samples=100,
            ),
        )
        
        alg = Brainalyzer(
            algorithm_config=algorithm_config,
            experiment_config=experiment_config,
            hardware_manager=hardware_manager,
        )
        
        # Set ROI
        alg.set_roi((0, 0, 128, 128))
        
        # Mock shared memory components (since we're not running worker)
        alg.shared_ndarray_list = [
            np.zeros((128, 128), dtype=np.uint16) for _ in range(10)
        ]
        alg.shared_image_count = [0]
        alg.parent_conn = None  # No worker in unit tests
        
        return alg
    
    def test_process_sample(self, brainalyzer):
        """Test frame processing."""
        # Create test frame
        img = np.random.randint(0, 1000, (128, 128), dtype=np.uint16)
        
        # Process frame
        brainalyzer.process_sample(img, sample_ndx=0)
        
        # Verify frame stored in shared memory
        assert np.array_equal(brainalyzer.shared_ndarray_list[0], img)
        
        # Verify image count incremented
        assert brainalyzer.shared_image_count[0] == 1
    
    def test_check_stim_pulse_event(self, brainalyzer):
        """Test stimulus checking with pulse event."""
        # Set current event (simulating GUI trigger)
        brainalyzer.current_event = {
            "event_type": "pulse-rect-roi-list",
            "stim_intensity": 20,
            "stim_duration_vols": 2,
            "stim_rect_roi_list": {"x": [10], "y": [20], "width": [30], "height": [40]},
        }
        
        # Check stimulus at frame 15 (middle of volume 1, z=10 planes)
        stim_params, cooldown = brainalyzer.check_stim(image_ndx=15, cooldown_counter=0)
        
        # Should trigger at start of next volume (frame 20)
        assert stim_params["stim_on"] == 20
        assert stim_params["stim_intensity"] == 20
        assert stim_params["stim_off"] == 40  # 20 + (2 vols * 10 planes)
        assert stim_params["event"]["event_type"] == "pulse-rect-roi-list"
        
        # Current event should be cleared
        assert brainalyzer.current_event is None
        
        # Event should be recorded
        assert len(brainalyzer.stim_param_list) == 1
    
    def test_check_stim_stream_event(self, brainalyzer):
        """Test stimulus checking with continuous stream event."""
        # Set stream event
        brainalyzer.current_event = {
            "event_type": "stream-rect-roi-list",
            "stim_intensity": 15,
            "stim_rect_roi_list": {"x": [10], "y": [20], "width": [30], "height": [40]},
        }
        
        # Check stimulus
        stim_params, cooldown = brainalyzer.check_stim(image_ndx=15, cooldown_counter=0)
        
        # Should turn on stimulus without off time
        assert stim_params["stim_on"] == 20
        assert stim_params["stim_intensity"] == 15
        assert "stim_off" not in stim_params
        
        # Should set stimulus_is_on flag
        assert brainalyzer.stimulus_is_on is True
    
    def test_check_stim_stop_stream(self, brainalyzer):
        """Test stopping continuous stimulus."""
        # Set stimulus as currently on
        brainalyzer.stimulus_is_on = True
        brainalyzer.stim_param_list = [
            {"stim_on": 20, "stim_intensity": 15}
        ]
        
        # Send stop event
        brainalyzer.current_event = {
            "event_type": "stream-rect-roi-list",
            "stim_intensity": 15,
        }
        
        # Check stimulus at frame 45
        stim_params, cooldown = brainalyzer.check_stim(image_ndx=45, cooldown_counter=0)
        
        # Should set stim_off
        assert stim_params["stim_off"] == 50  # Next volume start
        
        # Should update previous stim param
        assert brainalyzer.stim_param_list[0]["stim_off"] == 50
        
        # Should turn off flag
        assert brainalyzer.stimulus_is_on is False


class TestBrainalyzerMetadata:
    """Test Brainalyzer metadata collection."""
    
    @pytest.fixture
    def hardware_manager(self):
        """Create and initialize test hardware manager."""
        hardware_config = HardwareConfig(
            backend="dummy",
            stim_interface="dummy",
            microscope_name="test_scope",
        )
        hw_manager = HardwareManager(hardware_config)
        hw_manager.initialize()
        yield hw_manager
        hw_manager.close()
    
    def test_get_metadata(self, hardware_manager):
        """Test metadata collection."""
        from algorithms.brainalyzer.Brainalyzer import Brainalyzer
        
        algorithm_config = AlgorithmConfig(
            algorithm_type="Brainalyzer",
            gui_mode="neural_imaging",
            stimulus_params=StimulusParameters(
                enabled=False
            ),
        )
        
        experiment_config = ExperimentConfig(
            experiment_name="metadata_test",
            output_dir="./test",
            acquisition=AcquisitionConfig(num_samples=100, z_planes=10),
        )
        
        alg = Brainalyzer(
            algorithm_config=algorithm_config,
            experiment_config=experiment_config,
            hardware_manager=hardware_manager,
        )
        
        # Add some stimulus events
        alg.stim_param_list = [
            {"stim_on": 20, "stim_off": 40, "stim_intensity": 10},
            {"stim_on": 60, "stim_off": 80, "stim_intensity": 20},
        ]
        
        # Get metadata
        metadata = alg.get_metadata()
        
        # Verify contents
        assert "stim_param_list" in metadata
        assert len(metadata["stim_param_list"]) == 2
        assert "algorithm_config" in metadata
        assert "experiment_config" in metadata
        assert "hardware_config" in metadata


class TestBrainalyzerBehaviorMode:
    """Test Brainalyzer behavior mode features."""
    
    def test_behavior_mode_initialization(self):
        """Test initialization in behavior mode."""
        from algorithms.brainalyzer.Brainalyzer import Brainalyzer
        
        algorithm_config = AlgorithmConfig(
            algorithm_type="Brainalyzer",
            gui_mode="behavior",  # Behavior mode
            stimulus_params=StimulusParameters(
                enabled=False,
            ),
        )
        
        experiment_config = ExperimentConfig(
            experiment_name="behavior_test",
            output_dir="./test",
            acquisition=AcquisitionConfig(num_samples=100, z_planes=1),
        )
        
        hardware_config = HardwareConfig(
            backend="dummy",
            microscope_name="innovation core thunderscope",  # Specific scope
        )
        
        # Create hardware manager
        hardware_manager = HardwareManager(hardware_config)
        hardware_manager.initialize()
        
        # Mock MMC
        mock_mmc = Mock()
        local_handles = {"mmc": mock_mmc}
        
        try:
            # Create with behavior mode
            alg = Brainalyzer(
                algorithm_config=algorithm_config,
                experiment_config=experiment_config,
                hardware_manager=hardware_manager,
                local_handles=local_handles,
            )
            
            # Verify behavior mode setup
            assert alg.GUI_mode == "behavior"
            assert hasattr(alg, 'shared_stage_offset_xy')
            assert hasattr(alg, 'xy_stage_position_list')
        finally:
            hardware_manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])