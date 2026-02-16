"""
Comprehensive test suite for ClosedLoopEngine class.

Tests are organized into 8 categories:
1. Configuration Object Creation and Conversion
2. Initialization Tests
3. Acquisition Loop Tests
4. Metadata Tests
5. Cleanup Tests
6. Integration Tests
7. Error Handling Tests
8. Backward Compatibility Tests
"""

import pytest
import os
import time
import tempfile
import shutil
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

# Import the engine and supporting modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from engine.closed_loop_engine import (
    ClosedLoopEngine,
    # convert_gooey_args_to_configs,
    # launch_wblive_from_gooey,
    create_test_config,
)
from config.config_manager import (
    ConfigManager,
    HardwareConfig,
    ExperimentConfig,
    AlgorithmConfig,
    AcquisitionConfig,
    SubjectMetadata,
    AlgorithmParameters,
    StimulusParameters,
)
from algorithms.dummy import DummyAlg

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def minimal_configs(temp_output_dir):
    """Provide minimal valid Config objects for testing."""
    
    from config.config_manager import (
        HardwareConfig,
        ExperimentConfig,
        AlgorithmConfig,
        AcquisitionConfig,
        SubjectMetadata,
        SubjectDetails,
        TreatmentDetails,
        BackendConfiguration,
        StimulusConfiguration,
        SystemProperties,
        AlgorithmConfiguration,
        StimulusParameters,
    )
    
    # Hardware config
    backend_config = BackendConfiguration(
        backend_name="dummy"
    )
    
    stim_config = StimulusConfiguration(
        stim_interface="dummy"
    )
    
    system_props = SystemProperties(
        system_name="test"
    )
    
    hardware_config = HardwareConfig(
        backend_configuration=backend_config,
        stimulus_configuration=stim_config,
        system_properties=system_props,
    )
    
    # Experiment config
    acquisition_config = AcquisitionConfig(
        num_samples=100,
    )
    
    treatment_details = TreatmentDetails()
    
    subject_details = SubjectDetails(
        treatment_details=treatment_details
    )
    
    subject_metadata = SubjectMetadata(
        subject_id="test_subject",
        subject_type="test_strain",
        subject_details=subject_details,
        notes="Test run with dummy objects",
    )
    
    experiment_config = ExperimentConfig(
        experiment_name="test_experiment",
        output_dir=temp_output_dir,
        save_images=False,
        save_metadata=False,
        acquisition=acquisition_config,
        subject=subject_metadata,
    )
    
    # Algorithm config
    stimulus_params = StimulusParameters(
        enabled=False,
    )
    
    algorithm_configuration = AlgorithmConfiguration(
        enable_gui=False,
        gui_mode='none',
        stimulus_params=stimulus_params,
    )
    
    algorithm_config = AlgorithmConfig(
        algorithm_type="dummy",
        save_algorithm_plot=False,
        algorithm_configuration=algorithm_configuration,
    )
    
    return {
        "hardware": hardware_config,
        "experiment": experiment_config,
        "algorithm": algorithm_config,
    }

@pytest.fixture
def dummy_tiff_file(temp_output_dir):
    """Create a dummy TIFF file for testing with file input."""
    import tifffile as tf
    
    # Create small test dataset
    test_data = np.random.randint(0, 65536, size=(10, 50, 50), dtype=np.uint16)
    filepath = os.path.join(temp_output_dir, "test_recording.tiff")
    tf.imwrite(filepath, test_data)
    
    return filepath


@pytest.fixture
def engine_with_configs(minimal_configs):
    """Provide an engine instance with minimal Config objects."""
    return ClosedLoopEngine(
        hardware_config=minimal_configs["hardware"],
        experiment_config=minimal_configs["experiment"],
        algorithm_config=minimal_configs["algorithm"]
    )


# ============================================================================
# 1. Configuration Object Creation and Conversion Tests
# ============================================================================

class TestConfigObjectCreation:
    """Test creating Config objects from different sources."""
    
    def test_create_from_yaml(self, temp_output_dir):
        """Test loading Config objects from YAML files."""
        manager = ConfigManager()
        
        # Load all configs from defaults
        manager.load_all_configs()
        
        assert manager.hardware_config is not None
        assert manager.experiment_config is not None
        assert manager.algorithm_config is not None
        
        # Validate loaded configs
        assert manager.hardware_config.backend == "dummy"
        assert manager.experiment_config.acquisition.num_samples == 100
        assert manager.algorithm_config.algorithm_type == "dummy"
    
    def test_create_programmatically(self, minimal_configs):
        """Test creating Config objects programmatically."""
        assert minimal_configs["hardware"].backend == "dummy"
        assert minimal_configs["experiment"].acquisition.num_samples == 100
        assert minimal_configs["algorithm"].algorithm_type == "dummy"
    
    # def test_convert_from_legacy_gooey_args(self, legacy_gooey_args):
    #     """Test converting legacy gooey_args to Config objects."""
    #     configs = convert_gooey_args_to_configs(legacy_gooey_args)
        
    #     assert isinstance(configs["hardware"], HardwareConfig)
    #     assert isinstance(configs["experiment"], ExperimentConfig)
    #     assert isinstance(configs["algorithm"], AlgorithmConfig)
        
    #     # Verify conversion accuracy
    #     assert configs["hardware"].backend == "dummy"
    #     assert configs["experiment"].acquisition.num_samples == 100
    #     assert configs["algorithm"].algorithm_type == "dummy"
    
    def test_create_test_config_generates_valid_configs(self):
        """Test that create_test_config helper generates valid Config objects."""
        configs = create_test_config()
        
        # Should return dict with three config objects
        assert "hardware" in configs
        assert "experiment" in configs
        assert "algorithm" in configs
        
        # Should have sensible defaults
        assert configs["experiment"].acquisition.num_samples == 100
        assert configs["algorithm"].algorithm_type == "dummy"
        assert configs["hardware"].stim_interface == "dummy"


# ============================================================================
# 2. Initialization Tests
# ============================================================================

class TestInitialization:
    """Test suite for ClosedLoopEngine initialization."""
    
    def test_engine_accepts_config_objects(self, minimal_configs):
        """Test that engine accepts and stores Config objects."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_configs["hardware"],
            experiment_config=minimal_configs["experiment"],
            algorithm_config=minimal_configs["algorithm"]
        )
        
        assert engine.hardware_config == minimal_configs["hardware"]
        assert engine.experiment_config == minimal_configs["experiment"]
        assert engine.algorithm_config == minimal_configs["algorithm"]
    
    # def test_legacy_args_structure_built(self, minimal_configs):
    #     """Test that legacy args dictionary is built for backward compatibility."""
    #     engine = ClosedLoopEngine(
    #         hardware_config=minimal_configs["hardware"],
    #         experiment_config=minimal_configs["experiment"],
    #         algorithm_config=minimal_configs["algorithm"]
    #     )
        
    #     # Legacy args dict should exist
    #     assert "gooey_args" in engine.args
        
    #     # Should contain key fields from configs
    #     gooey_args = engine.args["gooey_args"]
    #     assert gooey_args["acquisition_backend"] == "dummy"
    #     assert gooey_args["total_frames"] == 100
    #     assert gooey_args["zsize"] == 10
    
    def test_initial_state(self, engine_with_configs):
        """Test that engine starts with correct initial state."""
        engine = engine_with_configs
        
        assert engine.is_running is False
        assert engine.sample_count == 0
        assert engine.sample_count == 0
        assert engine.cooldown_counter == 0
        # assert engine.mmc is None
        assert engine.alg is None
        assert engine.stim_controller is None
    
    def test_algorithm_factory_creates_dummy_alg(self, engine_with_configs):
        """Test that algorithm factory creates DummyAlg for dummy config."""
        engine = engine_with_configs
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        
        assert engine.alg is not None
        assert isinstance(engine.alg, DummyAlg)
    
    def test_stimulus_initialization_creates_interface(self, engine_with_configs):
        """Test that stimulus initialization creates interface."""
        engine = engine_with_configs
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_stimulus()
        
        assert engine.stim_controller is not None
    
    def test_config_field_access_patterns(self, minimal_configs):
        """Test that Config fields are accessible in expected patterns."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_configs["hardware"],
            experiment_config=minimal_configs["experiment"],
            algorithm_config=minimal_configs["algorithm"]
        )
        
        # Direct config access (new way)
        assert engine.hardware_config.backend == "dummy"
        assert engine.experiment_config.acquisition.num_samples == 100
        assert engine.algorithm_config.algorithm_type == "dummy"
        
        # # Legacy args access still works
        # assert engine.args["gooey_args"]["acquisition_backend"] == "dummy"
        # assert engine.args["gooey_args"]["total_frames"] == 100

    def test_hardware_initialization_creates_hardware_manager(self, engine_with_configs):
        """Test that hardware initialization creates HardwareManager."""
        engine = engine_with_configs
        engine.initialize_hardware()
        
        # Should create HardwareManager, not direct MMC
        assert engine.hardware is not None
        from hardware.hardware_manager import HardwareManager
        assert isinstance(engine.hardware, HardwareManager)
        
        # HardwareManager should be initialized
        assert engine.hardware.is_initialized
        
        # Legacy mmc should still be available for backward compatibility
        assert engine.mmc is not None
    
    def test_roi_setup_from_hardware_manager(self, engine_with_configs):
        """Test that ROI is properly retrieved from HardwareManager."""
        engine = engine_with_configs
        engine.initialize_hardware()
        
        # ROI should be retrieved through camera interface
        assert engine.roi is not None
        assert engine.data_interface.xsize == engine.roi[2]
        assert engine.data_interface.ysize == engine.roi[3]
        
        # ROI should match what camera interface returns
        camera_roi = engine.hardware.camera.get_roi()
        assert engine.roi == camera_roi


# ============================================================================
# 3. Acquisition Loop Tests
# ============================================================================

class TestAcquisitionLoop:
    """Test suite for acquisition loop functionality."""
    
    def test_prepare_acquisition_creates_directories(self, engine_with_configs):
        """Test that prepare_acquisition creates output directories."""
        engine = engine_with_configs
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        assert engine.savedir is not None
        assert os.path.exists(engine.savedir)
        assert engine.saveroot is not None
        assert engine.session_id is not None
    
    def test_prepare_acquisition_initializes_frame_storage(self, engine_with_configs):
        """Test that frame storage is initialized with correct dimensions."""
        engine = engine_with_configs
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        assert engine.samples is not None
        expected_frames = engine.experiment_config.acquisition.num_samples
        assert engine.samples.shape == (expected_frames, engine.data_interface.ysize, engine.data_interface.xsize)
        assert engine.samples.dtype == np.uint16
    
    def test_acquisition_loop_captures_frames(self, minimal_configs):
        """Test that acquisition loop captures the expected number of frames."""
        # Create config with small frame count
        configs = minimal_configs.copy()
        configs["experiment"].acquisition.num_samples = 20
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Run acquisition
        engine.run_acquisition_loop()
        
        assert engine.sample_count == 20
    
    def test_acquisition_loop_tracks_frame_times(self, minimal_configs):
        """Test that frame timestamps are recorded."""
        configs = minimal_configs.copy()
        configs["experiment"].acquisition.num_samples = 20
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        engine.run_acquisition_loop()
        
        assert len(engine.data_interface.sample_time_list) == 20
        assert all(isinstance(t, (float, np.float64)) for t in engine.data_interface.sample_time_list)
    
    def test_z_stack_indexing(self, minimal_configs):
        """Test that z-stack indexing cycles correctly."""
        configs = minimal_configs.copy()
        configs["experiment"].acquisition.num_samples = 25
        configs["experiment"].acquisition.z_planes = 5
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Mock the algorithm to track z indices
        z_indices_seen = []
        original_process_sample = engine.alg.process_sample
        
        def track_z_index(frame, sample_ndx):
            zndx = sample_ndx % configs['experiment'].acquisition.z_planes
            z_indices_seen.append(zndx)
            return original_process_sample(frame, zndx)
        
        engine.alg.process_sample = track_z_index
        
        engine.run_acquisition_loop()
        
        # Check that z indices cycle from 0 to z_planes-1
        expected_pattern = [i % 5 for i in range(25)]
        assert z_indices_seen == expected_pattern
    
    def test_stimulus_triggering_flow(self, minimal_configs):
        """Test that stimulus checking and submission occurs each frame."""
        configs = minimal_configs.copy()
        configs["experiment"].acquisition.num_samples = 10
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()

        # Test generate stimulus events
        def generate_stim(image_ndx, cooldown_counter):
            return {'test_stim': 1}, 0
        
        # Track stimulus submissions
        stim_submissions = []
        original_submit = engine.stim_controller.submit_stim_params
        
        def track_stim(stim_params, image_ndx):
            stim_submissions.append((stim_params, image_ndx))
            return original_submit(stim_params, image_ndx)
        
        # patch functions
        engine.alg.check_stim = generate_stim
        engine.stim_controller.submit_stim_params = track_stim
        
        engine.run_acquisition_loop()
        
        # Should have one submission per frame
        assert len(stim_submissions) == 10
    
    def test_cooldown_counter_decrements(self, minimal_configs):
        """Test that cooldown counter properly decrements."""
        configs = minimal_configs.copy()
        configs["experiment"].acquisition.num_samples = 10
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Set initial cooldown
        engine.cooldown_counter = 5
        initial_cooldown = engine.cooldown_counter
        
        # Mock to trigger after a few frames
        sample_count = [0]
        def mock_check_stim(image_ndx, cooldown):
            nonlocal sample_count
            sample_count[0] += 1
            if sample_count[0] >= initial_cooldown:
                return {}, 0  # Reset cooldown after it expires
            return {}, cooldown - 1 if cooldown > 0 else 0
        
        engine.alg.check_stim = mock_check_stim
        
        engine.run_acquisition_loop()
        
        # Cooldown should have decremented to zero
        assert engine.cooldown_counter == 0

    def test_acquisition_uses_camera_interface(self, minimal_configs):
        """Test that acquisition loop uses camera interface from HardwareManager."""
        configs = minimal_configs.copy()
        configs["experiment"].acquisition.num_samples = 20
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Track camera interface calls
        camera = engine.hardware.camera
        original_pop = camera.pop_next_image
        call_count = [0]
        
        def track_pop():
            call_count[0] += 1
            return original_pop()
        
        camera.pop_next_image = track_pop
        
        # Run acquisition
        engine.run_acquisition_loop()
        
        # Should have called camera interface 20 times
        assert call_count[0] == 20


# ============================================================================
# 4. Metadata Tests
# ============================================================================

class TestMetadata:
    """Test suite for metadata collection and saving."""
    
    def test_metadata_includes_config_objects(self, minimal_configs):
        """Test that metadata includes Config objects."""
        configs = minimal_configs.copy()
        configs["experiment"].save_metadata = True
        configs["experiment"].acquisition.num_samples = 10
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        engine.run_acquisition_loop()
        
        with patch('utils.wbliveUtils.save_metadata') as mock_save:
            metadata = engine.save_metadata()
            
            # Verify Config objects are in metadata
            assert "hardware_config" in metadata
            assert "experiment_config" in metadata
            assert "algorithm_config" in metadata
            
            # Should be dicts (from model_dump())
            assert isinstance(metadata["hardware_config"], dict)
            assert isinstance(metadata["experiment_config"], dict)
            assert isinstance(metadata["algorithm_config"], dict)
    
    def test_metadata_includes_timing_info(self, minimal_configs):
        """Test that metadata includes timing information."""
        configs = minimal_configs.copy()
        configs["experiment"].save_metadata = True
        configs["experiment"].acquisition.num_samples = 10
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        engine.run_acquisition_loop()
        
        with patch('utils.wbliveUtils.save_metadata') as mock_save:
            metadata = engine.save_metadata()
            
            assert metadata["t0"] is not None
            assert len(metadata["sample_time_list"]) == 10
    
    def test_algorithm_metadata_included(self, minimal_configs):
        """Test that algorithm metadata is collected."""
        configs = minimal_configs.copy()
        configs["experiment"].save_metadata = True
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        with patch('utils.wbliveUtils.save_metadata') as mock_save:
            metadata = engine.save_metadata()
            
            assert "alg_metadata" in metadata
            assert metadata["alg_metadata"]["is_dummy_alg"] is True
    
    def test_stimulus_metadata_included(self, minimal_configs):
        """Test that stimulus metadata is collected."""
        configs = minimal_configs.copy()
        configs["experiment"].save_metadata = True
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        with patch('utils.wbliveUtils.save_metadata') as mock_save:
            metadata = engine.save_metadata()
            
            assert "stim_metadata" in metadata

    def test_metadata_includes_hardware_manager_data(self, minimal_configs):
        """Test that metadata includes data from HardwareManager."""
        configs = minimal_configs.copy()
        configs["experiment"].save_metadata = True
        configs["experiment"].acquisition.num_samples = 10
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        engine.run_acquisition_loop()
        
        from unittest.mock import patch
        with patch('utils.wbliveUtils.save_metadata') as mock_save:
            metadata = engine.save_metadata()
            
            # Verify hardware metadata is collected through HardwareManager
            assert "hardware_metadata" in metadata
            assert isinstance(metadata["hardware_metadata"], dict)

# ============================================================================
# 5. Cleanup Tests
# ============================================================================

class TestCleanup:
    """Test suite for resource cleanup and management."""
    
    def test_cleanup_closes_hardware(self, engine_with_configs):
        """Test that cleanup stops the acquisition."""
        engine = engine_with_configs
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        # Mock the close method
        engine.hardware.close = Mock()

        engine.cleanup()

        engine.hardware.close.assert_called_once()
    
    def test_cleanup_closes_algorithm(self, engine_with_configs):
        """Test that cleanup closes the algorithm."""
        engine = engine_with_configs
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        
        # Mock the algorithm close method
        engine.alg.close = Mock()
        
        engine.cleanup()
        
        engine.alg.close.assert_called_once()
    
    def test_cleanup_closes_stimulus(self, engine_with_configs):
        """Test that cleanup closes stimulus interface."""
        engine = engine_with_configs
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_stimulus()
        
        # Mock the stimulus close method
        engine.stim_controller.close = Mock()
        
        engine.cleanup()
        
        engine.stim_controller.close.assert_called_once()
    
    def test_cleanup_handles_mmc_close_errors(self, engine_with_configs):
        """Test that cleanup gracefully handles MMC errors."""
        engine = engine_with_configs
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        # Make stopSequenceAcquisition raise an error
        engine.mmc.stopSequenceAcquisition = Mock(side_effect=Exception("MMC error"))
        
        # Should not raise - cleanup should handle gracefully
        try:
            engine.cleanup()
        except Exception as e:
            pytest.fail(f"Cleanup should handle errors gracefully, but raised: {e}")
    
    def test_cleanup_sends_sms_notification(self, minimal_configs):
        """Test that cleanup sends SMS notification when enabled."""
        configs = minimal_configs.copy()
        configs["experiment"].dev_options = {"send_sms_on_completion": True}
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        with patch('utils.wbliveUtils.notify') as mock_notify:
            engine.cleanup()
            
            mock_notify.assert_called_once()
            call_args = mock_notify.call_args
            assert "completed" in call_args[0][0].lower()
    
    def test_cleanup_skips_sms_when_disabled(self, engine_with_configs):
        """Test that cleanup skips SMS when disabled."""
        engine = engine_with_configs
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        with patch('utils.wbliveUtils.notify') as mock_notify:
            engine.cleanup()
            
            mock_notify.assert_not_called()

    def test_cleanup_closes_hardware_manager(self, engine_with_configs):
        """Test that cleanup properly closes HardwareManager."""
        engine = engine_with_configs
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        # Mock hardware manager close
        from unittest.mock import Mock
        engine.hardware.close = Mock()
        
        engine.cleanup()
        
        # Should call hardware.close() instead of direct MMC close
        engine.hardware.close.assert_called_once()
    
    def test_cleanup_handles_hardware_close_errors(self, engine_with_configs):
        """Test that cleanup gracefully handles HardwareManager errors."""
        engine = engine_with_configs
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        # Make hardware close raise an error
        from unittest.mock import Mock
        engine.hardware.close = Mock(side_effect=Exception("Hardware error"))
        
        # Should not raise - cleanup should handle gracefully
        try:
            engine.cleanup()
        except Exception as e:
            pytest.fail(f"Cleanup should handle errors gracefully, but raised: {e}")

# ============================================================================
# 6. Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests with dummy backend."""
    
    def test_full_workflow_with_hardware_manager(self, minimal_configs):
        """Test complete workflow using HardwareManager."""
        configs = minimal_configs.copy()
        configs["experiment"].acquisition.num_samples = 20
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        
        # Should complete without errors
        engine.run()
        
        # Verify final state
        assert engine.sample_count == 20
        assert os.path.exists(engine.savedir)
        
        # Verify hardware was properly closed
        assert not engine.hardware.is_initialized
    
    def test_acquisition_with_tiff_input(self, minimal_configs, dummy_tiff_file):
        """Test acquisition with TIFF file input."""
        configs = minimal_configs.copy()
        configs["experiment"].input_recording_path = dummy_tiff_file
        configs["experiment"].acquisition.num_samples = 10
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware() # needs to load virtual hardware settings after initialization
        engine.prepare_acquisition() # needs to overwrite default acquitision initialization 
        engine.run()
        
        assert engine.sample_count == 10
    
    def test_acquisition_with_z_stacks(self, minimal_configs):
        """Test acquisition with multiple z-planes."""
        configs = minimal_configs.copy()
        configs["experiment"].acquisition.num_samples = 30
        configs["experiment"].acquisition.z_planes = 10
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.run()
        
        # Should complete 3 full volumes
        assert engine.sample_count == 30
    
    def test_multiple_acquisitions_with_same_instance(self, minimal_configs):
        """Test that engine can be reused for multiple acquisitions."""
        configs = minimal_configs.copy()
        configs["experiment"].acquisition.num_samples = 10
        
        # First acquisition
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        engine.run_acquisition_loop()
        engine.cleanup()
        first_session_id = engine.session_id
        
        # Wait 1s for new session id
        time.sleep(1)
        
        # Reset for second acquisition
        engine.is_running = False
        engine.sample_count = 0
        engine.initialize_hardware() # reset hardware
        engine.prepare_acquisition()

        
        # Second acquisition
        engine.initialize_hardware()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        engine.run_acquisition_loop()
        engine.cleanup()
        
        second_session_id = engine.session_id
        
        # Session IDs should be different
        assert first_session_id != second_session_id

    def test_backend_switching(self, minimal_configs):
        """Test that backend can be switched via config."""
        # Test dummy backend
        configs_dummy = minimal_configs.copy()
        configs_dummy["hardware"].backend = "dummy"
        
        engine_dummy = ClosedLoopEngine(
            hardware_config=configs_dummy["hardware"],
            experiment_config=configs_dummy["experiment"],
            algorithm_config=configs_dummy["algorithm"]
        )
        engine_dummy.initialize_hardware()
        
        from hardware.backends.dummy_backend import DummyHardwareBackend
        assert isinstance(engine_dummy.hardware._backend, DummyHardwareBackend)
        
        engine_dummy.cleanup()


# ============================================================================
# 7. Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_unknown_algorithm_name(self, minimal_configs):
        """Test handling of unknown algorithm name."""
        configs = minimal_configs.copy()
        configs["algorithm"].algorithm_type = "NonexistentAlgorithm"
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        # Should raise exception
        with pytest.raises(Exception):
            engine.initialize_algorithm()
    
    def test_acquisition_interrupted_mid_loop(self, minimal_configs):
        """Test that interrupting acquisition is handled gracefully."""
        configs = minimal_configs.copy()
        configs["experiment"].acquisition.num_samples = 100
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Mock to interrupt after 10 frames
        original_process = engine.alg.process_sample
        call_count = [0]
        
        def interrupt_after_10(frame, zndx):
            call_count[0] += 1
            if call_count[0] > 10:
                raise KeyboardInterrupt("User interrupted")
            return original_process(frame, zndx)
        
        engine.alg.process_sample = interrupt_after_10
        
        # Should handle interrupt gracefully
        with pytest.raises(KeyboardInterrupt):
            engine.run_acquisition_loop()
    
    
    def test_algorithm_process_sample_error(self, minimal_configs):
        """Test handling of algorithm errors during frame processing."""
        configs = minimal_configs.copy()
        configs["experiment"].acquisition.num_samples = 10
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Make algorithm raise error
        engine.alg.process_sample = Mock(side_effect=Exception("Algorithm error"))
        
        # Should propagate error from acquisition loop
        with pytest.raises(Exception):
            engine.run_acquisition_loop()
    
    def test_cleanup_called_on_exception(self, minimal_configs):
        """Test that cleanup is called even when exception occurs."""
        configs = minimal_configs.copy()
        configs["experiment"].acquisition.num_samples = 10
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        
        # Mock run_acquisition_loop to raise error
        with patch.object(engine, 'run_acquisition_loop', side_effect=Exception("Test error")):
            with patch.object(engine, 'cleanup') as mock_cleanup:
                with pytest.raises(Exception):
                    engine.run()
                
                # Cleanup should still be called
                mock_cleanup.assert_called_once()
    
    def test_invalid_config_validation(self):
        """Test that invalid Config objects are caught by Pydantic."""
        # Test invalid backend
        with pytest.raises(Exception):  # ValidationError
            HardwareConfig(backend="invalid_backend")
        
        # Test negative num_samples
        with pytest.raises(Exception):  # ValidationError
            AcquisitionConfig(num_samples=-10)
        



# ============================================================================
# 8. Backward Compatibility Tests
# ============================================================================

# class TestBackwardCompatibility:
#     """Test backward compatibility with legacy gooey_args interface."""
    
    # def test_gooey_args_conversion(self, legacy_gooey_args):
    #     """Test that gooey_args can be converted and used."""
    #     configs = convert_gooey_args_to_configs(legacy_gooey_args)
        
    #     engine = ClosedLoopEngine(
    #         hardware_config=configs["hardware"],
    #         experiment_config=configs["experiment"],
    #         algorithm_config=configs["algorithm"]
    #     )
        
    #     # Verify engine is properly initialized
    #     assert engine.hardware_config.backend == "dummy"
    #     assert engine.experiment_config.acquisition.num_samples == 100
    
    # def test_launch_from_gooey_wrapper(self, legacy_gooey_args, monkeypatch):
    #     """Test the legacy launch_wblive_from_gooey wrapper."""
    #     # Mock sys.exit to prevent test from exiting
    #     def mock_exit(code=0):
    #         pass
        
    #     monkeypatch.setattr("sys.exit", mock_exit)
        
    #     # This should convert gooey_args and run without errors
    #     # (will fail at hardware init with dummy objects, which is expected)
    #     try:
    #         launch_wblive_from_gooey(legacy_gooey_args)
    #     except Exception as e:
    #         # Expected to fail at some point with dummy hardware
    #         # The important part is that conversion succeeded
    #         assert "gooey_args" not in str(e) or "Config" not in str(e)
    
    # def test_legacy_args_accessible_in_components(self, minimal_configs):
    #     """Test that legacy args dict is accessible for unrefactored components."""
    #     engine = ClosedLoopEngine(
    #         hardware_config=minimal_configs["hardware"],
    #         experiment_config=minimal_configs["experiment"],
    #         algorithm_config=minimal_configs["algorithm"]
    #     )
        
    #     # Legacy components expect args["gooey_args"]
    #     assert "gooey_args" in engine.args
        
    #     # Should contain all necessary fields
    #     gooey_args = engine.args["gooey_args"]
    #     assert "acquisition_backend" in gooey_args
    #     assert "total_frames" in gooey_args
    #     assert "trigger_algorithm" in gooey_args
    #     assert "stim_interface" in gooey_args
    
    # def test_field_mapping_accuracy(self, legacy_gooey_args):
    #     """Test that all fields are correctly mapped in conversion."""
    #     configs = convert_gooey_args_to_configs(legacy_gooey_args)
        
    #     # Hardware mappings
    #     assert configs["hardware"].backend == legacy_gooey_args["acquisition_backend"]
    #     assert configs["hardware"].stim_interface == legacy_gooey_args["stim_interface"]
        
    #     # Experiment mappings
    #     assert configs["experiment"].output_dir == legacy_gooey_args["output_folder"]
    #     assert configs["experiment"].acquisition.num_samples == legacy_gooey_args["total_frames"]
    #     assert configs["experiment"].acquisition.z_planes == legacy_gooey_args["zsize"]
        
    #     # Algorithm mappings
    #     assert configs["algorithm"].algorithm_type == legacy_gooey_args["trigger_algorithm"]
    #     assert configs["algorithm"].gui_mode == legacy_gooey_args["GUI_mode"]
    
    # def test_boolean_inversions_handled(self, legacy_gooey_args):
    #     """Test that boolean inversions (no_save_X → save_X) are handled correctly."""
    #     # Set to NOT save
    #     legacy_gooey_args["no_save_images"] = True
    #     legacy_gooey_args["no_save_metadata"] = True
        
    #     configs = convert_gooey_args_to_configs(legacy_gooey_args)
        
    #     # Should be inverted in new config
    #     assert configs["experiment"].save_images is False
    #     assert configs["experiment"].save_metadata is False
        
    #     # Test opposite
    #     legacy_gooey_args["no_save_images"] = False
    #     legacy_gooey_args["no_save_metadata"] = False
        
    #     configs = convert_gooey_args_to_configs(legacy_gooey_args)
        
    #     assert configs["experiment"].save_images is True
    #     assert configs["experiment"].save_metadata is True
    
    # def test_microscope_name_temporary_field(self, legacy_gooey_args):
    #     """Test that microscope_name is included but marked as temporary."""
    #     configs = convert_gooey_args_to_configs(legacy_gooey_args)
        
    #     # Should be present in hardware config
    #     assert configs["hardware"].microscope_name == "test"
        
    #     # But backend should be the primary field used
    #     assert configs["hardware"].backend == "dummy"


# ============================================================================
# 9. Config Access Pattern Tests
# ============================================================================

class TestConfigAccessPatterns:
    """Test that Config objects properly replace args dict access."""
    
    def test_hardware_config_replaces_args_access(self, minimal_configs):
        """Test accessing hardware config fields instead of args dict."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_configs["hardware"],
            experiment_config=minimal_configs["experiment"],
            algorithm_config=minimal_configs["algorithm"]
        )
        
        # New way: direct config access
        assert engine.hardware_config.backend == "dummy"
        assert engine.hardware_config.stim_interface == "dummy"
        
        # Old way still works through legacy args
        # assert engine.args["gooey_args"]["acquisition_backend"] == "dummy"
    
    def test_experiment_config_replaces_args_access(self, minimal_configs):
        """Test accessing experiment config fields instead of args dict."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_configs["hardware"],
            experiment_config=minimal_configs["experiment"],
            algorithm_config=minimal_configs["algorithm"]
        )
        
        # New way
        assert engine.experiment_config.acquisition.num_samples == 100
        # assert engine.experiment_config.acquisition.z_planes == 10
        assert engine.experiment_config.save_images is False
        # assert engine.experiment_config.subject.genotype == "test_strain"
        
        # # Old way
        # assert engine.args["gooey_args"]["total_frames"] == 100
        # assert engine.args["gooey_args"]["zsize"] == 10
    
    def test_algorithm_config_replaces_args_access(self, minimal_configs):
        """Test accessing algorithm config fields instead of args dict."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_configs["hardware"],
            experiment_config=minimal_configs["experiment"],
            algorithm_config=minimal_configs["algorithm"]
        )
        
        # New way
        assert engine.algorithm_config.algorithm_type == "dummy"
        assert engine.algorithm_config.enable_gui is False
        # assert engine.algorithm_config.algorithm_params.stimulus_diameter_pixels == 10
        
        # Old way
        # assert engine.args["gooey_args"]["trigger_algorithm"] == "dummy"
    
    def test_nested_config_access(self, minimal_configs):
        """Test accessing nested config structures."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_configs["hardware"],
            experiment_config=minimal_configs["experiment"],
            algorithm_config=minimal_configs["algorithm"]
        )
        
        # Nested experiment config
        assert engine.experiment_config.acquisition.num_samples == 100
        assert engine.experiment_config.subject.genotype == "test_strain"

        # Nested algorithm config
        assert engine.algorithm_config.stimulus_params.enabled is False


# ============================================================================
# 10. Microscope Name Removal Tests
# ============================================================================

class TestMicroscopeNameRemoval:
    """Test that microscope_name conditionals are no longer used."""
    
    def test_backend_selection_not_microscope_name(self, minimal_configs):
        """Test that backend field is used instead of microscope_name."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_configs["hardware"],
            experiment_config=minimal_configs["experiment"],
            algorithm_config=minimal_configs["algorithm"]
        )
        
        # Backend should be used for hardware selection
        assert engine.hardware_config.backend in ["dummy", "pycromanager", "pymmcore"]
        
        # microscope_name should only exist for backward compat
        if engine.hardware_config.microscope_name:
            # If present, it's just for legacy support
            assert isinstance(engine.hardware_config.microscope_name, str)
    
    def test_no_microscope_name_logic_in_engine(self, minimal_configs):
        """Test that engine doesn't use microscope_name for logic."""
        # Create configs with different microscope_name but same backend
        configs1 = minimal_configs.copy()
        configs1["hardware"].microscope_name = "scope_a"
        
        configs2 = minimal_configs.copy()
        configs2["hardware"].microscope_name = "scope_b"
        
        engine1 = ClosedLoopEngine(
            hardware_config=configs1["hardware"],
            experiment_config=configs1["experiment"],
            algorithm_config=configs1["algorithm"]
        )
        
        engine2 = ClosedLoopEngine(
            hardware_config=configs2["hardware"],
            experiment_config=configs2["experiment"],
            algorithm_config=configs2["algorithm"]
        )
        
        # Both should behave identically since backend is the same
        engine1.initialize_hardware()
        engine2.initialize_hardware()
        
        # Both should create same type of MMC
        assert type(engine1.mmc) == type(engine2.mmc)


# ====================
# 11. Hardware abstraction verification
# ====================

class TestHardwareAbstraction:
    """Test that hardware abstraction is properly implemented."""
    
    def test_no_direct_mmc_calls_in_acquisition_loop(self, minimal_configs):
        """Verify acquisition loop doesn't call MMC directly."""
        configs = minimal_configs.copy()
        configs["experiment"].acquisition.num_samples = 5
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Track all method calls on mmc
        from unittest.mock import Mock, patch
        mmc_calls = []
        original_mmc = engine.mmc
        
        def track_call(name):
            def wrapper(*args, **kwargs):
                mmc_calls.append(name)
                return getattr(original_mmc, name)(*args, **kwargs)
            return wrapper
        
        # Don't track these - they're allowed during initialization
        allowed_during_loop = []
        
        # Run acquisition
        engine.run_acquisition_loop()
        
        # During acquisition loop, should use camera interface, not direct MMC
        # (Some legacy calls may still exist temporarily)
        # This test documents the transition
        
    def test_camera_interface_methods_work(self, engine_with_configs):
        """Test that camera interface provides all needed methods."""
        engine = engine_with_configs
        engine.initialize_hardware()
        
        camera = engine.hardware.camera
        
        # Verify interface methods exist and work
        assert hasattr(camera, 'get_roi')
        assert hasattr(camera, 'get_image_size')
        assert hasattr(camera, 'start_acquisition')
        assert hasattr(camera, 'stop_acquisition')
        assert hasattr(camera, 'pop_next_image')
        assert hasattr(camera, 'get_remaining_image_count')
        assert hasattr(camera, 'clear_buffer')
        assert hasattr(camera, 'snap_image')
        assert hasattr(camera, 'get_image')
        
        # Test basic operations
        roi = camera.get_roi()
        assert len(roi) == 4
        
        size = camera.get_image_size()
        assert len(size) == 2
        
    def test_hardware_config_drives_initialization(self, minimal_configs):
        """Test that HardwareConfig properly drives hardware initialization."""
        # Test with dummy backend
        configs = minimal_configs.copy()
        configs["hardware"].backend = "dummy"
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        
        # Should create dummy backend
        from hardware.backends.dummy_backend import DummyHardwareBackend
        backend = engine.hardware._backend
        assert isinstance(backend, DummyHardwareBackend)
        
    def test_input_recording_passed_to_hardware(self, minimal_configs, dummy_tiff_file):
        """Test that input recording path is properly passed to hardware."""
        configs = minimal_configs.copy()
        configs["experiment"].input_recording_path = dummy_tiff_file
        
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        engine.initialize_hardware()
        
        # Hardware should be initialized with input file
        # For dummy backend, this means camera has loaded the file
        assert engine.hardware.is_initialized

# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])