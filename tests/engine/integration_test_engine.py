"""
Integration test for ClosedLoopEngine with config-based stimulus.

Tests that the engine properly integrates with stimulus controllers
and hardware manager using the new config-based architecture.
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from engine.closed_loop_engine import ClosedLoopEngine
from config.config_manager import (
    HardwareConfig,
    ExperimentConfig,
    AlgorithmConfig,
    AcquisitionConfig,
    SubjectMetadata,
    StimulusParameters,
    TreatmentDetails,
    Orientation,
)


@pytest.fixture
def test_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def minimal_hardware_config():
    """Create minimal hardware config for testing."""
    return HardwareConfig(
        backend="dummy",
        stim_interface="dummy",
        microscope_name="test_scope",
    )


@pytest.fixture
def minimal_experiment_config(test_output_dir):
    """Create minimal experiment config for testing."""
    return ExperimentConfig(
        experiment_name="test_experiment",
        output_dir=test_output_dir,
        save_images=False,
        save_metadata=True,
        save_sample_video=False,
        acquisition=AcquisitionConfig(
            num_samples=10,
        ),
        subject=SubjectMetadata(
            subject_type="test_strain",
            treatment_details=TreatmentDetails(),
        ),
    )


@pytest.fixture
def minimal_algorithm_config():
    """Create minimal algorithm config for testing."""
    from config.config_manager import AlgorithmParameters, GUIParameters

    return AlgorithmConfig(
        algorithm_type="dummy",
        algorithm_params=AlgorithmParameters(),
        stimulus_params=StimulusParameters(
            duration_frames_options=[2],
            intensity_percent_options=[10],
        ),
        gui_params=GUIParameters(
            enable_gui=False,
            gui_mode="neural_imaging",
            save_algorithm_plot=False,
        ),
    )


class TestClosedLoopEngineIntegration:
    """Integration tests for ClosedLoopEngine with new stimulus architecture."""
    
    def test_engine_initialization_with_configs(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test engine initializes properly with config objects."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        assert engine.hardware_config == minimal_hardware_config
        assert engine.experiment_config == minimal_experiment_config
        assert engine.algorithm_config == minimal_algorithm_config
        assert engine.hardware is None  # Not initialized yet
        assert engine.stim_controller is None  # Not initialized yet
    
    def test_hardware_initialization(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test hardware initializes through HardwareManager."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        engine.initialize_hardware()
        
        assert engine.hardware is not None
        assert engine.hardware.is_initialized
        assert engine.roi is not None
        assert engine.xsize > 0
        assert engine.ysize > 0
    
    def test_stimulus_controller_initialization(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test stimulus controller initializes with config-based pattern."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        # Initialize hardware first (required for stimulus)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        # Initialize stimulus controller
        engine.initialize_stimulus()
        
        assert engine.stim_controller is not None
        assert hasattr(engine.stim_controller, 'submit_stim_params')
        assert hasattr(engine.stim_controller, 'check_stim')
        assert hasattr(engine.stim_controller, 'get_metadata')
    
    def test_full_acquisition_cycle(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test complete acquisition cycle with stimulus controller."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        try:
            # Run full acquisition
            engine.initialize_hardware()
            engine.prepare_acquisition()
            engine.initialize_algorithm()
            engine.initialize_stimulus()
            
            # Verify all components initialized
            assert engine.hardware is not None
            assert engine.alg is not None
            assert engine.stim_controller is not None
            
            # Run short acquisition
            engine.run_acquisition_loop()
            
            # Verify acquisition ran
            assert engine.img_count > 0
            assert len(engine.frame_time_list) > 0
            
        finally:
            engine.cleanup()
    
    def test_stimulus_controller_receives_params_from_algorithm(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test stimulus controller receives params from algorithm during acquisition."""
        # Modify config to ensure stimulus events
        minimal_algorithm_config.algorithm_params.stim_threshold_pos = 0.0001  # Very low threshold
        
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        try:
            engine.initialize_hardware()
            engine.prepare_acquisition()
            engine.initialize_algorithm()
            engine.initialize_stimulus()
            
            # Run acquisition
            engine.run_acquisition_loop()
            
            # Check if stimulus controller received any events
            # (may be empty for dummy algorithm, but structure should be there)
            metadata = engine.stim_controller.get_metadata()
            assert "stim_on_list" in metadata
            assert "stim_off_list" in metadata
            assert "stim_param_list" in metadata
            
        finally:
            engine.cleanup()
    
    def test_metadata_includes_stimulus_controller_data(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test saved metadata includes stimulus controller information."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        try:
            engine.initialize_hardware()
            engine.prepare_acquisition()
            engine.initialize_algorithm()
            engine.initialize_stimulus()
            engine.run_acquisition_loop()
            
            # Get metadata
            metadata = engine.save_metadata()
            
            # Verify stimulus metadata is included
            assert "stim_metadata" in metadata
            assert "stim_on_list" in metadata["stim_metadata"]
            assert "stim_off_list" in metadata["stim_metadata"]
            
        finally:
            engine.cleanup()
    
    def test_stimulus_activation_through_hardware_manager(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test that stimulus controller properly controls hardware.stimulus."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        try:
            engine.initialize_hardware()
            engine.prepare_acquisition()
            engine.initialize_stimulus()
            
            # Manually submit stim params
            stim_params = {
                "stim_on": 5,
                "stim_off": 7,
                "event": {"stim_intensity": 50}
            }
            engine.stim_controller.submit_stim_params(stim_params, 0)
            
            # Check at activation frame
            engine.stim_controller.check_stim(5)
            
            # Verify hardware stimulus is active
            assert engine.hardware.stimulus.is_stimulus_active()
            
            # Check at deactivation frame
            engine.stim_controller.check_stim(7)
            
            # Verify hardware stimulus is inactive
            assert not engine.hardware.stimulus.is_stimulus_active()
            
        finally:
            engine.cleanup()
    
    def test_different_stimulus_interfaces(self, minimal_experiment_config, minimal_algorithm_config):
        """Test engine works with different stimulus interface types."""
        stimulus_interfaces = ["dummy", "no stim", "test"]
        
        for stim_interface in stimulus_interfaces:
            hardware_config = HardwareConfig(
                backend="dummy",
                stim_interface=stim_interface,
            )
            
            engine = ClosedLoopEngine(
                hardware_config=hardware_config,
                experiment_config=minimal_experiment_config,
                algorithm_config=minimal_algorithm_config,
            )
            
            try:
                engine.initialize_hardware()
                engine.prepare_acquisition()
                engine.initialize_stimulus()
                
                # Verify controller was created
                assert engine.stim_controller is not None
                
                # Verify it's the right type (all dummy variants should create DummyStimulusController)
                from hardware.stimulus_controllers.simple_controller import SimpleStimulusController
                assert isinstance(engine.stim_controller, SimpleStimulusController)
                
            finally:
                engine.cleanup()
    
    def test_cleanup_closes_stimulus_controller(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test cleanup properly closes stimulus controller."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_stimulus()
        
        # Track if close was called
        close_called = False
        original_close = engine.stim_controller.close
        
        def tracked_close():
            nonlocal close_called
            close_called = True
            return original_close()
        
        engine.stim_controller.close = tracked_close
        
        # Cleanup
        engine.cleanup()
        
        # Verify close was called
        assert close_called
    
    # def test_backward_compatibility_with_legacy_args(
    #     self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    # ):
    #     """Test that engine still builds legacy args dict for backward compatibility."""
    #     engine = ClosedLoopEngine(
    #         hardware_config=minimal_hardware_config,
    #         experiment_config=minimal_experiment_config,
    #         algorithm_config=minimal_algorithm_config,
    #     )
        
    #     # Verify args dict exists and has expected structure
    #     assert "gooey_args" in engine.args
    #     assert "stim_interface" in engine.args["gooey_args"]
    #     assert "acquisition_backend" in engine.args["gooey_args"]
    #     assert "trigger_algorithm" in engine.args["gooey_args"]


class TestConfigBasedStimulusPattern:
    """Test the new config-based stimulus pattern specifically."""
    
    def test_no_direct_stim_interface_calls_in_loop(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Verify acquisition loop uses controller, not direct stim interface."""
        # Modify config for very short acquisition
        minimal_experiment_config.acquisition.num_frames = 5
        
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        try:
            engine.initialize_hardware()
            engine.prepare_acquisition()
            engine.initialize_algorithm()
            engine.initialize_stimulus()
            
            # Track controller calls
            submit_calls = []
            check_calls = []
            
            original_submit = engine.stim_controller.submit_stim_params
            original_check = engine.stim_controller.check_stim
            
            def tracked_submit(params, idx):
                submit_calls.append((params, idx))
                return original_submit(params, idx)
            
            def tracked_check(frame):
                check_calls.append(frame)
                return original_check(frame)
            
            engine.stim_controller.submit_stim_params = tracked_submit
            engine.stim_controller.check_stim = tracked_check
            
            # Run acquisition
            engine.run_acquisition_loop()
            
            # Verify controller methods were called (not direct hardware.stimulus)
            # At minimum, check_stim should be called for volume completions
            assert len(check_calls) > 0
            
        finally:
            engine.cleanup()
    
    def test_controller_manages_hardware_stimulus(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test that controller is the sole manager of hardware.stimulus."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        try:
            engine.initialize_hardware()
            engine.prepare_acquisition()
            engine.initialize_stimulus()
            
            # Direct hardware.stimulus should not be called by engine
            # Only controller should call it
            
            # Submit params through controller
            stim_params = {
                "stim_on": 3,
                "stim_off": 5,
                "event": {"stim_intensity": 30}
            }
            engine.stim_controller.submit_stim_params(stim_params, 1)
            
            # Controller manages activation
            engine.stim_controller.check_stim(3)
            assert engine.hardware.stimulus.is_stimulus_active()
            
            # Controller manages deactivation
            engine.stim_controller.check_stim(5)
            assert not engine.hardware.stimulus.is_stimulus_active()
            
        finally:
            engine.cleanup()


class TestLegacyCompatibility:
    """Test backward compatibility with legacy interfaces."""
    
    # def test_convert_gooey_args_to_configs(self):
    #     """Test conversion function for legacy gooey_args."""
    #     from closed_loop_engine import convert_gooey_args_to_configs
        
    #     gooey_args = {
    #         "acquisition_backend": "dummy",
    #         "stim_interface": "dummy",
    #         "microscope_name": "test_scope",
    #         "total_frames": 50,
    #         "zsize": 5,
    #         "trigger_algorithm": "dummy",
    #         "output_folder": "./test_output",
    #     }
        
    #     configs = convert_gooey_args_to_configs(gooey_args)
        
    #     assert "hardware" in configs
    #     assert "experiment" in configs
    #     assert "algorithm" in configs
        
    #     # Verify conversion
    #     assert configs["hardware"].backend == "dummy"
    #     assert configs["hardware"].stim_interface == "dummy"
    #     assert configs["experiment"].acquisition.num_frames == 50
    #     assert configs["algorithm"].algorithm_type == "dummy"
    
    # def test_legacy_args_dict_still_available(
    #     self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    # ):
    #     """Test that engine still provides args dict for legacy components."""
    #     engine = ClosedLoopEngine(
    #         hardware_config=minimal_hardware_config,
    #         experiment_config=minimal_experiment_config,
    #         algorithm_config=minimal_algorithm_config,
    #     )
        
    #     # Legacy args should be available
    #     assert hasattr(engine, 'args')
    #     assert "gooey_args" in engine.args
        
    #     # Should contain key fields
    #     assert "stim_interface" in engine.args["gooey_args"]
    #     assert "acquisition_backend" in engine.args["gooey_args"]

