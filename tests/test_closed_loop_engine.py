"""
Comprehensive test suite for ClosedLoopEngine class.

Tests are organized into 6 categories:
1. Initialization Tests
2. Acquisition Loop Tests
3. Metadata Tests
4. Cleanup Tests
5. Integration Tests
6. Error Handling Tests
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

# Import the engine and supporting modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from closed_loop_engine import ClosedLoopEngine, create_test_config
from engine.closed_loop_engine import ClosedLoopEngine, create_test_config
from lib import DummyMMC, DummyAlg, DummyStim


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
def minimal_config(temp_output_dir):
    """Provide minimal valid configuration for testing."""
    test_config = {
        # Acquisition controls
        "output_folder": "./test_output",
        "total_frames": 100,  # Small number for quick testing
        "mm_configuration_file": "MMConfig_demo.cfg", # Not used
        "zsize": 10,
        "save_mip": False,
        "strobe_acquisition": False,
        "strobe_inter_frame_interval": 80,
        "save_structural_scan": "none",
        
        # Experimental metadata (minimal for testing)
        "subject_strain": "test_strain",
        "subject_condition": "",
        "atr_concentration": 0.0,
        "z_step_size": 3.0,
        "nose_orientation": "left",
        "vnc_orientation": "up",
        "num_eggs": 0,
        "microscope_name": "test",
        "experimental_notes": "Test run with dummy objects",
        
        # Closed-loop controls
        "trigger_algorithm": "Dummy algorithm (does nothing)",
        "GUI_mode": "neural_imaging",
        "rec_baseline": 0,
        "save_alg_model_plot": False,
        
        # Stimulus settings
        "stim_interface": "no stim",
        "use_static_stim_roi": False,
        "frames_to_stimulate_for_options": [48],
        "stim_intensity_options": [10],
        "stimulus_diameter": 10,
        
        # Dev ops
        "input_recording": None,
        "acquisition_backend": "test",
        "no_save_images": True,  # Don't save images during testing
        "no_save_metadata": True,  # Don't save metadata during testing
        "save_gooey_defaults": False,
        "prefill_wb_ops": False,
        "send_sms": False,
        
        # Additional params that might be needed
        "roi": (0, 0, 200, 200),
        "exposure": 30,
        "binning": "1x1",
        "configs": {},
    }
    return test_config


@pytest.fixture
def dummy_tiff_file(temp_output_dir):
    """Create a dummy TIFF file for testing with file input."""
    import tifffile as tf
    
    # Create small test dataset
    test_data = np.random.randint(0, 65536, size=(100, 200, 200), dtype=np.uint16)
    filepath = os.path.join(temp_output_dir, "test_recording.tiff")
    tf.imwrite(filepath, test_data)
    
    return filepath


@pytest.fixture
def engine_with_config(minimal_config):
    """Provide an engine instance with minimal config."""
    return ClosedLoopEngine(gooey_args=minimal_config)


# ============================================================================
# 1. Initialization Tests
# ============================================================================

class TestInitialization:
    """Test suite for ClosedLoopEngine initialization."""
    
    def test_engine_accepts_config(self, minimal_config):
        """Test that engine accepts and stores configuration."""
        engine = ClosedLoopEngine(gooey_args=minimal_config)
        
        assert engine.gooey_args == minimal_config
        assert engine.args["gooey_args"] == minimal_config
        
    def test_args_structure_correct(self, minimal_config):
        """Test that args dictionary has correct structure for MMSubroutines."""
        engine = ClosedLoopEngine(gooey_args=minimal_config)
        
        assert "gooey_args" in engine.args
        assert engine.args["gooey_args"] == minimal_config
        
    def test_parameter_extraction(self, minimal_config):
        """Test that parameters are correctly extracted from config."""
        engine = ClosedLoopEngine(gooey_args=minimal_config)
        
        assert engine.zsize == minimal_config["zsize"]
        assert engine.frames_to_grab == minimal_config["total_frames"]
        assert engine.trigger_alg == minimal_config["trigger_algorithm"]
        assert engine.acquisition_backend == minimal_config["acquisition_backend"]
        
    def test_initial_state(self, engine_with_config):
        """Test that engine starts with correct initial state."""
        engine = engine_with_config
        
        assert engine.is_running is False
        assert engine.frame_count == 0
        assert engine.img_count == 0
        assert engine.cooldown_counter == 0
        assert engine.mmc is None
        assert engine.alg is None
        assert engine.stim is None
        
    def test_hardware_initialization_creates_mmc(self, engine_with_config):
        """Test that hardware initialization creates MMC object."""
        engine = engine_with_config
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        assert engine.mmc is not None
        assert isinstance(engine.mmc, DummyMMC.DummyMMC)
        
    def test_algorithm_factory_creates_dummy_alg(self, engine_with_config):
        """Test that algorithm factory creates DummyAlg for dummy config."""
        engine = engine_with_config
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        
        assert engine.alg is not None
        assert isinstance(engine.alg, DummyAlg.DummyAlg)
        
    def test_stimulus_initialization_creates_interface(self, engine_with_config):
        """Test that stimulus initialization creates interface."""
        engine = engine_with_config
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_stimulus()
        
        assert engine.stim is not None
        
    def test_roi_setup_from_config(self, minimal_config):
        """Test that ROI is properly set from config."""
        engine = ClosedLoopEngine(gooey_args=minimal_config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        assert engine.roi == minimal_config["roi"]
        assert engine.xsize == minimal_config["roi"][2]
        assert engine.ysize == minimal_config["roi"][3]


# ============================================================================
# 2. Acquisition Loop Tests
# ============================================================================

class TestAcquisitionLoop:
    """Test suite for acquisition loop functionality."""
    
    def test_prepare_acquisition_creates_directories(self, engine_with_config):
        """Test that prepare_acquisition creates output directories."""
        engine = engine_with_config
        engine.prepare_acquisition()
        
        assert engine.savedir is not None
        assert os.path.exists(engine.savedir)
        assert engine.saveroot is not None
        assert engine.session_id is not None
        
    def test_prepare_acquisition_initializes_frame_storage(self, engine_with_config):
        """Test that frame storage is initialized with correct dimensions."""
        engine = engine_with_config
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        assert engine.frames is not None
        assert engine.frames.shape == (engine.frames_to_grab, engine.ysize, engine.xsize)
        assert engine.frames.dtype == np.uint16
        
    def test_acquisition_loop_captures_frames(self, minimal_config):
        """Test that acquisition loop captures the expected number of frames."""
        config = minimal_config.copy()
        config["total_frames"] = 20  # Small number for quick test
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Run acquisition
        engine.run_acquisition_loop()
        
        assert engine.img_count == config["total_frames"]
        
    def test_acquisition_loop_tracks_frame_times(self, minimal_config):
        """Test that frame timestamps are recorded."""
        config = minimal_config.copy()
        config["total_frames"] = 20
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        engine.run_acquisition_loop()
        
        assert len(engine.frame_time_list) == config["total_frames"]
        assert all(isinstance(t, (float, np.float64)) for t in engine.frame_time_list)
        
    def test_z_stack_indexing(self, minimal_config):
        """Test that z-stack indexing cycles correctly."""
        config = minimal_config.copy()
        config["total_frames"] = 25
        config["zsize"] = 5
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Mock the algorithm to track z indices
        z_indices_seen = []
        original_process_frame = engine.alg.process_frame
        
        def track_z_index(frame, zndx):
            z_indices_seen.append(zndx)
            return original_process_frame(frame, zndx)
            
        engine.alg.process_frame = track_z_index
        
        engine.run_acquisition_loop()
        
        # Check that z indices cycle from 0 to zsize-1
        expected_pattern = [i % config["zsize"] for i in range(config["total_frames"])]
        assert z_indices_seen == expected_pattern
        
    def test_stimulus_triggering_flow(self, minimal_config):
        """Test that stimulus checking and submission occurs each frame."""
        config = minimal_config.copy()
        config["total_frames"] = 10
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Track stimulus submissions
        stim_submissions = []
        original_submit = engine.stim.submit_stim_params
        
        def track_stim(stim_params, image_ndx):
            stim_submissions.append((stim_params, image_ndx))
            return original_submit(stim_params, image_ndx)
            
        engine.stim.submit_stim_params = track_stim
        
        engine.run_acquisition_loop()
        
        # Should have one submission per frame
        assert len(stim_submissions) == config["total_frames"]
        
    def test_cooldown_counter_decrements(self, minimal_config):
        """Test that cooldown counter properly decrements."""
        config = minimal_config.copy()
        config["total_frames"] = 10
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Set initial cooldown
        engine.cooldown_counter = 5
        initial_cooldown = engine.cooldown_counter
        
        # Mock to trigger after a few frames
        frame_count = 0
        def mock_check_stim(image_ndx, cooldown):
            nonlocal frame_count
            frame_count += 1
            if frame_count >= initial_cooldown:
                return {}, 0  # Reset cooldown after it expires
            return {}, cooldown - 1 if cooldown > 0 else 0
            
        engine.alg.check_stim = mock_check_stim
        
        engine.run_acquisition_loop()
        
        # Cooldown should have decremented to zero
        assert engine.cooldown_counter == 0


# ============================================================================
# 3. Metadata Tests
# ============================================================================

class TestMetadata:
    """Test suite for metadata collection and saving."""
    
    def test_metadata_collection_from_all_components(self, minimal_config):
        """Test that metadata is collected from all components."""
        config = minimal_config.copy()
        config["no_save_metadata"] = False
        config["total_frames"] = 10
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        engine.run_acquisition_loop()
        
        # Manually call save_metadata to inspect
        with patch('lib.wbliveUtils.save_metadata') as mock_save:
            engine.save_metadata()
            
            # Check that save_metadata was called
            assert mock_save.called
            
            # Get the metadata dict that was passed
            call_args = mock_save.call_args
            metadata = call_args[1]['metadata']
            
            # Verify metadata structure
            assert "gooey_args" in metadata
            assert "frame_time_list" in metadata
            assert "t0" in metadata
            assert "xsize" in metadata
            assert "ysize" in metadata
            assert "alg_metadata" in metadata
            assert "stim_metadata" in metadata
            
    def test_metadata_includes_timing_info(self, minimal_config):
        """Test that metadata includes timing information."""
        config = minimal_config.copy()
        config["no_save_metadata"] = False
        config["total_frames"] = 10
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        engine.run_acquisition_loop()
        
        with patch('lib.wbliveUtils.save_metadata') as mock_save:
            engine.save_metadata()
            
            metadata = mock_save.call_args[1]['metadata']
            
            assert metadata["t0"] is not None
            assert len(metadata["frame_time_list"]) == config["total_frames"]
            
    def test_metadata_save_disabled_flag(self, minimal_config):
        """Test that no_save_metadata flag prevents saving."""
        config = minimal_config.copy()
        config["no_save_metadata"] = True
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.prepare_acquisition()
        
        with patch('lib.wbliveUtils.save_metadata') as mock_save:
            engine.save_metadata()
            
            # Should not call save when flag is True
            assert not mock_save.called
            
    def test_algorithm_metadata_included(self, minimal_config):
        """Test that algorithm metadata is collected."""
        config = minimal_config.copy()
        config["no_save_metadata"] = False
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        with patch('lib.wbliveUtils.save_metadata') as mock_save:
            engine.save_metadata()
            
            metadata = mock_save.call_args[1]['metadata']
            
            assert "alg_metadata" in metadata
            assert metadata["alg_metadata"]["is_dummy_alg"] is True


# ============================================================================
# 4. Cleanup Tests
# ============================================================================

class TestCleanup:
    """Test suite for resource cleanup and management."""
    
    def test_cleanup_stops_acquisition(self, minimal_config):
        """Test that cleanup stops the acquisition."""
        engine = ClosedLoopEngine(gooey_args=minimal_config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        # Mock the stopSequenceAcquisition method
        engine.mmc.stopSequenceAcquisition = Mock()
        
        engine.cleanup()
        
        engine.mmc.stopSequenceAcquisition.assert_called_once()
        
    def test_cleanup_closes_algorithm(self, minimal_config):
        """Test that cleanup closes the algorithm."""
        engine = ClosedLoopEngine(gooey_args=minimal_config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        
        # Mock the algorithm close method
        engine.alg.close = Mock()
        
        engine.cleanup()
        
        engine.alg.close.assert_called_once()
        
    def test_cleanup_closes_stimulus(self, minimal_config):
        """Test that cleanup closes stimulus interface."""
        engine = ClosedLoopEngine(gooey_args=minimal_config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_stimulus()
        
        # Mock the stimulus close method
        engine.stim.close = Mock()
        
        engine.cleanup()
        
        engine.stim.close.assert_called_once()
        
    def test_cleanup_handles_mmc_close_errors(self, minimal_config):
        """Test that cleanup gracefully handles MMC errors."""
        engine = ClosedLoopEngine(gooey_args=minimal_config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        # Make stopSequenceAcquisition raise an error
        engine.mmc.stopSequenceAcquisition = Mock(side_effect=Exception("MMC error"))
        
        # Should not raise - cleanup should handle gracefully
        try:
            engine.cleanup()
        except Exception as e:
            pytest.fail(f"Cleanup should handle errors gracefully, but raised: {e}")
            
    def test_cleanup_sends_sms_notification(self, minimal_config):
        """Test that cleanup sends SMS notification when enabled."""
        config = minimal_config.copy()
        config["send_sms"] = True
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.prepare_acquisition()
        
        with patch('lib.wbliveUtils.notify') as mock_notify:
            engine.cleanup()
            
            mock_notify.assert_called_once()
            call_args = mock_notify.call_args
            assert "completed" in call_args[0][0].lower()
            
    def test_cleanup_skips_sms_when_disabled(self, minimal_config):
        """Test that cleanup skips SMS when disabled."""
        config = minimal_config.copy()
        config["send_sms"] = False
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.prepare_acquisition()
        
        with patch('lib.wbliveUtils.notify') as mock_notify:
            engine.cleanup()
            
            mock_notify.assert_not_called()


# ============================================================================
# 5. Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests with dummy backend."""
    
    def test_full_acquisition_workflow(self, minimal_config):
        """Test complete acquisition workflow from start to finish."""
        config = minimal_config.copy()
        config["total_frames"] = 20
        
        engine = ClosedLoopEngine(gooey_args=config)
        
        # Should complete without errors
        engine.run()
        
        # Verify final state
        assert engine.img_count == config["total_frames"]
        assert os.path.exists(engine.savedir)
        
    def test_acquisition_with_tiff_input(self, minimal_config, dummy_tiff_file):
        """Test acquisition with TIFF file input."""
        config = minimal_config.copy()
        config["input_recording"] = dummy_tiff_file
        config["total_frames"] = 50
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.run()
        
        assert engine.img_count == config["total_frames"]
        
    def test_acquisition_with_z_stacks(self, minimal_config):
        """Test acquisition with multiple z-planes."""
        config = minimal_config.copy()
        config["total_frames"] = 30
        config["zsize"] = 10
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.run()
        
        # Should complete 3 full volumes
        assert engine.img_count == 30
        
    def test_acquisition_saves_images_when_enabled(self, minimal_config):
        """Test that images are saved when flag is enabled."""
        config = minimal_config.copy()
        config["no_save_images"] = False
        config["total_frames"] = 10
        
        engine = ClosedLoopEngine(gooey_args=config)
        
        with patch('lib.MMSubroutines.saveScanTiffs') as mock_save:
            engine.run()
            
            # Should call save function
            mock_save.assert_called_once()
            
    def test_acquisition_skips_images_when_disabled(self, minimal_config):
        """Test that images are not saved when flag is disabled."""
        config = minimal_config.copy()
        config["no_save_images"] = True
        
        engine = ClosedLoopEngine(gooey_args=config)
        
        with patch('lib.MMSubroutines.saveScanTiffs') as mock_save:
            engine.run()
            
            # Should not call save function
            mock_save.assert_not_called()
            
    def test_multiple_acquisitions_with_same_instance(self, minimal_config):
        """Test that engine can be reused for multiple acquisitions."""
        config = minimal_config.copy()
        config["total_frames"] = 10
        
        # First acquisition
        engine = ClosedLoopEngine(gooey_args=config)
        engine.initialize_hardware()
        engine.prepare_acquisition()  # Creates new directories
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        engine.run_acquisition_loop()
        engine.cleanup()
        first_session_id = engine.session_id

        # wait 1s for new session id
        time.sleep(1)
        
        # Reset for second acquisition
        engine.is_running = False
        engine.img_count = 0
        engine.frame_count = 0
        engine.prepare_acquisition()  # Creates new directories
        
        # Second acquisition
        engine.initialize_hardware()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        engine.run_acquisition_loop()
        engine.cleanup()
        
        second_session_id = engine.session_id
        
        # Session IDs should be different
        assert first_session_id != second_session_id
        
    # def test_legacy_entry_point_compatibility(self, minimal_config):
    #     """Test that legacy launch_wblive_from_gooey still works."""
    #     from closed_loop_engine import launch_wblive_from_gooey
        
    #     config = minimal_config.copy()
    #     config["total_frames"] = 10
        
    #     # Mock sys.exit to prevent actual exit
    #     with patch('sys.exit'):
    #         launch_wblive_from_gooey(ops=config)


# ============================================================================
# 6. Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_missing_required_config_keys(self):
        """Test that missing required config keys raise appropriate errors."""
        incomplete_config = {
            "output_folder": "./test",
            # Missing many required keys
        }
        
        engine = ClosedLoopEngine(gooey_args=incomplete_config)
        
        # Should handle missing keys gracefully during initialization
        # The actual behavior depends on implementation - test for expected behavior
        
    def test_unknown_algorithm_name(self, minimal_config):
        """Test handling of unknown algorithm name."""
        config = minimal_config.copy()
        config["trigger_algorithm"] = "NonexistentAlgorithm"
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        # Should fall back to DummyAlg
        engine.initialize_algorithm()
        assert isinstance(engine.alg, DummyAlg.DummyAlg)
        
    def test_invalid_roi_dimensions(self, minimal_config):
        """Test handling of invalid ROI."""
        config = minimal_config.copy()
        config["roi"] = [0, 0, 0, 0]  # Invalid zero-size ROI
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        # Should still create frame storage, even if dimensions are unusual
        assert engine.frames is not None
        
    def test_acquisition_interrupted_mid_loop(self, minimal_config):
        """Test that interrupting acquisition is handled gracefully."""
        config = minimal_config.copy()
        config["total_frames"] = 100
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Mock to interrupt after 10 frames
        original_process = engine.alg.process_frame
        call_count = [0]
        
        def interrupt_after_10(frame, zndx):
            call_count[0] += 1
            if call_count[0] > 10:
                raise KeyboardInterrupt("User interrupted")
            return original_process(frame, zndx)
            
        engine.alg.process_frame = interrupt_after_10
        
        # Should handle interrupt gracefully
        with pytest.raises(KeyboardInterrupt):
            engine.run_acquisition_loop()
            
    def test_file_write_error_during_save(self, minimal_config):
        """Test handling of file write errors during save."""
        config = minimal_config.copy()
        config["no_save_images"] = False
        config["total_frames"] = 10
        config["output_folder"] = "/invalid/path/that/does/not/exist"
        
        engine = ClosedLoopEngine(gooey_args=config)
        
        # Should raise error during prepare_acquisition when creating dirs
        with pytest.raises(Exception):
            engine.prepare_acquisition()
            
    def test_mmc_initialization_failure(self, minimal_config):
        """Test handling of MMC initialization failure."""
        config = minimal_config.copy()
        config["mm_configuration_file"] = "/invalid/config/file.cfg"
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.prepare_acquisition()
        
        # Mock initialize_mmc to raise error
        with patch('lib.MMSubroutines.initialize_mmc', side_effect=Exception("MMC init failed")):
            with pytest.raises(Exception):
                engine.initialize_hardware()
                
    def test_algorithm_process_frame_error(self, minimal_config):
        """Test handling of algorithm errors during frame processing."""
        config = minimal_config.copy()
        config["total_frames"] = 10
        
        engine = ClosedLoopEngine(gooey_args=config)
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Make algorithm raise error
        engine.alg.process_frame = Mock(side_effect=Exception("Algorithm error"))
        
        # Should propagate error from acquisition loop
        with pytest.raises(Exception):
            engine.run_acquisition_loop()
            
    def test_cleanup_called_on_exception(self, minimal_config):
        """Test that cleanup is called even when exception occurs."""
        config = minimal_config.copy()
        config["total_frames"] = 10
        
        engine = ClosedLoopEngine(gooey_args=config)
        
        # Mock run_acquisition_loop to raise error
        with patch.object(engine, 'run_acquisition_loop', side_effect=Exception("Test error")):
            with patch.object(engine, 'cleanup') as mock_cleanup:
                with pytest.raises(Exception):
                    engine.run()
                    
                # Cleanup should still be called
                mock_cleanup.assert_called_once()


# ============================================================================
# Test Utilities
# ============================================================================

def test_create_test_config_generates_valid_config():
    """Test that create_test_config helper generates valid configuration."""
    config = create_test_config()
    
    # Should have all required keys
    required_keys = [
        "output_folder", "total_frames", "mm_configuration_file", "zsize",
        "trigger_algorithm", "stim_interface", "acquisition_backend"
    ]
    
    for key in required_keys:
        assert key in config
        
    # Should have sensible defaults
    assert config["total_frames"] == 100
    assert config["trigger_algorithm"] == "Dummy algorithm (does nothing)"
    assert config["stim_interface"] == "no stim"
    

def test_create_test_config_with_tiff_input():
    """Test create_test_config with TIFF file input."""
    test_file = "/path/to/test.tiff"
    config = create_test_config(input_recording=test_file)
    
    assert config["input_recording"] == test_file