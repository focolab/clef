"""
Unit tests for DummyAlg algorithm

Tests the minimal dummy algorithm implementation.
"""

import pytest
import os
import numpy as np
import logging
from unittest.mock import Mock

# Import the engine and supporting modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from algorithms.dummy import DummyAlg
from config.config_manager import HardwareConfig
from hardware.hardware_manager import HardwareManager


@pytest.fixture
def hardware_manager():
    """Create and initialize a hardware manager for testing."""
    hardware_config = HardwareConfig(
        backend="dummy",
        stim_interface="dummy",
        microscope_name="test",
    )
    hw_manager = HardwareManager(hardware_config)
    hw_manager.initialize()
    yield hw_manager
    hw_manager.close()


class TestDummyAlgBasics:
    """Test basic DummyAlg functionality."""
    
    def test_initialization_with_minimal_args(self):
        """Test DummyAlg can initialize with no args."""
        alg = DummyAlg()
        assert alg is not None
        assert alg.sample_count == 0
        assert alg.volume_count == 0
    
    # def test_initialization_with_full_args(self):
    #     """Test DummyAlg initialization with full args."""
    #     args = {
    #         "id": "test_session",
    #         "roi": [0, 0, 512, 512],
    #         "gooey_args": {
    #             "total_frames": 100,
    #             "zsize": 10,
    #         }
    #     }
        
    #     alg = DummyAlg(args=args)
    #     assert alg.samples_to_grab == 100
    #     assert alg.zsize == 10
    #     assert alg.xsize == 512
    #     assert alg.ysize == 512
    
    def test_initialization_with_local_handles(self):
        """Test DummyAlg accepts local_handles."""
        mock_mmc = Mock()
        alg = DummyAlg(local_handles={"mmc": mock_mmc})
        assert alg.local_handles["mmc"] == mock_mmc
    
    def test_initialize_model(self):
        """Test model initialization."""
        args = {"id": "test_session"}
        alg = DummyAlg(args=args)
        
        # Should not raise any errors
        alg.initialize_model()


class TestDummyAlgProcessing:
    """Test DummyAlg frame and volume processing."""
    
    def test_process_sample(self):
        """Test frame processing increments counter."""
        alg = DummyAlg()
        alg.initialize_model()
        
        img = np.zeros((512, 512), dtype=np.uint16)
        
        assert alg.sample_count == 0
        alg.process_sample(img, sample_ndx=0)
        assert alg.sample_count == 1
        alg.process_sample(img, sample_ndx=0)
        assert alg.sample_count == 2
    
    def test_process_volume_counting(self):
        """Test volume counting when z-stack completes."""
        args = {
            "gooey_args": {
                "zsize": 5,
            }
        }
        alg = DummyAlg(args=args)
        img = np.zeros((512, 512), dtype=np.uint16)
        sample_ndx = 0
        
        assert alg.zsize == 5
        assert alg.volume_count == 0
        
        # Process first volume (z=0 through z=4)
        for z in range(5):
            alg.process_sample(img, sample_ndx=sample_ndx)
            sample_ndx = sample_ndx + 1
        
        assert alg.volume_count == 1
        
        # Process second volume
        for z in range(5):
            alg.process_sample(img, sample_ndx=sample_ndx)
            sample_ndx = sample_ndx + 1
        
        assert alg.volume_count == 2
    
    def test_process_volume_method(self):
        """Test process_volume method (should do nothing)."""
        alg = DummyAlg()
        
        # Should not raise any errors
        alg.process_volume()


class TestDummyAlgStimulation:
    """Test DummyAlg stimulation behavior."""
    
    def test_check_stim_returns_empty(self):
        """Test that check_stim never triggers stimulation."""
        alg = DummyAlg()
        
        # Check stim at various image indices
        for i in range(100):
            stim_params, cooldown = alg.check_stim(i, cooldown_counter=0)
            
            # Should always return empty dict
            assert stim_params == {}
            assert cooldown == 0
    
    def test_check_stim_with_cooldown(self):
        """Test that cooldown doesn't affect dummy algorithm."""
        alg = DummyAlg()
        
        # Even with cooldown, should return empty
        stim_params, cooldown = alg.check_stim(0, cooldown_counter=100)
        assert stim_params == {}
        assert cooldown == 0


class TestDummyAlgMetadata:
    """Test DummyAlg metadata functionality."""
    
    def test_get_metadata_structure(self):
        """Test metadata has expected structure."""
        alg = DummyAlg()
        metadata = alg.get_metadata()
        
        assert "algorithm_type" in metadata
        assert "samples_processed" in metadata
        assert "volumes_processed" in metadata
        assert "description" in metadata
    
    def test_get_metadata_values(self):
        """Test metadata reflects actual processing."""
        args = {"gooey_args": {"zsize": 10}}
        alg = DummyAlg(args=args)
        img = np.zeros((512, 512), dtype=np.uint16)
        
        # Process some frames
        for i in range(20):
            alg.process_sample(img, sample_ndx=i)
        
        metadata = alg.get_metadata()
        
        assert metadata["algorithm_type"] == "DummyAlg"
        assert metadata["samples_processed"] == 20
        assert metadata["volumes_processed"] == 2  # 2 complete volumes


class TestDummyAlgClosing:
    """Test DummyAlg cleanup functionality."""
    
    def test_close(self, caplog):
        """Test close method logs correctly."""
        args = {"gooey_args": {"zsize": 5}}
        alg = DummyAlg(args=args)
        img = np.zeros((512, 512), dtype=np.uint16)
        
        # Process some frames
        for i in range(10):
            alg.process_sample(img, sample_ndx=i)
        
        with caplog.at_level(logging.INFO):
            alg.close()
        
        # Should log close message with counts
        assert "DummyAlg closing" in caplog.text
        assert "10 frames" in caplog.text
        assert "2 volumes" in caplog.text


class TestDummyAlgPlotting:
    """Test DummyAlg plotting functionality."""
    
    def test_plot_model(self, caplog):
        """Test plot_model logs appropriately."""
        alg = DummyAlg()
        
        with caplog.at_level(logging.INFO):
            alg.plot_model()
        
        assert "DummyAlg has no model to plot" in caplog.text
    
    def test_plot_model_with_save(self, caplog):
        """Test plot_model with save parameter."""
        alg = DummyAlg()
        
        with caplog.at_level(logging.INFO):
            alg.plot_model(savefilename="test.png")
        
        # Should still just log that there's nothing to plot
        assert "DummyAlg has no model to plot" in caplog.text


class TestDummyAlgIntegration:
    """Integration tests for DummyAlg with factory."""
    
    def test_create_from_factory(self, hardware_manager):
        """Test creating DummyAlg through factory."""
        from algorithms import create_algorithm
        
        # Create mock configs
        algorithm_config = Mock()
        algorithm_config.algorithm_type = "dummy"
        algorithm_config.gui_params = Mock()
        algorithm_config.gui_params.gui_mode = "neural_imaging"
        algorithm_config.gui_params.save_algorithm_plot = False
        algorithm_config.algorithm_params = Mock()
        algorithm_config.algorithm_params.stim_cooldown_frames = 900
        algorithm_config.algorithm_params.skip_stimulation_probability = 0.0
        algorithm_config.algorithm_params.delay_stimulation_probability = 0.0
        algorithm_config.algorithm_params.stim_delay_frames_options = []
        algorithm_config.algorithm_params.stim_onset_list = []
        algorithm_config.stimulus_params = Mock()
        algorithm_config.stimulus_params.duration_frames_options = [48]
        algorithm_config.stimulus_params.intensity_percent_options = [10]
        
        experiment_config = Mock()
        experiment_config.output_dir = "./test"
        experiment_config.save_images = True
        experiment_config.save_metadata = True
        experiment_config.save_sample_video = False
        experiment_config.input_recording_path = None
        experiment_config.z_step_size_um = 1.0
        experiment_config.acquisition = Mock()
        experiment_config.acquisition.num_samples = 100
        experiment_config.acquisition.z_planes = 10
        experiment_config.acquisition.save_structural_scan = "none"
        experiment_config.acquisition.baseline_frames = 0
        experiment_config.subject = Mock()
        experiment_config.subject.genotype = "test"
        experiment_config.subject.num_eggs = 0
        experiment_config.subject.notes = ""
        experiment_config.subject.treatment_details = Mock()
        experiment_config.subject.treatment_details.condition = ""
        experiment_config.subject.treatment_details.atr_concentration_uM = 0.0
        experiment_config.subject.orientation = Mock()
        experiment_config.subject.orientation.nose = "left"
        experiment_config.subject.orientation.vnc = "up"
        experiment_config.dev_options = Mock()
        # experiment_config.dev_options.prefill_wb_ops = False
        experiment_config.dev_options.send_sms_on_completion = False
        
        # Create algorithm through factory
        alg = create_algorithm(
            algorithm_config=algorithm_config,
            experiment_config=experiment_config,
            hardware_manager=hardware_manager,
        )
        
        # Should be a DummyAlg instance
        assert isinstance(alg, DummyAlg)
        
        # Should be usable
        alg.initialize_model()
        img = np.zeros((512, 512), dtype=np.uint16)
        alg.process_sample(img, sample_ndx=0)
        stim_params, cooldown = alg.check_stim(0, 0)
        assert stim_params == {}


class TestDummyAlgEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_gooey_args(self):
        """Test DummyAlg handles missing gooey_args."""
        args = {"id": "test"}
        alg = DummyAlg(args=args)
        
        # Should use defaults
        assert alg.samples_to_grab == 100
        assert alg.zsize == 1
    
    def test_missing_roi(self):
        """Test DummyAlg handles missing ROI."""
        alg = DummyAlg()
        
        # Should use default
        assert alg.roi == [0, 0, 512, 512]
    
    def test_process_large_number_of_frames(self):
        """Test processing many frames doesn't cause issues."""
        args = {
            "id": "test_session",
            "roi": [0, 0, 512, 512],
            "gooey_args": {
                # "total_frames": 1000,
                "zsize": 10,
            }
        }
        alg = DummyAlg(args=args)
        img = np.zeros((512, 512), dtype=np.uint16)
        
        # Process 1000 frames
        for i in range(1000):
            alg.process_sample(img, sample_ndx=i)
        
        assert alg.sample_count == 1000
        assert alg.volume_count == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])