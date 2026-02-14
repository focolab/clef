"""
Unit tests for Algorithm Factory

Tests the factory pattern implementation for creating algorithm instances.
"""

import pytest
import os
import logging
from unittest.mock import Mock, patch, MagicMock

# Import the engine and supporting modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import factory functions
from algorithms.algorithm_factory import (
    create_algorithm,
    list_available_algorithms,
    register_algorithm,
    _registry,
)
from hardware.hardware_manager import HardwareManager


@pytest.fixture
def mock_configs():
    """Create mock configuration objects for testing."""
    # Mock AlgorithmConfig
    algorithm_config = Mock()
    algorithm_config.algorithm_type = "dummy"
    algorithm_config.gui_mode = "neural_imaging"
    algorithm_config.save_algorithm_plot = False
    algorithm_config.algorithm_params = Mock()
    algorithm_config.algorithm_params.stim_cooldown_frames = 900
    algorithm_config.algorithm_params.skip_stimulation_probability = 0.1
    algorithm_config.algorithm_params.delay_stimulation_probability = 0.4
    algorithm_config.algorithm_params.stim_delay_frames_options = [200, 400]
    algorithm_config.algorithm_params.stim_onset_list = []
    algorithm_config.stimulus_params = Mock()
    algorithm_config.stimulus_params.duration_frames_options = [48]
    algorithm_config.stimulus_params.intensity_percent_options = [10]
    
    # Mock ExperimentConfig
    experiment_config = Mock()
    experiment_config.output_dir = "./test_output"
    experiment_config.save_images = True
    experiment_config.save_metadata = True
    experiment_config.save_sample_video = False
    experiment_config.input_recording_path = None
    # experiment_config.z_step_size_um = 1.0
    
    experiment_config.acquisition = Mock()
    experiment_config.acquisition.num_samples = 100
    # experiment_config.acquisition.z_planes = 10
    # experiment_config.acquisition.save_structural_scan = "none"
    # experiment_config.acquisition.baseline_samples = 0
    
    experiment_config.subject = Mock()
    experiment_config.subject.genotype = "test_strain"
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
    
    # Mock HardwareConfig
    hardware_config = Mock()
    hardware_config.backend = "dummy"
    hardware_config.stim_interface = "dummy"
    hardware_config.microscope_name = "test"
    hardware_config.use_static_stim_roi = False
    
    return {
        "algorithm": algorithm_config,
        "experiment": experiment_config,
        "hardware": hardware_config,
    }


@pytest.fixture
def hardware_manager(mock_configs):
    """Create and initialize a hardware manager for testing."""
    from config.config_manager import HardwareConfig
    
    hardware_config = HardwareConfig(
        backend="dummy",
        stim_interface="dummy",
        microscope_name="test",
    )
    
    hw_manager = HardwareManager(hardware_config)
    hw_manager.initialize()
    yield hw_manager
    hw_manager.close()


class TestAlgorithmFactory:
    """Test suite for algorithm factory."""
    
    def test_list_available_algorithms(self):
        """Test that we can list available algorithms."""
        algorithms = list_available_algorithms()
        
        # Should at least have dummy algorithm
        assert isinstance(algorithms, list)
        assert len(algorithms) > 0
        assert "dummy" in algorithms or "Dummy algorithm (does nothing)" in algorithms
    
    def test_create_dummy_algorithm(self, mock_configs, hardware_manager):
        """Test creation of dummy algorithm (actual instance, not mocked)."""
        mock_configs["algorithm"].algorithm_type = "dummy"
        
        # Create real dummy algorithm
        alg = create_algorithm(
            algorithm_config=mock_configs["algorithm"],
            experiment_config=mock_configs["experiment"],
            hardware_manager=hardware_manager,
        )
        
        # Should be actual DummyAlg instance
        from algorithms.dummy import DummyAlg
        assert isinstance(alg, DummyAlg)
        
        # Should be functional
        alg.initialize_model()
        import numpy as np
        img = np.zeros((512, 512), dtype=np.uint16)
        alg.process_sample(img, sample_ndx=0)
        stim_params, cooldown = alg.check_stim(0, 0)
        assert stim_params == {}
    
    def test_create_algorithm_with_local_handles(self, mock_configs, hardware_manager):
        """Test that local_handles are passed to algorithm."""
        mock_configs["algorithm"].algorithm_type = "dummy"
        mock_mmc = Mock()
        
        # Use real dummy algorithm
        alg = create_algorithm(
            algorithm_config=mock_configs["algorithm"],
            experiment_config=mock_configs["experiment"],
            hardware_manager=hardware_manager,
            local_handles={"mmc": mock_mmc}
        )
        
        # Should have local handles
        from algorithms.dummy import DummyAlg
        assert isinstance(alg, DummyAlg)
        assert alg.local_handles["mmc"] == mock_mmc
    
    def test_unknown_algorithm_raises_error(self, mock_configs, hardware_manager):
        """Test that unknown algorithm type raises ValueError."""
        mock_configs["algorithm"].algorithm_type = "nonexistent_algorithm"
        
        with pytest.raises(ValueError) as excinfo:
            create_algorithm(
                algorithm_config=mock_configs["algorithm"],
                experiment_config=mock_configs["experiment"],
                hardware_manager=hardware_manager,
            )
        
        assert "Unknown algorithm type" in str(excinfo.value)
        assert "nonexistent_algorithm" in str(excinfo.value)
    
    def test_register_custom_algorithm(self, mock_configs, hardware_manager):
        """Test registering a custom algorithm."""
        # Create mock custom algorithm class
        mock_custom_alg = Mock()
        mock_custom_instance = Mock()
        mock_custom_alg.return_value = mock_custom_instance
        
        # Register it
        register_algorithm("custom_test", mock_custom_alg)
        
        # Should now be able to create it
        mock_configs["algorithm"].algorithm_type = "custom_test"
        
        alg = create_algorithm(
            algorithm_config=mock_configs["algorithm"],
            experiment_config=mock_configs["experiment"],
            hardware_manager=hardware_manager,
        )
        
        assert alg == mock_custom_instance
        mock_custom_alg.assert_called_once()
    
    def test_algorithm_import_error_handling(self, mock_configs, hardware_manager):
        """Test graceful handling of import errors."""
        # Force an import error by using non-existent algorithm
        mock_configs["algorithm"].algorithm_type = "BrokenAlgorithm"
        
        # Should raise ValueError (algorithm not in registry)
        with pytest.raises(ValueError):
            create_algorithm(
                algorithm_config=mock_configs["algorithm"],
                experiment_config=mock_configs["experiment"],
                hardware_manager=hardware_manager,
            )


class TestAlgorithmRegistry:
    """Test the registry pattern implementation."""
    
    def test_registry_initialization(self):
        """Test that registry initializes on first access."""
        # Access should trigger initialization
        algorithms = _registry.list_algorithms()
        assert len(algorithms) > 0
    
    def test_registry_caching(self):
        """Test that registry doesn't re-initialize."""
        # First call
        algs1 = _registry.list_algorithms()
        
        # Second call should return same list
        algs2 = _registry.list_algorithms()
        
        assert algs1 == algs2
    
    def test_get_algorithm_class(self):
        """Test retrieving algorithm class from registry."""
        with patch("algorithms.algorithm_factory.DummyAlg") as mock_dummy:
            _registry.register("test_dummy", mock_dummy.DummyAlg)
            
            alg_class = _registry.get("test_dummy")
            assert alg_class == mock_dummy.DummyAlg


class TestFactoryLogging:
    """Test logging behavior of factory."""
    
    def test_logs_algorithm_creation(self, mock_configs, hardware_manager, caplog):
        """Test that algorithm creation is logged."""
        mock_configs["algorithm"].algorithm_type = "dummy"
        
        with caplog.at_level(logging.INFO):
            with patch("algorithms.algorithm_factory.DummyAlg"):
                create_algorithm(
                    algorithm_config=mock_configs["algorithm"],
                    experiment_config=mock_configs["experiment"],
                    hardware_manager=hardware_manager,
                )
        
        # Should log creation
        assert "Creating algorithm: dummy" in caplog.text
        assert "Successfully created algorithm" in caplog.text
    
    def test_logs_algorithm_not_found(self, mock_configs, hardware_manager, caplog):
        """Test that missing algorithms are logged."""
        mock_configs["algorithm"].algorithm_type = "missing_algorithm"
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                create_algorithm(
                    algorithm_config=mock_configs["algorithm"],
                    experiment_config=mock_configs["experiment"],
                    hardware_manager=hardware_manager,
                )
        
        assert "Unknown algorithm type" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])