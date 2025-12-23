"""
Comprehensive test suite for Data Interface Abstraction Layer.

Tests are organized into categories:
1. DataInterface Abstract Base Class Tests
2. ImageDataInterface Tests
3. HardwareManager Data Integration Tests
4. ClosedLoopEngine Data Refactoring Tests
5. Data Format Configuration Tests
6. Backward Compatibility Tests
7. Future Extensibility Tests
"""

import pytest
import os
import sys
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tifffile as tf

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hardware.data_interface import DataInterface
from hardware.image_data_interface import ImageDataInterface, create_image_data_interface
from hardware.camera_interface import CameraInterface
from hardware.backends.dummy_backend import DummyCamera
from hardware.hardware_manager import HardwareManager
from engine.closed_loop_engine import ClosedLoopEngine
from config.config_manager import (
    HardwareConfig,
    ExperimentConfig,
    AlgorithmConfig,
    AcquisitionConfig,
    SubjectMetadata,
    DevOptions,
    AlgorithmParameters,
    StimulusParameters,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def dummy_camera():
    """Create dummy camera for testing."""
    return DummyCamera(width=512, height=512)


@pytest.fixture
def dummy_camera_with_file(temp_output_dir):
    """Create dummy camera with test TIFF file."""
    # Create test TIFF
    test_data = np.random.randint(0, 65536, size=(10, 256, 256), dtype=np.uint16)
    filepath = os.path.join(temp_output_dir, "test_input.tiff")
    tf.imwrite(filepath, test_data)
    
    return DummyCamera(width=256, height=256, input_file=filepath)


@pytest.fixture
def image_data_interface(dummy_camera):
    """Create ImageDataInterface wrapping dummy camera."""
    return ImageDataInterface(dummy_camera)


@pytest.fixture
def minimal_hardware_config():
    """Create minimal hardware config."""
    return HardwareConfig(
        backend="dummy",
        stim_interface="dummy",
        microscope_name="test_scope",
    )


@pytest.fixture
def minimal_experiment_config(temp_output_dir):
    """Create minimal experiment config with data format."""
    return ExperimentConfig(
        experiment_name="test_data_interface",
        output_dir=temp_output_dir,
        save_images=False,
        save_metadata=True,
        acquisition=AcquisitionConfig(
            num_samples=20,
            z_planes=2,
        ),
        subject=SubjectMetadata(genotype="test_strain"),
        dev_options=DevOptions(
            prefill_wb_ops=False,
            send_sms_on_completion=False,
        ),
    )


@pytest.fixture
def minimal_algorithm_config():
    """Create minimal algorithm config."""
    return AlgorithmConfig(
        algorithm_type="dummy",
        algorithm_params=AlgorithmParameters(),
        stimulus_params=StimulusParameters(),
    )


# ============================================================================
# 1. DataInterface Abstract Base Class Tests
# ============================================================================

class TestDataInterfaceAbstractClass:
    """Test DataInterface abstract base class."""
    
    def test_datainterface_is_abstract(self):
        """Test that DataInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataInterface()
    
    def test_datainterface_requires_implementation(self):
        """Test that subclasses must implement required methods."""
        # Create incomplete subclass
        class IncompleteInterface(DataInterface):
            def sample_data(self):
                pass
        
        # Should still fail - missing other required methods
        with pytest.raises(TypeError):
            IncompleteInterface()
    
    def test_datainterface_method_signatures(self):
        """Test that DataInterface defines expected method signatures."""
        # Check abstract methods exist
        assert hasattr(DataInterface, 'sample_data')
        assert hasattr(DataInterface, 'start_sampling')
        assert hasattr(DataInterface, 'stop_sampling')
        assert hasattr(DataInterface, 'get_sample_shape')
        assert hasattr(DataInterface, 'get_sample_dtype')
        assert hasattr(DataInterface, 'get_metadata')
        assert hasattr(DataInterface, 'save_data')
        assert hasattr(DataInterface, 'configure_sampling')
        assert hasattr(DataInterface, 'get_remaining_sample_count')
        assert hasattr(DataInterface, 'clear_buffer')


# ============================================================================
# 2. ImageDataInterface Tests
# ============================================================================

class TestImageDataInterfaceBasics:
    """Test ImageDataInterface basic functionality."""
    
    def test_initialization_with_camera(self, dummy_camera):
        """Test ImageDataInterface initializes with camera."""
        interface = ImageDataInterface(dummy_camera)
        assert interface.camera == dummy_camera
        assert interface.metadata_cache == {}
    
    def test_factory_function(self, dummy_camera):
        """Test create_image_data_interface factory function."""
        interface = create_image_data_interface(dummy_camera)
        assert isinstance(interface, ImageDataInterface)
        assert interface.camera == dummy_camera
    
    def test_data_type_name(self, image_data_interface):
        """Test data_type_name property."""
        assert image_data_interface.data_type_name == "microscopy_image"


class TestImageDataInterfaceSampling:
    """Test ImageDataInterface sampling operations."""
    
    def test_sample_data_returns_array(self, image_data_interface):
        """Test sample_data returns numpy array."""
        sample = image_data_interface.sample_data()
        assert isinstance(sample, np.ndarray)
        assert sample.dtype == np.uint16
    
    def test_sample_data_shape_matches_camera(self, image_data_interface):
        """Test sample_data returns correct shape."""
        sample = image_data_interface.sample_data()
        expected_shape = image_data_interface.get_sample_shape()
        assert sample.shape == expected_shape
    
    def test_get_sample_shape(self, image_data_interface):
        """Test get_sample_shape returns tuple."""
        shape = image_data_interface.get_sample_shape()
        assert isinstance(shape, tuple)
        assert len(shape) == 2  # (height, width)
        assert shape == (512, 512)
    
    def test_get_sample_dtype(self, image_data_interface):
        """Test get_sample_dtype returns uint16."""
        dtype = image_data_interface.get_sample_dtype()
        assert dtype == np.dtype(np.uint16)
    
    def test_start_sampling(self, image_data_interface):
        """Test start_sampling delegates to camera."""
        image_data_interface.start_sampling(buffer_size=100)
        # Should not raise
        assert image_data_interface.camera._acquisition_running is True
    
    def test_stop_sampling(self, image_data_interface):
        """Test stop_sampling delegates to camera."""
        image_data_interface.start_sampling()
        image_data_interface.stop_sampling()
        assert image_data_interface.camera._acquisition_running is False
    
    def test_get_remaining_sample_count(self, image_data_interface):
        """Test get_remaining_sample_count delegates to camera."""
        count = image_data_interface.get_remaining_sample_count()
        assert isinstance(count, int)
        assert count >= 0
    
    def test_clear_buffer(self, image_data_interface):
        """Test clear_buffer delegates to camera."""
        image_data_interface.clear_buffer()
        # Should reset camera counters
        assert image_data_interface.camera._frame_count == 0


class TestImageDataInterfaceConfiguration:
    """Test ImageDataInterface configuration."""
    
    def test_configure_sampling(self, image_data_interface, minimal_experiment_config):
        """Test configure_sampling updates camera settings."""

        minimal_experiment_config.acquisition.num_samples = 5
        minimal_experiment_config.save_images = True
        
        image_data_interface.configure_sampling(minimal_experiment_config)
        
        assert image_data_interface.samples.shape[0] == 5


class TestImageDataInterfaceMetadata:
    """Test ImageDataInterface metadata operations."""
    
    def test_get_metadata_returns_dict(self, image_data_interface):
        """Test get_metadata returns dictionary."""
        metadata = image_data_interface.get_metadata()
        assert isinstance(metadata, dict)
    
    def test_get_metadata_includes_data_type(self, image_data_interface):
        """Test metadata includes data_type field."""
        metadata = image_data_interface.get_metadata()
        assert metadata['data_type'] == 'microscopy_image'
    
    def test_get_metadata_includes_camera_settings(self, image_data_interface):
        """Test metadata includes camera settings."""
        metadata = image_data_interface.get_metadata()
        assert 'exposure_ms' in metadata
        assert 'roi' in metadata
        assert 'image_size' in metadata
    
    def test_get_metadata_includes_shape_dtype(self, image_data_interface):
        """Test metadata includes shape and dtype."""
        metadata = image_data_interface.get_metadata()
        assert 'shape' in metadata
        assert 'dtype' in metadata
        assert metadata['shape'] == (512, 512)
        assert metadata['dtype'] == 'uint16'
    
    def test_get_metadata_caches_result(self, image_data_interface):
        """Test metadata is cached."""
        metadata1 = image_data_interface.get_metadata()
        # Check cache was populated
        assert image_data_interface.metadata_cache == metadata1


class TestImageDataInterfaceSaving:
    """Test ImageDataInterface data saving."""
    
    def test_save_data_creates_tiff(self, image_data_interface, temp_output_dir):
        """Test save_data creates TIFF file."""
        data = np.random.randint(0, 65536, size=(10, 512, 512), dtype=np.uint16)
        filepath = os.path.join(temp_output_dir, "test_save.tiff")
        
        image_data_interface.save_data(data, filepath)
        
        assert os.path.exists(filepath)
    
    def test_save_data_adds_extension(self, image_data_interface, temp_output_dir):
        """Test save_data adds .tiff extension if missing."""
        data = np.random.randint(0, 65536, size=(10, 512, 512), dtype=np.uint16)
        filepath = os.path.join(temp_output_dir, "test_save")
        
        image_data_interface.save_data(data, filepath)
        
        expected_path = filepath + ".tiff"
        assert os.path.exists(expected_path)
    
    def test_save_data_preserves_shape_dtype(self, image_data_interface, temp_output_dir):
        """Test saved data preserves shape and dtype."""
        data = np.random.randint(0, 65536, size=(10, 512, 512), dtype=np.uint16)
        filepath = os.path.join(temp_output_dir, "test_save.tiff")
        
        image_data_interface.save_data(data, filepath)
        
        # Read back and verify
        loaded = tf.imread(filepath)
        assert loaded.shape == data.shape
        assert loaded.dtype == data.dtype
        np.testing.assert_array_equal(loaded, data)
    
    def test_save_data_embeds_metadata(self, image_data_interface, temp_output_dir):
        """Test save_data embeds metadata in TIFF."""
        data = np.random.randint(0, 65536, size=(10, 512, 512), dtype=np.uint16)
        filepath = os.path.join(temp_output_dir, "test_save.tiff")
        
        extra_metadata = {'session_id': 'test_123'}
        image_data_interface.save_data(
            data, filepath, 
            metadata=extra_metadata
        )
        
        # Read back metadata
        with tf.TiffFile(filepath) as tif:
            # TIFF metadata is embedded in description
            assert tif.pages[0].description is not None



# ============================================================================
# 3. HardwareManager Data Integration Tests
# ============================================================================

class TestHardwareManagerDataProperty:
    """Test HardwareManager.data property."""
    
    def test_hardware_manager_has_data_property(self, minimal_hardware_config):
        """Test HardwareManager exposes data property."""
        hw_manager = HardwareManager(minimal_hardware_config)
        hw_manager.initialize()
        
        assert hasattr(hw_manager, 'data')
        assert hw_manager.data is not None
    
    def test_data_property_returns_data_interface(self, minimal_hardware_config):
        """Test data property returns DataInterface instance."""
        hw_manager = HardwareManager(minimal_hardware_config)
        hw_manager.initialize()
        
        data_interface = hw_manager.data
        assert isinstance(data_interface, DataInterface)
    
    def test_data_property_is_image_data_interface(self, minimal_hardware_config):
        """Test data property returns ImageDataInterface for camera backends."""
        hw_manager = HardwareManager(minimal_hardware_config)
        hw_manager.initialize()
        
        data_interface = hw_manager.data
        assert isinstance(data_interface, ImageDataInterface)
    
    def test_data_property_before_initialization(self, minimal_hardware_config):
        """Test data property raises error before initialization."""
        hw_manager = HardwareManager(minimal_hardware_config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = hw_manager.data
    
    def test_data_wraps_camera_interface(self, minimal_hardware_config):
        """Test data interface wraps hardware manager's camera."""
        hw_manager = HardwareManager(minimal_hardware_config)
        hw_manager.initialize()
        
        # Data interface should wrap the camera
        assert hw_manager.data.camera == hw_manager.camera


class TestHardwareManagerMetadataWithData:
    """Test HardwareManager metadata includes data interface info."""
    
    def test_get_metadata_includes_data_metadata(self, minimal_hardware_config):
        """Test hardware metadata includes data interface metadata."""
        hw_manager = HardwareManager(minimal_hardware_config)
        hw_manager.initialize()
        
        metadata = hw_manager.get_metadata()
        assert 'data' in metadata
        assert isinstance(metadata['data'], dict)
    
    def test_data_metadata_includes_data_type(self, minimal_hardware_config):
        """Test data metadata includes data_type field."""
        hw_manager = HardwareManager(minimal_hardware_config)
        hw_manager.initialize()
        
        metadata = hw_manager.get_metadata()
        assert metadata['data']['data_type'] == 'microscopy_image'
    
    def test_data_metadata_includes_shape_dtype(self, minimal_hardware_config):
        """Test data metadata includes shape and dtype."""
        hw_manager = HardwareManager(minimal_hardware_config)
        hw_manager.initialize()
        
        metadata = hw_manager.get_metadata()
        assert 'shape' in metadata['data']
        assert 'dtype' in metadata['data']


class TestHardwareManagerCameraBackwardCompat:
    """Test HardwareManager.camera backward compatibility."""
    
    def test_camera_property_still_exists(self, minimal_hardware_config):
        """Test camera property is still accessible."""
        hw_manager = HardwareManager(minimal_hardware_config)
        hw_manager.initialize()
        
        assert hasattr(hw_manager, 'camera')
        assert hw_manager.camera is not None
    
    def test_camera_property_returns_camera_interface(self, minimal_hardware_config):
        """Test camera property returns CameraInterface."""
        hw_manager = HardwareManager(minimal_hardware_config)
        hw_manager.initialize()
        
        camera = hw_manager.camera
        assert isinstance(camera, CameraInterface)
    
    def test_camera_and_data_use_same_camera(self, minimal_hardware_config):
        """Test camera property and data interface use same camera."""
        hw_manager = HardwareManager(minimal_hardware_config)
        hw_manager.initialize()
        
        # Should be the same underlying camera
        assert hw_manager.data.camera == hw_manager.camera


# ============================================================================
# 4. ClosedLoopEngine Data Refactoring Tests
# ============================================================================

class TestClosedLoopEngineDataTerminology:
    """Test ClosedLoopEngine uses new data-centric terminology."""
    
    def test_engine_has_samples_not_frames(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test engine uses 'samples' instead of 'frames'."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        # Should have samples attribute
        assert hasattr(engine, 'samples')
        # Legacy frames might still exist temporarily
    
    def test_engine_has_sample_count_not_img_count(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test engine uses 'sample_count' instead of 'img_count'."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        assert hasattr(engine, 'sample_count')
        assert engine.sample_count == 0
    
    def test_engine_has_sample_time_list(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test engine uses 'sample_time_list' instead of 'frame_time_list'."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )

        engine.initialize_hardware()
        
        assert hasattr(engine.data_interface, 'sample_time_list')
        assert isinstance(engine.data_interface.sample_time_list, list)
    
    def test_engine_tracks_sample_shape(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test engine tracks sample_shape from data interface."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        assert hasattr(engine, 'sample_shape')
        # Should be None before hardware init
        assert engine.sample_shape is None
    
    def test_engine_tracks_sample_dtype(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test engine tracks sample_dtype from data interface."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        assert hasattr(engine, 'sample_dtype')
        # Should be None before hardware init
        assert engine.sample_dtype is None


class TestClosedLoopEngineDataInterface:
    """Test ClosedLoopEngine uses data interface for acquisition."""
    
    def test_hardware_initialization_sets_sample_properties(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test hardware initialization sets sample shape/dtype."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        engine.initialize_hardware()
        
        # Should get shape and dtype from data interface
        assert engine.sample_shape is not None
        assert engine.sample_dtype is not None
        assert isinstance(engine.sample_shape, tuple)
        assert isinstance(engine.sample_dtype, np.dtype)
    
    def test_prepare_acquisition_creates_sample_storage(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test prepare_acquisition creates samples array with correct shape/dtype."""
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        # Should create samples array
        assert engine.samples is not None
        assert isinstance(engine.samples, np.ndarray)
        
        # Shape should include num_samples and sample dimensions
        expected_samples = minimal_experiment_config.acquisition.num_samples
        assert engine.samples.shape[0] == expected_samples
        assert engine.samples.shape[1:] == engine.sample_shape
        
        # Dtype should match sample_dtype
        assert engine.samples.dtype == engine.sample_dtype
    
    def test_acquisition_loop_uses_data_interface(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test acquisition loop samples via data interface."""
        # Use small frame count
        minimal_experiment_config.acquisition.num_samples = 10
        
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        
        # Track data interface calls
        data_interface = engine.hardware.data
        original_sample = data_interface.sample_data
        sample_calls = []
        
        def track_sample():
            sample_calls.append(True)
            return original_sample()
        
        data_interface.sample_data = track_sample
        
        # Run acquisition
        engine.run_acquisition_loop()
        
        # Should have called sample_data for each frame
        assert len(sample_calls) == 10
    
    def test_acquisition_saves_sample_metadata(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test acquisition saves sample shape/dtype in metadata."""
        minimal_experiment_config.acquisition.num_samples = 5
        
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        engine.run_acquisition_loop()
        
        with patch('utils.wbliveUtils.save_metadata') as mock_save:
            metadata = engine.save_metadata()
            
            # Should include sample properties
            assert 'sample_shape' in metadata
            assert 'sample_dtype' in metadata
            assert metadata['sample_shape'] == engine.sample_shape
            assert metadata['sample_dtype'] == str(engine.sample_dtype)


class TestClosedLoopEngineSaveData:
    """Test ClosedLoopEngine saves data via data interface."""
    
    def test_save_data_uses_data_interface(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test _save_data delegates to data interface."""
        minimal_experiment_config.acquisition.num_samples = 5
        minimal_experiment_config.save_images = True
        
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        engine.initialize_hardware()
        engine.prepare_acquisition()
        engine.initialize_algorithm()
        engine.initialize_stimulus()
        engine.run_acquisition_loop()
        
        # Mock data interface save_data
        with patch.object(engine.hardware.data, 'save_data') as mock_save:
            engine._save_data()
            
            # Should call data interface save_data
            mock_save.assert_called_once()
            
            # Check arguments
            call_args = mock_save.call_args
            assert isinstance(call_args[1]['data'], np.ndarray)
            assert 'filepath' in call_args[1]
    
    def test_save_data_respects_no_save_flag(
        self, minimal_hardware_config, minimal_experiment_config, minimal_algorithm_config
    ):
        """Test _save_data respects no_save_data flag."""
        minimal_experiment_config.save_images = False
        
        engine = ClosedLoopEngine(
            hardware_config=minimal_hardware_config,
            experiment_config=minimal_experiment_config,
            algorithm_config=minimal_algorithm_config,
        )
        
        engine.initialize_hardware()
        engine.prepare_acquisition()
        
        with patch.object(engine.hardware.data, 'save_data') as mock_save:
            engine._save_data()
            
            # Should not call save_data
            mock_save.assert_not_called()

