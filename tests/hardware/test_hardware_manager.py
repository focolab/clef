"""
Unit tests for HardwareManager class.

Tests the main hardware manager interface, backend selection, and initialization.
"""

import pytest
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the engine and supporting modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from hardware.hardware_manager import HardwareManager
from hardware.backends.dummy_backend import DummyHardwareBackend
from hardware.backends.micromanager_backend import MicroManagerBackend
from config.config_manager import HardwareConfig
from hardware.backends.lib import DummyMMC

@pytest.fixture
def minimal_hardware_config():
    """Create minimal hardware config for testing."""
    return HardwareConfig(
        backend="dummy",
        stim_interface="dummy"
    )


@pytest.fixture
def pycromanager_config():
    """Create pycromanager hardware config for testing."""
    return HardwareConfig(
        backend="pycromanager",
        stim_interface="dummy",
        mm_config_path="test_config.cfg"
    )


class TestHardwareManagerInitialization:
    """Test HardwareManager initialization and configuration."""
    
    def test_hardware_manager_initialization(self, minimal_hardware_config):
        """Test HardwareManager initializes with config."""
        hw_manager = HardwareManager(minimal_hardware_config)
        hw_manager.initialize()
        assert hw_manager.is_initialized is True
        assert hw_manager._initialized is True
    
    def test_hardware_manager_not_initialized_by_default(self, minimal_hardware_config):
        """Test HardwareManager is not initialized until initialize() is called."""
        hw_manager = HardwareManager(minimal_hardware_config)
        assert hw_manager.is_initialized is False
        assert hw_manager._initialized is False
    
    def test_backend_selection_dummy(self, minimal_hardware_config):
        """Test correct backend is selected from config for dummy."""
        manager = HardwareManager(minimal_hardware_config)
        manager.initialize()
        assert isinstance(manager.get_backend(), DummyHardwareBackend)
    
    def test_backend_selection_pycromanager(self, pycromanager_config):
        """Test correct backend is selected from config for pycromanager."""
        manager = HardwareManager(pycromanager_config)
        # Don't actually initialize - just check selection
        assert isinstance(manager._backend, MicroManagerBackend)
    
    def test_backend_selection_test_alias(self):
        """Test 'test' backend alias selects dummy backend."""
        config = HardwareConfig(backend="test", stim_interface="dummy")
        manager = HardwareManager(config)
        manager.initialize()
        assert isinstance(manager.get_backend(), DummyHardwareBackend)
    
    def test_backend_selection_invalid(self):
        """Test invalid backend raises ValueError."""
        config = HardwareConfig(backend="dummy", stim_interface="dummy")
        config.backend = "invalid_backend"  # overwrite after pydantic validation
        with pytest.raises(ValueError, match="Unknown backend type"):
            HardwareManager(config)
    
    def test_initialize_with_kwargs(self, minimal_hardware_config):
        """Test initialize accepts kwargs for backend-specific params."""
        manager = HardwareManager(minimal_hardware_config)
        # Should not raise - dummy backend accepts input_file kwarg
        manager.initialize(input_file="test.tiff")
        assert manager.is_initialized is True
    
    def test_initialize_idempotent(self, minimal_hardware_config):
        """Test initialize can be called multiple times safely."""
        manager = HardwareManager(minimal_hardware_config)
        manager.initialize()
        assert manager.is_initialized is True
        
        # Second call should not raise
        manager.initialize()
        assert manager.is_initialized is True
    
    def test_close_cleans_up(self, minimal_hardware_config):
        """Test close() properly cleans up resources."""
        manager = HardwareManager(minimal_hardware_config)
        manager.initialize()
        assert manager.is_initialized is True
        
        manager.close()
        assert manager.is_initialized is False
    
    def test_close_idempotent(self, minimal_hardware_config):
        """Test close() can be called multiple times safely."""
        manager = HardwareManager(minimal_hardware_config)
        manager.initialize()
        manager.close()
        assert manager.is_initialized is False
        
        # Second call should not raise
        manager.close()
        assert manager.is_initialized is False


class TestHardwareManagerInterfaces:
    """Test HardwareManager provides correct interfaces."""
    
    def test_camera_interface_access(self, minimal_hardware_config):
        """Test camera interface is accessible after initialization."""
        manager = HardwareManager(minimal_hardware_config)
        manager.initialize()
        
        camera = manager.camera
        assert camera is not None
        assert hasattr(camera, 'acquire_frame')
        assert hasattr(camera, 'start_acquisition')
        assert hasattr(camera, 'stop_acquisition')
    
    def test_camera_interface_before_initialization(self, minimal_hardware_config):
        """Test camera interface raises error before initialization."""
        manager = HardwareManager(minimal_hardware_config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = manager.camera
    
    def test_stage_interface_access(self, minimal_hardware_config):
        """Test stage interface is accessible after initialization."""
        manager = HardwareManager(minimal_hardware_config)
        manager.initialize()
        
        stage = manager.stage
        assert stage is not None
        assert hasattr(stage, 'get_position')
        assert hasattr(stage, 'move_to_position')
    
    def test_stage_interface_before_initialization(self, minimal_hardware_config):
        """Test stage interface raises error before initialization."""
        manager = HardwareManager(minimal_hardware_config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = manager.stage
    
    def test_stimulus_interface_access(self, minimal_hardware_config):
        """Test stimulus interface is accessible after initialization."""
        manager = HardwareManager(minimal_hardware_config)
        manager.initialize()
        
        stimulus = manager.stimulus
        assert stimulus is not None
        assert hasattr(stimulus, 'activate_stimulus')
        assert hasattr(stimulus, 'deactivate_stimulus')
    
    def test_stimulus_interface_before_initialization(self, minimal_hardware_config):
        """Test stimulus interface raises error before initialization."""
        manager = HardwareManager(minimal_hardware_config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = manager.stimulus


class TestHardwareManagerMetadata:
    """Test metadata retrieval."""
    
    def test_get_metadata_after_initialization(self, minimal_hardware_config):
        """Test get_metadata returns dict after initialization."""
        manager = HardwareManager(minimal_hardware_config)
        manager.initialize()
        
        metadata = manager.get_metadata()
        assert isinstance(metadata, dict)
        assert 'backend_metadata' in metadata
        assert metadata['backend_metadata']['backend'] == 'dummy'
    
    def test_get_metadata_before_initialization(self, minimal_hardware_config):
        """Test get_metadata returns empty dict before initialization."""
        manager = HardwareManager(minimal_hardware_config)
        
        metadata = manager.get_metadata()
        assert isinstance(metadata, dict)
        assert len(metadata) == 0


class TestHardwareManagerLegacyAccess:
    """Test legacy access methods (get_mmc, get_backend)."""
    
    def test_get_backend(self, minimal_hardware_config):
        """Test get_backend returns backend instance."""
        manager = HardwareManager(minimal_hardware_config)
        manager.initialize()
        
        backend = manager.get_backend()
        assert backend is not None
        assert isinstance(backend, DummyHardwareBackend)
    
    def test_get_backend_before_initialization(self, minimal_hardware_config):
        """Test get_backend raises error before initialization."""
        manager = HardwareManager(minimal_hardware_config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = manager.get_backend()
    
    def test_get_mmc_returns_none_for_dummy(self, minimal_hardware_config):
        """Test get_mmc returns None for non-MM backends."""
        manager = HardwareManager(minimal_hardware_config)
        manager.initialize()
        
        mmc = manager.get_mmc()
        assert isinstance(mmc, DummyMMC.DummyMMC)
    
    def test_get_mmc_returns_mmc_for_micromanager(self, pycromanager_config):
        """Test get_mmc returns MMC object for Micro-Manager backend."""
        mock_mmc = MagicMock()
        mock_core_cls = MagicMock(return_value=mock_mmc)
        mock_pycromanager = MagicMock()
        mock_pycromanager.Core = mock_core_cls

        import sys
        with patch.dict(sys.modules, {'pycromanager': mock_pycromanager}):
            manager = HardwareManager(pycromanager_config)
            manager.initialize()

            mmc = manager.get_mmc()
            assert mmc is not None

