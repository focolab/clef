"""
Test suite for ConfigManager

Tests loading, validation, and merging of configuration files.
Run with: pytest test_config_manager.py -v
"""

import pytest
import os
import yaml
import tempfile
import shutil
from pathlib import Path
from pydantic import ValidationError

# Import the engine and supporting modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.config_manager import (
    ConfigManager,
    HardwareConfig,
    ExperimentConfig,
    AlgorithmConfig,
    DeviceConfig,
    IlluminationChannel,
    AcquisitionConfig,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_config_dir():
    """Create temporary directory for test configs."""
    temp_dir = Path(tempfile.mkdtemp())
    defaults_dir = temp_dir / "config" / "defaults"
    defaults_dir.mkdir(parents=True)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def minimal_hardware_yaml(temp_config_dir):
    """Create minimal hardware YAML."""
    config = {
        'backend': 'dummy',
        'stim_interface': 'dummy',
    }
    
    path = temp_config_dir / "hardware_minimal.yaml"
    with open(path, 'w') as f:
        yaml.dump(config, f)
    
    return path


@pytest.fixture
def full_hardware_yaml(temp_config_dir):
    """Create complete hardware YAML."""
    config = {
        'backend': 'pycromanager',
        'mm_config_path': '/path/to/MM_config.cfg',
        'stim_interface': 'InvCoreLDIPolygon',
        'devices': {
            'camera': {
                'device_name': 'Prime BSI',
                'device_type': 'camera',
                'properties': {
                    'Exposure': 50.0,
                    'Binning': 2,
                    'TriggerMode': 'External'
                }
            },
            'stage': {
                'device_name': 'ASI XY Stage',
                'device_type': 'stage',
                'properties': {
                    'Speed': 2.0
                }
            }
        },
        'illumination_channels': [
            {
                'name': '488nm',
                'device': 'DAC488',
                'wavelength': 488,
                'power': 75.0,
                'exposure_ms': 50.0
            },
            {
                'name': '561nm',
                'device': 'DAC561',
                'wavelength': 561,
                'power': 60.0,
                'exposure_ms': 50.0
            }
        ],
        'polygon_calibration_path': '/path/to/calibration.json'
    }
    
    path = temp_config_dir / "hardware_full.yaml"
    with open(path, 'w') as f:
        yaml.dump(config, f)
    
    return path


@pytest.fixture
def experiment_yaml(temp_config_dir):
    """Create experiment YAML."""
    config = {
        'experiment_name': 'test_experiment',
        'experimenter': 'Test User',
        'output_dir': './test_data',
        'save_images': True,
        'save_metadata': True,
        'acquisition': {
            'num_frames': 200,
            'frame_rate': 20.0,
            'z_stack': True,
            'z_start': 0.0,
            'z_end': 10.0,
            'z_step': 2.0,
            'z_planes': 5
        },
        'subject': {
            'subject_id': 'subject_001',
            'subject_type': 'C. elegans',
            'genotype': 'N2',
            'treatment': 'ATR 1mM',
            'notes': 'Test notes'
        }
    }
    
    path = temp_config_dir / "experiment.yaml"
    with open(path, 'w') as f:
        yaml.dump(config, f)
    
    return path


@pytest.fixture
def algorithm_yaml(temp_config_dir):
    """Create algorithm YAML."""
    config = {
        'algorithm_type': 'brainalyzer',
        'enable_gui': True,
        'algorithm_params': {
            'threshold': 0.5,
            'window_size': 10
        },
        'stimulus_params': {
            'enabled': True,
            'randomize': False,
            'power': 80.0,
            'duration_ms': 100.0
        }
    }
    
    path = temp_config_dir / "algorithm.yaml"
    with open(path, 'w') as f:
        yaml.dump(config, f)
    
    return path


@pytest.fixture
def default_configs(temp_config_dir):
    """Create default config files."""
    defaults_dir = temp_config_dir / "config" / "defaults"
    
    # Hardware default
    hardware_default = {
        'backend': 'dummy',
        'stim_interface': 'dummy',
        'devices': {
            'camera': {
                'device_name': 'DefaultCamera',
                'device_type': 'camera',
                'properties': {'Exposure': 100.0}
            }
        }
    }
    with open(defaults_dir / "hardware_default.yaml", 'w') as f:
        yaml.dump(hardware_default, f)
    
    # Experiment default
    experiment_default = {
        'experiment_name': 'default_experiment',
        'experimenter': 'Unknown',
        'output_dir': './data',
        'acquisition': {
            'num_frames': 100,
            'z_stack': False
        }
    }
    with open(defaults_dir / "experiment_default.yaml", 'w') as f:
        yaml.dump(experiment_default, f)
    
    # Algorithm default
    algorithm_default = {
        'algorithm_type': 'dummy',
        'enable_gui': False,
        'stimulus_params': {'enabled': False}
    }
    with open(defaults_dir / "algorithm_default.yaml", 'w') as f:
        yaml.dump(algorithm_default, f)
    
    return defaults_dir


# ============================================================================
# Test ConfigManager Initialization
# ============================================================================

def test_config_manager_init(temp_config_dir):
    """Test ConfigManager initialization."""
    cm = ConfigManager(package_root=temp_config_dir)
    
    assert cm.package_root == temp_config_dir
    assert cm.defaults_dir == temp_config_dir / "config" / "defaults"
    assert cm.hardware_config is None
    assert cm.experiment_config is None
    assert cm.algorithm_config is None


# ============================================================================
# Test Hardware Config Loading
# ============================================================================

def test_load_minimal_hardware_config(temp_config_dir, minimal_hardware_yaml, default_configs):
    """Test loading minimal hardware config."""
    cm = ConfigManager(package_root=temp_config_dir)
    config = cm.load_hardware_config(minimal_hardware_yaml)
    
    assert isinstance(config, HardwareConfig)
    assert config.backend == 'dummy'
    assert config.stim_interface == 'dummy'


def test_load_full_hardware_config(temp_config_dir, full_hardware_yaml, default_configs):
    """Test loading full hardware config with devices and channels."""
    cm = ConfigManager(package_root=temp_config_dir)
    config = cm.load_hardware_config(full_hardware_yaml)
    
    assert config.backend == 'pycromanager'
    assert config.stim_interface == 'InvCoreLDIPolygon'
    assert 'camera' in config.devices
    assert config.devices['camera'].device_name == 'Prime BSI'
    assert config.devices['camera'].properties.Exposure == 50.0
    assert len(config.illumination_channels) == 2
    assert config.illumination_channels[0].wavelength == 488


def test_hardware_config_validation_invalid_backend(temp_config_dir):
    """Test hardware config validation rejects invalid backend."""
    invalid_config = {
        'backend': 'invalid_backend',
        'stim_interface': 'dummy'
    }
    
    path = temp_config_dir / "invalid_hardware.yaml"
    with open(path, 'w') as f:
        yaml.dump(invalid_config, f)
    
    cm = ConfigManager(package_root=temp_config_dir)
    
    with pytest.raises(ValidationError) as exc_info:
        cm.load_hardware_config(path)
    
    assert "Backend must be one of" in str(exc_info.value)


# ============================================================================
# Test Experiment Config Loading
# ============================================================================

def test_load_experiment_config(temp_config_dir, experiment_yaml, default_configs):
    """Test loading experiment config."""
    cm = ConfigManager(package_root=temp_config_dir)
    config = cm.load_experiment_config(experiment_yaml)
    
    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == 'test_experiment'
    assert config.acquisition.num_frames == 200
    assert config.acquisition.z_stack is True
    assert config.subject.subject_id == 'subject_001'


def test_experiment_config_defaults(temp_config_dir, default_configs):
    """Test experiment config loads with defaults when no user config."""
    cm = ConfigManager(package_root=temp_config_dir)
    config = cm.load_experiment_config()
    
    assert config.experiment_name == 'default_experiment'
    assert config.experimenter == 'Unknown'
    assert config.acquisition.num_frames == 100


def test_acquisition_config_validation_negative_frames(temp_config_dir):
    """Test acquisition config rejects negative frame count."""
    invalid_config = {
        'experiment_name': 'test',
        'acquisition': {
            'num_frames': -10
        }
    }
    
    path = temp_config_dir / "invalid_experiment.yaml"
    with open(path, 'w') as f:
        yaml.dump(invalid_config, f)
    
    cm = ConfigManager(package_root=temp_config_dir)
    
    with pytest.raises(ValidationError) as exc_info:
        cm.load_experiment_config(path)
    
    assert "greater than 0" in str(exc_info.value)


def test_z_stack_validation_invalid_range(temp_config_dir):
    """Test z-stack validation rejects invalid z-range."""
    invalid_config = {
        'experiment_name': 'test',
        'acquisition': {
            'num_frames': 100,
            'z_stack': True,
            'z_start': 10.0,
            'z_end': 5.0,  # end < start - invalid
            'z_planes': 5
        }
    }
    
    path = temp_config_dir / "invalid_zstack.yaml"
    with open(path, 'w') as f:
        yaml.dump(invalid_config, f)
    
    cm = ConfigManager(package_root=temp_config_dir)
    
    with pytest.raises(ValidationError) as exc_info:
        cm.load_experiment_config(path)
    
    assert "z_end must be greater than z_start" in str(exc_info.value)


# ============================================================================
# Test Algorithm Config Loading
# ============================================================================

def test_load_algorithm_config(temp_config_dir, algorithm_yaml, default_configs):
    """Test loading algorithm config."""
    cm = ConfigManager(package_root=temp_config_dir)
    config = cm.load_algorithm_config(algorithm_yaml)
    
    assert isinstance(config, AlgorithmConfig)
    assert config.algorithm_type == 'brainalyzer'
    assert config.enable_gui is True
    assert config.algorithm_params.threshold == 0.5
    assert config.stimulus_params.enabled is True


def test_algorithm_config_defaults(temp_config_dir, default_configs):
    """Test algorithm config loads with defaults."""
    cm = ConfigManager(package_root=temp_config_dir)
    config = cm.load_algorithm_config()
    
    assert config.algorithm_type == 'dummy'
    assert config.enable_gui is False
    assert config.stimulus_params.enabled is False


# ============================================================================
# Test Config Merging
# ============================================================================

def test_config_merge_deep_override(temp_config_dir, default_configs):
    """Test deep merging of user config with defaults."""
    # Create user override that only changes some fields
    user_override = {
        'backend': 'pycromanager',  # Override
        'devices': {
            'camera': {
                'device_name': 'UserCamera',  # Override
                'properties': {
                    'Exposure': 50.0  # Override
                }
            }
        }
        # stim_interface should come from default
    }
    
    user_path = temp_config_dir / "user_hardware.yaml"
    with open(user_path, 'w') as f:
        yaml.dump(user_override, f)
    
    cm = ConfigManager(package_root=temp_config_dir)
    config = cm.load_hardware_config(user_path)
    
    # Check overrides applied
    assert config.backend == 'pycromanager'
    assert config.devices['camera'].device_name == 'UserCamera'
    assert config.devices['camera'].properties.Exposure == 50.0
    
    # Check default preserved
    assert config.stim_interface == 'dummy'


def test_merge_configs_method(temp_config_dir):
    """Test merge_configs method directly."""
    cm = ConfigManager(package_root=temp_config_dir)
    
    default = {
        'a': 1,
        'b': {'c': 2, 'd': 3},
        'e': [1, 2, 3]
    }
    
    override = {
        'a': 10,  # Override scalar
        'b': {'c': 20},  # Override nested scalar, keep 'd'
        'f': 100  # Add new key
    }
    
    merged = cm.merge_configs(default, override)
    
    assert merged['a'] == 10
    assert merged['b']['c'] == 20
    assert merged['b']['d'] == 3  # Preserved from default
    assert merged['e'] == [1, 2, 3]  # Preserved from default
    assert merged['f'] == 100  # New key added


# ============================================================================
# Test Loading All Configs
# ============================================================================

def test_load_all_configs(temp_config_dir, minimal_hardware_yaml, 
                         experiment_yaml, algorithm_yaml, default_configs):
    """Test loading all three configs at once."""
    cm = ConfigManager(package_root=temp_config_dir)
    cm.load_all_configs(
        hardware_path=minimal_hardware_yaml,
        experiment_path=experiment_yaml,
        algorithm_path=algorithm_yaml
    )
    
    assert cm.hardware_config is not None
    assert cm.experiment_config is not None
    assert cm.algorithm_config is not None
    
    assert cm.hardware_config.backend == 'dummy'
    assert cm.experiment_config.experiment_name == 'test_experiment'
    assert cm.algorithm_config.algorithm_type == 'brainalyzer'


def test_load_all_configs_with_defaults_only(temp_config_dir, default_configs):
    """Test loading all configs using defaults only."""
    cm = ConfigManager(package_root=temp_config_dir)
    cm.load_all_configs()
    
    assert cm.hardware_config.backend == 'dummy'
    assert cm.experiment_config.experiment_name == 'default_experiment'
    assert cm.algorithm_config.algorithm_type == 'dummy'


# ============================================================================
# Test Config Validation
# ============================================================================

def test_validate_config_success(temp_config_dir, default_configs):
    """Test config validation passes for compatible configs."""
    cm = ConfigManager(package_root=temp_config_dir)
    cm.load_all_configs()
    
    assert cm.validate_config() is True


def test_validate_config_before_loading(temp_config_dir):
    """Test validate_config fails if configs not loaded."""
    cm = ConfigManager(package_root=temp_config_dir)
    
    assert cm.validate_config() is False


def test_validate_config_warns_backend_stim_mismatch(temp_config_dir, default_configs, caplog):
    """Test validation warns about backend/stim mismatch."""
    # Create config with dummy backend but real stim
    mismatch_config = {
        'backend': 'dummy',
        'stim_interface': 'InvCoreLDIPolygon'  # Real stim with dummy backend
    }
    
    path = temp_config_dir / "mismatch_hardware.yaml"
    with open(path, 'w') as f:
        yaml.dump(mismatch_config, f)
    
    cm = ConfigManager(package_root=temp_config_dir)
    cm.load_all_configs(hardware_path=path)
    
    cm.validate_config()
    
    # Check for warning in logs
    assert any("may not work" in record.message for record in caplog.records)


# ============================================================================
# Test Error Handling
# ============================================================================

def test_load_nonexistent_file(temp_config_dir):
    """Test loading non-existent file raises error."""
    cm = ConfigManager(package_root=temp_config_dir)
    
    with pytest.raises(FileNotFoundError):
        cm.load_hardware_config(temp_config_dir / "nonexistent.yaml")


def test_load_invalid_yaml(temp_config_dir):
    """Test loading invalid YAML raises error."""
    invalid_path = temp_config_dir / "invalid.yaml"
    with open(invalid_path, 'w') as f:
        f.write("invalid: yaml: content: [unclosed")
    
    cm = ConfigManager(package_root=temp_config_dir)
    
    with pytest.raises(yaml.YAMLError):
        cm.load_yaml(invalid_path)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])