# CLEF Configuration Manager

## Overview

The ConfigManager implements the YAML + Pydantic configuration strategy defined in the refactoring plan. It provides:

- **Type-safe configuration** with Pydantic validation
- **Default + override merging** for flexible configuration
- **Hardware abstraction** - no hardcoded device names in code
- **Clear error messages** pointing to configuration issues

## Directory Structure

```
config/
├── defaults/           # Package-provided default configs
│   ├── hardware_default.yaml
│   ├── experiment_default.yaml
│   └── algorithm_default.yaml
└── demo/              # Demo/test configurations
    ├── hardware_minimal.yaml
    ├── experiment_minimal.yaml
    └── algorithm_minimal.yaml
```

## Configuration Files

### hardware.yaml

Defines all hardware-specific settings:

```yaml
backend: "pycromanager"  # or "pymmcore", "dummy"
mm_config_path: "/path/to/MMConfig.cfg"
stim_interface: "InvCoreLDIPolygon"

devices:
  camera:
    device_name: "Prime BSI"
    device_type: "camera"
    properties:
      Exposure: 50.0
      Binning: 2

illumination_channels:
  - name: "488nm"
    device: "DAC488"
    wavelength: 488
    power: 75.0
```

### experiment.yaml

Defines experiment parameters:

```yaml
experiment_name: "my_experiment"
experimenter: "Jane Doe"
output_dir: "./data/exp001"

acquisition:
  num_frames: 500
  frame_rate: 10.0
  z_stack: true
  z_start: 0.0
  z_end: 20.0
  z_step: 2.0

subject:
  subject_id: "subject_001"
  genotype: "N2"
```

### algorithm.yaml

Defines closed-loop algorithm settings:

```yaml
algorithm_type: "brainalyzer"
enable_gui: true

algorithm_params:
  threshold: 0.5
  window_size: 10

stimulus_params:
  enabled: true
  power: 80.0
```

## Usage

### Basic Loading

```python
from config.config_manager import ConfigManager

# Initialize
cm = ConfigManager()

# Load all configs with defaults
cm.load_all_configs()

# Access configurations
print(cm.hardware_config.backend)
print(cm.experiment_config.acquisition.num_frames)
print(cm.algorithm_config.algorithm_type)
```

### Loading with User Overrides

```python
cm = ConfigManager()

# Load with user-specific configs
cm.load_all_configs(
    hardware_path="my_hardware.yaml",
    experiment_path="my_experiment.yaml",
    algorithm_path="my_algorithm.yaml"
)

# Validate compatibility
if cm.validate_config():
    print("✓ Configuration valid")
```

### Config Merging Example

**Default** (hardware_default.yaml):
```yaml
backend: "dummy"
devices:
  camera:
    device_name: "DefaultCamera"
    properties:
      Exposure: 100.0
      Binning: 1
```

**User Override** (my_hardware.yaml):
```yaml
backend: "pycromanager"
devices:
  camera:
    properties:
      Exposure: 50.0
```

**Merged Result**:
```yaml
backend: "pycromanager"          # ← from user
devices:
  camera:
    device_name: "DefaultCamera"  # ← from default
    properties:
      Exposure: 50.0              # ← from user
      Binning: 1                  # ← from default
```

### CLI Integration

```python
# Example CLI entry point
import argparse
from config.config_manager import ConfigManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--algorithm", type=str)
    args = parser.parse_args()
    
    cm = ConfigManager()
    cm.load_all_configs(
        hardware_path=args.hardware,
        experiment_path=args.experiment,
        algorithm_path=args.algorithm
    )
    
    # Use configs...
```

## Running Tests

```bash
# Run all tests
pytest test_config_manager.py -v

# Run specific test
pytest test_config_manager.py::test_load_minimal_hardware_config -v

# Run with coverage
pytest test_config_manager.py --cov=config --cov-report=term
```

## Validation

The ConfigManager uses Pydantic for validation:

### Type Validation
```python
# ✗ This will fail validation
acquisition:
  num_frames: "not a number"  # ValidationError: not a valid integer
```

### Range Validation
```python
# ✗ This will fail validation
acquisition:
  num_frames: -10  # ValidationError: greater than 0 required
```

### Custom Validation
```python
# ✗ This will fail validation
backend: "invalid_backend"  # ValidationError: must be pycromanager/pymmcore/dummy
```

### Z-Stack Validation
```python
# ✗ This will fail validation
acquisition:
  z_stack: true
  z_start: 10.0
  z_end: 5.0  # ValidationError: z_end must be > z_start
```

## Migration from Old System

### Before (args dict)
```python
def run_acquisition(args):
    num_frames = args["num-frames"]
    scope = args["microscope_name"]
    if scope == "innovation core":
        # hardware-specific code...
```

### After (Config objects)
```python
def run_acquisition(config: ExperimentConfig):
    num_frames = config.acquisition.num_frames
    # No scope name needed - HardwareManager handles this
```

## Error Messages

ConfigManager provides helpful error messages:

```
ValidationError: 1 validation error for HardwareConfig
backend
  Backend must be one of ['pycromanager', 'pymmcore', 'dummy'], got 'invalid'
  (type=value_error)
```

```
FileNotFoundError: Config file not found: /path/to/nonexistent.yaml
```

```
INFO: Overriding 'devices.camera.exposure': 100.0 → 50.0
```

## Best Practices

1. **Start with minimal configs** - Override only what you need
2. **Use comments** - YAML supports comments, document hardware-specific settings
3. **Validate early** - Call `validate_config()` before running experiments
4. **Version control configs** - Commit user configs for reproducibility
5. **Keep defaults generic** - Package defaults should work for demo/testing

## Troubleshooting

**Problem**: "Not all configs loaded"
- **Solution**: Call `load_all_configs()` before `validate_config()`

**Problem**: Validation error on valid config
- **Solution**: Check YAML indentation (use spaces, not tabs)

**Problem**: Configs not merging as expected
- **Solution**: Use `--validate-config` flag to preview merged config

**Problem**: "Backend is 'dummy' but stim_interface is '...'"
- **Solution**: Warning only - use matching backend/stim for testing