# CLEF Command-Line Interface

Command-line interface for running CLEF (Closed-Loop Experiment Framework) experiments with YAML configuration files.

## Installation

After adding the entry point to `setup.py`, install CLEF in development mode:

```bash
pip install -e .
```

This registers the `clef-cli` command globally.

## Usage

### Basic Usage

Run an experiment with custom configuration files:

```bash
clef-cli --hardware hw.yaml --experiment exp.yaml --algorithm alg.yaml
```

### Validate Configuration Only

Check that your configuration files are valid without running an experiment:

```bash
clef-cli --validate-config --hardware hw.yaml --experiment exp.yaml --algorithm alg.yaml
```

### Use Default Configurations

If you omit a configuration file, CLEF uses the default from `config/defaults/`:

```bash
# Use default experiment and algorithm configs, custom hardware
clef-cli --hardware my_hardware.yaml

# Use all defaults
clef-cli --validate-config
```

### Logging Options

Control output verbosity:

```bash
# Verbose debug output
clef-cli --verbose --hardware hw.yaml

# Suppress all output except errors
clef-cli --quiet --hardware hw.yaml
```

### Help and Version

```bash
# Display help message
clef-cli --help

# Show version
clef-cli --version
```

## Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--hardware PATH` | Optional | Path to hardware configuration YAML |
| `--experiment PATH` | Optional | Path to experiment configuration YAML |
| `--algorithm PATH` | Optional | Path to algorithm configuration YAML |
| `--validate-config` | Flag | Validate configs without running experiment |
| `--verbose`, `-v` | Flag | Enable verbose debug logging |
| `--quiet`, `-q` | Flag | Suppress all output except errors |
| `--version` | Flag | Show version and exit |
| `--help`, `-h` | Flag | Show help message and exit |

## Configuration Files

CLEF uses three YAML configuration files:

### 1. Hardware Configuration (`hardware.yaml`)

Defines hardware backend, devices, and stimulus interface:

```yaml
backend: "pycromanager"
mm_config_path: "/path/to/micromanager.cfg"
stim_interface: "InvCore-SpinningDisk-639"
microscope_name: "Nikon Ti2"

devices:
  camera:
    device_name: "Prime95B"
    device_type: "camera"
    properties:
      Exposure: 100.0

stimulus_devices:
  InvCore-SpinningDisk-639:
    type: "widefield_laser"
    voltage_device: "DAC639"
    voltage_property: "Volts"
    ttl_device: "TTL1-8"
    ttl_line: "TTL-4"
    max_volts: 3.5
```

### 2. Experiment Configuration (`experiment.yaml`)

Defines acquisition parameters and subject metadata:

```yaml
experiment_name: "my_experiment"
experimenter: "Jane Doe"
output_dir: "./data"
save_images: true
save_metadata: true

acquisition:
  num_frames: 1000
  frame_rate: 10.0
  z_stack: true
  z_planes: 5
  z_step: 1.0
  baseline_frames: 100

subject:
  subject_id: "animal_001"
  genotype: "wild_type"
  treatment: "vehicle"
```

### 3. Algorithm Configuration (`algorithm.yaml`)

Defines closed-loop algorithm and stimulus parameters:

```yaml
algorithm_type: "Brainalyzer"
enable_gui: true
gui_mode: "neural_imaging"

algorithm_params:
  stim_threshold_pos: 0.06
  stim_threshold_neg: 0.06
  stim_cooldown_frames: 900

stimulus_params:
  enabled: true
  duration_frames_options: [48, 96]
  intensity_percent_options: [10, 20, 50]
  randomize: true
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Configuration error (missing file, validation failure) |
| 1 | Experiment error (hardware failure, runtime error) |

## Examples

### Example 1: Quick Validation

Validate your configs before starting a long experiment:

```bash
clef-cli --validate-config \
  --hardware configs/inverted_scope.yaml \
  --experiment configs/calcium_imaging.yaml \
  --algorithm configs/threshold_trigger.yaml
```

### Example 2: Run with Custom Configs

Run a full experiment:

```bash
clef-cli \
  --hardware configs/my_hardware.yaml \
  --experiment configs/my_experiment.yaml \
  --algorithm configs/my_algorithm.yaml
```

### Example 3: Use Defaults with Override

Use default algorithm and experiment, custom hardware only:

```bash
clef-cli --hardware configs/special_hardware.yaml
```

### Example 4: Verbose Logging for Debugging

Enable debug output to troubleshoot issues:

```bash
clef-cli --verbose \
  --hardware configs/debug_hw.yaml \
  --validate-config
```

## Error Messages

The CLI provides helpful error messages:

### Missing File
```
ERROR - Hardware config file not found: /path/to/missing.yaml
Configuration file validation failed
```

### Invalid YAML
```
ERROR - YAML parsing error in experiment.yaml: ...
Failed to load configurations: ...
```

### Validation Failure
```
ERROR - Backend must be one of ['pycromanager', 'pymmcore', 'dummy'], got 'invalid'
Configuration validation error: ...
```

### Experiment Error
```
ERROR - Experiment failed: Hardware initialization error
Cleanup error: ...
```

## Integration with Tests

Run CLI tests:

```bash
pytest tests/cli/test_cli.py -v
```

Run integration tests:

```bash
pytest tests/engine/integration_test_engine.py -v
```

## Troubleshooting

### Command Not Found

If `clef-cli` command is not found after installation:

1. Ensure you installed with `pip install -e .`
2. Check that the entry point is in `setup.py`:
   ```python
   entry_points={
       'console_scripts': [
           'clef-cli=cli:cli_entry_point',
       ],
   }
   ```
3. Reinstall: `pip uninstall clef && pip install -e .`

### Import Errors

If you get import errors:

1. Ensure you're running from the correct directory
2. Check that `config/`, `engine/`, `hardware/` are present
3. Verify Python path includes the package root

### Configuration Not Loading

If configs aren't loading:

1. Check file paths are correct (absolute or relative to current directory)
2. Verify YAML syntax with a YAML validator
3. Run with `--verbose` to see detailed loading messages

## Development

### Running Tests

```bash
# All CLI tests
pytest tests/cli/test_cli.py

# Specific test
pytest tests/cli/test_cli.py::TestMainFunction::test_main_validate_only

# With coverage
pytest tests/cli/test_cli.py --cov=cli --cov-report=html
```

### Adding New Arguments

To add a new command-line argument:

1. Edit `cli.py` → `setup_argparser()`
2. Add argument to parser
3. Handle in `main()` or relevant function
4. Add tests in `test_cli.py`
5. Update this README

### Code Structure

```
cli.py
├── setup_argparser()      # Configure argument parser
├── validate_config_paths()  # Check file existence
├── load_configs()         # Load YAML with ConfigManager
├── validate_configs()     # Validate loaded configs
├── run_experiment()       # Execute CLEF experiment
├── main()                 # Main entry point
└── cli_entry_point()      # Wrapper for setup.py
```

## Future Enhancements

Planned features:

- [ ] `--list-devices` to show available hardware
- [ ] `--dry-run` to simulate without hardware
- [ ] `--resume` to continue interrupted experiments
- [ ] `--profile` for performance profiling
- [ ] Multiple experiment batch mode
- [ ] Real-time progress display

## Related Documentation

- [Configuration Manager](../config/README.md)
- [Hardware Abstraction](../hardware/README.md)
- [Algorithm Development](../algorithms/README.md)
- [Integration Testing](../tests/README.md)
