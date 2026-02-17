# CLEF

Closed-Loop Experimental Framework for real-time microscopy and stimulus control.

## Installation

```bash
# Base install (core engine only)
pip install clef

# With GUI demos
pip install clef[demos]

# Full environment (microscope + imaging + dev tools)
pip install clef[all]
```

## Usage

```bash
# Run with config files
clef --hardware config/demo/demo_ring_attractor_hardware.yaml \
     --experiment config/demo/demo_ring_attractor_experiment.yaml \
     --algorithm config/demo/demo_ring_attractor_algorithm.yaml

# Validate configs
clef --validate-config --hardware config/demo/demo_ring_attractor_hardware.yaml
```

## Development

```bash
# Install editable with all dependencies
pip install -e .[all]

# Run tests
pytest tests/ --ignore=tests/hardware_physical -v
```

## License

MIT
