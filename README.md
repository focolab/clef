# CLEF

Closed-Loop Experimental Framework for real-time microscopy and stimulus control.

## Installation

```bash
# Create a virtual environment
conda create -n clef python
conda activate clef

# Install with demo dependencies (while in clef directory)
pip install -e .[demos]
```

## Usage

### Ring attractor demo

Illustrates CLEF on a synthetic dynamical system:

```bash
clef --hardware config/demo/demo_ring_attractor_hardware.yaml \
     --experiment config/demo/demo_ring_attractor_experiment.yaml \
     --algorithm config/demo/demo_ring_attractor_algorithm.yaml
```

### Brainalyzer demo

Illustrates CLEF on a microscopy recording:

```bash
clef --hardware config/demo/demo_brainalyzer_hardware.yaml \
     --experiment config/demo/demo_brainalyzer_experiment.yaml \
     --algorithm config/demo/demo_brainalyzer_algorithm.yaml
```

### Alternatively, run demos with scripts

```bash
python demo/demo_ring_attractor.py
```

```bash
python demo/demo_brainalyzer.py
```

### Validate configs without running

```bash
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
