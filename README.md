# CLEF

Closed-Loop Experimental Framework for real-time microscopy and stimulus control.

## Installation

```bash
# Clone the repository
git clone https://github.com/focolab/clef.git
cd clef

# Create and activate a virtual environment (choose one):
# Using conda:
conda create -n clef python>=3.10
conda activate clef

# Using venv:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with package manager of choice:
# pip:
pip install -e .[demos]

# uv:
uv pip install -e ".[demos]"

# poetry:
poetry install --extras demos
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
