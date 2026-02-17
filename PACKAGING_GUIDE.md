# CLEF Packaging Guide

## Installation Options

### Base Install (minimal dependencies)
```bash
pip install clef
```
**Includes:** `numpy`, `pydantic`, `pyyaml`
**Use for:** Core engine only, custom algorithms without GUI/demos

### With Demos
```bash
pip install clef[demos]
```
**Adds:** `scipy`, `tifffile`, `pyqtgraph`, `pyside6`, `pyopengl`, `pyopengl-accelerate`, `matplotlib`
**Use for:** Running RingAttractor/Brainalyzer/DisplayRGB demos with GUI

### Everything (microscope + image processing + dev tools)
```bash
pip install clef[all]
```
**Adds:** Everything in `[demos]` plus `pycromanager`, `opencv-python`, `scikit-image`, `scikit-learn`, `numba`, `pytest`
**Use for:** Full development environment with real hardware support

### Development (editable install)
```bash
pip install -e .[all]  # editable install with all deps
```

---

## Testing with Optional Dependencies

### Run tests skipping demo dependencies:
```bash
pytest -m "not demos"
```

### Run only base tests (no optional deps):
```bash
pytest -m "not demos and not microscope and not imaging"
```

### Run all tests (requires all deps):
```bash
pytest
```

---

## Marking Tests for Optional Dependencies

### Fixture-level (recommended for tifffile/scipy usage):
```python
@pytest.fixture
def dummy_tiff_file(temp_output_dir):
    pytest.importorskip("tifffile")  # Skip if not installed
    import tifffile as tf
    # ... rest of fixture
```

### Test-level:
```python
@pytest.mark.demos
def test_ring_attractor_algorithm():
    from scipy import ndimage  # Will fail gracefully if scipy missing
    # ... test code
```

### Class-level:
```python
@pytest.mark.demos
class TestRingAttractorAlgorithm:
    def test_initialization(self):
        ...
    def test_processing(self):
        ...
```

---

## Files That Need scipy/tifffile

**scipy (demos only):**
- `algorithms/demo/ring_attractor_algorithm.py` - ndimage
- `algorithms/models/HeadCurvatureAndXYStageModel.py` - scipy.io (commented out)

**tifffile (demos + dummy backend):**
- `hardware/backends/dummy_backend.py` - loading test data
- `hardware/backends/lib/DummyMMC.py` - optional file loading
- `hardware/image_data_interface.py` - image I/O
- `utils/MMSubroutines.py` - saving images

**Tests using tifffile:**
- `tests/engine/test_closed_loop_engine.py:157` - fixture
- `tests/hardware/test_data_interface.py:23` - direct import
- `tests/hardware/test_dummy_backend.py:151` - fixture

---

## Patching Tests

Tests have been patched with `pytest.importorskip` and `@pytest.mark.demos` across the 3 affected test files. Example patterns used:

```python
# At top of file
import pytest

# For fixtures that import tifffile
@pytest.fixture
def dummy_tiff_file(temp_output_dir):
    pytest.importorskip("tifffile")  # <-- ADDED
    import tifffile as tf
    # ... rest unchanged
```

Or mark entire test functions:
```python
@pytest.mark.demos  # <-- ADDED
def test_that_needs_scipy_or_tifffile():
    ...
```

---

## Publishing Workflow

1. **Local test:**
   ```bash
   pip install build
   python -m build
   pip install dist/clef-0.1.0-py3-none-any.whl
   ```

2. **Test in clean environment:**
   ```bash
   python -m venv test_env
   source test_env/bin/activate  # or test_env\Scripts\activate on Windows
   pip install dist/clef-0.1.0-py3-none-any.whl
   clef --help
   ```

3. **Upload to TestPyPI:**
   ```bash
   pip install twine
   twine upload --repository testpypi dist/*
   ```

4. **Test install from TestPyPI:**
   ```bash
   pip install --index-url https://test.pypi.org/simple/ clef
   ```

5. **Upload to PyPI (production):**
   ```bash
   twine upload dist/*
   ```

---

## Current Status

- ✅ `pyproject.toml` created
- ✅ Tests patched with pytest.importorskip (3 files)
- ✅ Source files use lazy imports for tifffile
- ✅ All 377 tests passing
- ✅ Base install tested without optional deps
- ⏳ Update README.md with installation instructions
- ⏳ Add `__version__` to package (optional)
- ⏳ Fix README.md merge conflict
