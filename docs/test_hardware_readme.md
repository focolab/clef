# Physical Hardware Tests

These tests validate CLEF functionality with physical hardware connections.

## Requirements

- Physical access to Innovation Core microscope
- Micro-Manager installed with device drivers
- Polygon calibration file present
- Sample mounted (for acquisition tests)

## Running Tests

### Run all hardware tests:
```bash
pytest tests/hardware_physical/ -v --hardware
```

### Run specific test module:
```bash
pytest tests/hardware_physical/test_micromanager_backend.py -v --hardware
```

### Run specific test:
```bash
pytest tests/hardware_physical/test_micromanager_backend.py::TestCameraOperations::test_snap_single_frame -v --hardware
```

### Skip slow tests:
```bash
pytest tests/hardware_physical/ -v --hardware -m "not slow"
```

## Test Organization

- `test_micromanager_backend.py` - Low-level backend operations
  - Camera operations (exposure, ROI, acquisition)
  - Stage operations (positioning, Z-stack)
  - Stimulus operations (activation, polygon masks)
  - Data interface operations

- `test_brainalyzer_hardware.py` - Integration tests
  - Full Brainalyzer workflow
  - Stimulus controller operations
  - Complete acquisition with saving

## Safety Notes

- Tests use low laser intensities (5-10%)
- Stage movements are small (<10 µm)
- Acquisitions are brief (80-800 frames)
- Always verify sample is properly mounted before running

## Troubleshooting

**"Hardware not initialized"**
- Check that Micro-Manager is not already running
- Verify config file path is correct
- Check device connections

**"Calibration file not found"**
- Ensure polygon calibration is at: `./res/peripherals/Mightex Polygon P1000/calibrations.json`

**"Tests hang during acquisition"**
- Camera may need firmware reset
- Check that TTL connections are correct
- Verify stage is not hitting limits

## Adding New Tests

1. Mark test with `@pytest.mark.hardware`
2. Use fixtures from `conftest.py`
3. Add cleanup in `finally` blocks
4. Log progress for debugging
5. Use low intensities for safety
