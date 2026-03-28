"""
Integration tests for multi-device, multi-logic engine loop.

Verifies that the engine correctly orchestrates multiple input devices,
output devices, and logic algorithms over 100 iterations.
"""

import pytest
from pathlib import Path

from clef2.core.config.config_manager import ConfigManager
from clef2.core.config.IOConfig import IOConfig
from clef2.core.config.ClosedLoopLogicConfig import ClosedLoopLogicConfig
from clef2.core.io.io_manager import IOManager
from clef2.core.logic.logic_manager import LogicManager
from clef2.core.engine.closed_loop_engine import ClosedLoopEngine

# Import to trigger __init_subclass__ registration
from clef2.tests.core.integration.conftest import (  # noqa: F401
    CountingInputDevice,
    TrackingOutputDevice,
    PassthroughLogic,
    CyclingLogic,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "core" / "config"
NONEXISTENT = Path("__nonexistent__")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_device_config():
    """ConfigManager with 2 inputs, 3 outputs, 2 logic algorithms."""
    cm = ConfigManager(defaults_dir=CONFIG_DIR)

    cm.io_config = IOConfig(**{
        "input_devices": [
            {
                "input_device_name": "input_a",
                "input_device_type": "test",
                "input_device_class": "counting_input",
                "data_interface_class": "base_data_interface",
            },
            {
                "input_device_name": "input_b",
                "input_device_type": "test",
                "input_device_class": "counting_input",
                "data_interface_class": "base_data_interface",
            },
        ],
        "output_devices": [
            {
                "output_device_name": "output_0",
                "output_device_type": "test",
                "output_device_class": "tracking_output",
            },
            {
                "output_device_name": "output_1",
                "output_device_type": "test",
                "output_device_class": "tracking_output",
            },
            {
                "output_device_name": "output_2",
                "output_device_type": "test",
                "output_device_class": "tracking_output",
            },
        ],
    })

    cm.logic_config = ClosedLoopLogicConfig(**{
        "logic_algorithms": [
            {
                "logic_algorithm_name": "logic_a",
                "logic_class": "cycling_logic",
                "logic_parameters": {},
                "io_parameters": {
                    "output_device_names": ["output_0", "output_1", "output_2"],
                },
                "gui_parameters": {},
            },
            {
                "logic_algorithm_name": "logic_b",
                "logic_class": "passthrough_logic",
                "logic_parameters": {},
                "io_parameters": {
                    "output_device_names": ["output_1"],
                },
                "gui_parameters": {},
            },
        ],
    })

    return cm


@pytest.fixture
def io_manager(multi_device_config):
    return IOManager(config_manager=multi_device_config, apps_io_dir=NONEXISTENT)


@pytest.fixture
def logic_manager(multi_device_config, io_manager):
    return LogicManager(
        config_manager=multi_device_config,
        io_manager=io_manager,
        apps_logic_dir=NONEXISTENT,
    )


@pytest.fixture
def engine(multi_device_config, io_manager, logic_manager):
    return ClosedLoopEngine(
        config_manager=multi_device_config,
        io_manager=io_manager,
        logic_manager=logic_manager,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiDeviceEngine:

    def test_engine_completes_100_iterations(self, engine, io_manager):
        engine.loop(iterations=100)
        for name, dev in io_manager.input_devices.items():
            assert dev.counter == 100, f"{name} polled {dev.counter} times, expected 100"

    def test_both_logic_algorithms_process_every_sample(self, engine, logic_manager):
        engine.loop(iterations=100)
        for name, logic in logic_manager.logic_instances.items():
            assert logic.process_sample_count == 100, (
                f"{name} processed {logic.process_sample_count} samples, expected 100"
            )

    def test_cycling_logic_targets_correct_devices(self, engine, io_manager):
        engine.loop(iterations=100)
        out0 = io_manager.output_devices["output_0"]
        out1 = io_manager.output_devices["output_1"]
        out2 = io_manager.output_devices["output_2"]

        # Cycling pattern per 40-sample period:
        #   sample 10 → [output_0]        (cycle 0)
        #   sample 20 → [output_1]        (cycle 1)
        #   sample 30 → [output_2]        (cycle 2)
        #   sample 40 → [output_0,1,2]    (cycle 3)
        # Over 100 samples there are 10 trigger points (10,20,...,100).
        # Two full periods (80 samples) + partial: samples 90→[output_0], 100→[output_1]
        #
        # output_0 hits: 10, 40, 50, 80, 90  → 5
        # output_1 hits: 20, 40, 60, 80, 100 → 5
        # output_2 hits: 30, 40, 70, 80      → 4

        assert len(out0.calls) == 5, f"output_0 got {len(out0.calls)} calls, expected 5"
        assert len(out1.calls) == 5, f"output_1 got {len(out1.calls)} calls, expected 5"
        assert len(out2.calls) == 4, f"output_2 got {len(out2.calls)} calls, expected 4"

    def test_cycling_logic_passes_iteration_in_kwargs(self, engine, io_manager):
        engine.loop(iterations=100)
        out0 = io_manager.output_devices["output_0"]
        iterations_received = [c["iteration"] for c in out0.calls]
        assert iterations_received == [10, 40, 50, 80, 90]

    def test_passthrough_logic_does_not_trigger_output(self, engine, io_manager, logic_manager):
        engine.loop(iterations=100)
        # Total output calls should equal only what cycling_logic produces
        total = sum(len(d.calls) for d in io_manager.output_devices.values())
        assert total == 14  # 5 + 5 + 4

    def test_input_stores_contain_both_devices(self, engine, io_manager):
        engine.loop(iterations=1)
        stores = io_manager.input_stores
        assert "input_a" in stores
        assert "input_b" in stores

    def test_logic_receives_full_input_stores(self, engine, logic_manager):
        engine.loop(iterations=1)
        logic_b = logic_manager.get_logic("logic_b")
        assert "input_a" in logic_b.last_sample
        assert "input_b" in logic_b.last_sample
