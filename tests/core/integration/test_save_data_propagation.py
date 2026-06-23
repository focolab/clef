"""
Integration tests for save_data propagation from engine through managers to devices/logic.
"""

import pytest
from pathlib import Path

from core.config.config_manager import ConfigManager
from core.config.IOConfig import IOConfig
from core.config.ClosedLoopLogicConfig import ClosedLoopLogicConfig
from core.config.SessionConfig import SessionConfig
from core.io.io_manager import IOManager
from core.logic.logic_manager import LogicManager
from core.engine.closed_loop_engine import ClosedLoopEngine

# Import to trigger __init_subclass__ registration
from tests.core.integration.conftest import (  # noqa: F401
    SaveTrackingInputDevice,
    TrackingOutputDevice,
    SaveTrackingLogic,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "core" / "config"
NONEXISTENT = Path("__nonexistent__")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def save_config():
    """ConfigManager with 2 save-tracking inputs, 1 output, 2 save-tracking logic."""
    cm = ConfigManager(defaults_dir=CONFIG_DIR)

    cm.io_config = IOConfig(**{
        "input_devices": [
            {
                "input_device_name": "input_a",
                "input_device_type": "test",
                "input_device_class": "save_tracking_input",
                "data_interface_class": "base_data_interface",
            },
            {
                "input_device_name": "input_b",
                "input_device_type": "test",
                "input_device_class": "save_tracking_input",
                "data_interface_class": "base_data_interface",
            },
        ],
        "output_devices": [
            {
                "output_device_name": "output_0",
                "output_device_type": "test",
                "output_device_class": "tracking_output",
            },
        ],
    })

    cm.logic_config = ClosedLoopLogicConfig(**{
        "logic_algorithms": [
            {
                "logic_algorithm_name": "logic_a",
                "logic_class": "save_tracking_logic",
                "logic_parameters": {},
                "io_parameters": {"output_device_names": ["output_0"]},
                "gui_parameters": {},
            },
            {
                "logic_algorithm_name": "logic_b",
                "logic_class": "save_tracking_logic",
                "logic_parameters": {},
                "io_parameters": {"output_device_names": ["output_0"]},
                "gui_parameters": {},
            },
        ],
    })

    cm.session_config = SessionConfig(session_parameters={"save_samples": True})

    return cm


@pytest.fixture
def io_manager(save_config):
    return IOManager(config_manager=save_config, apps_io_dir=NONEXISTENT)


@pytest.fixture
def logic_manager(save_config, io_manager):
    return LogicManager(
        config_manager=save_config,
        io_manager=io_manager,
        apps_logic_dir=NONEXISTENT,
    )


@pytest.fixture
def engine(save_config, io_manager, logic_manager):
    return ClosedLoopEngine(
        config_manager=save_config,
        io_manager=io_manager,
        logic_manager=logic_manager,
    )


# ---------------------------------------------------------------------------
# Engine → both managers
# ---------------------------------------------------------------------------

class TestSaveDataFromEngine:

    def test_engine_save_data_reaches_all_input_devices(self, engine, io_manager):
        engine.save_data()
        for name, dev in io_manager.input_devices.items():
            assert len(dev.save_calls) == 1, f"{name} save_calls={len(dev.save_calls)}"

    def test_engine_save_data_reaches_all_logic(self, engine, logic_manager):
        engine.save_data()
        for name, logic in logic_manager.logic_instances.items():
            assert len(logic.save_calls) == 1, f"{name} save_calls={len(logic.save_calls)}"

    def test_kwargs_propagate_through_engine(self, engine, io_manager, logic_manager):
        engine.save_data(path="/tmp/out", fmt="tiff")
        # Devices receive a computed `filepath` plus the passthrough kwargs.
        for name, dev in io_manager.input_devices.items():
            call = dev.save_calls[0]
            assert call["path"] == "/tmp/out"
            assert call["fmt"] == "tiff"
            assert Path(call["filepath"]).name == name
        # Logic instances receive a computed `savefilename` plus the kwargs.
        for name, logic in logic_manager.logic_instances.items():
            call = logic.save_calls[0]
            assert call["path"] == "/tmp/out"
            assert call["fmt"] == "tiff"
            assert Path(call["savefilename"]).name == name


# ---------------------------------------------------------------------------
# IOManager save_data targeting
# ---------------------------------------------------------------------------

class TestSaveDataFromIOManager:

    def test_save_all_hits_both_devices(self, io_manager):
        io_manager.save_data()
        for dev in io_manager.input_devices.values():
            assert len(dev.save_calls) == 1

    def test_save_by_name_hits_only_target(self, io_manager):
        io_manager.save_data(name="input_a")
        assert len(io_manager.input_devices["input_a"].save_calls) == 1
        assert len(io_manager.input_devices["input_b"].save_calls) == 0

    def test_save_missing_device_raises(self, io_manager):
        with pytest.raises(KeyError):
            io_manager.save_data(name="nonexistent")


# ---------------------------------------------------------------------------
# LogicManager save_data targeting
# ---------------------------------------------------------------------------

class TestSaveDataFromLogicManager:

    def test_save_all_hits_both_logic(self, logic_manager):
        logic_manager.save_data()
        for logic in logic_manager.logic_instances.values():
            assert len(logic.save_calls) == 1

    def test_save_by_name_hits_only_target(self, logic_manager):
        logic_manager.save_data(name="logic_a")
        assert len(logic_manager.logic_instances["logic_a"].save_calls) == 1
        assert len(logic_manager.logic_instances["logic_b"].save_calls) == 0

    def test_save_missing_logic_raises(self, logic_manager):
        with pytest.raises(KeyError):
            logic_manager.save_data(name="nonexistent")

    def test_save_after_loop(self, engine, io_manager, logic_manager):
        engine.loop(iterations=100)
        engine.save_data()
        for dev in io_manager.input_devices.values():
            assert len(dev.save_calls) == 1
        for logic in logic_manager.logic_instances.values():
            assert len(logic.save_calls) == 1
