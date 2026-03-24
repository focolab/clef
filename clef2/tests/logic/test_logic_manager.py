"""
Tests for clef2 Logic system: BaseClosedLoopLogic and LogicManager.

All tests load the actual default YAML config files from clef2/core/config/
rather than hardcoding their contents.
"""

import pytest
import yaml
from pathlib import Path

from clef2.core.config.ClosedLoopLogicConfig import ClosedLoopLogicConfig
from clef2.core.config.IOConfig import IOConfig
from clef2.core.logic.BaseClosedLoopLogic import BaseClosedLoopLogic
from clef2.core.logic.logic_manager import LogicManager
from clef2.core.io.io_manager import IOManager

# Path to the default config directory
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "core" / "config"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def logic_yaml():
    """Raw YAML data from default_logic_config.yaml."""
    with open(CONFIG_DIR / "default_logic_config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def logic_config(logic_yaml):
    """ClosedLoopLogicConfig built from the default YAML."""
    return ClosedLoopLogicConfig(**logic_yaml)


@pytest.fixture
def io_manager():
    """Minimal IOManager with default config (no plugin scanning)."""
    with open(CONFIG_DIR / "default_io_config.yaml") as f:
        io_yaml = yaml.safe_load(f)
    io_config = IOConfig(**io_yaml)
    return IOManager(io_config=io_config, apps_io_dir=Path("__nonexistent__"))


@pytest.fixture
def logic_manager(logic_config, io_manager):
    """LogicManager built from default config (no apps/logic plugin scanning)."""
    return LogicManager(
        logic_config=logic_config,
        io_manager=io_manager,
        apps_logic_dir=Path("__nonexistent__"),
    )


# ---------------------------------------------------------------------------
# BaseClosedLoopLogic
# ---------------------------------------------------------------------------

class TestBaseClosedLoopLogic:

    def test_base_registered(self):
        assert "base_closed_loop_logic" in BaseClosedLoopLogic._registry

    def test_get_class_returns_base(self):
        cls = BaseClosedLoopLogic.get_class("base_closed_loop_logic")
        assert cls is BaseClosedLoopLogic

    def test_get_class_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown logic class"):
            BaseClosedLoopLogic.get_class("nonexistent_logic")

    def test_instantiate(self):
        logic = BaseClosedLoopLogic(name="test", config={"a": 1})
        assert logic.name == "test"
        assert logic.config == {"a": 1}

    def test_no_op_methods(self):
        logic = BaseClosedLoopLogic(name="noop")
        logic.initialize_model()
        result = logic.process_sample(sample=None)
        assert result == {"triggered": False, "trigger_value": 0.0}
        logic.close()

    def test_get_metadata(self):
        logic = BaseClosedLoopLogic(name="meta_test")
        meta = logic.get_metadata()
        assert meta["name"] == "meta_test"
        assert meta["logic_class"] == "base_closed_loop_logic"

    def test_list_registered(self):
        registered = BaseClosedLoopLogic.list_registered()
        assert "base_closed_loop_logic" in registered


# ---------------------------------------------------------------------------
# LogicManager — instantiation from default config
# ---------------------------------------------------------------------------

class TestLogicManagerInstantiation:

    def test_logic_instances_loaded(self, logic_manager, logic_yaml):
        assert len(logic_manager.logic_instances) == len(logic_yaml["logic_algorithms"])

    def test_logic_names_match(self, logic_manager, logic_yaml):
        for entry in logic_yaml["logic_algorithms"]:
            name = entry["logic_algorithm_name"]
            assert name in logic_manager.logic_instances

    def test_logic_instance_type(self, logic_manager):
        for logic in logic_manager.logic_instances.values():
            assert isinstance(logic, BaseClosedLoopLogic)


# ---------------------------------------------------------------------------
# LogicManager — access
# ---------------------------------------------------------------------------

class TestLogicManagerAccess:

    def test_get_logic(self, logic_manager, logic_yaml):
        name = logic_yaml["logic_algorithms"][0]["logic_algorithm_name"]
        logic = logic_manager.get_logic(name)
        assert logic.name == name

    def test_get_logic_missing_raises(self, logic_manager):
        with pytest.raises(KeyError, match="No logic named"):
            logic_manager.get_logic("no_such_logic")


# ---------------------------------------------------------------------------
# LogicManager — execution
# ---------------------------------------------------------------------------

class TestLogicManagerExecution:

    def test_initialize_model_all(self, logic_manager):
        logic_manager.initialize_model()

    def test_initialize_model_by_name(self, logic_manager, logic_yaml):
        name = logic_yaml["logic_algorithms"][0]["logic_algorithm_name"]
        logic_manager.initialize_model(name)

    def test_process_sample_all(self, logic_manager, logic_yaml):
        results = logic_manager.process_sample(sample=None)
        assert len(results) == len(logic_yaml["logic_algorithms"])
        for result in results:
            assert "triggered" in result
            assert "trigger_value" in result
            assert "logic_name" in result

    def test_process_sample_by_name(self, logic_manager, logic_yaml):
        name = logic_yaml["logic_algorithms"][0]["logic_algorithm_name"]
        results = logic_manager.process_sample(sample=None, name=name)
        assert len(results) == 1
        assert results[0]["logic_name"] == name


# ---------------------------------------------------------------------------
# LogicManager — lifecycle
# ---------------------------------------------------------------------------

class TestLogicManagerLifecycle:

    def test_close_all(self, logic_manager):
        logic_manager.close()

    def test_close_by_name(self, logic_manager, logic_yaml):
        name = logic_yaml["logic_algorithms"][0]["logic_algorithm_name"]
        logic_manager.close(name)

    def test_close_missing_raises(self, logic_manager):
        with pytest.raises(KeyError, match="No logic named"):
            logic_manager.close("nonexistent")


# ---------------------------------------------------------------------------
# LogicManager — metadata
# ---------------------------------------------------------------------------

class TestLogicManagerMetadata:

    def test_get_metadata_structure(self, logic_manager, logic_yaml):
        meta = logic_manager.get_metadata()
        assert "logic_algorithms" in meta
        for entry in logic_yaml["logic_algorithms"]:
            name = entry["logic_algorithm_name"]
            assert name in meta["logic_algorithms"]

    def test_get_metadata_contains_logic_class(self, logic_manager):
        meta = logic_manager.get_metadata()
        for name, logic_meta in meta["logic_algorithms"].items():
            assert "logic_class" in logic_meta
            assert "name" in logic_meta
