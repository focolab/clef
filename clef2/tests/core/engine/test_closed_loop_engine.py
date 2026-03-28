"""
Tests for clef2 ClosedLoopEngine.

All tests load the actual default YAML config files from clef2/core/config/
rather than hardcoding their contents.
"""

import pytest
from pathlib import Path

from clef2.core.config.config_manager import ConfigManager
from clef2.core.io.io_manager import IOManager
from clef2.core.logic.logic_manager import LogicManager
from clef2.core.engine.closed_loop_engine import ClosedLoopEngine

# Path to the default config directory
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "core" / "config"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_manager():
    cm = ConfigManager(defaults_dir=CONFIG_DIR)
    cm.load_io_config()
    cm.load_logic_config()
    cm.load_session_config()
    return cm


@pytest.fixture
def io_manager(config_manager):
    return IOManager(config_manager=config_manager, apps_io_dir=Path("__nonexistent__"))


@pytest.fixture
def logic_manager(config_manager, io_manager):
    return LogicManager(
        config_manager=config_manager,
        io_manager=io_manager,
        apps_logic_dir=Path("__nonexistent__"),
    )


@pytest.fixture
def engine(config_manager, io_manager, logic_manager):
    return ClosedLoopEngine(
        config_manager=config_manager,
        io_manager=io_manager,
        logic_manager=logic_manager,
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestEngineInstantiation:

    def test_creates(self, engine):
        assert engine.running is False

    def test_has_managers(self, engine):
        assert engine.config_manager is not None
        assert engine.io_manager is not None
        assert engine.logic_manager is not None


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

class TestEngineLoop:

    def test_loop_single_iteration(self, engine):
        engine.loop(iterations=1)

    def test_loop_multiple_iterations(self, engine):
        engine.loop(iterations=5)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class TestEngineMetadata:

    def test_get_metadata_structure(self, engine):
        meta = engine.get_metadata()
        assert "engine" in meta
        assert "io" in meta
        assert "logic" in meta

    def test_metadata_engine_section(self, engine):
        meta = engine.get_metadata()
        assert "running" in meta["engine"]


# ---------------------------------------------------------------------------
# Save data
# ---------------------------------------------------------------------------

class TestEngineSaveData:

    def test_save_data(self, engine):
        engine.save_data()


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------

class TestEngineClose:

    def test_close(self, engine):
        engine.close()
        assert engine.running is False
