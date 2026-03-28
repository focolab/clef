"""
Tests for clef2 ConfigManager and config validation.

All tests load the actual default YAML files from clef2/core/config/
rather than hardcoding their contents.
"""

import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from clef2.core.config.config_manager import ConfigManager
from clef2.core.config.SessionConfig import SessionConfig
from clef2.core.config.IOConfig import IOConfig, InputDeviceConfig, OutputDeviceConfig
from clef2.core.config.ClosedLoopLogicConfig import ClosedLoopLogicConfig

# Path to the default config directory
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "core" / "config"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_manager():
    """ConfigManager pointed at the real defaults directory."""
    return ConfigManager(defaults_dir=CONFIG_DIR)


@pytest.fixture
def session_yaml():
    """Raw YAML data from default_session_config.yaml."""
    with open(CONFIG_DIR / "default_session_config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def io_yaml():
    """Raw YAML data from default_io_config.yaml."""
    with open(CONFIG_DIR / "default_io_config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def logic_yaml():
    """Raw YAML data from default_logic_config.yaml."""
    with open(CONFIG_DIR / "default_logic_config.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# YAML file existence
# ---------------------------------------------------------------------------

class TestDefaultFilesExist:
    """Verify that all expected default config files are present."""

    def test_session_config_exists(self):
        assert (CONFIG_DIR / "default_session_config.yaml").is_file()

    def test_io_config_exists(self):
        assert (CONFIG_DIR / "default_io_config.yaml").is_file()

    def test_logic_config_exists(self):
        assert (CONFIG_DIR / "default_logic_config.yaml").is_file()


# ---------------------------------------------------------------------------
# SessionConfig validation
# ---------------------------------------------------------------------------

class TestSessionConfig:
    """Validate default_session_config.yaml against SessionConfig model."""

    def test_load_default(self, config_manager):
        config = config_manager.load_session_config()
        assert isinstance(config, SessionConfig)

    def test_session_name(self, config_manager, session_yaml):
        config = config_manager.load_session_config()
        assert config.session_name == session_yaml["session_name"]

    def test_user_name(self, config_manager, session_yaml):
        config = config_manager.load_session_config()
        assert config.user_name == session_yaml["user_name"]

    def test_sample_data_dir(self, config_manager, session_yaml):
        config = config_manager.load_session_config()
        assert config.sample_data_dir == session_yaml["sample_data_dir"]

    def test_session_parameters(self, config_manager, session_yaml):
        config = config_manager.load_session_config()
        assert config.session_parameters == session_yaml["session_parameters"]

    def test_extra_fields_allowed(self, session_yaml):
        """SessionConfig should accept extra fields from YAML (e.g. subject, dev_options)."""
        config = SessionConfig(**session_yaml)
        assert isinstance(config, SessionConfig)

    def test_stored_on_manager(self, config_manager):
        config_manager.load_session_config()
        assert config_manager.session_config is not None


# ---------------------------------------------------------------------------
# IOConfig validation
# ---------------------------------------------------------------------------

class TestIOConfig:
    """Validate default_io_config.yaml against IOConfig model."""

    def test_load_default(self, config_manager):
        config = config_manager.load_io_config()
        assert isinstance(config, IOConfig)

    def test_input_devices_list(self, config_manager, io_yaml):
        config = config_manager.load_io_config()
        assert len(config.input_devices) == len(io_yaml["input_devices"])

    def test_output_devices_list(self, config_manager, io_yaml):
        config = config_manager.load_io_config()
        assert len(config.output_devices) == len(io_yaml["output_devices"])

    def test_input_device_fields(self, config_manager, io_yaml):
        config = config_manager.load_io_config()
        first_input = config.input_devices[0]
        first_yaml = io_yaml["input_devices"][0]
        assert first_input.input_device_name == first_yaml["input_device_name"]
        assert first_input.input_device_type == first_yaml["input_device_type"]

    def test_output_device_fields(self, config_manager, io_yaml):
        config = config_manager.load_io_config()
        first_output = config.output_devices[0]
        first_yaml = io_yaml["output_devices"][0]
        assert first_output.output_device_name == first_yaml["output_device_name"]
        assert first_output.output_device_type == first_yaml["output_device_type"]

    def test_stored_on_manager(self, config_manager):
        config_manager.load_io_config()
        assert config_manager.io_config is not None


# ---------------------------------------------------------------------------
# ClosedLoopLogicConfig validation
# ---------------------------------------------------------------------------

class TestClosedLoopLogicConfig:
    """Validate default_logic_config.yaml against ClosedLoopLogicConfig model."""

    def test_load_default(self, config_manager):
        config = config_manager.load_logic_config()
        assert isinstance(config, ClosedLoopLogicConfig)

    def test_logic_algorithms_count(self, config_manager, logic_yaml):
        config = config_manager.load_logic_config()
        assert len(config.logic_algorithms) == len(logic_yaml["logic_algorithms"])

    def test_first_entry_fields(self, config_manager, logic_yaml):
        config = config_manager.load_logic_config()
        entry = config.logic_algorithms[0]
        yaml_entry = logic_yaml["logic_algorithms"][0]
        assert entry.logic_algorithm_name == yaml_entry["logic_algorithm_name"]
        assert entry.logic_class == yaml_entry["logic_class"]
        assert entry.logic_parameters == yaml_entry["logic_parameters"]
        assert entry.io_parameters == yaml_entry["io_parameters"]
        assert entry.gui_parameters == yaml_entry["gui_parameters"]

    def test_stored_on_manager(self, config_manager):
        config_manager.load_logic_config()
        assert config_manager.logic_config is not None


# ---------------------------------------------------------------------------
# ConfigManager.load_all_configs
# ---------------------------------------------------------------------------

class TestLoadAllConfigs:
    """Test loading all configs at once."""

    def test_load_all(self, config_manager):
        config_manager.load_all_configs()
        assert config_manager.session_config is not None
        assert config_manager.io_config is not None
        assert config_manager.logic_config is not None

    def test_load_all_returns_correct_types(self, config_manager):
        config_manager.load_all_configs()
        assert isinstance(config_manager.session_config, SessionConfig)
        assert isinstance(config_manager.io_config, IOConfig)
        assert isinstance(config_manager.logic_config, ClosedLoopLogicConfig)


# ---------------------------------------------------------------------------
# ConfigManager.merge_configs
# ---------------------------------------------------------------------------

class TestMergeConfigs:
    """Test deep-merge behavior."""

    def test_override_scalar(self):
        default = {"a": 1, "b": 2}
        override = {"b": 99}
        result = ConfigManager.merge_configs(default, override)
        assert result == {"a": 1, "b": 99}

    def test_deep_merge_nested(self):
        default = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 99}}
        result = ConfigManager.merge_configs(default, override)
        assert result == {"outer": {"a": 1, "b": 99}}

    def test_override_adds_new_keys(self):
        default = {"a": 1}
        override = {"b": 2}
        result = ConfigManager.merge_configs(default, override)
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_inputs(self):
        default = {"nested": {"x": 1}}
        override = {"nested": {"y": 2}}
        ConfigManager.merge_configs(default, override)
        assert default == {"nested": {"x": 1}}
        assert override == {"nested": {"y": 2}}


# ---------------------------------------------------------------------------
# ConfigManager with user override YAML (uses tmp_path)
# ---------------------------------------------------------------------------

class TestUserOverride:
    """Test that user configs merge on top of defaults."""

    def test_session_override(self, config_manager, tmp_path):
        override = tmp_path / "session_override.yaml"
        override.write_text("session_name: my_override\nuser_name: Tester\n")

        config = config_manager.load_session_config(user_path=override)
        assert config.session_name == "my_override"
        assert config.user_name == "Tester"
        # Fields not overridden should come from defaults
        assert config.sample_data_dir == "./data"

    def test_io_override(self, config_manager, tmp_path):
        override = tmp_path / "io_override.yaml"
        override.write_text(
            "input_devices:\n"
            "  - input_device_name: my_camera\n"
            "    input_device_type: camera\n"
            "    input_device_parameters:\n"
            "      exposure_ms: 50\n"
        )

        config = config_manager.load_io_config(user_path=override)
        assert config.input_devices[0].input_device_name == "my_camera"

    def test_logic_override(self, config_manager, tmp_path):
        override = tmp_path / "logic_override.yaml"
        override.write_text("logic_algorithm: my_algorithm\n")

        config = config_manager.load_logic_config(user_path=override)
        assert config.logic_algorithm == "my_algorithm"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test error paths."""

    def test_missing_file_raises(self, config_manager):
        with pytest.raises(FileNotFoundError):
            config_manager.load_session_config(user_path=Path("/nonexistent/file.yaml"))

    def test_invalid_yaml_raises(self, config_manager, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(": : : not valid yaml [[[")
        with pytest.raises(Exception):
            config_manager.load_session_config(user_path=bad)
