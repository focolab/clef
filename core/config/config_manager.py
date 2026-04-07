"""
CLEF Configuration Manager

Centralized configuration loading, validation, and merging for the CLEF platform.
Loads YAML configs, merges defaults with user overrides, validates with Pydantic models.
"""

import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from copy import deepcopy
from pydantic import ValidationError

from core.config.SessionConfig import SessionConfig
from core.config.IOConfig import IOConfig
from core.config.ClosedLoopLogicConfig import ClosedLoopLogicConfig

logger = logging.getLogger(__name__)

# Directory containing default YAML config files
DEFAULTS_DIR = Path(__file__).parent


class ConfigManager:
    """
    Centralized configuration management for CLEF.

    Loads YAML configs, validates with Pydantic, merges defaults with user overrides.
    """

    def __init__(self, defaults_dir: Optional[Path] = None):
        """Initialize ConfigManager.

        Args:
            defaults_dir: Directory containing default YAML configs.
                          Defaults to the core/config/ directory.
        """
        self.defaults_dir = Path(defaults_dir) if defaults_dir else DEFAULTS_DIR

        self.session_config: Optional[SessionConfig] = None
        self.io_config: Optional[IOConfig] = None
        self.logic_config: Optional[ClosedLoopLogicConfig] = None

        logger.info(f"ConfigManager initialized with defaults dir: {self.defaults_dir}")

    @staticmethod
    def load_yaml(path: Path) -> Dict[str, Any]:
        """Load and parse a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed YAML data as a dictionary.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            logger.debug(f"Loaded YAML from {path}")
            return data
        except FileNotFoundError:
            logger.error(f"Config file not found: {path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {path}: {e}")
            raise

    @staticmethod
    def merge_configs(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two config dictionaries. Override values take precedence.

        Args:
            default: Base configuration dictionary.
            override: Override configuration dictionary (wins on conflict).

        Returns:
            Merged configuration dictionary.
        """
        merged = deepcopy(default)

        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def load_session_config(self, user_path: Optional[Path] = None) -> SessionConfig:
        """Load session configuration.

        Args:
            user_path: Optional path to a user session config YAML that overrides defaults.

        Returns:
            Validated SessionConfig instance.
        """
        default_path = self.defaults_dir / "default_session_config.yaml"
        default_data = self.load_yaml(default_path) if default_path.exists() else {}

        if user_path:
            user_data = self.load_yaml(Path(user_path))
            config_data = self.merge_configs(default_data, user_data)
        else:
            config_data = default_data

        try:
            self.session_config = SessionConfig(**config_data)
            logger.info(f"Session config loaded: {self.session_config.session_name}")
            return self.session_config
        except ValidationError as e:
            logger.error(f"Session config validation failed: {e}")
            raise

    def load_io_config(self, user_path: Optional[Path] = None) -> IOConfig:
        """Load IO configuration.

        Args:
            user_path: Optional path to a user IO config YAML that overrides defaults.

        Returns:
            Validated IOConfig instance.
        """
        default_path = self.defaults_dir / "default_io_config.yaml"
        default_data = self.load_yaml(default_path) if default_path.exists() else {}

        if user_path:
            user_data = self.load_yaml(Path(user_path))
            config_data = self.merge_configs(default_data, user_data)
        else:
            config_data = default_data

        try:
            self.io_config = IOConfig(**config_data)
            logger.info(f"IO config loaded: {len(self.io_config.input_devices)} input, "
                        f"{len(self.io_config.output_devices)} output devices")
            return self.io_config
        except ValidationError as e:
            logger.error(f"IO config validation failed: {e}")
            raise

    def load_logic_config(self, user_path: Optional[Path] = None) -> ClosedLoopLogicConfig:
        """Load closed-loop logic configuration.

        Args:
            user_path: Optional path to a user logic config YAML that overrides defaults.

        Returns:
            Validated ClosedLoopLogicConfig instance.
        """
        default_path = self.defaults_dir / "default_logic_config.yaml"
        default_data = self.load_yaml(default_path) if default_path.exists() else {}

        if user_path:
            user_data = self.load_yaml(Path(user_path))
            config_data = self.merge_configs(default_data, user_data)
        else:
            config_data = default_data

        try:
            self.logic_config = ClosedLoopLogicConfig(**config_data)
            logger.info(f"Logic config loaded: {len(self.logic_config.logic_algorithms)} algorithm(s)")
            return self.logic_config
        except ValidationError as e:
            logger.error(f"Logic config validation failed: {e}")
            raise

    def load_all_configs(
        self,
        session_path: Optional[Path] = None,
        io_path: Optional[Path] = None,
        logic_path: Optional[Path] = None,
    ) -> None:
        """Load all three configuration files.

        Args:
            session_path: Optional user session config path.
            io_path: Optional user IO config path.
            logic_path: Optional user logic config path.
        """
        self.load_session_config(session_path)
        self.load_io_config(io_path)
        self.load_logic_config(logic_path)

        logger.info("All configurations loaded successfully")

    def get_metadata(self) -> Dict[str, Any]:
        """Return current in-memory configs as a serializable dict."""
        meta: Dict[str, Any] = {}
        if self.session_config is not None:
            meta["session"] = self.session_config.model_dump()
        if self.io_config is not None:
            meta["io"] = self.io_config.model_dump()
        if self.logic_config is not None:
            meta["logic"] = self.logic_config.model_dump()
        return meta
