"""
LogicManager for CLEF.

Responsible for:
1. Scanning apps/logic/ to discover user-defined logic subclasses
2. Instantiating logic algorithms from ClosedLoopLogicConfig
3. Resolving output device references via IOManager
4. Providing access to logic instances by name
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from core.logic.BaseClosedLoopLogic import BaseClosedLoopLogic
from core.config.config_manager import ConfigManager
from core.io.io_manager import IOManager

logger = logging.getLogger(__name__)

# Default location for user logic modules
APPS_LOGIC_DIR = Path(__file__).resolve().parent.parent.parent / "apps" / "logic"


class LogicManager:
    """Loads logic plugins from apps/logic and instantiates algorithms from config."""

    def __init__(
        self,
        config_manager: ConfigManager,
        io_manager: IOManager,
        apps_logic_dir: Optional[Path] = None,
    ):
        self.config_manager = config_manager
        self.logic_config = config_manager.logic_config
        self.io_manager = io_manager
        self.apps_logic_dir = Path(apps_logic_dir) if apps_logic_dir else APPS_LOGIC_DIR

        self.logic_instances: Dict[str, BaseClosedLoopLogic] = {}

        self._discover_plugins()
        self._instantiate_logic()

    # ------------------------------------------------------------------
    # Plugin discovery
    # ------------------------------------------------------------------

    def _discover_plugins(self):
        """Import all .py modules in apps/logic/ to trigger __init_subclass__ registration."""
        self._import_modules_from(self.apps_logic_dir)

    @staticmethod
    def _import_modules_from(directory: Path):
        """Import all .py files in a directory to trigger __init_subclass__ registration."""
        if not directory.is_dir():
            logger.debug(f"Plugin directory does not exist: {directory}")
            return

        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                parts = py_file.resolve().parts
                apps_idx = parts.index("apps")
                dotted = ".".join(parts[apps_idx:]).removesuffix(".py")
                importlib.import_module(dotted)
                logger.debug(f"Loaded plugin module: {dotted}")
            except Exception as e:
                logger.warning(f"Failed to import plugin {py_file}: {e}")

    # ------------------------------------------------------------------
    # Logic instantiation
    # ------------------------------------------------------------------

    def _instantiate_logic(self):
        """Create logic instances from config entries."""
        for idx, entry in enumerate(self.logic_config.logic_algorithms):
            logic = self._create_logic(entry, idx)
            if logic.name in self.logic_instances:
                raise ValueError(
                    f"Duplicate logic_algorithm_name: '{logic.name}'. Names must be unique."
                )
            self.logic_instances[logic.name] = logic

    def _create_logic(self, entry, idx: int) -> BaseClosedLoopLogic:
        logic_class_key = entry.logic_class
        name = entry.logic_algorithm_name or f"logic_{idx}"

        # Resolve output device references
        output_devices = {}
        output_device_names = entry.io_parameters.get("output_device_names", [])
        for device_name in output_device_names:
            output_devices[device_name] = self.io_manager.get_output_device(device_name)

        # Resolve input device references
        input_devices = {}
        input_device_names = entry.io_parameters.get("input_device_names", [])
        for device_name in input_device_names:
            try:
                input_devices[device_name] = self.io_manager.get_input_device(device_name)
            except KeyError:
                logger.warning(f"Input device '{device_name}' not found, skipping")

        kwargs = dict(
            name=name,
            config=entry.logic_parameters,
            output_devices=output_devices,
            gui_parameters=entry.gui_parameters,
            input_devices=input_devices,
            io_manager=self.io_manager,
            config_manager=self.config_manager,
        )

        if not logic_class_key:
            logger.info(f"No logic_class for '{name}', using BaseClosedLoopLogic")
            return BaseClosedLoopLogic(**kwargs)

        cls = BaseClosedLoopLogic.get_class(logic_class_key)
        logger.info(f"Creating logic '{name}' (class={logic_class_key})")
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Logic access
    # ------------------------------------------------------------------

    def get_logic(self, name: str) -> BaseClosedLoopLogic:
        """Get a logic instance by name."""
        if name not in self.logic_instances:
            available = list(self.logic_instances.keys())
            raise KeyError(f"No logic named '{name}'. Available: {available}")
        return self.logic_instances[name]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def initialize_model(self, name: Optional[str] = None):
        """Initialize model(s). If name given, initialize only that one; otherwise all."""
        if name is not None:
            self.get_logic(name).initialize_model()
            return
        for logic in self.logic_instances.values():
            logic.initialize_model()

    def process_sample(self, sample: Any, name: Optional[str] = None):
        """Process sample through logic algorithm(s).

        If name given, run only that algorithm; otherwise run all.
        """
        targets = (
            [self.get_logic(name)]
            if name is not None
            else list(self.logic_instances.values())
        )
        for logic in targets:
            logic.process_sample(sample)

    def check_logic(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Check logic state and collect output update requests.

        Returns a single dict keyed by output device name, formatted for
        io_manager.update_output(). Warns if multiple algorithms try to
        update the same output device.
        """
        targets = (
            [self.get_logic(name)]
            if name is not None
            else list(self.logic_instances.values())
        )
        merged: Dict[str, Any] = {}
        for logic in targets:
            logic_update = logic.check_logic()
            if logic_update is None:
                continue
            for device_name, device_kwargs in logic_update.items():
                if device_name in merged:
                    logger.warning(
                        f"Conflicting update for output device '{device_name}': "
                        f"logic '{logic.name}' overwrites previous update"
                    )
                merged[device_name] = device_kwargs
        return merged

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the LogicManager and all logic instances."""
        return {
            "logic_algorithms": {
                name: logic.get_metadata()
                for name, logic in self.logic_instances.items()
            },
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def save_data(self, name: Optional[str] = None, **kwargs):
        """Save data from logic instance(s). If name given, save only that one; otherwise all."""
        if name is not None:
            self.get_logic(name).save_data(**kwargs)
            return
        for logic in self.logic_instances.values():
            logic.save_data(**kwargs)

    def close(self, name: Optional[str] = None):
        """Close logic instance(s). If name given, close only that one; otherwise all."""
        if name is not None:
            self.get_logic(name).close()
            return
        for logic in self.logic_instances.values():
            logic.close()
