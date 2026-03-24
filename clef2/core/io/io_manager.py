"""
IOManager for CLEF.

Responsible for:
1. Scanning apps/io directories to discover user-defined device subclasses
2. Instantiating input and output devices from IOConfig
3. Providing access to device instances by name
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, Optional

from clef2.core.io.input_device.BaseInputDevice import BaseInputDevice
from clef2.core.io.output_device.BaseOutputDevice import BaseOutputDevice
from clef2.core.config.IOConfig import IOConfig

logger = logging.getLogger(__name__)

# Default location for user device modules
APPS_IO_DIR = Path(__file__).resolve().parent.parent.parent / "apps" / "io"


class IOManager:
    """Loads device plugins from apps/io and instantiates devices from config."""

    def __init__(
        self,
        io_config: IOConfig,
        apps_io_dir: Optional[Path] = None,
    ):
        self.io_config = io_config
        self.apps_io_dir = Path(apps_io_dir) if apps_io_dir else APPS_IO_DIR

        self.input_devices: Dict[str, BaseInputDevice] = {}
        self.output_devices: Dict[str, BaseOutputDevice] = {}

        self._discover_plugins()
        self._instantiate_devices()

    # ------------------------------------------------------------------
    # Plugin discovery
    # ------------------------------------------------------------------

    def _discover_plugins(self):
        """Import all .py modules in apps/io/input_device and apps/io/output_device."""
        self._import_modules_from(self.apps_io_dir / "input_device")
        self._import_modules_from(self.apps_io_dir / "output_device")

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
                clef2_idx = parts.index("clef2")
                dotted = ".".join(parts[clef2_idx:]).removesuffix(".py")
                importlib.import_module(dotted)
                logger.debug(f"Loaded plugin module: {dotted}")
            except Exception as e:
                logger.warning(f"Failed to import plugin {py_file}: {e}")

    # ------------------------------------------------------------------
    # Device instantiation
    # ------------------------------------------------------------------

    def _instantiate_devices(self):
        """Create device instances from config entries."""
        for dev_cfg in self.io_config.input_devices:
            device = self._create_input_device(dev_cfg)
            self.input_devices[device.name] = device

        for dev_cfg in self.io_config.output_devices:
            device = self._create_output_device(dev_cfg)
            self.output_devices[device.name] = device

    @staticmethod
    def _create_input_device(dev_cfg) -> BaseInputDevice:
        device_class_key = dev_cfg.input_device_class
        name = dev_cfg.input_device_name or "unnamed_input"
        params = dev_cfg.input_device_parameters

        if device_class_key is None:
            logger.info(f"No input_device_class for '{name}', using BaseInputDevice")
            return BaseInputDevice(name=name, config=params)

        cls = BaseInputDevice.get_class(device_class_key)
        logger.info(f"Creating input device '{name}' (class={device_class_key})")
        return cls(name=name, config=params)

    @staticmethod
    def _create_output_device(dev_cfg) -> BaseOutputDevice:
        device_class_key = dev_cfg.output_device_class
        name = dev_cfg.output_device_name or "unnamed_output"
        params = dev_cfg.output_device_parameters

        if device_class_key is None:
            logger.info(f"No output_device_class for '{name}', using BaseOutputDevice")
            return BaseOutputDevice(name=name, config=params)

        cls = BaseOutputDevice.get_class(device_class_key)
        logger.info(f"Creating output device '{name}' (class={device_class_key})")
        return cls(name=name, config=params)

    # ------------------------------------------------------------------
    # Device access
    # ------------------------------------------------------------------

    def get_input_device(self, name: str) -> BaseInputDevice:
        """Get an input device by name."""
        if name not in self.input_devices:
            available = list(self.input_devices.keys())
            raise KeyError(f"No input device named '{name}'. Available: {available}")
        return self.input_devices[name]

    def get_output_device(self, name: str) -> BaseOutputDevice:
        """Get an output device by name."""
        if name not in self.output_devices:
            available = list(self.output_devices.keys())
            raise KeyError(f"No output device named '{name}'. Available: {available}")
        return self.output_devices[name]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, name: Optional[str] = None):
        """Connect devices. If name is given, connect only that device; otherwise connect all."""
        if name is not None:
            self._get_device(name).connect()
            return
        for dev in self.input_devices.values():
            dev.connect()
        for dev in self.output_devices.values():
            dev.connect()

    def close(self, name: Optional[str] = None):
        """Close devices. If name is given, close only that device; otherwise close all."""
        if name is not None:
            self._get_device(name).close()
            return
        for dev in self.input_devices.values():
            dev.close()
        for dev in self.output_devices.values():
            dev.close()

    def _get_device(self, name: str):
        """Look up a device by name across both input and output devices."""
        if name in self.input_devices:
            return self.input_devices[name]
        if name in self.output_devices:
            return self.output_devices[name]
        available = list(self.input_devices.keys()) + list(self.output_devices.keys())
        raise KeyError(f"No device named '{name}'. Available: {available}")
