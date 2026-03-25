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
from clef2.core.io.input_device.BaseDataInterface import BaseDataInterface
from clef2.core.io.output_device.BaseOutputDevice import BaseOutputDevice
from clef2.core.config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Default location for user device modules
APPS_IO_DIR = Path(__file__).resolve().parent.parent.parent / "apps" / "io"


class IOManager:
    """Loads device plugins from apps/io and instantiates devices from config."""

    def __init__(
        self,
        config_manager: ConfigManager,
        apps_io_dir: Optional[Path] = None,
    ):
        self.config_manager = config_manager
        self.io_config = config_manager.io_config
        self.apps_io_dir = Path(apps_io_dir) if apps_io_dir else APPS_IO_DIR

        self.input_devices: Dict[str, BaseInputDevice] = {}
        self.output_devices: Dict[str, BaseOutputDevice] = {}
        self.data_interfaces: Dict[str, BaseDataInterface] = {}

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
            device = self._create_input_device(dev_cfg, io_manager=self)
            self.input_devices[device.name] = device

            # Wire up data interface
            di = self._create_data_interface(dev_cfg, device)
            device.data_interface = di
            self.data_interfaces[device.name] = di

        for dev_cfg in self.io_config.output_devices:
            device = self._create_output_device(dev_cfg, io_manager=self)
            self.output_devices[device.name] = device

    @staticmethod
    def _create_data_interface(dev_cfg, device: BaseInputDevice) -> BaseDataInterface:
        di_class_key = dev_cfg.data_interface_class
        if not di_class_key:
            logger.info(f"No data_interface_class for '{device.name}', using BaseDataInterface")
            return BaseDataInterface(input_device=device)

        cls = BaseDataInterface.get_class(di_class_key)
        logger.info(f"Creating data interface for '{device.name}' (class={di_class_key})")
        return cls(input_device=device)

    @staticmethod
    def _create_input_device(dev_cfg, io_manager=None) -> BaseInputDevice:
        device_class_key = dev_cfg.input_device_class
        name = dev_cfg.input_device_name or "unnamed_input"
        params = dev_cfg.input_device_parameters

        if device_class_key is None:
            logger.info(f"No input_device_class for '{name}', using BaseInputDevice")
            return BaseInputDevice(name=name, config=params, io_manager=io_manager)

        cls = BaseInputDevice.get_class(device_class_key)
        logger.info(f"Creating input device '{name}' (class={device_class_key})")
        return cls(name=name, config=params, io_manager=io_manager)

    @staticmethod
    def _create_output_device(dev_cfg, io_manager=None) -> BaseOutputDevice:
        device_class_key = dev_cfg.output_device_class
        name = dev_cfg.output_device_name or "unnamed_output"
        params = dev_cfg.output_device_parameters

        if device_class_key is None:
            logger.info(f"No output_device_class for '{name}', using BaseOutputDevice")
            return BaseOutputDevice(name=name, config=params, io_manager=io_manager)

        cls = BaseOutputDevice.get_class(device_class_key)
        logger.info(f"Creating output device '{name}' (class={device_class_key})")
        return cls(name=name, config=params, io_manager=io_manager)

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
    # Input / Output
    # ------------------------------------------------------------------

    def update_input(self):
        """Poll all data interfaces: get_input then store_input."""
        for di in self.data_interfaces.values():
            sample = di.get_input()
            di.store_input(sample)

    @property
    def input_stores(self) -> Dict[str, Any]:
        """Collect input_store from each data interface, keyed by device name."""
        return {name: di.input_store for name, di in self.data_interfaces.items()}

    def update_output(self, **kwargs):
        """Update output devices. Keys are device names, values are kwarg dicts.

        Raises KeyError if a device name is not found.
        """
        for device_name, device_kwargs in kwargs.items():
            dev = self.get_output_device(device_name)
            dev.update_output(**device_kwargs)

    def save_data(self, name: Optional[str] = None, **kwargs):
        """Save data from input device(s). If name given, save only that one; otherwise all."""
        if name is not None:
            if name not in self.input_devices:
                available = list(self.input_devices.keys())
                raise KeyError(f"No input device named '{name}'. Available: {available}")
            self.input_devices[name].save_data(**kwargs)
            return
        for dev in self.input_devices.values():
            dev.save_data(**kwargs)

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

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> dict:
        """Get metadata about the IOManager and all devices."""
        return {
            "input_devices": {
                name: dev.get_metadata() if hasattr(dev, "get_metadata") else {"name": name}
                for name, dev in self.input_devices.items()
            },
            "output_devices": {
                name: dev.get_metadata() if hasattr(dev, "get_metadata") else {"name": name}
                for name, dev in self.output_devices.items()
            },
        }

    def _get_device(self, name: str):
        """Look up a device by name across both input and output devices."""
        if name in self.input_devices:
            return self.input_devices[name]
        if name in self.output_devices:
            return self.output_devices[name]
        available = list(self.input_devices.keys()) + list(self.output_devices.keys())
        raise KeyError(f"No device named '{name}'. Available: {available}")
