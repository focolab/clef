"""
Tests for clef IO system: BaseInputDevice, BaseOutputDevice, and IOManager.

All tests load the actual default YAML config files from clef/core/config/
rather than hardcoding their contents.
"""

import pytest
import yaml
from pathlib import Path

from core.config.config_manager import ConfigManager
from core.io.input_device.BaseInputDevice import BaseInputDevice
from core.io.input_device.BaseDataInterface import BaseDataInterface
from core.io.output_device.BaseOutputDevice import BaseOutputDevice
from core.io.io_manager import IOManager

# Path to the default config directory
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "core" / "config"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def io_yaml():
    """Raw YAML data from default_io_config.yaml."""
    with open(CONFIG_DIR / "default_io_config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def config_manager():
    """ConfigManager with default IO config loaded."""
    cm = ConfigManager(defaults_dir=CONFIG_DIR)
    cm.load_io_config()
    return cm


@pytest.fixture
def io_manager(config_manager):
    """IOManager built from default config (no apps/io plugin scanning)."""
    return IOManager(config_manager=config_manager, apps_io_dir=Path("__nonexistent__"))


# ---------------------------------------------------------------------------
# BaseInputDevice
# ---------------------------------------------------------------------------

class TestBaseInputDevice:

    def test_base_registered(self):
        assert "base_input_device" in BaseInputDevice._registry

    def test_get_class_returns_base(self):
        cls = BaseInputDevice.get_class("base_input_device")
        assert cls is BaseInputDevice

    def test_get_class_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown input device class"):
            BaseInputDevice.get_class("nonexistent_device")

    def test_instantiate(self):
        dev = BaseInputDevice(name="test", config={"a": 1})
        assert dev.name == "test"
        assert dev.config == {"a": 1}

    def test_no_op_methods(self):
        dev = BaseInputDevice(name="noop")
        dev.connect()
        dev.configure()
        assert dev.get_input() is None
        dev.close()


# ---------------------------------------------------------------------------
# BaseOutputDevice
# ---------------------------------------------------------------------------

class TestBaseOutputDevice:

    def test_base_registered(self):
        assert "base_output_device" in BaseOutputDevice._registry

    def test_get_class_returns_base(self):
        cls = BaseOutputDevice.get_class("base_output_device")
        assert cls is BaseOutputDevice

    def test_get_class_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown output device class"):
            BaseOutputDevice.get_class("nonexistent_device")

    def test_instantiate(self):
        dev = BaseOutputDevice(name="test", config={"b": 2})
        assert dev.name == "test"
        assert dev.config == {"b": 2}

    def test_no_op_methods(self):
        dev = BaseOutputDevice(name="noop")
        dev.connect()
        dev.configure()
        dev.update_output()
        dev.close()


# ---------------------------------------------------------------------------
# IOManager — device instantiation from default config
# ---------------------------------------------------------------------------

class TestIOManagerDevices:

    def test_input_devices_loaded(self, io_manager, io_yaml):
        assert len(io_manager.input_devices) == len(io_yaml["input_devices"])

    def test_output_devices_loaded(self, io_manager, io_yaml):
        assert len(io_manager.output_devices) == len(io_yaml["output_devices"])

    def test_input_device_names_match(self, io_manager, io_yaml):
        for dev_cfg in io_yaml["input_devices"]:
            name = dev_cfg["input_device_name"]
            assert name in io_manager.input_devices

    def test_output_device_names_match(self, io_manager, io_yaml):
        for dev_cfg in io_yaml["output_devices"]:
            name = dev_cfg["output_device_name"]
            assert name in io_manager.output_devices

    def test_input_device_type(self, io_manager):
        for dev in io_manager.input_devices.values():
            assert isinstance(dev, BaseInputDevice)

    def test_output_device_type(self, io_manager):
        for dev in io_manager.output_devices.values():
            assert isinstance(dev, BaseOutputDevice)


# ---------------------------------------------------------------------------
# IOManager — device access
# ---------------------------------------------------------------------------

class TestIOManagerAccess:

    def test_get_input_device(self, io_manager, io_yaml):
        name = io_yaml["input_devices"][0]["input_device_name"]
        dev = io_manager.get_input_device(name)
        assert dev.name == name

    def test_get_output_device(self, io_manager, io_yaml):
        name = io_yaml["output_devices"][0]["output_device_name"]
        dev = io_manager.get_output_device(name)
        assert dev.name == name

    def test_get_input_device_missing_raises(self, io_manager):
        with pytest.raises(KeyError, match="No input device named"):
            io_manager.get_input_device("no_such_device")

    def test_get_output_device_missing_raises(self, io_manager):
        with pytest.raises(KeyError, match="No output device named"):
            io_manager.get_output_device("no_such_device")


# ---------------------------------------------------------------------------
# IOManager — connect / close
# ---------------------------------------------------------------------------

class TestIOManagerLifecycle:

    def test_connect_all(self, io_manager):
        io_manager.connect()

    def test_close_all(self, io_manager):
        io_manager.close()

    def test_connect_by_name(self, io_manager, io_yaml):
        name = io_yaml["input_devices"][0]["input_device_name"]
        io_manager.connect(name)

    def test_close_by_name(self, io_manager, io_yaml):
        name = io_yaml["output_devices"][0]["output_device_name"]
        io_manager.close(name)

    def test_connect_missing_raises(self, io_manager):
        with pytest.raises(KeyError, match="No device named"):
            io_manager.connect("nonexistent")

    def test_close_missing_raises(self, io_manager):
        with pytest.raises(KeyError, match="No device named"):
            io_manager.close("nonexistent")


# ---------------------------------------------------------------------------
# BaseDataInterface
# ---------------------------------------------------------------------------

class TestBaseDataInterface:

    def test_base_registered(self):
        assert "base_data_interface" in BaseDataInterface._registry

    def test_get_class_returns_base(self):
        cls = BaseDataInterface.get_class("base_data_interface")
        assert cls is BaseDataInterface

    def test_get_class_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown data interface class"):
            BaseDataInterface.get_class("nonexistent_interface")

    def test_instantiate(self):
        di = BaseDataInterface()
        assert di.input_device is None
        assert di.input_store is None

    def test_get_input_delegates_to_device(self):
        dev = BaseInputDevice(name="test")
        di = BaseDataInterface(input_device=dev)
        assert di.get_input() is None  # base device returns None

    def test_store_input_sets_input_store(self):
        di = BaseDataInterface()
        di.store_input(42)
        assert di.input_store == 42

    def test_save_data_noop(self):
        di = BaseDataInterface()
        di.save_data()  # should not raise


# ---------------------------------------------------------------------------
# IOManager — data interface wiring
# ---------------------------------------------------------------------------

class TestIOManagerDataInterfaces:

    def test_data_interfaces_created(self, io_manager, io_yaml):
        assert len(io_manager.data_interfaces) == len(io_yaml["input_devices"])

    def test_data_interface_type(self, io_manager):
        for di in io_manager.data_interfaces.values():
            assert isinstance(di, BaseDataInterface)

    def test_input_device_has_data_interface(self, io_manager):
        for dev in io_manager.input_devices.values():
            assert dev.data_interface is not None
            assert isinstance(dev.data_interface, BaseDataInterface)

    def test_data_interface_has_input_device(self, io_manager):
        for name, di in io_manager.data_interfaces.items():
            assert di.input_device is io_manager.input_devices[name]


# ---------------------------------------------------------------------------
# IOManager — update_input / input_stores
# ---------------------------------------------------------------------------

class TestIOManagerUpdateInput:

    def test_update_input_runs(self, io_manager):
        io_manager.update_input()

    def test_input_stores_keys_match(self, io_manager):
        io_manager.update_input()
        assert set(io_manager.input_stores.keys()) == set(io_manager.input_devices.keys())

    def test_input_stores_after_update(self, io_manager):
        io_manager.update_input()
        # base device returns None, so input_store is None
        for store in io_manager.input_stores.values():
            assert store is None


# ---------------------------------------------------------------------------
# IOManager — update_output
# ---------------------------------------------------------------------------

class TestIOManagerUpdateOutput:

    def test_update_output_valid_device(self, io_manager, io_yaml):
        name = io_yaml["output_devices"][0]["output_device_name"]
        io_manager.update_output(**{name: {}})

    def test_update_output_invalid_device_logs_critical(self, io_manager):
        """Invalid device name is logged as critical but does not raise."""
        io_manager.update_output(**{"nonexistent_device": {}})


# ---------------------------------------------------------------------------
# IOManager — save_data
# ---------------------------------------------------------------------------

class TestIOManagerSaveData:

    def test_save_data_all(self, io_manager):
        io_manager.save_data()

    def test_save_data_by_name(self, io_manager, io_yaml):
        name = io_yaml["input_devices"][0]["input_device_name"]
        io_manager.save_data(name=name)

    def test_save_data_missing_raises(self, io_manager):
        with pytest.raises(KeyError, match="No input device named"):
            io_manager.save_data(name="nonexistent")
