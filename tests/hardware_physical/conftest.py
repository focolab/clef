"""
Pytest configuration for physical hardware tests.

These fixtures provide hardware connections for testing.
Tests are skipped unless --hardware flag is provided.
"""

import pytest
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="Run tests that require physical hardware connections"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "hardware: mark test as requiring physical hardware (deselected by default)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip hardware tests unless --hardware flag provided."""
    if config.getoption("--hardware"):
        return
    
    skip_hardware = pytest.mark.skip(reason="need --hardware option to run")
    for item in items:
        if "hardware_physical" in item.keywords:
            item.add_marker(skip_hardware)


@pytest.fixture(scope="session")
def innovation_core_config():
    """Hardware configuration for Innovation Core microscope."""
    from config.config_manager import ConfigManager
    
    config_manager = ConfigManager()
    config_dir = Path("./config/demo")
    
    config_manager.load_hardware_config(
        config_dir / "hardware_physical_hardware.yaml"
    )
    
    return config_manager.hardware_config


@pytest.fixture(scope="session")
def hardware_manager_connected(innovation_core_config):
    """
    Create HardwareManager with physical hardware connection.
    
    This fixture initializes actual hardware and should be used sparingly.
    Scope is 'session' to avoid repeated initialization.
    """
    from hardware.hardware_manager import HardwareManager
    
    logger.info("Initializing physical hardware connection...")
    hardware = HardwareManager(innovation_core_config)
    
    try:
        hardware.initialize()
        logger.info("✓ Hardware initialized successfully")
        yield hardware
    finally:
        logger.info("Closing hardware connection...")
        hardware.close()
        logger.info("✓ Hardware closed")


@pytest.fixture
def camera_connected(hardware_manager_connected):
    """Get camera interface from connected hardware."""
    return hardware_manager_connected.camera


@pytest.fixture
def stage_connected(hardware_manager_connected):
    """Get stage interface from connected hardware."""
    return hardware_manager_connected.stage


@pytest.fixture
def stimulus_connected(hardware_manager_connected):
    """Get stimulus interface from connected hardware."""
    return hardware_manager_connected.stimulus


@pytest.fixture
def data_interface_connected(hardware_manager_connected):
    """Get data interface from connected hardware."""
    return hardware_manager_connected.data
