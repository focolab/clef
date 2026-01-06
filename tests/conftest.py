"""
Root pytest configuration for CLEF tests.

This file registers command-line options and markers that are available
to all test modules.

If you already have a tests/conftest.py file, merge the following:
1. The pytest_addoption function (add --hardware option)
2. The pytest_configure function (add hardware and slow markers)
3. The pytest_collection_modifyitems function (skip hardware tests by default)
"""

import pytest
import sys
from pathlib import Path

# Add project root to path so tests can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running (can be skipped with -m 'not slow')"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to skip hardware tests unless --hardware flag provided.
    
    This runs during test collection and adds skip markers to hardware tests
    unless the --hardware flag is present.
    """
    if config.getoption("--hardware"):
        # Hardware tests should run
        return
    
    skip_hardware = pytest.mark.skip(reason="need --hardware option to run")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hardware)
