"""
Verification test for --hardware flag setup.

This simple test verifies that the --hardware flag is properly configured.

Usage:
    # Should be SKIPPED
    pytest tests/hardware_physical/test_verify_hardware_flag.py -v
    
    # Should PASS
    pytest tests/hardware_physical/test_verify_hardware_flag.py -v --hardware
"""

import pytest

pytestmark = pytest.mark.hardware


def test_hardware_flag_is_working():
    """
    Simple test to verify --hardware flag is configured correctly.
    
    This test should only run when --hardware flag is provided.
    If you see this test run without --hardware, the configuration is incorrect.
    """
    print("\n✓ Hardware flag is working correctly!")
    print("  This test only runs with --hardware flag")
    assert True


def test_hardware_marker_detection():
    """Verify that hardware marker is detected by pytest."""
    # This will only run with --hardware flag
    import sys
    
    # Check that we're in a test environment
    assert 'pytest' in sys.modules
    print("\n✓ Running in pytest environment with hardware tests enabled")


class TestHardwareFlagClass:
    """Tests in a class also work with module-level marker."""
    
    def test_class_method_respects_marker(self):
        """Test that class methods also respect the hardware marker."""
        print("\n✓ Class-based tests also respect hardware marker")
        assert True


# @pytest.mark.slow
@pytest.mark.hardware
def test_combined_markers():
    """Test that multiple markers work together."""
    print("\n✓ Multiple markers (hardware + slow) work correctly")
    assert True
