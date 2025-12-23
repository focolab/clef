"""
Test script for polygon stimulus with config objects.

Run this to verify the polygon stimulus system works with the new
config-based architecture.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_polygon_stimulus():
    """Test polygon stimulus initialization with configs."""
    
    print("="*60)
    print("Testing Polygon Stimulus with Config Objects")
    print("="*60)
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Load configs
    # Update these paths to your actual config files
    hardware_config_path = Path("config/demo/hardware_config_ldi_polygon.yaml")
    experiment_config_path = Path("config/defaults/experiment_default.yaml")
    algorithm_config_path = Path("config/defaults/algorithm_default.yaml")
    
    try:
        # Load all configs
        config_manager.load_all_configs(
            hardware_path=hardware_config_path,
            experiment_path=experiment_config_path,
            algorithm_path=algorithm_config_path
        )
        
        # Validate
        if not config_manager.validate_config():
            logger.error("Config validation failed")
            return False
        
        print("\n✓ Configs loaded and validated")
        
        # Initialize hardware manager
        from hardware.hardware_manager import HardwareManager
        hardware_manager = HardwareManager(config_manager.hardware_config)
        
        print("\n✓ Hardware manager created")
        
        # Initialize hardware
        hardware_manager.initialize()
        
        print("\n✓ Hardware initialized")
        
        # Check stimulus interface
        stimulus = hardware_manager.stimulus
        
        # Verify polygon-specific methods are available
        if hasattr(stimulus, 'get_polygon_dimensions'):
            dims = stimulus.get_polygon_dimensions()
            print(f"\n✓ Polygon dimensions: {dims}")
        else:
            logger.error("Stimulus interface missing get_polygon_dimensions()")
            return False
        
        if hasattr(stimulus, 'get_calibration_points'):
            calib = stimulus.get_calibration_points()
            if calib:
                print(f"✓ Calibration points loaded: {list(calib.keys())}")
            else:
                logger.warning("Calibration points not loaded")
        else:
            logger.error("Stimulus interface missing get_calibration_points()")
            return False
        
        # Test stimulus configuration
        config = {
            "interface_type": "InvCore-LDI-Polygon-640",
            "intensity": 10,
        }
        stimulus.configure_stimulus(config)
        print("\n✓ Stimulus configured")
        
        # Test activation/deactivation
        params = {"intensity": 10}
        stimulus.activate_stimulus(params)
        print("✓ Stimulus activated")
        
        stimulus.deactivate_stimulus()
        print("✓ Stimulus deactivated")
        
        # Test polygon mask update
        stim_params = {
            'event': {
                'event_type': 'circle-button',
                'x': 100,
                'y': 100,
                'stim_diameter': 30,
                'stim_intensity': 10
            },
            'roi': [0, 0]
        }
        
        if hasattr(stimulus, 'update_polygon_mask'):
            stimulus.update_polygon_mask(stim_params)
            print("✓ Polygon mask updated")
        else:
            logger.warning("Stimulus interface missing update_polygon_mask()")
        
        # Clean up
        hardware_manager.close()
        print("\n✓ Hardware closed")
        
        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60)
        return True
        
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_polygon_stimulus()
    sys.exit(0 if success else 1)
