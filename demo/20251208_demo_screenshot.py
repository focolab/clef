"""
Screenshot Backend Demo

Demonstrates CLEF's data abstraction with RGB screen capture and keyboard/mouse input.

This demo:
1. Captures RGB screenshots from a specified screen region
2. Displays the captured frames in real-time with color statistics
3. Optionally triggers keyboard/mouse inputs based on visual analysis

Usage:
    python demo/20251208_demo_screenshot.py [--trigger]

Arguments:
    --trigger: Enable input triggers based on brightness threshold
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager
from engine.closed_loop_engine import ClosedLoopEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run screenshot capture demo."""
    
    # Parse command line arguments
    enable_trigger = "--trigger" in sys.argv
    
    print("="*70)
    print("CLEF Screenshot Backend Demo")
    print("="*70)
    print("\nThis demo captures RGB screenshots and displays them in real-time.")
    print("\nMake sure you have a window or application visible on screen")
    print("in the capture region (default: 100, 100, 800x600).")
    print("\nThe demo will capture 200 frames at ~30 Hz.")
    
    if enable_trigger:
        print("\n⚠️  INPUT TRIGGER ENABLED ⚠️")
        print("The demo will send keyboard/mouse inputs when brightness exceeds threshold!")
        print("Press Ctrl+C to stop early if needed.")
    else:
        print("\nInput triggers are DISABLED (display only mode).")
        print("Run with --trigger to enable input delivery.")
    
    print("\nPress Enter to start capture...")
    input()
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config_manager = ConfigManager()
        
        # Load config files
        config_dir = Path(__file__).parent.parent / "config" / "demo"
        
        hardware_config = config_manager.load_hardware_config(
            config_dir / "screenshot_hardware.yaml"
        )
        experiment_config = config_manager.load_experiment_config(
            config_dir / "screenshot_experiment.yaml"
        )
        algorithm_config = config_manager.load_algorithm_config(
            config_dir / "screenshot_algorithm.yaml"
        )
        
        # Enable trigger if requested
        if enable_trigger:
            algorithm_config.algorithm_params.trigger_enabled = True
            algorithm_config.stimulus_params.enabled = True
            logger.warning("Input triggers ENABLED - will send keyboard/mouse events!")
        
        # Validate configuration
        config_manager.validate_config()
        
        # Create and run engine
        logger.info("Initializing screenshot capture engine...")
        engine = ClosedLoopEngine(
            hardware_config=hardware_config,
            experiment_config=experiment_config,
            algorithm_config=algorithm_config
        )
        
        # Add screenshot-specific initialization parameters
        engine.hardware_config.screenshot = {
            'x': 100,
            'y': 100,
            'width': 800,
            'height': 600,
            'monitor': 1,
            'backend': 'mss',
        }
        
        logger.info("Starting screenshot capture...")
        engine.run()
        
        print("\n" + "="*70)
        print("Screenshot capture complete!")
        print(f"Data saved to: {engine.savedir}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nCapture interrupted by user.")
        logger.info("Screenshot capture interrupted")
        
    except Exception as e:
        logger.exception(f"Error during screenshot capture: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
