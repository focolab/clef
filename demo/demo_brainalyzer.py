"""
Brainalyzer Demo

End-to-end demonstration of Brainalyzer algorithm with GUI using:
- Config-based initialization from demo_brainalyzer_* YAML files
- DummyHardwareBackend (reads from TIFF file)
- Interactive GUI for ROI placement and stimulus triggering
- Full closed-loop workflow

Usage:
    python demo/20251126_demo_brainalyzer.py [path/to/input.tif]
"""

import sys
import os
import logging
import time
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

parent_dir = str(Path(__file__).parent.parent)


def run_brainalyzer_demo(input_tiff_path: str):
    """
    Run complete Brainalyzer demo workflow.

    Args:
        input_tiff_path: Path to input TIFF file
    """
    logger.info("=" * 60)
    logger.info("Brainalyzer Demo")
    logger.info("=" * 60)

    # Verify input file exists
    if not os.path.exists(input_tiff_path):
        logger.error(f"Input TIFF not found: {input_tiff_path}")
        logger.error("Please provide a valid input TIFF path")
        return False

    logger.info(f"Using input TIFF: {input_tiff_path}")

    try:
        # Load configurations
        logger.info("Loading configuration files...")
        config_manager = ConfigManager()

        config_dir = Path(f"{parent_dir}/config/demo")
        config_manager.load_all_configs(
            hardware_path=config_dir / "demo_brainalyzer_hardware.yaml",
            experiment_path=config_dir / "demo_brainalyzer_experiment.yaml",
            algorithm_path=config_dir / "demo_brainalyzer_algorithm.yaml",
        )

        # Override input recording path from command-line argument
        config_manager.experiment_config.input_recording_path = input_tiff_path

        # Validate configurations
        logger.info("Validating configurations...")
        if not config_manager.validate_config():
            logger.error("Configuration validation failed")
            return False

        logger.info("✓ Configurations loaded and validated")

        # Create engine
        logger.info("Creating ClosedLoopEngine...")
        engine = ClosedLoopEngine(
            hardware_config=config_manager.hardware_config,
            experiment_config=config_manager.experiment_config,
            algorithm_config=config_manager.algorithm_config,
        )

        # Initialize hardware
        logger.info("Initializing hardware...")
        engine.initialize_hardware()
        logger.info("✓ Hardware initialized")

        # Prepare acquisition
        logger.info("Preparing acquisition...")
        engine.prepare_acquisition()
        logger.info("✓ Acquisition prepared")

        # Initialize algorithm (launches GUI subprocess)
        logger.info("Initializing Brainalyzer algorithm...")
        logger.info("  - Creating GUI subprocess")
        logger.info("  - Setting up shared memory")
        logger.info("  - Configuring ROI tracking")
        engine.initialize_algorithm()
        logger.info("✓ Algorithm initialized")

        # Initialize stimulus
        logger.info("Initializing stimulus controller...")
        engine.initialize_stimulus()
        logger.info("✓ Stimulus controller ready")

        logger.info("")
        logger.info("=" * 60)
        logger.info("Ready to start acquisition")
        logger.info("=" * 60)
        logger.info("")
        logger.info("INTERACTIVE DEMO INSTRUCTIONS")
        logger.info("  1. GUI window opens with real-time z-plane image display")
        logger.info("  2. Click on any z-plane image to add an ROI at that location")
        logger.info("  3. ROIs appear as colored overlays; intensity is plotted in real-time")
        logger.info("  4. Use GUI controls to manually trigger stimulus or set thresholds")
        logger.info("  5. Close the window or press Ctrl+C to stop acquisition")
        logger.info("=" * 60)
        logger.info("")

        # Run acquisition
        start_time = time.time()
        engine.run_acquisition_loop()
        duration = time.time() - start_time

        # Save outputs
        logger.info("")
        logger.info("=" * 60)
        logger.info("SAVING OUTPUTS")
        logger.info("=" * 60)
        engine._save_data()
        engine.save_metadata()

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"✓ Demo completed in {duration:.1f} seconds")
        logger.info(f"Output directory: {engine.savedir}")
        logger.info("=" * 60)

        return True

    except KeyboardInterrupt:
        logger.warning("Demo interrupted by user (Ctrl+C)")
        return False

    except Exception as err:
        logger.exception(f"Error during demo: {err}")
        return False

    finally:
        logger.info("Cleaning up...")
        try:
            engine.cleanup()
            logger.info("✓ Cleanup complete")
        except Exception:
            pass


def main():
    """Main entry point for demo."""
    if len(sys.argv) > 1:
        input_tiff = sys.argv[1]
    elif os.environ.get("BRAINALYZER_DEMO_TIFF"):
        input_tiff = os.environ["BRAINALYZER_DEMO_TIFF"]
    else:
        print("Usage: python demo/demo_brainalyzer.py <path/to/input.tif>")
        print("  or set environment variable BRAINALYZER_DEMO_TIFF")
        sys.exit(1)

    success = run_brainalyzer_demo(input_tiff)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
