"""
Brainalyzer Hardware Demo - Innovation Core Microscope

End-to-end demonstration of Brainalyzer algorithm with physical hardware:
- Innovation Core spinning disk microscope
- Hamamatsu camera with 8-plane z-stack
- LDI Polygon stimulus interface
- Interactive GUI for ROI placement and stimulus triggering

This demo validates the complete CLEF hardware abstraction stack:
1. MicroManagerBackend initializes physical hardware
2. HardwareManager provides unified interface
3. Brainalyzer algorithm processes frames in real-time
4. PolygonStimulusController activates galvo-based stimulus
5. All metadata and images are saved

Prerequisites:
- Physical access to Innovation Core microscope
- Micro-Manager installed with appropriate device drivers
- Polygon calibration file present
- Sample mounted and in focus

Usage:
    python demo/20251201_demo_brainalyzer_hardware.py
    
Safety Notes:
    - Ensure sample is properly mounted before starting
    - Check that all shutters are configured correctly
    - Monitor first few acquisitions to verify focus tracking
    - Use low laser intensity for initial tests
    
Author: CLEF Team
Date: 2025-12-01
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

from config.config_manager import ConfigManager
from engine.closed_loop_engine import ClosedLoopEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./demo_output_hardware/brainalyzer_hardware_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def verify_hardware_prerequisites() -> bool:
    """
    Verify that all prerequisites for hardware demo are met.
    
    Returns:
        True if all checks pass, False otherwise
    """
    logger.info("Verifying hardware prerequisites...")
    
    checks = []
    
    # Check 1: Calibration file exists
    calib_path = Path(f"{parent_dir}/hardware/stimulus_controllers/stimulus_resources/Mightex Polygon P1000/calibrations.json")
    # cwd = os.getcwd()
    if calib_path.exists():
        logger.info(f"✓ Polygon calibration file found: {calib_path}")
        checks.append(True)
    else:
        logger.error(f"✗ Polygon calibration file not found: {calib_path}")
        logger.error(f"Is the pathing correct? Current working directory: f{parent_dir}")
        logger.error("  Please ensure calibration file is present before running demo")
        checks.append(False)
    
    # Check 2: Output directory can be created
    output_dir = Path("./demo_output_hardware")
    try:
        output_dir.mkdir(exist_ok=True)
        logger.info(f"✓ Output directory ready: {output_dir}")
        checks.append(True)
    except Exception as e:
        logger.error(f"✗ Cannot create output directory: {e}")
        checks.append(False)
    
    # Check 3: Config files exist
    config_dir = Path(f"{parent_dir}/config/demo")
    required_configs = [
        "hardware_physical_experiment.yaml",
        "hardware_physical_algorithm.yaml",
        "hardware_physical_hardware.yaml"
    ]
    
    for config_file in required_configs:
        config_path = config_dir / config_file
        if config_path.exists():
            logger.info(f"✓ Config file found: {config_file}")
            checks.append(True)
        else:
            logger.error(f"✗ Config file missing: {config_file}")
            checks.append(False)
    
    success = all(checks)
    if success:
        logger.info("=" * 60)
        logger.info("✓ All prerequisite checks passed")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("✗ Prerequisite checks failed")
        logger.error("=" * 60)
    
    return success


def print_hardware_instructions():
    """Print instructions for hardware setup."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("HARDWARE SETUP INSTRUCTIONS")
    logger.info("=" * 60)
    logger.info("1. Mount sample on microscope stage")
    logger.info("2. Focus sample using eyepiece or live view")
    logger.info("3. Set z-stack range to cover desired volume")
    logger.info("4. Verify laser alignment and intensity")
    logger.info("5. Check that polygon calibration matches current objective")
    logger.info("=" * 60)
    logger.info("")
    
    response = input("Are you ready to proceed? (yes/no): ").strip().lower()
    return response == "yes"


def print_demo_instructions():
    """Print instructions for using the demo."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DEMO USAGE INSTRUCTIONS")
    logger.info("=" * 60)
    logger.info("GUI Controls:")
    logger.info("  - Click on z-plane images to select plane")
    logger.info("  - 'Add quant ROI' - draw green ROI for intensity tracking")
    logger.info("  - 'Add stim ROI' - draw red dotted ROI for stimulus")
    logger.info("  - 'Pulse stimulate ROI(s)' - trigger stimulus at ROIs")
    logger.info("  - Adjust stimulus intensity and duration as needed")
    logger.info("")
    logger.info("What to Expect:")
    logger.info("  - Real-time frame display updates every ~100ms")
    logger.info("  - ROI intensity plots update with each volume")
    logger.info("  - Stimulus triggers will activate polygon and laser")
    logger.info("  - Demo runs for ~800 frames (80 seconds)")
    logger.info("=" * 60)
    logger.info("")


def run_hardware_demo():
    """
    Run complete Brainalyzer hardware demonstration.
    
    Returns:
        True if demo completed successfully
    """
    logger.info("=" * 60)
    logger.info("Brainalyzer Hardware Demo - Innovation Core")
    logger.info("=" * 60)
    
    try:
        # Verify prerequisites
        if not verify_hardware_prerequisites():
            logger.error("Prerequisites not met. Aborting demo.")
            return False
        
        # Check hardware readiness
        if not print_hardware_instructions():
            logger.info("Demo cancelled by user")
            return False
        
        # Print usage instructions
        print_demo_instructions()
        
        # Load configurations
        logger.info("Loading configuration files...")
        config_manager = ConfigManager()
        
        config_dir = Path(f"{parent_dir}/config/demo")
        config_manager.load_all_configs(
            hardware_path=config_dir / "hardware_physical_hardware.yaml",
            experiment_path=config_dir / "hardware_physical_experiment.yaml",
            algorithm_path=config_dir /  "hardware_physical_algorithm.yaml",
        )
        
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
            algorithm_config=config_manager.algorithm_config
        )
        
        # Initialize hardware
        logger.info("Initializing hardware (this may take 10-30 seconds)...")
        logger.info("  - Loading Micro-Manager configuration")
        logger.info("  - Configuring camera and stage")
        logger.info("  - Setting up polygon stimulus interface")
        engine.initialize_hardware()
        logger.info("✓ Hardware initialized successfully")
        
        # Prepare acquisition
        logger.info("Preparing acquisition...")
        engine.prepare_acquisition()
        logger.info("✓ Acquisition prepared")
        
        # Initialize algorithm
        logger.info("Initializing Brainalyzer algorithm...")
        logger.info("  - Creating GUI subprocess")
        logger.info("  - Setting up shared memory")
        logger.info("  - Configuring ROI tracking")
        engine.initialize_algorithm()
        logger.info("✓ Algorithm initialized")
        
        # Initialize stimulus
        logger.info("Initializing stimulus controller...")
        logger.info("  - Loading polygon calibration")
        logger.info("  - Configuring LDI intensity and TTL")
        logger.info("  - Spooling numba JIT functions")
        engine.initialize_stimulus()
        logger.info("✓ Stimulus controller ready")
        
        # Final check before acquisition
        logger.info("")
        logger.info("=" * 60)
        logger.info("Ready to start acquisition!")
        logger.info("=" * 60)
        response = input("Start acquisition now? (yes/no): ").strip().lower()
        
        if response != "yes":
            logger.info("Acquisition cancelled by user")
            engine.cleanup()
            return False
        
        # Run acquisition
        logger.info("")
        logger.info("=" * 60)
        logger.info("STARTING ACQUISITION")
        logger.info("=" * 60)
        logger.info("The GUI window should now be visible.")
        logger.info("Monitor the terminal for progress updates.")
        logger.info("=" * 60)
        logger.info("")
        
        import time
        start_time = time.time()
        
        engine.run_acquisition_loop()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Save outputs
        logger.info("")
        logger.info("=" * 60)
        logger.info("SAVING OUTPUTS")
        logger.info("=" * 60)
        
        engine._save_data()
        engine.save_metadata()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"✓ Demo completed successfully in {duration:.1f} seconds")
        logger.info("=" * 60)
        logger.info(f"Output directory: {engine.savedir}")
        logger.info("=" * 60)
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("Demo interrupted by user (Ctrl+C)")
        return False
        
    except Exception as e:
        logger.exception(f"Error during hardware demo: {e}")
        return False
        
    finally:
        # Always cleanup
        logger.info("Cleaning up hardware connections...")
        try:
            engine.cleanup()
            logger.info("✓ Cleanup complete")
        except:
            pass


def main():
    """Main entry point."""
    logger.info("Brainalyzer Hardware Demo Starting...")
    
    # Check if running on appropriate system
    if not sys.platform.startswith('win'):
        logger.warning("This demo is designed for Windows systems with Micro-Manager")
        response = input("Continue anyway? (yes/no): ").strip().lower()
        if response != "yes":
            logger.info("Demo cancelled")
            sys.exit(0)
    
    # Run demo
    success = run_hardware_demo()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
