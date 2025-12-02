"""
Lorenz Attractor Closed-Loop Demo

Complete demonstration of CLEF closed-loop capabilities using a Lorenz attractor
dynamical system. The system:

1. Simulates a camera observing Lorenz dynamics encoded as 3 bright pixels
2. Extracts state (x, y, z) from each frame by finding local maxima
3. Monitors phase space trajectory
4. Triggers stimulation when trajectory enters a defined 3D volume
5. Perturbation pushes the system's state, affecting future dynamics

This demonstrates:
- Custom hardware backend (LorenzDemoBackend)
- Custom algorithm with state extraction (LorenzDemoAlgorithm)
- Closed-loop triggering based on extracted features
- State-dependent perturbation (stimulus affects dynamics)
- Real-time monitoring of dynamical system evolution
- Metadata capture and visualization

Usage:
    python demo/20251201_demo_lorenz.py
    
    or via CLI:
    clef-cli --hardware config/demo/demo_lorenz_hardware.yaml \
             --experiment config/demo/demo_lorenz_experiment.yaml \
             --algorithm config/demo/demo_lorenz_algorithm.yaml
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


def print_demo_header():
    """Print informative demo header."""
    print("=" * 80)
    print("LORENZ ATTRACTOR CLOSED-LOOP DEMO")
    print("=" * 80)
    print()
    print("This demo showcases CLEF's closed-loop capabilities with a dynamical system.")
    print()
    print("What happens:")
    print("  1. Lorenz attractor evolves according to classic chaotic dynamics")
    print("  2. Each frame encodes state (x,y,z) as 3 bright pixels in a 20x20 image")
    print("  3. Algorithm extracts state by finding local maxima")
    print("  4. When trajectory enters trigger volume (x>0, y>0, z>25):")
    print("     → Stimulus perturbs state by [+2, +2, +2]")
    print("     → Cooldown period of 100 frames prevents rapid re-triggering")
    print("  5. System continues evolving from perturbed state")
    print()
    print("Expected behavior:")
    print("  - Trajectory explores classic butterfly-shaped attractor")
    print("  - ~3-5 stimulation events over 500 frames")
    print("  - Perturbations briefly shift trajectory but dynamics remain chaotic")
    print()
    print("Output:")
    print("  - Images saved as TIFF stack (3-pixel encoding of Lorenz state)")
    print("  - Metadata with extracted state timeseries and stimulus events")
    print("  - 3D phase space plot showing trajectory and trigger volume")
    print()
    print("=" * 80)
    print()


def validate_demo_outputs(output_dir: str, session_id: str) -> bool:
    """
    Validate that demo produced expected outputs.
    
    Args:
        output_dir: Directory where outputs were saved
        session_id: Session ID for this run
        
    Returns:
        True if all validations pass
    """
    logger.info("Validating demo outputs...")
    
    session_dir = Path(output_dir) / session_id
    checks = []
    
    # Check 1: Output directory
    if session_dir.exists():
        logger.info(f"✓ Output directory: {session_dir}")
        checks.append(True)
    else:
        logger.error(f"✗ Output directory not found: {session_dir}")
        checks.append(False)
        return False
    
    # Check 2: Metadata file
    metadata_file = session_dir / f"{session_id}_metadata.json"
    if metadata_file.exists():
        logger.info(f"✓ Metadata file: {metadata_file}")
        
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Validate Lorenz-specific metadata
        if 'alg_metadata' in metadata:
            alg_meta = metadata['alg_metadata']
            
            # Check for state timeseries
            if all(k in alg_meta for k in ['x_history', 'y_history', 'z_history']):
                num_states = len(alg_meta['x_history'])
                logger.info(f"  ✓ Extracted {num_states} state vectors")
                checks.append(True)
            else:
                logger.warning("  ✗ Missing state timeseries")
                checks.append(False)
            
            # Check for stimulus events
            if 'stim_events' in alg_meta:
                num_events = len(alg_meta['stim_events'])
                logger.info(f"  ✓ Recorded {num_events} stimulus events")
                
                if num_events > 0:
                    # Show first event details
                    first_event = alg_meta['stim_events'][0]
                    logger.info(f"    First event at frame {first_event['frame']}, "
                              f"state={first_event['state']}")
                checks.append(True)
            else:
                logger.warning("  ✗ No stimulus events recorded")
                checks.append(False)
            
            # Check for trigger volume configuration
            if 'trigger_volume' in alg_meta:
                vol = alg_meta['trigger_volume']
                logger.info(f"  ✓ Trigger volume: x=[{vol['x_min']}, {vol['x_max']}], "
                          f"y=[{vol['y_min']}, {vol['y_max']}], "
                          f"z=[{vol['z_min']}, {vol['z_max']}]")
                checks.append(True)
        
        # Check Lorenz parameters in hardware metadata
        if 'hardware_config' in metadata:
            hw_meta = metadata['hardware_config']
            if 'lorenz_sigma' in hw_meta:
                logger.info(f"  ✓ Lorenz parameters: σ={hw_meta['lorenz_sigma']}, "
                          f"ρ={hw_meta['lorenz_rho']}, β={hw_meta['lorenz_beta']:.3f}")
                checks.append(True)
    else:
        logger.error(f"✗ Metadata file not found: {metadata_file}")
        checks.append(False)
    
    # Check 3: Image TIFF
    tiff_file = session_dir / f"{session_id}.tiff"
    if tiff_file.exists():
        logger.info(f"✓ Image TIFF: {tiff_file}")
        
        # Try to read and validate
        try:
            import tifffile as tf
            img_stack = tf.imread(tiff_file)
            logger.info(f"  Shape: {img_stack.shape}")
            
            # Should be (T, H, W) with H=W=20
            if len(img_stack.shape) == 3 and img_stack.shape[1:] == (100, 100):
                logger.info(f"  ✓ Correct dimensions (100x100 images)")
                checks.append(True)
            else:
                logger.warning(f"  ⚠ Unexpected dimensions")
        except Exception as e:
            logger.warning(f"  Could not read TIFF: {e}")
    else:
        logger.warning(f"⚠ Image TIFF not found: {tiff_file}")
    
    # Check 4: Algorithm plot
    plot_file = session_dir / f"{session_id}_live_stim_fig.svg"
    if plot_file.exists():
        logger.info(f"✓ Algorithm plot: {plot_file}")
        checks.append(True)
    else:
        # Try PNG
        plot_file_png = session_dir / f"{session_id}_live_stim_fig.png"
        if plot_file_png.exists():
            logger.info(f"✓ Algorithm plot: {plot_file_png}")
            checks.append(True)
    
    # Overall result
    success = all(checks)
    print()
    print("=" * 80)
    if success:
        print("✓ ALL VALIDATION CHECKS PASSED")
        print()
        print("Demo completed successfully! Check output directory for:")
        print(f"  - Images: {session_dir / (session_id + '.tiff')}")
        print(f"  - Metadata: {metadata_file}")
        print(f"  - Plot: {plot_file}")
    else:
        print("⚠ SOME VALIDATION CHECKS FAILED")
        print("Check logs above for details")
    print("=" * 80)
    print()
    
    return success


def run_lorenz_demo(config_dir: Path = None):
    """
    Run complete Lorenz attractor demo.
    
    Args:
        config_dir: Directory containing config files (default: config/demo/)
    """
    print_demo_header()
    
    # Default config directory
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "config" / "demo"
    
    # Config file paths
    hw_config_path = config_dir / "demo_lorenz_hardware.yaml"
    exp_config_path = config_dir / "demo_lorenz_experiment.yaml"
    alg_config_path = config_dir / "demo_lorenz_algorithm.yaml"
    
    # Verify config files exist
    for path in [hw_config_path, exp_config_path, alg_config_path]:
        if not path.exists():
            logger.error(f"Config file not found: {path}")
            logger.error("Please ensure config files are in config/demo/ directory")
            return False
    
    logger.info(f"Loading configurations from {config_dir}")
    
    try:
        # Load configurations
        config_manager = ConfigManager()
        
        hardware_config = config_manager.load_hardware_config(hw_config_path)
        experiment_config = config_manager.load_experiment_config(exp_config_path)
        algorithm_config = config_manager.load_algorithm_config(alg_config_path)
        
        logger.info("Configurations loaded successfully")
        
        # Create ClosedLoopEngine
        logger.info("Initializing ClosedLoopEngine...")
        engine = ClosedLoopEngine(
            hardware_config=hardware_config,
            experiment_config=experiment_config,
            algorithm_config=algorithm_config
        )
        
        # Run acquisition
        logger.info("Starting Lorenz attractor closed-loop demo...")
        logger.info("")
        
        start_time = time.time()
        
        # Run the engine
        engine.run()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Demo completed in {duration:.1f} seconds")
        logger.info("=" * 80)
        logger.info("")
        
        # Validate outputs
        success = validate_demo_outputs(
            engine.experiment_config.output_dir,
            engine.session_id
        )
        
        return success
        
    except Exception as err:
        logger.exception(f"Error during demo: {err}")
        return False


def main():
    """Main entry point for demo."""
    
    # Check for custom config directory
    if len(sys.argv) > 1:
        config_dir = Path(sys.argv[1])
        logger.info(f"Using custom config directory: {config_dir}")
    else:
        config_dir = None
    
    # Run the demo
    success = run_lorenz_demo(config_dir)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
