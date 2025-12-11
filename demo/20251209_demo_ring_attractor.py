"""
Ring Attractor Closed-Loop Demo

Complete demonstration of CLEF closed-loop capabilities using a dual ring attractor
dynamical system. The system:

1. Simulates a bistable dynamical system with two stable limit cycles
2. Radial dynamics: dr/dt = -k*(r - r1)*(r - r2) + perturbation
3. Angular dynamics: dθ/dt = ω (constant rotation)
4. Camera observes puncta position on trajectory
5. Algorithm extracts (theta, ring_index) state from images
6. Interactive GUI for manual stimulus triggering with intensity control
7. Optional auto-triggering when theta enters defined range
8. Stimulus applies radial perturbation to push system between rings

This demonstrates:
- Custom hardware backend (RingAttractorBackend) with bistable limit cycles
- Classical dynamical system (not neural network)
- Custom algorithm with state extraction (RingAttractorAlgorithm)
- Interactive real-time visualization with PyQt
- Manual stimulus control with adjustable intensity
- Optional auto-triggering based on state
- Ring transitions via perturbation crossing potential barrier
- Trajectory tracking on stable limit cycles
- Metadata capture and visualization

Usage:
    python demo/20251209_demo_ring_attractor.py
    
    or via CLI:
    clef-cli --hardware config/demo/demo_ring_attractor_hardware.yaml \
             --experiment config/demo/demo_ring_attractor_experiment.yaml \
             --algorithm config/demo/demo_ring_attractor_algorithm.yaml
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
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # <--- Remove this if the debug output is overwhelming lol
)
logger = logging.getLogger(__name__)


def print_demo_header():
    """Print informative demo header."""
    print("=" * 80)
    print("RING ATTRACTOR CLOSED-LOOP DEMO")
    print("=" * 80)
    print()
    print("This demo showcases CLEF's closed-loop capabilities with ring attractors.")
    print()
    print("What happens:")
    print("  1. Bistable dynamical system with two stable limit cycles at r1=30px, r2=45px")
    print("  2. Radial dynamics: dr/dt = -k*(r-r1)*(r-r2) + u (double-well potential)")
    print("  3. Angular dynamics: dθ/dt = ω (constant rotation around current ring)")
    print("  4. Camera observes puncta position following the trajectory")
    print("  5. Algorithm extracts (theta, ring_index) state from each frame")
    print("  6. Interactive GUI provides:")
    print("     • 'Trigger Stimulus' button for manual control")
    print("     • Intensity slider (0-100%) to scale perturbation")
    print("     • Optional auto-trigger checkbox (theta ∈ [0, π/4])")
    print("  7. Stimulus effects:")
    print("     • Applies radial perturbation to push across potential barrier")
    print("     • System naturally relaxes to other ring's limit cycle")
    print("     • 50-frame cooldown prevents rapid re-triggering")
    print()
    print("Expected behavior:")
    print("  - Puncta smoothly orbits current ring (stable limit cycle)")
    print("  - Manual triggers apply radial push, causing ring transition")
    print("  - System settles onto new ring's orbit")
    print("  - State space plot shows circular trajectories on both rings")
    print("  - Color-coded by ring (blue=inner, red=outer)")
    print()
    print("Output:")
    print("  - Images saved as TIFF stack (100x100 uint16)")
    print("  - Metadata with state timeseries and stimulus events")
    print("  - Polar and Cartesian trajectory plots")
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
        
        # Validate ring-specific metadata
        if 'alg_metadata' in metadata:
            alg_meta = metadata['alg_metadata']
            
            # Check for state timeseries
            if all(k in alg_meta for k in ['theta_history', 'ring_history']):
                num_states = len(alg_meta['theta_history'])
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
                    first_event = alg_meta['stim_events'][0]
                    logger.info(
                        f"    First event at frame {first_event['frame']}, "
                        f"theta={first_event['theta']:.3f}, ring={first_event['ring']}"
                    )
                checks.append(True)
            
            # Check transitions
            if 'num_transitions' in alg_meta:
                num_trans = alg_meta['num_transitions']
                logger.info(f"  ✓ Detected {num_trans} ring transitions")
                checks.append(True)
        
        # Check ring parameters in hardware metadata
        if 'hardware_metadata' in metadata:
            hw_meta = metadata['hardware_metadata']
            if 'ring_inner_radius' in hw_meta:
                logger.info(
                    f"  ✓ Ring parameters: inner_r={hw_meta['ring_inner_radius']}, "
                    f"outer_r={hw_meta['ring_outer_radius']}"
                )
                checks.append(True)
    else:
        logger.error(f"✗ Metadata file not found: {metadata_file}")
        checks.append(False)
    
    # Check 3: Image TIFF
    tiff_file = session_dir / f"{session_id}.tiff"
    if tiff_file.exists():
        logger.info(f"✓ Image TIFF: {tiff_file}")
        
        try:
            import tifffile as tf
            img_stack = tf.imread(tiff_file)
            logger.info(f"  Shape: {img_stack.shape}")
            
            if len(img_stack.shape) == 3 and img_stack.shape[1:] == (100, 100):
                logger.info(f"  ✓ Correct dimensions (100x100 images)")
                checks.append(True)
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


def run_ring_attractor_demo(config_dir: Path = None):
    """
    Run complete ring attractor demo.
    
    Args:
        config_dir: Directory containing config files (default: config/demo/)
    """
    print_demo_header()
    
    # Default config directory
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "config" / "demo"
    
    # Config file paths
    hw_config_path = config_dir / "demo_ring_attractor_hardware.yaml"
    exp_config_path = config_dir / "demo_ring_attractor_experiment.yaml"
    alg_config_path = config_dir / "demo_ring_attractor_algorithm.yaml"
    
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
        logger.info("Starting ring attractor closed-loop demo...")
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
    success = run_ring_attractor_demo(config_dir)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
