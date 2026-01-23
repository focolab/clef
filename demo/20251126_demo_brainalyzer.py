"""
Brainalyzer Demo Integration Test

End-to-end demonstration of Brainalyzer algorithm with GUI using:
- Config-based initialization
- DummyHardwareBackend (reads from TIFF file)
- Interactive GUI for ROI placement and stimulus triggering
- Full closed-loop workflow

This demo validates:
1. Config object initialization
2. Hardware abstraction layer
3. Algorithm processing pipeline
4. GUI subprocess communication
5. Stimulus event handling
6. Metadata capture and saving

Usage:
    python demo/20251126_demo_brainalyzer.py
    
Requirements:
    - PyQt5/PyQt6 installed
    - Input TIFF file with image stack
    - ~2 minutes runtime
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import (
    HardwareConfig,
    ExperimentConfig,
    AlgorithmConfig,
    AcquisitionConfig,
    SubjectMetadata,
    AlgorithmParameters,
    StimulusParameters,
)
from engine.closed_loop_engine import ClosedLoopEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_configs(input_tiff_path: str) -> dict:
    """
    Create configuration objects for Brainalyzer demo.
    
    Args:
        input_tiff_path: Path to input TIFF file for dummy backend
        
    Returns:
        Dictionary with Config objects
    """
    
    # Hardware Config - Use dummy backend with TIFF input
    hardware_config = HardwareConfig(
        backend="dummy",
        stim_interface="dummy",
        microscope_name="demo_microscope",
        strobe_acquisition=False,
    )
    
    # Acquisition Config
    acquisition_config = AcquisitionConfig(
        num_frames=800,  # Will read from TIFF
        z_planes=8,
        z_step=3,
        baseline_frames=0,
        save_structural_scan="none",
    )
    
    # Subject Metadata
    subject_metadata = SubjectMetadata(
        genotype="demo_strain",
        notes="Brainalyzer demo with interactive GUI",
    )
    
    # Experiment Config
    experiment_config = ExperimentConfig(
        experiment_name="brainalyzer_demo",
        output_dir="./demo_output",
        save_images=True,
        save_metadata=True,
        save_sample_video=False,
        acquisition=acquisition_config,
        subject=subject_metadata,
        input_recording_path=input_tiff_path,  # Key for dummy backend
        dev_options={
            # "prefill_wb_ops": False,
            "send_sms_on_completion": False,
        }
    )
    
    # Algorithm Parameters
    algorithm_params = AlgorithmParameters(
        stim_cooldown_frames=50,  # Short cooldown for demo
        skip_stimulation_probability=0.0,
        delay_stimulation_probability=0.0,
        stimulus_diameter_pixels=30,
    )
    
    # Stimulus Parameters
    stimulus_params = StimulusParameters(
        enabled=True,
        duration_frames_options=[20, 40],  # 2-4 volumes
        intensity_percent_options=[10, 20, 30],
        duration_frames=20,
        intensity_percent=10,
    )
    
    # Algorithm Config - Enable GUI for interactive demo
    algorithm_config = AlgorithmConfig(
        algorithm_type="Brainalyzer",
        enable_gui=True,
        gui_mode="neural_imaging",
        save_algorithm_plot=True,
        algorithm_params=algorithm_params,
        stimulus_params=stimulus_params,
    )
    
    return {
        "hardware": hardware_config,
        "experiment": experiment_config,
        "algorithm": algorithm_config,
    }


def validate_demo_outputs(output_dir: str, session_id: str) -> bool:
    """
    Validate that demo produced expected outputs.
    
    Args:
        output_dir: Directory where outputs were saved
        session_id: Session ID for this run
        
    Returns:
        True if all validations pass, False otherwise
    """
    logger.info("Validating demo outputs...")
    
    checks = []
    
    # Check 1: Output directory exists
    session_dir = Path(output_dir) / session_id
    if session_dir.exists():
        logger.info(f"✓ Output directory created: {session_dir}")
        checks.append(True)
    else:
        logger.error(f"✗ Output directory not found: {session_dir}")
        checks.append(False)
    
    # Check 2: Metadata file exists
    metadata_file = session_dir / f"{session_id}_metadata.json"
    if metadata_file.exists():
        logger.info(f"✓ Metadata file created: {metadata_file}")
        
        # Validate metadata contents
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check for key metadata fields
        required_fields = [
            'algorithm_config',
            'experiment_config',
            'hardware_config',
            'stim_metadata',
            'alg_metadata',
        ]
        
        for field in required_fields:
            if field in metadata:
                logger.info(f"  ✓ Metadata contains '{field}'")
            else:
                logger.warning(f"  ✗ Metadata missing '{field}'")
        
        # Check for stimulus events in algorithm metadata
        if 'alg_metadata' in metadata and 'stim_param_list' in metadata['alg_metadata']:
            stim_events = metadata['alg_metadata']['stim_param_list']
            logger.info(f"  ✓ Algorithm recorded {len(stim_events)} stimulus events")
            checks.append(True)
        else:
            logger.warning("  ✗ No stimulus events recorded")
            checks.append(False)
    else:
        logger.error(f"✗ Metadata file not found: {metadata_file}")
        checks.append(False)
    
    # Check 3: Image TIFF exists (if saving enabled)
    tiff_file = session_dir / f"{session_id}.tiff"
    if tiff_file.exists():
        logger.info(f"✓ Image TIFF created: {tiff_file}")
        checks.append(True)
    else:
        logger.warning(f"⚠ Image TIFF not found (may be disabled): {tiff_file}")
    
    # Check 4: Algorithm plot exists (if enabled)
    plot_file = session_dir / f"{session_id}_live_stim_fig.svg"
    if plot_file.exists():
        logger.info(f"✓ Algorithm plot created: {plot_file}")
        checks.append(True)
    
    # Overall result
    success = all(checks)
    if success:
        logger.info("=" * 60)
        logger.info("✓ All validation checks passed!")
        logger.info("=" * 60)
    else:
        logger.warning("=" * 60)
        logger.warning("⚠ Some validation checks failed")
        logger.warning("=" * 60)
    
    return success


def run_brainalyzer_demo(input_tiff_path: str):
    """
    Run complete Brainalyzer demo workflow.
    
    Args:
        input_tiff_path: Path to input TIFF file
    """
    logger.info("=" * 60)
    logger.info("Brainalyzer Interactive Demo")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This demo will:")
    logger.info("1. Load configurations from YAML defaults")
    logger.info("2. Initialize hardware (DummyBackend with TIFF input)")
    logger.info("3. Launch Brainalyzer GUI (PyQt)")
    logger.info("4. Process frames through algorithm")
    logger.info("5. Allow interactive ROI placement and stimulus triggering")
    logger.info("6. Save metadata and validate outputs")
    logger.info("")
    logger.info("=" * 60)
    logger.info("")
    
    # Verify input file exists
    if not os.path.exists(input_tiff_path):
        logger.error(f"Input TIFF not found: {input_tiff_path}")
        logger.error("Please provide a valid input TIFF path")
        return False
    
    logger.info(f"Using input TIFF: {input_tiff_path}")
    
    try:
        # Create demo configurations
        logger.info("Creating demo configurations...")
        configs = create_demo_configs(input_tiff_path)
        
        # Create ClosedLoopEngine
        logger.info("Initializing ClosedLoopEngine...")
        engine = ClosedLoopEngine(
            hardware_config=configs["hardware"],
            experiment_config=configs["experiment"],
            algorithm_config=configs["algorithm"]
        )
        
        # Run acquisition
        logger.info("Starting acquisition with Brainalyzer GUI...")
        logger.info("")
        logger.info("=" * 60)
        logger.info("INTERACTIVE DEMO INSTRUCTIONS")
        logger.info("=" * 60)
        logger.info("1. GUI will open showing real-time image display")
        logger.info("2. Use mouse to click on z-plane images")
        logger.info("3. Add quantification ROIs (green) to track intensity")
        logger.info("4. Add stimulus ROIs (red, dotted) to define stim regions")
        logger.info("5. Click 'pulse stimulate ROI(s)' to trigger stimulus")
        logger.info("6. Watch ROI intensity plots update in real-time")
        logger.info("7. Demo will run for ~100 frames then save outputs")
        logger.info("=" * 60)
        logger.info("")
        
        # Record start time
        start_time = time.time()
        
        # Run the engine
        engine.run()
        
        # Record end time
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Demo completed in {duration:.1f} seconds")
        logger.info("=" * 60)
        
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
    
    # Default input TIFF path
    # Users should replace this with their own TIFF file
    # default_tiff = "./demo_data/sample_recording.tiff"
    default_tiff = 'C:/Users/rldun/Desktop/temp_render/example_data/20221106-21-47-31/20221106-21-47-1_minified_8z_100t.tif'
    
    # Check if custom path provided via command line
    if len(sys.argv) > 1:
        input_tiff = sys.argv[1]
    else:
        input_tiff = default_tiff
    
    # Check for environment variable override
    if os.environ.get("BRAINALYZER_DEMO_TIFF"):
        input_tiff = os.environ["BRAINALYZER_DEMO_TIFF"]
    
    logger.info(f"Demo input TIFF: {input_tiff}")
    
    # Run the demo
    success = run_brainalyzer_demo(input_tiff)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()