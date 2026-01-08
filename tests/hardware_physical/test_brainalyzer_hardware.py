"""
Integration tests for Brainalyzer with physical hardware.

Tests the complete workflow:
1. Initialize hardware
2. Start Brainalyzer algorithm with GUI
3. Simulate ROI placement and stimulus trigger
4. Verify stimulus activation
5. Complete acquisition and save

Run with: pytest tests/hardware_physical/test_brainalyzer_hardware.py -v --hardware
"""

import pytest
import numpy as np
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.hardware

# Add parent directory to path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

@pytest.fixture
def brainalyzer_configs():
    """Load configurations for Brainalyzer hardware test."""
    from config.config_manager import ConfigManager
    
    config_manager = ConfigManager()
    config_dir = Path("./config/demo")
    
    config_manager.load_all_configs(
        hardware_path=config_dir / "hardware_physical_hardware.yaml",
        experiment_path=config_dir / "hardware_physical_experiment.yaml",
        algorithm_path=config_dir / "hardware_physical_algorithm.yaml"
    )
    
    return config_manager


@pytest.fixture
def closed_loop_engine(brainalyzer_configs):
    """Create ClosedLoopEngine for testing."""
    from engine.closed_loop_engine import ClosedLoopEngine
    
    engine = ClosedLoopEngine(
        hardware_config=brainalyzer_configs.hardware_config,
        experiment_config=brainalyzer_configs.experiment_config,
        algorithm_config=brainalyzer_configs.algorithm_config
    )
    
    # Override num_samples for faster testing
    engine.experiment_config.acquisition.num_samples = 80  # Just 10 volumes
    engine.samples_to_grab = 80
    
    return engine


class TestBrainalyzerHardwareIntegration:
    """Test complete Brainalyzer workflow with hardware."""
    
    def test_engine_initialization(self, closed_loop_engine):
        """Test that engine initializes with hardware."""
        logger.info("Testing engine initialization...")
        
        closed_loop_engine.initialize_hardware()
        assert closed_loop_engine.hardware.is_initialized
        logger.info("✓ Hardware initialized")
        
        closed_loop_engine.prepare_acquisition()
        assert closed_loop_engine.savedir is not None
        logger.info("✓ Acquisition prepared")
        
        closed_loop_engine.initialize_algorithm()
        assert closed_loop_engine.alg is not None
        logger.info("✓ Algorithm initialized")
        
        closed_loop_engine.initialize_stimulus()
        assert closed_loop_engine.stim_controller is not None
        logger.info("✓ Stimulus controller initialized")
        
        # Cleanup
        closed_loop_engine.cleanup()
    
    @pytest.mark.slow
    def test_short_acquisition_with_stimulus(self, closed_loop_engine):
        """Test short acquisition with stimulus trigger."""
        logger.info("Testing short acquisition with stimulus...")
        
        # Initialize
        closed_loop_engine.initialize_hardware()
        closed_loop_engine.prepare_acquisition()
        closed_loop_engine.initialize_algorithm()
        closed_loop_engine.initialize_stimulus()
        
        # Simulate stimulus event after 20 frames
        def inject_stimulus_event():
            """Inject a stimulus event into the algorithm."""
            time.sleep(2.0)  # Wait for acquisition to start
            
            logger.info("Injecting stimulus event...")
            event = {
                "ts": time.time(),
                "event_type": "pulse-rect-roi-list",
                "stim_intensity": 10,
                "stim_duration_vols": 3,
                "stim_rect_roi_list": {
                    "x": [600],
                    "y": [400],
                    "width": [50],
                    "height": [50]
                }
            }
            
            # Inject into algorithm
            closed_loop_engine.alg.current_event = event
            logger.info("✓ Stimulus event injected")
        
        # Start event injection in background
        import threading
        event_thread = threading.Thread(target=inject_stimulus_event, daemon=True)
        event_thread.start()
        
        # Run acquisition
        try:
            closed_loop_engine.run_acquisition_loop()
            
            # Verify stimulus was triggered
            assert len(closed_loop_engine.alg.stim_param_list) > 0
            logger.info(f"✓ Stimulus triggered {len(closed_loop_engine.alg.stim_param_list)} times")
            
            # Save outputs
            closed_loop_engine._save_data()
            closed_loop_engine.save_metadata()
            
            logger.info("✓ Short acquisition completed successfully")
            
        finally:
            closed_loop_engine.cleanup()


class TestPolygonStimulusController:
    """Test polygon stimulus controller with hardware."""
    
    @pytest.fixture
    def polygon_controller(self, brainalyzer_configs):
        """Create polygon controller for testing."""
        from hardware.hardware_manager import HardwareManager
        from hardware.stimulus_controllers.polygon_controller import PolygonStimulusController
        
        hardware = HardwareManager(brainalyzer_configs.hardware_config)
        hardware.initialize()

        # Also need to configure stimulus with data after hardware e.g. camera is initialized
        # Get stimulus interface type from hardware config
        stim_interface = brainalyzer_configs.hardware_config.stim_interface

        # Configure device
        logger.info('Configuring stimulus device')
        hardware.stimulus.configure_stimulus(config={'interface_type': stim_interface})

        # Build legacy args for controller
        args = {
            "gooey_args": {
                "trigger_algorithm": "Brainalyzer",
                "stim_interface": stim_interface
            },
            # "roi": [0, 0, 2048, 2048]
            "roi": hardware.camera.get_roi()
        }
        
        controller = PolygonStimulusController(hardware, args)
        controller.spool()
        
        yield controller
        
        hardware.close()
    
    def test_controller_spooling(self, polygon_controller):
        """Test that controller spools JIT functions."""
        logger.info("Testing polygon controller spooling...")
        # Spooling happens in fixture
        logger.info("✓ Controller spooled successfully")
    
    def test_submit_rect_roi_stimulus(self, polygon_controller):
        """Test submitting rectangle ROI stimulus."""
        logger.info("Testing rectangle ROI stimulus submission...")
        
        stim_params = {
            "stim_on": 10,
            "stim_off": 20,
            "stim_intensity": 10,
            "event": {
                "event_type": "pulse-rect-roi-list",
                "stim_rect_roi_list": {
                    "x": [600, 700],
                    "y": [400, 500],
                    "width": [50, 50],
                    "height": [50, 50]
                }
            }
        }
        
        polygon_controller.submit_stim_params(stim_params, image_ndx=0)
        
        assert len(polygon_controller.stim_on_list) == 1
        assert len(polygon_controller.stim_off_list) == 1
        logger.info("✓ Rectangle ROI stimulus submitted")
    
    def test_check_stim_activation(self, polygon_controller):
        """Test checking stimulus activation at correct frame."""
        logger.info("Testing stimulus activation check...")
        
        # Submit stimulus
        stim_params = {
            "stim_on": 10,
            "stim_off": 20,
            "stim_intensity": 15,
            "event": {
                "event_type": "pulse-rect-roi-list",
                "stim_rect_roi_list": {
                    "x": [600],
                    "y": [400],
                    "width": [50],
                    "height": [50]
                }
            }
        }
        
        polygon_controller.submit_stim_params(stim_params, image_ndx=0)
        
        # Check activation before stim_on
        polygon_controller.check_stim(img_count=5)
        assert not polygon_controller.hardware_manager.stimulus.is_stimulus_active()
        logger.info("✓ Stimulus inactive before stim_on")
        
        # Check activation at stim_on
        polygon_controller.check_stim(img_count=10)
        assert polygon_controller.hardware_manager.stimulus.is_stimulus_active()
        logger.info("✓ Stimulus activated at stim_on")
        
        # Check deactivation at stim_off
        polygon_controller.check_stim(img_count=20)
        assert not polygon_controller.hardware_manager.stimulus.is_stimulus_active()
        logger.info("✓ Stimulus deactivated at stim_off")
