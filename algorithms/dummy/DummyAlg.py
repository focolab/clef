"""
Dummy Algorithm for CLEF

A minimal no-op algorithm for testing and as a template for new algorithms.
This algorithm does nothing - it processes frames without triggering any stimuli.
"""

import logging
import random
from typing import Optional, Dict, Any
logger = logging.getLogger(__name__)

# Import config models
from config.config_manager import (
    AlgorithmConfig,
    ExperimentConfig,
)
from hardware.hardware_manager import HardwareManager

class DummyAlg:
    """
    Dummy algorithm that does nothing.
    
    This is useful for:
    - Testing the acquisition system without closed-loop logic
    - Template for implementing new algorithms
    - Fallback when a specified algorithm fails to load
    """
    
    def __init__(
        self,
        algorithm_config: Optional[AlgorithmConfig] = None,
        experiment_config: Optional[ExperimentConfig] = None,
        hardware_manager: Optional[HardwareManager] = None,
        local_handles: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None
        ):
        """
        Initialize dummy algorithm.
        
        Args:
            algorithm_config: Algorithm configuration (type, params, etc.)
            experiment_config: Experiment configuration (for building args)
            hardware_config: Hardware configuration (for building args)
            local_handles: Dictionary of local handles (e.g., {'mmc': mmc_instance})
            args: Legacy args dictionary (optional, for compatibility)
        """
        if args is None:
            args = {}
        if local_handles is None:
            local_handles = {}
            
        self.args = args
        self.local_handles = local_handles
        
        # Extract commonly used parameters with safe defaults
        gooey_args = self.args.get("gooey_args", {})
        self.samples_to_grab = gooey_args.get("total_frames", 100)
        self.zsize = gooey_args.get("zsize", 1)
        
        # Get ROI info if available
        self.roi = self.args.get("roi", [0, 0, 512, 512])
        self.xsize = self.roi[2]
        self.ysize = self.roi[3]
        
        # State
        self.sample_count = 0
        self.volume_count = 0
        
        logger.info("DummyAlg initialized (no-op algorithm)")
    
    def initialize_model(self):
        """
        Initialize the algorithm model.
        
        For dummy algorithm, this just seeds the RNG for reproducibility.
        """
        # Seed RNG for reproducibility if we have a session ID
        session_id = self.args.get("id", "default")
        random.seed(session_id)
        logger.debug(f"DummyAlg model initialized with seed: {session_id}")
    
    def process_sample(self, img, sample_ndx):
        """
        Process a single frame.
        
        Args:
            img: Image array (numpy array)
            zndx: Z-plane index
        """
        self.sample_count = self.sample_count + 1
        
        # Check if we completed a volume
        zndx = sample_ndx % self.zsize
        if zndx == (self.zsize - 1):
            logging.debug(f'Finished volume with zndx: {zndx}, self.zsize: {self.zsize}, sample_ndx: {sample_ndx}')
            self.process_volume()
            
        # Log periodically
        if self.sample_count % 500 == 0:
            logger.debug(
                f"DummyAlg processed {self.sample_count} samples "
                f"({self.volume_count} volumes)"
            )
    
    def process_volume(self):
        """
        Process a completed volume (called after full z-stack).
        
        For dummy algorithm, this does nothing.
        """
        self.volume_count = self.volume_count + 1
    
    def check_stim(self, image_ndx, cooldown_counter=0):
        """
        Check if stimulus should be triggered.
        
        Args:
            image_ndx: Current image index
            cooldown_counter: Current cooldown counter value
            
        Returns:
            tuple: (stim_params dict, new_cooldown_counter)
                - stim_params: Empty dict (no stimulation)
                - new_cooldown_counter: 0 (no cooldown needed)
        """
        # Dummy algorithm never triggers stimulation
        return {}, 0
    
    def get_metadata(self, args=None):
        """
        Return metadata captured during runtime.
        
        Args:
            args: Optional args dict (for compatibility)
            
        Returns:
            dict: Metadata dictionary
        """
        return {
            "algorithm_type": "DummyAlg",
            "samples_processed": self.sample_count,
            "volumes_processed": self.volume_count,
            "description": "No-op algorithm for testing",
            "is_dummy_alg": True,
        }
    
    def plot_model(self, show_plot=False, savefilename=None):
        """
        Plot the current state of the model.
        
        Args:
            show_plot: Whether to display the plot
            savefilename: If provided, save plot to this file
        """
        logger.info("DummyAlg has no model to plot")
    
    def close(self):
        """
        Clean up resources and close the algorithm.
        """
        logger.info(
            f"DummyAlg closing. Processed {self.sample_count} frames, "
            f"{self.volume_count} volumes"
        )


# Standalone testing
if __name__ == "__main__":
    import numpy as np
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing DummyAlg...")
    
    # Create test args
    test_args = {
        "id": "test_session",
        "roi": [0, 0, 512, 512],
        "gooey_args": {
            "total_samples": 100,
            "zsize": 10,
        }
    }
    
    # Initialize algorithm
    alg = DummyAlg(test_args)
    alg.initialize_model()
    
    # Simulate processing some frames
    for i in range(50):
        img = np.random.randint(0, 255, (512, 512), dtype=np.uint16)
        zndx = i % 10
        
        alg.process_sample(img, i)
        
        # Check stim every frame
        stim_params, cooldown = alg.check_stim(i, 0)
        
        if stim_params:
            print(f"Frame {i}: Triggered stim (shouldn't happen!)")
    
    # Get metadata
    metadata = alg.get_metadata()
    print(f"\nMetadata: {metadata}")
    
    # Close
    alg.close()
    
    print("\nDummyAlg test complete!")