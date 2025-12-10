"""
Ring Attractor Stimulus Controller

Extends DummyStimulusController with ring-specific functionality.
Accepts intensity parameter and passes it to the stimulus interface.
"""

import logging
from typing import Dict, Any

from hardware.stimulus_controllers.dummy_controller import DummyStimulusController

logger = logging.getLogger(__name__)


class RingAttractorStimulusController(DummyStimulusController):
    """Ring attractor stimulus controller with intensity control."""
    
    def __init__(self, hardware_manager, config: Dict[str, Any]):
        """
        Initialize ring attractor stimulus controller.
        
        Args:
            hardware_manager: HardwareManager instance
            config: Configuration dictionary
        """
        super().__init__(hardware_manager, config)
        
        # Default intensity (can be updated by algorithm)
        self.stim_intensity = 50  # 0-100%
        
        logger.info("RingAttractorStimulusController initialized")
    
    def submit_stim_params(self, stim_params: Dict[str, Any], image_ndx: int) -> None:
        """
        Accept and process ring-specific stimulus parameters.
        
        Args:
            stim_params: Dictionary containing stimulus parameters
            image_ndx: Current image index
        """
        if not stim_params:
            return
        
        # Extract intensity from event if provided
        if 'event' in stim_params and 'stim_intensity' in stim_params['event']:
            self.stim_intensity = stim_params['event']['stim_intensity']
        
        # Call parent implementation
        super().submit_stim_params(stim_params, image_ndx)
        
        logger.debug(f"Ring stimulus submitted with intensity {self.stim_intensity}%")
    
    def _activate_hardware(self, intensity: float) -> None:
        """
        Activate stimulus hardware with ring-specific parameters.
        
        Args:
            intensity: Stimulus intensity (not used, we use self.stim_intensity)
        """
        # Pass intensity to stimulus interface
        params = {"intensity": self.stim_intensity}
        self.hardware_manager.stimulus.activate_stimulus(params)
        
        logger.debug(f"Activated ring stimulus at intensity {self.stim_intensity}%")
