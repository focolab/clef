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
    """Ring attractor stimulus controller with intensity and omega perturbation control."""
    
    def __init__(self, hardware_manager, config: Dict[str, Any]):
        """
        Initialize ring attractor stimulus controller.
        
        Args:
            hardware_manager: HardwareManager instance
            config: Configuration dictionary
        """
        super().__init__(hardware_manager, config)
        
        # Default parameters (can be updated by algorithm)
        self.stim_intensity = 0  # Radial perturbation: -30 to +30
        self.omega_perturbation = 0.0  # Angular velocity perturbation: -5 to +5
        
        logger.info("RingAttractorStimulusController initialized")
    
    def submit_stim_params(self, stim_params: Dict[str, Any], image_ndx: int) -> None:
        """
        Accept and process ring-specific stimulus parameters.
        
        Args:
            stim_params: Dictionary containing stimulus parameters including:
                - event: Dict with 'stim_intensity' and 'omega_perturbation'
            image_ndx: Current image index
        """
        if not stim_params:
            return
        
        logger.debug(f"RingAttractorStimulusController: Received stim params at frame {image_ndx}: {stim_params}")
        
        # Call parent implementation to handle timing
        super().submit_stim_params(stim_params, image_ndx)        

        # Extract ring-specific parameters
        event = stim_params.get('event', {})
        self.stim_intensity = event.get('stim_intensity', 0) 
        self.omega_perturbation = event.get('omega_perturbation', 0.0)
        
        logger.debug(
            f"Ring stimulus submitted: intensity={self.stim_intensity}, "
            f"omega_perturbation={self.omega_perturbation:.2f}"
        )
    
    def _activate_hardware(self, intensity: float) -> None:
        """
        Activate stimulus hardware with ring-specific parameters.
        
        Args:
            intensity: Stimulus intensity (from parent class, not used directly)
        """
        # Pass both radial and angular perturbations to stimulus interface
        params = {
            "intensity": self.stim_intensity,
            "omega_perturbation": self.omega_perturbation
        }
        self.hardware_manager.stimulus.activate_stimulus(params)
        
        logger.debug(
            f"Activated ring stimulus: intensity={self.stim_intensity}, "
            f"omega_perturbation={self.omega_perturbation:.2f}"
        )