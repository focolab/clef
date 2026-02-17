"""
Dummy stimulus controller for testing.

Designed for lorenz demo, demonstrating custom functionality without
real hardware. 
"""
import logging
from typing import Dict, Any

from hardware.stimulus_controllers.dummy_controller import DummyStimulusController

logger = logging.getLogger(__name__)

class LorenzStimulusController(DummyStimulusController):
    """Lorenz-specific stimulus controller extending dummy controller."""

    def __init__(self, hardware_manager):
        """
        Initialize Lorenz demo stimulus controller.

        Args:
            hardware_manager: HardwareManager instance
        """
        super().__init__(hardware_manager)
    
    def submit_stim_params(self, stim_params: Dict[str, Any], image_ndx: int) -> None:
        """
        Accept params and track them, with Lorenz-specific processing.
        
        Args:
            stim_params: Dictionary containing stimulus parameters
            image_ndx: Current image index
        """

        if not stim_params:
            return

        # Call parent implementation
        super().submit_stim_params(stim_params, image_ndx)
        
        # Here we're adding some additional functionality, storing a new value we expect in stim_params
        self.perturbation = stim_params.get('event').get('perturbation')

    def _activate_hardware(self, intensity: float) -> None:
        """
        Activate dummy stimulus hardware.
        
        Args:
            intensity: Stimulus intensity value
        """

        # Here we're making sure to pass that to our new stim interface
        params = {"intensity": intensity, "perturbation": self.perturbation}
        self.hardware_manager.stimulus.activate_stimulus(params)
        logger.debug(f"LorenzStimulusController: Activated stimulus at intensity {intensity}")
