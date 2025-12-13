"""
Dummy stimulus controller for testing.

Provides stimulus controller functionality without real hardware,
using the dummy hardware backend.
"""

import logging
from typing import Dict, Any

from hardware.stimulus_controllers.base_controller import BaseStimulusController

logger = logging.getLogger(__name__)


class DummyStimulusController(BaseStimulusController):
    """Dummy stimulus controller for testing."""
    
    def submit_stim_params(self, stim_params: Dict[str, Any], image_ndx: int) -> None:
        """
        Accept params and track them (but don't control real hardware).
        
        Args:
            stim_params: Dictionary containing stimulus parameters
            image_ndx: Current image index
        """
        if not stim_params:
            return
        
        logger.debug(f"DummyStimulusController: Received stim params at frame {image_ndx}: {stim_params}")
        
        # Extract intensity if present
        if "event" in stim_params and "stim_intensity" in stim_params["event"]:
            self.stim_intensity_list.append(stim_params["event"]["stim_intensity"])
        
        # Check if stimulus is pulsed (has both on and off times specified)
        if stim_params.get("stim_on") is not None and stim_params.get("stim_off") is not None:
            self.stim_param_list.append(stim_params)
            self.stim_on_list.append(stim_params["stim_on"])
            self.stim_off_list.append(stim_params["stim_off"])
    
    def _activate_hardware(self, intensity: float) -> None:
        """
        Activate dummy stimulus hardware.
        
        Args:
            intensity: Stimulus intensity value
        """
        params = {"intensity": intensity}
        self.hardware_manager.stimulus.activate_stimulus(params)
        logger.debug(f"DummyStimulusController: Activated stimulus at intensity {intensity}")
    
    def _deactivate_hardware(self) -> None:
        """Deactivate dummy stimulus hardware."""
        self.hardware_manager.stimulus.deactivate_stimulus()
        logger.debug("DummyStimulusController: Deactivated stimulus")