"""
Simple stimulus controller for testing and basic use cases.

Stores the most recent stim_params and delegates activation to hardware
without requiring any runtime configuration.
"""

import logging
from typing import Dict, Any

from hardware.stimulus_controllers.base_controller import BaseStimulusController

logger = logging.getLogger(__name__)


class SimpleStimulusController(BaseStimulusController):
    """Simple stimulus controller — no runtime configuration required."""

    def submit_stim_params(self, stim_params: Dict[str, Any], image_ndx: int) -> None:
        """
        Store stim_params and track on/off frame lists.

        Args:
            stim_params: Dictionary containing stimulus parameters
            image_ndx: Current image index
        """
        if not stim_params:
            return

        logger.debug(f"SimpleStimulusController: Received stim params at frame {image_ndx}: {stim_params}")

        self.last_stim_params = stim_params

        if stim_params.get("stim_on") is not None and stim_params.get("stim_off") is not None:
            self.stim_param_list.append(stim_params)
            self.stim_on_list.append(stim_params["stim_on"])
            self.stim_off_list.append(stim_params["stim_off"])

    def _activate_hardware(self, stim_params: Dict[str, Any]) -> None:
        """
        Activate stimulus hardware by passing stim_params to the hardware backend.

        Args:
            stim_params: The current stim_params dict
        """
        self.hardware_manager.stimulus.activate_stimulus(stim_params)
        logger.debug(f"SimpleStimulusController: Activated stimulus with params {stim_params}")

    def _deactivate_hardware(self) -> None:
        """Deactivate stimulus hardware."""
        self.hardware_manager.stimulus.deactivate_stimulus()
        logger.debug("SimpleStimulusController: Deactivated stimulus")
