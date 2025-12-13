"""
Widefield stimulus controller (e.g., 639nm laser).

Handles widefield-specific stimulus patterns including streaming and pulsed modes.
"""

import logging
from typing import Dict, Any

from hardware.stimulus_controllers.base_controller import BaseStimulusController

logger = logging.getLogger(__name__)


class WidefieldStimulusController(BaseStimulusController):
    """Controller for widefield stimulus (e.g., 639nm laser)."""
    
    def submit_stim_params(self, stim_params: Dict[str, Any], image_ndx: int) -> None:
        """
        Handle widefield-specific stimulus patterns.
        
        Args:
            stim_params: Dictionary containing stimulus parameters
            image_ndx: Current image index
        """
        if not stim_params:
            return
        
        logger.info(f"WidefieldController: received stim params: {stim_params}, image_ndx: {image_ndx}")
        
        # Extract event type
        event = stim_params.get('event', {})
        event_type = event.get('event_type')
        
        # Extract intensity if present
        if "stim_intensity" in event:
            self.stim_intensity_list.append(event["stim_intensity"])
        
        # Handle stream-widefield events specially
        if event_type == 'stream-widefield':
            self._process_stream_widefield(stim_params, image_ndx)
        else:
            # Standard pulsed stimulus
            if stim_params.get("stim_on") is not None and stim_params.get("stim_off") is not None:
                self.stim_param_list.append(stim_params)
                self.stim_on_list.append(stim_params["stim_on"])
                self.stim_off_list.append(stim_params["stim_off"])
    
    def _process_stream_widefield(self, stim_params: Dict[str, Any], image_ndx: int) -> None:
        """
        Handle streaming widefield stimulus.
        
        For streaming stimulus, we receive on/off separately and need to
        update the previous event object.
        
        Args:
            stim_params: Dictionary containing stimulus parameters
            image_ndx: Current image index
        """
        # Received on but no off
        if stim_params.get('stim_on') is not None and stim_params.get('stim_off') is None:
            self.stim_on_list.append(stim_params["stim_on"])
            self.stim_param_list.append(stim_params)
            logger.debug(f"WidefieldController: Stream stim ON at frame {stim_params['stim_on']}")
        
        # Received off but no on (update previous event)
        elif stim_params.get('stim_off') is not None:
            self.stim_off_list.append(stim_params["stim_off"])
            if self.stim_param_list:
                self.stim_param_list[-1]["stim_off"] = stim_params["stim_off"]
            logger.debug(f"WidefieldController: Stream stim OFF at frame {stim_params['stim_off']}")
    
    def _activate_hardware(self, intensity: float) -> None:
        """
        Activate widefield stimulus hardware.
        
        Args:
            intensity: Stimulus intensity value (typically 0-100%)
        """
        params = {"intensity": intensity}
        self.hardware_manager.stimulus.activate_stimulus(params)
        logger.debug(f"WidefieldController: Activated stimulus at intensity {intensity}")
    
    def _deactivate_hardware(self) -> None:
        """Deactivate widefield stimulus hardware."""
        self.hardware_manager.stimulus.deactivate_stimulus()
        logger.debug("WidefieldController: Deactivated stimulus")
