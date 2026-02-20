"""
Base stimulus controller for high-level stimulus coordination and timing.

Responsibilities:
- Track stimulus timing (on/off frames)
- Coordinate with algorithm triggers
- Manage stimulus metadata
- Delegate hardware control to HardwareManager
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseStimulusController(ABC):
    """
    High-level stimulus coordination and timing.
    
    Separates timing/coordination logic from hardware control.
    Hardware operations are delegated to HardwareManager.
    """
    
    def __init__(self, hardware_manager):
        """
        Initialize stimulus controller.
        
        Args:
            hardware_manager: HardwareManager instance for hardware operations
        """
        self.hardware_manager = hardware_manager

        # Timing tracking
        self.stim_on_list = []
        self.stim_off_list = []
        self.stim_on_time_list = []
        self.stim_off_time_list = []
        self.stim_param_list = []
        self.last_stim_params = None
    
    @abstractmethod
    def submit_stim_params(self, stim_params: Dict[str, Any], image_ndx: int) -> None:
        """
        Process stimulus parameters from algorithm.
        
        Args:
            stim_params: Dictionary containing stimulus parameters from algorithm
            image_ndx: Current image index
        """
        pass
    
    def check_stim(self, img_count: int) -> None:
        """
        Check if stimulus should be activated/deactivated at current frame.

        Args:
            img_count: Current image count
        """
        # Activation check
        if (
            self.last_stim_params is not None
            and img_count == self.last_stim_params.get("stim_on")
        ):
            logger.info(f"BaseStimulusController: activating stim on frame {img_count}")
            self._activate_hardware(self.last_stim_params)
            self.stim_on_time_list.append(time.time())

        # Deactivation check
        if img_count in self.stim_off_list:
            logger.info(
                f"BaseStimulusController: deactivating stim on frame {img_count}"
            )
            
            # Delegate to hardware
            self._deactivate_hardware()
            self.stim_off_time_list.append(time.time())
    
    @abstractmethod
    def _activate_hardware(self, stim_params: Dict[str, Any]) -> None:
        """
        Activate stimulus hardware with the current stim params.

        Args:
            stim_params: The stim_params dict from the most recent submit_stim_params call
        """
        pass
    
    @abstractmethod
    def _deactivate_hardware(self) -> None:
        """Deactivate stimulus hardware."""
        pass
    
    def spool(self) -> None:
        """
        Runtime initialization (called spooling).
        Override in subclasses if needed.
        """
        pass
    
    def get_metadata(self, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get stimulus metadata.

        Args:
            args: Optional arguments (e.g., t0 for relative timing)

        Returns:
            Dictionary containing stimulus metadata
        """
        args = args or {}
        
        # Get stim onset relative to recording start
        # t0 = self.config.get("t0") or args.get("t0")
        # stim_onset_times_simple = self.get_stim_time_onsets(t0=t0)
        
        t0 = args.get("t0")
        stim_onset_times_simple = self.get_stim_time_onsets(t0=t0)

        metadata = {
            "stim_onset_times_simple": stim_onset_times_simple,
            "stim_on_list": self.stim_on_list,
            "stim_off_list": self.stim_off_list,
            "stim_on_time_list": self.stim_on_time_list,
            "stim_off_time_list": self.stim_off_time_list,
            "stim_param_list": self.stim_param_list,
        }
        
        return metadata
    
    def get_stim_time_onsets(self, t0: Optional[float] = None) -> list:
        """
        Get stimulus onset timing in seconds.
        
        Args:
            t0: Optional starting time for relative timing
        
        Returns:
            List of stimulus onset times
        """
        onsets = self.stim_on_time_list.copy()
        
        # If we provide a starting time, return stim onsets relative to that
        if t0:
            onsets = [on - t0 for on in onsets]
        
        return onsets
    
    def close(self) -> None:
        """Close stimulus interface (cleanup)."""
        pass
