"""
StimBaseClass - DEPRECATED legacy interface for backward compatibility.

This class is maintained for backward compatibility with existing code.
New code should use stimulus controllers in lib/stimulus_controllers/ instead.

The refactored architecture separates concerns:
- Hardware control -> HardwareManager + backends
- Timing/coordination -> Stimulus controllers
- Legacy interface -> This class (delegates to new system)
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class StimBaseClass(ABC):
    """
    DEPRECATED: Legacy baseclass for interfacing with stimulus apparatus.
    
    This class is maintained for backward compatibility. New implementations
    should use the stimulus controller architecture (lib/stimulus_controllers/).
    
    The new architecture:
    - HardwareManager handles low-level hardware control
    - Stimulus controllers handle timing and coordination
    - This class provides a compatibility shim
    """

    def __init__(self, args, **kwargs):
        """ init stim base class """

        # Configuration
        self.args = args
        self.local_handles = kwargs.get('local_handles', {})
        self.rec_id = self.args.get("id")
        self.roi = self.args.get("roi")
        self.savedir = self.args.get("saveroot")

        # Grab params derived from input gooey
        gooey_args = self.args.get("gooey_args", {})
        self.zsize = gooey_args.get("zsize")
        self.stim_interface = gooey_args.get("stim_interface")
        self.acquisition_backend = gooey_args.get("acquisition_backend")
        self.trigger_alg = gooey_args.get("trigger_algorithm")
        self.microscope_name = gooey_args.get("microscope_name")

        # NEW: Use HardwareManager instead of direct MMC access
        self.hardware_manager = kwargs.get('hardware_manager')
        if not self.hardware_manager:
            # For backward compatibility, try to get from local_handles
            self.hardware_manager = self.local_handles.get('hardware_manager')
        
        # holder variables shared between stimulus interfaces
        self.stim_on_list = []
        self.stim_off_list = []
        self.stim_on_time_list = []
        self.stim_off_time_list = []
        self.stim_intensity_list = []
        self.stim_param_list = []

        # timers
        self.submit_stim_params_time_list = []
        self.process_stim_params_event_time_list = []

    @abstractmethod
    def submit_stim_params(self, stim_params: Dict[str, Any], image_ndx: int) -> None:
        """
        Process stimulus parameters submitted by algorithm.
        
        Args:
            stim_params: Dictionary containing stimulus parameters
            image_ndx: Current image index
        """
        pass

    @abstractmethod
    def activate_stim(self, intensity: float = 10) -> None:
        """
        Activate stimulus with given intensity.
        
        Args:
            intensity: Stimulus intensity value
        """
        pass

    @abstractmethod
    def inactivate_stim(self) -> None:
        """Deactivate stimulus."""
        pass

    def spool(self) -> None:
        """
        Runtime initialization (called spooling).
        Override in subclasses if needed.
        """
        pass

    def check_stim(self, img_count: int) -> None:
        """
        Check if current frame should trigger stimulus activation/deactivation.
        
        Args:
            img_count: Current image count
        """
        # Check to activate stimulation
        to_activate_stimulation = img_count in self.stim_on_list
        if to_activate_stimulation:
            logger.info(
                f"StimBaseClass::check_stim> activating {self.stim_interface} stim on frame {img_count}"
            )

            # Get stim intensity
            idx = self.stim_on_list.index(img_count)
            intensity = self.stim_intensity_list[idx] if idx < len(self.stim_intensity_list) else 10
            
            self.activate_stim(intensity)
            self.stim_on_time_list.append(time.time())

        # Check to deactivate stimulation
        to_inactivate_stimulation = img_count in self.stim_off_list
        if to_inactivate_stimulation:
            logger.info(
                f"StimBaseClass::check_stim> deactivating {self.stim_interface} stim on frame {img_count}"
            )
            
            self.inactivate_stim()
            self.stim_off_time_list.append(time.time())

    def get_mmc(self):
        """
        Return handle to internal micromanager object.
        
        DEPRECATED: Direct MMC access should be avoided. Use HardwareManager instead.
        This method is maintained for backward compatibility only.
        
        Returns:
            Micro-Manager Core object or None
        """
        # Try to get from local_handles (old way)
        mmc = self.local_handles.get('mmc', None)
        
        if not mmc and self.hardware_manager:
            # Try to get from HardwareManager (new way)
            mmc = self.hardware_manager.get_mmc()
        
        if not mmc:
            logger.warning(
                "No MMC object available. Direct MMC access is deprecated. "
                "Use HardwareManager stimulus interface instead."
            )
        
        return mmc

    def get_metadata(self, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Return metadata for this stimulus interface.
        
        Args:
            args: Optional arguments (e.g., t0 for relative timing)
        
        Returns:
            Dictionary containing stimulus metadata
        """
        args = args or {}
        
        # Get stim onset relative to recording start
        t0 = self.args.get("t0") or args.get("t0")
        stim_onset_times_simple = self.get_stim_time_onsets(t0=t0)

        # Add metadata attributes common to all stim interfaces
        metadata = {
            "stim_onset_times_simple": stim_onset_times_simple,
            "stim_on_list": self.stim_on_list,
            "stim_off_list": self.stim_off_list,
            "stim_on_time_list": self.stim_on_time_list,
            "stim_off_time_list": self.stim_off_time_list,
            "stim_param_list": self.stim_param_list,
        }

        return metadata

    def close(self) -> None:
        """Close stimulus interface (cleanup)."""
        pass

    def get_stim_time_onsets(self, t0: Optional[float] = None) -> list:
        """
        Convenience function to return stimulus onset timing in seconds.
        
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

    @staticmethod
    def initialize_stim_interface(args: Dict[str, Any], local_handles: Optional[Dict] = None):
        """
        Factory function to create relevant stimulus interface object.
        
        DEPRECATED: Use lib.stimulus_controllers.create_stimulus_controller instead.
        This method is maintained for backward compatibility.
        
        Args:
            args: Configuration dictionary
            local_handles: Optional local handles dictionary
        
        Returns:
            StimBaseClass subclass instance
        """
        local_handles = local_handles or {}
        
        # Get microscope and software backend
        gooey_args = args.get("gooey_args", {})
        stim_interface = gooey_args.get("stim_interface")
        acquisition_backend = gooey_args.get("acquisition_backend")

        logger.info(
            f"StimBaseClass.initialize_stim_interface: {stim_interface} on {acquisition_backend}"
        )

        # Return appropriate stim interface
        if stim_interface == "InvCore-LDI-Polygon-640" and acquisition_backend == 'pycromanager':
            from lib import InvCoreLDIPolygon
            stim = InvCoreLDIPolygon.InvCoreLDIPolygon(args, local_handles=local_handles)
        
        elif stim_interface == "InvCore-ThunderscopeLED3" and acquisition_backend == 'pycromanager':
            from lib import InvCoreThunderscopeLED3
            stim = InvCoreThunderscopeLED3.InvCoreThunderscopeLED3(args, local_handles=local_handles)
        
        elif stim_interface in ["no stim", "test", "dummy"]:
            from lib import DummyStim
            stim = DummyStim.DummyStim(args, local_handles=local_handles)
        
        elif stim_interface == "InvCore-SpinningDisk-639":
            from lib import InvCoreSpinningDisk639
            stim = InvCoreSpinningDisk639.InvCoreSpinningDisk639(args, local_handles=local_handles)
        
        else:
            raise ValueError(
                f"Unknown stimulus interface: {stim_interface}. "
                f"Consider using new stimulus controller architecture."
            )

        return stim