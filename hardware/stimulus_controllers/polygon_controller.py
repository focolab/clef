"""
Polygon stimulus controller for LDI (Laser Direct Illumination).

Handles complex polygon-based stimulus patterns including circles,
rectangles, and dynamic tracking (Hammer of Dawn).
"""

import logging
import time
import numpy as np
from typing import Dict, Any
from utils import numba_utils
from hardware.stimulus_controllers.base_controller import BaseStimulusController

logger = logging.getLogger(__name__)


class PolygonStimulusController(BaseStimulusController):
    """Controller for polygon-based LDI stimulus."""
    
    def __init__(self, hardware_manager):
        """
        Initialize polygon stimulus controller.
        
        Args:
            hardware_manager: HardwareManager instance
        """
        super().__init__(hardware_manager)
        
        # Polygon-specific attributes will be set by hardware backend
        self.calibration_points = {}

        # We need camera ROI for calibration
        self.roi = self.hardware_manager.camera.get_roi()
    
    def spool(self) -> None:
        """
        Initialize jitted functions with dummy routines.
        
        This pre-compiles numba JIT functions for better performance
        during actual stimulus generation.
        """
        tstart = time.time()
        logger.info("Spooling polygon stimulus controller functions...")
        
        # Import utilities (these should be available)
        try:
            
            # Get polygon dimensions from hardware
            polygon_dims = self.hardware_manager.stimulus.get_polygon_dimensions()
            if polygon_dims is not None:
                DSI_IMGWIDTH, DSI_IMGHEIGHT = polygon_dims
            else:
                # polygon unsuccessfully initialized. For testing, pass along default values
                DSI_IMGHEIGHT = 1140
                DSI_IMGWIDTH = 912
            logger.debug(f'Retrieved Polygon dimensions: {polygon_dims}')
            
            # Get calibration points from hardware
            calib = self.hardware_manager.stimulus.get_calibration_points() # todo is this the best way to find calibration points from config?
            logger.debug(f'Retrieved Polygon calibration: {calib}')
            pcx = np.array(calib['pcx'])
            pcy = np.array(calib['pcy'])
            icx = np.array(calib['icx'])
            icy = np.array(calib['icy'])
            logger.info(f'Spooling Polygon')
            # Spool multi-rectangle mask generation
            ix_arr = np.array([600, 700, 800, 900])
            iy_arr = np.array([100, 200, 300, 400])
            width_arr = np.array([20, 50, 20, 50])
            height_arr = np.array([20, 30, 40, 50])
            logger.info(f'Generating dummy mask for polygon with the following arguments: {ix_arr} {iy_arr} {width_arr} {height_arr} {pcx} {pcy} {icx} {icy} {self.roi[0]} {self.roi[1]} {DSI_IMGWIDTH} {DSI_IMGHEIGHT}')
            trash = numba_utils.generate_pg_multi_rectangle_mask(
                ix_arr, iy_arr, width_arr, height_arr,
                pcx, pcy, icx, icy,
                self.roi[0], self.roi[1],
                DSI_IMGWIDTH, DSI_IMGHEIGHT,
            )
            
            tend = time.time()
            logger.info(f"Spooling polygon functions took {tend - tstart:.3f}s")
        
        except Exception as e:
            logger.warning(f"Could not spool polygon functions: {e}")
    
    def submit_stim_params(self, stim_params: Dict[str, Any], image_ndx: int) -> None:
        """
        Handle polygon-specific stimulus patterns.
        
        Args:
            stim_params: Dictionary containing stimulus parameters
            image_ndx: Current image index
        """
        if not stim_params:
            return
        
        logger.info(f"PolygonController: received stim params: {stim_params}, image_ndx: {image_ndx}")

        self.last_stim_params = stim_params

        # Update polygon mask if event present
        if "event" in stim_params:
            self._update_polygon_mask(stim_params)
        
        # Extract event info
        event = stim_params.get('event', {})
        stim_type = event.get('event_type')
        
        # Check if stimulus is pulsed (has both on and off times)
        if stim_params.get("stim_on") is not None and stim_params.get("stim_off") is not None:
            self.stim_param_list.append(stim_params)
            self.stim_on_list.append(stim_params["stim_on"])
            self.stim_off_list.append(stim_params["stim_off"])
        
        # Handle dynamic events (incomplete timing info)
        else:
            if stim_type == 'hammer-of-dawn':
                self._process_hammer_of_dawn_event(stim_params, image_ndx)
            elif stim_type in ['pulse-rect-roi-list', 'stream-rect-roi-list']:
                self._process_stream_rect_roi_list_event(stim_params, image_ndx)
    
    def _process_hammer_of_dawn_event(self, stim_params: Dict[str, Any], image_ndx: int) -> None:
        """
        Handle Hammer of Dawn dynamic tracking events.
        
        Args:
            stim_params: Stimulus parameters
            image_ndx: Current image index
        """
        event = stim_params["event"]
        cx = int(event["x"])
        cy = int(event["y"])
        t = image_ndx + 1  # Next frame is first with set index
        
        # Received on but no off
        if stim_params.get("stim_on") is not None and stim_params.get('stim_off') is None:
            # Cast event xy to list for dynamic tracking
            event["x"] = [cx]
            event["y"] = [cy]
            event["dynamic_event_frame_ndx"] = [t]
            
            self.stim_on_list.append(stim_params["stim_on"])
            self.stim_param_list.append(stim_params)
        
        # Received off but no on
        elif stim_params.get('stim_on') is None and stim_params.get('stim_off') is not None:
            self.stim_off_list.append(stim_params["stim_off"])
            
            # Append to previous stim param
            if self.stim_param_list:
                self.stim_param_list[-1]["event"]["x"].append(cx)
                self.stim_param_list[-1]["event"]["y"].append(cy)
                self.stim_param_list[-1]["event"]["dynamic_event_frame_ndx"].append(t)
        
        # Neither on nor off - just updated xy
        else:
            if self.stim_param_list:
                self.stim_param_list[-1]["event"]["x"].append(cx)
                self.stim_param_list[-1]["event"]["y"].append(cy)
                self.stim_param_list[-1]["event"]["dynamic_event_frame_ndx"].append(t)
    
    def _process_stream_rect_roi_list_event(self, stim_params: Dict[str, Any], image_ndx: int) -> None:
        """
        Handle streaming multi-rectangle ROI events.
        
        Args:
            stim_params: Stimulus parameters
            image_ndx: Current image index
        """
        # Received on but no off
        if stim_params.get('stim_on') is not None and stim_params.get('stim_off') is None:
            self.stim_param_list.append(stim_params)
            self.stim_on_list.append(stim_params["stim_on"])
        
        # Received off but no on
        elif stim_params.get('stim_on') is None and stim_params.get('stim_off') is not None:
            if self.stim_param_list:
                self.stim_param_list[-1]["stim_off"] = stim_params["stim_off"]
            self.stim_off_list.append(stim_params["stim_off"])
    
    def _update_polygon_mask(self, stim_params: Dict[str, Any]) -> None:
        """
        Update polygon mask based on event type.
        
        Delegates mask generation and upload to hardware backend.
        
        Args:
            stim_params: Stimulus parameters containing event info
        """
        event = stim_params['event']
        event_type = event.get('event_type')
        
        logger.info(f"PolygonController: Updating mask for {event_type} event")
        
        # Delegate mask update to hardware backend
        self.hardware_manager.stimulus.update_polygon_mask(stim_params)
    
    def _activate_hardware(self, stim_params: Dict[str, Any]) -> None:
        """
        Activate polygon stimulus hardware.

        Args:
            stim_params: The current stim_params dict
        """
        self.hardware_manager.stimulus.activate_stimulus(stim_params)
        logger.debug(f"PolygonController: Activated stimulus with params {stim_params}")
    
    def _deactivate_hardware(self) -> None:
        """Deactivate polygon stimulus hardware."""
        self.hardware_manager.stimulus.deactivate_stimulus()
        logger.debug("PolygonController: Deactivated stimulus")
    
    def get_metadata(self, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get polygon stimulus metadata.
        
        Args:
            args: Optional arguments
        
        Returns:
            Dictionary containing stimulus metadata including calibration points
        """
        metadata = super().get_metadata(args)
        
        # Add polygon-specific metadata
        try:
            calib = self.hardware_manager.stimulus.get_calibration_points()

            # replace numpy array vals with lists
            calib['pcx'] = calib['pcx'].tolist()
            calib['pcy'] = calib['pcy'].tolist()
            calib['icx'] = calib['icx'].tolist()
            calib['icy'] = calib['icy'].tolist()

            metadata["calibration_points"] = calib
        except Exception as e:
            logger.warning(f"Could not get calibration points for metadata: {e}")
        
        return metadata
