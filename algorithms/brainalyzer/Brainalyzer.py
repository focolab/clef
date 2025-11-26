"""
Brainalyzer Algorithm

Interactive GUI-based algorithm for closed-loop microscopy with real-time
visualization and manual/automated ROI-based stimulus triggering.

UPDATED: Now accepts Config objects instead of legacy args dict.
"""

import time
import logging
import random
import numpy as np
from multiprocessing import Pipe, shared_memory
from typing import Optional, Dict, Any

# Import config models
from config.config_manager import (
    AlgorithmConfig,
    ExperimentConfig,
    HardwareConfig,
)

# Import worker process
try:
    from algorithms.brainalyzer import BrainalyzerWorker
except ImportError:
    try:
        from . import BrainalyzerWorker
    except ImportError:
        import BrainalyzerWorker  # for local testing

# Ignore numpy warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class Brainalyzer:
    """
    Interactive algorithm with GUI for closed-loop microscopy.
    
    Features:
    - Real-time visualization of acquired frames
    - Manual ROI placement for quantification and stimulation
    - Interactive stimulus triggering
    - Automated closed-loop models
    - Support for both neural imaging and behavior modes
    
    UPDATED: Now uses Config objects for initialization.
    """
    
    def __init__(
        self,
        algorithm_config: AlgorithmConfig,
        experiment_config: ExperimentConfig,
        hardware_config: Optional[HardwareConfig] = None,
        local_handles: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Brainalyzer algorithm with Config objects.
        
        Args:
            algorithm_config: Algorithm configuration (GUI mode, params, stimulus)
            experiment_config: Experiment configuration (acquisition, subject, output)
            hardware_config: Hardware configuration (optional, for behavior mode)
            local_handles: Dictionary with optional handles (e.g., {'mmc': mmc_instance})
        """
        if local_handles is None:
            local_handles = {}
        
        # Store configs
        self.algorithm_config = algorithm_config
        self.experiment_config = experiment_config
        self.hardware_config = hardware_config
        self.local_handles = local_handles
        
        # Extract core experiment params
        self.rec_id = experiment_config.experiment_name
        self.saveroot = experiment_config.output_dir
        self.frames_to_grab = experiment_config.acquisition.num_frames
        self.zsize = experiment_config.acquisition.z_planes
        
        # Extract algorithm params
        self.GUI_mode = algorithm_config.gui_mode
        self.stim_intensity_ops = algorithm_config.stimulus_params.intensity_percent_options
        self.stim_intensity = self.stim_intensity_ops[0]
        
        # Hardware params (with safe defaults)
        self.microscope_name = (
            hardware_config.microscope_name if hardware_config else "unknown"
        )
        
        # Data params - will be set by closed_loop_engine after hardware init
        self.roi = (0, 0, 200, 200)  # Default, will be updated
        self.xsize = 200
        self.ysize = 200
        self.camera_binning = None
        self.dtype = np.uint16
        
        # Data transmission with subprocess IPC
        self.ipc = "shared_memory"  # Only supported mode

        # If we're in behavior mode, initialize behavior-related hardware
        if self.GUI_mode == "behavior":
            self.initialize_behavior_mode()

        # State holders
        self.events = []
        self.current_event = None
        self.stimulus_is_on = False
        self.stim_param_list = []
        self.shared_frame_memory_list = []
        self.shared_ndarray_list = []

        # Worker will be initialized after hardware setup
        self.proc = None
        self.parent_conn = None
        self.child_conn = None
        self.shared_image_count = None

    def set_roi(self, roi: tuple):
        """
        Set ROI dimensions after hardware initialization.
        
        This must be called by the acquisition engine after hardware setup
        and before initializing the worker process.
        
        Args:
            roi: Tuple of (x_offset, y_offset, width, height)
        """
        self.roi = roi
        self.xsize = roi[2]
        self.ysize = roi[3]
        logger.info(f"Brainalyzer ROI set to: {roi}")

    def initialize_worker(self):
        """
        Initialize the GUI worker process.
        
        Must be called after set_roi() to ensure dimensions are correct.
        """
        # Build vis_args for worker subprocess
        self.vis_args = {
            "id": self.rec_id,
            "saveroot": self.saveroot,
            "ysize": self.ysize,
            "xsize": self.xsize,
            "total_frames": self.frames_to_grab,
            "zsize": self.zsize,
            "stim_intensity": self.stim_intensity,
            "data_ipc": self.ipc,
            "GUI_mode": self.GUI_mode,
            "camera_binning": self.camera_binning,
            "dtype": self.dtype,
        }

        # Initialize the visualizer
        try:
            # Create a pipe to visualizer
            self.parent_conn, self.child_conn = Pipe()

            if self.ipc == "shared_memory":
                # Make a shared memory buffer and associated ndarray for each z plane
                for z in range(self.zsize):
                    try:
                        shared_frame_memory = shared_memory.SharedMemory(
                            create=True,
                            size=self.ysize * self.xsize * 2,
                            name=f"shared_frame_memory_{z}",
                        )
                    except FileExistsError:
                        # Bad cleanup means file might already exist
                        shared_frame_memory = shared_memory.SharedMemory(
                            name=f"shared_frame_memory_{z}",
                            create=False,
                            size=self.ysize * self.xsize * 2
                        )

                    shared_ndarray = np.ndarray(
                        shape=(self.ysize, self.xsize),
                        buffer=shared_frame_memory.buf,
                        dtype=self.dtype,
                    )

                    self.shared_frame_memory_list.append(shared_frame_memory)
                    self.shared_ndarray_list.append(shared_ndarray)

                # Also make frame counter
                try:
                    self.shared_image_count = shared_memory.ShareableList(
                        [0], name="shared_image_count"
                    )
                except FileExistsError:
                    self.shared_image_count = shared_memory.ShareableList(
                        None, name="shared_image_count"
                    )
            else:
                raise Exception("ERROR: ipc must be shared_memory")

            # Create process and start it
            self.proc = BrainalyzerWorker.BrainalyzerWorker(
                self.child_conn, self.vis_args
            )
            self.proc.start()
            
            logger.info("BrainalyzerWorker subprocess started successfully")

        except Exception as err:
            logger.error(f"Error while initializing BrainalyzerWorker: {err}")
            raise
        
    def initialize_behavior_mode(self):
        """Initialize behavior mode specific hardware (e.g., stage control)."""

        # TODO: Fix this...
        if self.microscope_name == "innovation core thunderscope":
            # Shared memory for xy stage control
            try:
                self.shared_stage_offset_xy = shared_memory.ShareableList(
                    [0, 0], name="shared_stage_offset_xy"
                )
            except FileExistsError:
                self.shared_stage_offset_xy = shared_memory.ShareableList(
                    None, name="shared_stage_offset_xy"
                )

            # Data structure for xy stage position tracking
            self.xy_stage_position_list = []

        # MicroManager handle for stage control
        # TODO: Fix this...
        self.mmc = self.local_handles.get("mmc", None)
        if self.mmc is None:
            logger.warning(
                "Error while initializing GUI_mode: behavior, "
                "no local handle to micromanager found"
            )
        else:
            cam = self.mmc.getCameraDevice()
            self.camera_binning = self.mmc.getProperty(cam, "Binning")


    def get_xy_offset(self):
        """Get stage offset from shared memory."""
        if not hasattr(self, 'shared_stage_offset_xy'):
            return None
            
        offsetx, offsety = self.shared_stage_offset_xy

        if offsetx or offsety:
            # Reset offsets
            self.shared_stage_offset_xy[0] = 0
            self.shared_stage_offset_xy[1] = 0
            return int(offsetx), int(offsety)
        return None

    # TODO migrate this to stage controller
    def sync_stage(self):
        """Synchronize microscope stage position with GUI."""
        if (self.GUI_mode == "behavior" and 
            self.microscope_name == "innovation core thunderscope" and
            self.rec_id != "test"):
            
            self.xy_stage_position_list.append([
                self.mmc.getXPosition(),
                self.mmc.getYPosition()
            ])

            # Get xy offset
            xy_offset = self.get_xy_offset()
            if xy_offset is not None:
                # Move stage
                self.mmc.setRelativeXYPosition(xy_offset[0], xy_offset[1])


    def initialize_model(self):
        """Initialize the algorithm model."""
        # Seed RNG for reproducibility
        random.seed(self.rec_id)

    def get_metadata(self, args=None):
        """
        Return metadata captured during runtime.
        
        Args:
            args: Legacy parameter, ignored (kept for compatibility)
            
        Returns:
            Dictionary containing runtime metadata
        """
        metadata = {
            "stim_param_list": self.stim_param_list,
            "algorithm_config": self.algorithm_config.model_dump(mode='json'),
            "experiment_config": self.experiment_config.model_dump(mode='json'),
        }
        
        if self.hardware_config:
            metadata["hardware_config"] = self.hardware_config.model_dump(mode='json')

        # Optional metadata
        if (self.GUI_mode == "behavior" and
            
            # TODO remove scope ref, this should get pulled when migrating to stage
            self.microscope_name == "innovation core thunderscope"):
            metadata["xy_stage_position_list"] = self.xy_stage_position_list

        return metadata

    def process_frame(self, img, zndx):
        """Process each acquired frame."""
        # Store the frame in shared memory
        self.store_frame_in_shm(img, zndx)

        # Update image count (frames not volumes)
        self.shared_image_count[0] += 1

        # Check for events from GUI
        data = self.get_event()
        if data is not None:
            self.events.append(data)
            logger.info(f"Brainalyzer::process_frame> storing event: {data}")
            self.current_event = data

        # Adjust stage if necessary -- TODO this should be in hardware
        if self.GUI_mode == "behavior":
            self.sync_stage()

    def process_volume(self):
        """Process completed volume (called after full z-scan)."""
        pass

    def check_stim(self, image_ndx, cooldown_counter=0):
        """
        Check for stimulus events and format for stim_interface.
        
        Returns:
            tuple: (stim_params dict, new_cooldown_counter)
        """
        stim_params = {}
        new_cooldown = 0

        # Grab current event if present
        if not self.stimulus_is_on and self.current_event is not None:
            stim_intensity = self.current_event["stim_intensity"]

            # Trigger stim_on on next volume
            zndx = image_ndx % self.zsize
            stim_on = image_ndx + self.zsize - zndx

            stim_event = self.current_event
            event_type = stim_event["event_type"]

            # Build stim parameters
            stim_params["stim_on"] = stim_on
            stim_params["stim_intensity"] = stim_intensity
            stim_params["event"] = stim_event

            # Handle different types of stim event signals
            if event_type == "pulse-rect-roi-list" or event_type == "full-field-button":
                # Set number of stim frames
                num_stim_frames = self.current_event["stim_duration_vols"] * self.zsize
                stim_params["stim_off"] = stim_on + num_stim_frames

                # Check if recording is about to end
                if stim_params["stim_off"] > self.frames_to_grab:
                    stim_params = {}

            elif event_type == "stream-rect-roi-list" or event_type == "stream-widefield":
                # Continuous stimulation mode
                self.stimulus_is_on = True
                self.current_event = None
            else:
                logger.critical(f"check_stim> event type {event_type} not recognized!")

            # Reset current event
            self.current_event = None
            new_cooldown = 0
            self.stim_param_list.append(stim_params)

        # If we are still stimulating but no new event
        elif self.stimulus_is_on and self.current_event is None:
            pass  # Continue stimulating

        # If we receive an event while stimulation is ongoing (stop signal)
        elif self.stimulus_is_on and self.current_event is not None:
            zndx = image_ndx % self.zsize
            stim_off = image_ndx + self.zsize - zndx

            stim_params["stim_off"] = stim_off
            stim_params["event"] = self.current_event

            self.stimulus_is_on = False
            self.current_event = None

            # Update most recent stim param
            self.stim_param_list[-1]["stim_off"] = stim_off

        if stim_params:
            logger.info(f"Brainalyzer::check_stim> emitting stim params: {stim_params}")

        return stim_params, new_cooldown
    
    def plot_model(self, show_plot=False, savefilename=None):
        """Plot current state of model."""
        logger.warning("There is no alg model to plot!")

    def close(self):
        """Close the worker process and clean up resources."""
        # Close the shared memory buffers
        for shm in self.shared_frame_memory_list:
            shm.close()
            shm.unlink()
            
        if self.shared_image_count:
            self.shared_image_count.shm.close()
            self.shared_image_count.shm.unlink()

        # Close the worker process
        if self.parent_conn:
            self.parent_conn.send("close")

        # Close stage-related shared memory
        if self.GUI_mode == "behavior" and hasattr(self, 'shared_stage_offset_xy'):
            self.shared_stage_offset_xy.shm.close()
            self.shared_stage_offset_xy.shm.unlink()

    # Internal methods
    def get_event(self):
        """Check worker to see if there's any events to process."""
        if self.parent_conn and self.parent_conn.poll():
            data = self.parent_conn.recv()
            return data
        return None

    def store_frame_in_shm(self, img, zndx):
        """Store frame in shared memory buffer."""
        self.shared_ndarray_list[zndx][:] = img[:]

def create_brainalyzer_from_legacy_args(args: Dict[str, Any], local_handles: Optional[Dict[str, Any]] = None):
    """
    Backward compatibility wrapper: Create Brainalyzer from legacy args dict.
    
    This function allows existing code to continue using the old args format
    while the new code uses Config objects internally.
    
    Args:
        args: Legacy args dictionary with gooey_args structure
        local_handles: Optional dictionary of local handles (mmc, etc.)
        
    Returns:
        Brainalyzer instance
        
    Example:
        >>> # Old way (still works)
        >>> alg = create_brainalyzer_from_legacy_args(args, local_handles)
        >>> 
        >>> # New way (preferred)
        >>> alg = Brainalyzer(algorithm_config, experiment_config, hardware_config)
    """
    from closed_loop_engine import convert_gooey_args_to_configs
    
    # Convert legacy args to configs
    gooey_args = args.get("gooey_args", args)
    configs = convert_gooey_args_to_configs(gooey_args)
    
    # Create Brainalyzer with configs
    alg = Brainalyzer(
        algorithm_config=configs["algorithm"],
        experiment_config=configs["experiment"],
        hardware_config=configs["hardware"],
        local_handles=local_handles
    )
    
    # Extract ROI from args if present (for backward compatibility)
    if "roi" in args:
        alg.set_roi(args["roi"])
    
    return alg