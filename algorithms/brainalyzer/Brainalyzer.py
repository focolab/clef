"""
Brainalyzer Algorithm

Interactive GUI-based algorithm for closed-loop microscopy with real-time
visualization and manual/automated ROI-based stimulus triggering.
"""

import time
import logging
import random
import numpy as np
from multiprocessing import Pipe, shared_memory

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
    """
    
    def __init__(self, args, local_handles=None):
        """
        Initialize Brainalyzer algorithm.
        
        Args:
            args: Legacy args dictionary containing configuration
            local_handles: Dictionary with optional handles (e.g., {'mmc': mmc_instance})
        """
        if local_handles is None:
            local_handles = {}
        
        # Get general params
        self.args = args
        self.rec_id = self.args["id"]
        self.roi = self.args["roi"]
        self.frames_to_grab = self.args["gooey_args"]["total_frames"]
        self.microscope_name = self.args["gooey_args"]["microscope_name"]
        self.saveroot = self.args.get("saveroot", "")
        self.zsize = self.args["gooey_args"]["zsize"]
        self.xsize = self.args["roi"][2]
        self.ysize = self.args["roi"][3]
        self.camera_binning = None
        self.dtype = self.args.get("dtype", np.uint16)
        self.local_handles = local_handles

        # Data transmission with subprocess IPC
        self.ipc = self.args.get("data_ipc", "shared_memory")

        # Set behavior mode vs neural imaging mode
        self.GUI_mode = self.args["gooey_args"].get("GUI_mode", "neural_imaging")

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

        # Initialize the worker process
        self.initialize_worker()

    def initialize_worker(self):
        """Initialize the GUI worker process."""
        # Algorithm-specific params for subprocess
        self.stim_intensity_ops = self.args["gooey_args"]["stim_intensity_options"]
        self.stim_intensity = self.stim_intensity_ops[0]

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

        except Exception as err:
            logger.error(f"Error while initializing BrainalyzerWorker: {err}")
            raise

    def initialize_behavior_mode(self):
        """Initialize behavior mode specific hardware (e.g., stage control)."""
        if self.microscope_name == "innovation core thunderscope":
            # Shared memory for xy stage control
            self.shared_stage_offset_xy = shared_memory.ShareableList(
                [0, 0], name="shared_stage_offset_xy"
            )

            # Data structure for xy stage position tracking
            self.xy_stage_position_list = []

        # MicroManager handle for stage control
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
        offsetx, offsety = self.shared_stage_offset_xy

        if offsetx or offsety:
            # Reset offsets
            self.shared_stage_offset_xy[0] = 0
            self.shared_stage_offset_xy[1] = 0
            return int(offsetx), int(offsety)
        return None

    def sync_stage(self):
        """Synchronize microscope stage position with GUI."""
        if (self.GUI_mode == "behavior" and 
            self.microscope_name == "innovation core thunderscope" and
            self.args["id"] != "test"):
            
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
        fname_root = self.args["id"]
        random.seed(fname_root)

    def get_metadata(self, args=None):
        """Return metadata captured during runtime."""
        metadata = {
            "stim_param_list": self.stim_param_list,
        }

        # Optional metadata
        if (self.GUI_mode == "behavior" and
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

        # Adjust stage if necessary
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
        self.shared_image_count.shm.close()
        self.shared_image_count.shm.unlink()

        # Close the worker process
        self.parent_conn.send("close")

        # Close stage-related shared memory
        if self.GUI_mode == "behavior":
            self.shared_stage_offset_xy.shm.close()
            self.shared_stage_offset_xy.shm.unlink()

    # Internal methods
    def get_event(self):
        """Check worker to see if there's any events to process."""
        if self.parent_conn.poll():
            data = self.parent_conn.recv()
            return data
        return None

    def store_frame_in_shm(self, img, zndx):
        """Store frame in shared memory buffer."""
        self.shared_ndarray_list[zndx][:] = img[:]
