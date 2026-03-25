"""
Brainalyzer Logic for CLEF2.

Orchestrates the BrainalyzerWorker GUI subprocess for interactive
closed-loop microscopy. Manages shared memory for frame transfer,
relays stimulus events from the GUI to output devices (polygon, LDI).

The BrainalyzerWorker generates polygon masks directly and writes
them into the polygon output device's shared memory buffer.
"""

import json
import logging
import numpy as np
from multiprocessing import Pipe, shared_memory
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from clef2.core.logic.BaseClosedLoopLogic import BaseClosedLoopLogic

logger = logging.getLogger(__name__)


class BrainalyzerLogic(BaseClosedLoopLogic):
    """Interactive GUI-based closed-loop logic for microscopy experiments."""

    logic_class: ClassVar[Optional[str]] = "brainalyzer_logic"

    def __init__(
        self,
        name: str,
        config: Dict[str, Any] | None = None,
        output_devices: Dict[str, Any] | None = None,
        gui_parameters: Dict[str, Any] | None = None,
    ):
        super().__init__(name, config, output_devices, gui_parameters)

        cfg = self.config

        # Core experiment params
        self.gui_mode = cfg.get("gui_mode", "neural_imaging")
        self.stim_intensity = cfg.get("stim_intensity", 0)
        self.zsize = cfg.get("zsize", 1)
        self.num_samples = cfg.get("num_samples", 1000)
        self.input_device_name = cfg.get("input_device_name", "camera")

        # Calibration
        self.calibration_file = cfg.get("calibration_file", None)
        self.calibration_points = None

        # Camera state (set during initialize_model via output_devices -> io_manager)
        self.xsize = 0
        self.ysize = 0
        self.camera_roi = (0, 0, 0, 0)

        # Polygon state
        self.polygon_shm_name = None
        self.polygon_width = None
        self.polygon_height = None

        # IPC
        self.parent_conn = None
        self.child_conn = None
        self.shared_frame_memory_list = []
        self.shared_ndarray_list = []
        self.shared_image_count = None
        self.proc = None

        # Stimulus state
        self.events = []
        self.current_event = None
        self.stimulus_is_on = False
        self.stim_param_list = []
        self.sample_count = 0

        # Save dir
        self.saveroot = cfg.get("save_dir", "./output")

        # GUI screenshot freq
        self.gui_screenshot_freq = self.gui_parameters.get("gui_screenshot_freq", 0)

    def initialize_model(self):
        """Load calibration, resolve hardware info, create shared memory, start worker."""

        # Load calibration points
        self._load_calibration()

        # Resolve polygon output device info
        polygon_dev = self.output_devices.get("polygon")
        if polygon_dev is not None:
            dims = polygon_dev.get_dimensions()
            if dims is not None:
                self.polygon_width, self.polygon_height = dims
            self.polygon_shm_name = polygon_dev.shm_name
            logger.info(
                f"Polygon: shm={self.polygon_shm_name}, "
                f"dims=({self.polygon_width}, {self.polygon_height})"
            )

        # Resolve camera input device info via io_manager
        # Output devices hold a reference to io_manager
        io_manager = None
        for dev in self.output_devices.values():
            if hasattr(dev, "io_manager") and dev.io_manager is not None:
                io_manager = dev.io_manager
                break

        if io_manager is not None:
            try:
                camera_dev = io_manager.get_input_device(self.input_device_name)
                self.xsize = camera_dev.width
                self.ysize = camera_dev.height
                self.camera_roi = camera_dev.get_roi()
                logger.info(f"Camera: {self.xsize}x{self.ysize}, ROI={self.camera_roi}")
            except Exception as e:
                logger.warning(f"Could not get camera info from io_manager: {e}")

        # Fall back to config if io_manager didn't provide values
        if self.xsize == 0 or self.ysize == 0:
            self.xsize = self.config.get("xsize", 0)
            self.ysize = self.config.get("ysize", 0)
            self.camera_roi = self.config.get(
                "camera_roi", (0, 0, self.xsize, self.ysize)
            )

        if self.xsize == 0 or self.ysize == 0:
            logger.warning("xsize/ysize not set — worker may fail.")

        # Start worker subprocess
        self._start_worker()

    def _load_calibration(self):
        """Load polygon calibration points from JSON file."""
        if self.calibration_file is None:
            logger.warning("No calibration_file specified")
            return

        calib_path = Path(self.calibration_file)
        if not calib_path.exists():
            logger.error(f"Calibration file not found: {calib_path}")
            return

        with open(calib_path) as f:
            data = json.load(f)

        calibrations = data.get("calibrations", [])
        if not calibrations:
            logger.error("No calibrations found in file")
            return

        # Use latest calibration
        calib = calibrations[-1]
        self.calibration_points = {
            "pcx": calib["pcx"],
            "pcy": calib["pcy"],
            "icx": calib["icx"],
            "icy": calib["icy"],
        }
        logger.info(f"Loaded calibration from {calib.get('datetime', 'unknown')}")

    def _start_worker(self):
        """Create shared memory buffers, Pipe, and start BrainalyzerWorker."""
        from clef2.apps.logic.brainalyzer_worker import BrainalyzerWorker

        self.parent_conn, self.child_conn = Pipe()

        # Create shared memory for each z-plane
        for z in range(self.zsize):
            buf_size = self.ysize * self.xsize * 2  # uint16
            try:
                shm = shared_memory.SharedMemory(
                    create=True,
                    size=buf_size,
                    name=f"shared_frame_memory_{z}",
                )
            except FileExistsError:
                shm = shared_memory.SharedMemory(
                    name=f"shared_frame_memory_{z}",
                    create=False,
                    size=buf_size,
                )

            shared_ndarray = np.ndarray(
                shape=(self.ysize, self.xsize),
                buffer=shm.buf,
                dtype=np.uint16,
            )

            self.shared_frame_memory_list.append(shm)
            self.shared_ndarray_list.append(shared_ndarray)

        # Shared image counter
        try:
            self.shared_image_count = shared_memory.ShareableList(
                [0], name="shared_image_count"
            )
        except FileExistsError:
            self.shared_image_count = shared_memory.ShareableList(
                None, name="shared_image_count"
            )

        # Build vis_args for worker
        vis_args = {
            "id": self.name,
            "saveroot": self.saveroot,
            "ysize": self.ysize,
            "xsize": self.xsize,
            "total_frames": self.num_samples,
            "zsize": self.zsize,
            "stim_intensity": self.stim_intensity,
            "GUI_mode": self.gui_mode,
            "dtype": np.uint16,
            "gui_screenshot_freq": self.gui_screenshot_freq,
            # Polygon mask params
            "polygon_shm_name": self.polygon_shm_name,
            "polygon_width": self.polygon_width,
            "polygon_height": self.polygon_height,
            "camera_roi": self.camera_roi,
            "calibration_points": self.calibration_points,
        }

        self.proc = BrainalyzerWorker(self.child_conn, vis_args)
        self.proc.start()
        logger.info("BrainalyzerWorker subprocess started")

    def process_sample(self, sample: Any):
        """Write camera frame to shared memory, poll for GUI events."""
        img = sample.get(self.input_device_name)
        if img is None:
            return

        zndx = self.sample_count % self.zsize

        # Store frame in shared memory
        self.shared_ndarray_list[zndx][:] = img[:]

        # Update shared image count
        self.sample_count += 1
        self.shared_image_count[0] = self.sample_count

        # Poll for events from GUI
        self._poll_events()

    def _poll_events(self):
        """Check pipe for events from BrainalyzerWorker."""
        if self.parent_conn and self.parent_conn.poll():
            data = self.parent_conn.recv()
            self.events.append(data)
            logger.info(f"BrainalyzerLogic: received event: {data}")
            self.current_event = data

    def check_logic(self) -> Optional[Dict[str, Any]]:
        """Translate GUI events into output device updates.

        Returns dict like:
            {"polygon": {"action": "upload_mask"}, "ldi": {"intensity": N}}
        """
        if self.current_event is None and not self.stimulus_is_on:
            return None

        updates = {}

        # New event while not stimulating -> start stimulus
        if not self.stimulus_is_on and self.current_event is not None:
            event = self.current_event
            event_type = event.get("event_type")
            stim_intensity = event.get("stim_intensity", 0)

            # Mask was already written by BrainalyzerWorker, just tell polygon to upload
            updates["polygon"] = {"action": "upload_mask"}
            updates["ldi"] = {"intensity": stim_intensity}

            if event_type in ("pulse-rect-roi-list", "full-field-button"):
                stim_duration_vols = event.get("stim_duration_vols", 4)
                stim_frames = stim_duration_vols * self.zsize
                zndx = self.sample_count % self.zsize
                stim_on = self.sample_count + self.zsize - zndx
                stim_off = stim_on + stim_frames

                if stim_off <= self.num_samples:
                    self.stim_param_list.append({
                        "stim_on": stim_on,
                        "stim_off": stim_off,
                        "stim_intensity": stim_intensity,
                        "event": event,
                    })

            elif event_type in ("stream-rect-roi-list", "stream-widefield"):
                self.stimulus_is_on = True
                self.stim_param_list.append({
                    "stim_on": self.sample_count,
                    "stim_intensity": stim_intensity,
                    "event": event,
                })

            self.current_event = None

        # Stop signal while stimulating
        elif self.stimulus_is_on and self.current_event is not None:
            updates["polygon"] = {"action": "blank"}
            updates["ldi"] = {"intensity": 0}
            self.stimulus_is_on = False

            if self.stim_param_list:
                self.stim_param_list[-1]["stim_off"] = self.sample_count

            self.current_event = None

        if updates:
            logger.info(f"BrainalyzerLogic: sending updates: {updates}")

        return updates if updates else None

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata including stimulus event log."""
        return {
            "name": self.name,
            "logic_class": self.logic_class,
            "stim_param_list": self.stim_param_list,
            "total_events": len(self.events),
            "total_samples": self.sample_count,
        }

    def close(self):
        """Clean up shared memory and stop worker."""
        # Close shared frame memory
        try:
            for shm in self.shared_frame_memory_list:
                shm.close()
                shm.unlink()

            if self.shared_image_count:
                self.shared_image_count.shm.close()
                self.shared_image_count.shm.unlink()
        except Exception as err:
            logger.info(f"Error unlinking shared memory: {err}")

        # Tell worker to close
        if self.parent_conn:
            try:
                self.parent_conn.send("close")
            except Exception as err:
                logger.info(f"Error closing worker: {err}")

        logger.info(f"BrainalyzerLogic '{self.name}' closed")
