"""
Brainalyzer Logic for CLEF.

Orchestrates the BrainalyzerWorker GUI subprocess for interactive
closed-loop microscopy. Relays stimulus events from the GUI to
output devices (polygon, LDI). Frame shared memory is managed by
SharedMemoryUint16DataInterface.

The BrainalyzerWorker generates polygon masks directly and writes
them into the polygon output device's shared memory buffer.
"""

import json
import logging
import numpy as np
from multiprocessing import Pipe
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from core.logic.BaseClosedLoopLogic import BaseClosedLoopLogic

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
        input_devices: Dict[str, Any] | None = None,
        io_manager: Any = None,
        config_manager: Any = None,
    ):
        super().__init__(
            name,
            config,
            output_devices,
            gui_parameters,
            input_devices=input_devices,
            io_manager=io_manager,
            config_manager=config_manager,
        )

        cfg = self.config

        # Core experiment params
        self.gui_mode = cfg.get("gui_mode", "neural_imaging")
        self.stim_intensity = cfg.get("stim_intensity", 0)
        self.num_z_planes = cfg.get("num_z_planes", cfg.get("zsize", 1))
        self.input_device_name = next(iter(self.input_devices), "camera")

        # Calibration
        self.calibration_file = cfg.get("calibration_file", None)
        self.calibration_points = None

        # Polygon state
        self.polygon_shm_name = None
        self.polygon_width = None
        self.polygon_height = None

        # IPC
        self.parent_conn = None
        self.child_conn = None
        self.proc = None

        # Stimulus state
        self.events = []
        self.current_event = None
        self.stimulus_is_on = False
        self.active_pulse_stim_off = None  # frame index to turn off pulsed stim
        self.stim_param_list = []
        self.sample_count = 0

        # Save dir
        self.saveroot = cfg.get("save_dir", "./output")

        # GUI screenshot freq
        self.gui_screenshot_freq = self.gui_parameters.get("gui_screenshot_freq", 0)

    def initialize_model(self):
        """Load calibration, resolve hardware info, start worker."""

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

        # Resolve camera input device info
        self.xsize = 0
        self.ysize = 0
        self.camera_roi = (0, 0, 0, 0)

        camera_dev = self.input_devices.get(self.input_device_name)
        if camera_dev is not None:
            self.xsize = camera_dev.width
            self.ysize = camera_dev.height
            self.camera_roi = camera_dev.get_roi()
            logger.info(f"Camera: {self.xsize}x{self.ysize}, ROI={self.camera_roi}")

        # Fall back to config if device didn't provide values
        if self.xsize == 0 or self.ysize == 0:
            self.xsize = self.config.get("xsize", 0)
            self.ysize = self.config.get("ysize", 0)
            self.camera_roi = self.config.get(
                "camera_roi", (0, 0, self.xsize, self.ysize)
            )

        if self.xsize == 0 or self.ysize == 0:
            logger.warning("xsize/ysize not set — worker may fail.")

        # Read shm metadata from data interface (single source of truth)
        self.shm_names = []
        if camera_dev is not None and hasattr(camera_dev, "data_interface"):
            di = camera_dev.data_interface
            di_nz = di.config.get("shm_buffer_size", 0) if hasattr(di, "config") else 0
            if di_nz > 0:
                self.num_z_planes = di_nz
            if hasattr(di, "shm_names"):
                self.shm_names = di.shm_names
            if hasattr(di, "image_count_shm_name"):
                self.image_count_shm_name = di.image_count_shm_name

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
        """Create Pipe and start BrainalyzerWorker.

        Frame shared memory is owned by the data interface; we just read
        the shm names to pass to the worker.
        """
        from apps.logic.brainalyzer_worker import BrainalyzerWorker

        self.parent_conn, self.child_conn = Pipe()

        # Build vis_args for worker
        vis_args = {
            "id": self.name,
            "saveroot": self.saveroot,
            "ysize": self.ysize,
            "xsize": self.xsize,
            "zsize": self.num_z_planes,
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
            "shm_names": self.shm_names,
            "image_count_shm_name": self.image_count_shm_name,
        }

        self.proc = BrainalyzerWorker(self.child_conn, vis_args)
        self.proc.start()
        logger.info("BrainalyzerWorker subprocess started")

    def process_sample(self, sample: Any):
        """Poll for GUI events. Frame shm is handled by the data interface."""
        img = sample.get(self.input_device_name)
        if img is None:
            return

        self.sample_count += 1
        self._poll_events()

    def _poll_events(self):
        """Check pipe for events from BrainalyzerWorker."""
        if self.parent_conn and self.parent_conn.poll():
            data = self.parent_conn.recv()
            self.events.append(data)
            logger.info(f"BrainalyzerLogic: received event: {data}")
            self.current_event = data

    def _check_logic(self) -> Optional[Dict[str, Any]]:
        """Translate GUI events into output device updates.

        Returns dict like:
            {"polygon": {"action": "upload_mask"}, "ldi": {"intensity": N}}
        """
        # Check for pulsed stim expiry first
        if (
            self.active_pulse_stim_off is not None
            and self.sample_count >= self.active_pulse_stim_off
        ):
            updates = {
                "ldi": {"intensity": 0},
            }
            logger.info(
                f"BrainalyzerLogic: pulse stim off at frame {self.sample_count}"
            )
            self.active_pulse_stim_off = None
            return updates

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
                stim_frames = stim_duration_vols * self.num_z_planes
                zndx = self.sample_count % self.num_z_planes
                stim_on = self.sample_count + self.num_z_planes - zndx
                stim_off = stim_on + stim_frames

                self.active_pulse_stim_off = stim_off

                self.stim_param_list.append(
                    {
                        "stim_on": stim_on,
                        "stim_off": stim_off,
                        "stim_intensity": stim_intensity,
                        "event": event,
                    }
                )

            elif event_type in ("stream-rect-roi-list", "stream-widefield"):
                self.stimulus_is_on = True
                self.stim_param_list.append(
                    {
                        "stim_on": self.sample_count,
                        "stim_intensity": stim_intensity,
                        "event": event,
                    }
                )

            self.current_event = None

        # Stop signal while stimulating
        elif self.stimulus_is_on and self.current_event is not None:
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
        base = super().get_metadata()
        base.update(
            {
                "stim_param_list": self.stim_param_list,
                "total_events": len(self.events),
                "total_samples": self.sample_count,
            }
        )
        return base

    def close(self):
        """Stop worker. Frame shared memory is cleaned up by the data interface."""
        if self.parent_conn:
            try:
                self.parent_conn.send("close")
            except Exception as err:
                logger.info(f"Error sending close to worker: {err}")

        if self.proc is not None:
            self.proc.join(timeout=5)
            if self.proc.is_alive():
                logger.warning("Worker did not exit in time, terminating")
                self.proc.terminate()

        logger.info(f"BrainalyzerLogic '{self.name}' closed")
