"""
XY Tracking Logic for CLEF.

Orchestrates the XYTrackingWorker GUI subprocess for centroid-based
stage tracking of a single bright fluorescent blob. The worker accumulates
stage commands directly in shared memory; this logic class calls update_output
on the stage output device each sample to apply whatever motion is still owed,
and records the microns-per-pixel ratio the worker measures during calibration.

The control algorithm (proportional / PID / Kalman) is selected here from
config and switchable live in the worker's GUI.
"""

import json
import logging
import numpy as np
from multiprocessing import Pipe
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from core.logic.BaseClosedLoopLogic import BaseClosedLoopLogic

logger = logging.getLogger(__name__)


class XYTrackingLogic(BaseClosedLoopLogic):
    """Centroid-based XY stage tracking via GUI subprocess."""

    logic_class: ClassVar[Optional[str]] = "xy_tracking_logic"

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
            name, config, output_devices, gui_parameters,
            input_devices=input_devices, io_manager=io_manager,
            config_manager=config_manager,
        )

        cfg = self.config
        self.input_device_name = cfg.get("input_device_name", "camera")
        self.stage_output_name = cfg.get("stage_output_name", "xy_stage_output")

        # Tracking / calibration params passed to the worker
        self.micron_to_pix_ratio = cfg.get("micron_to_pix_ratio", 100.0 / 74.0)
        self.stage_dampening_factor = cfg.get("stage_dampening_factor", 0.5)
        self.threshold_frac = cfg.get("threshold_frac", 0.5)

        # Control algorithm and its per-algorithm parameter block. The worker
        # instantiates all of them so they can be switched live from the GUI.
        self.tracking_algorithm = cfg.get("tracking_algorithm", "kalman")
        self.algorithm_params = {
            name: dict(cfg.get(name, {}) or {})
            for name in ("proportional", "pid", "kalman")
        }
        # stage_dampening_factor was the old single proportional gain; keep it
        # working as that algorithm's Kp so existing configs behave the same.
        self.algorithm_params["proportional"].setdefault(
            "kp", self.stage_dampening_factor
        )
        self.reset_after_lost_frames = cfg.get("reset_after_lost_frames", 25)
        self.deadband_px = cfg.get("deadband_px", 2.0)
        self.centroid_smoothing = cfg.get("centroid_smoothing", 0.0)
        self.max_step_um = cfg.get("max_step_um", 200.0)
        self.min_puncta_snr = cfg.get("min_puncta_snr", 6.0)
        self.enable_axis0 = cfg.get("enable_axis0", True)
        self.enable_axis1 = cfg.get("enable_axis1", True)
        self.invert_axis0 = cfg.get("invert_axis0", False)
        self.invert_axis1 = cfg.get("invert_axis1", False)
        self.swap_axes = cfg.get("swap_axes", False)
        self.roi_radius_px = cfg.get("roi_radius_px", 30)
        self.jog_step_um = cfg.get("jog_step_um", 50)
        self.cal_step_um = cfg.get("cal_step_um", 100)
        self.cal_settle_frames = cfg.get("cal_settle_frames", 8)
        self.ratio_increment = cfg.get("ratio_increment", 0.02)
        self.save_dir = cfg.get("save_dir", "./output")

        # Resolved at initialize_model()
        self.image_count_shm_name = "shared_image_count"

        # Sub-acquisition epochs reported by the worker (Option B: we store the
        # frame ranges + timestamps as metadata; the full-session video is saved
        # by the data interface and sub-clips are sliced from it downstream).
        self.epochs = []
        self.open_epoch = None

        # IPC
        self.parent_conn = None
        self.child_conn = None
        self.proc = None

    def initialize_model(self):
        """Resolve device info and start XYTrackingWorker."""
        camera_dev = self.input_devices.get(self.input_device_name)
        self.xsize = 0
        self.ysize = 0
        self.shm_names = []
        self.ring_size = 1

        if camera_dev is not None:
            self.xsize = camera_dev.width
            self.ysize = camera_dev.height

            di = getattr(camera_dev, "data_interface", None)
            if di is not None:
                if hasattr(di, "shm_names"):
                    self.shm_names = di.shm_names
                if hasattr(di, "image_count_shm_name"):
                    self.image_count_shm_name = di.image_count_shm_name
                # Frame ring-buffer size: the worker must read the newest of
                # these segments, not always segment 0.
                if hasattr(di, "config"):
                    self.ring_size = di.config.get("shm_buffer_size", 1)
        if self.ring_size <= 0:
            self.ring_size = max(1, len(self.shm_names))

        # Fall back to config for frame dims
        if self.xsize == 0 or self.ysize == 0:
            self.xsize = self.config.get("xsize", 0)
            self.ysize = self.config.get("ysize", 0)
        if self.xsize == 0 or self.ysize == 0:
            logger.warning("xsize/ysize not set - worker may fail.")

        # Resolve stage output SHM name
        stage_shm_name = "shared_stage_offset_xy"
        stage_dev = self.output_devices.get(self.stage_output_name)
        if stage_dev is not None and hasattr(stage_dev, "_shm_name"):
            stage_shm_name = stage_dev._shm_name

        # Colocate the worker's CSVs with the session's other outputs.
        scfg = getattr(self.config_manager, "session_config", None) if self.config_manager else None
        session_dir = getattr(scfg, "sample_data_dir", None) if scfg else None
        if session_dir:
            self.save_dir = session_dir

        self._start_worker(stage_shm_name)

    def _start_worker(self, stage_shm_name):
        from apps.logic.xy_tracking_worker import XYTrackingWorker

        self.parent_conn, self.child_conn = Pipe()

        vis_args = {
            "ysize": self.ysize,
            "xsize": self.xsize,
            "ring_size": self.ring_size,
            "dtype": np.uint16,
            "shm_names": self.shm_names,
            "image_count_shm_name": self.image_count_shm_name,
            "stage_shm_name": stage_shm_name,
            "micron_to_pix_ratio": self.micron_to_pix_ratio,
            "tracking_algorithm": self.tracking_algorithm,
            "algorithm_params": self.algorithm_params,
            "reset_after_lost_frames": self.reset_after_lost_frames,
            "threshold_frac": self.threshold_frac,
            "deadband_px": self.deadband_px,
            "centroid_smoothing": self.centroid_smoothing,
            "max_step_um": self.max_step_um,
            "min_puncta_snr": self.min_puncta_snr,
            "enable_axis0": self.enable_axis0,
            "enable_axis1": self.enable_axis1,
            "invert_axis0": self.invert_axis0,
            "invert_axis1": self.invert_axis1,
            "swap_axes": self.swap_axes,
            "roi_radius_px": self.roi_radius_px,
            "jog_step_um": self.jog_step_um,
            "cal_step_um": self.cal_step_um,
            "cal_settle_frames": self.cal_settle_frames,
            "ratio_increment": self.ratio_increment,
            "save_dir": self.save_dir,
            "rec_id": self.name,
        }

        self.proc = XYTrackingWorker(self.child_conn, vis_args)
        self.proc.start()
        logger.info(
            f"XYTrackingWorker started (ring_size={self.ring_size}, "
            f"frame={self.xsize}x{self.ysize})"
        )

    def process_sample(self, sample: Any):
        """Drain worker events, then apply any pending stage offsets."""
        self._drain_events()

        stage_dev = self.output_devices.get(self.stage_output_name)
        if stage_dev is not None:
            stage_dev.update_output()

    def _drain_events(self):
        """Consume calibration + sub-acquisition events sent by the worker."""
        if self.parent_conn is None:
            return
        while self.parent_conn.poll():
            msg = self.parent_conn.recv()
            if not isinstance(msg, dict):
                continue
            kind = msg.get("type")
            if kind == "calibration_update":
                self.micron_to_pix_ratio = msg["micron_to_pix_ratio"]
                logger.info(
                    f"Calibration update: {self.micron_to_pix_ratio:.5f} um/px"
                )
            elif kind == "recording_start":
                self.open_epoch = {
                    k: msg.get(k) for k in ("index", "name", "start_frame", "start_ts")
                }
                logger.info(f"Sub-acquisition {msg.get('name')} started")
            elif kind == "recording_stop":
                self.epochs.append({
                    k: msg.get(k) for k in (
                        "index", "name", "start_frame", "stop_frame",
                        "start_ts", "stop_ts",
                    )
                })
                self.open_epoch = None
                logger.info(
                    f"Sub-acquisition {msg.get('name')} stopped: frames "
                    f"{msg.get('start_frame')}-{msg.get('stop_frame')}"
                )

    def get_metadata(self) -> Dict[str, Any]:
        base = super().get_metadata()
        base["micron_to_pix_ratio"] = self.micron_to_pix_ratio
        base["stage_dampening_factor"] = self.stage_dampening_factor
        base["tracking_algorithm"] = self.tracking_algorithm
        base["algorithm_params"] = self.algorithm_params
        base["sub_acquisitions"] = self.epochs
        if self.open_epoch is not None:
            base["sub_acquisition_open"] = self.open_epoch
        return base

    def save_data(self, **kwargs):
        """Persist tracking parameters + sub-acquisition frame ranges as JSON.

        The full-session video is saved by the camera's data interface; the
        frame ranges here identify each sub-acquisition within that video.
        """
        # Catch any recording-stop event still queued from the last frames.
        self._drain_events()

        savefilename = kwargs.get("savefilename") or kwargs.get("filepath")
        if savefilename is None:
            return
        path = Path(str(savefilename)).with_suffix(".json")
        try:
            with open(path, "w") as f:
                json.dump(self.get_metadata(), f, indent=2, default=str)
            logger.info(f"Saved tracking metadata to {path}")
        except Exception as err:
            logger.warning(f"Failed to save tracking metadata: {err}")

    def close(self):
        if self.parent_conn is not None:
            try:
                self.parent_conn.send("close")
            except Exception:
                pass
        if self.proc is not None:
            self.proc.join(timeout=5)
            if self.proc.is_alive():
                logger.warning("XYTrackingWorker did not exit, terminating")
                self.proc.terminate()
        logger.info("XYTrackingLogic closed")
