"""
XY Tracking Logic for CLEF.

Orchestrates the XYTrackingWorker GUI subprocess for centroid-based
stage tracking. The worker writes stage offsets directly into shared
memory; this logic class calls update_output on the stage output
device each sample to apply pending corrections.
"""

import logging
import numpy as np
from multiprocessing import Pipe
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

        # Tracking params passed to worker
        self.micron_to_pix_ratio = cfg.get("micron_to_pix_ratio", 100.0 / 74.0)
        self.stage_dampening_factor = cfg.get("stage_dampening_factor", 0.5)

        # IPC
        self.parent_conn = None
        self.child_conn = None
        self.proc = None

    def initialize_model(self):
        """Resolve device info and start XYTrackingWorker."""
        # Resolve camera input
        camera_dev = self.input_devices.get(self.input_device_name)
        self.xsize = 0
        self.ysize = 0
        self.shm_names = []
        self.image_count_shm_name = "shared_frame_memory_image_count"
        camera_binning = "1x1"

        if camera_dev is not None:
            self.xsize = camera_dev.width
            self.ysize = camera_dev.height
            if hasattr(camera_dev, "camera_binning"):
                camera_binning = camera_dev.camera_binning

            # Read SHM metadata from data interface
            if hasattr(camera_dev, "data_interface") and camera_dev.data_interface is not None:
                di = camera_dev.data_interface
                if hasattr(di, "shm_names"):
                    self.shm_names = di.shm_names
                if hasattr(di, "image_count_shm_name"):
                    self.image_count_shm_name = di.image_count_shm_name

        # Fall back to config
        if self.xsize == 0 or self.ysize == 0:
            self.xsize = self.config.get("xsize", 0)
            self.ysize = self.config.get("ysize", 0)

        if self.xsize == 0 or self.ysize == 0:
            logger.warning("xsize/ysize not set — worker may fail.")

        # Resolve stage output SHM name
        stage_shm_name = "shared_stage_offset_xy"
        stage_dev = self.output_devices.get(self.stage_output_name)
        if stage_dev is not None and hasattr(stage_dev, "_shm_name"):
            stage_shm_name = stage_dev._shm_name

        self._start_worker(camera_binning, stage_shm_name)

    def _start_worker(self, camera_binning, stage_shm_name):
        from apps.logic.xy_tracking_worker import XYTrackingWorker

        self.parent_conn, self.child_conn = Pipe()

        vis_args = {
            "ysize": self.ysize,
            "xsize": self.xsize,
            "zsize": 1,
            "dtype": np.uint16,
            "shm_names": self.shm_names,
            "image_count_shm_name": self.image_count_shm_name,
            "stage_shm_name": stage_shm_name,
            "camera_binning": camera_binning,
            "micron_to_pix_ratio": self.micron_to_pix_ratio,
            "stage_dampening_factor": self.stage_dampening_factor,
        }

        self.proc = XYTrackingWorker(self.child_conn, vis_args)
        self.proc.start()
        logger.info("XYTrackingWorker subprocess started")

    def process_sample(self, sample: Any):
        """Apply any pending stage offsets written by the worker."""
        # Poll worker for events (future extensibility)
        if self.parent_conn is not None:
            while self.parent_conn.poll():
                msg = self.parent_conn.recv()
                logger.debug(f"XYTrackingLogic received: {msg}")

        # Trigger stage output to apply offsets from SHM
        stage_dev = self.output_devices.get(self.stage_output_name)
        if stage_dev is not None:
            stage_dev.update_output()

    def close(self):
        if self.parent_conn is not None:
            try:
                self.parent_conn.send("close")
            except Exception:
                pass
        if self.proc is not None:
            self.proc.join(timeout=5)
        logger.info("XYTrackingLogic closed")
