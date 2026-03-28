"""
XY Tracking Stage Output Device for CLEF2.

Exposes a SharedMemory list with two elements (offset_x, offset_y).
On each update_output call, reads the offsets, and if nonzero,
applies them as a relative XY stage move via Micro-Manager, then resets.

An external subprocess (XYTrackingWorker) writes offsets directly
into the shared memory, bypassing the normal logic→instruction path.
"""

import logging
from multiprocessing import shared_memory
from typing import Any, ClassVar, Dict, Optional

from clef2.core.io.output_device.BaseOutputDevice import BaseOutputDevice

logger = logging.getLogger(__name__)


class XYTrackingStageOutput(BaseOutputDevice):
    """Output device that applies XY stage corrections from shared memory."""

    device_class: ClassVar[Optional[str]] = "xy_tracking_stage_output"
    device_type: ClassVar[Optional[str]] = "hardware"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name, config, io_manager=io_manager)
        self.mmc = None
        self.shared_stage_offset_xy = None
        self._shm_name = None

    def connect(self):
        from pycromanager import Core
        self.mmc = Core(convert_camel_case=False)
        logger.info(f"XYTrackingStageOutput '{self.name}' connected to pycromanager Core")

    def configure(self):
        self._shm_name = self.config.get("shm_name", "shared_stage_offset_xy")
        try:
            self.shared_stage_offset_xy = shared_memory.ShareableList(
                [0, 0], name=self._shm_name
            )
        except FileExistsError:
            self.shared_stage_offset_xy = shared_memory.ShareableList(
                name=self._shm_name
            )
        logger.info(f"XYTrackingStageOutput '{self.name}' configured with SHM '{self._shm_name}'")

    def _update_output(self, **kwargs):
        if self.shared_stage_offset_xy is None:
            return

        offset_x = self.shared_stage_offset_xy[0]
        offset_y = self.shared_stage_offset_xy[1]

        if offset_x or offset_y:
            self.shared_stage_offset_xy[0] = 0
            self.shared_stage_offset_xy[1] = 0
            self.mmc.setRelativeXYPosition(int(offset_x), int(offset_y))

    def close(self):
        if self.shared_stage_offset_xy is not None:
            try:
                self.shared_stage_offset_xy.shm.close()
                self.shared_stage_offset_xy.shm.unlink()
            except Exception as e:
                logger.warning(f"Error closing stage offset SHM: {e}")
        logger.info(f"XYTrackingStageOutput '{self.name}' closed")
