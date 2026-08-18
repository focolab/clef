"""
XY Tracking Stage Input Device for CLEF.

Reads XY stage position from Micro-Manager via pycromanager.
Accumulates position history for metadata/saving.
"""

import logging
import time
from typing import Any, ClassVar, Dict, Optional

from core.io.input_device.BaseInputDevice import BaseInputDevice

logger = logging.getLogger(__name__)


class XYTrackingStageInput(BaseInputDevice):
    """Input device that reads XY stage position via Micro-Manager."""

    device_class: ClassVar[Optional[str]] = "xy_tracking_stage_input"
    device_type: ClassVar[Optional[str]] = "hardware"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name, config, io_manager=io_manager)
        self.mmc = None
        self._xy_stage_device = None
        self.xy_stage_position_list = []
        # Position reads hit the stage controller (often a slow serial query,
        # tens of ms). The control loop doesn't use this value — it's metadata —
        # so poll every Nth frame instead of every frame to keep the loop fast.
        self._poll_interval = 1
        self._counter = 0
        self._last = {"x": 0.0, "y": 0.0}
        self._warned_slow = False

    def connect(self):
        from pycromanager import Core
        self.mmc = Core(convert_camel_case=False)
        logger.info(f"XYTrackingStageInput '{self.name}' connected to pycromanager Core")

    def configure(self):
        device = self.config.get("xy_stage_device", None)
        if device is None:
            self._xy_stage_device = self.mmc.getXYStageDevice()
        else:
            self._xy_stage_device = device
        self._poll_interval = max(1, int(self.config.get("poll_interval_frames", 1)))
        logger.info(
            f"XYTrackingStageInput '{self.name}' using XY stage device: "
            f"{self._xy_stage_device} (poll every {self._poll_interval} frame(s))"
        )

    def _get_input(self) -> Dict[str, float]:
        # Reuse the cached position on non-poll frames so we don't stall the loop
        # on a slow controller query.
        self._counter += 1
        if self._poll_interval > 1 and self._counter % self._poll_interval != 0:
            return self._last

        t0 = time.perf_counter()
        x = self.mmc.getXPosition()
        y = self.mmc.getYPosition()
        dt = time.perf_counter() - t0
        if dt > 0.02 and not self._warned_slow:
            logger.warning(
                f"XY stage position query took {dt * 1000:.1f} ms — this runs "
                f"every {self._poll_interval} frame(s) and can throttle the loop. "
                "Raise poll_interval_frames to reduce it."
            )
            self._warned_slow = True

        self._last = {"x": x, "y": y}
        self.xy_stage_position_list.append([x, y])
        return self._last

    def get_metadata(self) -> Dict[str, Any]:
        base = super().get_metadata()
        base["xy_stage_position_list"] = self.xy_stage_position_list
        return base

    def close(self):
        logger.info(f"XYTrackingStageInput '{self.name}' closed")
