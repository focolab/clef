"""
XY Tracking Stage Input Device for CLEF2.

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
        logger.info(f"XYTrackingStageInput '{self.name}' using XY stage device: {self._xy_stage_device}")

    def _get_input(self) -> Dict[str, float]:
        x = self.mmc.getXPosition()
        y = self.mmc.getYPosition()
        self.xy_stage_position_list.append([x, y])
        return {"x": x, "y": y}

    def get_metadata(self) -> Dict[str, Any]:
        base = super().get_metadata()
        base["xy_stage_position_list"] = self.xy_stage_position_list
        return base

    def close(self):
        logger.info(f"XYTrackingStageInput '{self.name}' closed")
