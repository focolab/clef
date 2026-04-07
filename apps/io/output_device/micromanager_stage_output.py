"""
Micro-Manager Stage Output Device for CLEF.

Uploads a z-position buffer to an ASI stage at configure-time.
The stage free-runs via TTL triggering during acquisition;
update_output() is a no-op for this demo.

Instantiates its own pycromanager Core object.
"""

import logging
from typing import Any, ClassVar, Dict, Optional

from core.io.output_device.BaseOutputDevice import BaseOutputDevice

logger = logging.getLogger(__name__)


class MicroManagerStageOutput(BaseOutputDevice):
    """Output device that configures a Micro-Manager z-stage with a position buffer."""

    device_class: ClassVar[Optional[str]] = "micromanager_stage"
    device_type: ClassVar[Optional[str]] = "hardware"

    def __init__(self, name: str, config: Dict[str, Any] | None = None, io_manager: Any = None):
        super().__init__(name, config, io_manager=io_manager)
        self.mmc = None
        self._stage_device = None

    def connect(self):
        """Instantiate pycromanager Core connection."""
        from pycromanager import Core
        self.mmc = Core(convert_camel_case=False)
        logger.info(f"MicroManagerStageOutput '{self.name}' connected to pycromanager Core")

    def configure(self):
        """Upload z-position buffer and TTL property sequences to ASI stage."""
        from pycromanager import JavaObject

        cfg = self.config
        num_z_planes = cfg.get("num_z_planes", 1)
        z_step = cfg.get("z_step_size_um", 1.0)
        ttl_device = cfg.get("ttl_device")
        ttl_state = cfg.get("ttl_state")

        if num_z_planes <= 1:
            logger.info(f"MicroManagerStageOutput '{self.name}': single plane, no buffer upload needed")
            return

        # Ensure z_step is nonzero to avoid stage hang
        if z_step == 0:
            z_step = 1.0
            logger.warning("z_step was 0, setting to 1 to avoid hanging")

        # Compute symmetric z range around 0
        z_start = -(num_z_planes - 1) * z_step / 2
        z_end = (num_z_planes - 1) * z_step / 2

        self._stage_device = self.mmc.getFocusDevice()

        # Build position and TTL vectors
        dv = JavaObject("mmcorej.DoubleVector")
        sv = JavaObject("mmcorej.StrVector")
        z_positions = []

        z = z_start
        while z <= z_end:
            dv.add(z)
            z_positions.append(z)
            sv.add(str(ttl_state))
            z += z_step

        logger.info(f"Uploading {len(z_positions)} z-positions to ASI stage: {z_positions}")

        # Upload and start sequences
        self.mmc.setPosition(self._stage_device, z_start)
        self.mmc.waitForDevice(self._stage_device)
        self.mmc.stopStageSequence(self._stage_device)
        self.mmc.loadStageSequence(self._stage_device, dv)
        self.mmc.stopPropertySequence(ttl_device, "State")
        self.mmc.loadPropertySequence(ttl_device, "State", sv)
        self.mmc.startStageSequence(self._stage_device)
        self.mmc.startPropertySequence(ttl_device, "State")

        logger.info(
            f"MicroManagerStageOutput '{self.name}' configured: "
            f"{len(z_positions)} planes from {z_start} to {z_end} um"
        )

    def _update_output(self, **kwargs):
        """No-op. Stage runs autonomously via TTL triggering."""
        pass

    def close(self):
        """Stop stage sequence, return to z=0, stop TTL sequence."""
        if self.mmc is None or self._stage_device is None:
            return

        cfg = self.config
        ttl_device = cfg.get("ttl_device")

        try:
            self.mmc.stopStageSequence(self._stage_device)
            self.mmc.waitForDevice(self._stage_device)
            self.mmc.setPosition(self._stage_device, 0)
            self.mmc.waitForDevice(self._stage_device)
            if ttl_device:
                self.mmc.stopPropertySequence(ttl_device, "State")
        except Exception as e:
            logger.warning(f"Error during stage close: {e}")

        logger.info(f"MicroManagerStageOutput '{self.name}' closed")
