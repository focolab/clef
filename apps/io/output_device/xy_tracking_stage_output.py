"""
XY Tracking Stage Output Device for CLEF.

Exposes a SharedMemory list that carries stage corrections from the
XYTrackingWorker subprocess, which writes into it directly rather than going
through the normal logic->instruction path. On each update_output call the
still-owed motion is applied as a relative XY stage move via Micro-Manager.

The buffer is a two-way handshake with one writer per slot, so neither side can
race the other and no command is lost:

  0,1  cmd_total      cumulative microns ever commanded   (written by the worker)
  2,3  applied_total  cumulative microns issued to the stage      (written here)
  4    applied_seq    number of moves issued                      (written here)
  5    applied_ts     perf_counter() of the most recent move      (written here)

The move owed at any moment is cmd_total - applied_total. Carrying totals rather
than a one-shot offset means a correction written while a previous move was in
flight stays owed instead of being overwritten, and it lets the worker see both
what is still pending and what has already been applied — which is what its
estimators need in order to not command the same error twice.

All slots are floats. Stage moves are issued in floating-point microns:
setRelativeXYPosition takes doubles and the stage resolves well under a micron,
so rounding corrections to whole microns would throw away most of them.
"""

import logging
import time
from multiprocessing import shared_memory
from typing import Any, ClassVar, Dict, Optional

from core.io.output_device.BaseOutputDevice import BaseOutputDevice

logger = logging.getLogger(__name__)

# Slot layout of the shared buffer (see module docstring).
CMD_0, CMD_1, APPLIED_0, APPLIED_1, APPLIED_SEQ, APPLIED_TS = range(6)
NUM_SLOTS = 6


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
        self.shared_stage_offset_xy = self._create_or_replace(self._shm_name)
        # A segment left by a prior run may still hold nonzero totals; zero them
        # so we never apply a spurious move on the first update.
        for i in range(NUM_SLOTS):
            self.shared_stage_offset_xy[i] = 0.0
        logger.info(
            f"XYTrackingStageOutput '{self.name}' configured with SHM "
            f"'{self._shm_name}'"
        )

    @staticmethod
    def _create_or_replace(name: str) -> shared_memory.ShareableList:
        """Attach to the buffer, replacing one with an incompatible layout.

        A segment left behind by an older build has a different slot count and
        packing format, and writing floats into it would fail or silently
        truncate — so it is discarded rather than reused.
        """
        try:
            return shared_memory.ShareableList([0.0] * NUM_SLOTS, name=name)
        except FileExistsError:
            existing = shared_memory.ShareableList(name=name)
            slots = len(existing)  # unreadable once the mapping is closed
            if slots == NUM_SLOTS:
                return existing
            logger.warning(
                f"Stage offset SHM '{name}' has {slots} slots, expected "
                f"{NUM_SLOTS}; recreating it."
            )
            existing.shm.close()
            existing.shm.unlink()
            try:
                return shared_memory.ShareableList([0.0] * NUM_SLOTS, name=name)
            except FileExistsError:
                # Unlinking only frees the name once every handle is closed, so
                # this means another process still has the old buffer mapped.
                # Its layout cannot carry fractional-micron commands, and
                # reusing it would silently truncate every correction.
                raise RuntimeError(
                    f"Stage offset SHM '{name}' is held open by another process "
                    f"with an incompatible {slots}-slot layout. Stop the other "
                    f"CLEF session and start this one again."
                ) from None

    def _update_output(self, **kwargs):
        shl = self.shared_stage_offset_xy
        if shl is None or self.mmc is None:
            return

        # Snapshot the worker's running total first: anything it adds after this
        # read stays in the difference and is applied on the next call.
        cmd_0 = float(shl[CMD_0])
        cmd_1 = float(shl[CMD_1])
        move_0 = cmd_0 - float(shl[APPLIED_0])
        move_1 = cmd_1 - float(shl[APPLIED_1])

        if move_0 == 0.0 and move_1 == 0.0:
            return

        self.mmc.setRelativeXYPosition(move_0, move_1)
        # Advance to the snapshot, not by the move, so a concurrent write is
        # carried rather than double-counted.
        shl[APPLIED_0] = cmd_0
        shl[APPLIED_1] = cmd_1
        shl[APPLIED_SEQ] = float(shl[APPLIED_SEQ]) + 1.0
        shl[APPLIED_TS] = time.perf_counter()
        logger.debug(f"Applied relative stage move ({move_0:.3f}, {move_1:.3f}) um")

    def close(self):
        if self.shared_stage_offset_xy is not None:
            try:
                self.shared_stage_offset_xy.shm.close()
                self.shared_stage_offset_xy.shm.unlink()
            except Exception as e:
                logger.warning(f"Error closing stage offset SHM: {e}")
        logger.info(f"XYTrackingStageOutput '{self.name}' closed")
