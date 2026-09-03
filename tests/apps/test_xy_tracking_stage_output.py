"""Tests for the XY tracking stage output device's command handshake."""

import pytest

from apps.io.output_device.xy_tracking_stage_output import (
    APPLIED_0,
    APPLIED_1,
    APPLIED_SEQ,
    APPLIED_TS,
    CMD_0,
    CMD_1,
    NUM_SLOTS,
    XYTrackingStageOutput,
)


class FakeCore:
    """Records relative moves instead of driving a stage."""

    def __init__(self):
        self.moves = []

    def setRelativeXYPosition(self, dx, dy):
        self.moves.append((dx, dy))


@pytest.fixture
def device():
    dev = XYTrackingStageOutput("stage", {"shm_name": "test_stage_offset_xy"})
    dev.mmc = FakeCore()
    dev.configure()
    yield dev
    dev.close()


def command(dev, axis0, axis1):
    """Stand in for the worker: add to the running command total."""
    shl = dev.shared_stage_offset_xy
    shl[CMD_0] = float(shl[CMD_0]) + axis0
    shl[CMD_1] = float(shl[CMD_1]) + axis1


def test_starts_zeroed(device):
    assert len(device.shared_stage_offset_xy) == NUM_SLOTS
    assert list(device.shared_stage_offset_xy) == [0.0] * NUM_SLOTS


def test_no_command_means_no_move(device):
    device.update_output()
    assert device.mmc.moves == []


def test_sub_micron_correction_is_issued(device):
    """The whole point of the float command path: a 0.3 um correction used to
    truncate to a 0 um move, leaving the blob permanently off center."""
    command(device, 0.3, -0.45)
    device.update_output()
    assert device.mmc.moves == [(0.3, -0.45)]


def test_tiny_corrections_are_issued(device):
    """Nothing is filtered out by size; the stage decides what it can resolve."""
    command(device, 0.001, -0.002)
    device.update_output()
    assert device.mmc.moves == [(0.001, -0.002)]


def test_applied_totals_track_the_commands(device):
    command(device, 10.0, 5.0)
    device.update_output()
    command(device, -2.5, 1.25)
    device.update_output()

    shl = device.shared_stage_offset_xy
    assert device.mmc.moves == [(10.0, 5.0), (-2.5, 1.25)]
    assert float(shl[APPLIED_0]) == pytest.approx(7.5)
    assert float(shl[APPLIED_1]) == pytest.approx(6.25)
    assert float(shl[APPLIED_0]) == pytest.approx(float(shl[CMD_0]))
    assert float(shl[APPLIED_SEQ]) == 2.0
    assert float(shl[APPLIED_TS]) > 0.0


def test_commands_between_updates_are_not_lost(device):
    """Two corrections written before the loop picks either up must both move
    the stage. The old one-shot offset silently dropped the first."""
    command(device, 3.0, 0.0)
    command(device, 4.0, 0.0)
    device.update_output()
    assert device.mmc.moves == [(7.0, 0.0)]
    assert float(device.shared_stage_offset_xy[APPLIED_0]) == pytest.approx(7.0)


def test_command_during_an_apply_stays_owed(device):
    """A correction written after the device snapshots the total is carried to
    the next update rather than being double-counted or dropped."""
    command(device, 6.0, 0.0)

    shl = device.shared_stage_offset_xy
    original = device.mmc.setRelativeXYPosition

    def move_and_interleave(dx, dy):
        original(dx, dy)
        command(device, 2.0, 0.0)  # worker writes while the move is in flight

    device.mmc.setRelativeXYPosition = move_and_interleave
    device.update_output()
    device.mmc.setRelativeXYPosition = original

    assert device.mmc.moves == [(6.0, 0.0)]
    assert float(shl[CMD_0]) - float(shl[APPLIED_0]) == pytest.approx(2.0)

    device.update_output()
    assert device.mmc.moves == [(6.0, 0.0), (2.0, 0.0)]


def test_incompatible_segment_held_open_is_reported():
    """A buffer in the old two-int layout cannot carry fractional-micron
    commands. If another process still holds it open the name cannot be
    reclaimed, so say what is wrong rather than truncating every correction."""
    from multiprocessing import shared_memory

    name = "test_stale_stage_offset_xy"
    stale = shared_memory.ShareableList([0, 0], name=name)
    try:
        dev = XYTrackingStageOutput("stage", {"shm_name": name})
        dev.mmc = FakeCore()
        with pytest.raises(RuntimeError, match="held open by another process"):
            dev.configure()
    finally:
        stale.shm.close()
        try:
            stale.shm.unlink()
        except FileNotFoundError:
            pass


def test_non_finite_move_is_refused(device):
    """A NaN command passes the zero-check, would be handed to the stage, and
    would leave applied_total NaN - after which every owed-motion difference is
    NaN and the loop silently stops moving. Refuse it and resync instead."""
    command(device, 5.0, 0.0)
    device.update_output()
    assert device.mmc.moves == [(5.0, 0.0)]

    shl = device.shared_stage_offset_xy
    shl[CMD_0] = float("nan")
    device.update_output()

    assert device.mmc.moves == [(5.0, 0.0)], "NaN must not reach the stage"
    # Totals stay finite, so the loop recovers instead of dying silently.
    assert float(shl[CMD_0]) == pytest.approx(5.0)
    assert float(shl[APPLIED_0]) == pytest.approx(5.0)

    command(device, 2.0, 0.0)
    device.update_output()
    assert device.mmc.moves == [(5.0, 0.0), (2.0, 0.0)], "loop recovered"


def test_infinite_move_is_refused(device):
    shl = device.shared_stage_offset_xy
    shl[CMD_0] = float("inf")
    device.update_output()
    assert device.mmc.moves == []
    assert float(shl[CMD_0]) == pytest.approx(0.0)
