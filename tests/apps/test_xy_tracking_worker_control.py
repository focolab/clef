"""Tests for XYTrackingWorker's control path, exercised without the GUI.

Covers the wiring the controller unit tests cannot see: pixel error converted to
stage-axis microns with the right sign, the command total accumulating in shared
memory, and the pending/applied handshake with the output device across the
process boundary.
"""

import numpy as np
import pytest
from multiprocessing import shared_memory

from apps.io.output_device.xy_tracking_stage_output import (
    APPLIED_0, APPLIED_1, APPLIED_SEQ, APPLIED_TS, CMD_0, CMD_1, NUM_SLOTS,
)
from apps.logic.xy_tracking_worker import XYTrackingWorker

YSIZE, XSIZE = 64, 80
COUNT_NAME = "test_worker_image_count"
STAGE_NAME = "test_worker_stage_offset"
FRAME_NAME = "test_worker_frame_0"


@pytest.fixture
def worker():
    """A worker attached to real shared memory, with the GUI never built."""
    frame = shared_memory.SharedMemory(
        create=True, size=YSIZE * XSIZE * 2, name=FRAME_NAME
    )
    count = shared_memory.ShareableList([0, float("nan"), 0.0], name=COUNT_NAME)
    stage = shared_memory.ShareableList([0.0] * NUM_SLOTS, name=STAGE_NAME)

    w = XYTrackingWorker(None, {
        "ysize": YSIZE, "xsize": XSIZE, "ring_size": 1, "dtype": np.uint16,
        "shm_names": [FRAME_NAME],
        "image_count_shm_name": COUNT_NAME,
        "stage_shm_name": STAGE_NAME,
        "micron_to_pix_ratio": 2.0,   # round number: 1 px = 2 um
        "tracking_algorithm": "proportional",
        "algorithm_params": {"proportional": {"kp": 1.0}},
        "deadband_px": 0.0,
        "max_step_um": 1000.0,
        "swap_axes": False,
        "invert_axis0": False,
        "invert_axis1": False,
    })
    w.initialize_shm()
    w._build_trackers()
    yield w

    # Close the worker's own mappings first: on Windows a name stays alive while
    # any handle is open, so unlinking underneath it would leak into the next test.
    w.close()
    frame.close()
    frame.unlink()
    for shl in (count, stage):
        shl.shm.close()
        shl.shm.unlink()


def place_blob(worker, cy, cx):
    """Pretend the centroid detector found the blob at (cy, cx)."""
    worker.sm_cy, worker.sm_cx = float(cy), float(cx)
    worker.blob_ok = True


def test_pixel_error_becomes_microns_on_the_right_axis(worker):
    """Blob 10 px above center: the stage must move by 10 px * 2 um/px on the
    axis the vertical direction maps to, with nothing on the other."""
    place_blob(worker, worker.cy - 10, worker.cx)
    worker.update_stage_offset(dt=0.05, frame_ts=1.0)

    assert float(worker.shared_stage_offset_xy[CMD_0]) == pytest.approx(20.0)
    assert float(worker.shared_stage_offset_xy[CMD_1]) == pytest.approx(0.0)


def test_swap_axes_routes_vertical_error_to_axis1(worker):
    """A stage rotated 90 deg vs the camera; same transform manual jog uses."""
    worker.swap_axes = True
    place_blob(worker, worker.cy - 10, worker.cx)
    worker.update_stage_offset(dt=0.05, frame_ts=1.0)

    assert float(worker.shared_stage_offset_xy[CMD_0]) == pytest.approx(0.0)
    assert float(worker.shared_stage_offset_xy[CMD_1]) == pytest.approx(20.0)


def test_invert_flips_a_physical_axis(worker):
    worker.invert_axis0 = True
    place_blob(worker, worker.cy - 10, worker.cx)
    worker.update_stage_offset(dt=0.05, frame_ts=1.0)

    assert float(worker.shared_stage_offset_xy[CMD_0]) == pytest.approx(-20.0)


def test_disabled_axis_is_not_driven(worker):
    worker.enable_axis0 = False
    place_blob(worker, worker.cy - 10, worker.cx - 10)
    worker.update_stage_offset(dt=0.05, frame_ts=1.0)

    assert float(worker.shared_stage_offset_xy[CMD_0]) == pytest.approx(0.0)
    assert float(worker.shared_stage_offset_xy[CMD_1]) != 0.0


def test_commands_accumulate_as_a_running_total(worker):
    """The output device applies the difference from what it has issued, so the
    worker's slot is a total, not a one-shot offset."""
    shl = worker.shared_stage_offset_xy
    for i in range(3):
        place_blob(worker, worker.cy - 5, worker.cx)
        worker.update_stage_offset(dt=0.05, frame_ts=1.0 + 0.1 * i)
        # Acknowledge the move, as the output device would, so the next frame's
        # unchanged error is a genuinely new correction rather than the pending one.
        shl[APPLIED_0] = float(shl[CMD_0])
        shl[APPLIED_1] = float(shl[CMD_1])
        shl[APPLIED_TS] = 1.0 + 0.1 * i + 0.01

    assert float(shl[CMD_0]) == pytest.approx(30.0)


def test_sub_micron_corrections_survive(worker):
    """A quarter-pixel error is worth 0.5 um and must reach the stage; the old
    integer command path truncated everything under a micron to zero."""
    place_blob(worker, worker.cy - 0.25, worker.cx)
    worker.update_stage_offset(dt=0.05, frame_ts=1.0)

    assert float(worker.shared_stage_offset_xy[CMD_0]) == pytest.approx(0.5)


def test_pending_motion_is_not_commanded_again(worker):
    """The error is unchanged on the next frame because the stage has not moved
    yet. Without the pending term the same correction would be issued twice."""
    place_blob(worker, worker.cy - 10, worker.cx)
    worker.update_stage_offset(dt=0.05, frame_ts=1.0)
    first = float(worker.shared_stage_offset_xy[CMD_0])

    worker.update_stage_offset(dt=0.05, frame_ts=1.05)
    assert float(worker.shared_stage_offset_xy[CMD_0]) == pytest.approx(first)


def test_applied_motion_frees_the_loop_to_correct_again(worker):
    """Once the output device reports the move as issued and the blob is still
    off center, the loop corrects the remaining error."""
    place_blob(worker, worker.cy - 10, worker.cx)
    worker.update_stage_offset(dt=0.05, frame_ts=1.0)

    shl = worker.shared_stage_offset_xy
    shl[APPLIED_0] = float(shl[CMD_0])       # output device applied it
    shl[APPLIED_1] = float(shl[CMD_1])
    shl[APPLIED_SEQ] = 1.0
    shl[APPLIED_TS] = 1.02                   # ...before the next frame

    place_blob(worker, worker.cy - 4, worker.cx)  # partially corrected
    worker.update_stage_offset(dt=0.05, frame_ts=1.05)
    assert float(shl[CMD_0]) == pytest.approx(20.0 + 8.0)


def test_latency_is_measured_from_capture_to_move(worker):
    place_blob(worker, worker.cy - 10, worker.cx)
    worker.update_stage_offset(dt=0.05, frame_ts=1.0)

    shl = worker.shared_stage_offset_xy
    shl[APPLIED_0] = float(shl[CMD_0])
    shl[APPLIED_TS] = worker._cmd_log[-1][0] + 0.001  # applied just after write

    place_blob(worker, worker.cy - 4, worker.cx)
    worker.update_stage_offset(dt=0.05, frame_ts=1.05)
    # Frame captured at 1.0, move issued shortly after the command was written.
    assert worker._latency_ms > 0.0
    assert np.isfinite(worker._latency_ms)


def test_deadband_holds_a_nearly_centered_blob_still(worker):
    worker.deadband_px = 5.0
    place_blob(worker, worker.cy - 2, worker.cx - 2)
    worker.update_stage_offset(dt=0.05, frame_ts=1.0)

    assert float(worker.shared_stage_offset_xy[CMD_0]) == pytest.approx(0.0)
    assert float(worker.shared_stage_offset_xy[CMD_1]) == pytest.approx(0.0)


def test_max_step_clamps_an_outlier(worker):
    worker.max_step_um = 15.0
    place_blob(worker, worker.cy - 40, worker.cx)
    worker.update_stage_offset(dt=0.05, frame_ts=1.0)

    assert float(worker.shared_stage_offset_xy[CMD_0]) == pytest.approx(15.0)


def test_jog_goes_through_the_same_command_total(worker):
    """Manual moves share the accounting, so the estimators are not surprised
    by a jog."""
    worker.jog_step_um = 25
    worker._jog_screen(vert=1)

    assert float(worker.shared_stage_offset_xy[CMD_0]) == pytest.approx(25.0)
    assert worker.cmd_total[0] == pytest.approx(25.0)


def test_frame_timing_prefers_the_camera_clock(worker):
    worker.shared_image_count[1] = 1000.0   # camera ms
    worker.shared_image_count[2] = 5.0      # host s
    worker._frame_timing()
    worker.shared_image_count[1] = 1040.0
    worker.shared_image_count[2] = 5.1      # host disagrees; camera wins

    frame_ts, dt = worker._frame_timing()
    assert worker._clock == "cam"
    assert dt == pytest.approx(0.040)
    assert frame_ts == pytest.approx(5.1)   # host clock, comparable with the stage


def test_frame_timing_falls_back_without_camera_metadata(worker):
    worker.shared_image_count[1] = float("nan")
    worker.shared_image_count[2] = 5.0
    worker._frame_timing()
    worker.shared_image_count[2] = 5.05

    _, dt = worker._frame_timing()
    assert worker._clock == "host"
    assert dt == pytest.approx(0.05)


def test_absurd_frame_interval_is_rejected(worker):
    """A stall or clock glitch must not make the controllers lurch."""
    worker.shared_image_count[1] = float("nan")
    worker.shared_image_count[2] = 5.0
    worker._frame_timing()
    worker.shared_image_count[2] = 95.0     # a 90 s gap

    _, dt = worker._frame_timing()
    assert 0.0 < dt <= 1.0


def test_switching_algorithm_keeps_the_command_total(worker):
    """Algorithms are interchangeable: switching must not jerk the stage by
    restarting the total or replaying old state."""
    place_blob(worker, worker.cy - 10, worker.cx)
    worker.update_stage_offset(dt=0.05, frame_ts=1.0)
    total = float(worker.shared_stage_offset_xy[CMD_0])

    worker.tracker = worker.trackers["kalman"]
    worker.tracking_algorithm = "kalman"
    worker._reset_control_state()

    assert float(worker.shared_stage_offset_xy[CMD_0]) == pytest.approx(total)
    assert worker._latency_ms is not None
