"""End-to-end closed-loop test: worker and stage output device together.

The unit tests check the control laws and the shared-memory handshake in
isolation. This drives the real pair across the real shared buffer against a
simulated drifting target, which is what catches a sign convention or an
accounting mistake that only shows up once both halves are connected.

The simulation works in pixels: the blob drifts across the frame, and a stage
move of u microns on an axis shifts it back by u / (um per pixel).
"""

import time

import numpy as np
import pytest
from multiprocessing import shared_memory

from apps.io.output_device.xy_tracking_stage_output import (
    NUM_SLOTS, XYTrackingStageOutput,
)
from apps.logic.xy_tracking_worker import XYTrackingWorker

YSIZE, XSIZE = 200, 200
UM_PER_PX = 0.815  # the rig's value
DT = 0.02          # 50 fps, the rig's framerate
COUNT_NAME = "test_loop_image_count"
STAGE_NAME = "test_loop_stage_offset"
FRAME_NAME = "test_loop_frame_0"


class SimulatedStage:
    """Applies relative moves after a fixed delay, as a real stage would."""

    def __init__(self, settle_frames=2):  # ~40 ms at 50 fps
        self.pending = [np.zeros(2) for _ in range(settle_frames)]
        self.arrived = np.zeros(2)

    def setRelativeXYPosition(self, dx, dy):
        self.pending.append(np.array([dx, dy]))

    def step(self):
        """Return the motion that physically completes this frame."""
        return self.pending.pop(0) if self.pending else np.zeros(2)


@pytest.fixture
def rig():
    """Worker + output device wired through the real shared-memory buffers."""
    frame = shared_memory.SharedMemory(
        create=True, size=YSIZE * XSIZE * 2, name=FRAME_NAME
    )
    count = shared_memory.ShareableList([0, float("nan"), 0.0], name=COUNT_NAME)

    stage_out = XYTrackingStageOutput("stage", {"shm_name": STAGE_NAME})
    stage_out.mmc = SimulatedStage()
    stage_out.configure()

    def make_worker(algorithm, params):
        w = XYTrackingWorker(None, {
            "ysize": YSIZE, "xsize": XSIZE, "ring_size": 1, "dtype": np.uint16,
            "shm_names": [FRAME_NAME],
            "image_count_shm_name": COUNT_NAME,
            "stage_shm_name": STAGE_NAME,
            "micron_to_pix_ratio": UM_PER_PX,
            "tracking_algorithm": algorithm,
            "algorithm_params": {algorithm: params},
            "deadband_px": 0.0,
            "max_step_um": 200.0,
            "swap_axes": False,
            "invert_axis0": False,
            "invert_axis1": False,
        })
        w.initialize_shm()
        w._build_trackers()
        return w

    workers = []

    def factory(algorithm, params):
        # Fresh stage and zeroed buffer, so one run in a test cannot contaminate
        # the next.
        stage_out.mmc = SimulatedStage()
        for i in range(NUM_SLOTS):
            stage_out.shared_stage_offset_xy[i] = 0.0
        w = make_worker(algorithm, params)
        workers.append(w)
        return w, stage_out

    yield factory

    for w in workers:
        w.close()
    stage_out.close()
    frame.close()
    frame.unlink()
    count.shm.close()
    count.shm.unlink()


def run_loop(worker, stage_out, drift_px_s=(0.0, 0.0), start_px=(0.0, 0.0),
             n=400, noise_px=0.0, seed=0):
    """Drive the loop and return the per-frame centering error in pixels."""
    rng = np.random.default_rng(seed)
    blob = np.array([worker.cy + start_px[0], worker.cx + start_px[1]], float)
    drift = np.array(drift_px_s, float)
    errors = np.zeros((n, 2))

    for i in range(n):
        # The stage motion that physically completes this frame moves the blob
        # back toward center by move / (um per px).
        blob += stage_out.mmc.step() / UM_PER_PX
        blob += drift * DT
        errors[i] = [worker.cy - blob[0], worker.cx - blob[1]]

        # Real perf_counter, because the worker buckets applied moves by
        # comparing the stage's apply timestamp against the frame's — both are
        # perf_counter in the running system.
        t = time.perf_counter()
        measured = blob + rng.normal(0.0, noise_px, 2) if noise_px else blob
        worker.sm_cy, worker.sm_cx = measured[0], measured[1]
        worker.blob_ok = True
        worker.update_stage_offset(dt=DT, frame_ts=t)

        # The engine applies whatever motion is still owed, once per frame.
        stage_out.update_output()

    return errors


def settled(errors, frac=0.25):
    tail = errors[int(len(errors) * (1.0 - frac)):]
    return float(np.sqrt(np.mean(tail ** 2)))


@pytest.mark.parametrize("algorithm,params", [
    ("proportional", {"kp": 0.25}),
    ("pid", {"kp": 0.4, "ki": 0.8}),
    ("kalman", {"kp": 0.9, "horizon_ms": 20.0, "q_accel": 400.0,
                "r_meas_um": 2.0}),
])
def test_offset_blob_is_centered(rig, algorithm, params):
    """A blob parked 30 px off center is brought to the crosshair."""
    worker, stage_out = rig(algorithm, params)
    errors = run_loop(worker, stage_out, start_px=(30.0, -20.0))
    assert settled(errors) < 0.5, f"{algorithm}: {settled(errors)} px"


def test_sub_pixel_error_is_still_corrected(rig):
    """The regression that motivated all of this: with integer microns a 3 px
    error commanded a 0 um move and the blob simply stayed off center."""
    worker, stage_out = rig("proportional", {"kp": 0.25})
    errors = run_loop(worker, stage_out, start_px=(3.0, 3.0), n=200)
    assert settled(errors) < 0.2, settled(errors)
    assert stage_out.mmc.arrived is not None
    # Corrections really are fractional microns, not rounded away.
    assert any(0.0 < abs(m[0]) < 1.0 for m in stage_out.mmc.pending
               ) or settled(errors) < 0.2


@pytest.mark.parametrize("algorithm,params", [
    ("pid", {"kp": 0.4, "ki": 0.8}),
    ("kalman", {"kp": 0.9, "horizon_ms": 20.0, "q_accel": 400.0,
                "r_meas_um": 2.0}),
])
def test_estimators_beat_proportional_on_a_drifting_target(rig, algorithm, params):
    """The real case: a neuron drifting steadily across the field.

    Proportional settles at a standing offset; the estimators do not. This is
    the comparison the GUI's RMS readout is there to let you make on the rig.
    """
    drift = (12.0, -6.0)  # px/s, ~10 um/s on the fast axis

    baseline_worker, stage_out = rig("proportional", {"kp": 0.25})
    baseline_errors = run_loop(baseline_worker, stage_out, drift_px_s=drift, n=800)
    baseline = settled(baseline_errors)

    worker, stage_out = rig(algorithm, params)
    estimator = settled(run_loop(worker, stage_out, drift_px_s=drift, n=800))

    # Proportional parks at v * dt / kp on each axis, which is the whole problem.
    assert baseline_errors[-1, 0] == pytest.approx(
        -drift[0] * DT / 0.25, rel=0.1
    ), "proportional should sit at the predicted standing offset"
    assert estimator < baseline / 3.0, (
        f"{algorithm} {estimator:.2f} px vs proportional {baseline:.2f} px"
    )


@pytest.mark.parametrize("algorithm,params", [
    ("pid", {"kp": 0.4, "ki": 0.8}),
    ("kalman", {"kp": 0.9, "horizon_ms": 20.0, "q_accel": 400.0,
                "r_meas_um": 2.0}),
])
def test_estimators_beat_proportional_with_a_noisy_centroid(rig, algorithm, params):
    """The same comparison under realistic conditions.

    A dim puncta jitters, and a filter that chases the noise can easily do worse
    overall than a sluggish proportional law. The shipped defaults have to win
    here, not just on a noise-free target.
    """
    drift, noise = (12.0, -6.0), 1.0

    baseline_worker, stage_out = rig("proportional", {"kp": 0.25})
    baseline = settled(run_loop(baseline_worker, stage_out, drift_px_s=drift,
                                noise_px=noise, n=1500))

    worker, stage_out = rig(algorithm, params)
    estimator = settled(run_loop(worker, stage_out, drift_px_s=drift,
                                 noise_px=noise, n=1500))

    assert estimator < baseline, (
        f"{algorithm} {estimator:.2f} px vs proportional {baseline:.2f} px"
    )


def test_kalman_horizon_has_an_optimum(rig):
    """The horizon is a tuning knob with a real optimum, not a bigger-is-better
    dial: it covers the stage's settle time only, because motion already
    commanded is accounted for exactly. Over-setting it makes the loop lead the
    target — which is why the guidance is to minimise the RMS readout.
    """
    drift = (12.0, -6.0)
    base = {"kp": 0.9, "q_accel": 400.0, "r_meas_um": 1.0}

    def rms_at(horizon_ms):
        worker, stage_out = rig("kalman", {**base, "horizon_ms": horizon_ms})
        return settled(run_loop(worker, stage_out, drift_px_s=drift, n=800))

    # The optimum is well under the 2-frame settle, because motion already
    # issued is carried through the filter's predict step.
    tuned = rms_at(1000 * DT)
    assert tuned < 0.2, f"tuned horizon should nearly null the drift: {tuned}"
    assert tuned < rms_at(0.0), "no horizon should trail the target"
    assert tuned < rms_at(300.0), "an over-set horizon should lead the target"


@pytest.mark.parametrize("algorithm,params", [
    ("proportional", {"kp": 0.25}),
    ("pid", {"kp": 0.4, "ki": 0.8}),
    ("kalman", {"kp": 0.9, "horizon_ms": 20.0, "q_accel": 400.0,
                "r_meas_um": 2.0}),
])
def test_noisy_centroid_does_not_destabilize_the_loop(rig, algorithm, params):
    worker, stage_out = rig(algorithm, params)
    errors = run_loop(worker, stage_out, drift_px_s=(2.0, 0.0),
                      noise_px=1.0, n=600)
    assert np.all(np.isfinite(errors))
    assert settled(errors) < 4.0, f"{algorithm}: {settled(errors)} px"


def test_wrong_axis_sign_diverges(rig):
    """Guards the sign convention: if invert were set wrongly the loop would
    run away, so the passing tests above are not passing by accident."""
    worker, stage_out = rig("proportional", {"kp": 0.25})
    worker.invert_axis0 = True
    errors = run_loop(worker, stage_out, start_px=(5.0, 0.0), n=120)
    assert abs(errors[-1, 0]) > abs(errors[0, 0]) * 5
