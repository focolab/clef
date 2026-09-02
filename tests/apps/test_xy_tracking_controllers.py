"""Tests for the XY tracking control laws.

Each controller is run against a simulated plant that reproduces the parts of
the real loop that make centering hard: the stage is an incremental positioner,
commands take a few frames to be applied, and the centroid is noisy.

The plant, per frame:
    e <- e + v * dt - (motion applied this frame)
where e is the centering error in microns and v is the target's drift velocity.
A command issued on frame k is applied on frame k + delay, which is what
produces the pending/in-flight motion the controllers are told about.
"""

import numpy as np
import pytest

from apps.logic.xy_tracking_controllers import (
    REGISTRY,
    KalmanTracker,
    PIDTracker,
    ProportionalTracker,
    make_tracker,
)

DT = 0.02  # 50 fps, the rig's framerate


def run_plant(tracker, n=600, dt=DT, drift=(0.0, 0.0), e0=(0.0, 0.0),
              delay=1, noise=0.0, seed=0, dt_jitter=0.0):
    """Simulate the closed loop; return the per-frame true error, shape (n, 2).

    delay is in frames between a command being written and being applied, so
    delay=1 matches the real loop's one-frame command pickup.
    """
    rng = np.random.default_rng(seed)
    e = np.array(e0, dtype=float)
    v = np.array(drift, dtype=float)
    queue = [np.zeros(2) for _ in range(delay)]  # commands not yet applied
    history = np.zeros((n, 2))

    for i in range(n):
        step = dt * (1.0 + dt_jitter * rng.uniform(-1.0, 1.0)) if dt_jitter else dt

        # Apply the command that has finished waiting, then let the target move.
        applied = queue.pop(0)
        e = e + v * step - applied
        history[i] = e

        measured = e + rng.normal(0.0, noise, 2) if noise else e.copy()
        pending = np.sum(queue, axis=0) if queue else np.zeros(2)
        # The move landed before this frame was measured, so it is u_pre.
        cmd = tracker.update(measured, step, applied, np.zeros(2), pending)
        queue.append(cmd)

    return history


def settled(history, frac=0.25):
    """RMS error over the last `frac` of the run, per axis."""
    tail = history[int(len(history) * (1.0 - frac)):]
    return np.sqrt(np.mean(tail ** 2, axis=0))


def all_trackers():
    return [
        ProportionalTracker(kp=0.25),
        PIDTracker(kp=0.4, ki=0.8),
        KalmanTracker(kp=0.9, horizon_ms=1000 * DT, q_accel=400.0, r_meas_um=2.0),
    ]


@pytest.mark.parametrize("tracker", all_trackers(), ids=lambda t: t.name)
def test_step_disturbance_converges(tracker):
    """A one-off displacement is driven back to center by every algorithm."""
    history = run_plant(tracker, e0=(20.0, -15.0))
    assert np.all(settled(history) < 0.5), f"{tracker.name}: {settled(history)}"


def test_proportional_trails_a_drifting_target():
    """The failure mode the estimators exist to fix.

    Pure proportional control is type-0 against a ramp, so a steadily drifting
    target sits at a fixed offset instead of centered.
    """
    kp = 0.25
    drift = 20.0  # um/s
    history = run_plant(ProportionalTracker(kp=kp), drift=(drift, 0.0))
    offset = np.mean(history[-100:, 0])

    # Each frame the target moves v*dt and the controller claws back kp of the
    # error, so it balances at e_ss = v*dt/kp — here 20 * 0.02 / 0.25 = 1.6 um,
    # about 2 px at the rig's 0.815 um/px. There is no extra term for the command
    # delay because pending motion is compensated; without that it would be worse.
    assert offset == pytest.approx(drift * DT / kp, rel=0.05)
    assert abs(offset) > 1.0, "expected a substantial standing offset"


@pytest.mark.parametrize(
    "tracker",
    [PIDTracker(kp=0.4, ki=0.8),
     KalmanTracker(kp=0.9, horizon_ms=1000 * DT, q_accel=400.0, r_meas_um=2.0)],
    ids=lambda t: t.name,
)
def test_estimators_remove_the_drift_offset(tracker):
    """PID's integral and the Kalman velocity state both drive the standing
    offset to zero, where proportional parks at v*dt/kp."""
    history = run_plant(tracker, drift=(20.0, -8.0), n=1200)
    assert np.all(settled(history, frac=0.2) < 0.5), settled(history, frac=0.2)


@pytest.mark.parametrize("tracker", all_trackers(), ids=lambda t: t.name)
def test_measurement_noise_is_not_amplified(tracker):
    """A stationary target with a noisy centroid must not be chased around."""
    noise = 1.0
    history = run_plant(tracker, noise=noise, n=1000)
    # Staying within a few times the measurement noise means the loop is
    # filtering rather than amplifying it.
    assert np.all(settled(history) < 4.0 * noise), settled(history)


@pytest.mark.parametrize("tracker", all_trackers(), ids=lambda t: t.name)
def test_stable_under_frame_interval_jitter(tracker):
    """Frame intervals wander in a real acquisition; timing comes from the
    camera precisely so the controllers can cope with that."""
    history = run_plant(
        tracker, drift=(10.0, 0.0), dt_jitter=0.4, noise=0.5, n=1000
    )
    assert np.all(np.isfinite(history))
    assert np.all(settled(history) < 6.0), settled(history)


def test_kalman_estimates_the_drift_velocity():
    """The velocity state is the thing doing the work; check it is right."""
    tracker = KalmanTracker(kp=0.9, horizon_ms=1000 * DT)
    run_plant(tracker, drift=(20.0, -8.0), n=1200)
    assert tracker.velocity[0] == pytest.approx(20.0, abs=3.0)
    assert tracker.velocity[1] == pytest.approx(-8.0, abs=3.0)


def test_pid_integral_is_clamped():
    """A blob that never centers (stage disabled) must not wind the integral up
    into a huge stored command."""
    tracker = PIDTracker(kp=0.4, ki=5.0, i_limit_um=10.0)
    err = np.array([100.0, 100.0])
    zero = np.zeros(2)
    for _ in range(500):
        tracker.update(err, DT, zero, zero, zero)  # nothing is ever applied
    assert np.all(np.abs(tracker._i) <= 10.0 + 1e-9)


@pytest.mark.parametrize("tracker", all_trackers(), ids=lambda t: t.name)
def test_pending_motion_is_not_commanded_twice(tracker):
    """With the full correction already written but not yet applied, there is
    nothing left to ask for."""
    err = np.array([10.0, 10.0])
    cmd = tracker.update(err, DT, np.zeros(2), np.zeros(2), err.copy())
    assert np.all(np.abs(cmd) < 1e-6), cmd


def test_reset_clears_accumulated_state():
    for tracker in all_trackers():
        run_plant(tracker, drift=(20.0, 20.0), n=200)
        tracker.reset()
        cmd = tracker.update(np.zeros(2), DT, np.zeros(2), np.zeros(2), np.zeros(2))
        assert np.all(np.abs(cmd) < 1e-6), f"{tracker.name} kept state: {cmd}"


def test_make_tracker_ignores_unknown_keys():
    """Config blocks are hand-edited; a stray key should not fail a session."""
    tracker = make_tracker("pid", {"kp": 0.7, "not_a_parameter": 1})
    assert tracker.kp == 0.7

    with pytest.raises(ValueError, match="Unknown tracking algorithm"):
        make_tracker("nope")


def test_registry_covers_the_three_algorithms():
    assert set(REGISTRY) == {"proportional", "pid", "kalman"}
    for name, cls in REGISTRY.items():
        assert cls.name == name
        assert cls.param_spec, f"{name} has no tunable parameters for the GUI"
