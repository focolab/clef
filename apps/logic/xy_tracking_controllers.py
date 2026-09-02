"""
Tracking control laws for closed-loop XY stage tracking.

Three interchangeable controllers that turn a centering error into a stage
command. Everything around them is shared by XYTrackingWorker — blob detection,
the SNR gate, the deadband, the pixel->micron conversion and axis swap/invert
upstream; the step clamp, per-axis enable and shared-memory write downstream —
so the controllers differ only in the XY position they ask for.

All quantities are in **stage-axis microns**: 2-vectors [axis0, axis1] holding
how far the stage must move to center the object. Because the worker applies the
same axis transform to the error that it applies to a manual jog, a commanded or
applied move of +u on an axis reduces the error on that axis by exactly u, which
is what lets the estimators reason about their own past commands.

update() takes the stage motion in flight, split three ways:

  u_pre     issued since the last frame and *before* this frame was captured, so
            it is already reflected in the measured error
  u_post    issued after this frame was captured, so it is not yet visible
  pending   written to shared memory but not yet issued to the stage at all

Proportional and PID subtract only ``pending`` — commanding an error twice
because the first correction has not been picked up yet is the mistake that
forces their gains down. The Kalman filter additionally carries ``u_pre``
through its predict step and ``u_post`` into its prediction horizon, so it also
accounts for motion the camera has not caught up with. That dead-time
compensation is the substantive difference between the three.
"""

import numpy as np


class BaseTracker:
    """Common interface. Parameters are plain attributes so the GUI sliders can
    write them live, the same way the worker's shared parameters work."""

    name = "base"

    # (attribute, slider label, min, max, decimals) — drives the GUI page.
    param_spec = ()

    def reset(self):
        """Drop all accumulated state (integrators, estimates)."""

    def update(self, err_um, dt, u_pre, u_post, pending):
        """Return the stage command for this frame, in stage-axis microns."""
        raise NotImplementedError

    def readout(self):
        """Short diagnostic string for the status bar, or '' if nothing to add."""
        return ""

    def describe(self):
        """Current parameter values, for session metadata."""
        return {attr: getattr(self, attr) for attr, *_ in self.param_spec}


class ProportionalTracker(BaseTracker):
    """Correct a fixed fraction of the current error each frame.

    Type-0 against a ramp: a target drifting at a constant velocity leaves a
    standing offset of roughly v * dt * (1/kp + dead-time frames). Kept as the
    baseline to compare the estimators against.
    """

    name = "proportional"
    param_spec = (("kp", "Kp (fraction of error)", 0.0, 2.0, 2),)

    def __init__(self, kp=0.25):
        self.kp = kp

    def update(self, err_um, dt, u_pre, u_post, pending):
        return self.kp * (err_um - pending)


class PIDTracker(BaseTracker):
    """Proportional-integral-derivative on the centering error.

    The integral term is what removes the standing offset the proportional law
    leaves against a steadily drifting target: it keeps accumulating until the
    error is actually zero. It is clamped to +/- i_limit_um so a lost blob or a
    saturated stage cannot wind it up into a large stored command.

    kd defaults to 0. The centroid is noisy enough at these frame rates that a
    raw derivative is mostly noise, so the derivative is low-pass filtered with
    time constant d_tau_s before use.
    """

    name = "pid"
    param_spec = (
        ("kp", "Kp (fraction of error)", 0.0, 2.0, 2),
        ("ki", "Ki (1/s)", 0.0, 10.0, 2),
        ("kd", "Kd (s)", 0.0, 1.0, 3),
        ("d_tau_s", "derivative filter (s)", 0.01, 1.0, 2),
        ("i_limit_um", "integral limit (um)", 0.0, 1000.0, 0),
    )

    def __init__(self, kp=0.4, ki=0.8, kd=0.0, d_tau_s=0.1, i_limit_um=100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.d_tau_s = d_tau_s
        self.i_limit_um = i_limit_um
        self.reset()

    def reset(self):
        self._i = np.zeros(2)
        self._d = np.zeros(2)
        self._prev_e = None

    def update(self, err_um, dt, u_pre, u_post, pending):
        e = err_um - pending

        if self._prev_e is not None and dt > 0.0:
            de = (e - self._prev_e) / dt
            # EMA toward the raw derivative; alpha from the filter time constant
            # so the smoothing is in seconds, not frames, and survives dt jitter.
            alpha = dt / (self.d_tau_s + dt) if self.d_tau_s > 0.0 else 1.0
            self._d += alpha * (de - self._d)
        self._prev_e = e

        if dt > 0.0:
            self._i = np.clip(
                self._i + self.ki * e * dt, -self.i_limit_um, self.i_limit_um
            )
        return self.kp * e + self._i + self.kd * self._d

    def readout(self):
        return f"I=({self._i[0]:+.1f}, {self._i[1]:+.1f}) um"


class KalmanTracker(BaseTracker):
    """Constant-velocity Kalman filter per axis, commanding where the target
    will be once the stage gets there.

    State is [error_um, velocity_um_per_s]: how far off center the object is,
    and how fast that is changing. Each axis runs its own 2-state filter — the
    axes are independent here, so one 2x2 filter each is cheaper and clearer
    than a coupled 4-state one.

    The velocity state does the job PID's integrator does, without having to
    wind up first. The horizon does the job neither can: it extrapolates the
    error to the moment the stage will actually have arrived, so the loop stops
    chasing a position the target has already left.

    The horizon covers only the lag that is *not* already accounted for: moves
    still waiting to be issued are subtracted exactly (pending), and moves
    already issued are carried through the predict step, so what is left is
    mainly how long the stage takes to physically arrive after being commanded.
    That makes it roughly the stage's settle time — a frame or two — not the
    whole capture-to-apply latency. Over-setting it makes the loop lead the
    target and is worse than leaving it at zero, so tune it by minimising the
    RMS error readout against a drifting target rather than by calculation.

    q_accel (assumed acceleration noise, um^2/s^3) and r_meas_um (centroid
    measurement noise, microns) act as a ratio: what matters is q_accel divided
    by r_meas_um squared. A high ratio tracks an agile target closely and passes
    centroid noise through; a low one is smooth and steady but lags a target
    that changes direction. If the blob is dim and the centroid jitters, raise
    r_meas_um — that is the single most effective knob for a noisy image.
    """

    name = "kalman"
    param_spec = (
        ("kp", "Kp (fraction of prediction)", 0.0, 1.5, 2),
        ("horizon_ms", "prediction horizon (ms)", 0.0, 500.0, 0),
        ("q_accel", "target agility q (um^2/s^3)", 1.0, 5000.0, 0),
        ("r_meas_um", "measurement noise (um)", 0.01, 20.0, 2),
    )

    def __init__(self, q_accel=400.0, r_meas_um=2.0, horizon_ms=20.0, kp=0.9):
        self.q_accel = q_accel
        self.r_meas_um = r_meas_um
        self.horizon_ms = horizon_ms
        self.kp = kp
        self.reset()

    def reset(self):
        self.x = np.zeros((2, 2))  # [axis, (error_um, velocity_um_per_s)]
        # Wide initial covariance: the first measurement should dominate.
        self.P = np.tile(np.diag([1e4, 1e4]).astype(float), (2, 1, 1))
        self._seeded = False

    def _predict(self, axis, dt, u):
        """Advance one axis by dt, minus the stage motion u that has landed."""
        e, v = self.x[axis]
        self.x[axis] = (e + v * dt - u, v)

        F = np.array([[1.0, dt], [0.0, 1.0]])
        # Continuous white-noise acceleration model.
        Q = self.q_accel * np.array(
            [[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]]
        )
        self.P[axis] = F @ self.P[axis] @ F.T + Q

    def _correct(self, axis, z):
        """Standard scalar-measurement update; H = [1, 0]."""
        P = self.P[axis]
        S = P[0, 0] + self.r_meas_um ** 2
        K = P[:, 0] / S
        self.x[axis] = self.x[axis] + K * (z - self.x[axis][0])
        self.P[axis] = P - np.outer(K, P[0, :])

    def update(self, err_um, dt, u_pre, u_post, pending):
        if not self._seeded:
            # Start on the first measurement rather than at zero, so the filter
            # does not command a spurious lunge on its first frame.
            self.x[:, 0] = err_um
            self._seeded = True

        horizon = self.horizon_ms / 1000.0
        cmd = np.zeros(2)
        for a in range(2):
            self._predict(a, dt, u_pre[a])
            self._correct(a, err_um[a])
            e, v = self.x[a]
            # Where the error will be once everything already in flight has
            # landed and the horizon has elapsed. Subtracting u_post and pending
            # here (rather than in the state) keeps the filter's own estimate
            # honest: those moves are corrected out on the next frame's predict,
            # when they show up in u_pre.
            cmd[a] = self.kp * (e + v * horizon - u_post[a] - pending[a])
        return cmd

    @property
    def velocity(self):
        return self.x[:, 1]

    def readout(self):
        v = self.velocity
        return f"v=({v[0]:+.1f}, {v[1]:+.1f}) um/s"


REGISTRY = {
    cls.name: cls for cls in (ProportionalTracker, PIDTracker, KalmanTracker)
}


def make_tracker(name, params=None):
    """Build a tracker by name, ignoring parameters it does not accept.

    Config blocks are hand-edited, so an unknown key is dropped with the rest of
    the block still applied rather than failing the whole session at startup.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown tracking algorithm '{name}'. Available: {sorted(REGISTRY)}"
        )
    cls = REGISTRY[name]
    known = {attr for attr, *_ in cls.param_spec}
    return cls(**{k: v for k, v in (params or {}).items() if k in known})
