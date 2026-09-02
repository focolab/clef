"""
XYTrackingWorker for CLEF.

GUI subprocess for real-time closed-loop XY stage tracking of a single small
bright fluorescent blob, with live fluorescence extraction. Reads camera
frames from shared memory, computes the blob centroid with numba, and writes
stage-correction offsets into the shared_stage_offset_xy buffer that
XYTrackingStageOutput applies each sample.

Tracking control law (all parameters live-adjustable):
  * intensity-weighted sub-pixel centroid of the bright blob (numba);
  * optional exponential smoothing of the centroid to reject pixel noise;
  * a per-axis deadband so a nearly-centered blob triggers no motion;
  * the pixel error converted to stage-axis microns, then handed to the
    selected control algorithm (proportional / PID / Kalman — see
    xy_tracking_controllers.py), which is chosen live from the GUI;
  * the resulting command clamped to a maximum step to reject outliers;
  * independent enable + invert toggles per stage axis.

Stage commands are accumulated as a running total in shared memory and the
output device reports back what it has actually issued, so the worker can see
both what is still pending and what has already moved. The estimators use that
to avoid commanding the same error twice while a correction is in flight, which
is the main thing that limits how hard the loop can be driven.

Fluorescence: the mean intensity inside a fixed-radius circular ROI centered
on the blob is extracted every frame (numba), plotted live, and saved to a
CSV trace on close.

The window has an always-visible image (raw or processed), a live
fluorescence plot, and two tabs:
  * "Acquisition"           - clean run interface: tracking + processed view.
  * "Calibration & Testing" - manual jog D-pad, guided um/pixel calibration,
                              and sliders for every tuning parameter.

Stage commands are written into shared memory as a running per-axis total in
microns; the output device moves by whatever difference it has not yet issued
via setRelativeXYPosition, and reports back what it applied and when. See
xy_tracking_stage_output.py for the slot layout.
"""

import csv
import json
import logging
import os
import time
import warnings
import numpy as np
from collections import deque
from datetime import datetime
from multiprocessing import Process, shared_memory

from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

from apps.logic.xy_tracking_controllers import REGISTRY, make_tracker
from utils import tracking_numba
from utils.style.demo_stylization import DemoStyle

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

PLOT_TSIZE = 500  # rolling fluorescence plot width, in frames


def _iso(ts):
    """Unix timestamp -> human-readable local ISO string (seconds precision)."""
    return datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")


class XYTrackingWorker(Process):

    def __init__(self, child_conn, vis_args):
        super().__init__()
        self.child_conn = child_conn
        self.vis_args = vis_args

        self.ysize = vis_args.get("ysize")
        self.xsize = vis_args.get("xsize")
        self.ring_size = max(1, vis_args.get("ring_size", 1))
        self.dtype = vis_args.get("dtype", np.uint16)

        # Tracking parameters (all live-adjustable via sliders)
        self.micron_to_pix_ratio = vis_args.get("micron_to_pix_ratio", 100.0 / 74.0)
        self.threshold_frac = vis_args.get("threshold_frac", 0.5)
        self.deadband_px = vis_args.get("deadband_px", 2.0)
        self.centroid_smoothing = vis_args.get("centroid_smoothing", 0.0)
        self.max_step_um = vis_args.get("max_step_um", 200.0)
        # Minimum peak SNR to accept a puncta; below this the frame has no real
        # blob and tracking holds still instead of chasing noise.
        self.min_puncta_snr = vis_args.get("min_puncta_snr", 6.0)

        # Control algorithm. Every algorithm sees the same error and writes the
        # same command path, so they can be compared by switching live.
        self.tracking_algorithm = vis_args.get("tracking_algorithm", "kalman")
        self.algorithm_params = vis_args.get("algorithm_params", {})
        self.trackers = {}   # built after the process starts, in initialize_display
        self.tracker = None
        # How long the blob may be missing before the estimator's state is too
        # stale to trust and is thrown away.
        self.reset_after_lost_frames = vis_args.get("reset_after_lost_frames", 25)

        # Per-axis control
        self.enable_axis0 = vis_args.get("enable_axis0", True)
        self.enable_axis1 = vis_args.get("enable_axis1", True)
        self.invert_axis0 = vis_args.get("invert_axis0", False)
        self.invert_axis1 = vis_args.get("invert_axis1", False)
        # Stage axes rotated 90 deg vs the camera: route vertical image error to
        # stage axis1 and horizontal to axis0 (instead of the default 1:1). Jog
        # and tracking share this transform, so an intuitive D-pad guarantees
        # correct tracking signs.
        self.swap_axes = vis_args.get("swap_axes", False)

        # Fluorescence ROI
        self.roi_radius_px = vis_args.get("roi_radius_px", 30)

        # Testing / calibration parameters
        self.jog_step_um = vis_args.get("jog_step_um", 50)
        self.cal_step_um = vis_args.get("cal_step_um", 100)
        self.cal_settle_frames = max(1, vis_args.get("cal_settle_frames", 8))
        self.ratio_increment = vis_args.get("ratio_increment", 0.02)

        # Saving
        self.save_dir = vis_args.get("save_dir", "./output")
        self.rec_id = vis_args.get("rec_id", "xy_tracking")

        # Frame center = tracking target (row, col)
        self.cy = self.ysize // 2
        self.cx = self.xsize // 2

        # Runtime state
        self.image_count = 0
        self.first_img_flag = True
        self.raw_cy = float(self.cy)
        self.raw_cx = float(self.cx)
        self.sm_cy = float(self.cy)
        self.sm_cx = float(self.cx)
        self.blob_ok = False
        self.blob_snr = 0.0
        self._last_track_log = 0.0  # throttle for the tracking-diagnostic log

        # Control-loop bookkeeping (all in stage-axis microns / seconds)
        self.cmd_total = np.zeros(2)      # cumulative motion we have asked for
        self._prev_applied = np.zeros(2)  # applied total as of the previous frame
        self._prev_cam_s = None
        self._prev_host_ts = None
        self._nominal_dt = 0.02           # fallback frame period, refined live
        self._dt_s = 0.02
        self._clock = "host"
        self._lost_frames = 0
        self._have_frame_ts = False
        # Frame capture -> stage move issued. A diagnostic for how much dead
        # time the loop is carrying; it is an upper bound on, not the value for,
        # the Kalman horizon, since commanded motion is already accounted for.
        self._latency_ms = float("nan")
        self._cmd_log = deque(maxlen=64)  # (write_time, frame_ts) pairs
        # ~5 s of history at 50 fps: long enough to average out centroid noise,
        # short enough to respond when you switch algorithm.
        self._err_hist = deque(maxlen=250)  # |error| px, for the RMS readout
        self._last_cmd = np.zeros(2)

        # Fluorescence trace: rolling display buffer + full per-frame record
        self.recent_f = np.full(PLOT_TSIZE, np.nan)
        self.rec_rows = []

        # Sub-acquisitions: recording epochs triggered live during the session
        self.recording = False
        self.open_epoch = None
        self.epochs = []

        # Calibration state machine: "idle" or "settling"
        self.cal_state = "idle"
        self.cal_start_count = 0
        self.cal_centroid_before = (0.0, 0.0)

        # SHM references (attached in initialize_shm)
        self.shared_frame_memory_list = []
        self.img_list = []
        self.shared_image_count = None
        self.shared_stage_offset_xy = None

    # ------------------------------------------------------------------ SHM

    def initialize_shm(self):
        """Attach to the camera frame ring buffer, image count, stage offset."""
        shm_names = self.vis_args.get("shm_names", [])
        for z in range(self.ring_size):
            name = shm_names[z] if z < len(shm_names) else f"shared_frame_memory_{z}"
            shm = shared_memory.SharedMemory(name=name)
            self.shared_frame_memory_list.append(shm)
            self.img_list.append(
                np.ndarray((self.ysize, self.xsize), dtype=self.dtype, buffer=shm.buf)
            )

        image_count_name = self.vis_args.get(
            "image_count_shm_name", "shared_image_count"
        )
        self.shared_image_count = shared_memory.ShareableList(name=image_count_name)

        # Slot 0 is the image count; slots 1-2 carry the camera and host
        # timestamps of that frame. An older layout has only the count, in which
        # case we fall back to timing frames as they arrive here.
        self._have_frame_ts = len(self.shared_image_count) >= 3

        stage_shm_name = self.vis_args.get("stage_shm_name", "shared_stage_offset_xy")
        self.shared_stage_offset_xy = shared_memory.ShareableList(name=stage_shm_name)
        # Start from whatever the output device has already accounted for, so
        # the pending/applied difference is zero before the first command.
        self.cmd_total = np.array(
            [float(self.shared_stage_offset_xy[0]),
             float(self.shared_stage_offset_xy[1])]
        )
        self._prev_applied = np.array(
            [float(self.shared_stage_offset_xy[2]),
             float(self.shared_stage_offset_xy[3])]
        )

    def current_frame(self):
        """Return the most-recently-written ring-buffer frame."""
        buf = (self.image_count - 1) % self.ring_size
        return self.img_list[buf]

    def _command(self, axis0, axis1, frame_ts=None):
        """Add a move to the running command total in shared memory.

        The output device applies the difference between this total and what it
        has already issued, so a command written while a previous move is still
        in flight is carried rather than overwritten. Jog, calibration and
        tracking all go through here, which is what keeps the estimators from
        being surprised by a manual move.
        """
        self.cmd_total += (axis0, axis1)
        self.shared_stage_offset_xy[0] = float(self.cmd_total[0])
        self.shared_stage_offset_xy[1] = float(self.cmd_total[1])
        if frame_ts is not None:
            self._cmd_log.append((time.perf_counter(), frame_ts))

    def _frame_timing(self):
        """Return (frame_ts, dt) in seconds for the frame just published.

        dt comes from the camera's own acquisition clock when it is available,
        since that is immune to jitter in the host loop; otherwise from the host
        time the frame was stored. frame_ts is always a host perf_counter value
        so it is comparable with the stage's applied timestamp.
        """
        cam_s = None
        if self._have_frame_ts:
            cam_ms = float(self.shared_image_count[1])
            frame_ts = float(self.shared_image_count[2])
            if np.isfinite(cam_ms):
                cam_s = cam_ms / 1000.0
        else:
            frame_ts = time.perf_counter()
        if not np.isfinite(frame_ts) or frame_ts <= 0.0:
            frame_ts = time.perf_counter()

        if cam_s is not None:
            self._clock = "cam"
            dt = cam_s - self._prev_cam_s if self._prev_cam_s is not None else np.nan
            self._prev_cam_s = cam_s
        else:
            self._clock = "host"
            dt = (
                frame_ts - self._prev_host_ts
                if self._prev_host_ts is not None else np.nan
            )
        self._prev_host_ts = frame_ts

        # Reject a nonsensical interval (first frame, a clock glitch, a long
        # stall) in favour of the running estimate, so one bad dt cannot make
        # the controllers lurch.
        if np.isfinite(dt) and 0.0 < dt <= 1.0:
            self._nominal_dt += 0.1 * (dt - self._nominal_dt)
        else:
            dt = self._nominal_dt
        self._dt_s = dt
        return frame_ts, dt

    def _stage_feedback(self, frame_ts):
        """Return (u_pre, u_post, pending), all in stage-axis microns.

        Motion issued since the previous frame is split on whether it landed
        before or after this frame was captured: motion applied before capture
        is already reflected in the measured error, motion applied after it is
        not yet visible. pending is written but not yet issued at all.

        The split uses the timestamp of the most recent move, which is exact at
        the usual one move per frame and approximate if several were issued.
        """
        shl = self.shared_stage_offset_xy
        applied = np.array([float(shl[2]), float(shl[3])])
        applied_ts = float(shl[5])
        delta = applied - self._prev_applied
        self._prev_applied = applied
        pending = self.cmd_total - applied

        # Attribute the move to the frame whose command produced it, so the
        # reported latency is capture -> stage move, not just apply time.
        if np.any(delta):
            for write_ts, cmd_frame_ts in reversed(self._cmd_log):
                if write_ts <= applied_ts:
                    self._latency_ms = (applied_ts - cmd_frame_ts) * 1000.0
                    break

        if applied_ts <= frame_ts:
            return delta, np.zeros(2), pending
        return np.zeros(2), delta, pending

    # ------------------------------------------------------------------ GUI

    def initialize_display(self):
        self.initialize_shm()
        self._build_trackers()

        self.app = pg.Qt.mkQApp(name="XYTrackingWorker")
        DemoStyle.load_qss(self.app)

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("XY Stage Tracking")
        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setSpacing(8)

        main_layout.addWidget(self._build_image_panel(), stretch=3)
        main_layout.addWidget(self._build_fluorescence_plot(), stretch=1)
        main_layout.addWidget(self._build_status_bar())

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_acquisition_tab(), "Acquisition")
        self.tabs.addTab(self._build_calibration_tab(), "Calibration && Testing")
        main_layout.addWidget(self.tabs)

        self.window.resize(860, 1000)
        self.window.show()
        self._refresh_readouts()

    def _build_trackers(self):
        """Instantiate every algorithm up front so switching between them is
        instant and each keeps its own tuning."""
        self.trackers = {
            name: make_tracker(name, self.algorithm_params.get(name, {}))
            for name in REGISTRY
        }
        if self.tracking_algorithm not in self.trackers:
            logger.warning(
                f"Unknown tracking_algorithm '{self.tracking_algorithm}', "
                f"falling back to 'proportional'"
            )
            self.tracking_algorithm = "proportional"
        self.tracker = self.trackers[self.tracking_algorithm]

    def _build_image_panel(self):
        self.graphics_layout_widget = pg.GraphicsLayoutWidget()
        self.vb = pg.ViewBox(lockAspect=True, enableMouse=True, enableMenu=True)
        self.vb.setBorder({"color": DemoStyle.COLOR_NEUTRAL, "width": 2})
        self.vb.invertY()  # row 0 at top

        # Row-major so array (row, col) maps to scene (x=col, y=row).
        self.ii = pg.ImageItem()
        self.ii.setOpts(axisOrder="row-major")
        self.ii.setImage(self.current_frame())
        self.vb.addItem(self.ii)

        # Fixed crosshair at the tracking target (frame center).
        self.target_marker = pg.ScatterPlotItem(
            x=[self.cx], y=[self.cy], pen=None,
            brush=pg.mkBrush(DemoStyle.COLOR_NEUTRAL), size=14, symbol="+",
        )
        self.vb.addItem(self.target_marker)

        # Fluorescence ROI outline (updated each frame around the centroid).
        self.roi_circle = pg.PlotCurveItem(
            pen=pg.mkPen(DemoStyle.COLOR_SUCCESS, width=2)
        )
        self._unit_circle = np.linspace(0, 2 * np.pi, 48)
        self.vb.addItem(self.roi_circle)

        # Live blob centroid.
        self.centroid_marker = pg.ScatterPlotItem(
            x=[], y=[], pen=pg.mkPen(DemoStyle.COLOR_DANGER, width=2),
            brush=None, size=16, symbol="o",
        )
        self.vb.addItem(self.centroid_marker)

        self.graphics_layout_widget.addItem(self.vb, row=0, col=0)

        self.lut_histo = pg.HistogramLUTItem(
            image=self.ii, fillHistogram=False,
            orientation="vertical", levelMode="mono",
        )
        self.graphics_layout_widget.addItem(self.lut_histo, row=0, col=1)
        return self.graphics_layout_widget

    def _build_fluorescence_plot(self):
        self.f_plot = pg.PlotWidget()
        self.f_plot.setTitle("blob fluorescence (ROI mean)")
        self.f_plot.setLabel("bottom", "recent frames")
        self.f_plot.setLabel("left", "mean intensity")
        self.f_plot.setXRange(0, PLOT_TSIZE)
        self.f_curve = self.f_plot.plot(
            self.recent_f, pen=pg.mkPen(DemoStyle.COLOR_SUCCESS, width=2),
            connect="finite",
        )
        return self.f_plot

    def _build_status_bar(self):
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet(DemoStyle.INFO_BOX_STYLE)
        return self.status_label

    def _build_acquisition_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.show_processed_button = DemoStyle.make_action_button(
            "Show Processed (blob mask)", QtWidgets
        )
        self.show_processed_button.setCheckable(True)
        # Re-auto-level the next frame on toggle: raw (uint16) and mask (0/255)
        # views span very different ranges, so the LUT must reset.
        self.show_processed_button.toggled.connect(
            lambda _: setattr(self, "first_img_flag", True)
        )
        layout.addWidget(self.show_processed_button)

        # Auto-adjust contrast each frame (re-levels the LUT from the current
        # image). Off by default so the manual histogram sliders stay in control
        # and we skip the per-frame min/max scan.
        self.auto_contrast_button = DemoStyle.make_action_button(
            "Auto-adjust Contrast", QtWidgets, color=DemoStyle.COLOR_NEUTRAL
        )
        self.auto_contrast_button.setCheckable(True)
        self.auto_contrast_button.toggled.connect(
            lambda on: self.auto_contrast_button.setStyleSheet(
                "background-color: "
                f"{DemoStyle.COLOR_SUCCESS if on else DemoStyle.COLOR_NEUTRAL}"
            )
        )
        layout.addWidget(self.auto_contrast_button)

        self.enable_tracking_button = DemoStyle.make_action_button(
            "Enable Stage Tracking", QtWidgets, color=DemoStyle.COLOR_NEUTRAL
        )
        self.enable_tracking_button.setCheckable(True)
        self.enable_tracking_button.toggled.connect(self._tracking_toggled)
        layout.addWidget(self.enable_tracking_button)

        layout.addWidget(self._build_recording_group())
        layout.addStretch()
        return tab

    def _build_recording_group(self):
        group = QtWidgets.QGroupBox("Experiment Recording (sub-acquisitions)")
        group.setStyleSheet(DemoStyle.GROUP_BOX_STYLE)
        layout = QtWidgets.QVBoxLayout(group)

        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("name:"))
        self.rec_name_edit = QtWidgets.QLineEdit()
        self.rec_name_edit.setPlaceholderText("optional sub-acquisition label")
        name_row.addWidget(self.rec_name_edit)
        layout.addLayout(name_row)

        self.record_button = DemoStyle.make_action_button(
            "Start Recording", QtWidgets, color=DemoStyle.COLOR_SUCCESS
        )
        self.record_button.setCheckable(True)
        self.record_button.toggled.connect(self._toggle_recording)
        layout.addWidget(self.record_button)

        self.record_status = DemoStyle.make_info_box(
            "No sub-acquisitions recorded yet.", QtWidgets
        )
        layout.addWidget(self.record_status)
        return group

    def _build_calibration_tab(self):
        tab = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(tab)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(inner)
        layout.setSpacing(8)

        instructions = DemoStyle.make_info_box(
            "1. Disable tracking on the Acquisition tab.\n"
            "2. Bring the blob near the center crosshair, jog the stage with "
            "the D-pad, and confirm which way it moves. Enable/invert each axis "
            "so tracking drives the blob toward center.\n"
            "3. Set a calibration step (um) and click Run Calibration; um/pixel "
            "is measured from how far the blob moved, then the stage returns.\n"
            "4. Pick a tracking algorithm and tune it for smooth, artifact-free "
            "centering. Compare algorithms on the RMS error in the status bar.",
            QtWidgets, style=DemoStyle.INSTRUCTIONS_BOX_STYLE,
        )
        layout.addWidget(instructions)

        layout.addWidget(self._build_jog_group())
        layout.addWidget(self._build_calibration_group())
        layout.addWidget(self._build_tuning_group())
        layout.addWidget(self._build_algorithm_group())
        layout.addStretch()

        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return tab

    def _build_jog_group(self):
        group = QtWidgets.QGroupBox("Manual Stage Jog & Per-Axis Control")
        group.setStyleSheet(DemoStyle.GROUP_BOX_STYLE)
        grid = QtWidgets.QGridLayout(group)

        grid.addWidget(QtWidgets.QLabel("jog step (um):"), 0, 0)
        self.jog_step_spinbox = QtWidgets.QSpinBox()
        self.jog_step_spinbox.setRange(1, 10000)  # cap to avoid fat-finger runaway
        self.jog_step_spinbox.setValue(int(self.jog_step_um))
        self.jog_step_spinbox.valueChanged.connect(
            lambda v: setattr(self, "jog_step_um", v)
        )
        grid.addWidget(self.jog_step_spinbox, 0, 1)

        # D-pad in screen directions. The swap/invert transform below maps these
        # to physical stage axes; adjust it until the buttons feel intuitive.
        up = DemoStyle.make_action_button("Up", QtWidgets)
        down = DemoStyle.make_action_button("Down", QtWidgets)
        left = DemoStyle.make_action_button("Left", QtWidgets)
        right = DemoStyle.make_action_button("Right", QtWidgets)
        # Jog convention is intentionally inverted from tracking: pressing a
        # direction pans the stage that way, so the sample appears to move the
        # opposite way (standard microscopy feel). Do NOT re-align these with
        # _screen_to_axes — tracking depends on that transform and is correct.
        up.clicked.connect(lambda: self._jog_screen(vert=+1))
        down.clicked.connect(lambda: self._jog_screen(vert=-1))
        left.clicked.connect(lambda: self._jog_screen(horiz=+1))
        right.clicked.connect(lambda: self._jog_screen(horiz=-1))
        grid.addWidget(up, 1, 1)
        grid.addWidget(left, 2, 0)
        grid.addWidget(right, 2, 2)
        grid.addWidget(down, 3, 1)

        # Independent enable + invert per axis.
        self.enable0_checkbox = QtWidgets.QCheckBox("Track axis0")
        self.enable0_checkbox.setChecked(self.enable_axis0)
        self.enable0_checkbox.toggled.connect(
            lambda v: setattr(self, "enable_axis0", v)
        )
        self.enable1_checkbox = QtWidgets.QCheckBox("Track axis1")
        self.enable1_checkbox.setChecked(self.enable_axis1)
        self.enable1_checkbox.toggled.connect(
            lambda v: setattr(self, "enable_axis1", v)
        )
        self.invert0_checkbox = QtWidgets.QCheckBox("Invert axis0")
        self.invert0_checkbox.setChecked(self.invert_axis0)
        self.invert0_checkbox.toggled.connect(
            lambda v: setattr(self, "invert_axis0", v)
        )
        self.invert1_checkbox = QtWidgets.QCheckBox("Invert axis1")
        self.invert1_checkbox.setChecked(self.invert_axis1)
        self.invert1_checkbox.toggled.connect(
            lambda v: setattr(self, "invert_axis1", v)
        )
        # Swap which physical axis each screen direction drives (90 deg
        # stage/camera rotation). Applies to both jog and tracking.
        self.swap_checkbox = QtWidgets.QCheckBox("Swap axes (90 deg rotation)")
        self.swap_checkbox.setChecked(self.swap_axes)
        self.swap_checkbox.toggled.connect(
            lambda v: setattr(self, "swap_axes", v)
        )

        grid.addWidget(self.enable0_checkbox, 4, 0)
        grid.addWidget(self.invert0_checkbox, 4, 1)
        grid.addWidget(self.enable1_checkbox, 5, 0)
        grid.addWidget(self.invert1_checkbox, 5, 1)
        grid.addWidget(self.swap_checkbox, 6, 0, 1, 2)
        return group

    def _build_calibration_group(self):
        group = QtWidgets.QGroupBox("Calibration (microns per pixel)")
        group.setStyleSheet(DemoStyle.GROUP_BOX_STYLE)
        grid = QtWidgets.QGridLayout(group)

        grid.addWidget(QtWidgets.QLabel("calibration step (um):"), 0, 0)
        self.cal_step_spinbox = QtWidgets.QSpinBox()
        self.cal_step_spinbox.setRange(1, 10000)  # cap to avoid fat-finger runaway
        self.cal_step_spinbox.setValue(int(self.cal_step_um))
        self.cal_step_spinbox.valueChanged.connect(
            lambda v: setattr(self, "cal_step_um", v)
        )
        grid.addWidget(self.cal_step_spinbox, 0, 1)

        grid.addWidget(QtWidgets.QLabel("settle frames:"), 0, 2)
        self.settle_spinbox = QtWidgets.QSpinBox()
        self.settle_spinbox.setRange(1, 1000)
        self.settle_spinbox.setValue(int(self.cal_settle_frames))
        self.settle_spinbox.valueChanged.connect(
            lambda v: setattr(self, "cal_settle_frames", v)
        )
        grid.addWidget(self.settle_spinbox, 0, 3)

        self.run_cal_button = DemoStyle.make_action_button(
            "Run Calibration", QtWidgets, color=DemoStyle.COLOR_SUCCESS
        )
        self.run_cal_button.clicked.connect(self._start_calibration)
        grid.addWidget(self.run_cal_button, 1, 0, 1, 2)

        grid.addWidget(QtWidgets.QLabel("um / pixel:"), 2, 0)
        self.ratio_edit = QtWidgets.QLineEdit(f"{self.micron_to_pix_ratio:.5f}")
        self.ratio_edit.returnPressed.connect(self._ratio_edited)
        grid.addWidget(self.ratio_edit, 2, 1)

        minus = DemoStyle.make_action_button("-", QtWidgets)
        plus = DemoStyle.make_action_button("+", QtWidgets)
        minus.clicked.connect(lambda: self._nudge_ratio(-1))
        plus.clicked.connect(lambda: self._nudge_ratio(+1))
        self.increment_spinbox = QtWidgets.QDoubleSpinBox()
        self.increment_spinbox.setDecimals(4)
        self.increment_spinbox.setRange(0.0001, 100.0)
        self.increment_spinbox.setSingleStep(0.001)
        self.increment_spinbox.setValue(self.ratio_increment)
        self.increment_spinbox.valueChanged.connect(
            lambda v: setattr(self, "ratio_increment", v)
        )
        grid.addWidget(minus, 2, 2)
        grid.addWidget(self.increment_spinbox, 2, 3)
        grid.addWidget(plus, 2, 4)

        self.cal_readout = DemoStyle.make_info_box("", QtWidgets)
        grid.addWidget(self.cal_readout, 3, 0, 1, 5)
        return group

    def _build_tuning_group(self):
        group = QtWidgets.QGroupBox("Tracking & Fluorescence Parameters")
        group.setStyleSheet(DemoStyle.GROUP_BOX_STYLE)
        grid = QtWidgets.QGridLayout(group)
        row = 0
        self._add_slider(grid, row, "blob threshold frac", 0.0, 1.0, 2,
                         self.threshold_frac, "threshold_frac"); row += 1
        self._add_slider(grid, row, "centroid smoothing", 0.0, 0.95, 2,
                         self.centroid_smoothing, "centroid_smoothing"); row += 1
        self._add_slider(grid, row, "deadband (px)", 0.0, 50.0, 1,
                         self.deadband_px, "deadband_px"); row += 1
        self._add_slider(grid, row, "min puncta SNR", 0.0, 50.0, 1,
                         self.min_puncta_snr, "min_puncta_snr"); row += 1
        self._add_slider(grid, row, "max step (um)", 1.0, 2000.0, 0,
                         self.max_step_um, "max_step_um"); row += 1
        self._add_slider(grid, row, "ROI radius (px)", 1.0, 300.0, 0,
                         float(self.roi_radius_px), "roi_radius_px"); row += 1
        return group

    def _build_algorithm_group(self):
        """Algorithm selector over a stack of per-algorithm parameter pages.

        The algorithms are interchangeable — same error in, same command path
        out — so switching is safe mid-run; only the accumulated estimator state
        is dropped, so the new algorithm starts from the current frame.
        """
        group = QtWidgets.QGroupBox("Tracking Algorithm")
        group.setStyleSheet(DemoStyle.GROUP_BOX_STYLE)
        layout = QtWidgets.QVBoxLayout(group)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("algorithm:"))
        self.algorithm_combo = QtWidgets.QComboBox()
        self.algorithm_names = list(REGISTRY)
        self.algorithm_combo.addItems(self.algorithm_names)
        self.algorithm_combo.setCurrentIndex(
            self.algorithm_names.index(self.tracking_algorithm)
        )
        self.algorithm_combo.currentIndexChanged.connect(self._algorithm_changed)
        row.addWidget(self.algorithm_combo, stretch=1)
        layout.addLayout(row)

        self.algorithm_stack = QtWidgets.QStackedWidget()
        for name in self.algorithm_names:
            tracker = self.trackers[name]
            page = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(page)
            for i, (attr, label, lo, hi, decimals) in enumerate(tracker.param_spec):
                self._add_slider(grid, i, label, lo, hi, decimals,
                                 getattr(tracker, attr), attr, target=tracker)
            self.algorithm_stack.addWidget(page)
        self.algorithm_stack.setCurrentIndex(
            self.algorithm_names.index(self.tracking_algorithm)
        )
        layout.addWidget(self.algorithm_stack)

        layout.addWidget(DemoStyle.make_info_box(
            "Proportional corrects a fraction of the current error and will trail "
            "a steadily drifting target by a fixed offset.\n"
            "PID adds an integral term that walks that offset out over a few "
            "seconds.\n"
            "Kalman estimates the target's velocity and commands where it will "
            "be once the stage arrives.\n"
            "Tune the horizon by minimising the RMS error against a drifting "
            "target: raise it while RMS falls, and back off once it climbs "
            "again. It only covers the stage's settle time (typically a frame "
            "or two) because motion already commanded is accounted for exactly, "
            "so it is much shorter than the latency shown above. Leave centroid "
            "smoothing at 0 so the filter is not fed a pre-lagged position.",
            QtWidgets, style=DemoStyle.INSTRUCTIONS_BOX_STYLE,
        ))
        return group

    def _algorithm_changed(self, index):
        self.tracking_algorithm = self.algorithm_names[index]
        self.tracker = self.trackers[self.tracking_algorithm]
        self.algorithm_stack.setCurrentIndex(index)
        self._reset_control_state()
        logger.info(f"Tracking algorithm -> {self.tracking_algorithm}")

    def _add_slider(self, grid, row, label, lo, hi, decimals, value, attr,
                    target=None):
        """A labeled slider + spinbox that live-writes <target>.<attr>.

        Defaults to writing on the worker itself; algorithm pages pass their
        tracker so its parameters can be tuned while the loop runs.
        """
        scale = 10 ** decimals
        grid.addWidget(QtWidgets.QLabel(label + ":"), row, 0)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(lo * scale), int(hi * scale))
        slider.setValue(int(value * scale))
        grid.addWidget(slider, row, 1)

        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(lo, hi)
        spin.setSingleStep(1.0 / scale)
        spin.setValue(value)
        grid.addWidget(spin, row, 2)

        def on_slider(v):
            val = v / scale
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
            self._set_param(attr, val, target)

        def on_spin(val):
            slider.blockSignals(True)
            slider.setValue(int(val * scale))
            slider.blockSignals(False)
            self._set_param(attr, val, target)

        slider.valueChanged.connect(on_slider)
        spin.valueChanged.connect(on_spin)

    def _set_param(self, attr, val, target=None):
        # roi_radius_px is used as an int extent
        setattr(target or self, attr, int(val) if attr == "roi_radius_px" else val)

    # -------------------------------------------------------------- callbacks

    def _tracking_toggled(self, checked):
        color = DemoStyle.COLOR_WARNING if checked else DemoStyle.COLOR_NEUTRAL
        self.enable_tracking_button.setStyleSheet(f"background-color: {color}")
        if checked:
            # Start clean: whatever the stage did while tracking was off (jogs,
            # a calibration step) is history, not control input.
            self._reset_control_state()

    def _reset_control_state(self):
        """Drop estimator state and re-baseline against the stage's totals."""
        if self.tracker is not None:
            self.tracker.reset()
        self._prev_applied = np.array(
            [float(self.shared_stage_offset_xy[2]),
             float(self.shared_stage_offset_xy[3])]
        )
        self._lost_frames = 0
        self._cmd_log.clear()
        self._err_hist.clear()

    def _send(self, msg):
        try:
            if self.child_conn is not None:
                self.child_conn.send(msg)
        except Exception:
            pass

    def _toggle_recording(self, checked):
        if checked:
            idx = len(self.epochs)
            name = self.rec_name_edit.text().strip() or f"sub{idx:02d}"
            self.open_epoch = {
                "index": idx,
                "name": name,
                "start_frame": self.image_count,
                "start_ts": time.time(),
            }
            self.recording = True
            self.record_button.setText("Stop Recording")
            self.record_button.setStyleSheet(
                f"background-color: {DemoStyle.COLOR_DANGER}"
            )
            self._send({"type": "recording_start", **self.open_epoch})
        else:
            self._finalize_epoch()
            self.record_button.setText("Start Recording")
            self.record_button.setStyleSheet(
                f"background-color: {DemoStyle.COLOR_SUCCESS}"
            )
        self._refresh_readouts()

    def _finalize_epoch(self):
        """Close the open recording epoch (on Stop, or when the worker exits)."""
        if self.open_epoch is None:
            return
        ep = self.open_epoch
        ep["stop_frame"] = self.image_count
        ep["stop_ts"] = time.time()
        self.epochs.append(ep)
        self.recording = False
        self.open_epoch = None
        self._send({"type": "recording_stop", **ep})

    def _screen_to_axes(self, vert, horiz):
        """Map a desired screen motion to (axis0, axis1) stage offsets.

        vert/horiz are signed magnitudes in the image frame (+vert = toward
        larger row / down on screen, +horiz = toward larger col / right).
        swap_axes handles a 90 deg stage/camera rotation; invert flips each
        physical axis. Shared by manual jog and automatic tracking so both stay
        sign-consistent.
        """
        if self.swap_axes:
            a0, a1 = horiz, vert
        else:
            a0, a1 = vert, horiz
        if self.invert_axis0:
            a0 = -a0
        if self.invert_axis1:
            a1 = -a1
        return a0, a1

    def _jog_screen(self, vert=0, horiz=0):
        """Jog the stage one step in a screen direction (Up/Down/Left/Right)."""
        a0, a1 = self._screen_to_axes(vert, horiz)
        self._command(a0 * self.jog_step_um, a1 * self.jog_step_um)
        logger.info(f"Jog (vert={vert}, horiz={horiz}) -> axes ({a0}, {a1})")

    def _ratio_edited(self):
        try:
            self.micron_to_pix_ratio = float(self.ratio_edit.text())
            self._emit_calibration()
        except ValueError:
            self.ratio_edit.setText(f"{self.micron_to_pix_ratio:.5f}")

    def _nudge_ratio(self, sign):
        self.micron_to_pix_ratio = max(
            0.0, self.micron_to_pix_ratio + sign * self.ratio_increment
        )
        self.ratio_edit.setText(f"{self.micron_to_pix_ratio:.5f}")
        self._emit_calibration()

    def _start_calibration(self):
        if self.cal_state != "idle":
            return
        if self.enable_tracking_button.isChecked():
            self.cal_readout.setText("Disable tracking before calibrating.")
            return
        if not self.blob_ok:
            self.cal_readout.setText(
                "No blob detected - center a visible blob before calibrating."
            )
            return
        self.cal_centroid_before = (self.raw_cy, self.raw_cx)
        self.cal_start_count = self.image_count
        # Command a known move on stage axis0.
        self._command(self.cal_step_um, 0.0)
        self.cal_state = "settling"
        self.cal_readout.setText(
            f"Moving stage {self.cal_step_um} um, waiting to settle..."
        )

    def _advance_calibration(self):
        """Called each new frame while a calibration step is in flight."""
        if self.cal_state != "settling":
            return
        if self.image_count - self.cal_start_count < self.cal_settle_frames:
            return

        dy = self.raw_cy - self.cal_centroid_before[0]
        dx = self.raw_cx - self.cal_centroid_before[1]
        pix = float(np.hypot(dy, dx))
        self.cal_state = "idle"

        # Return the stage to where it started so the blob ends up recentered.
        self._command(-self.cal_step_um, 0.0)

        if pix < 1.0 or not np.isfinite(pix):
            self.cal_readout.setText(
                f"Calibration failed: blob moved only {pix:.2f} px. "
                "Increase the step or check the blob is visible."
            )
            return

        self.micron_to_pix_ratio = self.cal_step_um / pix
        self.ratio_edit.setText(f"{self.micron_to_pix_ratio:.5f}")
        self.cal_readout.setText(
            f"Measured {pix:.1f} px for {self.cal_step_um} um "
            f"-> {self.micron_to_pix_ratio:.5f} um/px "
            f"(dy={dy:.1f}, dx={dx:.1f})"
        )
        self._emit_calibration()

    def _emit_calibration(self):
        try:
            self.child_conn.send(
                {"type": "calibration_update",
                 "micron_to_pix_ratio": self.micron_to_pix_ratio}
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ track

    def update_stage_offset(self, dt, frame_ts):
        """Run the selected control algorithm and command the stage.

        Everything except the middle line is shared by all algorithms: the same
        error goes in, the same clamp and per-axis enable come out, so switching
        algorithm changes only the position asked for.
        """
        dy = self.cy - self.sm_cy
        dx = self.cx - self.sm_cx

        # Deadband: ignore sub-threshold error so a centered blob is left still.
        if abs(dy) < self.deadband_px:
            dy = 0.0
        if abs(dx) < self.deadband_px:
            dx = 0.0

        # Pixel error -> microns on each physical stage axis. swap_axes routes
        # vertical error to axis1 when the stage is rotated 90 deg vs the camera,
        # and invert flips a physical axis; this is the same transform manual jog
        # uses, so a D-pad that feels right means correct tracking signs. The
        # ratio is microns per pixel at the current imaging condition (what
        # calibration measures), so binning needs no extra factor.
        a0, a1 = self._screen_to_axes(dy, dx)
        err_um = np.array([a0, a1]) * self.micron_to_pix_ratio

        u_pre, u_post, pending = self._stage_feedback(frame_ts)
        cmd = self.tracker.update(err_um, dt, u_pre, u_post, pending)

        # Clamp to reject outlier corrections (e.g. a momentarily lost blob).
        cmd = np.clip(cmd, -self.max_step_um, self.max_step_um)
        if not self.enable_axis0:
            cmd[0] = 0.0
        if not self.enable_axis1:
            cmd[1] = 0.0
        self._last_cmd = cmd
        self._command(cmd[0], cmd[1], frame_ts=frame_ts)

        # Throttled diagnostic: watch the error sequence. Shrinking |dy|,|dx| =
        # healthy negative feedback; growing = wrong sign on that axis; ping-
        # ponging = dead-time/gain (loop commanding faster than the blob's new
        # position becomes visible, or um/px miscalibrated).
        now = time.time()
        if now - self._last_track_log >= 0.25:
            self._last_track_log = now
            extra = self.tracker.readout()
            # print (not logger): this runs in the GUI subprocess, which on
            # Windows spawn has no console log handler.
            print(
                f"[track:{self.tracking_algorithm}] "
                f"err(dy={dy:+.1f}, dx={dx:+.1f}) px  "
                f"cmd(axis0={cmd[0]:+.2f}, axis1={cmd[1]:+.2f}) um  "
                f"pending=({pending[0]:+.2f}, {pending[1]:+.2f})  "
                f"rms={self._rms_error_px():.2f} px  "
                f"dt={dt * 1000:.1f} ms  latency={self._latency_ms:.0f} ms"
                + (f"  {extra}" if extra else ""),
                flush=True,
            )

    def _rms_error_px(self):
        """RMS centering error over the recent window — the number to compare
        algorithms on."""
        if not self._err_hist:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(self._err_hist))))

    def update_display(self):
        frame_ts, dt = self._frame_timing()
        frame = self.current_frame()

        cy, cx, npix, snr = tracking_numba.bright_blob_centroid(frame, self.threshold_frac)
        self.blob_snr = snr
        self.blob_ok = npix > 0 and np.isfinite(cy) and snr >= self.min_puncta_snr
        if self.blob_ok:
            self.raw_cy, self.raw_cx = cy, cx
            s = self.centroid_smoothing
            self.sm_cy = s * self.sm_cy + (1.0 - s) * cy
            self.sm_cx = s * self.sm_cx + (1.0 - s) * cx
            self.centroid_marker.setData(x=[self.sm_cx], y=[self.sm_cy])
            self._err_hist.append(np.hypot(self.cy - self.sm_cy, self.cx - self.sm_cx))
        else:
            self.centroid_marker.setData(x=[], y=[])

        f_mean = self._extract_fluorescence(frame)
        self._update_image(frame)

        # Closed-loop correction (skipped while a calibration step is settling).
        self._last_cmd = np.zeros(2)
        if self.cal_state == "idle" and self.enable_tracking_button.isChecked():
            if self.blob_ok:
                self._lost_frames = 0
                self.update_stage_offset(dt, frame_ts)
            else:
                # Hold still while the blob is missing, and once it has been gone
                # long enough that the estimate is stale, start the algorithm over
                # rather than resuming from a stale velocity or integral.
                self._lost_frames += 1
                if self._lost_frames == self.reset_after_lost_frames:
                    self.tracker.reset()

        self._record_row(frame_ts, f_mean)
        self._advance_calibration()
        self._refresh_readouts()

    def _extract_fluorescence(self, frame):
        """Mean intensity in a fixed-radius circle on the blob; plot + record."""
        if self.blob_ok:
            f_mean, _, _ = tracking_numba.circle_roi_stats(
                frame, self.sm_cy, self.sm_cx, self.roi_radius_px
            )
            xs = self.sm_cx + self.roi_radius_px * np.cos(self._unit_circle)
            ys = self.sm_cy + self.roi_radius_px * np.sin(self._unit_circle)
            self.roi_circle.setData(xs, ys)
        else:
            f_mean = np.nan
            self.roi_circle.setData([], [])

        self.recent_f[:-1] = self.recent_f[1:]
        self.recent_f[-1] = f_mean
        self.f_curve.setData(self.recent_f, connect="finite")
        return f_mean

    def _record_row(self, frame_ts, f_mean):
        """Append this frame's trace sample: fluorescence, where the blob was,
        the error that implies, and what the controller did about it."""
        cam_ms = float(self.shared_image_count[1]) if self._have_frame_ts else np.nan
        ok = self.blob_ok
        nan = float("nan")
        self.rec_rows.append((
            self.image_count,
            f"{frame_ts:.6f}",
            f"{cam_ms:.3f}",
            f"{f_mean:.4f}",
            f"{self.sm_cy if ok else nan:.4f}",
            f"{self.sm_cx if ok else nan:.4f}",
            f"{self.cy - self.sm_cy if ok else nan:.4f}",
            f"{self.cx - self.sm_cx if ok else nan:.4f}",
            f"{self._last_cmd[0]:.4f}",
            f"{self._last_cmd[1]:.4f}",
            self.tracking_algorithm,
            self.open_epoch["index"] if self.recording else -1,
        ))

    def _update_image(self, frame):
        if self.show_processed_button.isChecked():
            display = tracking_numba.bright_blob_mask(frame, self.threshold_frac)
        else:
            display = frame
        auto = self.first_img_flag or self.auto_contrast_button.isChecked()
        self.ii.setImage(display, autoLevels=auto)
        self.first_img_flag = False

    def _refresh_readouts(self):
        tracking = self.enable_tracking_button.isChecked()
        f_last = self.recent_f[-1]
        f_str = "--" if not np.isfinite(f_last) else f"{f_last:.1f}"
        rec = "REC" if self.recording else "off"
        puncta = "puncta" if self.blob_ok else "no puncta"
        lat = (
            "--" if not np.isfinite(self._latency_ms)
            else f"{self._latency_ms:.0f} ms"
        )
        rms = self._rms_error_px()
        rms_str = "--" if not np.isfinite(rms) else f"{rms:.2f} px"
        extra = self.tracker.readout() if self.tracker is not None else ""
        self.status_label.setText(
            f"frame: {self.image_count}   "
            f"blob: ({self.sm_cx:.1f}, {self.sm_cy:.1f})   "
            f"target: ({self.cx}, {self.cy})   "
            f"SNR: {self.blob_snr:.1f} ({puncta})   "
            f"F: {f_str}   "
            f"um/px: {self.micron_to_pix_ratio:.4f}   "
            f"tracking: {'ON' if tracking else 'off'}   "
            f"rec: {rec}"
            "\n"
            f"alg: {self.tracking_algorithm}   "
            f"RMS error: {rms_str}   "
            f"latency: {lat}   "
            f"dt: {self._dt_s * 1000:.1f} ms ({self._clock} clock)"
            + (f"   {extra}" if extra else "")
        )
        if hasattr(self, "record_status"):
            self.record_status.setText(self._recording_summary())

    def _recording_summary(self):
        lines = [f"completed sub-acquisitions: {len(self.epochs)}"]
        if self.recording and self.open_epoch is not None:
            n = self.image_count - self.open_epoch["start_frame"]
            dur = time.time() - self.open_epoch["start_ts"]
            lines.append(
                f"RECORDING '{self.open_epoch['name']}': "
                f"{n} frames, {dur:.1f}s (started at frame "
                f"{self.open_epoch['start_frame']})"
            )
        for ep in self.epochs[-3:]:
            lines.append(
                f"  [{ep['index']}] {ep['name']}: frames "
                f"{ep['start_frame']}-{ep['stop_frame']}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------- loop

    def run(self):
        try:
            self.initialize_display()
            self._spool()
            self.app.processEvents()
            self.image_count = self.shared_image_count[0]

            while True:
                if self.child_conn.poll():
                    if self.child_conn.recv() == "close":
                        break

                try:
                    if self.image_count == self.shared_image_count[0]:
                        self.app.processEvents()
                        continue
                    self.image_count = self.shared_image_count[0]
                except ValueError:
                    continue

                self.update_display()
                self.app.processEvents()

        except EOFError:
            logger.warning("XYTrackingWorker pipe closed (EOF)")
        except BrokenPipeError:
            logger.warning("XYTrackingWorker pipe closed (BrokenPipe)")
        except Exception as err:
            logger.error(f"XYTrackingWorker error: {err}")
            raise
        finally:
            self.close()

    def _spool(self):
        """Pre-compile numba kernels so the first tracked frame has no hitch."""
        dummy = np.zeros((self.ysize, self.xsize), dtype=self.dtype)
        dummy[self.cy, self.cx] = 1000
        try:
            tracking_numba.bright_blob_centroid(dummy, self.threshold_frac)
            tracking_numba.bright_blob_mask(dummy, self.threshold_frac)
            tracking_numba.circle_roi_stats(dummy, self.cy, self.cx, self.roi_radius_px)
        except Exception as err:
            logger.warning(f"numba spool failed: {err}")

        zero = np.zeros(2)
        for tracker in self.trackers.values():
            tracker.update(zero, self._nominal_dt, zero, zero, zero)
            tracker.reset()

    def _save_trace(self):
        if not self.rec_rows:
            return
        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception as err:
            logger.warning(f"Cannot create save dir '{self.save_dir}': {err}")
            return

        # Per-frame fluorescence and tracking record, tagged with its
        # sub-acquisition (-1 = none). The error and command columns are what
        # you post-hoc compare algorithms on.
        try:
            path = os.path.join(self.save_dir, f"{self.rec_id}_fluorescence.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame", "host_time_s", "camera_time_ms", "fluorescence",
                    "centroid_y", "centroid_x", "error_y_px", "error_x_px",
                    "command_axis0_um", "command_axis1_um", "algorithm",
                    "sub_acquisition",
                ])
                writer.writerows(self.rec_rows)
            logger.info(f"Saved {len(self.rec_rows)} tracking samples to {path}")
        except Exception as err:
            logger.warning(f"Failed to save fluorescence trace: {err}")

        # Tuning actually used. The sliders are live, so the config file is not
        # a record of what ran — this is, and it is what makes an algorithm
        # comparison reproducible after the fact.
        try:
            path = os.path.join(self.save_dir, f"{self.rec_id}_tracking_params.json")
            with open(path, "w") as f:
                json.dump({
                    "tracking_algorithm": self.tracking_algorithm,
                    "micron_to_pix_ratio": self.micron_to_pix_ratio,
                    "deadband_px": self.deadband_px,
                    "centroid_smoothing": self.centroid_smoothing,
                    "max_step_um": self.max_step_um,
                    "min_puncta_snr": self.min_puncta_snr,
                    "threshold_frac": self.threshold_frac,
                    "swap_axes": self.swap_axes,
                    "invert_axis0": self.invert_axis0,
                    "invert_axis1": self.invert_axis1,
                    "enable_axis0": self.enable_axis0,
                    "enable_axis1": self.enable_axis1,
                    "algorithms": {
                        name: tracker.describe()
                        for name, tracker in self.trackers.items()
                    },
                    "measured_latency_ms": self._latency_ms,
                    "measured_frame_interval_ms": self._dt_s * 1000.0,
                    "rms_error_px": self._rms_error_px(),
                }, f, indent=2)
            logger.info(f"Saved tracking parameters to {path}")
        except Exception as err:
            logger.warning(f"Failed to save tracking parameters: {err}")

        # Sub-acquisition summary: frame ranges + timestamps for video subselection.
        try:
            path = os.path.join(self.save_dir, f"{self.rec_id}_subacquisitions.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "index", "name", "start_frame", "stop_frame", "n_frames",
                    "start_time", "stop_time", "start_unix", "stop_unix",
                    "duration_s",
                ])
                for ep in self.epochs:
                    writer.writerow([
                        ep["index"], ep["name"],
                        ep["start_frame"], ep["stop_frame"],
                        ep["stop_frame"] - ep["start_frame"],
                        _iso(ep["start_ts"]), _iso(ep["stop_ts"]),
                        f"{ep['start_ts']:.3f}", f"{ep['stop_ts']:.3f}",
                        f"{ep['stop_ts'] - ep['start_ts']:.3f}",
                    ])
            logger.info(f"Saved {len(self.epochs)} sub-acquisitions to {path}")
        except Exception as err:
            logger.warning(f"Failed to save sub-acquisition summary: {err}")

    def close(self):
        # Finalize a still-open recording so an unstopped epoch is still saved.
        self._finalize_epoch()
        self._save_trace()
        for shm in self.shared_frame_memory_list:
            try:
                shm.close()
            except Exception:
                pass
        for shl in (self.shared_image_count, self.shared_stage_offset_xy):
            try:
                if shl is not None:
                    shl.shm.close()
            except Exception:
                pass
        try:
            self.child_conn.close()
        except Exception:
            pass
