"""
Ring Attractor Logic for CLEF2.

Extracts puncta position from uint16 images, tracks XY/theta/ring state,
provides interactive GUI with manual and closed-loop stimulus control.
"""

import logging
from pathlib import Path
import numpy as np
from typing import Any, ClassVar, Dict, Optional, Tuple

from core.logic.BaseClosedLoopLogic import BaseClosedLoopLogic
from utils.style.demo_stylization import DemoStyle

logger = logging.getLogger(__name__)


class RingAttractorLogic(BaseClosedLoopLogic):
    """
    Closed-loop logic for ring attractor demo.

    Extracts puncta position, monitors trajectory, and provides
    manual/ROI-based stimulus control with visualization.
    """

    logic_class: ClassVar[Optional[str]] = "ring_attractor_logic"

    def __init__(
        self,
        name: str,
        config: Dict[str, Any] | None = None,
        output_devices: Dict[str, Any] | None = None,
        gui_parameters: Dict[str, Any] | None = None,
        input_devices: Dict[str, Any] | None = None,
        io_manager: Any = None,
        config_manager: Any = None,
    ):
        super().__init__(
            name, config, output_devices, gui_parameters,
            input_devices=input_devices, io_manager=io_manager,
            config_manager=config_manager,
        )

        cfg = self.config
        self.inner_radius = cfg.get("inner_radius", 3.0)
        self.outer_radius = cfg.get("outer_radius", 6.0)
        self.image_width = cfg.get("image_width", 100)
        self.image_height = cfg.get("image_height", 100)
        self.stim_cooldown_frames = cfg.get("stim_cooldown_frames", 50)
        self.visualize_real_time = cfg.get("visualize_real_time", False)
        self.fading_trajectory_samples = cfg.get("fading_trajectory_samples", 100)

        # Which input device to read images from
        self.input_device_name = next(iter(self.input_devices), None)

        # Image center
        self.center_x = self.image_width / 2.0
        self.center_y = self.image_height / 2.0

        # State tracking
        self.frame_count = 0
        self.cooldown_counter = 0
        self.transition_count = 0

        # Extracted state timeseries
        self.x_history = []
        self.y_history = []
        self.theta_history = []
        self.ring_history = []
        self.frame_indices = []
        self.stim_events = []

        # Manual stimulus control
        self.manual_stim_pending = False
        self.current_stim_intensity = 0
        self.current_omega_perturbation = 0

        # GUI screenshot freq
        self.gui_screenshot_freq = self.gui_parameters.get("gui_screenshot_freq", 0)
        self.saveroot = cfg.get("save_dir", "./demo_output/ring_attractor")

        # Initialize visualizer
        self.visualizer = None
        if self.visualize_real_time:
            try:
                self.visualizer = RingVisualizer(self)
            except Exception as e:
                logger.warning(f"Could not initialize visualizer: {e}")

        logger.info(
            f"RingAttractorLogic '{name}' initialized with rings at "
            f"r={self.inner_radius}, {self.outer_radius}"
        )

    def initialize_model(self):
        logger.info("RingAttractorLogic model initialized")

    def _find_puncta_centroid(self, image: np.ndarray) -> Tuple[float, float]:
        threshold = np.percentile(image, 99)
        mask = image > threshold

        if not np.any(mask):
            return self.image_width / 2.0, self.image_height / 2.0

        coords = np.argwhere(mask)
        centroid_y = np.mean(coords[:, 0])
        centroid_x = np.mean(coords[:, 1])

        return centroid_x, centroid_y

    def _pixel_to_polar(self, x: float, y: float) -> Tuple[float, float, int]:
        dx = x - self.center_x
        dy = y - self.center_y

        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        if theta < 0:
            theta += 2 * np.pi

        dist_inner = abs(r - self.inner_radius)
        dist_outer = abs(r - self.outer_radius)
        ring_index = 0 if dist_inner < dist_outer else 1

        return theta, r, ring_index

    def process_sample(self, sample: Any):
        """Process input_stores dict — extract image and find puncta."""
        self.frame_count += 1

        # Get image from input stores
        if self.input_device_name and isinstance(sample, dict):
            img = sample.get(self.input_device_name)
        elif isinstance(sample, dict):
            # Use first available input
            img = next(iter(sample.values()), None)
        else:
            img = sample

        if img is None:
            return

        x_pixel, y_pixel = self._find_puncta_centroid(img)

        x_centered = x_pixel - self.center_x
        y_centered = y_pixel - self.center_y

        theta, r, ring_idx = self._pixel_to_polar(x_pixel, y_pixel)

        self.x_history.append(x_centered)
        self.y_history.append(y_centered)
        self.theta_history.append(theta)
        self.ring_history.append(ring_idx)
        self.frame_indices.append(self.frame_count)

        # Check for ring transitions
        if len(self.ring_history) > 1:
            if self.ring_history[-1] != self.ring_history[-2]:
                self.transition_count += 1
                logger.info(f"Frame {self.frame_count}: Ring transition (total: {self.transition_count})")

        # Update visualizer
        if self.visualizer:
            self.visualizer.update_image(img, x_pixel, y_pixel)
            self.visualizer.update_trajectory()
            self.visualizer.update_info_text()
            self.visualizer.process_events()

        self._maybe_save_screenshot()

        if self.frame_count % 100 == 0:
            logger.info(
                f"Frame {self.frame_count}: x={x_centered:.1f}, y={y_centered:.1f}, "
                f"r={r:.1f}, ring={ring_idx}"
            )

    def _maybe_save_screenshot(self):
        if not self.gui_screenshot_freq or self.frame_count % self.gui_screenshot_freq != 0:
            return
        if not self.visualizer:
            return
        try:
            screenshot_dir = Path(self.saveroot) / "gui_screenshot"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            path = screenshot_dir / f"screenshot_{self.frame_count:06d}.png"
            pixmap = self.visualizer.window.grab()
            pixmap.save(str(path), "PNG")
            logger.debug(f"Saved GUI screenshot: {path}")
        except Exception as err:
            logger.warning(f"Failed to save GUI screenshot: {err}")

    def _check_logic(self) -> Optional[Dict[str, Any]]:
        """Check for manual or ROI-triggered stimulus."""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None

        # Check manual trigger
        if self.manual_stim_pending:
            self.manual_stim_pending = False
            self._record_stim_event("manual")
            self.cooldown_counter = self.stim_cooldown_frames

            output_name = self._get_output_device_name()
            if output_name:
                return {
                    output_name: {
                        "radial_perturbation": self.current_stim_intensity,
                        "omega_perturbation": self.current_omega_perturbation,
                    }
                }

        # Check closed-loop ROI trigger
        if self.visualizer and len(self.x_history) > 0:
            x = self.x_history[-1]
            y = self.y_history[-1]
            if self.visualizer.is_state_in_any_roi(x, y):
                intensity = self.visualizer.intensity_slider.value()
                omega_perturbation = self.visualizer.omega_slider.value() / 10.0
                self.current_stim_intensity = intensity
                self.current_omega_perturbation = omega_perturbation
                self._record_stim_event("roi")
                self.cooldown_counter = self.stim_cooldown_frames

                output_name = self._get_output_device_name()
                if output_name:
                    return {
                        output_name: {
                            "radial_perturbation": intensity,
                            "omega_perturbation": omega_perturbation,
                        }
                    }

        return None

    def _get_output_device_name(self) -> Optional[str]:
        if self.output_devices:
            return next(iter(self.output_devices.keys()))
        return None

    def _record_stim_event(self, trigger_type: str):
        self.stim_events.append({
            "frame": self.frame_count,
            "x": self.x_history[-1] if self.x_history else 0,
            "y": self.y_history[-1] if self.y_history else 0,
            "theta": self.theta_history[-1] if self.theta_history else 0,
            "ring": self.ring_history[-1] if self.ring_history else 0,
            "radial_perturbation": self.current_stim_intensity,
            "omega_perturbation": self.current_omega_perturbation,
            "trigger_type": trigger_type,
        })
        logger.info(
            f"{trigger_type.capitalize()} stimulus at frame {self.frame_count}, "
            f"radial={self.current_stim_intensity}, omega={self.current_omega_perturbation}"
        )

    def trigger_manual_stimulus(self, intensity: float, omega_perturbation: float):
        self.current_stim_intensity = intensity
        self.current_omega_perturbation = omega_perturbation
        self.manual_stim_pending = True

    def get_metadata(self) -> Dict[str, Any]:
        base = super().get_metadata()
        base.update({
            "frames_processed": self.frame_count,
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "x_history": self.x_history,
            "y_history": self.y_history,
            "theta_history": self.theta_history,
            "ring_history": self.ring_history,
            "frame_indices": self.frame_indices,
            "stim_events": self.stim_events,
            "num_stim_events": len(self.stim_events),
            "num_transitions": self.transition_count,
        })
        return base

    def save_data(self, **kwargs):
        """Save trajectory plot via matplotlib."""
        if len(self.x_history) < 2:
            return

        savefilename = kwargs.get("savefilename")
        if savefilename is None:
            savefilename = str(Path(self.saveroot) / "ring_attractor_trajectory.png")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # XY state space
        ax = axes[0, 0]
        colors = np.arange(len(self.x_history))
        scatter = ax.scatter(self.x_history, self.y_history, c=colors, s=1, cmap="viridis", alpha=0.5)
        circle_inner = plt.Circle((0, 0), self.inner_radius, fill=False, color="blue", linestyle="--", alpha=0.5)
        circle_outer = plt.Circle((0, 0), self.outer_radius, fill=False, color="red", linestyle="--", alpha=0.5)
        ax.add_patch(circle_inner)
        ax.add_patch(circle_outer)
        if self.stim_events:
            stim_x = [e["x"] for e in self.stim_events]
            stim_y = [e["y"] for e in self.stim_events]
            ax.scatter(stim_x, stim_y, c="gold", s=100, marker="*", zorder=5, label="Stimulus")
        ax.set_xlabel("X (px, centered)")
        ax.set_ylabel("Y (px, centered)")
        ax.set_title("XY State Space Trajectory")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(scatter, ax=ax, label="Time (frame)")

        # X time series
        ax = axes[0, 1]
        ax.plot(self.frame_indices, self.x_history, "r-", alpha=0.7, linewidth=0.5, label="X")
        for event in self.stim_events:
            ax.axvline(event["frame"], color="gold", alpha=0.3, linestyle="--")
        ax.set_xlabel("Frame")
        ax.set_ylabel("X Position (px)")
        ax.set_title("X Position Over Time")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Y time series
        ax = axes[1, 0]
        ax.plot(self.frame_indices, self.y_history, "b-", alpha=0.7, linewidth=0.5, label="Y")
        for event in self.stim_events:
            ax.axvline(event["frame"], color="gold", alpha=0.3, linestyle="--")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Y Position (px)")
        ax.set_title("Y Position Over Time")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Ring index
        ax = axes[1, 1]
        ax.plot(self.frame_indices, self.ring_history, "k-", linewidth=0.5)
        for event in self.stim_events:
            ax.axvline(event["frame"], color="gold", alpha=0.3, linestyle="--")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Ring Index")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Inner", "Outer"])
        ax.set_title(f"Ring Transitions (Total: {self.transition_count})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(savefilename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savefilename, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {savefilename}")
        plt.close()

    def close(self):
        if self.visualizer:
            self.visualizer.close()
        logger.info(
            f"RingAttractorLogic closing. {self.frame_count} frames, "
            f"{self.transition_count} transitions, {len(self.stim_events)} stimuli"
        )


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class _RotatedLabel:
    """QLabel that draws its text rotated 90 degrees counter-clockwise."""
    pass  # Defined at runtime when Qt is available


class RingVisualizer:
    """Real-time visualization with XY state space plot."""

    COLOR_INNER_RING = "#2EC4B6"
    COLOR_OUTER_RING = "#FF9F1C"
    COLOR_TRAJECTORY = "#A8DADC"
    COLOR_HEAD_NORMAL = "#F4D35E"
    COLOR_HEAD_COOLDOWN = "#E84855"
    COLOR_STIM_MARKER = "#E84855"

    def __init__(self, logic: RingAttractorLogic):
        self.logic = logic

        try:
            from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
            import pyqtgraph as pg
        except ImportError:
            logger.error("PyQt or pyqtgraph not available")
            raise

        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.pg = pg

        self._rois = []

        self.app = pg.mkQApp("RingVisualizer")
        DemoStyle.load_qss(self.app)

        # --- Build _RotatedLabel using the actual Qt classes ---
        class RotatedLabel(QtWidgets.QLabel):
            def paintEvent(self_, event):
                painter = QtGui.QPainter(self_)
                painter.setPen(QtGui.QColor("#555555"))
                painter.setFont(self_.font())
                painter.translate(self_.width() / 2, self_.height() / 2)
                painter.rotate(-90)
                rect = QtCore.QRectF(
                    -self_.height() / 2, -self_.width() / 2,
                    self_.height(), self_.width(),
                )
                painter.drawText(rect, QtCore.Qt.AlignCenter, self_.text())
                painter.end()

        self._RotatedLabel = RotatedLabel

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Ring Attractor — Closed-Loop Demo")
        self.window.resize(1280, 780)

        self.layout = QtWidgets.QGridLayout()
        self.layout.setColumnStretch(0, 0)
        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 0)
        self.window.setLayout(self.layout)

        # ── LEFT: Image column
        img_col = QtWidgets.QVBoxLayout()
        img_heading = DemoStyle.make_heading("Raw 'Microscopy' Images", QtWidgets)
        img_col.addWidget(img_heading)

        eq_label = QtWidgets.QLabel(
            "Gaussian puncta drawn according to:<br>"
            "<small>dr/dt = −k(r−r₁)(r−mid)(r−r₂) + u<br>"
            "dθ/dt = ω + u_ω</small>"
        )
        eq_label.setStyleSheet(DemoStyle.MUTED_TEXT_STYLE)
        img_col.addWidget(eq_label)

        self.image_widget = pg.ImageView()
        self.image_widget.ui.roiBtn.hide()
        self.image_widget.ui.menuBtn.hide()
        self.image_widget.ui.histogram.hide()
        self.image_widget.setFixedSize(320, 320)

        img_row = QtWidgets.QHBoxLayout()
        img_row.setSpacing(2)
        y_axis_label = RotatedLabel("Y (px)")
        y_axis_label.setFixedWidth(16)
        y_axis_label.setFixedHeight(320)
        img_row.addWidget(y_axis_label)
        img_row.addWidget(self.image_widget)
        img_col.addLayout(img_row)

        x_axis_label = QtWidgets.QLabel("X (px)")
        x_axis_label.setAlignment(QtCore.Qt.AlignHCenter)
        x_axis_label.setStyleSheet("font-size: 11px; color: #555; padding-left: 16px;")
        img_col.addWidget(x_axis_label)
        img_col.addStretch()

        img_container = QtWidgets.QWidget()
        img_container.setLayout(img_col)
        self.layout.addWidget(img_container, 0, 0, 2, 1)

        # ── CENTER: State space
        plot_col = QtWidgets.QVBoxLayout()
        plot_heading = DemoStyle.make_heading("Extracted Data — State Space", QtWidgets)
        plot_col.addWidget(plot_heading)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabel("left", "Y position (px, centered)")
        self.plot_widget.setLabel("bottom", "X position (px, centered)")
        plot_col.addWidget(self.plot_widget)

        plot_container = QtWidgets.QWidget()
        plot_container.setLayout(plot_col)
        self.layout.addWidget(plot_container, 0, 1, 2, 1)

        self._draw_ring_circles()

        self.trajectory_plot = self.plot_widget.plot(
            pen=pg.mkPen(self.COLOR_TRAJECTORY, width=2)
        )
        self.current_pos_plot = self.plot_widget.plot(
            pen=None, symbol="o", symbolSize=12,
            symbolBrush=self.COLOR_HEAD_NORMAL, symbolPen=None,
        )
        self.stim_marker_plot = self.plot_widget.plot(
            pen=None, symbol="x", symbolSize=14,
            symbolBrush=self.COLOR_STIM_MARKER,
            symbolPen=pg.mkPen(self.COLOR_STIM_MARKER, width=2),
        )
        self._stim_marker_x = []
        self._stim_marker_y = []
        self._stim_events_offset = 0

        # ── RIGHT: Control panel
        ctrl_col = QtWidgets.QVBoxLayout()
        ctrl_col.setSpacing(6)

        instructions = DemoStyle.make_info_box(
            "<b>Instructions</b><br>"
            "1. Watch the puncta orbit the ring attractors.<br>"
            "2. Use <i>Manual Stimulation</i> to push<br>"
            "&nbsp;&nbsp;&nbsp;the system between rings.<br>"
            "3. Draw a <i>Closed-Loop ROI</i> on the state<br>"
            "&nbsp;&nbsp;&nbsp;space to trigger automatically<br>"
            "&nbsp;&nbsp;&nbsp;when the state enters that region.<br>"
            "4. Adjust sliders to tune the perturbation.",
            QtWidgets, style=DemoStyle.INSTRUCTIONS_BOX_STYLE,
        )
        ctrl_col.addWidget(instructions)

        self.trigger_button = QtWidgets.QPushButton("Manual Stimulation")
        self.trigger_button.setFixedHeight(28)
        self.trigger_button.clicked.connect(self._on_trigger_clicked)
        ctrl_col.addWidget(self.trigger_button)

        roi_layout = QtWidgets.QHBoxLayout()
        self.add_roi_button = QtWidgets.QPushButton("Add Closed-Loop ROI")
        self.add_roi_button.setFixedHeight(28)
        self.add_roi_button.clicked.connect(self._on_add_roi_clicked)
        roi_layout.addWidget(self.add_roi_button)
        self.delete_rois_button = QtWidgets.QPushButton("Delete ROIs")
        self.delete_rois_button.setFixedHeight(28)
        self.delete_rois_button.clicked.connect(self._on_delete_rois_clicked)
        roi_layout.addWidget(self.delete_rois_button)
        ctrl_col.addLayout(roi_layout)

        cl_box = DemoStyle.make_info_box(
            "<b>Closed-Loop Trigger</b><br>"
            "When the system state (X, Y) enters a<br>"
            "drawn ROI, a stimulus is automatically<br>"
            "delivered (subject to cooldown).<br>"
            "Drag corners to resize; drag body to move.",
            QtWidgets, style=DemoStyle.CALLOUT_BOX_STYLE,
        )
        ctrl_col.addWidget(cl_box)

        ctrl_col.addWidget(DemoStyle.make_separator(QtWidgets))

        stim_heading = DemoStyle.make_heading("Stimulus Parameters", QtWidgets, style=DemoStyle.SUBHEADING_STYLE)
        ctrl_col.addWidget(stim_heading)

        # Radial slider
        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(QtWidgets.QLabel("r:"))
        self.intensity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.intensity_slider.setMinimum(-100)
        self.intensity_slider.setMaximum(100)
        self.intensity_slider.setValue(0)
        self.intensity_slider.setFixedHeight(20)
        self.intensity_slider.valueChanged.connect(self._on_intensity_changed)
        slider_layout.addWidget(self.intensity_slider)
        self.intensity_label = QtWidgets.QLabel("0")
        self.intensity_label.setFixedWidth(80)
        slider_layout.addWidget(self.intensity_label)
        ctrl_col.addLayout(slider_layout)

        # Omega slider
        omega_layout = QtWidgets.QHBoxLayout()
        omega_layout.addWidget(QtWidgets.QLabel("ω:"))
        self.omega_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.omega_slider.setMinimum(-100)
        self.omega_slider.setMaximum(100)
        self.omega_slider.setValue(0)
        self.omega_slider.setFixedHeight(20)
        self.omega_slider.valueChanged.connect(self._on_omega_changed)
        omega_layout.addWidget(self.omega_slider)
        self.omega_label = QtWidgets.QLabel("0.0")
        self.omega_label.setFixedWidth(80)
        omega_layout.addWidget(self.omega_label)
        ctrl_col.addLayout(omega_layout)

        slider_help = QtWidgets.QLabel(
            "<small>r: + push out, − pull in (~±15 switches rings)<br>"
            "ω: + speed up, − slow down rotation (±30)</small>"
        )
        slider_help.setStyleSheet(DemoStyle.HELP_TEXT_STYLE)
        ctrl_col.addWidget(slider_help)

        sep2 = DemoStyle.make_separator(QtWidgets)
        ctrl_col.addWidget(sep2)

        self.clear_markers_button = QtWidgets.QPushButton("Clear Stim Markers")
        self.clear_markers_button.setFixedHeight(24)
        self.clear_markers_button.clicked.connect(self._on_clear_markers_clicked)
        ctrl_col.addWidget(self.clear_markers_button)

        self.info_text = QtWidgets.QLabel()
        self.info_text.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.info_text.setWordWrap(True)
        self.info_text.setStyleSheet(DemoStyle.INFO_BOX_STYLE)
        ctrl_col.addWidget(self.info_text)
        ctrl_col.addStretch()

        ctrl_container = QtWidgets.QWidget()
        ctrl_container.setFixedWidth(260)
        ctrl_container.setLayout(ctrl_col)
        self.layout.addWidget(ctrl_container, 0, 2, 2, 1)

        self.window.show()
        logger.info("RingVisualizer initialized")

    def _draw_ring_circles(self):
        theta = np.linspace(0, 2 * np.pi, 200)
        x_inner = self.logic.inner_radius * np.cos(theta)
        y_inner = self.logic.inner_radius * np.sin(theta)
        self.plot_widget.plot(
            x_inner, y_inner,
            pen=self.pg.mkPen(self.COLOR_INNER_RING, width=2, style=self.QtCore.Qt.DotLine),
        )
        x_outer = self.logic.outer_radius * np.cos(theta)
        y_outer = self.logic.outer_radius * np.sin(theta)
        self.plot_widget.plot(
            x_outer, y_outer,
            pen=self.pg.mkPen(self.COLOR_OUTER_RING, width=2, style=self.QtCore.Qt.DotLine),
        )

    # ── ROI helpers
    def _on_add_roi_clicked(self):
        r = self.logic.inner_radius
        roi = self.pg.RectROI(
            pos=[-r * 0.5, -r * 0.5], size=[r, r],
            pen=self.pg.mkPen("#F4D35E", width=2),
            handlePen=self.pg.mkPen("#F4D35E", width=2),
            movable=True, resizable=True,
        )
        self.plot_widget.addItem(roi)
        self._rois.append(roi)

    def _on_delete_rois_clicked(self):
        for roi in self._rois:
            self.plot_widget.removeItem(roi)
        self._rois.clear()

    def is_state_in_any_roi(self, x: float, y: float) -> bool:
        for roi in self._rois:
            pos = roi.pos()
            size = roi.size()
            if pos.x() <= x <= pos.x() + size.x() and pos.y() <= y <= pos.y() + size.y():
                return True
        return False

    # ── Button callbacks
    def _on_trigger_clicked(self):
        intensity = self.intensity_slider.value()
        omega_perturbation = self.omega_slider.value() / 10.0
        self.logic.trigger_manual_stimulus(intensity, omega_perturbation)

    def _on_intensity_changed(self, value):
        label = f"+{value} (out)" if value > 0 else (f"{value} (in)" if value < 0 else "0")
        self.intensity_label.setText(label)
        self.logic.current_stim_intensity = value

    def _on_omega_changed(self, value):
        omega_value = value / 10.0
        label = f"+{omega_value:.1f} (↑)" if omega_value > 0 else (f"{omega_value:.1f} (↓)" if omega_value < 0 else "0.0")
        self.omega_label.setText(label)
        self.logic.current_omega_perturbation = omega_value

    def _on_clear_markers_clicked(self):
        self._stim_marker_x.clear()
        self._stim_marker_y.clear()
        self._stim_events_offset = len(self.logic.stim_events)
        self.stim_marker_plot.setData([], [])

    # ── Per-frame update
    def update_image(self, img: np.ndarray, puncta_x: float, puncta_y: float):
        self.image_widget.setImage(
            img.T, autoLevels=False, autoRange=False,
            levels=(0, int(self.logic.image_width)),
        )

    def update_trajectory(self):
        if len(self.logic.x_history) < 2:
            return

        N = min(self.logic.fading_trajectory_samples, len(self.logic.x_history))
        x_coords = self.logic.x_history[-N:]
        y_coords = self.logic.y_history[-N:]

        self.trajectory_plot.setData(x_coords, y_coords)

        in_cooldown = self.logic.cooldown_counter > 0
        head_color = self.COLOR_HEAD_COOLDOWN if in_cooldown else self.COLOR_HEAD_NORMAL
        self.current_pos_plot.setData(
            [x_coords[-1]], [y_coords[-1]],
            symbolBrush=head_color, symbolPen=None,
        )

        shown_count = self._stim_events_offset + len(self._stim_marker_x)
        if len(self.logic.stim_events) > shown_count:
            for event in self.logic.stim_events[shown_count:]:
                self._stim_marker_x.append(event["x"])
                self._stim_marker_y.append(event["y"])
            self.stim_marker_plot.setData(self._stim_marker_x, self._stim_marker_y)

    def update_info_text(self):
        if not self.logic.x_history:
            return

        x = self.logic.x_history[-1]
        y = self.logic.y_history[-1]
        theta = self.logic.theta_history[-1] if self.logic.theta_history else 0
        ring_idx = self.logic.ring_history[-1] if self.logic.ring_history else 0
        r = np.sqrt(x**2 + y**2)
        cooldown = self.logic.cooldown_counter

        info = (
            f"<b>Frame:</b> {self.logic.frame_count}<br>"
            f"<b>X</b> = {x:.1f} px &nbsp; <b>Y</b> = {y:.1f} px<br>"
            f"<b>R</b> = {r:.1f} px &nbsp; <b>θ</b> = {np.degrees(theta):.1f}°<br>"
            f"<b>Ring:</b> {'Inner' if ring_idx == 0 else 'Outer'}<br>"
            f"<b>Cooldown:</b> {cooldown} frames<br>"
            f"<b>Stim events:</b> {len(self.logic.stim_events)}<br>"
            f"<b>Transitions:</b> {self.logic.transition_count}<br>"
            f"<b>Active ROIs:</b> {len(self._rois)}"
        )
        self.info_text.setText(info)

    def process_events(self):
        self.app.processEvents()

    def close(self):
        self.window.close()
        logger.info("RingVisualizer closed")
