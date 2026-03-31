"""
BCI Visualization Logic for CLEF.

Displays real-time neural traces, trial progress, ground truth text,
and decoded predictions using PyQtGraph.
"""

import logging
import time
from typing import Any, ClassVar, Dict, Optional

import numpy as np

from core.logic.BaseClosedLoopLogic import BaseClosedLoopLogic
from utils.style.demo_stylization import DemoStyle

logger = logging.getLogger(__name__)

N_CHANNELS = 256


class BCIVisualizationLogic(BaseClosedLoopLogic):
    logic_class: ClassVar[Optional[str]] = "bci_visualization_logic"

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
            name=name, config=config, output_devices=output_devices,
            gui_parameters=gui_parameters, input_devices=input_devices,
            io_manager=io_manager, config_manager=config_manager,
        )
        self.input_device_name = next(iter(self.input_devices), None)
        self.channel_step = config.get("channel_step", 8)

        self.app = None
        self.win = None
        self.trace_plot = None
        self.curves = []
        self.decoded_label = None
        self.ground_truth_label = None
        self.progress_label = None
        self.trial_info_label = None
        self.speed_label = None

        self.trial_buf = None
        self.bin_count = 0
        self.current_transcription = ""
        self.input_device = None

        # Timing for speed calculation
        self.trial_start_wall_time = 0.0
        self.trial_start_bin_idx = 0

    def initialize_model(self):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        DemoStyle.load_qss(self.app)

        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle("BCI Speech Decoder - CLEF")
        self.win.resize(1000, 600)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)
        self.win.setLayout(main_layout)

        # ── Header row: trial info + speed ──
        header_layout = QtWidgets.QHBoxLayout()
        self.trial_info_label = QtWidgets.QLabel("Waiting for decoder to load...")
        self.trial_info_label.setStyleSheet(DemoStyle.MUTED_TEXT_STYLE)
        header_layout.addWidget(self.trial_info_label)
        header_layout.addStretch()
        self.speed_label = QtWidgets.QLabel("")
        self.speed_label.setStyleSheet(DemoStyle.MUTED_TEXT_STYLE)
        header_layout.addWidget(self.speed_label)
        main_layout.addLayout(header_layout)

        # ── Text display: decoded (top) + ground truth (bottom) ──
        text_group = QtWidgets.QGroupBox("Speech Decoding")
        text_group.setStyleSheet(DemoStyle.GROUP_BOX_STYLE)
        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setSpacing(6)

        pred_heading = DemoStyle.make_heading("Predicted", QtWidgets)
        text_layout.addWidget(pred_heading)
        self.decoded_label = QtWidgets.QLabel("(waiting for data)")
        self.decoded_label.setWordWrap(True)
        self.decoded_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #FFC107; "
            "padding: 8px; background: #f8f8f8; border: 1px solid #ddd; border-radius: 4px;"
        )
        self.decoded_label.setMinimumHeight(50)
        text_layout.addWidget(self.decoded_label)

        truth_heading = DemoStyle.make_heading("Ground Truth", QtWidgets)
        text_layout.addWidget(truth_heading)
        self.ground_truth_label = QtWidgets.QLabel("-")
        self.ground_truth_label.setWordWrap(True)
        self.ground_truth_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #2196F3; "
            "padding: 8px; background: #f8f8f8; border: 1px solid #ddd; border-radius: 4px;"
        )
        self.ground_truth_label.setMinimumHeight(50)
        text_layout.addWidget(self.ground_truth_label)

        text_group.setLayout(text_layout)
        main_layout.addWidget(text_group)

        # ── Progress bar ──
        self.progress_label = QtWidgets.QLabel("Progress: -")
        self.progress_label.setStyleSheet(DemoStyle.MUTED_TEXT_STYLE)
        main_layout.addWidget(self.progress_label)

        main_layout.addWidget(DemoStyle.make_separator(QtWidgets))

        # ── Neural traces plot (compact) ──
        self.trace_plot = pg.PlotWidget(title="Neural Activity (subsampled)")
        self.trace_plot.setLabel("bottom", "Time bin")
        self.trace_plot.setLabel("left", "Channel")
        self.trace_plot.showGrid(x=True, y=True, alpha=0.3)
        main_layout.addWidget(self.trace_plot, stretch=1)

        n_display = N_CHANNELS // self.channel_step
        colors_broca = pg.mkColor(100, 180, 255, 150)
        colors_premotor = pg.mkColor(255, 150, 100, 150)
        for i in range(n_display):
            ch = i * self.channel_step
            color = colors_broca if ch < 128 else colors_premotor
            curve = self.trace_plot.plot(pen=pg.mkPen(color, width=1))
            self.curves.append(curve)

        self.win.show()
        self.app.processEvents()

        if self.input_device_name and self.io_manager:
            self.input_device = self.io_manager.get_input_device(self.input_device_name)

        logger.info("BCIVisualizationLogic initialized")

    def _reset_for_trial(self):
        total_bins = self.input_device.total_bins
        self.trial_buf = np.zeros((total_bins, N_CHANNELS), dtype=np.float32)
        self.bin_count = 0
        self.trial_start_wall_time = time.time()
        self.trial_start_bin_idx = 0
        self.trace_plot.setXRange(0, total_bins, padding=0)
        for curve in self.curves:
            curve.setData([], [])

    def process_sample(self, sample: Any):
        if self.input_device is None:
            return

        frame = None
        if isinstance(sample, dict) and self.input_device_name:
            frame = sample.get(self.input_device_name)

        # Detect trial change
        if self.input_device.current_transcription != self.current_transcription:
            self.current_transcription = self.input_device.current_transcription
            self.ground_truth_label.setText(self.current_transcription)
            self.decoded_label.setText("(waiting for data)")
            self.decoded_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #FFC107; "
                "padding: 8px; background: #f8f8f8; border: 1px solid #ddd; border-radius: 4px;"
            )
            self._reset_for_trial()

        # Update progress + speed
        if self.input_device.total_bins > 0:
            trial_num = self.input_device.current_trial_list_idx + 1
            total_trials = len(self.input_device.trials)
            bin_idx = self.input_device.current_bin_idx
            total_bins = self.input_device.total_bins
            pct = int(100 * bin_idx / total_bins)
            bar_len = 30
            filled = int(bar_len * bin_idx / total_bins)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
            self.progress_label.setText(
                f"Progress: {bar} bin {bin_idx}/{total_bins} ({pct}%)"
            )
            self.trial_info_label.setText(
                f"Trial {trial_num}/{total_trials} | Day {self.input_device.current_day_idx}"
            )

            # Speed: bins are 20ms each in real time
            elapsed_wall = time.time() - self.trial_start_wall_time
            if elapsed_wall > 0.1 and bin_idx > 0:
                simulated_time_s = bin_idx * 0.020  # 20ms per bin
                speed_ratio = simulated_time_s / elapsed_wall
                self.speed_label.setText(f"Speed: {speed_ratio:.1f}x realtime")

        # Accumulate into trial buffer and update traces
        if frame is not None and frame.shape == (N_CHANNELS,) and self.trial_buf is not None:
            if self.bin_count < self.trial_buf.shape[0]:
                self.trial_buf[self.bin_count] = frame
                self.bin_count += 1

                x = np.arange(self.bin_count)
                data = self.trial_buf[:self.bin_count]
                for i, curve in enumerate(self.curves):
                    ch = i * self.channel_step
                    curve.setData(x, data[:, ch] + i * 2.0)

        # Check for decoded text
        result = self.input_device.get_decoded_text()
        if result is not None:
            decoded, is_final = result
            if is_final:
                self.decoded_label.setText(decoded)
                self.decoded_label.setStyleSheet(
                    "font-size: 16px; font-weight: bold; color: #009900; "
                    "padding: 8px; background: #f8f8f8; border: 1px solid #ddd; border-radius: 4px;"
                )
                logger.info(f"Final decoded: '{decoded}'")
            else:
                self.decoded_label.setText(f"{decoded} ...")
                self.decoded_label.setStyleSheet(
                    "font-size: 16px; font-weight: bold; color: #FFC107; "
                    "padding: 8px; background: #f8f8f8; border: 1px solid #ddd; border-radius: 4px;"
                )

        self.app.processEvents()

    def close(self):
        if self.win is not None:
            self.win.close()
        logger.info("BCIVisualizationLogic closed")
