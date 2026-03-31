"""
BCI Visualization Logic for CLEF.

Displays real-time neural traces, trial progress, ground truth text,
and decoded predictions using PyQtGraph.
"""

import logging
from typing import Any, ClassVar, Dict, Optional

import numpy as np

from core.logic.BaseClosedLoopLogic import BaseClosedLoopLogic

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
        self.ground_truth_label = None
        self.decoded_label = None
        self.progress_label = None
        self.trial_info_label = None

        self.trial_buf = None  # (total_bins, 256) allocated per trial
        self.bin_count = 0
        self.current_transcription = ""
        self.input_device = None

    def initialize_model(self):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets

        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])

        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle("BCI Speech Decoder - CLEF")
        self.win.resize(1200, 800)

        layout = QtWidgets.QVBoxLayout()
        self.win.setLayout(layout)

        # Trial info
        self.trial_info_label = QtWidgets.QLabel("Waiting for decoder to load...")
        self.trial_info_label.setStyleSheet("font-size: 14px; color: #888;")
        layout.addWidget(self.trial_info_label)

        # Ground truth
        self.ground_truth_label = QtWidgets.QLabel("Ground truth: -")
        self.ground_truth_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2196F3;")
        self.ground_truth_label.setWordWrap(True)
        layout.addWidget(self.ground_truth_label)

        # Progress
        self.progress_label = QtWidgets.QLabel("Progress: -")
        self.progress_label.setStyleSheet("font-size: 14px; color: #aaa;")
        layout.addWidget(self.progress_label)

        # Neural traces plot - fixed x-axis per trial
        self.trace_plot = pg.PlotWidget(title="Neural Activity (256 channels, subsampled)")
        self.trace_plot.setLabel("bottom", "Time bin")
        self.trace_plot.setLabel("left", "Channel")
        self.trace_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.trace_plot, stretch=3)

        # Create curves for subsampled channels
        n_display = N_CHANNELS // self.channel_step
        colors_broca = pg.mkColor(100, 180, 255, 150)    # blue for Area 44
        colors_premotor = pg.mkColor(255, 150, 100, 150)  # orange for Area 6v
        for i in range(n_display):
            ch = i * self.channel_step
            color = colors_broca if ch < 128 else colors_premotor
            curve = self.trace_plot.plot(pen=pg.mkPen(color, width=1))
            self.curves.append(curve)

        # Decoded text
        self.decoded_label = QtWidgets.QLabel("Decoded: (waiting for trial to complete)")
        self.decoded_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")
        self.decoded_label.setWordWrap(True)
        layout.addWidget(self.decoded_label)

        self.win.show()
        self.app.processEvents()

        # Get reference to input device
        if self.input_device_name and self.io_manager:
            self.input_device = self.io_manager.get_input_device(self.input_device_name)

        logger.info("BCIVisualizationLogic initialized")

    def _reset_for_trial(self):
        total_bins = self.input_device.total_bins
        self.trial_buf = np.zeros((total_bins, N_CHANNELS), dtype=np.float32)
        self.bin_count = 0
        # Fix x-axis to full trial length
        self.trace_plot.setXRange(0, total_bins, padding=0)
        # Clear curves
        for curve in self.curves:
            curve.setData([], [])

    def process_sample(self, sample: Any):
        if self.input_device is None:
            return

        # Get neural frame from input_stores
        frame = None
        if isinstance(sample, dict) and self.input_device_name:
            frame = sample.get(self.input_device_name)

        # Detect trial change
        if self.input_device.current_transcription != self.current_transcription:
            self.current_transcription = self.input_device.current_transcription
            self.ground_truth_label.setText(f"Ground truth: \"{self.current_transcription}\"")
            self.decoded_label.setText("Decoded: (waiting for trial to complete)")
            self._reset_for_trial()

        # Update progress
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

        # Accumulate into trial buffer and update traces
        if frame is not None and frame.shape == (N_CHANNELS,) and self.trial_buf is not None:
            if self.bin_count < self.trial_buf.shape[0]:
                self.trial_buf[self.bin_count] = frame
                self.bin_count += 1

                # Update traces up to current bin
                x = np.arange(self.bin_count)
                data = self.trial_buf[:self.bin_count]
                for i, curve in enumerate(self.curves):
                    ch = i * self.channel_step
                    curve.setData(x, data[:, ch] + i * 2.0)

        # Check for decoded text
        decoded = self.input_device.get_decoded_text()
        if decoded is not None:
            self.decoded_label.setText(f"Decoded: \"{decoded}\"")
            logger.info(f"Decoded: '{decoded}'")

        self.app.processEvents()

    def close(self):
        if self.win is not None:
            self.win.close()
        logger.info("BCIVisualizationLogic closed")
