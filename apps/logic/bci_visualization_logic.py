"""
BCI Visualization Logic for CLEF.

Displays real-time neural traces, trial progress, ground truth text,
and decoded predictions using PyQtGraph.
"""

import logging
import math
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

        # Adaptive decode interval state
        self._latency_ema = None
        self._last_emitted_interval = None
        self._last_emitted_pause = None
        self._last_adaptive_emit_time = None
        self._min_decode_interval = None  # set from config on first use
        self._min_inter_trial_pause = None

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
        self.latency_label = QtWidgets.QLabel("")
        self.latency_label.setStyleSheet(DemoStyle.MUTED_TEXT_STYLE + " font-family: monospace;")
        header_layout.addWidget(self.latency_label)
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

        # Update latency display
        if hasattr(self.input_device, 'decoder_mode') and self.input_device.decoder_mode == "cloud":
            decode_ms = self.input_device.last_decode_latency_ms
            rt_ms = self.input_device.last_roundtrip_ms
            d = f"{decode_ms:6.0f}" if decode_ms is not None else "     -"
            r = f"{rt_ms:6.0f}" if rt_ms is not None else "     -"
            n = f"{rt_ms - decode_ms:6.0f}" if (rt_ms is not None and decode_ms is not None) else "     -"
            self.latency_label.setText(f"Cloud | decode:{d}ms | roundtrip:{r}ms | network:{n}ms")
        else:
            if hasattr(self, 'latency_label'):
                self.latency_label.setText("Local (SHM)")

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

    def _get_output_device_name(self) -> Optional[str]:
        if self.output_devices:
            return next(iter(self.output_devices.keys()))
        return None

    def _check_logic(self) -> Optional[Dict[str, Any]]:
        """Adaptively adjust decode_interval_bins based on observed decode latency."""
        if self.input_device is None:
            return None
        if not hasattr(self.input_device, 'decoder_mode') or self.input_device.decoder_mode != "cloud":
            return None

        decode_ms = self.input_device.last_decode_latency_ms
        if decode_ms is None:
            return None

        # Capture config values as minimums on first use
        if self._min_decode_interval is None:
            self._min_decode_interval = self.input_device.decode_interval_bins
            self._min_inter_trial_pause = self.input_device.inter_trial_pause_s

        # EMA smoothing
        if self._latency_ema is None:
            self._latency_ema = decode_ms
        else:
            self._latency_ema = 0.5 * decode_ms + 0.5 * self._latency_ema

        # Each bin = 1000/playback_rate_hz ms. Need interval >= decode_time / bin_period
        bin_period_ms = 1000.0 / self.input_device.playback_rate_hz
        target_interval = int(math.ceil(self._latency_ema / bin_period_ms * 1.2))
        target_interval = max(target_interval, self._min_decode_interval)

        # Inter-trial pause: enough for one final decode
        target_pause = max(self._latency_ema / 1000.0 * 1.5, self._min_inter_trial_pause)

        # Throttle: emit at most once per decode cycle
        now = time.time()
        if self._last_adaptive_emit_time is not None:
            elapsed = now - self._last_adaptive_emit_time
            if elapsed < self._latency_ema / 1000.0:
                return None

        # Only emit when values change
        interval_changed = self._last_emitted_interval != target_interval
        pause_changed = self._last_emitted_pause is None or abs(target_pause - self._last_emitted_pause) > 0.5

        if interval_changed or pause_changed:
            self._last_emitted_interval = target_interval
            self._last_emitted_pause = target_pause
            self._last_adaptive_emit_time = now
            output_name = self._get_output_device_name()
            if output_name:
                logger.info(
                    f"Adaptive: decode_interval_bins={target_interval}, "
                    f"inter_trial_pause_s={target_pause:.1f} (latency_ema={self._latency_ema:.0f}ms)"
                )
                return {
                    output_name: {
                        "decode_interval_bins": target_interval,
                        "inter_trial_pause_s": target_pause,
                    }
                }
        return None

    def close(self):
        if self.win is not None:
            self.win.close()
        logger.info("BCIVisualizationLogic closed")
