"""
XYTrackingWorker for CLEF2.

GUI subprocess for real-time centroid-based XY stage tracking.
Reads camera frames from shared memory, computes centroid,
and writes stage correction offsets into shared_stage_offset_xy.
"""

import logging
import time
import warnings
import numpy as np
from multiprocessing import Process, shared_memory

from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

from clef2.utils.style.demo_stylization import DemoStyle

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class XYTrackingWorker(Process):

    def __init__(self, child_conn, vis_args):
        super().__init__()
        self.child_conn = child_conn
        self.vis_args = vis_args

        self.ysize = vis_args.get("ysize")
        self.xsize = vis_args.get("xsize")
        self.zsize = vis_args.get("zsize", 1)
        self.dtype = vis_args.get("dtype", np.uint16)

        # Tracking parameters
        self.micron_to_pix_ratio = vis_args.get("micron_to_pix_ratio", 100.0 / 74.0)
        self.stage_dampening_factor = vis_args.get("stage_dampening_factor", 0.5)
        camera_binning = vis_args.get("camera_binning", "1x1")
        if camera_binning == "2x2":
            self.camera_binning_multiplier = 2
        else:
            self.camera_binning_multiplier = 1

        # Frame center (tracking target)
        self.cy = self.ysize // 2
        self.cx = self.xsize // 2

        # SHM references (attached in initialize_shm)
        self.shared_frame_memory_list = []
        self.img_list = []
        self.shared_image_count = None
        self.shared_stage_offset_xy = None
        self.image_count = 0
        self.first_img_flag = True

    def initialize_shm(self):
        """Attach to camera frame SHM, image count, and stage offset SHM."""
        # Frame buffers
        shm_names = self.vis_args.get("shm_names", [])
        for z in range(self.zsize):
            name = shm_names[z] if z < len(shm_names) else f"shared_frame_memory_{z}"
            shm = shared_memory.SharedMemory(name=name)
            self.shared_frame_memory_list.append(shm)
            self.img_list.append(
                np.ndarray((self.ysize, self.xsize), dtype=self.dtype, buffer=shm.buf)
            )

        # Image count
        image_count_name = self.vis_args.get("image_count_shm_name", "shared_frame_memory_image_count")
        self.shared_image_count = shared_memory.ShareableList(name=image_count_name)

        # Stage offset
        stage_shm_name = self.vis_args.get("stage_shm_name", "shared_stage_offset_xy")
        self.shared_stage_offset_xy = shared_memory.ShareableList(name=stage_shm_name)

    def initialize_display(self):
        self.initialize_shm()

        self.app = pg.Qt.mkQApp(name="XYTrackingWorker")
        DemoStyle.load_qss(self.app)

        # Main window
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("XY Stage Tracking")
        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setSpacing(8)

        # Image viewer
        self.graphics_layout_widget = pg.GraphicsLayoutWidget()
        self.vb = pg.ViewBox(
            lockAspect=True, enableMouse=True, border=None, enableMenu=True
        )
        self.vb.setLimits(xMin=0, xMax=self.ysize, yMin=0, yMax=self.xsize)
        self.vb.setBorder({"color": DemoStyle.COLOR_NEUTRAL, "width": 2})

        self.ii = pg.ImageItem()
        self.ii.setImage(self.img_list[0])
        self.vb.addItem(self.ii)

        # Centroid marker
        self.centroid_marker = pg.ScatterPlotItem(
            x=[], y=[], pen=None,
            brush=pg.mkBrush(DemoStyle.COLOR_DANGER),
            size=10, symbol="+"
        )
        self.vb.addItem(self.centroid_marker)

        self.graphics_layout_widget.addItem(self.vb, row=0, col=0)

        # Histogram
        self.lut_histo = pg.HistogramLUTItem(
            image=self.ii, fillHistogram=False,
            orientation="vertical", levelMode="mono"
        )
        self.lut_histo.setLevels(90, 150)
        self.lut_histo.setHistogramRange(60, 200)
        self.graphics_layout_widget.addItem(self.lut_histo, row=0, col=1)

        main_layout.addWidget(self.graphics_layout_widget, stretch=1)

        # Control panel
        control_group = QtWidgets.QGroupBox("Stage Tracking")
        control_group.setStyleSheet(DemoStyle.GROUP_BOX_STYLE)
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(4)

        self.enable_tracking_button = QtWidgets.QPushButton("Enable Stage Tracking")
        self.enable_tracking_button.setCheckable(True)
        self.enable_tracking_button.setFixedHeight(28)
        self.enable_tracking_button.setStyleSheet(
            f"background-color: {DemoStyle.COLOR_NEUTRAL}"
        )
        self.enable_tracking_button.toggled.connect(self._tracking_button_callback)
        control_layout.addWidget(self.enable_tracking_button)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        self.window.resize(800, 700)
        self.window.show()

    def _tracking_button_callback(self, checked):
        if checked:
            self.enable_tracking_button.setStyleSheet(
                f"background-color: {DemoStyle.COLOR_WARNING}"
            )
        else:
            self.enable_tracking_button.setStyleSheet(
                f"background-color: {DemoStyle.COLOR_NEUTRAL}"
            )

    def compute_centroid(self, frame):
        """Compute centroid of thresholded frame (dark object on light background)."""
        threshold = np.percentile(frame, 1.0)
        binary = frame < threshold
        ys, xs = np.nonzero(binary)
        if len(ys) == 0:
            return self.cy, self.cx
        return float(np.mean(ys)), float(np.mean(xs))

    def update_stage_offset(self, centroid_y, centroid_x):
        """Compute and write stage correction to shared memory."""
        dy = self.cy - int(centroid_y)
        dx = self.cx - int(centroid_x)

        correction_x = int(dx * self.micron_to_pix_ratio * self.camera_binning_multiplier * self.stage_dampening_factor)
        correction_y = int(dy * self.micron_to_pix_ratio * self.camera_binning_multiplier * self.stage_dampening_factor)

        self.shared_stage_offset_xy[0] = correction_y
        self.shared_stage_offset_xy[1] = correction_x

    def update_display(self):
        frame = self.img_list[0].T
        frame = np.flip(frame, axis=1)

        auto_levels = self.first_img_flag
        if self.first_img_flag:
            self.first_img_flag = False

        self.ii.setImage(frame, autoLevels=auto_levels)

        # Compute centroid
        cy, cx = self.compute_centroid(frame)
        self.centroid_marker.setData(x=[cx], y=[cy])

        # Apply tracking if enabled
        if self.enable_tracking_button.isChecked():
            self.update_stage_offset(cy, cx)

    def run(self):
        try:
            self.initialize_display()
            self.app.processEvents()
            self.image_count = self.shared_image_count[0]

            while True:
                if self.child_conn.poll():
                    msg = self.child_conn.recv()
                    if msg == "close":
                        break

                try:
                    if self.image_count == self.shared_image_count[0]:
                        self.app.processEvents()
                        continue
                except ValueError:
                    continue

                self.update_display()

                try:
                    self.image_count = self.shared_image_count[0]
                except ValueError:
                    self.image_count = self.shared_image_count[0]

                self.app.processEvents()

        except EOFError:
            logger.warning("XYTrackingWorker pipe closed (EOF)")
        except BrokenPipeError:
            logger.warning("XYTrackingWorker pipe closed (BrokenPipe)")
        except Exception as err:
            logger.error(f"XYTrackingWorker error: {err}")
            raise
