"""
BrainalyzerWorker for CLEF2.

GUI subprocess for real-time visualization and interactive stimulus control.
Reads camera frames from shared memory, displays them, and sends stimulus
events back to BrainalyzerLogic via Pipe.

Generates polygon masks directly and writes them into the polygon output
device's shared memory buffer (polygon_mask_buffer).
"""

import time
import logging
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, shared_memory

from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

from clef2.utils import numba_utils
from clef2.utils.style.demo_stylization import DemoStyle

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class BrainalyzerWorker(Process):

    def __init__(self, child_conn, vis_args):
        super().__init__()

        self.vis_args = vis_args
        self.child_conn = child_conn

        # general params
        self.initialization_t0 = time.time()
        self.first_img_flag = True
        self.image_count = 0
        self.stim_refractory_vols = 0
        self.stim_duration_vols = self.vis_args.get("stim_duration_vols", 4)
        self.stimulus_is_on = False
        self.image_downsample = 1
        self.refractory_constant = 1
        self.rec_id = self.vis_args.get("id", "test")
        self.saveroot = self.vis_args.get("saveroot", "")
        self.dtype = self.vis_args.get("dtype", np.uint16)
        self.ysize = self.vis_args.get("ysize", None)
        self.xsize = self.vis_args.get("xsize", None)
        self.zsize = self.vis_args.get("zsize", 1)
        self.stim_diameter = self.vis_args.get("stim_diameter", 30)
        self.stim_intensity = self.vis_args.get("stim_intensity", 10)
        self.GUI_mode = self.vis_args.get("GUI_mode", "neural_imaging")
        self.camera_binning = self.vis_args.get("camera_binning", "1x1")
        self.gui_screenshot_freq = self.vis_args.get("gui_screenshot_freq", 0)
        self.num_rois_added = 0
        self.stim_cmap_list = [np.array((255, 0, 0), dtype=np.uint8)]
        self.shared_frame_memory_list = []
        self.img_list = []
        self.quant_roi_dict_list = []
        self.stim_roi_dict_list = []
        self.vbox_z_label_list = []
        self.stop_rendering_image = False

        # Polygon mask shared memory params (from vis_args)
        self.polygon_shm_name = self.vis_args.get("polygon_shm_name", None)
        self.polygon_width = self.vis_args.get("polygon_width", None)
        self.polygon_height = self.vis_args.get("polygon_height", None)
        self.camera_roi = self.vis_args.get("camera_roi", (0, 0, 0, 0))

        # Calibration
        self.calibration_points = self.vis_args.get("calibration_points", None)

        # Polygon shared memory (attached in initialize_shm)
        self._polygon_shm = None
        self._polygon_mask_array = None


    def initialize_display(self):

        # initialize shared memory objects
        self.initialize_shm()

        # initialize qt app
        self.app = pg.Qt.mkQApp(name="BrainalyzerWorker")
        DemoStyle.load_qss(self.app)

        # ── Image viewer ────────────────────────────────────────────
        self.graphics_layout_widget = pg.GraphicsLayoutWidget()
        self.image_vbox_list = []
        self.ii_list = []
        for z in range(self.zsize):

            lab = pg.LabelItem(text="z={}".format(z))
            self.vbox_z_label_list.append(lab)
            self.graphics_layout_widget.addItem(lab, row=0, col=z)

            if z == 0:
                vb = pg.ViewBox(lockAspect=True, enableMouse=True, name="first_viewbox", border=None, enableMenu=True)
            else:
                vb = pg.ViewBox(lockAspect=True, enableMouse=True, border=None, enableMenu=True)
                vb.linkView(vb.XAxis, "first_viewbox")
                vb.linkView(vb.YAxis, "first_viewbox")

            vb.setLimits(xMin=0, xMax=self.ysize, yMin=0, yMax=self.xsize)
            vb.setBorder({"color": DemoStyle.COLOR_NEUTRAL, "width": 2})

            ii = BrainalyzerImageItem(imageitem_id=z)
            ii.setImage(self.img_list[z])

            vb.addItem(ii)
            self.graphics_layout_widget.addItem(vb, row=1, col=z)
            self.image_vbox_list.append(vb)
            self.ii_list.append(ii)

        self.graphics_layout_widget.scene().sigMouseClicked.connect(self.graphics_layout_mouse_clicked_callback)

        # histogram
        self.lut_histo = pg.HistogramLUTItem(image=self.ii_list[0], fillHistogram=False, orientation="vertical", levelMode="mono")
        self.histo_sig_proxy = pg.SignalProxy(self.lut_histo.sigLevelsChanged, rateLimit=3, slot=self.update_image_LUTs)
        self.lut_histo.setLevels(90, 150)
        self.lut_histo.setHistogramRange(60, 200)
        self.graphics_layout_widget.addItem(self.lut_histo, row=1, col=self.zsize)

        # ── ROI intensity plot ──────────────────────────────────────
        self.roi_plot_item = pg.PlotItem()
        self.roi_plot_item.disableAutoRange()
        self.roi_plot_item.setRange(xRange=[0, 100], yRange=[90, 180])
        self.roi_plot_item.setLabel(axis="left", units="avg grey count")
        self.roi_plot_item.setLabel(axis="bottom", units="frame (vols)")
        self.roi_plot_item.setTitle("roi intensity")

        colspan = (self.zsize + 1) // 2
        if self.GUI_mode == "neural_imaging":
            self.graphics_layout_widget.addItem(self.roi_plot_item, row=2, col=0, rowspan=1, colspan=colspan)

        # ── Command panel — 3 grouped columns ──────────────────────
        command_panel = QtWidgets.QHBoxLayout()
        command_panel.setSpacing(8)

        # --- Stimulus Control group ---
        stim_group = QtWidgets.QGroupBox("Stimulus Control")
        stim_group.setStyleSheet(DemoStyle.GROUP_BOX_STYLE)
        stim_layout = QtWidgets.QVBoxLayout()
        stim_layout.setSpacing(4)

        self.enter_stimulus_intensity_button = DemoStyle.make_action_button(
            "stim intensity (%): {}".format(self.stim_intensity), QtWidgets)
        self.enter_stimulus_intensity_button.clicked.connect(self.my_get_stimulus_intensity)
        stim_layout.addWidget(self.enter_stimulus_intensity_button)

        self.enter_stim_duration_vols_button = DemoStyle.make_action_button(
            "stimulus duration (vols): {}".format(self.stim_duration_vols), QtWidgets)
        self.enter_stim_duration_vols_button.clicked.connect(self.my_get_stimulus_duration_in_vols)
        stim_layout.addWidget(self.enter_stim_duration_vols_button)

        stim_layout.addWidget(DemoStyle.make_separator(QtWidgets))

        stim_layout.addWidget(DemoStyle.make_heading("Pulse Stimulus", QtWidgets, style=DemoStyle.SUBHEADING_STYLE))

        self.pulse_stimulus_rois_button = DemoStyle.make_action_button(
            "pulse stimulate ROI(s)", QtWidgets, color=DemoStyle.COLOR_SUCCESS)
        self.pulse_stimulus_rois_button.clicked.connect(self.pulse_stimulus_rois)
        stim_layout.addWidget(self.pulse_stimulus_rois_button)

        self.pulse_full_field_button = DemoStyle.make_action_button(
            "pulse stimulate full-field", QtWidgets, color=DemoStyle.COLOR_SUCCESS)
        self.pulse_full_field_button.clicked.connect(self.pulse_full_field)
        stim_layout.addWidget(self.pulse_full_field_button)

        stim_layout.addStretch()
        stim_group.setLayout(stim_layout)
        command_panel.addWidget(stim_group)

        # --- Quantification ROIs group ---
        quant_group = QtWidgets.QGroupBox("Quantification ROIs")
        quant_group.setStyleSheet(DemoStyle.GROUP_BOX_STYLE)
        quant_layout = QtWidgets.QVBoxLayout()
        quant_layout.setSpacing(4)

        self.quant_roi_combobox = pg.ComboBox(items=[str(x) for x in range(self.zsize)])
        self.quant_roi_combobox.setEditable(True)
        self.quant_roi_combobox_lineEdit = self.quant_roi_combobox.lineEdit()
        self.quant_roi_combobox_lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.quant_roi_combobox_lineEdit.setReadOnly(True)
        self.quant_roi_combobox.textActivated.connect(self.update_add_delete_quant_roi_button_text)
        quant_layout.addWidget(self.quant_roi_combobox)

        self.add_quant_roi_button = DemoStyle.make_action_button("add quant roi to z-plane 0", QtWidgets)
        self.add_quant_roi_button.clicked.connect(self.add_quant_roi_to_image)
        quant_layout.addWidget(self.add_quant_roi_button)

        self.delete_quant_roi_button = DemoStyle.make_action_button("delete quant roi from z-plane 0", QtWidgets)
        self.delete_quant_roi_button.clicked.connect(self.delete_quant_roi_from_image)
        quant_layout.addWidget(self.delete_quant_roi_button)

        quant_layout.addStretch()
        quant_group.setLayout(quant_layout)
        command_panel.addWidget(quant_group)

        # --- Stimulation ROIs group ---
        stim_roi_group = QtWidgets.QGroupBox("Stimulation ROIs")
        stim_roi_group.setStyleSheet(DemoStyle.GROUP_BOX_STYLE)
        stim_roi_layout = QtWidgets.QVBoxLayout()
        stim_roi_layout.setSpacing(4)

        self.stim_roi_combobox = pg.ComboBox(items=[str(x) for x in range(self.zsize)])
        self.stim_roi_combobox.setEditable(True)
        self.stim_roi_combobox_lineEdit = self.stim_roi_combobox.lineEdit()
        self.stim_roi_combobox_lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.stim_roi_combobox_lineEdit.setReadOnly(True)
        self.stim_roi_combobox.textActivated.connect(self.update_add_delete_stim_roi_button_text)
        stim_roi_layout.addWidget(self.stim_roi_combobox)

        self.add_stim_roi_button = DemoStyle.make_action_button("add stim roi to z-plane 0", QtWidgets)
        self.add_stim_roi_button.clicked.connect(self.add_stim_roi_to_image)
        stim_roi_layout.addWidget(self.add_stim_roi_button)

        self.delete_stim_roi_button = DemoStyle.make_action_button("remove stim roi from z-plane 0", QtWidgets)
        self.delete_stim_roi_button.clicked.connect(self.delete_stim_roi_from_image)
        stim_roi_layout.addWidget(self.delete_stim_roi_button)

        stim_roi_layout.addStretch()
        stim_roi_group.setLayout(stim_roi_layout)
        command_panel.addWidget(stim_roi_group)

        # ── Status bar ──────────────────────────────────────────────
        status_bar = QtWidgets.QHBoxLayout()

        self.vol_count_text = QtWidgets.QLabel("received vols:        0")
        self.vol_count_text.setFont(QtGui.QFont("Monospace"))
        self.vol_count_text.setMinimumWidth(200)
        status_bar.addWidget(self.vol_count_text)

        status_bar.addStretch()

        self.stim_refractory_text = QtWidgets.QLabel(
            "stim refractory period: {}".format(self.stim_refractory_vols)
        )
        self.stim_refractory_text.setAlignment(QtCore.Qt.AlignRight)
        status_bar.addWidget(self.stim_refractory_text)

        status_widget = QtWidgets.QWidget()
        status_widget.setLayout(status_bar)
        status_widget.setStyleSheet(DemoStyle.INFO_BOX_STYLE)

        # ── Main layout ─────────────────────────────────────────────
        self.grid_layout = QtWidgets.QVBoxLayout()
        self.grid_layout.addWidget(self.graphics_layout_widget, stretch=1)
        self.grid_layout.addLayout(command_panel)
        self.grid_layout.addWidget(status_widget)

        self.window = QtWidgets.QWidget()
        self.window.resize(1400, 800)
        self.window.setLayout(self.grid_layout)
        self.window.show()

    def initialize_shm(self):
        """Attach to frame shared memory buffers and polygon mask buffer."""
        # Frame buffers (created by BrainalyzerLogic)
        for z in range(self.zsize):
            shm = shared_memory.SharedMemory(name="shared_frame_memory_{}".format(z))
            self.shared_frame_memory_list.append(shm)
            self.img_list.append(
                np.ndarray((self.ysize, self.xsize), dtype=self.dtype, buffer=shm.buf)
            )

        # Shared image count
        self.shared_image_count = shared_memory.ShareableList(name="shared_image_count")

        # Polygon mask buffer (created by MightexPolygonOutput)
        if self.polygon_shm_name and self.polygon_width and self.polygon_height:
            try:
                self._polygon_shm = shared_memory.SharedMemory(
                    name=self.polygon_shm_name, create=False
                )
                self._polygon_mask_array = np.ndarray(
                    shape=(self.polygon_height, self.polygon_width),
                    dtype=np.uint8,
                    buffer=self._polygon_shm.buf,
                )
                logger.info(
                    f"Attached to polygon mask buffer '{self.polygon_shm_name}' "
                    f"({self.polygon_width}x{self.polygon_height})"
                )
            except Exception as e:
                logger.warning(f"Could not attach to polygon mask buffer: {e}")
                self._polygon_shm = None
                self._polygon_mask_array = None

    def run(self):
        try:
            self.initialize_display()
            self.app.processEvents()
            self.image_count = self.shared_image_count[0]

            # main loop — runs until parent sends "close" on the pipe
            try:
                while True:
                    # Check for close signal from parent
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

                    self.update_dashboard()

                    try:
                        self.image_count = self.shared_image_count[0]
                    except ValueError:
                        self.image_count = self.shared_image_count[0]

                    self.app.processEvents()

            except Exception as err:
                logging.error("Error while running BrainalyzerWorker: {}".format(err))
                raise
        except EOFError as err:
            logging.warning("Visualization multiprocess socket closed EOF {}".format(err))
        except BrokenPipeError as err:
            logging.warning("Visualization multiprocess socket closed BrokenPipe {}".format(err))
        except Exception as err:
            logging.error("Unknown error in BrainalyzerWorker: {}".format(err))
        finally:
            self.close()

    def close(self):
        # close IPC
        self.shared_image_count.shm.close()
        for z in range(self.zsize):
            self.shared_frame_memory_list[z].close()

        if self._polygon_shm is not None:
            self._polygon_shm.close()

        self.child_conn.close()

    #####################################################################################
    # Mask generation — writes directly into polygon shared memory

    def generate_and_write_mask(self, event):
        """Generate a polygon mask from event data and write into shared memory."""
        if self._polygon_mask_array is None:
            logger.warning("Polygon mask buffer not available, cannot generate mask")
            return

        event_type = event.get("event_type")
        calib = self.calibration_points
        if calib is None:
            logger.warning("No calibration points, cannot generate mask")
            return

        pcx = np.array(calib["pcx"])
        pcy = np.array(calib["pcy"])
        icx = np.array(calib["icx"])
        icy = np.array(calib["icy"])
        xoffset = self.camera_roi[0]
        yoffset = self.camera_roi[1]
        width = self.polygon_width
        height = self.polygon_height

        if event_type in ("circle-click", "circle-button", "hammer-of-dawn"):
            cx = int(event["x"])
            cy = int(event["y"])
            diameter = int(event.get("stim_diameter", 20))

            mask = numba_utils.generate_pg_ellipse_mask(
                cx, cy, pcx, pcy, icx, icy,
                diameter, xoffset, yoffset, width, height
            )
            mask = mask * 255

        elif event_type in ("pulse-rect-roi-list", "stream-rect-roi-list"):
            stim_rect_roi_list = event.get("stim_rect_roi_list", {})
            x_list = np.array(stim_rect_roi_list.get("x", []))
            y_list = np.array(stim_rect_roi_list.get("y", []))
            width_list = np.array(stim_rect_roi_list.get("width", []))
            height_list = np.array(stim_rect_roi_list.get("height", []))

            mask = numba_utils.generate_pg_multi_rectangle_mask(
                x_list, y_list, width_list, height_list,
                pcx, pcy, icx, icy,
                xoffset, yoffset, width, height
            )
            mask = mask * 255

        elif event_type == "full-field-button":
            mask = np.ones((height, width), dtype=np.uint8) * 255

        else:
            logger.warning(f"Unknown polygon event type: {event_type}")
            return

        # Write mask into polygon shared memory
        self._polygon_mask_array[:] = mask.astype(np.uint8)
        logger.debug(f"Wrote mask for {event_type} into polygon shared memory")

    def blank_polygon_mask(self):
        """Zero out the polygon mask buffer."""
        if self._polygon_mask_array is not None:
            self._polygon_mask_array[:] = 0

    #####################################################################################
    # callbacks

    def graphics_layout_mouse_clicked_callback(self, event):
        items = self.graphics_layout_widget.scene().items(event.scenePos())

        selected_z = None
        for item in items:
            if isinstance(item, BrainalyzerImageItem):
                selected_z = item.imageitem_id
                self.stim_roi_combobox.setValue(str(selected_z))
                self.quant_roi_combobox.setValue(str(selected_z))
                self.update_add_delete_quant_roi_button_text(None)
                self.update_add_delete_stim_roi_button_text(None)
                break

        if selected_z is not None:
            self.update_viewbox_border_color(selected_z)

    def my_get_stimulus_intensity(self, event):
        self.stimulus_intensity_dialog = QtWidgets.QInputDialog()
        self.stimulus_intensity_dialog.setInputMode(QtWidgets.QInputDialog.IntInput)
        self.stimulus_intensity_dialog.setIntMinimum(0)
        self.stimulus_intensity_dialog.setIntMaximum(100)
        self.stimulus_intensity_dialog.intValueSelected.connect(self.set_stimulus_intensity)
        self.stimulus_intensity_dialog.show()

    def set_stimulus_intensity(self, intensity):
        self.stim_intensity = intensity
        self.enter_stimulus_intensity_button.setText(
            "stim intensity (%): {}".format(intensity)
        )

    def update_dashboard(self):
        self.update_display_images()
        self.update_display_text()
        self.update_refractory_counter()

        if self.GUI_mode == "neural_imaging":
            self.update_quant_roi_plot()

        self._maybe_save_screenshot()

    def _maybe_save_screenshot(self):
        if not self.gui_screenshot_freq or self.image_count % self.gui_screenshot_freq != 0:
            return
        try:
            screenshot_dir = Path(self.saveroot) / "gui_screenshot"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            path = screenshot_dir / f"screenshot_{self.image_count:06d}.png"
            pixmap = self.window.grab()
            pixmap.save(str(path), "PNG")
            logger.debug(f"Saved GUI screenshot: {path}")
        except Exception as err:
            logger.warning(f"Failed to save GUI screenshot at frame {self.image_count}: {err}")

    def update_refractory_counter(self):
        if self.image_count % self.zsize == 0:
            if self.stim_refractory_vols >= 1:
                self.stim_refractory_vols -= 1
            if self.stim_refractory_vols == 1:
                self.check_stim_refractory_vols()

    def update_display_text(self):
        if self.image_count % self.zsize == 0:
            self.vol_count_text.setText(
                "received vols: {:>8d}".format(self.image_count // self.zsize)
            )
            self.stim_refractory_text.setText(
                "stim refractory vols: {}".format(self.stim_refractory_vols)
            )

    def update_display_images(self):
        z_ndx = self.image_count % self.zsize

        if self.first_img_flag:
            auto_levels = True
            self.first_img_flag = False
        else:
            auto_levels = False

        if not self.stop_rendering_image:
            self.ii_list[z_ndx].setImage(
                self.img_list[z_ndx][::self.image_downsample, ::self.image_downsample],
                autoLevels=auto_levels,
            )

        self.image_count = self.shared_image_count[0]

    def update_image_LUTs(self):
        lev = self.lut_histo.getLevels()
        for z in range(self.zsize):
            self.ii_list[z].setLevels(lev)

    def get_random_pen_color(self):
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        return color

    def add_quant_roi_to_image(self, event):
        z_ndx = self.quant_roi_combobox.value()
        if not z_ndx:
            print("please select a z-plane to add roi to")
            return
        z_ndx = int(z_ndx)

        roi_id = hash(datetime.now())
        pen_color = self.get_random_pen_color()
        quant_roi = pg.RectROI(
            [self.ysize // 2, self.xsize // 2],
            [40, 40],
            pen=pg.mkPen(pen_color, width=2),
        )
        self.num_rois_added += 1

        roi_dict = {
            "roi_id": roi_id,
            "quant_roi": quant_roi,
            "z_ndx": z_ndx,
            "color": pen_color,
            "roi_dataitem": self.roi_plot_item.plot([], pen=pg.mkPen(pen_color, width=2)),
            "xvals": [],
            "yvals": [],
            "yvals_derivs": [],
        }

        self.quant_roi_dict_list.append(roi_dict)
        self.image_vbox_list[z_ndx].addItem(quant_roi)

    def add_stim_roi_to_image(self, event):
        z_ndx = self.stim_roi_combobox.value()
        if not z_ndx:
            print("please select a z-plane to add roi to")
            return
        z_ndx = int(z_ndx)

        roi_id = hash(datetime.now())
        pen_color = self.stim_cmap_list[len(self.stim_roi_dict_list) % len(self.stim_cmap_list)]
        stim_roi = pg.RectROI(
            [self.ysize // 2, self.xsize // 2],
            [40, 40],
            pen=pg.mkPen(pen_color, width=2, style=QtCore.Qt.DotLine),
        )

        roi_dict = {
            "roi_id": roi_id,
            "stim_roi": stim_roi,
            "z_ndx": z_ndx,
            "color": pen_color,
        }

        self.stim_roi_dict_list.append(roi_dict)
        self.image_vbox_list[z_ndx].addItem(stim_roi)

    def build_stim_roi_event(self):
        event_data = {"x": [], "y": [], "width": [], "height": []}
        for roi_dict in self.stim_roi_dict_list:
            y, x = roi_dict["stim_roi"].pos()
            width, height = roi_dict["stim_roi"].size()
            event_data["x"].append(int(x))
            event_data["y"].append(int(y))
            event_data["width"].append(int(width))
            event_data["height"].append(int(height))
        return event_data

    def pulse_stimulus_rois(self, event):
        if self.GUI_mode == "neural_imaging":
            if len(self.stim_roi_dict_list) == 0:
                print("Please add stimulus rois before trying to stimulate!")
                return
            event_type = "pulse-rect-roi-list"
            event_data = self.build_stim_roi_event()
        elif self.GUI_mode == "behavior":
            event_type = "stream-widefield"
            event_data = {}
        else:
            return

        self.write_event(event_type, event_data)

    def pulse_full_field(self, event):
        self.write_event("full-field-button", None)

    def stream_stimulus_rois(self):
        if self.GUI_mode == "neural_imaging":
            if len(self.stim_roi_dict_list) == 0:
                print("Please add stimulus rois before trying to stimulate!")
                return
            event_type = "pulse-rect-roi-list"
            event_data = self.build_stim_roi_event()
        elif self.GUI_mode == "behavior":
            event_type = "stream-widefield"
            event_data = {}
        else:
            return

        self.write_event(event_type, event_data)

    def write_event(self, event_type, event_data):
        """Build event dict, generate mask, write to polygon shm, send via pipe."""
        if self.stim_refractory_vols > 0:
            return

        data = {
            "ts": time.time(),
            "event_type": event_type,
            "stim_intensity": self.stim_intensity,
        }

        if event_type in ("pulse-rect-roi-list", "full-field-button"):
            data["stim_duration_vols"] = int(self.stim_duration_vols)

        if event_type in ("pulse-rect-roi-list", "stream-rect-roi-list"):
            data["stim_rect_roi_list"] = event_data

        # Generate mask and write directly into polygon shared memory
        self.generate_and_write_mask(data)

        # Send event to parent (BrainalyzerLogic) for timing/coordination
        self.child_conn.send(data)

        # visual aid if stim is on — fixed duration update refractory counter
        if event_type in ("pulse-rect-roi-list", "full-field-button"):
            self.stim_refractory_vols = self.stim_duration_vols + self.refractory_constant
            self.check_stim_refractory_vols()

    def change_stim_buttons_color(self, active):
        DemoStyle.set_button_state_color(self.pulse_stimulus_rois_button, active)
        DemoStyle.set_button_state_color(self.pulse_full_field_button, active)

    def check_stim_refractory_vols(self):
        if self.stim_refractory_vols > 1:
            self.change_stim_buttons_color(False)
        else:
            self.change_stim_buttons_color(True)

    def update_quant_roi_plot(self):
        curr_vol = self.image_count // self.zsize

        for roi_dict in self.quant_roi_dict_list:
            roi_z_ndx = roi_dict["z_ndx"]
            if self.image_count % self.zsize != roi_z_ndx:
                continue

            try:
                arr_mean = roi_dict["quant_roi"].getArrayRegion(
                    self.img_list[roi_z_ndx], self.ii_list[roi_z_ndx]
                ).mean()

                roi_dict["xvals"].append(curr_vol)
                roi_dict["yvals"].append(arr_mean)

                if len(roi_dict["yvals"]) == 1:
                    roi_dict["yvals_derivs"].append(0)
                else:
                    roi_dict["yvals_derivs"].append(roi_dict["yvals"][-1] - roi_dict["yvals"][-2])

                roi_dict["roi_dataitem"].setData(roi_dict["xvals"], roi_dict["yvals"])

            except ValueError as err:
                print("Error while trying to update ROI plot: {}".format(err))

        xstart = max(0, curr_vol - 100)
        self.roi_plot_item.setRange(xRange=[xstart, curr_vol + 1])

    def update_add_delete_quant_roi_button_text(self, event):
        self.add_quant_roi_button.setText("add quant roi to z-plane: {}".format(self.quant_roi_combobox.value()))
        self.delete_quant_roi_button.setText("delete quant roi from z-plane: {}".format(self.quant_roi_combobox.value()))
        self.update_viewbox_border_color(int(self.quant_roi_combobox.value()))

    def update_add_delete_stim_roi_button_text(self, event):
        self.add_stim_roi_button.setText("add stim roi to z-plane: {}".format(self.stim_roi_combobox.value()))
        self.delete_stim_roi_button.setText("delete stim roi from z-plane: {}".format(self.stim_roi_combobox.value()))
        self.update_viewbox_border_color(int(self.stim_roi_combobox.value()))

    def update_viewbox_border_color(self, selected_z):
        for z in range(self.zsize):
            if z == selected_z:
                self.image_vbox_list[z].setBorder({"color": DemoStyle.COLOR_WARNING, "width": 2})
            else:
                self.image_vbox_list[z].setBorder({"color": DemoStyle.COLOR_NEUTRAL, "width": 2})
            self.image_vbox_list[z].update()

    def delete_quant_roi_from_image(self, event):
        z_to_cleanse = int(self.quant_roi_combobox.value())

        roi_to_delete = []
        for r in range(len(self.quant_roi_dict_list)):
            roi_z_ndx = self.quant_roi_dict_list[r]["z_ndx"]
            if z_to_cleanse != roi_z_ndx:
                continue
            roi_to_delete.append(self.quant_roi_dict_list[r])

        for roi_dict in roi_to_delete:
            self.roi_plot_item.removeItem(roi_dict["roi_dataitem"])
            self.image_vbox_list[roi_dict["z_ndx"]].removeItem(roi_dict["quant_roi"])
            self.quant_roi_dict_list.remove(roi_dict)

    def delete_stim_roi_from_image(self, event):
        z_to_cleanse = int(self.stim_roi_combobox.value())

        roi_to_delete = []
        for r in range(len(self.stim_roi_dict_list)):
            roi_z_ndx = self.stim_roi_dict_list[r]["z_ndx"]
            if z_to_cleanse != roi_z_ndx:
                continue
            roi_to_delete.append(self.stim_roi_dict_list[r])

        for roi_dict in roi_to_delete:
            self.image_vbox_list[roi_dict["z_ndx"]].removeItem(roi_dict["stim_roi"])
            self.stim_roi_dict_list.remove(roi_dict)

    def my_get_stimulus_duration_in_vols(self):
        self.stimulus_duration_vols_dialog = QtWidgets.QInputDialog()
        self.stimulus_duration_vols_dialog.setInputMode(QtWidgets.QInputDialog.IntInput)
        self.stimulus_duration_vols_dialog.intValueSelected.connect(self.set_stim_delay_volumes)
        self.stimulus_duration_vols_dialog.show()

    def set_stim_delay_volumes(self, dur):
        self.enter_stim_duration_vols_button.setText(
            "stimulus duration (vols): {}".format(dur)
        )
        self.stim_duration_vols = dur


class BrainalyzerImageItem(pg.ImageItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imageitem_id = kwargs.get("imageitem_id", None)
