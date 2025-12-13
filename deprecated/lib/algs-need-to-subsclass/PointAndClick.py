# for alg
import numpy as np
import scipy.ndimage
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import tifffile as tf
import os
import cv2
from datetime import datetime

# for qtvisualizer
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph.opengl as gl
from matplotlib import cm
import pyqtgraph as pg
import numpy as np
from multiprocessing import Process, Pipe, SimpleQueue
import sys
import os

# registration
from image_registration import cross_correlation_shifts

# from skimage.feature import register_translation
# from skimage.registration import phase_cross_correlation
# import imreg_dft as ird

# ignore numpy warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# conditional import, necessary for standalone testing
try:
    from lib import wbliveUtils as utils
except ImportError as err:

    # change path and try again
    import wbliveUtils as utils


# logging.basicConfig(level=logging.DEBUG)
sns.set_style("darkgrid")


class PointAndClick:
    def __init__(self, args):

        # general alg params
        self.args = args

        # self.structural_scan_dir = args["structural_scan_dir"]
        self.rec_id = self.args["id"]
        self.roi = self.args["roi"]
        self.frames_to_grab = args["gooey_args"]["total_frames"]
        self.zsize = args["gooey_args"]["zsize"]
        self.xsize = args["roi"][2]
        self.ysize = args["roi"][3]

        # handle user-provided alg params. should clean up and do better because these may be ignored based on specifics of alg
        self.stim_intensity_ops = args["gooey_args"]["stim_intensity_options"]
        self.frames_to_stimulate_for_options = args["gooey_args"][
            "frames_to_stimulate_for_options"
        ]

        # alg-specific params or useful data structures
        self.stim_diameter = int(args["gooey_args"]["stimulus_diameter"])
        total_frames = int(args["gooey_args"]["total_frames"])
        self.vis_args = {
            "stim_diameter": self.stim_diameter,
            "ysize": self.ysize,
            "xsize": self.xsize,
            "total_frames": total_frames,
            "zsize": self.zsize,
            "stim_duration_vols": self.frames_to_stimulate_for_options[0] // self.zsize,
            "stim_intensity": self.stim_intensity_ops[0],
        }
        self.blank_image = np.zeros((self.ysize, self.xsize), dtype=np.uint16)
        self.current_image = self.blank_image
        self.events = []
        self.current_event = None

        # algs might want to generate semantic warnings based on user input to flag user to potentially undesired behavior
        self.parse_alg_semantic_warnings()

        # output holder vars
        # self.live_data = []
        # self.raw_signal_list = []
        # self.skipped_stimulation_list = []
        # self.delayed_stimulation_list = []
        self.process_time_list = []
        self.model_time_list = []
        self.stim_param_list = []

    # whatever setup needs to be done with model
    def initialize_model(self):

        # for reproducible psudorandomness
        fname_root = self.args["id"]
        random.seed(fname_root)

        # spawn subprocess to handle rendering
        self.vis = QtVisualizer(self.vis_args)

    # function to grab algorithm arguments, spitting out semantic warnings (i.e. based on input arguments, may lead to different behavior than what the user expects)
    def parse_alg_semantic_warnings(self):
        pass

    def get_metadata(self, args=None):
        """Get metadata which we might want to save for later e.g. for
        writing to file"""

        metadata = {
            # "live_data": self.live_data,
            # "raw_signal": self.raw_signal_list,
            # "skipped_stimulation_list": self.skipped_stimulation_list,
            "process_time_list": self.process_time_list,
            "model_time_list": self.model_time_list,
            "stim_param_list": self.stim_param_list,
            # "delayed_stimulation_list": self.delayed_stimulation_list,
            # important metadata specific to this alg
            # "user_provided_stim_onsets": self.user_provided_stim_onsets,
        }

        return metadata

    # we often have a period at the beginning of the expr where we need to collect more data before we can start doing things
    def skip(self):

        # just do nothing, preserve indexing or intermediate variable states
        # self.live_data.append(0)
        # self.raw_signal_list.append(0)
        self.model_time_list.append(0)

    def process_frame(self, img, zndx):

        # start timer
        process_tstart = time.time()

        # set current image state
        if self.zsize > 1:

            # add image to mip
            self.current_image = np.maximum(self.current_image, img)

        else:
            self.current_image = img

        # send image off as long as we're not in the middle of zscan
        if zndx == self.zsize - 1:
            self.vis.update_display_images(self.current_image)

            # if zsize > 1, reset current image
            if self.zsize > 1:
                self.current_image = self.blank_image

        # end and append timer
        process_tend = time.time()
        self.process_time_list.append(process_tend - process_tstart)

    def process_volume(self):

        # start timer
        model_tstart = time.time()

        # just place some zeros for continuity
        # self.raw_signal_list.append(0)
        # self.live_data.append(0)

        # get events from gui object
        data = self.vis.get_event()

        # incorporate events into internal state
        if data is not None:
            self.events.append(data)
            self.current_event = data

        # end and append timer
        model_tend = time.time()
        self.model_time_list.append(model_tend - model_tstart)

    # check for data
    def check_stim(self, image_ndx, cooldown_counter):

        # to keep consistent with outter loop
        # img_count = image_ndx + 1

        # initialize output var
        stim_params = {}
        new_cooldown = 0

        # grab current measurement
        if self.current_event is not None:

            # get stim params from randomly initialized lists or from event
            # stim_intensity = random.choice(self.stim_intensity_ops)
            stim_intensity = self.current_event["stim_intensity"]
            # num_stim_frames = random.choice(self.frames_to_stimulate_for_options)

            # add data to stim params
            # trigger stim_on on next volume
            zndx = image_ndx % self.zsize

            # if we've done a full volume which happens on zndx=zsize-1
            stim_on = image_ndx + self.zsize - zndx

            # trigger event-based stimulation one volume later to allow for polygon to update
            stim_on += self.zsize

            # add delay if need be
            stim_on += self.current_event["stim_delay_vols"] * self.zsize

            # set number of stim in frames
            num_stim_frames = self.current_event["stim_duration_vols"] * self.zsize

            # build stim parameters
            stim_params["stim_on"] = stim_on
            stim_params["stim_off"] = stim_on + num_stim_frames
            stim_params["stim_intensity"] = stim_intensity
            stim_params["event"] = self.current_event

            # reset current event
            self.current_event = None

            # if we had a stimulation
            if stim_params:

                # set new cooldown timeout, in this case we're ignoring
                # new_cooldown = self.frames_to_cool_down_after_stim
                new_cooldown = 0

                # check if recording is about to end, in which case we don't want to stimulate
                if stim_params["stim_off"] > self.frames_to_grab:

                    # remove stimulus info
                    stim_params = {}

                self.stim_param_list.append(stim_params)

        return stim_params, new_cooldown

    def plot_model(self, show_plot=False, savefilename=None):
        """Plot current state of model"""

        logging.warning("There is no alg model to plot!")

    def close(self):
        self.vis.close()


class QtVisualizer:
    def __init__(self, vis_args):

        try:
            # create a pipe to visualizer
            self.parent_conn, self.child_conn = Pipe()

            # object to hold incoming events
            self.events = []

            # make queue if it's not already there
            # if "queue" not in vis_args.keys():
            #    self.q = SimpleQueue()
            #    vis_args["queue"] = self.q
            # else:
            #    self.q = vis_args["queue"]

            # create process and start it
            self.proc = QtVisualizerWorker(self.child_conn, vis_args)
            self.proc.start()

        except Exception as err:
            logging.error("Problem while initializing QtVisualizer: {}".format(err))
            raise (err)

    def update_display_images(self, payload):

        # set ranges of image
        self.parent_conn.send(payload)

    def get_event(self):

        """
        if not self.q.empty():

            # return data from subprocess
            data = self.q.get()

            # append to internal data structure for bookkeeping
            self.events.append(data)

            return data

        else:

            return None
        """
        if self.parent_conn.poll():

            data = self.parent_conn.recv()

            # append to internal data structure for bookkeeping
            self.events.append(data)

            return data

    def close(self):
        # self.parent_conn.send("Lollll closing")
        self.parent_conn.close()


class QtVisualizerWorker(Process):
    def __init__(self, child_conn, vis_args):

        # call superconstructor
        super(QtVisualizerWorker, self).__init__()
        self.child_conn = child_conn
        self.vis_args = vis_args

        # flag for if the visualizer has received any input yet
        # useful for triggering scaling to first image
        self.first_img_flag = False

        # for global motion estimation
        self.last_image = None
        self.globmo_xoff = 0
        self.globmo_yoff = 0

        # stimulus refractory period
        self.cl_stim_delay_ops = [0]
        self.event_queue = []

        # stimulus delay
        self.stim_duration_vols = vis_args.get("stim_duration_vols")

        # stimulus intensity, usually in %
        self.stim_intensity = vis_args.get("stim_intensity", 20)

        # template matcher
        self.template_filter = gaussian_2d(19, 3)

        # refractory counter after stim
        self.stim_refractory_vols = 0

        # constant value after which control of sending
        # events is reenabled
        self.refractory_constant = 1

        # elements
        self.xsize = self.vis_args.get("xsize")
        self.ysize = self.vis_args.get("ysize")
        self.zsize = self.vis_args.get("zsize")
        self.total_vols = self.vis_args.get("total_frames") / self.zsize
        self.stim_diameter = int(self.vis_args.get("stim_diameter", 10))
        # self.q = self.vis_args.get("queue", None)
        self.img_count = 0

        # set stdout for debugging from subprocess
        # sys.stdout = open(str('QtVisualizerWorker') + ".stout", "a")
        # sys.stderr = open(str('QtVisualizerWorker') + ".stderr", "a")

    def initialize_display(self):

        # make app
        self.app = QtGui.QApplication([])

        # make graphics widget
        graphics = pg.GraphicsLayoutWidget()

        #####################
        # prep real-time display of images

        # Get the colormap
        # maxcolor = 255
        # colormap = cm.get_cmap("magma")
        # colormap._init()
        # lut = (colormap._lut * maxcolor).view(
        #    np.ndarray
        # )  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # create view box for image item with scrolling locked
        box = pg.ViewBox(lockAspect=True, enableMouse=False)

        # create image item, maintain reference to update later
        ii = pg.ImageItem()

        # adjust image showing to be row-major and origin in top left, removing need for transpose
        ii.setOpts(axisOrder="row-major")
        box.invertY()

        # Apply the colormap
        # ii.setLookupTable(lut, update=False)

        # add image item to box
        box.addItem(ii)

        # make histogram lookup widget
        lut_histo = pg.HistogramLUTItem(image=ii, fillHistogram=False)
        # lut_histo.disableAutoHistogramRange()

        ##################################
        # mouse click for stimulations
        self.enable_mouse_clicks_for_stim = False

        if self.enable_mouse_clicks_for_stim:

            # add cross hair to box
            vLine = pg.InfiniteLine(angle=90, movable=False, pen={"alpha": 0.5})
            hLine = pg.InfiniteLine(angle=0, movable=False, pen={"alpha": 0.5})
            box.addItem(vLine, ignoreBounds=False)
            box.addItem(hLine, ignoreBounds=False)

            # render graphic object at mouse cursor
            cursor_graphic = QtGui.QGraphicsEllipseItem()
            cursor_pen = QtGui.QPen()
            cursor_pen.setWidth(2)
            cursor_pen.setColor(QtGui.QColor("green"))
            cursor_pen.setStyle(QtCore.Qt.DotLine)
            cursor_graphic.setPen(cursor_pen)
            box.addItem(cursor_graphic, ignoreBounds=False)

            # map events
            ii.hoverEvent = self.my_hover_event
            ii.mouseClickEvent = self.my_click_event

            # set internal refs
            self.vLine = vLine
            self.hLine = hLine
            self.cursor_graphic = cursor_graphic

        else:

            # create stim roi circle
            stim_roi = pg.CircleROI(
                [self.xsize // 1.9, self.ysize // 1.9],
                radius=self.stim_diameter / 2,
                pen=pg.mkPen(color="r", alpha=0.3, width=4),
            )

            # add stim roi to image
            box.addItem(stim_roi)

            # internal ref
            self.stim_roi = stim_roi

        ###################################

        # add view box to graphic
        graphics.addItem(box)

        # add histogram lut to graphic
        graphics.addItem(lut_histo)

        #######################
        # set data plots
        # create layout for roi data plots
        graphics_plots = pg.GraphicsLayout()

        # add plotitem for roi data
        roi_plot_item = pg.PlotItem()
        roi_plot_item.disableAutoRange()
        roi_plot_item.setRange(xRange=[0, self.total_vols // 2], yRange=[90, 180])
        roi_plot_dataitem = roi_plot_item.plot([])
        graphics_plots.addItem(roi_plot_item)

        # add plotitem for roi deriv data
        roi_plot_deriv_item = pg.PlotItem()
        roi_plot_deriv_item.disableAutoRange()
        roi_plot_deriv_item.setRange(xRange=[0, self.total_vols], yRange=[-20, 30])
        roi_plot_deriv_dataitem = roi_plot_deriv_item.plot([])
        graphics_plots.addItem(roi_plot_deriv_item)

        # set horizontal line on deriv plot and add to plot
        self.closedloop_pos_thresh = 35.0
        self.closedloop_neg_thresh = -25.0
        self.closedloop_pos_thresh_line = roi_plot_deriv_item.addLine(
            x=None, y=self.closedloop_pos_thresh, pen=pg.mkPen("r"), movable=False
        )
        self.closedloop_neg_thresh_line = roi_plot_deriv_item.addLine(
            x=None, y=self.closedloop_neg_thresh, pen=pg.mkPen("b"), movable=False
        )

        # add internal refs
        self.roi_plot_item = roi_plot_item
        self.roi_plot_dataitem = roi_plot_dataitem
        self.roi_plot_deriv_item = roi_plot_deriv_item
        self.roi_plot_deriv_dataitem = roi_plot_deriv_dataitem

        # internal list to store incoming data items
        self.roi_xvals = []
        self.roi_yvals = []
        self.roi_deriv_yvals = []

        ##########
        # set command panel layout
        command_panel_layout = QtGui.QGridLayout()

        # buttons
        enable_moco_button = QtGui.QPushButton("enable moco")
        enable_cl_stimulation_pos_button = QtGui.QPushButton(
            "enable closed-loop stimulation (+ threshold)"
        )
        enable_cl_stimulation_neg_button = QtGui.QPushButton(
            "enable closed-loop stimulation (- threshold)"
        )
        enter_cl_thresh_button = QtGui.QPushButton(
            "set closed-loop (+/-) threshold: {}/{}".format(
                self.closedloop_pos_thresh, self.closedloop_neg_thresh
            )
        )
        roi_stim_button = QtGui.QPushButton("stimulate ROI")
        full_field_stim_button = QtGui.QPushButton("stimulate full-field")
        enter_cl_stim_delay_ops_button = QtGui.QPushButton(
            "CL stim delay frames options: {}".format(self.cl_stim_delay_ops)
        )
        enter_stim_duration_vols_button = QtGui.QPushButton(
            "set stimulus duration (vols): {}".format(self.stim_duration_vols)
        )
        clear_event_queue_button = QtGui.QPushButton("clear event queue")
        enable_cl_events_clear_event_queue_button = QtGui.QPushButton(
            "CL events clear event queue"
        )
        enter_stimulus_intensity_button = QtGui.QPushButton(
            "stim intensity (%): {}".format(self.stim_intensity)
        )

        # additional tweaks
        enable_moco_button.setCheckable(True)
        # enable_moco_button.setChecked(True)
        enable_cl_stimulation_pos_button.setCheckable(True)
        enable_cl_stimulation_neg_button.setCheckable(True)
        enable_cl_events_clear_event_queue_button.setCheckable(True)
        roi_stim_button.setStyleSheet("background-color: green")
        full_field_stim_button.setStyleSheet("background-color: green")

        # add to layout
        command_panel_layout.addWidget(enable_moco_button, 1, 1)
        command_panel_layout.addWidget(enable_cl_stimulation_pos_button, 2, 1)
        command_panel_layout.addWidget(enable_cl_stimulation_neg_button, 3, 1)
        command_panel_layout.addWidget(enter_cl_thresh_button, 1, 2)
        command_panel_layout.addWidget(enter_cl_stim_delay_ops_button, 2, 2)
        command_panel_layout.addWidget(enter_stim_duration_vols_button, 3, 2)
        command_panel_layout.addWidget(roi_stim_button, 1, 3)
        command_panel_layout.addWidget(full_field_stim_button, 2, 3)
        command_panel_layout.addWidget(clear_event_queue_button, 3, 3)
        command_panel_layout.addWidget(enable_cl_events_clear_event_queue_button, 4, 3)
        command_panel_layout.addWidget(enter_stimulus_intensity_button, 4, 2)

        # map events
        roi_stim_button.clicked.connect(self.my_roi_stim_button_event)
        full_field_stim_button.clicked.connect(self.my_full_field_stim_button_event)
        enter_cl_thresh_button.clicked.connect(self.my_get_closedloop_threshold_event)
        enter_cl_stim_delay_ops_button.clicked.connect(
            self.my_get_closedloop_delay_options_event
        )
        enter_stim_duration_vols_button.clicked.connect(
            self.my_get_stimulus_duration_in_vols
        )
        clear_event_queue_button.clicked.connect(self.clear_event_queue_event)
        enter_stimulus_intensity_button.clicked.connect(self.my_get_stimulus_intensity)

        # set internal refs
        self.full_field_stim_button = full_field_stim_button
        self.roi_stim_button = roi_stim_button
        self.enable_moco_button = enable_moco_button
        self.enable_cl_stimulation_pos_button = enable_cl_stimulation_pos_button
        self.enable_cl_stimulation_neg_button = enable_cl_stimulation_neg_button
        self.enter_cl_thresh_button = enter_cl_thresh_button
        self.enter_cl_stim_delay_ops_button = enter_cl_stim_delay_ops_button
        self.enter_stim_duration_vols_button = enter_stim_duration_vols_button
        self.enable_cl_events_clear_event_queue_button = (
            enable_cl_events_clear_event_queue_button
        )
        self.enter_stimulus_intensity_button = enter_stimulus_intensity_button

        ##############
        # add ROI to imageitem
        quant_roi = pg.RectROI(
            [self.xsize // 2, self.ysize // 2],
            [20, 20],
        )
        box.addItem(quant_roi)

        ################

        # create text items
        vol_count_text = QtWidgets.QLabel("frames: {}/{}".format(0, self.total_vols))
        stim_refractory_vols_text = QtWidgets.QLabel(
            "refractory vols: {}".format(self.stim_refractory_vols)
        )
        event_queue_text = QtWidgets.QLabel("event queue:<br>")
        last_stim_sent_text = QtWidgets.QLabel("last stim:")

        # add text items to text layout
        text = QtGui.QGridLayout()
        # text.setColumnStretch(0, 1)
        # text.setColumnStretch(1, 1)
        text.addWidget(vol_count_text, 0, 0)
        text.addWidget(stim_refractory_vols_text, 0, 1)
        text.addWidget(event_queue_text, 0, 2)
        text.addWidget(last_stim_sent_text, 1, 1)

        # add plots to larger graphics window
        graphics.addItem(graphics_plots)

        # add graphics to overall layout
        grid = QtGui.QGridLayout()
        grid.addWidget(graphics)
        grid.addItem(command_panel_layout)
        grid.addItem(text)

        # add layout to window
        window = QtGui.QWidget()
        window.resize(1400, 600)
        window.setLayout(grid)

        # keep reference so that it doesn't get garbage collected
        self.ii = ii
        self.window = window
        self.box = box
        self.graphics = graphics
        self.grid = grid
        self.text = text
        self.vol_count_text = vol_count_text
        self.stim_refractory_vols_text = stim_refractory_vols_text
        self.event_queue_text = event_queue_text
        self.last_stim_sent_text = last_stim_sent_text
        self.quant_roi = quant_roi

        # show
        self.window.show()

    def update_display_images(self, img):

        # get minimum and maximum, should be faster than autoleveling in pyqtgraph
        # mn, mx = np.min(img), np.max(img)
        # img = ((img - mn) / (mx - mn) * 254).astype(np.uint8)

        # update gui
        self.ii.setImage(img, autoLevels=False)

    def update_display_text(self):

        # update frame count
        self.vol_count_text.setText(
            "received vols: {}/{}".format(self.img_count, self.total_vols)
        )

        # update refractory volumes
        self.stim_refractory_vols_text.setText(
            "refractory vols: {}".format(self.stim_refractory_vols)
        )

    def update_roi_plot(self, img, img_count):

        # grab roi pixel values
        arr = self.quant_roi.getArrayRegion(img, self.ii)
        arr_mean = arr.mean()

        # store xvals of trace
        self.roi_xvals.append(img_count)

        # store derivative of trace, done before storing normal yvalue to make life slightly easier
        if len(self.roi_deriv_yvals) == 0:
            self.roi_deriv_yvals = [0]
        else:
            self.roi_deriv_yvals.append(arr_mean - self.roi_yvals[-1])

        # store original yvale
        self.roi_yvals.append(arr_mean)

        # update plot
        # self.roi_plot_item.plot(self.roi_xvals, self.roi_yvals)
        self.roi_plot_dataitem.setData(self.roi_xvals, self.roi_yvals)

        # update deriv plot
        self.roi_plot_deriv_dataitem.setData(self.roi_xvals, self.roi_deriv_yvals)

    def run(self):

        try:
            # initialize display, has to happen outside of init because
            # qt objects aren't serializable
            self.initialize_display()
            # pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=callback)
            try:
                while 1:
                    # process event loop unless there's data to work with
                    while not self.child_conn.poll():
                        self.app.processEvents()

                    # get image and update display
                    img = self.child_conn.recv()
                    self.img_count += 1
                    self.update_roi_plot(img, self.img_count)

                    # process any queued up events
                    if len(self.event_queue) > 0:
                        self.process_event_queue()

                    # check refractory for stimulus,
                    if self.stim_refractory_vols > 0:
                        self.stim_refractory_vols -= 1

                        # reseting buttons coloring,
                        # hack: on refrac==1 to limit subfunction calls
                        if self.stim_refractory_vols == 1:
                            self.check_stim_refractory_vols()

                    # check closed loop
                    self.check_closedloop_thresh_cross()

                    # update display
                    self.update_display_images(img)
                    self.update_display_text()

                    # calculate drift and apply to rois
                    if self.enable_moco_button.isChecked():
                        self.apply_globmo_to_rois(img)

                    # remove viewbox padding, only need to do once with locked scrolling
                    # also autolevel image
                    if not self.first_img_flag:
                        self.first_img_flag = True
                        self.box.setRange(rect=self.ii.boundingRect(), padding=0)
                        self.ii.setImage(img)

                    # continue event loop
                    self.app.processEvents()

            except BrokenPipeError as err:
                logging.warning(
                    "Broken pipe error, was this because you closed?: {}".format(err)
                )
            except Exception as err:
                logging.warning("Error in PointAndClick: {}".format(err))
            finally:
                self.child_conn.close()

        except EOFError as err:
            logging.warning(
                "Visualization multiprocess socket has been closed {}".format(err)
            )

            # don't close the window right away, allow user to do so
            self.app.exec_()

        except BrokenPipeError as err:
            logging.warning(
                "Visualization multiprocess socket has been closed {}".format(err)
            )

        except ValueError as err:
            logging.warning("Invalid value received: {}".format(err))

    # events
    # see https://github.com/pyqtgraph/pyqtgraph/blob/develop/examples/imageAnalysis.py#L97
    # see http://www.pyqtgraph.org/downloads/0.10.0/pyqtgraph-0.10.0-deb/pyqtgraph-0.10.0/examples/crosshair.py
    def my_click_event(self, event):

        # get click position
        pos = event.pos()
        i = pos.y()
        j = pos.x()

        # get pixel position
        ppos = self.ii.mapToParent(pos)
        x = int(ppos.x())
        y = int(ppos.y())

        # set window title
        self.window.setWindowTitle("{}".format(data))

        # write event data to parent process
        self.write_event(
            data, event_type="circle-click", stim_diameter=self.stim_diameter, x=x, y=y
        )

    def write_event(self, event_type, stim_diameter=None, x=None, y=None):

        # get delay options
        # stim_delay_vols = random.choice(self.cl_stim_delay_ops)
        # stim_delay_vols = self.cl_stim_delay_ops[0]

        # don't write events if in refractory period
        if self.stim_refractory_vols > 0:
            return

        # if we have multiple delays, write first event and queue up seubsequent events with a flag for when to trigger them
        ts = time.time()

        # set event
        for op in range(len(self.cl_stim_delay_ops)):
            data = {
                "ts": ts,
                "event_type": event_type,
                "stim_delay_vols": int(self.cl_stim_delay_ops[op]),
                "stim_duration_vols": int(self.stim_duration_vols),
                "stim_intensity": self.stim_intensity,
            }

            # write conditional data
            if event_type == "circle-click" or event_type == "circle-button":
                data["stim_diameter"] = int(stim_diameter)
                data["x"] = int(x)
                data["y"] = int(y)

            # if we're writing multiple events, queue events after the first to send out later
            if op == 0:
                self.child_conn.send(data)

                # play a sound =)
                utils.play_wblive_sound("laser")

                # update last stim sent
                self.update_last_stim_sent_text(data)

            else:

                # set when event can be released from gui, one volume after previous delay
                event_embargo = (
                    self.img_count
                    + self.cl_stim_delay_ops[op - 1]
                    + self.stim_duration_vols
                    + self.refractory_constant
                )
                data["event_embargo"] = event_embargo

                # subtract embargo from delay to get stimulation events
                # which are appropriately spaced
                data["stim_delay_vols"] -= (
                    self.cl_stim_delay_ops[op - 1] + self.stim_duration_vols
                )

                # append to event queue
                self.event_queue.append(data)

        # change button colors and set cooldown
        self.stim_refractory_vols = (
            self.stim_duration_vols
            + self.cl_stim_delay_ops[0]
            + self.refractory_constant
        )
        self.check_stim_refractory_vols()

        # update event queue text display
        self.update_event_queue_display()

    def check_stim_refractory_vols(self):

        if self.stim_refractory_vols > 1:
            self.roi_stim_button.setStyleSheet("background-color: red")
            self.full_field_stim_button.setStyleSheet("background-color: red")
        else:
            self.roi_stim_button.setStyleSheet("background-color: green")
            self.full_field_stim_button.setStyleSheet("background-color: green")

    def my_hover_event(self, event):

        # get click position
        try:
            pos = event.pos()
        except AttributeError as err:
            return

        i = pos.y()
        j = pos.x()

        # get pixel position
        ppos = self.ii.mapToParent(pos)
        x = int(ppos.x())
        y = int(ppos.y())

        # set window title
        # window.setWindowTitle("Current position: y={} x={}".format(y, x))
        self.vLine.setPos(x)
        self.hLine.setPos(y)

        # add a marker in the middle of the lines
        # h = ii.height()
        # self.hLine.clearMarkers()
        # self.hLine.addMarker("o", position=x / self.xsize, size=self.stimdiameter * 2)
        # self.vLine.clearMarkers()
        # self.vLine.addMarker("o", position=y / self.ysize, size=self.stim_diameter)

        # draw shape at pointer
        self.cursor_graphic.setRect(
            x - self.stim_diameter // 2,
            y - self.stim_diameter // 2,
            self.stim_diameter,
            self.stim_diameter,
        )

    def my_full_field_stim_button_event(self, event):

        # update display
        # self.window.setWindowTitle("full-field button clicked")
        self.write_event("full-field-button")

    def my_roi_stim_button_event(self, event):

        # self.window.setWindowTitle("roi-stim button clicked")

        # get position of stim roi
        x, y = self.stim_roi.pos()

        # get radius of stim roi
        stim_diameter = self.stim_roi.size()[0]

        # circle roi .pos is corner of bounding box, adjust with radius
        x += stim_diameter / 2
        y += stim_diameter / 2

        # write data
        self.write_event("circle-button", x=x, y=y, stim_diameter=stim_diameter)

    def apply_globmo_to_rois(self, img, globmo_interval=10):

        # corner case for first image
        if self.last_image is None:

            self.last_image = image_filtering(
                img.astype(np.float32), template_filter=self.template_filter
            )

        else:
            # if interval
            if self.img_count % globmo_interval == 0:

                # filter image
                frame = image_filtering(
                    img.astype(np.float32), template_filter=self.template_filter
                )

                # calculate globmo
                xoff, yoff = cross_correlation_shifts(self.last_image, frame)

                # sum shifts over multiple timepoints
                self.globmo_xoff += xoff
                self.globmo_yoff += yoff

                # print("x: {}".format(self.globmo_xoff))
                # print("y: {}".format(self.globmo_yoff))

                # apply globmo to rois if it's greater than 1 px, then reset running sum
                if np.abs(self.globmo_xoff) > 1:
                    self.quant_roi.translate(self.globmo_xoff, 0, snap=False)
                    self.stim_roi.translate(self.globmo_xoff, 0, snap=False)
                    self.globmo_xoff = 0

                if np.abs(self.globmo_yoff) > 1:
                    self.quant_roi.translate(0, self.globmo_yoff, snap=False)
                    self.stim_roi.translate(0, self.globmo_yoff, snap=False)
                    self.globmo_yoff = 0

                # set next last image
                self.last_image = frame

    def my_get_closedloop_threshold_event(self, event):

        # grab value from user for float
        self.closedloop_thresh_dialog = QtGui.QInputDialog()

        # paramaterize dialog
        self.closedloop_thresh_dialog.setInputMode(QtGui.QInputDialog.DoubleInput)
        self.closedloop_thresh_dialog.setDoubleMinimum(-1000)
        self.closedloop_thresh_dialog.setDoubleMaximum(1000)

        # set event handler
        self.closedloop_thresh_dialog.doubleValueSelected.connect(
            self.set_closedloop_threshold
        )

        # show dialog
        self.closedloop_thresh_dialog.show()

    def my_get_stimulus_intensity(self, event):

        # grab value from user
        self.stimulus_intensity_dialog = QtGui.QInputDialog()

        # paramaterize dialog
        self.stimulus_intensity_dialog.setInputMode(QtGui.QInputDialog.IntInput)
        self.stimulus_intensity_dialog.setIntMinimum(0)
        self.stimulus_intensity_dialog.setIntMaximum(100)

        # set event handler
        self.stimulus_intensity_dialog.intValueSelected.connect(
            self.set_stimulus_intensity
        )

        # show dialog
        self.stimulus_intensity_dialog.show()

    def set_stimulus_intensity(self, intensity):

        # set internal var
        self.stim_intensity = intensity

        # set button text
        self.enter_stimulus_intensity_button.setText(
            "stim intensity (%): {}".format(intensity)
        )

    def set_closedloop_threshold(self, thresh):

        if thresh >= 0:
            # set internal var
            self.closedloop_pos_thresh = thresh

            # set deriv line
            self.closedloop_pos_thresh_line.setValue(self.closedloop_pos_thresh)

        else:
            self.closedloop_neg_thresh = thresh

            # set deriv line
            self.closedloop_neg_thresh_line.setValue(self.closedloop_neg_thresh)

        # set text of button if user hit ok
        self.enter_cl_thresh_button.setText(
            "set closed-loop (+/-) threshold: {}/{}".format(
                self.closedloop_pos_thresh, self.closedloop_neg_thresh
            )
        )

    def my_get_closedloop_delay_options_event(self):

        # grab value from user for float
        self.closedloop_delay_ops_dialog = QtGui.QInputDialog()

        # paramaterize dialog
        self.closedloop_delay_ops_dialog.setInputMode(QtGui.QInputDialog.TextInput)

        # set event handler
        self.closedloop_delay_ops_dialog.textValueSelected.connect(
            self.set_closedloop_delay_options
        )

        # show dialog
        self.closedloop_delay_ops_dialog.show()

    def set_closedloop_delay_options(self, txt):

        # split string into list of ints
        textlist = txt.split(",")
        try:

            opslist = [int(x) for x in textlist]

            # stort
            opslist.sort()

            # set internal var
            self.cl_stim_delay_ops = opslist

            # set button text
            self.enter_cl_stim_delay_ops_button.setText(
                "CL stim delay frames options: {}".format(self.cl_stim_delay_ops)
            )

        except Exception as err:
            logging.warning(
                "Error while trying to set closedloop delay options: value {} not neatly split into integers.".format(
                    txt
                )
            )

    def my_get_stimulus_duration_in_vols(self):

        # grab value from user for float
        self.stimulus_duration_vols_dialog = QtGui.QInputDialog()

        # paramaterize dialog
        self.stimulus_duration_vols_dialog.setInputMode(QtGui.QInputDialog.IntInput)

        # set event handler
        self.stimulus_duration_vols_dialog.intValueSelected.connect(
            self.set_stim_delay_volumes
        )

        # show dialog
        self.stimulus_duration_vols_dialog.show()

    def set_stim_delay_volumes(self, dur):
        self.enter_stim_duration_vols_button.setText(
            "set stimulus duration (vols): {}".format(dur)
        )
        self.stim_duration_vols = dur

    def check_closedloop_thresh_cross(self):

        # threshold cross flag
        thresh_cross = False

        # check positive threshold crosses
        if self.enable_cl_stimulation_pos_button.isChecked():

            # if threshold is positive, take highpass. if negative, take lowpass
            last_sample = self.roi_deriv_yvals[-1]
            if last_sample > self.closedloop_pos_thresh:

                thresh_cross = True

                # roi stim function, event arg doesn't matter
                self.my_roi_stim_button_event(0)

        # check negative
        if self.enable_cl_stimulation_neg_button.isChecked():
            last_sample = self.roi_deriv_yvals[-1]
            if last_sample < self.closedloop_neg_thresh:

                thresh_cross = True

                # roi stim function, event arg doesn't matter
                self.my_roi_stim_button_event(0)

        # if enabled, clear events queue
        if thresh_cross:
            if self.enable_cl_events_clear_event_queue_button.isChecked():
                self.clear_event_queue_event(0)

    def process_event_queue(self):

        # non-pythonic syntax because we gotta dynamically update indexing
        num_events = len(self.event_queue)
        e = 0
        while e < num_events and num_events != 0:

            data = self.event_queue[e]

            # if event embargo is over write event
            if data["event_embargo"] == self.img_count:
                self.child_conn.send(data)

                # play a sound =)
                utils.play_wblive_sound("laser")

                # update last tim sent
                self.update_last_stim_sent_text(data)

                # remove event from queue
                self.event_queue.pop(e)

                # num_events and don't increment counter
                num_events -= 1

                # update refractory counter
                self.stim_refractory_vols = (
                    data["stim_delay_vols"]
                    + self.stim_duration_vols
                    + self.refractory_constant
                )
                self.check_stim_refractory_vols()

            # otherwise increment e
            else:
                e += 1

        # update event queue display
        self.update_event_queue_display()

    def update_event_queue_display(self):

        to_display = ""
        for e in self.event_queue:
            release = e["event_embargo"]
            delay = e["stim_delay_vols"]
            duration = e["stim_duration_vols"]
            event_type = e["event_type"]
            to_display += "release: {}, delay: {}, duration: {}, type: {}<br>".format(
                release, delay, duration, event_type
            )

        # update text
        self.event_queue_text.setText("event queue: <br>{}".format(to_display))

    def clear_event_queue_event(self, event):

        # clear internal reference
        self.event_queue = []

        # update text
        self.update_event_queue_display()

    def update_last_stim_sent_text(self, data):

        # get timestamp and set qlabel text
        timestamp = datetime.fromtimestamp(data["ts"]).strftime("%H:%M:%S")
        to_display = "last stim: [{}, type={}, dur={}, delay={}]".format(
            timestamp,
            data["event_type"],
            data["stim_duration_vols"],
            data["stim_delay_vols"],
        )
        self.last_stim_sent_text.setText(to_display)


def image_filtering(
    frame,
    fb_post_threshold=50,
    fb_threshold_margin=50,
    med_filt_size=5,
    template_filter=None,
):
    """
    :param frame:
    :param args:
    :return:
    """

    # apply threshold
    threshold = np.median(frame) + fb_threshold_margin
    frame = (frame > threshold) * frame

    # apply median filter
    frame = cv2.medianBlur(frame, med_filt_size)

    # template filtering
    frame = cv2.matchTemplate(frame, template_filter, cv2.TM_CCOEFF)

    # pad frame b/c of template matching
    # p = (len(template_filter) // 2) + 1
    # frame = np.pad(frame, (p, 0), "constant")
    # frame = (frame > fb_post_threshold) * frame

    return frame


# gaussian 2d to generate template to match
def gaussian_2d(im_width, sigma):
    """

    :param im_width:
    :param sigma:
    :return:
    """

    g = np.zeros((im_width, im_width), dtype=np.float32)

    # gaussian filter
    for i in range(int(-(im_width - 1) / 2), int((im_width + 1) / 2)):
        for j in range(int(-(im_width - 1) / 2), int((im_width + 1) / 2)):
            x0 = int((im_width) / 2)  # center
            y0 = int((im_width) / 2)  # center
            x = i + x0  # row
            y = j + y0  # col
            g[y, x] = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2 / sigma / sigma)
            # g[x, y] = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2 / sigma / sigma)

    return g


# standalone testing
if __name__ == "__main__":

    # DEBUG = True
    DEBUG = False

    # load tiff file for streaming
    # fname = "C:/Users/rldun/data/RLD_TEMP_DATA_HOLDER/20201124_RLD_1/behavior/wormb_0atr_2/wormb_0atr_2_MMStack_Pos0.ome.tif"
    # fname = "C:/Users/rldun/Desktop/temp_render/20210104-00-58-02.tif"
    # fname = "C:/Users/rldun/Desktop/temp_render/MAX_20210103-21-29-18.tif"
    fname = "C:/Users/rldun/Desktop/temp_render/20210717-16-52-11.tiff"
    # fname = "E:/RLD/20220116_RLD_1/20220116-19-12-47/20220116-19-12-47.tiff"
    data = tf.imread(fname)

    # initialize subprocess
    fps = 5
    zsize = 12
    total_vols = 250
    vis_args = {
        "stim_diameter": 30,
        "ysize": data.shape[1],
        "xsize": data.shape[2],
        "total_frames": total_vols * zsize,
        "zsize": zsize,
        "stim_duration_vols": 4,
        "stim_intensity": 10,
    }
    vis = QtVisualizer(vis_args)

    # initialize visualizer with blank frame
    frame = data[0, :, :]
    vis.update_display_images(frame)

    #  display images for some amount of time
    # zmip = np.zeros((data.shape[1], data.shape[2], zsize), dtype=np.uint16)
    zmip = np.zeros((data.shape[1], data.shape[2]), dtype=np.uint16)
    if not DEBUG:

        # iterate volumes
        for i in range(0, total_vols - 1):

            # if linear indexing, take zmip
            for z in range(0, zsize):
                # zmip[:, :, z] = data[i * zsize + z, :, :]
                zmip = np.maximum(zmip, data[i * zsize + z, :, :])

            # mip = zmip.max(axis=2)
            # vis.update_display_images(mip)
            vis.update_display_images(zmip)

            # reset zmip
            zmip[:] = 0

            # get events from vis
            event = vis.get_event()
            if event is not None:
                print(event)

            # pause for fps sim
            if fps is not None:
                time.sleep(1 / fps)

    # close
    vis.close()
