# for alg
from multiprocessing.sharedctypes import Value
from xml.dom.expatbuilder import theDOMImplementation
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
import sys
import os
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph.opengl as gl
from matplotlib import cm
import pyqtgraph as pg
from multiprocessing import Process, Pipe, shared_memory
import time
import json

# ignore numpy warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# conditional import, necessary for standalone testing
# try:
# from lib import wbliveUtils as utils
# except ImportError as err:
# change path and try again
# import wbliveUtils as utils


class HammerOfDawn:
    def __init__(self, args):

        # get general params
        self.args = args
        self.rec_id = self.args["id"]
        self.roi = self.args["roi"]
        self.frames_to_grab = self.args["gooey_args"]["total_frames"]
        self.zsize = self.args["gooey_args"]["zsize"]
        self.xsize = self.args["roi"][2]
        self.ysize = self.args["roi"][3]
        self.dtype = self.args.get("dtype", np.uint16)

        # data transmission with subprocess ipc. default to shm
        self.ipc = self.args.get("data_ipc", "shared_memory")

        # holders
        self.events = []
        self.current_event = None
        self.stimulus_is_on = False
        self.process_time_list = []
        self.model_time_list = []
        self.stim_param_list = []

        # alg-specific params for subprocess
        self.stim_diameter = int(args["gooey_args"]["stimulus_diameter"])
        self.stim_intensity_ops = args["gooey_args"]["stim_intensity_options"]
        self.stim_intensity = self.stim_intensity_ops[0]
        self.vis_args = {
            "stim_diameter": self.stim_diameter,
            "ysize": self.ysize,
            "xsize": self.xsize,
            "total_frames": self.frames_to_grab,
            "zsize": self.zsize,
            # "stim_duration_vols": self.frames_to_stimulate_for_options[0] // self.zsize,
            "stim_intensity": self.stim_intensity,
            "data_ipc": self.ipc,
        }

        try:

            # create a pipe to visualizer
            self.parent_conn, self.child_conn = Pipe()

            # if ipc is shared memory, we only use the pipe for subprocess->main thread events
            if self.ipc == "shared_memory":
                self.shared_frame_memory = shared_memory.SharedMemory(
                    create=True,
                    size=self.ysize * self.xsize * 2,
                    name="shared_frame_memory",
                )
                self.shared_image_count = shared_memory.ShareableList(
                    [0], name="shared_image_count"
                )
                self.shared_hod_xy = shared_memory.ShareableList(
                    [0, 0], name="shared_hod_xy"
                )
                self.shared_ndarray = np.ndarray(
                    shape=(self.ysize, self.xsize),
                    buffer=self.shared_frame_memory.buf,
                    dtype=self.dtype,
                )

            # create process and start it
            self.proc = HammerOfDawnWorker(self.child_conn, self.vis_args)
            self.proc.start()

        except Exception as err:
            logging.error("Problem while initializing QtVisualizer: {}".format(err))
            raise (err)

    def initialize_model(self):

        # for reproducible psudorandomness
        fname_root = self.args["id"]
        random.seed(fname_root)

    def get_metadata(self, args=None):
        """Get metadata which we might want to save for later e.g. for
        writing to file"""

        metadata = {
            "process_time_list": self.process_time_list,
            "model_time_list": self.model_time_list,
            "stim_param_list": self.stim_param_list,
        }

        return metadata

    def skip(self):
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
            self.update_display_images(self.current_image)

            # if zsize > 1, reset current image
            if self.zsize > 1:
                self.current_image = self.blank_image

        # end and append timer
        process_tend = time.time()
        self.process_time_list.append(process_tend - process_tstart)

    def process_volume(self):

        # start timer
        model_tstart = time.time()

        # get events from gui object
        data = self.get_event()

        # incorporate events into internal state
        if data is not None:
            self.events.append(data)
            self.current_event = data

        # end and append timer
        model_tend = time.time()
        self.model_time_list.append(model_tend - model_tstart)

    def get_event(self):

        if self.parent_conn.poll():
            data = self.parent_conn.recv()
            self.events.append(data)
            return data

    def check_stim(self, image_ndx, cooldown_counter):

        # initialize output var
        stim_params = {}
        new_cooldown = 0

        # if this is initiating stimulus event
        if not self.stimulus_is_on and self.current_event is not None:

            # get stim params from randomly initialized lists or from event
            stim_intensity = self.current_event["stim_intensity"]
            stim_diameter = self.current_event["stim_diameter"]

            # add data to stim params
            # trigger stim_on on next volume
            zndx = image_ndx % self.zsize

            # trigger stim on next full volume
            stim_on = image_ndx + self.zsize - zndx

            # trigger event-based stimulation one volume later to allow for polygon to update
            # stim_on += self.zsize

            # add delay if need be, here not necessary because setSLMImage is blocking
            # stim_on += self.current_event["stim_delay_vols"] * self.zsize
            # set number of stim in frames
            # num_stim_frames = self.current_event["stim_duration_vols"] * self.zsize

            # try to update xy of the current event

            # build stim parameters
            stim_params["stim_on"] = stim_on
            # stim_params["stim_off"] = None
            stim_params["stim_intensity"] = stim_intensity
            stim_params["event"] = self.current_event

            # update holders
            self.stimulus_is_on = True
            self.current_event = None
            self.stim_diameter = stim_diameter
            self.stim_intensity = stim_intensity

            # add stim to stim param list
            self.stim_param_list.append(stim_params)

        # updating hammerofdawn xy position without event
        elif self.stimulus_is_on and self.current_event is None:

            try:

                # try to get more updated cursor xy and fill in event with holders
                nowx, nowy = self.shared_hod_xy
                stim_params["event"] = {
                    "event_type": "hammer-of-dawn",
                    "x": nowx,
                    "y": nowy,
                    "stim_diameter": self.stim_diameter,
                    "stim_intensity": self.stim_intensity,
                }

                # don't worry about updating stim param list - it gets updated in stim interface
                # self.stim_param_list.append(stim_params)

            except ValueError as err:
                logging.warning(
                    "Error while trying to grab cursor XY in HOD: {}".format(err)
                )

        # got event while stimulus is on
        elif self.stimulus_is_on and self.current_event is not None:

            # build stim parameters
            # stim_params["stim_on"] = stim_on

            # add data to stim params
            # trigger stim_on on next volume
            zndx = image_ndx % self.zsize

            # if we've done a full volume which happens on zndx=zsize-1
            stim_off = image_ndx + self.zsize - zndx

            # set
            stim_params["stim_off"] = stim_off
            stim_params["event"] = self.current_event

            # set holder
            self.stimulus_is_on = False
            self.current_event = None

            # update most recent stim param with accompanying info
            # so will only contain on/off frames and start xy
            self.stim_param_list[-1]["stim_off"] = stim_off

        return stim_params, new_cooldown

    def update_display_images(self, payload):

        if self.ipc == "sockets":
            self.parent_conn.send(payload)
        elif self.ipc == "shared_memory":
            self.shared_ndarray[:] = payload[:]
            self.shared_image_count[0] += 1

    def close(self):
        if self.ipc == "shared_memory":
            self.shared_frame_memory.close()
            self.shared_frame_memory.unlink()  # Free and release the shared memory block at the very end
            self.shared_image_count.shm.close()
            self.shared_image_count.shm.unlink()
            self.shared_hod_xy.shm.close()
            self.shared_hod_xy.shm.unlink()
            # alternatively use a manager
        else:
            # self.vis.close()
            self.parent_conn.close()


class HammerOfDawnWorker(Process):
    def __init__(self, child_conn, vis_args):

        super(HammerOfDawnWorker, self).__init__()

        # general params
        self.vis_args = vis_args
        self.child_conn = child_conn
        self.initialization_t0 = time.time()
        self.first_img_flag = False
        self.image_count = 0
        self.cl_stim_delay_ops = [0]
        self.stim_refractory_vols = 0
        self.stim_duration_vols = None
        self.stimulus_is_on = False
        self.ipc = self.vis_args.get("data_ipc", "shared_memory")
        self.dtype = self.vis_args.get("dtype", np.uint16)
        self.ysize = self.vis_args.get("ysize", None)
        self.xsize = self.vis_args.get("xsize", None)
        self.zsize = self.vis_args.get("zsize")
        self.total_frames = self.vis_args.get("total_frames", 100)
        self.total_vols = self.vis_args.get("total_frames") / self.zsize
        self.stim_diameter = self.vis_args.get("stim_diameter", 30)
        self.stim_intensity = self.vis_args.get("stim_intensity", 50)

    def close(self):
        self.child_conn.close()
        if self.ipc == "shared_memory":
            self.shared_frame_memory.close()
            self.shared_image_count.shm.close()
            self.shared_hod_xy.shm.close()

    def initialize_display(self):

        # initialize image holder, if shared mem has to be outside init
        if self.ipc == "shared_memory":

            # grab shared memory addresses
            self.shared_frame_memory = shared_memory.SharedMemory(
                name="shared_frame_memory"
            )
            self.shared_image_count = shared_memory.ShareableList(
                name="shared_image_count"
            )
            self.shared_hod_xy = shared_memory.ShareableList(name="shared_hod_xy")
            self.img = np.ndarray(
                shape=(self.ysize, self.xsize),
                buffer=self.shared_frame_memory.buf,
                dtype=self.dtype,
            )

        # app <- layout <- vbox <- image
        self.app = QtWidgets.QApplication([])
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.image_vbox = pg.ViewBox(lockAspect=True, enableMouse=False)
        self.ii = pg.ImageItem()

        # adjust image showing to be row-major and origin in top left, removing need for transpose
        # also add lut item
        self.ii.setOpts(axisOrder="row-major")
        self.image_vbox.invertY()
        # self.image_vbox.invertX()
        self.lut_histo = pg.HistogramLUTItem(image=self.ii, fillHistogram=False)

        # add cross hair to box
        self.crosshair_line_pen = pg.mkPen(color="green", alpha=0.5)
        self.vLine = pg.InfiniteLine(
            angle=90, movable=False, pen=self.crosshair_line_pen
        )
        self.hLine = pg.InfiniteLine(
            angle=0, movable=False, pen=self.crosshair_line_pen
        )

        # render graphic object at mouse cursor
        self.cursor_graphic = QtGui.QGraphicsEllipseItem()
        self.cursor_pen = pg.mkPen(
            color="green", alpha=0.8, width=2, style=QtCore.Qt.DotLine
        )
        self.cursor_graphic.setPen(self.cursor_pen)

        # map events
        self.ii.hoverEvent = self.my_hover_event
        self.ii.mouseClickEvent = self.my_click_event

        # set command panel layout
        self.command_panel_layout = QtGui.QGridLayout()

        # button for intensity
        self.enter_stimulus_intensity_button = QtGui.QPushButton(
            "stim intensity (%): {}".format(self.stim_intensity)
        )
        self.enter_stimulus_intensity_button.clicked.connect(
            self.my_get_stimulus_intensity
        )
        self.command_panel_layout.addWidget(self.enter_stimulus_intensity_button, 0, 0)

        # add text
        self.text_layout = QtGui.QGridLayout()
        self.vol_count_text = QtWidgets.QLabel(
            "frames: {}/{}".format(0, self.total_vols)
        )
        self.text_layout.addWidget(self.vol_count_text, 0, 0)

        # assemble all containers and widgets
        self.image_vbox.addItem(self.ii)
        self.image_vbox.addItem(self.vLine, ignoreBounds=False)
        self.image_vbox.addItem(self.hLine, ignoreBounds=False)
        self.image_vbox.addItem(self.cursor_graphic, ignoreBounds=False)
        self.graphics_layout.addItem(self.image_vbox)
        self.graphics_layout.addItem(self.lut_histo)
        self.grid_layout = QtGui.QGridLayout()
        self.grid_layout.addWidget(self.graphics_layout)
        self.grid_layout.addItem(self.command_panel_layout)
        self.grid_layout.addItem(self.text_layout)

        # add layout to window
        self.window = QtGui.QWidget()
        self.window.resize(1400, 600)
        self.window.setLayout(self.grid_layout)
        self.window.show()

    def update_display_images(self, img):

        # update gui
        self.ii.setImage(img, autoLevels=False)
        self.image_count += 1

    def update_display_text(self):

        # update frame count
        self.vol_count_text.setText(
            "received vols: {}/{}".format(self.image_count, self.total_vols)
        )

    def run(self):

        try:

            # initialize display, has to happen outside of init because
            self.initialize_display()
            self.app.processEvents()

            # set initial imagecount in case visualization is lagging
            # try:
            self.image_count = self.shared_image_count[0]
            # except Exception as err:
            # self.image_count = self.shared_image_count[0]

            try:
                while self.image_count < self.total_frames:

                    # process event loop unless there's data to work with
                    if self.ipc == "sockets":
                        while not self.child_conn.poll():
                            self.app.processEvents()
                        self.img = self.child_conn.recv()
                    elif self.ipc == "shared_memory":
                        try:
                            while self.image_count == self.shared_image_count[0]:
                                self.app.processEvents()
                        except ValueError as err:
                            continue

                    # if self.image_count != self.shared_image_count[0]:
                    # print('Internal image count: {}, shared image count: {}'.format(self.image_count, self.shared_image_count[0]))

                    # internal image buffer has been updated
                    self.update_dashboard()

                    # remove viewbox padding, only need to do once with locked scrolling
                    # also autolevel image
                    if not self.first_img_flag:
                        self.first_img_flag = True
                        self.image_vbox.setRange(rect=self.ii.boundingRect(), padding=0)
                        self.ii.setImage(self.img)

                        # sync image counts, can throw error
                        self.image_count = self.shared_image_count[0]

                    # continue event loop
                    self.app.processEvents()

            except Exception as err:
                logging.error("Error in inner loop in PointAndClick: {}".format(err))
        except EOFError as err:
            logging.warning(
                "Visualization multiprocess socket has been closed {}".format(err)
            )
        except BrokenPipeError as err:
            logging.warning(
                "Visualization multiprocess socket has been closed {}".format(err)
            )
        except ValueError as err:
            logging.warning("Invalid value received: {}".format(err))
        except Exception as err:
            logging.error(
                "Unknown error in initialization of HammerOfDawn: {}".format(err)
            )
        finally:
            self.close()

    def update_dashboard(self):
        self.update_display_images(self.img)
        self.update_display_text()

    def my_click_event(self, event):

        # get click position
        pos = event.pos()
        i = pos.y()
        j = pos.x()

        # get pixel position
        ppos = self.ii.mapToParent(pos)
        x = int(ppos.x())
        y = int(ppos.y())

        # write event data to parent process
        self.write_event(
            event_type="hammer-of-dawn", stim_diameter=self.stim_diameter, x=x, y=y
        )

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

        # update shared memory
        try:
            self.shared_hod_xy[0] = x
            self.shared_hod_xy[1] = y
        except ValueError as err:
            logging.warning(
                "Error writing cursor xy position to shared memory: {}".format(err)
            )

        # set window title
        self.vLine.setPos(x)
        self.hLine.setPos(y)

        # draw shape at pointer
        self.cursor_graphic.setRect(
            x - self.stim_diameter // 2,
            y - self.stim_diameter // 2,
            self.stim_diameter,
            self.stim_diameter,
        )

    def write_event(self, event_type, stim_diameter=None, x=None, y=None):

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
                # "stim_duration_vols": int(self.stim_duration_vols),
                "stim_intensity": self.stim_intensity,
            }

            # write conditional data
            if (
                event_type == "circle-click"
                or event_type == "circle-button"
                or event_type == "hammer-of-dawn"
            ):
                data["stim_diameter"] = int(stim_diameter)
                data["x"] = int(x)
                data["y"] = int(y)

            # if we're writing multiple events, queue events after the first to send out later
            if op == 0:
                self.child_conn.send(data)

                # play a sound =)
                # utils.play_wblive_sound("laser")

                # update last stim sent
                # self.update_last_stim_sent_text(data)

        # change button colors and set cooldown
        # self.stim_refractory_vols = (
        #    self.stim_duration_vols
        #    + self.refractory_constant
        # )
        # self.check_stim_refractory_vols()

        # visual aid that stimulus is on
        if not self.stimulus_is_on:
            crosshair_color = "red"
            self.stimulus_is_on = True
        else:
            crosshair_color = "green"
            self.stimulus_is_on = False
        self.set_crosshair_color(crosshair_color=crosshair_color)

    def set_crosshair_color(self, crosshair_color):

        # fix coloring
        self.cursor_pen.setColor(QtGui.QColor(crosshair_color))
        self.crosshair_line_pen.setColor(QtGui.QColor(crosshair_color))
        self.cursor_graphic.setPen(self.cursor_pen)
        self.vLine.setPen(self.crosshair_line_pen)
        self.hLine.setPen(self.crosshair_line_pen)

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

    def check_stim_refractory_vols(self):

        if self.stim_refractory_vols >= 1:
            self.roi_stim_button.setStyleSheet("background-color: red")
            self.full_field_stim_button.setStyleSheet("background-color: red")
        else:
            self.roi_stim_button.setStyleSheet("background-color: green")
            self.full_field_stim_button.setStyleSheet("background-color: green")


# standalone testing
if __name__ == "__main__":

    # DEBUG = True
    DEBUG = False

    # load tiff file for streaming
    # fname = "C:/Users/rldun/data/RLD_TEMP_DATA_HOLDER/20201124_RLD_1/behavior/wormb_0atr_2/wormb_0atr_2_MMStack_Pos0.ome.tif"
    # fname = "C:/Users/rldun/Desktop/temp_render/20210104-00-58-02.tif"
    # fname = "C:/Users/rldun/Desktop/temp_render/MAX_20210103-21-29-18.tif"
    # fname = "C:/Users/rldun/Desktop/temp_render/20210717-16-52-11.tiff"
    # fname = "E:/RLD/20220116_RLD_1/20220116-19-12-47/20220116-19-12-47.tiff"
    fname = (
        "C:/Users/rldun/Desktop/temp_render/20210114-00-24-49/20210114-00-24-49.tiff"
    )
    data = tf.imread(fname)

    # initialize subprocess
    fps = 10
    zsize = 1
    total_vols = 250
    stim_diameter = 30
    ysize = data.shape[1]
    xsize = data.shape[2]
    vis_args = {
        "stim_diameter": stim_diameter,
        "ysize": ysize,
        "xsize": xsize,
        "total_frames": total_vols * zsize,
        "zsize": zsize,
        # "stim_duration_vols": 4,
        "stim_intensity": 10,
        "id": 1111111111,
        "roi": [0, 0, xsize, ysize],
        "data_ipc": "shared_memory",
        "gooey_args": {
            "total_frames": total_vols * zsize,
            "zsize": zsize,
            "stimulus_diameter": stim_diameter,
            "stim_intensity_options": [50],
        },
    }

    hod = HammerOfDawn(args=vis_args)

    # vis = HammerOfDawnWorker(vis_args=vis_args)

    # initialize visualizer with blank frame
    # frame = data[0, :, :]
    # vis.update_display_images(frame)
    time.sleep(10)

    #  display images for some amount of time
    # zmip = np.zeros((data.shape[1], data.shape[2], zsize), dtype=np.uint16)
    # zmip = np.zeros((data.shape[1], data.shape[2]), dtype=np.uint16)

    # iterate volumes
    for i in range(total_vols):

        # if linear indexing, take zmip
        # for z in range(0, zsize):
        #    # zmip[:, :, z] = data[i * zsize + z, :, :]
        #    zmip = np.maximum(zmip, data[i * zsize + z, :, :])
        # mip = zmip.max(axis=2)
        # vis.update_display_images(mip)
        # vis.update_display_images(zmip)

        hod.update_display_images(data[i, :, :])

        # reset zmip
        # zmip[:] = 0

        # get events from vis
        event = hod.get_event()
        if event is not None:
            print(event)

        # pause for fps sim
        if fps is not None:
            time.sleep(1 / fps)

    # close
    hod.close()


"""
# examine hammer of dawn success
import tifffile as tf
import json
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np
fontprops = fm.FontProperties(size=18)

# params
linewidth = 2
linestyle = '-'
color = 'white'
pre_post_t_offset = 20

%matplotlib qt


# fnames
# fname_tiff = 'C:/Users/rldun/data/TEMP_DATA_HOLDER/20220914_RLD_1/20220914-17-15-01/20220914-17-15-01.tiff'
fname_tiff = 'C:/Users/rldun/data/TEMP_DATA_HOLDER/20220914_RLD_1/20220914-17-15-57/20220914-17-15-57.tiff'
fname_md = fname_tiff[:-5] + '_metadata.json'

# load data
d = tf.imread(fname_tiff)
with open(fname_md, 'r') as j:
    md = json.load(j)
stim_md = md['stim_metadata']
stim_param_list = stim_md['stim_param_list']



# create figure
fig = plt.figure()
ax = plt.gca()
image_counter = 0
err_dict = {'err_dxy_um': [], 'cursor_dxy_um': [], 'err_dxy_tminus1_um': []}
for stim_ndx in range(len(stim_param_list)):

    sp = stim_param_list[stim_ndx]
    son = sp['stim_on']
    soff = sp.get('stim_off', d.shape[0] - 1)
    event = sp['event']
    diameter = event['stim_diameter']
    event_frame_ndx = event['dynamic_event_frame_ndx']
    xs = event['x']
    ys = event['y']
    dmd_fstart = event_frame_ndx[0]
    dmd_fend = event_frame_ndx[-1]
    exposure = int(md['mmc_metadata']['exposure'])

    # offset t if there's difference between frame dmd is set and stim is turned on
    dmd_stim_on_offset = son - event_frame_ndx[0]

    for t in range(-pre_post_t_offset, len(event_frame_ndx) + pre_post_t_offset):

        # handle case where we want to render area where stim will be or just was
        if t >= 0 and t < len(event_frame_ndx) - dmd_stim_on_offset:
            cx = xs[t + dmd_stim_on_offset]
            cy = ys[t + dmd_stim_on_offset]
            circ = plt.Circle((cx, cy), radius=diameter/2, fill=False, color=color, linewidth=linewidth, linestyle=linestyle)
            ax.add_patch(circ)
        else:
            if t < 0:
                cx = xs[0]
                cy = ys[0]
            else:
                cx = xs[-1]
                cy = ys[-1]

        # show image
        frame = d[son + t,:,:]
        plt.imshow(frame, vmin=100, vmax=300)

        # adjust image crop halfwidth
        cropx = int(100/2)
        cropy = int(100/2)
        plt.xlim([cx-cropx, cx+cropx])
        plt.ylim([cy-cropy, cy+cropy])

        # add scale bar
        scalebar = AnchoredSizeBar(ax.transData,
                                30, '9.3um', 'lower center', 
                                pad=0.1,
                                color='white',
                                frameon=False,
                                size_vertical=1,
                                fontproperties=fontprops)
        ax.add_artist(scalebar)

        # add title
        ts = np.round(t*(exposure/1000), decimals=1)
        plt.title('stim_ndx: {}; t: {}s'.format(stim_ndx, ts))
        
        # get center of mass and plot point there
        img = d[son + t, cy-cropy:cy+cropy, cx-cropx:cx+cropx]
        median_threshold_offset = 10
        ret,img = cv2.threshold(img,np.median(img) + median_threshold_offset, 255, cv2.THRESH_BINARY)
        img = cv2.medianBlur(img, ksize=5)
        img = cv2.erode(img, np.ones((5, 5), np.uint8))
        comy, comx = ndimage.center_of_mass(img)
        plt.scatter(x=[cx+comx-cropx], y=[cy+comy-cropy], color='r', label='center of mass')
        plt.legend(loc='upper right')

        # get difference between goal in time
        if t > 0 and t < len(event_frame_ndx) - dmd_stim_on_offset:

            # get difference between com and current point
            dx = comx - cropx
            dy = comy - cropy
            err_dxy_um = np.sqrt(dx**2 + dy**2) / ppm
            err_dict['err_dxy_um'].append(err_dxy_um)

            # get absolute position of center of mass
            comx_abs = cx + dx
            comy_abs = cy + dy

            x_prev = xs[t-1 + dmd_stim_on_offset]
            y_prev = ys[t-1 + dmd_stim_on_offset]
            dx = cx - x_prev
            dy = cy - y_prev
            cursor_dxy_um = np.sqrt(dx**2 + dy**2) / ppm
            err_dict['cursor_dxy_um'].append(cursor_dxy_um)

            # get difference between center of mass and previous point
            dx = comx_abs - x_prev
            dy = comy_abs - y_prev
            err_dxy_tminus1_um = np.sqrt(dx**2 + dy**2) / ppm
            err_dict['err_dxy_tminus1_um'].append(err_dxy_tminus1_um)

        # save image
        plt.savefig('C:/Users/rldun/Desktop/temp_render/20220909/fig_{}.png'.format(str(image_counter).zfill(4)))
        image_counter += 1

        # increment counter and clear figure
        ax.cla()
    
err_dxy_arr = err_dict['err_dxy_um']
cursor_dxy_arr = err_dict['cursor_dxy_um']
err_dxy_tminus1_arr = err_dict['err_dxy_tminus1_um']


plt.figure()
plt.scatter(x=cursor_dxy_arr, y=err_dxy_arr)
plt.xlabel('stim displacement across 1 frame (um)')
plt.ylabel('error (um)')

plt.figure()
plt.plot(err_dxy_arr, label='error')
plt.plot(cursor_dxy_arr, label='stim displacement across 1 frame')
plt.legend()

plt.figure()
plt.scatter(x=cursor_dxy_arr, y=err_dxy_tminus1_arr)
plt.xlabel('stim displacement across 1 frame (um)')
plt.ylabel('error (um)')

plt.figure()
plt.hist(err_dxy_arr)
plt.xlabel('error (um)')
plt.xlim([0, 25])
plt.ylabel('count')

plt.figure()
plt.hist(err_dxy_tminus1_arr)
plt.xlabel('error (um)')
plt.ylabel('count')
plt.xlim([0, 25])

# now we check timings, i thinks setImage blocks
frame_time_arr = np.array(md['frame_time_list'])
frame_interval_arr = frame_time_arr[1:] - frame_time_arr[:-1]

# check exposure
# print('exposure: {}'.format(md['mmc_metadata']['exposure']))

# timings
stim_param_list = md['stim_metadata']['stim_param_list']
# model_time_arr = np.array(md['alg_metadata']['model_time_list'])
# process_time_arr = np.array(md['alg_metadata']['process_time_list'])
stim_on_time_arr = np.array(md['stim_metadata']['stim_on_time_list'])
stim_event_time_arr = np.array([sp['event']['ts'] for sp in stim_param_list])
process_stim_params_event_time_arr = np.array(md['stim_metadata']['process_stim_params_event_time_list']) * 1000
submit_stim_params_time_arr = np.array(md['stim_metadata']['submit_stim_params_time_list']) * 1000
alg_model_time_list = np.array(md['alg_metadata']['model_time_list']) * 1000
alg_process_time_list = np.array(md['alg_metadata']['process_time_list']) * 1000

plt.figure()
plt.hist(frame_interval_arr)
plt.xlabel('frame interval (s)')

plt.figure()
plt.hist(process_stim_params_event_time_arr)
plt.xlabel('time (ms)')
plt.title('process_stim_params_event duration')

plt.figure()
plt.hist(submit_stim_params_time_arr)
plt.xlabel('time (ms)')
plt.title('submit_stim_params duration')

plt.figure()
plt.hist(alg_model_time_list)
plt.xlabel('time (ms)')
plt.title('alg_model duration')

plt.figure()
plt.hist(alg_process_time_list)
plt.xlabel('time (ms)')
plt.title('alg_process duration')


"""
