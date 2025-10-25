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
from multiprocessing import Process, Pipe, shared_memory
import time
import json

# ignore numpy warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class XYStageTracker:
    def __init__(self, args, local_handles={}):

        # get general params
        self.args = args
        self.rec_id = self.args["id"]
        self.roi = self.args["roi"]
        self.frames_to_grab = self.args["gooey_args"]["total_frames"]
        self.zsize = self.args["gooey_args"]["zsize"]
        self.xsize = self.args["roi"][2]
        self.ysize = self.args["roi"][3]
        self.dtype = self.args.get("dtype", np.uint16)
        self.STANDALONE_MODE = self.args.get("STANDALONE_MODE", False)
        self.HEADLESS_MODE = self.args.get("HEADLESS_MODE", False)

        # load handle to microscope hardware
        self.mmc = local_handles.get('mmc', None)
        if self.mmc is None:
            logging.critical(
                "Warning: No MMC oject submitted to XYStageTracker."
            )

        # set args for subprocess
        self.vis_args = {
            "ysize": self.ysize,
            "xsize": self.xsize,
            "total_frames": self.frames_to_grab,
            "zsize": self.zsize,
            "dtype": self.dtype,
            "HEADLESS_MODE": self.HEADLESS_MODE,
        }

        try:

            # create a pipe to visualizer
            self.parent_conn, self.child_conn = Pipe()

            # if ipc is shared memory, we only use the pipe for subprocess->main thread events
            self.shared_stage_offset_xy = shared_memory.ShareableList(
                [0, 0], name="shared_stage_offset_xy"
            )

            # if we're running standalone, we have to create shared memory and update it
            if self.STANDALONE_MODE:
                self.shared_frame_memory = shared_memory.SharedMemory(
                    create=True,
                    size=self.ysize * self.xsize * 2,
                    name="shared_frame_memory",
                )
                self.shared_image_count = shared_memory.ShareableList(
                    [0], name="shared_image_count"
                )
                self.shared_ndarray = np.ndarray(
                    shape=(self.ysize, self.xsize),
                    buffer=self.shared_frame_memory.buf,
                    dtype=self.dtype,
                )

            # create process and start it
            self.proc = XYStageTrackerWorker(self.child_conn, self.vis_args)
            self.proc.start()

        except Exception as err:
            logging.error("problem while initializing XYStageTracker: {}".format(err))
            raise (err)

    def close(self):

        # Free and release the shared memory block at the very end
        self.shared_stage_offset_xy.shm.close()
        self.shared_stage_offset_xy.shm.unlink()

        if self.STANDALONE_MODE:
            self.shared_frame_memory.close()
            self.shared_frame_memory.unlink()
            self.shared_image_count.shm.close()
            self.shared_image_count.shm.unlink()

    def update_display_images(self, payload):
        if self.STANDALONE_MODE:
            self.shared_ndarray[:] = payload[:]
            self.shared_image_count[0] += 1
        else:
            pass


    def get_xy_offset(self):

        # grab offset values
        offsetx, offsety = self.shared_stage_offset_xy

        # if either value is nonzero
        if offsetx or offsety:

            # reset offsets
            self.shared_stage_offset_xy[0] = 0
            self.shared_stage_offset_xy[1] = 0

            # return
            return offsetx, offsety

        else:

            return None

    # 20240129 this part still needs to be done
    def sync_stage(self):

        # get xy offset
        xy_offset = self.get_xy_offset()
        if xy_offset is not None:

            print('This is where we would move the stage.')
            # micro-manager core api (http lol) https://valelab4.ucsf.edu/~MM/doc/MMCore/html/class_c_m_m_core.html
            # see section "XY stage control."
            # e.g. mmc.setXYPosition('XYStage', 10, 10)
            # or mmc.setRelativeXYPosition('XYStage', 10, 10)

    

class XYStageTrackerWorker(Process):
    def __init__(self, child_conn, vis_args):

        super(XYStageTrackerWorker, self).__init__()

        # general params
        self.vis_args = vis_args
        self.child_conn = child_conn
        self.initialization_t0 = time.time()
        self.first_img_flag = False
        self.image_count = 0
        self.dtype = self.vis_args.get("dtype", np.uint16)
        self.ysize = self.vis_args.get("ysize", None)
        self.xsize = self.vis_args.get("xsize", None)
        self.zsize = self.vis_args.get("zsize")
        self.total_frames = self.vis_args.get("total_frames", 100)
        self.total_vols = self.vis_args.get("total_frames") / self.zsize

        # more specific vars
        self.image_proc_delay = 0
        self.center_of_mass_graphic_radius = 5
        self.image_cx = self.xsize / 2
        self.image_cy = self.ysize / 2
        self.HEADLESS_MODE = vis_args.get("HEADLESS_MODE", False)

    def close(self):
        self.child_conn.close()
        self.shared_frame_memory.close()
        self.shared_image_count.shm.close()
        self.shared_stage_offset_xy.shm.close()

    def initialize_display(self):

        # grab shared memory addresses
        self.shared_frame_memory = shared_memory.SharedMemory(
            name="shared_frame_memory"
        )
        self.shared_image_count = shared_memory.ShareableList(name="shared_image_count")
        self.shared_stage_offset_xy = shared_memory.ShareableList(
            name="shared_stage_offset_xy"
        )
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
        self.ii.setOpts(axisOrder="row-major")
        self.image_vbox.invertY()
        # self.image_vbox.invertX()

        # add cursor at center of mass
        self.com_graphic = QtWidgets.QGraphicsEllipseItem()
        self.com_graphic_pen = pg.mkPen(
            color="#FA8128", alpha=0.8, width=2, style=QtCore.Qt.DotLine
        )
        self.com_graphic.setPen(self.com_graphic_pen)

        # set command panel layout
        self.command_panel_layout = QtWidgets.QGridLayout()

        # add text
        self.text_layout = QtWidgets.QGridLayout()
        self.vol_count_text = QtWidgets.QLabel(
            "received vols: {}/{}".format(self.image_count, self.total_vols)
        )
        self.image_processing_latency_text = QtWidgets.QLabel(
            "image processing latency: {}ms".format(self.image_proc_delay)
        )

        # assemble all containers and widgets
        self.image_vbox.addItem(self.ii)
        self.image_vbox.addItem(self.com_graphic, ignoreBounds=False)
        self.graphics_layout.addItem(self.image_vbox)
        self.text_layout.addWidget(self.vol_count_text, 0, 0)
        self.text_layout.addWidget(self.image_processing_latency_text, 1, 0)
        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.addWidget(self.graphics_layout)
        self.grid_layout.addItem(self.command_panel_layout)
        self.grid_layout.addItem(self.text_layout)

        # add layout to window
        self.window = QtWidgets.QWidget()
        self.window.resize(1400, 600)
        self.window.setLayout(self.grid_layout)
        if not self.HEADLESS_MODE:
            self.window.show()

    def update_display_images(self, img):

        # update gui
        img = self.process_image(img)

        # display
        if not self.HEADLESS_MODE:
            self.ii.setImage(img, autoLevels=False)

        #
        self.image_count += 1

    def process_image(self, img):

        # cast to uint8
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

        #
        # thresh = cv2.adaptiveThreshold(
        #    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        # )

        # calculate contour
        # cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # calculate image filtering
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # calculate center of mass
        offsety, offsetx = scipy.ndimage.measurements.center_of_mass(img)

        # set shared variable
        self.shared_stage_offset_xy[0] = int(offsetx) - self.image_cx
        self.shared_stage_offset_xy[1] = int(offsety) - self.image_cy

        return img

    def update_display_graphics(self):

        # update cursor graphic at center of mass
        offsetx, offsety = self.shared_stage_offset_xy
        self.com_graphic.setRect(
            int(offsetx - self.center_of_mass_graphic_radius),
            int(offsety - self.center_of_mass_graphic_radius),
            int(self.center_of_mass_graphic_radius * 2),
            int(self.center_of_mass_graphic_radius * 2),
        )

    def update_display_text(self):

        # update frame count
        self.vol_count_text.setText(
            "received vols: {}/{}".format(self.image_count, self.total_vols)
        )

        # update processing delay
        self.image_processing_latency_text.setText(
            "image processing latency: {}ms".format(self.image_proc_delay)
        )

    def update_dashboard(self):

        # update image processing and calculations
        self.update_display_images(self.img)

        # optionally update gui
        if not self.HEADLESS_MODE:
            self.update_display_graphics()
            self.update_display_text()

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
                    try:
                        while self.image_count == self.shared_image_count[0]:
                            self.app.processEvents()
                    except ValueError as err:
                        continue

                    # internal image buffer has been updated
                    t_pre = time.time()
                    self.update_dashboard()
                    t_post = time.time()

                    # update real-time display of processing delay
                    self.image_proc_delay = np.round(
                        (t_post - t_pre) * 1000, decimals=1
                    )


                    # continue event loop
                    self.app.processEvents()

            except Exception as err:
                logging.error(
                    "Error in inner loop in XYStageTrackerWorker: {}".format(err)
                )
        except Exception as err:
            logging.error(
                "Error in outter loop in XYStageTrackerWorker: {}".format(err)
            )

        finally:
            self.close()


# standalone testing
if __name__ == "__main__":

    # DEBUG = True
    DEBUG = False

    # load tiff file for streaming
    fname = 'C:/Users/rldun/Desktop/temp_render/20221017-17-22-01/20221017-17-22-01.tiff'
    data = tf.imread(fname)

    # arguments for subprocess
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
        "STANDALONE_MODE": True, # if we want to instantiate shared-memory 
        "HEADLESS_MODE": False, # in theory, we might not care about gooey for simply tracking worms, so this alg could run headless as well
    }

    # initialize alg/gui
    xytracker = XYStageTracker(args=vis_args)

    # iterate frames
    for i in range(total_vols):

        # send image to xystage tracker (by updating memory array shared between distinct subprocesses)
        xytracker.update_display_images(data[i, :, :])

        # print output provided by xytracker subprocessor
        event = xytracker.get_xy_offset()
        # xytracker.sync_stage()
        if event is not None:
            print(event)

        # pause for fps sim
        if fps is not None:
            time.sleep(1 / fps)

    # clean up
    xytracker.close()
