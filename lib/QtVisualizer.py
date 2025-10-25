from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph.opengl as gl
from matplotlib import cm
import pyqtgraph as pg
import numpy as np
from multiprocessing import Process, Pipe
import pyqtgraph.opengl as gl
from matplotlib import cm
import pyqtgraph as pg
import sys, os, warnings, time, json, logging
import pyqtgraph.opengl as gl


class QtVisualizer:
    def __init__(self):

        try:
            # create a pipe to visualizer
            self.parent_conn, child_conn = Pipe()

            # create process and start it
            self.proc = QtVisualizerWorker(child_conn)
            self.proc.start()

        except Exception as err:
            print("Problem while initializing QtVisualizer: {}".format(err))
            raise (err)

    def update_display_images(self, payload):
        self.parent_conn.send(payload)

    def close(self):
        # self.parent_conn.send("Lollll closing")
        self.parent_conn.close()


class QtVisualizerWorker(Process):
    def __init__(self, child_conn):

        # call superconstructor
        super(QtVisualizerWorker, self).__init__()
        self.child_conn = child_conn

    def initialize_display(self):

        # make app
        self.app = pg.Qt.mkQApp(name='QtVisualizer')

        # make graphics widget
        graphics = pg.GraphicsLayoutWidget()

        # Get the colormap
        maxcolor = 255
        colormap = cm.get_cmap("magma")
        colormap._init()
        lut = (colormap._lut * maxcolor).view(
            np.ndarray
        )  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # create view box for image item
        box = pg.ViewBox(lockAspect=True)

        # create image item, maintain reference to update later
        self.ii = pg.ImageItem()

        # adjust image showing to be row-major, removing need for transpose
        self.ii.setOpts(axisOrder="row-major")
        box.invertY()

        # Apply the colormap
        self.ii.setLookupTable(lut)

        # add image item to box
        box.addItem(self.ii)

        # add box to graphic
        graphics.addItem(box)

        # add graphics to layout
        grid = QtWidgets.QGridLayout()
        grid.addWidget(graphics)

        # show window and keep reference so that it doesn't get garbage collected
        self.window = QtWidgets.QWidget()
        self.window.setLayout(grid)
        # self.window.showMaximized()
        self.window.show()

    def update_display_images(self, img):

        # set ranges of image
        # img = (img / np.amax(img) * 254).astype(np.uint8)

        # flip image to match microscope coords
        # img = img.T
        # img = np.flip(img, axis=0)

        # update gui
        self.ii.setImage(img)

    def run(self):

        try:
            # initialize display, has to happen outside of init because
            # qt objects aren't serializable
            self.initialize_display()

            while 1:
                # process event loop unless there's data to work with
                while not self.child_conn.poll():
                    self.app.processEvents()

                # get image and update display
                img = self.child_conn.recv()
                self.update_display_images(img)
                self.app.processEvents()

        except EOFError as err:
            print("Visualization multiprocess socket has been closed {}".format(err))

            # don't close the window right away, allow user to do so
            self.app.exec_()

        except BrokenPipeError as err:
            print("Visualization multiprocess socket has been closed {}".format(err))

        except ValueError as err:
            print("Invalid value received: {}".format(err))



# standalone testing
if __name__ == "__main__":

    import tifffile as tf

    # params
    fps = 10 
    total_vols = 250
    fname = 'C:/Users/rldun/Desktop/temp_render/20221017-17-22-01/20221017-17-22-01.tiff'

    # load tiff file for streaming
    data = tf.imread(fname)

    # create visualizer
    viz = QtVisualizer()

    # iterate frames
    for i in range(total_vols):

        # send image to subprocess
        viz.update_display_images(data[i, :, :])

        # pause for fps sim
        if fps is not None:
            time.sleep(1 / fps)
    
    vis.close()
