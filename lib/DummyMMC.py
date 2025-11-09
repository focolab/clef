
import logging
import numpy as np

# logging
logger = logging.getLogger(__name__)

class DummyMMC:
    def __init__(self, fname=None):
        self.ndx = 0
        self.fname = fname
        self.data = None
        self.xsize = None
        self.ysize = None
        self.is_dummy = True

        # if we're supplying frames, load the data file
        if self.fname:
            import tifffile as tf

            self.data = tf.imread(self.fname)
            self.ysize = self.data.shape[1]
            self.xsize = self.data.shape[2]
            self.tsize = self.data.shape[0]
            logstr = "Loading datafile {} of dimensions: t:{}, x:{}, y:{}".format(
                    self.fname, self.tsize, self.xsize, self.ysize
                )
            logger.info(logstr)
        
        else:

            # hardcoded params for now
            self.ysize = 200
            self.xsize = 200
            self.tsize = 1000

            logstr = "Running DummyMMC with no input file, default to dimensions: t:{}, x:{}, y:{}".format(
                    self.fname, self.tsize, self.xsize, self.ysize
                )
            logger.info(logstr)


    def startSequenceAcquisition(self, foo, bar, junk):
        pass

    def stopSequenceAcquisition(self):
        pass

    def isSequenceRunning(self):
        return False

    def startContinuousSequenceAcquisition(self, foo):
        pass

    def clearCircularBuffer(self):
        pass

    def getRemainingImageCount(self):
        images_left = self.tsize - self.ndx
        return images_left

    def getROI(self):

        return 0, 0, self.xsize, self.ysize

    def popNextImage(self):

        # if we have data, return image and increment internal counter
        if self.fname:
            try:
                retval = self.data[self.ndx, :, :]
                self.ndx += 1
                return retval

            except IndexError:
                print(
                    "Unable to get index {} from data of shape {}".format(
                        self.ndx, (self.tsize, self.ysize, self.xsize)
                    )
                )
                return None
        else:

            # fall back on white noise
            return np.random.randint(0, 65536, size=(self.ysize, self.xsize), dtype=np.uint16)


    def setExposure(self, exposure):
        pass

    def getImage(self):
        return self.popNextImage()

    def snapImage(self):
        pass

    def setConfig(self, *argv):
        pass

    def setShutterOpen(self, shutter_state):
        pass
