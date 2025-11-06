import tifffile as tf


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
            self.data = tf.imread(self.fname)
            self.ysize = self.data.shape[1]
            self.xsize = self.data.shape[2]
            self.tsize = self.data.shape[0]
            print(
                "Loading datafile {} of dimensions: t:{}, x:{}, y:{}".format(
                    self.fname, self.tsize, self.xsize, self.ysize
                )
            )

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
        images_left = self.data.shape[0] - self.ndx
        return images_left

    def getROI(self):

        return 0, 0, self.xsize, self.ysize

    def popNextImage(self):

        # if we have data, return image and increment internal counter
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
