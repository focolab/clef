from pycromanager import Core
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
import sys
import time
import json
from datetime import datetime
 
from hardware.stimulus_controllers.stimulus_resources.MightexPolygon1000 import ImageProcessor


##########################
AUTO_THRESHOLD = True
circle_radius = 7
calibration_fname = 'C:/Users/Leica/code/clef/hardware/stimulus_controllers/stimulus_resources/MightexPolygon1000/calibrations.json'
ip_ops =  {
    "fb_post_threshold": 100,
    "fb_threshold_margin": 100,
    "med_filt_size": 5,
    "fb_min_blob_spacing": 10,
    "template_filter_width": 7,
    "cb_maxdist": 9
}
##########################


# load core connection
mmc = Core(convert_camel_case=False)

# first we want to calibrate with full field camera
mmc.clearROI()

# get image params for reshaping
res = mmc.getROI()
xsize = res.getWidth()
ysize = res.getHeight()

# set slm device
slm = mmc.getSLMDevice()
mmc.setSLMDevice(slm)
slm_height = mmc.getSLMHeight(slm)
slm_width = mmc.getSLMWidth(slm)
#slm_height = 1140
#slm_width = 912
slm_shape = (slm_height, slm_width)

# set slm to off
mmc.setSLMPixelsTo(slm, 0)

# draw circle, happens in-place
pcx = [250, 500, 750]
pcy = [200, 500, 800]


# function to draw potentially multiple circles and return a snap of the slm patterened result
def draw_and_snap(points_x, points_y, slm_width=1180, slm_height=960):
    slm_shape = (slm_height, slm_width)

    # set mask
    mask = np.zeros(slm_shape)

    # draw circles for measuring x
    for i in range(len(points_x)):
        cv2.circle(mask, (points_x[i],points_y[i]), circle_radius, 255, -1)

    # display image
    mmc.setSLMImage(slm, mask.astype(np.uint8).flatten())

    # sleep
    time.sleep(1)

    # snap image
    mmc.snapImage()

    # get image and reformate
    img = mmc.getImage().astype(np.uint16).reshape((ysize, xsize))

    return img


# pattern x and y calibration points
img_x1 = draw_and_snap(pcx, [500, 500, 500], slm_width=slm_width, slm_height=slm_height)
img_x2 = draw_and_snap(pcx, [500, 500, 500], slm_width=slm_width, slm_height=slm_height)
img_x3 = draw_and_snap(pcx, [500, 500, 500], slm_width=slm_width, slm_height=slm_height)
img_y1 = draw_and_snap([500, 500, 500], pcy, slm_width=slm_width, slm_height=slm_height)
img_y2 = draw_and_snap([500, 500, 500], pcy, slm_width=slm_width, slm_height=slm_height)
img_y3 = draw_and_snap([500, 500, 500], pcy, slm_width=slm_width, slm_height=slm_height)
img_x = np.array([img_x1, img_x2, img_x3]).max(axis=0)
img_y = np.array([img_y1, img_y2, img_y3]).max(axis=0)

# auto threshold as 0.2 x max
if AUTO_THRESHOLD:
    thresh = int((img_x.max() - np.median(img_x)) / 5)
    ip_ops['fb_post_threshold'] = thresh
    ip_ops['fb_threshold_margin'] = thresh

# peakfind x and y
ip = ImageProcessor.ImageProcessor(ip_ops)
xys_x = ip.segmentchunk(img_x.astype(np.float32))
xys_y = ip.segmentchunk(img_y.astype(np.float32))

# grab image coordinates of patterened points
icx = np.array(xys_x)[:, 1]
icy = np.array(xys_y)[:, 0]

# cast to int from int32
icx = [int(x) for x in icx]
icy = [int(y) for y in icy]

# 20240315 sort because... sometimes gets reversed?
icx.sort()
icy.sort()

# printouts
print('pcx = {}'.format(pcx))
print('pcy = {}'.format(pcy))
print('icx = {}'.format(icx))
print('icy = {}'.format(icy))

# icx = [400, 1235, 2080]
# pcy = [451, 958, 1460]

###########################################################################################

# write to json file
# calibration_fname = "calibrations.json"
if not os.path.exists(calibration_fname):
    with open(calibration_fname, "w") as j:
        json.dump({"calibrations": []}, j, indent=4, sort_keys=True)
        
# load calibrations
with open(calibration_fname, "r") as j:

    # load calibrations
    md = json.load(j)

# add calibration points
cali = {
    'pcx': pcx,
    'pcy': pcy,
    'icx': icx,
    'icy': icy
}


# add some other info
dt = datetime.today().strftime("%Y%m%d-%H-%M-%S")
cam = mmc.getCameraDevice()
binning = mmc.getProperty(cam, 'Binning')
roi_j = mmc.getROI()
roi = [int(roi_j.getX()), int(roi_j.getY()), int(roi_j.getWidth()), int(roi_j.getHeight())]
obj = mmc.getProperty('ObjectiveTurret', 'Label')

# add to object
cali['datetime'] = dt
cali['objective'] = obj
cali['camera'] = cam
cali['binning'] = binning
cali['roi'] = roi

md['calibrations'].append(cali)

print('Existing calibrations: {}'.format(md))

# save
with open(calibration_fname, 'w') as j:
    json.dump(md, j, indent=4, sort_keys=True)