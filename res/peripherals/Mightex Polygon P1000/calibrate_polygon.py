from pycromanager import Bridge
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
import sys
import time
import json
from datetime import datetime

# cd to parent dir to import imageprocessor utility
cwd = os.getcwd()
os.chdir('../../../lib')
path_to_lib = os.getcwd()
sys.path.append(path_to_lib) 
import ImageProcessor
os.chdir(cwd)


##########################
AUTO_THRESHOLD = True
circle_radius = 7
ip_ops =  {
    "fb_post_threshold": 100,
    "fb_threshold_margin": 100,
    "med_filt_size": 5,
    "fb_min_blob_spacing": 10,
    "template_filter_width": 7,
    "cb_maxdist": 9
}
##########################


#### Setup pycromanager ####
bridge = Bridge(convert_camel_case=False)
mmc = bridge.get_core()

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

# write to json file
calibration_fname = "calibrations.json"
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


#############################################################################
# 20240315 test stim writing
#############################################################################
import sys
import matplotlib.pyplot as plt
mypath = 'C:/Users/Leica/code/wb-live/wb-live-v4/lib/'
sys.path.append(mypath)
import wbliveUtils as utils

# various params, 20240315
pcx = [250, 500, 750]
pcy = [200, 500, 800]
icx = [931, 1762, 2617]
icy = [986, 1490, 1992]
DSI_IMGWIDTH = 912
DSI_IMGHEIGHT = 1140

# full frame
# roi = [0, 0, 3200, 3200] 
# x_list = [200, 500, 800]
# y_list = [250, 500, 750]
# width_list = [10, 30, 60]
# height_list = [60, 50, 40]

# standard imaging roi
roi = [600,1360,2000, 480]
x_list = [350, 1000, 1500]
y_list = [100, 250, 400]
width_list = [60, 40, 20]
height_list = [20, 80, 20]

# test main function
res = utils.generate_pg_multi_rectangle_mask(np.array(x_list),np.array(y_list),np.array(width_list),np.array(height_list),pcx,pcy,icx,icy,roi[0],roi[1],DSI_IMGWIDTH,DSI_IMGHEIGHT)

plt.imshow(res)

# then put it on the dmd
mask = res * 255  # mask is uint8, but values 1-255 are valid and specify dithering
mmc.setSLMImage(mmc.getSLMDevice(), mask.astype(np.uint8).flatten())

#################
# test generate_pg_multi_rectangle_mask
ix_arr = np.array(x_list)
iy_arr = np.array(y_list)

pxy_arr = np.zeros((len(ix_arr), 2))
for i in range(len(ix_arr)):
    pxy_arr[i,:] = utils.three_point_xy_transform(ix_arr[i], iy_arr[i], pcx, pcy, icx, icy)


###################
# test three_point_xy_transform
ix = ix_arr[0]
iy = ix_arr[0]

# call subfunction for each point
px = utils.three_point_calibration_transform(ix, pcx, icx)
py = utils.three_point_calibration_transform(iy, pcy, icy)
res = [px, py]

###################
# test three_point_calibration_transform

# unpack points from vectors
pc1x, pc2x, pc3x = pcx
ic1x, ic2x, ic3x = icx

# tranform x
# get which two points for transform
if ix <= ic2x:
    p1 = pc1x
    p2 = pc2x
    i1 = ic1x
    i2 = ic2x
else:
    p1 = pc2x
    p2 = pc3x
    i1 = ic2x
    i2 = ic3x

# transform
ux = (ix - i1) / (i2 - i1)
px = p1 + (ux * (p2 - p1))