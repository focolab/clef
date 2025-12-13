import numpy as np
import napari
import tifffile as tf
import socket
import sys
import subprocess


# flags
DEBUG = False
use_img = 0
use_shapes_layer = 1
launch_exe = 1
PORT = 5007

# to launch exe externally
if launch_exe and not DEBUG:
    path_to_exe = "C:/Users/confocal/source/repos/polygon-app/x64/Debug/polygon-app.exe"
    polygon_process = subprocess.Popen(path_to_exe, stdout=subprocess.PIPE)

# connect to cpp program
if not DEBUG:
    HOST = "localhost"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))

# params for polygon image
DSI_IMGWIDTH = 912
DSI_IMGHEIGHT = 1140
polygon_dims = [DSI_IMGWIDTH, DSI_IMGHEIGHT]


def draw(img_fname="", xoffset=0, yoffset=0):

    # intialize background image
    if use_img:
        if not img_fname:
            #img_fname = "C:/Users/rldun/data/RLD TEMP DATA HOLDER/20200923_RLD_1/calibration attempt 3/20201004_img.tif"
            img_fname = 'C:/Users/confocal/Desktop/mmsnap.tif'

        data = tf.imread(img_fname)
    else:
        data = np.zeros((polygon_dims))

    # calculated calibration points
    # image pixel space
    ic1 = [181, 159]
    ic2 = [590, 826]
    # polygon pixel space
    pc1 = [291, 240]
    pc2 = [772, 635]

    #    pts1_list = [
    #    [[213, 209], [328, 271]],
    #    [[181, 159], [291, 240]],
    # ]
    # pts2_list = [[[548, 775], [722, 605]], [[590, 826], [772, 635]]]

    # start event loop/gui
    with napari.gui_qt():

        # intialize viewer
        viewer = napari.Viewer(ndisplay=2)
        viewer.theme = "light"
        # viewer.palette['background'] = 'rgb(220, 220, 220)'
        # viewer.palette['canvas'] = 'grey'
        @viewer.bind_key("enter")
        def print_message(viewer):
            print("taking screenshot")
            viewer.screenshot("C:/Users/rldun/Desktop/lol.png")
            print("submitting mask")
            viewer.close()

        # add the volume
        img_layer = viewer.add_image(data)

        # add the shapes layer and do some presets
        if use_shapes_layer:
            shape_layer = viewer.add_shapes(name="polygon_mask")
            shape_layer.current_face_color = "#FFA500"
            shape_layer.mode = "add_rectangle"

        # alternatively use a labels layer
        else:
            labels_layer = viewer.add_labels(np.zeros(polygon_dims))
            labels_layer.mode = "paint"

    # convert individual shapes to single mask
    if use_shapes_layer:

        if use_img:

            # apply correction to each vertex in label layer to transform into polygon space
            new_shape_list = []
            if len(shape_layer.data) > 0:
                for shape in shape_layer.data:
                    new_vertex_list = []
                    for v in range(shape.shape[0]):

                        # grab image pixel space xy
                        iy, ix = shape[v, :]

                        ix += xoffset
                        iy += yoffset

                        # do conversion to polygon pixel space
                        ux = (ix - ic1[0]) / (ic2[0] - ic1[0])
                        uy = (iy - ic1[1]) / (ic2[1] - ic1[1])
                        px = pc1[0] + (ux * (pc2[0] - pc1[0]))
                        py = pc1[1] + (uy * (pc2[1] - pc1[1]))

                        # save that vertex
                        new_vertex_list.append(np.array((py, px)))

                    # shape layer is list of vertices x 2
                    new_shape_list.append(np.array(new_vertex_list))

                # create new shape layer for to_masks function
                shape_layer = napari.layers.shapes.shapes.Shapes(data=new_shape_list)

        # combine multiple masks for each polygon into a single monochrome
        masks_list = shape_layer.to_masks(mask_shape=polygon_dims)
        mask = np.zeros(shape=polygon_dims, dtype=np.bool)
        for m in masks_list:
            mask = mask + m

    # alternatively take labels layer data
    else:
        if use_img:
            print("Not yet implemented!")
        else:
            mask = labels_layer.data.astype(np.bool)

    # boolify and pack for single bit encoding
    processed_mask = mask.astype(np.bool).T
    processed_mask = np.flip(processed_mask, axis=0)
    processed_mask = np.flip(processed_mask, axis=1)
    msg = np.packbits(processed_mask)

    # send message
    if not DEBUG:
        s.send(msg)

        # grab response and reshape
        #buff_size = 1024
        #resp = s.recv(buff_size)
        # returned = np.frombuffer(res, dtype=np.bool).reshape(msg.shape)

        # print
        #print("Response: {}".format(resp))

    ret = {
        "mask": mask,
        "pgons": {
            "shape_list": shape_layer.data,
            "shape_type": shape_layer.shape_type,
            "dims": 2,
        },
    }

    return ret


mask = draw()