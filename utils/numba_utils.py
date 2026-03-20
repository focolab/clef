import numpy as np
import json
import logging

# import numba, patch for testing 
try:
    import numba
    from numba import jit
except ImportError as err:
    numba = None
    def jit(nopython):
        def decorator(func):
            return func
        return decorator
    
logger = logging.getLogger(__name__)

# function to apply three point calibration
@jit(nopython=True)
def three_point_calibration_transform(ix, pcx, icx):

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

    return px


# function to transform xy coordinates given 3 sets of calibration points
@jit(nopython=True)
def three_point_xy_transform(ix, iy, pcx, pcy, icx, icy):

    # call subfunction for each point
    px = three_point_calibration_transform(ix, pcx, icx)
    py = three_point_calibration_transform(iy, pcy, icy)
    res = [px, py]

    return res


# function to draw a circle
@jit(nopython=True)
def circle_2d(im_width):

    g = np.zeros((im_width, im_width), dtype=np.uint8)
    r = im_width // 2

    # iterate points
    for i in range(int(-(im_width) / 2), int((im_width) / 2 + 1)):
        for j in range(int(-(im_width) / 2), int((im_width) / 2 + 1)):
            x0 = int((im_width) / 2)  # center
            y0 = int((im_width) / 2)  # center
            x = i + x0  # row
            y = j + y0  # col

            dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            if dist <= r:
                g[y, x] = 1

    return g


# function to draw an ellipse mask given dims
@jit(nopython=True)
def ellipse_2d(maskwidth, maskheight):

    cx = maskwidth // 2
    cy = maskheight // 2

    a = cx
    b = cy

    # build ellipse by multiplying column array and taking distance
    x = np.linspace(-cx, cx, int(maskwidth))

    # assemble into column vector
    # y = np.linspace(-cy, cy, int(maskheight))[:,None]
    y = np.zeros((int(maskheight), 1))
    y[:, 0] = np.linspace(-cy, cy, int(maskheight))

    ellipse = ((x) / a) ** 2 + ((y) / b) ** 2 <= 1

    return ellipse.astype(np.uint8)


# function to draw a circle mask given dims
@jit(nopython=True)
def make_circle_mask(ix, iy, diameter, xsize, ysize):

    if diameter % 2 == 0:
        diameter = diameter + 1

    mask = np.zeros((xsize, ysize), dtype=np.uint8)
    mask[
        iy - diameter // 2 : iy + diameter // 2 + 1,
        ix - diameter // 2 : ix + diameter // 2 + 1,
    ] = circle_2d(diameter)
    return mask


@jit(nopython=True)
def make_ellipse_mask(ix, iy, diameter, ddx, ddy, xsize, ysize):

    # adjust even diameters make indexing easier
    # if diameter % 2 == 0:
    #    diameter = diameter + 1

    # calculate scaled mask dimensions
    # for taking imagespace size (px) and converting to pixel space size
    maskwidth = round(diameter / ddx)
    maskheight = round(diameter / ddy)

    # set mask params to even
    if maskwidth % 2 == 0:
        maskwidth = maskwidth - 1
    if maskheight % 2 == 0:
        maskheight = maskheight - 1

    # initialize output array
    # mask = np.zeros((xsize, ysize), dtype=np.uint8) this worked on torstoscope
    mask = np.zeros((ysize, xsize), dtype=np.uint8)
    ellipse = ellipse_2d(maskwidth, maskheight)
    e_height = ellipse.shape[0]
    e_width = ellipse.shape[1]
    mask[
        iy - e_height // 2 : iy + e_height // 2 + 1,
        ix - e_width // 2 : ix + e_width // 2 + 1,
    ] = ellipse
    return mask


@jit(nopython=True)
def generate_pg_ellipse_mask(
    ix, iy, pcx, pcy, icx, icy, diameter, xoffset, yoffset, DSI_IMGWIDTH, DSI_IMGHEIGHT
):

    # apply roi offset
    ix = ix + xoffset
    iy = iy + yoffset

    # transform imagespace coords into polygon space
    pxy = three_point_xy_transform(ix, iy, pcx, pcy, icx, icy)

    # unpack calibration points
    p1y = pcy[0]
    p3y = pcy[2]
    i1y = icy[0]
    i3y = icy[2]
    p1x = pcx[0]
    p3x = pcx[2]
    i1x = icx[0]
    i3x = icx[2]

    # calculate xy ansiotropy aka xy scaling
    # get magnitude
    pdy = p3y - p1y
    pdx = p3x - p1x
    idy = i3y - i1y
    idx = i3x - i1x

    # get scaling ratio
    ddy = idy / pdy
    ddx = idx / pdx

    # plot mask inside appropriate dimensions
    # mask = make_circle_mask(pxy[0], pxy[1], diameter, DSI_IMGWIDTH, DSI_IMGHEIGHT)
    mask = make_ellipse_mask(
        int(pxy[0]), int(pxy[1]), diameter, ddx, ddy, DSI_IMGWIDTH, DSI_IMGHEIGHT
    )

    return mask


# function to transform imagespace coords into polygonspace and draw a circle there
@jit(nopython=True)
def generate_pg_circle_mask(
    ix, iy, pcx, pcy, icx, icy, diameter, xoffset, yoffset, DSI_IMGWIDTH, DSI_IMGHEIGHT
):

    # apply roi offset
    ix = ix + xoffset
    iy = iy + yoffset

    # transform imagespace coords into polygon space
    pxy = three_point_xy_transform(ix, iy, pcx, pcy, icx, icy)

    # plot circular mask inside appropriate dimensions
    mask = make_circle_mask(pxy[0], pxy[1], diameter, DSI_IMGWIDTH, DSI_IMGHEIGHT)

    return mask

@jit(nopython=True)
def make_rectangle_mask(ix, iy, width, height, ddx, ddy, xsize, ysize):

    # calculate scaled mask dimensions
    # for taking imagespace size (px) and converting to pixel space size
    maskwidth = round(width / ddx)
    maskheight = round(height / ddy)

    # initialize output array
    mask = np.zeros((ysize, xsize), dtype=np.uint8)

    # return masked value
    mask[iy:iy+maskheight, ix:ix+maskwidth] = 1

    return mask


@jit(nopython=True)
def make_multi_rectangle_mask(ix_arr, iy_arr, width_arr, height_arr, ddx, ddy, xsize, ysize):

    # initialize output array
    mask = np.zeros((ysize, xsize), dtype=np.uint8)

    # fill mask with each rectangle
    for x in range(len(ix_arr)):
        y1 = int(iy_arr[x])
        y2 = int(iy_arr[x]+round(height_arr[x]/ddy))
        x1 = int(ix_arr[x])
        x2 = int(ix_arr[x]+round(width_arr[x]/ddx))
        mask[y1:y2, x1:x2] = 1

    return mask


@jit(nopython=True)
def generate_pg_multi_rectangle_mask(ix_arr, iy_arr, width_arr, height_arr,  pcx, pcy, icx, icy, xoffset, yoffset, DSI_IMGWIDTH, DSI_IMGHEIGHT):

    # apply roi offset
    ix_arr = ix_arr + xoffset
    iy_arr = iy_arr + yoffset

    # transform imagespace coords into polygon space
    # pxy = three_point_xy_transform(ix, iy, pcx, pcy, icx, icy)
    pxy_arr = np.zeros((len(ix_arr), 2))
    for i in range(len(ix_arr)):
        pxy_arr[i,:] = three_point_xy_transform(ix_arr[i], iy_arr[i], pcx, pcy, icx, icy)
    
    # unpack calibration points
    p1y = pcy[0]
    p3y = pcy[2]
    i1y = icy[0]
    i3y = icy[2]
    p1x = pcx[0]
    p3x = pcx[2]
    i1x = icx[0]
    i3x = icx[2]

    # calculate xy ansiotropy aka xy scaling
    # get magnitude
    pdy = p3y - p1y
    pdx = p3x - p1x
    idy = i3y - i1y
    idx = i3x - i1x

    # get scaling ratio
    ddy = idy / pdy
    ddx = idx / pdx

    # make multi rectangle mask with transformed coords
    mask = make_multi_rectangle_mask(
        pxy_arr[:,0], pxy_arr[:,1], width_arr, height_arr, ddx, ddy, DSI_IMGWIDTH, DSI_IMGHEIGHT
    )
    return mask

# def draw_polygons_on_structural_image(
#     datadir="", rec_id="", use_mip=True, zsize=12, finfo=""
# ):

#     # load structural images
#     data = MMSubroutines.load_structural_images(datadir, rec_id)
#     # data = load_structural_images(datadir, rec_id)

#     # shape into
#     img = data.reshape((3, data.shape[1] // zsize, zsize, data.shape[2], data.shape[3]))
#     img = img.max(axis=1)
#     logging.debug("Structural data reshaped into {}".format(img.shape))

#     if use_mip:
#         logging.debug("Using structural MIP for ROI")
#         img = img.max(axis=1)

#     logging.debug("Please enter a shape ROI")

#     # start event loop/gui
#     with napari.gui_qt():

#         # intialize viewer
#         viewer = napari.Viewer(ndisplay=2)
#         viewer.theme = "light"

#         @viewer.bind_key("u")
#         def print_message(viewer):
#             try:

#                 screenshot_fname = datadir + "\\{}_polygon-mask{}.png".format(
#                     rec_id, finfo
#                 )
#                 logging.debug("Saving screenshot of napari window")

#                 viewer.screenshot(screenshot_fname)

#                 viewer.close()
#             except Exception as err:
#                 logging.exception(
#                     "Error while trying to save a screenshot during napari drawing: {}".format(
#                         err
#                     )
#                 )

#         # add the structural image data
#         layerbf = viewer.add_image(img[2], colormap="gray")
#         layer561 = viewer.add_image(img[1], colormap="red", blending="additive")
#         layer488 = viewer.add_image(img[0], colormap="green", blending="additive")

#         # add the shapes layer and do some presets
#         shape_layer = viewer.add_shapes(name="polygon_mask")
#         shape_layer.current_face_color = "#FFA500"
#         shape_layer.mode = "add_rectangle"

#     # get drawn polygons
#     shape_list = shape_layer.data

#     # get dimensionality of shapes
#     if use_mip or zsize == 1:
#         dims = 2
#     else:
#         dims = 3

#     pgons = {
#         "shape_list": shape_list,
#         "shape_type": shape_layer.shape_type,
#         "dims": dims,
#     }

#     return pgons


def polygons_to_masks(
    pgons,
    convert_imagespace_to_polygonspace=True,
    xoffset=0,
    yoffset=0,
    frame_dims=None,
    pcx=None,
    pcy=None,
    icx=None,
    icy=None,
):

    try:

        shape_list = pgons["shape_list"]
        shape_ndims = pgons["dims"]
        shape_type = pgons["shape_type"]

        if shape_ndims != 2 and shape_ndims != 3:
            raise ("Napari polygon shape dimensions are not 2 or 3")

        # apply correction to each vertex in label layer to transform into polygon space
        new_shape_list = []
        if len(shape_list) > 0:
            for shape in shape_list:
                new_vertex_list = []
                for v in range(shape.shape[0]):

                    # logging.info("Shape to convert is: {}".format(shape))

                    # grab image pixel space xy
                    vertices = shape[v, :]

                    # check for 2d or 3d object
                    if shape_ndims == 3:
                        iz, iy, ix = vertices
                    else:
                        iy, ix = vertices

                    # shift according to offset
                    newx = ix + xoffset
                    newy = iy + yoffset

                    # if we need coordinate transform, apply
                    if convert_imagespace_to_polygonspace:
                        newx, newy = three_point_xy_transform(
                            newx, newy, pcx, pcy, icx, icy
                        )

                    # add new coords to new polygon list
                    if shape_ndims == 3:
                        new_vertex_list.append(np.array((iz, newy, newx)))
                    else:
                        new_vertex_list.append(np.array((newy, newx)))

                # shape layer is list of vertices
                new_shape_list.append(np.array(new_vertex_list))

            # create new shape layer for to_masks function
            shape_layer = napari.layers.shapes.shapes.Shapes(
                data=new_shape_list, shape_type=shape_type
            )

        # combine multiple masks for each polygon into a single monochrome
        masks_list = shape_layer.to_masks(mask_shape=frame_dims)
        mask = np.zeros(shape=frame_dims, dtype=np.bool)
        for m in masks_list:
            mask = mask + m

    except Exception as err:
        logging.warning(
            "Exception while trying to convert polygons {} to masks: {}. Defaulting to full frame stim.".format(
                pgons, err
            )
        )
        mask = np.ones(shape=frame_dims, dtype=np.bool)

    return mask



if __name__ == "__main__":

    import numpy, json
    
    # calibrations_fname = "C:/Users/rldun/code/wb-live/wb-live-v4/res/peripherals/Mightex Polygon P1000/calibrations.json"
    calibrations_fname = '../res/peripherals/Mightex Polygon P1000/calibrations.json'
    DSI_IMGWIDTH = 912
    DSI_IMGHEIGHT = 1140
    roi = [600,1360,2000,480]
    
    # pick some stims to make
    ix_arr = [600, 700, 800, 900]
    iy_arr = [100, 200, 300, 400]
    width_arr = [20, 50, 20, 50]
    height_arr = [20, 30, 40, 50]

    # open calibration file packged with wb-live
    with open(calibrations_fname) as j:
        md = json.load(j)
        calibrations = md["calibrations"]

    # grab latest calibration points
    cali = calibrations[-1]
    print('grabbed calibration: {}'.format(cali))

    #  unpack calibration points
    pcx, pcy, icx, icy = cali["pcx"], cali["pcy"], cali["icx"], cali["icy"]
    pcx = np.array(pcx)
    pcy = np.array(pcy)
    icx = np.array(icx)
    icy = np.array(icy)

    # turn lists into arrays
    ix_arr = np.array(ix_arr)
    iy_arr = np.array(iy_arr)
    width_arr = np.array(width_arr)
    height_arr = np.array(height_arr)

    # unpack xy offsets from roi
    xoffset = roi[0]
    yoffset = roi[1]

    # make circular mask
    # mask = utils.generate_pg_ellipse_mask(cx,cy,pcx,pcy,icx,icy,diameter,roi[0],roi[1],DSI_IMGWIDTH, DSI_IMGHEIGHT)

    # make rectangle mask, ever so slightly different syntax
    res = generate_pg_multi_rectangle_mask(ix_arr, iy_arr, width_arr, height_arr, pcx, pcy, icx, icy, roi[0], roi[1], DSI_IMGWIDTH, DSI_IMGHEIGHT)
