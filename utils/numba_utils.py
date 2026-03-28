"""
Numba-accelerated mask generation utilities for polygon/SLM stimulus control.

Provides calibrated coordinate transforms (camera space -> polygon space)
and mask generation (ellipses, circles, rectangles) for the Mightex Polygon DMD.
"""

import numpy as np
import logging

try:
    import numba
    from numba import jit
except ImportError:
    numba = None
    def jit(nopython):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


@jit(nopython=True)
def three_point_calibration_transform(ix, pcx, icx):
    """Piecewise linear calibration transform for one axis."""
    pc1x, pc2x, pc3x = pcx
    ic1x, ic2x, ic3x = icx

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

    ux = (ix - i1) / (i2 - i1)
    px = p1 + (ux * (p2 - p1))

    return px


@jit(nopython=True)
def three_point_xy_transform(ix, iy, pcx, pcy, icx, icy):
    """Transform XY coordinates using 3-point calibration."""
    px = three_point_calibration_transform(ix, pcx, icx)
    py = three_point_calibration_transform(iy, pcy, icy)
    res = [px, py]
    return res


@jit(nopython=True)
def circle_2d(im_width):
    """Draw a filled circle in a square array."""
    g = np.zeros((im_width, im_width), dtype=np.uint8)
    r = im_width // 2

    for i in range(int(-(im_width) / 2), int((im_width) / 2 + 1)):
        for j in range(int(-(im_width) / 2), int((im_width) / 2 + 1)):
            x0 = int((im_width) / 2)
            y0 = int((im_width) / 2)
            x = i + x0
            y = j + y0

            dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            if dist <= r:
                g[y, x] = 1

    return g


@jit(nopython=True)
def ellipse_2d(maskwidth, maskheight):
    """Draw a filled ellipse mask."""
    cx = maskwidth // 2
    cy = maskheight // 2

    a = cx
    b = cy

    x = np.linspace(-cx, cx, int(maskwidth))
    y = np.zeros((int(maskheight), 1))
    y[:, 0] = np.linspace(-cy, cy, int(maskheight))

    ellipse = ((x) / a) ** 2 + ((y) / b) ** 2 <= 1

    return ellipse.astype(np.uint8)


@jit(nopython=True)
def make_circle_mask(ix, iy, diameter, xsize, ysize):
    """Draw a circle mask at (ix, iy) with given diameter."""
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
    """Draw an anisotropic ellipse mask scaled by ddx/ddy."""
    maskwidth = round(diameter / ddx)
    maskheight = round(diameter / ddy)

    if maskwidth % 2 == 0:
        maskwidth = maskwidth - 1
    if maskheight % 2 == 0:
        maskheight = maskheight - 1

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
def make_rectangle_mask(ix, iy, width, height, ddx, ddy, xsize, ysize):
    """Draw a single rectangle mask scaled by ddx/ddy."""
    maskwidth = round(width / ddx)
    maskheight = round(height / ddy)

    mask = np.zeros((ysize, xsize), dtype=np.uint8)
    mask[iy:iy+maskheight, ix:ix+maskwidth] = 1

    return mask


@jit(nopython=True)
def make_multi_rectangle_mask(ix_arr, iy_arr, width_arr, height_arr, ddx, ddy, xsize, ysize):
    """Draw multiple rectangles into a single mask."""
    mask = np.zeros((ysize, xsize), dtype=np.uint8)

    for x in range(len(ix_arr)):
        y1 = int(iy_arr[x])
        y2 = int(iy_arr[x]+round(height_arr[x]/ddy))
        x1 = int(ix_arr[x])
        x2 = int(ix_arr[x]+round(width_arr[x]/ddx))
        mask[y1:y2, x1:x2] = 1

    return mask


@jit(nopython=True)
def generate_pg_ellipse_mask(
    ix, iy, pcx, pcy, icx, icy, diameter, xoffset, yoffset, DSI_IMGWIDTH, DSI_IMGHEIGHT
):
    """Generate calibrated ellipse mask for polygon SLM."""
    ix = ix + xoffset
    iy = iy + yoffset

    pxy = three_point_xy_transform(ix, iy, pcx, pcy, icx, icy)

    p1y = pcy[0]
    p3y = pcy[2]
    i1y = icy[0]
    i3y = icy[2]
    p1x = pcx[0]
    p3x = pcx[2]
    i1x = icx[0]
    i3x = icx[2]

    pdy = p3y - p1y
    pdx = p3x - p1x
    idy = i3y - i1y
    idx = i3x - i1x

    ddy = idy / pdy
    ddx = idx / pdx

    mask = make_ellipse_mask(
        int(pxy[0]), int(pxy[1]), diameter, ddx, ddy, DSI_IMGWIDTH, DSI_IMGHEIGHT
    )

    return mask


@jit(nopython=True)
def generate_pg_circle_mask(
    ix, iy, pcx, pcy, icx, icy, diameter, xoffset, yoffset, DSI_IMGWIDTH, DSI_IMGHEIGHT
):
    """Generate calibrated circle mask for polygon SLM."""
    ix = ix + xoffset
    iy = iy + yoffset

    pxy = three_point_xy_transform(ix, iy, pcx, pcy, icx, icy)

    mask = make_circle_mask(pxy[0], pxy[1], diameter, DSI_IMGWIDTH, DSI_IMGHEIGHT)

    return mask


@jit(nopython=True)
def generate_pg_multi_rectangle_mask(
    ix_arr, iy_arr, width_arr, height_arr, pcx, pcy, icx, icy, xoffset, yoffset, DSI_IMGWIDTH, DSI_IMGHEIGHT
):
    """Generate calibrated multi-rectangle mask for polygon SLM."""
    ix_arr = ix_arr + xoffset
    iy_arr = iy_arr + yoffset

    pxy_arr = np.zeros((len(ix_arr), 2))
    for i in range(len(ix_arr)):
        pxy_arr[i,:] = three_point_xy_transform(ix_arr[i], iy_arr[i], pcx, pcy, icx, icy)

    p1y = pcy[0]
    p3y = pcy[2]
    i1y = icy[0]
    i3y = icy[2]
    p1x = pcx[0]
    p3x = pcx[2]
    i1x = icx[0]
    i3x = icx[2]

    pdy = p3y - p1y
    pdx = p3x - p1x
    idy = i3y - i1y
    idx = i3x - i1x

    ddy = idy / pdy
    ddx = idx / pdx

    mask = make_multi_rectangle_mask(
        pxy_arr[:,0], pxy_arr[:,1], width_arr, height_arr, ddx, ddy, DSI_IMGWIDTH, DSI_IMGHEIGHT
    )
    return mask
