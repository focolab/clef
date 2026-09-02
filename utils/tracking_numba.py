"""
Numba-accelerated blob tracking kernels for XY stage tracking.

Tracks a single bright fluorescent blob on a dark background. The threshold
is computed relative to each frame (mean + frac * (max - mean)) so it adapts
to intensity drift without a per-frame sort. Falls back to pure Python if
numba is not installed.
"""

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba optional
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


@njit(cache=True)
def _relative_threshold(frame, frac):
    """One-pass mean/max, return mean + frac * (max - mean)."""
    h, w = frame.shape
    fmax = frame[0, 0]
    fsum = 0.0
    for i in range(h):
        for j in range(w):
            v = frame[i, j]
            if v > fmax:
                fmax = v
            fsum += v
    fmean = fsum / (h * w)
    return fmean + frac * (fmax - fmean)


@njit(cache=True)
def bright_blob_centroid(frame, frac):
    """Intensity-weighted centroid of pixels brighter than the threshold.

    Returns (row, col, npix, snr). Weighting by (intensity - threshold) gives a
    sub-pixel centroid biased toward the brightest core of the blob. snr is the
    peak prominence over background, (max - mean) / std: a real puncta sits many
    sigma above the mean, whereas a blank/noisy frame peaks only ~4-5 sigma, so
    snr lets the caller reject frames with no genuine puncta. The relative
    threshold alone can't — some pixels always clear mean + frac*(max - mean).
    Returns (nan, nan, 0, snr) if no pixel clears the threshold.
    """
    h, w = frame.shape
    n = h * w
    fmax = frame[0, 0]
    fsum = 0.0
    fsqsum = 0.0
    for i in range(h):
        for j in range(w):
            v = frame[i, j]
            if v > fmax:
                fmax = v
            fsum += v
            fsqsum += v * v
    fmean = fsum / n
    var = fsqsum / n - fmean * fmean
    fstd = np.sqrt(var) if var > 0.0 else 0.0
    snr = (fmax - fmean) / fstd if fstd > 0.0 else 0.0
    thr = fmean + frac * (fmax - fmean)

    wsum = 0.0
    ysum = 0.0
    xsum = 0.0
    count = 0
    for i in range(h):
        for j in range(w):
            v = frame[i, j]
            if v > thr:
                weight = v - thr
                wsum += weight
                ysum += weight * i
                xsum += weight * j
                count += 1
    if wsum <= 0.0:
        return np.nan, np.nan, 0, snr
    return ysum / wsum, xsum / wsum, count, snr


@njit(cache=True)
def bright_blob_mask(frame, frac):
    """Binary (0/255 uint8) mask of pixels brighter than the threshold."""
    thr = _relative_threshold(frame, frac)
    h, w = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if frame[i, j] > thr:
                mask[i, j] = 255
    return mask


@njit(cache=True)
def circle_roi_stats(frame, cy, cx, radius):
    """Mean/sum/pixel-count inside a circle of given radius centered at (cy, cx).

    Iterates only the circle's bounding box, so cost scales with the ROI area,
    not the whole frame. Returns (mean, total, npix); (nan, 0, 0) if empty.
    """
    h, w = frame.shape
    ci = int(round(cy))
    cj = int(round(cx))
    r = int(radius)
    r2 = radius * radius
    i0 = max(0, ci - r)
    i1 = min(h, ci + r + 1)
    j0 = max(0, cj - r)
    j1 = min(w, cj + r + 1)
    total = 0.0
    count = 0
    for i in range(i0, i1):
        di = i - ci
        for j in range(j0, j1):
            dj = j - cj
            if di * di + dj * dj <= r2:
                total += frame[i, j]
                count += 1
    if count == 0:
        return np.nan, 0.0, 0
    return total / count, total, count
