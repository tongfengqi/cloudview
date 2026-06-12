"""Color mapping utilities for point cloud visualization."""

import numpy as np


# Viridis-like colormap: blue -> cyan -> green -> yellow
_VIRIDIS_STOPS = [
    (0.0,  (0.267, 0.004, 0.329)),
    (0.25, (0.282, 0.140, 0.458)),
    (0.5,  (0.127, 0.566, 0.551)),
    (0.75, (0.544, 0.772, 0.247)),
    (1.0,  (0.993, 0.906, 0.144)),
]

# Plasma colormap: purple -> red -> yellow
_PLASMA_STOPS = [
    (0.0,  (0.050, 0.030, 0.528)),
    (0.25, (0.494, 0.012, 0.658)),
    (0.5,  (0.798, 0.280, 0.468)),
    (0.75, (0.968, 0.584, 0.232)),
    (1.0,  (0.940, 0.975, 0.131)),
]


def _interp_colormap(t, stops):
    """Interpolate a colormap at position t in [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return (
                c0[0] + f * (c1[0] - c0[0]),
                c0[1] + f * (c1[1] - c0[1]),
                c0[2] + f * (c1[2] - c0[2]),
            )
    return stops[-1][1]


def values_to_colors(values, colormap='viridis'):
    """Map a 1D array of values to Nx3 RGB colors."""
    stops = _VIRIDIS_STOPS if colormap == 'viridis' else _PLASMA_STOPS
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-10:
        t = np.zeros_like(values)
    else:
        t = (values - vmin) / (vmax - vmin)

    colors = np.empty((len(values), 3), dtype=np.float32)
    for i, ti in enumerate(t):
        colors[i] = _interp_colormap(ti, stops)
    return colors


def values_to_colors_fast(values, colormap='viridis'):
    """Vectorized color mapping (faster for large arrays)."""
    stops = _VIRIDIS_STOPS if colormap == 'viridis' else _PLASMA_STOPS
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-10:
        t = np.zeros_like(values)
    else:
        t = (values - vmin) / (vmax - vmin)

    # Build lookup table
    lut_size = 256
    lut = np.empty((lut_size, 3), dtype=np.float32)
    for i in range(lut_size):
        lut[i] = _interp_colormap(i / (lut_size - 1), stops)

    indices = (t * (lut_size - 1)).astype(np.int32)
    indices = np.clip(indices, 0, lut_size - 1)
    return lut[indices]
