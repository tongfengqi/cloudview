"""Point cloud file loader with validation. Supports .bin and .pcd formats."""

import os
import struct
import numpy as np


class PointCloudData:
    """Holds loaded point cloud data and metadata."""

    __slots__ = ('points', 'intensities', 'filepath', 'filename', 'num_points',
                 'x_range', 'y_range', 'z_range', 'file_size', 'format',
                 'header_points')

    def __init__(self, points, filepath, fmt='unknown', header_points=None, intensities=None):
        self.points = points  # Nx3 float32 array
        self.intensities = intensities
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.file_size = os.path.getsize(filepath)
        self.num_points = len(points)
        self.x_range = (points[:, 0].min(), points[:, 0].max())
        self.y_range = (points[:, 1].min(), points[:, 1].max())
        self.z_range = (points[:, 2].min(), points[:, 2].max())
        self.format = fmt
        self.header_points = header_points  # original count from file header (before NaN filtering)


# ============================================================
#  BIN format
# ============================================================

def validate_bin(filepath):
    """Validate a bin file as float32 x4."""
    if not os.path.isfile(filepath):
        return False, f"File not found: {filepath}", None

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return False, "File is empty", None
    if file_size < 16:
        return False, f"File too small ({file_size} bytes)", None

    if file_size % 16 != 0:
        if file_size % 32 == 0:
            return False, f"Not float32 x4 format.\nFile size ({file_size:,} bytes) suggests float64 data.", None
        elif file_size % 12 == 0:
            return False, f"Not float32 x4 format.\nFile appears to be float32 x3 ({file_size // 12:,} points).", None
        else:
            return False, f"Not float32 x4 format.\nFile size ({file_size:,} bytes) is not divisible by 16.", None

    num_points = file_size // 16
    try:
        data = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    except Exception as e:
        return False, f"Failed to read as float32 x4: {e}", None

    xyz = data[:, :3]
    max_val = np.abs(xyz).max()
    if max_val > 100000:
        return False, (
            f"Not float32 x4 format.\n"
            f"Max coordinate value ({max_val:.0f}) is unreasonably large.\n"
            f"File may be float64 or a different column count."
        ), None

    if np.all(np.isnan(xyz)) or np.all(xyz == 0):
        return False, "Data appears invalid (all zeros or NaN).", None

    return None, None, {'num_points': num_points, 'file_size': file_size}


def load_bin(filepath, max_points=None):
    """Load a bin file as float32 x4."""
    ok, msg, info = validate_bin(filepath)
    if ok is False:
        raise ValueError(msg)

    data = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    points = data[:, :3].copy()
    intensities = data[:, 3].copy()

    if max_points and len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        intensities = intensities[idx]

    return PointCloudData(points, filepath, fmt='bin(float32 x4)', intensities=intensities)


# ============================================================
#  PCD format
# ============================================================

def parse_pcd_header(filepath):
    """Parse PCD file header. Returns header dict or raises ValueError."""
    header = {}
    with open(filepath, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file in PCD header")
            line = line.decode('ascii', errors='replace').strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            key = parts[0].upper()
            if key == 'VERSION':
                header['version'] = parts[1]
            elif key == 'FIELDS':
                header['fields'] = parts[1:]
            elif key == 'SIZE':
                header['size'] = [int(x) for x in parts[1:]]
            elif key == 'TYPE':
                header['type'] = parts[1:]
            elif key == 'COUNT':
                header['count'] = [int(x) for x in parts[1:]]
            elif key == 'WIDTH':
                header['width'] = int(parts[1])
            elif key == 'HEIGHT':
                header['height'] = int(parts[1])
            elif key == 'VIEWPOINT':
                header['viewpoint'] = parts[1:]
            elif key == 'POINTS':
                header['points'] = int(parts[1])
            elif key == 'DATA':
                header['data'] = parts[1].lower()
                header['_data_offset'] = f.tell()
                break
    return header


def _get_pcd_dtype(type_char, size):
    """Convert PCD type/size to numpy dtype."""
    if type_char == 'F':
        return {4: np.float32, 8: np.float64}[size]
    elif type_char == 'U':
        return {1: np.uint8, 2: np.uint16, 4: np.uint32}[size]
    elif type_char == 'I':
        return {1: np.int8, 2: np.int16, 4: np.int32}[size]
    else:
        raise ValueError(f"Unknown PCD type: {type_char}")


def validate_pcd(filepath):
    """Validate a PCD file."""
    if not os.path.isfile(filepath):
        return False, f"File not found: {filepath}", None

    try:
        header = parse_pcd_header(filepath)
    except Exception as e:
        return False, f"Invalid PCD header: {e}", None

    # Check required fields
    required = ['fields', 'size', 'type', 'count', 'points', 'data']
    missing = [k for k in required if k not in header]
    if missing:
        return False, f"PCD header missing: {', '.join(missing)}", None

    # Check for XYZ fields
    fields_lower = [f.lower() for f in header['fields']]
    if 'x' not in fields_lower or 'y' not in fields_lower or 'z' not in fields_lower:
        return False, f"PCD missing XYZ fields. Found: {header['fields']}", None

    # Check data type
    data_type = header['data']
    if data_type not in ('ascii', 'binary'):
        return False, f"Unsupported PCD data type: '{data_type}' (only ascii/binary supported)", None

    num_points = header['points']
    info = {
        'num_points': num_points,
        'file_size': os.path.getsize(filepath),
        'header': header,
    }
    return None, None, info


def load_pcd(filepath, max_points=None):
    """Load a PCD file. Returns PointCloudData."""
    ok, msg, info = validate_pcd(filepath)
    if ok is False:
        raise ValueError(msg)

    header = info['header']
    fields = [f.lower() for f in header['fields']]
    sizes = header['size']
    types = header['type']
    counts = header['count']
    num_points = header['points']
    data_type = header['data']

    # Build dtype for structured array
    dtype_list = []
    for f, s, t, c in zip(fields, sizes, types, counts):
        dt = _get_pcd_dtype(t, s)
        if c > 1:
            dtype_list.append((f, dt, c))
        else:
            dtype_list.append((f, dt))
    dtype = np.dtype(dtype_list)

    if data_type == 'binary':
        data = np.fromfile(filepath, dtype=dtype, offset=header['_data_offset'])
    else:
        # ASCII
        data = np.loadtxt(
            filepath, dtype=dtype,
            skiprows=header['_data_offset'],
            max_rows=num_points
        )

    # Extract XYZ and filter out NaN points
    raw_x = data['x'].astype(np.float32)
    raw_y = data['y'].astype(np.float32)
    raw_z = data['z'].astype(np.float32)

    valid = ~(np.isnan(raw_x) | np.isnan(raw_y) | np.isnan(raw_z))
    points = np.column_stack([raw_x[valid], raw_y[valid], raw_z[valid]])

    intensities = None
    if 'intensity' in data.dtype.names:
        intensities = data['intensity'].astype(np.float32)[valid]
    elif 'i' in data.dtype.names:
        intensities = data['i'].astype(np.float32)[valid]

    if max_points and len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        if intensities is not None:
            intensities = intensities[idx]

    return PointCloudData(points, filepath, fmt=f"pcd({header.get('version','?')}, {data_type})",
                          header_points=num_points, intensities=intensities)


# ============================================================
#  Unified interface
# ============================================================

def validate_file(filepath):
    """Auto-detect format and validate. Returns (ok, msg, info)."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pcd':
        return validate_pcd(filepath)
    elif ext == '.bin':
        return validate_bin(filepath)
    else:
        return False, f"Unsupported file format: '{ext}'\nSupported: .bin, .pcd", None


def load_file(filepath, max_points=None):
    """Auto-detect format and load. Returns PointCloudData."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pcd':
        return load_pcd(filepath, max_points)
    elif ext == '.bin':
        return load_bin(filepath, max_points)
    else:
        raise ValueError(f"Unsupported file format: '{ext}'")
