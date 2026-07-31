"""3D bounding box annotation loader."""

import os
import json
import math
import numpy as np


# Category colors (R, G, B)
CATEGORY_COLORS = {
    'Mine_truck':        (0.90, 0.25, 0.25),  # red
    'Excavator':         (0.95, 0.75, 0.15),  # yellow
    'movable_object.barrier': (0.50, 0.50, 0.50),  # gray
    'Dump_truck':        (0.25, 0.60, 0.90),  # blue
    'Loader':            (0.20, 0.80, 0.40),  # green
    'Forklift':          (0.80, 0.40, 0.90),  # purple
    'Person':            (1.00, 0.50, 0.30),  # orange
}
DEFAULT_COLOR = (0.7, 0.7, 0.7)


class BBox3D:
    """A single 3D bounding box."""

    __slots__ = ('obj_id', 'obj_type', 'center', 'yaw', 'size', 'vertices', 'color')

    def __init__(self, obj_id, obj_type, center, yaw, size):
        self.obj_id = obj_id
        self.obj_type = obj_type
        self.center = np.array(center, dtype=np.float32)
        self.yaw = yaw
        self.size = np.array(size, dtype=np.float32)
        
        # Try exact match
        color = CATEGORY_COLORS.get(obj_type)
        if color is None:
            # Try case-insensitive match
            for k, v in CATEGORY_COLORS.items():
                if k.lower() == obj_type.lower():
                    color = v
                    break
        if color is None:
            # Generate deterministic fallback color
            import hashlib
            h = hashlib.md5(obj_type.encode('utf-8')).hexdigest()
            r = int(h[0:2], 16) / 255.0 * 0.6 + 0.2
            g = int(h[2:4], 16) / 255.0 * 0.6 + 0.2
            b = int(h[4:6], 16) / 255.0 * 0.6 + 0.2
            color = (r, g, b)
            
        self.color = color
        self.vertices = self._compute_vertices()

    def _compute_vertices(self):
        """Compute 8 corner vertices of the oriented bbox."""
        hx, hy, hz = self.size / 2.0
        # 8 corners in local frame (before rotation)
        corners = np.array([
            [-hx, -hy, -hz],
            [+hx, -hy, -hz],
            [+hx, +hy, -hz],
            [-hx, +hy, -hz],
            [-hx, -hy, +hz],
            [+hx, -hy, +hz],
            [+hx, +hy, +hz],
            [-hx, +hy, +hz],
        ], dtype=np.float32)

        # Rotate around Z axis by yaw
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        rot = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1],
        ], dtype=np.float32)

        corners = corners @ rot.T  # (8, 3)
        corners += self.center
        return corners

    def get_edges(self):
        """Return 12 edge pairs as list of (v1, v2) tuples."""
        v = self.vertices
        edges = [
            # bottom face
            (v[0], v[1]), (v[1], v[2]), (v[2], v[3]), (v[3], v[0]),
            # top face
            (v[4], v[5]), (v[5], v[6]), (v[6], v[7]), (v[7], v[4]),
            # vertical edges
            (v[0], v[4]), (v[1], v[5]), (v[2], v[6]), (v[3], v[7]),
        ]
        return edges

    def get_bottom_face(self):
        """Return bottom face vertices (4 vertices, CCW from outside)."""
        return [self.vertices[i] for i in [0, 1, 2, 3]]

    def get_top_face(self):
        """Return top face vertices."""
        return [self.vertices[i] for i in [4, 5, 6, 7]]

    def get_yaw_arrow(self, length=None):
        """
        Return arrow endpoints for yaw direction visualization.
        Returns (start, end, arrow_left, arrow_right) — all np.array(3).
        Arrow extends from bbox front edge outward in the yaw direction.
        """
        if length is None:
            length = max(self.size[0], self.size[1]) * 0.8

        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        direction = np.array([cos_y, sin_y, 0], dtype=np.float32)

        cx, cy, cz = self.center
        # Arrow starts at the front edge of bbox, extends outward
        half_len = max(self.size[0], self.size[1]) * 0.5
        start = np.array([cx + direction[0] * half_len,
                          cy + direction[1] * half_len,
                          cz], dtype=np.float32)
        end = np.array([cx + direction[0] * (half_len + length),
                        cy + direction[1] * (half_len + length),
                        cz], dtype=np.float32)

        # Arrowhead: two small lines at 25 degrees
        arrow_len = length * 0.25
        arrow_angle = math.radians(25)
        cos_a = math.cos(arrow_angle)
        sin_a = math.sin(arrow_angle)

        # Left arrowhead line
        left_dir = np.array([
            -cos_y * cos_a + sin_y * sin_a,
            -sin_y * cos_a - cos_y * sin_a,
            0
        ], dtype=np.float32)
        arrow_left = end + left_dir * arrow_len

        # Right arrowhead line
        right_dir = np.array([
            -cos_y * cos_a - sin_y * sin_a,
            -sin_y * cos_a + cos_y * sin_a,
            0
        ], dtype=np.float32)
        arrow_right = end + right_dir * arrow_len

        return start, end, arrow_left, arrow_right


def quaternion_to_yaw(w, x, y, z):
    """Convert quaternion [w, x, y, z] to yaw angle (radians around Z axis)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _parse_position(obj):
    if 'psr' in obj and isinstance(obj['psr'], dict) and 'position' in obj['psr']:
        pos = obj['psr']['position']
        if isinstance(pos, dict):
            return (pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0))
        elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return (pos[0], pos[1], pos[2])

    if 'translation' in obj and isinstance(obj['translation'], (list, tuple)) and len(obj['translation']) >= 3:
        t = obj['translation']
        return (t[0], t[1], t[2])

    if 'position' in obj:
        pos = obj['position']
        if isinstance(pos, dict):
            return (pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0))
        elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return (pos[0], pos[1], pos[2])

    return (0.0, 0.0, 0.0)


def _parse_size(obj):
    if 'psr' in obj and isinstance(obj['psr'], dict) and 'scale' in obj['psr']:
        scl = obj['psr']['scale']
        if isinstance(scl, dict):
            return (scl.get('x', 1.0), scl.get('y', 1.0), scl.get('z', 1.0))
        elif isinstance(scl, (list, tuple)) and len(scl) >= 3:
            return (scl[0], scl[1], scl[2])

    if 'size' in obj:
        s = obj['size']
        if isinstance(s, dict):
            return (s.get('x', 1.0), s.get('y', 1.0), s.get('z', 1.0))
        elif isinstance(s, (list, tuple)) and len(s) >= 3:
            return (s[0], s[1], s[2])

    if 'scale' in obj:
        scl = obj['scale']
        if isinstance(scl, dict):
            return (scl.get('x', 1.0), scl.get('y', 1.0), scl.get('z', 1.0))
        elif isinstance(scl, (list, tuple)) and len(scl) >= 3:
            return (scl[0], scl[1], scl[2])

    return (1.0, 1.0, 1.0)


def _parse_rotation(rot_data, default_heading=0.0):
    if rot_data is not None:
        if isinstance(rot_data, dict):
            return float(rot_data.get('z', default_heading))
        elif isinstance(rot_data, (list, tuple)):
            if len(rot_data) == 4:
                # Quaternion [w, x, y, z]
                w, x, y, z = [float(v) for v in rot_data]
                return quaternion_to_yaw(w, x, y, z)
            elif len(rot_data) == 3:
                return float(rot_data[2])
            elif len(rot_data) == 1:
                return float(rot_data[0])
        elif isinstance(rot_data, (int, float)):
            return float(rot_data)
    return float(default_heading)


def load_annotation(filepath):
    """
    Load a JSON annotation file (supports both converted format and raw dataset format).
    Returns list of BBox3D objects.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        objects = data.get('objects', [])
    elif isinstance(data, list):
        objects = data
    else:
        objects = []

    bboxes = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue

        obj_id = obj.get('obj_id')
        if obj_id is None:
            obj_id = obj.get('ID', obj.get('id', ''))

        obj_type = obj.get('obj_type') or obj.get('category') or obj.get('type') or 'unknown'

        center = _parse_position(obj)
        size = _parse_size(obj)

        default_heading = float(obj.get('obj_heading', 0.0) or 0.0)
        rot_data = None
        if 'psr' in obj and isinstance(obj['psr'], dict) and 'rotation' in obj['psr']:
            rot_data = obj['psr']['rotation']
        elif 'rotation' in obj:
            rot_data = obj['rotation']

        yaw = _parse_rotation(rot_data, default_heading)

        bbox = BBox3D(
            obj_id=obj_id,
            obj_type=obj_type,
            center=center,
            yaw=yaw,
            size=size,
        )
        bboxes.append(bbox)

    return bboxes


def find_matching_json(bin_path):
    """Find the JSON annotation file matching a bin/pcd file."""
    base = os.path.splitext(bin_path)[0]
    fname = os.path.basename(base) + '.json'
    parent = os.path.dirname(bin_path)

    candidates = [
        base + '.json',                                          # same dir, same name
        os.path.join(parent, 'json', fname),                     # json/ subdirectory
        os.path.join(os.path.dirname(parent), 'json', fname),    # sibling json/
        os.path.join(parent, 'json_0', fname),                   # json_0/ subdirectory
        os.path.join(os.path.dirname(parent), 'json_0', fname),  # sibling json_0/
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None
