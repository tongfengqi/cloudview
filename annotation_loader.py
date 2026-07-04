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
        """Return top face vertices."""
        return [self.vertices[i] for i in [4, 5, 6, 7]]


def load_annotation(filepath):
    """
    Load a JSON annotation file.
    Returns list of BBox3D objects.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        objects = json.load(f)

    bboxes = []
    for obj in objects:
        psr = obj.get('psr', {})
        pos = psr.get('position', {})
        rot = psr.get('rotation', {})
        scl = psr.get('scale', {})

        center = (pos.get('x', 0), pos.get('y', 0), pos.get('z', 0))
        yaw = rot.get('z', 0)
        size = (scl.get('x', 1), scl.get('y', 1), scl.get('z', 1))

        bbox = BBox3D(
            obj_id=obj.get('obj_id', ''),
            obj_type=obj.get('obj_type', 'unknown'),
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
