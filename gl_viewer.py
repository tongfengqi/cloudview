"""OpenGL 3D point cloud viewer widget."""

import math
import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import QOpenGLFunctions_3_3_Compatibility
from PySide6.QtCore import Qt


# GL constants
GL_DEPTH_TEST = 0x0B71
GL_POINT_SMOOTH = 0x0B10
GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_COLOR_BUFFER_BIT = 0x4000
GL_DEPTH_BUFFER_BIT = 0x0100
GL_PROJECTION = 0x1701
GL_MODELVIEW = 0x1700
GL_POINTS = 0x0000
GL_LINES = 0x0001
GL_QUADS = 0x0007
GL_FLOAT = 0x1406
GL_VERTEX_ARRAY = 0x8074
GL_COLOR_ARRAY = 0x8076


class GLViewer(QOpenGLWidget):
    """3D point cloud viewer with trackball camera."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gl = None  # QOpenGLFunctions_3_3_Compatibility

        self._points = None
        self._colors = None
        self._center = np.zeros(3)
        self._extent = 10.0

        # Camera
        self._cam_yaw = 45.0
        self._cam_pitch = 30.0
        self._cam_dist = 50.0
        self._cam_target = np.zeros(3)
        self._cam_fov = 60.0

        # Interaction
        self._last_mouse = None
        self._mouse_button = None

        # Display
        self._point_size = 2.0
        self._bg_color = (0.15, 0.15, 0.18)
        self._show_axes = True
        self._show_grid = True

        # Bounding boxes
        self._bboxes = None   # list of BBox3D
        self._show_bboxes = True

        self.setFocusPolicy(Qt.StrongFocus)

    # -- Public API --

    def set_point_cloud(self, points, colors=None):
        """Set point cloud data. points: Nx3 float32, colors: Nx3 float32."""
        self._points = points.astype(np.float32) if points is not None else None
        self._colors = colors.astype(np.float32) if colors is not None else None
        if self._points is not None and len(self._points) > 0:
            # Filter NaN before computing bounds
            valid = np.isfinite(self._points).all(axis=1)
            pts = self._points[valid] if valid.any() else self._points
            self._center = (pts.min(axis=0) + pts.max(axis=0)) / 2
            ext = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
            self._extent = ext if math.isfinite(ext) and ext > 0 else 10.0
            self._cam_target = self._center.copy()
            self._cam_dist = self._extent * 1.8
            if self._cam_dist < 1.0:
                self._cam_dist = 10.0
        self.update()

    def set_point_size(self, size):
        self._point_size = max(0.5, min(20.0, size))
        self.update()

    def set_background(self, color):
        self._bg_color = color
        self.update()

    def reset_view(self):
        if self._points is not None:
            self._cam_target = self._center.copy()
            self._cam_dist = self._extent * 1.8
        self._cam_yaw = 45.0
        self._cam_pitch = 30.0
        self.update()

    def toggle_axes(self):
        self._show_axes = not self._show_axes
        self.update()

    def toggle_grid(self):
        self._show_grid = not self._show_grid
        self.update()

    def set_bboxes(self, bboxes):
        """Set bounding boxes for visualization."""
        self._bboxes = bboxes
        self.update()

    def toggle_bboxes(self):
        self._show_bboxes = not self._show_bboxes
        self.update()

    # -- OpenGL --

    def initializeGL(self):
        self._gl = QOpenGLFunctions_3_3_Compatibility()
        self._gl.initializeOpenGLFunctions()
        self._gl.glEnable(GL_DEPTH_TEST)
        self._gl.glEnable(GL_POINT_SMOOTH)
        self._gl.glEnable(GL_BLEND)
        self._gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def paintGL(self):
        gl = self._gl
        if gl is None:
            return

        gl.glClearColor(*self._bg_color, 1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Projection
        gl.glMatrixMode(GL_PROJECTION)
        gl.glLoadIdentity()
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        aspect = w / h
        near = max(self._cam_dist * 0.001, 0.01)
        far = max(self._cam_dist * 100, 1000)
        self._perspective(gl, self._cam_fov, aspect, near, far)

        # View
        gl.glMatrixMode(GL_MODELVIEW)
        gl.glLoadIdentity()
        eye = self._cam_eye()
        self._look_at(gl, eye, self._cam_target, np.array([0, 0, 1], dtype=np.float32))

        # Grid
        if self._show_grid:
            self._draw_grid(gl)

        # Axes
        if self._show_axes:
            self._draw_axes(gl)

        # Points
        if self._points is not None and len(self._points) > 0:
            self._draw_points(gl)

        # Bounding boxes
        if self._show_bboxes and self._bboxes:
            self._draw_bboxes(gl)

    def resizeGL(self, w, h):
        if self._gl:
            self._gl.glViewport(0, 0, w, h)

    # -- Drawing helpers --

    def _draw_points(self, gl):
        gl.glPointSize(self._point_size)
        gl.glEnableClientState(GL_VERTEX_ARRAY)

        # Ensure arrays are C-contiguous
        verts = np.ascontiguousarray(self._points, dtype=np.float32)
        gl.glVertexPointer(3, GL_FLOAT, 0, verts)

        if self._colors is not None:
            gl.glEnableClientState(GL_COLOR_ARRAY)
            cols = np.ascontiguousarray(self._colors, dtype=np.float32)
            gl.glColorPointer(3, GL_FLOAT, 0, cols)
        else:
            gl.glColor3f(0.2, 0.8, 0.4)

        gl.glDrawArrays(GL_POINTS, 0, len(self._points))

        gl.glDisableClientState(GL_VERTEX_ARRAY)
        if self._colors is not None:
            gl.glDisableClientState(GL_COLOR_ARRAY)

    def _draw_axes(self, gl):
        length = max(self._extent * 0.3, 1.0)
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        gl.glLineWidth(2.5)
        gl.glBegin(GL_LINES)
        # X - Red
        gl.glColor3f(0.9, 0.2, 0.2)
        gl.glVertex3f(*origin)
        gl.glVertex3f(origin[0] + length, origin[1], origin[2])
        # Y - Green
        gl.glColor3f(0.2, 0.8, 0.2)
        gl.glVertex3f(*origin)
        gl.glVertex3f(origin[0], origin[1] + length, origin[2])
        # Z - Blue
        gl.glColor3f(0.3, 0.4, 0.9)
        gl.glVertex3f(*origin)
        gl.glVertex3f(origin[0], origin[1], origin[2] + length)
        gl.glEnd()

        # Axis endpoint markers
        self._draw_marker(gl, origin[0] + length * 1.1, origin[1], origin[2], (0.9, 0.2, 0.2))
        self._draw_marker(gl, origin[0], origin[1] + length * 1.1, origin[2], (0.2, 0.8, 0.2))
        self._draw_marker(gl, origin[0], origin[1], origin[2] + length * 1.1, (0.3, 0.4, 0.9))

    def _draw_marker(self, gl, x, y, z, color):
        s = max(self._extent * 0.02, 0.1)
        gl.glLineWidth(2.0)
        gl.glColor3f(*color)
        gl.glBegin(GL_LINES)
        gl.glVertex3f(x - s, y, z)
        gl.glVertex3f(x + s, y, z)
        gl.glVertex3f(x, y - s, z)
        gl.glVertex3f(x, y + s, z)
        gl.glEnd()

    def _draw_grid(self, gl):
        center = self._cam_target
        size = max(self._extent * 1.5, 10.0)
        step = self._grid_step(size)
        n_lines = int(size / step) + 1
        z_level = self._center[2] - self._extent * 0.5 if self._points is not None else 0.0

        gl.glLineWidth(0.5)
        gl.glColor4f(0.4, 0.4, 0.4, 0.3)
        gl.glBegin(GL_LINES)
        x0 = center[0] - size / 2
        y0 = center[1] - size / 2
        for i in range(n_lines + 1):
            x = x0 + i * step
            gl.glVertex3f(x, y0, z_level)
            gl.glVertex3f(x, y0 + size, z_level)
        for i in range(n_lines + 1):
            y = y0 + i * step
            gl.glVertex3f(x0, y, z_level)
            gl.glVertex3f(x0 + size, y, z_level)
        gl.glEnd()

    def _grid_step(self, size):
        if not math.isfinite(size) or size <= 0:
            return 1.0
        target_lines = 10
        raw = size / target_lines
        if raw <= 0 or not math.isfinite(raw):
            return 1.0
        magnitude = 10 ** math.floor(math.log10(raw))
        for m in [1, 2, 5, 10]:
            if m * magnitude >= raw:
                return m * magnitude
        return magnitude

    def _draw_bboxes(self, gl):
        """Draw 3D bounding boxes."""
        for bbox in self._bboxes:
            r, g, b = bbox.color
            edges = bbox.get_edges()

            # Draw edges
            gl.glLineWidth(2.0)
            gl.glColor4f(r, g, b, 0.9)
            gl.glBegin(GL_LINES)
            for v1, v2 in edges:
                gl.glVertex3f(*v1)
                gl.glVertex3f(*v2)
            gl.glEnd()

            # Draw semi-transparent bottom face
            gl.glEnable(GL_BLEND)
            gl.glColor4f(r, g, b, 0.12)
            gl.glBegin(GL_QUADS)
            for v in bbox.get_bottom_face():
                gl.glVertex3f(*v)
            gl.glEnd()

            # Draw semi-transparent top face
            gl.glColor4f(r, g, b, 0.08)
            gl.glBegin(GL_QUADS)
            for v in bbox.get_top_face():
                gl.glVertex3f(*v)
            gl.glEnd()

            # Draw yaw direction arrow
            start, end, arrow_left, arrow_right = bbox.get_yaw_arrow()
            gl.glLineWidth(3.0)
            gl.glColor4f(1.0, 1.0, 1.0, 0.9)
            gl.glBegin(GL_LINES)
            gl.glVertex3f(*start)
            gl.glVertex3f(*end)
            gl.glVertex3f(*end)
            gl.glVertex3f(*arrow_left)
            gl.glVertex3f(*end)
            gl.glVertex3f(*arrow_right)
            gl.glEnd()

    # -- Camera --

    def _cam_eye(self):
        yaw = math.radians(self._cam_yaw)
        pitch = math.radians(self._cam_pitch)
        cp = math.cos(pitch)
        dx = self._cam_dist * cp * math.cos(yaw)
        dy = self._cam_dist * cp * math.sin(yaw)
        dz = self._cam_dist * math.sin(pitch)
        return self._cam_target + np.array([dx, dy, dz], dtype=np.float32)

    def _perspective(self, gl, fov, aspect, near, far):
        f = 1.0 / math.tan(math.radians(fov) / 2)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2 * far * near) / (near - far)
        m[3, 2] = -1
        gl.glLoadMatrixf(m.T.flatten().tolist())

    def _look_at(self, gl, eye, target, up):
        f = target - eye
        f = f / np.linalg.norm(f)
        u = up / np.linalg.norm(up)
        s = np.cross(f, u)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)

        m = np.eye(4, dtype=np.float32)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        m[0, 3] = -np.dot(s, eye)
        m[1, 3] = -np.dot(u, eye)
        m[2, 3] = np.dot(f, eye)
        gl.glLoadMatrixf(m.T.flatten().tolist())

    # -- Mouse --

    def mousePressEvent(self, event):
        self._last_mouse = event.position().toPoint()
        self._mouse_button = event.button()

    def mouseMoveEvent(self, event):
        if self._last_mouse is None:
            return
        pos = event.position().toPoint()
        dx = pos.x() - self._last_mouse.x()
        dy = pos.y() - self._last_mouse.y()
        self._last_mouse = pos

        if self._mouse_button == Qt.LeftButton:
            self._cam_yaw -= dx * 0.4
            self._cam_pitch += dy * 0.4
            self._cam_pitch = max(-89, min(89, self._cam_pitch))
        elif self._mouse_button == Qt.RightButton:
            scale = self._cam_dist * 0.002
            yaw = math.radians(self._cam_yaw)
            right = np.array([-math.sin(yaw), math.cos(yaw), 0], dtype=np.float32)
            up_vec = np.array([0, 0, 1], dtype=np.float32)
            self._cam_target -= right * dx * scale
            self._cam_target += up_vec * dy * scale

        self.update()

    def mouseReleaseEvent(self, event):
        self._last_mouse = None
        self._mouse_button = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self._cam_dist *= factor
        self._cam_dist = max(0.1, min(self._cam_dist, 100000))
        self.update()

    def mouseDoubleClickEvent(self, event):
        if self._points is not None:
            self._cam_target = self._center.copy()
            self.update()
