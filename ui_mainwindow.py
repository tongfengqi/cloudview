"""Main window UI with file info panel, 3D viewer, and toolbar."""

import os
import glob
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QFileDialog,
    QMessageBox, QGroupBox, QFrame, QApplication
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence, QShortcut

from gl_viewer import GLViewer
from pointcloud_loader import validate_file, load_file
from annotation_loader import load_annotation, find_matching_json
from colormap import values_to_colors_fast


class InfoPanel(QFrame):
    """Left panel showing file info and controls."""

    frame_changed = Signal(int)  # emitted when slider changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(260)
        self._suppress_slider = False
        self.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                border: 1px solid #333344;
                border-radius: 6px;
            }
            QLabel { color: #cdd6f4; }
            QGroupBox {
                color: #89b4fa;
                border: 1px solid #333344;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 12px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #45475a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: -5px 0;
                background: #89b4fa;
                border-radius: 7px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # File info group
        info_group = QGroupBox("File Info")
        info_layout = QVBoxLayout(info_group)
        self.lbl_frame = self._add_row(info_layout, "Frame: ", "—")
        self.lbl_filename = self._add_row(info_layout, "File: ", "—")
        self.lbl_size = self._add_row(info_layout, "Size: ", "—")
        self.lbl_points = self._add_row(info_layout, "Points: ", "—")
        self.lbl_dtype = self._add_row(info_layout, "Format: ", "—")
        layout.addWidget(info_group)

        # Frame slider
        slider_group = QGroupBox("Frame Slider")
        slider_layout = QVBoxLayout(slider_group)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.frame_slider)
        layout.addWidget(slider_group)

        # Range group
        range_group = QGroupBox("Coordinate Range")
        range_layout = QVBoxLayout(range_group)
        self.lbl_x = self._add_row(range_layout, "X (fwd): ", "—")
        self.lbl_y = self._add_row(range_layout, "Y (left): ", "—")
        self.lbl_z = self._add_row(range_layout, "Z (up):  ", "—")
        layout.addWidget(range_group)

        # Annotation group
        anno_group = QGroupBox("Annotation")
        anno_layout = QVBoxLayout(anno_group)
        self.lbl_anno = self._add_row(anno_layout, "BBox: ", "—")
        self.lbl_anno_detail = QLabel("")
        self.lbl_anno_detail.setWordWrap(True)
        self.lbl_anno_detail.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 2px 0;")
        anno_layout.addWidget(self.lbl_anno_detail)
        layout.addWidget(anno_group)

        # Selected bbox group
        sel_group = QGroupBox("Selected Object")
        sel_layout = QVBoxLayout(sel_group)
        self.lbl_sel_type = self._add_row(sel_layout, "Type: ", "—")
        self.lbl_sel_pos = self._add_row(sel_layout, "Pos: ", "—")
        self.lbl_sel_size = self._add_row(sel_layout, "Size: ", "—")
        self.lbl_sel_yaw = self._add_row(sel_layout, "Yaw: ", "—")
        layout.addWidget(sel_group)

        # Status
        self.lbl_status = QLabel("Drag & drop a .bin or .pcd file to start")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #a6adc8; font-style: italic; padding: 4px;")
        layout.addWidget(self.lbl_status)

        layout.addStretch()

    def _add_row(self, parent_layout, label_text, default):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setStyleSheet("font-weight: bold; color: #89b4fa;")
        val = QLabel(default)
        val.setStyleSheet("color: #cdd6f4;")
        val.setWordWrap(True)
        row.addWidget(lbl)
        row.addWidget(val, 1)
        parent_layout.addLayout(row)
        return val

    def _on_slider_changed(self, value):
        if not self._suppress_slider:
            self.frame_changed.emit(value)

    def update_frame(self, current, total):
        self._suppress_slider = True
        if total > 0:
            self.lbl_frame.setText(f"{current + 1} / {total}")
            self.frame_slider.setRange(0, total - 1)
            self.frame_slider.setValue(current)
        else:
            self.lbl_frame.setText("—")
            self.frame_slider.setRange(0, 0)
        self._suppress_slider = False

    def update_info(self, pc_data):
        self.lbl_filename.setText(pc_data.filename)
        self.lbl_size.setText(f"{pc_data.file_size / 1024 / 1024:.1f} MB")
        if pc_data.header_points and pc_data.header_points != pc_data.num_points:
            self.lbl_points.setText(f"{pc_data.num_points:,} / {pc_data.header_points:,}")
        else:
            self.lbl_points.setText(f"{pc_data.num_points:,}")
        self.lbl_dtype.setText(pc_data.format)
        self.lbl_x.setText(f"[{pc_data.x_range[0]:.2f}, {pc_data.x_range[1]:.2f}]")
        self.lbl_y.setText(f"[{pc_data.y_range[0]:.2f}, {pc_data.y_range[1]:.2f}]")
        self.lbl_z.setText(f"[{pc_data.z_range[0]:.2f}, {pc_data.z_range[1]:.2f}]")
        self.lbl_status.setText("Loaded successfully")
        self.lbl_status.setStyleSheet("color: #a6e3a1; font-style: normal; padding: 4px;")

    def show_error(self, msg):
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet("color: #f38ba8; font-style: normal; padding: 4px;")

    def update_annotation(self, bboxes):
        if not bboxes:
            self.lbl_anno.setText("—")
            self.lbl_anno_detail.setText("")
            return
        self.lbl_anno.setText(f"{len(bboxes)} objects")
        types = {}
        for b in bboxes:
            types[b.obj_type] = types.get(b.obj_type, 0) + 1
        detail = ", ".join(f"{v}x {k}" for k, v in types.items())
        self.lbl_anno_detail.setText(detail)

    def show_selected_bbox(self, bbox):
        """Show details of a selected bounding box."""
        if bbox is None:
            self.lbl_sel_type.setText("—")
            self.lbl_sel_pos.setText("—")
            self.lbl_sel_size.setText("—")
            self.lbl_sel_yaw.setText("—")
            return
        import math
        self.lbl_sel_type.setText(bbox.obj_type)
        self.lbl_sel_pos.setText(f"({bbox.center[0]:.1f}, {bbox.center[1]:.1f}, {bbox.center[2]:.1f})")
        self.lbl_sel_size.setText(f"{bbox.size[0]:.1f} x {bbox.size[1]:.1f} x {bbox.size[2]:.1f}")
        self.lbl_sel_yaw.setText(f"{math.degrees(bbox.yaw):.1f} deg")

    def clear_info(self):
        for lbl in [self.lbl_filename, self.lbl_size, self.lbl_points,
                     self.lbl_dtype, self.lbl_x, self.lbl_y, self.lbl_z]:
            lbl.setText("—")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CloudView")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)
        self.setStyleSheet("""
            QMainWindow { background-color: #11111b; }
            QToolBar { background-color: #181825; border: none; padding: 4px; }
            QToolButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 12px;
            }
            QToolButton:hover { background-color: #45475a; }
            QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #45475a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                margin: -5px 0;
                background: #89b4fa;
                border-radius: 7px;
            }
        """)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Info panel
        self.info_panel = InfoPanel()
        self.info_panel.frame_changed.connect(self._on_slider_jump)
        main_layout.addWidget(self.info_panel)

        # 3D Viewer
        self.viewer = GLViewer()
        self.viewer.bbox_selected.connect(self.info_panel.show_selected_bbox)
        main_layout.addWidget(self.viewer, 1)

        # Toolbar
        self._create_toolbar()

        # Accept drag & drop
        self.setAcceptDrops(True)

        # Current data
        self._pc_data = None

        # File list for navigation
        self._file_list = []
        self._current_index = -1
        self._navigating = False  # True when loading via prev/next/slider

        # Keyboard shortcuts
        for key, slot in [
            (Qt.Key_Right, self._next_frame),
            (Qt.Key_D, self._next_frame),
            (Qt.Key_Left, self._prev_frame),
            (Qt.Key_A, self._prev_frame),
            (Qt.Key_Escape, self._clear_selection),
        ]:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(slot)

    def _create_toolbar(self):
        toolbar = self.addToolBar("Controls")
        toolbar.setMovable(False)

        # Open file
        btn_open = QPushButton(" Open File")
        btn_open.clicked.connect(self._open_file_dialog)
        toolbar.addWidget(btn_open)

        toolbar.addSeparator()

        # Prev / Next frame
        btn_prev = QPushButton("◀ Prev (A)")
        btn_prev.clicked.connect(self._prev_frame)
        toolbar.addWidget(btn_prev)

        btn_next = QPushButton("Next (D) ▶")
        btn_next.clicked.connect(self._next_frame)
        toolbar.addWidget(btn_next)

        toolbar.addSeparator()

        # Color mode
        toolbar.addWidget(QLabel(" Color: "))
        self.combo_color = QComboBox()
        self.combo_color.addItems(["Uniform", "Z Height"])
        self.combo_color.currentTextChanged.connect(self._on_color_mode_changed)
        toolbar.addWidget(self.combo_color)

        # Colormap
        toolbar.addWidget(QLabel(" Map: "))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["viridis", "plasma"])
        self.combo_cmap.currentTextChanged.connect(self._on_color_mode_changed)
        toolbar.addWidget(self.combo_cmap)

        toolbar.addSeparator()

        # Point size
        toolbar.addWidget(QLabel(" Size: "))
        self.slider_size = QSlider(Qt.Horizontal)
        self.slider_size.setRange(1, 10)
        self.slider_size.setValue(2)
        self.slider_size.setFixedWidth(120)
        self.slider_size.valueChanged.connect(lambda v: self.viewer.set_point_size(v))
        toolbar.addWidget(self.slider_size)

        toolbar.addSeparator()

        # Background
        btn_bg = QPushButton("BG Toggle")
        btn_bg.clicked.connect(self._toggle_bg)
        toolbar.addWidget(btn_bg)

        # Axes
        btn_axes = QPushButton("Axes")
        btn_axes.clicked.connect(self.viewer.toggle_axes)
        toolbar.addWidget(btn_axes)

        # Grid
        btn_grid = QPushButton("Grid")
        btn_grid.clicked.connect(self.viewer.toggle_grid)
        toolbar.addWidget(btn_grid)

        # Reset
        btn_reset = QPushButton("Reset View")
        btn_reset.clicked.connect(self.viewer.reset_view)
        toolbar.addWidget(btn_reset)

        toolbar.addSeparator()

        # Load JSON annotation
        btn_json = QPushButton("Load JSON")
        btn_json.clicked.connect(self._open_json_dialog)
        toolbar.addWidget(btn_json)

        # BBox toggle
        self.btn_bbox = QPushButton("BBox: ON")
        self.btn_bbox.clicked.connect(self._toggle_bboxes)
        toolbar.addWidget(self.btn_bbox)

    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Point Cloud", "",
            "Point Cloud (*.bin *.pcd);;Bin Files (*.bin);;PCD Files (*.pcd);;All Files (*)"
        )
        if path:
            self._load_file(path)

    def _build_file_list(self, filepath):
        directory = os.path.dirname(os.path.abspath(filepath))
        extensions = ('.bin', '.pcd')
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        self._file_list = sorted(os.path.normpath(f) for f in files)
        target = os.path.normpath(os.path.abspath(filepath))
        try:
            self._current_index = self._file_list.index(target)
        except ValueError:
            self._current_index = -1

    def _load_file(self, filepath, skip_popup=False):
        ok, msg, info = validate_file(filepath)
        if ok is False:
            if not skip_popup:
                QMessageBox.warning(self, "Invalid Format", msg)
            self.info_panel.show_error(msg.split('\n')[0])
            return

        norm = os.path.normpath(os.path.abspath(filepath))
        if not self._file_list or norm not in self._file_list:
            self._build_file_list(filepath)

        try:
            max_pts = None
            num_points = info['num_points']
            if num_points > 5_000_000 and not skip_popup:
                reply = QMessageBox.question(
                    self, "Large File",
                    f"This file has {num_points:,} points.\n"
                    f"Load all points or subsample to 2,000,000?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                if reply == QMessageBox.StandardButton.No:
                    max_pts = 2_000_000

            pc_data = load_file(filepath, max_pts)
            self._pc_data = pc_data
            self.info_panel.update_info(pc_data)
            self.info_panel.update_frame(self._current_index, len(self._file_list))
            self.setWindowTitle(f"CloudView — {os.path.basename(filepath)}")

            self._apply_colors()

            json_path = find_matching_json(filepath)
            if json_path:
                if skip_popup:
                    self._load_json(json_path)
                else:
                    reply = QMessageBox.question(
                        self, "Annotation Found",
                        f"Found matching annotation:\n{os.path.basename(json_path)}\n\nLoad it?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self._load_json(json_path)

        except Exception as e:
            if not skip_popup:
                QMessageBox.critical(self, "Error", str(e))
            self.info_panel.show_error(str(e))

    def _on_slider_jump(self, index):
        """Jump to frame when slider is dragged."""
        if 0 <= index < len(self._file_list):
            self._current_index = index
            self._navigating = True
            self._load_file(self._file_list[self._current_index], skip_popup=True)
            self._navigating = False

    def _next_frame(self):
        if not self._file_list or self._current_index < 0:
            return
        if self._current_index < len(self._file_list) - 1:
            self._current_index += 1
            self._navigating = True
            self._load_file(self._file_list[self._current_index], skip_popup=True)
            self._navigating = False

    def _prev_frame(self):
        if not self._file_list or self._current_index < 0:
            return
        if self._current_index > 0:
            self._current_index -= 1
            self._navigating = True
            self._load_file(self._file_list[self._current_index], skip_popup=True)
            self._navigating = False

    def _open_json_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Annotation", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self._load_json(path)

    def _load_json(self, filepath):
        try:
            bboxes = load_annotation(filepath)
            self.viewer.set_bboxes(bboxes)
            self.info_panel.update_annotation(bboxes)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load annotation:\n{e}")

    def _toggle_bboxes(self):
        self.viewer.toggle_bboxes()
        if self.viewer._show_bboxes:
            self.btn_bbox.setText("BBox: ON")
        else:
            self.btn_bbox.setText("BBox: OFF")

    def _apply_colors(self):
        if self._pc_data is None:
            return
        mode = self.combo_color.currentText()
        cmap = self.combo_cmap.currentText()
        if mode == "Z Height":
            values = self._pc_data.points[:, 2]
            colors = values_to_colors_fast(values, cmap)
        else:
            colors = None
        self.viewer.set_point_cloud(self._pc_data.points, colors, reset_camera=not self._navigating)

    def _on_color_mode_changed(self, _):
        self._apply_colors()

    def _clear_selection(self):
        self.viewer._selected_bbox = None
        self.viewer.update()
        self.info_panel.show_selected_bbox(None)

    def _toggle_bg(self):
        bg = self.viewer._bg_color
        if bg[0] < 0.3:
            self.viewer.set_background((0.92, 0.92, 0.90))
        else:
            self.viewer.set_background((0.15, 0.15, 0.18))

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = os.path.splitext(url.toLocalFile())[1].lower()
                if ext in ('.bin', '.pcd'):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.bin', '.pcd'):
                self._load_file(path)
                break
