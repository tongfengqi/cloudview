"""Main window UI with file info panel, 3D viewer, and toolbar."""

import os
import re
import glob
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QFileDialog,
    QMessageBox, QGroupBox, QFrame, QApplication, QMenu
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence, QShortcut

from gl_viewer import GLViewer
from pointcloud_loader import validate_file, load_file
from annotation_loader import load_annotation, find_matching_json, index_annotation_dir
from colormap import values_to_colors_fast


class InfoPanel(QFrame):
    """Left panel showing file info and controls."""

    frame_changed = Signal(int)  # emitted when slider changes
    anno_clear_requested = Signal()   # emitted when clear anno folder button is clicked
    infer_clear_requested = Signal()  # emitted when clear infer folder button is clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(270)
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
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # File info group
        info_group = QGroupBox("File Info")
        info_layout = QVBoxLayout(info_group)
        self.lbl_frame = self._add_row(info_layout, "Frame: ", "—")
        self.lbl_pcdir = self._add_row(info_layout, "PC Dir: ", "—")
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

        # Ground truth Annotation group
        anno_group = QGroupBox("Ground Truth (GT)")
        anno_layout = QVBoxLayout(anno_group)

        # Folder row with clear button
        row_dir = QHBoxLayout()
        lbl_dir = QLabel("Folder: ")
        lbl_dir.setStyleSheet("font-weight: bold; color: #89b4fa;")
        self.lbl_anno_dir = QLabel("Auto")
        self.lbl_anno_dir.setStyleSheet("color: #cdd6f4;")
        self.lbl_anno_dir.setWordWrap(True)
        self.btn_clear_anno = QPushButton("✕")
        self.btn_clear_anno.setFixedSize(16, 16)
        self.btn_clear_anno.setToolTip("Reset to auto-detection (clear custom annotation folder)")
        self.btn_clear_anno.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #f38ba8;
                border: none;
                font-weight: bold;
                font-size: 11px;
                padding: 0;
            }
            QPushButton:hover {
                background-color: #45475a;
                border-radius: 8px;
            }
        """)
        self.btn_clear_anno.setVisible(False)
        self.btn_clear_anno.clicked.connect(self.anno_clear_requested.emit)
        row_dir.addWidget(lbl_dir)
        row_dir.addWidget(self.lbl_anno_dir, 1)
        row_dir.addWidget(self.btn_clear_anno)
        anno_layout.addLayout(row_dir)

        self.lbl_anno_file = self._add_row(anno_layout, "JSON: ", "—")
        self.lbl_anno = self._add_row(anno_layout, "BBox: ", "—")
        self.lbl_anno_detail = QLabel("")
        self.lbl_anno_detail.setWordWrap(True)
        self.lbl_anno_detail.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 1px 0;")
        anno_layout.addWidget(self.lbl_anno_detail)
        layout.addWidget(anno_group)

        # Model Inference group
        infer_group = QGroupBox("Model Inference (Pred)")
        infer_layout = QVBoxLayout(infer_group)

        row_infer = QHBoxLayout()
        lbl_inf_dir = QLabel("Folder: ")
        lbl_inf_dir.setStyleSheet("font-weight: bold; color: #fab387;")
        self.lbl_infer_dir = QLabel("—")
        self.lbl_infer_dir.setStyleSheet("color: #cdd6f4;")
        self.lbl_infer_dir.setWordWrap(True)
        self.btn_clear_infer = QPushButton("✕")
        self.btn_clear_infer.setFixedSize(16, 16)
        self.btn_clear_infer.setToolTip("Clear model inference folder")
        self.btn_clear_infer.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #f38ba8;
                border: none;
                font-weight: bold;
                font-size: 11px;
                padding: 0;
            }
            QPushButton:hover {
                background-color: #45475a;
                border-radius: 8px;
            }
        """)
        self.btn_clear_infer.setVisible(False)
        self.btn_clear_infer.clicked.connect(self.infer_clear_requested.emit)
        row_infer.addWidget(lbl_inf_dir)
        row_infer.addWidget(self.lbl_infer_dir, 1)
        row_infer.addWidget(self.btn_clear_infer)
        infer_layout.addLayout(row_infer)

        self.lbl_infer_file = self._add_row(infer_layout, "JSON: ", "—")
        self.lbl_infer_pred = self._add_row(infer_layout, "Pred: ", "—")
        self.lbl_infer_detail = QLabel("")
        self.lbl_infer_detail.setWordWrap(True)
        self.lbl_infer_detail.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 1px 0;")
        infer_layout.addWidget(self.lbl_infer_detail)
        layout.addWidget(infer_group)

        # Selected bbox group
        sel_group = QGroupBox("Selected Object")
        sel_layout = QVBoxLayout(sel_group)
        self.lbl_sel_type = self._add_row(sel_layout, "Type: ", "—")
        self.lbl_sel_score = self._add_row(sel_layout, "Score: ", "—")
        self.lbl_sel_pos = self._add_row(sel_layout, "Pos: ", "—")
        self.lbl_sel_size = self._add_row(sel_layout, "Size: ", "—")
        self.lbl_sel_yaw = self._add_row(sel_layout, "Yaw: ", "—")
        layout.addWidget(sel_group)

        # Status
        self.lbl_status = QLabel("Drag & drop a .bin, .pcd, or folder to start")
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

    def update_frame(self, current, total, suffix=""):
        self._suppress_slider = True
        if total > 0:
            txt = f"{current + 1} / {total}"
            if suffix:
                txt += f" ({suffix})"
            self.lbl_frame.setText(txt)
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

    def update_inference(self, bboxes):
        if not bboxes:
            self.lbl_infer_pred.setText("—")
            self.lbl_infer_detail.setText("")
            return
        self.lbl_infer_pred.setText(f"{len(bboxes)} predictions")
        types = {}
        for b in bboxes:
            types[b.obj_type] = types.get(b.obj_type, 0) + 1
        detail = ", ".join(f"{v}x {k}" for k, v in types.items())
        self.lbl_infer_detail.setText(detail)

    def show_selected_bbox(self, bbox):
        """Show details of a selected bounding box."""
        if bbox is None:
            self.lbl_sel_type.setText("—")
            self.lbl_sel_score.setText("—")
            self.lbl_sel_pos.setText("—")
            self.lbl_sel_size.setText("—")
            self.lbl_sel_yaw.setText("—")
            return
        import math
        self.lbl_sel_type.setText(bbox.obj_type)
        if getattr(bbox, 'score', None) is not None:
            self.lbl_sel_score.setText(f"{bbox.score:.3f} ({bbox.score * 100:.1f}%) [Pred]")
        else:
            self.lbl_sel_score.setText("Ground Truth (GT)")
        self.lbl_sel_pos.setText(f"({bbox.center[0]:.1f}, {bbox.center[1]:.1f}, {bbox.center[2]:.1f})")
        self.lbl_sel_size.setText(f"{bbox.size[0]:.1f} x {bbox.size[1]:.1f} x {bbox.size[2]:.1f}")
        self.lbl_sel_yaw.setText(f"{math.degrees(bbox.yaw):.1f} deg")

    def set_pc_folder(self, name, full_path=""):
        self.lbl_pcdir.setText(name)
        if full_path:
            self.lbl_pcdir.setToolTip(full_path)
        else:
            self.lbl_pcdir.setToolTip("")

    def set_anno_folder(self, name, full_path="", count=0):
        if full_path and count > 0:
            self.lbl_anno_dir.setText(f"{name} ({count})")
            self.lbl_anno_dir.setToolTip(f"{full_path}\n({count} JSON files indexed)")
            self.btn_clear_anno.setVisible(True)
        else:
            self.lbl_anno_dir.setText(name)
            self.lbl_anno_dir.setToolTip("Auto-detecting annotations from adjacent directories")
            self.btn_clear_anno.setVisible(False)

    def set_anno_file(self, filename, full_path=""):
        self.lbl_anno_file.setText(filename)
        if full_path:
            self.lbl_anno_file.setToolTip(full_path)
        else:
            self.lbl_anno_file.setToolTip("")

    def set_infer_folder(self, name, full_path="", count=0):
        if full_path and count > 0:
            self.lbl_infer_dir.setText(f"{name} ({count})")
            self.lbl_infer_dir.setToolTip(f"{full_path}\n({count} JSON files indexed)")
            self.btn_clear_infer.setVisible(True)
        else:
            self.lbl_infer_dir.setText("—")
            self.lbl_infer_dir.setToolTip("")
            self.btn_clear_infer.setVisible(False)

    def set_infer_file(self, filename, full_path=""):
        self.lbl_infer_file.setText(filename)
        if full_path:
            self.lbl_infer_file.setToolTip(full_path)
        else:
            self.lbl_infer_file.setToolTip("")

    def show_status(self, msg, color=None):
        self.lbl_status.setText(msg)
        if color:
            self.lbl_status.setStyleSheet(f"color: {color}; font-style: normal; padding: 4px;")
        else:
            self.lbl_status.setStyleSheet("color: #a6adc8; font-style: normal; padding: 4px;")

    def clear_info(self):
        for lbl in [self.lbl_filename, self.lbl_size, self.lbl_points,
                     self.lbl_dtype, self.lbl_x, self.lbl_y, self.lbl_z,
                     self.lbl_anno_file, self.lbl_anno,
                     self.lbl_infer_file, self.lbl_infer_pred]:
            lbl.setText("—")
        self.lbl_anno_detail.setText("")
        self.lbl_infer_detail.setText("")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CloudView")
        self.setMinimumSize(1150, 720)
        self.resize(1420, 860)
        self.setStyleSheet("""
            QMainWindow { background-color: #11111b; }
            QToolBar { background-color: #181825; border: none; padding: 4px; }
            QToolButton, QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 12px;
            }
            QToolButton:hover, QPushButton:hover { background-color: #45475a; }
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

        # Current data
        self._pc_data = None
        self._current_pc_path = None

        # Point cloud & annotation folder states
        self._pc_dir = None
        self._anno_dir = None
        self._anno_map = {}
        self._anno_numeric_map = {}

        # Model Inference results states
        self._infer_dir = None
        self._infer_map = {}
        self._infer_numeric_map = {}
        self._filter_infer_only = False

        # File list for navigation
        self._all_file_list = []  # all files in PC directory
        self._file_list = []      # active navigable files (may be filtered)
        self._current_index = -1
        self._navigating = False  # True when loading via prev/next/slider

        # Toolbar
        self._create_toolbar()

        # Connect info panel clear buttons
        self.info_panel.anno_clear_requested.connect(self._clear_anno_directory)
        self.info_panel.infer_clear_requested.connect(self._clear_infer_directory)

        # Accept drag & drop
        self.setAcceptDrops(True)

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
        btn_open.setToolTip("Open single point cloud file (.bin, .pcd)")
        btn_open.clicked.connect(self._open_file_dialog)
        toolbar.addWidget(btn_open)

        # Open PC folder
        self.btn_pc_folder = QPushButton(" PC Folder")
        self.btn_pc_folder.setToolTip("Select point cloud folder (.bin, .pcd)\n智能支持识别根目录或 bin/pcd 子目录")
        self.btn_pc_folder.clicked.connect(self._open_pc_folder_dialog)
        toolbar.addWidget(self.btn_pc_folder)

        # Open Anno folder (Ground Truth)
        self.btn_anno_folder = QPushButton(" Anno Folder")
        self.btn_anno_folder.setToolTip("Select Ground Truth annotation folder (.json)\n选择真实标注文件夹（右键可清除重置为自动探测）")
        self.btn_anno_folder.clicked.connect(self._open_anno_folder_dialog)
        self.btn_anno_folder.setContextMenuPolicy(Qt.CustomContextMenu)
        self.btn_anno_folder.customContextMenuRequested.connect(self._show_anno_folder_menu)
        toolbar.addWidget(self.btn_anno_folder)

        # Open Infer folder (Model Predictions)
        self.btn_infer_folder = QPushButton(" Infer Folder")
        self.btn_infer_folder.setToolTip("Select model inference results folder (.json)\n选择模型推理结果文件夹（右键可清除）")
        self.btn_infer_folder.clicked.connect(self._open_infer_folder_dialog)
        self.btn_infer_folder.setContextMenuPolicy(Qt.CustomContextMenu)
        self.btn_infer_folder.customContextMenuRequested.connect(self._show_infer_folder_menu)
        toolbar.addWidget(self.btn_infer_folder)

        toolbar.addSeparator()

        # Prev / Next frame
        btn_prev = QPushButton("◀ Prev (A)")
        btn_prev.clicked.connect(self._prev_frame)
        toolbar.addWidget(btn_prev)

        btn_next = QPushButton("Next (D) ▶")
        btn_next.clicked.connect(self._next_frame)
        toolbar.addWidget(btn_next)

        # Only Infer Filter toggle button
        self.btn_filter_infer = QPushButton(" Only Infer: OFF")
        self.btn_filter_infer.setToolTip("仅显示有推理结果的点云帧 (过滤无结果帧，左右切换只浏览有推理结果的帧)\nFilter to only point cloud frames that have inference results")
        self.btn_filter_infer.clicked.connect(self._toggle_infer_filter)
        toolbar.addWidget(self.btn_filter_infer)

        toolbar.addSeparator()

        # Color mode
        toolbar.addWidget(QLabel(" Color: "))
        self.combo_color = QComboBox()
        self.combo_color.addItems(["Uniform", "Z Height", "Intensity"])
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
        self.slider_size.setFixedWidth(110)
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
        btn_json.setToolTip("Load a single annotation or inference JSON file")
        btn_json.clicked.connect(self._open_json_dialog)
        toolbar.addWidget(btn_json)

        # Ground Truth BBox toggle
        self.btn_bbox = QPushButton("GT: ON")
        self.btn_bbox.setToolTip("Toggle Ground Truth bounding boxes visibility")
        self.btn_bbox.clicked.connect(self._toggle_bboxes)
        toolbar.addWidget(self.btn_bbox)

        # Model Prediction BBox toggle
        self.btn_pred_bbox = QPushButton("Pred: ON")
        self.btn_pred_bbox.setToolTip("Toggle Model Prediction bounding boxes visibility")
        self.btn_pred_bbox.clicked.connect(self._toggle_pred_bboxes)
        toolbar.addWidget(self.btn_pred_bbox)

    def _find_pc_files_in_directory(self, directory):
        """
        Find .bin or .pcd files in directory or its common subdirectories.
        Returns (actual_dir, file_list).
        """
        extensions = ('.bin', '.pcd')
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        if files:
            return directory, files

        # Check common subdirectories
        for sub in ['bin', 'pcd', 'lidar', 'pointcloud', 'points', 'data', 'velodyne']:
            sub_dir = os.path.join(directory, sub)
            if os.path.isdir(sub_dir):
                sub_files = []
                for ext in extensions:
                    sub_files.extend(glob.glob(os.path.join(sub_dir, f'*{ext}')))
                if sub_files:
                    return sub_dir, sub_files

        # Check 1 level of subdirectories
        try:
            for entry in os.scandir(directory):
                if entry.is_dir():
                    sub_files = []
                    for ext in extensions:
                        sub_files.extend(glob.glob(os.path.join(entry.path, f'*{ext}')))
                    if sub_files:
                        return entry.path, sub_files
        except Exception:
            pass

        return directory, []

    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Point Cloud", "",
            "Point Cloud (*.bin *.pcd);;Bin Files (*.bin);;PCD Files (*.pcd);;All Files (*)"
        )
        if path:
            self._load_file(path)

    def _open_pc_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Open Point Cloud Folder")
        if path:
            self._load_pc_directory(path)

    def _open_folder_dialog(self):
        self._open_pc_folder_dialog()

    def _load_pc_directory(self, directory):
        actual_dir, files = self._find_pc_files_in_directory(directory)
        if not files:
            QMessageBox.warning(
                self, "No Files",
                f"No .bin or .pcd files found in:\n{directory}\n(including subdirectories)"
            )
            return

        self._pc_dir = actual_dir
        self.info_panel.set_pc_folder(os.path.basename(actual_dir), actual_dir)
        self.btn_pc_folder.setText(f" PC: {os.path.basename(actual_dir)[:10]}")
        # Build the file list using actual_dir
        if files:
            self._build_file_list(files[0])
            if self._file_list:
                self._load_file(self._file_list[0], skip_popup=True)

    def _load_directory(self, directory):
        self._load_pc_directory(directory)

    # -- Ground Truth Annotation Folder --

    def _open_anno_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Open Ground Truth Annotation Folder")
        if path:
            self._set_anno_directory(path)

    def _set_anno_directory(self, directory):
        directory = os.path.normpath(os.path.abspath(directory))
        anno_map, numeric_map = index_annotation_dir(directory)
        if not anno_map:
            QMessageBox.warning(
                self, "No Annotations Found",
                f"No .json files found in:\n{directory}\n(including subdirectories)"
            )
            return

        self._anno_dir = directory
        self._anno_map = anno_map
        self._anno_numeric_map = numeric_map
        count = len(set(anno_map.values()))

        self.info_panel.set_anno_folder(os.path.basename(directory), directory, count)
        self.btn_anno_folder.setText(f" Anno: {os.path.basename(directory)[:8]}")
        self.btn_anno_folder.setToolTip(
            f"Annotation folder: {directory}\n"
            f"Indexed {count} JSON files\n"
            f"Right-click to clear / reset to auto-detect"
        )
        self.info_panel.show_status(
            f"Loaded annotation folder: {os.path.basename(directory)} ({count} files)",
            color="#a6e3a1"
        )

        # If infer filter is active and using anno source, update
        if self._filter_infer_only:
            self._apply_infer_filter()

        # If a point cloud is currently loaded, reload annotation
        if self._current_pc_path:
            self._reload_current_annotation(self._current_pc_path)

    def _clear_anno_directory(self):
        self._anno_dir = None
        self._anno_map = {}
        self._anno_numeric_map = {}
        self.info_panel.set_anno_folder("Auto", "", 0)
        self.btn_anno_folder.setText(" Anno Folder")
        self.btn_anno_folder.setToolTip(
            "Select annotation folder (.json) to match with point clouds\n"
            "Right-click to clear / reset to auto-detect"
        )
        self.info_panel.show_status("Annotation folder reset to auto-detect")

        if self._current_pc_path:
            self._reload_current_annotation(self._current_pc_path)

    def _show_anno_folder_menu(self, pos):
        menu = QMenu(self)
        act_select = menu.addAction("Select Annotation Folder...")
        act_select.triggered.connect(self._open_anno_folder_dialog)
        if getattr(self, '_anno_dir', None):
            act_clear = menu.addAction(f"Clear Annotation Folder ({os.path.basename(self._anno_dir)})")
            act_clear.triggered.connect(self._clear_anno_directory)
        menu.exec(self.btn_anno_folder.mapToGlobal(pos))

    # -- Model Inference Folder --

    def _open_infer_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Open Model Inference Results Folder")
        if path:
            self._set_infer_directory(path)

    def _set_infer_directory(self, directory):
        directory = os.path.normpath(os.path.abspath(directory))
        infer_map, numeric_map = index_annotation_dir(directory)
        if not infer_map:
            QMessageBox.warning(
                self, "No Inference Results Found",
                f"No .json files found in:\n{directory}\n(including subdirectories)"
            )
            return

        self._infer_dir = directory
        self._infer_map = infer_map
        self._infer_numeric_map = numeric_map
        count = len(set(infer_map.values()))

        self.info_panel.set_infer_folder(os.path.basename(directory), directory, count)
        self.btn_infer_folder.setText(f" Infer: {os.path.basename(directory)[:8]}")
        self.btn_infer_folder.setToolTip(
            f"Inference folder: {directory}\n"
            f"Indexed {count} JSON files\n"
            f"Right-click to clear"
        )
        self.info_panel.show_status(
            f"Loaded inference folder: {os.path.basename(directory)} ({count} files)",
            color="#a6e3a1"
        )

        # If infer filter is active, update
        if self._filter_infer_only:
            self._apply_infer_filter()

        # If a point cloud is currently loaded, reload predictions
        if self._current_pc_path:
            self._reload_current_inference(self._current_pc_path)

    def _clear_infer_directory(self):
        self._infer_dir = None
        self._infer_map = {}
        self._infer_numeric_map = {}
        self.info_panel.set_infer_folder("—", "", 0)
        self.btn_infer_folder.setText(" Infer Folder")
        self.btn_infer_folder.setToolTip("Select model inference folder (.json)\nRight-click to clear")
        self.viewer.set_pred_bboxes([])
        self.info_panel.update_inference([])
        self.info_panel.set_infer_file("—", "")

        # If filter was on, turn it off
        if self._filter_infer_only:
            self._toggle_infer_filter()

        self.info_panel.show_status("Inference folder cleared")

    def _show_infer_folder_menu(self, pos):
        menu = QMenu(self)
        act_select = menu.addAction("Select Inference Folder...")
        act_select.triggered.connect(self._open_infer_folder_dialog)
        if getattr(self, '_infer_dir', None):
            act_clear = menu.addAction(f"Clear Inference Folder ({os.path.basename(self._infer_dir)})")
            act_clear.triggered.connect(self._clear_infer_directory)
        menu.exec(self.btn_infer_folder.mapToGlobal(pos))

    # -- Matching & File List --

    def _find_json_for_pointcloud(self, pc_path):
        """Find matching Ground Truth JSON file for a point cloud."""
        if self._anno_map:
            stem = os.path.splitext(os.path.basename(pc_path))[0]
            stem_lower = stem.lower()
            if stem_lower in self._anno_map:
                return self._anno_map[stem_lower]
            stripped = re.sub(r'^\d+_', '', stem_lower)
            if stripped in self._anno_map:
                return self._anno_map[stripped]
            if stem.isdigit():
                try:
                    int_val = int(stem)
                    if int_val in self._anno_numeric_map:
                        return self._anno_numeric_map[int_val]
                except ValueError:
                    pass
            # Also check direct file in _anno_dir
            fname = stem + '.json'
            direct = os.path.join(self._anno_dir, fname)
            if os.path.isfile(direct):
                return direct
            return None
        else:
            json_hint = getattr(self, '_custom_json_hint', None)
            return find_matching_json(pc_path, json_hint)

    def _find_infer_for_pointcloud(self, pc_path):
        """Find matching Model Inference JSON file for a point cloud."""
        if not self._infer_map:
            return None
        stem = os.path.splitext(os.path.basename(pc_path))[0]
        stem_lower = stem.lower()
        if stem_lower in self._infer_map:
            return self._infer_map[stem_lower]
        stripped = re.sub(r'^\d+_', '', stem_lower)
        if stripped in self._infer_map:
            return self._infer_map[stripped]
        if stem.isdigit():
            try:
                int_val = int(stem)
                if int_val in self._infer_numeric_map:
                    return self._infer_numeric_map[int_val]
            except ValueError:
                pass
        # Check direct file in _infer_dir
        fname = stem + '.json'
        direct = os.path.join(self._infer_dir, fname)
        if os.path.isfile(direct):
            return direct
        return None

    def _has_infer_for_pointcloud(self, pc_path):
        """Check whether a point cloud file has matching inference results."""
        if self._infer_map:
            return self._find_infer_for_pointcloud(pc_path) is not None
        elif self._anno_map:
            # If user loaded inference results into _anno_map
            return self._find_json_for_pointcloud(pc_path) is not None
        return False

    def _reload_current_annotation(self, pc_path):
        """Match and reload GT annotation for current frame."""
        json_path = self._find_json_for_pointcloud(pc_path)
        if json_path:
            bboxes = load_annotation(json_path)
            # If this is actually an inference predictions file loaded via anno folder
            if bboxes and getattr(bboxes[0], 'is_prediction', False):
                self.viewer.set_pred_bboxes(bboxes)
                self.info_panel.update_inference(bboxes)
                self.info_panel.set_infer_file(os.path.basename(json_path), json_path)
            else:
                self.viewer.set_bboxes(bboxes)
                self.info_panel.update_annotation(bboxes)
                self.info_panel.set_anno_file(os.path.basename(json_path), json_path)
            self.info_panel.show_status(f"Matched: {os.path.basename(json_path)}")
        else:
            self.viewer.set_bboxes([])
            self.info_panel.update_annotation([])
            self.info_panel.set_anno_file("—", "")
            if self._anno_dir:
                self.info_panel.show_status(f"No match for {os.path.basename(pc_path)} in annotation folder")

    def _reload_current_inference(self, pc_path):
        """Match and reload Model Inference predictions for current frame."""
        infer_path = self._find_infer_for_pointcloud(pc_path)
        if infer_path:
            pred_bboxes = load_annotation(infer_path)
            self.viewer.set_pred_bboxes(pred_bboxes)
            self.info_panel.update_inference(pred_bboxes)
            self.info_panel.set_infer_file(os.path.basename(infer_path), infer_path)
            self.info_panel.show_status(f"Matched Infer: {os.path.basename(infer_path)}")
        else:
            self.viewer.set_pred_bboxes([])
            self.info_panel.update_inference([])
            self.info_panel.set_infer_file("—", "")

    # -- Filter Logic: "Only Infer" --

    def _toggle_infer_filter(self):
        has_source = bool(self._infer_map or self._anno_map)
        if not has_source:
            QMessageBox.information(
                self, "No Inference Results",
                "请先通过 [Infer Folder] 或 [Anno Folder] 加载推理结果文件夹！\nPlease load an inference or annotation folder first."
            )
            return

        if not self._all_file_list:
            return

        self._filter_infer_only = not self._filter_infer_only
        self._apply_infer_filter()

    def _apply_infer_filter(self):
        if self._filter_infer_only:
            filtered = [f for f in self._all_file_list if self._has_infer_for_pointcloud(f)]
            if not filtered:
                QMessageBox.information(
                    self, "No Matches",
                    "当前点云列表中没有与推理结果匹配的帧。\nNo point cloud frames match the loaded inference results."
                )
                self._filter_infer_only = False
                self.btn_filter_infer.setText(" Only Infer: OFF")
                self.btn_filter_infer.setStyleSheet("")
                return

            self._file_list = filtered
            self.btn_filter_infer.setText(" Only Infer: ON")
            self.btn_filter_infer.setStyleSheet("background-color: #a6e3a1; color: #11111b; font-weight: bold;")

            # Keep current frame if in filtered list, otherwise jump to first filtered frame
            if self._current_pc_path and self._current_pc_path in self._file_list:
                self._current_index = self._file_list.index(self._current_pc_path)
                self.info_panel.update_frame(self._current_index, len(self._file_list), suffix="Infer Only")
            else:
                self._current_index = 0
                self._load_file(self._file_list[0], skip_popup=True)

            self.info_panel.show_status(
                f"Filter ON: viewing {len(self._file_list)} / {len(self._all_file_list)} inference frames",
                color="#a6e3a1"
            )
        else:
            self._file_list = list(self._all_file_list)
            self.btn_filter_infer.setText(" Only Infer: OFF")
            self.btn_filter_infer.setStyleSheet("")

            if self._current_pc_path and self._current_pc_path in self._file_list:
                self._current_index = self._file_list.index(self._current_pc_path)
                self.info_panel.update_frame(self._current_index, len(self._file_list))
            else:
                self._current_index = 0
                self._load_file(self._file_list[0], skip_popup=True)

            self.info_panel.show_status(f"Filter OFF: viewing all {len(self._file_list)} frames")

    def _build_file_list(self, filepath):
        directory = os.path.dirname(os.path.abspath(filepath))
        extensions = ('.bin', '.pcd')
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, f'*{ext}')))

        def natural_keys(filepath):
            basename = os.path.basename(filepath)
            stem = os.path.splitext(basename)[0]

            group = 0 if stem.isdigit() and len(stem) == 6 else 1

            parts = re.split(r'(\d+)', filepath)
            result = [group]
            for p in parts:
                if p.isdigit():
                    if p.startswith('0'):
                        result.append((1, p))
                    else:
                        result.append((2, int(p)))
                else:
                    result.append((0, p))
            return result

        self._all_file_list = sorted((os.path.normpath(f) for f in files), key=natural_keys)

        if self._filter_infer_only:
            self._file_list = [f for f in self._all_file_list if self._has_infer_for_pointcloud(f)]
            if not self._file_list:
                # If no matching frames in new folder, reset filter
                self._file_list = list(self._all_file_list)
                self._filter_infer_only = False
                self.btn_filter_infer.setText(" Only Infer: OFF")
                self.btn_filter_infer.setStyleSheet("")
        else:
            self._file_list = list(self._all_file_list)

        target = os.path.normpath(os.path.abspath(filepath))
        try:
            self._current_index = self._file_list.index(target)
        except ValueError:
            self._current_index = 0 if self._file_list else -1

    def _load_file(self, filepath, skip_popup=False, custom_json_hint=None):
        if custom_json_hint:
            self._custom_json_hint = custom_json_hint

        ok, msg, info = validate_file(filepath)
        if ok is False:
            if not skip_popup:
                QMessageBox.warning(self, "Invalid Format", msg)
            self.info_panel.show_error(msg.split('\n')[0])
            return

        norm = os.path.normpath(os.path.abspath(filepath))
        if not self._file_list or norm not in self._file_list:
            self._build_file_list(filepath)
            if not self._pc_dir:
                parent_dir = os.path.dirname(norm)
                self.info_panel.set_pc_folder(os.path.basename(parent_dir), parent_dir)
        else:
            try:
                self._current_index = self._file_list.index(norm)
            except ValueError:
                self._current_index = -1

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
            self._current_pc_path = filepath
            self.info_panel.update_info(pc_data)
            suffix = "Infer Only" if self._filter_infer_only else ""
            self.info_panel.update_frame(self._current_index, len(self._file_list), suffix=suffix)
            self.setWindowTitle(f"CloudView — {os.path.basename(filepath)}")

            self._apply_colors()

            # 1. Match and load Ground Truth annotation
            json_path = self._find_json_for_pointcloud(filepath)
            if json_path:
                if skip_popup or getattr(self, '_anno_dir', None) or getattr(self, '_custom_json_hint', None):
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
                    else:
                        self.viewer.set_bboxes([])
                        self.info_panel.update_annotation([])
                        self.info_panel.set_anno_file("—", "")
            else:
                self.viewer.set_bboxes([])
                self.info_panel.update_annotation([])
                self.info_panel.set_anno_file("—", "")

            # 2. Match and load Model Inference prediction
            if self._infer_map:
                infer_path = self._find_infer_for_pointcloud(filepath)
                if infer_path:
                    pred_bboxes = load_annotation(infer_path)
                    self.viewer.set_pred_bboxes(pred_bboxes)
                    self.info_panel.update_inference(pred_bboxes)
                    self.info_panel.set_infer_file(os.path.basename(infer_path), infer_path)
                else:
                    self.viewer.set_pred_bboxes([])
                    self.info_panel.update_inference([])
                    self.info_panel.set_infer_file("—", "")

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
            self, "Open Annotation / Inference JSON", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self._load_json(path)

    def _load_json(self, filepath):
        try:
            bboxes = load_annotation(filepath)
            # Route to predictions or GT
            if bboxes and getattr(bboxes[0], 'is_prediction', False):
                self.viewer.set_pred_bboxes(bboxes)
                self.info_panel.update_inference(bboxes)
                self.info_panel.set_infer_file(os.path.basename(filepath), filepath)
            else:
                self.viewer.set_bboxes(bboxes)
                self.info_panel.update_annotation(bboxes)
                self.info_panel.set_anno_file(os.path.basename(filepath), filepath)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load annotation:\n{e}")

    def _toggle_bboxes(self):
        self.viewer.toggle_bboxes()
        if self.viewer._show_bboxes:
            self.btn_bbox.setText("GT: ON")
        else:
            self.btn_bbox.setText("GT: OFF")

    def _toggle_pred_bboxes(self):
        self.viewer.toggle_pred()
        if self.viewer._show_pred:
            self.btn_pred_bbox.setText("Pred: ON")
        else:
            self.btn_pred_bbox.setText("Pred: OFF")

    def _apply_colors(self):
        if self._pc_data is None:
            return
        mode = self.combo_color.currentText()
        cmap = self.combo_cmap.currentText()
        if mode == "Z Height":
            values = self._pc_data.points[:, 2]
            colors = values_to_colors_fast(values, cmap)
        elif mode == "Intensity" and getattr(self._pc_data, 'intensities', None) is not None:
            values = self._pc_data.intensities
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
                path = url.toLocalFile()
                if os.path.isdir(path):
                    event.acceptProposedAction()
                    return
                ext = os.path.splitext(path)[1].lower()
                if ext in ('.bin', '.pcd', '.json'):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent):
        dropped_paths = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile()]
        if not dropped_paths:
            return

        for path in dropped_paths:
            if os.path.isdir(path):
                # 1. Check if it has point clouds
                actual_pc_dir, pc_files = self._find_pc_files_in_directory(path)
                if pc_files:
                    self._load_pc_directory(path)
                    continue

                # 2. Check if it has json annotations / inference
                anno_map, _ = index_annotation_dir(path)
                if anno_map:
                    # Peek at a sample JSON to determine whether it's inference predictions or GT
                    first_json = next(iter(anno_map.values()))
                    is_pred = False
                    try:
                        import json
                        with open(first_json, 'r', encoding='utf-8') as jf:
                            jd = json.load(jf)
                            if isinstance(jd, dict) and ('pred_bboxes_3d' in jd or 'boxes_3d' in jd):
                                is_pred = True
                    except Exception:
                        pass

                    if is_pred:
                        self._set_infer_directory(path)
                    else:
                        self._set_anno_directory(path)
                    continue
            else:
                ext = os.path.splitext(path)[1].lower()
                if ext in ('.bin', '.pcd'):
                    self._load_file(path)
                elif ext == '.json':
                    self._load_json(path)
