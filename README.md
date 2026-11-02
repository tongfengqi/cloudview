# CloudView

A lightweight 3D point cloud viewer for `.bin` and `.pcd` files, with JSON ground-truth annotation and model inference visualization.

## Features

- **File formats**: Load `.bin` (float32 x4) and `.pcd` (binary/ASCII) point clouds
- **3D rendering**: OpenGL with trackball camera (rotate, pan, zoom)
- **Color modes**: Z-height coloring (viridis/plasma colormaps) or uniform color
- **Scene aids**: RGB coordinate axes (X=red, Y=green, Z=blue), adjustable ground grid
- **Bounding boxes**: Load JSON annotations and model predictions, display 3D BBox with semi-transparent faces and yaw direction arrows
- **3D ID & Score Labels**: Real-time projected 2D badges displayed next to/above each 3D box with category-colored borders and prediction confidence scores
- **Independent Folder Loading**: Load point cloud folder, Ground Truth annotation folder, and Model Inference results folder independently
- **Fast In-Memory Indexing**: Millisecond-level indexing of tens of thousands of JSON files, supporting multi-level subdirectories (`bin/`, `pcd/`, `json/`, `bev_lidar/`, index prefixes `0_`, `1_`, etc.)
- **Only Infer Filtering**: One-click filter button (`Only Infer: ON/OFF`) to scrub strictly through point cloud frames that have inference results (e.g. 22 frames out of 25,000+)
- **Dual / Triple Annotation Formats**: Supports converted format (List with `psr`), raw dataset format (Dict with `translation`, `size` [w,l,h], quaternion `[w,x,y,z]`), and model inference format (`pred_bboxes_3d`, `pred_scores`, `pred_names`)
- **Auto-matching & Frame Navigation**: Seamless Prev/Next frame switching with synchronized GT and Pred loading
- **Validation**: Detects non-float32 or non-4-column files with clear error messages
- **Drag & drop**: Drop `.bin`, `.pcd`, `.json` files or entire folders directly into the window

## Screenshot

![screenshot](screenshot.png)

## Requirements

- Python 3.9+
- PySide6

```bash
pip install PySide6
```

## Usage

```bash
# Open the viewer
python main.py

# Open with a specific file
python main.py path/to/pointcloud.bin
python main.py path/to/pointcloud.pcd
```

### Controls & Toolbar
 
| Action | Control / Shortcut | Description |
|--------|--------------------|-------------|
| Open File | `Open File` button | Open a single `.bin` or `.pcd` point cloud file |
| Open PC Folder | `PC Folder` button | Select folder with point clouds (auto-detects `bin/`, `pcd/`, etc.) |
| Open Anno Folder | `Anno Folder` button | Select folder with Ground Truth JSON annotations |
| Open Infer Folder | `Infer Folder` button | Select folder with Model Inference JSON results |
| Clear Anno / Infer Folder | Right-click folder button or click `✕` in Info Panel | Clear custom annotation / inference directory |
| Only Infer Filter | `Only Infer: OFF/ON` button | Filter point cloud sequence down strictly to frames with inference results |
| Next Frame | `D` or `Right Arrow` or `Next (D)` | Jump to next point cloud frame (respects active filter) |
| Prev Frame | `A` or `Left Arrow` or `Prev (A)` | Jump to previous point cloud frame (respects active filter) |
| Frame Slider | Horizontal Slider | Scrub across frames in the active sequence |
| Rotate | Left mouse drag | Orbit view around center |
| Pan | Right mouse drag | Translate view |
| Zoom | Mouse wheel | Zoom in/out |
| Reset view | Double-click / `Reset View` | Reset camera to default perspective |
| Select BBox | Left click on 3D box | Highlight box and view dimensions, coordinates, & confidence score in left panel |
| Clear Selection | `Escape` | Deselect active 3D bounding box |
| Toggle GT BBoxes | `GT: ON/OFF` button | Toggle Ground Truth 3D bounding boxes visibility |
| Toggle Pred BBoxes | `Pred: ON/OFF` button | Toggle Model Prediction 3D bounding boxes visibility |

## Point Cloud Format

The viewer expects float32 binary files with 4 columns per point:

```
[x0, y0, z0, intensity0, x1, y1, z1, intensity1, ...]
```

Each point is 16 bytes (4 × float32).

## Annotation & Inference Formats

CloudView natively supports three 3D annotation / prediction JSON formats:

### Format A: Converted / Standard PSR Format

```json
[
  {
    "obj_id": "1",
    "obj_type": "truck.dump_truck",
    "psr": {
      "position": {"x": 10.5, "y": -3.2, "z": 0.8},
      "rotation": {"x": 0, "y": 0, "z": 1.23},
      "scale": {"x": 4.5, "y": 1.8, "z": 1.5}
    }
  }
]
```

- `position`: Center coordinates (meters)
- `rotation.z`: Yaw angle (radians, counterclockwise around Z axis)
- `scale`: Dimensions (length × width × height, meters)

### Format B: Raw Dataset Format (Quaternion & Size)

```json
{
  "file_name": "00028765_record...pcd",
  "objects": [
    {
      "ID": 165,
      "category": "road_static_object.traffic_signs",
      "translation": [92.54, -10.13, 0.81],
      "size": [1.72, 1.32, 2.03],
      "rotation": [0.02681, 0, 0, -0.999641],
      "obj_heading": 0.0
    }
  ]
}
```

- `translation`: Center coordinates `[x, y, z]` (meters)
- `size`: Box dimensions `[width, length, height]` (meters, automatically mapped to local coordinates)
- `rotation`: Unit Quaternion `[w, x, y, z]` (automatically converted to yaw angle)

### Format C: Model Inference Results Format

```json
{
  "pred_bboxes_3d": [
    [10.5, -3.2, 0.05, 4.5, 1.8, 1.5, 1.23]
  ],
  "pred_scores": [0.89],
  "pred_names": ["truck.dump_truck"]
}
```

- `pred_bboxes_3d`: Each box is `[x, y, z_bottom, dx, dy, dz, yaw]`. CloudView automatically converts bottom `z` to geometric center `z = z_bottom + dz / 2.0`.
- `pred_scores`: Prediction confidence score (displayed on 2D badge and details panel).
- `pred_names`: Predicted category name.
- Index prefixes (e.g. `0_00187585_...json`) are automatically stripped and matched to the corresponding point cloud file (`00187585_...pcd`).

## Project Structure

```
tool/
├── main.py              Entry point
├── ui_mainwindow.py     Main window with toolbar, filtering, and info panel
├── gl_viewer.py         OpenGL 3D renderer with camera controls, dual GT/Pred rendering, and projected 2D badges
├── pointcloud_loader.py Bin/PCD file loading and validation
├── annotation_loader.py Multi-format JSON annotation/prediction loading and BBox computation
└── colormap.py          Color mapping (viridis, plasma)
```

## License

MIT
