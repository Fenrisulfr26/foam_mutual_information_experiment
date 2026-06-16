r"""
Compare experimental and PMCX-simulated 3 x 3 scan photon curves.

Run with:
    python pmcx_3x3_curve_compare_gui.py

Or call:
    from pmcx_3x3_curve_compare_gui import pmcx_3x3_curve_compare_gui
    pmcx_3x3_curve_compare_gui(exp_folder, sim_folder, smooth_experiment=False, smooth_window_bins=5)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("qtagg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.io import loadmat


DEFAULT_EXP_FOLDER = (
    r"F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\data"
    r"\3x3_grid_scan_20260603_104733_deg_2_exp_2us_frames_100000_avg_10"
)
DEFAULT_SIM_FOLDER = (
    r"F:\OneDrive\foam_imaging_project\experiment_setup\MCX_simulation\obj_sim_results"
    r"\20260603_105147_obj_pmcx_3x3_scan"
)
DEFAULT_TIME_COMPENSATION_PATH = (
    r"F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\IRF"
    r"\IRF_noLens_10avg_20260612_2210_compensation.mat"
)
POINT_NAMES = [
    "left top",
    "top center",
    "right top",
    "left middle",
    "center",
    "right middle",
    "left bottom",
    "bottom center",
    "right bottom",
]
MAT_PREFERRED_VARS = ("hist", "cal", "data", "histogram")
COMPENSATION_PREFERRED_VARS = ("compensation_matrix", "compensation", "offsets", "time_offsets", "shift_matrix")
DEFAULT_HOT_PIXELS_1BASED = np.asarray(
    [
        [1, 8],
        [1, 9],
        [1, 31],
        [2, 9],
        [2, 13],
        [2, 21],
        [2, 30],
        [3, 1],
        [3, 6],
        [3, 8],
        [3, 25],
        [4, 6],
        [4, 9],
        [4, 17],
        [4, 22],
        [5, 4],
        [5, 7],
        [5, 9],
        [5, 25],
        [5, 26],
        [6, 4],
        [7, 3],
        [7, 9],
        [7, 11],
        [7, 16],
        [7, 18],
        [8, 10],
        [8, 12],
        [8, 23],
        [8, 28],
        [9, 3],
        [9, 5],
        [9, 25],
        [10, 9],
        [10, 18],
        [10, 23],
        [10, 24],
        [10, 28],
        [11, 3],
        [11, 29],
        [12, 4],
        [12, 5],
        [12, 12],
        [12, 25],
        [12, 28],
        [13, 20],
        [13, 21],
        [13, 23],
        [13, 24],
        [14, 3],
        [14, 4],
        [14, 12],
        [14, 14],
        [14, 18],
        [14, 31],
        [15, 2],
        [15, 6],
        [15, 9],
        [15, 18],
        [15, 29],
        [16, 5],
        [16, 23],
        [17, 27],
        [17, 28],
        [17, 30],
        [18, 2],
        [19, 3],
        [19, 19],
        [19, 30],
        [19, 31],
        [20, 5],
        [20, 17],
        [20, 30],
        [20, 31],
        [21, 7],
        [21, 23],
        [21, 24],
        [21, 28],
        [21, 30],
        [22, 15],
        [22, 27],
        [22, 28],
        [23, 4],
        [23, 19],
        [23, 20],
        [23, 32],
        [24, 2],
        [24, 31],
        [25, 4],
        [25, 9],
        [25, 23],
        [25, 24],
        [26, 13],
        [26, 18],
        [26, 23],
        [27, 3],
        [27, 9],
        [27, 11],
        [27, 13],
        [27, 15],
        [27, 16],
        [27, 27],
        [27, 28],
        [27, 32],
        [28, 17],
        [29, 2],
        [29, 3],
        [29, 4],
        [29, 14],
        [29, 19],
        [29, 31],
        [30, 1],
        [30, 11],
        [30, 12],
        [30, 18],
        [30, 20],
        [30, 23],
        [30, 26],
        [30, 29],
        [31, 4],
        [31, 17],
        [31, 18],
        [31, 21],
        [32, 4],
        [32, 21],
        [32, 28],
    ],
    dtype=int,
)


def parse_index_list(text: str, max_value: int, label: str) -> np.ndarray:
    text = text.strip().lower()
    if text in {"", ":", "all"}:
        return np.arange(max_value, dtype=int)

    values = []
    for part in re.split(r"[,;\s]+", text):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            nums = [float(x) for x in part.split(":")]
            if len(nums) not in {2, 3} or not all(np.isfinite(nums)):
                raise ValueError(f"{label} format error: {part}")
            if len(nums) == 2:
                start, stop = nums
                step = 1 if stop >= start else -1
            else:
                start, step, stop = nums
                if step == 0:
                    raise ValueError(f"{label} step cannot be zero: {part}")
            if step > 0:
                seq = np.arange(start, stop + 0.5, step)
            else:
                seq = np.arange(start, stop - 0.5, step)
            values.extend(seq.tolist())
        else:
            value = float(part)
            if not np.isfinite(value):
                raise ValueError(f"{label} format error: {part}")
            values.append(value)

    idx = []
    seen = set()
    for value in values:
        rounded = int(round(value))
        if 1 <= rounded <= max_value and rounded not in seen:
            idx.append(rounded - 1)
            seen.add(rounded)
    if not idx:
        raise ValueError(f"{label} has no valid index in 1..{max_value}")
    return np.asarray(idx, dtype=int)


def normalize_curve_set(curves: np.ndarray) -> np.ndarray:
    curves = np.asarray(curves, dtype=float).copy()
    curves[~np.isfinite(curves)] = 0
    max_value = float(np.nanmax(curves)) if curves.size else 0.0
    return curves / max_value if max_value > 0 else curves


def smooth_curve_set(curves: np.ndarray, window_bins: int) -> np.ndarray:
    curves = np.asarray(curves, dtype=float).copy()
    curves[~np.isfinite(curves)] = 0
    window_bins = int(window_bins)
    if window_bins <= 1 or curves.ndim != 2 or curves.shape[1] <= 1:
        return curves

    window_bins = min(window_bins, curves.shape[1])
    if window_bins % 2 == 0:
        window_bins -= 1
    if window_bins <= 1:
        return curves

    pad = window_bins // 2
    kernel = np.ones(window_bins, dtype=float) / float(window_bins)
    padded = np.pad(curves, ((0, 0), (pad, pad)), mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)


def load_compensation_matrix(path: str | Path) -> tuple[np.ndarray, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Time compensation MAT not found: {path}")

    mat = loadmat(path)
    var_name = next((name for name in COMPENSATION_PREFERRED_VARS if name in mat), None)
    if var_name is None:
        best_name, best_size = None, -1
        for name in public_mat_vars(mat):
            arr = np.asarray(mat[name]).squeeze()
            if arr.ndim == 2 and arr.size > best_size:
                best_name, best_size = name, arr.size
        var_name = best_name
    if var_name is None:
        raise ValueError(f"No public 2D compensation matrix found in {path}")

    offsets = np.asarray(mat[var_name], dtype=float).squeeze()
    if offsets.shape != (32, 32):
        raise ValueError(f"Expected compensation matrix shape (32,32), got {offsets.shape} from {var_name!r}")
    offsets[~np.isfinite(offsets)] = 0
    return offsets, var_name


def apply_time_compensation_cube(cube: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3 or cube.shape[:2] != (32, 32):
        raise ValueError(f"Expected experiment cube shape 32 x 32 x time, got {cube.shape}")

    shifts = np.rint(offsets).astype(int)
    compensated = np.empty_like(cube, dtype=float)
    for row in range(32):
        for col in range(32):
            compensated[row, col, :] = np.roll(cube[row, col, :], shifts[row, col])
    return compensated


def apply_time_compensation_scan_cube(scan_cube: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    scan_cube = np.asarray(scan_cube, dtype=float)
    if scan_cube.ndim != 4 or scan_cube.shape[0] != 9:
        raise ValueError(f"Expected scan cube shape 9 x 32 x 32 x time, got {scan_cube.shape}")
    return np.stack([apply_time_compensation_cube(scan_cube[idx], offsets) for idx in range(9)], axis=0)


def default_hot_pixel_mask(ny: int, nx: int) -> np.ndarray:
    if ny != 32 or nx != 32:
        raise ValueError("Default hot/dark pixel mask is only defined for 32 x 32 histograms.")
    mask = np.zeros((ny, nx), dtype=bool)
    hot0 = DEFAULT_HOT_PIXELS_1BASED - 1
    mask[hot0[:, 0], hot0[:, 1]] = True
    return mask


def find_good_neighbors(row: int, col: int, hot_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = hot_mask.shape
    for radius in range(1, 5):
        row_start = max(0, row - radius)
        row_stop = min(ny, row + radius + 1)
        col_start = max(0, col - radius)
        col_stop = min(nx, col + radius + 1)
        rr, cc = np.meshgrid(
            np.arange(row_start, row_stop),
            np.arange(col_start, col_stop),
            indexing="ij",
        )
        is_self = (rr == row) & (cc == col)
        is_good = (~is_self) & (~hot_mask[row_start:row_stop, col_start:col_stop])
        rows = rr[is_good]
        cols = cc[is_good]
        if rows.size:
            return rows, cols
    return np.asarray([], dtype=int), np.asarray([], dtype=int)


def correct_hot_dark_pixels(hist_in: np.ndarray) -> tuple[np.ndarray, dict]:
    hist = np.asarray(hist_in, dtype=float)
    if hist.ndim != 3:
        raise ValueError(f"hist_in must be 3D, got shape {hist.shape}")
    ny, nx, _ = hist.shape
    hot_mask = default_hot_pixel_mask(ny, nx)
    hist_out = hist.copy()
    hot_rows, hot_cols = np.where(hot_mask)
    neighbor_count = []
    for row, col in zip(hot_rows, hot_cols):
        rows, cols = find_good_neighbors(int(row), int(col), hot_mask)
        neighbor_count.append(int(rows.size))
        if rows.size == 0:
            continue
        hist_out[row, col, :] = np.mean(hist[rows, cols, :], axis=0)
    info = {
        "num_hot_pixels": int(hot_rows.size),
        "neighbor_count": neighbor_count,
        "method": "Hot/dark pixels replaced by time-bin-wise average of surrounding non-hot pixels.",
    }
    return hist_out, info


def point_number_from_name(path: Path) -> int | None:
    match = re.search(r"point0?([1-9])(?=\D|$)", path.name, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def public_mat_vars(mat_dict):
    return [k for k in mat_dict.keys() if not k.startswith("__")]


def pick_mat_cube(mat_dict, path: Path) -> tuple[np.ndarray, str]:
    for name in MAT_PREFERRED_VARS:
        if name in mat_dict:
            arr = np.asarray(mat_dict[name]).squeeze()
            if arr.ndim >= 3:
                return arr, name

    best_name = None
    best_arr = None
    best_size = -1
    for name in public_mat_vars(mat_dict):
        arr = np.asarray(mat_dict[name]).squeeze()
        if arr.ndim >= 3 and arr.size > best_size:
            best_name = name
            best_arr = arr
            best_size = arr.size
    if best_arr is None:
        raise ValueError(f"No 3D histogram variable found in {path}")
    return best_arr, best_name


def load_experiment_scan_cube(folder: str | Path) -> tuple[np.ndarray, list[str]]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Experiment folder not found: {folder}")

    point_files: dict[int, Path] = {}
    for path in folder.glob("*.mat"):
        idx = point_number_from_name(path)
        if idx is not None and "obj" in path.stem.lower():
            point_files[idx] = path

    missing = [idx for idx in range(1, 10) if idx not in point_files]
    if missing:
        raise ValueError(f"Experiment folder is missing OBJ point files: {missing}")

    cubes = []
    labels = []
    for idx in range(1, 10):
        path = point_files[idx]
        mat = loadmat(path)
        cube, var_name = pick_mat_cube(mat, path)
        cube = np.asarray(cube, dtype=float).squeeze()
        if cube.ndim != 3:
            raise ValueError(f"Expected 3D cube in {path}, got shape {cube.shape}")
        if cube.shape[0] != 32 or cube.shape[1] != 32:
            raise ValueError(f"Expected 32 x 32 x time in {path}, got shape {cube.shape}")
        cube[~np.isfinite(cube)] = 0
        cube, _ = correct_hot_dark_pixels(cube)
        cubes.append(cube)
        labels.append(f"P{idx:02d}: {path.name} [{var_name}, dark-corrected]")

    return np.stack(cubes, axis=0), labels


def load_simulation_scan_cube(folder: str | Path) -> tuple[np.ndarray, list[str]]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Simulation folder not found: {folder}")

    npz_path = folder / "pmcx_obj_scan_result.npz"
    meta_path = folder / "scan_settings_and_meta.json"
    irf_norm_npy_path = folder / "scan_tpsf_cube_9x32x32xt.npy"
    raw_npy_path = folder / "scan_tpsf_cube_raw_9x32x32xt.npy"

    storage_mode = None
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "storage_mode" in data.files:
                    storage_mode = str(np.asarray(data["storage_mode"]).item())
        except Exception:
            storage_mode = None
    if storage_mode is None and meta_path.exists():
        try:
            import json

            storage_mode = json.loads(meta_path.read_text(encoding="utf-8")).get("storage_mode")
        except Exception:
            storage_mode = None

    is_new_raw_plus_irf_format = storage_mode == "raw_and_irf_global_max_norm_npy"

    if is_new_raw_plus_irf_format and irf_norm_npy_path.exists():
        cube = np.asarray(np.load(irf_norm_npy_path), dtype=float)
        source_label = "IRF global max-normalized PMCX"
    elif npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        if not is_new_raw_plus_irf_format and "scan_tpsf_cube_irf_raw_9x32x32xt" in data.files:
            cube = np.asarray(data["scan_tpsf_cube_irf_raw_9x32x32xt"], dtype=float)
            source_label = "legacy IRF unnormalized PMCX"
        elif not is_new_raw_plus_irf_format and "scan_tpsf_cube_raw_9x32x32xt" in data.files:
            # Legacy 3x3 scan used this name for IRF-convolved, unnormalized data.
            cube = np.asarray(data["scan_tpsf_cube_raw_9x32x32xt"], dtype=float)
            source_label = "legacy IRF unnormalized PMCX"
        elif "scan_tpsf_cube_9x32x32xt" in data.files:
            cube = np.asarray(data["scan_tpsf_cube_9x32x32xt"], dtype=float)
            source_label = "IRF global max-normalized PMCX"
        elif "scan_tpsf_cube_raw_9x32x32xt" in data.files:
            cube = np.asarray(data["scan_tpsf_cube_raw_9x32x32xt"], dtype=float)
            source_label = "raw PMCX"
        elif raw_npy_path.exists():
            cube = np.asarray(np.load(raw_npy_path), dtype=float)
            source_label = "raw PMCX"
        else:
            raise ValueError(f"Cannot find simulation cube in {npz_path}, {irf_norm_npy_path}, or {raw_npy_path}")
    elif irf_norm_npy_path.exists():
        cube = np.asarray(np.load(irf_norm_npy_path), dtype=float)
        source_label = "IRF max-normalized PMCX"
    elif raw_npy_path.exists():
        cube = np.asarray(np.load(raw_npy_path), dtype=float)
        source_label = "raw PMCX"
    else:
        raise FileNotFoundError(f"Cannot find {npz_path}, {irf_norm_npy_path}, or {raw_npy_path}")

    if cube.ndim != 4 or cube.shape[0] != 9 or cube.shape[1] != 32 or cube.shape[2] != 32:
        raise ValueError(f"Expected simulation cube shape 9 x 32 x 32 x time, got {cube.shape}")
    cube = cube.copy()
    cube[~np.isfinite(cube)] = 0
    labels = [f"P{idx:02d}: {source_label}" for idx in range(1, 10)]
    return cube, labels


def sum_selected_pixels(cube: np.ndarray, rows0: np.ndarray, cols0: np.ndarray) -> np.ndarray:
    if cube.ndim != 4 or cube.shape[0] != 9:
        raise ValueError(f"Expected scan cube shape 9 x rows x cols x time, got {cube.shape}")
    return np.sum(cube[:, rows0, :, :][:, :, cols0, :], axis=(1, 2))


class CurveCompareWindow(QMainWindow):
    def __init__(
        self,
        exp_folder: str | None = None,
        sim_folder: str | None = None,
        smooth_experiment: bool = False,
        smooth_window_bins: int = 5,
    ):
        super().__init__()
        self.setWindowTitle("3x3 Experiment vs PMCX Curve Compare")
        self.exp_cube = None
        self.sim_cube = None
        self.exp_labels = []
        self.sim_labels = []
        self.init_ui(exp_folder, sim_folder, smooth_experiment, smooth_window_bins)

    def init_ui(self, exp_folder, sim_folder, smooth_experiment, smooth_window_bins):
        central = QWidget()
        layout = QVBoxLayout(central)

        grid = QGridLayout()
        self.exp_edit = QLineEdit(exp_folder or DEFAULT_EXP_FOLDER)
        self.sim_edit = QLineEdit(sim_folder or DEFAULT_SIM_FOLDER)
        exp_browse = QPushButton("Browse")
        sim_browse = QPushButton("Browse")
        exp_browse.clicked.connect(lambda: self.browse_folder(self.exp_edit, "Select experiment 3x3 folder"))
        sim_browse.clicked.connect(lambda: self.browse_folder(self.sim_edit, "Select simulation 3x3 folder"))
        grid.addWidget(QLabel("Experiment folder"), 0, 0)
        grid.addWidget(self.exp_edit, 0, 1)
        grid.addWidget(exp_browse, 0, 2)
        grid.addWidget(QLabel("Simulation folder"), 1, 0)
        grid.addWidget(self.sim_edit, 1, 1)
        grid.addWidget(sim_browse, 1, 2)
        layout.addLayout(grid)

        controls = QHBoxLayout()
        self.row_edit = QLineEdit("16")
        self.col_edit = QLineEdit("16")
        self.row_edit.setMaximumWidth(160)
        self.col_edit.setMaximumWidth(160)
        self.same_y_check = QCheckBox("same y")
        self.same_y_check.setChecked(True)
        self.log_y_check = QCheckBox("log y")
        self.exp_smooth_check = QCheckBox("smooth experiment")
        self.exp_smooth_window = QSpinBox()
        self.exp_smooth_window.setRange(1, 999)
        self.exp_smooth_window.setSingleStep(2)
        self.exp_smooth_window.setValue(int(smooth_window_bins))
        self.exp_smooth_window.setMaximumWidth(72)
        self.exp_smooth_check.setChecked(bool(smooth_experiment))
        self.exp_smooth_window.setEnabled(bool(smooth_experiment))
        self.exp_comp_check = QCheckBox("compensate experiment time")
        self.exp_comp_path = QLineEdit(DEFAULT_TIME_COMPENSATION_PATH)
        self.exp_comp_path.setMinimumWidth(360)
        comp_browse = QPushButton("Browse comp")
        comp_browse.clicked.connect(lambda: self.browse_file(self.exp_comp_path, "Select time compensation MAT"))
        load_btn = QPushButton("Load folders")
        plot_btn = QPushButton("Plot comparison")
        exp_maps_btn = QPushButton("Show experiment maps")
        sim_maps_btn = QPushButton("Show simulation maps")
        error_btn = QPushButton("Show error curves")
        load_btn.clicked.connect(self.load_folders)
        plot_btn.clicked.connect(self.plot_comparison)
        exp_maps_btn.clicked.connect(lambda: self.show_intensity_maps("experiment"))
        sim_maps_btn.clicked.connect(lambda: self.show_intensity_maps("simulation"))
        error_btn.clicked.connect(self.show_error_curves)
        self.row_edit.editingFinished.connect(self.plot_comparison)
        self.col_edit.editingFinished.connect(self.plot_comparison)
        self.same_y_check.stateChanged.connect(self.plot_comparison)
        self.log_y_check.stateChanged.connect(self.plot_comparison)
        self.exp_smooth_check.stateChanged.connect(lambda state: self.exp_smooth_window.setEnabled(bool(state)))
        self.exp_smooth_check.stateChanged.connect(self.plot_comparison)
        self.exp_smooth_window.valueChanged.connect(self.plot_comparison)
        self.exp_comp_check.stateChanged.connect(self.plot_comparison)
        self.exp_comp_path.editingFinished.connect(self.plot_comparison)
        controls.addWidget(QLabel("Rows Y"))
        controls.addWidget(self.row_edit)
        controls.addWidget(QLabel("Cols X"))
        controls.addWidget(self.col_edit)
        controls.addWidget(self.same_y_check)
        controls.addWidget(self.log_y_check)
        controls.addWidget(self.exp_smooth_check)
        controls.addWidget(QLabel("bins"))
        controls.addWidget(self.exp_smooth_window)
        controls.addWidget(self.exp_comp_check)
        controls.addWidget(self.exp_comp_path)
        controls.addWidget(comp_browse)
        controls.addWidget(load_btn)
        controls.addWidget(plot_btn)
        controls.addWidget(exp_maps_btn)
        controls.addWidget(sim_maps_btn)
        controls.addWidget(error_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.figure = Figure(figsize=(10, 7.6), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(110)
        layout.addWidget(self.log_box)

        self.setCentralWidget(central)
        self.resize(1220, 920)

    def browse_folder(self, line_edit: QLineEdit, title: str):
        current = Path(line_edit.text().strip())
        start = str(current if current.exists() else Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, title, start)
        if folder:
            line_edit.setText(folder)

    def browse_file(self, line_edit: QLineEdit, title: str):
        current = Path(line_edit.text().strip())
        start = str(current.parent if current.exists() else Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, title, str(start), "MAT files (*.mat);;All files (*)")
        if path:
            line_edit.setText(path)

    def log(self, text: str):
        self.log_box.append(text)

    def load_folders(self):
        try:
            self.exp_cube, self.exp_labels = load_experiment_scan_cube(self.exp_edit.text().strip())
            self.sim_cube, self.sim_labels = load_simulation_scan_cube(self.sim_edit.text().strip())
            self.log(f"Loaded experiment cube: {self.exp_cube.shape} (hot/dark pixels corrected)")
            self.log(f"Loaded simulation raw cube: {self.sim_cube.shape}")
            self.plot_comparison()
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            self.log(f"[ERROR] Load failed: {exc}")

    def experiment_cube_for_display(self):
        if self.exp_cube is None:
            return None
        cube = self.exp_cube
        if self.exp_comp_check.isChecked():
            offsets, var_name = load_compensation_matrix(self.exp_comp_path.text().strip())
            cube = apply_time_compensation_scan_cube(cube, offsets)
        return cube

    def current_curves(self):
        if self.exp_cube is None or self.sim_cube is None:
            self.load_folders()
            if self.exp_cube is None or self.sim_cube is None:
                return None

        exp_cube = self.experiment_cube_for_display()
        if exp_cube is None:
            return None

        ny = min(exp_cube.shape[1], self.sim_cube.shape[1])
        nx = min(exp_cube.shape[2], self.sim_cube.shape[2])
        rows0 = parse_index_list(self.row_edit.text(), ny, "Rows Y")
        cols0 = parse_index_list(self.col_edit.text(), nx, "Cols X")

        exp_curves = sum_selected_pixels(exp_cube, rows0, cols0)
        sim_curves = sum_selected_pixels(self.sim_cube, rows0, cols0)
        common_bins = min(exp_curves.shape[1], sim_curves.shape[1])
        if exp_curves.shape[1] != sim_curves.shape[1]:
            self.log(
                f"[WARN] Time bins differ, cropping to common length {common_bins}: "
                f"experiment={exp_curves.shape[1]}, simulation={sim_curves.shape[1]}"
            )
        exp_curves = exp_curves[:, :common_bins]
        sim_curves = sim_curves[:, :common_bins]
        if self.exp_smooth_check.isChecked():
            exp_curves = smooth_curve_set(exp_curves, self.exp_smooth_window.value())
        return normalize_curve_set(exp_curves), normalize_curve_set(sim_curves), rows0, cols0

    def plot_comparison(self, *_):
        try:
            current = self.current_curves()
            if current is None:
                return
            exp_curves, sim_curves, rows0, cols0 = current
        except Exception as exc:
            self.log(f"[ERROR] Plot failed: {exc}")
            return

        self.figure.clear()
        x_axis = np.arange(exp_curves.shape[1])
        y_max = max(float(np.nanmax(exp_curves)), float(np.nanmax(sim_curves)), 1.0)
        positive = np.concatenate([exp_curves[exp_curves > 0], sim_curves[sim_curves > 0]])
        y_min_positive = float(np.nanmin(positive)) if positive.size else 1e-12

        for idx in range(9):
            ax = self.figure.add_subplot(3, 3, idx + 1)
            ax.plot(x_axis, exp_curves[idx], color="tab:blue", linewidth=1.4, label="Experiment")
            ax.plot(x_axis, sim_curves[idx], color="tab:orange", linewidth=1.4, label="Simulation")
            ax.set_title(f"P{idx + 1:02d} {POINT_NAMES[idx]}", fontsize=9)
            ax.grid(True)
            ax.tick_params(labelsize=8)
            if idx // 3 == 2:
                ax.set_xlabel("Bin", fontsize=8)
            if idx % 3 == 0:
                ax.set_ylabel("Normalized", fontsize=8)
            if self.log_y_check.isChecked():
                ax.set_yscale("log")
                ax.set_ylim(max(y_min_positive * 0.8, 1e-12), y_max * 1.2)
            elif self.same_y_check.isChecked():
                ax.set_ylim(0, y_max * 1.08)
            if idx == 0:
                ax.legend(fontsize=8)

        self.figure.suptitle(
            f"Raw-read -> selected-pixel sum -> per-folder global max normalization | "
            f"Rows Y={self.row_edit.text().strip()}, Cols X={self.col_edit.text().strip()}, "
            f"pixels summed={len(rows0) * len(cols0)}, "
            f"exp smoothing={'on' if self.exp_smooth_check.isChecked() else 'off'}"
            f"{' (' + str(self.exp_smooth_window.value()) + ' bins)' if self.exp_smooth_check.isChecked() else ''}, "
            f"exp compensation={'on' if self.exp_comp_check.isChecked() else 'off'}",
            fontsize=11,
        )
        self.canvas.draw_idle()
        self.log(
            f"Plotted comparison: rows={self.row_edit.text().strip()}, cols={self.col_edit.text().strip()}, "
            f"bins={exp_curves.shape[1]}, "
            f"exp_smoothing={'on' if self.exp_smooth_check.isChecked() else 'off'}, "
            f"exp_compensation={'on' if self.exp_comp_check.isChecked() else 'off'}"
        )

    def show_intensity_maps(self, source: str):
        try:
            if self.exp_cube is None or self.sim_cube is None:
                self.load_folders()
            if source == "experiment":
                cube = self.experiment_cube_for_display()
                title = (
                    "Experiment 3x3 Intensity Maps"
                    + (" (time compensated)" if self.exp_comp_check.isChecked() else "")
                )
                color = "Experiment"
            else:
                cube = self.sim_cube
                title = "Simulation 3x3 Intensity Maps"
                color = "Simulation"
            if cube is None:
                return

            maps = np.sum(cube, axis=3)
            maps = np.asarray(maps, dtype=float)
            maps[~np.isfinite(maps)] = 0
            max_value = float(np.nanmax(maps)) if maps.size else 0.0
            maps_norm = maps / max_value if max_value > 0 else maps
            self.open_intensity_window(maps_norm, title, color, max_value)
        except Exception as exc:
            QMessageBox.critical(self, "Intensity map failed", str(exc))
            self.log(f"[ERROR] Intensity map failed: {exc}")

    def open_intensity_window(self, maps: np.ndarray, title: str, source_label: str, raw_max: float):
        win = QMainWindow(self)
        win.setWindowTitle(title)
        fig = Figure(figsize=(9.2, 8.0), constrained_layout=True)
        canvas = FigureCanvas(fig)
        axes = []
        for idx in range(9):
            ax = fig.add_subplot(3, 3, idx + 1)
            im = ax.imshow(
                maps[idx],
                origin="upper",
                cmap="jet",
                vmin=0,
                vmax=1 if raw_max > 0 else None,
                extent=[0.5, maps.shape[2] + 0.5, maps.shape[1] + 0.5, 0.5],
            )
            ax.set_title(f"P{idx + 1:02d} {POINT_NAMES[idx]}", fontsize=9)
            ax.set_xlabel("X", fontsize=8)
            ax.set_ylabel("Y", fontsize=8)
            ax.tick_params(labelsize=7)
            axes.append(ax)

        fig.colorbar(im, ax=axes, shrink=0.82, label="Normalized intensity")
        fig.suptitle(
            f"{source_label}: sum over time, normalized by folder global max = {raw_max:.6g}",
            fontsize=11,
        )
        win.setCentralWidget(canvas)
        win.resize(980, 880)
        win.show()
        if not hasattr(self, "_child_windows"):
            self._child_windows = []
        self._child_windows.append(win)

    def show_error_curves(self):
        try:
            current = self.current_curves()
            if current is None:
                return
            exp_curves, sim_curves, rows0, cols0 = current
            error_curves = sim_curves - exp_curves
            self.open_error_window(error_curves, exp_curves, sim_curves, rows0, cols0)
        except Exception as exc:
            QMessageBox.critical(self, "Error view failed", str(exc))
            self.log(f"[ERROR] Error view failed: {exc}")

    def open_error_window(self, error_curves, exp_curves, sim_curves, rows0, cols0):
        win = QMainWindow(self)
        win.setWindowTitle("3x3 Simulation - Experiment Error Curves")
        fig = Figure(figsize=(10.2, 8.0), constrained_layout=True)
        canvas = FigureCanvas(fig)

        x_axis = np.arange(error_curves.shape[1])
        max_abs = float(np.nanmax(np.abs(error_curves))) if error_curves.size else 0.0
        if max_abs <= 0:
            max_abs = 1.0

        rmse = np.sqrt(np.mean(error_curves**2, axis=1))
        mae = np.mean(np.abs(error_curves), axis=1)
        max_abs_each = np.max(np.abs(error_curves), axis=1)
        corr = np.zeros(9, dtype=float)
        for idx in range(9):
            if np.std(exp_curves[idx]) > 0 and np.std(sim_curves[idx]) > 0:
                corr[idx] = np.corrcoef(exp_curves[idx], sim_curves[idx])[0, 1]
            else:
                corr[idx] = np.nan

        axes = []
        for idx in range(9):
            ax = fig.add_subplot(3, 3, idx + 1)
            ax.plot(x_axis, error_curves[idx], color="tab:red", linewidth=1.2)
            ax.axhline(0, color="0.25", linewidth=0.8)
            ax.set_ylim(-max_abs * 1.08, max_abs * 1.08)
            ax.set_title(
                f"P{idx + 1:02d} {POINT_NAMES[idx]}\nRMSE={rmse[idx]:.4f}, MAE={mae[idx]:.4f}",
                fontsize=8,
            )
            ax.grid(True)
            ax.tick_params(labelsize=7)
            if idx // 3 == 2:
                ax.set_xlabel("Bin", fontsize=8)
            if idx % 3 == 0:
                ax.set_ylabel("Sim - Exp", fontsize=8)
            axes.append(ax)

        fig.suptitle(
            f"Final normalized curve error | Rows Y={self.row_edit.text().strip()}, "
            f"Cols X={self.col_edit.text().strip()}, pixels summed={len(rows0) * len(cols0)} | "
            f"mean RMSE={np.nanmean(rmse):.4f}, mean MAE={np.nanmean(mae):.4f}, "
            f"mean corr={np.nanmean(corr):.4f}",
            fontsize=11,
        )

        win.setCentralWidget(canvas)
        win.resize(1100, 880)
        win.show()
        if not hasattr(self, "_child_windows"):
            self._child_windows = []
        self._child_windows.append(win)
        self.log(
            "Error metrics: "
            f"mean_RMSE={np.nanmean(rmse):.6g}, mean_MAE={np.nanmean(mae):.6g}, "
            f"max_abs={np.nanmax(max_abs_each):.6g}, mean_corr={np.nanmean(corr):.6g}"
        )


def pmcx_3x3_curve_compare_gui(
    exp_folder: str | None = None,
    sim_folder: str | None = None,
    smooth_experiment: bool = False,
    smooth_window_bins: int = 5,
):
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])
    win = CurveCompareWindow(exp_folder, sim_folder, smooth_experiment, smooth_window_bins)
    win.show()
    if owns_app:
        app.exec()
    return win


def main():
    exp_folder = sys.argv[1] if len(sys.argv) >= 2 else None
    sim_folder = sys.argv[2] if len(sys.argv) >= 3 else None
    pmcx_3x3_curve_compare_gui(exp_folder, sim_folder)


if __name__ == "__main__":
    main()
