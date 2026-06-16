r"""
Viewer for PMCX 3 x 3 object-source scan results.

Run with:
    python pmcx_obj_scan_viewer.py
    python pmcx_obj_scan_viewer.py path\to\YYYYMMDD_HHMMSS_obj_pmcx_3x3_scan
"""

from __future__ import annotations

import json
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
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from my_display_hist import my_display_hist


SCAN_FILE = "pmcx_obj_scan_result.npz"
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
CUBE_KEYS = {
    "max-normalized IRF cube": "scan_tpsf_cube_9x32x32xt",
    "raw TPSF cube": "scan_tpsf_cube_raw_9x32x32xt",
}


class PMCXObjScanViewer(QMainWindow):
    def __init__(self, initial_folder: str | None = None):
        super().__init__()
        self.setWindowTitle("PMCX 3x3 Source Scan Viewer")
        self.folder = Path(initial_folder) if initial_folder else Path.cwd() / "obj_sim_results"
        self.data = None
        self.cube = None
        self.positions_yz = None
        self.meta = {}
        self.axes = []
        self.images = []
        self.selection_artists = []
        self.overview_cid = None
        self.init_ui()
        if initial_folder:
            self.load_folder(initial_folder)

    def init_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        top = QGridLayout()
        self.folder_edit = QLineEdit(str(self.folder))
        browse_btn = QPushButton("Browse")
        load_btn = QPushButton("Load")
        browse_btn.clicked.connect(self.browse_folder)
        load_btn.clicked.connect(lambda: self.load_folder(self.folder_edit.text().strip()))
        top.addWidget(QLabel("Scan folder"), 0, 0)
        top.addWidget(self.folder_edit, 0, 1)
        top.addWidget(browse_btn, 0, 2)
        top.addWidget(load_btn, 0, 3)

        self.cube_combo = QComboBox()
        self.cube_combo.addItems(list(CUBE_KEYS.keys()))
        self.cube_combo.currentIndexChanged.connect(self.refresh_overview)
        top.addWidget(QLabel("Display cube"), 1, 0)
        top.addWidget(self.cube_combo, 1, 1)

        pixel_row = QHBoxLayout()
        self.row_edit = QLineEdit("16")
        self.col_edit = QLineEdit("16")
        self.row_edit.setMaximumWidth(150)
        self.col_edit.setMaximumWidth(150)
        self.log_y_check = QCheckBox("log y")
        self.normalize_curves_check = QCheckBox("normalize curves")
        self.same_y_check = QCheckBox("same y")
        self.same_y_check.setChecked(True)
        self.row_edit.editingFinished.connect(self.refresh_curves)
        self.col_edit.editingFinished.connect(self.refresh_curves)
        self.log_y_check.stateChanged.connect(self.refresh_curves)
        self.normalize_curves_check.stateChanged.connect(self.refresh_curves)
        self.same_y_check.stateChanged.connect(self.refresh_curves)
        pixel_row.addWidget(QLabel("Rows Y"))
        pixel_row.addWidget(self.row_edit)
        pixel_row.addWidget(QLabel("Cols X"))
        pixel_row.addWidget(self.col_edit)
        pixel_row.addWidget(self.log_y_check)
        pixel_row.addWidget(self.normalize_curves_check)
        pixel_row.addWidget(self.same_y_check)
        pixel_row.addStretch(1)
        layout.addLayout(top)
        layout.addLayout(pixel_row)

        self.figure = Figure(figsize=(9.2, 7.4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)

        self.curve_figure = Figure(figsize=(9.2, 5.8), constrained_layout=True)
        self.curve_canvas = FigureCanvas(self.curve_figure)
        layout.addWidget(self.curve_canvas, stretch=1)

        bottom = QHBoxLayout()
        self.open_point_btn = QPushButton("Open selected point")
        self.open_point_btn.clicked.connect(self.open_selected_point)
        self.selected_label = QLabel("Selected: P01")
        bottom.addWidget(self.open_point_btn)
        bottom.addWidget(self.selected_label)
        bottom.addStretch(1)
        layout.addLayout(bottom)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(90)
        layout.addWidget(self.log_box)

        self.selected_point = 0
        self.selected_y = 15
        self.selected_x = 15
        self.setCentralWidget(central)
        self.resize(1180, 1100)

    def browse_folder(self):
        start = self.folder_edit.text().strip()
        if not Path(start).exists():
            start = str(Path.cwd() / "obj_sim_results")
        folder = QFileDialog.getExistingDirectory(self, "Select PMCX 3x3 scan folder", start)
        if folder:
            self.folder_edit.setText(folder)
            self.load_folder(folder)

    def log(self, text):
        self.log_box.append(text)

    def load_folder(self, folder):
        folder_path = Path(folder)
        result_path = folder_path / SCAN_FILE
        if not result_path.exists():
            QMessageBox.critical(self, "Load failed", f"Cannot find {result_path}")
            return

        try:
            self.data = np.load(result_path, allow_pickle=True)
            self.folder = folder_path
            self.folder_edit.setText(str(folder_path))
            self.positions_yz = np.asarray(self.data["source_positions_yz_mm"], dtype=float)
            self.meta = self.load_meta(folder_path)
            self.refresh_overview()
            self.log(f"Loaded {result_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            self.log(f"[ERROR] {exc}")

    def load_meta(self, folder_path):
        meta_path = folder_path / "scan_settings_and_meta.json"
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def current_cube_key(self):
        label = self.cube_combo.currentText()
        return CUBE_KEYS[label]

    def refresh_overview(self, *_):
        if self.data is None:
            return

        key = self.current_cube_key()
        try:
            self.cube = self.load_cube(key)
        except Exception as exc:
            QMessageBox.critical(self, "Missing cube", str(exc))
            return
        if self.cube.ndim != 4 or self.cube.shape[0] != 9:
            QMessageBox.critical(self, "Invalid cube", f"Expected shape 9 x 32 x 32 x time, got {self.cube.shape}")
            return
        ny, nx = self.cube.shape[1:3]
        self.selected_y = min(max(self.selected_y, 0), ny - 1)
        self.selected_x = min(max(self.selected_x, 0), nx - 1)

        maps = np.sum(self.cube, axis=3)
        vmax = float(np.nanmax(maps)) if maps.size else 0.0
        if vmax <= 0:
            vmax = None

        self.figure.clear()
        self.axes = []
        self.images = []
        self.selection_artists = []
        for idx in range(9):
            ax = self.figure.add_subplot(3, 3, idx + 1)
            image = ax.imshow(
                maps[idx],
                origin="upper",
                cmap="jet",
                vmin=0,
                vmax=vmax,
                extent=[0.5, nx + 0.5, ny + 0.5, 0.5],
            )
            ax.set_xticks([])
            ax.set_yticks([])
            title = f"P{idx + 1:02d} {POINT_NAMES[idx]}"
            if self.positions_yz is not None and self.positions_yz.shape[0] > idx:
                y_mm, z_mm = self.positions_yz[idx]
                title += f"\ny={y_mm:.1f}, z={z_mm:.1f} mm"
            ax.set_title(title, fontsize=9)
            ax.set_picker(True)
            self.axes.append(ax)
            self.images.append(image)

        self.figure.colorbar(self.images[-1], ax=self.axes, shrink=0.80, label="Accumulated intensity")
        self.figure.suptitle(f"3 x 3 source scan: {key}, shape {self.cube.shape}", fontsize=11)
        if self.overview_cid is not None:
            self.canvas.mpl_disconnect(self.overview_cid)
        self.overview_cid = self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.draw_idle()
        self.selected_point = 0
        self.update_selected_label()
        self.refresh_curves()
        self.log(f"Showing {key}, cube shape={self.cube.shape}")

    def load_cube(self, key):
        if self.data is not None and key in self.data.files:
            return np.asarray(self.data[key], dtype=float)
        npy_path = self.folder / f"{key}.npy"
        if npy_path.exists():
            return np.asarray(np.load(npy_path), dtype=float)
        raise FileNotFoundError(f"Cannot find {key} in {SCAN_FILE} or {npy_path}")

    def on_click(self, event):
        if event.inaxes not in self.axes:
            return
        self.selected_point = self.axes.index(event.inaxes)
        if event.xdata is not None and event.ydata is not None and self.cube is not None:
            ny, nx = self.cube.shape[1:3]
            x = int(np.rint(event.xdata))
            y = int(np.rint(event.ydata))
            if 1 <= x <= nx and 1 <= y <= ny:
                self.selected_x = x - 1
                self.selected_y = y - 1
                self.row_edit.blockSignals(True)
                self.col_edit.blockSignals(True)
                self.row_edit.setText(str(y))
                self.col_edit.setText(str(x))
                self.row_edit.blockSignals(False)
                self.col_edit.blockSignals(False)
                self.refresh_curves()
        self.update_selected_label()

    def update_selected_label(self):
        point_text = f"P{self.selected_point + 1:02d} {POINT_NAMES[self.selected_point]}"
        if self.positions_yz is not None and self.positions_yz.shape[0] > self.selected_point:
            y_mm, z_mm = self.positions_yz[self.selected_point]
            point_text += f"  y={y_mm:.2f} mm, z={z_mm:.2f} mm"
        self.selected_label.setText(
            f"Selected: {point_text} | rows Y={self.row_edit.text().strip()}, cols X={self.col_edit.text().strip()}"
        )

    def update_selection_overlay(self, rows0, cols0):
        for artist in self.selection_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.selection_artists = []
        if not self.axes:
            return

        xx, yy = np.meshgrid(cols0 + 1, rows0 + 1)
        for ax in self.axes:
            artist = ax.scatter(
                xx.reshape(-1),
                yy.reshape(-1),
                s=18,
                marker="s",
                facecolors="none",
                edgecolors="white",
                linewidths=0.9,
            )
            self.selection_artists.append(artist)
        self.canvas.draw_idle()

    def parse_index_list(self, text, max_value, label):
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

    def selected_curves(self):
        if self.cube is None:
            return None
        ny, nx = self.cube.shape[1:3]
        rows0 = self.parse_index_list(self.row_edit.text(), ny, "Rows Y")
        cols0 = self.parse_index_list(self.col_edit.text(), nx, "Cols X")
        self.selected_y = int(rows0[0])
        self.selected_x = int(cols0[0])
        curves = np.sum(self.cube[:, rows0, :, :][:, :, cols0, :], axis=(1, 2))
        if self.normalize_curves_check.isChecked():
            denom = np.nanmax(curves, axis=1)
            denom[denom <= 0] = 1.0
            curves = curves / denom[:, None]
        return curves, rows0, cols0

    def refresh_curves(self, *_):
        if self.cube is None:
            return

        try:
            selected = self.selected_curves()
        except Exception as exc:
            self.log(f"[ERROR] Pixel selection failed: {exc}")
            return
        if selected is None:
            return
        curves, rows0, cols0 = selected
        self.update_selected_label()
        self.update_selection_overlay(rows0, cols0)

        self.curve_figure.clear()
        axes = []
        x_axis = np.arange(curves.shape[1])
        y_max = float(np.nanmax(curves)) if curves.size else 0.0
        positive = curves[curves > 0]
        y_min_positive = float(np.nanmin(positive)) if positive.size else 1.0
        for idx in range(9):
            ax = self.curve_figure.add_subplot(3, 3, idx + 1)
            label = f"P{idx + 1:02d}"
            if self.positions_yz is not None and self.positions_yz.shape[0] > idx:
                y_mm, z_mm = self.positions_yz[idx]
                label += f"\ny={y_mm:.1f}, z={z_mm:.1f}"
            color = "tab:red" if idx == self.selected_point else "tab:blue"
            linewidth = 1.9 if idx == self.selected_point else 1.2
            ax.plot(x_axis, curves[idx], linewidth=linewidth, color=color)
            ax.set_title(label, fontsize=8)
            ax.grid(True)
            ax.tick_params(labelsize=7)
            if idx // 3 == 2:
                ax.set_xlabel("Bin", fontsize=8)
            if idx % 3 == 0:
                ax.set_ylabel("Norm." if self.normalize_curves_check.isChecked() else "Counts", fontsize=8)
            if self.log_y_check.isChecked():
                ax.set_yscale("log")
                if positive.size:
                    ax.set_ylim(max(y_min_positive * 0.8, 1e-12), max(y_min_positive * 10, y_max * 1.2))
            elif self.same_y_check.isChecked() and y_max > 0:
                ax.set_ylim(0, y_max * 1.1)
            axes.append(ax)

        self.curve_figure.suptitle(
            f"9 source curves in 3 x 3 scan layout | rows Y={self.row_edit.text().strip()}, "
            f"cols X={self.col_edit.text().strip()} | pixels summed={len(rows0) * len(cols0)}"
        )
        self.curve_canvas.draw_idle()

    def open_selected_point(self):
        if self.cube is None:
            return
        idx = self.selected_point
        title = f"PMCX scan P{idx + 1:02d} {POINT_NAMES[idx]} - {self.current_cube_key()}"
        my_display_hist(self.cube[idx], figure_name=title, show=True)


def pmcx_obj_scan_viewer(initial_folder: str | None = None):
    app = QApplication([])
    win = PMCXObjScanViewer(initial_folder)
    win.show()
    app.exec()


def main():
    initial_folder = sys.argv[1] if len(sys.argv) > 1 else None
    pmcx_obj_scan_viewer(initial_folder)


if __name__ == "__main__":
    main()
