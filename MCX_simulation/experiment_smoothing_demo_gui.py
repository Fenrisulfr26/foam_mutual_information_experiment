"""
Demo GUI for comparing raw and time-smoothed experiment histograms.

Run with:
    D:\codings\anaconda\envs\diffusion\python.exe experiment_smoothing_demo_gui.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("qtagg")
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d


NUM_PIX = 32
DEFAULT_EXPERIMENT_PATH = (
    r"F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\data"
    r"\3x3_grid_scan_20260520_202613_deg_neg3_exp_2us_frames_100000_avg_20"
    r"\hist_2us_100000_avg20_point05_center_cal.mat"
)


def _public_vars(mat_dict):
    return [key for key in mat_dict.keys() if not key.startswith("__")]


def _auto_pick_main_var(mat_dict):
    best_name, best_size = None, -1
    for name in _public_vars(mat_dict):
        arr = np.asarray(mat_dict[name])
        if arr.size > best_size:
            best_name, best_size = name, arr.size
    return best_name


def load_experiment_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mat = loadmat(path)
    preferred = ["hist", "cal", "data", "histogram"]
    var = next((name for name in preferred if name in mat), None)
    if var is None:
        var = _auto_pick_main_var(mat)
    if var is None:
        raise ValueError(f"No public variable found in {path}")
    return np.asarray(mat[var], dtype=float).squeeze(), var


def load_experiment_cube(path, point_index):
    data, var = load_experiment_data(path)
    if data.ndim == 4:
        idx0 = int(point_index) - 1
        if idx0 < 0 or idx0 >= data.shape[3]:
            raise ValueError(f"Point index {point_index} is outside 1..{data.shape[3]}")
        data = np.squeeze(data[:, :, :, idx0])
    if data.ndim != 3 or data.shape[:2] != (NUM_PIX, NUM_PIX):
        raise ValueError(f"Expected experiment shape (32,32,time), got {data.shape}")
    data = np.asarray(data, dtype=float)
    data[~np.isfinite(data)] = 0
    return data, var


def normalize_for_display(arr):
    arr = np.asarray(arr, dtype=float)
    vmax = np.nanmax(arr)
    if not np.isfinite(vmax) or vmax <= 0:
        return arr
    return arr / vmax


class SmoothingDemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Experiment Smoothing Demo")
        self.raw_cube = None
        self.smooth_cube = None
        self.exp_var = None
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        controls = QGroupBox("Experiment and smoothing")
        grid = QGridLayout(controls)
        self.exp_path = QLineEdit(DEFAULT_EXPERIMENT_PATH)
        self.point_index = self._spin_int(1, 100, 5)
        self.sigma_bins = self._spin_float(0.0, 20.0, 0.5, 2)
        self.pixel_x = self._spin_int(1, NUM_PIX, 16)
        self.pixel_y = self._spin_int(1, NUM_PIX, 16)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_experiment)
        load_btn = QPushButton("Load / refresh")
        load_btn.clicked.connect(self._load_and_update)
        update_btn = QPushButton("Update plot")
        update_btn.clicked.connect(self._update_plot)

        grid.addWidget(QLabel("Experiment MAT"), 0, 0)
        grid.addWidget(self.exp_path, 0, 1)
        grid.addWidget(browse_btn, 0, 2)
        grid.addWidget(QLabel("4D point index"), 1, 0)
        grid.addWidget(self.point_index, 1, 1)
        grid.addWidget(QLabel("sigma bins"), 2, 0)
        grid.addWidget(self.sigma_bins, 2, 1)

        pixel_row = QHBoxLayout()
        pixel_row.addWidget(QLabel("X"))
        pixel_row.addWidget(self.pixel_x)
        pixel_row.addWidget(QLabel("Y"))
        pixel_row.addWidget(self.pixel_y)
        grid.addWidget(QLabel("Pixel"), 3, 0)
        grid.addLayout(pixel_row, 3, 1)

        button_row = QHBoxLayout()
        button_row.addWidget(load_btn)
        button_row.addWidget(update_btn)
        grid.addLayout(button_row, 4, 1)
        layout.addWidget(controls)

        self.status = QLabel("Load an experiment MAT file to begin.")
        layout.addWidget(self.status)

        self.figure = Figure(figsize=(11, 7))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        layout.addWidget(self.canvas, stretch=1)

        self.setCentralWidget(central)
        self.resize(1220, 860)
        self._load_and_update()

    def _spin_int(self, low, high, value):
        box = QSpinBox()
        box.setRange(low, high)
        box.setValue(value)
        return box

    def _spin_float(self, low, high, value, decimals):
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setDecimals(decimals)
        box.setSingleStep(10 ** -decimals)
        box.setValue(value)
        return box

    def _browse_experiment(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select experiment MAT", "", "MAT files (*.mat);;All files (*)")
        if path:
            self.exp_path.setText(path)

    def _load_and_update(self):
        try:
            self.raw_cube, self.exp_var = load_experiment_cube(self.exp_path.text().strip(), self.point_index.value())
            self._update_plot()
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def _update_plot(self):
        if self.raw_cube is None:
            return

        sigma = float(self.sigma_bins.value())
        if sigma > 0:
            self.smooth_cube = gaussian_filter1d(self.raw_cube, sigma=sigma, axis=2, mode="nearest")
        else:
            self.smooth_cube = self.raw_cube.copy()

        row = self.pixel_y.value() - 1
        col = self.pixel_x.value() - 1
        raw_curve = self.raw_cube[row, col, :]
        smooth_curve = self.smooth_cube[row, col, :]
        diff_curve = smooth_curve - raw_curve

        raw_map = np.sum(self.raw_cube, axis=2)
        smooth_map = np.sum(self.smooth_cube, axis=2)
        diff_map = smooth_map - raw_map

        self.figure.clear()
        axes = self.figure.subplots(2, 3)

        im0 = axes[0, 0].imshow(normalize_for_display(raw_map), origin="lower", cmap="jet")
        axes[0, 0].set_title("Raw integrated map")
        self.figure.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(normalize_for_display(smooth_map), origin="lower", cmap="jet")
        axes[0, 1].set_title(f"Smoothed map, sigma={sigma:g}")
        self.figure.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        vmax = np.nanmax(np.abs(diff_map))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        im2 = axes[0, 2].imshow(diff_map, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[0, 2].set_title("Smoothed - raw map")
        self.figure.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        for ax in axes[0, :2]:
            ax.scatter([col], [row], s=70, facecolors="none", edgecolors="white", linewidths=1.6)
            ax.set_xlabel("col / X")
            ax.set_ylabel("row / Y")

        t = np.arange(raw_curve.size)
        axes[1, 0].plot(t, raw_curve, label="raw", lw=1.4)
        axes[1, 0].plot(t, smooth_curve, label="smoothed", lw=1.8)
        axes[1, 0].set_title(f"Pixel X={col + 1}, Y={row + 1}")
        axes[1, 0].set_xlabel("time bin")
        axes[1, 0].set_ylabel("counts / intensity")
        axes[1, 0].legend()

        axes[1, 1].plot(t, normalize_for_display(raw_curve), label="raw", lw=1.4)
        axes[1, 1].plot(t, normalize_for_display(smooth_curve), label="smoothed", lw=1.8)
        axes[1, 1].set_title("Normalized curve shape")
        axes[1, 1].set_xlabel("time bin")
        axes[1, 1].legend()

        axes[1, 2].plot(t, diff_curve, color="tab:red", lw=1.4)
        axes[1, 2].axhline(0, color="black", lw=0.8, alpha=0.5)
        axes[1, 2].set_title("Smoothed - raw curve")
        axes[1, 2].set_xlabel("time bin")

        self.figure.tight_layout()
        self.canvas.draw_idle()

        raw_peak = int(np.nanargmax(raw_curve))
        smooth_peak = int(np.nanargmax(smooth_curve))
        self.status.setText(
            f"Loaded variable {self.exp_var!r}, cube={self.raw_cube.shape}, "
            f"sigma={sigma:g}, raw peak={raw_peak}, smoothed peak={smooth_peak}"
        )

    def _on_click(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if event.inaxes not in self.figure.axes[:3]:
            return
        col = int(np.floor(event.xdata + 0.5))
        row = int(np.floor(event.ydata + 0.5))
        if 0 <= row < NUM_PIX and 0 <= col < NUM_PIX:
            self.pixel_x.setValue(col + 1)
            self.pixel_y.setValue(row + 1)
            self._update_plot()


def main():
    app = QApplication(sys.argv)
    window = SmoothingDemoWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
