"""
GUI for PMCX slab simulation with a binary absorbing object mask.

Run with:
    python pmcx_obj_gui.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg" if "--run-settings" in sys.argv else "qtagg")
import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import map_coordinates, zoom
from scipy.io import loadmat

import pmcx_sim
from my_display_hist import compare_hist


NUM_PIX = 32
BASELINE_BINS = 20


def _mat_public_vars(mat_dict):
    return [k for k in mat_dict.keys() if not k.startswith("__")]


def _auto_pick_main_var(mat_dict):
    best_name, best_size = None, -1
    for name in _mat_public_vars(mat_dict):
        arr = np.asarray(mat_dict[name])
        if arr.size > best_size:
            best_name, best_size = name, arr.size
    return best_name


def load_experiment_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Experiment MAT not found: {path}")
    mat = loadmat(path)
    preferred = ["hist", "cal", "data", "histogram"]
    var = next((k for k in preferred if k in mat), None)
    if var is None:
        var = _auto_pick_main_var(mat)
    arr = np.asarray(mat[var], dtype=float).squeeze()
    return arr, var


def load_irf_curve(path, matlab_index=(16, 16)):
    if not os.path.exists(path):
        raise FileNotFoundError(f"IRF MAT not found: {path}")

    mat = loadmat(path)
    preferred = ["IRF", "irf", "hist", "histogram", "irf_hist"]
    var = next((k for k in preferred if k in mat), None)
    if var is None:
        var = _auto_pick_main_var(mat)

    data = np.asarray(mat[var], dtype=float)
    if data.ndim < 2:
        curve = data.reshape(-1)
    elif data.ndim == 2:
        center = data.shape[0] // 2
        curve = data[center, :].reshape(-1)
    else:
        y = max(0, min(data.shape[0] - 1, matlab_index[0] - 1))
        z = max(0, min(data.shape[1] - 1, matlab_index[1] - 1))
        curve = data[y, z, :].reshape(-1)

    curve = curve - np.median(curve[: min(BASELINE_BINS, curve.size)])
    curve[curve < 0] = 0
    if np.sum(curve) <= 0:
        raise ValueError("IRF curve sum is zero, cannot normalize")
    return curve / np.max(curve), var


@dataclass
class ObjSimSettings:
    experiment_mat_path: str
    experiment_point_index: int
    irf_mat_path: str
    mask_image_path: str
    selected_mask_path: str
    selected_crop_path: str
    selected_quad_xy: list[list[float]]
    output_root: str
    nphoton: int
    voxel_size_mm: float
    slab_thickness_mm: float
    slab_width_mm: float
    slab_height_mm: float
    source_y_mm: float
    source_z_mm: float
    detector_center_y_mm: float
    detector_center_z_mm: float
    fov_mm: float
    detector_diameter_mm: float
    object_x_mm: float
    object_center_y_mm: float
    object_center_z_mm: float
    object_size_y_mm: float
    object_size_z_mm: float
    threshold: int
    mua: float
    mus: float
    g: float
    n: float
    gpuid: int
    seed: int

# X：depth; Y：width; Z：height

def build_detector_array_centered(
    slab_thickness_mm: float,
    center_y_mm: float,
    center_z_mm: float,
    fov_mm: float,
    num_pix: int,
    detector_diameter_mm: float,
    voxel_size_mm: float,
):
    pitch_mm = fov_mm / num_pix
    offsets_mm = (np.arange(num_pix) + 0.5) * pitch_mm - fov_mm / 2 # from low 2 high, centered around zero
    yy_mm = center_y_mm + offsets_mm[::-1] # from galvo's view, right to left
    zz_mm = center_z_mm + offsets_mm[::-1] # from galvo's view, from top to bottom
    det_radius_mm = detector_diameter_mm / 2

    detpos = []
    for z_mm in zz_mm:
        for y_mm in yy_mm:
            detpos.append(
                [
                    slab_thickness_mm / voxel_size_mm,
                    y_mm / voxel_size_mm,
                    z_mm / voxel_size_mm,
                    det_radius_mm / voxel_size_mm,
                ]
            )
    return np.asarray(detpos, dtype=np.float32), yy_mm, zz_mm


def _clip_slice(center_mm: float, size_mm: float, voxel_size_mm: float, max_vox: int):
    start = int(round((center_mm - size_mm / 2) / voxel_size_mm))
    stop = int(round((center_mm + size_mm / 2) / voxel_size_mm))
    start = max(0, min(max_vox, start))
    stop = max(start + 1, min(max_vox, stop))
    return slice(start, stop)


def normalize_cube(cube: np.ndarray):
    cube = np.asarray(cube, dtype=float)
    cube = cube.copy()
    cube[~np.isfinite(cube)] = 0
    max_value = float(np.nanmax(cube)) if cube.size else 0.0
    return cube / max_value if max_value > 0 else cube


def sum_normalize(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = 0
    total = float(np.sum(arr))
    return arr / total if total > 0 else arr


# camera_view_cube function was removed as we now generate all simulation cubes directly in camera view.


def crop_cache_path(output_root: str, image_path: str, threshold: int):
    cache_dir = Path(output_root) / "_crop_cache"
    image_abs = str(Path(image_path).resolve()).lower()
    key_text = f"{image_abs}|threshold={int(threshold)}"
    key = hashlib.sha1(key_text.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"crop_{key}.npz"


def load_crop_cache(cache_path: Path):
    data = np.load(cache_path, allow_pickle=True)
    return (
        np.asarray(data["mask"], dtype=np.uint8),
        np.asarray(data["crop"], dtype=float),
        np.asarray(data["quad_xy"], dtype=float),
    )


def save_crop_cache(cache_path: Path, image_path: str, threshold: int, mask, crop, quad_xy):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        image_path=str(Path(image_path).resolve()),
        threshold=int(threshold),
        mask=np.asarray(mask, dtype=np.uint8),
        crop=np.asarray(crop, dtype=float),
        quad_xy=np.asarray(quad_xy, dtype=float),
    )


def load_experiment_cube(path: str, point_index: int) -> tuple[np.ndarray, str]:
    exp_raw, exp_var = load_experiment_data(path)
    if exp_raw.ndim == 4:
        idx0 = point_index - 1
        if not (0 <= idx0 < exp_raw.shape[3]):
            raise ValueError(f"Experiment point index {point_index} is outside 1..{exp_raw.shape[3]}")
        exp_raw = np.squeeze(exp_raw[:, :, :, idx0])
    if exp_raw.shape[:2] != (NUM_PIX, NUM_PIX) or exp_raw.ndim != 3:
        raise ValueError(f"Expected experiment shape (32,32,time), got {exp_raw.shape}")
    exp_raw = np.asarray(exp_raw, dtype=float)
    exp_raw[~np.isfinite(exp_raw)] = 0
    return exp_raw, exp_var


def match_time_bins(cube: np.ndarray, target_bins: int):
    cube = np.asarray(cube, dtype=float)
    if cube.shape[2] == target_bins:
        return cube
    if cube.shape[2] > target_bins:
        return cube[:, :, :target_bins]
    out = np.zeros((cube.shape[0], cube.shape[1], target_bins), dtype=float)
    out[:, :, : cube.shape[2]] = cube
    return out


def _fold_last_axis_to_period(arr: np.ndarray, period_bins: int):
    arr = np.asarray(arr, dtype=float)
    out = np.zeros(arr.shape[:-1] + (period_bins,), dtype=float)
    for k in range(arr.shape[-1]):
        out[..., k % period_bins] += arr[..., k]
    return out


def _fold_1d_to_period(x: np.ndarray, period_bins: int):
    x = np.asarray(x, dtype=float).reshape(-1)
    out = np.zeros(period_bins, dtype=float)
    for k, val in enumerate(x):
        out[k % period_bins] += val
    return out


def convolve_irf_all_pixels_tcspc(cube, irf, period_bins=None, normalize_irf=True, irf_zero_idx=0):
    cube = np.asarray(cube, dtype=float)
    irf = np.asarray(irf, dtype=float).reshape(-1)
    if cube.ndim != 3:
        raise ValueError("cube must have shape (ny,nx,nt)")
    if irf.size == 0:
        raise ValueError("irf is empty")
    if period_bins is None:
        period_bins = cube.shape[-1]
    period_bins = int(period_bins)
    if period_bins <= 0:
        raise ValueError("period_bins must be positive")

    cube_periodic = _fold_last_axis_to_period(cube, period_bins)
    irf_periodic = _fold_1d_to_period(irf, period_bins)
    if irf_zero_idx != 0:
        irf_periodic = np.roll(irf_periodic, -int(irf_zero_idx))
    if normalize_irf:
        s = np.sum(irf_periodic)
        if not np.isfinite(s) or s <= 0:
            raise ValueError("IRF sum must be positive and finite")
        irf_periodic = irf_periodic / s

    x_fft = np.fft.rfft(cube_periodic, n=period_bins, axis=-1)
    h_fft = np.fft.rfft(irf_periodic, n=period_bins)
    out = np.fft.irfft(x_fft * h_fft.reshape(1, 1, -1), n=period_bins, axis=-1)
    out = np.real(out)
    tiny = 1e-12 * max(1.0, float(np.nanmax(np.abs(out))))
    out[np.abs(out) < tiny] = 0.0
    out[out < 0] = 0.0
    return out


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def load_gray_image(image_path: str):
    image = plt.imread(image_path)
    if image.ndim == 3:
        rgb = image[:, :, :3].astype(float)
        if rgb.max() > 1:
            rgb = rgb / 255.0
        gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    else:
        gray = image.astype(float)
        if gray.max() > 1:
            gray = gray / 255.0
    gray[~np.isfinite(gray)] = 0
    return np.clip(gray, 0, 1)


def order_quad_points(points):
    pts = np.asarray(points, dtype=float)
    if pts.shape != (4, 2):
        raise ValueError("Exactly four corner points are required.")
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(np.sum(ordered, axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if ordered[1, 0] < ordered[-1, 0]:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def homography_from_points(src, dst):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    rows = []
    rhs = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.extend([u, v])
    h = np.linalg.solve(np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float))
    return np.asarray([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]], dtype=float)


def warp_quad_to_square(gray: np.ndarray, quad_xy: np.ndarray, output_pixels: int = 256):
    output_pixels = int(max(8, output_pixels))
    square = np.asarray(
        [
            [0, 0],
            [output_pixels - 1, 0],
            [output_pixels - 1, output_pixels - 1],
            [0, output_pixels - 1],
        ],
        dtype=float,
    )
    h_dst_to_src = homography_from_points(square, quad_xy)
    yy, xx = np.mgrid[0:output_pixels, 0:output_pixels]
    coords = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=0)
    src = h_dst_to_src @ coords
    src_x = src[0] / src[2]
    src_y = src[1] / src[2]
    warped = map_coordinates(gray, [src_y, src_x], order=1, mode="nearest")
    return warped.reshape(output_pixels, output_pixels)


def select_quad_mask_from_image(image_path: str, threshold: int):
    gray = load_gray_image(image_path)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(gray, cmap="gray", origin="upper")
    ax.set_title("Click four corners of the 50 x 50 mm object area")
    points = plt.ginput(4, timeout=0)
    if len(points) != 4:
        plt.close(fig)
        raise ValueError("Please click exactly four corner points.")
    quad_xy = order_quad_points(points)
    closed = np.vstack([quad_xy, quad_xy[0]])
    ax.plot(closed[:, 0], closed[:, 1], "r-", linewidth=1.6)
    ax.plot(quad_xy[:, 0], quad_xy[:, 1], "ro", markersize=5)
    fig.canvas.draw_idle()
    plt.pause(0.2)
    plt.close(fig)

    crop = warp_quad_to_square(gray, quad_xy, output_pixels=256)
    mask = (crop >= (threshold / 255.0)).astype(np.uint8)
    return mask, crop, quad_xy


def resize_mask(mask: np.ndarray, out_shape: tuple[int, int]):
    if mask.shape == out_shape:
        return mask.astype(np.uint8)
    factors = (out_shape[0] / mask.shape[0], out_shape[1] / mask.shape[1])
    resized = zoom(mask.astype(float), factors, order=0)
    return resized[: out_shape[0], : out_shape[1]].astype(np.uint8)


def make_object_cfg(settings: ObjSimSettings, mask: np.ndarray):
    nx = int(round(settings.slab_thickness_mm / settings.voxel_size_mm))
    ny = int(round(settings.slab_width_mm / settings.voxel_size_mm))
    nz = int(round(settings.slab_height_mm / settings.voxel_size_mm))
    vol = np.ones((nx, ny, nz), dtype=np.uint8)

    x_idx = int(round(settings.object_x_mm / settings.voxel_size_mm))
    x_idx = max(0, min(nx - 1, x_idx))
    # slice from low 2 high
    y_slice = _clip_slice(settings.object_center_y_mm, settings.object_size_y_mm, settings.voxel_size_mm, ny)
    z_slice = _clip_slice(settings.object_center_z_mm, settings.object_size_z_mm, settings.voxel_size_mm, nz)

    target_shape = (z_slice.stop - z_slice.start, y_slice.stop - y_slice.start)
    mask_zy = resize_mask(mask, target_shape)
    # Correctly align Y to Y and Z to Z, reversing both axes to match camera view from -X
    vol[x_idx, y_slice, z_slice] = mask_zy[::-1, :].T

    detpos, yy_mm, zz_mm = build_detector_array_centered(
        slab_thickness_mm=settings.slab_thickness_mm,
        center_y_mm=settings.detector_center_y_mm,
        center_z_mm=settings.detector_center_z_mm,
        fov_mm=settings.fov_mm,
        num_pix=NUM_PIX,
        detector_diameter_mm=settings.detector_diameter_mm,
        voxel_size_mm=settings.voxel_size_mm,
    )

    tstart = 0.0
    tend = 12.5e-9
    tstep = 55e-12
    cfg = {
        "nphoton": int(settings.nphoton),
        "vol": vol,
        "unitinmm": settings.voxel_size_mm,
        "issrcfrom0": 1,
        "prop": [
            [0.0, 0.0, 1.0, 1.0],
            [settings.mua, settings.mus, settings.g, settings.n],
        ],
        "srcpos": [
            0.0,
            settings.source_y_mm / settings.voxel_size_mm,
            settings.source_z_mm / settings.voxel_size_mm,
        ],
        "srcdir": [1.0, 0.0, 0.0],
        "srctype": "pencil",
        "detpos": detpos,
        "tstart": tstart,
        "tend": tend,
        "tstep": tstep,
        "seed": settings.seed,
        "gpuid": settings.gpuid,
        "autopilot": 1,
        # Only detected photon histories are needed for TPSF/intensity.
        # Avoid saving the full time-resolved fluence field, which is large
        # for a 50 x 250 x 250 x 227 grid and can crash the native MCX layer.
        "issave2pt": 0,
        "issavedet": 1,
        "savedetflag": "dp",
        "debuglevel": "P"

    }
    meta = {
        "volume_shape_voxels": (nx, ny, nz),
        "object_x_index": x_idx,
        "object_y_slice": (y_slice.start, y_slice.stop),
        "object_z_slice": (z_slice.start, z_slice.stop),
        "object_mask_shape_zy": mask_zy.shape,
        "detector_y_mm": yy_mm,
        "detector_z_mm": zz_mm,
        "detector_pitch_mm": settings.fov_mm / NUM_PIX,
        "num_pix": NUM_PIX,
        "tstart_s": tstart,
        "tend_s": tend,
        "tstep_s": tstep,
    }
    return cfg, meta, mask_zy


def detp_to_detector_outputs(res, cfg, nt: int):
    detp = res.get("detp") if isinstance(res, dict) else None
    if detp is None:
        raise ValueError("No detected photon data found in PMCX result.")

    detid = pmcx_sim.extract_detector_id_from_detp(detp)
    ppath = pmcx_sim.extract_partial_path_from_detp(detp)
    if detid is None or ppath is None:
        raise ValueError("Cannot extract detector id or partial paths from PMCX result.")

    detid0 = pmcx_sim.detector_id_to_zero_based(detid, NUM_PIX * NUM_PIX)
    valid = (detid0 >= 0) & (detid0 < NUM_PIX * NUM_PIX)
    detid0 = detid0[valid]
    ppath = ppath[valid]
    weights = pmcx_sim.detected_photon_weights(detp, cfg)[valid]

    prop = np.asarray(cfg["prop"], dtype=float)
    unit_mm = float(cfg.get("unitinmm", 1.0))
    media_n = prop[1 : 1 + ppath.shape[1], 3]
    tof_ns = np.sum(ppath * unit_mm * media_n[None, :], axis=1) / 299.792458

    tstart_ns = float(cfg["tstart"]) * 1e9
    tstep_ns = float(cfg["tstep"]) * 1e9
    edges = tstart_ns + np.arange(nt + 1) * tstep_ns
    t_idx = np.searchsorted(edges, tof_ns, side="right") - 1
    t_valid = (t_idx >= 0) & (t_idx < nt)

    y_idx = detid0[t_valid] % NUM_PIX
    z_idx = detid0[t_valid] // NUM_PIX

    cube = np.zeros((NUM_PIX, NUM_PIX, nt), dtype=float)
    np.add.at(cube, (z_idx, y_idx, t_idx[t_valid]), weights[t_valid])

    intensity = np.sum(cube, axis=2)
    return intensity, cube


def plot_intensity(intensity, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(intensity, origin="upper", cmap="jet")
    ax.set_title("32 x 32 detector weighted intensity")
    ax.set_xlabel("Detector y index")
    ax.set_ylabel("Detector z index")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mask_preview(crop, mask, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(crop, cmap="gray", origin="upper")
    axes[0].set_title("Perspective-corrected crop")
    axes[1].imshow(mask, cmap="gray", origin="upper", vmin=0, vmax=1)
    axes[1].set_title("Binary mask")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_experiment_sim_comparison(exp_cube, sim_cube, out_path: Path):
    exp_norm = normalize_cube(exp_cube)
    sim_norm = normalize_cube(sim_cube)
    exp_map = np.sum(exp_norm, axis=2)
    sim_map = np.sum(sim_norm, axis=2)
    max_map = max(float(np.nanmax(exp_map)), float(np.nanmax(sim_map)), 1.0)
    diff = exp_map - sim_map

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    image_specs = [
        (axes[0, 0], exp_map, "Experiment accumulated, max-normalized", "jet", 0, max_map),
        (axes[0, 1], sim_map, "Simulation accumulated, max-normalized", "jet", 0, max_map),
        (axes[1, 0], np.abs(diff), "Absolute map difference, max-normalized", "hot", 0, None),
    ]
    for ax, image, title, cmap, vmin, vmax in image_specs:
        im = ax.imshow(image, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")
        fig.colorbar(im, ax=ax)

    ax_curve = axes[1, 1]
    ax_curve.plot(np.sum(exp_norm, axis=(0, 1)), label="Experiment", linewidth=1.5)
    ax_curve.plot(np.sum(sim_norm, axis=(0, 1)), label="Simulation", linewidth=1.5)
    ax_curve.set_title("Total time curve")
    ax_curve.set_xlabel("Bin index")
    ax_curve.set_ylabel("Max-normalized counts")
    ax_curve.grid(True)
    ax_curve.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def scan_source_positions(center_y_mm: float, center_z_mm: float, spacing_mm: float):
    positions = []
    for row, z_offset in enumerate([spacing_mm, 0.0, -spacing_mm], start=1):
        for col, y_offset in enumerate([-spacing_mm, 0.0, spacing_mm], start=1):
            point_index = (row - 1) * 3 + col
            positions.append(
                {
                    "point_index": point_index,
                    "row": row,
                    "col": col,
                    "source_y_mm": float(center_y_mm + y_offset),
                    "source_z_mm": float(center_z_mm + z_offset),
                }
            )
    return positions


def plot_scan_overview(cubes, positions, out_path: Path):
    cubes = np.asarray(cubes, dtype=float)
    maps = np.sum(cubes, axis=3)
    vmax = float(np.nanmax(maps)) if maps.size else 0.0
    if vmax <= 0:
        vmax = None

    fig, axes = plt.subplots(3, 3, figsize=(9, 8))
    for idx, ax in enumerate(axes.flat):
        im = ax.imshow(maps[idx], origin="upper", cmap="jet", vmin=0, vmax=vmax)
        pos = positions[idx]
        ax.set_title(
            f"P{pos['point_index']:02d}  y={pos['source_y_mm']:.1f}, z={pos['source_z_mm']:.1f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="Accumulated intensity")
    fig.suptitle("3 x 3 source scan overview", fontsize=12)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_simulation(settings: ObjSimSettings, log=print, result_dir: str | Path | None = None):
    started = time.perf_counter()
    if settings.selected_mask_path and os.path.exists(settings.selected_mask_path):
        log("Loading selected binary mask...")
        mask = np.load(settings.selected_mask_path).astype(np.uint8)
        crop = np.load(settings.selected_crop_path) if settings.selected_crop_path and os.path.exists(settings.selected_crop_path) else mask
        quad_xy = np.asarray(settings.selected_quad_xy, dtype=float)
        object_present = bool(np.any(mask == 0))
    elif settings.mask_image_path:
        log("Select four corner points from mask image...")
        mask, crop, quad_xy = select_quad_mask_from_image(settings.mask_image_path, settings.threshold)
        object_present = bool(np.any(mask == 0))
    else:
        log("No target image selected; running a homogeneous scatterer simulation without an object.")
        mask = np.ones((2, 2), dtype=np.uint8)
        crop = mask.copy()
        quad_xy = np.empty((0, 2), dtype=float)
        object_present = False
    cfg, meta, mask_zy = make_object_cfg(settings, mask)
    meta["object_present"] = object_present
    meta["object_mode"] = "mask_image" if object_present else "homogeneous_scatterer_no_object"

    log(f"Volume shape: {meta['volume_shape_voxels']}, object slice x={meta['object_x_index']}")
    log(f"Object mask black pixels as vol=0: {int(np.count_nonzero(mask_zy == 0))}")
    log("Running pmcx.mcxlab...")
    res = pmcx_sim.pmcx.mcxlab(cfg)
    log("pmcx.mcxlab returned; extracting detected photons...")

    nt = int(np.ceil((cfg["tend"] - cfg["tstart"]) / cfg["tstep"]))
    intensity_raw, cube_raw = detp_to_detector_outputs(res, cfg, nt=nt) # get 32x32x227 cube from sim result
    detp = res.get("detp") if isinstance(res, dict) else None
    detid = pmcx_sim.extract_detector_id_from_detp(detp)
    ppath = pmcx_sim.extract_partial_path_from_detp(detp)

    exp_cube = None
    exp_var = None
    period_bins = nt
    if settings.experiment_mat_path and os.path.exists(settings.experiment_mat_path):
        exp_cube, exp_var = load_experiment_cube(settings.experiment_mat_path, settings.experiment_point_index)
        period_bins = exp_cube.shape[2]

    log("Loading IRF and convolving simulated TPSF...")
    irf, irf_var = load_irf_curve(settings.irf_mat_path, matlab_index=(16, 16))
    cube_irf = convolve_irf_all_pixels_tcspc(cube_raw, irf, period_bins=period_bins)
    cube_irf_max_norm = normalize_cube(cube_irf)
    cube_irf_sum_norm = sum_normalize(cube_irf)

    # Since cube_irf is already in camera view, we assign directly
    cube_irf_camera = cube_irf
    cube_irf_max_norm_camera = cube_irf_max_norm
    cube_irf_sum_norm_camera = cube_irf_sum_norm

    intensity_irf_raw = np.sum(cube_irf, axis=2)
    intensity_max_norm = np.sum(cube_irf_max_norm, axis=2)
    intensity_sum_norm = np.sum(cube_irf_sum_norm, axis=2)

    intensity_irf_raw_camera = intensity_irf_raw
    intensity_max_norm_camera = intensity_max_norm
    intensity_sum_norm_camera = intensity_sum_norm

    if result_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(settings.output_root) / f"{timestamp}_obj_pmcx"
    else:
        result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=False)

    plot_intensity(intensity_sum_norm_camera, result_dir / "detector_intensity_sum_normalized.png")
    plot_intensity(intensity_max_norm_camera, result_dir / "detector_intensity_normalized.png")
    plot_mask_preview(crop, mask, result_dir / "selected_mask_preview.png")
    np.save(result_dir / "detector_intensity.npy", intensity_max_norm_camera)
    np.save(result_dir / "detector_intensity_sum_normalized.npy", intensity_sum_norm_camera)
    np.save(result_dir / "detector_intensity_max_normalized.npy", intensity_max_norm_camera)
    np.save(result_dir / "detector_intensity_sum_normalized_mcx.npy", intensity_sum_norm)
    np.save(result_dir / "detector_intensity_max_normalized_mcx.npy", intensity_max_norm)
    np.save(result_dir / "detector_intensity_raw.npy", intensity_raw)
    np.save(result_dir / "detector_intensity_irf_raw.npy", intensity_irf_raw)
    np.save(result_dir / "detector_intensity_irf_raw_camera.npy", intensity_irf_raw_camera)
    np.save(result_dir / "tpsf_cube_yzt.npy", cube_irf_max_norm_camera)
    np.save(result_dir / "tpsf_cube_raw_yzt.npy", cube_raw)
    np.save(result_dir / "tpsf_cube_irf_yzt.npy", cube_irf)
    np.save(result_dir / "tpsf_cube_irf_camera_yzt.npy", cube_irf_camera)
    np.save(result_dir / "tpsf_cube_irf_norm_yzt.npy", cube_irf_max_norm_camera)
    np.save(result_dir / "tpsf_cube_irf_sum_norm_yzt.npy", cube_irf_sum_norm_camera)
    np.save(result_dir / "tpsf_cube_irf_norm_mcx_yzt.npy", cube_irf_max_norm)
    np.save(result_dir / "tpsf_cube_irf_sum_norm_mcx_yzt.npy", cube_irf_sum_norm)
    np.save(result_dir / "object_mask_zy.npy", mask_zy)
    np.save(result_dir / "selected_image_crop.npy", crop)
    np.save(result_dir / "vol_uint8.npy", cfg["vol"])
    np.savez(
        result_dir / "pmcx_obj_result.npz",
        detector_intensity=intensity_max_norm_camera,
        detector_intensity_sum_normalized=intensity_sum_norm_camera,
        detector_intensity_max_normalized=intensity_max_norm_camera,
        detector_intensity_sum_normalized_mcx=intensity_sum_norm,
        detector_intensity_max_normalized_mcx=intensity_max_norm,
        detector_intensity_raw=intensity_raw,
        detector_intensity_irf_raw=intensity_irf_raw,
        detector_intensity_irf_raw_camera=intensity_irf_raw_camera,
        tpsf_cube_yzt=cube_irf_max_norm_camera,
        tpsf_cube_raw_yzt=cube_raw,
        tpsf_cube_irf_yzt=cube_irf,
        tpsf_cube_irf_camera_yzt=cube_irf_camera,
        tpsf_cube_irf_norm_yzt=cube_irf_max_norm_camera,
        tpsf_cube_irf_sum_norm_yzt=cube_irf_sum_norm_camera,
        tpsf_cube_irf_norm_mcx_yzt=cube_irf_max_norm,
        tpsf_cube_irf_sum_norm_mcx_yzt=cube_irf_sum_norm,
        object_mask_zy=mask_zy,
        selected_image_crop=crop,
        selected_quad_xy=np.asarray(quad_xy, dtype=float),
        detpos_vox=cfg["detpos"],
        detector_y_mm=meta["detector_y_mm"],
        detector_z_mm=meta["detector_z_mm"],
        detector_pitch_mm=meta["detector_pitch_mm"],
        srcpos_vox=np.asarray(cfg["srcpos"], dtype=float),
        optical_prop=np.asarray(cfg["prop"], dtype=float),
        irf=irf,
        irf_variable=irf_var,
        period_bins=int(period_bins),
        detid=np.asarray(detid) if detid is not None else np.asarray([]),
        ppath=np.asarray(ppath) if ppath is not None else np.asarray([]),
    )

    if exp_cube is not None:
        log("Building experiment/simulation comparison...")
        sim_for_exp = match_time_bins(cube_irf_max_norm_camera, exp_cube.shape[2])
        exp_for_display = normalize_cube(exp_cube)
        sim_for_display = normalize_cube(sim_for_exp)
        plot_experiment_sim_comparison(exp_cube, sim_for_exp, result_dir / "experiment_simulation_comparison.png")
        np.savez(
            result_dir / "experiment_simulation_compare.npz",
            exp_cube=exp_cube,
            sim_cube=sim_for_exp,
            exp_compare=exp_for_display,
            sim_compare=sim_for_display,
            experiment_variable=exp_var,
            irf_variable=irf_var,
            normalization="max",
            simulation_orientation="camera_view_fliplr_from_mcx",
        )

    with open(result_dir / "settings_and_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            json_ready({"settings": asdict(settings), "meta": meta, "selected_quad_xy": quad_xy}),
            f,
            ensure_ascii=False,
            indent=2,
        )

    elapsed = time.perf_counter() - started
    log(f"Saved result to {result_dir}")
    log(f"Finished in {elapsed:.1f}s")
    return result_dir


def run_from_settings_file(settings_path: str):
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = ObjSimSettings(**json.load(f))
    result_dir = run_simulation(settings, log=lambda text: print(text, flush=True))
    print(f"RESULT_DIR={result_dir}", flush=True)


def run_scan_from_settings_file(settings_path: str):
    with open(settings_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    settings = ObjSimSettings(**payload["settings"])
    scan = payload["scan"]
    center_y = float(scan["center_y_mm"])
    center_z = float(scan["center_z_mm"])
    spacing = float(scan["spacing_mm"])

    scan_dir = run_source_scan(
        settings,
        center_y_mm=center_y,
        center_z_mm=center_z,
        spacing_mm=spacing,
        log=lambda text: print(text, flush=True),
    )
    print(f"SCAN_DIR={scan_dir}", flush=True)


def run_source_scan(settings: ObjSimSettings, center_y_mm: float, center_z_mm: float, spacing_mm: float, log=print):
    started = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scan_dir = Path(settings.output_root) / f"{timestamp}_obj_pmcx_3x3_scan"
    point_root = scan_dir / "point_runs"
    scan_dir.mkdir(parents=True, exist_ok=False)
    point_root.mkdir(parents=True, exist_ok=False)

    positions = scan_source_positions(center_y_mm, center_z_mm, spacing_mm)
    cubes = []
    raw_cubes = []
    sum_norm_cubes = []
    point_dirs = []

    log(
        "Starting 3 x 3 source scan: "
        f"center y={center_y_mm:.2f} mm, center z={center_z_mm:.2f} mm, spacing={spacing_mm:.2f} mm"
    )
    for pos in positions:
        point_settings = ObjSimSettings(**asdict(settings))
        point_settings.source_y_mm = pos["source_y_mm"]
        point_settings.source_z_mm = pos["source_z_mm"]
        point_settings.seed = int(settings.seed) + pos["point_index"] - 1
        point_dir = point_root / (
            f"point{pos['point_index']:02d}_row{pos['row']}_col{pos['col']}"
            f"_y{pos['source_y_mm']:.2f}_z{pos['source_z_mm']:.2f}".replace(".", "p")
        )
        log(
            f"[{pos['point_index']}/9] Running source y={pos['source_y_mm']:.2f} mm, "
            f"z={pos['source_z_mm']:.2f} mm"
        )
        result_dir = run_simulation(point_settings, log=log, result_dir=point_dir)
        point_dirs.append(str(result_dir))
        data = np.load(Path(result_dir) / "pmcx_obj_result.npz", allow_pickle=True)
        cubes.append(np.asarray(data["tpsf_cube_irf_norm_yzt"], dtype=float))
        raw_cubes.append(np.asarray(data["tpsf_cube_irf_yzt"], dtype=float))
        sum_norm_cubes.append(np.asarray(data["tpsf_cube_irf_sum_norm_yzt"], dtype=float))

    scan_cube = np.stack(cubes, axis=0)
    scan_raw_cube = np.stack(raw_cubes, axis=0)
    scan_sum_norm_cube = np.stack(sum_norm_cubes, axis=0)
    intensity = np.sum(scan_cube, axis=3)

    np.save(scan_dir / "scan_tpsf_cube_9x32x32xt.npy", scan_cube)
    np.save(scan_dir / "scan_tpsf_cube_raw_9x32x32xt.npy", scan_raw_cube)
    np.save(scan_dir / "scan_tpsf_cube_irf_raw_9x32x32xt.npy", scan_raw_cube)
    np.save(scan_dir / "scan_tpsf_cube_sum_norm_9x32x32xt.npy", scan_sum_norm_cube)
    np.save(scan_dir / "scan_detector_intensity_9x32x32.npy", intensity)
    np.savez(
        scan_dir / "pmcx_obj_scan_result.npz",
        scan_tpsf_cube_9x32x32xt=scan_cube,
        scan_tpsf_cube_raw_9x32x32xt=scan_raw_cube,
        scan_tpsf_cube_irf_raw_9x32x32xt=scan_raw_cube,
        scan_tpsf_cube_sum_norm_9x32x32xt=scan_sum_norm_cube,
        scan_detector_intensity_9x32x32=intensity,
        source_positions_yz_mm=np.asarray(
            [[p["source_y_mm"], p["source_z_mm"]] for p in positions],
            dtype=float,
        ),
        point_row_col=np.asarray([[p["row"], p["col"]] for p in positions], dtype=int),
        point_dirs=np.asarray(point_dirs, dtype=object),
        scan_center_yz_mm=np.asarray([center_y_mm, center_z_mm], dtype=float),
        scan_spacing_mm=float(spacing_mm),
    )
    plot_scan_overview(scan_cube, positions, scan_dir / "scan_3x3_overview.png")

    with open(scan_dir / "scan_settings_and_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            json_ready(
                {
                    "settings": asdict(settings),
                    "scan": {
                        "center_y_mm": center_y_mm,
                        "center_z_mm": center_z_mm,
                        "spacing_mm": spacing_mm,
                        "order": "row-major from top-left; z high to low, y low to high",
                        "positions": positions,
                    },
                    "point_dirs": point_dirs,
                    "cube_shape": scan_cube.shape,
                }
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    elapsed = time.perf_counter() - started
    log(f"Saved 3 x 3 scan to {scan_dir}")
    log(f"Scan cube shape: {scan_cube.shape}")
    log(f"Finished scan in {elapsed:.1f}s")
    return scan_dir


class PMCXObjectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PMCX Object Mask Simulation GUI")
        self.process = None
        self.current_result_dir = None
        self.current_result_kind = "single"
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        paths = QGroupBox("Data and output")
        path_layout = QGridLayout(paths)
        self.exp_path = QLineEdit(r"F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\data")
        self.exp_point_index = self.spin_int(1, 100, 1)
        self.irf_path = QLineEdit(r"F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\data\IRF_20260601_165629_deg_2_exp_2us_frames_100000_avg_1\hist_2us_100000_avg1_point05_center_obj.mat")
        self.mask_path = QLineEdit("")
        self.mask_path.setPlaceholderText("Optional: leave empty for homogeneous scatterer without an object")
        self.reuse_crop = QCheckBox("Reuse cached crop for this image")
        self.reuse_crop.setChecked(True)
        self.output_root = QLineEdit(r"F:\OneDrive\foam_imaging_project\experiment_setup\MCX_simulation\obj_sim_results")
        self.add_path_row(path_layout, 0, "Experiment MAT", self.exp_path, "MAT files (*.mat);;All files (*)")
        path_layout.addWidget(QLabel("4D experiment point index"), 1, 0)
        path_layout.addWidget(self.exp_point_index, 1, 1)
        self.add_path_row(path_layout, 2, "IRF MAT", self.irf_path, "MAT files (*.mat);;All files (*)")
        self.add_path_row(
            path_layout,
            3,
            "Mask image",
            self.mask_path,
            "Image files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
        )
        path_layout.addWidget(QLabel("Crop cache"), 4, 0)
        path_layout.addWidget(self.reuse_crop, 4, 1)
        self.add_dir_row(path_layout, 5, "Output root", self.output_root)
        layout.addWidget(paths)

        params_row = QHBoxLayout()
        params_row.addWidget(self.sim_group())
        params_row.addWidget(self.geometry_group())
        params_row.addWidget(self.optical_group())
        layout.addLayout(params_row)

        buttons = QHBoxLayout()
        self.run_btn = QPushButton("Run object simulation")
        self.scan_btn = QPushButton("Run 3x3 source scan")
        self.stop_btn = QPushButton("Stop simulation")
        self.compare_btn = QPushButton("Compare result folder")
        self.scan_viewer_btn = QPushButton("Open scan viewer")
        self.run_btn.clicked.connect(self.start_run)
        self.scan_btn.clicked.connect(self.start_scan)
        self.stop_btn.clicked.connect(self.stop_run)
        self.compare_btn.clicked.connect(self.on_compare_clicked)
        self.scan_viewer_btn.clicked.connect(self.open_scan_viewer)
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.scan_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addWidget(self.compare_btn)
        buttons.addWidget(self.scan_viewer_btn)
        layout.addLayout(buttons)
        self.stop_btn.setEnabled(False)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box, stretch=1)

        self.setCentralWidget(central)
        self.resize(1180, 760)

    def sim_group(self):
        group = QGroupBox("PMCX simulation")
        form = QFormLayout(group)
        self.nphoton = self.spin_int(1, 1_000_000_000, 100_000_000)
        self.voxel = self.spin_float(0.1, 10, 1.0, 2)
        self.thickness = self.spin_float(1, 500, 50.0, 2)
        self.width = self.spin_float(1, 1000, 250.0, 2)
        self.height = self.spin_float(1, 1000, 250.0, 2)
        self.fov = self.spin_float(1, 1000, 140.0, 2)
        self.det_diam = self.spin_float(0.01, 100, 2.0, 3)
        self.gpuid = self.spin_int(0, 16, 1)
        self.seed = self.spin_int(1, 2_000_000_000, 123456789)
        for label, widget in [
            ("nphoton", self.nphoton),
            ("voxel size mm", self.voxel),
            ("thickness mm", self.thickness),
            ("width mm", self.width),
            ("height mm", self.height),
            ("camera / detector FOV mm", self.fov),
            ("detector diameter mm", self.det_diam),
            ("GPU id", self.gpuid),
            ("seed", self.seed),
        ]:
            form.addRow(label, widget)
        return group

    def geometry_group(self):
        group = QGroupBox("Source / detector / object geometry")
        form = QFormLayout(group)
        self.source_y = self.spin_float(0, 1000, 125.0, 2)
        self.source_z = self.spin_float(0, 1000, 105.0, 2)
        self.scan_center_y = self.spin_float(0, 1000, 125.0, 2)
        self.scan_center_z = self.spin_float(0, 1000, 105.0, 2)
        self.scan_spacing = self.spin_float(0, 1000, 15.0, 2)
        self.det_center_y = self.spin_float(0, 1000, 125.0, 2)
        self.det_center_z = self.spin_float(0, 1000, 105.0, 2)
        self.obj_x = self.spin_float(0, 500, 25.0, 2)
        self.obj_center_y = self.spin_float(0, 1000, 125.0, 2)
        self.obj_center_z = self.spin_float(0, 1000, 105.0, 2)
        self.obj_size_y = self.spin_float(0.1, 1000, 50.0, 2)
        self.obj_size_z = self.spin_float(0.1, 1000, 50.0, 2)
        self.threshold = self.spin_int(0, 255, 128)
        for label, widget in [
            ("source y mm", self.source_y),
            ("source height z mm", self.source_z),
            ("3x3 scan center y mm", self.scan_center_y),
            ("3x3 scan center height z mm", self.scan_center_z),
            ("3x3 scan spacing mm", self.scan_spacing),
            ("detector center y mm", self.det_center_y),
            ("detector center height z mm", self.det_center_z),
            ("object x depth mm", self.obj_x),
            ("object center y mm", self.obj_center_y),
            ("object center height z mm", self.obj_center_z),
            ("object size y mm", self.obj_size_y),
            ("object size z mm", self.obj_size_z),
            ("binary threshold 0-255", self.threshold),
        ]:
            form.addRow(label, widget)
        return group

    def optical_group(self):
        group = QGroupBox("Scatterer optical parameters")
        form = QFormLayout(group)
        self.mua = self.spin_float(1e-6, 1, 0.0009153, 6)
        self.mus = self.spin_float(1e-4, 100, 1.7874, 5)
        self.g = self.spin_float(-0.99, 0.99, 0, 3)
        self.n = self.spin_float(1.0, 3.0, 1.05, 4)
        for label, widget in [
            ("mua mm^-1", self.mua),
            ("mus mm^-1", self.mus),
            ("g", self.g),
            ("n", self.n),
        ]:
            form.addRow(label, widget)
        return group

    def add_path_row(self, layout, row, label, line_edit, file_filter):
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(line_edit, row, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.browse_file(line_edit, file_filter))
        layout.addWidget(btn, row, 2)

    def add_dir_row(self, layout, row, label, line_edit):
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(line_edit, row, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.browse_dir(line_edit))
        layout.addWidget(btn, row, 2)

    def spin_int(self, low, high, value):
        box = QSpinBox()
        box.setRange(low, high)
        box.setValue(value)
        return box

    def spin_float(self, low, high, value, decimals):
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setDecimals(decimals)
        box.setSingleStep(10 ** -decimals)
        box.setValue(value)
        return box

    def browse_file(self, line_edit, file_filter):
        current_path = line_edit.text().strip()
        initial_dir = ""
        if current_path:
            try:
                p = Path(current_path)
                if p.is_file():
                    initial_dir = str(p.parent)
                elif p.is_dir():
                    initial_dir = str(p)
                else:
                    for parent in p.parents:
                        if parent.exists():
                            initial_dir = str(parent)
                            break
            except Exception:
                pass
        if not initial_dir:
            initial_dir = r"F:\OneDrive\foam_imaging_project"

        path, _ = QFileDialog.getOpenFileName(self, "Select file", initial_dir, file_filter)
        if path:
            line_edit.setText(path)

    def browse_dir(self, line_edit):
        current_path = line_edit.text().strip()
        initial_dir = ""
        if current_path:
            try:
                p = Path(current_path)
                if p.is_dir() and p.exists():
                    initial_dir = str(p)
                else:
                    for parent in p.parents:
                        if parent.exists():
                            initial_dir = str(parent)
                            break
            except Exception:
                pass
        if not initial_dir:
            initial_dir = r"F:\OneDrive\foam_imaging_project"

        path = QFileDialog.getExistingDirectory(self, "Select folder", initial_dir)
        if path:
            line_edit.setText(path)

    def collect_settings(self):
        return ObjSimSettings(
            experiment_mat_path=self.exp_path.text().strip(),
            experiment_point_index=self.exp_point_index.value(),
            irf_mat_path=self.irf_path.text().strip(),
            mask_image_path=self.mask_path.text().strip(),
            selected_mask_path="",
            selected_crop_path="",
            selected_quad_xy=[],
            output_root=self.output_root.text().strip(),
            nphoton=self.nphoton.value(),
            voxel_size_mm=self.voxel.value(),
            slab_thickness_mm=self.thickness.value(),
            slab_width_mm=self.width.value(),
            slab_height_mm=self.height.value(),
            source_y_mm=self.source_y.value(),
            source_z_mm=self.source_z.value(),
            detector_center_y_mm=self.det_center_y.value(),
            detector_center_z_mm=self.det_center_z.value(),
            fov_mm=self.fov.value(),
            detector_diameter_mm=self.det_diam.value(),
            object_x_mm=self.obj_x.value(),
            object_center_y_mm=self.obj_center_y.value(),
            object_center_z_mm=self.obj_center_z.value(),
            object_size_y_mm=self.obj_size_y.value(),
            object_size_z_mm=self.obj_size_z.value(),
            threshold=self.threshold.value(),
            mua=self.mua.value(),
            mus=self.mus.value(),
            g=self.g.value(),
            n=self.n.value(),
            gpuid=self.gpuid.value(),
            seed=self.seed.value(),
        )

    def append_log(self, text):
        self.log_box.append(text)

    def start_run(self):
        try:
            settings = self.collect_settings()
            if settings.experiment_mat_path and not os.path.exists(settings.experiment_mat_path):
                raise ValueError("Please select a valid experiment MAT file, or leave it empty.")
            if not settings.irf_mat_path or not os.path.exists(settings.irf_mat_path):
                raise ValueError("Please select a valid IRF MAT file.")
            if settings.mask_image_path and not os.path.isfile(settings.mask_image_path):
                raise ValueError("Target image must be a valid image file, or leave it empty for no object.")
            if settings.fov_mm <= 0:
                raise ValueError("FOV must be positive.")
            run_dir = Path(settings.output_root) / "_gui_runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            if not settings.mask_image_path:
                self.append_log("No target image selected: using a homogeneous scatterer without an object.")
                settings.selected_mask_path = ""
                settings.selected_crop_path = ""
                settings.selected_quad_xy = []
            else:
                cache_path = crop_cache_path(settings.output_root, settings.mask_image_path, settings.threshold)
                if self.reuse_crop.isChecked() and cache_path.exists():
                    self.append_log(f"Reusing cached crop: {cache_path}")
                    mask, crop, quad_xy = load_crop_cache(cache_path)
                else:
                    self.append_log("Click four corners corresponding to the 50 x 50 mm object area.")
                    mask, crop, quad_xy = select_quad_mask_from_image(settings.mask_image_path, settings.threshold)
                    save_crop_cache(cache_path, settings.mask_image_path, settings.threshold, mask, crop, quad_xy)
                    self.append_log(f"Saved crop cache: {cache_path}")
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mask_path = run_dir / f"mask_{stamp}.npy"
                crop_path = run_dir / f"crop_{stamp}.npy"
                np.save(mask_path, mask)
                np.save(crop_path, crop)
                settings.selected_mask_path = str(mask_path)
                settings.selected_crop_path = str(crop_path)
                settings.selected_quad_xy = np.asarray(quad_xy, dtype=float).tolist()
            settings_path = run_dir / f"settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(asdict(settings), f, ensure_ascii=False, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return

        self.run_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.current_result_dir = None
        self.current_result_kind = "single"
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments([str(Path(__file__).resolve()), "--run-settings", str(settings_path)])
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parent))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_process_output)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.append_log(f"Starting simulation subprocess with settings: {settings_path}")
        self.process.start()

    def start_scan(self):
        try:
            settings = self.collect_settings()
            if not settings.irf_mat_path or not os.path.exists(settings.irf_mat_path):
                raise ValueError("Please select a valid IRF MAT file.")
            if settings.mask_image_path and not os.path.isfile(settings.mask_image_path):
                raise ValueError("Target image must be a valid image file, or leave it empty for no object.")
            if settings.fov_mm <= 0:
                raise ValueError("FOV must be positive.")
            if self.scan_spacing.value() <= 0:
                raise ValueError("3x3 scan spacing must be positive.")

            run_dir = Path(settings.output_root) / "_gui_runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not settings.mask_image_path:
                self.append_log("No target image selected: 3x3 scan will use a homogeneous scatterer without an object.")
                settings.selected_mask_path = ""
                settings.selected_crop_path = ""
                settings.selected_quad_xy = []
            else:
                cache_path = crop_cache_path(settings.output_root, settings.mask_image_path, settings.threshold)
                if self.reuse_crop.isChecked() and cache_path.exists():
                    self.append_log(f"Reusing cached crop: {cache_path}")
                    mask, crop, quad_xy = load_crop_cache(cache_path)
                else:
                    self.append_log("Click four corners corresponding to the 50 x 50 mm object area.")
                    mask, crop, quad_xy = select_quad_mask_from_image(settings.mask_image_path, settings.threshold)
                    save_crop_cache(cache_path, settings.mask_image_path, settings.threshold, mask, crop, quad_xy)
                    self.append_log(f"Saved crop cache: {cache_path}")
                mask_path = run_dir / f"scan_mask_{stamp}.npy"
                crop_path = run_dir / f"scan_crop_{stamp}.npy"
                np.save(mask_path, mask)
                np.save(crop_path, crop)
                settings.selected_mask_path = str(mask_path)
                settings.selected_crop_path = str(crop_path)
                settings.selected_quad_xy = np.asarray(quad_xy, dtype=float).tolist()
            settings.experiment_mat_path = ""
            payload = {
                "settings": asdict(settings),
                "scan": {
                    "center_y_mm": self.scan_center_y.value(),
                    "center_z_mm": self.scan_center_z.value(),
                    "spacing_mm": self.scan_spacing.value(),
                },
            }
            settings_path = run_dir / f"scan_settings_{stamp}.json"
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid scan settings", str(exc))
            return

        self.run_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.current_result_dir = None
        self.current_result_kind = "scan"
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments([str(Path(__file__).resolve()), "--run-scan-settings", str(settings_path)])
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parent))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_process_output)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.append_log(f"Starting 3x3 scan subprocess with settings: {settings_path}")
        self.process.start()

    def stop_run(self):
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.append_log("Stop requested. Terminating the simulation subprocess...")
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()
                self.append_log("Subprocess did not terminate quickly; killed it.")
        self.stop_btn.setEnabled(False)

    def read_process_output(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        for line in text.splitlines():
            if line.startswith("RESULT_DIR="):
                self.current_result_dir = line.split("=", 1)[1].strip()
                self.current_result_kind = "single"
            elif line.startswith("SCAN_DIR="):
                self.current_result_dir = line.split("=", 1)[1].strip()
                self.current_result_kind = "scan"
            self.append_log(line)

    def process_finished(self, exit_code, exit_status):
        self.run_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if exit_code == 0 and self.current_result_dir:
            self.append_log(f"Finished. Result folder: {self.current_result_dir}")
            if self.current_result_kind == "single":
                self.try_auto_compare(self.current_result_dir)
                QMessageBox.information(self, "Simulation finished", f"Saved to:\n{self.current_result_dir}")
            else:
                QMessageBox.information(self, "3x3 scan finished", f"Saved to:\n{self.current_result_dir}")
        else:
            message = f"Simulation subprocess exited with code {exit_code}, status {exit_status.name}"
            self.append_log(f"[ERROR] {message}")
            QMessageBox.critical(self, "Simulation failed", message)
        self.process = None

    def process_error(self, error):
        self.run_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        message = f"Simulation subprocess error: {error.name}"
        self.append_log(f"[ERROR] {message}")
        QMessageBox.critical(self, "Simulation failed", message)

    def try_auto_compare(self, folder):
        if not self.exp_path.text().strip():
            return
        try:
            self.compare_result_folder(folder)
        except Exception as exc:
            self.append_log(f"[WARN] Could not open comparison view: {exc}")

    def on_compare_clicked(self):
        try:
            self.compare_result_folder()
        except Exception as exc:
            QMessageBox.critical(self, "Compare failed", str(exc))
            self.append_log(f"[ERROR] Compare failed: {exc}")

    def open_scan_viewer(self):
        viewer_path = Path(__file__).resolve().with_name("pmcx_obj_scan_viewer.py")
        if not viewer_path.exists():
            QMessageBox.critical(self, "Viewer missing", f"Cannot find {viewer_path}")
            return

        args = [str(viewer_path)]
        if self.current_result_kind == "scan" and self.current_result_dir:
            args.append(self.current_result_dir)
        ok = QProcess.startDetached(sys.executable, args, str(Path(__file__).resolve().parent))
        if not ok:
            QMessageBox.critical(self, "Viewer failed", "Could not start scan viewer.")

    def compare_result_folder(self, folder=None):
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Select object simulation result folder")
            if not folder:
                return

        folder_path = Path(folder)
        result_path = folder_path / "pmcx_obj_result.npz"
        if not result_path.exists():
            raise ValueError(f"Cannot find {result_path}")

        exp_path = self.exp_path.text().strip()
        exp_point_index = self.exp_point_index.value()
        report_path = folder_path / "settings_and_meta.json"
        if not exp_path and report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            saved_settings = report.get("settings", {})
            exp_path = saved_settings.get("experiment_mat_path", "")
            exp_point_index = int(saved_settings.get("experiment_point_index", exp_point_index))

        if not exp_path or not os.path.exists(exp_path):
            raise ValueError("Please select a valid experiment MAT file before comparing.")

        data = np.load(result_path, allow_pickle=True)
        if "tpsf_cube_irf_norm_yzt" in data.files:
            sim_key = "tpsf_cube_irf_norm_yzt"
            sim_is_camera_view = "tpsf_cube_irf_norm_mcx_yzt" in data.files
        elif "tpsf_cube_irf_sum_norm_yzt" in data.files:
            sim_key = "tpsf_cube_irf_sum_norm_yzt"
            sim_is_camera_view = "tpsf_cube_irf_sum_norm_mcx_yzt" in data.files
        else:
            sim_key = "tpsf_cube_yzt"
            sim_is_camera_view = False
        sim_cube = np.asarray(data[sim_key], dtype=float)
        if not sim_is_camera_view:
            # Inline legacy camera view conversion (Transpose YxZxT -> ZxYxT, then flip Z and Y)
            sim_cube = sim_cube.transpose(1, 0, 2)[::-1, ::-1, :]
        exp_cube, exp_var = load_experiment_cube(exp_path, exp_point_index)
        sim_cube = match_time_bins(sim_cube, exp_cube.shape[2])
        exp_display = normalize_cube(exp_cube)
        sim_display = normalize_cube(sim_cube)

        plot_experiment_sim_comparison(exp_cube, sim_cube, folder_path / "experiment_simulation_comparison.png")
        compare_hist(
            exp_display,
            sim_display,
            label_a="Experiment",
            label_b="Simulation",
            figure_name="Object simulation comparison",
        )
        self.append_log(
            f"Opened comparison: experiment variable={exp_var}, sim shape={sim_cube.shape}, exp shape={exp_cube.shape}"
        )


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--run-settings":
        try:
            run_from_settings_file(sys.argv[2])
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)
            sys.exit(1)
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--run-scan-settings":
        try:
            run_scan_from_settings_file(sys.argv[2])
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)
            sys.exit(1)
        return

    app = QApplication([])
    win = PMCXObjectWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
