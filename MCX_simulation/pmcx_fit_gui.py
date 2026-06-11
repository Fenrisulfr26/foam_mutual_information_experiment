"""
GUI for fitting PMCX simulations to measured 32 x 32 x time histograms.

Run with:
    D:\\codings\\anaconda\\envs\\diffusion\\python.exe pmcx_fit_gui.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg" if "--run-fit-settings" in sys.argv else "qtagg")
import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import QObject, QProcess, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
from scipy.optimize import OptimizeResult, curve_fit, differential_evolution, minimize

import pmcx_sim
from my_display_hist import compare_hist
from pmcx_fit import (
    convolve_irf_all_pixels_tcspc,
    load_experiment_data,
    load_irf_curve,
    sum_normalize,
    weighted_3d_rmse,
)


NUM_PIX = 32


@dataclass
class FitSettings:
    experiment_mat_path: str
    experiment_point_index: int
    irf_mat_path: str
    output_root: str
    nphoton: int
    voxel_size_mm: float
    slab_thickness_mm: float
    slab_width_mm: float
    slab_height_mm: float
    fov_mm: float
    detector_diameter_mm: float
    mua0: float
    mus0: float
    n0: float
    g: float
    mua_min: float
    mua_max: float
    mus_min: float
    mus_max: float
    n_min: float
    n_max: float
    fit_n: bool
    optimizer: str
    max_evals: int
    grid_mua_steps: int
    grid_mus_steps: int
    pixel_mode: str
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    pixel_list: str
    gpuid: int
    seed: int
    loss_func: str = "composite"



def parse_pixel_selection(settings: FitSettings) -> list[tuple[int, int]]:
    """Return selected pixels as 0-based (y, x) pairs."""

    if settings.pixel_mode == "Full 32x32":
        return [(y, x) for y in range(NUM_PIX) for x in range(NUM_PIX)]

    if settings.pixel_mode == "Rectangle":
        x_min = max(1, min(NUM_PIX, settings.x_min))
        x_max = max(1, min(NUM_PIX, settings.x_max))
        y_min = max(1, min(NUM_PIX, settings.y_min))
        y_max = max(1, min(NUM_PIX, settings.y_max))
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        return [(y - 1, x - 1) for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1)]

    pixels = []
    for item in settings.pixel_list.replace("\n", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.replace(" ", ",").split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Bad pixel entry {item!r}; use X,Y like 16,16")
        x, y = int(parts[0]), int(parts[1])
        if not (1 <= x <= NUM_PIX and 1 <= y <= NUM_PIX):
            raise ValueError(f"Pixel {item!r} is outside 1..{NUM_PIX}")
        pixels.append((y - 1, x - 1))
    if not pixels:
        raise ValueError("Pixel list is empty.")
    return sorted(set(pixels))


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


def selected_detector_positions(settings: FitSettings, selected_pixels: list[tuple[int, int]]):
    detpos_full, yy_mm, zz_mm = pmcx_sim.build_detector_array_32x32(
        slab_thickness_mm=settings.slab_thickness_mm,
        slab_width_mm=settings.slab_width_mm,
        slab_height_mm=settings.slab_height_mm,
        fov_mm=settings.fov_mm,
        num_pix=NUM_PIX,
        detector_diameter_mm=settings.detector_diameter_mm,
        voxel_size_mm=settings.voxel_size_mm,
        put_on_boundary=True,
    )
    # build_detector_array_32x32 loops z first, then y:
    #   detid0 = z_index * NUM_PIX + y_index
    # The data cube convention inherited from pmcx_fit.py is cube[y, z, t].
    selected_indices = np.array([z * NUM_PIX + y for y, z in selected_pixels], dtype=int)
    return detpos_full[selected_indices], selected_indices, yy_mm, zz_mm


def detp_to_selected_cube(res, cfg, selected_indices, nt=227):
    """Build a full 32x32xnt cube; unselected pixels remain zero."""

    detp = res.get("detp") if isinstance(res, dict) else None
    if detp is None:
        raise ValueError("No detected photon data found in PMCX result")

    detid = pmcx_sim.extract_detector_id_from_detp(detp)
    ppath = pmcx_sim.extract_partial_path_from_detp(detp)
    if detid is None or ppath is None:
        raise ValueError("Cannot extract detid/ppath from PMCX result")

    prop = np.asarray(cfg["prop"], dtype=float)
    unit_mm = float(cfg.get("unitinmm", 1.0))
    photon_weight = pmcx_sim.detected_photon_weights(detp, cfg)

    detid0_local = pmcx_sim.detector_id_to_zero_based(detid, len(selected_indices))
    valid = (detid0_local >= 0) & (detid0_local < len(selected_indices))
    detid0_local = detid0_local[valid]
    ppath = ppath[valid]
    photon_weight = photon_weight[valid]

    media_n = prop[1 : 1 + ppath.shape[1], 3]
    tof_ns = np.sum(ppath * unit_mm * media_n[None, :], axis=1) / 299.792458

    tstart_ns = float(cfg["tstart"]) * 1e9
    tstep_ns = float(cfg["tstep"]) * 1e9
    edges = tstart_ns + np.arange(nt + 1) * tstep_ns
    t_idx = np.searchsorted(edges, tof_ns, side="right") - 1
    t_valid = (t_idx >= 0) & (t_idx < nt)

    global_det_indices = selected_indices[detid0_local[t_valid]]
    y_idx = global_det_indices % NUM_PIX
    z_idx = global_det_indices // NUM_PIX

    cube = np.zeros((NUM_PIX, NUM_PIX, nt), dtype=float)
    np.add.at(cube, (y_idx, z_idx, t_idx[t_valid]), photon_weight[t_valid])
    return cube


def normalize_cube(cube):
    m = np.nanmax(cube)
    return cube / m if m > 0 else cube


def fit_loss(sim_cube, exp_cube, selected_pixels, loss_func="composite"):
    mask = np.zeros((NUM_PIX, NUM_PIX), dtype=bool)
    for y, x in selected_pixels:
        mask[y, x] = True

    sim = normalize_cube(sim_cube)
    exp = normalize_cube(exp_cube)

    sim_sel = sim[mask, :]
    exp_sel = exp[mask, :]

    if loss_func == "simple_rmse":
        return np.sqrt(np.mean((sim_sel - exp_sel) ** 2))

    loss_3d = weighted_3d_rmse(sim_sel, exp_sel)

    sim_map = sum_normalize(np.sum(sim, axis=2)[mask])
    exp_map = sum_normalize(np.sum(exp, axis=2)[mask])
    loss_map = np.sqrt(np.mean((sim_map - exp_map) ** 2))

    sim_time = sum_normalize(np.sum(sim[mask, :], axis=0))
    exp_time = sum_normalize(np.sum(exp[mask, :], axis=0))
    loss_time = np.sqrt(np.mean((sim_time - exp_time) ** 2))

    return 0.60 * loss_3d + 0.25 * loss_map + 0.15 * loss_time



def plot_grid_loss_surface(loss_grid, mua_values, mus_values, show=True):
    loss_grid = np.asarray(loss_grid, dtype=float)
    mua_values = np.asarray(mua_values, dtype=float)
    mus_values = np.asarray(mus_values, dtype=float)
    best_i, best_j = np.unravel_index(np.nanargmin(loss_grid), loss_grid.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        loss_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[mus_values[0], mus_values[-1], mua_values[0], mua_values[-1]],
    )
    ax.scatter([mus_values[best_j]], [mua_values[best_i]], c="red", s=55, marker="x", linewidths=2)
    ax.set_xlabel("mu_s mm^-1")
    ax.set_ylabel("mu_a mm^-1")
    ax.set_title(
        f"Grid search loss, best={float(loss_grid[best_i, best_j]):.6g} "
        f"at mu_a={mua_values[best_i]:.6g}, mu_s={mus_values[best_j]:.6g}"
    )
    plt.colorbar(im, ax=ax, label="loss")
    fig.tight_layout()
    if show:
        plt.show(block=False)
    return fig


class FitEngine:
    def __init__(self, settings: FitSettings, log):
        self.settings = settings
        self.log = log
        self.eval_count = 0
        self.stop_requested = False
        self.best_loss = np.inf
        self.best_params = None
        self.best_sim_cube = None
        self.exp_cube, self.exp_var = load_experiment_cube(settings.experiment_mat_path, settings.experiment_point_index)
        self.irf, self.irf_var = load_irf_curve(settings.irf_mat_path, matlab_index=(16, 16))
        self.selected_pixels = parse_pixel_selection(settings)
        self.detpos, self.selected_indices, self.yy_mm, self.zz_mm = selected_detector_positions(
            settings, self.selected_pixels
        )
        self.selected_time_mask = self.make_selected_time_mask()

    def make_selected_time_mask(self):
        mask2d = np.zeros((NUM_PIX, NUM_PIX), dtype=bool)
        for y, x in self.selected_pixels:
            mask2d[y, x] = True
        return np.repeat(mask2d[:, :, None], self.exp_cube.shape[2], axis=2)

    def parameter_names(self):
        names = ["mua", "mus"]
        if self.settings.fit_n:
            names.append("n")
        return names

    def p0(self):
        vals = [self.settings.mua0, self.settings.mus0]
        if self.settings.fit_n:
            vals.append(self.settings.n0)
        return np.array(vals, dtype=float)

    def bounds(self):
        low = [self.settings.mua_min, self.settings.mus_min]
        high = [self.settings.mua_max, self.settings.mus_max]
        if self.settings.fit_n:
            low.append(self.settings.n_min)
            high.append(self.settings.n_max)
        return np.array(low, dtype=float), np.array(high, dtype=float)

    def request_stop(self):
        self.stop_requested = True

    def run_sim(self, params):
        mua = float(params[0])
        mus = float(params[1])
        n = float(params[2]) if self.settings.fit_n and len(params) >= 3 else self.settings.n0

        cfg, _ = pmcx_sim.make_foam_slab_cfg(
            nphoton=self.settings.nphoton,
            voxel_size_mm=self.settings.voxel_size_mm,
            slab_thickness_mm=self.settings.slab_thickness_mm,
            slab_width_mm=self.settings.slab_width_mm,
            slab_height_mm=self.settings.slab_height_mm,
            fov_mm=self.settings.fov_mm,
            num_pix=NUM_PIX,
            detector_diameter_mm=self.settings.detector_diameter_mm,
            mua=mua,
            mus=mus,
            g=self.settings.g,
            n=n,
            gpuid=self.settings.gpuid,
            seed=self.settings.seed,
        )
        cfg["detpos"] = self.detpos
        cfg["issave2pt"] = 0
        cfg.pop("outputtype", None)
        # cfg.pop("debuglevel", None) # display the prograss bar

        res = pmcx_sim.pmcx.mcxlab(cfg)
        cube = detp_to_selected_cube(res, cfg, self.selected_indices, nt=self.exp_cube.shape[2])
        return convolve_irf_all_pixels_tcspc(cube, self.irf, period_bins=self.exp_cube.shape[2])

    def objective(self, params):
        if self.stop_requested:
            raise StopIteration("Fit stopped by user.")

        low, high = self.bounds()
        params = np.clip(np.asarray(params, dtype=float), low, high)
        self.eval_count += 1
        param_text = ", ".join(f"{n}={v:.6g}" for n, v in zip(self.parameter_names(), params))
        self.log(f"[eval {self.eval_count}/{self.settings.max_evals}] start {param_text}")
        started = time.perf_counter()
        try:
            sim_cube = self.run_sim(params)
            loss = float(fit_loss(sim_cube, self.exp_cube, self.selected_pixels, self.settings.loss_func))
        except Exception as exc:
            self.log(f"[warn] simulation failed at {params}: {exc}")
            return 1e12
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = params.copy()
            self.best_sim_cube = sim_cube
        elapsed = time.perf_counter() - started
        self.log(f"[eval {self.eval_count}/{self.settings.max_evals}] done rmse={loss:.8g}, time={elapsed:.1f}s")
        return loss

    def fit(self):
        low, high = self.bounds()
        p0 = np.clip(self.p0(), low, high)

        if self.settings.optimizer == "grid_search":
            return self.fit_grid_search()

        if self.settings.optimizer == "curve_fit":
            return self.fit_curve_fit(p0, low, high)

        bounds = list(zip(low, high))
        p0_loss = self.objective(p0)

        result_de = differential_evolution(
            lambda x: self.objective(x),
            bounds=bounds,
            seed=self.settings.seed,
            maxiter=max(1, self.settings.max_evals // 12),
            popsize=6,
            polish=False,
            workers=1,
            updating="immediate",
            x0=p0,
        )

        result_nm = minimize(
            lambda x: self.objective(np.clip(x, low, high)),
            result_de.x,
            method="Nelder-Mead",
            options={"maxiter": max(10, self.settings.max_evals // 3), "xatol": 1e-4, "fatol": 1e-5},
        )

        nm_x = np.clip(result_nm.x, low, high)
        candidates = [
            (p0_loss, p0, "p0"),
            (float(result_de.fun), result_de.x, "differential_evolution"),
            (float(result_nm.fun), nm_x, "nelder_mead"),
        ]

        loss, params, source = min(candidates, key=lambda item: item[0])
        if self.best_params is not None and np.allclose(params, self.best_params):
            sim_cube = self.best_sim_cube
            loss = float(self.best_loss)
        else:
            sim_cube = self.run_sim(params)
            loss = float(fit_loss(sim_cube, self.exp_cube, self.selected_pixels, self.settings.loss_func))
        return OptimizeResult(x=params, fun=loss, success=True, message=f"best candidate from {source}", sim_cube=sim_cube)

    def fit_grid_search(self):
        if self.settings.fit_n:
            self.log("[grid] fit n is ignored in grid_search; using fixed n=n initial.")

        mua_values = np.linspace(self.settings.mua_min, self.settings.mua_max, self.settings.grid_mua_steps)
        mus_values = np.linspace(self.settings.mus_min, self.settings.mus_max, self.settings.grid_mus_steps)
        loss_grid = np.full((mua_values.size, mus_values.size), np.nan, dtype=float)
        total = int(mua_values.size * mus_values.size)

        exp_compare = normalize_cube(self.exp_cube)
        best_loss = np.inf
        best_params = None
        best_sim_cube = None

        for i, mua in enumerate(mua_values):
            for j, mus in enumerate(mus_values):
                if self.stop_requested:
                    raise StopIteration("Fit stopped by user.")
                self.eval_count += 1
                params = np.array([float(mua), float(mus)], dtype=float)
                self.log(
                    f"[grid {self.eval_count}/{total}] start mua={mua:.8g}, mus={mus:.8g}, n={self.settings.n0:.8g}"
                )
                started = time.perf_counter()
                try:
                    sim_cube = self.run_sim(params)
                    sim_compare = normalize_cube(sim_cube)
                    loss = float(fit_loss(sim_compare, exp_compare, self.selected_pixels, self.settings.loss_func))
                except Exception as exc:
                    self.log(f"[warn] grid simulation failed at mua={mua:.8g}, mus={mus:.8g}: {exc}")
                    loss = 1e12
                    sim_cube = None
                loss_grid[i, j] = loss
                if loss < best_loss and sim_cube is not None:
                    best_loss = loss
                    best_params = params.copy()
                    best_sim_cube = sim_cube
                elapsed = time.perf_counter() - started
                self.log(f"[grid {self.eval_count}/{total}] done loss={loss:.8g}, time={elapsed:.1f}s")

        if best_params is None:
            raise RuntimeError("Grid search failed: no valid PMCX simulation result.")

        return OptimizeResult(
            x=best_params,
            fun=float(best_loss),
            success=True,
            message=f"grid_search {mua_values.size}x{mus_values.size}, fixed n={self.settings.n0}",
            sim_cube=best_sim_cube,
            loss_grid=loss_grid,
            grid_mua_values=mua_values,
            grid_mus_values=mus_values,
        )

    def fit_curve_fit(self, p0, low, high):
        y_full = normalize_cube(self.exp_cube).reshape(-1)
        valid = np.isfinite(y_full) & self.selected_time_mask.reshape(-1)
        xdata = np.arange(np.count_nonzero(valid), dtype=float)

        def model_func(_x, *params):
            if self.stop_requested:
                raise StopIteration("Fit stopped by user.")
            if self.eval_count >= self.settings.max_evals:
                raise RuntimeError(f"Reached max simulations: {self.settings.max_evals}")
            self.eval_count += 1
            param_text = ", ".join(f"{n}={v:.6g}" for n, v in zip(self.parameter_names(), params))
            self.log(f"[eval {self.eval_count}/{self.settings.max_evals}] start {param_text}")
            started = time.perf_counter()
            sim_cube = normalize_cube(self.run_sim(params)).reshape(-1)
            elapsed = time.perf_counter() - started
            self.log(f"[eval {self.eval_count}/{self.settings.max_evals}] done curve_fit model, time={elapsed:.1f}s")
            return sim_cube[valid]

        popt, pcov = curve_fit(
            model_func,
            xdata,
            y_full[valid],
            p0=p0,
            bounds=(low, high),
            method="trf",
            max_nfev=self.settings.max_evals,
        )
        sim_cube = self.run_sim(popt)
        loss = float(fit_loss(sim_cube, self.exp_cube, self.selected_pixels, self.settings.loss_func))
        return OptimizeResult(x=np.asarray(popt), fun=loss, success=True, message="curve_fit finished", pcov=pcov, sim_cube=sim_cube)

    def save_result(self, result):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(self.settings.output_root) / f"{timestamp}_rmse_{result.fun:.6g}"
        result_dir.mkdir(parents=True, exist_ok=False)

        names = self.parameter_names()
        params = {name: float(value) for name, value in zip(names, result.x)}
        if "n" not in params:
            params["n"] = self.settings.n0

        exp_compare = normalize_cube(self.exp_cube)
        sim_compare = normalize_cube(result.sim_cube)
        exp_map = normalize_cube(np.sum(exp_compare, axis=2))
        sim_map = normalize_cube(np.sum(sim_compare, axis=2))
        diff = np.abs(exp_map - sim_map)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, data, title, cmap in [
            (axes[0], exp_map, "Experiment", "jet"),
            (axes[1], sim_map, "Simulation", "jet"),
            (axes[2], diff, "Absolute difference", "hot"),
        ]:
            im = ax.imshow(data, origin="upper", cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel("Y index")
            ax.set_ylabel("Z index")
            plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(result_dir / "fitting_comparison.png", dpi=150)
        plt.close(fig)

        has_grid = all(hasattr(result, name) for name in ["loss_grid", "grid_mua_values", "grid_mus_values"])
        if has_grid:
            self.save_grid_surface(result, result_dir)

        extra_arrays = {}
        if has_grid:
            extra_arrays = {
                "loss_grid": result.loss_grid,
                "grid_mua_values": result.grid_mua_values,
                "grid_mus_values": result.grid_mus_values,
            }

        np.savez(
            result_dir / "fitting_result.npz",
            exp_cube=self.exp_cube,
            sim_cube=result.sim_cube,
            exp_compare=exp_compare,
            sim_compare=sim_compare,
            exp_map2d=exp_map,
            sim_map2d=sim_map,
            selected_indices=self.selected_indices,
            selected_pixels=np.array(self.selected_pixels, dtype=int),
            rmse=float(result.fun),
            **extra_arrays,
            **params,
        )

        metadata = {
            "settings": asdict(self.settings),
            "experiment_variable": self.exp_var,
            "irf_variable": self.irf_var,
            "selected_detector_count": int(len(self.selected_indices)),
            "selected_pixels_yx_zero_based": self.selected_pixels,
            "result": {
                "rmse": float(result.fun),
                "parameters": params,
                "message": str(result.message),
            },
            "loss_normalization": "sim_compare and exp_compare are global-max normalized before loss comparison.",
        }
        with open(result_dir / "fit_report.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        with open(result_dir / "fit_report.txt", "w", encoding="utf-8") as f:
            f.write("PMCX FIT RESULT\n")
            f.write("================\n")
            f.write(f"experiment: {self.settings.experiment_mat_path}\n")
            f.write(f"irf: {self.settings.irf_mat_path}\n")
            f.write(f"optimizer: {self.settings.optimizer}\n")
            f.write(f"selected_detector_count: {len(self.selected_indices)}\n")
            f.write(f"rmse: {result.fun:.8e}\n")
            f.write("loss_normalization: global max normalization on sim_compare and exp_compare before loss\n")
            for name, value in params.items():
                f.write(f"{name}: {value:.8e}\n")
            if has_grid:
                f.write(f"grid_mua_steps: {len(result.grid_mua_values)}\n")
                f.write(f"grid_mus_steps: {len(result.grid_mus_values)}\n")
                f.write("grid_surface: grid_loss_surface.png\n")
        return result_dir

    def save_grid_surface(self, result, result_dir: Path):
        loss_grid = np.asarray(result.loss_grid, dtype=float)
        mua_values = np.asarray(result.grid_mua_values, dtype=float)
        mus_values = np.asarray(result.grid_mus_values, dtype=float)
        fig = plot_grid_loss_surface(loss_grid, mua_values, mus_values, show=False)
        fig.savefig(result_dir / "grid_loss_surface.png", dpi=180)
        plt.close(fig)


class FitWorker(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, settings: FitSettings):
        super().__init__()
        self.settings = settings
        self.engine = None

    def stop(self):
        if self.engine is not None:
            self.engine.request_stop()

    def run(self):
        try:
            self.engine = FitEngine(self.settings, self.log.emit)
            self.log.emit(
                f"Loaded experiment: {self.engine.exp_cube.shape}, selected detectors: {len(self.engine.selected_indices)}"
            )
            result = self.engine.fit()
            result_dir = self.engine.save_result(result)
            self.finished.emit(str(result_dir))
        except Exception as exc:
            self.failed.emit(str(exc))


def run_fit_from_settings_file(settings_path: str):
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = FitSettings(**json.load(f))

    def log(message):
        print(message, flush=True)

    engine = FitEngine(settings, log)
    log(f"Loaded experiment: {engine.exp_cube.shape}, selected detectors: {len(engine.selected_indices)}")
    result = engine.fit()
    result_dir = engine.save_result(result)
    log(f"RESULT_DIR={result_dir}")


class PMCXFitWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PMCX Fitting GUI")
        self.thread = None
        self.worker = None
        self.process = None
        self.current_result_dir = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        paths = QGroupBox("Data and output")
        path_layout = QGridLayout(paths)
        self.exp_path = QLineEdit(
            r"F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\data\3x3_grid_scan_20260520_202613_deg_neg3_exp_2us_frames_100000_avg_20\hist_2us_100000_avg20_point05_center_cal.mat"
        )
        self.exp_point_index = self.spin_int(1, 100, 5)
        self.irf_path = QLineEdit(r"F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\data\IRF_20260601_165629_deg_2_exp_2us_frames_100000_avg_1\hist_2us_100000_avg1_point05_center_obj.mat")
        self.output_root = QLineEdit(r"F:\OneDrive\foam_imaging_project\experiment_setup\MCX_simulation\fit_results")
        self.add_path_row(path_layout, 0, "Experiment MAT", self.exp_path)
        path_layout.addWidget(QLabel("4D experiment point index"), 1, 0)
        path_layout.addWidget(self.exp_point_index, 1, 1)
        self.add_path_row(path_layout, 2, "IRF MAT", self.irf_path)
        self.add_dir_row(path_layout, 3, "Output root", self.output_root)
        layout.addWidget(paths)

        params_row = QHBoxLayout()
        params_row.addWidget(self.sim_group())
        params_row.addWidget(self.fit_group())
        params_row.addWidget(self.pixel_group())
        layout.addLayout(params_row)

        buttons = QHBoxLayout()
        self.run_btn = QPushButton("Run fit")
        self.stop_btn = QPushButton("Stop after current simulation")
        self.check_btn = QPushButton("Check fitting result folder")
        self.run_btn.clicked.connect(self.start_fit)
        self.stop_btn.clicked.connect(self.stop_fit)
        self.check_btn.clicked.connect(self.check_result_folder)
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addWidget(self.check_btn)
        layout.addLayout(buttons)
        self.stop_btn.setEnabled(False)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box, stretch=1)

        self.setCentralWidget(central)
        self.resize(1180, 780)

    def add_path_row(self, layout, row, label, line_edit):
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(line_edit, row, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.browse_file(line_edit))
        layout.addWidget(btn, row, 2)

    def add_dir_row(self, layout, row, label, line_edit):
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(line_edit, row, 1)
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self.browse_dir(line_edit))
        layout.addWidget(btn, row, 2)

    def sim_group(self):
        group = QGroupBox("PMCX simulation")
        form = QFormLayout(group)
        self.nphoton = self.spin_int(1, 1_000_000_000, 500_000_000)
        self.voxel = self.spin_float(0.1, 10, 1.0, 2)
        self.thickness = self.spin_float(1, 500, 50.0, 2)
        self.width = self.spin_float(1, 1000, 250.0, 2)
        self.height = self.spin_float(1, 1000, 250.0, 2)
        self.fov = self.spin_float(1, 1000, 140.0, 2)
        self.det_diam = self.spin_float(0.01, 100, 1.0, 3)
        self.g = self.spin_float(-0.99, 0.99, 0.0, 3)
        self.gpuid = self.spin_int(0, 16, 1)
        self.seed = self.spin_int(1, 2_000_000_000, 123456789)
        for label, widget in [
            ("nphoton", self.nphoton),
            ("voxel size mm", self.voxel),
            ("thickness mm", self.thickness),
            ("width mm", self.width),
            ("height mm", self.height),
            ("FOV mm", self.fov),
            ("detector diameter mm", self.det_diam),
            ("g fixed", self.g),
            ("GPU id", self.gpuid),
            ("seed", self.seed),
        ]:
            form.addRow(label, widget)
        return group

    def fit_group(self):
        group = QGroupBox("Fit parameters")
        form = QFormLayout(group)
        self.mua0 = self.spin_float(1e-6, 1, 0.0019, 6)
        self.mus0 = self.spin_float(1e-4, 100, 1.4, 5)
        self.n0 = self.spin_float(1.0, 3.0, 1.05, 4)
        self.mua_min = self.spin_float(1e-6, 1, 1e-5, 6)
        self.mua_max = self.spin_float(1e-6, 1, 0.1, 6)
        self.mus_min = self.spin_float(1e-4, 100, 0.1, 5)
        self.mus_max = self.spin_float(1e-4, 100, 10.0, 5)
        self.n_min = self.spin_float(1.0, 3.0, 1.0, 4)
        self.n_max = self.spin_float(1.0, 3.0, 2.2, 4)
        self.fit_n = QCheckBox("fit n")
        self.fit_n.setChecked(False)
        self.optimizer = QComboBox()
        self.optimizer.addItems(["differential", "curve_fit", "grid_search"])
        self.max_evals = self.spin_int(1, 10000, 80)
        self.grid_mua_steps = self.spin_int(2, 1000, 100)
        self.grid_mus_steps = self.spin_int(2, 1000, 100)
        self.loss_func = QComboBox()
        self.loss_func.addItems(["composite", "simple_rmse"])
        for label, widget in [
            ("mua initial", self.mua0),
            ("mus initial", self.mus0),
            ("n initial", self.n0),
            ("mua min", self.mua_min),
            ("mua max", self.mua_max),
            ("mus min", self.mus_min),
            ("mus max", self.mus_max),
            ("n min", self.n_min),
            ("n max", self.n_max),
            ("fit n", self.fit_n),
            ("optimizer", self.optimizer),
            ("max evals", self.max_evals),
            ("grid mu_a steps", self.grid_mua_steps),
            ("grid mu_s steps", self.grid_mus_steps),
            ("loss function", self.loss_func),
        ]:
            form.addRow(label, widget)
        return group

    def pixel_group(self):
        group = QGroupBox("Fitting pixels")
        form = QFormLayout(group)
        self.pixel_mode = QComboBox()
        self.pixel_mode.addItems(["Full 32x32", "Rectangle", "Pixel list"])
        self.x_min = self.spin_int(1, 32, 1)
        self.x_max = self.spin_int(1, 32, 32)
        self.y_min = self.spin_int(1, 32, 1)
        self.y_max = self.spin_int(1, 32, 32)
        self.pixel_list = QLineEdit("16,16")
        for label, widget in [
            ("mode", self.pixel_mode),
            ("X min", self.x_min),
            ("X max", self.x_max),
            ("Y min", self.y_min),
            ("Y max", self.y_max),
            ("pixel list", self.pixel_list),
        ]:
            form.addRow(label, widget)
        return group

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

    def browse_file(self, line_edit):
        path, _ = QFileDialog.getOpenFileName(self, "Select MAT file", "", "MAT files (*.mat);;All files (*)")
        if path:
            line_edit.setText(path)

    def browse_dir(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path:
            line_edit.setText(path)

    def collect_settings(self):
        return FitSettings(
            experiment_mat_path=self.exp_path.text().strip(),
            experiment_point_index=self.exp_point_index.value(),
            irf_mat_path=self.irf_path.text().strip(),
            output_root=self.output_root.text().strip(),
            nphoton=self.nphoton.value(),
            voxel_size_mm=self.voxel.value(),
            slab_thickness_mm=self.thickness.value(),
            slab_width_mm=self.width.value(),
            slab_height_mm=self.height.value(),
            fov_mm=self.fov.value(),
            detector_diameter_mm=self.det_diam.value(),
            mua0=self.mua0.value(),
            mus0=self.mus0.value(),
            n0=self.n0.value(),
            g=self.g.value(),
            mua_min=self.mua_min.value(),
            mua_max=self.mua_max.value(),
            mus_min=self.mus_min.value(),
            mus_max=self.mus_max.value(),
            n_min=self.n_min.value(),
            n_max=self.n_max.value(),
            fit_n=self.fit_n.isChecked(),
            optimizer=self.optimizer.currentText(),
            max_evals=self.max_evals.value(),
            grid_mua_steps=self.grid_mua_steps.value(),
            grid_mus_steps=self.grid_mus_steps.value(),
            pixel_mode=self.pixel_mode.currentText(),
            x_min=self.x_min.value(),
            x_max=self.x_max.value(),
            y_min=self.y_min.value(),
            y_max=self.y_max.value(),
            pixel_list=self.pixel_list.text(),
            gpuid=self.gpuid.value(),
            seed=self.seed.value(),
            loss_func=self.loss_func.currentText(),
        )

    def append_log(self, text):
        self.log_box.append(text)

    def start_fit(self):
        try:
            settings = self.collect_settings()
            if not settings.experiment_mat_path or not os.path.exists(settings.experiment_mat_path):
                raise ValueError("Please select a valid experiment MAT file.")
            if not settings.irf_mat_path or not os.path.exists(settings.irf_mat_path):
                raise ValueError("Please select a valid IRF MAT file.")
            parse_pixel_selection(settings)
            if settings.optimizer == "grid_search":
                total = settings.grid_mua_steps * settings.grid_mus_steps
                if total >= 1000 or settings.nphoton >= 10_000_000:
                    reply = QMessageBox.question(
                        self,
                        "Confirm grid search",
                        (
                            f"Grid search will run {total} PMCX simulations with "
                            f"nphoton={settings.nphoton:,} each.\n\nContinue?"
                        ),
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
            run_dir = Path(settings.output_root) / "_gui_runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            settings_path = run_dir / f"settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(asdict(settings), f, ensure_ascii=False, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.current_result_dir = None
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments([str(Path(__file__).resolve()), "--run-fit-settings", str(settings_path)])
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parent))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_fit_process_output)
        self.process.finished.connect(self.fit_process_finished)
        self.process.errorOccurred.connect(self.fit_process_error)
        self.append_log(f"Starting fit subprocess with settings: {settings_path}")
        self.process.start()

    def stop_fit(self):
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            self.append_log("Stop requested. Terminating the fitting subprocess...")
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()
                self.append_log("Subprocess did not terminate quickly; killed it.")
        self.stop_btn.setEnabled(False)

    def read_fit_process_output(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        for line in text.splitlines():
            if line.startswith("RESULT_DIR="):
                self.current_result_dir = line.split("=", 1)[1].strip()
            self.append_log(line)

    def fit_process_finished(self, exit_code, exit_status):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if exit_code == 0 and self.current_result_dir:
            self.append_log(f"Finished. Result folder: {self.current_result_dir}")
            QMessageBox.information(self, "Fit finished", f"Saved to:\n{self.current_result_dir}")
        else:
            message = f"Fit subprocess exited with code {exit_code}, status {exit_status.name}"
            self.append_log(f"[ERROR] {message}")
            QMessageBox.critical(self, "Fit failed", message)
        self.process = None

    def fit_process_error(self, error):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        message = f"Fit subprocess error: {error.name}"
        self.append_log(f"[ERROR] {message}")
        QMessageBox.critical(self, "Fit failed", message)

    def check_result_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select fitting result folder")
        if not folder:
            return
        npz_path = Path(folder) / "fitting_result.npz"
        report_path = Path(folder) / "fit_report.json"
        if not npz_path.exists():
            QMessageBox.critical(self, "Missing result", f"Cannot find {npz_path}")
            return
        data = np.load(npz_path, allow_pickle=True)
        if "exp_compare" in data.files and "sim_compare" in data.files:
            exp_cube = data["exp_compare"]
            sim_cube = data["sim_compare"]
            compare_note = "normalized exp_compare/sim_compare"
        else:
            exp_cube = normalize_cube(data["exp_cube"])
            sim_cube = normalize_cube(data["sim_cube"])
            compare_note = "normalized legacy exp_cube/sim_cube"
        info = [f"Folder: {folder}", f"RMSE: {float(data['rmse']):.8g}"]
        for key in ["mua", "mus", "n"]:
            if key in data.files:
                info.append(f"{key}: {float(data[key]):.8g}")
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            info.append(f"Optimizer: {report['settings']['optimizer']}")
            info.append(f"Experiment: {report['settings']['experiment_mat_path']}")
        info.append(f"Compare display: {compare_note}")
        self.append_log("\n".join(info))
        QMessageBox.information(self, "Fit result", "\n".join(info[:5]))
        if all(key in data.files for key in ["loss_grid", "grid_mua_values", "grid_mus_values"]):
            plot_grid_loss_surface(data["loss_grid"], data["grid_mua_values"], data["grid_mus_values"], show=True)
        compare_hist(exp_cube, sim_cube, label_a="Experiment", label_b="Simulation", figure_name="Fit result comparison")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--run-fit-settings":
        try:
            run_fit_from_settings_file(sys.argv[2])
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)
            sys.exit(1)
        return

    app = QApplication([])
    win = PMCXFitWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
