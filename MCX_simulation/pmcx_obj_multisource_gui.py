r"""
GUI for PMCX object simulation with 9 simultaneous 3 x 3 source points.

This differs from pmcx_obj_gui's scan mode: all 9 sources are launched in
one PMCX run with cfg["srcid"] = -1, then detected photons are split by
detp["srcid"] and saved as 9 separate 32 x 32 x time cubes.

Run with:
    python pmcx_obj_multisource_gui.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg" if "--run-multisource-settings" in sys.argv else "qtagg")
import numpy as np
from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QApplication, QMessageBox

import pmcx_sim
from pmcx_obj_gui import (
    NUM_PIX,
    ObjSimSettings,
    PMCXObjectWindow,
    convolve_irf_all_pixels_tcspc,
    crop_cache_path,
    json_ready,
    load_crop_cache,
    load_irf_curve,
    make_object_cfg,
    normalize_cube,
    plot_intensity,
    plot_mask_preview,
    plot_scan_overview,
    save_crop_cache,
    scan_source_positions,
    select_quad_mask_from_image,
    sum_normalize,
)


def extract_source_id_from_detp(detp):
    if detp is None:
        return None
    if isinstance(detp, dict):
        for key in ["srcid", "src", "sourceid", "sid"]:
            if key in detp:
                return np.asarray(detp[key]).astype(int).reshape(-1)
    if isinstance(detp, np.ndarray) and detp.ndim == 2 and detp.shape[1] >= 3:
        return detp[:, 2].astype(int)
    return None


def load_or_create_object_mask(settings: ObjSimSettings, log=print):
    if settings.selected_mask_path and os.path.exists(settings.selected_mask_path):
        log("Loading selected binary mask...")
        mask = np.load(settings.selected_mask_path).astype(np.uint8)
        crop = (
            np.load(settings.selected_crop_path)
            if settings.selected_crop_path and os.path.exists(settings.selected_crop_path)
            else mask
        )
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
    return mask, crop, quad_xy, object_present


def make_multisource_cfg(settings: ObjSimSettings, mask: np.ndarray, positions):
    cfg, meta, mask_zy = make_object_cfg(settings, mask)
    source_positions_vox = [
        [0.0, pos["source_y_mm"] / settings.voxel_size_mm, pos["source_z_mm"] / settings.voxel_size_mm]
        for pos in positions
    ]
    cfg["srcpos"] = source_positions_vox
    cfg["srcdir"] = [[1.0, 0.0, 0.0] for _ in positions]
    cfg["srcid"] = -1
    cfg["nphoton"] = int(settings.nphoton) * len(positions)
    meta["source_positions"] = positions
    meta["srcpos_vox"] = source_positions_vox
    meta["num_sources"] = len(positions)
    meta["nphoton_per_source"] = int(settings.nphoton)
    meta["nphoton_total_submitted"] = int(cfg["nphoton"])
    meta["srcid_mode"] = -1
    return cfg, meta, mask_zy


def detp_to_source_detector_outputs(res, cfg, nt: int, num_sources: int):
    detp = res.get("detp") if isinstance(res, dict) else None
    if detp is None:
        raise ValueError("No detected photon data found in PMCX result.")

    detid = pmcx_sim.extract_detector_id_from_detp(detp)
    srcid = extract_source_id_from_detp(detp)
    ppath = pmcx_sim.extract_partial_path_from_detp(detp)
    if detid is None or srcid is None or ppath is None:
        raise ValueError('Cannot extract detector id, source id, or partial paths. Check cfg["srcid"] = -1.')

    detid0 = pmcx_sim.detector_id_to_zero_based(detid, NUM_PIX * NUM_PIX)
    srcid0 = srcid.astype(int) - 1
    weights = pmcx_sim.detected_photon_weights(detp, cfg)
    valid = (
        (detid0 >= 0)
        & (detid0 < NUM_PIX * NUM_PIX)
        & (srcid0 >= 0)
        & (srcid0 < num_sources)
    )
    detid0 = detid0[valid]
    srcid0 = srcid0[valid]
    ppath = ppath[valid]
    weights = weights[valid]

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
    source_idx = srcid0[t_valid]

    cubes = np.zeros((num_sources, NUM_PIX, NUM_PIX, nt), dtype=float)
    np.add.at(cubes, (source_idx, z_idx, y_idx, t_idx[t_valid]), weights[t_valid])
    intensities = np.sum(cubes, axis=3)

    counts = np.zeros((num_sources, NUM_PIX * NUM_PIX), dtype=int)
    np.add.at(counts, (source_idx, detid0[t_valid]), 1)
    return intensities, cubes, counts, detid, srcid, ppath


def run_multisource_simulation(
    settings: ObjSimSettings,
    center_y_mm: float,
    center_z_mm: float,
    spacing_mm: float,
    log=print,
):
    started = time.perf_counter()
    mask, crop, quad_xy, object_present = load_or_create_object_mask(settings, log=log)
    positions = scan_source_positions(center_y_mm, center_z_mm, spacing_mm)
    cfg, meta, mask_zy = make_multisource_cfg(settings, mask, positions)
    meta["object_present"] = object_present
    meta["object_mode"] = "mask_image" if object_present else "homogeneous_scatterer_no_object"

    log(
        "Running one PMCX simulation with 9 simultaneous sources: "
        f"nphoton per source={settings.nphoton}, total submitted={cfg['nphoton']}"
    )
    log(f"Volume shape: {meta['volume_shape_voxels']}, object slice x={meta['object_x_index']}")
    res = pmcx_sim.pmcx.mcxlab(cfg)
    log("pmcx.mcxlab returned; splitting detected photons by source id...")

    nt = int(np.ceil((cfg["tend"] - cfg["tstart"]) / cfg["tstep"]))
    intensity_raw, cubes_raw, source_detector_counts, detid, srcid, ppath = detp_to_source_detector_outputs(
        res,
        cfg,
        nt=nt,
        num_sources=len(positions),
    )

    log("Loading IRF and convolving source-separated TPSF cubes...")
    irf, irf_var = load_irf_curve(settings.irf_mat_path, matlab_index=(16, 16))
    period_bins = nt
    cubes_irf = np.stack(
        [convolve_irf_all_pixels_tcspc(cubes_raw[idx], irf, period_bins=period_bins) for idx in range(len(positions))],
        axis=0,
    )
    cubes_irf_norm = normalize_cube(cubes_irf)
    cubes_irf_sum_norm = sum_normalize(cubes_irf)
    intensity_irf_raw = np.sum(cubes_irf, axis=3)
    intensity_irf_norm = np.sum(cubes_irf_norm, axis=3)
    intensity_irf_sum_norm = np.sum(cubes_irf_sum_norm, axis=3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(settings.output_root) / f"{timestamp}_obj_pmcx_3x3_multisource"
    result_dir.mkdir(parents=True, exist_ok=False)

    plot_mask_preview(crop, mask, result_dir / "selected_mask_preview.png")
    plot_scan_overview(cubes_irf_norm, positions, result_dir / "multisource_3x3_overview.png")
    np.save(result_dir / "multisource_tpsf_cube_raw_9x32x32xt.npy", cubes_raw)
    np.save(result_dir / "multisource_tpsf_cube_irf_9x32x32xt.npy", cubes_irf)
    np.save(result_dir / "multisource_tpsf_cube_irf_norm_9x32x32xt.npy", cubes_irf_norm)
    np.save(result_dir / "multisource_tpsf_cube_irf_sum_norm_9x32x32xt.npy", cubes_irf_sum_norm)
    np.save(result_dir / "multisource_detector_intensity_raw_9x32x32.npy", intensity_raw)
    np.save(result_dir / "source_detector_counts_9x1024.npy", source_detector_counts)
    np.save(result_dir / "object_mask_zy.npy", mask_zy)
    np.save(result_dir / "selected_image_crop.npy", crop)
    np.save(result_dir / "vol_uint8.npy", cfg["vol"])

    for idx, pos in enumerate(positions):
        point_name = f"source{idx + 1:02d}_row{pos['row']}_col{pos['col']}"
        np.save(result_dir / f"{point_name}_tpsf_cube_raw_yzt.npy", cubes_raw[idx])
        np.save(result_dir / f"{point_name}_tpsf_cube_irf_yzt.npy", cubes_irf[idx])
        np.save(result_dir / f"{point_name}_tpsf_cube_irf_norm_yzt.npy", cubes_irf_norm[idx])
        np.save(result_dir / f"{point_name}_detector_intensity_raw.npy", intensity_raw[idx])
        np.save(result_dir / f"{point_name}_detector_intensity_irf_norm.npy", intensity_irf_norm[idx])
        plot_intensity(intensity_irf_sum_norm[idx], result_dir / f"{point_name}_intensity_sum_normalized.png")
        np.savez(
            result_dir / f"{point_name}_pmcx_obj_multisource_result.npz",
            tpsf_cube_raw_yzt=cubes_raw[idx],
            tpsf_cube_irf_yzt=cubes_irf[idx],
            tpsf_cube_irf_norm_yzt=cubes_irf_norm[idx],
            tpsf_cube_irf_sum_norm_yzt=cubes_irf_sum_norm[idx],
            detector_intensity_raw=intensity_raw[idx],
            detector_intensity_irf_raw=intensity_irf_raw[idx],
            detector_intensity_irf_norm=intensity_irf_norm[idx],
            detector_intensity_irf_sum_norm=intensity_irf_sum_norm[idx],
            source_position_yz_mm=np.asarray([pos["source_y_mm"], pos["source_z_mm"]], dtype=float),
            source_index=idx + 1,
        )

    np.savez(
        result_dir / "pmcx_obj_multisource_result.npz",
        multisource_tpsf_cube_raw_9x32x32xt=cubes_raw,
        multisource_tpsf_cube_irf_9x32x32xt=cubes_irf,
        multisource_tpsf_cube_irf_norm_9x32x32xt=cubes_irf_norm,
        multisource_tpsf_cube_irf_sum_norm_9x32x32xt=cubes_irf_sum_norm,
        multisource_detector_intensity_raw_9x32x32=intensity_raw,
        multisource_detector_intensity_irf_norm_9x32x32=intensity_irf_norm,
        source_detector_counts_9x1024=source_detector_counts,
        source_positions_yz_mm=np.asarray([[p["source_y_mm"], p["source_z_mm"]] for p in positions], dtype=float),
        point_row_col=np.asarray([[p["row"], p["col"]] for p in positions], dtype=int),
        detid=np.asarray(detid) if detid is not None else np.asarray([]),
        srcid=np.asarray(srcid) if srcid is not None else np.asarray([]),
        ppath=np.asarray(ppath) if ppath is not None else np.asarray([]),
        irf=irf,
        irf_variable=irf_var,
        period_bins=int(period_bins),
    )

    with open(result_dir / "settings_and_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            json_ready(
                {
                    "settings": asdict(settings),
                    "meta": meta,
                    "selected_quad_xy": quad_xy,
                    "source_positions": positions,
                    "output_note": "One PMCX run with 9 simultaneous sources; cubes split by detected photon srcid.",
                }
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    elapsed = time.perf_counter() - started
    log(f"Saved simultaneous 3x3 source result to {result_dir}")
    log(f"Split cube shape: {cubes_raw.shape}")
    log(f"Finished in {elapsed:.1f}s")
    return result_dir


def run_multisource_from_settings_file(settings_path: str):
    with open(settings_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    settings = ObjSimSettings(**payload["settings"])
    scan = payload["scan"]
    result_dir = run_multisource_simulation(
        settings,
        center_y_mm=float(scan["center_y_mm"]),
        center_z_mm=float(scan["center_z_mm"]),
        spacing_mm=float(scan["spacing_mm"]),
        log=lambda text: print(text, flush=True),
    )
    print(f"MULTISOURCE_DIR={result_dir}", flush=True)


class PMCXObjectMultiSourceWindow(PMCXObjectWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PMCX Object Simultaneous 3x3 Source GUI")

    def init_ui(self):
        super().init_ui()
        self.run_btn.setText("Run simultaneous 3x3 sources")
        try:
            self.run_btn.clicked.disconnect()
        except TypeError:
            pass
        self.run_btn.clicked.connect(self.start_multisource_run)
        self.scan_btn.setVisible(False)
        self.compare_btn.setVisible(False)
        self.scan_viewer_btn.setVisible(False)

    def start_multisource_run(self):
        try:
            settings = self.collect_settings()
            settings.experiment_mat_path = ""
            if not settings.irf_mat_path or not os.path.exists(settings.irf_mat_path):
                raise ValueError("Please select a valid IRF MAT file.")
            if settings.mask_image_path and not os.path.isfile(settings.mask_image_path):
                raise ValueError("Target image must be a valid image file, or leave it empty for no object.")
            if settings.fov_mm <= 0:
                raise ValueError("FOV must be positive.")
            if self.scan_spacing.value() <= 0:
                raise ValueError("3x3 source spacing must be positive.")

            run_dir = Path(settings.output_root) / "_gui_runs"
            run_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
                mask_path = run_dir / f"multisource_mask_{stamp}.npy"
                crop_path = run_dir / f"multisource_crop_{stamp}.npy"
                np.save(mask_path, mask)
                np.save(crop_path, crop)
                settings.selected_mask_path = str(mask_path)
                settings.selected_crop_path = str(crop_path)
                settings.selected_quad_xy = np.asarray(quad_xy, dtype=float).tolist()

            payload = {
                "settings": asdict(settings),
                "scan": {
                    "center_y_mm": self.scan_center_y.value(),
                    "center_z_mm": self.scan_center_z.value(),
                    "spacing_mm": self.scan_spacing.value(),
                },
            }
            settings_path = run_dir / f"multisource_settings_{stamp}.json"
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid multisource settings", str(exc))
            return

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.current_result_dir = None
        self.current_result_kind = "multisource"
        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments([str(Path(__file__).resolve()), "--run-multisource-settings", str(settings_path)])
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parent))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_process_output)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)
        self.append_log(f"Starting simultaneous 3x3 source subprocess with settings: {settings_path}")
        self.process.start()

    def read_process_output(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        for line in text.splitlines():
            if line.startswith("MULTISOURCE_DIR="):
                self.current_result_dir = line.split("=", 1)[1].strip()
                self.current_result_kind = "multisource"
            self.append_log(line)

    def process_finished(self, exit_code, exit_status):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if exit_code == 0 and self.current_result_dir:
            self.append_log(f"Finished. Result folder: {self.current_result_dir}")
            QMessageBox.information(
                self,
                "Simultaneous 3x3 source simulation finished",
                f"Saved to:\n{self.current_result_dir}",
            )
        else:
            message = f"Simulation subprocess exited with code {exit_code}, status {exit_status.name}"
            self.append_log(f"[ERROR] {message}")
            QMessageBox.critical(self, "Simulation failed", message)
        self.process = None

    def process_error(self, error):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        message = f"Simulation subprocess error: {error.name}"
        self.append_log(f"[ERROR] {message}")
        QMessageBox.critical(self, "Simulation failed", message)


def pmcx_obj_multisource_gui():
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])
    win = PMCXObjectMultiSourceWindow()
    win.show()
    if owns_app:
        app.exec()
    return win


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--run-multisource-settings":
        try:
            run_multisource_from_settings_file(sys.argv[2])
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)
            sys.exit(1)
        return
    pmcx_obj_multisource_gui()


if __name__ == "__main__":
    main()
