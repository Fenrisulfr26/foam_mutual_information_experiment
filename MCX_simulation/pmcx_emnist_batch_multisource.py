"""
Batch PMCX object simulation using EMNIST handwritten uppercase letters.

This script is intentionally self-contained: it does not import the local
pmcx_obj_gui.py, pmcx_obj_multisource_gui.py, or pmcx_sim.py files.

Example:
    1. Edit USER_CONFIG below.
    2. Run:
        python pmcx_emnist_batch_multisource.py
"""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pmcx
from numpy.lib.format import open_memmap
from scipy.ndimage import zoom


NUM_PIX = 32
ALL_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# =========================
# User-editable run settings
# =========================
# Edit this block directly before running the script. No command-line
# arguments are needed for normal use.
USER_CONFIG = {
    # Dataset and batch selection.
    "total_images": 260,
    "letters": "ALL",  # "ALL", "ABC", or "A,C,Z"
    "data_root": "/root/autodl-tmp/pmcx_project/data",
    "output_root": "/root/autodl-tmp/pmcx_project/obj_sim_results",
    "sample_with_replacement": False,
    "seed": 123456789,
    "emnist_threshold": 0,

    # PMCX runtime.
    "nphoton": 1_000_000,  # photons per source; PMCX submits this times 9
    "gpuid": 1,
    "raw_tpsf_dtype": "float32",

    # 3 x 3 simultaneous source positions.
    "scan_center_y": 125.0,
    "scan_center_z": 105.0,
    "scan_spacing": 15.0,

    # Slab and detector geometry, kept consistent with the GUI defaults.
    "voxel_size_mm": 1.0,
    "slab_thickness_mm": 50.0,
    "slab_width_mm": 250.0,
    "slab_height_mm": 250.0,
    "detector_center_y_mm": 125.0,
    "detector_center_z_mm": 105.0,
    "fov_mm": 140.0,
    "detector_diameter_mm": 2.0,
    # Keep the original 32 x 32 detector grid positions, but only place this
    # central inclusive 1-based index range. 11..22 gives 12 x 12 detectors.
    "detector_roi_start_1based": 11,
    "detector_roi_end_1based": 22,

    # Embedded EMNIST object geometry.
    "object_x_mm": 25.0,
    "object_center_y_mm": 125.0,
    "object_center_z_mm": 105.0,
    "object_size_y_mm": 50.0,
    "object_size_z_mm": 50.0,

    # Scatterer optical properties.
    "mua": 0.0007716,
    "mus": 1.7699,
    "g": 0.0,
    "n": 1.05,
}


@dataclass
class ObjSimSettings:
    output_root: str = "/root/autodl-tmp/pmcx_project/obj_sim_results"
    nphoton: int = 1_000_000
    voxel_size_mm: float = 1.0
    slab_thickness_mm: float = 50.0
    slab_width_mm: float = 250.0
    slab_height_mm: float = 250.0
    source_y_mm: float = 125.0
    source_z_mm: float = 105.0
    detector_center_y_mm: float = 125.0
    detector_center_z_mm: float = 105.0
    fov_mm: float = 140.0
    detector_diameter_mm: float = 2.0
    detector_roi_start_1based: int = 11
    detector_roi_end_1based: int = 22
    object_x_mm: float = 25.0
    object_center_y_mm: float = 125.0
    object_center_z_mm: float = 105.0
    object_size_y_mm: float = 50.0
    object_size_z_mm: float = 50.0
    threshold: int = 0
    mua: float = 0.0007716
    mus: float = 1.7699
    g: float = 0.0
    n: float = 1.05
    gpuid: int = 1
    seed: int = 123456789


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


def build_detector_array_centered(
    slab_thickness_mm,
    center_y_mm,
    center_z_mm,
    fov_mm,
    num_pix,
    detector_diameter_mm,
    voxel_size_mm,
    roi_start_1based,
    roi_end_1based,
):
    roi_start = int(roi_start_1based) - 1
    roi_end = int(roi_end_1based) - 1
    if roi_start < 0 or roi_end >= num_pix or roi_start > roi_end:
        raise ValueError(f"Invalid detector ROI: {roi_start_1based}..{roi_end_1based} for {num_pix} x {num_pix} grid")

    pitch_mm = fov_mm / num_pix
    offsets_mm = (np.arange(num_pix) + 0.5) * pitch_mm - fov_mm / 2
    yy_mm = center_y_mm + offsets_mm[::-1]
    zz_mm = center_z_mm + offsets_mm[::-1]
    roi_indices = np.arange(roi_start, roi_end + 1, dtype=int)
    yy_roi_mm = yy_mm[roi_indices]
    zz_roi_mm = zz_mm[roi_indices]
    det_radius_mm = detector_diameter_mm / 2

    detpos = []
    for z_mm in zz_roi_mm:
        for y_mm in yy_roi_mm:
            detpos.append(
                [
                    slab_thickness_mm / voxel_size_mm,
                    y_mm / voxel_size_mm,
                    z_mm / voxel_size_mm,
                    det_radius_mm / voxel_size_mm,
                ]
            )
    return np.asarray(detpos, dtype=np.float32), yy_roi_mm, zz_roi_mm, roi_indices


def _clip_slice(center_mm, size_mm, voxel_size_mm, max_vox):
    start = int(round((center_mm - size_mm / 2) / voxel_size_mm))
    stop = int(round((center_mm + size_mm / 2) / voxel_size_mm))
    start = max(0, min(max_vox, start))
    stop = max(start + 1, min(max_vox, stop))
    return slice(start, stop)


def resize_mask(mask, out_shape):
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
    y_slice = _clip_slice(settings.object_center_y_mm, settings.object_size_y_mm, settings.voxel_size_mm, ny)
    z_slice = _clip_slice(settings.object_center_z_mm, settings.object_size_z_mm, settings.voxel_size_mm, nz)

    target_shape = (z_slice.stop - z_slice.start, y_slice.stop - y_slice.start)
    mask_zy = resize_mask(mask, target_shape)
    vol[x_idx, y_slice, z_slice] = mask_zy[::-1, :].T

    detpos, yy_mm, zz_mm, roi_indices = build_detector_array_centered(
        slab_thickness_mm=settings.slab_thickness_mm,
        center_y_mm=settings.detector_center_y_mm,
        center_z_mm=settings.detector_center_z_mm,
        fov_mm=settings.fov_mm,
        num_pix=NUM_PIX,
        detector_diameter_mm=settings.detector_diameter_mm,
        voxel_size_mm=settings.voxel_size_mm,
        roi_start_1based=settings.detector_roi_start_1based,
        roi_end_1based=settings.detector_roi_end_1based,
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
        "maxdetphoton": 100_000_000,
        "issave2pt": 0,
        "issavedet": 1,
        "savedetflag": "dp",
        "debuglevel": "P",
    }
    meta = {
        "volume_shape_voxels": (nx, ny, nz),
        "object_x_index": x_idx,
        "object_y_slice": (y_slice.start, y_slice.stop),
        "object_z_slice": (z_slice.start, z_slice.stop),
        "object_mask_shape_zy": mask_zy.shape,
        "detector_y_mm": yy_mm,
        "detector_z_mm": zz_mm,
        "detector_roi_indices_zero_based": roi_indices,
        "detector_roi_indices_one_based": roi_indices + 1,
        "num_detectors_placed": int(detpos.shape[0]),
        "detector_pitch_mm": settings.fov_mm / NUM_PIX,
        "num_pix": NUM_PIX,
        "tstart_s": tstart,
        "tend_s": tend,
        "tstep_s": tstep,
    }
    return cfg, meta, mask_zy


def scan_source_positions(center_y_mm, center_z_mm, spacing_mm):
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


def make_multisource_cfg(settings: ObjSimSettings, mask: np.ndarray, positions):
    cfg, meta, mask_zy = make_object_cfg(settings, mask)
    hist_tstart = float(cfg["tstart"])
    hist_tend = float(cfg["tend"])
    hist_tstep = float(cfg["tstep"])
    cfg["tstep"] = hist_tend - hist_tstart

    source_positions_vox = np.asarray(
        [
            [0.0, pos["source_y_mm"] / settings.voxel_size_mm, pos["source_z_mm"] / settings.voxel_size_mm]
            for pos in positions
        ],
        dtype=np.float32,
    )
    source_directions = np.asarray([[1.0, 0.0, 0.0] for _ in positions], dtype=np.float32)
    total_photons = int(settings.nphoton) * len(positions)
    max_detected = min(max(total_photons, 1_000_000), 20_000_000)
    cfg["srcpos"] = source_positions_vox
    cfg["srcdir"] = source_directions
    cfg["srcid"] = -1
    cfg["nphoton"] = total_photons
    cfg["maxdetphoton"] = int(max_detected)

    meta["source_positions"] = positions
    meta["srcpos_vox"] = source_positions_vox.tolist()
    meta["num_sources"] = len(positions)
    meta["nphoton_per_source"] = int(settings.nphoton)
    meta["nphoton_total_submitted"] = int(cfg["nphoton"])
    meta["maxdetphoton"] = int(cfg["maxdetphoton"])
    meta["srcid_mode"] = -1
    meta["hist_tstart_s"] = hist_tstart
    meta["hist_tend_s"] = hist_tend
    meta["hist_tstep_s"] = hist_tstep
    meta["pmcx_gate_tstep_s"] = float(cfg["tstep"])
    return cfg, meta, mask_zy


def extract_detector_id_from_detp(detp):
    if detp is None:
        return None
    if isinstance(detp, dict):
        for key in ["detid", "det", "detid_data", "d"]:
            if key in detp:
                return np.asarray(detp[key]).astype(int).reshape(-1)
    if isinstance(detp, np.ndarray) and detp.ndim == 2 and detp.shape[1] >= 1:
        return detp[:, 0].astype(int)
    return None


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


def extract_partial_path_from_detp(detp):
    if detp is None:
        return None
    if isinstance(detp, dict):
        for key in ["ppath", "p"]:
            if key in detp:
                ppath = np.asarray(detp[key], dtype=float)
                if ppath.ndim == 1:
                    ppath = ppath[:, None]
                return ppath
        if "data" in detp:
            data = np.asarray(detp["data"], dtype=float)
            if data.ndim == 2 and data.shape[0] >= 2:
                return data[1:, :].T
    if isinstance(detp, np.ndarray) and detp.ndim == 2 and detp.shape[1] >= 2:
        return detp[:, 1:]
    return None


def detector_id_to_zero_based(detid, num_detectors):
    detid = np.asarray(detid).astype(int)
    if detid.size == 0:
        return detid
    if detid.min() >= 1 and detid.max() <= num_detectors:
        return detid - 1
    return detid


def detected_photon_weights(detp, cfg):
    ppath = extract_partial_path_from_detp(detp)
    if ppath is None:
        raise ValueError("Cannot extract partial path lengths from detected photon data")
    prop = np.asarray(cfg["prop"], dtype=float)
    unit_mm = float(cfg.get("unitinmm", 1.0))
    mua = prop[1 : 1 + ppath.shape[1], 0]
    if mua.size != ppath.shape[1]:
        raise ValueError("cfg['prop'] does not match detected photon partial-path columns")
    if isinstance(detp, dict) and "w0" in detp:
        weight = np.asarray(detp["w0"], dtype=float).reshape(-1)
    else:
        weight = np.ones(ppath.shape[0], dtype=float)
    weight = weight * np.exp(-np.sum(ppath * unit_mm * mua[None, :], axis=1))
    weight[~np.isfinite(weight)] = 0.0
    return weight


def detp_to_source_summed_tpsf(res, cfg, nt, num_sources, hist_tstart_s, hist_tstep_s):
    detp = res.get("detp") if isinstance(res, dict) else None
    if detp is None:
        raise ValueError("No detected photon data found in PMCX result.")

    detid = extract_detector_id_from_detp(detp)
    srcid = extract_source_id_from_detp(detp)
    ppath = extract_partial_path_from_detp(detp)
    if detid is None or srcid is None or ppath is None:
        raise ValueError('Cannot extract detector id, source id, or partial paths. Check cfg["srcid"] = -1.')

    num_detectors = int(np.asarray(cfg["detpos"]).shape[0])
    detid0 = detector_id_to_zero_based(detid, num_detectors)
    srcid0 = srcid.astype(int) - 1
    weights = detected_photon_weights(detp, cfg)
    valid = (detid0 >= 0) & (detid0 < num_detectors) & (srcid0 >= 0) & (srcid0 < num_sources)
    detid0 = detid0[valid]
    srcid0 = srcid0[valid]
    ppath = ppath[valid]
    weights = weights[valid]

    prop = np.asarray(cfg["prop"], dtype=float)
    unit_mm = float(cfg.get("unitinmm", 1.0))
    media_n = prop[1 : 1 + ppath.shape[1], 3]
    tof_ns = np.sum(ppath * unit_mm * media_n[None, :], axis=1) / 299.792458

    tstart_ns = float(hist_tstart_s) * 1e9
    tstep_ns = float(hist_tstep_s) * 1e9
    edges = tstart_ns + np.arange(nt + 1) * tstep_ns
    t_idx = np.searchsorted(edges, tof_ns, side="right") - 1
    t_valid = (t_idx >= 0) & (t_idx < nt)

    source_idx = srcid0[t_valid]
    t_idx = t_idx[t_valid]
    weights = weights[t_valid]

    tpsf = np.zeros((num_sources, nt), dtype=float)
    np.add.at(tpsf, (source_idx, t_idx), weights)

    counts = np.zeros(num_sources, dtype=int)
    np.add.at(counts, source_idx, 1)
    return tpsf, counts, detid, srcid, ppath


def parse_letters(text):
    text = (text or "ALL").strip().upper()
    if text in {"ALL", "*"}:
        return list(ALL_LETTERS)
    letters = []
    for part in text.replace(",", " ").split():
        for ch in part:
            if ch in ALL_LETTERS and ch not in letters:
                letters.append(ch)
    if not letters:
        raise ValueError("No valid uppercase letters selected.")
    return letters


def corrected_emnist_image(raw):
    # EMNIST is stored transposed/rotated relative to normal reading direction.
    return np.fliplr(np.rot90(np.asarray(raw), k=3))


def load_emnist_uppercase_pool(data_root, letters, threshold, seed, train=True):
    try:
        from torchvision.datasets import EMNIST
    except Exception as exc:
        raise RuntimeError("torchvision is required to download/read EMNIST. Install torch and torchvision first.") from exc

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = EMNIST(root=str(root), split="byclass", train=train, download=True)

    selected_label_ids = {10 + ALL_LETTERS.index(letter): letter for letter in letters}
    pool = {letter: [] for letter in letters}
    for idx, target in enumerate(dataset.targets.tolist()):
        target = int(target)
        if target in selected_label_ids:
            pool[selected_label_ids[target]].append(idx)

    rng = random.Random(seed)
    for values in pool.values():
        rng.shuffle(values)

    def get_item(index):
        raw = dataset.data[index].numpy()
        corrected = corrected_emnist_image(raw)
        mask = np.where(corrected > threshold, 0, 1).astype(np.uint8)
        return mask

    return pool, get_item


def balanced_sample_indices(pool, total_images, seed, replace=False):
    letters = list(pool.keys())
    rng = random.Random(seed)
    base = total_images // len(letters)
    remainder = total_images % len(letters)
    remainder_letters = set(rng.sample(letters, remainder))
    selection = []
    per_letter = {}
    for letter in letters:
        count = base + (1 if letter in remainder_letters else 0)
        available = pool[letter]
        if count > len(available) and not replace:
            raise ValueError(
                f"Requested {count} images for letter {letter}, but only {len(available)} are available. "
                "Use fewer total images or add --sample-with-replacement."
            )
        if replace:
            chosen = [rng.choice(available) for _ in range(count)]
        else:
            chosen = available[:count]
        per_letter[letter] = len(chosen)
        selection.extend((letter, idx) for idx in chosen)
    rng.shuffle(selection)
    return selection, per_letter


def run_one_simulation(settings, mask, positions, emnist_meta, raw_dtype=np.float32, log=print):
    started = time.perf_counter()
    cfg, meta, mask_zy = make_multisource_cfg(settings, mask, positions)

    log(
        "Running PMCX multisource: "
        f"letter={emnist_meta['letter']}, dataset_index={emnist_meta['dataset_index']}, "
        f"nphoton/source={settings.nphoton}, total={cfg['nphoton']}"
    )
    res = pmcx.mcxlab(cfg)

    nt = int(np.ceil((meta["hist_tend_s"] - meta["hist_tstart_s"]) / meta["hist_tstep_s"]))
    source_tpsf, _, _, _, _ = detp_to_source_summed_tpsf(
        res,
        cfg,
        nt=nt,
        num_sources=len(positions),
        hist_tstart_s=meta["hist_tstart_s"],
        hist_tstep_s=meta["hist_tstep_s"],
    )

    raw_tpsf = source_tpsf.reshape(3, 3, nt).astype(raw_dtype, copy=False)
    template = mask_zy.astype(np.uint8, copy=False)
    elapsed = time.perf_counter() - started
    log(f"Finished raw TPSF extraction in {elapsed:.1f}s")
    return template, raw_tpsf, elapsed


def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    whole = int(round(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def validate_config(config):
    if int(config["total_images"]) <= 0:
        raise ValueError('USER_CONFIG["total_images"] must be positive')
    if float(config["scan_spacing"]) <= 0:
        raise ValueError('USER_CONFIG["scan_spacing"] must be positive')
    roi_start = int(config["detector_roi_start_1based"])
    roi_end = int(config["detector_roi_end_1based"])
    if roi_start < 1 or roi_end > NUM_PIX or roi_start > roi_end:
        raise ValueError(f"Detector ROI must be within 1..{NUM_PIX}, got {roi_start}..{roi_end}")


def write_timing_summary(path, summary):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_ready(summary), f, ensure_ascii=False, indent=2)


def main():
    config = dict(USER_CONFIG)
    validate_config(config)
    batch_wall_start = datetime.now()
    batch_perf_start = time.perf_counter()

    letters = parse_letters(config["letters"])
    settings = ObjSimSettings(
        output_root=config["output_root"],
        nphoton=int(config["nphoton"]),
        voxel_size_mm=float(config["voxel_size_mm"]),
        slab_thickness_mm=float(config["slab_thickness_mm"]),
        slab_width_mm=float(config["slab_width_mm"]),
        slab_height_mm=float(config["slab_height_mm"]),
        source_y_mm=float(config["scan_center_y"]),
        source_z_mm=float(config["scan_center_z"]),
        detector_center_y_mm=float(config["detector_center_y_mm"]),
        detector_center_z_mm=float(config["detector_center_z_mm"]),
        fov_mm=float(config["fov_mm"]),
        detector_diameter_mm=float(config["detector_diameter_mm"]),
        detector_roi_start_1based=int(config["detector_roi_start_1based"]),
        detector_roi_end_1based=int(config["detector_roi_end_1based"]),
        object_x_mm=float(config["object_x_mm"]),
        object_center_y_mm=float(config["object_center_y_mm"]),
        object_center_z_mm=float(config["object_center_z_mm"]),
        object_size_y_mm=float(config["object_size_y_mm"]),
        object_size_z_mm=float(config["object_size_z_mm"]),
        threshold=int(config["emnist_threshold"]),
        mua=float(config["mua"]),
        mus=float(config["mus"]),
        g=float(config["g"]),
        n=float(config["n"]),
        gpuid=int(config["gpuid"]),
        seed=int(config["seed"]),
    )

    print(f"Loading/downloading EMNIST ByClass uppercase letters to {config['data_root']}", flush=True)
    pool, get_item = load_emnist_uppercase_pool(
        config["data_root"],
        letters,
        int(config["emnist_threshold"]),
        int(config["seed"]),
        train=True,
    )
    selection, per_letter = balanced_sample_indices(
        pool,
        int(config["total_images"]),
        int(config["seed"]),
        bool(config["sample_with_replacement"]),
    )
    print(f"Selected letters: {''.join(letters)}", flush=True)
    print(f"Per-letter counts: {per_letter}", flush=True)

    timestamp = batch_wall_start.strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(config["output_root"]) / f"{timestamp}_emnist_pmcx_3x3_multisource_batch"
    batch_dir.mkdir(parents=True, exist_ok=False)
    positions = scan_source_positions(
        float(config["scan_center_y"]),
        float(config["scan_center_z"]),
        float(config["scan_spacing"]),
    )

    manifest_path = batch_dir / "batch_manifest.csv"
    timing_summary_path = batch_dir / "timing_summary.json"
    simulation_elapsed_values = []
    total_images = len(selection)
    raw_dtype = np.dtype(config["raw_tpsf_dtype"])

    dummy_mask = np.ones((28, 28), dtype=np.uint8)
    _, dummy_meta, dummy_template = make_multisource_cfg(settings, dummy_mask, positions)
    nt = int(np.ceil((dummy_meta["hist_tend_s"] - dummy_meta["hist_tstart_s"]) / dummy_meta["hist_tstep_s"]))
    detector_roi_size = int(config["detector_roi_end_1based"]) - int(config["detector_roi_start_1based"]) + 1
    if dummy_template.shape != (50, 50):
        print(f"Template shape is {dummy_template.shape}, not 50x50. Check object size and voxel size.", flush=True)

    templates_path = batch_dir / f"templates_{dummy_template.shape[0]}x{dummy_template.shape[1]}_uint8.npy"
    raw_tpsf_path = batch_dir / f"raw_tpsf_3x3x{nt}_{raw_dtype.name}.npy"
    templates_mm = open_memmap(
        templates_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_images, dummy_template.shape[0], dummy_template.shape[1]),
    )
    raw_tpsf_mm = open_memmap(
        raw_tpsf_path,
        mode="w+",
        dtype=raw_dtype,
        shape=(total_images, 3, 3, nt),
    )

    with open(batch_dir / "batch_settings.json", "w", encoding="utf-8") as f:
        json.dump(
            json_ready(
                {
                    "user_config": config,
                    "settings": asdict(settings),
                    "letters": letters,
                    "per_letter_counts": per_letter,
                    "positions": positions,
                    "templates_path": str(templates_path),
                    "templates_shape": templates_mm.shape,
                    "raw_tpsf_path": str(raw_tpsf_path),
                    "raw_tpsf_shape": raw_tpsf_mm.shape,
                    "raw_tpsf_note": (
                        f"For each 3 x 3 source position, TPSF is summed over the placed "
                        f"{detector_roi_size} x {detector_roi_size} detector ROI. Shape is N x 3 x 3 x T."
                    ),
                    "detector_roi_size": [detector_roi_size, detector_roi_size],
                    "manifest": str(manifest_path),
                    "timing_summary": str(timing_summary_path),
                    "batch_started_at": batch_wall_start.isoformat(timespec="seconds"),
                }
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(manifest_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "batch_index",
                "matrix_index",
                "letter",
                "dataset_index",
                "seed",
                "status",
                "started_at",
                "finished_at",
                "elapsed_s",
                "elapsed_text",
                "avg_sim_elapsed_s",
                "estimated_remaining_s",
                "estimated_remaining_text",
                "batch_elapsed_s",
                "batch_elapsed_text",
                "error",
            ],
        )
        writer.writeheader()
        for batch_index, (letter, dataset_index) in enumerate(selection, start=1):
            matrix_index = batch_index - 1
            image_seed = int(config["seed"]) + batch_index - 1
            settings.seed = image_seed
            mask = get_item(dataset_index)
            emnist_meta = {
                "split": "byclass",
                "train": True,
                "letter": letter,
                "label_id": 10 + ALL_LETTERS.index(letter),
                "dataset_index": int(dataset_index),
                "orientation_correction": "np.fliplr(np.rot90(raw, k=3))",
                "binary_rule": f"mask = 0 where corrected_pixel > {int(config['emnist_threshold'])}, else 1",
                "batch_index": batch_index,
                "batch_total": total_images,
            }
            image_wall_start = datetime.now()
            image_perf_start = time.perf_counter()
            completed_before = batch_index - 1
            print(
                f"[{batch_index}/{total_images}] Start {letter} index={dataset_index} "
                f"at {image_wall_start.strftime('%Y-%m-%d %H:%M:%S')}",
                flush=True,
            )
            try:
                template, raw_tpsf, sim_elapsed = run_one_simulation(
                    settings,
                    mask,
                    positions,
                    emnist_meta,
                    raw_dtype=raw_dtype,
                    log=lambda s: print(s, flush=True),
                )
                templates_mm[matrix_index] = template
                raw_tpsf_mm[matrix_index] = raw_tpsf
                templates_mm.flush()
                raw_tpsf_mm.flush()
                image_elapsed = time.perf_counter() - image_perf_start
                image_wall_end = datetime.now()
                simulation_elapsed_values.append(float(sim_elapsed))
                avg_sim_elapsed = float(np.mean(simulation_elapsed_values))
                remaining = total_images - batch_index
                estimated_remaining = avg_sim_elapsed * remaining
                batch_elapsed = time.perf_counter() - batch_perf_start
                print(
                    f"[{batch_index}/{total_images}] Done in {format_duration(image_elapsed)}; "
                    f"avg simulation {format_duration(avg_sim_elapsed)}; "
                    f"ETA {format_duration(estimated_remaining)}",
                    flush=True,
                )
                row = {
                    "batch_index": batch_index,
                    "matrix_index": matrix_index,
                    "letter": letter,
                    "dataset_index": dataset_index,
                    "seed": image_seed,
                    "status": "ok",
                    "started_at": image_wall_start.isoformat(timespec="seconds"),
                    "finished_at": image_wall_end.isoformat(timespec="seconds"),
                    "elapsed_s": f"{image_elapsed:.3f}",
                    "elapsed_text": format_duration(image_elapsed),
                    "avg_sim_elapsed_s": f"{avg_sim_elapsed:.3f}",
                    "estimated_remaining_s": f"{estimated_remaining:.3f}",
                    "estimated_remaining_text": format_duration(estimated_remaining),
                    "batch_elapsed_s": f"{batch_elapsed:.3f}",
                    "batch_elapsed_text": format_duration(batch_elapsed),
                    "error": "",
                }
                writer.writerow(row)
                csv_file.flush()
                write_timing_summary(
                    timing_summary_path,
                    {
                        "status": "running" if remaining else "complete",
                        "batch_started_at": batch_wall_start.isoformat(timespec="seconds"),
                        "last_updated_at": image_wall_end.isoformat(timespec="seconds"),
                        "total_images": total_images,
                        "completed_images": batch_index,
                        "failed_images": 0,
                        "remaining_images": remaining,
                        "batch_elapsed_s": batch_elapsed,
                        "batch_elapsed_text": format_duration(batch_elapsed),
                        "last_image_elapsed_s": image_elapsed,
                        "last_image_elapsed_text": format_duration(image_elapsed),
                        "avg_sim_elapsed_s": avg_sim_elapsed,
                        "avg_sim_elapsed_text": format_duration(avg_sim_elapsed),
                        "estimated_remaining_s": estimated_remaining,
                        "estimated_remaining_text": format_duration(estimated_remaining),
                        "templates_path": str(templates_path),
                        "templates_shape": templates_mm.shape,
                        "raw_tpsf_path": str(raw_tpsf_path),
                        "raw_tpsf_shape": raw_tpsf_mm.shape,
                        "manifest": str(manifest_path),
                    },
                )
            except Exception as exc:
                image_elapsed = time.perf_counter() - image_perf_start
                image_wall_end = datetime.now()
                batch_elapsed = time.perf_counter() - batch_perf_start
                writer.writerow(
                    {
                        "batch_index": batch_index,
                        "matrix_index": matrix_index,
                        "letter": letter,
                        "dataset_index": dataset_index,
                        "seed": image_seed,
                        "status": "failed",
                        "started_at": image_wall_start.isoformat(timespec="seconds"),
                        "finished_at": image_wall_end.isoformat(timespec="seconds"),
                        "elapsed_s": f"{image_elapsed:.3f}",
                        "elapsed_text": format_duration(image_elapsed),
                        "avg_sim_elapsed_s": "",
                        "estimated_remaining_s": "",
                        "estimated_remaining_text": "",
                        "batch_elapsed_s": f"{batch_elapsed:.3f}",
                        "batch_elapsed_text": format_duration(batch_elapsed),
                        "error": repr(exc),
                    }
                )
                csv_file.flush()
                write_timing_summary(
                    timing_summary_path,
                    {
                        "status": "failed",
                        "batch_started_at": batch_wall_start.isoformat(timespec="seconds"),
                        "last_updated_at": image_wall_end.isoformat(timespec="seconds"),
                        "total_images": total_images,
                        "completed_images": completed_before,
                        "failed_images": 1,
                        "failed_batch_index": batch_index,
                        "batch_elapsed_s": batch_elapsed,
                        "batch_elapsed_text": format_duration(batch_elapsed),
                        "error": repr(exc),
                        "templates_path": str(templates_path),
                        "templates_shape": templates_mm.shape,
                        "raw_tpsf_path": str(raw_tpsf_path),
                        "raw_tpsf_shape": raw_tpsf_mm.shape,
                        "manifest": str(manifest_path),
                    },
                )
                raise

    batch_elapsed = time.perf_counter() - batch_perf_start
    batch_wall_end = datetime.now()
    templates_mm.flush()
    raw_tpsf_mm.flush()
    write_timing_summary(
        timing_summary_path,
        {
            "status": "complete",
            "batch_started_at": batch_wall_start.isoformat(timespec="seconds"),
            "batch_finished_at": batch_wall_end.isoformat(timespec="seconds"),
            "total_images": total_images,
            "completed_images": total_images,
            "failed_images": 0,
            "batch_elapsed_s": batch_elapsed,
            "batch_elapsed_text": format_duration(batch_elapsed),
            "avg_sim_elapsed_s": float(np.mean(simulation_elapsed_values)) if simulation_elapsed_values else 0.0,
            "avg_sim_elapsed_text": format_duration(np.mean(simulation_elapsed_values)) if simulation_elapsed_values else "0s",
            "templates_path": str(templates_path),
            "templates_shape": templates_mm.shape,
            "raw_tpsf_path": str(raw_tpsf_path),
            "raw_tpsf_shape": raw_tpsf_mm.shape,
            "manifest": str(manifest_path),
        },
    )
    print(f"Batch finished in {format_duration(batch_elapsed)}", flush=True)
    print(f"BATCH_DIR={batch_dir}", flush=True)


if __name__ == "__main__":
    main()
