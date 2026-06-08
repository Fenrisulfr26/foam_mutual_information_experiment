"""
Compare PMCX time-of-flight curves for media with the same mus' = mus * (1 - g).

Geometry:
    250 mm x 250 mm x 50 mm slab.
    x is the thickness direction, source enters at x = 0 toward +x.

Primary output:
    A TPSF curve formed by summing all photons exiting the back face near x = 50 mm.

Notes:
    PMCX exposes MCX's save-reflectance field as cfg["issaveref"] -> res["dref"].
    In pmcx 0.7.1, dref is useful to inspect but may not contain the transmitted
    back-face signal. This script therefore also places a detector grid across the
    whole back face and reconstructs the TPSF from detected partial path lengths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pmcx


C_MM_PER_NS = 299.792458


def build_back_face_detectors(
    slab_thickness_mm: float,
    slab_width_mm: float,
    slab_height_mm: float,
    voxel_size_mm: float,
    detector_grid: int,
) -> np.ndarray:
    """Create a detector grid covering the output face."""

    nx = int(round(slab_thickness_mm / voxel_size_mm))
    pitch_y_vox = (slab_width_mm / voxel_size_mm) / detector_grid
    pitch_z_vox = (slab_height_mm / voxel_size_mm) / detector_grid

    y = (np.arange(detector_grid, dtype=np.float32) + 0.5) * pitch_y_vox
    z = (np.arange(detector_grid, dtype=np.float32) + 0.5) * pitch_z_vox
    yy, zz = np.meshgrid(y, z, indexing="xy")

    detpos = np.empty((detector_grid * detector_grid, 4), dtype=np.float32)
    detpos[:, 0] = nx
    detpos[:, 1] = yy.ravel()
    detpos[:, 2] = zz.ravel()
    detpos[:, 3] = 0.5 * float(np.hypot(pitch_y_vox, pitch_z_vox)) * 1.01
    return detpos


def make_cfg(
    *,
    nphoton: int,
    voxel_size_mm: float,
    tstep_ns: float,
    tend_ns: float,
    mua: float,
    mus: float,
    g: float,
    n: float,
    gpuid: int,
    seed: int,
    detector_grid: int,
) -> dict:
    slab_thickness_mm = 50.0
    slab_width_mm = 250.0
    slab_height_mm = 250.0

    nx = int(round(slab_thickness_mm / voxel_size_mm))
    ny = int(round(slab_width_mm / voxel_size_mm))
    nz = int(round(slab_height_mm / voxel_size_mm))

    vol = np.ones((nx, ny, nz), dtype=np.uint8)
    detpos = build_back_face_detectors(
        slab_thickness_mm, slab_width_mm, slab_height_mm, voxel_size_mm, detector_grid
    )

    return {
        "nphoton": int(nphoton),
        "vol": vol,
        "unitinmm": float(voxel_size_mm),
        "issrcfrom0": 1,
        "prop": [
            [0.0, 0.0, 1.0, 1.0],
            [float(mua), float(mus), float(g), float(n)],
        ],
        "srcpos": [
            0.0,
            (slab_width_mm / 2.0) / voxel_size_mm,
            (slab_height_mm / 2.0) / voxel_size_mm,
        ],
        "srcdir": [1.0, 0.0, 0.0],
        "srctype": "pencil",
        "detpos": detpos,
        "tstart": 0.0,
        "tend": float(tend_ns) * 1e-9,
        "tstep": float(tstep_ns) * 1e-9,
        "seed": int(seed),
        "gpuid": int(gpuid),
        "autopilot": 1,
        "issavedet": 1,
        "savedetflag": "dpw",
        "issaveref": 1,
        "isreflect": 1,
        "outputtype": "flux",
        "debuglevel": "P",
    }


def extract_detp_tpsf(res: dict, cfg: dict) -> tuple[np.ndarray, int, float]:
    detp = res.get("detp")
    if not isinstance(detp, dict) or "ppath" not in detp:
        raise RuntimeError("No detected photon partial path data found in res['detp'].")

    ppath = np.asarray(detp["ppath"], dtype=float)
    if ppath.ndim == 1:
        ppath = ppath[:, None]

    unit_mm = float(cfg["unitinmm"])
    prop = np.asarray(cfg["prop"], dtype=float)
    mua = prop[1 : 1 + ppath.shape[1], 0]
    refractive_index = prop[1 : 1 + ppath.shape[1], 3]

    if "w0" in detp:
        weight = np.asarray(detp["w0"], dtype=float).reshape(-1)
    else:
        weight = np.ones(ppath.shape[0], dtype=float)

    # MCX ppath is in voxel units. Convert to mm for absorption and time.
    path_mm_by_medium = ppath * unit_mm
    weight = weight * np.exp(-np.sum(path_mm_by_medium * mua[None, :], axis=1))
    tof_ns = np.sum(path_mm_by_medium * refractive_index[None, :], axis=1) / C_MM_PER_NS

    edges_ns = time_edges_ns(cfg)
    hist, _ = np.histogram(tof_ns, bins=edges_ns, weights=weight)
    return hist.astype(float), int(ppath.shape[0]), float(np.sum(weight))


def time_edges_ns(cfg: dict) -> np.ndarray:
    tstart_ns = float(cfg["tstart"]) * 1e9
    tend_ns = float(cfg["tend"]) * 1e9
    tstep_ns = float(cfg["tstep"]) * 1e9
    nt = int(round((tend_ns - tstart_ns) / tstep_ns))
    return tstart_ns + np.arange(nt + 1, dtype=float) * tstep_ns


def extract_dref_back_tpsf(res: dict) -> tuple[np.ndarray | None, dict]:
    dref = res.get("dref")
    if dref is None:
        return None, {"available": False}

    dref = np.asarray(dref)
    if dref.size == 0 or dref.ndim != 4:
        return None, {"available": False, "shape": list(dref.shape)}

    x0 = dref[0, :, :, :].sum(axis=(0, 1))
    xlast = dref[-1, :, :, :].sum(axis=(0, 1))
    all_faces = dref.sum(axis=(0, 1, 2))
    info = {
        "available": True,
        "shape": list(dref.shape),
        "sum_x0": float(np.sum(x0)),
        "sum_xlast": float(np.sum(xlast)),
        "sum_all": float(np.sum(all_faces)),
    }
    return xlast.astype(float), info


def run_case(case: dict, args: argparse.Namespace) -> dict:
    cfg = make_cfg(
        nphoton=args.nphoton,
        voxel_size_mm=args.voxel_size_mm,
        tstep_ns=args.tstep_ns,
        tend_ns=args.tend_ns,
        mua=args.mua,
        mus=case["mus"],
        g=case["g"],
        n=args.refractive_index,
        gpuid=args.gpuid,
        seed=args.seed + case["index"] * 1009,
        detector_grid=args.detector_grid,
    )

    print(
        f"Running g={case['g']:.2f}, mus={case['mus']:.6g}, "
        f"mus'={case['mus_prime']:.6g}, photons={args.nphoton}"
    )
    res = pmcx.mcxlab(cfg)

    det_tpsf, detected_count, detected_weight = extract_detp_tpsf(res, cfg)
    dref_tpsf, dref_info = extract_dref_back_tpsf(res)

    return {
        "label": f"g={case['g']:.2f}, mus={case['mus']:.3g}",
        "g": case["g"],
        "mus": case["mus"],
        "mus_prime": case["mus_prime"],
        "det_tpsf": det_tpsf,
        "dref_back_tpsf": dref_tpsf,
        "detected_count": detected_count,
        "detected_weight": detected_weight,
        "dref_info": dref_info,
        "runtime_ms": float(res.get("stat", {}).get("runtime", np.nan)),
    }


def plot_results(results: list[dict], edges_ns: np.ndarray, outdir: Path) -> tuple[Path, Path]:
    centers_ns = 0.5 * (edges_ns[:-1] + edges_ns[1:])

    raw_png = outdir / "mus_g_same_musp_tpsf_raw.png"
    norm_png = outdir / "mus_g_same_musp_tpsf_normalized.png"

    plt.figure(figsize=(8.5, 5.2))
    for item in results:
        plt.plot(centers_ns, item["det_tpsf"], lw=1.8, label=item["label"])
    plt.xlabel("Time of flight (ns)")
    plt.ylabel("Back-face summed weighted response")
    plt.title("TPSF, same mus' = 1.5 1/mm")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(raw_png, dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    for item in results:
        y = item["det_tpsf"].astype(float)
        total = float(np.sum(y))
        if total > 0:
            y = y / total
        plt.plot(centers_ns, y, lw=1.8, label=item["label"])
    plt.xlabel("Time of flight (ns)")
    plt.ylabel("Normalized back-face response")
    plt.title("Normalized TPSF shape, same mus' = 1.5 1/mm")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(norm_png, dpi=180)
    plt.close()

    return raw_png, norm_png


def summarize(results: list[dict], edges_ns: np.ndarray, outdir: Path, args: argparse.Namespace) -> Path:
    centers_ns = 0.5 * (edges_ns[:-1] + edges_ns[1:])
    rows = []
    for item in results:
        y = item["det_tpsf"]
        total = float(np.sum(y))
        if total > 0:
            mean_ns = float(np.sum(centers_ns * y) / total)
            peak_ns = float(centers_ns[int(np.argmax(y))])
        else:
            mean_ns = float("nan")
            peak_ns = float("nan")
        rows.append(
            {
                "g": item["g"],
                "mus": item["mus"],
                "mus_prime": item["mus_prime"],
                "detected_photons": item["detected_count"],
                "detected_weight_sum": item["detected_weight"],
                "tpsf_sum": total,
                "peak_time_ns": peak_ns,
                "mean_time_ns": mean_ns,
                "dref_info": item["dref_info"],
            }
        )

    summary = {
        "geometry_mm": [250.0, 250.0, 50.0],
        "voxel_size_mm": args.voxel_size_mm,
        "mua_1_per_mm": args.mua,
        "mus_prime_1_per_mm": args.mus_prime,
        "refractive_index": args.refractive_index,
        "nphoton": args.nphoton,
        "tstep_ns": args.tstep_ns,
        "tend_ns": args.tend_ns,
        "rows": rows,
    }
    path = outdir / "mus_g_same_musp_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def save_npz(results: list[dict], edges_ns: np.ndarray, outdir: Path) -> Path:
    path = outdir / "mus_g_same_musp_tpsf.npz"
    np.savez(
        path,
        time_edges_ns=edges_ns,
        time_centers_ns=0.5 * (edges_ns[:-1] + edges_ns[1:]),
        g=np.asarray([item["g"] for item in results], dtype=float),
        mus=np.asarray([item["mus"] for item in results], dtype=float),
        mus_prime=np.asarray([item["mus_prime"] for item in results], dtype=float),
        det_tpsf=np.vstack([item["det_tpsf"] for item in results]),
        detected_count=np.asarray([item["detected_count"] for item in results], dtype=int),
        detected_weight=np.asarray([item["detected_weight"] for item in results], dtype=float),
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nphoton", type=int, default=100_000_000)
    parser.add_argument("--voxel-size-mm", type=float, default=2.0)
    parser.add_argument("--tstep-ns", type=float, default=0.1)
    parser.add_argument("--tend-ns", type=float, default=15.0)
    parser.add_argument("--mua", type=float, default=0.001)
    parser.add_argument("--mus-prime", type=float, default=1.5)
    parser.add_argument("--refractive-index", type=float, default=1.05)
    parser.add_argument("--g-values", type=float, nargs="+", default=[0.0, 0.5, 0.8, 0.9])
    parser.add_argument("--detector-grid", type=int, default=50)
    parser.add_argument("--gpuid", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--outdir", type=Path, default=Path("mus_g_tpsf_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cases = []
    for index, g in enumerate(args.g_values):
        if not (0.0 <= g < 1.0):
            raise ValueError(f"g must be in [0, 1), got {g}")
        cases.append(
            {
                "index": index,
                "g": float(g),
                "mus": float(args.mus_prime / (1.0 - g)),
                "mus_prime": float(args.mus_prime),
            }
        )

    results = [run_case(case, args) for case in cases]
    cfg0 = make_cfg(
        nphoton=args.nphoton,
        voxel_size_mm=args.voxel_size_mm,
        tstep_ns=args.tstep_ns,
        tend_ns=args.tend_ns,
        mua=args.mua,
        mus=cases[0]["mus"],
        g=cases[0]["g"],
        n=args.refractive_index,
        gpuid=args.gpuid,
        seed=args.seed,
        detector_grid=args.detector_grid,
    )
    edges_ns = time_edges_ns(cfg0)

    raw_png, norm_png = plot_results(results, edges_ns, args.outdir)
    npz_path = save_npz(results, edges_ns, args.outdir)
    summary_path = summarize(results, edges_ns, args.outdir, args)

    print("\nSaved outputs:")
    print(f"  Raw TPSF:        {raw_png}")
    print(f"  Normalized TPSF: {norm_png}")
    print(f"  Data:            {npz_path}")
    print(f"  Summary:         {summary_path}")
    print("\nPer-case summary:")
    for item in results:
        y = item["det_tpsf"]
        total = float(np.sum(y))
        peak = float(0.5 * (edges_ns[:-1] + edges_ns[1:])[int(np.argmax(y))]) if total > 0 else float("nan")
        print(
            f"  {item['label']}: detected={item['detected_count']}, "
            f"weight_sum={item['detected_weight']:.6g}, tpsf_sum={total:.6g}, peak={peak:.3f} ns, "
            f"dref_back_sum={item['dref_info'].get('sum_xlast', float('nan')):.6g}"
        )


if __name__ == "__main__":
    main()
