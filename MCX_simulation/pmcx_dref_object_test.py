"""
PMCX dref test for fixed mus' with and without a central absorbing object.

Geometry:
    Tissue slab: 250 mm x 250 mm x 50 mm.
    x is thickness, source enters from x = 0 toward +x.
    One extra vol=0 layer is appended after the output face, so dref records
    photons escaping from the 50 mm back face into background.

Object:
    A centered 50 mm x 50 mm absorbing square at x = 25 mm.
    By default it is one voxel thick in x and has higher mua, while mus/g/n
    are kept the same as the surrounding slab for each case.
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


def make_volume(
    *,
    voxel_size_mm: float,
    with_object: bool,
    object_size_mm: float,
    object_thickness_mm: float,
) -> np.ndarray:
    nx_tissue = int(round(50.0 / voxel_size_mm))
    ny = int(round(250.0 / voxel_size_mm))
    nz = int(round(250.0 / voxel_size_mm))

    # Extra x layer with label 0 acts as the output-side background.
    vol = np.ones((nx_tissue + 1, ny, nz), dtype=np.uint8)
    vol[-1, :, :] = 0

    if not with_object:
        return vol

    obj_ny = max(1, int(round(object_size_mm / voxel_size_mm)))
    obj_nz = max(1, int(round(object_size_mm / voxel_size_mm)))
    obj_nx = max(1, int(round(object_thickness_mm / voxel_size_mm)))

    x_center = int(round(25.0 / voxel_size_mm))
    x0 = max(0, x_center - obj_nx // 2)
    x1 = min(nx_tissue, x0 + obj_nx)

    y0 = max(0, ny // 2 - obj_ny // 2)
    y1 = min(ny, y0 + obj_ny)
    z0 = max(0, nz // 2 - obj_nz // 2)
    z1 = min(nz, z0 + obj_nz)

    vol[x0:x1, y0:y1, z0:z1] = 0
    return vol


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
    with_object: bool,
    object_mua: float,
    object_size_mm: float,
    object_thickness_mm: float,
    gpuid: int,
    seed: int,
) -> dict:
    vol = make_volume(
        voxel_size_mm=voxel_size_mm,
        with_object=with_object,
        object_size_mm=object_size_mm,
        object_thickness_mm=object_thickness_mm,
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
            (250.0 / 2.0) / voxel_size_mm,
            (250.0 / 2.0) / voxel_size_mm,
        ],
        "srcdir": [1.0, 0.0, 0.0],
        "srctype": "pencil",
        "tstart": 0.0,
        "tend": float(tend_ns) * 1e-9,
        "tstep": float(tstep_ns) * 1e-9,
        "seed": int(seed),
        "gpuid": int(gpuid),
        "autopilot": 1,
        "issaveref": 1,
        "isreflect": 1,
        "outputtype": "flux",
        "debuglevel": "P",
    }


def time_edges_ns(cfg: dict) -> np.ndarray:
    tstart_ns = float(cfg["tstart"]) * 1e9
    tend_ns = float(cfg["tend"]) * 1e9
    tstep_ns = float(cfg["tstep"]) * 1e9
    nt = int(round((tend_ns - tstart_ns) / tstep_ns))
    return tstart_ns + np.arange(nt + 1, dtype=float) * tstep_ns


def dref_back_tpsf(res: dict) -> tuple[np.ndarray, dict]:
    dref = np.asarray(res["dref"], dtype=float)
    if dref.ndim != 4:
        raise RuntimeError(f"Expected dref to be 4D, got shape {dref.shape}")

    x_sums = dref.sum(axis=(1, 2, 3))
    back_tpsf = dref[-1, :, :, :].sum(axis=(0, 1))
    return back_tpsf, {
        "dref_shape": list(dref.shape),
        "dref_sum_all": float(dref.sum()),
        "dref_sum_back_layer": float(back_tpsf.sum()),
        "x_layer_sums": [float(v) for v in x_sums],
    }


def run_case(case: dict, with_object: bool, args: argparse.Namespace) -> dict:
    cfg = make_cfg(
        nphoton=args.nphoton,
        voxel_size_mm=args.voxel_size_mm,
        tstep_ns=args.tstep_ns,
        tend_ns=args.tend_ns,
        mua=args.mua,
        mus=case["mus"],
        g=case["g"],
        n=args.refractive_index,
        with_object=with_object,
        object_mua=args.object_mua,
        object_size_mm=args.object_size_mm,
        object_thickness_mm=args.object_thickness_mm,
        gpuid=args.gpuid,
        seed=args.seed + case["index"] * 1009 + (500_000 if with_object else 0),
    )

    tag = "object" if with_object else "baseline"
    print(
        f"Running {tag}: g={case['g']:.2f}, mus={case['mus']:.6g}, "
        f"mus'={case['mus_prime']:.6g}, photons={args.nphoton}"
    )
    res = pmcx.mcxlab(cfg)
    tpsf, dref_info = dref_back_tpsf(res)

    return {
        "tag": tag,
        "label": f"{tag}, g={case['g']:.2f}, mus={case['mus']:.3g}",
        "g": case["g"],
        "mus": case["mus"],
        "mus_prime": case["mus_prime"],
        "tpsf": tpsf,
        "dref_info": dref_info,
    }


def plot_results(results: list[dict], edges_ns: np.ndarray, outdir: Path) -> tuple[Path, Path, Path]:
    centers_ns = 0.5 * (edges_ns[:-1] + edges_ns[1:])
    baseline = [r for r in results if r["tag"] == "baseline"]
    objected = [r for r in results if r["tag"] == "object"]

    raw_png = outdir / "dref_tpsf_baseline_vs_object_raw.png"
    norm_png = outdir / "dref_tpsf_baseline_vs_object_normalized.png"
    ratio_png = outdir / "dref_total_transmission_ratio.png"

    plt.figure(figsize=(9.0, 5.3))
    for item in results:
        ls = "-" if item["tag"] == "baseline" else "--"
        plt.plot(centers_ns, item["tpsf"], lw=1.7, ls=ls, label=item["label"])
    plt.xlabel("Time of flight (ns)")
    plt.ylabel("Back-face dref summed response")
    plt.title("Back-face dref TPSF, baseline vs absorbing object")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(raw_png, dpi=180)
    plt.close()

    plt.figure(figsize=(9.0, 5.3))
    for item in results:
        y = item["tpsf"].astype(float)
        total = float(np.sum(y))
        if total > 0:
            y = y / total
        ls = "-" if item["tag"] == "baseline" else "--"
        plt.plot(centers_ns, y, lw=1.7, ls=ls, label=item["label"])
    plt.xlabel("Time of flight (ns)")
    plt.ylabel("Normalized back-face dref response")
    plt.title("Normalized dref TPSF shape")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(norm_png, dpi=180)
    plt.close()

    g_values = [r["g"] for r in baseline]
    ratios = []
    for b, o in zip(baseline, objected):
        bsum = float(np.sum(b["tpsf"]))
        osum = float(np.sum(o["tpsf"]))
        ratios.append(osum / bsum if bsum > 0 else np.nan)

    plt.figure(figsize=(7.0, 4.5))
    plt.bar([str(g) for g in g_values], ratios, color="#3b82f6")
    plt.axhline(1.0, color="black", lw=1.0, alpha=0.5)
    plt.xlabel("g")
    plt.ylabel("Object / baseline total dref")
    plt.title("Total transmitted dref ratio")
    plt.ylim(0, max(1.05, np.nanmax(ratios) * 1.1))
    plt.tight_layout()
    plt.savefig(ratio_png, dpi=180)
    plt.close()

    return raw_png, norm_png, ratio_png


def summarize(results: list[dict], edges_ns: np.ndarray, outdir: Path, args: argparse.Namespace) -> Path:
    centers_ns = 0.5 * (edges_ns[:-1] + edges_ns[1:])
    rows = []
    for item in results:
        y = item["tpsf"].astype(float)
        total = float(np.sum(y))
        if total > 0:
            peak_ns = float(centers_ns[int(np.argmax(y))])
            mean_ns = float(np.sum(centers_ns * y) / total)
        else:
            peak_ns = float("nan")
            mean_ns = float("nan")
        rows.append(
            {
                "tag": item["tag"],
                "g": item["g"],
                "mus": item["mus"],
                "mus_prime": item["mus_prime"],
                "tpsf_sum": total,
                "peak_time_ns": peak_ns,
                "mean_time_ns": mean_ns,
                "dref_info": item["dref_info"],
            }
        )

    summary = {
        "geometry_mm": {
            "tissue_slab": [250.0, 250.0, 50.0],
            "extra_output_background_layer_mm": args.voxel_size_mm,
            "absorbing_object": {
                "center_x_mm": 25.0,
                "size_yz_mm": [args.object_size_mm, args.object_size_mm],
                "thickness_x_mm": args.object_thickness_mm,
                "mua_1_per_mm": args.object_mua,
            },
        },
        "voxel_size_mm": args.voxel_size_mm,
        "mua_1_per_mm": args.mua,
        "mus_prime_1_per_mm": args.mus_prime,
        "refractive_index": args.refractive_index,
        "nphoton": args.nphoton,
        "tstep_ns": args.tstep_ns,
        "tend_ns": args.tend_ns,
        "rows": rows,
    }
    path = outdir / "dref_object_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def save_npz(results: list[dict], edges_ns: np.ndarray, outdir: Path) -> Path:
    path = outdir / "dref_object_tpsf.npz"
    np.savez(
        path,
        time_edges_ns=edges_ns,
        time_centers_ns=0.5 * (edges_ns[:-1] + edges_ns[1:]),
        tag=np.asarray([item["tag"] for item in results]),
        g=np.asarray([item["g"] for item in results], dtype=float),
        mus=np.asarray([item["mus"] for item in results], dtype=float),
        mus_prime=np.asarray([item["mus_prime"] for item in results], dtype=float),
        tpsf=np.vstack([item["tpsf"] for item in results]),
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
    parser.add_argument("--object-mua", type=float, default=0.05)
    parser.add_argument("--object-size-mm", type=float, default=50.0)
    parser.add_argument("--object-thickness-mm", type=float, default=2.0)
    parser.add_argument("--gpuid", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--outdir", type=Path, default=Path("dref_object_results"))
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

    results = []
    for case in cases:
        results.append(run_case(case, False, args))
    for case in cases:
        results.append(run_case(case, True, args))

    cfg0 = make_cfg(
        nphoton=args.nphoton,
        voxel_size_mm=args.voxel_size_mm,
        tstep_ns=args.tstep_ns,
        tend_ns=args.tend_ns,
        mua=args.mua,
        mus=cases[0]["mus"],
        g=cases[0]["g"],
        n=args.refractive_index,
        with_object=False,
        object_mua=args.object_mua,
        object_size_mm=args.object_size_mm,
        object_thickness_mm=args.object_thickness_mm,
        gpuid=args.gpuid,
        seed=args.seed,
    )
    edges_ns = time_edges_ns(cfg0)

    raw_png, norm_png, ratio_png = plot_results(results, edges_ns, args.outdir)
    npz_path = save_npz(results, edges_ns, args.outdir)
    summary_path = summarize(results, edges_ns, args.outdir, args)

    print("\nSaved outputs:")
    print(f"  Raw TPSF:        {raw_png}")
    print(f"  Normalized TPSF: {norm_png}")
    print(f"  Total ratio:     {ratio_png}")
    print(f"  Data:            {npz_path}")
    print(f"  Summary:         {summary_path}")
    print("\nPer-case summary:")
    for item in results:
        y = item["tpsf"]
        total = float(np.sum(y))
        print(
            f"  {item['tag']:8s} g={item['g']:.2f}, mus={item['mus']:.6g}: "
            f"dref_sum={total:.6g}, nonzero_back={np.count_nonzero(y)}"
        )


if __name__ == "__main__":
    main()
