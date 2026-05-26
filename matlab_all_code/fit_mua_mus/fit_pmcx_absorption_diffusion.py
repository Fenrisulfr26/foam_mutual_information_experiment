#!/usr/bin/env python
"""
Fit absorption coefficient mua and diffusion coefficient D using PMCX only.

Forward model:
  - PMCX/MCX transient simulation
  - no detector is defined
  - issaveref=1 saves boundary reflectance/transmittance as dref
  - the fitted simulated curve is the outgoing +z face center pixel dref(t)
  - dref(t) is convolved with the measured IRF before comparison

Run with your diffusion environment:
  D:\\codings\\anaconda\\envs\\diffusion\\python.exe fit_pmcx_absorption_diffusion.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pmcx
import scipy.io
import scipy.optimize


def load_mat_hist(path: Path, variable: str = "hist") -> np.ndarray:
    """Load hist from MATLAB v7/v7.2 or v7.3 .mat accurately in Python."""
    try:
        mat = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
        if variable not in mat:
            keys = sorted(k for k in mat if not k.startswith("__"))
            raise KeyError(f"{variable!r} not found in {path}. Available variables: {keys}")
        arr = np.asarray(mat[variable], dtype=np.float64)
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            if variable not in f:
                raise KeyError(f"{variable!r} not found in {path}. Available variables: {list(f.keys())}")
            arr = np.asarray(f[variable], dtype=np.float64)
        # h5py exposes MATLAB column-major arrays reversed for common v7.3 files.
        arr = np.transpose(arr)

    if arr.ndim != 3:
        raise ValueError(f"{path}::{variable} must be 3-D, got {arr.shape}")
    if arr.shape[0] == 32 and arr.shape[1] == 32:
        return arr
    if arr.shape[-2:] == (32, 32):
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"{path}::{variable} has unexpected shape {arr.shape}; expected 32x32xT")


def pixel_curve_1based(hist: np.ndarray, y: int, x: int) -> np.ndarray:
    curve = np.asarray(hist[y - 1, x - 1, :], dtype=np.float64).reshape(-1)
    curve[~np.isfinite(curve)] = 0
    curve[curve < 0] = 0
    return curve


def normalize_area(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y[~np.isfinite(y)] = 0
    y[y < 0] = 0
    s = float(y.sum())
    return y / s if s > 0 else y


def shift_curve(y: np.ndarray, shift_bins: float) -> np.ndarray:
    x = np.arange(y.size, dtype=np.float64)
    return np.interp(x - shift_bins, x, y, left=0.0, right=0.0)


def make_padded_volume(sample_size_mm: tuple[float, float, float], voxel_size_mm: float) -> tuple[np.ndarray, np.ndarray]:
    sample_vox = np.rint(np.asarray(sample_size_mm, dtype=np.float64) / voxel_size_mm).astype(int)
    if np.any(sample_vox <= 0):
        raise ValueError(f"Invalid sample voxel size: {sample_vox}")
    if np.any(np.abs(sample_vox * voxel_size_mm - np.asarray(sample_size_mm)) > 1e-9):
        raise ValueError("sample_size_mm must be exactly divisible by voxel_size_mm.")

    vol = np.zeros(tuple((sample_vox + 2).tolist()), dtype=np.uint8)
    vol[1:-1, 1:-1, 1:-1] = 1
    return vol, sample_vox


def make_base_cfg(args: argparse.Namespace, n_bins: int) -> tuple[dict[str, Any], tuple[int, int, int]]:
    vol, sample_vox = make_padded_volume(args.sample_size_mm, args.voxel_size_mm)

    # Padded volume layout:
    #   z=0      background layer above the slab
    #   z=1..Nz  tissue slab, thickness 50 mm
    #   z=Nz+1   background layer below the slab
    #
    # Source is centered in x/y at the top tissue surface and travels along +z.
    center_x = int(sample_vox[0] // 2 + 1)
    center_y = int(sample_vox[1] // 2 + 1)
    source_z = 1

    cfg: dict[str, Any] = {
        "nphoton": int(args.nphoton),
        "vol": vol,
        "unitinmm": float(args.voxel_size_mm),
        "srcpos": [center_x, center_y, source_z],
        "srcdir": [0.0, 0.0, 1.0],
        "prop": np.asarray(
            [[0.0, 0.0, 1.0, 1.0], [args.initial_mua, 1.0, args.g, args.n]],
            dtype=np.float32,
        ),
        "tstart": 0.0,
        "tend": float(n_bins * args.time_bin_s),
        "tstep": float(args.time_bin_s),
        "isreflect": 0,
        "issaveref": 1,
        "bc": "aaaaaa",
        "gpuid": int(args.gpuid),
        "autopilot": 1,
        "maxdetphoton": 0,
        "seed": int(args.seed),
    }
    center_index = (center_x, center_y, int(sample_vox[2] + 1))
    return cfg, center_index


def mua_d_to_mus(mua: float, diffusion: float, g: float) -> tuple[float, float]:
    musp = 1.0 / (3.0 * diffusion) - mua
    if not np.isfinite(musp) or musp <= 0:
        raise ValueError(f"invalid musp={musp:g} from mua={mua:g}, D={diffusion:g}")
    mus = musp / (1.0 - g)
    return musp, mus


def run_pmcx_dref_center(
    base_cfg: dict[str, Any],
    center_index: tuple[int, int, int],
    mua: float,
    diffusion: float,
    g: float,
) -> tuple[np.ndarray, dict[str, float]]:
    musp, mus = mua_d_to_mus(mua, diffusion, g)

    cfg = dict(base_cfg)
    cfg["prop"] = np.array(base_cfg["prop"], dtype=np.float32, copy=True)
    cfg["prop"][1, :] = [mua, mus, g, cfg["prop"][1, 3]]

    res = pmcx.mcxlab(cfg)
    if "dref" not in res:
        raise RuntimeError("PMCX result does not contain dref. Make sure issaveref=1 is supported.")

    dref = np.asarray(res["dref"], dtype=np.float64)
    if dref.ndim != 4:
        raise RuntimeError(f"Expected dref as 4-D [x,y,z,t], got shape {dref.shape}")

    cx, cy, cz_bottom_bg = center_index
    curve = dref[cx, cy, cz_bottom_bg, :].reshape(-1)
    curve[~np.isfinite(curve)] = 0
    curve[curve < 0] = 0

    meta = {
        "musp_1_per_mm": float(musp),
        "mus_1_per_mm": float(mus),
        "dref_sum": float(curve.sum()),
    }
    return curve, meta


def model_curve(
    base_cfg: dict[str, Any],
    center_index: tuple[int, int, int],
    mua: float,
    diffusion: float,
    g: float,
    irf_curve: np.ndarray,
    target_curve: np.ndarray,
    shift_bins: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    raw, meta = run_pmcx_dref_center(base_cfg, center_index, mua, diffusion, g)
    if raw.sum() <= 0:
        raise RuntimeError("Center-pixel dref curve is zero; increase nphoton or check source/face geometry.")

    conv = np.convolve(normalize_area(raw), normalize_area(irf_curve), mode="same")
    conv = normalize_area(shift_curve(normalize_area(conv), shift_bins))

    a = np.column_stack([conv, np.ones(conv.size)])
    coeff, _ = scipy.optimize.nnls(a, target_curve)
    fitted = coeff[0] * conv + coeff[1]
    meta["scale"] = float(coeff[0])
    meta["background"] = float(coeff[1])
    return fitted, raw, meta


def unpack(z: np.ndarray, fit_shift: bool) -> tuple[float, float, float]:
    mua = float(np.exp(z[0]))
    diffusion = float(np.exp(z[1]))
    shift = float(z[2]) if fit_shift else 0.0
    return mua, diffusion, shift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=Path("data/50mm_60d_foam/hist_5us_1e+06_38.5deg_obj.mat"))
    parser.add_argument("--irf", type=Path, default=Path("data/IRF.mat"))
    parser.add_argument("--pixel-yx", type=int, nargs=2, default=(16, 16), help="MATLAB 1-based y x pixel")
    parser.add_argument("--sample-size-mm", type=float, nargs=3, default=(250.0, 250.0, 50.0))
    parser.add_argument("--voxel-size-mm", type=float, default=2.5)
    parser.add_argument("--n", type=float, default=1.48)
    parser.add_argument("--g", type=float, default=0.90)
    parser.add_argument("--time-bin-s", type=float, default=55e-12)
    parser.add_argument("--nphoton", type=float, default=1e6)
    parser.add_argument("--gpuid", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1648335518)
    parser.add_argument("--initial-mua", type=float, default=0.005)
    parser.add_argument("--initial-d", type=float, default=2.0)
    parser.add_argument("--bounds-mua", type=float, nargs=2, default=(1e-5, 0.20))
    parser.add_argument("--bounds-d", type=float, nargs=2, default=(0.01, 100.0))
    parser.add_argument("--fit-shift", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shift-bounds", type=float, nargs=2, default=(-60.0, 60.0))
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--output-prefix", type=Path, default=Path("fit_pmcx_absorption_diffusion_result"))
    parser.add_argument("--check-data-only", action="store_true")
    args = parser.parse_args()

    target_hist = load_mat_hist(args.target)
    irf_hist = load_mat_hist(args.irf)
    target_curve = pixel_curve_1based(target_hist, args.pixel_yx[0], args.pixel_yx[1])
    irf_curve = pixel_curve_1based(irf_hist, args.pixel_yx[0], args.pixel_yx[1])
    if target_curve.size != irf_curve.size:
        raise ValueError(f"target bins {target_curve.size} != IRF bins {irf_curve.size}")

    print(f"target hist shape: {target_hist.shape}, file: {args.target}")
    print(f"IRF hist shape:    {irf_hist.shape}, file: {args.irf}")
    print(f"using MATLAB pixel y,x = {tuple(args.pixel_yx)}, bins = {target_curve.size}")

    if args.check_data_only:
        return

    base_cfg, center_index = make_base_cfg(args, target_curve.size)
    print(f"PMCX source position: {base_cfg['srcpos']}, source direction: {base_cfg['srcdir']}")
    print(f"PMCX dref output pixel [x,y,z] = {center_index} on outgoing +z face")
    print(f"GPU info: {pmcx.gpuinfo()}")

    cache: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}

    def objective(z: np.ndarray) -> float:
        mua, diffusion, shift = unpack(z, args.fit_shift)
        key = tuple(np.round(z, 10))

        if not (args.bounds_mua[0] <= mua <= args.bounds_mua[1]):
            return 1e12 + 1e8 * abs(mua)
        if not (args.bounds_d[0] <= diffusion <= args.bounds_d[1]):
            return 1e12 + 1e8 * abs(diffusion)
        if args.fit_shift and not (args.shift_bounds[0] <= shift <= args.shift_bounds[1]):
            return 1e12 + 1e6 * abs(shift)

        try:
            if key not in cache:
                cache[key] = model_curve(base_cfg, center_index, mua, diffusion, args.g, irf_curve, target_curve, shift)
            fitted = cache[key][0]
        except Exception as exc:
            print(f"failed: mua={mua:.6g}, D={diffusion:.6g}, shift={shift:.3f}: {exc}")
            return 1e12

        y = normalize_area(target_curve)
        m = normalize_area(fitted)
        residual = (m - y) / np.sqrt(np.maximum(y, float(y.max()) * 0.01) + np.finfo(float).eps)
        value = float(np.sum(residual * residual))
        print(f"mua={mua:.6g}, D={diffusion:.6g}, shift={shift:.3f}, obj={value:.6g}")
        return value

    z0 = [math.log(args.initial_mua), math.log(args.initial_d)]
    if args.fit_shift:
        z0.append(0.0)

    result = scipy.optimize.minimize(
        objective,
        np.asarray(z0, dtype=np.float64),
        method="Nelder-Mead",
        options={"maxiter": args.maxiter, "xatol": 1e-3, "fatol": 1e-3, "disp": True},
    )

    mua_best, d_best, shift_best = unpack(result.x, args.fit_shift)
    fitted_best, raw_best, meta_best = model_curve(
        base_cfg, center_index, mua_best, d_best, args.g, irf_curve, target_curve, shift_best
    )

    fit_result = {
        "mua_1_per_mm": float(mua_best),
        "D_mm": float(d_best),
        "musp_1_per_mm": meta_best["musp_1_per_mm"],
        "mus_1_per_mm": meta_best["mus_1_per_mm"],
        "g": float(args.g),
        "n": float(args.n),
        "time_shift_bins": float(shift_best),
        "time_shift_ns": float(shift_best * args.time_bin_s * 1e9),
        "dref_sum": meta_best["dref_sum"],
        "scale": meta_best["scale"],
        "background": meta_best["background"],
        "objective": float(result.fun),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "source_position_voxel": base_cfg["srcpos"],
        "dref_center_pixel_xyz": center_index,
    }

    args.output_prefix.with_suffix(".json").write_text(json.dumps(fit_result, indent=2), encoding="utf-8")
    np.savez(
        args.output_prefix.with_suffix(".npz"),
        target_curve=target_curve,
        irf_curve=irf_curve,
        raw_dref_curve=raw_best,
        fitted_curve=fitted_best,
        time_axis_s=(np.arange(target_curve.size) + 0.5) * args.time_bin_s,
        fit_result_json=json.dumps(fit_result),
    )

    try:
        import matplotlib.pyplot as plt

        t_ns = (np.arange(target_curve.size) + 0.5) * args.time_bin_s * 1e9
        fig, ax = plt.subplots(2, 1, figsize=(8, 7), constrained_layout=True)
        ax[0].plot(t_ns, target_curve, "k-", label="measured")
        ax[0].plot(t_ns, fitted_best, "r-", label="PMCX dref + IRF")
        ax[0].set_xlabel("Time (ns)")
        ax[0].set_ylabel("Counts")
        ax[0].grid(True)
        ax[0].legend()
        ax[1].plot(t_ns, normalize_area(target_curve), "k-", label="measured")
        ax[1].plot(t_ns, normalize_area(fitted_best), "r-", label="fit")
        ax[1].plot(t_ns, normalize_area(irf_curve), "b--", label="IRF")
        ax[1].set_xlabel("Time (ns)")
        ax[1].set_ylabel("Area normalized")
        ax[1].grid(True)
        ax[1].legend()
        fig.suptitle(f"mua={mua_best:.4g} 1/mm, D={d_best:.4g} mm")
        fig.savefig(args.output_prefix.with_suffix(".png"), dpi=200)
    except Exception as exc:
        print(f"plot skipped: {exc}")

    print("\nBest fit:")
    print(json.dumps(fit_result, indent=2))


if __name__ == "__main__":
    main()
