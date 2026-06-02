"""
Fit PMCX simulation to experimental data (fit mua/mus only, g fixed).

Workflow:
1. Load experiment MAT data.
2. Load IRF MAT data and use IRF histogram curve at [16,16] (MATLAB indexing).
3. Run PMCX with candidate mua/mus.
4. Build per-pixel time histogram, convolve IRF for all pixels.
5. Global-max normalize convolved simulation data.
6. Optimize mua/mus by minimizing MSE to experiment.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import differential_evolution, minimize
from scipy.signal import convolve
from scipy.optimize import OptimizeResult
from scipy.optimize import curve_fit
from my_display_hist import my_display_hist, display_hist_mat, compare_hist

import pmcx_sim

# 100 density foam data
# EXPERIMENT_MAT_PATH = (
#     r"F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\data"
#     r"\3x3_grid_scan_20260518_145708_deg_10_exp_2us_frames_100000_avg_1\hist_2us_100000_avg1_point05_center_cal.mat"
# )

# EXPERIMENT_MAT_PATH = (
#     r"F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\data"
#     r"\3x3_grid_scan_20260518_145337_deg_6_exp_2us_frames_100000_avg_1\hist_2us_100000_avg1_point05_center_cal.mat"
# )

# 80 density data 3x3_grid_scan_20260520_202613_deg_neg3_exp_2us_frames_100000_avg_20
EXPERIMENT_MAT_PATH = (
    r"F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\data"
    r"\3x3_grid_scan_20260520_202613_deg_neg3_exp_2us_frames_100000_avg_20\hist_2us_100000_avg20_point05_center_cal.mat"
)

# 60 density foam data
# EXPERIMENT_MAT_PATH = (
#     r"F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\data"
#     r"\3x3_grid_scan_20260518_210020_deg_neg2_exp_2us_frames_100000_avg_1\hist_2us_100000_avg1_point05_center_cal.mat"
# )

IRF_MAT_PATH = r"F:\OneDrive\foam_imaging_project\experiment_setup\matlab_all_code\data\IRF.mat"
META_PATH = "foam_slab_32x32_pmcx_result_meta.npz"

NUM_PIX = 32
NPHOTON = 50_000_000
VOXEL_SIZE_MM = 1.0
EXPERIMENT_POINT_INDEX = 5  # MATLAB-style point index: 1..9, center point is 5
BASELINE_BINS = 20
BASELINE_SIDE = "head"  # This dataset has the signal peak near the end.
MAX_TIME_SHIFT_BINS = 226
USE_CIRCULAR_TIME_SHIFT = True
DISPLAY_GATE_HALF_WIDTH_BINS = 20
DETECTOR_DIAMETER_MM = 1
LOSS_SPACE_WEIGHT = 0.75
LOSS_TIME_WEIGHT = 0.25
FIT_REFRACTIVE_INDEX = False
N_BOUNDS = (1.0, 2.2)
LOSS_FUNC = "composite"  # "composite" or "simple_rmse"



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
        # If 2D, treat as [detector,time] and pick center detector.
        center = data.shape[0] // 2
        curve = data[center, :].reshape(-1)
    else:
        # MATLAB [16,16] -> Python [15,15]
        y = max(0, min(data.shape[0] - 1, matlab_index[0] - 1))
        z = max(0, min(data.shape[1] - 1, matlab_index[1] - 1))
        curve = data[y, z, :].reshape(-1)

    curve = curve - np.median(curve[: min(BASELINE_BINS, curve.size)])
    curve[curve < 0] = 0
    s = np.sum(curve)
    if s <= 0:
        raise ValueError("IRF curve sum is zero, cannot normalize")
    curve = curve / curve.max() # normalize to sum=1 for convolution
    return curve, var

def extract_sim_cube(res, cfg, num_pix=NUM_PIX):
    # get 32x32x227 data from sim results
    detp = None
    for key in ["detp", "detphoton", "detphotons"]:
        if key in res:
            detp = res[key]
            break
    if detp is None:
        raise ValueError("No detected photon data found in PMCX result")

    detid = pmcx_sim.extract_detector_id_from_detp(detp)
    ppath = pmcx_sim.extract_partial_path_from_detp(detp)
    if detid is None or ppath is None:
        raise ValueError("Cannot extract detid/ppath from PMCX result")

    photon_weight = pmcx_sim.detected_photon_weights(detp, cfg)

    detid0 = pmcx_sim.detector_id_to_zero_based(detid, num_pix * num_pix)
    valid = (detid0 >= 0) & (detid0 < num_pix * num_pix)
    detid0 = detid0[valid]
    ppath = ppath[valid]
    photon_weight = photon_weight[valid]

    unit_mm = float(cfg.get("unitinmm", 1.0))
    prop = np.asarray(cfg["prop"], dtype=float)
    if prop.ndim != 2 or prop.shape[1] < 4:
        raise ValueError("cfg['prop'] format invalid")

    media_n = prop[1 : 1 + ppath.shape[1], 3]
    c_mm_per_ns = 299.792458
    tof_ns = np.sum(ppath * unit_mm * media_n[None, :], axis=1) / c_mm_per_ns

    tstart_ns = float(cfg["tstart"]) * 1e9
    tend_ns = float(cfg["tend"]) * 1e9
    tstep_ns = float(cfg["tstep"]) * 1e9
    nt = int(np.ceil((tend_ns - tstart_ns) / tstep_ns))
    
    # Truncate to 227 bins to match experiment
    # Experiment data size is 32x32x227
    nt = 227
    
    edges = tstart_ns + np.arange(nt + 1) * tstep_ns

    y_idx = detid0 % num_pix
    z_idx = detid0 // num_pix
    t_idx = np.searchsorted(edges, tof_ns, side="right") - 1
    t_valid = (t_idx >= 0) & (t_idx < nt)

    # Directly map Z and Y to the camera perspective (from -X axis view)
    z_img = num_pix - 1 - z_idx[t_valid]
    y_img = num_pix - 1 - y_idx[t_valid]
    t_idx = t_idx[t_valid]

    cube = np.zeros((num_pix, num_pix, nt), dtype=float)
    np.add.at(cube, (z_img, y_img, t_idx), photon_weight[t_valid])
    return cube

# align_to_camera_view function was removed as we now generate the simulation cube directly in camera view in extract_sim_cube.

def sum_normalize(arr, eps=1e-12):
    arr = np.asarray(arr, dtype=float)
    s = np.nansum(arr)
    return arr / max(s, eps)


def weighted_3d_rmse(sim, exp, eps=1e-12):
    sim = sum_normalize(sim, eps)
    exp = sum_normalize(exp, eps)

    # 权重集中在实验有信号的地方，避免大量背景 0 主导 loss
    w = np.sqrt(exp + eps)
    w = w / np.mean(w)

    return np.sqrt(np.mean(w * (sim - exp) ** 2))

def _fold_last_axis_to_period(arr, period_bins):
    """
    Fold the last axis of arr into one periodic TCSPC window.
    For example, bin 227 folds to bin 0 if period_bins=227.
    """
    arr = np.asarray(arr, dtype=float)
    out = np.zeros(arr.shape[:-1] + (period_bins,), dtype=float)

    arr2 = arr.reshape(-1, arr.shape[-1])
    out2 = out.reshape(-1, period_bins)

    for k in range(arr.shape[-1]):
        out2[:, k % period_bins] += arr2[:, k]

    return out

def compare_results():
    results = np.load('F:/OneDrive/foam_imaging_project/experiment_setup/MCX_simulation/fitting_results.npz')
    compare_hist(results['sim_compare'],results["exp_compare"])


def _fold_1d_to_period(x, period_bins):
    """
    Fold a 1D IRF or TPSF into one periodic TCSPC window.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    out = np.zeros(period_bins, dtype=float)

    for k, val in enumerate(x):
        out[k % period_bins] += val

    return out


def convolve_irf_all_pixels_tcspc(
    cube,
    irf,
    period_bins=None,
    normalize_irf=True,
    irf_zero_idx=0,
    fold_cube=True,
):
    """
    Periodic / circular convolution for TCSPC data.

    This is appropriate when the measured TCSPC histogram is sampled modulo
    the laser repetition period.

    Parameters
    ----------
    cube : ndarray
        Simulated TPSF data. Shape can be (ny, nz, nt_sim).

    irf : ndarray
        Measured IRF. It can be shorter, equal to, or longer than one period.

    period_bins : int or None
        Number of TCSPC bins in one laser period.
        If None, use cube.shape[-1].

    normalize_irf : bool
        If True, normalize IRF so sum(irf) = 1.
        This preserves total photon counts after circular convolution.

    irf_zero_idx : int
        Index in the IRF corresponding to time zero.
        If your IRF already starts at time zero, use 0.
        If the IRF peak should be treated as time zero, use np.argmax(irf).

    fold_cube : bool
        If True, fold the simulated cube into one TCSPC period before convolution.
        This is important if cube has a longer time window than period_bins.

    Returns
    -------
    out : ndarray
        Periodic convolution result, shape (ny, nz, period_bins).
    """

    cube = np.asarray(cube, dtype=float)
    irf = np.asarray(irf, dtype=float).reshape(-1)

    if cube.ndim != 3:
        raise ValueError("cube must have shape (ny, nz, nt)")

    if irf.size == 0:
        raise ValueError("irf is empty")

    if period_bins is None:
        period_bins = cube.shape[-1]

    period_bins = int(period_bins)

    if period_bins <= 0:
        raise ValueError("period_bins must be positive")

    # Fold simulated TPSF into one TCSPC period.
    if fold_cube:
        cube_periodic = _fold_last_axis_to_period(cube, period_bins)
    else:
        if cube.shape[-1] != period_bins:
            raise ValueError(
                "If fold_cube=False, cube.shape[-1] must equal period_bins"
            )
        cube_periodic = cube

    # Fold IRF into one period as well.
    irf_periodic = _fold_1d_to_period(irf, period_bins)

    # Align IRF time zero to bin 0.
    # Example: if IRF peak is time zero, set irf_zero_idx=np.argmax(irf).
    if irf_zero_idx != 0:
        irf_periodic = np.roll(irf_periodic, -int(irf_zero_idx))

    if normalize_irf:
        s = np.sum(irf_periodic)
        if not np.isfinite(s) or s <= 0:
            raise ValueError("IRF sum must be positive and finite")
        irf_periodic = irf_periodic / s

    # Circular convolution along the time axis:
    # out[t] = sum_tau cube[tau] * irf[(t - tau) mod period_bins]
    X = np.fft.rfft(cube_periodic, n=period_bins, axis=-1)
    H = np.fft.rfft(irf_periodic, n=period_bins)

    out = np.fft.irfft(
        X * H.reshape(1, 1, -1),
        n=period_bins,
        axis=-1,
    )

    # Remove tiny numerical imaginary/negative artifacts from FFT roundoff.
    out = np.real(out)
    tiny = 1e-12 * max(1.0, np.nanmax(np.abs(out)))
    out[np.abs(out) < tiny] = 0.0

    return out

def global_max_normalize(arr):
    m = np.nanmax(arr)
    if m <= 0:
        return arr
    return arr / m

class PMCXFitter:
    def __init__(self):
        self.meta = np.load(META_PATH)
        self.default_mua = float(self.meta["mua_mm_inv"])
        self.default_mus = float(self.meta["mus_mm_inv"])
        self.fixed_g = float(self.meta["g"])
        self.fixed_n = float(self.meta["n"])

        self.exp_raw, self.exp_var = load_experiment_data(EXPERIMENT_MAT_PATH)
        self.exp_loaded_shape = self.exp_raw.shape
        
        if self.exp_raw.ndim == 4:
            point_idx0 = EXPERIMENT_POINT_INDEX - 1
            if point_idx0 < 0 or point_idx0 >= self.exp_raw.shape[3]:
                raise ValueError(
                    f"EXPERIMENT_POINT_INDEX={EXPERIMENT_POINT_INDEX} is outside "
                    f"1..{self.exp_raw.shape[3]}"
                )
            self.exp_raw = np.squeeze(self.exp_raw[:, :, :, point_idx0])
            
        self.irf, self.irf_var = load_irf_curve(IRF_MAT_PATH, matlab_index=(16, 16))
        
        self.exp_compare = self.exp_raw
        self.exp_map2d = self.exp_raw.sum(axis = 2)
        self.exp_compare = global_max_normalize(self.exp_compare) # normalize
        self.exp_peak_bin = int(np.nanargmax(np.sum(self.exp_compare, axis=(0, 1))))
        self.exp_map2d = global_max_normalize(self.exp_map2d)

    def run_once(self, mua, mus, n=None):
        if n is None:
            n = self.fixed_n

        cfg, _ = pmcx_sim.make_foam_slab_cfg(
            nphoton=NPHOTON,
            voxel_size_mm=VOXEL_SIZE_MM,
            detector_diameter_mm=DETECTOR_DIAMETER_MM,
            mua=float(mua),
            mus=float(mus),
            g=self.fixed_g,
            n=float(n),
            gpuid=1,
        )

        # Fitting only needs detected photon histories. Avoid saving the full
        # 4D flux/fluence volume, which is several GB for this grid.
        cfg["issave2pt"] = 0
        cfg.pop("outputtype", None)
        cfg.pop("debuglevel", None)

        res = pmcx_sim.pmcx.mcxlab(cfg)
        cube = extract_sim_cube(res, cfg, num_pix=NUM_PIX)
        cube_conv = convolve_irf_all_pixels_tcspc(cube, self.irf)

        # Since cube_conv is already in camera view, we assign directly
        sim_compare = cube_conv
        sim_compare = global_max_normalize(sim_compare)
        
        # get 2D map and then normalize
        sim_map2d = np.sum(sim_compare, axis=2)
        sim_map2d = global_max_normalize(sim_map2d)
        
        return sim_compare, sim_map2d

    def compare_loss(self, sim_compare):
        sim = np.asarray(sim_compare, dtype=float)
        exp = np.asarray(self.exp_compare, dtype=float)
    
        if LOSS_FUNC == "simple_rmse":
            return np.sqrt(np.mean((sim - exp) ** 2))

        loss_3d = weighted_3d_rmse(sim, exp)
    
        # 额外保证 2D 空间积分图相似
        sim_map = sum_normalize(np.sum(sim, axis=2))
        exp_map = sum_normalize(np.sum(exp, axis=2))
        loss_map = np.sqrt(np.mean((sim_map - exp_map) ** 2))
    
        # 额外保证总体时间曲线相似
        sim_time = sum_normalize(np.sum(sim, axis=(0, 1)))
        exp_time = sum_normalize(np.sum(exp, axis=(0, 1)))
        loss_time = np.sqrt(np.mean((sim_time - exp_time) ** 2))
    
        return 0.60 * loss_3d + 0.25 * loss_map + 0.15 * loss_time
# =============================================================================
#     def fit(self, p0=None, max_nfev=80):
#         """
#         Fit mua and mus using scipy.optimize.curve_fit.
#     
#         This assumes self.run_once(mua, mus) returns sim_compare,
#         whose shape matches self.exp_compare.
#         """
#     
#         # 参数范围：和你原来的 differential_evolution bounds 一致
#         lower = np.array([1e-5, 0.1], dtype=float)   # mua_min, mus_min
#         upper = np.array([0.1, 10.0], dtype=float)   # mua_max, mus_max
#     
#         if p0 is None:
#             p0 = [self.default_mua,self.default_mus]
#         p0 = np.asarray(p0, dtype=float)
#     
#         # 保证初始值在边界内
#         p0 = np.clip(p0, lower, upper)
#     
#         # 实验数据拉平成一维
#         y_full = np.asarray(self.exp_compare, dtype=float).reshape(-1)
#     
#         # 只拟合有限值，排除 nan / inf
#         valid = np.isfinite(y_full)
#         ydata = y_full[valid]
#     
#         if ydata.size == 0:
#             raise ValueError("self.exp_compare has no finite values.")
#     
#         # curve_fit 必须要 xdata，但你的模型其实不依赖 x
#         # 所以这里用一个 dummy xdata
#         xdata = np.arange(ydata.size, dtype=float)
#     
#         call_counter = {"n": 0}
#     
#         def model_func(x_dummy, mua, mus):
#             """
#             curve_fit calls this function repeatedly.
#     
#             x_dummy is unused.
#             The returned array must have the same shape as ydata.
#             """
#             call_counter["n"] += 1
#     
#             mua = float(mua)
#             mus = float(mus)
#     
#             if mua <= 0 or mus <= 0:
#                 return np.full_like(ydata, 1e12, dtype=float)
#     
#             try:
#                 sim_compare, _ = self.run_once(mua, mus)
#             except Exception as exc:
#                 print(
#                     f"[warn] simulation failed at "
#                     f"mua={mua:.6g}, mus={mus:.6g}: {exc}"
#                 )
#                 return np.full_like(ydata, 1e12, dtype=float)
#     
#             sim_full = np.asarray(sim_compare, dtype=float).reshape(-1)
#     
#             if sim_full.size != y_full.size:
#                 raise ValueError(
#                     f"sim_compare size {sim_full.size} does not match "
#                     f"exp_compare size {y_full.size}"
#                 )
#     
#             yhat = sim_full[valid]
#     
#             # curve_fit 不能处理 nan / inf，所以要替换掉
#             if not np.all(np.isfinite(yhat)):
#                 yhat = np.nan_to_num(
#                     yhat,
#                     nan=1e12,
#                     posinf=1e12,
#                     neginf=-1e12,
#                 )
#     
#             return yhat
#     
#         # popt, pcov = curve_fit(
#         #     model_func,
#         #     xdata,
#         #     ydata,
#         #     p0=p0,
#         #     method="lm",
#         #     maxfev=max_nfev,
#         # )
#         
#         popt, pcov = curve_fit(
#             model_func,
#             xdata,
#             ydata,
#             p0=p0,
#             bounds=(lower, upper),
#             method="trf",
#             max_nfev=max_nfev,
#         )
#     
#     
#         mua_fit, mus_fit = popt
#     
#         # 用最终参数再跑一次，得到最终模拟结果和 mse
#         sim_compare, aux = self.run_once(float(mua_fit), float(mus_fit))
#         sim_full = np.asarray(sim_compare, dtype=float).reshape(-1)
#         residual = sim_full[valid] - ydata
#         mse = float(np.mean(residual ** 2))
#     
#         # 为了尽量保持和 scipy.optimize 的返回格式类似
#         result = OptimizeResult(
#             x=np.asarray(popt, dtype=float),
#             fun=mse,
#             success=True,
#             message="curve_fit finished",
#             popt=popt,
#             pcov=pcov,
#             nfev=call_counter["n"],
#             sim_compare=sim_compare,
#             aux=aux,
#             exp_compare = self.exp_compare
#         )
#     
#         return result
# =============================================================================
    
    

    def objective(self, params, verbose=True):
        params = np.asarray(params, dtype=float)
        mua, mus = params[:2]
        n = params[2] if FIT_REFRACTIVE_INDEX and params.size >= 3 else self.fixed_n

        if mua <= 0 or mus <= 0 or n <= 0:
            return 1e12

        try:
            sim_compare, _ = self.run_once(float(mua), float(mus), float(n))
        except Exception as exc:
            if verbose:
                print(f"[warn] simulation failed at mua={mua:.6g}, mus={mus:.6g}, n={n:.6g}: {exc}")
            return 1e12

        mse = float(self.compare_loss(sim_compare))
        if verbose:
            print(f"[eval] mua={mua:.6g}, mus={mus:.6g}, n={n:.6g}, loss={mse:.8g}")
        return mse

    def fit(self, p0=None, max_nfev=80):
        """
        Fit mua/mus, and optionally refractive index n, with a derivative-free
        black-box optimizer.

        The MCX simulation is stochastic and the simulated cube is globally
        normalized. Local least-squares methods such as curve_fit/trf often see
        nearly zero or noisy gradients and can stop at p0. Differential
        evolution samples the bounded parameter space directly, then
        Nelder-Mead polishes the best sampled point.
        """

        lower = [1e-5, 0.1]
        upper = [0.1, 10.0]
        if FIT_REFRACTIVE_INDEX:
            lower.append(N_BOUNDS[0])
            upper.append(N_BOUNDS[1])

        lower = np.array(lower, dtype=float)
        upper = np.array(upper, dtype=float)

        if p0 is None:
            p0 = [self.default_mua, self.default_mus]
            if FIT_REFRACTIVE_INDEX:
                p0.append(self.fixed_n)
        p0 = np.clip(np.asarray(p0, dtype=float), lower, upper)

        bounds = list(zip(lower, upper))
        p0_mse = self.objective(p0)

        result_de = differential_evolution(
            lambda x: self.objective(x),
            bounds=bounds,
            seed=42,
            maxiter=max(1, max_nfev // 12),
            popsize=6,
            polish=False,
            workers=1,
            updating="immediate",
            x0=p0,
        )

        result_nm = minimize(
            lambda x: self.objective(np.clip(x, lower, upper)),
            result_de.x,
            method="Nelder-Mead",
            options={"maxiter": max(10, max_nfev // 3), "xatol": 1e-4, "fatol": 1e-5},
        )

        nm_x = np.clip(result_nm.x, lower, upper)
        candidates = [
            (p0_mse, p0, "p0"),
            (float(result_de.fun), result_de.x, "differential_evolution"),
            (float(result_nm.fun), nm_x, "nelder_mead"),
        ]
        mse, best_x, source = min(candidates, key=lambda item: item[0])

        best_n = float(best_x[2]) if FIT_REFRACTIVE_INDEX and len(best_x) >= 3 else self.fixed_n
        sim_compare, aux = self.run_once(float(best_x[0]), float(best_x[1]), best_n)
        mse = float(self.compare_loss(sim_compare))

        return OptimizeResult(
            x=np.asarray(best_x, dtype=float),
            fun=mse,
            success=True,
            message=f"best candidate from {source}",
            nfev=getattr(result_de, "nfev", 0) + getattr(result_nm, "nfev", 0) + 1,
            sim_compare=sim_compare,
            aux=aux,
            exp_compare=self.exp_compare,
            p0_mse=p0_mse,
            de_result=result_de,
            nm_result=result_nm,
        )

    def save_outputs(self, mua, mus, n, mse, sim_map2d,sim_comapre):
        exp_map2d = self.exp_map2d
        diff = np.abs(exp_map2d - sim_map2d)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(exp_map2d, origin="upper", cmap="jet")
        axes[0].set_title("Experiment (normalized)")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(sim_map2d, origin="upper", cmap="jet")
        axes[1].set_title("Simulation after IRF + norm")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(diff, origin="upper", cmap="hot")
        axes[2].set_title("Absolute difference")
        plt.colorbar(im2, ax=axes[2])

        for ax in axes:
            ax.set_xlabel("Y index")
            ax.set_ylabel("Z index")
        plt.tight_layout()
        fig.savefig("fitting_comparison.png", dpi=150)
        plt.close(fig)

        np.savez(
            "fitting_results.npz",
            mua=mua,
            mus=mus,
            g=self.fixed_g,
            n=n,
            mse=mse,
            exp_map2d=exp_map2d,
            sim_map2d=sim_map2d,
            exp_compare = self.exp_compare,
            sim_compare = sim_comapre,
        )

        with open("fitting_results_report.txt", "w", encoding="utf-8") as f:
            f.write("PMCX FIT REPORT\n")
            f.write("====================\n")
            f.write(f"experiment_path: {EXPERIMENT_MAT_PATH}\n")
            f.write(f"experiment_var: {self.exp_var}\n")
            f.write(f"experiment_loaded_shape: {self.exp_loaded_shape}\n")
            f.write(f"experiment_peak_bin: {self.exp_peak_bin}\n")
            f.write(f"irf_var: {self.irf_var}\n")
            f.write("irf_curve_index: [16,16] (MATLAB index)\n")
            f.write("\n")
            f.write(f"mua: {mua:.8e} mm^-1\n")
            f.write(f"mus: {mus:.8e} mm^-1\n")
            f.write(f"n: {n:.8e}\n")
            f.write(f"g_fixed: {self.fixed_g:.6f}\n")
            f.write(f"fit_refractive_index: {FIT_REFRACTIVE_INDEX}\n")
            f.write(f"n_initial_from_meta: {self.fixed_n:.6f}\n")
            f.write(f"n_bounds: [{N_BOUNDS[0]:.6f}, {N_BOUNDS[1]:.6f}]\n")
            f.write(f"mse: {mse:.8e}\n")
            f.write(f"loss_space_weight: {LOSS_SPACE_WEIGHT:.6f}\n")
            f.write(f"loss_time_weight: {LOSS_TIME_WEIGHT:.6f}\n")
            f.write("\nOutputs:\n")
            f.write("- fitting_comparison.png\n")
            f.write("- experiment_orientation_diagnostic.png\n")
            f.write("- fitting_results.npz\n")


if __name__ == "__main__":
    
        print("=" * 60)
        print("PMCX FITTING SCRIPT")
        print("=" * 60)
        print("\n[1/5] Loading data...")
        fitter = PMCXFitter()
        print("  ✓ Data loaded successfully.")
        print(f"      Experiment variable: {fitter.exp_var}, shape={fitter.exp_raw.shape}")
        print(f"      After reshape: exp_compare shape={fitter.exp_compare.shape}")
        print(f"      IRF variable: {fitter.irf_var}, length={len(fitter.irf)}")
        print(f"      Fixed g={fitter.fixed_g:.4f}, n={fitter.fixed_n:.4f}")

        if FIT_REFRACTIVE_INDEX:
            print("\n[2/5] Starting optimization (mua, mus, n) ...")
        else:
            print("\n[2/5] Starting optimization (mua, mus) ...")
        result = fitter.fit()
        mua, mus = result.x[:2]
        n_fit = float(result.x[2]) if FIT_REFRACTIVE_INDEX and len(result.x) >= 3 else fitter.fixed_n
        mse = float(result.fun)
        print(f"  ✓ Fit done.")
        print(f"      mua={mua:.8e} mm^-1")
        print(f"      mus={mus:.8e} mm^-1")
        print(f"      n={n_fit:.8e}")
        print(f"      mse={mse:.8e}")

        print("\n[3/5] Running final simulation...")
        sim_comapre, sim_map2d = fitter.run_once(mua, mus, n_fit)
        print(f"  ✓ Simulation complete. sim_map2d shape={sim_map2d.shape}")

        print("\n[4/5] Saving outputs...")
        fitter.save_outputs(mua, mus, n_fit, mse, sim_map2d,sim_comapre)
        print("  ✓ Saved: fitting_comparison.png")
        print("  ✓ Saved: fitting_results.npz")
        print("  ✓ Saved: fitting_results_report.txt")

        print("\n[5/5] Done!")
        print("=" * 60)
        
        my_display_hist(result.sim_compare)
        my_display_hist(result.exp_compare)
        compare_hist(result.sim_compare, result.exp_compare)




