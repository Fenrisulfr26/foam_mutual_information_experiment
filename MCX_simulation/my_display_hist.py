"""
Python version of my_display_hist.m.

Interactive viewer for a 3D histogram cube.

Usage
-----
    from my_display_hist import my_display_hist, display_hist_mat

    my_display_hist(histgram)
    display_hist_mat(r"path\to\hist_file.mat")

Input convention
----------------
histgram is a ny x nx x nBins array.
histgram[y, x, :] is the histogram curve of pixel (x + 1, y + 1).
The displayed X/Y labels are 1-based, matching the MATLAB function.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


def my_display_hist(histgram, gaussian_contour_fraction=0.60, figure_name=None, show=True):
    """
    Interactive viewer for a 3D histogram cube.

    Parameters
    ----------
    histgram : array_like, shape (ny, nx, nBins)
        Histogram cube. histgram[y, x, :] is the curve of pixel (x + 1, y + 1).
    gaussian_contour_fraction : float
        Contour level for the fitted 2D Gaussian:
        offset + fraction * amplitude.
    figure_name : str, optional
        Window title. A new figure is created for every call even when the
        title is reused.
    show : bool
        If True, request Matplotlib to show the figure without blocking.

    Returns
    -------
    fig, state
        Matplotlib figure and a state dictionary containing handles/callback id.
    """

    histgram = np.asarray(histgram, dtype=float)
    if histgram.ndim != 3:
        raise ValueError("histgram must be a 3D array, for example 32 x 32 x nBins.")

    ny, nx, n_bins = histgram.shape
    if ny < 1 or nx < 1 or n_bins < 1:
        raise ValueError("histgram has invalid size.")

    histgram = histgram.copy()
    histgram[~np.isfinite(histgram)] = 0

    intensity_image = np.sum(histgram, axis=2)
    center_x, center_y, fitted_image, contour_level = fit_gaussian_center_2d(
        intensity_image,
        gaussian_contour_fraction,
    )
    bin_axis = np.arange(n_bins)

    if figure_name is None:
        figure_name = "my_display_hist"

    fig = plt.figure(figsize=(10.5, 4.8), facecolor="w")
    try:
        fig.canvas.manager.set_window_title(figure_name)
    except Exception:
        pass
    ax_img = fig.add_axes([0.07, 0.16, 0.35, 0.75])
    ax_curve = fig.add_axes([0.54, 0.28, 0.41, 0.58])

    # Match MATLAB imagesc orientation: row 1 is at the top.
    im = ax_img.imshow(
        intensity_image,
        cmap="jet",
        origin="upper",
        extent=[0.5, nx + 0.5, ny + 0.5, 0.5],
        aspect="equal",
    )
    fig.colorbar(im, ax=ax_img)
    ax_img.set_title("Accumulated photon count")
    ax_img.set_xlabel("X pixel")
    ax_img.set_ylabel("Y pixel")
    ax_img.set_xlim(0.5, nx + 0.5)
    ax_img.set_ylim(ny + 0.5, 0.5)

    marker, = ax_img.plot(1, 1, "wo", markersize=10, markerfacecolor="none", linewidth=1.5)

    if np.isfinite(center_x) and np.isfinite(center_y):
        if fitted_image is not None and np.isfinite(contour_level):
            x_grid = np.arange(1, nx + 1)
            y_grid = np.arange(1, ny + 1)
            ax_img.contour(
                x_grid,
                y_grid,
                fitted_image,
                levels=[contour_level],
                colors="k",
                linewidths=1,
            )
        ax_img.plot(center_x, center_y, "wx", markersize=8, linewidth=1)

    curve_line, = ax_curve.plot(bin_axis, np.zeros(n_bins), linewidth=1.5)
    ax_curve.grid(True)
    ax_curve.set_xlim(bin_axis[0], bin_axis[-1] if n_bins > 1 else 1)
    ax_curve.set_xlabel("Bin index")
    ax_curve.set_ylabel("Counts")
    ax_curve.set_title("Pixel histogram")

    text_x = fig.text(0.54, 0.18, "X: 1", fontsize=10)
    text_y = fig.text(0.64, 0.18, "Y: 1", fontsize=10)
    text_sum = fig.text(0.74, 0.18, "Total counts: 0", fontsize=10)

    global_ymax = float(np.nanmax(histgram)) if histgram.size else 0.0
    state = {
        "last_x": None,
        "last_y": None,
        "callback_id": None,
        "histgram": histgram,
        "intensity_image": intensity_image,
        "gaussian_center": (center_x, center_y),
    }

    def update_pixel(x_pos, y_pos):
        state["last_x"] = x_pos
        state["last_y"] = y_pos

        selected_curve = histgram[y_pos - 1, x_pos - 1, :].reshape(-1)
        total_counts = float(np.sum(selected_curve))

        curve_line.set_data(bin_axis, selected_curve)
        if global_ymax <= 0:
            ax_curve.set_ylim(0, 1)
        else:
            ax_curve.set_ylim(0, global_ymax * 1.1)

        ax_curve.set_title(f"Histogram at X = {x_pos}, Y = {y_pos}")
        text_x.set_text(f"X: {x_pos}")
        text_y.set_text(f"Y: {y_pos}")
        text_sum.set_text(f"Total counts: {total_counts:.0f}")
        marker.set_data([x_pos], [y_pos])
        fig.canvas.draw_idle()

    def mouse_move_callback(event):
        if event.inaxes is not ax_img or event.xdata is None or event.ydata is None:
            return

        x_pos = int(np.rint(event.xdata))
        y_pos = int(np.rint(event.ydata))

        if x_pos < 1 or x_pos > nx or y_pos < 1 or y_pos > ny:
            return

        if x_pos == state["last_x"] and y_pos == state["last_y"]:
            return

        update_pixel(x_pos, y_pos)

    state["callback_id"] = fig.canvas.mpl_connect("motion_notify_event", mouse_move_callback)
    update_pixel(1, 1)
    if show:
        plt.show(block=False)

    return fig, state


def compare_hist(
    hist_a,
    hist_b,
    label_a="A",
    label_b="B",
    figure_name="compare_hist",
    gaussian_contour_fraction=0.60,
    show=True,
):
    """
    Compare two 3D histogram cubes interactively.

    The top row shows accumulated intensity images for hist_a and hist_b.
    Moving the mouse over either image selects that pixel. The bottom row
    shows the two histogram curves for the same selected pixel.

    Parameters
    ----------
    hist_a, hist_b : array_like, shape (ny, nx, nBins)
        Histogram cubes to compare. They must have the same shape.
    label_a, label_b : str
        Labels used in titles and legends.
    figure_name : str
        Window title.
    gaussian_contour_fraction : float
        Contour level for fitted Gaussian overlays on the intensity images.
    show : bool
        If True, request Matplotlib to show the figure without blocking.

    Returns
    -------
    fig, state
        Matplotlib figure and callback state.
    """

    hist_a = np.asarray(hist_a, dtype=float)
    hist_b = np.asarray(hist_b, dtype=float)

    if hist_a.ndim != 3 or hist_b.ndim != 3:
        raise ValueError("hist_a and hist_b must both be 3D arrays.")
    if hist_a.shape != hist_b.shape:
        raise ValueError(f"hist_a shape {hist_a.shape} does not match hist_b shape {hist_b.shape}.")

    ny, nx, n_bins = hist_a.shape
    if ny < 1 or nx < 1 or n_bins < 1:
        raise ValueError("histograms have invalid size.")

    hist_a = hist_a.copy()
    hist_b = hist_b.copy()
    hist_a[~np.isfinite(hist_a)] = 0
    hist_b[~np.isfinite(hist_b)] = 0

    image_a = np.sum(hist_a, axis=2)
    image_b = np.sum(hist_b, axis=2)
    bin_axis = np.arange(n_bins)

    fit_a = fit_gaussian_center_2d(image_a, gaussian_contour_fraction)
    fit_b = fit_gaussian_center_2d(image_b, gaussian_contour_fraction)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), facecolor="w")
    try:
        fig.canvas.manager.set_window_title(figure_name)
    except Exception:
        pass

    ax_img_a, ax_img_b = axes[0]
    ax_curve, ax_diff = axes[1]

    vmax = max(float(np.nanmax(image_a)), float(np.nanmax(image_b)), 1.0)

    def setup_image_axis(ax, image, label, fit):
        im = ax.imshow(
            image,
            cmap="jet",
            origin="upper",
            extent=[0.5, nx + 0.5, ny + 0.5, 0.5],
            aspect="equal",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(f"{label}: accumulated photon count")
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")
        ax.set_xlim(0.5, nx + 0.5)
        ax.set_ylim(ny + 0.5, 0.5)

        center_x, center_y, fitted_image, contour_level = fit
        if np.isfinite(center_x) and np.isfinite(center_y):
            if fitted_image is not None and np.isfinite(contour_level):
                x_grid = np.arange(1, nx + 1)
                y_grid = np.arange(1, ny + 1)
                ax.contour(
                    x_grid,
                    y_grid,
                    fitted_image,
                    levels=[contour_level],
                    colors="k",
                    linewidths=1,
                )
            ax.plot(center_x, center_y, "wx", markersize=8, linewidth=1)

        marker, = ax.plot(1, 1, "wo", markersize=10, markerfacecolor="none", linewidth=1.5)
        return im, marker

    im_a, marker_a = setup_image_axis(ax_img_a, image_a, label_a, fit_a)
    im_b, marker_b = setup_image_axis(ax_img_b, image_b, label_b, fit_b)
    fig.colorbar(im_a, ax=ax_img_a, fraction=0.046, pad=0.04)
    fig.colorbar(im_b, ax=ax_img_b, fraction=0.046, pad=0.04)

    line_a, = ax_curve.plot(bin_axis, np.zeros(n_bins), linewidth=1.6, label=label_a)
    line_b, = ax_curve.plot(bin_axis, np.zeros(n_bins), linewidth=1.6, label=label_b)
    ax_curve.grid(True)
    ax_curve.set_xlim(bin_axis[0], bin_axis[-1] if n_bins > 1 else 1)
    ax_curve.set_xlabel("Bin index")
    ax_curve.set_ylabel("Counts")
    ax_curve.legend()

    line_diff, = ax_diff.plot(bin_axis, np.zeros(n_bins), linewidth=1.4, color="k")
    ax_diff.axhline(0, color="0.5", linewidth=0.8)
    ax_diff.grid(True)
    ax_diff.set_xlim(bin_axis[0], bin_axis[-1] if n_bins > 1 else 1)
    ax_diff.set_xlabel("Bin index")
    ax_diff.set_ylabel(f"{label_a} - {label_b}")

    info_text = fig.text(0.5, 0.02, "", ha="center", fontsize=10)

    global_ymax = max(float(np.nanmax(hist_a)), float(np.nanmax(hist_b)), 1.0)
    global_diff_abs = max(float(np.nanmax(np.abs(hist_a - hist_b))), 1.0)

    state = {
        "last_x": None,
        "last_y": None,
        "callback_id": None,
        "hist_a": hist_a,
        "hist_b": hist_b,
        "image_a": image_a,
        "image_b": image_b,
    }

    def update_pixel(x_pos, y_pos):
        state["last_x"] = x_pos
        state["last_y"] = y_pos

        curve_a = hist_a[y_pos - 1, x_pos - 1, :].reshape(-1)
        curve_b = hist_b[y_pos - 1, x_pos - 1, :].reshape(-1)
        diff = curve_a - curve_b

        line_a.set_data(bin_axis, curve_a)
        line_b.set_data(bin_axis, curve_b)
        line_diff.set_data(bin_axis, diff)

        ax_curve.set_ylim(0, global_ymax * 1.1 if global_ymax > 0 else 1)
        ax_curve.set_title(f"Histogram comparison at X = {x_pos}, Y = {y_pos}")
        ax_diff.set_ylim(-global_diff_abs * 1.1, global_diff_abs * 1.1)
        ax_diff.set_title("Difference curve")

        marker_a.set_data([x_pos], [y_pos])
        marker_b.set_data([x_pos], [y_pos])

        total_a = float(np.sum(curve_a))
        total_b = float(np.sum(curve_b))
        info_text.set_text(
            f"X: {x_pos}   Y: {y_pos}    "
            f"{label_a} total: {total_a:.0f}    {label_b} total: {total_b:.0f}"
        )

        fig.canvas.draw_idle()

    def mouse_move_callback(event):
        if event.inaxes not in (ax_img_a, ax_img_b) or event.xdata is None or event.ydata is None:
            return

        x_pos = int(np.rint(event.xdata))
        y_pos = int(np.rint(event.ydata))

        if x_pos < 1 or x_pos > nx or y_pos < 1 or y_pos > ny:
            return
        if x_pos == state["last_x"] and y_pos == state["last_y"]:
            return

        update_pixel(x_pos, y_pos)

    state["callback_id"] = fig.canvas.mpl_connect("motion_notify_event", mouse_move_callback)
    update_pixel(1, 1)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if show:
        plt.show(block=False)

    return fig, state


def display_hist_mat(path, var_name="hist", figure_name=None, **kwargs):
    """
    Load a MAT file and display its histogram cube.

    Parameters
    ----------
    path : str
        MAT file path.
    var_name : str
        Variable name to load. Defaults to "hist".
    **kwargs
        Extra keyword arguments passed to my_display_hist.
    """

    mat = loadmat(path)
    if var_name not in mat:
        public_vars = [name for name in mat if not name.startswith("__")]
        raise KeyError(f"Variable {var_name!r} not found. Available variables: {public_vars}")

    if figure_name is None:
        figure_name = str(path)

    return my_display_hist(mat[var_name], figure_name=figure_name, **kwargs)


def fit_gaussian_center_2d(intensity_image, contour_fraction=0.60):
    """
    Estimate the center of a 2D Gaussian on an image.

    This mirrors the MATLAB helper that uses fminsearch, using scipy's
    Nelder-Mead optimizer instead.
    """

    center_x = np.nan
    center_y = np.nan
    fitted_image = None
    contour_level = np.nan

    contour_fraction = float(contour_fraction)
    if not np.isfinite(contour_fraction):
        contour_fraction = 0.60
    contour_fraction = min(max(contour_fraction, 0.0), 1.0)

    image_data = np.asarray(intensity_image, dtype=float)
    image_data = image_data.copy()
    image_data[~np.isfinite(image_data)] = 0

    if image_data.ndim != 2:
        raise ValueError("intensity_image must be a 2D array.")

    ny, nx = image_data.shape
    if ny < 1 or nx < 1:
        return center_x, center_y, fitted_image, contour_level

    min_value = float(np.min(image_data))
    signal_data = image_data - min_value
    signal_sum = float(np.sum(signal_data))

    y_grid, x_grid = np.mgrid[1 : ny + 1, 1 : nx + 1]

    if signal_sum <= 0:
        max_idx = np.unravel_index(np.argmax(image_data), image_data.shape)
        center_y = float(max_idx[0] + 1)
        center_x = float(max_idx[1] + 1)
        return center_x, center_y, fitted_image, contour_level

    x0 = float(np.sum(x_grid * signal_data) / signal_sum)
    y0 = float(np.sum(y_grid * signal_data) / signal_sum)

    x_variance = float(np.sum(((x_grid - x0) ** 2) * signal_data) / signal_sum)
    y_variance = float(np.sum(((y_grid - y0) ** 2) * signal_data) / signal_sum)

    sigma_x0 = max(np.sqrt(max(x_variance, np.finfo(float).eps)), 1.0)
    sigma_y0 = max(np.sqrt(max(y_variance, np.finfo(float).eps)), 1.0)
    offset0 = min_value
    amplitude0 = max(float(np.max(image_data)) - offset0, np.finfo(float).eps)

    initial_params = np.array(
        [
            offset0,
            np.log(amplitude0),
            x0,
            y0,
            np.log(sigma_x0),
            np.log(sigma_y0),
        ],
        dtype=float,
    )

    def objective(params):
        residual = image_data.ravel() - gaussian_model(params, x_grid, y_grid)
        return float(np.sum(residual**2))

    try:
        result = minimize(
            objective,
            initial_params,
            method="Nelder-Mead",
            options={"maxiter": 1000, "maxfev": 3000, "disp": False},
        )
        fitted_params = result.x if result.success else initial_params
        center_x = float(np.clip(fitted_params[2], 1, nx))
        center_y = float(np.clip(fitted_params[3], 1, ny))
    except Exception:
        fitted_params = initial_params
        center_x = float(np.clip(x0, 1, nx))
        center_y = float(np.clip(y0, 1, ny))

    fitted_image = gaussian_model(fitted_params, x_grid, y_grid).reshape(ny, nx)
    contour_level = float(fitted_params[0] + contour_fraction * np.exp(fitted_params[1]))
    return center_x, center_y, fitted_image, contour_level


def gaussian_model(params, x_grid, y_grid):
    offset = params[0]
    amplitude = np.exp(params[1])
    center_x = params[2]
    center_y = params[3]
    sigma_x = max(np.exp(params[4]), np.finfo(float).eps)
    sigma_y = max(np.exp(params[5]), np.finfo(float).eps)

    values = offset + amplitude * np.exp(
        -0.5
        * (
            ((x_grid.ravel() - center_x) / sigma_x) ** 2
            + ((y_grid.ravel() - center_y) / sigma_y) ** 2
        )
    )
    return values
