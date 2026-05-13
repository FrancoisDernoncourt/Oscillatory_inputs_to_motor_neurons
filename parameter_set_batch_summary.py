import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
from scipy.signal import hilbert
from shared_run_analysis import (
    ANALYSIS_PARAM_FEATURE_COLUMNS,
    ANALYSIS_PARAM_FILTER_COLUMNS,
    ANALYSIS_PARAM_SPECTRUM_COLUMNS,
    ANALYSIS_PARAM_SYNC_COLUMNS,
    ANALYSIS_PARAM_WINDOW_COLUMNS,
    classify_active_units_by_rate,
    DEFAULT_MODULATION_SPECTRUM_CONFIG,
    DEFAULT_OBSERVATION_WINDOW_DEFS_REL_CUE_S,
    EXPORT_TABLE_COLUMNS,
    OBS_BASELINE_EXPORT_COLUMNS,
    OBS_SPECTRUM_EXPORT_COLUMNS,
    OBS_WINDOW_SPECS,
    OBS_WINDOW_EXPORT_COLUMNS,
    compute_observation_features_from_cst_sync,
    resolve_modulation_spectrum_config,
)
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


DEFAULT_INPUT_COMPONENT_COLORS = {
    "lf": "#2ca02c",
    "alpha": "#1f77b4",
    "beta": "#9467bd",
    "common": "#111111",
    "cst": "#ff7f0e",
    "sync": "#d62728",
    "envelope": "#444444",
    "trend": "#8c8c8c",
    "burst": "#FF0080",
    "cue": "#7a7a7a",
}

DEFAULT_PLOT_STYLE = {
    "figure_size": (19, 11),
    "figure_dpi": 120,
    "individual_trace_lw": 0.8,
    "individual_trace_alpha": 0.10,
    "mean_trace_lw": 1.6,
    "std_band_alpha": 0.18,
    "envelope_lw": 1.0,
    "envelope_alpha": 0.9,
    "trend_lw": 1.2,
    "trend_alpha": 0.95,
    "trend_ls": ":",
    "burst_kernel_lw": 1.1,
    "burst_kernel_alpha": 0.95,
    "reference_std_lw": 1.0,
    "reference_std_alpha": 0.45,
    "burst_epoch_alpha": 0.10,
    "burst_center_lw": 1.1,
    "burst_center_alpha": 0.95,
    "get_ready_lw": 1.0,
    "get_ready_alpha": 0.8,
    "go_cue_lw": 1.1,
    "go_cue_alpha": 0.8,
    "feature_baseline_lw": 1.3,
    "feature_baseline_ls": "--",
    "feature_baseline_alpha": 1.0,
    "feature_baseline_color": "#111111",
    "feature_baseline_backdrop_color": "#BDBDBD",
    "feature_baseline_backdrop_lw": 3.0,
    "feature_baseline_backdrop_alpha": 0.85,
    "feature_outside_alpha": 0.12,
    "feature_outside_color": "#D9D9D9",
    "feature_analysis_mean_lw": 1.2,
    "feature_analysis_mean_ls": "--",
    "feature_analysis_mean_alpha": 1.0,
    "feature_text_fontsize": 9,
    "feature_text_weight": "bold",
    "feature_text_box_alpha": 0.75,
    "feature_text_box_facecolor": "white",
}

DEFAULT_DISTRIBUTION_STYLE = {
    "figure_size": (14, 7),
    "figure_dpi": 130,
    "scatter_alpha": 0.55,
    "scatter_size": 22,
    "hist_alpha": 0.75,
    "hist_edgecolor": "white",
    "title_fontsize": 12,
    "panel_title_fontsize": 9,
}

DEFAULT_DISTRIBUTION_COLORS = {
    "firing_rate": "#E67E22",
    "isi_cv": "#2E86C1",
}


@dataclass
class BaselineSplineFeatureResult:
    t_fit_window: np.ndarray
    y_fit_window: np.ndarray
    baseline_fit_window: np.ndarray
    residual_fit_window: np.ndarray
    feature_mean_residual: float
    feature_signed_area: float
    fit_r2: float
    linear_fit_r2: float
    fit_r2_gain_over_linear: float
    trend_pre_mean: float
    trend_post_mean: float
    trend_post_minus_pre: float
    fitting_mask: np.ndarray
    analysis_mask: np.ndarray
    exclusion_mask: np.ndarray
    fit_support_mask: np.ndarray
    spline_object: object


def fit_baseline_spline_and_extract_feature(
    t,
    y,
    fitting_window,
    analysis_window,
    buffer_s=0.0,
    spline_smoothing=None,
    spline_order=3,
    n_knots=None,
    trend_pre_window=None,
    trend_post_window=None,
    extrapolate=False,
    return_debug_info=True,
):
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if t.shape != y.shape:
        raise ValueError("t and y must have the same shape")
    if t.ndim != 1:
        raise ValueError("t and y must be 1D arrays")

    finite_mask = np.isfinite(t) & np.isfinite(y)
    if not np.any(finite_mask):
        raise ValueError("No finite samples are available for spline fitting")
    t = t[finite_mask]
    y = y[finite_mask]

    sort_idx = np.argsort(t)
    t = t[sort_idx]
    y = y[sort_idx]
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing after preprocessing")

    fit_start_s, fit_end_s = map(float, fitting_window)
    analysis_start_s, analysis_end_s = map(float, analysis_window)
    if fit_start_s >= fit_end_s:
        raise ValueError("fitting_window must be increasing")
    if analysis_start_s >= analysis_end_s:
        raise ValueError("analysis_window must be increasing")
    if analysis_start_s < fit_start_s or analysis_end_s > fit_end_s:
        raise ValueError("analysis_window must lie within fitting_window")

    fitting_mask = (t >= fit_start_s) & (t <= fit_end_s)
    if not np.any(fitting_mask):
        raise ValueError("No samples fall inside fitting_window")
    t_fit = t[fitting_mask]
    y_fit = y[fitting_mask]

    analysis_mask = (t_fit >= analysis_start_s) & (t_fit <= analysis_end_s)
    if not np.any(analysis_mask):
        raise ValueError("No samples fall inside analysis_window")

    exclusion_start_s = analysis_start_s - float(buffer_s)
    exclusion_end_s = analysis_end_s + float(buffer_s)
    exclusion_mask = (t_fit >= exclusion_start_s) & (t_fit <= exclusion_end_s)
    fit_support_mask = ~exclusion_mask
    if np.count_nonzero(fit_support_mask) <= int(spline_order):
        raise ValueError("Too few points remain to fit the spline after exclusion masking")

    t_support = t_fit[fit_support_mask]
    y_support = y_fit[fit_support_mask]

    if n_knots is not None:
        n_knots = int(n_knots)
        if n_knots < 0:
            raise ValueError("n_knots must be >= 0")
        if n_knots == 0:
            n_knots = None

    if n_knots:
        interior = np.linspace(t_support[0], t_support[-1], n_knots + 2, dtype=float)[1:-1]
        valid_knots = interior[(interior > t_support[0]) & (interior < t_support[-1])]
        if valid_knots.size == 0:
            spline = UnivariateSpline(
                t_support,
                y_support,
                k=int(spline_order),
                s=float(0.0 if spline_smoothing is None else spline_smoothing),
                ext=0 if extrapolate else 3,
            )
        else:
            spline = LSQUnivariateSpline(
                t_support,
                y_support,
                t=valid_knots,
                k=int(spline_order),
                ext=0 if extrapolate else 3,
            )
    else:
        if spline_smoothing is None:
            spline_smoothing = float(len(t_support) * np.var(y_support))
        spline = UnivariateSpline(
            t_support,
            y_support,
            k=int(spline_order),
            s=float(spline_smoothing),
            ext=0 if extrapolate else 3,
        )

    baseline_fit = np.asarray(spline(t_fit), dtype=float)
    residual_fit = y_fit - baseline_fit
    feature_mean_residual = float(np.mean(residual_fit[analysis_mask]))
    if np.count_nonzero(analysis_mask) >= 2:
        feature_signed_area = float(np.trapz(residual_fit[analysis_mask], t_fit[analysis_mask]))
    else:
        feature_signed_area = 0.0
    ss_res = float(np.sum((y_fit - baseline_fit) ** 2))
    ss_tot = float(np.sum((y_fit - float(np.mean(y_fit))) ** 2))
    if ss_tot <= 0.0:
        fit_r2 = 1.0 if ss_res <= 1e-12 else np.nan
    else:
        fit_r2 = float(1.0 - ss_res / ss_tot)
    if t_fit.size < 2:
        linear_fit_r2 = np.nan
    else:
        linear_coeffs = np.polyfit(t_fit, y_fit, deg=1)
        linear_fit = np.polyval(linear_coeffs, t_fit)
        linear_ss_res = float(np.sum((y_fit - linear_fit) ** 2))
        if ss_tot <= 0.0:
            linear_fit_r2 = 1.0 if linear_ss_res <= 1e-12 else np.nan
        else:
            linear_fit_r2 = float(1.0 - linear_ss_res / ss_tot)
    fit_r2_gain_over_linear = (
        float(fit_r2 - linear_fit_r2)
        if np.isfinite(fit_r2) and np.isfinite(linear_fit_r2)
        else np.nan
    )

    if trend_pre_window is None:
        trend_pre_window = (fit_start_s, min(0.0, fit_end_s))
    if trend_post_window is None:
        trend_post_window = (max(0.0, fit_start_s), fit_end_s)
    trend_pre_start_s, trend_pre_end_s = map(float, trend_pre_window)
    trend_post_start_s, trend_post_end_s = map(float, trend_post_window)
    pre_mask = (t_fit >= trend_pre_start_s) & (t_fit <= trend_pre_end_s)
    post_mask = (t_fit >= trend_post_start_s) & (t_fit <= trend_post_end_s)
    if not np.any(pre_mask):
        trend_pre_mean = np.nan
    else:
        trend_pre_mean = float(np.mean(baseline_fit[pre_mask]))
    if not np.any(post_mask):
        trend_post_mean = np.nan
    else:
        trend_post_mean = float(np.mean(baseline_fit[post_mask]))
    trend_post_minus_pre = (
        float(trend_post_mean - trend_pre_mean)
        if np.isfinite(trend_pre_mean) and np.isfinite(trend_post_mean)
        else np.nan
    )

    return BaselineSplineFeatureResult(
        t_fit_window=t_fit,
        y_fit_window=y_fit,
        baseline_fit_window=baseline_fit,
        residual_fit_window=residual_fit,
        feature_mean_residual=feature_mean_residual,
        feature_signed_area=feature_signed_area,
        fit_r2=fit_r2,
        linear_fit_r2=linear_fit_r2,
        fit_r2_gain_over_linear=fit_r2_gain_over_linear,
        trend_pre_mean=trend_pre_mean,
        trend_post_mean=trend_post_mean,
        trend_post_minus_pre=trend_post_minus_pre,
        fitting_mask=fitting_mask,
        analysis_mask=analysis_mask,
        exclusion_mask=exclusion_mask,
        fit_support_mask=fit_support_mask,
        spline_object=spline if return_debug_info else None,
    )


def build_synthetic_baseline_feature_example():
    t = np.linspace(-2.0, 1.0, 601)
    baseline = 0.2 * np.sin(2 * np.pi * 0.3 * t) + 0.15 * t
    bump = 0.8 * np.exp(-0.5 * ((t - 0.25) / 0.08) ** 2)
    y = baseline + bump
    result = fit_baseline_spline_and_extract_feature(
        t,
        y,
        fitting_window=(-2.0, 1.0),
        analysis_window=(0.0, 0.5),
        buffer_s=0.1,
        spline_smoothing=None,
        spline_order=3,
        n_knots=None,
        extrapolate=False,
        return_debug_info=True,
    )
    return {"t": t, "baseline": baseline, "signal": y, "result": result}


def _resolve_plot_style(plot_style=None):
    resolved = dict(DEFAULT_PLOT_STYLE)
    if plot_style:
        resolved.update(plot_style)
    return resolved


def _resolve_distribution_style(plot_style=None):
    resolved = dict(DEFAULT_DISTRIBUTION_STYLE)
    if plot_style:
        resolved.update(plot_style)
    return resolved


def _resolve_feature_config(feature_config=None):
    resolved = {
        "enabled": True,
        "fitting_window": (-2.0, 1.0),
        "analysis_window": (0.0, 0.5),
        "buffer_s": 0.1,
        "spline_smoothing": None,
        "spline_order": 3,
        "n_knots": None,
        "trend_pre_window": None,
        "trend_post_window": None,
        "extrapolate": False,
    }
    if feature_config:
        resolved.update(feature_config)
    return resolved


def _safe_array(x):
    arr = np.asarray(x, dtype=float)
    if arr.size == 1 and not np.isfinite(arr[0]):
        return None
    return arr


def _load_dataset_if_present(group, key):
    if group is None or key not in group:
        return None
    return np.asarray(group[key], dtype=float)


def _merge_colors(params, color_overrides=None):
    colors = dict(DEFAULT_INPUT_COMPONENT_COLORS)
    colors.update(params.get("input_component_colors", {}) or {})
    if color_overrides:
        colors.update(color_overrides)
    return colors


def _find_parameter_set_dir(path):
    for parent in [path.parent, *path.parents]:
        if re.fullmatch(r"parameter_set_\d+", parent.name) or re.fullmatch(
            r"parameter_set_posterior_mode__.+", parent.name
        ):
            return parent
    return None


def _parameter_set_id_from_path(path):
    match = re.fullmatch(r"parameter_set_(\d+)", Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def _posterior_mode_label_from_path(path):
    match = re.fullmatch(r"parameter_set_posterior_mode__(.+)", Path(path).name)
    if match is None:
        return None
    return str(match.group(1))


def _parameter_set_is_posterior_mode(path):
    return _posterior_mode_label_from_path(path) is not None


def _parameter_set_is_selected(parameter_dir, selected_parameter_set_ids=None, parameter_set_id_range=None):
    if _parameter_set_is_posterior_mode(parameter_dir):
        return selected_parameter_set_ids is None and parameter_set_id_range is None
    parameter_id = _parameter_set_id_from_path(parameter_dir)
    if parameter_id is None:
        return False
    if selected_parameter_set_ids is not None and parameter_id not in set(selected_parameter_set_ids):
        return False
    if parameter_set_id_range is not None:
        start_id, end_id = parameter_set_id_range
        if parameter_id < int(start_id) or parameter_id > int(end_id):
            return False
    return True


def _progress_iter(iterable, *, desc, show_progress=True):
    items = list(iterable)
    if not show_progress:
        for item in items:
            yield item
        return
    if tqdm is not None:
        progress = tqdm(items, desc=desc)
        for item in progress:
            label = None
            if isinstance(item, tuple) and item and hasattr(item[0], "name"):
                label = item[0].name
            elif hasattr(item, "name"):
                label = item.name
            if label:
                progress.set_postfix_str(label)
            yield item
        return
    total = len(items)
    for index, item in enumerate(items, start=1):
        if isinstance(item, tuple) and item and hasattr(item[0], "name"):
            folder_name = item[0].name
            print(f"[{index}/{total}] {desc}: {folder_name}")
        elif hasattr(item, "name"):
            print(f"[{index}/{total}] {desc}: {item.name}")
        else:
            print(f"[{index}/{total}] {desc}")
        yield item


def _get_event_times(params, driving_inputs):
    event_times = {}
    task_event_times = params.get("task_event_times_s", {})
    if isinstance(task_event_times, dict):
        for key, value in task_event_times.items():
            event_times[key] = float(value)
    if driving_inputs is not None:
        for key in driving_inputs.keys():
            if key.startswith("task_"):
                val = _safe_array(driving_inputs[key])
                if val is not None and val.size:
                    event_times[key.replace("task_", "")] = float(val.flat[0])
    return event_times


def _resolve_zoom_window(params, t_s, event_times, xlim_rel_cue_s=None):
    if xlim_rel_cue_s is not None:
        xlim_arr = np.asarray(xlim_rel_cue_s, dtype=float).reshape(-1)
        if xlim_arr.size != 2 or not (xlim_arr[0] < xlim_arr[1]):
            raise ValueError("xlim_rel_cue_s must contain exactly two increasing values")
        cue_time_s = float(event_times.get("go_nogo_cue_s", 0.0))
        zoom_start_s = cue_time_s + float(xlim_arr[0])
        zoom_end_s = cue_time_s + float(xlim_arr[1])
        zoom_mask = (t_s >= zoom_start_s) & (t_s <= zoom_end_s)
        burst_center_s = event_times.get("burst_center_s")
        burst_sigma_s = float(params.get("input_burst_sigma_ms", 250.0)) / 1000.0
        start_marker_n_sigma = float(params.get("input_burst_start_marker_n_sigma", 2.0))
        burst_start_s = None if burst_center_s is None else float(burst_center_s) - start_marker_n_sigma * burst_sigma_s
        burst_end_s = None if burst_center_s is None else float(burst_center_s) + start_marker_n_sigma * burst_sigma_s
        return {
            "zoom_start_s": float(zoom_start_s),
            "zoom_end_s": float(zoom_end_s),
            "zoom_mask": zoom_mask,
            "burst_center_s": burst_center_s,
            "burst_start_s": burst_start_s,
            "burst_end_s": burst_end_s,
        }

    usable_start_s = float(params["edges_ignore_duration"])
    usable_end_s = float(params["duration_with_ignored_window"] - params["edges_ignore_duration"])
    zoom_duration_s = float(params.get("input_component_diag_zoom_duration_s", 3.0))
    burst_center_s = event_times.get("burst_center_s")
    burst_sigma_s = float(params.get("input_burst_sigma_ms", 250.0)) / 1000.0
    zoom_start_n_sigma = float(params.get("input_burst_zoom_start_n_sigma", 4.0))
    start_marker_n_sigma = float(params.get("input_burst_start_marker_n_sigma", 2.0))

    if burst_center_s is None:
        zoom_start_s = usable_start_s + 0.5 * max(usable_end_s - usable_start_s - zoom_duration_s, 0.0)
        burst_start_s = None
        burst_end_s = None
    else:
        zoom_start_s = float(burst_center_s) - zoom_start_n_sigma * burst_sigma_s
        burst_start_s = float(burst_center_s) - start_marker_n_sigma * burst_sigma_s
        burst_end_s = float(burst_center_s) + start_marker_n_sigma * burst_sigma_s

    zoom_start_s = max(usable_start_s, zoom_start_s)
    zoom_end_s = zoom_start_s + zoom_duration_s
    if zoom_end_s > usable_end_s:
        zoom_end_s = usable_end_s
        zoom_start_s = max(usable_start_s, zoom_end_s - zoom_duration_s)

    zoom_mask = (t_s >= zoom_start_s) & (t_s <= zoom_end_s)
    return {
        "zoom_start_s": float(zoom_start_s),
        "zoom_end_s": float(zoom_end_s),
        "zoom_mask": zoom_mask,
        "burst_center_s": burst_center_s,
        "burst_start_s": burst_start_s,
        "burst_end_s": burst_end_s,
    }


def _auto_limits(data, zero_floor=False, pad_frac=0.08):
    data = np.asarray(data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return (0.0, 1.0) if zero_floor else (-1.0, 1.0)
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if zero_floor:
        y_min = 0.0
    span = y_max - y_min
    pad = max(0.05, span * pad_frac) if span > 1e-12 else max(0.05, abs(y_max) * pad_frac, 0.05)
    return y_min - (0.0 if zero_floor else pad), y_max + pad


def _draw_shared_markers(ax, event_times, colors, style, burst_center_s=None, burst_start_s=None, burst_end_s=None,
                         time_offset_s=0.0, show_burst=True):
    if show_burst and burst_start_s is not None and burst_end_s is not None:
        ax.axvspan(burst_start_s - time_offset_s, burst_end_s - time_offset_s, color=colors["burst"], alpha=style["burst_epoch_alpha"], zorder=0)
    if show_burst and burst_center_s is not None:
        ax.axvline(burst_center_s - time_offset_s, color=colors["burst"], ls="--", lw=style["burst_center_lw"], alpha=style["burst_center_alpha"], zorder=1)
    get_ready_s = event_times.get("get_ready_cue_s")
    if get_ready_s is not None:
        ax.axvline(get_ready_s - time_offset_s, color=colors["cue"], ls="--", lw=style["get_ready_lw"], alpha=style["get_ready_alpha"], zorder=1)
    go_cue_s = event_times.get("go_nogo_cue_s")
    if go_cue_s is not None:
        ax.axvline(go_cue_s - time_offset_s, color=colors["cue"], ls="-", lw=style["go_cue_lw"], alpha=style["go_cue_alpha"], zorder=1)


def _stack_runs(runs, key):
    return np.stack([np.asarray(run[key], dtype=float) for run in runs], axis=0)


def _validate_same_shape(runs, key):
    shapes = {np.asarray(run[key]).shape for run in runs}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent shapes for '{key}' across repeats: {sorted(shapes)}")


def _interp_to_reference(time_s, signal, ref_time_s):
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    ref_time_s = np.asarray(ref_time_s, dtype=float)
    if time_s.shape == ref_time_s.shape and np.allclose(time_s, ref_time_s):
        return signal
    return np.interp(ref_time_s, time_s, signal, left=np.nan, right=np.nan)


def _format_panel_title(title):
    title = str(title)
    match = re.match(r"^(.*?)(\s*\([^()]+\))$", title)
    if match is None:
        return title
    main = match.group(1).rstrip()
    suffix = match.group(2).strip()
    return f"{main}\n{suffix}"


def _annotate_feature_panel(ax, time_s_rel, mean_signal, signal_color, feature_result, feature_config, style, feature_unit_label):
    analysis_start_s, analysis_end_s = map(float, feature_config["analysis_window"])
    trend_pre_window = feature_config.get("trend_pre_window")
    trend_post_window = feature_config.get("trend_post_window")
    if trend_pre_window is None:
        fitting_start_s, fitting_end_s = map(float, feature_config["fitting_window"])
        trend_pre_window = (fitting_start_s, min(0.0, fitting_end_s))
    if trend_post_window is None:
        fitting_start_s, fitting_end_s = map(float, feature_config["fitting_window"])
        trend_post_window = (max(0.0, fitting_start_s), fitting_end_s)
    plot_start_s = float(np.min(time_s_rel))
    plot_end_s = float(np.max(time_s_rel))
    if analysis_start_s > plot_start_s:
        ax.axvspan(
            plot_start_s,
            analysis_start_s,
            color=style["feature_outside_color"],
            alpha=style["feature_outside_alpha"],
            zorder=0,
        )
    if analysis_end_s < plot_end_s:
        ax.axvspan(
            analysis_end_s,
            plot_end_s,
            color=style["feature_outside_color"],
            alpha=style["feature_outside_alpha"],
            zorder=0,
        )

    fit_mask = (
        (feature_result.t_fit_window >= plot_start_s)
        & (feature_result.t_fit_window <= plot_end_s)
    )
    if np.any(fit_mask):
        ax.plot(
            feature_result.t_fit_window[fit_mask],
            feature_result.baseline_fit_window[fit_mask],
            color=style.get("feature_baseline_backdrop_color", "#BDBDBD"),
            lw=float(style.get("feature_baseline_backdrop_lw", 3.0)),
            ls="-",
            alpha=float(style.get("feature_baseline_backdrop_alpha", 0.85)),
            zorder=4.5,
        )

    for baseline_window, linestyle in ((trend_pre_window, "--"), (trend_post_window, ":")):
        baseline_window_start_s, baseline_window_end_s = map(float, baseline_window)
        baseline_mask = (
            (feature_result.t_fit_window >= max(plot_start_s, baseline_window_start_s))
            & (feature_result.t_fit_window <= min(plot_end_s, baseline_window_end_s))
        )
        if np.any(baseline_mask):
            ax.plot(
                feature_result.t_fit_window[baseline_mask],
                feature_result.baseline_fit_window[baseline_mask],
                color=style["feature_baseline_color"],
                lw=style["feature_baseline_lw"],
                ls=linestyle,
                alpha=style["feature_baseline_alpha"],
                zorder=5,
            )

    analysis_mask = (time_s_rel >= analysis_start_s) & (time_s_rel <= analysis_end_s)
    if np.any(analysis_mask):
        analysis_mean = float(np.mean(mean_signal[analysis_mask]))
        ax.hlines(
            analysis_mean,
            xmin=analysis_start_s,
            xmax=analysis_end_s,
            color=signal_color,
            lw=style["feature_analysis_mean_lw"],
            ls=style["feature_analysis_mean_ls"],
            alpha=style["feature_analysis_mean_alpha"],
            zorder=5,
        )

    def _fmt_feature_value(value):
        if not np.isfinite(value) or abs(float(value)) < 0.005:
            return "~=0"
        return f"{float(value):.2f}"

    def _fmt_signed_delta(value):
        if not np.isfinite(value) or abs(float(value)) < 0.005:
            return "~=0"
        return f"{float(value):+.2f}"

    ax.set_title(
        (
            f"Mean burst window residuals: {_fmt_feature_value(feature_result.feature_mean_residual)} {feature_unit_label}".rstrip()
            + "\n"
            + f"Post VS pre mean difference: {_fmt_feature_value(feature_result.trend_post_minus_pre)} {feature_unit_label}".rstrip()
            + "\n"
            + f"Spline fit R²: {_fmt_feature_value(feature_result.fit_r2)} ({_fmt_signed_delta(feature_result.fit_r2_gain_over_linear)} over linear fit)"
        ),
        loc="left",
        fontsize=style["feature_text_fontsize"],
        fontweight=style["feature_text_weight"],
        color=signal_color,
        pad=8,
    )


def _append_limits(limit_map, key, data):
    data = np.asarray(data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    curr = limit_map.get(key)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if curr is None:
        limit_map[key] = [lo, hi]
    else:
        curr[0] = min(curr[0], lo)
        curr[1] = max(curr[1], hi)


def _finalize_limits(limit_map):
    finalized = {}
    for key, (lo, hi) in limit_map.items():
        finalized[key] = _auto_limits(np.array([lo, hi], dtype=float), zero_floor=False)
    return finalized


def _warn_unreadable_h5(sim_h5_path, exc):
    warnings.warn(
        f"Skipping unreadable HDF5 file: {sim_h5_path}\nReason: {exc}",
        stacklevel=2,
    )


def load_parameter_set_run(run_dir, color_overrides=None):
    run_dir = Path(run_dir)
    sim_json = run_dir / "sim_parameters.json"
    sim_h5 = run_dir / "simulation_output.h5"
    if not sim_json.exists() or not sim_h5.exists():
        return None

    with sim_json.open("r", encoding="utf-8") as f:
        params = json.load(f)

    try:
        with h5py.File(sim_h5, "r") as f:
            input_grp = f["input"] if "input" in f else None
            input_components = f["input_components"] if "input_components" in f else None
            cst_diag = f["cst_diagnostics"] if "cst_diagnostics" in f else None
            sync_diag = f["sync_diagnostics"] if "sync_diagnostics" in f else None
            driving_inputs = f["driving_inputs"] if "driving_inputs" in f else None

            saved_fsamp_hz = float(
                (input_grp.attrs.get("saved_fsamp_Hz") if input_grp is not None else None)
                or params["fsamp"]
            )
            common_input = np.asarray(f["input"]["common_input"], dtype=float)
            common_input = np.squeeze(common_input, axis=0) if common_input.ndim == 2 and common_input.shape[0] == 1 else np.asarray(common_input, dtype=float)
            t_s = np.arange(common_input.shape[-1], dtype=float) / float(saved_fsamp_hz)

            lf_signal = _load_dataset_if_present(input_components, "lf_final_with_burst")
            if lf_signal is None:
                lf_signal = _load_dataset_if_present(input_components, "lf_final")
            alpha_signal = _load_dataset_if_present(input_components, "alpha_final_with_burst")
            if alpha_signal is None:
                alpha_signal = _load_dataset_if_present(input_components, "alpha_final")
            beta_signal = _load_dataset_if_present(input_components, "beta_final_with_burst")
            if beta_signal is None:
                beta_signal = _load_dataset_if_present(input_components, "beta_final")
            common_signal = _load_dataset_if_present(input_components, "common_input_with_burst")
            if common_signal is None:
                common_signal = common_input
            if lf_signal is None or alpha_signal is None or beta_signal is None:
                raise KeyError(
                    f"{sim_h5} does not contain the saved final input-component traces needed for batch summaries."
                )

            lf_env = _load_dataset_if_present(input_components, "lf_effective_envelope_nA")
            alpha_env = _load_dataset_if_present(input_components, "alpha_effective_envelope_nA")
            beta_env = _load_dataset_if_present(input_components, "beta_effective_envelope_nA")
            lf_burst = _load_dataset_if_present(input_components, "lf_burst_kernel_nA")
            alpha_burst = _load_dataset_if_present(input_components, "alpha_burst_kernel_nA")
            beta_burst = _load_dataset_if_present(input_components, "beta_burst_kernel_nA")
            lf_trend = _load_dataset_if_present(input_components, "lf_trend_kernel_nA")
            alpha_trend = _load_dataset_if_present(input_components, "alpha_trend_kernel_nA")
            beta_trend = _load_dataset_if_present(input_components, "beta_trend_kernel_nA")

            if driving_inputs is not None:
                if lf_env is None:
                    lf_env = _load_dataset_if_present(driving_inputs, "lf_effective_envelope_nA")
                if alpha_env is None:
                    alpha_env = _load_dataset_if_present(driving_inputs, "alpha_effective_envelope_nA")
                if beta_env is None:
                    beta_env = _load_dataset_if_present(driving_inputs, "beta_effective_envelope_nA")
                if lf_burst is None:
                    lf_burst = _load_dataset_if_present(driving_inputs, "lf_burst_kernel_nA")
                if alpha_burst is None:
                    alpha_burst = _load_dataset_if_present(driving_inputs, "alpha_burst_kernel_nA")
                if beta_burst is None:
                    beta_burst = _load_dataset_if_present(driving_inputs, "beta_burst_kernel_nA")
                if lf_trend is None:
                    lf_trend = _load_dataset_if_present(driving_inputs, "lf_trend_kernel_nA")
                if alpha_trend is None:
                    alpha_trend = _load_dataset_if_present(driving_inputs, "alpha_trend_kernel_nA")
                if beta_trend is None:
                    beta_trend = _load_dataset_if_present(driving_inputs, "beta_trend_kernel_nA")

            if lf_env is None:
                lf_env = np.zeros_like(lf_signal if lf_signal is not None else common_signal)
            if alpha_env is None:
                alpha_env = np.abs(hilbert(alpha_signal)) if alpha_signal is not None else np.zeros_like(common_signal)
            if beta_env is None:
                beta_env = np.abs(hilbert(beta_signal)) if beta_signal is not None else np.zeros_like(common_signal)
            if lf_burst is None:
                lf_burst = np.zeros_like(common_signal)
            if alpha_burst is None:
                alpha_burst = np.zeros_like(common_signal)
            if beta_burst is None:
                beta_burst = np.zeros_like(common_signal)
            if lf_trend is None:
                lf_trend = np.zeros_like(common_signal)
            if alpha_trend is None:
                alpha_trend = np.zeros_like(common_signal)
            if beta_trend is None:
                beta_trend = np.zeros_like(common_signal)

            event_times = _get_event_times(params, driving_inputs)
            cst_modulation_mode = str(params.get("cst_modulation_mode", "zscore")).lower()
            sync_display_mode = str(params.get("sync_trace_display_mode", "absolute")).lower()

            cst_global = _load_dataset_if_present(cst_diag, "cst_normalized") if cst_diag is not None else None
            cst_lf = _load_dataset_if_present(cst_diag, "cst_lf") if cst_diag is not None else None
            cst_hann = _load_dataset_if_present(cst_diag, "cst_hann") if cst_diag is not None else None
            cst_alpha = _load_dataset_if_present(cst_diag, "cst_alpha") if cst_diag is not None else None
            cst_beta = _load_dataset_if_present(cst_diag, "cst_beta") if cst_diag is not None else None
            cst_alpha_mod = _load_dataset_if_present(
                cst_diag,
                "cst_alpha_envelope_z" if cst_modulation_mode == "zscore" else "cst_alpha_envelope_pct",
            ) if cst_diag is not None else None
            cst_beta_mod = _load_dataset_if_present(
                cst_diag,
                "cst_beta_envelope_z" if cst_modulation_mode == "zscore" else "cst_beta_envelope_pct",
            ) if cst_diag is not None else None

            sync_time_rel = _load_dataset_if_present(sync_diag, "t_sync") if sync_diag is not None else None
            sync_trace = _load_dataset_if_present(
                sync_diag,
                "sync_trace" if sync_display_mode == "absolute" else "sync_trace_zscore",
            ) if sync_diag is not None else None
            if sync_time_rel is not None:
                go_cue_s = float(event_times.get("go_nogo_cue_s", 0.0))
                sync_time_abs = sync_time_rel + go_cue_s
            else:
                sync_time_abs = None
    except OSError as exc:
        _warn_unreadable_h5(sim_h5, exc)
        return None

    return {
        "run_dir": run_dir,
        "params": params,
        "saved_trace_fsamp_hz": saved_fsamp_hz,
        "colors": _merge_colors(params, color_overrides=color_overrides),
        "t_s": t_s,
        "event_times": event_times,
        "lf_signal": np.asarray(lf_signal, dtype=float),
        "alpha_signal": np.asarray(alpha_signal, dtype=float),
        "beta_signal": np.asarray(beta_signal, dtype=float),
        "common_signal": np.asarray(common_signal, dtype=float),
        "lf_env": np.asarray(lf_env, dtype=float),
        "alpha_env": np.asarray(alpha_env, dtype=float),
        "beta_env": np.asarray(beta_env, dtype=float),
        "lf_burst": np.asarray(lf_burst, dtype=float),
        "alpha_burst": np.asarray(alpha_burst, dtype=float),
        "beta_burst": np.asarray(beta_burst, dtype=float),
        "lf_trend": np.asarray(lf_trend, dtype=float),
        "alpha_trend": np.asarray(alpha_trend, dtype=float),
        "beta_trend": np.asarray(beta_trend, dtype=float),
        "cst_global": None if cst_global is None else np.asarray(cst_global, dtype=float),
        "cst_lf": None if cst_lf is None else np.asarray(cst_lf, dtype=float),
        "cst_hann": None if cst_hann is None else np.asarray(cst_hann, dtype=float),
        "cst_alpha": None if cst_alpha is None else np.asarray(cst_alpha, dtype=float),
        "cst_beta": None if cst_beta is None else np.asarray(cst_beta, dtype=float),
        "cst_alpha_mod": None if cst_alpha_mod is None else np.asarray(cst_alpha_mod, dtype=float),
        "cst_beta_mod": None if cst_beta_mod is None else np.asarray(cst_beta_mod, dtype=float),
        "sync_time_abs": None if sync_time_abs is None else np.asarray(sync_time_abs, dtype=float),
        "sync_trace": None if sync_trace is None else np.asarray(sync_trace, dtype=float),
        "cst_modulation_mode": cst_modulation_mode,
        "sync_display_mode": sync_display_mode,
    }


def collect_parameter_set_runs(root_dir, color_overrides=None, selected_parameter_set_ids=None, parameter_set_id_range=None):
    root_dir = Path(root_dir)
    grouped = {}
    for h5_path in sorted(root_dir.rglob("simulation_output.h5")):
        run_dir = h5_path.parent
        parameter_dir = _find_parameter_set_dir(h5_path)
        if parameter_dir is None:
            continue
        if not _parameter_set_is_selected(
            parameter_dir,
            selected_parameter_set_ids=selected_parameter_set_ids,
            parameter_set_id_range=parameter_set_id_range,
        ):
            continue
        run = load_parameter_set_run(run_dir, color_overrides=color_overrides)
        if run is None:
            continue
        grouped.setdefault(parameter_dir, []).append(run)
    return grouped


def _plot_input_panel(ax, zoom_t, mean_signal, std_signal, indiv_signals, mean_env, mean_burst, mean_trend, color, colors,
                      title, burst_center_s, burst_start_s, burst_end_s, event_times,
                      show_individual_traces, show_std_band, style, show_envelope=True, reference_std_uA=None,
                      fixed_ylim=None, time_offset_s=0.0):
    mean_plot = mean_signal / 1000.0
    std_plot = std_signal / 1000.0
    env_pos = mean_env / 1000.0
    env_neg = -mean_env / 1000.0
    modulation_plot = (mean_burst + mean_trend) / 1000.0
    indiv_plot = indiv_signals / 1000.0

    _draw_shared_markers(ax, event_times, colors, style, burst_center_s, burst_start_s, burst_end_s, time_offset_s=time_offset_s, show_burst=True)
    if show_individual_traces and indiv_plot.size:
        for trace in indiv_plot:
            ax.plot(zoom_t, trace, color=color, lw=style["individual_trace_lw"], alpha=style["individual_trace_alpha"], zorder=2)
    if show_std_band:
        ax.fill_between(zoom_t, mean_plot - std_plot, mean_plot + std_plot, color=color, alpha=style["std_band_alpha"], zorder=3)
    if reference_std_uA is not None and np.isfinite(reference_std_uA) and reference_std_uA > 0:
        ax.axhline(reference_std_uA, color=color, lw=style["reference_std_lw"], ls="--", alpha=style["reference_std_alpha"], zorder=1)
        ax.axhline(-reference_std_uA, color=color, lw=style["reference_std_lw"], ls="--", alpha=style["reference_std_alpha"], zorder=1)
    ax.plot(zoom_t, mean_plot, color=color, lw=style["mean_trace_lw"], zorder=4)
    if show_envelope:
        ax.plot(zoom_t, env_pos, color=colors["envelope"], lw=style["envelope_lw"], alpha=style["envelope_alpha"], zorder=3)
        ax.plot(zoom_t, env_neg, color=colors["envelope"], lw=style["envelope_lw"], alpha=style["envelope_alpha"], zorder=3)
    if np.any(np.abs(mean_burst + mean_trend) > 1e-12):
        ax.plot(zoom_t, modulation_plot, color=colors["burst"], lw=style["burst_kernel_lw"], ls="--", alpha=style["burst_kernel_alpha"], zorder=5)
    ylim_data = [mean_plot]
    if show_individual_traces and indiv_plot.size:
        ylim_data.append(indiv_plot.reshape(-1))
    if show_std_band:
        ylim_data.extend([mean_plot - std_plot, mean_plot + std_plot])
    if np.any(np.abs(mean_burst + mean_trend) > 1e-12):
        ylim_data.append(modulation_plot)
    if reference_std_uA is not None and np.isfinite(reference_std_uA) and reference_std_uA > 0:
        ylim_data.append(np.array([reference_std_uA, -reference_std_uA], dtype=float))
    ymin, ymax = fixed_ylim if fixed_ylim is not None else _auto_limits(np.concatenate(ylim_data) if ylim_data else np.zeros(1), zero_floor=False)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Input (µA)")
    ax.set_title(_format_panel_title(title), fontsize=10, loc="right")
    ax.grid(alpha=0.25)


def _plot_raw_panel(ax, zoom_t, mean_signal, std_signal, indiv_signals, color, ylabel, title,
                    burst_center_s, burst_start_s, burst_end_s, event_times,
                    show_individual_traces, show_std_band, colors, style, reference_std=None, fixed_ylim=None,
                    reference_center=0.0, time_offset_s=0.0, show_burst=False):
    _draw_shared_markers(ax, event_times, colors, style, burst_center_s, burst_start_s, burst_end_s, time_offset_s=time_offset_s, show_burst=show_burst)
    if show_individual_traces and indiv_signals.size:
        for trace in indiv_signals:
            ax.plot(zoom_t, trace, color=color, lw=style["individual_trace_lw"], alpha=style["individual_trace_alpha"], zorder=2)
    if show_std_band:
        ax.fill_between(zoom_t, mean_signal - std_signal, mean_signal + std_signal, color=color, alpha=style["std_band_alpha"], zorder=3)
    if reference_std is not None and np.isfinite(reference_std) and reference_std > 0:
        ax.axhline(reference_center + reference_std, color=color, lw=style["reference_std_lw"], ls="--", alpha=style["reference_std_alpha"], zorder=1)
        ax.axhline(reference_center - reference_std, color=color, lw=style["reference_std_lw"], ls="--", alpha=style["reference_std_alpha"], zorder=1)
    ax.plot(zoom_t, mean_signal, color=color, lw=style["mean_trace_lw"], zorder=4)
    ylim_parts = [mean_signal]
    if indiv_signals.size:
        ylim_parts.append(indiv_signals.reshape(-1))
    if reference_std is not None and np.isfinite(reference_std) and reference_std > 0:
        ylim_parts.append(np.array([reference_center + reference_std, reference_center - reference_std], dtype=float))
    ymin, ymax = fixed_ylim if fixed_ylim is not None else _auto_limits(np.concatenate(ylim_parts), zero_floor=False)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel)
    ax.set_title(_format_panel_title(title), fontsize=10, loc="right")
    ax.grid(alpha=0.25)


def _compute_global_panel_limits(grouped_runs, show_individual_traces=True, show_std_band=True, show_reference_std_lines=True,
                                 xlim_rel_cue_s=None):
    limits = {}
    for _, runs in grouped_runs.items():
        if not runs:
            continue
        ref = runs[0]
        params = ref["params"]
        t_s = np.asarray(ref["t_s"], dtype=float)
        event_times = dict(ref["event_times"])
        zoom = _resolve_zoom_window(params, t_s, event_times, xlim_rel_cue_s=xlim_rel_cue_s)
        zoom_mask = zoom["zoom_mask"]

        def _zoom_stack(key):
            return _stack_runs(runs, key)[:, zoom_mask]

        def _mean_full_std(key):
            return float(np.mean([np.std(np.asarray(run[key], dtype=float)) for run in runs]))

        for panel_key, signal_key, env_key, burst_key, trend_key in (
            ("input_lf", "lf_signal", "lf_env", "lf_burst", "lf_trend"),
            ("input_alpha", "alpha_signal", "alpha_env", "alpha_burst", "alpha_trend"),
            ("input_beta", "beta_signal", "beta_env", "beta_burst", "beta_trend"),
            ("input_common", "common_signal", None, None, None),
        ):
            sig = _zoom_stack(signal_key) / 1000.0
            if show_individual_traces:
                _append_limits(limits, panel_key, sig)
            else:
                _append_limits(limits, panel_key, np.mean(sig, axis=0))
            if show_std_band:
                _append_limits(limits, panel_key, np.mean(sig, axis=0) - np.std(sig, axis=0))
                _append_limits(limits, panel_key, np.mean(sig, axis=0) + np.std(sig, axis=0))
            if burst_key is not None:
                burst = _zoom_stack(burst_key) / 1000.0
                _append_limits(limits, panel_key, burst)
            if trend_key is not None:
                trend = _zoom_stack(trend_key) / 1000.0
                _append_limits(limits, panel_key, trend)
            if show_reference_std_lines and panel_key != "input_common":
                ref_std = _mean_full_std(signal_key) / 1000.0
                _append_limits(limits, panel_key, np.array([ref_std, -ref_std], dtype=float))

        for panel_key, key in (
            ("cst_lf", "cst_lf"),
            ("cst_alpha", "cst_alpha"),
            ("cst_beta", "cst_beta"),
            ("mod_alpha", "cst_alpha_mod"),
            ("mod_beta", "cst_beta_mod"),
        ):
            available = [run[key] is not None for run in runs]
            if not any(available):
                continue
            traces = np.stack([np.asarray(run[key], dtype=float)[zoom_mask] for run in runs], axis=0)
            if show_individual_traces:
                _append_limits(limits, panel_key, traces)
            else:
                _append_limits(limits, panel_key, np.mean(traces, axis=0))
            if show_std_band:
                _append_limits(limits, panel_key, np.mean(traces, axis=0) - np.std(traces, axis=0))
                _append_limits(limits, panel_key, np.mean(traces, axis=0) + np.std(traces, axis=0))
            if show_reference_std_lines and panel_key in ("cst_lf", "cst_alpha", "cst_beta"):
                ref_std = float(np.mean([np.std(np.asarray(run[key], dtype=float)) for run in runs]))
                ref_center = float(np.mean(np.mean(traces, axis=0))) if panel_key == "cst_lf" else 0.0
                _append_limits(limits, panel_key, np.array([ref_center + ref_std, ref_center - ref_std], dtype=float))

        sync_available = [run["sync_trace"] is not None and run["sync_time_abs"] is not None for run in runs]
        if any(sync_available):
            ref_sync_time = None
            traces = []
            for run in runs:
                if run["sync_trace"] is None or run["sync_time_abs"] is None:
                    continue
                if ref_sync_time is None:
                    ref_sync_time = np.asarray(run["sync_time_abs"], dtype=float)
                traces.append(_interp_to_reference(run["sync_time_abs"], run["sync_trace"], ref_sync_time))
            traces = np.stack(traces, axis=0)
            sync_mask = (ref_sync_time >= zoom["zoom_start_s"]) & (ref_sync_time <= zoom["zoom_end_s"])
            traces = traces[:, sync_mask]
            if show_individual_traces:
                _append_limits(limits, "sync", traces)
            else:
                _append_limits(limits, "sync", np.nanmean(traces, axis=0))
            if show_std_band:
                _append_limits(limits, "sync", np.nanmean(traces, axis=0) - np.nanstd(traces, axis=0))
                _append_limits(limits, "sync", np.nanmean(traces, axis=0) + np.nanstd(traces, axis=0))
    return _finalize_limits(limits)


def plot_parameter_set_summary(parameter_set_dir, runs, output_dir=None, show_individual_traces=True, show_std_band=True,
                               show_reference_std_lines=True, fixed_panel_limits=None, plot_style=None,
                               xlim_rel_cue_s=None, feature_config=None, window_overlay_specs=None, save_figure=True):
    if not runs:
        raise ValueError("No runs were provided for this parameter set")

    ref = runs[0]
    params = ref["params"]
    colors = ref["colors"]
    style = _resolve_plot_style(plot_style)
    feature_config = _resolve_feature_config(feature_config)
    t_s = np.asarray(ref["t_s"], dtype=float)
    event_times = dict(ref["event_times"])
    zoom = _resolve_zoom_window(params, t_s, event_times, xlim_rel_cue_s=xlim_rel_cue_s)
    zoom_mask = zoom["zoom_mask"]
    cue_time_s = float(event_times.get("go_nogo_cue_s", 0.0))
    zoom_t = t_s[zoom_mask] - cue_time_s
    display_window_rel_cue_s = (float(np.min(zoom_t)), float(np.max(zoom_t)))
    features = {}

    for key in ("lf_signal", "alpha_signal", "beta_signal", "common_signal", "lf_env", "alpha_env", "beta_env", "lf_burst", "alpha_burst", "beta_burst", "lf_trend", "alpha_trend", "beta_trend"):
        _validate_same_shape(runs, key)

    fig, axes = plt.subplots(4, 3, figsize=style["figure_size"], dpi=style["figure_dpi"], sharex=True)

    def _zoom_stack(key):
        return _stack_runs(runs, key)[:, zoom_mask]

    def _mean_full_std(key):
        return float(np.mean([np.std(np.asarray(run[key], dtype=float)) for run in runs]))

    _plot_input_panel(
        axes[0, 0], zoom_t,
        mean_signal=np.mean(_zoom_stack("lf_signal"), axis=0),
        std_signal=np.std(_zoom_stack("lf_signal"), axis=0),
        indiv_signals=_zoom_stack("lf_signal"),
        mean_env=np.mean(_zoom_stack("lf_env"), axis=0),
        mean_burst=np.mean(_zoom_stack("lf_burst"), axis=0),
        mean_trend=np.mean(_zoom_stack("lf_trend"), axis=0),
        color=colors["lf"], colors=colors, title="LF input",
        burst_center_s=zoom["burst_center_s"], burst_start_s=zoom["burst_start_s"], burst_end_s=zoom["burst_end_s"],
        event_times=event_times, show_individual_traces=show_individual_traces, show_std_band=show_std_band, style=style,
        show_envelope=False,
        reference_std_uA=(_mean_full_std("lf_signal") / 1000.0) if show_reference_std_lines else None,
        fixed_ylim=None if fixed_panel_limits is None else fixed_panel_limits.get("input_lf"),
        time_offset_s=cue_time_s,
    )
    _draw_sim_window_overlays(axes[0, 0], params, feature_config, display_window_rel_cue_s, window_overlay_specs)
    _plot_input_panel(
        axes[1, 0], zoom_t,
        mean_signal=np.mean(_zoom_stack("alpha_signal"), axis=0),
        std_signal=np.std(_zoom_stack("alpha_signal"), axis=0),
        indiv_signals=_zoom_stack("alpha_signal"),
        mean_env=np.mean(_zoom_stack("alpha_env"), axis=0),
        mean_burst=np.mean(_zoom_stack("alpha_burst"), axis=0),
        mean_trend=np.mean(_zoom_stack("alpha_trend"), axis=0),
        color=colors["alpha"], colors=colors, title="Alpha input",
        burst_center_s=zoom["burst_center_s"], burst_start_s=zoom["burst_start_s"], burst_end_s=zoom["burst_end_s"],
        event_times=event_times, show_individual_traces=show_individual_traces, show_std_band=show_std_band, style=style,
        show_envelope=True,
        reference_std_uA=(_mean_full_std("alpha_signal") / 1000.0) if show_reference_std_lines else None,
        fixed_ylim=None if fixed_panel_limits is None else fixed_panel_limits.get("input_alpha"),
        time_offset_s=cue_time_s,
    )
    _draw_sim_window_overlays(axes[1, 0], params, feature_config, display_window_rel_cue_s, window_overlay_specs)
    _plot_input_panel(
        axes[2, 0], zoom_t,
        mean_signal=np.mean(_zoom_stack("beta_signal"), axis=0),
        std_signal=np.std(_zoom_stack("beta_signal"), axis=0),
        indiv_signals=_zoom_stack("beta_signal"),
        mean_env=np.mean(_zoom_stack("beta_env"), axis=0),
        mean_burst=np.mean(_zoom_stack("beta_burst"), axis=0),
        mean_trend=np.mean(_zoom_stack("beta_trend"), axis=0),
        color=colors["beta"], colors=colors, title="Beta input",
        burst_center_s=zoom["burst_center_s"], burst_start_s=zoom["burst_start_s"], burst_end_s=zoom["burst_end_s"],
        event_times=event_times, show_individual_traces=show_individual_traces, show_std_band=show_std_band, style=style,
        show_envelope=True,
        reference_std_uA=(_mean_full_std("beta_signal") / 1000.0) if show_reference_std_lines else None,
        fixed_ylim=None if fixed_panel_limits is None else fixed_panel_limits.get("input_beta"),
        time_offset_s=cue_time_s,
    )
    _draw_sim_window_overlays(axes[2, 0], params, feature_config, display_window_rel_cue_s, window_overlay_specs)
    _plot_input_panel(
        axes[3, 0], zoom_t,
        mean_signal=np.mean(_zoom_stack("common_signal"), axis=0),
        std_signal=np.std(_zoom_stack("common_signal"), axis=0),
        indiv_signals=_zoom_stack("common_signal"),
        mean_env=np.zeros_like(zoom_t),
        mean_burst=np.zeros_like(zoom_t),
        mean_trend=np.zeros_like(zoom_t),
        color=colors["common"], colors=colors, title="Delivered common input",
        burst_center_s=zoom["burst_center_s"], burst_start_s=zoom["burst_start_s"], burst_end_s=zoom["burst_end_s"],
        event_times=event_times, show_individual_traces=show_individual_traces, show_std_band=show_std_band, style=style,
        show_envelope=False,
        reference_std_uA=None,
        fixed_ylim=None if fixed_panel_limits is None else fixed_panel_limits.get("input_common"),
        time_offset_s=cue_time_s,
    )
    _draw_sim_window_overlays(axes[3, 0], params, feature_config, display_window_rel_cue_s, window_overlay_specs)

    def _plot_cst_key(ax, key, color, title, reference_std=None, ylabel="spikes/s/MU", panel_limit_key=None):
        available = [run[key] is not None for run in runs]
        if not any(available):
            ax.text(0.5, 0.5, "Not saved\n(compact output)", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="0.4")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return
        traces = np.stack([np.asarray(run[key], dtype=float)[zoom_mask] for run in runs], axis=0)
        mean_traces = np.mean(traces, axis=0)
        reference_center = float(np.mean(mean_traces)) if key == "cst_lf" else 0.0
        _plot_raw_panel(
            ax, zoom_t, mean_traces, np.std(traces, axis=0), traces,
            color=color, ylabel=ylabel, title=title,
            burst_center_s=zoom["burst_center_s"], burst_start_s=zoom["burst_start_s"], burst_end_s=zoom["burst_end_s"],
            event_times=event_times, show_individual_traces=show_individual_traces, show_std_band=show_std_band, colors=colors, style=style,
            reference_std=reference_std,
            fixed_ylim=None if fixed_panel_limits is None else fixed_panel_limits.get(panel_limit_key or key),
            reference_center=reference_center,
            time_offset_s=cue_time_s,
            show_burst=False,
        )
        _draw_sim_window_overlays(ax, params, feature_config, display_window_rel_cue_s, window_overlay_specs)
        if feature_config["enabled"] and key in ("cst_lf", "cst_alpha_mod", "cst_beta_mod", "sync"):
            try:
                full_time_rel = np.asarray(ref["t_s"], dtype=float) - cue_time_s
                full_mean_signal = np.nanmean(np.stack([np.asarray(run[key], dtype=float) for run in runs], axis=0), axis=0)
                feature_result = fit_baseline_spline_and_extract_feature(
                    full_time_rel,
                    full_mean_signal,
                    fitting_window=feature_config["fitting_window"],
                    analysis_window=feature_config["analysis_window"],
                    buffer_s=feature_config["buffer_s"],
                    spline_smoothing=feature_config["spline_smoothing"],
                    spline_order=feature_config["spline_order"],
                    n_knots=feature_config["n_knots"],
                    trend_pre_window=feature_config.get("trend_pre_window"),
                    trend_post_window=feature_config.get("trend_post_window"),
                    extrapolate=feature_config["extrapolate"],
                    return_debug_info=True,
                )
                if key == "cst_lf":
                    feature_unit_label = "spikes/s/MU"
                elif key in ("cst_alpha_mod", "cst_beta_mod"):
                    feature_unit_label = "z"
                else:
                    feature_unit_label = "a.u."
                _annotate_feature_panel(ax, zoom_t, mean_traces, color, feature_result, feature_config, style, feature_unit_label)
                features[f"{key}_feature_mean_residual"] = feature_result.feature_mean_residual
                features[f"{key}_feature_signed_area"] = feature_result.feature_signed_area
                features[f"{key}_fit_r2"] = feature_result.fit_r2
                features[f"{key}_fit_r2_gain_over_linear"] = feature_result.fit_r2_gain_over_linear
                features[f"{key}_trend_post_minus_pre"] = feature_result.trend_post_minus_pre
            except Exception as exc:
                warnings.warn(f"Could not extract baseline feature for {Path(parameter_set_dir).name} / {key}: {exc}")

    _plot_cst_key(
        axes[0, 1], "cst_lf", colors["lf"],
        f"CST low-pass ({params['lf_band_hz'][0]:.2f}-{params['lf_band_hz'][1]:.2f} Hz)",
        reference_std=_mean_full_std("cst_lf") if show_reference_std_lines else None,
    )
    _plot_cst_key(
        axes[1, 1], "cst_alpha", colors["alpha"],
        f"CST band-pass alpha ({params['alpha_band_hz'][0]:.1f}-{params['alpha_band_hz'][1]:.1f} Hz)",
        reference_std=_mean_full_std("cst_alpha") if show_reference_std_lines else None,
    )
    _plot_cst_key(
        axes[2, 1], "cst_beta", colors["beta"],
        f"CST band-pass beta ({params['beta_band_hz'][0]:.1f}-{params['beta_band_hz'][1]:.1f} Hz)",
        reference_std=_mean_full_std("cst_beta") if show_reference_std_lines else None,
    )
    axes[3, 1].axis("off")

    axes[0, 2].axis("off")
    legend_handles = [
        Line2D([0], [0], color=colors["cue"], lw=style["get_ready_lw"], ls="--", label="Get-ready cue"),
        Line2D([0], [0], color=colors["cue"], lw=style["go_cue_lw"], ls="-", label="Go cue"),
        Patch(facecolor=colors["burst"], edgecolor="none", alpha=style["burst_epoch_alpha"], label="Burst epoch"),
        Line2D([0], [0], color=colors["burst"], lw=style["burst_center_lw"], ls="--", label="Burst center"),
        Line2D([0], [0], color=colors["burst"], lw=style["burst_kernel_lw"], ls="--", alpha=style["burst_kernel_alpha"], label="Burst + trend modulation"),
        Line2D([0], [0], color=colors["common"], lw=style["mean_trace_lw"], label="Mean trace"),
        Patch(facecolor=style["feature_outside_color"], edgecolor="none", alpha=style["feature_outside_alpha"], label="Curve-fitting window (not analyzed)"),
        Line2D([0], [0], color=style["feature_baseline_color"], lw=style["feature_baseline_lw"], ls="--", label="Fitted curve (pre-cue)"),
        Line2D([0], [0], color=style["feature_baseline_color"], lw=style["feature_baseline_lw"], ls=":", label="Fitted curve (post-cue)"),
    ]
    if show_std_band:
        legend_handles.append(Patch(facecolor=colors["common"], edgecolor="none", alpha=style["std_band_alpha"], label="Mean ± SD"))
    if show_individual_traces:
        legend_handles.append(Line2D([0], [0], color=colors["common"], lw=style["individual_trace_lw"], alpha=style["individual_trace_alpha"], label="Individual repeats"))
    if show_reference_std_lines:
        legend_handles.append(Line2D([0], [0], color=colors["common"], lw=style["reference_std_lw"], ls="--", alpha=style["reference_std_alpha"], label="± mean run SD"))
    axes[0, 2].legend(
        handles=legend_handles,
        loc="center",
        frameon=True,
        title="Shared markers and traces",
    )

    modulation_ylabel = (
        "Modulation relative to baseline (z)"
        if ref["cst_modulation_mode"] == "zscore"
        else "Modulation relative to baseline (%)"
    )
    _plot_cst_key(
        axes[1, 2], "cst_alpha_mod", colors["alpha"],
        f"Alpha modulation ({'z-score' if ref['cst_modulation_mode'] == 'zscore' else '% baseline'} of Hilbert transform)",
        ylabel=modulation_ylabel,
        panel_limit_key="mod_alpha",
    )
    _plot_cst_key(
        axes[2, 2], "cst_beta_mod", colors["beta"],
        f"Beta modulation ({'z-score' if ref['cst_modulation_mode'] == 'zscore' else '% baseline'} of Hilbert transform)",
        ylabel=modulation_ylabel,
        panel_limit_key="mod_beta",
    )

    sync_available = [run["sync_trace"] is not None and run["sync_time_abs"] is not None for run in runs]
    if any(sync_available):
        ref_sync_time = None
        traces = []
        for run in runs:
            if run["sync_trace"] is None or run["sync_time_abs"] is None:
                continue
            if ref_sync_time is None:
                ref_sync_time = np.asarray(run["sync_time_abs"], dtype=float)
            traces.append(_interp_to_reference(run["sync_time_abs"], run["sync_trace"], ref_sync_time))
        traces = np.stack(traces, axis=0)
        sync_mask = (ref_sync_time >= zoom["zoom_start_s"]) & (ref_sync_time <= zoom["zoom_end_s"])
        _plot_raw_panel(
            axes[3, 2],
            ref_sync_time[sync_mask] - cue_time_s,
            np.nanmean(traces[:, sync_mask], axis=0),
            np.nanstd(traces[:, sync_mask], axis=0),
            traces[:, sync_mask],
            color=colors["sync"],
            ylabel="Excess coincidence index" if ref["sync_display_mode"] == "absolute" else "Excess coincidence index (z)",
            title=f"Sliding synchrony ({'absolute' if ref['sync_display_mode'] == 'absolute' else 'baseline-zscored'})",
            burst_center_s=zoom["burst_center_s"], burst_start_s=zoom["burst_start_s"], burst_end_s=zoom["burst_end_s"],
            event_times=event_times, show_individual_traces=show_individual_traces, show_std_band=show_std_band, colors=colors, style=style,
            fixed_ylim=None if fixed_panel_limits is None else fixed_panel_limits.get("sync"),
            time_offset_s=cue_time_s,
            show_burst=False,
        )
        _draw_sim_window_overlays(axes[3, 2], params, feature_config, display_window_rel_cue_s, window_overlay_specs)
        if feature_config["enabled"]:
            try:
                full_sync_time_rel = np.asarray(ref_sync_time, dtype=float) - cue_time_s
                full_sync_mean = np.nanmean(traces, axis=0)
                feature_result = fit_baseline_spline_and_extract_feature(
                    full_sync_time_rel,
                    full_sync_mean,
                    fitting_window=feature_config["fitting_window"],
                    analysis_window=feature_config["analysis_window"],
                    buffer_s=feature_config["buffer_s"],
                    spline_smoothing=feature_config["spline_smoothing"],
                    spline_order=feature_config["spline_order"],
                    n_knots=feature_config["n_knots"],
                    trend_pre_window=feature_config.get("trend_pre_window"),
                    trend_post_window=feature_config.get("trend_post_window"),
                    extrapolate=feature_config["extrapolate"],
                    return_debug_info=True,
                )
                _annotate_feature_panel(
                    axes[3, 2],
                    ref_sync_time[sync_mask] - cue_time_s,
                    np.nanmean(traces[:, sync_mask], axis=0),
                    colors["sync"],
                    feature_result,
                    feature_config,
                    style,
                    "a.u.",
                )
                features["sync_feature_mean_residual"] = feature_result.feature_mean_residual
                features["sync_feature_signed_area"] = feature_result.feature_signed_area
                features["sync_fit_r2"] = feature_result.fit_r2
                features["sync_fit_r2_gain_over_linear"] = feature_result.fit_r2_gain_over_linear
                features["sync_trend_post_minus_pre"] = feature_result.trend_post_minus_pre
            except Exception as exc:
                warnings.warn(f"Could not extract baseline feature for {Path(parameter_set_dir).name} / sync: {exc}")
    else:
        axes[3, 2].axis("off")

    tick_start = np.floor(np.min(zoom_t) / 0.5) * 0.5
    tick_end = np.ceil(np.max(zoom_t) / 0.5) * 0.5
    xticks = np.arange(tick_start, tick_end + 0.25, 0.5, dtype=float)
    for row_axes in axes:
        for ax in row_axes:
            if ax.axison:
                ax.set_xlim(float(np.min(zoom_t)), float(np.max(zoom_t)))
                ax.xaxis.set_major_locator(MultipleLocator(0.5))
                ax.set_xticks(xticks)

    axes[3, 0].set_xlabel("Time relative to go/no-go cue (s)")
    axes[2, 1].set_xlabel("Time relative to go/no-go cue (s)")
    axes[3, 2].set_xlabel("Time relative to go/no-go cue (s)")
    fig.suptitle(f"Input and CST summary diagnostics | {Path(parameter_set_dir).name} | n={len(runs)} repeats", fontsize=12)
    plt.tight_layout()

    figure_path = None
    if save_figure:
        output_dir = Path(output_dir) if output_dir is not None else Path(parameter_set_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = output_dir / f"{Path(parameter_set_dir).name}__summary.png"
        plt.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path, features


def _resolve_sim_window_overlay_specs(window_overlay_specs=None):
    default_specs = {
        "firing_rate_isi_window_rel_cue_s": {"enabled": True, "label": "FR / ISI window", "color": "#37a055", "lw": 2.0, "alpha": 1.0, "y_axes": 0.02},
        "cst_modulation_normalization_window_rel_cue_s": {"enabled": True, "label": "CST normalization window", "color": "#ffa600", "lw": 2.0, "alpha": 0.95, "y_axes": 0.05},
        "sync_baseline_win_rel_cue_s": {"enabled": True, "label": "Sync baseline window", "color": "#7e2179", "lw": 2.0, "alpha": 1.0, "y_axes": 0.08},
        "feature_fitting_window": {"enabled": True, "label": "Spline fitting window", "color": "#E09ED8", "lw": 2.0, "alpha": 1.0, "y_axes": 0.15},
        "feature_analysis_window": {"enabled": True, "label": "Feature window", "color": "#E900CA", "lw": 2.0, "alpha": 1.0, "y_axes": 0.15},
    }
    if window_overlay_specs:
        for key, value in window_overlay_specs.items():
            base = dict(default_specs.get(key, {}))
            base.update(value or {})
            default_specs[key] = base
    return default_specs


def _draw_sim_window_overlays(ax, params, feature_config, display_window_rel_cue_s, window_overlay_specs):
    overlay_specs = _resolve_sim_window_overlay_specs(window_overlay_specs)
    buffer_s = float(feature_config["buffer_s"])
    fitting_start_s, fitting_end_s = map(float, feature_config["fitting_window"])
    analysis_start_s, analysis_end_s = map(float, feature_config["analysis_window"])
    exclusion_start_s = analysis_start_s - buffer_s
    exclusion_end_s = analysis_end_s + buffer_s
    segments_by_key = {
        "firing_rate_isi_window_rel_cue_s": [tuple(map(float, params.get("firing_rate_isi_window_rel_cue_s", (-4.0, -1.0))))],
        "cst_modulation_normalization_window_rel_cue_s": [tuple(map(float, params.get("cst_modulation_normalization_window_rel_cue_s", _resolve_cst_normalization_window_rel_go_cue(params))))],
        "sync_baseline_win_rel_cue_s": [tuple(map(float, params.get("sync_baseline_win_rel_cue_s", (-3.0, -2.0))))],
        "feature_fitting_window": [
            (fitting_start_s, min(exclusion_start_s, fitting_end_s)),
            (max(exclusion_end_s, fitting_start_s), fitting_end_s),
        ],
        "feature_analysis_window": [(analysis_start_s, analysis_end_s)],
    }
    trans = plt.matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for key, spec in overlay_specs.items():
        if not bool(spec.get("enabled", True)):
            continue
        for segment in segments_by_key.get(key, []):
            seg_start_s, seg_end_s = map(float, segment)
            clipped_start_s = max(float(display_window_rel_cue_s[0]), seg_start_s)
            clipped_end_s = min(float(display_window_rel_cue_s[1]), seg_end_s)
            if clipped_start_s >= clipped_end_s:
                continue
            ax.plot(
                [clipped_start_s, clipped_end_s],
                [float(spec.get("y_axes", 0.02)), float(spec.get("y_axes", 0.02))],
                color=str(spec.get("color", "#111111")),
                lw=float(spec.get("lw", 2.0)),
                alpha=float(spec.get("alpha", 0.95)),
                solid_capstyle="butt",
                transform=trans,
                clip_on=False,
                zorder=7,
            )


def summarize_parameter_set_batch(root_dir, output_dir=None, show_individual_traces=True, show_std_band=True,
                                  show_reference_std_lines=True, fixed_ylim_across_parameter_sets=False,
                                  color_overrides=None, plot_style=None,
                                  selected_parameter_set_ids=None, parameter_set_id_range=None,
                                  show_progress=True, xlim_rel_cue_s=None, feature_config=None,
                                  window_overlay_specs=None, export_figures=True):
    grouped_runs = collect_parameter_set_runs(
        root_dir,
        color_overrides=color_overrides,
        selected_parameter_set_ids=selected_parameter_set_ids,
        parameter_set_id_range=parameter_set_id_range,
    )
    if not grouped_runs:
        raise FileNotFoundError("No simulation_output.h5 files were found under parameter_set_* folders.")

    summary_rows = []
    output_dir = Path(output_dir) if output_dir is not None else Path(root_dir) / "parameter_set_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_panel_limits = None
    if fixed_ylim_across_parameter_sets:
        fixed_panel_limits = _compute_global_panel_limits(
            grouped_runs,
            show_individual_traces=show_individual_traces,
            show_std_band=show_std_band,
            show_reference_std_lines=show_reference_std_lines,
            xlim_rel_cue_s=xlim_rel_cue_s,
        )

    sorted_items = sorted(grouped_runs.items(), key=lambda item: item[0].name)
    for parameter_dir, runs in _progress_iter(sorted_items, desc="Analyzing parameter sets", show_progress=show_progress):
        figure_path, features = plot_parameter_set_summary(
            parameter_set_dir=parameter_dir,
            runs=runs,
            output_dir=output_dir,
            show_individual_traces=show_individual_traces,
            show_std_band=show_std_band,
            show_reference_std_lines=show_reference_std_lines,
            fixed_panel_limits=fixed_panel_limits,
            plot_style=plot_style,
            xlim_rel_cue_s=xlim_rel_cue_s,
            feature_config=feature_config,
            window_overlay_specs=window_overlay_specs,
            save_figure=export_figures,
        )
        row = {
            "parameter_set_dir": str(parameter_dir),
            "n_repeats": len(runs),
            "figure_path": np.nan if figure_path is None else str(figure_path),
        }
        row.update(features)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def _summarize_distribution(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "sd": np.nan, "skewness": np.nan}
    return {
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite)),
        "skewness": float(pd.Series(finite).skew()),
    }


def _dataset_by_path(h5_file, path):
    try:
        obj = h5_file
        for part in path.split("/"):
            obj = obj[part]
        return np.asarray(obj, dtype=float)
    except Exception:
        return None


def _extract_saved_rate_isi_arrays(h5_file):
    candidate_pairs = [
        ("firing_rate_diagnostics/mn_mean_firing_rate_hz", "firing_rate_diagnostics/mn_isi_cv"),
        ("analysis/firing_rate_diagnostics/mn_mean_firing_rate_hz", "analysis/firing_rate_diagnostics/mn_isi_cv"),
        ("analysis/Firing_rates/mn_mean_firing_rate_hz", "analysis/Firing_rates/mn_isi_cv"),
    ]
    for fr_path, isi_path in candidate_pairs:
        fr = _dataset_by_path(h5_file, fr_path)
        isi = _dataset_by_path(h5_file, isi_path)
        if fr is not None and isi is not None:
            return np.asarray(fr, dtype=float), np.asarray(isi, dtype=float), fr_path
    return None, None, None


def _resolve_cst_analysis_window_from_params(params):
    usable_start_s = float(params.get("edges_ignore_duration", 0.0))
    usable_end_s = float(params.get("duration_with_ignored_window", params.get("duration", np.nan))) - usable_start_s
    window_start_s = usable_start_s + float(params.get("cst_analysis_window_start_s", 0.0))
    raw_window_end_s = float(params.get("cst_analysis_window_end_s", -1.0))
    if raw_window_end_s <= 0:
        window_end_s = usable_end_s
    else:
        window_end_s = usable_start_s + raw_window_end_s
    return float(window_start_s), float(window_end_s)


def _compute_fr_and_isi_cv_from_spikes(spike_trains_s, active_unit_ids_for_isi_cv=None):
    firing_rate_means = []
    isi_covs = []
    active_id_set = None if active_unit_ids_for_isi_cv is None else set(np.asarray(active_unit_ids_for_isi_cv, dtype=int).tolist())
    for spike_times in spike_trains_s:
        unit_i = len(firing_rate_means)
        times = np.asarray(spike_times, dtype=float)
        if times.size < 2:
            firing_rate_means.append(0.0)
            isi_covs.append(np.nan)
            continue
        inst_rates = 1.0 / np.diff(times)
        firing_rate_means.append(float(np.mean(inst_rates)))
        if active_id_set is not None and unit_i not in active_id_set:
            isi_covs.append(np.nan)
            continue
        isis = np.diff(times)
        mean_isi = float(np.mean(isis))
        std_isi = float(np.std(isis))
        raw_cov = (std_isi / mean_isi) if mean_isi > 0 else np.nan
        isi_covs.append(raw_cov if (np.isfinite(raw_cov) and raw_cov >= 0.01) else 0.0)
    return np.asarray(firing_rate_means, dtype=float), np.asarray(isi_covs, dtype=float)


def collect_rate_isi_moments(root_dir, selected_parameter_set_ids=None, parameter_set_id_range=None, show_progress=True):
    root_dir = Path(root_dir)
    rows = []
    skipped_runs = []
    h5_paths = sorted(root_dir.rglob("simulation_output.h5"))
    for h5_path in _progress_iter(h5_paths, desc="Scanning simulation outputs", show_progress=show_progress):
        parameter_dir = _find_parameter_set_dir(h5_path)
        if parameter_dir is None:
            continue
        if not _parameter_set_is_selected(
            parameter_dir,
            selected_parameter_set_ids=selected_parameter_set_ids,
            parameter_set_id_range=parameter_set_id_range,
        ):
            continue
        run_dir = h5_path.parent
        sim_json = run_dir / "sim_parameters.json"
        if not sim_json.exists():
            continue
        with sim_json.open("r", encoding="utf-8") as f:
            params = json.load(f)
        usable_start_s = float(params["edges_ignore_duration"])
        usable_end_s = float(params["duration_with_ignored_window"] - params["edges_ignore_duration"])
        try:
            with h5py.File(h5_path, "r") as f:
                fr_values, isi_cv_values, value_source = _extract_saved_rate_isi_arrays(f)
                if fr_values is None or isi_cv_values is None:
                    spike_trains_group = f.get("spike_trains", None)
                    mn_group = spike_trains_group.get("MN", None) if spike_trains_group is not None else None
                    if mn_group is not None:
                        spike_trains_s = []
                        for key in sorted(mn_group.keys(), key=lambda name: int(name.split("_")[-1])):
                            spikes = np.asarray(mn_group[key], dtype=float)
                            spikes = spikes[(spikes >= usable_start_s) & (spikes <= usable_end_s)]
                            spike_trains_s.append(spikes)
                        active_unit_ids = None
                        cst_grp = f.get("cst_diagnostics", None)
                        if cst_grp is not None and "active_unit_ids" in cst_grp:
                            active_unit_ids = np.asarray(cst_grp["active_unit_ids"], dtype=int)
                        else:
                            analysis_start_s, analysis_end_s = _resolve_cst_analysis_window_from_params(params)
                            active_selection = classify_active_units_by_rate(
                                spike_trains_s,
                                analysis_start_s=analysis_start_s,
                                analysis_end_s=analysis_end_s,
                                rate_threshold_hz=float(params.get("active_unit_rate_threshold_hz", 3.0)),
                            )
                            active_unit_ids = active_selection["active_unit_ids"]
                        fr_values, isi_cv_values = _compute_fr_and_isi_cv_from_spikes(
                            spike_trains_s,
                            active_unit_ids_for_isi_cv=active_unit_ids,
                        )
                        value_source = "spike_trains/MN"
                    else:
                        skipped_runs.append(str(run_dir))
                        continue
        except OSError as exc:
            _warn_unreadable_h5(h5_path, exc)
            skipped_runs.append(str(run_dir))
            continue
        fr_stats = _summarize_distribution(fr_values)
        isi_stats = _summarize_distribution(isi_cv_values)
        rows.append(
            {
                "parameter_set_dir": str(parameter_dir),
                "run_dir": str(run_dir),
                "value_source": value_source,
                "firing_rate_mean": fr_stats["mean"],
                "firing_rate_sd": fr_stats["sd"],
                "firing_rate_skewness": fr_stats["skewness"],
                "isi_cv_mean": isi_stats["mean"],
                "isi_cv_sd": isi_stats["sd"],
                "isi_cv_skewness": isi_stats["skewness"],
            }
        )
    return pd.DataFrame(rows), skipped_runs


def _plot_metric_pairgrid(fig, subgridspec, df, column_keys, axis_labels, color, style, title):
    axes = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(subgridspec[i, j])
            axes[i, j] = ax
            x = df[column_keys[j]].to_numpy(dtype=float)
            y = df[column_keys[i]].to_numpy(dtype=float)
            if i == j:
                finite = x[np.isfinite(x)]
                if finite.size:
                    ax.hist(
                        finite,
                        bins="auto",
                        color=color,
                        alpha=style["hist_alpha"],
                        edgecolor=style["hist_edgecolor"],
                    )
                ax.set_xlabel(axis_labels[j])
                ax.set_ylabel("Count")
            else:
                mask = np.isfinite(x) & np.isfinite(y)
                if np.any(mask):
                    ax.scatter(
                        x[mask],
                        y[mask],
                        color=color,
                        alpha=style["scatter_alpha"],
                        s=style["scatter_size"],
                        linewidths=0,
                    )
                ax.set_xlabel(axis_labels[j])
                ax.set_ylabel(axis_labels[i])

            if i == 0:
                ax.set_title(axis_labels[j], fontsize=style["panel_title_fontsize"])
            ax.grid(alpha=0.2)
    axes[0, 1].text(
        0.5,
        1.28,
        title,
        transform=axes[0, 1].transAxes,
        ha="center",
        va="bottom",
        fontsize=style["title_fontsize"],
    )


def plot_rate_isi_moment_summary(metrics_df, output_dir=None, distribution_style=None, distribution_colors=None, save_figure=True):
    if metrics_df.empty:
        raise ValueError("metrics_df is empty")
    style = _resolve_distribution_style(distribution_style)
    colors = dict(DEFAULT_DISTRIBUTION_COLORS)
    if distribution_colors:
        colors.update(distribution_colors)

    fig = plt.figure(figsize=style["figure_size"], dpi=style["figure_dpi"])
    outer = fig.add_gridspec(1, 2, wspace=0.18)
    fr_grid = outer[0].subgridspec(3, 3, wspace=0.15, hspace=0.15)
    isi_grid = outer[1].subgridspec(3, 3, wspace=0.15, hspace=0.15)

    _plot_metric_pairgrid(
        fig,
        fr_grid,
        metrics_df,
        ["firing_rate_mean", "firing_rate_sd", "firing_rate_skewness"],
        ["Mean (Hz)", "SD (Hz)", "Skewness"],
        color=colors["firing_rate"],
        style=style,
        title="Mean firing rate distribution across motor neurons",
    )
    _plot_metric_pairgrid(
        fig,
        isi_grid,
        metrics_df,
        ["isi_cv_mean", "isi_cv_sd", "isi_cv_skewness"],
        ["Mean (CV)", "SD (CV)", "Skewness"],
        color=colors["isi_cv"],
        style=style,
        title="ISI CV distribution across motor neurons",
    )

    parameter_set_name = Path(str(metrics_df["parameter_set_dir"].iloc[0])).name
    fig.suptitle(f"Per-simulation motor-unit summary moments | {parameter_set_name} | n={len(metrics_df)} simulations", fontsize=style["title_fontsize"] + 1)
    plt.tight_layout()
    figure_path = None
    if save_figure:
        output_dir = Path(output_dir) if output_dir is not None else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = output_dir / f"{parameter_set_name}__firing_rate_isi_moment_summary.png"
        plt.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def summarize_rate_isi_batch(root_dir, output_dir=None, distribution_style=None, distribution_colors=None,
                             selected_parameter_set_ids=None, parameter_set_id_range=None,
                             show_progress=True, export_figures=True):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir) if output_dir is not None else root_dir / "parameter_set_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df, skipped_runs = collect_rate_isi_moments(
        root_dir,
        selected_parameter_set_ids=selected_parameter_set_ids,
        parameter_set_id_range=parameter_set_id_range,
        show_progress=show_progress,
    )
    if skipped_runs:
        warnings.warn(
            "Could not compute firing-rate / ISI-CV summaries for some runs because neither saved per-MU metrics nor spike trains were available:\n"
            + "\n".join(skipped_runs[:10])
            + ("\n..." if len(skipped_runs) > 10 else "")
        )
    if metrics_df.empty:
        warnings.warn(
            "Firing-rate / ISI-CV batch summaries could not be computed because neither saved per-MU metrics nor spike trains were available."
        )
        empty_figures_df = pd.DataFrame(columns=["parameter_set_dir", "figure_path", "n_runs"])
        return metrics_df, empty_figures_df

    figure_rows = []
    grouped_subsets = list(metrics_df.groupby("parameter_set_dir", sort=True))
    for parameter_set_dir, subset in _progress_iter(grouped_subsets, desc="Summarizing rate / ISI moments", show_progress=show_progress):
        figure_path = None
        if export_figures:
            figure_path = plot_rate_isi_moment_summary(
                subset.reset_index(drop=True),
                output_dir=output_dir,
                distribution_style=distribution_style,
                distribution_colors=distribution_colors,
                save_figure=True,
            )
        figure_rows.append(
            {
                "parameter_set_dir": parameter_set_dir,
                "figure_path": np.nan if figure_path is None else str(figure_path),
                "n_runs": int(len(subset)),
            }
        )
    figures_df = pd.DataFrame(figure_rows)
    return metrics_df, figures_df


def _iter_selected_parameter_set_dirs(root_dir, selected_parameter_set_ids=None, parameter_set_id_range=None):
    root_dir = Path(root_dir)
    parameter_dirs = []
    for parameter_dir in sorted(root_dir.glob("parameter_set_*")):
        if not parameter_dir.is_dir():
            continue
        if not _parameter_set_is_selected(
            parameter_dir,
            selected_parameter_set_ids=selected_parameter_set_ids,
            parameter_set_id_range=parameter_set_id_range,
        ):
            continue
        parameter_dirs.append(parameter_dir)
    return parameter_dirs


def _find_reference_sim_json(parameter_set_dir):
    sim_json_paths = sorted(Path(parameter_set_dir).rglob("sim_parameters.json"))
    return sim_json_paths[0] if sim_json_paths else None


def _load_reference_params(parameter_set_dir):
    sim_json_path = _find_reference_sim_json(parameter_set_dir)
    if sim_json_path is None:
        return {}
    with sim_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _count_parameter_set_runs(parameter_set_dir):
    return int(len(list(Path(parameter_set_dir).rglob("simulation_output.h5"))))


def _native_export_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, list):
        return [_native_export_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_native_export_value(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _native_export_value(v) for k, v in value.items()}
    return value


def _csv_safe_value(value):
    if value is None:
        return np.nan
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(_native_export_value(value))
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _mean_series_or_nan(df, column_name):
    if df is None or df.empty or column_name not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column_name], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(np.mean(finite))


def _resolve_feature_export_value(feature_config, key):
    if not feature_config:
        return np.nan
    value = feature_config.get(key, np.nan)
    if value is None:
        return np.nan
    if key in {"fitting_window", "analysis_window", "trend_pre_window", "trend_post_window"}:
        return tuple(value)
    return _native_export_value(value)


def _resolve_cst_normalization_window_rel_go_cue(params):
    explicit_window = params.get("cst_modulation_normalization_window_rel_cue_s", np.nan)
    try:
        arr = np.asarray(explicit_window, dtype=float).reshape(-1)
        if arr.size == 2 and np.isfinite(arr).all() and arr[0] < arr[1]:
            return (float(arr[0]), float(arr[1]))
    except Exception:
        pass
    try:
        usable_start_s = float(params["edges_ignore_duration"])
        analysis_start_rel_s = float(params.get("cst_analysis_window_start_s", 0.0))
        normalization_start_abs_s = usable_start_s + analysis_start_rel_s
        get_ready_abs_s = usable_start_s + float(params["get_ready_cue_time_s"])
        go_cue_abs_s = get_ready_abs_s + float(params["go_nogo_cue_delay_s"])
    except Exception:
        return np.nan
    return (
        float(normalization_start_abs_s - go_cue_abs_s),
        float(get_ready_abs_s - go_cue_abs_s),
    )


def _scalar_from_dataset(group, dataset_name):
    if group is None or dataset_name not in group:
        return np.nan
    try:
        value = np.asarray(group[dataset_name], dtype=float).reshape(-1)
    except Exception:
        return np.nan
    if value.size == 0:
        return np.nan
    scalar = float(value[0])
    return scalar if np.isfinite(scalar) else np.nan


def _normalize_rel_window_or_nan(bounds):
    try:
        arr = np.asarray(bounds, dtype=float).reshape(-1)
    except Exception:
        return np.nan
    if arr.size != 2 or not np.isfinite(arr).all() or not (arr[0] < arr[1]):
        return np.nan
    return (float(arr[0]), float(arr[1]))


def _default_sim_summary_metadata_from_params(params):
    window_defs = dict(DEFAULT_OBSERVATION_WINDOW_DEFS_REL_CUE_S)
    raw_window_defs = params.get("observation_window_defs_rel_cue_s", {})
    if isinstance(raw_window_defs, dict):
        window_defs.update(raw_window_defs)
    return {
        "normalization_win_rel_cue_s": _normalize_rel_window_or_nan(window_defs.get("normalization_win", (-4.0, -3.0))),
        "baseline_win_rel_cue_s": _normalize_rel_window_or_nan(window_defs.get("baseline_win", (-3.0, -2.0))),
        "baseline_extended_win_rel_cue_s": _normalize_rel_window_or_nan(window_defs.get("baseline_extended_win", (-2.0, -1.0))),
        "ready_win_rel_cue_s": _normalize_rel_window_or_nan(window_defs.get("ready_win", (-1.0, 0.0))),
        "post_win_rel_cue_s": _normalize_rel_window_or_nan(window_defs.get("post_win", (0.0, 1.0))),
        "obs_baseline_window_rel_cue_s": _normalize_rel_window_or_nan(params.get("obs_baseline_window_rel_cue_s", (-3.0, -1.0))),
        "cst_modulation_normalization_window_rel_cue_s": _resolve_cst_normalization_window_rel_go_cue(params),
        "firing_rate_isi_window_rel_cue_s": _normalize_rel_window_or_nan(params.get("firing_rate_isi_window_rel_cue_s", (-4.0, -1.0))),
        "sync_analysis_window_rel_cue_s": _normalize_rel_window_or_nan(params.get("sync_analysis_window_rel_cue_s", (-4.0, 1.0))),
        "sync_baseline_win_rel_cue_s": _normalize_rel_window_or_nan(params.get("sync_baseline_win_rel_cue_s", (-3.0, -2.0))),
        "sync_post_win_rel_cue_s": _normalize_rel_window_or_nan(params.get("sync_post_win_rel_cue_s", (0.3, 0.6))),
        "modulation_spectrum_window_rel_cue_s": _normalize_rel_window_or_nan(
            params.get(
                "modulation_spectrum_window_rel_cue_s",
                DEFAULT_MODULATION_SPECTRUM_CONFIG["window_rel_cue_s"],
            )
        ),
        "modulation_spectrum_max_freq_hz": params.get("modulation_spectrum_max_freq_hz", DEFAULT_MODULATION_SPECTRUM_CONFIG["max_freq_hz"]),
        "modulation_spectrum_interval_mass_pct": params.get("modulation_spectrum_interval_mass_pct", DEFAULT_MODULATION_SPECTRUM_CONFIG["interval_mass_pct"]),
        "lf_band_hz": _native_export_value(params.get("lf_band_hz", np.nan)),
        "alpha_band_hz": _native_export_value(params.get("alpha_band_hz", np.nan)),
        "beta_band_hz": _native_export_value(params.get("beta_band_hz", np.nan)),
        "active_unit_rate_threshold_hz": params.get("active_unit_rate_threshold_hz", np.nan),
        "sync_win_ms": params.get("sync_win_ms", np.nan),
        "sync_step_ms": params.get("sync_step_ms", np.nan),
        "sync_coinc_lag_ms": params.get("sync_coinc_lag_ms", np.nan),
        "sync_direction_mode": params.get("sync_direction_mode", np.nan),
        "sync_expectation_mode": params.get("sync_expectation_mode", np.nan),
    }


def _load_analysis_summary_metadata_for_run(run_dir):
    run_dir = Path(run_dir)
    sim_json = run_dir / "sim_parameters.json"
    sim_h5 = run_dir / "simulation_output.h5"
    if not sim_json.exists() or not sim_h5.exists():
        return _default_sim_summary_metadata_from_params({})
    with sim_json.open("r", encoding="utf-8") as f:
        params = json.load(f)
    metadata = _default_sim_summary_metadata_from_params(params)
    try:
        with h5py.File(sim_h5, "r") as f:
            summary_grp = f.get("analysis/summary_metadata", None)
            if summary_grp is not None:
                for key in (
                    "normalization_win_rel_cue_s",
                    "baseline_win_rel_cue_s",
                    "baseline_extended_win_rel_cue_s",
                    "ready_win_rel_cue_s",
                    "post_win_rel_cue_s",
                    "obs_baseline_window_rel_cue_s",
                    "cst_modulation_normalization_window_rel_cue_s",
                    "firing_rate_isi_window_rel_cue_s",
                    "sync_analysis_window_rel_cue_s",
                    "sync_baseline_win_rel_cue_s",
                    "sync_post_win_rel_cue_s",
                    "modulation_spectrum_window_rel_cue_s",
                    "lf_band_hz",
                    "alpha_band_hz",
                    "beta_band_hz",
                ):
                    if key in summary_grp:
                        arr = np.asarray(summary_grp[key], dtype=float).reshape(-1)
                        metadata[key] = tuple(arr.tolist()) if arr.size > 1 else (float(arr[0]),) if arr.size == 1 else np.nan
                for key in (
                    "modulation_spectrum_max_freq_hz",
                    "modulation_spectrum_interval_mass_pct",
                    "sync_win_ms",
                    "sync_step_ms",
                    "sync_coinc_lag_ms",
                    "active_unit_rate_threshold_hz",
                    "n_units_total",
                    "n_units_kept",
                    "unit_fraction_kept",
                ):
                    if key in summary_grp:
                        metadata[key] = _scalar_from_dataset(summary_grp, key)
                for key in ("sync_direction_mode", "sync_expectation_mode"):
                    if key in summary_grp.attrs:
                        metadata[key] = str(summary_grp.attrs[key])
            cst_grp = f.get("cst_diagnostics", None)
            if cst_grp is not None:
                if not np.isfinite(float(metadata.get("n_units_kept", np.nan))) and "active_unit_ids" in cst_grp:
                    metadata["n_units_kept"] = float(np.asarray(cst_grp["active_unit_ids"], dtype=int).size)
                if not np.isfinite(float(metadata.get("n_units_total", np.nan))):
                    if "active_unit_ids" in cst_grp and "excluded_unit_ids" in cst_grp:
                        metadata["n_units_total"] = float(
                            np.asarray(cst_grp["active_unit_ids"], dtype=int).size
                            + np.asarray(cst_grp["excluded_unit_ids"], dtype=int).size
                        )
                    else:
                        metadata["n_units_total"] = float(params.get("nb_motoneurons", np.nan))
                if (not np.isfinite(float(metadata.get("unit_fraction_kept", np.nan)))
                        and np.isfinite(float(metadata.get("n_units_total", np.nan)))
                        and float(metadata.get("n_units_total", np.nan)) > 0
                        and np.isfinite(float(metadata.get("n_units_kept", np.nan)))):
                    metadata["unit_fraction_kept"] = float(metadata["n_units_kept"] / metadata["n_units_total"])
    except OSError as exc:
        _warn_unreadable_h5(sim_h5, exc)
    return metadata


def _load_sim_observation_features_for_run(run_dir):
    run_dir = Path(run_dir)
    sim_json = run_dir / "sim_parameters.json"
    sim_h5 = run_dir / "simulation_output.h5"
    feature_names = [*OBS_WINDOW_EXPORT_COLUMNS, *OBS_BASELINE_EXPORT_COLUMNS, *OBS_SPECTRUM_EXPORT_COLUMNS]
    features = {name: np.nan for name in feature_names}
    if not sim_json.exists() or not sim_h5.exists():
        return features
    try:
        with h5py.File(sim_h5, "r") as f:
            obs_grp = f.get("analysis/observation_features", None)
            if obs_grp is not None:
                for name in feature_names:
                    if name in obs_grp:
                        features[name] = _scalar_from_dataset(obs_grp, name)
                return features
    except OSError as exc:
        _warn_unreadable_h5(sim_h5, exc)
        return features

    with sim_json.open("r", encoding="utf-8") as f:
        params = json.load(f)
    try:
        with h5py.File(sim_h5, "r") as f:
            cst_grp = f.get("cst_diagnostics", None)
            if cst_grp is None:
                return features
            cst_diagnostics = {}
            for key in cst_grp.keys():
                cst_diagnostics[key] = np.asarray(cst_grp[key])
            if "time_s" not in cst_diagnostics and "time_rel_cue_s" in cst_diagnostics:
                event_times = _get_event_times(params, f.get("driving_inputs", None))
                cue_time_abs_s = float(event_times.get("go_nogo_cue_s", 0.0))
                cst_diagnostics["time_s"] = np.asarray(cst_diagnostics["time_rel_cue_s"], dtype=float) + cue_time_abs_s
            sync_grp = f.get("sync_diagnostics", None)
            sync_diagnostics = None
            if sync_grp is not None:
                sync_diagnostics = {}
                for key in sync_grp.keys():
                    sync_diagnostics[key] = np.asarray(sync_grp[key])
            event_times = _get_event_times(params, f.get("driving_inputs", None))
            cue_time_abs_s = float(event_times.get("go_nogo_cue_s", 0.0))
            summary_metadata = _default_sim_summary_metadata_from_params(params)
            observation_window_defs_rel_cue_s = {}
            for alias in ("normalization_win", "baseline_win", "baseline_extended_win", "ready_win", "post_win"):
                bounds = summary_metadata.get(f"{alias}_rel_cue_s", np.nan)
                normalized_bounds = _normalize_rel_window_or_nan(bounds)
                if isinstance(normalized_bounds, tuple):
                    observation_window_defs_rel_cue_s[alias] = normalized_bounds
            computed = compute_observation_features_from_cst_sync(
                cst_diagnostics,
                sync_diagnostics,
                cue_time_abs_s=cue_time_abs_s,
                observation_window_defs_rel_cue_s=observation_window_defs_rel_cue_s,
                obs_baseline_window_rel_cue_s=summary_metadata["obs_baseline_window_rel_cue_s"],
                modulation_spectrum_config=resolve_modulation_spectrum_config(
                    {
                        "window_rel_cue_s": summary_metadata["modulation_spectrum_window_rel_cue_s"],
                        "max_freq_hz": summary_metadata["modulation_spectrum_max_freq_hz"],
                        "interval_mass_pct": summary_metadata["modulation_spectrum_interval_mass_pct"],
                    }
                ),
            )
            if computed:
                features.update(computed)
    except OSError as exc:
        _warn_unreadable_h5(sim_h5, exc)
    return features


def _collect_parameter_set_observation_features(parameter_set_dir):
    run_dirs = sorted(path.parent for path in Path(parameter_set_dir).rglob("simulation_output.h5"))
    rows = [_load_sim_observation_features_for_run(run_dir) for run_dir in run_dirs]
    if not rows:
        return {name: np.nan for name in (*OBS_WINDOW_EXPORT_COLUMNS, *OBS_BASELINE_EXPORT_COLUMNS, *OBS_SPECTRUM_EXPORT_COLUMNS)}
    df = pd.DataFrame(rows)
    return {name: _mean_series_or_nan(df, name) for name in rows[0].keys()}


def _baseline_window_for_saved_cst(params, n_samples, saved_fsamp_hz):
    usable_start_s = float(params["edges_ignore_duration"])
    analysis_start_rel_s = float(params.get("cst_analysis_window_start_s", 0.0))
    baseline_start_s = usable_start_s + analysis_start_rel_s
    baseline_end_s = float(params.get("get_ready_cue_time_s", np.nan))
    if not np.isfinite(baseline_end_s):
        return None
    t_s = np.arange(int(n_samples), dtype=float) / float(saved_fsamp_hz)
    mask = (t_s >= baseline_start_s) & (t_s < baseline_end_s)
    if not np.any(mask):
        return None
    return mask


def _load_baseline_observations_for_run(run_dir):
    run_dir = Path(run_dir)
    sim_json = run_dir / "sim_parameters.json"
    sim_h5 = run_dir / "simulation_output.h5"
    if not sim_json.exists() or not sim_h5.exists():
        return {
            "cst_low_pass_baseline_mean": np.nan,
            "cst_alpha_hilbert_baseline_mean": np.nan,
            "cst_beta_hilbert_baseline_mean": np.nan,
        }

    with sim_json.open("r", encoding="utf-8") as f:
        params = json.load(f)

    low_pass_mean = np.nan
    alpha_mean = np.nan
    beta_mean = np.nan
    try:
        with h5py.File(sim_h5, "r") as f:
            cst_diag = f.get("cst_diagnostics", None)
            if cst_diag is not None:
                alpha_mean = _scalar_from_dataset(cst_diag, "cst_alpha_baseline_mean")
                beta_mean = _scalar_from_dataset(cst_diag, "cst_beta_baseline_mean")
                cst_lf = _load_dataset_if_present(cst_diag, "cst_lf")
                if cst_lf is not None:
                    input_grp = f["input"] if "input" in f else None
                    saved_fsamp_hz = float(
                        (input_grp.attrs.get("saved_fsamp_Hz") if input_grp is not None else None)
                        or params["fsamp"]
                    )
                    baseline_mask = _baseline_window_for_saved_cst(params, len(cst_lf), saved_fsamp_hz)
                    if baseline_mask is not None:
                        finite = np.asarray(cst_lf, dtype=float)[baseline_mask]
                        finite = finite[np.isfinite(finite)]
                        if finite.size > 0:
                            low_pass_mean = float(np.mean(finite))

                if not np.isfinite(alpha_mean):
                    alpha_env = _load_dataset_if_present(cst_diag, "cst_alpha_envelope")
                    if alpha_env is not None:
                        input_grp = f["input"] if "input" in f else None
                        saved_fsamp_hz = float(
                            (input_grp.attrs.get("saved_fsamp_Hz") if input_grp is not None else None)
                            or params["fsamp"]
                        )
                        baseline_mask = _baseline_window_for_saved_cst(params, len(alpha_env), saved_fsamp_hz)
                        if baseline_mask is not None:
                            finite = np.asarray(alpha_env, dtype=float)[baseline_mask]
                            finite = finite[np.isfinite(finite)]
                            if finite.size > 0:
                                alpha_mean = float(np.mean(finite))

                if not np.isfinite(beta_mean):
                    beta_env = _load_dataset_if_present(cst_diag, "cst_beta_envelope")
                    if beta_env is not None:
                        input_grp = f["input"] if "input" in f else None
                        saved_fsamp_hz = float(
                            (input_grp.attrs.get("saved_fsamp_Hz") if input_grp is not None else None)
                            or params["fsamp"]
                        )
                        baseline_mask = _baseline_window_for_saved_cst(params, len(beta_env), saved_fsamp_hz)
                        if baseline_mask is not None:
                            finite = np.asarray(beta_env, dtype=float)[baseline_mask]
                            finite = finite[np.isfinite(finite)]
                            if finite.size > 0:
                                beta_mean = float(np.mean(finite))
    except OSError as exc:
        _warn_unreadable_h5(sim_h5, exc)

    return {
        "cst_low_pass_baseline_mean": low_pass_mean,
        "cst_alpha_hilbert_baseline_mean": alpha_mean,
        "cst_beta_hilbert_baseline_mean": beta_mean,
    }


def _collect_parameter_set_baseline_observations(parameter_set_dir):
    run_dirs = sorted(path.parent for path in Path(parameter_set_dir).rglob("simulation_output.h5"))
    if not run_dirs:
        return {
            "cst_low_pass_baseline_mean": np.nan,
            "cst_alpha_hilbert_baseline_mean": np.nan,
            "cst_beta_hilbert_baseline_mean": np.nan,
        }
    rows = [_load_baseline_observations_for_run(run_dir) for run_dir in run_dirs]
    df = pd.DataFrame(rows)
    return {
        "cst_low_pass_baseline_mean": _mean_series_or_nan(df, "cst_low_pass_baseline_mean"),
        "cst_alpha_hilbert_baseline_mean": _mean_series_or_nan(df, "cst_alpha_hilbert_baseline_mean"),
        "cst_beta_hilbert_baseline_mean": _mean_series_or_nan(df, "cst_beta_hilbert_baseline_mean"),
    }


def build_parameter_set_export_table(
    root_dir,
    parameter_set_summary_df=None,
    rate_isi_metrics_df=None,
    feature_config=None,
    selected_parameter_set_ids=None,
    parameter_set_id_range=None,
):
    root_dir = Path(root_dir)
    parameter_dirs = _iter_selected_parameter_set_dirs(
        root_dir,
        selected_parameter_set_ids=selected_parameter_set_ids,
        parameter_set_id_range=parameter_set_id_range,
    )

    summary_lookup = {}
    if parameter_set_summary_df is not None and not parameter_set_summary_df.empty and "parameter_set_dir" in parameter_set_summary_df.columns:
        summary_lookup = {
            str(row["parameter_set_dir"]): row.to_dict()
            for _, row in parameter_set_summary_df.iterrows()
        }
        known_summary_dirs = [Path(p) for p in summary_lookup.keys()]
        if not parameter_dirs:
            parameter_dirs = sorted(known_summary_dirs, key=lambda p: p.name)

    rate_groups = {}
    if rate_isi_metrics_df is not None and not rate_isi_metrics_df.empty and "parameter_set_dir" in rate_isi_metrics_df.columns:
        rate_groups = {
            str(parameter_set_dir): subset.reset_index(drop=True)
            for parameter_set_dir, subset in rate_isi_metrics_df.groupby("parameter_set_dir", sort=True)
        }
        if not parameter_dirs:
            parameter_dirs = sorted((Path(p) for p in rate_groups.keys()), key=lambda p: p.name)

    rows = []
    for parameter_dir in parameter_dirs:
        parameter_dir = Path(parameter_dir)
        params = _load_reference_params(parameter_dir)
        summary_row = summary_lookup.get(str(parameter_dir), {})
        rate_subset = rate_groups.get(str(parameter_dir))
        observation_feature_means = _collect_parameter_set_observation_features(parameter_dir)
        run_dirs = sorted(path.parent for path in parameter_dir.rglob("simulation_output.h5"))
        run_analysis_metadata = [_load_analysis_summary_metadata_for_run(run_dir) for run_dir in run_dirs]
        analysis_metadata = run_analysis_metadata[0] if run_analysis_metadata else _default_sim_summary_metadata_from_params(params)
        kept_counts = np.asarray([meta.get("n_units_kept", np.nan) for meta in run_analysis_metadata], dtype=float)
        total_counts = np.asarray([meta.get("n_units_total", np.nan) for meta in run_analysis_metadata], dtype=float)
        kept_fractions = np.asarray([meta.get("unit_fraction_kept", np.nan) for meta in run_analysis_metadata], dtype=float)
        n_sims = summary_row.get("n_repeats", np.nan)
        if pd.isna(n_sims):
            n_sims = len(rate_subset) if rate_subset is not None else np.nan
        if pd.isna(n_sims):
            n_sims = _count_parameter_set_runs(parameter_dir)

        row = {column: np.nan for column in EXPORT_TABLE_COLUMNS}
        mode_label = _posterior_mode_label_from_path(parameter_dir)
        row.update({
            "DATASET_CONTEXT_participant_id": np.nan,
            "DATASET_CONTEXT_session_tag": np.nan,
            "DATASET_CONTEXT_condition_id": np.nan,
            "SIM_PARAM_GENERAL_batch_sim_index": _parameter_set_id_from_path(parameter_dir),
            "SIM_SUMMARY_is_posterior_mode": bool(mode_label is not None),
            "SIM_SUMMARY_mode_observation_label": mode_label if mode_label is not None else np.nan,
            "SIM_PARAM_GENERAL_minimal_output": params.get("minimal_output", np.nan),
            "SIM_PARAM_GENERAL_enable_force_model": params.get("enable_force_model", np.nan),
            "SIM_PARAM_GENERAL_n_sims": int(n_sims) if np.isfinite(float(n_sims)) else np.nan,
            "SIM_PARAM_GENERAL_duration": params.get("duration", np.nan),
            "SIM_PARAM_GENERAL_get_ready_cue_time_s": params.get("get_ready_cue_time_s", np.nan),
            "SIM_PARAM_GENERAL_go_nogo_cue_delay_s": params.get("go_nogo_cue_delay_s", np.nan),
            "SIM_PARAM_GENERAL_nb_motoneurons": params.get("nb_motoneurons", np.nan),
            "SIM_PARAM_GENERAL_min_soma_diameter": params.get("min_soma_diameter", np.nan),
            "SIM_PARAM_GENERAL_max_soma_diameter": params.get("max_soma_diameter", np.nan),
            "SIM_PARAM_GENERAL_tau_size_mode": params.get("tau_size_mode", np.nan),
            "SIM_PARAM_GENERAL_tau_size_ratio": params.get("tau_size_ratio", np.nan),
            "SIM_PARAM_GENERAL_tau_size_um": params.get("tau_size_um", np.nan),
            "SIM_PARAM_BASELINE_INPUT_excitatory_input_baseline": params.get("excitatory_input_baseline", np.nan),
            "SIM_PARAM_BASELINE_INPUT_lf_target_sd": params.get("lf_target_sd", np.nan),
            "SIM_PARAM_BASELINE_INPUT_lf_band_hz": _native_export_value(params.get("lf_band_hz", np.nan)),
            "SIM_PARAM_BASELINE_INPUT_alpha_target_sd_final": params.get("alpha_target_sd_final", np.nan),
            "SIM_PARAM_BASELINE_INPUT_alpha_band_hz": _native_export_value(params.get("alpha_band_hz", np.nan)),
            "SIM_PARAM_BASELINE_INPUT_alpha_envelope_band_hz": _native_export_value(params.get("alpha_envelope_band_hz", np.nan)),
            "SIM_PARAM_BASELINE_INPUT_beta_target_sd_final": params.get("beta_target_sd_final", np.nan),
            "SIM_PARAM_BASELINE_INPUT_beta_band_hz": _native_export_value(params.get("beta_band_hz", np.nan)),
            "SIM_PARAM_BASELINE_INPUT_beta_envelope_band_hz": _native_export_value(params.get("beta_envelope_band_hz", np.nan)),
            "SIM_PARAM_BASELINE_INPUT_common_input_weight_sd": params.get("common_input_weight_sd", np.nan),
            "SIM_PARAM_BASELINE_INPUT_common_input_weight_min": params.get("common_input_weight_min", np.nan),
            "SIM_PARAM_BASELINE_INPUT_common_input_weight_max": params.get("common_input_weight_max", np.nan),
            "SIM_PARAM_BASELINE_INPUT_independent_input_weight_sd": params.get("independent_input_weight_sd", np.nan),
            "SIM_PARAM_BASELINE_INPUT_independent_input_weight_max": params.get("independent_input_weight_max", np.nan),
            "SIM_PARAM_BASELINE_INPUT_independent_input_absolute_or_ratio": params.get("independent_input_absolute_or_ratio", np.nan),
            "SIM_PARAM_BASELINE_INPUT_independent_input_power": params.get("independent_input_power", np.nan),
            "SIM_PARAM_BURST_INPUT_burst_delay_after_go_nogo_cue_s": params.get("burst_delay_after_go_nogo_cue_s", np.nan),
            "SIM_PARAM_BURST_INPUT_input_burst_sigma_ms": params.get("input_burst_sigma_ms", np.nan),
            "SIM_PARAM_BURST_INPUT_lf_burst_peak_nA": params.get("lf_burst_peak_nA", np.nan),
            "SIM_PARAM_BURST_INPUT_alpha_burst_peak_nA": params.get("alpha_burst_peak_nA", np.nan),
            "SIM_PARAM_BURST_INPUT_beta_burst_peak_nA": params.get("beta_burst_peak_nA", np.nan),
            "SIM_PARAM_TREND_INPUT_lf_trend_start_rel_go_cue_s": params.get("lf_trend_start_rel_go_cue_s", np.nan),
            "SIM_PARAM_TREND_INPUT_alpha_trend_start_rel_go_cue_s": params.get("alpha_trend_start_rel_go_cue_s", np.nan),
            "SIM_PARAM_TREND_INPUT_beta_trend_start_rel_go_cue_s": params.get("beta_trend_start_rel_go_cue_s", np.nan),
            "SIM_PARAM_TREND_INPUT_lf_trend_slope_nA_per_s": params.get("lf_trend_slope_nA_per_s", np.nan),
            "SIM_PARAM_TREND_INPUT_alpha_trend_slope_nA_per_s": params.get("alpha_trend_slope_nA_per_s", np.nan),
            "SIM_PARAM_TREND_INPUT_beta_trend_slope_nA_per_s": params.get("beta_trend_slope_nA_per_s", np.nan),
            "OBS_FIRING_STATISTICS_firing_rate_mean": _mean_series_or_nan(rate_subset, "firing_rate_mean"),
            "OBS_FIRING_STATISTICS_firing_rate_sd": _mean_series_or_nan(rate_subset, "firing_rate_sd"),
            "OBS_FIRING_STATISTICS_firing_rate_skew": _mean_series_or_nan(rate_subset, "firing_rate_skewness"),
            "OBS_FIRING_STATISTICS_isi_cv_mean": _mean_series_or_nan(rate_subset, "isi_cv_mean"),
            "OBS_FIRING_STATISTICS_isi_cv_sd": _mean_series_or_nan(rate_subset, "isi_cv_sd"),
            "OBS_FIRING_STATISTICS_isi_cv_skew": _mean_series_or_nan(rate_subset, "isi_cv_skewness"),
            "OBS_BURST_FEATURES_resid_cst_low_frequency": summary_row.get("cst_lf_feature_mean_residual", np.nan),
            "OBS_BURST_FEATURES_resid_cst_alpha_modulation_hilbert_zscore": summary_row.get("cst_alpha_mod_feature_mean_residual", np.nan),
            "OBS_BURST_FEATURES_resid_cst_beta_modulation_hilbert_zscore": summary_row.get("cst_beta_mod_feature_mean_residual", np.nan),
            "OBS_BURST_FEATURES_resid_sync_coincidence_curve": summary_row.get("sync_feature_mean_residual", np.nan),
            "OBS_TREND_FEATURE_cst_low_frequency": summary_row.get("cst_lf_trend_post_minus_pre", np.nan),
            "OBS_TREND_FEATURE_cst_alpha_modulation_hilbert_zscore": summary_row.get("cst_alpha_mod_trend_post_minus_pre", np.nan),
            "OBS_TREND_FEATURE_cst_beta_modulation_hilbert_zscore": summary_row.get("cst_beta_mod_trend_post_minus_pre", np.nan),
            "OBS_TREND_FEATURE_sync_coincidence_curve": summary_row.get("sync_trend_post_minus_pre", np.nan),
            "ANALYSIS_PARAM_FEATURE_fitting_window": _resolve_feature_export_value(feature_config, "fitting_window"),
            "ANALYSIS_PARAM_FEATURE_analysis_window": _resolve_feature_export_value(feature_config, "analysis_window"),
            "ANALYSIS_PARAM_FEATURE_buffer_s": _resolve_feature_export_value(feature_config, "buffer_s"),
            "ANALYSIS_PARAM_FEATURE_spline_smoothing": _resolve_feature_export_value(feature_config, "spline_smoothing"),
            "ANALYSIS_PARAM_FEATURE_spline_order": _resolve_feature_export_value(feature_config, "spline_order"),
            "ANALYSIS_PARAM_FEATURE_n_knots": _resolve_feature_export_value(feature_config, "n_knots"),
            "ANALYSIS_PARAM_FEATURE_trend_pre_window": _resolve_feature_export_value(feature_config, "trend_pre_window"),
            "ANALYSIS_PARAM_FEATURE_trend_post_window": _resolve_feature_export_value(feature_config, "trend_post_window"),
            "ANALYSIS_PARAM_WINDOW_normalization_win_rel_go_cue_s": analysis_metadata.get("normalization_win_rel_cue_s", np.nan),
            "ANALYSIS_PARAM_WINDOW_baseline_win_rel_go_cue_s": analysis_metadata.get("baseline_win_rel_cue_s", np.nan),
            "ANALYSIS_PARAM_WINDOW_baseline_extended_win_rel_go_cue_s": analysis_metadata.get("baseline_extended_win_rel_cue_s", np.nan),
            "ANALYSIS_PARAM_WINDOW_ready_win_rel_go_cue_s": analysis_metadata.get("ready_win_rel_cue_s", np.nan),
            "ANALYSIS_PARAM_WINDOW_post_win_rel_go_cue_s": analysis_metadata.get("post_win_rel_cue_s", np.nan),
            "ANALYSIS_PARAM_WINDOW_obs_baseline_win_rel_go_cue_s": analysis_metadata.get("obs_baseline_window_rel_cue_s", np.nan),
            "ANALYSIS_PARAM_WINDOW_cst_modulation_normalization_win_rel_go_cue_s": analysis_metadata.get("cst_modulation_normalization_window_rel_cue_s", np.nan),
            "ANALYSIS_PARAM_FILTER_cst_lf_band_hz": analysis_metadata.get("lf_band_hz", _native_export_value(params.get("lf_band_hz", np.nan))),
            "ANALYSIS_PARAM_FILTER_cst_alpha_band_hz": analysis_metadata.get("alpha_band_hz", _native_export_value(params.get("alpha_band_hz", np.nan))),
            "ANALYSIS_PARAM_FILTER_cst_beta_band_hz": analysis_metadata.get("beta_band_hz", _native_export_value(params.get("beta_band_hz", np.nan))),
            "ANALYSIS_PARAM_FILTER_active_unit_rate_threshold_hz": analysis_metadata.get("active_unit_rate_threshold_hz", params.get("active_unit_rate_threshold_hz", np.nan)),
            "ANALYSIS_PARAM_SYNC_win_ms": analysis_metadata.get("sync_win_ms", params.get("sync_win_ms", np.nan)),
            "ANALYSIS_PARAM_SYNC_step_ms": analysis_metadata.get("sync_step_ms", params.get("sync_step_ms", np.nan)),
            "ANALYSIS_PARAM_SYNC_coinc_lag_ms": analysis_metadata.get("sync_coinc_lag_ms", params.get("sync_coinc_lag_ms", np.nan)),
            "ANALYSIS_PARAM_SYNC_direction_mode": analysis_metadata.get("sync_direction_mode", params.get("sync_direction_mode", np.nan)),
            "ANALYSIS_PARAM_SYNC_expectation_mode": analysis_metadata.get("sync_expectation_mode", params.get("sync_expectation_mode", np.nan)),
            "ANALYSIS_PARAM_SPECTRUM_window_rel_go_cue_s": analysis_metadata.get("modulation_spectrum_window_rel_cue_s", np.nan),
            "ANALYSIS_PARAM_SPECTRUM_max_freq_hz": analysis_metadata.get("modulation_spectrum_max_freq_hz", np.nan),
            "ANALYSIS_PARAM_SPECTRUM_interval_mass_pct": analysis_metadata.get("modulation_spectrum_interval_mass_pct", np.nan),
            "ANALYSIS_DIAGNOSIS_n_units_total": _mean_series_or_nan(pd.DataFrame({"x": total_counts}), "x"),
            "ANALYSIS_DIAGNOSIS_n_units_kept_mean": _mean_series_or_nan(pd.DataFrame({"x": kept_counts}), "x"),
            "ANALYSIS_DIAGNOSIS_n_units_kept_min": float(np.nanmin(kept_counts)) if np.any(np.isfinite(kept_counts)) else np.nan,
            "ANALYSIS_DIAGNOSIS_unit_fraction_kept_mean": _mean_series_or_nan(pd.DataFrame({"x": kept_fractions}), "x"),
            "ANALYSIS_DIAGNOSIS_fit_r2_cst_low_frequency": summary_row.get("cst_lf_fit_r2", np.nan),
            "ANALYSIS_DIAGNOSIS_fit_r2_cst_alpha_modulation_hilbert_zscore": summary_row.get("cst_alpha_mod_fit_r2", np.nan),
            "ANALYSIS_DIAGNOSIS_fit_r2_cst_beta_modulation_hilbert_zscore": summary_row.get("cst_beta_mod_fit_r2", np.nan),
            "ANALYSIS_DIAGNOSIS_fit_r2_sync_coincidence_curve": summary_row.get("sync_fit_r2", np.nan),
            "ANALYSIS_DIAGNOSIS_fit_r2_gain_over_linear_cst_low_frequency": summary_row.get("cst_lf_fit_r2_gain_over_linear", np.nan),
            "ANALYSIS_DIAGNOSIS_fit_r2_gain_over_linear_cst_alpha_modulation_hilbert_zscore": summary_row.get("cst_alpha_mod_fit_r2_gain_over_linear", np.nan),
            "ANALYSIS_DIAGNOSIS_fit_r2_gain_over_linear_cst_beta_modulation_hilbert_zscore": summary_row.get("cst_beta_mod_fit_r2_gain_over_linear", np.nan),
            "ANALYSIS_DIAGNOSIS_fit_r2_gain_over_linear_sync_coincidence_curve": summary_row.get("sync_fit_r2_gain_over_linear", np.nan),
        })
        row.update(observation_feature_means)
        rows.append(row)

    export_df = pd.DataFrame(rows)
    if export_df.empty:
        export_df = pd.DataFrame(columns=EXPORT_TABLE_COLUMNS)
    else:
        export_df = export_df.reindex(columns=EXPORT_TABLE_COLUMNS)
    return export_df


def export_parameter_set_summary_table(
    root_dir,
    output_dir=None,
    parameter_set_summary_df=None,
    rate_isi_metrics_df=None,
    feature_config=None,
    selected_parameter_set_ids=None,
    parameter_set_id_range=None,
    csv_name="parameter_set_feature_summary.csv",
    pickle_name="parameter_set_feature_summary.pkl",
):
    export_df = build_parameter_set_export_table(
        root_dir=root_dir,
        parameter_set_summary_df=parameter_set_summary_df,
        rate_isi_metrics_df=rate_isi_metrics_df,
        feature_config=feature_config,
        selected_parameter_set_ids=selected_parameter_set_ids,
        parameter_set_id_range=parameter_set_id_range,
    )
    output_dir = Path(output_dir) if output_dir is not None else Path(root_dir) / "parameter_set_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_df = export_df.copy()
    for column_name in csv_df.columns:
        csv_df[column_name] = csv_df[column_name].map(_csv_safe_value)

    csv_path = output_dir / csv_name
    pickle_path = output_dir / pickle_name
    csv_df.to_csv(csv_path, index=False)
    export_df.to_pickle(pickle_path)
    return export_df, {
        "csv_path": str(csv_path),
        "pickle_path": str(pickle_path),
    }
