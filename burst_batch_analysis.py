import json
import os
import warnings
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


FORCE_COLOR = "tab:blue"
TARGET_COLOR = "#08306B"
INPUT_COLOR = "tab:red"
BURST_COLOR = "#7f0000"
NEUTRAL_COLOR = "0.25"


DEFAULT_PLOT_STYLE = {
    "figure_dpi": 140,
    "summary_figsize": (17, 9),
    "summary_wspace": 0.28,
    "summary_hspace": 0.18,
    "grid_panel_size": (4.2, 3.2),
    "force_color": FORCE_COLOR,
    "target_color": TARGET_COLOR,
    "input_color": INPUT_COLOR,
    "burst_default_color": INPUT_COLOR,
    "lf_burst_color": "#2A9D8F",
    "alpha_burst_color": "#E63973",
    "beta_burst_color": "#E76F51",
    "neutral_color": NEUTRAL_COLOR,
    "target_lw": 2.0,
    "target_ls": "--",
    "target_alpha": 0.35,
    "force_trace_lw": 1.1,
    "force_trace_alpha": 0.12,
    "force_mean_lw": 2.4,
    "force_mean_alpha": 0.95,
    "input_trace_lw": 1.0,
    "input_trace_alpha": 0.10,
    "input_mean_lw": 2.0,
    "input_mean_alpha": 0.90,
    "burst_trace_lw": 1.0,
    "burst_trace_alpha": 0.10,
    "burst_mean_lw": 2.0,
    "burst_mean_alpha": 0.95,
    "deriv_trace_lw": 1.1,
    "deriv_trace_alpha": 0.12,
    "deriv_mean_lw": 2.0,
    "deriv_mean_alpha": 0.95,
    "sync_color": "#2166AC",
    "sync_grid_trace_lw": 0.9,
    "sync_grid_trace_alpha": 0.12,
    "sync_grid_mean_lw": 2.0,
    "sync_grid_mean_alpha": 0.95,
    "grid_force_trace_lw": 0.9,
    "grid_force_trace_alpha": 0.12,
    "grid_force_mean_lw": 2.0,
    "grid_force_mean_alpha": 0.95,
    "grid_deriv_trace_lw": 0.9,
    "grid_deriv_trace_alpha": 0.12,
    "grid_deriv_mean_lw": 2.0,
    "grid_deriv_mean_alpha": 0.95,
    "marker_center_color": "black",
    "marker_center_lw": 1.3,
    "marker_center_ls": "-",
    "marker_center_alpha": 0.90,
    "marker_start_color": "0.35",
    "marker_start_lw": 1.1,
    "marker_start_ls": "--",
    "marker_start_alpha": 0.90,
    "zoom_force_box_color": NEUTRAL_COLOR,
    "zoom_force_box_lw": 1.8,
    "zoom_force_box_ls": "--",
    "zoom_input_box_color": "0.35",
    "zoom_input_box_lw": 1.6,
    "zoom_input_box_ls": ":",
    "grid_zero_line_color": "black",
    "grid_zero_line_lw": 1.1,
    "grid_zero_line_alpha": 0.90,
    "grid_baseline_line_color": NEUTRAL_COLOR,
    "grid_baseline_line_lw": 0.8,
    "grid_baseline_line_alpha": 0.50,
    "envelope_fill_alpha": 0.10,
    "envelope_edge_lw": 1.15,
    "envelope_edge_alpha": 0.65,
    "grid_title_fontsize": 9,
    "grid_suptitle_y": 0.995,
}


def _resolve_plot_style(plot_style=None):
    resolved = dict(DEFAULT_PLOT_STYLE)
    if plot_style:
        resolved.update(plot_style)
    return resolved


def _resolve_component_color(component_name, plot_style, component_color_map=None):
    if component_color_map and component_name in component_color_map:
        return component_color_map[component_name]
    return plot_style.get(f"{component_name}_burst_color", plot_style["burst_default_color"])


def _safe_array(dataset):
    arr = np.asarray(dataset, dtype=float)
    if arr.size == 1 and not np.isfinite(arr[0]):
        return None
    return arr


def _auto_limits(data, zero_floor=False, pad_frac=0.08, min_span=1.0):
    data = np.asarray(data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return (0.0, 1.0) if zero_floor else (-1.0, 1.0)
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if zero_floor:
        y_min = 0.0
    span = y_max - y_min
    if span < 1e-12:
        pad = max(min_span * 0.5, abs(y_max) * pad_frac, 0.5)
    else:
        pad = span * pad_frac
    return y_min - (0.0 if zero_floor else pad), y_max + pad


def _format_band(band):
    return f"{band[0]:g}-{band[1]:g}Hz"


def _active_burst_components(combo, tol=1e-12):
    _, lf_peak, alpha_peak, beta_peak = combo
    components = []
    for name, peak in (("lf", lf_peak), ("alpha", alpha_peak), ("beta", beta_peak)):
        if abs(float(peak)) > tol:
            components.append((name, float(peak)))
    return components


def _combo_signature_label(combo):
    active_components = _active_burst_components(combo)
    if not active_components:
        return "no burst"
    return ", ".join(f"{name}={peak:+g}" for name, peak in active_components)


def _combo_slug(combo):
    sigma_ms, lf_peak, alpha_peak, beta_peak = combo
    return (
        f"sigma_{sigma_ms:g}ms__"
        f"lf_{lf_peak:+g}__alpha_{alpha_peak:+g}__beta_{beta_peak:+g}"
    ).replace("+", "p").replace("-", "m")


def _combo_title(combo):
    sigma_ms, _, _, _ = combo
    return f"Burst sigma={sigma_ms:g} ms | {_combo_signature_label(combo)}"


def _signature_key(combo):
    _, lf_peak, alpha_peak, beta_peak = combo
    return (float(lf_peak), float(alpha_peak), float(beta_peak))


def _signature_label(signature):
    combo = (np.nan, signature[0], signature[1], signature[2])
    return _combo_signature_label(combo)


def _load_simulation_run(run_dir):
    return _load_simulation_run_core(run_dir, load_spikes=False)


def _load_simulation_run_core(run_dir, load_spikes=False):
    run_dir = Path(run_dir)
    sim_json = run_dir / "sim_parameters.json"
    sim_h5 = run_dir / "simulation_output.h5"
    if not sim_json.exists() or not sim_h5.exists():
        return None

    with sim_json.open("r", encoding="utf-8") as f:
        params = json.load(f)

    with h5py.File(sim_h5, "r") as f:
        fsamp = int(f["forces"].attrs["fsamp_Hz"])
        force = np.asarray(f["forces"]["pool_force_percent"], dtype=float)
        common_input = np.asarray(f["input"]["common_input"], dtype=float)
        di = f["driving_inputs"]
        comp = f["input_components"] if "input_components" in f else None
        target = _safe_array(di["target_percent_ts"]) if "target_percent_ts" in di else None
        baseline_ts_nA = _safe_array(di["final_baseline_ts_nA"]) if "final_baseline_ts_nA" in di else None
        burst_trace_nA = _safe_array(di["input_burst_trace_nA"]) if "input_burst_trace_nA" in di else None
        burst_center_arr = _safe_array(di["input_burst_center_s"]) if "input_burst_center_s" in di else None
        burst_start_arr = _safe_array(di["input_burst_start_s"]) if "input_burst_start_s" in di else None
        burst_zoom_start_arr = _safe_array(di["input_burst_zoom_start_s"]) if "input_burst_zoom_start_s" in di else None
        burst_kernel_by_component = {}
        if comp is not None:
            for component_name in ("lf", "alpha", "beta"):
                dataset_name = f"{component_name}_burst_kernel"
                if dataset_name in comp:
                    burst_kernel_by_component[component_name] = np.asarray(comp[dataset_name], dtype=float)
        mn_spike_trains = None
        if load_spikes:
            mn_grp = f["spike_trains"]["MN"]
            mn_spike_trains = {
                key: np.asarray(mn_grp[key], dtype=float)
                for key in sorted(mn_grp.keys(), key=lambda name: int(name.split("_")[-1]))
            }

    if target is None:
        target = np.zeros_like(force)

    if baseline_ts_nA is None:
        baseline_vals = params.get("excitatory_input_baseline", [0.0])
        baseline_ts_nA = np.full(force.shape[0], float(baseline_vals[0]), dtype=float)
    else:
        baseline_ts_nA = np.asarray(baseline_ts_nA, dtype=float).squeeze()
        if baseline_ts_nA.ndim == 0:
            baseline_ts_nA = np.full(force.shape[0], float(baseline_ts_nA), dtype=float)
        elif baseline_ts_nA.size != force.shape[0]:
            baseline_ts_nA = np.full(force.shape[0], float(baseline_ts_nA.flat[0]), dtype=float)

    T = min(force.shape[0], common_input.shape[1], baseline_ts_nA.shape[0], target.shape[0])
    force = force[:T]
    target = target[:T]
    baseline_ts_nA = baseline_ts_nA[:T]
    common_input_pool0_nA = common_input[0, :T]
    net_input_uA = (common_input_pool0_nA + baseline_ts_nA) / 1000.0
    time_s = np.arange(T, dtype=float) / float(fsamp)

    burst_trace_uA = None
    if burst_trace_nA is not None:
        burst_trace_uA = np.asarray(burst_trace_nA, dtype=float).squeeze()[:T] / 1000.0
    burst_kernel_by_component = {
        component_name: np.asarray(kernel, dtype=float).squeeze()[:T]
        for component_name, kernel in burst_kernel_by_component.items()
    }

    burst_center_s = float(burst_center_arr[0]) if burst_center_arr is not None else None
    burst_start_s = float(burst_start_arr[0]) if burst_start_arr is not None else None
    if burst_zoom_start_arr is not None:
        burst_zoom_start_s = float(burst_zoom_start_arr[0])
    elif burst_center_s is not None:
        burst_zoom_start_s = (
            burst_center_s
            - float(params.get("input_burst_zoom_start_n_sigma", 4.0))
            * float(params.get("input_burst_sigma_ms", 250.0))
            / 1000.0
        )
    else:
        burst_zoom_start_s = None

    return {
        "run_dir": run_dir,
        "params": params,
        "fsamp": fsamp,
        "time_s": time_s,
        "force": force,
        "target": target,
        "net_input_uA": net_input_uA,
        "burst_trace_uA": burst_trace_uA,
        "burst_kernel_by_component": burst_kernel_by_component,
        "burst_center_s": burst_center_s,
        "burst_start_s": burst_start_s,
        "burst_zoom_start_s": burst_zoom_start_s,
        "dforce_dt": np.gradient(force, 1.0 / float(fsamp)),
        "mn_spike_trains": mn_spike_trains,
    }


def collect_burst_runs(root_dir, require_burst_enabled=True, load_spikes=False,
                       show_progress=False, progress_desc="Loading runs"):
    root = Path(root_dir)
    if show_progress:
        print(f"Scanning '{root}' for simulation_output.h5 files...")
    h5_paths = sorted(root.rglob("simulation_output.h5"))
    if show_progress:
        print(f"Found {len(h5_paths)} simulation file(s).")
    runs = []
    iterator = h5_paths
    if show_progress and tqdm is not None:
        iterator = tqdm(
            h5_paths,
            total=len(h5_paths),
            desc=progress_desc,
            unit="file",
            smoothing=0.1,
        )
    for idx, h5_path in enumerate(iterator, start=1):
        run = _load_simulation_run_core(h5_path.parent, load_spikes=load_spikes)
        if run is None:
            continue
        if require_burst_enabled and not bool(run["params"].get("enable_input_burst", False)):
            continue
        runs.append(run)
        if show_progress and tqdm is None and (idx % 100 == 0 or idx == len(h5_paths)):
            print(f"{progress_desc}: loaded {idx}/{len(h5_paths)} file(s)...")
    return runs


def group_runs_by_burst_combo(runs):
    groups = {}
    for run in runs:
        params = run["params"]
        combo = _burst_combo_from_params(params)
        groups.setdefault(combo, []).append(run)
    return groups


def _burst_combo_from_params(params):
    return (
        float(params.get("input_burst_sigma_ms", np.nan)),
        float(params.get("lf_burst_peak_nA", params.get("lf_burst_peak_delta", 0.0))),
        float(params.get("alpha_burst_peak_nA", params.get("alpha_burst_peak_delta", 0.0))),
        float(params.get("beta_burst_peak_nA", params.get("beta_burst_peak_delta", 0.0))),
    )


DEFAULT_SPIKE_SYNC_CONFIG = {
    "analysis_window_start": 0.0,
    "analysis_window_end": 1.0,
    "match_force_grid_window_when_burst_present": True,
    "zoom_duration_s": 3.0,
    "padding_before": 0.25,
    "padding_after": 0.25,
    "active_unit_rate_threshold_hz": 5.0,
    "profile_dt": 0.005,
    "selected_unit_ids": None,
    "min_active_units": 2,
}


def _resolve_spike_sync_config(spike_sync_config=None):
    resolved = dict(DEFAULT_SPIKE_SYNC_CONFIG)
    if spike_sync_config:
        resolved.update(spike_sync_config)
    return resolved


def _resolve_analysis_window_for_run(run, config):
    sim_start = float(run["time_s"][0])
    sim_end = float(run["time_s"][-1])
    if (
        bool(config.get("match_force_grid_window_when_burst_present", False))
        and run.get("burst_center_s") is not None
        and run.get("burst_zoom_start_s") is not None
    ):
        zoom_duration_s = config.get("zoom_duration_s", None)
        if zoom_duration_s is None or float(zoom_duration_s) <= 0:
            raise ValueError(
                "SPIKE-synchronization config requires a positive 'zoom_duration_s' when "
                "'match_force_grid_window_when_burst_present' is enabled."
            )
        analysis_start = max(sim_start, float(run["burst_zoom_start_s"]))
        analysis_end = min(sim_end, analysis_start + float(zoom_duration_s))
        analysis_source = "burst_zoom_window"
    else:
        analysis_start = float(config["analysis_window_start"])
        analysis_end = float(config["analysis_window_end"])
        analysis_source = "explicit_window"
    return analysis_start, analysis_end, analysis_source


def _normalize_unit_id(unit_id):
    if isinstance(unit_id, str):
        if unit_id.startswith("MN_"):
            return f"MN_{int(unit_id.split('_')[-1])}"
        return f"MN_{int(unit_id)}"
    return f"MN_{int(unit_id)}"


def _normalize_selected_units(selected_unit_ids, available_unit_ids):
    if selected_unit_ids is None:
        return list(available_unit_ids)
    requested = [_normalize_unit_id(unit_id) for unit_id in selected_unit_ids]
    missing = [unit_id for unit_id in requested if unit_id not in available_unit_ids]
    if missing:
        raise ValueError(f"Requested motor units are not present in the run: {missing}")
    return requested


def _restrict_spikes_to_window(spike_times_s, window_start, window_end):
    spikes = np.asarray(spike_times_s, dtype=float)
    if spikes.size == 0:
        return spikes
    return spikes[(spikes >= window_start) & (spikes <= window_end)]


def _compute_window_rate_hz(spike_times_s, window_start, window_end):
    duration = float(window_end) - float(window_start)
    if duration <= 0:
        raise ValueError("Analysis window duration must be positive.")
    return _restrict_spikes_to_window(spike_times_s, window_start, window_end).size / duration


def _neighbor_intervals(spike_times_s, window_start, window_end):
    spikes = np.asarray(spike_times_s, dtype=float)
    prev_isi = np.empty(spikes.size, dtype=float)
    next_isi = np.empty(spikes.size, dtype=float)
    if spikes.size == 0:
        return prev_isi, next_isi
    prev_isi[0] = max(spikes[0] - window_start, 0.0)
    next_isi[-1] = max(window_end - spikes[-1], 0.0)
    if spikes.size > 1:
        diffs = np.diff(spikes)
        prev_isi[1:] = diffs
        next_isi[:-1] = diffs
    else:
        prev_isi[0] = max(spikes[0] - window_start, 0.0)
        next_isi[0] = max(window_end - spikes[0], 0.0)
    return prev_isi, next_isi


def _nearest_spike_index(spike_times_s, spike_time_s):
    spikes = np.asarray(spike_times_s, dtype=float)
    if spikes.size == 0:
        return None
    insert_at = int(np.searchsorted(spikes, spike_time_s))
    candidates = []
    if insert_at < spikes.size:
        candidates.append(insert_at)
    if insert_at > 0:
        candidates.append(insert_at - 1)
    if not candidates:
        return None
    return min(candidates, key=lambda idx: abs(spikes[idx] - spike_time_s))


def _piecewise_profile_from_spikes(spike_times_s, spike_values, computation_start, computation_end, sample_times_s):
    spikes = np.asarray(spike_times_s, dtype=float)
    values = np.asarray(spike_values, dtype=float)
    sample_times_s = np.asarray(sample_times_s, dtype=float)
    if spikes.size == 0:
        return np.zeros(sample_times_s.size, dtype=float)
    if spikes.size == 1:
        return np.full(sample_times_s.size, values[0], dtype=float)
    boundaries = np.empty(spikes.size + 1, dtype=float)
    boundaries[0] = float(computation_start)
    boundaries[-1] = float(computation_end)
    boundaries[1:-1] = 0.5 * (spikes[:-1] + spikes[1:])
    interval_index = np.digitize(sample_times_s, boundaries[1:-1], right=False)
    return values[interval_index]


def _compute_bivariate_spike_sync_profile(spike_train_a, spike_train_b, computation_start, computation_end, sample_times_s):
    spikes_a = np.asarray(spike_train_a, dtype=float)
    spikes_b = np.asarray(spike_train_b, dtype=float)
    if spikes_a.size == 0 or spikes_b.size == 0:
        return np.zeros_like(sample_times_s, dtype=float), np.zeros(spikes_a.size), np.zeros(spikes_b.size)

    prev_a, next_a = _neighbor_intervals(spikes_a, computation_start, computation_end)
    prev_b, next_b = _neighbor_intervals(spikes_b, computation_start, computation_end)
    coinc_a = np.zeros(spikes_a.size, dtype=float)
    coinc_b = np.zeros(spikes_b.size, dtype=float)

    for i, spike_a in enumerate(spikes_a):
        j = _nearest_spike_index(spikes_b, spike_a)
        if j is None:
            continue
        if _nearest_spike_index(spikes_a, spikes_b[j]) != i:
            continue
        tau = 0.5 * min(prev_a[i], next_a[i], prev_b[j], next_b[j])
        if abs(spike_a - spikes_b[j]) <= tau + 1e-12:
            coinc_a[i] = 1.0
            coinc_b[j] = 1.0

    profile_a = _piecewise_profile_from_spikes(spikes_a, coinc_a, computation_start, computation_end, sample_times_s)
    profile_b = _piecewise_profile_from_spikes(spikes_b, coinc_b, computation_start, computation_end, sample_times_s)
    return 0.5 * (profile_a + profile_b), coinc_a, coinc_b


def _compute_multivariate_spike_sync_profile(spike_trains_by_unit, computation_start, computation_end, sample_times_s):
    unit_ids = list(spike_trains_by_unit.keys())
    if len(unit_ids) < 2:
        raise ValueError("At least two active units are required for multivariate SPIKE-synchronization.")
    pair_profiles = []
    for unit_a, unit_b in combinations(unit_ids, 2):
        pair_profile, _, _ = _compute_bivariate_spike_sync_profile(
            spike_trains_by_unit[unit_a],
            spike_trains_by_unit[unit_b],
            computation_start,
            computation_end,
            sample_times_s,
        )
        pair_profiles.append(pair_profile)
    return _nanmean_no_warning(np.stack(pair_profiles, axis=0), axis=0)


def _mean_profile_value(sample_times_s, sample_values):
    sample_times_s = np.asarray(sample_times_s, dtype=float)
    sample_values = np.asarray(sample_values, dtype=float)
    if sample_values.size == 0:
        return np.nan
    if sample_values.size == 1:
        return float(sample_values[0])
    duration = float(sample_times_s[-1] - sample_times_s[0])
    if duration <= 0:
        return float(np.nanmean(sample_values))
    return float(np.trapz(sample_values, sample_times_s) / duration)


def compute_run_spike_synchrony(run, spike_sync_config=None):
    config = _resolve_spike_sync_config(spike_sync_config)
    if run.get("mn_spike_trains") is None:
        raise ValueError("Run does not include motor-unit spike trains. Reload it with load_spikes=True.")

    analysis_start, analysis_end, analysis_source = _resolve_analysis_window_for_run(run, config)
    padding_before = float(config["padding_before"])
    padding_after = float(config["padding_after"])
    rate_threshold = float(config["active_unit_rate_threshold_hz"])
    profile_dt = float(config["profile_dt"])
    min_active_units = int(config["min_active_units"])

    if analysis_end <= analysis_start:
        raise ValueError("analysis_window_end must be greater than analysis_window_start.")
    if padding_before < 0 or padding_after < 0:
        raise ValueError("padding_before and padding_after must be non-negative.")
    if profile_dt <= 0:
        raise ValueError("profile_dt must be strictly positive.")
    if min_active_units < 2:
        raise ValueError("min_active_units must be at least 2.")

    sim_start = float(run["time_s"][0])
    sim_end = float(run["time_s"][-1])
    if analysis_start < sim_start or analysis_end > sim_end + (1.0 / max(float(run["fsamp"]), 1.0)):
        raise ValueError(
            f"Analysis window [{analysis_start:.3f}, {analysis_end:.3f}] is outside the available simulation time "
            f"[{sim_start:.3f}, {sim_end:.3f}]."
        )

    computation_start = max(sim_start, analysis_start - padding_before)
    computation_end = min(sim_end, analysis_end + padding_after)
    if computation_end <= computation_start:
        raise ValueError("Padded computation window is empty after clipping to the simulation duration.")

    available_unit_ids = list(run["mn_spike_trains"].keys())
    candidate_unit_ids = _normalize_selected_units(config["selected_unit_ids"], available_unit_ids)
    candidate_trains = {unit_id: np.asarray(run["mn_spike_trains"][unit_id], dtype=float) for unit_id in candidate_unit_ids}

    active_unit_ids = []
    excluded_unit_ids = []
    rates_hz = {}
    for unit_id in candidate_unit_ids:
        rate_hz = _compute_window_rate_hz(candidate_trains[unit_id], analysis_start, analysis_end)
        rates_hz[unit_id] = rate_hz
        if rate_hz >= rate_threshold:
            active_unit_ids.append(unit_id)
        else:
            excluded_unit_ids.append(unit_id)

    result = {
        "run_dir": run["run_dir"],
        "status": "ok",
        "message": "",
        "analysis_window_source": analysis_source,
        "analysis_window_start": analysis_start,
        "analysis_window_end": analysis_end,
        "computation_window_start": computation_start,
        "computation_window_end": computation_end,
        "candidate_unit_ids": candidate_unit_ids,
        "active_unit_ids": active_unit_ids,
        "excluded_unit_ids": excluded_unit_ids,
        "candidate_unit_rates_hz": rates_hz,
        "n_candidate_units": len(candidate_unit_ids),
        "n_active_units": len(active_unit_ids),
        "sync_time_s": np.array([], dtype=float),
        "sync_profile": np.array([], dtype=float),
        "average_spike_sync": np.nan,
        "burst_center_s": run["burst_center_s"],
        "burst_start_s": run["burst_start_s"],
        "burst_kernel_by_component": run["burst_kernel_by_component"],
        "time_s": run["time_s"],
        "combo": _burst_combo_from_params(run["params"]),
    }

    if len(active_unit_ids) == 0:
        result["status"] = "skipped"
        result["message"] = "No units passed the active-unit firing-rate threshold."
        return result
    if len(active_unit_ids) < min_active_units:
        result["status"] = "skipped"
        result["message"] = "Too few active units for multivariate SPIKE-synchronization."
        return result

    active_spike_trains = {
        unit_id: _restrict_spikes_to_window(candidate_trains[unit_id], computation_start, computation_end)
        for unit_id in active_unit_ids
    }
    empty_units = [unit_id for unit_id, spikes in active_spike_trains.items() if spikes.size == 0]
    if empty_units:
        result["status"] = "skipped"
        result["message"] = f"Active units had empty spike trains in the padded computation window: {empty_units}"
        return result

    sync_time_s = np.arange(analysis_start, analysis_end + 0.5 * profile_dt, profile_dt, dtype=float)
    if sync_time_s.size == 0 or sync_time_s[-1] < analysis_end - 1e-12:
        sync_time_s = np.append(sync_time_s, analysis_end)
    elif sync_time_s[-1] > analysis_end + 1e-12:
        sync_time_s[-1] = analysis_end

    sync_profile = _compute_multivariate_spike_sync_profile(
        active_spike_trains,
        computation_start,
        computation_end,
        sync_time_s,
    )
    result["sync_time_s"] = sync_time_s
    result["sync_profile"] = sync_profile
    result["average_spike_sync"] = _mean_profile_value(sync_time_s, sync_profile)
    return result


def _mean_stack(series_list):
    min_len = min(len(x) for x in series_list)
    stack = np.stack([np.asarray(x[:min_len], dtype=float) for x in series_list], axis=0)
    return stack, _nanmean_no_warning(stack, axis=0)


def _nanmean_no_warning(arr, axis=0):
    with np.errstate(invalid="ignore"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            return np.nanmean(arr, axis=axis)


def plot_burst_combo_summary(combo, runs, out_dir, zoom_duration_s=3.0, plot_style=None, component_color_map=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_style = _resolve_plot_style(plot_style)

    time_s = runs[0]["time_s"]
    force_stack, force_mean = _mean_stack([r["force"] for r in runs])
    target_stack, target_mean = _mean_stack([r["target"] for r in runs])
    input_stack, input_mean = _mean_stack([r["net_input_uA"] for r in runs])
    dforce_stack, dforce_mean = _mean_stack([r["dforce_dt"] for r in runs])

    burst_stack = None
    burst_mean = None
    if all(r["burst_trace_uA"] is not None for r in runs):
        burst_stack, burst_mean = _mean_stack([r["burst_trace_uA"] for r in runs])
    burst_kernels_by_component = {}
    for component_name in ("lf", "alpha", "beta"):
        eligible = [r for r in runs if component_name in r["burst_kernel_by_component"]]
        if eligible:
            stack, mean_kernel = _mean_stack([r["burst_kernel_by_component"][component_name] for r in eligible])
            burst_kernels_by_component[component_name] = (stack, mean_kernel)

    burst_center_s = np.nanmean([r["burst_center_s"] for r in runs if r["burst_center_s"] is not None])
    burst_start_s = np.nanmean([r["burst_start_s"] for r in runs if r["burst_start_s"] is not None])
    burst_zoom_start_s = np.nanmean([r["burst_zoom_start_s"] for r in runs if r["burst_zoom_start_s"] is not None])
    if not np.isfinite(burst_zoom_start_s):
        burst_zoom_start_s = burst_center_s - 0.5 * zoom_duration_s

    zoom_start_s = max(float(time_s[0]), float(burst_zoom_start_s))
    zoom_end_s = min(float(time_s[-1]), zoom_start_s + float(zoom_duration_s))
    zoom_mask = (time_s >= zoom_start_s) & (time_s <= zoom_end_s)
    if not np.any(zoom_mask):
        zoom_mask[max(0, len(time_s) // 2 - 1):min(len(time_s), len(time_s) // 2 + 2)] = True

    fig = plt.figure(figsize=plot_style["summary_figsize"], dpi=plot_style["figure_dpi"])
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        width_ratios=[1.7, 1.0],
        height_ratios=[1, 1, 1],
        wspace=plot_style["summary_wspace"], hspace=plot_style["summary_hspace"]
    )
    ax_force_main = fig.add_subplot(gs[:, 0])
    ax_input_main = ax_force_main.twinx()
    ax_zoom_input = fig.add_subplot(gs[0, 1])
    ax_zoom_force = fig.add_subplot(gs[1, 1], sharex=ax_zoom_input)
    ax_zoom_deriv = fig.add_subplot(gs[2, 1], sharex=ax_zoom_input)

    def _add_markers(ax):
        ax.axvline(
            burst_center_s,
            color=plot_style["marker_center_color"],
            lw=plot_style["marker_center_lw"],
            ls=plot_style["marker_center_ls"],
            alpha=plot_style["marker_center_alpha"],
            zorder=40,
        )
        ax.axvline(
            burst_start_s,
            color=plot_style["marker_start_color"],
            lw=plot_style["marker_start_lw"],
            ls=plot_style["marker_start_ls"],
            alpha=plot_style["marker_start_alpha"],
            zorder=40,
        )

    ax_force_main.plot(
        time_s, target_mean,
        color=plot_style["target_color"],
        lw=plot_style["target_lw"],
        ls=plot_style["target_ls"],
        alpha=plot_style["target_alpha"],
        label="Target",
    )
    for trace in force_stack:
        ax_force_main.plot(
            time_s, trace,
            color=plot_style["force_color"],
            lw=plot_style["force_trace_lw"],
            alpha=plot_style["force_trace_alpha"],
            zorder=2,
        )
    ax_force_main.plot(
        time_s, force_mean,
        color=plot_style["force_color"],
        lw=plot_style["force_mean_lw"],
        alpha=plot_style["force_mean_alpha"],
        label="Mean force",
        zorder=5,
    )
    ax_force_main.set_xlim(time_s[0], time_s[-1])
    ax_force_main.set_ylim(0, 100)
    ax_force_main.set_xlabel("Time (s)")
    ax_force_main.set_ylabel("Force (% MVC)")
    ax_force_main.set_title(_combo_title(combo))
    ax_force_main.grid(alpha=0.25)

    for trace in input_stack:
        ax_input_main.plot(
            time_s, trace,
            color=plot_style["input_color"],
            lw=plot_style["input_trace_lw"],
            alpha=plot_style["input_trace_alpha"],
            zorder=1,
        )
    ax_input_main.plot(
        time_s, input_mean,
        color=plot_style["input_color"],
        lw=plot_style["input_mean_lw"],
        alpha=plot_style["input_mean_alpha"],
        label="Mean net input",
        zorder=3,
    )
    _, input_ymax = _auto_limits(input_stack, zero_floor=True, min_span=0.5)
    ax_input_main.set_ylim(0, input_ymax)
    ax_input_main.set_ylabel("Net common input (uA)")

    force_zoom_y0, force_zoom_y1 = _auto_limits(
        np.concatenate([force_stack[:, zoom_mask], target_mean[zoom_mask][None, :]], axis=0),
        zero_floor=False,
        min_span=2.0,
    )
    ax_force_main.add_patch(
        Rectangle(
            (zoom_start_s, force_zoom_y0), zoom_end_s - zoom_start_s, force_zoom_y1 - force_zoom_y0,
            fill=False,
            ec=plot_style["zoom_force_box_color"],
            lw=plot_style["zoom_force_box_lw"],
            ls=plot_style["zoom_force_box_ls"],
            zorder=30,
        )
    )
    input_zoom_y0, input_zoom_y1 = _auto_limits(input_stack[:, zoom_mask], zero_floor=False, min_span=0.5)
    ax_input_main.add_patch(
        Rectangle(
            (zoom_start_s, max(0.0, input_zoom_y0)), zoom_end_s - zoom_start_s, input_zoom_y1 - max(0.0, input_zoom_y0),
            fill=False,
            ec=plot_style["zoom_input_box_color"],
            lw=plot_style["zoom_input_box_lw"],
            ls=plot_style["zoom_input_box_ls"],
            zorder=30,
        )
    )
    _add_markers(ax_force_main)

    for trace in input_stack:
        ax_zoom_input.plot(
            time_s[zoom_mask], trace[zoom_mask],
            color=plot_style["input_color"],
            lw=plot_style["input_trace_lw"],
            alpha=plot_style["input_trace_alpha"],
            zorder=2,
        )
    ax_zoom_input.plot(
        time_s[zoom_mask], input_mean[zoom_mask],
        color=plot_style["input_color"],
        lw=plot_style["input_mean_lw"],
        alpha=plot_style["input_mean_alpha"],
        zorder=4,
    )
    ax_zoom_input.set_ylim(input_zoom_y0, input_zoom_y1)
    ax_zoom_input.set_ylabel("Input (uA)")
    ax_zoom_input.set_title("Zoomed common input")
    ax_zoom_input.grid(alpha=0.25)
    if burst_mean is not None:
        overlay_center = 0.5 * sum(ax_zoom_input.get_ylim())
        for trace in burst_stack:
            ax_zoom_input.plot(
                time_s[zoom_mask], overlay_center + trace[zoom_mask],
                color=plot_style["burst_default_color"],
                lw=plot_style["burst_trace_lw"],
                alpha=plot_style["burst_trace_alpha"],
                zorder=3,
            )
        ax_zoom_input.plot(
            time_s[zoom_mask], overlay_center + burst_mean[zoom_mask],
            color=plot_style["burst_default_color"],
            lw=plot_style["burst_mean_lw"],
            alpha=plot_style["burst_mean_alpha"],
            zorder=6,
        )
    for component_name, (_, mean_kernel) in burst_kernels_by_component.items():
        component_color = _resolve_component_color(component_name, plot_style, component_color_map)
        ax_zoom_input.plot(
            time_s[zoom_mask],
            np.nanmean(ax_zoom_input.get_ylim()) + mean_kernel[zoom_mask],
            color=component_color,
            lw=1.0,
            alpha=0.8,
            zorder=5,
        )

    for trace in force_stack:
        ax_zoom_force.plot(
            time_s[zoom_mask], trace[zoom_mask],
            color=plot_style["force_color"],
            lw=plot_style["force_trace_lw"],
            alpha=plot_style["force_trace_alpha"],
            zorder=2,
        )
    ax_zoom_force.plot(
        time_s[zoom_mask], target_mean[zoom_mask],
        color=plot_style["target_color"],
        lw=plot_style["target_lw"],
        ls=plot_style["target_ls"],
        alpha=plot_style["target_alpha"],
        zorder=3,
    )
    ax_zoom_force.plot(
        time_s[zoom_mask], force_mean[zoom_mask],
        color=plot_style["force_color"],
        lw=plot_style["force_mean_lw"],
        alpha=plot_style["force_mean_alpha"],
        zorder=4,
    )
    ax_zoom_force.set_ylim(force_zoom_y0, force_zoom_y1)
    ax_zoom_force.set_ylabel("Force (%MVC)")
    ax_zoom_force.set_title("Zoomed force")
    ax_zoom_force.grid(alpha=0.25)

    deriv_zoom_y0, deriv_zoom_y1 = _auto_limits(dforce_stack[:, zoom_mask], zero_floor=False, min_span=5.0)
    for trace in dforce_stack:
        ax_zoom_deriv.plot(
            time_s[zoom_mask], trace[zoom_mask],
            color=plot_style["force_color"],
            lw=plot_style["deriv_trace_lw"],
            alpha=plot_style["deriv_trace_alpha"],
            zorder=2,
        )
    ax_zoom_deriv.plot(
        time_s[zoom_mask], dforce_mean[zoom_mask],
        color=plot_style["force_color"],
        lw=plot_style["deriv_mean_lw"],
        alpha=plot_style["deriv_mean_alpha"],
        zorder=4,
    )
    ax_zoom_deriv.axhline(
        0,
        color=plot_style["grid_baseline_line_color"],
        lw=plot_style["grid_baseline_line_lw"],
        alpha=plot_style["grid_baseline_line_alpha"],
        zorder=1,
    )
    ax_zoom_deriv.set_ylim(deriv_zoom_y0, deriv_zoom_y1)
    ax_zoom_deriv.set_xlabel("Time (s)")
    ax_zoom_deriv.set_ylabel("dForce/dt\n(%MVC/s)")
    ax_zoom_deriv.set_title("Zoomed force derivative")
    ax_zoom_deriv.grid(alpha=0.25)

    for ax in (ax_zoom_input, ax_zoom_force, ax_zoom_deriv):
        ax.set_xlim(zoom_start_s, zoom_end_s)
        _add_markers(ax)

    fig.tight_layout()
    out_path = out_dir / f"{_combo_slug(combo)}__summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _relative_zoom_grid(runs, zoom_duration_s):
    fsamp = float(runs[0]["fsamp"])
    dt = 1.0 / fsamp
    start_candidates = []
    end_candidates = []
    for run in runs:
        left = float(run["burst_zoom_start_s"] - run["burst_center_s"])
        right = left + float(zoom_duration_s)
        start_candidates.append(left)
        end_candidates.append(right)
    x_common = np.arange(min(start_candidates), max(end_candidates) + 0.5 * dt, dt)
    return x_common


def _interp_to_relative_grid(run, x_common, key):
    x_rel = run["time_s"] - float(run["burst_center_s"])
    y = np.asarray(run[key], dtype=float)
    return np.interp(x_common, x_rel, y, left=np.nan, right=np.nan)


def _hide_env_axis_spines(ax_env):
    ax_env.patch.set_alpha(0.0)
    ax_env.spines["left"].set_visible(False)
    ax_env.spines["top"].set_visible(False)
    ax_env.spines["bottom"].set_visible(False)


def plot_burst_combo_grids(groups, out_dir, zoom_duration_s=3.0, plot_style=None, component_color_map=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_style = _resolve_plot_style(plot_style)
    combos = sorted(groups.keys(), key=lambda c: (c[0], c[1], c[2], c[3]))
    sigmas = sorted({combo[0] for combo in combos})
    signatures = sorted({_signature_key(combo) for combo in combos})
    all_runs = [run for runs in groups.values() for run in runs]
    x_common = _relative_zoom_grid(all_runs, zoom_duration_s=zoom_duration_s)

    force_by_combo = {}
    dforce_by_combo = {}
    burst_kernels_by_combo = {}
    all_force_vals = []
    all_dforce_vals = []
    all_kernel_vals = []
    for combo, runs in groups.items():
        force_stack = np.stack([_interp_to_relative_grid(run, x_common, "force") for run in runs], axis=0)
        dforce_stack = np.stack([_interp_to_relative_grid(run, x_common, "dforce_dt") for run in runs], axis=0)
        force_by_combo[combo] = (force_stack, _nanmean_no_warning(force_stack, axis=0))
        dforce_by_combo[combo] = (dforce_stack, _nanmean_no_warning(dforce_stack, axis=0))
        all_force_vals.append(force_stack)
        all_dforce_vals.append(dforce_stack)

        component_kernels = {}
        for component_name in ("lf", "alpha", "beta"):
            eligible = [run for run in runs if component_name in run["burst_kernel_by_component"]]
            if eligible:
                kernel_stack = np.stack(
                    [
                        np.interp(
                            x_common,
                            np.asarray(run["time_s"], dtype=float) - float(run["burst_center_s"]),
                            np.asarray(run["burst_kernel_by_component"][component_name], dtype=float),
                            left=np.nan,
                            right=np.nan,
                        )
                        for run in eligible
                    ],
                    axis=0,
                )
                component_kernels[component_name] = (kernel_stack, _nanmean_no_warning(kernel_stack, axis=0))
                all_kernel_vals.append(kernel_stack)
        burst_kernels_by_combo[combo] = component_kernels

    force_ylim = _auto_limits(np.concatenate(all_force_vals, axis=0), zero_floor=False, min_span=2.0)
    dforce_ylim = _auto_limits(np.concatenate(all_dforce_vals, axis=0), zero_floor=False, min_span=5.0)
    if all_kernel_vals:
        kernel_vals = np.concatenate(all_kernel_vals, axis=0)
        kernel_abs_max = float(np.nanmax(np.abs(kernel_vals)))
        kernel_limit = max(kernel_abs_max * 1.08, 0.25)
        kernel_ylim = (-kernel_limit, kernel_limit)
    else:
        kernel_ylim = (-1.0, 1.0)

    def _make_grid(data_map, ylabel, title, file_suffix):
        panel_w, panel_h = plot_style["grid_panel_size"]
        fig, axes = plt.subplots(
            nrows=len(sigmas), ncols=len(signatures),
            figsize=(panel_w * len(signatures), panel_h * len(sigmas)),
            sharex=True, sharey=True, squeeze=False, dpi=plot_style["figure_dpi"]
        )
        for row_i, sigma_ms in enumerate(sigmas):
            for col_i, signature in enumerate(signatures):
                ax = axes[row_i, col_i]
                ax_env = ax.twinx()
                ax_env.set_zorder(1)
                _hide_env_axis_spines(ax_env)
                ax_env.set_ylim(*kernel_ylim)
                combo = next((c for c in combos if c[0] == sigma_ms and _signature_key(c) == signature), None)
                ax.axvline(
                    0.0,
                    color=plot_style["marker_center_color"],
                    lw=plot_style["marker_center_lw"],
                    ls=plot_style["marker_center_ls"],
                    alpha=plot_style["marker_center_alpha"],
                    zorder=30,
                )
                if combo is not None:
                    start_rel = -float(groups[combo][0]["params"].get("input_burst_start_marker_n_sigma", 2.0)) * sigma_ms / 1000.0
                    ax.axvline(
                        start_rel,
                        color=plot_style["marker_start_color"],
                        lw=plot_style["marker_start_lw"],
                        ls=plot_style["marker_start_ls"],
                        alpha=plot_style["marker_start_alpha"],
                        zorder=30,
                    )
                    stack, mean_trace = data_map[combo]
                    trace_lw = plot_style["grid_force_trace_lw"] if "dForce" not in ylabel else plot_style["grid_deriv_trace_lw"]
                    trace_alpha = plot_style["grid_force_trace_alpha"] if "dForce" not in ylabel else plot_style["grid_deriv_trace_alpha"]
                    mean_lw = plot_style["grid_force_mean_lw"] if "dForce" not in ylabel else plot_style["grid_deriv_mean_lw"]
                    mean_alpha = plot_style["grid_force_mean_alpha"] if "dForce" not in ylabel else plot_style["grid_deriv_mean_alpha"]
                    for trace in stack:
                        ax.plot(x_common, trace, color=plot_style["force_color"], lw=trace_lw, alpha=trace_alpha, zorder=2)
                    ax.plot(x_common, mean_trace, color=plot_style["force_color"], lw=mean_lw, alpha=mean_alpha, zorder=4)

                    for component_name, (_, mean_kernel) in burst_kernels_by_combo.get(combo, {}).items():
                        kernel_mask = np.isfinite(mean_kernel)
                        if not np.any(kernel_mask):
                            continue
                        kernel_x = x_common[kernel_mask]
                        kernel_y = mean_kernel[kernel_mask]
                        component_color = _resolve_component_color(component_name, plot_style, component_color_map)
                        ax_env.fill_between(
                            kernel_x,
                            0.0,
                            kernel_y,
                            color=component_color,
                            alpha=plot_style["envelope_fill_alpha"],
                            zorder=1,
                        )
                        ax_env.plot(
                            kernel_x,
                            kernel_y,
                            color=component_color,
                            lw=plot_style["envelope_edge_lw"],
                            alpha=plot_style["envelope_edge_alpha"],
                            zorder=2,
                        )

                ax.set_zorder(2)
                ax.patch.set_alpha(0.0)
                ax.set_ylim(*(force_ylim if "dForce" not in ylabel else dforce_ylim))
                ax.grid(alpha=0.25)
                ax.set_title(
                    f"sigma={sigma_ms:g} ms | {_signature_label(signature)}",
                    fontsize=plot_style["grid_title_fontsize"],
                )
                if col_i == len(signatures) - 1:
                    ax_env.set_ylabel("Burst kernel (delta multiplier)")
                else:
                    ax_env.set_yticklabels([])
                    ax_env.tick_params(axis="y", length=0)
        for ax in axes[-1, :]:
            ax.set_xlabel("Time relative to burst center (s)")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        fig.suptitle(title, y=plot_style["grid_suptitle_y"])
        fig.tight_layout()
        out_path = out_dir / file_suffix
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    output_paths = []
    output_paths.append(
        _make_grid(
            force_by_combo,
            "Force (%MVC)",
            "Zoomed force grid by burst signature",
            "burst_force_grid__component_bursts.png",
        )
    )
    output_paths.append(
        _make_grid(
            dforce_by_combo,
            "dForce/dt (%MVC/s)",
            "Zoomed force derivative grid by burst signature",
            "burst_force_derivative_grid__component_bursts.png",
        )
    )
    return output_paths


def _relative_spike_sync_grid(spike_sync_results):
    dt_values = []
    start_candidates = []
    end_candidates = []
    for result in spike_sync_results:
        sync_time_s = np.asarray(result["sync_time_s"], dtype=float)
        if sync_time_s.size < 1:
            continue
        if sync_time_s.size > 1:
            dt_values.append(float(np.median(np.diff(sync_time_s))))
        start_candidates.append(float(sync_time_s[0] - result["burst_center_s"]))
        end_candidates.append(float(sync_time_s[-1] - result["burst_center_s"]))
    if not start_candidates:
        raise ValueError("No valid SPIKE-synchronization results were available to build the plotting grid.")
    dt = min(dt_values) if dt_values else 1e-3
    return np.arange(min(start_candidates), max(end_candidates) + 0.5 * dt, dt)


def _interp_spike_sync_to_relative_grid(result, x_common):
    x_rel = np.asarray(result["sync_time_s"], dtype=float) - float(result["burst_center_s"])
    y = np.asarray(result["sync_profile"], dtype=float)
    return np.interp(x_common, x_rel, y, left=np.nan, right=np.nan)


def plot_burst_spike_sync_grids(spike_sync_groups, out_dir, plot_style=None, component_color_map=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_style = _resolve_plot_style(plot_style)
    combos = sorted(spike_sync_groups.keys(), key=lambda c: (c[0], c[1], c[2], c[3]))
    sigmas = sorted({combo[0] for combo in combos})
    signatures = sorted({_signature_key(combo) for combo in combos})
    all_results = [result for results in spike_sync_groups.values() for result in results if result["status"] == "ok"]
    if not all_results:
        return []
    x_common = _relative_spike_sync_grid(all_results)

    sync_by_combo = {}
    burst_kernels_by_combo = {}
    kernel_vals = []
    for combo, results in spike_sync_groups.items():
        valid_results = [result for result in results if result["status"] == "ok"]
        if not valid_results:
            continue
        sync_stack = np.stack([_interp_spike_sync_to_relative_grid(result, x_common) for result in valid_results], axis=0)
        sync_by_combo[combo] = (sync_stack, _nanmean_no_warning(sync_stack, axis=0))
        component_kernels = {}
        for component_name in ("lf", "alpha", "beta"):
            eligible = [result for result in valid_results if component_name in result["burst_kernel_by_component"]]
            if eligible:
                kernel_stack = np.stack(
                    [
                        np.interp(
                            x_common,
                            np.asarray(result["time_s"], dtype=float) - float(result["burst_center_s"]),
                            np.asarray(result["burst_kernel_by_component"][component_name], dtype=float),
                            left=np.nan,
                            right=np.nan,
                        )
                        for result in eligible
                    ],
                    axis=0,
                )
                component_kernels[component_name] = (kernel_stack, _nanmean_no_warning(kernel_stack, axis=0))
                kernel_vals.append(kernel_stack)
        burst_kernels_by_combo[combo] = component_kernels

    kernel_limit = max(float(np.nanmax(np.abs(np.concatenate(kernel_vals, axis=0)))), 0.25) * 1.08 if kernel_vals else 1.0
    panel_w, panel_h = plot_style["grid_panel_size"]
    fig, axes = plt.subplots(
        nrows=len(sigmas), ncols=len(signatures),
        figsize=(panel_w * len(signatures), panel_h * len(sigmas)),
        sharex=True, sharey=True, squeeze=False, dpi=plot_style["figure_dpi"]
    )

    for row_i, sigma_ms in enumerate(sigmas):
        for col_i, signature in enumerate(signatures):
            ax = axes[row_i, col_i]
            ax_env = ax.twinx()
            ax_env.set_zorder(1)
            _hide_env_axis_spines(ax_env)
            ax_env.set_ylim(-kernel_limit, kernel_limit)
            combo = next((c for c in combos if c[0] == sigma_ms and _signature_key(c) == signature), None)
            ax.axvline(
                0.0,
                color=plot_style["marker_center_color"],
                lw=plot_style["marker_center_lw"],
                ls=plot_style["marker_center_ls"],
                alpha=plot_style["marker_center_alpha"],
                zorder=30,
            )
            if combo is not None and combo in sync_by_combo:
                sample_result = spike_sync_groups[combo][0]
                start_rel = float(sample_result["burst_start_s"] - sample_result["burst_center_s"])
                ax.axvline(
                    start_rel,
                    color=plot_style["marker_start_color"],
                    lw=plot_style["marker_start_lw"],
                    ls=plot_style["marker_start_ls"],
                    alpha=plot_style["marker_start_alpha"],
                    zorder=30,
                )
                for component_name, (_, mean_kernel) in burst_kernels_by_combo.get(combo, {}).items():
                    kernel_mask = np.isfinite(mean_kernel)
                    if not np.any(kernel_mask):
                        continue
                    kernel_x = x_common[kernel_mask]
                    kernel_y = mean_kernel[kernel_mask]
                    component_color = _resolve_component_color(component_name, plot_style, component_color_map)
                    ax_env.fill_between(
                        kernel_x, 0.0, kernel_y,
                        color=component_color,
                        alpha=plot_style["envelope_fill_alpha"],
                        zorder=1,
                    )
                    ax_env.plot(
                        kernel_x, kernel_y,
                        color=component_color,
                        lw=plot_style["envelope_edge_lw"],
                        alpha=plot_style["envelope_edge_alpha"],
                        zorder=2,
                    )

                sync_stack, sync_mean = sync_by_combo[combo]
                for trace in sync_stack:
                    ax.plot(
                        x_common, trace,
                        color=plot_style["sync_color"],
                        lw=plot_style["sync_grid_trace_lw"],
                        alpha=plot_style["sync_grid_trace_alpha"],
                        zorder=3,
                    )
                ax.plot(
                    x_common, sync_mean,
                    color=plot_style["sync_color"],
                    lw=plot_style["sync_grid_mean_lw"],
                    alpha=plot_style["sync_grid_mean_alpha"],
                    zorder=4,
                )

            ax.set_zorder(2)
            ax.patch.set_alpha(0.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.25)
            ax.set_title(
                f"sigma={sigma_ms:g} ms | {_signature_label(signature)}",
                fontsize=plot_style["grid_title_fontsize"],
            )
            if col_i == len(signatures) - 1:
                ax_env.set_ylabel("Burst kernel (delta multiplier)")
            else:
                ax_env.set_yticklabels([])
                ax_env.tick_params(axis="y", length=0)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time relative to burst center (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("SPIKE-synchronization")
    fig.suptitle("SPIKE-synchronization grid by burst signature", y=plot_style["grid_suptitle_y"])
    fig.tight_layout()
    out_path = out_dir / "burst_spike_sync_grid__component_bursts.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def run_burst_spike_sync_summary(root_dir, output_subdir="burst_batch_summary",
                                 spike_sync_config=None, plot_style=None, component_color_map=None,
                                 show_progress=True):
    config = _resolve_spike_sync_config(spike_sync_config)
    if show_progress:
        print("Collecting burst runs and loading spike trains...")
    runs = collect_burst_runs(
        root_dir,
        load_spikes=True,
        show_progress=show_progress,
        progress_desc="Loading spike runs",
    )
    if not runs:
        raise FileNotFoundError("No burst-enabled simulation_output.h5 files were found under the requested root folder.")

    output_dir = Path(root_dir) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    if show_progress:
        print(f"Loaded {len(runs)} burst-enabled run(s).")
        print("Computing SPIKE-synchronization per simulation...")
    runs_iterator = runs
    if show_progress and tqdm is not None:
        runs_iterator = tqdm(
            runs,
            total=len(runs),
            desc="SPIKE-sync",
            unit="sim",
            smoothing=0.1,
        )
    spike_sync_results = []
    for idx, run in enumerate(runs_iterator, start=1):
        spike_sync_results.append(compute_run_spike_synchrony(run, config))
        if show_progress and tqdm is None and (idx % 50 == 0 or idx == len(runs)):
            print(f"SPIKE-sync: analyzed {idx}/{len(runs)} simulation(s)...")
    rows = []
    grouped_results = {}
    if show_progress:
        print("Building SPIKE-synchronization summary table...")
    for result in spike_sync_results:
        combo = result["combo"]
        grouped_results.setdefault(combo, []).append(result)
        rows.append(
            {
                "run_dir": str(result["run_dir"]),
                "status": result["status"],
                "message": result["message"],
                "burst_sigma_ms": combo[0],
                "lf_burst_peak_nA": combo[1],
                "alpha_burst_peak_nA": combo[2],
                "beta_burst_peak_nA": combo[3],
                "burst_signature": _combo_signature_label(combo),
                "analysis_window_start": result["analysis_window_start"],
                "analysis_window_end": result["analysis_window_end"],
                "computation_window_start": result["computation_window_start"],
                "computation_window_end": result["computation_window_end"],
                "n_candidate_units": result["n_candidate_units"],
                "n_active_units": result["n_active_units"],
                "retained_unit_ids": ",".join(result["active_unit_ids"]),
                "excluded_unit_ids": ",".join(result["excluded_unit_ids"]),
                "average_spike_sync": result["average_spike_sync"],
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(
        [
            "burst_sigma_ms",
            "lf_burst_peak_nA",
            "alpha_burst_peak_nA",
            "beta_burst_peak_nA",
            "run_dir",
        ]
    ).reset_index(drop=True)
    summary_csv = output_dir / "burst_spike_sync_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    valid_grouped_results = {
        combo: [result for result in results if result["status"] == "ok"]
        for combo, results in grouped_results.items()
    }
    valid_grouped_results = {combo: results for combo, results in valid_grouped_results.items() if results}
    if show_progress:
        print("Rendering SPIKE-synchronization grid figure(s)...")
    grid_paths = plot_burst_spike_sync_grids(
        valid_grouped_results,
        output_dir,
        plot_style=plot_style,
        component_color_map=component_color_map,
    )
    if show_progress:
        print("SPIKE-synchronization summary complete.")
    return summary_df, grid_paths, valid_grouped_results


def run_burst_batch_summary(root_dir, output_subdir="burst_batch_summary", zoom_duration_s=3.0,
                            plot_style=None, component_color_map=None):
    runs = collect_burst_runs(root_dir)
    if not runs:
        raise FileNotFoundError("No burst-enabled simulation_output.h5 files were found under the requested root folder.")

    output_dir = Path(root_dir) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = group_runs_by_burst_combo(runs)

    rows = []
    combo_paths = []
    for combo, combo_runs in sorted(groups.items(), key=lambda kv: kv[0]):
        combo_paths.append(
            plot_burst_combo_summary(
                combo,
                combo_runs,
                output_dir,
                zoom_duration_s=zoom_duration_s,
                plot_style=plot_style,
                component_color_map=component_color_map,
            )
        )
        rows.append(
            {
                "burst_sigma_ms": combo[0],
                "lf_burst_peak_nA": combo[1],
                "alpha_burst_peak_nA": combo[2],
                "beta_burst_peak_nA": combo[3],
                "burst_signature": _combo_signature_label(combo),
                "n_simulations": len(combo_runs),
                "figure_file": combo_paths[-1].name,
            }
        )

    grid_paths = plot_burst_combo_grids(
        groups,
        output_dir,
        zoom_duration_s=zoom_duration_s,
        plot_style=plot_style,
        component_color_map=component_color_map,
    )
    summary_df = pd.DataFrame(rows).sort_values(
        ["burst_sigma_ms", "lf_burst_peak_nA", "alpha_burst_peak_nA", "beta_burst_peak_nA"]
    ).reset_index(drop=True)
    summary_csv = output_dir / "burst_batch_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    return summary_df, combo_paths, grid_paths
