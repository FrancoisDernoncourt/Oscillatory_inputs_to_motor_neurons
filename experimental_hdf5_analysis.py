import warnings
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde

from synchrony_analysis import run_sliding_sync_index_analysis
from parameter_set_batch_summary import (
    DEFAULT_INPUT_COMPONENT_COLORS,
    _annotate_feature_panel,
    _auto_limits,
    _csv_safe_value,
    _draw_shared_markers,
    _format_panel_title,
    _interp_to_reference,
    _mean_series_or_nan,
    _progress_iter,
    _resolve_feature_config,
    _resolve_feature_export_value,
    _resolve_plot_style,
    fit_baseline_spline_and_extract_feature,
)
from shared_run_analysis import (
    ANALYSIS_PARAM_FEATURE_COLUMNS,
    ANALYSIS_PARAM_FILTER_COLUMNS,
    ANALYSIS_PARAM_SPECTRUM_COLUMNS,
    ANALYSIS_PARAM_SYNC_COLUMNS,
    ANALYSIS_PARAM_WINDOW_COLUMNS,
    DEFAULT_MODULATION_SPECTRUM_CONFIG,
    DEFAULT_OBSERVATION_WINDOW_DEFS_REL_CUE_S,
    EXPORT_TABLE_COLUMNS,
    OBS_BASELINE_EXPORT_COLUMNS,
    OBS_SPECTRUM_EXPORT_COLUMNS,
    OBS_WINDOW_ALIAS_TO_CODE,
    OBS_WINDOW_CODE_TO_ALIAS,
    OBS_WINDOW_EXPORT_COLUMNS,
    OBS_WINDOW_SPECS,
    OBS_WINDOW_STAT_FAMILIES,
    classify_active_units_by_rate,
    compute_modulation_spectra_for_save,
    compute_observation_features_from_cst_sync,
    compute_psd_full_resolution,
    resolve_modulation_spectrum_config as shared_resolve_modulation_spectrum_config,
    smallest_contiguous_psd_mass_interval as shared_smallest_contiguous_psd_mass_interval,
)
from simulator import (
    _copy_value_for_save,
    _resample_uniform_time_and_traces,
    _set_filter_params,
    compute_baseline_normalized_envelope,
    compute_cst_diagnostics,
    compute_firing_rate_diagnostics,
)


ANALYSIS_VERSION = "experimental_sidecar_v2"


DEFAULT_RATE_ISI_DENSITY_CONFIG = {
    "enabled": True,
    "hist_bins": 40,
    "use_kde": True,
    "kde_bw_method": None,
    "density_grid_n": 256,
    "normalize_each_density_to_peak": True,
    "line_lw": 1.8,
    "firing_rate_color": "#E67E22",
    "firing_rate_fill_alpha": 0.18,
    "isi_cv_color": "#2E86C1",
    "isi_cv_fill_alpha": 0.18,
    "ylabel": "Density",
    "bottom_xlabel": "Mean firing rate (Hz)",
    "top_xlabel": "ISI CV",
    "title": "MN firing-rate / ISI-CV distributions",
    "stats_fontsize": 8.5,
}


def _resolve_observation_window_config(trial_data, analysis_config):
    cfg = dict(analysis_config or {})
    full_window_rel_cue_s = (
        float(np.min(trial_data["time_rel_cue_s"])),
        float(np.max(trial_data["time_rel_cue_s"])),
    )
    window_cfg = dict(DEFAULT_OBSERVATION_WINDOW_DEFS_REL_CUE_S)
    window_cfg.update(dict(cfg.get("observation_window_defs_rel_cue_s", {})))
    resolved_windows = {
        alias: _validate_and_clip_rel_window(bounds, full_window_rel_cue_s, name=f"{alias}_rel_cue_s")
        for alias, bounds in window_cfg.items()
    }
    obs_baseline_window_rel_cue_s = _validate_and_clip_rel_window(
        cfg.get("obs_baseline_window_rel_cue_s", (-3.0, -1.0)),
        full_window_rel_cue_s,
        name="obs_baseline_window_rel_cue_s",
    )
    return {
        "window_defs_rel_cue_s": resolved_windows,
        "obs_baseline_window_rel_cue_s": tuple(map(float, obs_baseline_window_rel_cue_s)),
    }


def _window_obs_feature_name(window_code, signal_family, stat_family):
    return f"OBS_WINDOW_{window_code}_{signal_family}_{stat_family}"


def _baseline_obs_feature_name(feature_family):
    return f"OBS_BASELINE_{feature_family}"


def _spectrum_obs_feature_name(feature_family):
    return f"OBS_SPECTRUM_{feature_family}"


def find_latest_experimental_manifest(input_root):
    input_root = Path(input_root)
    manifests = sorted(input_root.glob("experimental_trial_export_manifest_*.csv"))
    if not manifests:
        raise FileNotFoundError(f"No experimental manifest CSV was found under {input_root}")
    return manifests[-1]


def load_experimental_manifest(
    input_root,
    manifest_csv_path=None,
    *,
    exported_only=True,
    participant_ids=None,
    session_tags=None,
    condition_ids=None,
    max_trials=None,
):
    input_root = Path(input_root)
    manifest_csv_path = Path(manifest_csv_path) if manifest_csv_path is not None else find_latest_experimental_manifest(input_root)
    df = pd.read_csv(manifest_csv_path)
    if exported_only:
        included_mask = pd.to_numeric(df.get("included", 0), errors="coerce").fillna(0).astype(int) == 1
        status_mask = df.get("status", pd.Series(index=df.index, dtype=object)).astype(str).str.lower() == "exported"
        df = df[included_mask & status_mask].copy()
    if participant_ids is not None:
        df = df[df["participant_id"].astype(str).isin({str(v) for v in participant_ids})].copy()
    if session_tags is not None:
        df = df[df["session_tag"].astype(str).isin({str(v) for v in session_tags})].copy()
    if condition_ids is not None:
        df = df[df["condition_id"].astype(str).isin({str(v) for v in condition_ids})].copy()
    df = df.reset_index(drop=True)
    if max_trials is not None:
        df = df.iloc[: int(max_trials)].copy()
    df["manifest_csv_path"] = str(manifest_csv_path)
    return df


def _flatten_numeric_dataset(dataset):
    return np.asarray(dataset, dtype=float).reshape(-1)


def _scalar_numeric(group, key):
    return float(np.asarray(group[key], dtype=float).reshape(-1)[0])


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_and_clip_rel_window(window_rel_cue_s, full_window_rel_cue_s, *, name):
    arr = np.asarray(window_rel_cue_s, dtype=float).reshape(-1)
    if arr.size != 2 or not (arr[0] < arr[1]):
        raise ValueError(f"{name} must contain exactly two increasing values")
    full_start_s, full_end_s = map(float, full_window_rel_cue_s)
    clipped = (
        max(full_start_s, float(arr[0])),
        min(full_end_s, float(arr[1])),
    )
    if clipped[0] >= clipped[1]:
        raise ValueError(
            f"{name} does not overlap the available cue-relative trial support "
            f"{tuple(map(float, full_window_rel_cue_s))}"
        )
    return clipped


def _crop_spike_trains_to_abs_window(spike_trains_s, start_s, end_s):
    start_s = float(start_s)
    end_s = float(end_s)
    if not start_s < end_s:
        raise ValueError("Spike-train crop window must be strictly increasing")
    cropped = []
    for spikes_s in spike_trains_s:
        spikes_s = np.asarray(spikes_s, dtype=float)
        cropped.append(spikes_s[(spikes_s >= start_s) & (spikes_s < end_s)].copy())
    return cropped


def _recompute_cst_modulation_normalization(cst_diagnostics, trial_data, params):
    if cst_diagnostics is None or str(cst_diagnostics.get("status", "")).lower() != "ok":
        return cst_diagnostics
    t_s = np.asarray(cst_diagnostics.get("time_s"), dtype=float)
    if t_s.size == 0:
        return cst_diagnostics
    cue_time_abs_s = float(trial_data["cue_time_abs_s"])
    normalization_window_rel_cue_s = tuple(map(float, params.cst_modulation_normalization_window_rel_cue_s))
    normalization_start_s = cue_time_abs_s + normalization_window_rel_cue_s[0]
    normalization_end_s = cue_time_abs_s + normalization_window_rel_cue_s[1]
    normalization_mask = (t_s >= normalization_start_s) & (t_s < normalization_end_s)
    if not np.any(normalization_mask):
        raise ValueError(
            "CST modulation normalization window does not contain any samples after clipping. "
            f"Requested cue-relative normalization window={normalization_window_rel_cue_s}."
        )

    alpha_mod = compute_baseline_normalized_envelope(
        np.asarray(cst_diagnostics["cst_alpha_envelope"], dtype=float),
        normalization_mask,
        name="experimental_cst_alpha_envelope",
        logger=None,
    )
    beta_mod = compute_baseline_normalized_envelope(
        np.asarray(cst_diagnostics["cst_beta_envelope"], dtype=float),
        normalization_mask,
        name="experimental_cst_beta_envelope",
        logger=None,
    )
    cst_diagnostics = dict(cst_diagnostics)
    cst_diagnostics["normalization_window_start_s"] = float(normalization_start_s)
    cst_diagnostics["normalization_window_end_s"] = float(normalization_end_s)
    cst_diagnostics["cst_alpha_baseline_mean"] = np.array([alpha_mod["baseline_mean"]], dtype=float)
    cst_diagnostics["cst_alpha_baseline_sd"] = np.array([alpha_mod["baseline_sd"]], dtype=float)
    cst_diagnostics["cst_alpha_envelope_z"] = np.asarray(alpha_mod["envelope_z"], dtype=float)
    cst_diagnostics["cst_alpha_envelope_pct"] = np.asarray(alpha_mod["envelope_pct"], dtype=float)
    cst_diagnostics["cst_beta_baseline_mean"] = np.array([beta_mod["baseline_mean"]], dtype=float)
    cst_diagnostics["cst_beta_baseline_sd"] = np.array([beta_mod["baseline_sd"]], dtype=float)
    cst_diagnostics["cst_beta_envelope_z"] = np.asarray(beta_mod["envelope_z"], dtype=float)
    cst_diagnostics["cst_beta_envelope_pct"] = np.asarray(beta_mod["envelope_pct"], dtype=float)
    return cst_diagnostics


def _resolve_trial_h5_path(manifest_row, input_root):
    input_root = Path(input_root)
    raw_path_value = manifest_row.get("exported_h5_file", np.nan)
    candidate_paths = []
    expected_name = (
        f"{manifest_row['participant_id']}_{manifest_row['condition_id']}"
        f"_run_{int(manifest_row['run_num']):03d}_trial_{int(manifest_row['trial_num']):03d}.h5"
    )
    candidate_paths.append(
        input_root
        / str(manifest_row["participant_id"])
        / str(manifest_row["session_tag"])
        / str(manifest_row["condition_id"])
        / expected_name
    )
    if isinstance(raw_path_value, str) and raw_path_value.strip():
        candidate_paths.append(input_root / Path(raw_path_value).name)
        candidate_paths.append(
            input_root
            / str(manifest_row["participant_id"])
            / str(manifest_row["session_tag"])
            / str(manifest_row["condition_id"])
            / Path(raw_path_value).name
        )
        candidate_paths.append(Path(raw_path_value))
    for path in candidate_paths:
        if path.exists():
            return path.resolve()
    matches = list(input_root.rglob(expected_name))
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(
        "Could not resolve the exported experimental HDF5 for manifest row "
        f"{manifest_row['participant_id']} / {manifest_row['session_tag']} / "
        f"{manifest_row['condition_id']} / run {manifest_row['run_num']} / trial {manifest_row['trial_num']}"
    )


def load_experimental_trial(raw_h5_path):
    raw_h5_path = Path(raw_h5_path)
    with h5py.File(raw_h5_path, "r") as f:
        root_attrs = {str(k): _decode_attr(v) for k, v in f.attrs.items()}
        trial_meta = f["trial_metadata"]
        events = f["events"]
        unit_metadata = f["unit_metadata"]

        fsamp_hz = _scalar_numeric(trial_meta, "fsamp_hz")
        n_samples = int(np.asarray(trial_meta["n_samples_in_trial"], dtype=float).reshape(-1)[0])
        time_rel_ready_s = _flatten_numeric_dataset(trial_meta["trial_time_rel_ready_s"])
        cue_time_rel_ready_s = _scalar_numeric(events, "cue_time_rel_ready_s")
        ready_time_rel_ready_s = _scalar_numeric(events, "ready_time_rel_ready_s")
        time_rel_cue_s = time_rel_ready_s - cue_time_rel_ready_s
        dt_s = 1.0 / float(fsamp_hz)
        duration_s = float(n_samples) * dt_s
        trial_start_rel_ready_s = float(time_rel_ready_s[0])
        ready_time_abs_s = float(-trial_start_rel_ready_s + ready_time_rel_ready_s)
        cue_time_abs_s = float(ready_time_abs_s + cue_time_rel_ready_s)

        spike_group = f["spike_trains"]["MN"]
        unit_names = sorted(spike_group.keys(), key=lambda name: int(str(name).split("_")[-1]))
        spike_trains_rel_ready_s = []
        spike_trains_trial_s = []
        for key in unit_names:
            spikes_rel_ready = _flatten_numeric_dataset(spike_group[key])
            spikes_trial = spikes_rel_ready - trial_start_rel_ready_s
            spike_trains_rel_ready_s.append(np.asarray(spikes_rel_ready, dtype=float))
            spike_trains_trial_s.append(np.asarray(spikes_trial, dtype=float))

        task_event_times_s = {
            "simulation_start_s": 0.0,
            "usable_window_start_s": 0.0,
            "baseline_segment_start_s": 0.0,
            "get_ready_cue_s": float(ready_time_abs_s),
            "baseline_segment_end_s": float(ready_time_abs_s),
            "go_nogo_cue_s": float(cue_time_abs_s),
            "usable_window_end_s": float(duration_s),
            "simulation_end_s": float(duration_s),
        }

        return {
            "raw_h5_path": raw_h5_path.resolve(),
            "root_attrs": root_attrs,
            "participant_id": str(root_attrs.get("participant_id", "")),
            "session_tag": str(root_attrs.get("session_tag", "")),
            "condition_id": str(root_attrs.get("condition_id", "")),
            "run_num": int(root_attrs.get("run_num", 0)),
            "trial_num": int(root_attrs.get("trial_num", 0)),
            "fsamp_hz": float(fsamp_hz),
            "n_samples": int(n_samples),
            "duration_s": float(duration_s),
            "dt_s": float(dt_s),
            "time_rel_ready_s": np.asarray(time_rel_ready_s, dtype=float),
            "time_rel_cue_s": np.asarray(time_rel_cue_s, dtype=float),
            "trial_start_rel_ready_s": float(trial_start_rel_ready_s),
            "ready_time_abs_s": float(ready_time_abs_s),
            "cue_time_abs_s": float(cue_time_abs_s),
            "task_event_times_s": task_event_times_s,
            "spike_trains_rel_ready_s": spike_trains_rel_ready_s,
            "spike_trains_trial_s": spike_trains_trial_s,
            "unit_names": list(unit_names),
            "n_units": int(np.asarray(unit_metadata["n_units"], dtype=float).reshape(-1)[0]),
        }


def build_experimental_analysis_params(trial_data, analysis_config):
    cfg = dict(analysis_config or {})
    full_window_rel_cue_s = (
        float(np.min(trial_data["time_rel_cue_s"])),
        float(np.max(trial_data["time_rel_cue_s"])),
    )
    firing_rate_isi_window_rel_cue_s = _validate_and_clip_rel_window(
        cfg.get("firing_rate_isi_window_rel_cue_s", (-4.0, -1.0)),
        full_window_rel_cue_s,
        name="firing_rate_isi_window_rel_cue_s",
    )
    cst_modulation_normalization_window_rel_cue_s = _validate_and_clip_rel_window(
        cfg.get("cst_modulation_normalization_window_rel_cue_s", firing_rate_isi_window_rel_cue_s),
        full_window_rel_cue_s,
        name="cst_modulation_normalization_window_rel_cue_s",
    )
    sync_analysis_window_rel_cue_s = _validate_and_clip_rel_window(
        cfg.get("sync_analysis_window_rel_cue_s", full_window_rel_cue_s),
        full_window_rel_cue_s,
        name="sync_analysis_window_rel_cue_s",
    )
    observation_window_config = _resolve_observation_window_config(trial_data, cfg)

    cue_time_abs_s = float(trial_data["cue_time_abs_s"])
    cst_active_window_start_abs_s = cue_time_abs_s + cst_modulation_normalization_window_rel_cue_s[0]
    cst_active_window_end_abs_s = cue_time_abs_s + cst_modulation_normalization_window_rel_cue_s[1]
    ready_abs_s = float(trial_data["ready_time_abs_s"])

    return SimpleNamespace(
        fsamp=float(trial_data["fsamp_hz"]),
        duration=float(trial_data["duration_s"]),
        duration_with_ignored_window=float(trial_data["duration_s"]),
        edges_ignore_duration=0.0,
        get_ready_cue_time_s=float(ready_abs_s),
        go_nogo_cue_delay_s=float(cue_time_abs_s - ready_abs_s),
        enable_go_nogo_cue_diagnostics=True,
        enable_step_current=False,
        enable_step_sync_index_analysis=False,
        input_component_colors=dict(cfg.get("input_component_colors", DEFAULT_INPUT_COMPONENT_COLORS)),
        cst_analysis_window_start_s=float(cst_active_window_start_abs_s),
        cst_analysis_window_end_s=float(cst_active_window_end_abs_s),
        active_unit_rate_threshold_hz=float(cfg.get("active_unit_rate_threshold_hz", 3.0)),
        cst_modulation_mode=str(cfg.get("cst_modulation_mode", "zscore")),
        sync_trace_display_mode=str(cfg.get("sync_trace_display_mode", "absolute")),
        lf_band_hz=[float(v) for v in cfg.get("lf_band_hz", (0.0, 5.0))],
        alpha_band_hz=[float(v) for v in cfg.get("alpha_band_hz", (8.0, 13.0))],
        beta_band_hz=[float(v) for v in cfg.get("beta_band_hz", (13.0, 30.0))],
        scale_filter_order_to_frequency=bool(cfg.get("scale_filter_order_to_frequency", True)),
        filter_order_scaling_coeff=float(cfg.get("filter_order_scaling_coeff", 1.0)),
        default_freq_filter_order=int(cfg.get("default_freq_filter_order", 6)),
        lowest_freq_filter_order=int(cfg.get("lowest_freq_filter_order", 2)),
        max_freq_filter_order=int(cfg.get("max_freq_filter_order", 10)),
        input_burst_sigma_ms=250.0,
        input_burst_start_marker_n_sigma=2.0,
        input_burst_zoom_start_n_sigma=4.0,
        full_window_rel_cue_s=tuple(map(float, full_window_rel_cue_s)),
        firing_rate_isi_window_rel_cue_s=tuple(map(float, firing_rate_isi_window_rel_cue_s)),
        cst_modulation_normalization_window_rel_cue_s=tuple(map(float, cst_modulation_normalization_window_rel_cue_s)),
        sync_analysis_window_rel_cue_s=tuple(map(float, sync_analysis_window_rel_cue_s)),
        observation_window_defs_rel_cue_s=dict(observation_window_config["window_defs_rel_cue_s"]),
        obs_baseline_window_rel_cue_s=tuple(map(float, observation_window_config["obs_baseline_window_rel_cue_s"])),
        analysis_window_rel_cue_s=tuple(map(float, cst_modulation_normalization_window_rel_cue_s)),
        task_event_times_s=dict(trial_data["task_event_times_s"]),
    )


def _copy_group_if_present(src_file, dst_file, group_name):
    if group_name in src_file and group_name not in dst_file:
        src_file.copy(src_file[group_name], dst_file, name=group_name)


def _compute_modulation_spectra_for_save(cst_diagnostics, *, spectrum_window_rel_cue_s, cue_time_abs_s):
    return compute_modulation_spectra_for_save(
        cst_diagnostics,
        spectrum_window_rel_cue_s=spectrum_window_rel_cue_s,
        cue_time_abs_s=cue_time_abs_s,
    )


def _prepare_cst_diagnostics_for_save(cst_diagnostics, cue_time_abs_s, params, *, modulation_spectrum_window_rel_cue_s, minimal_output=False, target_fsamp_hz=None):
    if cst_diagnostics is None:
        return None, None
    cst_to_save = {k: _copy_value_for_save(v) for k, v in cst_diagnostics.items()}
    cst_to_save.update(
        _compute_modulation_spectra_for_save(
            cst_diagnostics,
            spectrum_window_rel_cue_s=modulation_spectrum_window_rel_cue_s,
            cue_time_abs_s=cue_time_abs_s,
        )
    )
    if "time_s" in cst_to_save:
        cst_to_save["time_rel_cue_s"] = np.asarray(cst_to_save["time_s"], dtype=float) - float(cue_time_abs_s)
    cst_to_save["analysis_window_rel_cue_s"] = np.asarray(
        [
            float(cst_diagnostics.get("analysis_window_start_s", np.nan) - cue_time_abs_s),
            float(cst_diagnostics.get("analysis_window_end_s", np.nan) - cue_time_abs_s),
        ],
        dtype=float,
    )
    cst_to_save["normalization_window_rel_cue_s"] = np.asarray(
        [
            float(cst_diagnostics.get("normalization_window_start_s", np.nan) - cue_time_abs_s),
            float(cst_diagnostics.get("normalization_window_end_s", np.nan) - cue_time_abs_s),
        ],
        dtype=float,
    )
    cst_to_save.pop("time_s", None)
    cst_to_save.pop("analysis_mask", None)
    cst_to_save.pop("zoom_mask", None)

    saved_fsamp_hz = float(params.fsamp)
    if minimal_output and target_fsamp_hz is not None and cst_diagnostics.get("status") == "ok":
        time_rel_cue_s = np.asarray(cst_to_save.pop("time_rel_cue_s"), dtype=float)
        trace_keys = {}
        meta_keys = {}
        for key, value in cst_to_save.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[-1] == time_rel_cue_s.size:
                trace_keys[key] = np.asarray(value, dtype=float)
            else:
                meta_keys[key] = _copy_value_for_save(value)
        time_new, trace_keys_new = _resample_uniform_time_and_traces(
            time_rel_cue_s,
            trace_keys,
            target_fsamp_hz=float(target_fsamp_hz),
        )
        cst_to_save = {"time_rel_cue_s": time_new, **trace_keys_new, **meta_keys}
        if time_new.size > 1:
            saved_fsamp_hz = 1.0 / float(np.median(np.diff(time_new)))
    return cst_to_save, float(saved_fsamp_hz)


def _prepare_sync_diagnostics_for_save(sync_diagnostics, *, minimal_output=False, target_fsamp_hz=None):
    if sync_diagnostics is None:
        return None, None
    sync_to_save = {k: _copy_value_for_save(v) for k, v in sync_diagnostics.items()}
    saved_fsamp_hz = None
    if minimal_output and target_fsamp_hz is not None and sync_to_save.get("t_sync") is not None:
        time_rel_cue_s = np.asarray(sync_to_save.pop("t_sync"), dtype=float)
        trace_keys = {}
        meta_keys = {}
        for key, value in sync_to_save.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[-1] == time_rel_cue_s.size:
                trace_keys[key] = np.asarray(value, dtype=float)
            else:
                meta_keys[key] = _copy_value_for_save(value)
        time_new, trace_keys_new = _resample_uniform_time_and_traces(
            time_rel_cue_s,
            trace_keys,
            target_fsamp_hz=float(target_fsamp_hz),
        )
        sync_to_save = {"t_sync": time_new, **trace_keys_new, **meta_keys}
        if time_new.size > 1:
            saved_fsamp_hz = 1.0 / float(np.median(np.diff(time_new)))
    elif sync_to_save.get("t_sync") is not None:
        time_rel_cue_s = np.asarray(sync_to_save["t_sync"], dtype=float)
        if time_rel_cue_s.size > 1:
            saved_fsamp_hz = 1.0 / float(np.median(np.diff(time_rel_cue_s)))
    return sync_to_save, saved_fsamp_hz


def _compute_observation_features_for_save(
    cst_diagnostics,
    sync_diagnostics,
    *,
    cue_time_abs_s,
    params,
    modulation_spectrum_config,
):
    return compute_observation_features_from_cst_sync(
        cst_diagnostics,
        sync_diagnostics,
        cue_time_abs_s=cue_time_abs_s,
        observation_window_defs_rel_cue_s=params.observation_window_defs_rel_cue_s,
        obs_baseline_window_rel_cue_s=params.obs_baseline_window_rel_cue_s,
        modulation_spectrum_config=modulation_spectrum_config,
    )


def _validate_minimal_output_target_fsamp(analysis_config):
    if not bool(analysis_config.get("minimal_output", False)):
        return
    if not bool(analysis_config.get("minimal_output_downsample_saved_traces", True)):
        return
    target_fsamp_hz = float(analysis_config.get("minimal_output_target_fsamp", 100.0))
    highest_band_hz = max(
        float(np.asarray(analysis_config.get("lf_band_hz", [0.0, 5.0]), dtype=float).reshape(-1)[1]),
        float(np.asarray(analysis_config.get("alpha_band_hz", [8.0, 13.0]), dtype=float).reshape(-1)[1]),
        float(np.asarray(analysis_config.get("beta_band_hz", [13.0, 30.0]), dtype=float).reshape(-1)[1]),
    )
    if 0.5 * target_fsamp_hz <= highest_band_hz:
        warnings.warn(
            "minimal_output_target_fsamp may be too low for the saved band-limited experimental traces: "
            f"Nyquist={0.5 * target_fsamp_hz:.1f} Hz, highest band upper bound={highest_band_hz:.1f} Hz."
        )


def _resolve_sync_center_support_window(analysis_window_rel_cue_s, sync_win_ms):
    analysis_start_rel_cue_s, analysis_end_rel_cue_s = map(float, analysis_window_rel_cue_s)
    half_win_s = 0.5 * float(sync_win_ms) / 1000.0
    return (
        analysis_start_rel_cue_s + half_win_s,
        analysis_end_rel_cue_s - half_win_s,
    )


def _sync_window_has_samples(target_window_rel_cue_s, available_center_window_rel_cue_s):
    target_start, target_end = map(float, target_window_rel_cue_s)
    avail_start, avail_end = map(float, available_center_window_rel_cue_s)
    return max(target_start, avail_start) <= min(target_end, avail_end)


def build_experimental_sidecar_path(raw_h5_path, input_root, analyzed_root):
    raw_h5_path = Path(raw_h5_path).resolve()
    input_root = Path(input_root).resolve()
    analyzed_root = Path(analyzed_root)
    try:
        relative = raw_h5_path.relative_to(input_root)
    except ValueError:
        relative = Path(raw_h5_path.parts[-4]) / raw_h5_path.parts[-3] / raw_h5_path.parts[-2] / raw_h5_path.parts[-1]
    return analyzed_root / relative


def save_experimental_analysis_sidecar(
    sidecar_path,
    trial_data,
    analysis_config,
    *,
    raw_input_root,
    manifest_csv_path,
    cst_diagnostics,
    sync_diagnostics,
    firing_rate_diagnostics,
):
    sidecar_path = Path(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    raw_h5_path = Path(trial_data["raw_h5_path"])
    minimal_output = bool(analysis_config.get("minimal_output", False))
    save_spike_trains = bool(analysis_config.get("minimal_output_save_spike_trains", False))
    target_fsamp_hz = (
        float(analysis_config.get("minimal_output_target_fsamp", 100.0))
        if minimal_output and bool(analysis_config.get("minimal_output_downsample_saved_traces", True))
        else None
    )
    params = build_experimental_analysis_params(trial_data, analysis_config)
    modulation_spectrum_config = _resolve_modulation_spectrum_config(
        {
            "window_rel_cue_s": analysis_config.get(
                "modulation_spectrum_window_rel_cue_s",
                params.obs_baseline_window_rel_cue_s,
            ),
            "max_freq_hz": analysis_config.get("modulation_spectrum_max_freq_hz", 12.0),
            "interval_mass_pct": analysis_config.get("modulation_spectrum_interval_mass_pct", 80.0),
        }
    )
    modulation_spectrum_config["window_rel_cue_s"] = _validate_and_clip_rel_window(
        modulation_spectrum_config["window_rel_cue_s"],
        params.full_window_rel_cue_s,
        name="modulation_spectrum_window_rel_cue_s",
    )
    cst_to_save, cst_saved_fsamp_hz = _prepare_cst_diagnostics_for_save(
        cst_diagnostics,
        cue_time_abs_s=trial_data["cue_time_abs_s"],
        params=params,
        modulation_spectrum_window_rel_cue_s=modulation_spectrum_config["window_rel_cue_s"],
        minimal_output=minimal_output,
        target_fsamp_hz=target_fsamp_hz,
    )
    sync_to_save, sync_saved_fsamp_hz = _prepare_sync_diagnostics_for_save(
        sync_diagnostics,
        minimal_output=minimal_output,
        target_fsamp_hz=target_fsamp_hz,
    )
    observation_features = _compute_observation_features_for_save(
        cst_diagnostics,
        sync_diagnostics,
        cue_time_abs_s=trial_data["cue_time_abs_s"],
        params=params,
        modulation_spectrum_config=modulation_spectrum_config,
    )

    with h5py.File(raw_h5_path, "r") as src, h5py.File(sidecar_path, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["raw_trial_h5_path"] = str(raw_h5_path)
        dst.attrs["analysis_version"] = ANALYSIS_VERSION
        dst.attrs["manifest_csv_path"] = str(manifest_csv_path)

        for group_name in ("trial_metadata", "events", "unit_metadata", "motoneurons_and_pools_indices"):
            _copy_group_if_present(src, dst, group_name)
        if "trial_metadata" in dst and "trial_time_rel_cue_s" not in dst["trial_metadata"]:
            dst["trial_metadata"].create_dataset("trial_time_rel_cue_s", data=np.asarray(trial_data["time_rel_cue_s"], dtype=float))
        if save_spike_trains and "spike_trains" in src:
            _copy_group_if_present(src, dst, "spike_trains")

        analysis_grp = dst.create_group("analysis")
        summary_grp = analysis_grp.create_group("summary_metadata")
        summary_grp.attrs["participant_id"] = str(trial_data["participant_id"])
        summary_grp.attrs["session_tag"] = str(trial_data["session_tag"])
        summary_grp.attrs["condition_id"] = str(trial_data["condition_id"])
        summary_grp.attrs["run_num"] = int(trial_data["run_num"])
        summary_grp.attrs["trial_num"] = int(trial_data["trial_num"])
        summary_grp.attrs["minimal_output"] = minimal_output
        summary_grp.attrs["minimal_output_turn_off_figure_output"] = bool(
            analysis_config.get("minimal_output_turn_off_figure_output", True)
        )
        summary_grp.attrs["minimal_output_save_spike_trains"] = save_spike_trains
        summary_grp.attrs["cst_modulation_mode"] = str(analysis_config.get("cst_modulation_mode", "zscore"))
        summary_grp.attrs["sync_trace_display_mode"] = str(analysis_config.get("sync_trace_display_mode", "absolute"))
        summary_grp.attrs["sync_direction_mode"] = str(analysis_config.get("sync_direction_mode", "legacy_forward"))
        summary_grp.attrs["sync_expectation_mode"] = str(analysis_config.get("sync_expectation_mode", "analytic"))
        summary_grp.attrs["raw_input_root"] = str(Path(raw_input_root).resolve())
        summary_grp.create_dataset("full_window_rel_cue_s", data=np.asarray(params.full_window_rel_cue_s, dtype=float))
        summary_grp.create_dataset("firing_rate_isi_window_rel_cue_s", data=np.asarray(params.firing_rate_isi_window_rel_cue_s, dtype=float))
        summary_grp.create_dataset("cst_modulation_normalization_window_rel_cue_s", data=np.asarray(params.cst_modulation_normalization_window_rel_cue_s, dtype=float))
        summary_grp.create_dataset("sync_analysis_window_rel_cue_s", data=np.asarray(params.sync_analysis_window_rel_cue_s, dtype=float))
        summary_grp.create_dataset("obs_baseline_window_rel_cue_s", data=np.asarray(params.obs_baseline_window_rel_cue_s, dtype=float))
        summary_grp.create_dataset("modulation_spectrum_window_rel_cue_s", data=np.asarray(modulation_spectrum_config["window_rel_cue_s"], dtype=float))
        summary_grp.create_dataset("modulation_spectrum_max_freq_hz", data=np.asarray([float(modulation_spectrum_config["max_freq_hz"])], dtype=float))
        summary_grp.create_dataset("modulation_spectrum_interval_mass_pct", data=np.asarray([float(modulation_spectrum_config["interval_mass_pct"])], dtype=float))
        summary_grp.create_dataset("analysis_window_rel_cue_s", data=np.asarray(params.analysis_window_rel_cue_s, dtype=float))
        summary_grp.create_dataset("active_unit_rate_threshold_hz", data=np.asarray([float(analysis_config.get("active_unit_rate_threshold_hz", 3.0))], dtype=float))
        if cst_diagnostics is not None:
            for key in ("n_units_total", "n_units_kept", "unit_fraction_kept", "active_unit_ids", "excluded_unit_ids"):
                if key in cst_diagnostics and cst_diagnostics[key] is not None:
                    summary_grp.create_dataset(key, data=np.asarray(cst_diagnostics[key]))
        summary_grp.create_dataset("sync_win_ms", data=np.asarray([float(analysis_config.get("sync_win_ms", 80.0))], dtype=float))
        summary_grp.create_dataset("sync_step_ms", data=np.asarray([float(analysis_config.get("sync_step_ms", 10.0))], dtype=float))
        summary_grp.create_dataset("sync_coinc_lag_ms", data=np.asarray([float(analysis_config.get("sync_coinc_lag_ms", 6.0))], dtype=float))
        for alias, window_rel_cue_s in params.observation_window_defs_rel_cue_s.items():
            summary_grp.create_dataset(f"{alias}_rel_cue_s", data=np.asarray(window_rel_cue_s, dtype=float))
        for key in ("lf_band_hz", "alpha_band_hz", "beta_band_hz", "sync_baseline_win_rel_cue_s", "sync_post_win_rel_cue_s"):
            if key in analysis_config:
                summary_grp.create_dataset(key, data=np.asarray(analysis_config[key], dtype=float))

        if cst_to_save is not None:
            cst_grp = analysis_grp.create_group("cst_diagnostics")
            if cst_saved_fsamp_hz is not None:
                cst_grp.attrs["saved_fsamp_Hz"] = float(cst_saved_fsamp_hz)
            for key, value in cst_to_save.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    cst_grp.attrs[key] = value
                else:
                    cst_grp.create_dataset(key, data=np.asarray(value))

        if sync_to_save is not None:
            sync_grp = analysis_grp.create_group("sync_diagnostics")
            if sync_saved_fsamp_hz is not None:
                sync_grp.attrs["saved_fsamp_Hz"] = float(sync_saved_fsamp_hz)
            for key, value in sync_to_save.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    sync_grp.attrs[key] = value
                else:
                    sync_grp.create_dataset(key, data=np.asarray(value))

        if firing_rate_diagnostics is not None:
            fr_grp = analysis_grp.create_group("firing_rate_diagnostics")
            for key, value in firing_rate_diagnostics.items():
                if value is None:
                    continue
                fr_grp.create_dataset(key, data=np.asarray(value, dtype=float))

        if observation_features is not None:
            obs_grp = analysis_grp.create_group("observation_features")
            for key, value in observation_features.items():
                obs_grp.create_dataset(key, data=np.asarray([float(value)], dtype=float))

    return sidecar_path


def analyze_experimental_manifest(
    input_root,
    analyzed_root,
    *,
    manifest_csv_path=None,
    analysis_config=None,
    exported_only=True,
    participant_ids=None,
    session_tags=None,
    condition_ids=None,
    max_trials=None,
    overwrite=False,
    show_progress=True,
):
    analysis_config = dict(analysis_config or {})
    _validate_minimal_output_target_fsamp(analysis_config)
    manifest_df = load_experimental_manifest(
        input_root,
        manifest_csv_path=manifest_csv_path,
        exported_only=exported_only,
        participant_ids=participant_ids,
        session_tags=session_tags,
        condition_ids=condition_ids,
        max_trials=max_trials,
    )
    rows = []
    for row in _progress_iter(list(manifest_df.to_dict(orient="records")), desc="Analyzing experimental trials", show_progress=show_progress):
        raw_h5_path = _resolve_trial_h5_path(row, input_root)
        sidecar_path = build_experimental_sidecar_path(raw_h5_path, input_root, analyzed_root)
        if sidecar_path.exists() and not overwrite:
            rows.append(
                {
                    "participant_id": str(row["participant_id"]),
                    "session_tag": str(row["session_tag"]),
                    "condition_id": str(row["condition_id"]),
                    "run_num": int(row["run_num"]),
                    "trial_num": int(row["trial_num"]),
                    "raw_h5_path": str(raw_h5_path),
                    "sidecar_h5_path": str(sidecar_path),
                    "status": "skipped_existing",
                }
            )
            continue

        trial_data = load_experimental_trial(raw_h5_path)
        params = build_experimental_analysis_params(trial_data, analysis_config)
        _set_filter_params(params)
        active_unit_selection = classify_active_units_by_rate(
            trial_data["spike_trains_trial_s"],
            analysis_start_s=float(params.cst_analysis_window_start_s),
            analysis_end_s=float(params.cst_analysis_window_end_s),
            rate_threshold_hz=float(params.active_unit_rate_threshold_hz),
        )
        cst_diagnostics = compute_cst_diagnostics(
            spike_trains_MN=trial_data["spike_trains_trial_s"],
            params=params,
            burst_center_s=None,
            active_unit_selection=active_unit_selection,
        )
        cst_diagnostics = _recompute_cst_modulation_normalization(cst_diagnostics, trial_data, params)

        firing_rate_window_start_s = float(trial_data["cue_time_abs_s"] + params.firing_rate_isi_window_rel_cue_s[0])
        firing_rate_window_end_s = float(trial_data["cue_time_abs_s"] + params.firing_rate_isi_window_rel_cue_s[1])
        firing_rate_spike_trains = _crop_spike_trains_to_abs_window(
            trial_data["spike_trains_trial_s"],
            firing_rate_window_start_s,
            firing_rate_window_end_s,
        )
        firing_rate_diagnostics = compute_firing_rate_diagnostics(
            firing_rate_spike_trains,
            active_unit_ids_for_isi_cv=active_unit_selection["active_unit_ids"],
        )
        sync_diagnostics = None
        if bool(analysis_config.get("enable_sync_index_analysis", True)):
            sync_baseline_win_rel_cue_s = tuple(map(float, analysis_config.get("sync_baseline_win_rel_cue_s", (-3.0, -2.0))))
            sync_post_win_rel_cue_s = tuple(map(float, analysis_config.get("sync_post_win_rel_cue_s", (0.3, 0.6))))
            sync_center_support_rel_cue_s = _resolve_sync_center_support_window(
                params.sync_analysis_window_rel_cue_s,
                analysis_config.get("sync_win_ms", 80.0),
            )
            if not _sync_window_has_samples(sync_baseline_win_rel_cue_s, sync_center_support_rel_cue_s):
                warnings.warn(
                    "Skipping sync analysis for this trial because sync_baseline_win_rel_cue_s does not overlap "
                    "the available synchrony-window centers. "
                    f"Requested baseline window={sync_baseline_win_rel_cue_s}, "
                    f"available center range={sync_center_support_rel_cue_s}."
                )
            elif not _sync_window_has_samples(sync_post_win_rel_cue_s, sync_center_support_rel_cue_s):
                warnings.warn(
                    "Skipping sync analysis for this trial because sync_post_win_rel_cue_s does not overlap "
                    "the available synchrony-window centers. "
                    f"Requested post window={sync_post_win_rel_cue_s}, "
                    f"available center range={sync_center_support_rel_cue_s}. "
                    "This error is about the synchrony summary timing, not about silent motor units."
                )
            else:
                try:
                    sync_active_ids = np.asarray(active_unit_selection["active_unit_ids"], dtype=int)
                    sync_spike_trains = [trial_data["spike_trains_trial_s"][int(unit_i)] for unit_i in sync_active_ids]
                    if len(sync_spike_trains) < 2:
                        raise ValueError("At least two active units are required for synchrony analysis")
                    sync_diagnostics = run_sliding_sync_index_analysis(
                        sync_spike_trains,
                        cue_time_sec=float(trial_data["cue_time_abs_s"]),
                        baseline_win=sync_baseline_win_rel_cue_s,
                        post_win=sync_post_win_rel_cue_s,
                        generate_figure=False,
                        display_mode=str(analysis_config.get("sync_trace_display_mode", "absolute")),
                        bin_ms=float(analysis_config.get("sync_bin_ms", 1.0)),
                        sync_win_ms=float(analysis_config.get("sync_win_ms", 80.0)),
                        sync_step_ms=float(analysis_config.get("sync_step_ms", 10.0)),
                        coinc_lag_ms=float(analysis_config.get("sync_coinc_lag_ms", 6.0)),
                        direction_mode=str(analysis_config.get("sync_direction_mode", "legacy_forward")),
                        expectation_mode=str(analysis_config.get("sync_expectation_mode", "analytic")),
                        n_surrogates=int(analysis_config.get("sync_n_surrogates", 100)),
                        surrogate_min_shift_ms=float(analysis_config.get("sync_surrogate_min_shift_ms", 1.0)),
                        duration_s=float(trial_data["duration_s"]),
                        analysis_start_s=float(trial_data["cue_time_abs_s"] + params.sync_analysis_window_rel_cue_s[0]),
                        analysis_end_s=float(trial_data["cue_time_abs_s"] + params.sync_analysis_window_rel_cue_s[1]),
                        unit_ids=sync_active_ids,
                    )
                except Exception as exc:
                    warnings.warn(f"Skipping sync analysis for this trial because it failed with: {exc}")

        saved_sidecar = save_experimental_analysis_sidecar(
            sidecar_path=sidecar_path,
            trial_data=trial_data,
            analysis_config=analysis_config,
            raw_input_root=input_root,
            manifest_csv_path=manifest_df["manifest_csv_path"].iloc[0],
            cst_diagnostics=cst_diagnostics,
            sync_diagnostics=sync_diagnostics,
            firing_rate_diagnostics=firing_rate_diagnostics,
        )
        rows.append(
            {
                "participant_id": trial_data["participant_id"],
                "session_tag": trial_data["session_tag"],
                "condition_id": trial_data["condition_id"],
                "run_num": int(trial_data["run_num"]),
                "trial_num": int(trial_data["trial_num"]),
                "raw_h5_path": str(raw_h5_path),
                "sidecar_h5_path": str(saved_sidecar),
                "status": "analyzed",
            }
        )
    return pd.DataFrame(rows)


def _batch_dir_from_key(analyzed_root, batch_key):
    participant_id, session_tag, condition_id = batch_key
    return Path(analyzed_root) / participant_id / session_tag / condition_id


def _load_group_dataset_if_present(group, key):
    if group is None or key not in group:
        return None
    return np.asarray(group[key], dtype=float)


def load_analyzed_experimental_trial(sidecar_path, color_overrides=None):
    sidecar_path = Path(sidecar_path)
    with h5py.File(sidecar_path, "r") as f:
        analysis_grp = f["analysis"]
        summary_grp = analysis_grp["summary_metadata"]
        cst_grp = analysis_grp.get("cst_diagnostics")
        sync_grp = analysis_grp.get("sync_diagnostics")
        fr_grp = analysis_grp.get("firing_rate_diagnostics")
        event_grp = f["events"]

        colors = dict(DEFAULT_INPUT_COMPONENT_COLORS)
        if color_overrides:
            colors.update({str(k): str(v) for k, v in color_overrides.items()})

        batch_key = (
            str(summary_grp.attrs.get("participant_id", "")),
            str(summary_grp.attrs.get("session_tag", "")),
            str(summary_grp.attrs.get("condition_id", "")),
        )
        return {
            "sidecar_path": sidecar_path,
            "batch_key": batch_key,
            "event_times": {
                "get_ready_cue_s": float(np.asarray(event_grp["ready_time_rel_cue_s"], dtype=float).reshape(-1)[0]),
                "go_nogo_cue_s": float(np.asarray(event_grp["cue_time_rel_cue_s"], dtype=float).reshape(-1)[0]),
            },
            "colors": colors,
            "full_window_rel_cue_s": tuple(np.asarray(summary_grp["full_window_rel_cue_s"], dtype=float).reshape(-1).tolist()) if "full_window_rel_cue_s" in summary_grp else tuple(np.asarray(summary_grp["display_window_rel_cue_s"], dtype=float).reshape(-1).tolist()),
            "firing_rate_isi_window_rel_cue_s": tuple(np.asarray(summary_grp["firing_rate_isi_window_rel_cue_s"], dtype=float).reshape(-1).tolist()) if "firing_rate_isi_window_rel_cue_s" in summary_grp else tuple(np.asarray(summary_grp["analysis_window_rel_cue_s"], dtype=float).reshape(-1).tolist()),
            "cst_modulation_normalization_window_rel_cue_s": tuple(np.asarray(summary_grp["cst_modulation_normalization_window_rel_cue_s"], dtype=float).reshape(-1).tolist()) if "cst_modulation_normalization_window_rel_cue_s" in summary_grp else tuple(np.asarray(summary_grp["analysis_window_rel_cue_s"], dtype=float).reshape(-1).tolist()),
            "modulation_spectrum_window_rel_cue_s": tuple(np.asarray(summary_grp["modulation_spectrum_window_rel_cue_s"], dtype=float).reshape(-1).tolist()) if "modulation_spectrum_window_rel_cue_s" in summary_grp else tuple(np.asarray(summary_grp["cst_modulation_normalization_window_rel_cue_s"], dtype=float).reshape(-1).tolist()),
            "sync_analysis_window_rel_cue_s": tuple(np.asarray(summary_grp["sync_analysis_window_rel_cue_s"], dtype=float).reshape(-1).tolist()) if "sync_analysis_window_rel_cue_s" in summary_grp else tuple(np.asarray(summary_grp["analysis_window_rel_cue_s"], dtype=float).reshape(-1).tolist()),
            "analysis_window_rel_cue_s": tuple(np.asarray(summary_grp["analysis_window_rel_cue_s"], dtype=float).reshape(-1).tolist()),
            "cst_modulation_mode": str(summary_grp.attrs.get("cst_modulation_mode", "zscore")),
            "sync_display_mode": str(summary_grp.attrs.get("sync_trace_display_mode", "absolute")),
            "lf_band_hz": tuple(np.asarray(summary_grp["lf_band_hz"], dtype=float).reshape(-1).tolist()) if "lf_band_hz" in summary_grp else (0.0, 5.0),
            "alpha_band_hz": tuple(np.asarray(summary_grp["alpha_band_hz"], dtype=float).reshape(-1).tolist()) if "alpha_band_hz" in summary_grp else (8.0, 13.0),
            "beta_band_hz": tuple(np.asarray(summary_grp["beta_band_hz"], dtype=float).reshape(-1).tolist()) if "beta_band_hz" in summary_grp else (13.0, 30.0),
            "sync_baseline_win_rel_cue_s": tuple(np.asarray(summary_grp["sync_baseline_win_rel_cue_s"], dtype=float).reshape(-1).tolist()) if "sync_baseline_win_rel_cue_s" in summary_grp else None,
            "sync_post_win_rel_cue_s": tuple(np.asarray(summary_grp["sync_post_win_rel_cue_s"], dtype=float).reshape(-1).tolist()) if "sync_post_win_rel_cue_s" in summary_grp else None,
            "t_rel_cue_s": _load_group_dataset_if_present(cst_grp, "time_rel_cue_s"),
            "cst_global": _load_group_dataset_if_present(cst_grp, "cst_normalized"),
            "cst_lf": _load_group_dataset_if_present(cst_grp, "cst_lf"),
            "cst_alpha": _load_group_dataset_if_present(cst_grp, "cst_alpha"),
            "cst_beta": _load_group_dataset_if_present(cst_grp, "cst_beta"),
            "cst_alpha_envelope": _load_group_dataset_if_present(cst_grp, "cst_alpha_envelope"),
            "cst_beta_envelope": _load_group_dataset_if_present(cst_grp, "cst_beta_envelope"),
            "cst_alpha_baseline_mean": _load_group_dataset_if_present(cst_grp, "cst_alpha_baseline_mean"),
            "cst_alpha_baseline_sd": _load_group_dataset_if_present(cst_grp, "cst_alpha_baseline_sd"),
            "cst_beta_baseline_mean": _load_group_dataset_if_present(cst_grp, "cst_beta_baseline_mean"),
            "cst_beta_baseline_sd": _load_group_dataset_if_present(cst_grp, "cst_beta_baseline_sd"),
            "cst_modulation_spectrum_freqs_hz": _load_group_dataset_if_present(cst_grp, "cst_modulation_spectrum_freqs_hz"),
            "cst_alpha_modulation_spectrum_psd": _load_group_dataset_if_present(cst_grp, "cst_alpha_modulation_spectrum_psd"),
            "cst_beta_modulation_spectrum_psd": _load_group_dataset_if_present(cst_grp, "cst_beta_modulation_spectrum_psd"),
            "cst_alpha_mod": _load_group_dataset_if_present(
                cst_grp,
                "cst_alpha_envelope_z" if str(summary_grp.attrs.get("cst_modulation_mode", "zscore")).lower() == "zscore" else "cst_alpha_envelope_pct",
            ),
            "cst_beta_mod": _load_group_dataset_if_present(
                cst_grp,
                "cst_beta_envelope_z" if str(summary_grp.attrs.get("cst_modulation_mode", "zscore")).lower() == "zscore" else "cst_beta_envelope_pct",
            ),
            "sync_time_rel": _load_group_dataset_if_present(sync_grp, "t_sync") if sync_grp is not None else None,
            "sync_trace": _load_group_dataset_if_present(
                sync_grp,
                "sync_trace" if str(summary_grp.attrs.get("sync_trace_display_mode", "absolute")).lower() == "absolute" else "sync_trace_zscore",
            ) if sync_grp is not None else None,
            "normalization_window_rel_cue_s": _load_group_dataset_if_present(cst_grp, "normalization_window_rel_cue_s") if cst_grp is not None else None,
            "firing_rate_diagnostics_present": fr_grp is not None,
        }


def collect_experimental_batch_runs(
    analyzed_root,
    *,
    participant_ids=None,
    session_tags=None,
    condition_ids=None,
    max_batches=None,
    color_overrides=None,
):
    analyzed_root = Path(analyzed_root)
    grouped = {}
    for sidecar_path in sorted(analyzed_root.rglob("*.h5")):
        try:
            run = load_analyzed_experimental_trial(sidecar_path, color_overrides=color_overrides)
        except Exception:
            continue
        participant_id, session_tag, condition_id = run["batch_key"]
        if participant_ids is not None and participant_id not in {str(v) for v in participant_ids}:
            continue
        if session_tags is not None and session_tag not in {str(v) for v in session_tags}:
            continue
        if condition_ids is not None and condition_id not in {str(v) for v in condition_ids}:
            continue
        batch_dir = _batch_dir_from_key(analyzed_root, run["batch_key"])
        grouped.setdefault(batch_dir, []).append(run)
    sorted_items = sorted(grouped.items(), key=lambda item: item[0].as_posix())
    if max_batches is not None:
        sorted_items = sorted_items[: int(max_batches)]
    return dict(sorted_items)


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
    return {key: _auto_limits(np.asarray(values, dtype=float), zero_floor=False) for key, values in limit_map.items()}


def _collect_available_traces(runs, signal_key, time_key, reference_time):
    traces = []
    for run in runs:
        signal = run.get(signal_key)
        time = run.get(time_key)
        if signal is None or time is None:
            continue
        traces.append(_interp_to_reference(time, signal, reference_time))
    if not traces:
        return None
    return np.stack(traces, axis=0)


def _compute_experimental_global_limits(grouped_runs, show_individual_traces=True, show_std_band=True, show_reference_std_lines=True, xlim_rel_cue_s=None):
    limits = {}
    for _, runs in grouped_runs.items():
        ref = next((run for run in runs if run.get("t_rel_cue_s") is not None), None)
        if ref is None:
            continue
        time_rel_cue = np.asarray(ref["t_rel_cue_s"], dtype=float)
        xlim = tuple(map(float, xlim_rel_cue_s)) if xlim_rel_cue_s is not None else ref["full_window_rel_cue_s"]
        zoom_mask = (time_rel_cue >= xlim[0]) & (time_rel_cue <= xlim[1])
        for panel_key, run_key in (
            ("cst_lf", "cst_lf"),
            ("cst_alpha", "cst_alpha"),
            ("cst_beta", "cst_beta"),
            ("mod_alpha", "cst_alpha_mod"),
            ("mod_beta", "cst_beta_mod"),
        ):
            traces_full = _collect_available_traces(runs, run_key, "t_rel_cue_s", time_rel_cue)
            if traces_full is None:
                continue
            traces = traces_full[:, zoom_mask]
            mean_trace = np.nanmean(traces, axis=0)
            if show_individual_traces:
                _append_limits(limits, panel_key, traces)
            else:
                _append_limits(limits, panel_key, mean_trace)
            if show_std_band:
                _append_limits(limits, panel_key, mean_trace - np.nanstd(traces, axis=0))
                _append_limits(limits, panel_key, mean_trace + np.nanstd(traces, axis=0))
            if show_reference_std_lines and panel_key in ("cst_lf", "cst_alpha", "cst_beta"):
                ref_std = float(np.mean([np.nanstd(trace) for trace in traces]))
                if panel_key == "cst_lf":
                    normalization_window = tuple(map(float, ref.get("normalization_window_rel_cue_s", ref.get("cst_modulation_normalization_window_rel_cue_s", (np.nan, np.nan)))))
                    normalization_mask = (time_rel_cue >= normalization_window[0]) & (time_rel_cue < normalization_window[1])
                    ref_center = float(np.nanmean(traces_full[:, normalization_mask])) if np.any(normalization_mask) else float(np.nanmean(mean_trace))
                else:
                    ref_center = 0.0
                _append_limits(limits, panel_key, np.array([ref_center + ref_std, ref_center - ref_std], dtype=float))

        ref_sync = next((run for run in runs if run.get("sync_time_rel") is not None and run.get("sync_trace") is not None), None)
        if ref_sync is not None:
            sync_time = np.asarray(ref_sync["sync_time_rel"], dtype=float)
            sync_mask = (sync_time >= xlim[0]) & (sync_time <= xlim[1])
            traces = _collect_available_traces(runs, "sync_trace", "sync_time_rel", sync_time)
            if traces is not None:
                traces = traces[:, sync_mask]
                mean_trace = np.nanmean(traces, axis=0)
                if show_individual_traces:
                    _append_limits(limits, "sync", traces)
                else:
                    _append_limits(limits, "sync", mean_trace)
                if show_std_band:
                    _append_limits(limits, "sync", mean_trace - np.nanstd(traces, axis=0))
                    _append_limits(limits, "sync", mean_trace + np.nanstd(traces, axis=0))
    return _finalize_limits(limits)


def _resolve_window_overlay_specs(window_overlay_specs=None):
    default_specs = {
        "firing_rate_isi_window_rel_cue_s": {
            "enabled": True,
            "label": "FR / ISI window",
            "color": "#2ca02c",
            "lw": 2.0,
            "alpha": 0.95,
            "y_axes": 0.02,
        },
        "cst_modulation_normalization_window_rel_cue_s": {
            "enabled": True,
            "label": "CST normalization window",
            "color": "#1f77b4",
            "lw": 2.0,
            "alpha": 0.95,
            "y_axes": 0.05,
        },
        "sync_baseline_win_rel_cue_s": {
            "enabled": True,
            "label": "Sync baseline window",
            "color": "#d62728",
            "lw": 2.0,
            "alpha": 0.95,
            "y_axes": 0.08,
        },
        "feature_fitting_window": {
            "enabled": True,
            "label": "Spline fitting window",
            "color": "#111111",
            "lw": 1.8,
            "alpha": 0.90,
            "y_axes": 0.11,
        },
        "feature_analysis_window": {
            "enabled": True,
            "label": "Feature window",
            "color": "#7f7f7f",
            "lw": 2.0,
            "alpha": 0.95,
            "y_axes": 0.14,
        },
    }
    if window_overlay_specs:
        for key, value in window_overlay_specs.items():
            base = dict(default_specs.get(key, {}))
            base.update(value or {})
            default_specs[key] = base
    return default_specs


def _resolve_modulation_spectrum_config(config=None):
    resolved = dict(DEFAULT_MODULATION_SPECTRUM_CONFIG)
    if config:
        resolved.update(config)
    resolved = shared_resolve_modulation_spectrum_config(resolved)
    resolved.setdefault("ylabel", "PSD (a.u.^2/Hz)")
    resolved.setdefault("interval_alpha", 0.18)
    return resolved


def _window_segments_from_run_and_feature(run, feature_config):
    feature_config = _resolve_feature_config(feature_config)
    buffer_s = float(feature_config["buffer_s"])
    fitting_start_s, fitting_end_s = map(float, feature_config["fitting_window"])
    analysis_start_s, analysis_end_s = map(float, feature_config["analysis_window"])
    exclusion_start_s = analysis_start_s - buffer_s
    exclusion_end_s = analysis_end_s + buffer_s
    return {
        "firing_rate_isi_window_rel_cue_s": [tuple(map(float, run["firing_rate_isi_window_rel_cue_s"]))],
        "cst_modulation_normalization_window_rel_cue_s": [tuple(map(float, run["cst_modulation_normalization_window_rel_cue_s"]))],
        "sync_baseline_win_rel_cue_s": [tuple(map(float, run["sync_baseline_win_rel_cue_s"]))] if run.get("sync_baseline_win_rel_cue_s") is not None else [],
        "feature_fitting_window": [
            (fitting_start_s, min(exclusion_start_s, fitting_end_s)),
            (max(exclusion_end_s, fitting_start_s), fitting_end_s),
        ],
        "feature_analysis_window": [(analysis_start_s, analysis_end_s)],
    }


def _draw_window_overlays(ax, display_window, run, feature_config, window_overlay_specs):
    overlay_specs = _resolve_window_overlay_specs(window_overlay_specs)
    segments_by_key = _window_segments_from_run_and_feature(run, feature_config)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    handles = []
    seen_labels = set()
    for key, spec in overlay_specs.items():
        if not bool(spec.get("enabled", True)):
            continue
        segments = segments_by_key.get(key, [])
        label = str(spec.get("label", key))
        for segment in segments:
            seg_start_s, seg_end_s = map(float, segment)
            clipped_start_s = max(float(display_window[0]), seg_start_s)
            clipped_end_s = min(float(display_window[1]), seg_end_s)
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
            if label not in seen_labels:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=str(spec.get("color", "#111111")),
                        lw=float(spec.get("lw", 2.0)),
                        alpha=float(spec.get("alpha", 0.95)),
                        label=label,
                    )
                )
                seen_labels.add(label)
    return handles


def _compute_baseline_modulation_spectrum(run, band_key, spectrum_config):
    saved_freqs = run.get("cst_modulation_spectrum_freqs_hz")
    saved_psd = run.get(f"cst_{band_key}_modulation_spectrum_psd")
    if saved_freqs is not None and saved_psd is not None:
        freqs = np.asarray(saved_freqs, dtype=float)
        psd = np.asarray(saved_psd, dtype=float)
        if freqs.size and psd.size and freqs.shape == psd.shape:
            max_freq_hz = spectrum_config.get("max_freq_hz")
            if max_freq_hz is not None:
                keep_mask = freqs <= float(max_freq_hz)
                freqs = freqs[keep_mask]
                psd = psd[keep_mask]
            return freqs, psd

    time_rel = np.asarray(run.get("t_rel_cue_s"), dtype=float)
    if time_rel.size < 4:
        return None, None
    if band_key == "alpha":
        envelope = run.get("cst_alpha_envelope")
        baseline_mean = run.get("cst_alpha_baseline_mean")
        baseline_sd = run.get("cst_alpha_baseline_sd")
    elif band_key == "beta":
        envelope = run.get("cst_beta_envelope")
        baseline_mean = run.get("cst_beta_baseline_mean")
        baseline_sd = run.get("cst_beta_baseline_sd")
    else:
        raise ValueError(f"Unsupported band_key: {band_key}")

    if envelope is None:
        return None, None
    envelope = np.asarray(envelope, dtype=float)
    if envelope.shape != time_rel.shape:
        return None, None

    spectrum_window = run.get("modulation_spectrum_window_rel_cue_s")
    if spectrum_window is None:
        spectrum_window = run.get("cst_modulation_spectrum_window_rel_cue_s")
    if spectrum_window is None:
        spectrum_window = run.get("cst_modulation_normalization_window_rel_cue_s")
    if spectrum_window is None:
        spectrum_window = run.get("normalization_window_rel_cue_s")
    if spectrum_window is None:
        return None, None
    spectrum_start_s, spectrum_end_s = map(float, spectrum_window)
    spectrum_mask = (time_rel >= spectrum_start_s) & (time_rel < spectrum_end_s)
    if np.count_nonzero(spectrum_mask) < 8:
        return None, None

    baseline_mean = np.nan if baseline_mean is None else float(np.asarray(baseline_mean, dtype=float).reshape(-1)[0])
    baseline_sd = np.nan if baseline_sd is None else float(np.asarray(baseline_sd, dtype=float).reshape(-1)[0])
    if not np.isfinite(baseline_mean) or not np.isfinite(baseline_sd) or baseline_sd <= 1e-9:
        return None, None

    envelope_z = (envelope - baseline_mean) / baseline_sd
    spectrum_signal = envelope_z[spectrum_mask]
    spectrum_time = time_rel[spectrum_mask]
    if spectrum_time.size < 4:
        return None, None
    dt = float(np.median(np.diff(spectrum_time)))
    if not np.isfinite(dt) or dt <= 0:
        return None, None
    fsamp = 1.0 / dt
    freqs, psd = _compute_psd_full_resolution(spectrum_signal, fsamp=fsamp)
    max_freq_hz = spectrum_config.get("max_freq_hz")
    if max_freq_hz is not None:
        keep_mask = np.asarray(freqs, dtype=float) <= float(max_freq_hz)
        freqs = np.asarray(freqs, dtype=float)[keep_mask]
        psd = np.asarray(psd, dtype=float)[keep_mask]
    return np.asarray(freqs, dtype=float), np.asarray(psd, dtype=float)


def _collect_modulation_spectra(runs, band_key, spectrum_config):
    valid = []
    ref_freqs = None
    for run in runs:
        freqs, psd = _compute_baseline_modulation_spectrum(run, band_key, spectrum_config)
        if freqs is None or psd is None or freqs.size == 0:
            continue
        if ref_freqs is None:
            ref_freqs = np.asarray(freqs, dtype=float)
            valid.append(np.asarray(psd, dtype=float))
        else:
            valid.append(np.interp(ref_freqs, freqs, psd, left=np.nan, right=np.nan))
    if ref_freqs is None or not valid:
        return None, None
    return ref_freqs, np.stack(valid, axis=0)


def _smallest_contiguous_psd_mass_interval(freqs, psd, mass_pct):
    return shared_smallest_contiguous_psd_mass_interval(freqs, psd, mass_pct)


def _resolve_rate_isi_density_config(rate_isi_density_config=None):
    resolved = dict(DEFAULT_RATE_ISI_DENSITY_CONFIG)
    if rate_isi_density_config:
        resolved.update(rate_isi_density_config)
    return resolved


def _load_rate_isi_arrays_for_run(run):
    sidecar_path = Path(run["sidecar_path"])
    with h5py.File(sidecar_path, "r") as f:
        fr_values, isi_values = _extract_saved_rate_isi_arrays(f)
        if fr_values is not None and isi_values is not None:
            return np.asarray(fr_values, dtype=float), np.asarray(isi_values, dtype=float)
        raw_trial_h5_path = Path(f.attrs.get("raw_trial_h5_path", ""))
    if raw_trial_h5_path.exists():
        trial_data = load_experimental_trial(raw_trial_h5_path)
        return _compute_fr_and_isi_cv_from_spikes(trial_data["spike_trains_trial_s"])
    return None, None


def _collect_rate_isi_density_data(runs):
    firing_rate_values = []
    isi_cv_values = []
    per_trial_rows = []
    for run in runs:
        fr_values, isi_values = _load_rate_isi_arrays_for_run(run)
        if fr_values is None or isi_values is None:
            continue
        fr_values = np.asarray(fr_values, dtype=float)
        isi_values = np.asarray(isi_values, dtype=float)
        fr_finite = fr_values[np.isfinite(fr_values)]
        isi_finite = isi_values[np.isfinite(isi_values)]
        if fr_finite.size:
            firing_rate_values.append(fr_finite)
        if isi_finite.size:
            isi_cv_values.append(isi_finite)
        fr_stats = _summarize_distribution(fr_finite)
        isi_stats = _summarize_distribution(isi_finite)
        per_trial_rows.append(
            {
                "firing_rate_mean": fr_stats["mean"],
                "firing_rate_sd": fr_stats["sd"],
                "firing_rate_skewness": fr_stats["skewness"],
                "isi_cv_mean": isi_stats["mean"],
                "isi_cv_sd": isi_stats["sd"],
                "isi_cv_skewness": isi_stats["skewness"],
            }
        )
    return {
        "firing_rate_values": np.concatenate(firing_rate_values) if firing_rate_values else np.asarray([], dtype=float),
        "isi_cv_values": np.concatenate(isi_cv_values) if isi_cv_values else np.asarray([], dtype=float),
        "per_trial_stats": pd.DataFrame(per_trial_rows),
    }


def _compute_density_curve(values, config):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    if values.size == 1 or float(np.nanstd(values)) <= 0.0:
        center = float(values[0])
        eps = max(abs(center) * 0.05, 1e-3)
        x = np.array([center - eps, center, center + eps], dtype=float)
        y = np.array([0.0, 1.0, 0.0], dtype=float)
        return x, y
    if bool(config.get("use_kde", True)):
        x = np.linspace(float(np.min(values)), float(np.max(values)), int(config.get("density_grid_n", 256)), dtype=float)
        kde = gaussian_kde(values, bw_method=config.get("kde_bw_method", None))
        y = np.asarray(kde(x), dtype=float)
        return x, y
    hist, bin_edges = np.histogram(values, bins=int(config.get("hist_bins", 40)), density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return np.asarray(centers, dtype=float), np.asarray(hist, dtype=float)


def _plot_rate_isi_density_panel(ax, runs, config):
    if not bool(config.get("enabled", True)):
        ax.axis("off")
        return None
    density_data = _collect_rate_isi_density_data(runs)
    fr_values = density_data["firing_rate_values"]
    isi_values = density_data["isi_cv_values"]
    stats_df = density_data["per_trial_stats"]
    if fr_values.size == 0 and isi_values.size == 0:
        ax.axis("off")
        return None

    fr_x, fr_y = _compute_density_curve(fr_values, config)
    isi_x, isi_y = _compute_density_curve(isi_values, config)
    ax_top = ax.twiny()
    ymax_candidates = []
    if fr_y is not None:
        ymax_candidates.append(np.nanmax(fr_y))
    if isi_y is not None:
        ymax_candidates.append(np.nanmax(isi_y))
    if bool(config.get("normalize_each_density_to_peak", True)):
        if fr_y is not None and np.nanmax(fr_y) > 0:
            fr_y = fr_y / float(np.nanmax(fr_y))
        if isi_y is not None and np.nanmax(isi_y) > 0:
            isi_y = isi_y / float(np.nanmax(isi_y))
        ymax_candidates = []
        if fr_y is not None:
            ymax_candidates.append(np.nanmax(fr_y))
        if isi_y is not None:
            ymax_candidates.append(np.nanmax(isi_y))
    ymax = 1.18 * float(np.nanmax(ymax_candidates)) if ymax_candidates else 1.0

    if fr_x is not None and fr_y is not None:
        fr_color = str(config["firing_rate_color"])
        ax.plot(fr_x, fr_y, color=fr_color, lw=float(config["line_lw"]), zorder=4)
        ax.fill_between(fr_x, 0.0, fr_y, color=fr_color, alpha=float(config["firing_rate_fill_alpha"]), zorder=2)
        ax.set_xlim(float(np.nanmin(fr_x)), float(np.nanmax(fr_x)))
    if isi_x is not None and isi_y is not None:
        isi_color = str(config["isi_cv_color"])
        ax_top.plot(isi_x, isi_y, color=isi_color, lw=float(config["line_lw"]), zorder=4)
        ax_top.fill_between(isi_x, 0.0, isi_y, color=isi_color, alpha=float(config["isi_cv_fill_alpha"]), zorder=1)
        ax_top.set_xlim(float(np.nanmin(isi_x)), float(np.nanmax(isi_x)))

    ax.set_ylim(0.0, ymax)
    ax_top.set_ylim(0.0, ymax)
    ax.set_ylabel(str(config["ylabel"]))
    ax.set_xlabel(str(config["bottom_xlabel"]), color=str(config["firing_rate_color"]))
    ax_top.set_xlabel(str(config["top_xlabel"]), color=str(config["isi_cv_color"]))
    ax.tick_params(axis="x", colors=str(config["firing_rate_color"]))
    ax_top.tick_params(axis="x", colors=str(config["isi_cv_color"]))
    ax_top.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    ax.set_title(_format_panel_title(config["title"]), fontsize=10, loc="right")
    ax.grid(alpha=0.25)

    def _fmt(value):
        return "~=0" if (not np.isfinite(value) or abs(float(value)) < 0.005) else f"{float(value):.2f}"

    fr_text = (
        f"FR mean: {_fmt(_mean_series_or_nan(stats_df, 'firing_rate_mean'))} Hz\n"
        f"FR sd: {_fmt(_mean_series_or_nan(stats_df, 'firing_rate_sd'))} Hz\n"
        f"FR skew: {_fmt(_mean_series_or_nan(stats_df, 'firing_rate_skewness'))}"
    )
    isi_text = (
        f"ISI CV mean: {_fmt(_mean_series_or_nan(stats_df, 'isi_cv_mean'))}\n"
        f"ISI CV sd: {_fmt(_mean_series_or_nan(stats_df, 'isi_cv_sd'))}\n"
        f"ISI CV skew: {_fmt(_mean_series_or_nan(stats_df, 'isi_cv_skewness'))}"
    )
    ax.text(
        0.01,
        0.98,
        isi_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=str(config["isi_cv_color"]),
        fontsize=float(config["stats_fontsize"]),
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
        zorder=5,
    )
    ax.text(
        0.99,
        0.98,
        fr_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=str(config["firing_rate_color"]),
        fontsize=float(config["stats_fontsize"]),
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
        zorder=5,
    )
    return {
        "firing_rate_mean": _mean_series_or_nan(stats_df, "firing_rate_mean"),
        "firing_rate_sd": _mean_series_or_nan(stats_df, "firing_rate_sd"),
        "firing_rate_skewness": _mean_series_or_nan(stats_df, "firing_rate_skewness"),
        "isi_cv_mean": _mean_series_or_nan(stats_df, "isi_cv_mean"),
        "isi_cv_sd": _mean_series_or_nan(stats_df, "isi_cv_sd"),
        "isi_cv_skewness": _mean_series_or_nan(stats_df, "isi_cv_skewness"),
    }


def _plot_mean_panel(
    ax,
    time_rel,
    traces,
    *,
    color,
    ylabel,
    title,
    event_times,
    colors,
    style,
    show_individual_traces,
    show_std_band,
    show_reference_std_lines=False,
    fixed_ylim=None,
    reference_center=0.0,
):
    mean_trace = np.nanmean(traces, axis=0)
    std_trace = np.nanstd(traces, axis=0)
    _draw_shared_markers(ax, event_times, colors, style, burst_center_s=None, burst_start_s=None, burst_end_s=None, time_offset_s=0.0, show_burst=False)
    if show_individual_traces:
        for trace in traces:
            ax.plot(time_rel, trace, color=color, lw=style["individual_trace_lw"], alpha=style["individual_trace_alpha"], zorder=2)
    if show_std_band:
        ax.fill_between(time_rel, mean_trace - std_trace, mean_trace + std_trace, color=color, alpha=style["std_band_alpha"], zorder=3)
    if show_reference_std_lines:
        reference_std = float(np.mean([np.nanstd(trace) for trace in traces]))
        ax.axhline(reference_center + reference_std, color=color, lw=style["reference_std_lw"], ls="--", alpha=style["reference_std_alpha"], zorder=1)
        ax.axhline(reference_center - reference_std, color=color, lw=style["reference_std_lw"], ls="--", alpha=style["reference_std_alpha"], zorder=1)
    ax.plot(time_rel, mean_trace, color=color, lw=style["mean_trace_lw"], zorder=4)
    ylim_parts = [mean_trace]
    if show_individual_traces:
        ylim_parts.append(traces.reshape(-1))
    if show_std_band:
        ylim_parts.extend([mean_trace - std_trace, mean_trace + std_trace])
    if show_reference_std_lines:
        reference_std = float(np.mean([np.nanstd(trace) for trace in traces]))
        ylim_parts.append(np.array([reference_center + reference_std, reference_center - reference_std], dtype=float))
    ymin, ymax = fixed_ylim if fixed_ylim is not None else _auto_limits(np.concatenate(ylim_parts), zero_floor=False)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel)
    ax.set_title(_format_panel_title(title), fontsize=10, loc="right")
    ax.grid(alpha=0.25)
    return mean_trace


def _plot_spectrum_panel(
    ax,
    freqs,
    spectra,
    *,
    color,
    ylabel,
    title,
    style,
    show_individual_traces,
    show_std_band,
    fixed_ylim=None,
    xlim=None,
    interval_bounds=None,
    interval_alpha=0.18,
    interval_label=None,
):
    mean_spectrum = np.nanmean(spectra, axis=0)
    std_spectrum = np.nanstd(spectra, axis=0)
    if show_individual_traces:
        for spectrum in spectra:
            ax.plot(freqs, spectrum, color=color, lw=style["individual_trace_lw"], alpha=style["individual_trace_alpha"], zorder=2)
    if show_std_band:
        ax.fill_between(freqs, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, color=color, alpha=style["std_band_alpha"], zorder=3)
    ax.plot(freqs, mean_spectrum, color=color, lw=style["mean_trace_lw"], zorder=4)
    if interval_bounds is not None:
        interval_low_hz, interval_high_hz = map(float, interval_bounds)
        interval_mask = (freqs >= interval_low_hz) & (freqs <= interval_high_hz)
        if np.any(interval_mask):
            ax.fill_between(freqs[interval_mask], 0.0, mean_spectrum[interval_mask], color=color, alpha=float(interval_alpha), zorder=1)
    ylim_parts = [mean_spectrum]
    if show_individual_traces:
        ylim_parts.append(spectra.reshape(-1))
    if show_std_band:
        ylim_parts.extend([mean_spectrum - std_spectrum, mean_spectrum + std_spectrum])
    ymin, ymax = fixed_ylim if fixed_ylim is not None else _auto_limits(np.concatenate(ylim_parts), zero_floor=True)
    ax.set_ylim(ymin, ymax)
    if xlim is not None:
        ax.set_xlim(*map(float, xlim))
    ax.set_ylabel(ylabel)
    ax.set_title(_format_panel_title(title), fontsize=10, loc="right")
    if interval_bounds is not None and interval_label:
        ax.text(
            0.02,
            0.95,
            interval_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="black",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.70, "edgecolor": "none"},
            zorder=5,
        )
    ax.grid(alpha=0.25)
    return mean_spectrum


def _resolve_lf_reference_center(time_rel, traces, ref_run):
    normalization_window = ref_run.get("normalization_window_rel_cue_s")
    if normalization_window is None:
        normalization_window = ref_run.get("cst_modulation_normalization_window_rel_cue_s")
    if normalization_window is None:
        return float(np.nanmean(traces))
    normalization_start_s, normalization_end_s = map(float, normalization_window)
    normalization_mask = (time_rel >= normalization_start_s) & (time_rel < normalization_end_s)
    if not np.any(normalization_mask):
        return float(np.nanmean(traces))
    return float(np.nanmean(traces[:, normalization_mask]))


def plot_experimental_batch_summary(
    batch_dir,
    runs,
    *,
    output_dir=None,
    show_individual_traces=True,
    show_std_band=True,
    show_reference_std_lines=True,
    fixed_panel_limits=None,
    plot_style=None,
    xlim_rel_cue_s=None,
    feature_config=None,
    modulation_spectrum_config=None,
    rate_isi_density_config=None,
    window_overlay_specs=None,
    save_figure=True,
):
    if not runs:
        raise ValueError("No analyzed experimental runs were provided")
    style = _resolve_plot_style(plot_style)
    feature_config = _resolve_feature_config(feature_config)
    modulation_spectrum_config = _resolve_modulation_spectrum_config(modulation_spectrum_config)
    rate_isi_density_config = _resolve_rate_isi_density_config(rate_isi_density_config)
    ref = next((run for run in runs if run.get("t_rel_cue_s") is not None), None)
    if ref is None:
        raise ValueError("No CST traces are available in this experimental batch")

    colors = dict(ref["colors"])
    full_time_rel = np.asarray(ref["t_rel_cue_s"], dtype=float)
    display_window = tuple(map(float, xlim_rel_cue_s)) if xlim_rel_cue_s is not None else tuple(map(float, ref["full_window_rel_cue_s"]))
    zoom_mask = (full_time_rel >= display_window[0]) & (full_time_rel <= display_window[1])
    zoom_t = full_time_rel[zoom_mask]
    features = {}
    window_legend_handles = []

    fig, axes = plt.subplots(4, 3, figsize=style["figure_size"], dpi=style["figure_dpi"])
    _plot_rate_isi_density_panel(axes[0, 0], runs, rate_isi_density_config)
    axes[3, 0].axis("off")

    def _plot_panel_from_key(ax, run_key, title, color, ylabel="spikes/s/MU", panel_limit_key=None, feature_unit_label=None):
        traces = _collect_available_traces(runs, run_key, "t_rel_cue_s", full_time_rel)
        if traces is None:
            ax.axis("off")
            return
        traces_zoom = traces[:, zoom_mask]
        mean_trace = _plot_mean_panel(
            ax,
            zoom_t,
            traces_zoom,
            color=color,
            ylabel=ylabel,
            title=title,
            event_times=ref["event_times"],
            colors=colors,
            style=style,
            show_individual_traces=show_individual_traces,
            show_std_band=show_std_band,
            show_reference_std_lines=show_reference_std_lines and run_key in ("cst_lf", "cst_alpha", "cst_beta"),
            fixed_ylim=None if fixed_panel_limits is None else fixed_panel_limits.get(panel_limit_key or run_key),
            reference_center=_resolve_lf_reference_center(full_time_rel, traces, ref) if run_key == "cst_lf" else 0.0,
        )
        if feature_config["enabled"] and run_key in ("cst_lf", "cst_alpha_mod", "cst_beta_mod"):
            try:
                full_mean = np.nanmean(traces, axis=0)
                feature_result = fit_baseline_spline_and_extract_feature(
                    full_time_rel,
                    full_mean,
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
                _annotate_feature_panel(ax, zoom_t, mean_trace, color, feature_result, feature_config, style, feature_unit_label)
                features[f"{run_key}_feature_mean_residual"] = feature_result.feature_mean_residual
                features[f"{run_key}_feature_signed_area"] = feature_result.feature_signed_area
                features[f"{run_key}_fit_r2"] = feature_result.fit_r2
                features[f"{run_key}_fit_r2_gain_over_linear"] = feature_result.fit_r2_gain_over_linear
                features[f"{run_key}_trend_post_minus_pre"] = feature_result.trend_post_minus_pre
            except Exception as exc:
                warnings.warn(f"Could not extract feature for {Path(batch_dir).name} / {run_key}: {exc}")
        for handle in _draw_window_overlays(ax, display_window, ref, feature_config, window_overlay_specs):
            if handle.get_label() not in {h.get_label() for h in window_legend_handles}:
                window_legend_handles.append(handle)

    spectrum_ylim = None
    spectrum_xlim = None
    if modulation_spectrum_config["enabled"]:
        alpha_freqs, alpha_spectra = _collect_modulation_spectra(runs, "alpha", modulation_spectrum_config)
        beta_freqs, beta_spectra = _collect_modulation_spectra(runs, "beta", modulation_spectrum_config)
        alpha_interval_bounds = None
        beta_interval_bounds = None
        alpha_interval_label = None
        beta_interval_label = None
        if alpha_freqs is not None or beta_freqs is not None:
            spectrum_bounds = []
            if alpha_spectra is not None:
                alpha_mean = np.nanmean(alpha_spectra, axis=0)
                spectrum_bounds.append(alpha_mean)
                if show_individual_traces:
                    spectrum_bounds.append(alpha_spectra.reshape(-1))
                if show_std_band:
                    alpha_std = np.nanstd(alpha_spectra, axis=0)
                    spectrum_bounds.extend([alpha_mean - alpha_std, alpha_mean + alpha_std])
            if beta_spectra is not None:
                beta_mean = np.nanmean(beta_spectra, axis=0)
                spectrum_bounds.append(beta_mean)
                if show_individual_traces:
                    spectrum_bounds.append(beta_spectra.reshape(-1))
                if show_std_band:
                    beta_std = np.nanstd(beta_spectra, axis=0)
                    spectrum_bounds.extend([beta_mean - beta_std, beta_mean + beta_std])
            if spectrum_bounds:
                spectrum_ylim = _auto_limits(np.concatenate(spectrum_bounds), zero_floor=True)
            all_freqs = []
            if alpha_freqs is not None:
                all_freqs.append(alpha_freqs)
            if beta_freqs is not None:
                all_freqs.append(beta_freqs)
            if all_freqs:
                max_freq = max(float(np.nanmax(freqs)) for freqs in all_freqs)
                spectrum_xlim = (0.0, max_freq)
        if alpha_freqs is None or alpha_spectra is None:
            axes[1, 0].axis("off")
        else:
            alpha_mean_spectrum = np.nanmean(alpha_spectra, axis=0)
            alpha_peak_freq_hz = float(alpha_freqs[int(np.nanargmax(alpha_mean_spectrum))]) if np.any(np.isfinite(alpha_mean_spectrum)) else np.nan
            alpha_interval_bounds = _smallest_contiguous_psd_mass_interval(alpha_freqs, alpha_mean_spectrum, modulation_spectrum_config["interval_mass_pct"])
            alpha_interval_label = (
                f"{alpha_interval_bounds[0]:.1f}-{alpha_interval_bounds[1]:.1f} Hz"
                if np.all(np.isfinite(alpha_interval_bounds)) else None
            )
            features["cst_alpha_modulation_spectrum_peak_hz"] = alpha_peak_freq_hz
            features["cst_alpha_modulation_spectrum_interval_low_hz"] = alpha_interval_bounds[0]
            features["cst_alpha_modulation_spectrum_interval_high_hz"] = alpha_interval_bounds[1]
            _plot_spectrum_panel(
                axes[1, 0],
                alpha_freqs,
                alpha_spectra,
                color=colors["alpha"],
                ylabel=modulation_spectrum_config["ylabel"],
                title="Alpha amplitude modulation spectrum",
                style=style,
                show_individual_traces=show_individual_traces,
                show_std_band=show_std_band,
                fixed_ylim=spectrum_ylim,
                xlim=spectrum_xlim,
                interval_bounds=alpha_interval_bounds if modulation_spectrum_config["show_interval"] else None,
                interval_alpha=modulation_spectrum_config["interval_alpha"],
                interval_label=alpha_interval_label if modulation_spectrum_config["show_interval"] else None,
            )
        if beta_freqs is None or beta_spectra is None:
            axes[2, 0].axis("off")
        else:
            beta_mean_spectrum = np.nanmean(beta_spectra, axis=0)
            beta_peak_freq_hz = float(beta_freqs[int(np.nanargmax(beta_mean_spectrum))]) if np.any(np.isfinite(beta_mean_spectrum)) else np.nan
            beta_interval_bounds = _smallest_contiguous_psd_mass_interval(beta_freqs, beta_mean_spectrum, modulation_spectrum_config["interval_mass_pct"])
            beta_interval_label = (
                f"{beta_interval_bounds[0]:.1f}-{beta_interval_bounds[1]:.1f} Hz"
                if np.all(np.isfinite(beta_interval_bounds)) else None
            )
            features["cst_beta_modulation_spectrum_peak_hz"] = beta_peak_freq_hz
            features["cst_beta_modulation_spectrum_interval_low_hz"] = beta_interval_bounds[0]
            features["cst_beta_modulation_spectrum_interval_high_hz"] = beta_interval_bounds[1]
            _plot_spectrum_panel(
                axes[2, 0],
                beta_freqs,
                beta_spectra,
                color=colors["beta"],
                ylabel=modulation_spectrum_config["ylabel"],
                title="Beta amplitude modulation spectrum",
                style=style,
                show_individual_traces=show_individual_traces,
                show_std_band=show_std_band,
                fixed_ylim=spectrum_ylim,
                xlim=spectrum_xlim,
                interval_bounds=beta_interval_bounds if modulation_spectrum_config["show_interval"] else None,
                interval_alpha=modulation_spectrum_config["interval_alpha"],
                interval_label=beta_interval_label if modulation_spectrum_config["show_interval"] else None,
            )
    else:
        axes[1, 0].axis("off")
        axes[2, 0].axis("off")

    _plot_panel_from_key(axes[0, 1], "cst_lf", f"CST low-pass ({ref['lf_band_hz'][0]:.2f}-{ref['lf_band_hz'][1]:.2f} Hz)", colors["lf"], ylabel="spikes/s/MU", panel_limit_key="cst_lf", feature_unit_label="spikes/s/MU")
    _plot_panel_from_key(axes[1, 1], "cst_alpha", f"CST band-pass alpha ({ref['alpha_band_hz'][0]:.1f}-{ref['alpha_band_hz'][1]:.1f} Hz)", colors["alpha"], ylabel="spikes/s/MU", panel_limit_key="cst_alpha")
    _plot_panel_from_key(axes[2, 1], "cst_beta", f"CST band-pass beta ({ref['beta_band_hz'][0]:.1f}-{ref['beta_band_hz'][1]:.1f} Hz)", colors["beta"], ylabel="spikes/s/MU", panel_limit_key="cst_beta")
    axes[3, 1].axis("off")

    axes[0, 2].axis("off")
    legend_handles = [
        Line2D([0], [0], color=colors["cue"], lw=style["get_ready_lw"], ls="--", label="READY cue"),
        Line2D([0], [0], color=colors["cue"], lw=style["go_cue_lw"], ls="-", label="GO / NOGO cue"),
        Patch(facecolor=style["feature_outside_color"], edgecolor="none", alpha=style["feature_outside_alpha"], label="Outside burst window"),
        Line2D([0], [0], color=style.get("feature_baseline_backdrop_color", "#BDBDBD"), lw=float(style.get("feature_baseline_backdrop_lw", 3.0)), ls="-", alpha=float(style.get("feature_baseline_backdrop_alpha", 0.85)), label="Fitted curve"),
        Line2D([0], [0], color=style["feature_baseline_color"], lw=style["feature_baseline_lw"], ls="--", label="Baseline"),
        Line2D([0], [0], color=style["feature_baseline_color"], lw=style["feature_baseline_lw"], ls=":", label="Post-cue trend"),
    ]
    if show_individual_traces:
        legend_handles.append(Line2D([0], [0], color=colors["cst"], lw=style["individual_trace_lw"], alpha=style["individual_trace_alpha"], label="Individual trials"))
    if show_reference_std_lines:
        legend_handles.append(Line2D([0], [0], color=colors["cst"], lw=style["reference_std_lw"], ls="--", alpha=style["reference_std_alpha"], label="+/- mean trial SD"))
    legend_handles.extend(window_legend_handles)
    axes[0, 2].legend(handles=legend_handles, loc="center", frameon=True, title="Shared markers and traces")

    modulation_ylabel = "Modulation relative to baseline (z)" if ref["cst_modulation_mode"] == "zscore" else "Modulation relative to baseline (%)"
    _plot_panel_from_key(axes[1, 2], "cst_alpha_mod", f"Alpha modulation ({'z-score' if ref['cst_modulation_mode'] == 'zscore' else '% baseline'} of Hilbert transform)", colors["alpha"], ylabel=modulation_ylabel, panel_limit_key="mod_alpha", feature_unit_label="z" if ref["cst_modulation_mode"] == "zscore" else "%")
    _plot_panel_from_key(axes[2, 2], "cst_beta_mod", f"Beta modulation ({'z-score' if ref['cst_modulation_mode'] == 'zscore' else '% baseline'} of Hilbert transform)", colors["beta"], ylabel=modulation_ylabel, panel_limit_key="mod_beta", feature_unit_label="z" if ref["cst_modulation_mode"] == "zscore" else "%")

    ref_sync = next((run for run in runs if run.get("sync_time_rel") is not None and run.get("sync_trace") is not None), None)
    if ref_sync is None:
        axes[3, 2].axis("off")
    else:
        sync_time_rel = np.asarray(ref_sync["sync_time_rel"], dtype=float)
        sync_mask = (sync_time_rel >= display_window[0]) & (sync_time_rel <= display_window[1])
        traces = _collect_available_traces(runs, "sync_trace", "sync_time_rel", sync_time_rel)
        if traces is None:
            axes[3, 2].axis("off")
        else:
            traces_zoom = traces[:, sync_mask]
            mean_sync = _plot_mean_panel(
                axes[3, 2],
                sync_time_rel[sync_mask],
                traces_zoom,
                color=colors["sync"],
                ylabel="Excess coincidence index" if ref["sync_display_mode"] == "absolute" else "Excess coincidence index (z)",
                title=f"Sliding synchrony ({'absolute' if ref['sync_display_mode'] == 'absolute' else 'baseline-zscored'})",
                event_times=ref["event_times"],
                colors=colors,
                style=style,
                show_individual_traces=show_individual_traces,
                show_std_band=show_std_band,
                show_reference_std_lines=False,
                fixed_ylim=None if fixed_panel_limits is None else fixed_panel_limits.get("sync"),
            )
            if feature_config["enabled"]:
                try:
                    full_sync_mean = np.nanmean(traces, axis=0)
                    feature_result = fit_baseline_spline_and_extract_feature(
                        sync_time_rel,
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
                    _annotate_feature_panel(axes[3, 2], sync_time_rel[sync_mask], mean_sync, colors["sync"], feature_result, feature_config, style, "a.u.")
                    features["sync_feature_mean_residual"] = feature_result.feature_mean_residual
                    features["sync_feature_signed_area"] = feature_result.feature_signed_area
                    features["sync_fit_r2"] = feature_result.fit_r2
                    features["sync_fit_r2_gain_over_linear"] = feature_result.fit_r2_gain_over_linear
                    features["sync_trend_post_minus_pre"] = feature_result.trend_post_minus_pre
                except Exception as exc:
                    warnings.warn(f"Could not extract feature for {Path(batch_dir).name} / sync: {exc}")
            for handle in _draw_window_overlays(axes[3, 2], display_window, ref, feature_config, window_overlay_specs):
                if handle.get_label() not in {h.get_label() for h in window_legend_handles}:
                    window_legend_handles.append(handle)

    tick_start = np.floor(float(np.min(zoom_t)) / 0.5) * 0.5
    tick_end = np.ceil(float(np.max(zoom_t)) / 0.5) * 0.5
    xticks = np.arange(tick_start, tick_end + 0.25, 0.5, dtype=float)
    for row_axes in axes:
        for ax in row_axes:
            if ax.axison and ax not in (axes[0, 0], axes[1, 0], axes[2, 0]):
                ax.set_xlim(float(np.min(zoom_t)), float(np.max(zoom_t)))
                ax.xaxis.set_major_locator(MultipleLocator(0.5))
                ax.set_xticks(xticks)

    if axes[2, 0].axison:
        if spectrum_xlim is not None:
            spectrum_tick_step = 5.0 if float(spectrum_xlim[1]) > 10.0 else 1.0
            spectrum_ticks = np.arange(0.0, float(spectrum_xlim[1]) + 0.5 * spectrum_tick_step, spectrum_tick_step, dtype=float)
            axes[1, 0].set_xticks(spectrum_ticks)
            axes[2, 0].set_xticks(spectrum_ticks)
            axes[1, 0].tick_params(axis="x", labelbottom=True)
            axes[2, 0].tick_params(axis="x", labelbottom=True)
        axes[2, 0].set_xlabel("Frequency (Hz)")
    axes[2, 1].set_xlabel("Time relative to go / no-go cue (s)")
    axes[3, 2].set_xlabel("Time relative to go / no-go cue (s)")
    participant_id, session_tag, condition_id = ref["batch_key"]
    fig.suptitle(f"Experimental CST and synchrony summary | {participant_id} | {session_tag} | {condition_id} | n={len(runs)} trials", fontsize=12)
    plt.tight_layout()

    figure_path = None
    if save_figure:
        output_dir = Path(output_dir) if output_dir is not None else Path(batch_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = output_dir / f"{participant_id}__{session_tag}__{condition_id}__summary.png"
        plt.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path, features


def summarize_experimental_batches(
    analyzed_root,
    *,
    output_dir=None,
    participant_ids=None,
    session_tags=None,
    condition_ids=None,
    max_batches=None,
    show_individual_traces=True,
    show_std_band=True,
    show_reference_std_lines=True,
    fixed_ylim_across_batches=False,
    color_overrides=None,
    plot_style=None,
    xlim_rel_cue_s=None,
    feature_config=None,
    modulation_spectrum_config=None,
    rate_isi_density_config=None,
    window_overlay_specs=None,
    export_figures=True,
    show_progress=True,
):
    grouped_runs = collect_experimental_batch_runs(analyzed_root, participant_ids=participant_ids, session_tags=session_tags, condition_ids=condition_ids, max_batches=max_batches, color_overrides=color_overrides)
    if not grouped_runs:
        raise FileNotFoundError("No analyzed experimental sidecar HDF5 files were found.")
    output_dir = Path(output_dir) if output_dir is not None else Path(analyzed_root) / "_batch_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_panel_limits = None
    if fixed_ylim_across_batches:
        fixed_panel_limits = _compute_experimental_global_limits(grouped_runs, show_individual_traces=show_individual_traces, show_std_band=show_std_band, show_reference_std_lines=show_reference_std_lines, xlim_rel_cue_s=xlim_rel_cue_s)

    rows = []
    for batch_dir, runs in _progress_iter(sorted(grouped_runs.items(), key=lambda item: item[0].as_posix()), desc="Summarizing experimental batches", show_progress=show_progress):
        figure_path, features = plot_experimental_batch_summary(
            batch_dir,
            runs,
            output_dir=output_dir,
            show_individual_traces=show_individual_traces,
            show_std_band=show_std_band,
            show_reference_std_lines=show_reference_std_lines,
            fixed_panel_limits=fixed_panel_limits,
            plot_style=plot_style,
            xlim_rel_cue_s=xlim_rel_cue_s,
            feature_config=feature_config,
            modulation_spectrum_config=modulation_spectrum_config,
            rate_isi_density_config=rate_isi_density_config,
            window_overlay_specs=window_overlay_specs,
            save_figure=export_figures,
        )
        participant_id, session_tag, condition_id = runs[0]["batch_key"]
        row = {
            "batch_dir": str(batch_dir),
            "participant_id": participant_id,
            "session_tag": session_tag,
            "condition_id": condition_id,
            "n_trials": int(len(runs)),
            "figure_path": np.nan if figure_path is None else str(figure_path),
        }
        row.update(features)
        rows.append(row)
    return pd.DataFrame(rows)


def _extract_saved_rate_isi_arrays(h5_file):
    if "analysis" not in h5_file or "firing_rate_diagnostics" not in h5_file["analysis"]:
        return None, None
    fr_grp = h5_file["analysis"]["firing_rate_diagnostics"]
    return _load_group_dataset_if_present(fr_grp, "mn_mean_firing_rate_hz"), _load_group_dataset_if_present(fr_grp, "mn_isi_cv")


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
        isis = np.diff(times)
        inst_rates = 1.0 / isis
        firing_rate_means.append(float(np.mean(inst_rates)))
        if active_id_set is not None and unit_i not in active_id_set:
            isi_covs.append(np.nan)
            continue
        mean_isi = float(np.mean(isis))
        std_isi = float(np.std(isis))
        raw_cov = std_isi / mean_isi if mean_isi > 0 else np.nan
        isi_covs.append(raw_cov if (np.isfinite(raw_cov) and raw_cov >= 0.01) else 0.0)
    return np.asarray(firing_rate_means, dtype=float), np.asarray(isi_covs, dtype=float)


def _summarize_distribution(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "sd": np.nan, "skewness": np.nan}
    return {"mean": float(np.mean(finite)), "sd": float(np.std(finite)), "skewness": float(pd.Series(finite).skew())}


def collect_experimental_rate_isi_moments(
    analyzed_root,
    *,
    participant_ids=None,
    session_tags=None,
    condition_ids=None,
    max_batches=None,
    show_progress=True,
):
    grouped_runs = collect_experimental_batch_runs(analyzed_root, participant_ids=participant_ids, session_tags=session_tags, condition_ids=condition_ids, max_batches=max_batches)
    rows = []
    skipped_runs = []
    for _, runs in _progress_iter(sorted(grouped_runs.items(), key=lambda item: item[0].as_posix()), desc="Collecting experimental rate / ISI metrics", show_progress=show_progress):
        for run in runs:
            sidecar_path = Path(run["sidecar_path"])
            participant_id, session_tag, condition_id = run["batch_key"]
            with h5py.File(sidecar_path, "r") as f:
                fr_values, isi_values = _extract_saved_rate_isi_arrays(f)
                value_source = "analysis/firing_rate_diagnostics"
                if fr_values is None or isi_values is None:
                    raw_trial_h5_path = Path(f.attrs.get("raw_trial_h5_path", ""))
                    if raw_trial_h5_path.exists():
                        trial_data = load_experimental_trial(raw_trial_h5_path)
                        active_unit_ids = None
                        summary_grp = f.get("analysis/summary_metadata", None)
                        if summary_grp is not None and "active_unit_ids" in summary_grp:
                            active_unit_ids = np.asarray(summary_grp["active_unit_ids"], dtype=int)
                        else:
                            threshold_hz = (
                                float(np.asarray(summary_grp["active_unit_rate_threshold_hz"], dtype=float).reshape(-1)[0])
                                if summary_grp is not None and "active_unit_rate_threshold_hz" in summary_grp else 3.0
                            )
                            cst_window = (
                                tuple(np.asarray(summary_grp["analysis_window_rel_cue_s"], dtype=float).reshape(-1).tolist())
                                if summary_grp is not None and "analysis_window_rel_cue_s" in summary_grp else (-4.0, 1.0)
                            )
                            active_selection = classify_active_units_by_rate(
                                trial_data["spike_trains_trial_s"],
                                analysis_start_s=float(trial_data["cue_time_abs_s"] + cst_window[0]),
                                analysis_end_s=float(trial_data["cue_time_abs_s"] + cst_window[1]),
                                rate_threshold_hz=threshold_hz,
                            )
                            active_unit_ids = active_selection["active_unit_ids"]
                        fr_values, isi_values = _compute_fr_and_isi_cv_from_spikes(
                            trial_data["spike_trains_trial_s"],
                            active_unit_ids_for_isi_cv=active_unit_ids,
                        )
                        value_source = "raw_trial_h5"
                    else:
                        skipped_runs.append(str(sidecar_path))
                        continue
            fr_stats = _summarize_distribution(fr_values)
            isi_stats = _summarize_distribution(isi_values)
            rows.append(
                {
                    "batch_dir": str(sidecar_path.parent),
                    "participant_id": participant_id,
                    "session_tag": session_tag,
                    "condition_id": condition_id,
                    "sidecar_path": str(sidecar_path),
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


def _scalar_from_group_dataset(group, key):
    if group is None or key not in group:
        return np.nan
    arr = np.asarray(group[key], dtype=float).reshape(-1)
    return np.nan if arr.size == 0 else float(arr[0])


def _load_experimental_observation_features_for_sidecar(sidecar_path):
    sidecar_path = Path(sidecar_path)
    feature_names = [
        *OBS_WINDOW_EXPORT_COLUMNS,
        *OBS_BASELINE_EXPORT_COLUMNS,
        *OBS_SPECTRUM_EXPORT_COLUMNS,
    ]
    features = {name: np.nan for name in feature_names}
    with h5py.File(sidecar_path, "r") as f:
        if "analysis" not in f or "observation_features" not in f["analysis"]:
            return features
        obs_grp = f["analysis"]["observation_features"]
        for name in feature_names:
            if name in obs_grp:
                features[name] = _scalar_from_group_dataset(obs_grp, name)
    return features


def _collect_experimental_batch_observation_features(batch_dir):
    sidecar_paths = sorted(Path(batch_dir).glob("*.h5"))
    rows = [_load_experimental_observation_features_for_sidecar(path) for path in sidecar_paths]
    df = pd.DataFrame(rows)
    return {name: _mean_series_or_nan(df, name) for name in rows[0].keys()} if rows else {name: np.nan for name in (*OBS_WINDOW_EXPORT_COLUMNS, *OBS_BASELINE_EXPORT_COLUMNS, *OBS_SPECTRUM_EXPORT_COLUMNS)}


def _load_experimental_unit_count_summary_for_sidecar(sidecar_path):
    sidecar_path = Path(sidecar_path)
    out = {
        "n_units_total": np.nan,
        "n_units_kept": np.nan,
        "unit_fraction_kept": np.nan,
        "active_unit_rate_threshold_hz": np.nan,
    }
    with h5py.File(sidecar_path, "r") as f:
        summary_grp = f.get("analysis/summary_metadata", None)
        if summary_grp is None:
            return out
        for key in out.keys():
            if key in summary_grp:
                out[key] = _scalar_from_group_dataset(summary_grp, key)
    return out


def _collect_experimental_batch_unit_count_summaries(batch_dir):
    sidecar_paths = sorted(Path(batch_dir).glob("*.h5"))
    rows = [_load_experimental_unit_count_summary_for_sidecar(path) for path in sidecar_paths]
    if not rows:
        return {
            "n_units_total": np.nan,
            "n_units_kept_mean": np.nan,
            "n_units_kept_min": np.nan,
            "unit_fraction_kept_mean": np.nan,
            "active_unit_rate_threshold_hz": np.nan,
        }
    df = pd.DataFrame(rows)
    kept = pd.to_numeric(df.get("n_units_kept"), errors="coerce")
    return {
        "n_units_total": _mean_series_or_nan(df, "n_units_total"),
        "n_units_kept_mean": _mean_series_or_nan(df, "n_units_kept"),
        "n_units_kept_min": float(np.nanmin(kept.to_numpy(dtype=float))) if np.any(np.isfinite(kept.to_numpy(dtype=float))) else np.nan,
        "unit_fraction_kept_mean": _mean_series_or_nan(df, "unit_fraction_kept"),
        "active_unit_rate_threshold_hz": _mean_series_or_nan(df, "active_unit_rate_threshold_hz"),
    }


def build_experimental_export_table(
    analyzed_root,
    *,
    batch_summary_df=None,
    rate_isi_metrics_df=None,
    feature_config=None,
    modulation_spectrum_config=None,
    participant_ids=None,
    session_tags=None,
    condition_ids=None,
    max_batches=None,
):
    modulation_spectrum_config = _resolve_modulation_spectrum_config(modulation_spectrum_config)
    grouped_runs = collect_experimental_batch_runs(analyzed_root, participant_ids=participant_ids, session_tags=session_tags, condition_ids=condition_ids, max_batches=max_batches)
    summary_lookup = {}
    if batch_summary_df is not None and not batch_summary_df.empty and "batch_dir" in batch_summary_df.columns:
        summary_lookup = {str(row["batch_dir"]): row.to_dict() for _, row in batch_summary_df.iterrows()}
    rate_groups = {}
    if rate_isi_metrics_df is not None and not rate_isi_metrics_df.empty and "batch_dir" in rate_isi_metrics_df.columns:
        rate_groups = {str(batch_dir): subset.reset_index(drop=True) for batch_dir, subset in rate_isi_metrics_df.groupby("batch_dir", sort=True)}

    rows = []
    for batch_dir, runs in sorted(grouped_runs.items(), key=lambda item: item[0].as_posix()):
        batch_dir = Path(batch_dir)
        participant_id, session_tag, condition_id = runs[0]["batch_key"]
        summary_row = summary_lookup.get(str(batch_dir), {})
        rate_subset = rate_groups.get(str(batch_dir))
        observation_feature_means = _collect_experimental_batch_observation_features(batch_dir)
        unit_count_summary = _collect_experimental_batch_unit_count_summaries(batch_dir)
        first_sidecar = Path(runs[0]["sidecar_path"])
        with h5py.File(first_sidecar, "r") as f:
            summary_grp = f["analysis"]["summary_metadata"] if "analysis" in f and "summary_metadata" in f["analysis"] else None
            normalization_window = (
                tuple(np.asarray(summary_grp["normalization_win_rel_cue_s"], dtype=float).reshape(-1).tolist())
                if summary_grp is not None and "normalization_win_rel_cue_s" in summary_grp else np.nan
            )
            baseline_window = (
                tuple(np.asarray(summary_grp["baseline_win_rel_cue_s"], dtype=float).reshape(-1).tolist())
                if summary_grp is not None and "baseline_win_rel_cue_s" in summary_grp else np.nan
            )
            baseline_extended_window = (
                tuple(np.asarray(summary_grp["baseline_extended_win_rel_cue_s"], dtype=float).reshape(-1).tolist())
                if summary_grp is not None and "baseline_extended_win_rel_cue_s" in summary_grp else np.nan
            )
            ready_window = (
                tuple(np.asarray(summary_grp["ready_win_rel_cue_s"], dtype=float).reshape(-1).tolist())
                if summary_grp is not None and "ready_win_rel_cue_s" in summary_grp else np.nan
            )
            post_window = (
                tuple(np.asarray(summary_grp["post_win_rel_cue_s"], dtype=float).reshape(-1).tolist())
                if summary_grp is not None and "post_win_rel_cue_s" in summary_grp else np.nan
            )
            obs_baseline_window = (
                tuple(np.asarray(summary_grp["obs_baseline_window_rel_cue_s"], dtype=float).reshape(-1).tolist())
                if summary_grp is not None and "obs_baseline_window_rel_cue_s" in summary_grp else np.nan
            )
            cst_normalization_window = (
                tuple(np.asarray(summary_grp["cst_modulation_normalization_window_rel_cue_s"], dtype=float).reshape(-1).tolist())
                if summary_grp is not None and "cst_modulation_normalization_window_rel_cue_s" in summary_grp else np.nan
            )
            spectrum_window = (
                tuple(np.asarray(summary_grp["modulation_spectrum_window_rel_cue_s"], dtype=float).reshape(-1).tolist())
                if summary_grp is not None and "modulation_spectrum_window_rel_cue_s" in summary_grp else np.nan
            )
            spectrum_max_freq_hz = (
                float(np.asarray(summary_grp["modulation_spectrum_max_freq_hz"], dtype=float).reshape(-1)[0])
                if summary_grp is not None and "modulation_spectrum_max_freq_hz" in summary_grp else modulation_spectrum_config.get("max_freq_hz", np.nan)
            )
            spectrum_interval_mass_pct = (
                float(np.asarray(summary_grp["modulation_spectrum_interval_mass_pct"], dtype=float).reshape(-1)[0])
                if summary_grp is not None and "modulation_spectrum_interval_mass_pct" in summary_grp else modulation_spectrum_config.get("interval_mass_pct", np.nan)
            )
            lf_band_hz = (
                tuple(np.asarray(summary_grp["lf_band_hz"], dtype=float).reshape(-1).tolist())
                if summary_grp is not None and "lf_band_hz" in summary_grp else np.nan
            )
            alpha_band_hz = (
                tuple(np.asarray(summary_grp["alpha_band_hz"], dtype=float).reshape(-1).tolist())
                if summary_grp is not None and "alpha_band_hz" in summary_grp else np.nan
            )
            beta_band_hz = (
                tuple(np.asarray(summary_grp["beta_band_hz"], dtype=float).reshape(-1).tolist())
                if summary_grp is not None and "beta_band_hz" in summary_grp else np.nan
            )
            sync_win_ms = (
                float(np.asarray(summary_grp["sync_win_ms"], dtype=float).reshape(-1)[0])
                if summary_grp is not None and "sync_win_ms" in summary_grp else np.nan
            )
            sync_step_ms = (
                float(np.asarray(summary_grp["sync_step_ms"], dtype=float).reshape(-1)[0])
                if summary_grp is not None and "sync_step_ms" in summary_grp else np.nan
            )
            sync_coinc_lag_ms = (
                float(np.asarray(summary_grp["sync_coinc_lag_ms"], dtype=float).reshape(-1)[0])
                if summary_grp is not None and "sync_coinc_lag_ms" in summary_grp else np.nan
            )
            sync_direction_mode = (
                str(summary_grp.attrs.get("sync_direction_mode", np.nan))
                if summary_grp is not None else np.nan
            )
            sync_expectation_mode = (
                str(summary_grp.attrs.get("sync_expectation_mode", np.nan))
                if summary_grp is not None else np.nan
            )
            active_unit_rate_threshold_hz = (
                float(np.asarray(summary_grp["active_unit_rate_threshold_hz"], dtype=float).reshape(-1)[0])
                if summary_grp is not None and "active_unit_rate_threshold_hz" in summary_grp else unit_count_summary.get("active_unit_rate_threshold_hz", np.nan)
            )
        row = {column: np.nan for column in EXPORT_TABLE_COLUMNS}
        row.update(
            {
                "DATASET_CONTEXT_participant_id": participant_id,
                "DATASET_CONTEXT_session_tag": session_tag,
                "DATASET_CONTEXT_condition_id": condition_id,
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
                "ANALYSIS_PARAM_WINDOW_normalization_win_rel_go_cue_s": normalization_window,
                "ANALYSIS_PARAM_WINDOW_baseline_win_rel_go_cue_s": baseline_window,
                "ANALYSIS_PARAM_WINDOW_baseline_extended_win_rel_go_cue_s": baseline_extended_window,
                "ANALYSIS_PARAM_WINDOW_ready_win_rel_go_cue_s": ready_window,
                "ANALYSIS_PARAM_WINDOW_post_win_rel_go_cue_s": post_window,
                "ANALYSIS_PARAM_WINDOW_obs_baseline_win_rel_go_cue_s": obs_baseline_window,
                "ANALYSIS_PARAM_WINDOW_cst_modulation_normalization_win_rel_go_cue_s": cst_normalization_window,
                "ANALYSIS_PARAM_FEATURE_fitting_window": _resolve_feature_export_value(feature_config, "fitting_window"),
                "ANALYSIS_PARAM_FEATURE_analysis_window": _resolve_feature_export_value(feature_config, "analysis_window"),
                "ANALYSIS_PARAM_FEATURE_buffer_s": _resolve_feature_export_value(feature_config, "buffer_s"),
                "ANALYSIS_PARAM_FEATURE_spline_smoothing": _resolve_feature_export_value(feature_config, "spline_smoothing"),
                "ANALYSIS_PARAM_FEATURE_spline_order": _resolve_feature_export_value(feature_config, "spline_order"),
                "ANALYSIS_PARAM_FEATURE_n_knots": _resolve_feature_export_value(feature_config, "n_knots"),
                "ANALYSIS_PARAM_FEATURE_trend_pre_window": _resolve_feature_export_value(feature_config, "trend_pre_window"),
                "ANALYSIS_PARAM_FEATURE_trend_post_window": _resolve_feature_export_value(feature_config, "trend_post_window"),
                "ANALYSIS_PARAM_FILTER_cst_lf_band_hz": lf_band_hz,
                "ANALYSIS_PARAM_FILTER_cst_alpha_band_hz": alpha_band_hz,
                "ANALYSIS_PARAM_FILTER_cst_beta_band_hz": beta_band_hz,
                "ANALYSIS_PARAM_FILTER_active_unit_rate_threshold_hz": active_unit_rate_threshold_hz,
                "ANALYSIS_PARAM_SYNC_win_ms": sync_win_ms,
                "ANALYSIS_PARAM_SYNC_step_ms": sync_step_ms,
                "ANALYSIS_PARAM_SYNC_coinc_lag_ms": sync_coinc_lag_ms,
                "ANALYSIS_PARAM_SYNC_direction_mode": sync_direction_mode,
                "ANALYSIS_PARAM_SYNC_expectation_mode": sync_expectation_mode,
                "ANALYSIS_PARAM_SPECTRUM_window_rel_go_cue_s": spectrum_window,
                "ANALYSIS_PARAM_SPECTRUM_max_freq_hz": spectrum_max_freq_hz,
                "ANALYSIS_PARAM_SPECTRUM_interval_mass_pct": spectrum_interval_mass_pct,
                "ANALYSIS_DIAGNOSIS_n_units_total": unit_count_summary.get("n_units_total", np.nan),
                "ANALYSIS_DIAGNOSIS_n_units_kept_mean": unit_count_summary.get("n_units_kept_mean", np.nan),
                "ANALYSIS_DIAGNOSIS_n_units_kept_min": unit_count_summary.get("n_units_kept_min", np.nan),
                "ANALYSIS_DIAGNOSIS_unit_fraction_kept_mean": unit_count_summary.get("unit_fraction_kept_mean", np.nan),
                "ANALYSIS_DIAGNOSIS_fit_r2_cst_low_frequency": summary_row.get("cst_lf_fit_r2", np.nan),
                "ANALYSIS_DIAGNOSIS_fit_r2_cst_alpha_modulation_hilbert_zscore": summary_row.get("cst_alpha_mod_fit_r2", np.nan),
                "ANALYSIS_DIAGNOSIS_fit_r2_cst_beta_modulation_hilbert_zscore": summary_row.get("cst_beta_mod_fit_r2", np.nan),
                "ANALYSIS_DIAGNOSIS_fit_r2_sync_coincidence_curve": summary_row.get("sync_fit_r2", np.nan),
                "ANALYSIS_DIAGNOSIS_fit_r2_gain_over_linear_cst_low_frequency": summary_row.get("cst_lf_fit_r2_gain_over_linear", np.nan),
                "ANALYSIS_DIAGNOSIS_fit_r2_gain_over_linear_cst_alpha_modulation_hilbert_zscore": summary_row.get("cst_alpha_mod_fit_r2_gain_over_linear", np.nan),
                "ANALYSIS_DIAGNOSIS_fit_r2_gain_over_linear_cst_beta_modulation_hilbert_zscore": summary_row.get("cst_beta_mod_fit_r2_gain_over_linear", np.nan),
                "ANALYSIS_DIAGNOSIS_fit_r2_gain_over_linear_sync_coincidence_curve": summary_row.get("sync_fit_r2_gain_over_linear", np.nan),
            }
        )
        row.update(observation_feature_means)
        rows.append(row)
    export_df = pd.DataFrame(rows)
    if export_df.empty:
        export_df = pd.DataFrame(columns=EXPORT_TABLE_COLUMNS)
    else:
        export_df = export_df.reindex(columns=EXPORT_TABLE_COLUMNS)
    return export_df


def export_experimental_summary_table(
    analyzed_root,
    *,
    output_dir=None,
    batch_summary_df=None,
    rate_isi_metrics_df=None,
    feature_config=None,
    modulation_spectrum_config=None,
    participant_ids=None,
    session_tags=None,
    condition_ids=None,
    max_batches=None,
    csv_name="experimental_batch_feature_summary.csv",
    pickle_name="experimental_batch_feature_summary.pkl",
):
    export_df = build_experimental_export_table(
        analyzed_root,
        batch_summary_df=batch_summary_df,
        rate_isi_metrics_df=rate_isi_metrics_df,
        feature_config=feature_config,
        modulation_spectrum_config=modulation_spectrum_config,
        participant_ids=participant_ids,
        session_tags=session_tags,
        condition_ids=condition_ids,
        max_batches=max_batches,
    )
    output_dir = Path(output_dir) if output_dir is not None else Path(analyzed_root) / "_batch_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_df = export_df.copy()
    for column_name in csv_df.columns:
        csv_df[column_name] = csv_df[column_name].map(_csv_safe_value)
    csv_path = output_dir / csv_name
    pickle_path = output_dir / pickle_name
    csv_df.to_csv(csv_path, index=False)
    export_df.to_pickle(pickle_path)
    return export_df, {"csv_path": str(csv_path), "pickle_path": str(pickle_path)}
