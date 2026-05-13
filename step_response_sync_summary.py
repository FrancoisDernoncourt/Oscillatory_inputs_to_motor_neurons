from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


PARAMETER_COLUMNS = [
    "excitatory_input_baseline",
    "lf_target_sd",
    "alpha_target_sd_final",
    "beta_target_sd_final",
    "alpha_beta_envelope_variability_scale",
    "step_current_amplitude_nA",
]

PAIRING_COLUMNS = [
    "paired_repeat_seed",
    "step_index",
    "is_zero_step_reference",
    "parameter_set_folder_name",
    "repeat_folder_name",
    "step_folder_name",
    "paired_reference_folder_name",
]

STEP_SCALAR_COLUMNS = [
    "baseline_tonic_input_nA",
    "lf_input_amplitude_nA",
    "alpha_input_amplitude_nA",
    "beta_input_amplitude_nA",
    "step_current_amplitude_nA",
    "step_current_amplitude_percent_baseline",
    "step_current_time_abs_s",
    "step_current_time_usable_rel_s",
    "step_current_duration_s",
    "matched_step_effect_available",
    "zero_step_condition",
    "step_response_cst_smoothing_tau_s",
    "step_response_cst_hann_half_width_s",
    "cst_smooth_causal_hann_half_width_s",
    "cst_smooth_causal_hann_half_width_samples",
    "cst_smooth_causal_hann_kernel_samples",
    "pre_step_cst_smooth_mean",
    "pre_step_cst_smooth_median",
    "pre_step_cst_smooth_sd",
    "post_step_cst_smooth_mean",
    "post_step_cst_smooth_median",
    "post_step_cst_smooth_sd",
    "late_post_step_cst_smooth_mean",
    "late_post_step_cst_smooth_median",
    "late_post_step_cst_smooth_sd",
    "delta_cst_smooth_raw",
    "response_gain_cst_smooth_raw_per_nA",
    "response_gain_cst_smooth_raw_per_uA",
    "pre_step_mean_firing_rate_hz",
    "pre_step_sd_firing_rate_hz",
    "pre_step_isi_cv_mean",
    "pre_step_isi_cv_sd",
    "post_step_mean_firing_rate_hz",
    "post_step_sd_firing_rate_hz",
    "post_step_isi_cv_mean",
    "post_step_isi_cv_sd",
    "pre_step_cst_lf_mean",
    "pre_step_cst_lf_sd",
    "post_step_cst_lf_mean",
    "post_step_cst_lf_sd",
    "delta_cst_lf",
    "response_gain_cst_lf_per_nA",
    "response_gain_cst_per_nA",
    "response_gain_cst_per_uA",
    "pre_step_cst_hann_mean",
    "pre_step_cst_hann_sd",
    "post_step_cst_hann_mean",
    "post_step_cst_hann_sd",
    "delta_cst_hann",
    "response_gain_cst_hann_per_nA",
    "pre_step_effect_mean",
    "pre_step_effect_median",
    "pre_step_effect_sd",
    "late_post_step_effect_mean",
    "late_post_step_effect_median",
    "late_post_step_effect_sd",
    "late_post_step_effect_sign_corrected_median",
    "delta_cst_effect",
    "response_gain_sign_corrected_cst_per_nA",
    "response_gain_sign_corrected_cst_per_uA",
    "response_has_expected_sign",
    "response_magnitude_sufficient",
    "linear_initial_slope_cst_per_s",
    "linear_initial_slope_normalized_per_nA",
    "linear_initial_slope_normalized_per_uA",
    "linear_selected_intercept",
    "linear_selected_window_end_s",
    "linear_selected_window_r2",
    "linear_selected_window_response_fraction",
    "linear_slope_fit_valid",
    "response_shape_tau_ms",
    "response_shape_gamma",
    "response_shape_fraction_50ms",
    "response_shape_fraction_150ms",
    "response_shape_fitted_t50_ms",
    "response_shape_fitted_t90_ms",
    "response_shape_fit_r2",
    "response_shape_fit_rmse",
    "response_shape_R_inf",
    "xcorr_lag_s",
    "xcorr_peak_corr",
    "xcorr_lag_sign_corrected_s",
    "baseline_sync_mean",
    "baseline_sync_sd",
    "pre_step_sync_mean",
    "pre_step_sync_sd",
    "post_step_sync_mean",
    "post_step_sync_sd",
    "post_step_sync_mean_actual_step",
    "post_step_sync_sd_actual_step",
    "pre_step_sync_mean_baseline_matched",
    "pre_step_sync_sd_baseline_matched",
    "local_step_sync_mean_baseline_matched",
    "local_step_sync_sd_baseline_matched",
]

STEP_SCALAR_ALIASES = {
    "baseline_sync_mean": "pre_step_sync_mean",
    "baseline_sync_sd": "pre_step_sync_sd",
    "post_step_sync_mean_actual_step": "post_step_sync_mean",
    "post_step_sync_sd_actual_step": "post_step_sync_sd",
    "linear_initial_slope_cst_per_s": "initial_slope_cst_per_s",
    "linear_initial_slope_normalized_per_nA": "initial_slope_normalized_per_nA",
    "linear_initial_slope_normalized_per_uA": "initial_slope_normalized_per_uA",
    "linear_selected_intercept": "selected_slope_intercept",
    "linear_selected_window_end_s": "selected_slope_window_end_s",
    "linear_selected_window_r2": "selected_slope_window_r2",
    "linear_selected_window_response_fraction": "selected_slope_window_response_fraction",
    "linear_slope_fit_valid": "slope_fit_valid",
}


def find_priors_pickle(batch_root: str | Path) -> Path:
    batch_root = Path(batch_root)
    candidates = sorted(batch_root.glob("*priors*.pkl"))
    if not candidates:
        candidates = sorted(batch_root.rglob("*priors*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No priors pickle found under {batch_root}")
    if len(candidates) > 1:
        warnings.warn(f"Multiple priors pickles found under {batch_root}; using {candidates[-1]}")
    return candidates[-1]


def load_step_response_priors(batch_root: str | Path) -> dict:
    priors_path = find_priors_pickle(batch_root)
    with priors_path.open("rb") as f:
        payload = pickle.load(f)
    payload = dict(payload)
    payload["priors_path"] = priors_path
    return payload


def _find_parameter_set_dir(path: Path) -> Path | None:
    for parent in [path, *path.parents]:
        if parent.name.startswith("parameter_set_"):
            return parent
    return None


def _find_named_parent(path: Path, prefix: str) -> Path | None:
    for parent in [path, *path.parents]:
        if parent.name.startswith(prefix):
            return parent
    return None


def _parse_parameter_set_id(parameter_set_dir: Path | None) -> float:
    if parameter_set_dir is None:
        return np.nan
    try:
        return float(int(parameter_set_dir.name.split("_")[-1]))
    except Exception:
        return np.nan


def _as_scalar(value) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.nan
    if arr.size == 0:
        return np.nan
    return float(arr[0])


def _dataset_scalar(group, key: str) -> float:
    if group is None or key not in group:
        return np.nan
    return _as_scalar(group[key][()])


def _dataset_vector(group, key: str) -> np.ndarray:
    if group is None or key not in group:
        return np.asarray([], dtype=float)
    return np.asarray(group[key][()], dtype=float).reshape(-1)


def _read_spike_trains_mn(h5_file: h5py.File, active_unit_ids: np.ndarray | None = None) -> list[np.ndarray]:
    mn_group = h5_file.get("spike_trains/MN")
    if mn_group is None:
        return []
    if active_unit_ids is None or np.asarray(active_unit_ids).size == 0:
        keys = sorted(mn_group.keys(), key=lambda name: int(name.split("_")[-1]))
        return [np.asarray(mn_group[key], dtype=float) for key in keys]
    spike_trains = []
    for unit_i in np.asarray(active_unit_ids, dtype=int).reshape(-1):
        key = f"MN_{int(unit_i)}"
        if key in mn_group:
            spike_trains.append(np.asarray(mn_group[key], dtype=float))
    return spike_trains


def _recompute_baseline_sync_from_spikes(f: h5py.File, params: dict, step_group) -> tuple[float, float, str]:
    from analyzer_with_force import compute_sliding_sync_trace

    active_ids = _dataset_vector(step_group, "active_unit_ids").astype(int)
    spike_trains = _read_spike_trains_mn(f, active_unit_ids=active_ids)
    if len(spike_trains) < 2:
        return np.nan, np.nan, "recompute_failed_lt2_units"

    step_abs_s = _dataset_scalar(step_group, "step_current_time_abs_s")
    pre_rel = _dataset_vector(step_group, "step_pre_window_rel_s")
    if not np.isfinite(step_abs_s) or pre_rel.size != 2:
        return np.nan, np.nan, "recompute_failed_missing_window"

    baseline_abs = step_abs_s + pre_rel.astype(float)
    baseline_rel = tuple(pre_rel.astype(float).tolist())

    sync_win_s = float(params.get("sync_win_ms", 100.0)) / 1000.0
    sync_step_ms = float(params.get("sync_step_ms", 10.0))
    duration_s = float(params.get("duration_with_ignored_window", params.get("duration", np.nan)))
    if not np.isfinite(duration_s):
        duration_s = max(float(baseline_abs[1] + sync_win_s), float(step_abs_s + 1.0))
    analysis_start_s = max(0.0, float(baseline_abs[0]) - sync_win_s)
    analysis_end_s = min(float(duration_s), float(baseline_abs[1]) + sync_win_s)
    if analysis_end_s <= analysis_start_s:
        return np.nan, np.nan, "recompute_failed_bad_analysis_window"

    try:
        result = compute_sliding_sync_trace(
            spike_trains,
            cue_time_sec=step_abs_s,
            bin_ms=float(params.get("sync_bin_ms", 1.0)),
            sync_win_ms=float(params.get("sync_win_ms", 100.0)),
            sync_step_ms=sync_step_ms,
            coinc_lag_ms=float(params.get("sync_coinc_lag_ms", 6.0)),
            direction_mode=str(params.get("sync_direction_mode", "legacy_forward")),
            expectation_mode=str(params.get("sync_expectation_mode", "analytic")),
            n_surrogates=int(params.get("sync_n_surrogates", 100)),
            surrogate_min_shift_ms=float(params.get("sync_surrogate_min_shift_ms", 1.0)),
            duration_s=duration_s,
            analysis_start_s=analysis_start_s,
            analysis_end_s=analysis_end_s,
            unit_ids=active_ids,
        )
        t_sync = np.asarray(result["t_sync"], dtype=float)
        sync_trace = np.asarray(result["sync_trace"], dtype=float)
        baseline_mask = (t_sync >= baseline_rel[0]) & (t_sync <= baseline_rel[1]) & np.isfinite(sync_trace)
        if not np.any(baseline_mask):
            return np.nan, np.nan, "recompute_failed_empty_baseline_window"
        baseline_values = sync_trace[baseline_mask]
        baseline_mean = float(np.mean(baseline_values))
        baseline_sd = float(np.std(baseline_values))
    except Exception as exc:
        message = str(exc).strip().replace(" ", "_")[:80]
        return np.nan, np.nan, f"recompute_failed_{type(exc).__name__}_{message}"
    return baseline_mean, baseline_sd, "recomputed_from_spikes"


def load_step_response_run_row(
    h5_path: str | Path,
    *,
    recompute_baseline_sync_if_missing: bool = True,
) -> dict:
    h5_path = Path(h5_path)
    run_dir = h5_path.parent
    parameter_set_dir = _find_parameter_set_dir(h5_path)
    repeat_dir = _find_named_parent(h5_path, "repeat_")
    step_dir = _find_named_parent(h5_path, "step_")
    sim_json = run_dir / "sim_parameters.json"
    params = {}
    if sim_json.exists():
        with sim_json.open("r", encoding="utf-8") as f:
            params = json.load(f)

    row = {
        "run_dir": str(run_dir),
        "simulation_output_h5": str(h5_path),
        "parameter_set_dir": "" if parameter_set_dir is None else str(parameter_set_dir),
        "parameter_set_id": params.get("parameter_set_id", _parse_parameter_set_id(parameter_set_dir)),
        "repeat_index": params.get("repeat_index", np.nan),
        "requested_random_seed": params.get("requested_random_seed", params.get("random_seed", np.nan)),
        "paired_repeat_seed": params.get(
            "paired_repeat_seed",
            params.get("requested_random_seed", params.get("random_seed", np.nan)),
        ),
        "step_index": params.get("step_index", np.nan),
        "is_zero_step_reference": params.get("is_zero_step_reference", np.nan),
        "parameter_set_folder_name": params.get(
            "parameter_set_folder_name",
            "" if parameter_set_dir is None else parameter_set_dir.name,
        ),
        "repeat_folder_name": params.get(
            "repeat_folder_name",
            "" if repeat_dir is None else repeat_dir.name,
        ),
        "step_folder_name": params.get(
            "step_folder_name",
            "" if step_dir is None else step_dir.name,
        ),
        "paired_reference_folder_name": params.get("paired_reference_folder_name", "step_p00000_nA"),
    }
    for key in PARAMETER_COLUMNS:
        row[key] = params.get(key, np.nan)
    if not isinstance(row["is_zero_step_reference"], bool):
        try:
            row["is_zero_step_reference"] = bool(np.isclose(float(row["step_current_amplitude_nA"]), 0.0))
        except Exception:
            row["is_zero_step_reference"] = False

    try:
        with h5py.File(h5_path, "r") as f:
            step_group = f.get("step_response_diagnostics")
            row["step_response_status"] = "" if step_group is None else str(step_group.attrs.get("status", ""))
            row["diagnostics_stage"] = "" if step_group is None else str(step_group.attrs.get("diagnostics_stage", ""))
            row["finalization_status"] = "" if step_group is None else str(step_group.attrs.get("finalization_status", ""))
            row["linear_slope_selection_status"] = "" if step_group is None else str(
                step_group.attrs.get(
                    "linear_slope_selection_status",
                    step_group.attrs.get("slope_selection_status", ""),
                )
            )
            row["response_shape_fit_status"] = "" if step_group is None else str(step_group.attrs.get("response_shape_fit_status", ""))
            row["xcorr_status"] = "" if step_group is None else str(step_group.attrs.get("xcorr_status", ""))
            row["xcorr_lag_sign_convention"] = "" if step_group is None else str(step_group.attrs.get("xcorr_lag_sign_convention", ""))
            row["sync_predictor_source"] = "" if step_group is None else str(step_group.attrs.get("sync_predictor_source", ""))
            row["matched_sync_trace_alignment_status"] = "" if step_group is None else str(step_group.attrs.get("matched_sync_trace_alignment_status", ""))
            row["matched_no_step_h5_path"] = "" if step_group is None else str(step_group.attrs.get("matched_no_step_h5_path", ""))
            for key in STEP_SCALAR_COLUMNS:
                row[key] = _dataset_scalar(step_group, key)
                if not np.isfinite(row[key]) and key in STEP_SCALAR_ALIASES:
                    row[key] = _dataset_scalar(step_group, STEP_SCALAR_ALIASES[key])

            if np.isfinite(row.get("pre_step_sync_mean", np.nan)) and np.isfinite(row.get("pre_step_sync_sd", np.nan)):
                row["baseline_sync_source"] = "saved"
            elif recompute_baseline_sync_if_missing and step_group is not None:
                mean_val, sd_val, source = _recompute_baseline_sync_from_spikes(f, params, step_group)
                row["pre_step_sync_mean"] = mean_val
                row["pre_step_sync_sd"] = sd_val
                row["baseline_sync_mean"] = mean_val
                row["baseline_sync_sd"] = sd_val
                row["baseline_sync_source"] = source
            else:
                row["baseline_sync_source"] = "missing"
    except OSError as exc:
        row["step_response_status"] = f"h5_error_{type(exc).__name__}"
        row["baseline_sync_source"] = "h5_error"
        for key in STEP_SCALAR_COLUMNS:
            row.setdefault(key, np.nan)
    return row


def collect_step_response_repeat_table(
    batch_root: str | Path,
    *,
    recompute_baseline_sync_if_missing: bool = True,
    max_runs: int | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    batch_root = Path(batch_root)
    h5_paths = sorted(batch_root.rglob("simulation_output.h5"))
    if max_runs is not None:
        h5_paths = h5_paths[: int(max_runs)]
    if not h5_paths:
        raise FileNotFoundError(f"No simulation_output.h5 files found under {batch_root}")

    n_jobs = int(n_jobs)
    def _run_serial(paths):
        try:
            from tqdm.auto import tqdm
        except Exception:
            tqdm = lambda x, **_: x
        return [
            load_step_response_run_row(
                h5_path,
                recompute_baseline_sync_if_missing=recompute_baseline_sync_if_missing,
            )
            for h5_path in tqdm(paths, desc="Scanning step-response simulations")
        ]

    if n_jobs <= 1:
        rows = _run_serial(h5_paths)
    else:
        try:
            from joblib import Parallel, delayed

            rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
                delayed(load_step_response_run_row)(
                    h5_path,
                    recompute_baseline_sync_if_missing=recompute_baseline_sync_if_missing,
                )
                for h5_path in h5_paths
            )
        except Exception as exc:
            warnings.warn(
                f"Parallel extraction with n_jobs={n_jobs} failed ({type(exc).__name__}: {exc}); falling back to serial.",
                stacklevel=2,
            )
            rows = _run_serial(h5_paths)
    df = pd.DataFrame(rows)
    for key in ["parameter_set_id", "repeat_index", "paired_repeat_seed", "step_index"]:
        df[key] = pd.to_numeric(df[key], errors="coerce")
    if "is_zero_step_reference" in df:
        df["is_zero_step_reference"] = df["is_zero_step_reference"].fillna(False).astype(bool)
    for key in [*PARAMETER_COLUMNS, *STEP_SCALAR_COLUMNS]:
        if key in df:
            df[key] = pd.to_numeric(df[key], errors="coerce")
    return df


def _parameter_lookup_from_parameter_set_df(parameter_set_df: pd.DataFrame | None) -> dict:
    if parameter_set_df is None or len(parameter_set_df) == 0 or "parameter_set_id" not in parameter_set_df.columns:
        return {}
    baseline_cols = [col for col in PARAMETER_COLUMNS if col != "step_current_amplitude_nA" and col in parameter_set_df.columns]
    keep_cols = ["parameter_set_id", *baseline_cols]
    table = parameter_set_df[keep_cols].copy()
    table["parameter_set_id"] = pd.to_numeric(table["parameter_set_id"], errors="coerce")
    table = table.dropna(subset=["parameter_set_id"]).drop_duplicates("parameter_set_id")
    lookup = {}
    for _, row in table.iterrows():
        lookup[float(row["parameter_set_id"])] = {col: row[col] for col in baseline_cols}
    return lookup


def _parameter_lookup_from_priors(priors: dict | None) -> dict:
    if not isinstance(priors, dict):
        return {}
    parameter_sets_df = priors.get("parameter_sets_df")
    if parameter_sets_df is None:
        parameter_sets_df = priors.get("baseline_parameter_sets_df")
    return _parameter_lookup_from_parameter_set_df(parameter_sets_df if isinstance(parameter_sets_df, pd.DataFrame) else None)


def load_step_response_mean_trace_row(
    h5_path: str | Path,
    *,
    parameter_lookup: dict | None = None,
) -> dict:
    h5_path = Path(h5_path)
    parameter_set_dir = _find_parameter_set_dir(h5_path)
    parameter_lookup = dict(parameter_lookup or {})
    row = {
        "run_dir": str(h5_path.parent),
        "simulation_output_h5": str(h5_path),
        "mean_trace_h5": str(h5_path),
        "parameter_set_dir": "" if parameter_set_dir is None else str(parameter_set_dir),
        "parameter_set_id": _parse_parameter_set_id(parameter_set_dir),
        "repeat_index": np.nan,
        "requested_random_seed": np.nan,
        "paired_repeat_seed": np.nan,
        "step_index": np.nan,
        "is_zero_step_reference": False,
        "parameter_set_folder_name": "" if parameter_set_dir is None else parameter_set_dir.name,
        "repeat_folder_name": "averaged_repeats",
        "step_folder_name": h5_path.parent.name,
        "paired_reference_folder_name": "step_p00000_nA",
        "reactivity_data_source": "mean_trace",
        "baseline_sync_source": "mean_trace",
    }
    try:
        with h5py.File(h5_path, "r") as f:
            step_group = f.get("step_response_diagnostics")
            row["step_response_status"] = "" if step_group is None else str(step_group.attrs.get("status", ""))
            row["diagnostics_stage"] = "" if step_group is None else str(step_group.attrs.get("diagnostics_stage", ""))
            row["finalization_status"] = "" if step_group is None else str(step_group.attrs.get("finalization_status", ""))
            row["linear_slope_selection_status"] = "" if step_group is None else str(
                step_group.attrs.get(
                    "linear_slope_selection_status",
                    step_group.attrs.get("slope_selection_status", ""),
                )
            )
            row["response_shape_fit_status"] = "" if step_group is None else str(step_group.attrs.get("response_shape_fit_status", ""))
            row["xcorr_status"] = "" if step_group is None else str(step_group.attrs.get("xcorr_status", ""))
            row["xcorr_lag_sign_convention"] = "" if step_group is None else str(step_group.attrs.get("xcorr_lag_sign_convention", ""))
            row["sync_predictor_source"] = "" if step_group is None else str(step_group.attrs.get("sync_predictor_source", ""))
            row["matched_sync_trace_alignment_status"] = "" if step_group is None else str(step_group.attrs.get("matched_sync_trace_alignment_status", ""))
            row["matched_no_step_h5_path"] = "" if step_group is None else str(step_group.attrs.get("matched_no_step_h5_path", ""))
            for key in STEP_SCALAR_COLUMNS:
                row[key] = _dataset_scalar(step_group, key)
                if not np.isfinite(row[key]) and key in STEP_SCALAR_ALIASES:
                    row[key] = _dataset_scalar(step_group, STEP_SCALAR_ALIASES[key])
            row["parameter_set_id"] = row["parameter_set_id"] if np.isfinite(row["parameter_set_id"]) else _dataset_scalar(step_group, "parameter_set_id")
            row["step_current_amplitude_nA"] = _dataset_scalar(step_group, "step_current_amplitude_nA")
            row["is_zero_step_reference"] = bool(np.isfinite(row["step_current_amplitude_nA"]) and np.isclose(row["step_current_amplitude_nA"], 0.0))
            row["matched_step_effect_available"] = _dataset_scalar(step_group, "matched_step_effect_available")
            if not np.isfinite(row["matched_step_effect_available"]):
                row["matched_step_effect_available"] = 0.0 if row["is_zero_step_reference"] else 1.0
            meta_group = f.get("repeat_metadata")
            row["n_repeats_available"] = _dataset_scalar(meta_group, "n_repeats_available")
            row["n_repeats_used"] = _dataset_scalar(meta_group, "n_repeats_used")
    except OSError as exc:
        row["step_response_status"] = f"h5_error_{type(exc).__name__}"
        for key in STEP_SCALAR_COLUMNS:
            row.setdefault(key, np.nan)

    lookup_values = parameter_lookup.get(float(row["parameter_set_id"]), {}) if np.isfinite(row.get("parameter_set_id", np.nan)) else {}
    fallback_values = {
        "excitatory_input_baseline": row.get("baseline_tonic_input_nA", np.nan),
        "lf_target_sd": row.get("lf_input_amplitude_nA", np.nan),
        "alpha_target_sd_final": row.get("alpha_input_amplitude_nA", np.nan),
        "beta_target_sd_final": row.get("beta_input_amplitude_nA", np.nan),
        "alpha_beta_envelope_variability_scale": np.nan,
    }
    for key in PARAMETER_COLUMNS:
        if key == "step_current_amplitude_nA":
            row[key] = row.get("step_current_amplitude_nA", np.nan)
        else:
            row[key] = lookup_values.get(key, fallback_values.get(key, np.nan))
    return row


def collect_step_response_mean_trace_table(
    batch_root: str | Path,
    *,
    parameter_set_df: pd.DataFrame | None = None,
    priors: dict | None = None,
    max_runs: int | None = None,
) -> pd.DataFrame:
    batch_root = Path(batch_root)
    h5_paths = sorted(batch_root.rglob("averaged_repeats/step_*/averaged_step_response_diagnostics.h5"))
    if max_runs is not None:
        h5_paths = h5_paths[: int(max_runs)]
    if not h5_paths:
        raise FileNotFoundError(
            f"No averaged_step_response_diagnostics.h5 files found under {batch_root}. "
            "Run mean-trace finalization first."
        )
    lookup = _parameter_lookup_from_parameter_set_df(parameter_set_df)
    if not lookup:
        lookup = _parameter_lookup_from_priors(priors)
    rows = [load_step_response_mean_trace_row(path, parameter_lookup=lookup) for path in h5_paths]
    df = pd.DataFrame(rows)
    for key in ["parameter_set_id", "repeat_index", "paired_repeat_seed", "step_index", "n_repeats_available", "n_repeats_used"]:
        if key in df:
            df[key] = pd.to_numeric(df[key], errors="coerce")
    if "is_zero_step_reference" in df:
        df["is_zero_step_reference"] = df["is_zero_step_reference"].fillna(False).astype(bool)
    for key in [*PARAMETER_COLUMNS, *STEP_SCALAR_COLUMNS]:
        if key in df:
            df[key] = pd.to_numeric(df[key], errors="coerce")
    return df


def build_parameter_set_summary_table(repeat_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["parameter_set_id", *PARAMETER_COLUMNS]
    agg_cols = [col for col in STEP_SCALAR_COLUMNS if col in repeat_df.columns]
    grouped = repeat_df.groupby(group_cols, dropna=False)
    out = grouped[agg_cols].agg(["mean", "std", "count"]).reset_index()
    out.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col
        for col in out.columns
    ]
    out["n_runs"] = grouped.size().to_numpy()
    return out


def _format_bin_value_uA(value_nA):
    if not np.isfinite(value_nA):
        return "NA"
    return f"{float(value_nA) / 1000.0:g}"


def _uniform_parameter_names_from_priors(priors: dict | None) -> set[str]:
    if not isinstance(priors, dict):
        return set()
    specs = priors.get("normalized_baseline_parameter_uniform_specs")
    if not specs:
        specs = priors.get("baseline_parameter_uniform_specs")
    return set(specs) if isinstance(specs, dict) else set()


def _unique_parameter_values_for_binning(df: pd.DataFrame, value_col: str) -> np.ndarray:
    if "parameter_set_id" in df.columns:
        values = (
            df[["parameter_set_id", value_col]]
            .drop_duplicates("parameter_set_id")[value_col]
        )
    else:
        values = df[value_col]
    values = pd.to_numeric(values, errors="coerce")
    values = np.asarray(values[np.isfinite(values)].unique(), dtype=float)
    return np.array(sorted(values), dtype=float)


def _exact_axis_metadata(values: np.ndarray, parameter_name: str) -> dict:
    values = np.array(sorted(np.asarray(values, dtype=float)), dtype=float)
    return {
        "parameter": parameter_name,
        "mode": "exact",
        "values_nA": values.tolist(),
        "centers_nA": values.tolist(),
        "labels": [_format_bin_value_uA(value) for value in values],
        "edges_nA": None,
        "n_bins": int(values.size),
    }


def _quantile_axis_metadata(values: np.ndarray, parameter_name: str, n_bins: int) -> dict:
    values = np.array(sorted(np.asarray(values, dtype=float)), dtype=float)
    n_bins = int(max(1, min(int(n_bins), values.size)))
    if values.size <= 1 or n_bins <= 1:
        return _exact_axis_metadata(values, parameter_name)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if edges.size < 3:
        return _exact_axis_metadata(values, parameter_name)
    centers = 0.5 * (edges[:-1] + edges[1:])
    labels = [
        f"{_format_bin_value_uA(low)}-{_format_bin_value_uA(high)}"
        for low, high in zip(edges[:-1], edges[1:])
    ]
    return {
        "parameter": parameter_name,
        "mode": "quantile",
        "values_nA": values.tolist(),
        "centers_nA": centers.tolist(),
        "labels": labels,
        "edges_nA": edges.tolist(),
        "n_bins": int(edges.size - 1),
    }


def resolve_alpha_beta_heatmap_binning(
    repeat_df: pd.DataFrame,
    *,
    priors: dict | None = None,
    binning: str = "auto",
    quantile_bins: int = 4,
) -> dict:
    mode = str(binning).strip().lower()
    if mode not in {"auto", "exact", "quantile"}:
        raise ValueError("alpha/beta heatmap binning must be one of 'auto', 'exact', or 'quantile'")
    uniform_params = _uniform_parameter_names_from_priors(priors)
    axes = {}
    for parameter in ("alpha_target_sd_final", "beta_target_sd_final"):
        values = _unique_parameter_values_for_binning(repeat_df, parameter)
        if values.size == 0:
            raise ValueError(f"Cannot build heatmap binning; no finite values for {parameter}")
        axis_mode = mode
        if mode == "auto":
            looks_continuous = values.size > max(8, int(quantile_bins) * 2)
            axis_mode = "quantile" if parameter in uniform_params or looks_continuous else "exact"
        if axis_mode == "quantile":
            axis_meta = _quantile_axis_metadata(values, parameter, quantile_bins)
            if axis_meta["mode"] != "quantile":
                warnings.warn(
                    f"Falling back to exact heatmap axis for {parameter}; not enough unique values for quantile bins.",
                    stacklevel=2,
                )
        else:
            axis_meta = _exact_axis_metadata(values, parameter)
        axes[parameter] = axis_meta
    effective = "quantile" if any(axis["mode"] == "quantile" for axis in axes.values()) else "exact"
    if axes["alpha_target_sd_final"]["mode"] != axes["beta_target_sd_final"]["mode"]:
        effective = "mixed"
    slug = (
        "alpha_beta_"
        f"alpha_{axes['alpha_target_sd_final']['mode']}{axes['alpha_target_sd_final']['n_bins']}_"
        f"beta_{axes['beta_target_sd_final']['mode']}{axes['beta_target_sd_final']['n_bins']}"
    )
    return {
        "requested_binning": mode,
        "effective_binning": effective,
        "quantile_bins_requested": int(quantile_bins),
        "uniform_parameter_names": sorted(uniform_params),
        "axes": axes,
        "slug": slug,
    }


def apply_alpha_beta_heatmap_binning(repeat_df: pd.DataFrame, binning_metadata: dict) -> pd.DataFrame:
    out = repeat_df.copy()
    for prefix, parameter in (("alpha", "alpha_target_sd_final"), ("beta", "beta_target_sd_final")):
        axis = binning_metadata["axes"][parameter]
        values = pd.to_numeric(out[parameter], errors="coerce")
        if axis["mode"] == "quantile":
            edges = np.asarray(axis["edges_nA"], dtype=float)
            cut_edges = edges.copy()
            cut_edges[0] = np.nextafter(cut_edges[0], -np.inf)
            cut_edges[-1] = np.nextafter(cut_edges[-1], np.inf)
            bins = pd.cut(values, bins=cut_edges, labels=False, include_lowest=True)
            bins = pd.Series(bins, index=out.index, dtype="float")
        else:
            exact_values = np.asarray(axis["values_nA"], dtype=float)
            bins = pd.Series(np.nan, index=out.index, dtype=float)
            for idx, exact_value in enumerate(exact_values):
                bins[np.isclose(values, exact_value, atol=1e-9, rtol=0.0)] = float(idx)
        centers = np.asarray(axis["centers_nA"], dtype=float)
        labels = list(axis["labels"])
        out[f"{prefix}_heatmap_bin"] = bins
        out[f"{prefix}_heatmap_center_nA"] = bins.map(lambda item: centers[int(item)] if np.isfinite(item) else np.nan)
        out[f"{prefix}_heatmap_label"] = bins.map(lambda item: labels[int(item)] if np.isfinite(item) else "")
        out[f"{prefix}_heatmap_axis_mode"] = axis["mode"]
    out["alpha_beta_heatmap_binning"] = str(binning_metadata.get("effective_binning", "exact"))
    return out


def _add_heatmap_axis_fields(row: dict, group: pd.DataFrame) -> None:
    for prefix, parameter in (("alpha", "alpha_target_sd_final"), ("beta", "beta_target_sd_final")):
        row[f"{prefix}_heatmap_label"] = str(group[f"{prefix}_heatmap_label"].iloc[0])
        row[f"{prefix}_heatmap_center_nA"] = float(group[f"{prefix}_heatmap_center_nA"].iloc[0])
        row[f"{prefix}_heatmap_axis_mode"] = str(group[f"{prefix}_heatmap_axis_mode"].iloc[0])
        row[parameter] = row[f"{prefix}_heatmap_center_nA"]
    row["alpha_beta_heatmap_binning"] = str(group["alpha_beta_heatmap_binning"].iloc[0])


def build_baseline_sync_heatmap_table(
    repeat_df: pd.DataFrame,
    *,
    pool_step_amplitudes: bool = True,
    use_zero_step_reference_only: bool = False,
    alpha_beta_binning_metadata: dict | None = None,
) -> pd.DataFrame:
    if use_zero_step_reference_only and "is_zero_step_reference" in repeat_df.columns:
        repeat_df = repeat_df[repeat_df["is_zero_step_reference"].astype(bool)].copy()
    if alpha_beta_binning_metadata is None:
        alpha_beta_binning_metadata = resolve_alpha_beta_heatmap_binning(repeat_df, binning="exact")
    repeat_df = apply_alpha_beta_heatmap_binning(repeat_df, alpha_beta_binning_metadata)
    group_cols = ["lf_target_sd", "alpha_heatmap_bin", "beta_heatmap_bin"]
    if not pool_step_amplitudes:
        group_cols.append("step_current_amplitude_nA")
    grouped = repeat_df.groupby(group_cols, dropna=False)
    rows = []
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_cols, group_key))
        _add_heatmap_axis_fields(row, group)
        mean_values = pd.to_numeric(group["pre_step_sync_mean"], errors="coerce")
        sd_values = pd.to_numeric(group["pre_step_sync_sd"], errors="coerce")
        finite_mean = mean_values[np.isfinite(mean_values)]
        finite_sd = sd_values[np.isfinite(sd_values)]
        row.update({
            "n_runs": int(len(group)),
            "n_finite_pre_step_sync_mean": int(finite_mean.size),
            "n_finite_pre_step_sync_sd": int(finite_sd.size),
            "mean_pre_step_sync_mean": float(finite_mean.mean()) if finite_mean.size else np.nan,
            "sd_across_runs_pre_step_sync_mean": float(finite_mean.std(ddof=0)) if finite_mean.size else np.nan,
            "sem_pre_step_sync_mean": float(finite_mean.std(ddof=0) / np.sqrt(finite_mean.size)) if finite_mean.size else np.nan,
            "mean_pre_step_sync_sd": float(finite_sd.mean()) if finite_sd.size else np.nan,
            "sd_across_runs_pre_step_sync_sd": float(finite_sd.std(ddof=0)) if finite_sd.size else np.nan,
            "sem_pre_step_sync_sd": float(finite_sd.std(ddof=0) / np.sqrt(finite_sd.size)) if finite_sd.size else np.nan,
            "pooled_step_amplitudes": bool(pool_step_amplitudes),
            "used_zero_step_reference_only": bool(use_zero_step_reference_only),
        })
        mean_sync = row["mean_pre_step_sync_mean"]
        sd_sync = row["sd_across_runs_pre_step_sync_mean"]
        row["cv_across_runs_pre_step_sync_mean"] = (
            float(sd_sync / abs(mean_sync))
            if np.isfinite(mean_sync) and abs(mean_sync) > 1e-12 and np.isfinite(sd_sync)
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _numeric_series_with_fallback(df: pd.DataFrame, preferred: str, fallback: str | None = None) -> pd.Series:
    if preferred in df.columns:
        values = pd.to_numeric(df[preferred], errors="coerce")
    else:
        values = pd.Series(np.nan, index=df.index, dtype=float)
    if fallback is not None and fallback in df.columns:
        fallback_values = pd.to_numeric(df[fallback], errors="coerce")
        values = values.where(np.isfinite(values), fallback_values)
    return values


def build_post_step_net_input_sync_heatmap_table(
    repeat_df: pd.DataFrame,
    *,
    alpha_beta_binning_metadata: dict | None = None,
) -> pd.DataFrame:
    required = [
        "lf_target_sd",
        "alpha_target_sd_final",
        "beta_target_sd_final",
        "step_current_amplitude_nA",
    ]
    missing = [col for col in required if col not in repeat_df.columns]
    if missing:
        raise ValueError(f"Cannot build post-step heatmap table; missing columns: {missing}")

    if alpha_beta_binning_metadata is None:
        alpha_beta_binning_metadata = resolve_alpha_beta_heatmap_binning(repeat_df, binning="exact")
    repeat_df = apply_alpha_beta_heatmap_binning(repeat_df, alpha_beta_binning_metadata)
    baseline_input = _numeric_series_with_fallback(
        repeat_df,
        "excitatory_input_baseline",
        "baseline_tonic_input_nA",
    )
    step_input = pd.to_numeric(repeat_df["step_current_amplitude_nA"], errors="coerce")
    repeat_df["post_step_net_input_nA"] = baseline_input + step_input
    repeat_df["post_step_net_input_uA"] = repeat_df["post_step_net_input_nA"] / 1000.0
    repeat_df["_post_step_sync_mean_for_heatmap"] = _numeric_series_with_fallback(
        repeat_df,
        "post_step_sync_mean_actual_step",
        "post_step_sync_mean",
    )
    repeat_df["_post_step_sync_sd_for_heatmap"] = _numeric_series_with_fallback(
        repeat_df,
        "post_step_sync_sd_actual_step",
        "post_step_sync_sd",
    )
    repeat_df["_post_step_mean_fr_for_heatmap"] = _numeric_series_with_fallback(
        repeat_df,
        "post_step_mean_firing_rate_hz",
    )

    group_cols = [
        "lf_target_sd",
        "post_step_net_input_nA",
        "post_step_net_input_uA",
        "alpha_heatmap_bin",
        "beta_heatmap_bin",
    ]
    grouped = repeat_df.groupby(group_cols, dropna=False)
    rows = []
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_cols, group_key))
        _add_heatmap_axis_fields(row, group)
        mean_values = pd.to_numeric(group["_post_step_sync_mean_for_heatmap"], errors="coerce")
        sd_values = pd.to_numeric(group["_post_step_sync_sd_for_heatmap"], errors="coerce")
        fr_values = pd.to_numeric(group["_post_step_mean_fr_for_heatmap"], errors="coerce")
        finite_mean = mean_values[np.isfinite(mean_values)]
        finite_sd = sd_values[np.isfinite(sd_values)]
        finite_fr = fr_values[np.isfinite(fr_values)]
        row.update({
            "n_runs": int(len(group)),
            "n_finite_post_step_sync_mean": int(finite_mean.size),
            "n_finite_post_step_sync_sd": int(finite_sd.size),
            "n_finite_post_step_mean_firing_rate_hz": int(finite_fr.size),
            "mean_post_step_sync_mean": float(finite_mean.mean()) if finite_mean.size else np.nan,
            "sd_across_runs_post_step_sync_mean": float(finite_mean.std(ddof=0)) if finite_mean.size else np.nan,
            "sem_post_step_sync_mean": float(finite_mean.std(ddof=0) / np.sqrt(finite_mean.size)) if finite_mean.size else np.nan,
            "mean_post_step_sync_sd": float(finite_sd.mean()) if finite_sd.size else np.nan,
            "sd_across_runs_post_step_sync_sd": float(finite_sd.std(ddof=0)) if finite_sd.size else np.nan,
            "sem_post_step_sync_sd": float(finite_sd.std(ddof=0) / np.sqrt(finite_sd.size)) if finite_sd.size else np.nan,
            "mean_post_step_mean_firing_rate_hz": float(finite_fr.mean()) if finite_fr.size else np.nan,
            "sd_across_runs_post_step_mean_firing_rate_hz": float(finite_fr.std(ddof=0)) if finite_fr.size else np.nan,
        })
        mean_sync = row["mean_post_step_sync_mean"]
        sd_sync = row["sd_across_runs_post_step_sync_mean"]
        row["cv_across_runs_post_step_sync_mean"] = (
            float(sd_sync / abs(mean_sync))
            if np.isfinite(mean_sync) and abs(mean_sync) > 1e-12 and np.isfinite(sd_sync)
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _heatmap_binning_slug(metadata: dict | None) -> str:
    if not metadata:
        return "alpha_beta_exact"
    return _slugify_label(str(metadata.get("slug", "alpha_beta_exact")))


def _jsonable_binning_metadata(metadata: dict | None) -> dict:
    if not metadata:
        return {}
    return json.loads(json.dumps(metadata, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))


def save_heatmap_binning_metadata(output_dir: str | Path, metadata: dict | None, filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable_binning_metadata(metadata), f, indent=2)
    return path


def save_baseline_heatmap_table(output_dir: str | Path, heatmap_df: pd.DataFrame, binning_metadata: dict | None = None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _heatmap_binning_slug(binning_metadata)
    pkl_path = output_dir / f"baseline_sync_heatmap_table_{slug}.pkl"
    csv_path = output_dir / f"baseline_sync_heatmap_table_{slug}.csv"
    heatmap_df.to_pickle(pkl_path)
    heatmap_df.to_csv(csv_path, index=False)
    legacy_pkl_path = output_dir / "baseline_sync_heatmap_table.pkl"
    legacy_csv_path = output_dir / "baseline_sync_heatmap_table.csv"
    heatmap_df.to_pickle(legacy_pkl_path)
    heatmap_df.to_csv(legacy_csv_path, index=False)
    metadata_path = save_heatmap_binning_metadata(
        output_dir,
        binning_metadata,
        "baseline_sync_heatmap_binning_metadata.json",
    )
    return {
        "pkl_path": pkl_path,
        "csv_path": csv_path,
        "legacy_pkl_path": legacy_pkl_path,
        "legacy_csv_path": legacy_csv_path,
        "metadata_path": metadata_path,
    }


def save_post_step_net_input_heatmap_table(
    output_dir: str | Path,
    heatmap_df: pd.DataFrame,
    binning_metadata: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _heatmap_binning_slug(binning_metadata)
    pkl_path = output_dir / f"post_step_net_input_sync_heatmap_table_{slug}.pkl"
    csv_path = output_dir / f"post_step_net_input_sync_heatmap_table_{slug}.csv"
    heatmap_df.to_pickle(pkl_path)
    heatmap_df.to_csv(csv_path, index=False)
    legacy_pkl_path = output_dir / "post_step_net_input_sync_heatmap_table.pkl"
    legacy_csv_path = output_dir / "post_step_net_input_sync_heatmap_table.csv"
    heatmap_df.to_pickle(legacy_pkl_path)
    heatmap_df.to_csv(legacy_csv_path, index=False)
    metadata_path = save_heatmap_binning_metadata(
        output_dir,
        binning_metadata,
        "post_step_net_input_sync_heatmap_binning_metadata.json",
    )
    return {
        "pkl_path": pkl_path,
        "csv_path": csv_path,
        "legacy_pkl_path": legacy_pkl_path,
        "legacy_csv_path": legacy_csv_path,
        "metadata_path": metadata_path,
    }


def load_post_step_net_input_heatmap_table(output_dir: str | Path, binning_metadata: dict | None = None) -> pd.DataFrame | None:
    if binning_metadata is not None:
        path = Path(output_dir) / f"post_step_net_input_sync_heatmap_table_{_heatmap_binning_slug(binning_metadata)}.pkl"
    else:
        path = Path(output_dir) / "post_step_net_input_sync_heatmap_table.pkl"
    if not path.exists():
        return None
    return pd.read_pickle(path)


def save_tables(output_dir: str | Path, repeat_df: pd.DataFrame, parameter_set_df: pd.DataFrame, heatmap_df: pd.DataFrame) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (
        ("step_response_repeat_table_raw", repeat_df),
        ("step_response_parameter_set_table", parameter_set_df),
    ):
        df.to_pickle(output_dir / f"{name}.pkl")
        df.to_csv(output_dir / f"{name}.csv", index=False)
    if heatmap_df is not None:
        heatmap_df.to_pickle(output_dir / "baseline_sync_heatmap_table.pkl")
        heatmap_df.to_csv(output_dir / "baseline_sync_heatmap_table.csv", index=False)


def load_cached_tables(output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    output_dir = Path(output_dir)
    core_paths = [
        output_dir / "step_response_repeat_table_raw.pkl",
        output_dir / "step_response_parameter_set_table.pkl",
    ]
    if not all(path.exists() for path in core_paths):
        return None
    heatmap_path = output_dir / "baseline_sync_heatmap_table.pkl"
    heatmap_df = pd.read_pickle(heatmap_path) if heatmap_path.exists() else None
    return pd.read_pickle(core_paths[0]), pd.read_pickle(core_paths[1]), heatmap_df


def build_or_load_step_response_tables(
    batch_root: str | Path,
    *,
    output_subdir_name: str = "step_response_summary",
    force_rebuild: bool = False,
    recompute_baseline_sync_if_missing: bool = True,
    pool_step_amplitudes: bool = True,
    use_zero_step_reference_only: bool = False,
    alpha_beta_heatmap_binning: str = "auto",
    alpha_beta_heatmap_quantile_bins: int = 4,
    max_runs: int | None = None,
    n_jobs: int = 1,
) -> dict:
    batch_root = Path(batch_root)
    output_dir = batch_root / output_subdir_name
    if max_runs is not None:
        output_dir = output_dir / f"_debug_first_{int(max_runs)}_runs"
    cached = None if force_rebuild else load_cached_tables(output_dir)
    priors = load_step_response_priors(batch_root)
    if cached is None:
        repeat_df = collect_step_response_repeat_table(
            batch_root,
            recompute_baseline_sync_if_missing=recompute_baseline_sync_if_missing,
            max_runs=max_runs,
            n_jobs=n_jobs,
        )
        parameter_set_df = build_parameter_set_summary_table(repeat_df)
        save_tables(output_dir, repeat_df, parameter_set_df, None)
        cache_source = "rebuilt"
    else:
        repeat_df, parameter_set_df, _cached_heatmap_df = cached
        cache_source = "cached"
    binning_metadata = resolve_alpha_beta_heatmap_binning(
        repeat_df,
        priors=priors,
        binning=alpha_beta_heatmap_binning,
        quantile_bins=alpha_beta_heatmap_quantile_bins,
    )
    heatmap_df = build_baseline_sync_heatmap_table(
        repeat_df,
        pool_step_amplitudes=pool_step_amplitudes,
        use_zero_step_reference_only=use_zero_step_reference_only,
        alpha_beta_binning_metadata=binning_metadata,
    )
    baseline_heatmap_paths = save_baseline_heatmap_table(output_dir, heatmap_df, binning_metadata)
    return {
        "batch_root": batch_root,
        "output_dir": output_dir,
        "priors": priors,
        "repeat_df": repeat_df,
        "parameter_set_df": parameter_set_df,
        "heatmap_df": heatmap_df,
        "alpha_beta_binning_metadata": binning_metadata,
        "baseline_heatmap_paths": baseline_heatmap_paths,
        "cache_source": cache_source,
    }


def _format_input_value(value, scale=1000.0):
    if not np.isfinite(value):
        return ""
    return f"{value / scale:g}"


def _format_uA_value(value):
    if not np.isfinite(value):
        return ""
    return f"{value:g}"


def _input_token_uA(value):
    if not np.isfinite(value):
        return "nan"
    rounded = int(round(float(value) * 1000.0))
    sign = "p" if rounded >= 0 else "m"
    return f"{sign}{abs(rounded):05d}"


def _resolve_color_limits(values, vmin=None, vmax=None, percentiles=None):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    percentile_vmin = None
    percentile_vmax = None
    if percentiles is not None:
        if len(percentiles) != 2:
            raise ValueError("percentiles must contain exactly two values, e.g. (2, 98)")
        pct_low, pct_high = float(percentiles[0]), float(percentiles[1])
        if not (0.0 <= pct_low < pct_high <= 100.0):
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
        if finite.size:
            percentile_vmin, percentile_vmax = np.percentile(finite, [pct_low, pct_high])
    resolved_vmin = (
        float(vmin)
        if vmin is not None
        else (float(percentile_vmin) if percentile_vmin is not None else (float(np.min(finite)) if finite.size else 0.0))
    )
    resolved_vmax = (
        float(vmax)
        if vmax is not None
        else (float(percentile_vmax) if percentile_vmax is not None else (float(np.max(finite)) if finite.size else 1.0))
    )
    if not np.isfinite(resolved_vmin):
        resolved_vmin = 0.0
    if not np.isfinite(resolved_vmax):
        resolved_vmax = 1.0
    if resolved_vmax <= resolved_vmin:
        pad = max(1e-6, 0.05 * abs(resolved_vmin) if resolved_vmin != 0 else 1e-3)
        resolved_vmin -= pad
        resolved_vmax += pad
    return resolved_vmin, resolved_vmax


def _annotation_color_for_value(value, *, cmap, norm):
    rgba = cmap(norm(float(value)))
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance >= 0.55 else "white"


def _heatmap_axis_info(heatmap_df: pd.DataFrame) -> dict:
    if {"alpha_heatmap_bin", "beta_heatmap_bin", "alpha_heatmap_label", "beta_heatmap_label"}.issubset(heatmap_df.columns):
        alpha_values = np.array(sorted(pd.to_numeric(heatmap_df["alpha_heatmap_bin"], errors="coerce").dropna().unique()), dtype=float)
        beta_values = np.array(sorted(pd.to_numeric(heatmap_df["beta_heatmap_bin"], errors="coerce").dropna().unique()), dtype=float)
        alpha_label_map = (
            heatmap_df.dropna(subset=["alpha_heatmap_bin"])
            .drop_duplicates("alpha_heatmap_bin")
            .set_index("alpha_heatmap_bin")["alpha_heatmap_label"]
            .to_dict()
        )
        beta_label_map = (
            heatmap_df.dropna(subset=["beta_heatmap_bin"])
            .drop_duplicates("beta_heatmap_bin")
            .set_index("beta_heatmap_bin")["beta_heatmap_label"]
            .to_dict()
        )
        alpha_mode = str(heatmap_df.get("alpha_heatmap_axis_mode", pd.Series(["exact"])).dropna().iloc[0])
        beta_mode = str(heatmap_df.get("beta_heatmap_axis_mode", pd.Series(["exact"])).dropna().iloc[0])
        return {
            "alpha_col": "alpha_heatmap_bin",
            "beta_col": "beta_heatmap_bin",
            "alpha_values": alpha_values,
            "beta_values": beta_values,
            "alpha_labels": [str(alpha_label_map.get(value, "")) for value in alpha_values],
            "beta_labels": [str(beta_label_map.get(value, "")) for value in beta_values],
            "alpha_axis_label": "alpha_target_sd_final bin (uA)" if alpha_mode == "quantile" else "alpha_target_sd_final (uA)",
            "beta_axis_label": "beta_target_sd_final bin (uA)" if beta_mode == "quantile" else "beta_target_sd_final (uA)",
        }
    alpha_values = np.array(sorted(pd.to_numeric(heatmap_df["alpha_target_sd_final"], errors="coerce").dropna().unique()), dtype=float)
    beta_values = np.array(sorted(pd.to_numeric(heatmap_df["beta_target_sd_final"], errors="coerce").dropna().unique()), dtype=float)
    return {
        "alpha_col": "alpha_target_sd_final",
        "beta_col": "beta_target_sd_final",
        "alpha_values": alpha_values,
        "beta_values": beta_values,
        "alpha_labels": [_format_input_value(value) for value in alpha_values],
        "beta_labels": [_format_input_value(value) for value in beta_values],
        "alpha_axis_label": "alpha_target_sd_final (uA)",
        "beta_axis_label": "beta_target_sd_final (uA)",
    }


def plot_baseline_sync_heatmaps(
    heatmap_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    filename_base: str = "baseline_sync_alpha_beta_by_lf_heatmaps",
    figsize_per_row=(9.5, 3.2),
    mean_cmap="viridis",
    sd_cmap="magma",
    across_run_cv_cmap="cividis",
    mean_vmin=None,
    mean_vmax=None,
    sd_vmin=None,
    sd_vmax=None,
    across_run_cv_vmin=None,
    across_run_cv_vmax=None,
    mean_color_percentiles=None,
    sd_color_percentiles=None,
    across_run_cv_color_percentiles=None,
    annotate=True,
    annotation_fmt=".3f",
    show_cell_n=True,
    annotation_fontsize=7.0,
    title_fontsize=10.0,
    axis_label_fontsize=9.0,
    tick_label_fontsize=8.0,
    colorbar_label_fontsize=8.0,
    suptitle_fontsize=13.0,
    title="Baseline synchrony across alpha/beta input amplitudes",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_df = heatmap_df.copy()
    if "cv_across_runs_pre_step_sync_mean" not in heatmap_df.columns:
        denominator = np.abs(pd.to_numeric(heatmap_df["mean_pre_step_sync_mean"], errors="coerce"))
        numerator = pd.to_numeric(heatmap_df["sd_across_runs_pre_step_sync_mean"], errors="coerce")
        heatmap_df["cv_across_runs_pre_step_sync_mean"] = np.where(
            np.isfinite(denominator) & (denominator > 1e-12) & np.isfinite(numerator),
            numerator / denominator,
            np.nan,
        )
    lf_values = np.array(sorted(pd.to_numeric(heatmap_df["lf_target_sd"], errors="coerce").dropna().unique()), dtype=float)
    axis_info = _heatmap_axis_info(heatmap_df)
    alpha_values = axis_info["alpha_values"]
    beta_values = axis_info["beta_values"]
    if lf_values.size == 0 or alpha_values.size == 0 or beta_values.size == 0:
        raise ValueError("Heatmap table does not contain finite LF/alpha/beta grid values")

    n_rows = int(lf_values.size)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(float(figsize_per_row[0]), float(figsize_per_row[1]) * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=suptitle_fontsize)

    cmap_mean = plt.get_cmap(mean_cmap).copy()
    cmap_sd = plt.get_cmap(sd_cmap).copy()
    cmap_across_run_cv = plt.get_cmap(across_run_cv_cmap).copy()
    cmap_mean.set_bad("#eeeeee")
    cmap_sd.set_bad("#eeeeee")
    cmap_across_run_cv.set_bad("#eeeeee")

    mean_vmin, mean_vmax = _resolve_color_limits(
        heatmap_df["mean_pre_step_sync_mean"],
        mean_vmin,
        mean_vmax,
        mean_color_percentiles,
    )
    sd_vmin, sd_vmax = _resolve_color_limits(
        heatmap_df["mean_pre_step_sync_sd"],
        sd_vmin,
        sd_vmax,
        sd_color_percentiles,
    )
    across_run_cv_vmin, across_run_cv_vmax = _resolve_color_limits(
        heatmap_df["cv_across_runs_pre_step_sync_mean"],
        across_run_cv_vmin,
        across_run_cv_vmax,
        across_run_cv_color_percentiles,
    )
    plot_specs = [
        ("mean_pre_step_sync_mean", "Mean of baseline mean synchrony", cmap_mean, mean_vmin, mean_vmax),
        ("mean_pre_step_sync_sd", "Mean of baseline SD synchrony", cmap_sd, sd_vmin, sd_vmax),
        (
            "cv_across_runs_pre_step_sync_mean",
            "Across-repeats CV of baseline mean synchrony",
            cmap_across_run_cv,
            across_run_cv_vmin,
            across_run_cv_vmax,
        ),
    ]
    for row_i, lf_value in enumerate(lf_values):
        sub = heatmap_df[np.isclose(pd.to_numeric(heatmap_df["lf_target_sd"], errors="coerce"), lf_value)]
        for col_i, (value_col, col_title, cmap, vmin, vmax) in enumerate(plot_specs):
            ax = axes[row_i, col_i]
            pivot = (
                sub.pivot_table(
                    index=axis_info["alpha_col"],
                    columns=axis_info["beta_col"],
                    values=value_col,
                    aggfunc="mean",
                )
                .reindex(index=alpha_values, columns=beta_values)
            )
            n_finite_pivot = (
                sub.pivot_table(
                    index=axis_info["alpha_col"],
                    columns=axis_info["beta_col"],
                    values="n_finite_pre_step_sync_mean",
                    aggfunc="mean",
                )
                .reindex(index=alpha_values, columns=beta_values)
            )
            n_total_pivot = (
                sub.pivot_table(
                    index=axis_info["alpha_col"],
                    columns=axis_info["beta_col"],
                    values="n_runs",
                    aggfunc="mean",
                )
                .reindex(index=alpha_values, columns=beta_values)
            )
            matrix = np.asarray(pivot, dtype=float)
            n_finite_matrix = np.asarray(n_finite_pivot, dtype=float)
            n_total_matrix = np.asarray(n_total_pivot, dtype=float)
            image = ax.imshow(
                np.ma.masked_invalid(matrix),
                origin="lower",
                aspect="equal",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            norm = Normalize(vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(beta_values.size))
            ax.set_xticklabels(axis_info["beta_labels"], rotation=45 if axis_info["beta_col"] == "beta_heatmap_bin" else 0, ha="right")
            ax.set_yticks(np.arange(alpha_values.size))
            ax.set_yticklabels(axis_info["alpha_labels"])
            ax.tick_params(axis="both", labelsize=tick_label_fontsize)
            ax.set_xlabel(axis_info["beta_axis_label"], fontsize=axis_label_fontsize)
            ax.set_ylabel(axis_info["alpha_axis_label"], fontsize=axis_label_fontsize)
            ax.set_title(f"{col_title}\nLF={lf_value / 1000:g} uA", fontsize=title_fontsize)
            if annotate:
                for y_i in range(alpha_values.size):
                    for x_i in range(beta_values.size):
                        value = matrix[y_i, x_i]
                        if np.isfinite(value):
                            label = format(float(value), annotation_fmt)
                            if show_cell_n:
                                n_finite = n_finite_matrix[y_i, x_i]
                                n_total = n_total_matrix[y_i, x_i]
                                if np.isfinite(n_finite) and np.isfinite(n_total):
                                    n_finite_int = int(round(float(n_finite)))
                                    n_total_int = int(round(float(n_total)))
                                    n_label = f"n={n_finite_int}" if n_finite_int == n_total_int else f"n={n_finite_int}/{n_total_int}"
                                    label = f"{label}\n{n_label}"
                            ax.text(
                                x_i,
                                y_i,
                                label,
                                ha="center",
                                va="center",
                                fontsize=annotation_fontsize,
                                color=_annotation_color_for_value(value, cmap=cmap, norm=norm),
                            )
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=tick_label_fontsize)
            cbar.ax.set_ylabel(value_col, rotation=90, fontsize=colorbar_label_fontsize)

    png_path = output_dir / f"{filename_base}.png"
    pdf_path = output_dir / f"{filename_base}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"figure": fig, "png_path": png_path, "pdf_path": pdf_path}


def plot_post_step_net_input_sync_heatmaps(
    heatmap_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    filename_base: str = "post_step_sync_alpha_beta_by_net_input",
    figsize_per_row=(10.5, 3.2),
    mean_cmap="viridis",
    sd_cmap="magma",
    across_run_cv_cmap="cividis",
    mean_vmin=None,
    mean_vmax=None,
    sd_vmin=None,
    sd_vmax=None,
    across_run_cv_vmin=None,
    across_run_cv_vmax=None,
    mean_color_percentiles=None,
    sd_color_percentiles=None,
    across_run_cv_color_percentiles=None,
    annotate=True,
    annotation_fmt=".3f",
    show_cell_n=True,
    annotation_fontsize=7.0,
    title_fontsize=10.0,
    axis_label_fontsize=9.0,
    tick_label_fontsize=8.0,
    colorbar_label_fontsize=8.0,
    suptitle_fontsize=13.0,
    title="Post-step synchrony across alpha/beta input amplitudes",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_df = heatmap_df.copy()
    if "cv_across_runs_post_step_sync_mean" not in heatmap_df.columns:
        denominator = np.abs(pd.to_numeric(heatmap_df["mean_post_step_sync_mean"], errors="coerce"))
        numerator = pd.to_numeric(heatmap_df["sd_across_runs_post_step_sync_mean"], errors="coerce")
        heatmap_df["cv_across_runs_post_step_sync_mean"] = np.where(
            np.isfinite(denominator) & (denominator > 1e-12) & np.isfinite(numerator),
            numerator / denominator,
            np.nan,
        )
    lf_values = np.array(sorted(pd.to_numeric(heatmap_df["lf_target_sd"], errors="coerce").dropna().unique()), dtype=float)
    net_values = np.array(sorted(pd.to_numeric(heatmap_df["post_step_net_input_uA"], errors="coerce").dropna().unique()), dtype=float)
    axis_info = _heatmap_axis_info(heatmap_df)
    alpha_values = axis_info["alpha_values"]
    beta_values = axis_info["beta_values"]
    if lf_values.size == 0 or net_values.size == 0 or alpha_values.size == 0 or beta_values.size == 0:
        raise ValueError("Heatmap table does not contain finite LF/net-input/alpha/beta grid values")

    cmap_mean = plt.get_cmap(mean_cmap).copy()
    cmap_sd = plt.get_cmap(sd_cmap).copy()
    cmap_across_run_cv = plt.get_cmap(across_run_cv_cmap).copy()
    cmap_mean.set_bad("#eeeeee")
    cmap_sd.set_bad("#eeeeee")
    cmap_across_run_cv.set_bad("#eeeeee")

    mean_vmin, mean_vmax = _resolve_color_limits(
        heatmap_df["mean_post_step_sync_mean"],
        mean_vmin,
        mean_vmax,
        mean_color_percentiles,
    )
    sd_vmin, sd_vmax = _resolve_color_limits(
        heatmap_df["mean_post_step_sync_sd"],
        sd_vmin,
        sd_vmax,
        sd_color_percentiles,
    )
    across_run_cv_vmin, across_run_cv_vmax = _resolve_color_limits(
        heatmap_df["cv_across_runs_post_step_sync_mean"],
        across_run_cv_vmin,
        across_run_cv_vmax,
        across_run_cv_color_percentiles,
    )
    plot_specs = [
        ("mean_post_step_sync_mean", "Mean of post-step mean synchrony", cmap_mean, mean_vmin, mean_vmax),
        ("mean_post_step_sync_sd", "Mean of post-step SD synchrony", cmap_sd, sd_vmin, sd_vmax),
        (
            "cv_across_runs_post_step_sync_mean",
            "Across-repeats CV of post-step mean synchrony",
            cmap_across_run_cv,
            across_run_cv_vmin,
            across_run_cv_vmax,
        ),
    ]

    figures = []
    for lf_value in lf_values:
        lf_sub = heatmap_df[np.isclose(pd.to_numeric(heatmap_df["lf_target_sd"], errors="coerce"), lf_value)]
        lf_net_values = np.array(
            sorted(pd.to_numeric(lf_sub["post_step_net_input_uA"], errors="coerce").dropna().unique()),
            dtype=float,
        )
        if lf_net_values.size == 0:
            continue
        n_rows = int(lf_net_values.size)
        fig, axes = plt.subplots(
            n_rows,
            3,
            figsize=(float(figsize_per_row[0]), float(figsize_per_row[1]) * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        fig.suptitle(f"{title}\nLF={lf_value / 1000:g} uA", fontsize=suptitle_fontsize)
        for row_i, net_value in enumerate(lf_net_values):
            sub = lf_sub[np.isclose(pd.to_numeric(lf_sub["post_step_net_input_uA"], errors="coerce"), net_value)]
            fr_values = pd.to_numeric(sub["mean_post_step_mean_firing_rate_hz"], errors="coerce")
            finite_fr = fr_values[np.isfinite(fr_values)]
            mean_fr = float(finite_fr.mean()) if finite_fr.size else np.nan
            row_label = f"net={_format_uA_value(net_value)} uA"
            if np.isfinite(mean_fr):
                row_label += f" | FR={mean_fr:.2f} Hz"
            for col_i, (value_col, col_title, cmap, vmin, vmax) in enumerate(plot_specs):
                ax = axes[row_i, col_i]
                pivot = (
                    sub.pivot_table(
                        index=axis_info["alpha_col"],
                        columns=axis_info["beta_col"],
                        values=value_col,
                        aggfunc="mean",
                    )
                    .reindex(index=alpha_values, columns=beta_values)
                )
                n_finite_pivot = (
                    sub.pivot_table(
                        index=axis_info["alpha_col"],
                        columns=axis_info["beta_col"],
                        values="n_finite_post_step_sync_mean",
                        aggfunc="mean",
                    )
                    .reindex(index=alpha_values, columns=beta_values)
                )
                n_total_pivot = (
                    sub.pivot_table(
                        index=axis_info["alpha_col"],
                        columns=axis_info["beta_col"],
                        values="n_runs",
                        aggfunc="mean",
                    )
                    .reindex(index=alpha_values, columns=beta_values)
                )
                matrix = np.asarray(pivot, dtype=float)
                n_finite_matrix = np.asarray(n_finite_pivot, dtype=float)
                n_total_matrix = np.asarray(n_total_pivot, dtype=float)
                image = ax.imshow(
                    np.ma.masked_invalid(matrix),
                    origin="lower",
                    aspect="equal",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                norm = Normalize(vmin=vmin, vmax=vmax)
                ax.set_xticks(np.arange(beta_values.size))
                ax.set_xticklabels(axis_info["beta_labels"], rotation=45 if axis_info["beta_col"] == "beta_heatmap_bin" else 0, ha="right")
                ax.set_yticks(np.arange(alpha_values.size))
                ax.set_yticklabels(axis_info["alpha_labels"])
                ax.tick_params(axis="both", labelsize=tick_label_fontsize)
                ax.set_xlabel(axis_info["beta_axis_label"], fontsize=axis_label_fontsize)
                ax.set_ylabel(axis_info["alpha_axis_label"], fontsize=axis_label_fontsize)
                ax.set_title(f"{col_title}\n{row_label}", fontsize=title_fontsize)
                if annotate:
                    for y_i in range(alpha_values.size):
                        for x_i in range(beta_values.size):
                            value = matrix[y_i, x_i]
                            if np.isfinite(value):
                                label = format(float(value), annotation_fmt)
                                if show_cell_n:
                                    n_finite = n_finite_matrix[y_i, x_i]
                                    n_total = n_total_matrix[y_i, x_i]
                                    if np.isfinite(n_finite) and np.isfinite(n_total):
                                        n_finite_int = int(round(float(n_finite)))
                                        n_total_int = int(round(float(n_total)))
                                        n_label = f"n={n_finite_int}" if n_finite_int == n_total_int else f"n={n_finite_int}/{n_total_int}"
                                        label = f"{label}\n{n_label}"
                                ax.text(
                                    x_i,
                                    y_i,
                                    label,
                                    ha="center",
                                    va="center",
                                    fontsize=annotation_fontsize,
                                    color=_annotation_color_for_value(value, cmap=cmap, norm=norm),
                                )
                cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=tick_label_fontsize)
                cbar.ax.set_ylabel(value_col, rotation=90, fontsize=colorbar_label_fontsize)

        lf_token = _input_token_uA(float(lf_value) / 1000.0)
        png_path = output_dir / f"{filename_base}_lf_{lf_token}uA.png"
        pdf_path = output_dir / f"{filename_base}_lf_{lf_token}uA.pdf"
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        figures.append({"lf_target_sd": float(lf_value), "png_path": png_path, "pdf_path": pdf_path})
    return {"figures": figures}


def _slugify_label(value: str) -> str:
    text = str(value).strip().lower()
    keep = []
    previous_was_sep = False
    for char in text:
        if char.isalnum():
            keep.append(char)
            previous_was_sep = False
        elif not previous_was_sep:
            keep.append("_")
            previous_was_sep = True
    return "".join(keep).strip("_") or "value"


def _finite_pearson_summary(x, y) -> tuple[float, float, int]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    n = int(mask.sum())
    if n < 2:
        return np.nan, np.nan, n
    x_valid = x_arr[mask]
    y_valid = y_arr[mask]
    if np.nanstd(x_valid) <= 0 or np.nanstd(y_valid) <= 0:
        return np.nan, np.nan, n
    r = float(np.corrcoef(x_valid, y_valid)[0, 1])
    return r, float(r * r), n


def _format_corr_text(x, y) -> str:
    r, r2, n = _finite_pearson_summary(x, y)
    if np.isfinite(r):
        return f"r={r:.2f}\nR2={r2:.2f}\nn={n}"
    return f"r=NA\nR2=NA\nn={n}"


def _split_label_units(label: str) -> str:
    label = str(label)
    if "\n" in label:
        return label
    if label.endswith(")") and "(" in label:
        prefix, suffix = label.rsplit("(", 1)
        return f"{prefix.strip()}\n({suffix}"
    return label


def _draw_regression_line(
    ax,
    x,
    y,
    *,
    color="black",
    lw=1.5,
    alpha=0.45,
    ls="-",
    zorder=5,
):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 2:
        return None
    x_valid = x_arr[mask]
    y_valid = y_arr[mask]
    if np.nanstd(x_valid) <= 0 or np.nanstd(y_valid) <= 0:
        return None
    slope, intercept = np.polyfit(x_valid, y_valid, deg=1)
    x_line = np.linspace(float(np.min(x_valid)), float(np.max(x_valid)), 100)
    y_line = intercept + slope * x_line
    return ax.plot(x_line, y_line, color=color, lw=lw, alpha=alpha, ls=ls, zorder=zorder)


def _padded_limits(values, limits=None, pad_fraction=0.05):
    if limits is not None:
        return tuple(limits)
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (0.0, 1.0)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax <= vmin:
        pad = max(1e-6, abs(vmin) * pad_fraction if vmin != 0 else 1.0)
    else:
        pad = max(1e-9, (vmax - vmin) * pad_fraction)
    return (vmin - pad, vmax + pad)


def resolve_step_reactivity_metric(repeat_df: pd.DataFrame, reactivity_metric: str = "gain") -> tuple[pd.Series, dict]:
    metric_key = str(reactivity_metric).strip().lower().replace("-", "_").replace(" ", "_")
    step_nA = pd.to_numeric(repeat_df.get("step_current_amplitude_nA", np.nan), errors="coerce")
    step_sign = np.sign(step_nA)
    step_abs_uA = np.abs(step_nA / 1000.0)

    aliases = {
        "gain": "gain",
        "response_gain": "gain",
        "cst_gain": "gain",
        "linear_slope": "linear_slope",
        "slope": "linear_slope",
        "shape_tau": "shape_tau",
        "tau": "shape_tau",
        "response_tau": "shape_tau",
        "shape_t50": "shape_t50",
        "t50": "shape_t50",
        "shape_t90": "shape_t90",
        "t90": "shape_t90",
        "response_50ms": "response_50ms",
        "response_50": "response_50ms",
        "fraction_50ms": "response_50ms",
        "f50": "response_50ms",
        "response_150ms": "response_150ms",
        "response_150": "response_150ms",
        "fraction_150ms": "response_150ms",
        "f150": "response_150ms",
        "xcorr_lag": "xcorr_lag",
        "lag": "xcorr_lag",
        "xcorr_lag_sign_corrected": "xcorr_lag_sign_corrected",
        "sign_corrected_xcorr_lag": "xcorr_lag_sign_corrected",
    }
    canonical = aliases.get(metric_key)
    if canonical is None:
        raise ValueError(f"Unknown reactivity_metric={reactivity_metric!r}. Available aliases: {sorted(aliases)}")

    if canonical == "gain":
        delta = pd.to_numeric(repeat_df.get("delta_cst_effect", np.nan), errors="coerce")
        values = delta / step_abs_uA.replace(0.0, np.nan)
        label = "signed CST effect gain (spikes/s/MN/uA)"
    elif canonical == "linear_slope":
        base = pd.to_numeric(repeat_df.get("linear_initial_slope_normalized_per_uA", np.nan), errors="coerce")
        values = step_sign * base
        label = "signed linear initial slope (spikes/s/MN/s/uA)"
    elif canonical == "shape_tau":
        base = pd.to_numeric(repeat_df.get("response_shape_tau_ms", np.nan), errors="coerce")
        values = step_sign * base
        label = "signed response-shape tau (ms)"
    elif canonical == "shape_t50":
        base = pd.to_numeric(repeat_df.get("response_shape_fitted_t50_ms", np.nan), errors="coerce")
        values = step_sign * base
        label = "signed response-shape fitted t50 (ms)"
    elif canonical == "shape_t90":
        base = pd.to_numeric(repeat_df.get("response_shape_fitted_t90_ms", np.nan), errors="coerce")
        values = step_sign * base
        label = "signed response-shape fitted t90 (ms)"
    elif canonical == "response_50ms":
        base = pd.to_numeric(repeat_df.get("response_shape_fraction_50ms", np.nan), errors="coerce")
        values = step_sign * base * 100.0
        label = "signed fitted response at 50 ms (% full gain)"
    elif canonical == "response_150ms":
        base = pd.to_numeric(repeat_df.get("response_shape_fraction_150ms", np.nan), errors="coerce")
        values = step_sign * base * 100.0
        label = "signed fitted response at 150 ms (% full gain)"
    elif canonical == "xcorr_lag":
        base = pd.to_numeric(repeat_df.get("xcorr_lag_s", np.nan), errors="coerce")
        values = base * 1000.0
        label = "xcorr lag (ms; negative means step CST leads)"
    elif canonical == "xcorr_lag_sign_corrected":
        base = pd.to_numeric(repeat_df.get("xcorr_lag_sign_corrected_s", np.nan), errors="coerce")
        values = base * 1000.0
        label = "sign-corrected xcorr lag (ms)"
    else:
        raise AssertionError(f"Unhandled canonical metric: {canonical}")

    metadata = {
        "metric_key": canonical,
        "metric_slug": _slugify_label(canonical),
        "metric_label": label,
    }
    return pd.Series(values, index=repeat_df.index, dtype=float), metadata


def resolve_step_reactivity_sync_source(repeat_df: pd.DataFrame, sync_source: str = "pre_matched") -> tuple[pd.Series, dict]:
    source_key = str(sync_source).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "baseline": ("baseline_sync_mean", "baseline synchrony"),
        "baseline_mean": ("baseline_sync_mean", "baseline synchrony"),
        "baseline_sync": ("baseline_sync_mean", "baseline synchrony"),
        "baseline_sd": ("baseline_sync_sd", "baseline synchrony SD"),
        "pre_actual": ("pre_step_sync_mean", "pre-step actual synchrony"),
        "pre_step_actual": ("pre_step_sync_mean", "pre-step actual synchrony"),
        "pre_actual_sd": ("pre_step_sync_sd", "pre-step actual synchrony SD"),
        "pre_step_actual_sd": ("pre_step_sync_sd", "pre-step actual synchrony SD"),
        "pre_matched": ("pre_step_sync_mean_baseline_matched", "pre-step matched synchrony"),
        "sync_pre_matched": ("pre_step_sync_mean_baseline_matched", "pre-step matched synchrony"),
        "pre_step_matched": ("pre_step_sync_mean_baseline_matched", "pre-step matched synchrony"),
        "pre_matched_sd": ("pre_step_sync_sd_baseline_matched", "pre-step matched synchrony SD"),
        "pre_step_matched_sd": ("pre_step_sync_sd_baseline_matched", "pre-step matched synchrony SD"),
        "local_matched": ("local_step_sync_mean_baseline_matched", "local step matched synchrony"),
        "sync_local_matched": ("local_step_sync_mean_baseline_matched", "local step matched synchrony"),
        "local_step_matched": ("local_step_sync_mean_baseline_matched", "local step matched synchrony"),
        "local_matched_sd": ("local_step_sync_sd_baseline_matched", "local step matched synchrony SD"),
        "local_step_matched_sd": ("local_step_sync_sd_baseline_matched", "local step matched synchrony SD"),
        "post_actual": ("post_step_sync_mean_actual_step", "post-step actual synchrony"),
        "post_step_actual": ("post_step_sync_mean_actual_step", "post-step actual synchrony"),
        "post_actual_sd": ("post_step_sync_sd_actual_step", "post-step actual synchrony SD"),
        "post_step_actual_sd": ("post_step_sync_sd_actual_step", "post-step actual synchrony SD"),
    }
    if source_key not in aliases:
        raise ValueError(f"Unknown sync_source={sync_source!r}. Available aliases: {sorted(aliases)}")
    column, label = aliases[source_key]
    if column not in repeat_df.columns:
        values = pd.Series(np.nan, index=repeat_df.index, dtype=float)
    else:
        values = pd.to_numeric(repeat_df[column], errors="coerce")
    metadata = {
        "sync_key": source_key,
        "sync_slug": _slugify_label(source_key),
        "sync_column": column,
        "sync_label": label,
    }
    return pd.Series(values, index=repeat_df.index, dtype=float), metadata


def _resolve_reactivity_color_norm(
    values,
    *,
    metric_vmin=None,
    metric_vmax=None,
    metric_color_percentiles=None,
    metric_color_scale=None,
    center_color_scale_on_zero=True,
):
    if metric_color_scale is not None:
        scale_key = str(metric_color_scale).strip().lower().replace("-", "_").replace(" ", "_")
        if scale_key in {"symmetric", "symmetric_zero", "zero_centered", "centered"}:
            center_color_scale_on_zero = True
        elif scale_key in {"auto", "asymmetric", "non_symmetric", "nonsymmetric"}:
            center_color_scale_on_zero = False
        else:
            raise ValueError("metric_color_scale must be 'symmetric' or 'auto'")
    vmin, vmax = _resolve_color_limits(values, metric_vmin, metric_vmax, metric_color_percentiles)
    if center_color_scale_on_zero:
        max_abs = max(abs(float(vmin)), abs(float(vmax)), 1e-12)
        if metric_vmin is None:
            vmin = -max_abs
        if metric_vmax is None:
            vmax = max_abs
    if center_color_scale_on_zero and vmin < 0.0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax), float(vmin), float(vmax)
    return Normalize(vmin=vmin, vmax=vmax), float(vmin), float(vmax)


def plot_step_reactivity_summary(
    repeat_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    reactivity_metric: str = "gain",
    sync_source: str = "pre_matched",
    step_amplitudes_uA=None,
    filename_base: str | None = None,
    figsize_per_step=(12.5, 4.0),
    cmap="coolwarm",
    point_size=28.0,
    point_alpha=0.82,
    point_edgecolor="none",
    show_regression_lines=True,
    regression_line_color="black",
    regression_line_lw=1.5,
    regression_line_alpha=0.45,
    regression_line_ls="-",
    wspace=0.35,
    hspace=0.38,
    metric_vmin=None,
    metric_vmax=None,
    metric_color_percentiles=None,
    metric_color_scale=None,
    center_color_scale_on_zero=True,
    alpha_xlim=None,
    beta_xlim=None,
    sync_xlim=None,
    metric_ylim=None,
    axis_pad_fraction=0.05,
    title_fontsize=10.0,
    axis_label_fontsize=9.0,
    tick_label_fontsize=8.0,
    annotation_fontsize=8.0,
    suptitle_fontsize=13.0,
    colorbar_label_fontsize=9.0,
    require_matched_step_effect=True,
    save_plot_table=True,
    title: str | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = repeat_df.copy()
    step_nA = pd.to_numeric(df.get("step_current_amplitude_nA", np.nan), errors="coerce")
    df = df[np.isfinite(step_nA) & ~np.isclose(step_nA, 0.0)].copy()
    if require_matched_step_effect and "matched_step_effect_available" in df.columns:
        matched = pd.to_numeric(df["matched_step_effect_available"], errors="coerce")
        df = df[matched.fillna(0.0) > 0.0].copy()
    if df.empty:
        raise ValueError("No nonzero finalized step-response rows available for reactivity plotting")

    metric_values, metric_meta = resolve_step_reactivity_metric(df, reactivity_metric)
    sync_values, sync_meta = resolve_step_reactivity_sync_source(df, sync_source)
    df["alpha_target_sd_final_uA"] = pd.to_numeric(df["alpha_target_sd_final"], errors="coerce") / 1000.0
    df["beta_target_sd_final_uA"] = pd.to_numeric(df["beta_target_sd_final"], errors="coerce") / 1000.0
    df["step_current_amplitude_uA"] = pd.to_numeric(df["step_current_amplitude_nA"], errors="coerce") / 1000.0
    df["reactivity_value"] = metric_values
    df["sync_predictor_value"] = sync_values
    df["reactivity_metric"] = metric_meta["metric_key"]
    df["reactivity_metric_label"] = metric_meta["metric_label"]
    df["sync_source"] = sync_meta["sync_key"]
    df["sync_source_column"] = sync_meta["sync_column"]
    if "reactivity_data_source" not in df.columns:
        df["reactivity_data_source"] = "repeats"

    required_cols = [
        "alpha_target_sd_final_uA",
        "beta_target_sd_final_uA",
        "sync_predictor_value",
        "reactivity_value",
        "step_current_amplitude_uA",
    ]
    mask = np.ones(len(df), dtype=bool)
    for col in required_cols:
        mask &= np.isfinite(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
    plot_df = df[mask].copy()
    if plot_df.empty:
        raise ValueError("No rows have finite alpha, beta, selected sync, and selected reactivity metric values")

    step_values = np.array(sorted(plot_df["step_current_amplitude_uA"].unique()), dtype=float)
    if step_amplitudes_uA is not None:
        requested_steps = np.asarray(step_amplitudes_uA, dtype=float).reshape(-1)
        if requested_steps.size == 0:
            raise ValueError("step_amplitudes_uA was provided but empty")
        available_steps = np.asarray(step_values, dtype=float)
        missing_steps = [
            float(value)
            for value in requested_steps
            if not np.any(np.isclose(available_steps, value, atol=1e-9, rtol=0.0))
        ]
        if missing_steps:
            raise ValueError(
                f"Requested step amplitudes in uA are not present: {missing_steps}. "
                f"Available nonzero steps: {available_steps.tolist()}"
            )
        keep = np.zeros(len(plot_df), dtype=bool)
        for value in requested_steps:
            keep |= np.isclose(plot_df["step_current_amplitude_uA"], float(value), atol=1e-9, rtol=0.0)
        plot_df = plot_df[keep].copy()
        step_values = requested_steps.astype(float)
    n_steps = int(step_values.size)
    alpha_limits = _padded_limits(plot_df["alpha_target_sd_final_uA"], alpha_xlim, axis_pad_fraction)
    beta_limits = _padded_limits(plot_df["beta_target_sd_final_uA"], beta_xlim, axis_pad_fraction)
    sync_limits = _padded_limits(plot_df["sync_predictor_value"], sync_xlim, axis_pad_fraction)
    metric_limits = _padded_limits(plot_df["reactivity_value"], metric_ylim, axis_pad_fraction)
    norm, metric_vmin_resolved, metric_vmax_resolved = _resolve_reactivity_color_norm(
        plot_df["reactivity_value"],
        metric_vmin=metric_vmin,
        metric_vmax=metric_vmax,
        metric_color_percentiles=metric_color_percentiles,
        metric_color_scale=metric_color_scale,
        center_color_scale_on_zero=center_color_scale_on_zero,
    )
    color_map = plt.get_cmap(cmap)

    fig = plt.figure(
        figsize=(float(figsize_per_step[0]), float(figsize_per_step[1]) * n_steps),
        constrained_layout=False,
    )
    gs = fig.add_gridspec(2 * n_steps, 4, width_ratios=[1.25, 1.0, 1.0, 1.0])
    gs.update(wspace=float(wspace), hspace=float(hspace))
    if title is None:
        title = f"Step-response reactivity: {metric_meta['metric_label']} vs {sync_meta['sync_label']}"
    fig.suptitle(title, fontsize=suptitle_fontsize)
    all_axes = []

    top_specs = [
        ("alpha_target_sd_final_uA", "sync_predictor_value", "alpha input SD (uA)", sync_meta["sync_label"]),
        ("beta_target_sd_final_uA", "sync_predictor_value", "beta input SD (uA)", sync_meta["sync_label"]),
    ]
    bottom_specs = [
        ("alpha_target_sd_final_uA", "alpha input SD (uA)"),
        ("beta_target_sd_final_uA", "beta input SD (uA)"),
        ("sync_predictor_value", sync_meta["sync_label"]),
    ]

    for step_i, step_value in enumerate(step_values):
        sub = plot_df[np.isclose(plot_df["step_current_amplitude_uA"], step_value)].copy()
        row0 = 2 * step_i
        ax3d = fig.add_subplot(gs[row0:row0 + 2, 0], projection="3d")
        all_axes.append(ax3d)
        ax3d.scatter(
            sub["alpha_target_sd_final_uA"],
            sub["beta_target_sd_final_uA"],
            sub["sync_predictor_value"],
            c=sub["reactivity_value"],
            cmap=color_map,
            norm=norm,
            s=point_size,
            alpha=point_alpha,
            edgecolors=point_edgecolor,
        )
        ax3d.set_xlim(alpha_limits)
        ax3d.set_ylim(beta_limits)
        ax3d.set_zlim(sync_limits)
        ax3d.set_xlabel("alpha input SD (uA)", fontsize=axis_label_fontsize)
        ax3d.set_ylabel("beta input SD (uA)", fontsize=axis_label_fontsize)
        ax3d.set_zlabel(sync_meta["sync_label"], fontsize=axis_label_fontsize)
        ax3d.tick_params(axis="both", labelsize=tick_label_fontsize)
        ax3d.set_title(f"Step={step_value:g} uA\nn={len(sub)}", fontsize=title_fontsize)

        for col_i, (x_col, y_col, x_label, y_label) in enumerate(top_specs, start=1):
            ax = fig.add_subplot(gs[row0, col_i])
            all_axes.append(ax)
            ax.scatter(
                sub[x_col],
                sub[y_col],
                c=sub["reactivity_value"],
                cmap=color_map,
                norm=norm,
                s=point_size,
                alpha=point_alpha,
                edgecolors=point_edgecolor,
            )
            if x_col == "alpha_target_sd_final_uA":
                ax.set_xlim(alpha_limits)
            elif x_col == "beta_target_sd_final_uA":
                ax.set_xlim(beta_limits)
            elif x_col == "sync_predictor_value":
                ax.set_xlim(sync_limits)
            if y_col == "alpha_target_sd_final_uA":
                ax.set_ylim(alpha_limits)
            elif y_col == "beta_target_sd_final_uA":
                ax.set_ylim(beta_limits)
            elif y_col == "sync_predictor_value":
                ax.set_ylim(sync_limits)
            ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
            ax.set_ylabel(_split_label_units(y_label), fontsize=axis_label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_label_fontsize)
            if show_regression_lines:
                _draw_regression_line(
                    ax,
                    sub[x_col],
                    sub[y_col],
                    color=regression_line_color,
                    lw=regression_line_lw,
                    alpha=regression_line_alpha,
                    ls=regression_line_ls,
                )
            ax.text(
                0.03,
                0.97,
                _format_corr_text(sub[x_col], sub[y_col]),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=annotation_fontsize,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

        for col_i, (x_col, x_label) in enumerate(bottom_specs, start=1):
            ax = fig.add_subplot(gs[row0 + 1, col_i])
            all_axes.append(ax)
            ax.scatter(
                sub[x_col],
                sub["reactivity_value"],
                c=sub["reactivity_value"],
                cmap=color_map,
                norm=norm,
                s=point_size,
                alpha=point_alpha,
                edgecolors=point_edgecolor,
            )
            if x_col == "alpha_target_sd_final_uA":
                ax.set_xlim(alpha_limits)
            elif x_col == "beta_target_sd_final_uA":
                ax.set_xlim(beta_limits)
            elif x_col == "sync_predictor_value":
                ax.set_xlim(sync_limits)
            ax.set_ylim(metric_limits)
            ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.5)
            ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
            ax.set_ylabel(_split_label_units(metric_meta["metric_label"]), fontsize=axis_label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_label_fontsize)
            if show_regression_lines:
                _draw_regression_line(
                    ax,
                    sub[x_col],
                    sub["reactivity_value"],
                    color=regression_line_color,
                    lw=regression_line_lw,
                    alpha=regression_line_alpha,
                    ls=regression_line_ls,
                )
            ax.text(
                0.03,
                0.97,
                _format_corr_text(sub[x_col], sub["reactivity_value"]),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=annotation_fontsize,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=color_map)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, ax=all_axes, fraction=0.018, pad=0.012)
    cbar.ax.tick_params(labelsize=tick_label_fontsize)
    cbar.ax.set_ylabel(metric_meta["metric_label"], rotation=90, fontsize=colorbar_label_fontsize)

    if filename_base is None:
        source_values = sorted(str(value) for value in plot_df["reactivity_data_source"].dropna().unique())
        source_slug = _slugify_label(source_values[0]) if len(source_values) == 1 else "mixed_source"
        filename_base = f"step_reactivity_summary_{metric_meta['metric_slug']}_{sync_meta['sync_slug']}_{source_slug}"
    filename_base = _slugify_label(filename_base)
    png_path = output_dir / f"{filename_base}.png"
    pdf_path = output_dir / f"{filename_base}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    table_paths = {}
    if save_plot_table:
        table_pkl_path = output_dir / f"{filename_base}.pkl"
        table_csv_path = output_dir / f"{filename_base}.csv"
        plot_df.to_pickle(table_pkl_path)
        plot_df.to_csv(table_csv_path, index=False)
        table_paths = {"table_pkl_path": table_pkl_path, "table_csv_path": table_csv_path}

    return {
        "png_path": png_path,
        "pdf_path": pdf_path,
        "plot_df": plot_df,
        "metric_metadata": metric_meta,
        "sync_metadata": sync_meta,
        "metric_vmin": metric_vmin_resolved,
        "metric_vmax": metric_vmax_resolved,
        **table_paths,
    }
