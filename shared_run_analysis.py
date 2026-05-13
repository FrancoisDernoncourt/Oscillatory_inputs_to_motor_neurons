import numpy as np
from scipy.signal import welch


OBS_WINDOW_SPECS = [
    ("m4_to_m3", "normalization_win", (-4.0, -3.0)),
    ("m3_to_m2", "baseline_win", (-3.0, -2.0)),
    ("m2_to_m1", "baseline_extended_win", (-2.0, -1.0)),
    ("m1_to_0", "ready_win", (-1.0, 0.0)),
    ("0_to_1", "post_win", (0.0, 1.0)),
]

OBS_WINDOW_SIGNAL_FAMILIES = [
    "CST_LF_raw",
    "ALPHA_raw",
    "ALPHA_mod_z",
    "BETA_raw",
    "BETA_mod_z",
    "SYNC",
]

OBS_WINDOW_STAT_FAMILIES = [
    "mean",
    "min",
    "max",
    "t_min_rel_cue_s",
    "t_max_rel_cue_s",
]

OBS_BASELINE_FEATURE_FAMILIES = [
    "CST_LF_raw_mean",
    "CST_LF_raw_sd",
    "ALPHA_raw_mean",
    "ALPHA_raw_sd",
    "ALPHA_mod_z_mean",
    "ALPHA_mod_z_sd",
    "BETA_raw_mean",
    "BETA_raw_sd",
    "BETA_mod_z_mean",
    "BETA_mod_z_sd",
]

OBS_SPECTRUM_FEATURE_FAMILIES = [
    "ALPHA_peak_hz",
    "ALPHA_interval_low_hz",
    "ALPHA_interval_high_hz",
    "BETA_peak_hz",
    "BETA_interval_low_hz",
    "BETA_interval_high_hz",
]

ANALYSIS_PARAM_WINDOW_COLUMNS = [
    "ANALYSIS_PARAM_WINDOW_normalization_win_rel_go_cue_s",
    "ANALYSIS_PARAM_WINDOW_baseline_win_rel_go_cue_s",
    "ANALYSIS_PARAM_WINDOW_baseline_extended_win_rel_go_cue_s",
    "ANALYSIS_PARAM_WINDOW_ready_win_rel_go_cue_s",
    "ANALYSIS_PARAM_WINDOW_post_win_rel_go_cue_s",
    "ANALYSIS_PARAM_WINDOW_obs_baseline_win_rel_go_cue_s",
    "ANALYSIS_PARAM_WINDOW_cst_modulation_normalization_win_rel_go_cue_s",
]

ANALYSIS_PARAM_FEATURE_COLUMNS = [
    "ANALYSIS_PARAM_FEATURE_fitting_window",
    "ANALYSIS_PARAM_FEATURE_analysis_window",
    "ANALYSIS_PARAM_FEATURE_buffer_s",
    "ANALYSIS_PARAM_FEATURE_spline_smoothing",
    "ANALYSIS_PARAM_FEATURE_spline_order",
    "ANALYSIS_PARAM_FEATURE_n_knots",
    "ANALYSIS_PARAM_FEATURE_trend_pre_window",
    "ANALYSIS_PARAM_FEATURE_trend_post_window",
]

ANALYSIS_PARAM_FILTER_COLUMNS = [
    "ANALYSIS_PARAM_FILTER_cst_lf_band_hz",
    "ANALYSIS_PARAM_FILTER_cst_alpha_band_hz",
    "ANALYSIS_PARAM_FILTER_cst_beta_band_hz",
    "ANALYSIS_PARAM_FILTER_active_unit_rate_threshold_hz",
]

ANALYSIS_PARAM_SYNC_COLUMNS = [
    "ANALYSIS_PARAM_SYNC_win_ms",
    "ANALYSIS_PARAM_SYNC_step_ms",
    "ANALYSIS_PARAM_SYNC_coinc_lag_ms",
    "ANALYSIS_PARAM_SYNC_direction_mode",
    "ANALYSIS_PARAM_SYNC_expectation_mode",
]

ANALYSIS_PARAM_SPECTRUM_COLUMNS = [
    "ANALYSIS_PARAM_SPECTRUM_window_rel_go_cue_s",
    "ANALYSIS_PARAM_SPECTRUM_max_freq_hz",
    "ANALYSIS_PARAM_SPECTRUM_interval_mass_pct",
]


def _build_obs_window_export_columns():
    columns = []
    for window_code, _, _ in OBS_WINDOW_SPECS:
        for signal_family in OBS_WINDOW_SIGNAL_FAMILIES:
            for stat_family in OBS_WINDOW_STAT_FAMILIES:
                columns.append(f"OBS_WINDOW_{window_code}_{signal_family}_{stat_family}")
    return columns


def _build_obs_baseline_export_columns():
    return [f"OBS_BASELINE_{feature_family}" for feature_family in OBS_BASELINE_FEATURE_FAMILIES]


def _build_obs_spectrum_export_columns():
    return [f"OBS_SPECTRUM_{feature_family}" for feature_family in OBS_SPECTRUM_FEATURE_FAMILIES]


OBS_WINDOW_EXPORT_COLUMNS = _build_obs_window_export_columns()
OBS_BASELINE_EXPORT_COLUMNS = _build_obs_baseline_export_columns()
OBS_SPECTRUM_EXPORT_COLUMNS = _build_obs_spectrum_export_columns()

EXPORT_TABLE_COLUMNS = [
    "DATASET_CONTEXT_participant_id",
    "DATASET_CONTEXT_session_tag",
    "DATASET_CONTEXT_condition_id",
    "SIM_PARAM_GENERAL_batch_sim_index",
    "SIM_SUMMARY_is_posterior_mode",
    "SIM_SUMMARY_mode_observation_label",
    "SIM_PARAM_GENERAL_minimal_output",
    "SIM_PARAM_GENERAL_enable_force_model",
    "SIM_PARAM_GENERAL_n_sims",
    "SIM_PARAM_GENERAL_duration",
    "SIM_PARAM_GENERAL_get_ready_cue_time_s",
    "SIM_PARAM_GENERAL_go_nogo_cue_delay_s",
    "SIM_PARAM_GENERAL_nb_motoneurons",
    "SIM_PARAM_GENERAL_min_soma_diameter",
    "SIM_PARAM_GENERAL_max_soma_diameter",
    "SIM_PARAM_GENERAL_tau_size_mode",
    "SIM_PARAM_GENERAL_tau_size_ratio",
    "SIM_PARAM_GENERAL_tau_size_um",
    "SIM_PARAM_BASELINE_INPUT_excitatory_input_baseline",
    "SIM_PARAM_BASELINE_INPUT_lf_target_sd",
    "SIM_PARAM_BASELINE_INPUT_lf_band_hz",
    "SIM_PARAM_BASELINE_INPUT_alpha_target_sd_final",
    "SIM_PARAM_BASELINE_INPUT_alpha_band_hz",
    "SIM_PARAM_BASELINE_INPUT_alpha_envelope_band_hz",
    "SIM_PARAM_BASELINE_INPUT_beta_target_sd_final",
    "SIM_PARAM_BASELINE_INPUT_beta_band_hz",
    "SIM_PARAM_BASELINE_INPUT_beta_envelope_band_hz",
    "SIM_PARAM_BASELINE_INPUT_common_input_weight_sd",
    "SIM_PARAM_BASELINE_INPUT_common_input_weight_min",
    "SIM_PARAM_BASELINE_INPUT_common_input_weight_max",
    "SIM_PARAM_BASELINE_INPUT_independent_input_weight_sd",
    "SIM_PARAM_BASELINE_INPUT_independent_input_weight_max",
    "SIM_PARAM_BASELINE_INPUT_independent_input_absolute_or_ratio",
    "SIM_PARAM_BASELINE_INPUT_independent_input_power",
    "SIM_PARAM_BURST_INPUT_burst_delay_after_go_nogo_cue_s",
    "SIM_PARAM_BURST_INPUT_input_burst_sigma_ms",
    "SIM_PARAM_BURST_INPUT_lf_burst_peak_nA",
    "SIM_PARAM_BURST_INPUT_alpha_burst_peak_nA",
    "SIM_PARAM_BURST_INPUT_beta_burst_peak_nA",
    "SIM_PARAM_TREND_INPUT_lf_trend_start_rel_go_cue_s",
    "SIM_PARAM_TREND_INPUT_alpha_trend_start_rel_go_cue_s",
    "SIM_PARAM_TREND_INPUT_beta_trend_start_rel_go_cue_s",
    "SIM_PARAM_TREND_INPUT_lf_trend_slope_nA_per_s",
    "SIM_PARAM_TREND_INPUT_alpha_trend_slope_nA_per_s",
    "SIM_PARAM_TREND_INPUT_beta_trend_slope_nA_per_s",
    "OBS_FIRING_STATISTICS_firing_rate_mean",
    "OBS_FIRING_STATISTICS_firing_rate_sd",
    "OBS_FIRING_STATISTICS_firing_rate_skew",
    "OBS_FIRING_STATISTICS_isi_cv_mean",
    "OBS_FIRING_STATISTICS_isi_cv_sd",
    "OBS_FIRING_STATISTICS_isi_cv_skew",
    *OBS_WINDOW_EXPORT_COLUMNS,
    *OBS_BASELINE_EXPORT_COLUMNS,
    *OBS_SPECTRUM_EXPORT_COLUMNS,
    "OBS_BURST_FEATURES_resid_cst_low_frequency",
    "OBS_BURST_FEATURES_resid_cst_alpha_modulation_hilbert_zscore",
    "OBS_BURST_FEATURES_resid_cst_beta_modulation_hilbert_zscore",
    "OBS_BURST_FEATURES_resid_sync_coincidence_curve",
    "OBS_TREND_FEATURE_cst_low_frequency",
    "OBS_TREND_FEATURE_cst_alpha_modulation_hilbert_zscore",
    "OBS_TREND_FEATURE_cst_beta_modulation_hilbert_zscore",
    "OBS_TREND_FEATURE_sync_coincidence_curve",
    *ANALYSIS_PARAM_WINDOW_COLUMNS,
    *ANALYSIS_PARAM_FEATURE_COLUMNS,
    *ANALYSIS_PARAM_FILTER_COLUMNS,
    *ANALYSIS_PARAM_SYNC_COLUMNS,
    *ANALYSIS_PARAM_SPECTRUM_COLUMNS,
    "ANALYSIS_DIAGNOSIS_n_units_total",
    "ANALYSIS_DIAGNOSIS_n_units_kept_mean",
    "ANALYSIS_DIAGNOSIS_n_units_kept_min",
    "ANALYSIS_DIAGNOSIS_unit_fraction_kept_mean",
    "ANALYSIS_DIAGNOSIS_fit_r2_cst_low_frequency",
    "ANALYSIS_DIAGNOSIS_fit_r2_cst_alpha_modulation_hilbert_zscore",
    "ANALYSIS_DIAGNOSIS_fit_r2_cst_beta_modulation_hilbert_zscore",
    "ANALYSIS_DIAGNOSIS_fit_r2_sync_coincidence_curve",
    "ANALYSIS_DIAGNOSIS_fit_r2_gain_over_linear_cst_low_frequency",
    "ANALYSIS_DIAGNOSIS_fit_r2_gain_over_linear_cst_alpha_modulation_hilbert_zscore",
    "ANALYSIS_DIAGNOSIS_fit_r2_gain_over_linear_cst_beta_modulation_hilbert_zscore",
    "ANALYSIS_DIAGNOSIS_fit_r2_gain_over_linear_sync_coincidence_curve",
]

OBS_WINDOW_CODE_TO_ALIAS = {
    code: alias
    for code, alias, _ in OBS_WINDOW_SPECS
}
OBS_WINDOW_ALIAS_TO_CODE = {
    alias: code
    for code, alias, _ in OBS_WINDOW_SPECS
}

DEFAULT_OBSERVATION_WINDOW_DEFS_REL_CUE_S = {
    alias: tuple(map(float, bounds))
    for _, alias, bounds in OBS_WINDOW_SPECS
}

DEFAULT_MODULATION_SPECTRUM_CONFIG = {
    "enabled": True,
    "window_rel_cue_s": (-3.0, -1.0),
    "max_freq_hz": 12.0,
    "interval_mass_pct": 80.0,
    "show_interval": True,
    "interval_alpha": 0.2,
    "ylabel": "PSD of baseline-zscored envelope (a.u.^2/Hz)",
}


def _window_obs_feature_name(window_code, signal_family, stat_family):
    return f"OBS_WINDOW_{window_code}_{signal_family}_{stat_family}"


def _baseline_obs_feature_name(feature_family):
    return f"OBS_BASELINE_{feature_family}"


def _spectrum_obs_feature_name(feature_family):
    return f"OBS_SPECTRUM_{feature_family}"


def resolve_modulation_spectrum_config(config=None):
    resolved = dict(DEFAULT_MODULATION_SPECTRUM_CONFIG)
    if config:
        resolved.update(config)
    if resolved.get("window_rel_cue_s") is not None:
        resolved["window_rel_cue_s"] = tuple(map(float, resolved["window_rel_cue_s"]))
    if resolved.get("max_freq_hz") is not None:
        resolved["max_freq_hz"] = float(resolved["max_freq_hz"])
    if resolved.get("interval_mass_pct") is not None:
        resolved["interval_mass_pct"] = float(resolved["interval_mass_pct"])
    resolved["show_interval"] = bool(resolved.get("show_interval", True))
    resolved["interval_alpha"] = float(resolved.get("interval_alpha", 0.2))
    return resolved


def classify_active_units_by_rate(
    spike_trains_s,
    *,
    analysis_start_s,
    analysis_end_s,
    rate_threshold_hz,
):
    analysis_start_s = float(analysis_start_s)
    analysis_end_s = float(analysis_end_s)
    rate_threshold_hz = float(rate_threshold_hz)
    window_duration_s = analysis_end_s - analysis_start_s
    if window_duration_s <= 0:
        raise ValueError("Active-unit classification window must have strictly positive duration")

    active_ids = []
    excluded_ids = []
    active_rates_hz = []
    all_rates_hz = np.zeros(len(spike_trains_s), dtype=float)
    for unit_i, spikes_s in enumerate(spike_trains_s):
        spikes_s = np.asarray(spikes_s, dtype=float)
        n_in_window = int(np.count_nonzero((spikes_s >= analysis_start_s) & (spikes_s < analysis_end_s)))
        mean_rate_hz = n_in_window / window_duration_s
        all_rates_hz[unit_i] = float(mean_rate_hz)
        if mean_rate_hz >= rate_threshold_hz:
            active_ids.append(unit_i)
            active_rates_hz.append(mean_rate_hz)
        else:
            excluded_ids.append(unit_i)

    n_total = int(len(spike_trains_s))
    n_kept = int(len(active_ids))
    return {
        "analysis_window_start_s": float(analysis_start_s),
        "analysis_window_end_s": float(analysis_end_s),
        "rate_threshold_hz": float(rate_threshold_hz),
        "n_units_total": n_total,
        "n_units_kept": n_kept,
        "unit_fraction_kept": float(n_kept / n_total) if n_total > 0 else np.nan,
        "active_unit_ids": np.asarray(active_ids, dtype=int),
        "excluded_unit_ids": np.asarray(excluded_ids, dtype=int),
        "active_unit_rates_hz": np.asarray(active_rates_hz, dtype=float),
        "all_unit_rates_hz": np.asarray(all_rates_hz, dtype=float),
    }


def compute_psd_full_resolution(signal, fsamp):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    finite_mask = np.isfinite(signal)
    signal = signal[finite_mask]
    if signal.size < 4:
        return np.array([], dtype=float), np.array([], dtype=float)
    nperseg = int(signal.size)
    freqs_hz, psd = welch(
        signal,
        fs=float(fsamp),
        window="hann",
        nperseg=nperseg,
        noverlap=0,
        scaling="density",
        detrend="constant",
    )
    return np.asarray(freqs_hz, dtype=float), np.asarray(psd, dtype=float)


def smallest_contiguous_psd_mass_interval(freqs, psd, mass_pct):
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    psd = np.asarray(psd, dtype=float).reshape(-1)
    mask = np.isfinite(freqs) & np.isfinite(psd) & (psd >= 0)
    freqs = freqs[mask]
    psd = psd[mask]
    if freqs.size == 0 or psd.size == 0:
        return (np.nan, np.nan)
    if freqs.size == 1:
        freq = float(freqs[0])
        return (freq, freq)
    total_mass = float(np.trapz(psd, freqs))
    if not np.isfinite(total_mass) or total_mass <= 0:
        peak_freq = float(freqs[int(np.nanargmax(psd))])
        return (peak_freq, peak_freq)
    target_mass = (float(mass_pct) / 100.0) * total_mass
    segment_masses = 0.5 * (psd[:-1] + psd[1:]) * np.diff(freqs)
    best_width = np.inf
    best_bounds = (float(freqs[0]), float(freqs[-1]))
    for start_idx in range(freqs.size - 1):
        accumulated = 0.0
        end_idx = start_idx
        while end_idx < segment_masses.size and accumulated < target_mass:
            accumulated += float(segment_masses[end_idx])
            end_idx += 1
        if accumulated >= target_mass:
            lo = float(freqs[start_idx])
            hi = float(freqs[end_idx])
            width = hi - lo
            if width < best_width:
                best_width = width
                best_bounds = (lo, hi)
    return best_bounds


def compute_modulation_spectra_for_save(cst_diagnostics, *, spectrum_window_rel_cue_s, cue_time_abs_s):
    if cst_diagnostics is None or str(cst_diagnostics.get("status", "")).lower() != "ok":
        return {}
    time_s = np.asarray(cst_diagnostics.get("time_s"), dtype=float)
    if time_s.size < 4:
        return {}
    dt_s = float(np.median(np.diff(time_s)))
    if not np.isfinite(dt_s) or dt_s <= 0:
        return {}
    spectrum_start_s = float(cue_time_abs_s) + float(spectrum_window_rel_cue_s[0])
    spectrum_end_s = float(cue_time_abs_s) + float(spectrum_window_rel_cue_s[1])
    spectrum_mask = (time_s >= spectrum_start_s) & (time_s < spectrum_end_s)
    if np.count_nonzero(spectrum_mask) < 8:
        return {}

    spectra = {}
    shared_freqs = None
    for prefix in ("alpha", "beta"):
        envelope = np.asarray(cst_diagnostics.get(f"cst_{prefix}_envelope"), dtype=float)
        baseline_mean_arr = np.asarray(cst_diagnostics.get(f"cst_{prefix}_baseline_mean"), dtype=float).reshape(-1)
        baseline_sd_arr = np.asarray(cst_diagnostics.get(f"cst_{prefix}_baseline_sd"), dtype=float).reshape(-1)
        if envelope.shape != time_s.shape or baseline_mean_arr.size == 0 or baseline_sd_arr.size == 0:
            continue
        baseline_mean = float(baseline_mean_arr[0])
        baseline_sd = float(baseline_sd_arr[0])
        if not np.isfinite(baseline_mean) or not np.isfinite(baseline_sd) or baseline_sd <= 1e-9:
            continue
        envelope_z = (envelope - baseline_mean) / baseline_sd
        freqs_hz, psd = compute_psd_full_resolution(envelope_z[spectrum_mask], fsamp=1.0 / dt_s)
        if freqs_hz.size == 0 or psd.size == 0:
            continue
        if shared_freqs is None:
            shared_freqs = np.asarray(freqs_hz, dtype=float)
            spectra["cst_modulation_spectrum_freqs_hz"] = shared_freqs
        else:
            psd = np.interp(shared_freqs, freqs_hz, psd, left=np.nan, right=np.nan)
        spectra[f"cst_{prefix}_modulation_spectrum_psd"] = np.asarray(psd, dtype=float)
    spectra["cst_modulation_spectrum_window_rel_cue_s"] = np.asarray(spectrum_window_rel_cue_s, dtype=float)
    return spectra


def compute_window_stat_features(time_rel_cue_s, signal, *, window_rel_cue_s, window_code, signal_family):
    time_rel_cue_s = np.asarray(time_rel_cue_s, dtype=float).reshape(-1)
    signal = np.asarray(signal, dtype=float).reshape(-1)
    features = {}
    if time_rel_cue_s.shape != signal.shape or time_rel_cue_s.size == 0:
        for stat_family in OBS_WINDOW_STAT_FAMILIES:
            features[_window_obs_feature_name(window_code, signal_family, stat_family)] = np.nan
        return features
    mask = (
        np.isfinite(time_rel_cue_s)
        & np.isfinite(signal)
        & (time_rel_cue_s >= float(window_rel_cue_s[0]))
        & (time_rel_cue_s < float(window_rel_cue_s[1]))
    )
    if not np.any(mask):
        for stat_family in OBS_WINDOW_STAT_FAMILIES:
            features[_window_obs_feature_name(window_code, signal_family, stat_family)] = np.nan
        return features
    t_window = time_rel_cue_s[mask]
    y_window = signal[mask]
    min_idx = int(np.nanargmin(y_window))
    max_idx = int(np.nanargmax(y_window))
    features[_window_obs_feature_name(window_code, signal_family, "mean")] = float(np.nanmean(y_window))
    features[_window_obs_feature_name(window_code, signal_family, "min")] = float(y_window[min_idx])
    features[_window_obs_feature_name(window_code, signal_family, "max")] = float(y_window[max_idx])
    features[_window_obs_feature_name(window_code, signal_family, "t_min_rel_cue_s")] = float(t_window[min_idx])
    features[_window_obs_feature_name(window_code, signal_family, "t_max_rel_cue_s")] = float(t_window[max_idx])
    return features


def compute_baseline_summary_features(time_rel_cue_s, cst_diagnostics, *, obs_baseline_window_rel_cue_s):
    time_rel_cue_s = np.asarray(time_rel_cue_s, dtype=float).reshape(-1)
    features = {_baseline_obs_feature_name(feature_family): np.nan for feature_family in OBS_BASELINE_FEATURE_FAMILIES}
    baseline_mask = (
        np.isfinite(time_rel_cue_s)
        & (time_rel_cue_s >= float(obs_baseline_window_rel_cue_s[0]))
        & (time_rel_cue_s < float(obs_baseline_window_rel_cue_s[1]))
    )
    if not np.any(baseline_mask):
        return features
    signal_map = {
        "CST_LF_raw": np.asarray(cst_diagnostics.get("cst_lf"), dtype=float),
        "ALPHA_raw": np.asarray(cst_diagnostics.get("cst_alpha"), dtype=float),
        "ALPHA_mod_z": np.asarray(cst_diagnostics.get("cst_alpha_envelope_z"), dtype=float),
        "BETA_raw": np.asarray(cst_diagnostics.get("cst_beta"), dtype=float),
        "BETA_mod_z": np.asarray(cst_diagnostics.get("cst_beta_envelope_z"), dtype=float),
    }
    for signal_family, signal in signal_map.items():
        if signal.shape != time_rel_cue_s.shape:
            continue
        y_window = np.asarray(signal, dtype=float)[baseline_mask]
        y_window = y_window[np.isfinite(y_window)]
        if y_window.size == 0:
            continue
        features[_baseline_obs_feature_name(f"{signal_family}_mean")] = float(np.mean(y_window))
        features[_baseline_obs_feature_name(f"{signal_family}_sd")] = float(np.std(y_window))
    return features


def compute_single_trial_modulation_spectrum_features(spectra_dict, spectrum_config):
    features = {_spectrum_obs_feature_name(feature_family): np.nan for feature_family in OBS_SPECTRUM_FEATURE_FAMILIES}
    freqs = np.asarray(spectra_dict.get("cst_modulation_spectrum_freqs_hz"), dtype=float)
    if freqs.size == 0:
        return features
    max_freq_hz = spectrum_config.get("max_freq_hz")
    if max_freq_hz is not None:
        keep_mask = freqs <= float(max_freq_hz)
        freqs = freqs[keep_mask]
    for band_key in ("alpha", "beta"):
        psd = np.asarray(spectra_dict.get(f"cst_{band_key}_modulation_spectrum_psd"), dtype=float)
        if psd.size == 0:
            continue
        if max_freq_hz is not None:
            psd = psd[keep_mask]
        finite_mask = np.isfinite(freqs) & np.isfinite(psd)
        freqs_band = freqs[finite_mask]
        psd_band = psd[finite_mask]
        if freqs_band.size == 0 or psd_band.size == 0:
            continue
        peak_freq_hz = float(freqs_band[int(np.nanargmax(psd_band))])
        interval_bounds = smallest_contiguous_psd_mass_interval(
            freqs_band,
            psd_band,
            spectrum_config["interval_mass_pct"],
        )
        prefix = band_key.upper()
        features[_spectrum_obs_feature_name(f"{prefix}_peak_hz")] = peak_freq_hz
        features[_spectrum_obs_feature_name(f"{prefix}_interval_low_hz")] = float(interval_bounds[0])
        features[_spectrum_obs_feature_name(f"{prefix}_interval_high_hz")] = float(interval_bounds[1])
    return features


def compute_observation_features_from_cst_sync(
    cst_diagnostics,
    sync_diagnostics,
    *,
    cue_time_abs_s,
    observation_window_defs_rel_cue_s,
    obs_baseline_window_rel_cue_s,
    modulation_spectrum_config,
):
    if cst_diagnostics is None or str(cst_diagnostics.get("status", "")).lower() != "ok":
        return None
    time_s = np.asarray(cst_diagnostics.get("time_s"), dtype=float)
    time_rel_cue_s = time_s - float(cue_time_abs_s)
    observation_features = {}
    signal_map = {
        "CST_LF_raw": np.asarray(cst_diagnostics.get("cst_lf"), dtype=float),
        "ALPHA_raw": np.asarray(cst_diagnostics.get("cst_alpha"), dtype=float),
        "ALPHA_mod_z": np.asarray(cst_diagnostics.get("cst_alpha_envelope_z"), dtype=float),
        "BETA_raw": np.asarray(cst_diagnostics.get("cst_beta"), dtype=float),
        "BETA_mod_z": np.asarray(cst_diagnostics.get("cst_beta_envelope_z"), dtype=float),
    }
    for alias, window_rel_cue_s in observation_window_defs_rel_cue_s.items():
        window_code = OBS_WINDOW_ALIAS_TO_CODE[alias]
        for signal_family, signal in signal_map.items():
            observation_features.update(
                compute_window_stat_features(
                    time_rel_cue_s,
                    signal,
                    window_rel_cue_s=window_rel_cue_s,
                    window_code=window_code,
                    signal_family=signal_family,
                )
            )
    observation_features.update(
        compute_baseline_summary_features(
            time_rel_cue_s,
            cst_diagnostics,
            obs_baseline_window_rel_cue_s=obs_baseline_window_rel_cue_s,
        )
    )
    resolved_spectrum_config = resolve_modulation_spectrum_config(modulation_spectrum_config)
    spectra_dict = compute_modulation_spectra_for_save(
        cst_diagnostics,
        spectrum_window_rel_cue_s=resolved_spectrum_config["window_rel_cue_s"],
        cue_time_abs_s=cue_time_abs_s,
    )
    observation_features.update(
        compute_single_trial_modulation_spectrum_features(spectra_dict, resolved_spectrum_config)
    )
    if sync_diagnostics is not None:
        sync_time_rel = np.asarray(sync_diagnostics.get("t_sync"), dtype=float)
        sync_trace = np.asarray(sync_diagnostics.get("sync_trace"), dtype=float)
        if sync_time_rel.shape == sync_trace.shape and sync_time_rel.size > 0:
            for alias, window_rel_cue_s in observation_window_defs_rel_cue_s.items():
                window_code = OBS_WINDOW_ALIAS_TO_CODE[alias]
                observation_features.update(
                    compute_window_stat_features(
                        sync_time_rel,
                        sync_trace,
                        window_rel_cue_s=window_rel_cue_s,
                        window_code=window_code,
                        signal_family="SYNC",
                    )
                )
    for window_code, _, _ in OBS_WINDOW_SPECS:
        for stat_family in OBS_WINDOW_STAT_FAMILIES:
            observation_features.setdefault(_window_obs_feature_name(window_code, "SYNC", stat_family), np.nan)
    return observation_features
