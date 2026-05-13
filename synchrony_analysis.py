import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


def _coerce_unit_ids(spike_data, unit_ids=None):
    if unit_ids is not None:
        return np.asarray(unit_ids)
    if isinstance(spike_data, dict):
        parsed = []
        for key in spike_data.keys():
            try:
                parsed.append(int(str(key).split("_")[-1]))
            except Exception:
                parsed.append(len(parsed))
        return np.asarray(parsed)
    if isinstance(spike_data, np.ndarray):
        return np.arange(spike_data.shape[0], dtype=int)
    return np.arange(len(spike_data), dtype=int)


def bin_spike_trains(
    spike_data,
    *,
    bin_ms=1.0,
    sample_dt_s=None,
    time_vector_s=None,
    duration_s=None,
    analysis_start_s=0.0,
    analysis_end_s=None,
):
    """Convert spike trains into binary binned trains of shape (n_units, n_bins)."""
    bin_s = float(bin_ms) / 1000.0
    if bin_s <= 0:
        raise ValueError("bin_ms must be strictly positive")
    analysis_start_s = float(analysis_start_s)
    if analysis_end_s is None:
        if duration_s is None and time_vector_s is None and not isinstance(spike_data, np.ndarray):
            raise ValueError("duration_s or time_vector_s is required for spike-time inputs")
        if duration_s is None:
            tv = np.asarray(time_vector_s, dtype=float)
            if tv.ndim != 1 or tv.size < 2:
                raise ValueError("time_vector_s must be 1D with at least two samples")
            duration_s = float(tv[-1] - tv[0] + np.median(np.diff(tv)))
        analysis_end_s = float(duration_s)
    else:
        analysis_end_s = float(analysis_end_s)
    if analysis_end_s <= analysis_start_s:
        raise ValueError("analysis window is empty or inverted")

    n_bins = int(np.ceil((analysis_end_s - analysis_start_s) / bin_s))
    t_bin_centers_abs = analysis_start_s + (np.arange(n_bins, dtype=float) + 0.5) * bin_s

    if isinstance(spike_data, np.ndarray):
        spike_matrix = np.asarray(spike_data)
        if spike_matrix.ndim != 2:
            raise ValueError("Spike matrix input must have shape (n_units, n_samples)")
        n_units, n_samples = spike_matrix.shape
        if time_vector_s is not None:
            sample_times = np.asarray(time_vector_s, dtype=float)
            if sample_times.ndim != 1 or sample_times.size != n_samples:
                raise ValueError("time_vector_s length must match the spike matrix sample dimension")
        else:
            if sample_dt_s is None:
                raise ValueError("sample_dt_s is required when time_vector_s is not provided for matrix input")
            sample_times = np.arange(n_samples, dtype=float) * float(sample_dt_s)
        sample_mask = (sample_times >= analysis_start_s) & (sample_times < analysis_end_s)
        sample_times = sample_times[sample_mask]
        selected = (spike_matrix[:, sample_mask] > 0).astype(np.uint8)
        binned = np.zeros((n_units, n_bins), dtype=np.uint8)
        if sample_times.size:
            bin_idx = np.floor((sample_times - analysis_start_s) / bin_s).astype(int)
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            for unit_i in range(n_units):
                active_bins = bin_idx[selected[unit_i] > 0]
                if active_bins.size:
                    binned[unit_i, np.unique(active_bins)] = 1
    else:
        spike_iter = list(spike_data.values()) if isinstance(spike_data, dict) else list(spike_data)
        n_units = len(spike_iter)
        binned = np.zeros((n_units, n_bins), dtype=np.uint8)
        for unit_i, spikes in enumerate(spike_iter):
            spikes_s = np.asarray(spikes, dtype=float)
            if spikes_s.size == 0:
                continue
            spikes_s = spikes_s[(spikes_s >= analysis_start_s) & (spikes_s < analysis_end_s)]
            if spikes_s.size == 0:
                continue
            bin_idx = np.floor((spikes_s - analysis_start_s) / bin_s).astype(int)
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            binned[unit_i, np.unique(bin_idx)] = 1

    return {
        "binary_spikes": binned,
        "bin_size_s": bin_s,
        "time_bin_centers_abs_s": t_bin_centers_abs,
        "analysis_start_s": analysis_start_s,
        "analysis_end_s": analysis_end_s,
    }


def sliding_window_indices(n_bins, window_bins, step_bins):
    if window_bins <= 0 or step_bins <= 0:
        raise ValueError("window_bins and step_bins must be strictly positive")
    if n_bins < window_bins:
        raise ValueError("The binned spike train is shorter than one synchrony window")
    starts = np.arange(0, n_bins - window_bins + 1, step_bins, dtype=int)
    return starts, starts + window_bins


def pairwise_coincidence_count_forward(x, y, lag_bins):
    x = np.asarray(x, dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    lag_bins = int(lag_bins)
    if lag_bins < 0:
        raise ValueError("lag_bins must be >= 0")
    if x.size == 0:
        return 0.0
    x_idx = np.flatnonzero(x > 0)
    y_idx = np.flatnonzero(y > 0)
    if x_idx.size == 0 or y_idx.size == 0:
        return 0.0
    count = 0
    for xi in x_idx:
        insert_pos = np.searchsorted(y_idx, xi)
        matched = False
        if insert_pos < y_idx.size and abs(int(y_idx[insert_pos]) - int(xi)) <= lag_bins:
            matched = True
        elif insert_pos > 0 and abs(int(y_idx[insert_pos - 1]) - int(xi)) <= lag_bins:
            matched = True
        if matched:
            count += 1
    return float(count)


def pairwise_coincidence_count_symmetric(x, y, lag_bins):
    obs_forward = pairwise_coincidence_count_forward(x, y, lag_bins)
    obs_reverse = pairwise_coincidence_count_forward(y, x, lag_bins)
    return 0.5 * (obs_forward + obs_reverse)


def expected_coincidence_analytic(x, y, lag_bins):
    n_bins = int(len(x))
    if n_bins <= 0:
        return 0.0
    rx = float(np.sum(x)) / float(n_bins)
    ry = float(np.sum(y)) / float(n_bins)
    return float(n_bins) * rx * ry * float(2 * int(lag_bins) + 1)


def circular_shift_surrogate(x, y, rng, min_abs_shift_bins=1):
    x = np.asarray(x, dtype=np.uint8)
    y = np.asarray(y, dtype=np.uint8)
    n_bins = int(len(x))
    if n_bins != len(y):
        raise ValueError("x and y must have the same length")
    if n_bins <= 1:
        return x.copy(), y.copy()
    min_abs_shift_bins = max(0, int(min_abs_shift_bins))

    def _draw_shift():
        if n_bins <= 2 * min_abs_shift_bins:
            choices = np.arange(1, n_bins, dtype=int)
        else:
            choices = np.arange(min_abs_shift_bins, n_bins, dtype=int)
            choices = choices[choices % n_bins != 0]
        if choices.size == 0:
            return 0
        return int(rng.choice(choices))

    return np.roll(x, _draw_shift()), np.roll(y, _draw_shift())


def expected_coincidence_surrogate_circular_shift(
    x,
    y,
    lag_bins,
    *,
    direction_mode,
    n_surrogates=100,
    min_abs_shift_bins=1,
    rng=None,
):
    rng = np.random.default_rng() if rng is None else rng
    surrogate_counts = np.zeros(int(n_surrogates), dtype=float)
    for surrogate_i in range(int(n_surrogates)):
        x_shift, y_shift = circular_shift_surrogate(
            x, y, rng=rng, min_abs_shift_bins=min_abs_shift_bins
        )
        if direction_mode == "legacy_forward":
            surrogate_counts[surrogate_i] = pairwise_coincidence_count_forward(x_shift, y_shift, lag_bins)
        elif direction_mode == "symmetric_mean":
            surrogate_counts[surrogate_i] = pairwise_coincidence_count_symmetric(x_shift, y_shift, lag_bins)
        else:
            raise ValueError(f"Unsupported direction_mode {direction_mode!r}")
    return float(np.mean(surrogate_counts)), float(np.std(surrogate_counts, ddof=0))


def pairwise_sync_index(obs_used, expected_used, eps=1e-12):
    if expected_used <= eps or not np.isfinite(expected_used):
        return 0.0
    return float((obs_used - expected_used) / expected_used)


def compute_sliding_sync_trace(
    spike_data,
    *,
    cue_time_sec=0.0,
    bin_ms=1.0,
    sync_win_ms=80.0,
    sync_step_ms=10.0,
    coinc_lag_ms=6.0,
    direction_mode="legacy_forward",
    expectation_mode="analytic",
    n_surrogates=100,
    surrogate_min_shift_ms=1.0,
    sample_dt_s=None,
    time_vector_s=None,
    duration_s=None,
    analysis_start_s=0.0,
    analysis_end_s=None,
    unit_ids=None,
    rng=None,
):
    if direction_mode not in ("legacy_forward", "symmetric_mean"):
        raise ValueError("direction_mode must be 'legacy_forward' or 'symmetric_mean'")
    if expectation_mode not in ("analytic", "surrogate_circular_shift"):
        raise ValueError("expectation_mode must be 'analytic' or 'surrogate_circular_shift'")

    binned = bin_spike_trains(
        spike_data,
        bin_ms=bin_ms,
        sample_dt_s=sample_dt_s,
        time_vector_s=time_vector_s,
        duration_s=duration_s,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
    )
    binary_spikes = binned["binary_spikes"]
    n_units, n_bins = binary_spikes.shape
    if n_units < 2:
        raise ValueError("At least two units are required for pairwise synchrony analysis")

    unit_ids = _coerce_unit_ids(spike_data, unit_ids=unit_ids)
    win_bins = max(1, int(round(float(sync_win_ms) / float(bin_ms))))
    step_bins = max(1, int(round(float(sync_step_ms) / float(bin_ms))))
    lag_bins = max(0, int(round(float(coinc_lag_ms) / float(bin_ms))))
    min_abs_shift_bins = max(1, int(round(float(surrogate_min_shift_ms) / float(bin_ms))))
    starts, stops = sliding_window_indices(n_bins, win_bins, step_bins)
    pair_list = list(combinations(range(n_units), 2))
    n_pairs = len(pair_list)
    if n_pairs < 1:
        raise ValueError("No unordered MU pairs available for synchrony analysis")

    sync_trace = np.zeros(len(starts), dtype=float)
    expected_trace_mean = np.zeros(len(starts), dtype=float)
    expected_trace_std = np.zeros(len(starts), dtype=float)
    t_sync_abs = np.zeros(len(starts), dtype=float)
    rng = np.random.default_rng() if rng is None else rng

    for window_i, (start, stop) in enumerate(zip(starts, stops)):
        window_pair_sync = np.zeros(n_pairs, dtype=float)
        window_pair_expected = np.zeros(n_pairs, dtype=float)
        window_pair_expected_std = np.zeros(n_pairs, dtype=float)
        t_sync_abs[window_i] = float(np.mean(binned["time_bin_centers_abs_s"][start:stop]))
        for pair_i, (ix, iy) in enumerate(pair_list):
            x = binary_spikes[ix, start:stop]
            y = binary_spikes[iy, start:stop]
            if direction_mode == "legacy_forward":
                obs_used = pairwise_coincidence_count_forward(x, y, lag_bins)
            else:
                obs_used = pairwise_coincidence_count_symmetric(x, y, lag_bins)

            if expectation_mode == "analytic":
                expected_used = expected_coincidence_analytic(x, y, lag_bins)
                expected_std = 0.0
            else:
                expected_used, expected_std = expected_coincidence_surrogate_circular_shift(
                    x,
                    y,
                    lag_bins,
                    direction_mode=direction_mode,
                    n_surrogates=n_surrogates,
                    min_abs_shift_bins=min_abs_shift_bins,
                    rng=rng,
                )
            window_pair_expected[pair_i] = expected_used
            window_pair_expected_std[pair_i] = expected_std
            window_pair_sync[pair_i] = pairwise_sync_index(obs_used, expected_used)

        sync_trace[window_i] = float(np.mean(window_pair_sync))
        expected_trace_mean[window_i] = float(np.mean(window_pair_expected))
        expected_trace_std[window_i] = float(np.mean(window_pair_expected_std))

    return {
        "t_sync": t_sync_abs - float(cue_time_sec),
        "t_sync_absolute_sec": t_sync_abs,
        "sync_trace": sync_trace,
        "expected_trace_mean": expected_trace_mean,
        "expected_trace_std": expected_trace_std,
        "analysis_start_rel_cue_sec": float(binned["analysis_start_s"] - float(cue_time_sec)),
        "analysis_end_rel_cue_sec": float(binned["analysis_end_s"] - float(cue_time_sec)),
        "cue_time_sec": float(cue_time_sec),
        "n_units": int(n_units),
        "n_pairs": int(n_pairs),
        "unit_ids": np.asarray(unit_ids),
        "direction_mode": direction_mode,
        "expectation_mode": expectation_mode,
        "bin_ms": float(bin_ms),
        "sync_win_ms": float(sync_win_ms),
        "sync_step_ms": float(sync_step_ms),
        "coinc_lag_ms": float(coinc_lag_ms),
        "n_surrogates": int(n_surrogates),
        "surrogate_min_shift_ms": float(surrogate_min_shift_ms),
    }


def summarize_sync_trace(sync_result, baseline_win=(-3.0, -2.0), post_win=(0.3, 0.6)):
    t_sync = np.asarray(sync_result["t_sync"], dtype=float)
    sync_trace = np.asarray(sync_result["sync_trace"], dtype=float)
    baseline_win = tuple(map(float, baseline_win))
    post_win = tuple(map(float, post_win))
    baseline_mask = (t_sync >= baseline_win[0]) & (t_sync <= baseline_win[1])
    post_mask = (t_sync >= post_win[0]) & (t_sync <= post_win[1])
    if not np.any(baseline_mask):
        raise ValueError("Baseline window does not contain any synchrony samples")
    if not np.any(post_mask):
        raise ValueError("Post window does not contain any synchrony samples")

    baseline_values = sync_trace[baseline_mask]
    post_times = t_sync[post_mask]
    post_values = sync_trace[post_mask]
    baseline_mean = float(np.nanmean(baseline_values))
    baseline_std = float(np.nanstd(baseline_values))
    if baseline_std > 1e-12 and np.isfinite(baseline_std):
        sync_trace_zscore = (sync_trace - baseline_mean) / baseline_std
    else:
        sync_trace_zscore = np.zeros_like(sync_trace, dtype=float)

    peak_local_idx = int(np.nanargmax(post_values))
    post_peak = float(post_values[peak_local_idx])
    post_peak_time_sec = float(post_times[peak_local_idx])
    half_height = baseline_mean + 0.5 * (post_peak - baseline_mean)
    above_half = np.isfinite(post_values) & (post_values >= half_height)
    if np.any(above_half):
        peak_post_idx = peak_local_idx
        mask_post = np.isfinite(post_values) & (post_values >= half_height)
        left = peak_post_idx
        right = peak_post_idx
        while left - 1 >= 0 and mask_post[left - 1]:
            left -= 1
        while right + 1 < len(post_values) and mask_post[right + 1]:
            right += 1
        step_s = float(np.median(np.diff(t_sync))) if len(t_sync) > 1 else 0.0
        post_peak_width_sec = float((post_times[right] - post_times[left]) + step_s)
    else:
        post_peak_width_sec = 0.0

    post_integral_above_baseline = float(np.trapz(np.maximum(post_values - baseline_mean, 0.0), post_times))
    return {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "post_peak": post_peak,
        "post_peak_time_sec": post_peak_time_sec,
        "post_peak_width_sec": post_peak_width_sec,
        "post_integral_above_baseline": post_integral_above_baseline,
        "sync_peak_rel_base": float(post_peak - baseline_mean),
        "sync_mean_rel_base": float(np.nanmean(post_values) - baseline_mean),
        "baseline_win_rel_cue_s": np.asarray(baseline_win, dtype=float),
        "post_win_rel_cue_s": np.asarray(post_win, dtype=float),
        "sync_trace_zscore": np.asarray(sync_trace_zscore, dtype=float),
    }


def plot_sync_trace_diagnostic(
    sync_result,
    savepath=None,
    figure_name="ANALYSIS_sync_index_trace.png",
    display_mode="absolute",
):
    t_sync = np.asarray(sync_result["t_sync"], dtype=float)
    display_mode = str(display_mode).lower()
    if display_mode not in ("absolute", "zscore"):
        raise ValueError("display_mode must be 'absolute' or 'zscore'")
    sync_trace = np.asarray(
        sync_result["sync_trace"] if display_mode == "absolute" else sync_result["sync_trace_zscore"],
        dtype=float,
    )
    baseline_win = np.asarray(sync_result["baseline_win_rel_cue_s"], dtype=float)
    post_win = np.asarray(sync_result["post_win_rel_cue_s"], dtype=float)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(t_sync, sync_trace, color="tab:blue", lw=1.8)
    ax.axvspan(baseline_win[0], baseline_win[1], color="0.85", alpha=0.7, label="Baseline window")
    ax.axvspan(post_win[0], post_win[1], color="tab:orange", alpha=0.15, label="Post window")
    ax.axvline(0.0, color="black", lw=1.2, ls="--", label="Cue onset")
    if "analysis_start_rel_cue_sec" in sync_result:
        ax.axvline(float(sync_result["analysis_start_rel_cue_sec"]), color="0.5", lw=1.0, ls=":")
    ax.set_xlabel("Time relative to cue (s)")
    ax.set_ylabel("Synchrony index" if display_mode == "absolute" else "Synchrony index (z)")
    ax.set_title(
        f"Sliding sync_index ({display_mode}) | dir={sync_result['direction_mode']} | "
        f"exp={sync_result['expectation_mode']} | win={sync_result['sync_win_ms']:.0f} ms | "
        f"step={sync_result['sync_step_ms']:.0f} ms | lag={sync_result['coinc_lag_ms']:.0f} ms | "
        f"nMU={sync_result['n_units']}"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if savepath is not None:
        plt.savefig(os.path.join(savepath, figure_name), bbox_inches="tight", dpi=200)
    plt.close(fig)


def run_sliding_sync_index_analysis(
    spike_data,
    *,
    cue_time_sec=0.0,
    baseline_win=(-3.0, -2.0),
    post_win=(0.3, 0.6),
    generate_figure=False,
    savepath=None,
    figure_name="ANALYSIS_sync_index_trace.png",
    display_mode="absolute",
    **kwargs,
):
    sync_result = compute_sliding_sync_trace(
        spike_data,
        cue_time_sec=cue_time_sec,
        **kwargs,
    )
    sync_result.update(summarize_sync_trace(sync_result, baseline_win=baseline_win, post_win=post_win))
    sync_result["sync_trace_display_mode"] = str(display_mode)
    if generate_figure:
        plot_sync_trace_diagnostic(
            sync_result,
            savepath=savepath,
            figure_name=figure_name,
            display_mode=display_mode,
        )
    return sync_result


def get_firing_rate(
    spike_trains_MN,
    spike_trains_RC=None,
    generate_figure=False,
    savepath=None,
    figure_name="ANALYSIS_Firing_rate_and_ISI.png",
):
    """Calculate firing-rate and ISI-CV summaries, with an optional diagnostic figure."""
    firing_rates_MN = {"mean": {}, "std": {}, "max": {}, "min": {}}
    cov_MN = {}
    for mn_id, spike_times in spike_trains_MN.items():
        if len(spike_times) < 2:
            firing_rates_MN["mean"][mn_id] = 0.0
            firing_rates_MN["std"][mn_id] = 0.0
            firing_rates_MN["max"][mn_id] = 0.0
            firing_rates_MN["min"][mn_id] = 0.0
            cov_MN[mn_id] = np.nan
            continue
        times = np.asarray(spike_times)
        inst_rates = 1.0 / np.diff(times)
        firing_rates_MN["mean"][mn_id] = np.mean(inst_rates)
        firing_rates_MN["std"][mn_id] = np.std(inst_rates)
        firing_rates_MN["max"][mn_id] = np.max(inst_rates)
        firing_rates_MN["min"][mn_id] = np.min(inst_rates)
        isis = np.diff(times)
        mean_isi = np.mean(isis)
        raw_cov = (np.std(isis) / mean_isi) if mean_isi > 0 else np.nan
        cov_MN[mn_id] = raw_cov if (not np.isnan(raw_cov) and raw_cov >= 0.01) else 0.0

    firing_rates_RC = {"mean": {}, "std": {}, "max": {}, "min": {}}
    cov_RC = {}
    if spike_trains_RC is not None:
        for rc_id, spike_times in spike_trains_RC.items():
            if len(spike_times) < 2:
                firing_rates_RC["mean"][rc_id] = 0.0
                firing_rates_RC["std"][rc_id] = 0.0
                firing_rates_RC["max"][rc_id] = 0.0
                firing_rates_RC["min"][rc_id] = 0.0
                cov_RC[rc_id] = np.nan
                continue
            times = np.asarray(spike_times)
            inst_rates = 1.0 / np.diff(times)
            firing_rates_RC["mean"][rc_id] = np.mean(inst_rates)
            firing_rates_RC["std"][rc_id] = np.std(inst_rates)
            firing_rates_RC["max"][rc_id] = np.max(inst_rates)
            firing_rates_RC["min"][rc_id] = np.min(inst_rates)
            isis = np.diff(times)
            mean_isi = np.mean(isis)
            raw_cov = np.std(isis) / mean_isi if mean_isi > 0 else np.nan
            cov_RC[rc_id] = raw_cov if (not np.isnan(raw_cov) and raw_cov >= 0.01) else 0.0

    if generate_figure:
        mn_ids = list(firing_rates_MN["mean"].keys())
        fr_means = np.array([firing_rates_MN["mean"][i] for i in mn_ids])
        fr_stds = np.array([firing_rates_MN["std"][i] for i in mn_ids])
        fr_max = np.array([firing_rates_MN["max"][i] for i in mn_ids])
        fr_min = np.array([firing_rates_MN["min"][i] for i in mn_ids])
        idx_MN = np.arange(len(mn_ids))

        isi_means = []
        isi_stds = []
        isi_max = []
        isi_min = []
        for mn_id in mn_ids:
            times = np.asarray(spike_trains_MN[mn_id])
            if len(times) < 2:
                isi_means.append(0.0)
                isi_stds.append(0.0)
                isi_max.append(0.0)
                isi_min.append(0.0)
            else:
                isis = np.diff(times)
                isi_means.append(np.mean(isis))
                isi_stds.append(np.std(isis))
                isi_max.append(np.max(isis))
                isi_min.append(np.min(isis))
        isi_means = np.array(isi_means)
        isi_stds = np.array(isi_stds)
        isi_max = np.array(isi_max)
        isi_min = np.array(isi_min)

        rc_ids = list(firing_rates_RC["mean"].keys()) if spike_trains_RC is not None else []
        cov_vals_RC = np.array([cov_RC[r] for r in rc_ids]) if spike_trains_RC is not None and cov_RC else np.array([])

        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(
            nrows=2,
            ncols=3,
            width_ratios=[3, 1, 1],
            height_ratios=[1, 1],
            hspace=0.4,
            wspace=0.3,
        )

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(idx_MN, fr_means, color="C1", label="Mean FR")
        ax0.fill_between(idx_MN, fr_means - fr_stds, fr_means + fr_stds, color="C1", alpha=0.3, label="+/- STD")
        ax0.plot(idx_MN, fr_max, color="C1", linestyle="--", label="Max FR")
        ax0.plot(idx_MN, fr_min, color="C1", linestyle=":", label="Min FR")
        ax0.set_xlabel("Motoneuron index")
        ax0.set_ylabel("Firing rate (Hz)")
        ax0.set_title("Firing rates across MNs")
        ax0.legend(loc="upper right")

        fr_means_finite = fr_means[np.isfinite(fr_means)]
        if fr_means_finite.size > 0:
            ax1 = fig.add_subplot(gs[0, 1])
            ax1.hist(fr_means_finite, bins="auto", color="C1", edgecolor="k", alpha=0.7)
            ax1.set_xlabel("Mean firing rate (Hz)")
            ax1.set_ylabel("Count")
            ax1.set_title("MN mean-FR histogram")

        if spike_trains_RC is not None:
            fr_means_RC = np.array([firing_rates_RC["mean"][r] for r in rc_ids])
            fr_means_RC = fr_means_RC[np.isfinite(fr_means_RC)]
            if fr_means_RC.size > 0 and fr_means_RC[fr_means_RC > 0.1].size > 0:
                ax2 = fig.add_subplot(gs[0, 2])
                ax2.hist(fr_means_RC, bins="auto", color="C4", edgecolor="k", alpha=0.7)
                ax2.set_xlabel("Mean firing rate (Hz)")
                ax2.set_ylabel("Count")
                ax2.set_title("RC mean-FR histogram")

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(idx_MN, isi_means, color="C2", label="Mean ISI")
        ax3.fill_between(idx_MN, isi_means - isi_stds, isi_means + isi_stds, color="C2", alpha=0.3, label="+/- STD")
        ax3.plot(idx_MN, isi_max, color="C2", linestyle="--", label="Max ISI")
        ax3.plot(idx_MN, isi_min, color="C2", linestyle=":", label="Min ISI")
        ax3.set_xlabel("Motoneuron index")
        ax3.set_ylabel("Inter-spike interval (s)")
        ax3.set_title("ISIs across MNs")
        ax3.legend(loc="upper right")
        if isi_max.size and np.isfinite(isi_max).any():
            ax3.set_ylim(bottom=0, top=np.min(np.array([1, np.nanmax(isi_max)])))

        cov_vals_MN = np.array([cov_MN[m] for m in mn_ids])
        cov_vals_MN = cov_vals_MN[np.isfinite(cov_vals_MN)]
        if cov_vals_MN.size > 0:
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.hist(cov_vals_MN, bins="auto", color="C2", edgecolor="k", alpha=0.7)
            ax4.set_xlabel("ISI CV (std/mean)")
            ax4.set_ylabel("Count")
            ax4.set_title("MN ISI-CV histogram")

        cov_vals_RC = cov_vals_RC[np.isfinite(cov_vals_RC)]
        if cov_vals_RC.size > 0:
            ax5 = fig.add_subplot(gs[1, 2])
            ax5.hist(cov_vals_RC, bins="auto", color="C4", edgecolor="k", alpha=0.7)
            ax5.set_xlabel("ISI CV (std/mean)")
            ax5.set_ylabel("Count")
            ax5.set_title("RC ISI-CV histogram")

        if savepath is not None:
            plt.savefig(os.path.join(savepath, figure_name), bbox_inches="tight")
        plt.close(fig)

    return firing_rates_MN, firing_rates_RC, cov_MN, cov_RC
