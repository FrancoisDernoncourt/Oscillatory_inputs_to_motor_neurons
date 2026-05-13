from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from experimental_hdf5_analysis import (
    _compute_fr_and_isi_cv_from_spikes,
    _extract_saved_rate_isi_arrays,
    _interp_to_reference,
    collect_experimental_batch_runs,
    load_experimental_trial,
)


DEFAULT_ACROSS_CONDITION_STYLE = {
    "figure_size": (22, 18),
    "figure_dpi": 130,
    "curve_width_ratios": (2.0, 1.0),
    "projection_width_ratios": (1.0, 1.0),
    "condition_wspace": 0.18,
    "pair_wspace": 0.22,
    "row_hspace": 0.38,
    "participant_trace_lw": 1.0,
    "participant_trace_alpha": 0.18,
    "mean_trace_lw": 2.0,
    "ready_lw": 1.0,
    "ready_alpha": 0.9,
    "go_lw": 1.2,
    "go_alpha": 0.9,
    "event_color": "#7A7A7A",
    "reference_line_color": "#7A7A7A",
    "reference_line_lw": 1.0,
    "reference_line_alpha": 0.9,
    "time_tick_step_s": 0.5,
    "violin_width": 0.7,
    "violin_alpha": 0.18,
    "violin_edge_lw": 1.2,
    "feature_mean_lw": 2.2,
    "feature_mean_alpha": 0.95,
    "feature_mean_half_width": 0.28,
    "feature_point_size": 34,
    "feature_point_alpha": 0.85,
    "feature_point_jitter": 0.08,
    "zero_line_color": "#7A7A7A",
    "zero_line_lw": 1.0,
    "zero_line_alpha": 0.9,
    "condition_title_fontsize": 13,
    "condition_subtitle_fontsize": 10,
    "panel_title_fontsize": 10,
    "axis_label_fontsize": 10,
    "tick_labelsize": 9,
    "projection_marker_size": 24,
    "projection_alpha": 0.75,
    "projection_plane_alpha": 0.06,
    "projection_grid_alpha": 0.18,
    "projection_edge_color": "#555555",
    "projection_edge_lw": 1.4,
    "projection_edge_alpha": 0.9,
    "projection_elev": 22,
    "projection_azim": -54,
}


DEFAULT_ROW_COLORS = {
    "cst_lf": "#ff3300",
    "cst_alpha_mod": "#ffa600",
    "cst_beta_mod": "#008cff",
    "sync_trace": "#7e2179",
    "firing_rate": "#FF7300",
    "isi_cv": "#00CFAD",
}


ROW_SPECS = [
    {
        "key": "cst_lf",
        "time_key": "t_rel_cue_s",
        "title": "CST low-pass",
        "ylabel": "spikes/s/MU",
        "burst_col": "OBS_BURST_FEATURES_resid_cst_low_frequency",
        "trend_col": "OBS_TREND_FEATURE_cst_low_frequency",
    },
    {
        "key": "cst_alpha_mod",
        "time_key": "t_rel_cue_s",
        "title": "Alpha modulation",
        "ylabel": "z",
        "burst_col": "OBS_BURST_FEATURES_resid_cst_alpha_modulation_hilbert_zscore",
        "trend_col": "OBS_TREND_FEATURE_cst_alpha_modulation_hilbert_zscore",
    },
    {
        "key": "cst_beta_mod",
        "time_key": "t_rel_cue_s",
        "title": "Beta modulation",
        "ylabel": "z",
        "burst_col": "OBS_BURST_FEATURES_resid_cst_beta_modulation_hilbert_zscore",
        "trend_col": "OBS_TREND_FEATURE_cst_beta_modulation_hilbert_zscore",
    },
    {
        "key": "sync_trace",
        "time_key": "sync_time_rel",
        "title": "Synchrony",
        "ylabel": "Excess coincidence index",
        "burst_col": "OBS_BURST_FEATURES_resid_sync_coincidence_curve",
        "trend_col": "OBS_TREND_FEATURE_sync_coincidence_curve",
    },
]


PROJECTION_SPECS = [
    {
        "title": "Firing-rate moments",
        "color_key": "firing_rate",
        "columns": (
            "OBS_FIRING_STATISTICS_firing_rate_mean",
            "OBS_FIRING_STATISTICS_firing_rate_sd",
            "OBS_FIRING_STATISTICS_firing_rate_skew",
        ),
        "labels": ("Mean (Hz)", "SD (Hz)", "Skewness"),
    },
    {
        "title": "ISI-CV moments",
        "color_key": "isi_cv",
        "columns": (
            "OBS_FIRING_STATISTICS_isi_cv_mean",
            "OBS_FIRING_STATISTICS_isi_cv_sd",
            "OBS_FIRING_STATISTICS_isi_cv_skew",
        ),
        "labels": ("Mean (CV)", "SD (CV)", "Skewness"),
    },
]


def _resolve_style(style=None):
    resolved = dict(DEFAULT_ACROSS_CONDITION_STYLE)
    if style:
        resolved.update(style)
    return resolved


def _resolve_colors(colors=None):
    resolved = dict(DEFAULT_ROW_COLORS)
    if colors:
        resolved.update(colors)
    return resolved


def _load_export_table(export_table_path):
    export_table_path = Path(export_table_path)
    if export_table_path.suffix.lower() == ".pkl":
        return pd.read_pickle(export_table_path)
    if export_table_path.suffix.lower() == ".csv":
        return pd.read_csv(export_table_path)
    raise ValueError(f"Unsupported export table format: {export_table_path.suffix}")


def _auto_limits(values, zero_floor=False, pad_frac=0.08):
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0) if not zero_floor else (0.0, 1.0)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if zero_floor:
        lo = min(0.0, lo)
        hi = max(0.0, hi)
    if np.isclose(lo, hi):
        span = max(abs(lo), 1.0)
        return lo - 0.2 * span, hi + 0.2 * span
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def _aggregate_subject_traces(runs, signal_key, time_key):
    reference_time = None
    aligned = []
    for run in runs:
        signal = run.get(signal_key)
        time = run.get(time_key)
        if signal is None or time is None:
            continue
        time = np.asarray(time, dtype=float)
        signal = np.asarray(signal, dtype=float)
        if reference_time is None:
            reference_time = time
            aligned.append(signal)
        else:
            aligned.append(_interp_to_reference(time, signal, reference_time))
    if reference_time is None or not aligned:
        return None, None
    stack = np.stack(aligned, axis=0)
    return np.asarray(reference_time, dtype=float), np.nanmean(stack, axis=0)


def _load_rate_isi_arrays_from_run(run):
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


def _build_subject_records(analyzed_root, export_df, condition_order, participant_ids=None, session_tags=None):
    grouped_runs = collect_experimental_batch_runs(
        analyzed_root,
        participant_ids=participant_ids,
        session_tags=session_tags,
        condition_ids=condition_order,
    )
    records = []
    for _, runs in grouped_runs.items():
        if not runs:
            continue
        ref = runs[0]
        participant_id, session_tag, condition_id = ref["batch_key"]
        row_match = export_df[
            (export_df["DATASET_CONTEXT_participant_id"].astype(str) == str(participant_id))
            & (export_df["DATASET_CONTEXT_session_tag"].astype(str) == str(session_tag))
            & (export_df["DATASET_CONTEXT_condition_id"].astype(str) == str(condition_id))
        ]
        export_row = row_match.iloc[0].to_dict() if not row_match.empty else {}
        subject_record = {
            "participant_id": participant_id,
            "session_tag": session_tag,
            "condition_id": condition_id,
            "subject_key": f"{participant_id} | {session_tag}",
            "n_trials": int(len(runs)),
            "event_times": dict(ref["event_times"]),
            "features": export_row,
        }
        for spec in ROW_SPECS:
            time_rel, mean_trace = _aggregate_subject_traces(runs, spec["key"], spec["time_key"])
            subject_record[f"{spec['key']}_time"] = time_rel
            subject_record[spec["key"]] = mean_trace
        fr_values = []
        isi_values = []
        for run in runs:
            run_fr, run_isi = _load_rate_isi_arrays_from_run(run)
            if run_fr is not None:
                run_fr = np.asarray(run_fr, dtype=float)
                run_fr = run_fr[np.isfinite(run_fr)]
                if run_fr.size:
                    fr_values.append(run_fr)
            if run_isi is not None:
                run_isi = np.asarray(run_isi, dtype=float)
                run_isi = run_isi[np.isfinite(run_isi)]
                if run_isi.size:
                    isi_values.append(run_isi)
        subject_record["firing_rate_values"] = np.concatenate(fr_values) if fr_values else np.asarray([], dtype=float)
        subject_record["isi_cv_values"] = np.concatenate(isi_values) if isi_values else np.asarray([], dtype=float)
        records.append(subject_record)
    return records


def _compute_row_limits(records_by_condition, display_window):
    limits = {"trace": {}, "feature": {}}
    for spec in ROW_SPECS:
        trace_values = []
        feature_values = []
        for records in records_by_condition.values():
            for record in records:
                time = record.get(f"{spec['key']}_time")
                trace = record.get(spec["key"])
                if time is not None and trace is not None:
                    time = np.asarray(time, dtype=float)
                    trace = np.asarray(trace, dtype=float)
                    mask = (time >= display_window[0]) & (time <= display_window[1])
                    if np.any(mask):
                        trace_values.append(trace[mask])
                feature_row = record.get("features", {})
                feature_values.append(np.asarray([feature_row.get(spec["burst_col"], np.nan), feature_row.get(spec["trend_col"], np.nan)], dtype=float))
        limits["trace"][spec["key"]] = _auto_limits(np.concatenate(trace_values) if trace_values else np.asarray([0.0]), zero_floor=False)
        limits["feature"][spec["key"]] = _auto_limits(np.concatenate(feature_values) if feature_values else np.asarray([0.0]), zero_floor=False)
    return limits


def _compute_projection_limits(records_by_condition):
    limits = {}
    for proj in PROJECTION_SPECS:
        per_axis = [[], [], []]
        for records in records_by_condition.values():
            for record in records:
                row = record.get("features", {})
                for axis_idx, column_name in enumerate(proj["columns"]):
                    per_axis[axis_idx].append(row.get(column_name, np.nan))
        limits[proj["title"]] = tuple(_auto_limits(axis_values, zero_floor=False) for axis_values in per_axis)
    return limits


def _merge_ylim_overrides(default_limits, ylim_overrides=None):
    merged = {
        "trace": dict(default_limits.get("trace", {})),
        "feature": dict(default_limits.get("feature", {})),
        "projection": dict(default_limits.get("projection", {})),
    }
    if not ylim_overrides:
        return merged
    for section in ("trace", "feature"):
        for key, value in (ylim_overrides.get(section, {}) or {}).items():
            if value is not None:
                merged[section][key] = tuple(map(float, value))
    for key, value in (ylim_overrides.get("projection", {}) or {}).items():
        if value is not None:
            merged["projection"][key] = tuple(tuple(map(float, axis_limits)) for axis_limits in value)
    return merged


def _draw_time_markers(ax, event_times, style):
    ready_s = event_times.get("get_ready_cue_s")
    cue_s = event_times.get("go_nogo_cue_s")
    if ready_s is not None:
        ax.axvline(float(ready_s), color=style["event_color"], ls="--", lw=style["ready_lw"], alpha=style["ready_alpha"], zorder=1)
    if cue_s is not None:
        ax.axvline(float(cue_s), color=style["event_color"], ls="-", lw=style["go_lw"], alpha=style["go_alpha"], zorder=1)


def _plot_trace_panel(ax, records, spec, color, display_window, y_limits, style, reference_window=None):
    plotted = []
    ref_time = None
    for record in records:
        time = record.get(f"{spec['key']}_time")
        trace = record.get(spec["key"])
        if time is None or trace is None:
            continue
        time = np.asarray(time, dtype=float)
        trace = np.asarray(trace, dtype=float)
        mask = (time >= display_window[0]) & (time <= display_window[1])
        if not np.any(mask):
            continue
        time_zoom = time[mask]
        trace_zoom = trace[mask]
        ref_time = time_zoom if ref_time is None else ref_time
        plotted.append((time_zoom, trace_zoom))
        ax.plot(time_zoom, trace_zoom, color=color, lw=style["participant_trace_lw"], alpha=style["participant_trace_alpha"], zorder=2)
    if plotted:
        common_time = plotted[0][0]
        mean_stack = []
        for time_zoom, trace_zoom in plotted:
            if time_zoom.shape == common_time.shape and np.allclose(time_zoom, common_time):
                mean_stack.append(trace_zoom)
            else:
                mean_stack.append(np.interp(common_time, time_zoom, trace_zoom, left=np.nan, right=np.nan))
        mean_trace = np.nanmean(np.stack(mean_stack, axis=0), axis=0)
        ax.plot(common_time, mean_trace, color=color, lw=style["mean_trace_lw"], alpha=1.0, zorder=4)
        _draw_time_markers(ax, records[0]["event_times"], style)
        if spec["key"] in ("cst_alpha_mod", "cst_beta_mod"):
            ax.axhline(
                0.0,
                color=style["reference_line_color"],
                lw=style["reference_line_lw"],
                ls="--",
                alpha=style["reference_line_alpha"],
                zorder=1,
            )
        if reference_window is not None and spec["key"] in ("cst_lf", "sync_trace"):
            ref_start_s, ref_end_s = map(float, reference_window)
            ref_mask = (common_time >= ref_start_s) & (common_time <= ref_end_s)
            if np.any(ref_mask):
                ref_value = float(np.nanmean(mean_trace[ref_mask]))
                ax.axhline(
                    ref_value,
                    color=style["reference_line_color"],
                    lw=style["reference_line_lw"],
                    ls="--",
                    alpha=style["reference_line_alpha"],
                    zorder=1,
                )
    ax.set_xlim(*map(float, display_window))
    ax.set_ylim(*map(float, y_limits))
    ax.set_ylabel(spec["ylabel"], fontsize=style["axis_label_fontsize"])
    xticks = np.arange(
        np.floor(display_window[0] / style["time_tick_step_s"]) * style["time_tick_step_s"],
        display_window[1] + 0.5 * style["time_tick_step_s"],
        style["time_tick_step_s"],
        dtype=float,
    )
    ax.set_xticks(xticks)
    ax.tick_params(axis="both", labelsize=style["tick_labelsize"])
    ax.grid(alpha=0.24)
    ax.set_title(spec["title"], loc="left", fontsize=style["panel_title_fontsize"])


def _plot_feature_panel(ax, records, spec, color, y_limits, style):
    burst_values = np.asarray([record["features"].get(spec["burst_col"], np.nan) for record in records], dtype=float)
    trend_values = np.asarray([record["features"].get(spec["trend_col"], np.nan) for record in records], dtype=float)
    datasets = [burst_values[np.isfinite(burst_values)], trend_values[np.isfinite(trend_values)]]
    positions = [1.0, 2.0]
    valid_positions = [pos for pos, vals in zip(positions, datasets) if vals.size > 0]
    valid_data = [vals for vals in datasets if vals.size > 0]
    if valid_data:
        violin = ax.violinplot(valid_data, positions=valid_positions, widths=style["violin_width"], showmeans=False, showextrema=False, showmedians=False)
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(style["violin_alpha"])
            body.set_linewidth(style["violin_edge_lw"])
    rng = np.random.default_rng(0)
    for pos, values in zip(positions, datasets):
        if values.size == 0:
            continue
        jitter = rng.uniform(-style["feature_point_jitter"], style["feature_point_jitter"], size=values.size)
        ax.scatter(
            np.full(values.shape, pos, dtype=float) + jitter,
            values,
            s=style["feature_point_size"],
            alpha=style["feature_point_alpha"],
            color=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
        mean_value = float(np.mean(values))
        half_width = float(style["feature_mean_half_width"])
        ax.hlines(
            mean_value,
            xmin=pos - half_width,
            xmax=pos + half_width,
            color=color,
            lw=style["feature_mean_lw"],
            alpha=style["feature_mean_alpha"],
            zorder=5,
        )
    ax.axhline(0.0, color=style["zero_line_color"], lw=style["zero_line_lw"], ls="--", alpha=style["zero_line_alpha"], zorder=1)
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(*map(float, y_limits))
    ax.set_xticks(positions)
    ax.set_xticklabels(["Burst", "Trend"], rotation=0)
    ax.tick_params(axis="both", labelsize=style["tick_labelsize"])
    ax.grid(alpha=0.22, axis="y")


def _plot_projection_panel(ax, records, proj_spec, color, limits, style):
    x_col, y_col, z_col = proj_spec["columns"]
    xyz = np.asarray(
        [
            [
                record["features"].get(x_col, np.nan),
                record["features"].get(y_col, np.nan),
                record["features"].get(z_col, np.nan),
            ]
            for record in records
        ],
        dtype=float,
    )
    finite_mask = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite_mask]
    (xlim, ylim, zlim) = limits
    ax.view_init(elev=style["projection_elev"], azim=style["projection_azim"])
    if xyz.size:
        ax.scatter(xyz[:, 0], xyz[:, 1], np.full(xyz.shape[0], zlim[0]), s=style["projection_marker_size"], color=color, alpha=style["projection_alpha"], depthshade=False)
        ax.scatter(xyz[:, 0], np.full(xyz.shape[0], ylim[1]), xyz[:, 2], s=style["projection_marker_size"], color=color, alpha=style["projection_alpha"], depthshade=False)
        ax.scatter(np.full(xyz.shape[0], xlim[0]), xyz[:, 1], xyz[:, 2], s=style["projection_marker_size"], color=color, alpha=style["projection_alpha"], depthshade=False)
    ax.set_xlim(*map(float, xlim))
    ax.set_ylim(*map(float, ylim))
    ax.set_zlim(*map(float, zlim))
    ax.xaxis.pane.set_alpha(style["projection_plane_alpha"])
    ax.yaxis.pane.set_alpha(style["projection_plane_alpha"])
    ax.zaxis.pane.set_alpha(style["projection_plane_alpha"])
    ax.grid(alpha=style["projection_grid_alpha"])
    ax.set_xlabel(proj_spec["labels"][0], fontsize=style["tick_labelsize"])
    ax.set_ylabel(proj_spec["labels"][1], fontsize=style["tick_labelsize"])
    ax.set_zlabel(proj_spec["labels"][2], fontsize=style["tick_labelsize"])
    ax.tick_params(axis="both", labelsize=style["tick_labelsize"] - 1)
    ax.set_title(proj_spec["title"], loc="left", fontsize=style["panel_title_fontsize"])
    edge_color = style["projection_edge_color"]
    edge_lw = style["projection_edge_lw"]
    edge_alpha = style["projection_edge_alpha"]
    background_edge_segments = {
        # Floor / bottom projection plane: z = zmin
        ((xlim[0], ylim[0], zlim[0]), (xlim[1], ylim[0], zlim[0])),
        ((xlim[0], ylim[1], zlim[0]), (xlim[1], ylim[1], zlim[0])),
        ((xlim[0], ylim[0], zlim[0]), (xlim[0], ylim[1], zlim[0])),
        ((xlim[1], ylim[0], zlim[0]), (xlim[1], ylim[1], zlim[0])),
        # Back wall projection plane: y = ymax
        ((xlim[0], ylim[1], zlim[1]), (xlim[1], ylim[1], zlim[1])),
        ((xlim[0], ylim[1], zlim[0]), (xlim[0], ylim[1], zlim[1])),
        ((xlim[1], ylim[1], zlim[0]), (xlim[1], ylim[1], zlim[1])),
        # Side wall projection plane: x = xmin
        ((xlim[0], ylim[0], zlim[1]), (xlim[0], ylim[1], zlim[1])),
        ((xlim[0], ylim[0], zlim[0]), (xlim[0], ylim[0], zlim[1])),
    }
    for (x0, y0, z0), (x1, y1, z1) in sorted(background_edge_segments):
        ax.plot([x0, x1], [y0, y1], [z0, z1], color=edge_color, lw=edge_lw, alpha=edge_alpha, zorder=6)


def plot_experimental_across_conditions_summary(
    analyzed_root,
    export_table_path,
    *,
    output_path=None,
    condition_order=None,
    display_window_rel_cue_s=(-4.0, 1.0),
    participant_ids=None,
    session_tags=None,
    row_colors=None,
    style=None,
    ylim_overrides=None,
    trace_reference_window_rel_cue_s=(-4.0, -2.0),
    show_progress=False,
):
    analyzed_root = Path(analyzed_root)
    export_df = _load_export_table(export_table_path)
    condition_order = list(condition_order or ["HAND_GO", "HAND_NOGO", "TA_NOGO"])
    style = _resolve_style(style)
    colors = _resolve_colors(row_colors)

    records = _build_subject_records(
        analyzed_root,
        export_df,
        condition_order=condition_order,
        participant_ids=participant_ids,
        session_tags=session_tags,
    )
    records_by_condition = {
        condition: [record for record in records if str(record["condition_id"]) == str(condition)]
        for condition in condition_order
    }

    row_limits = _compute_row_limits(records_by_condition, tuple(map(float, display_window_rel_cue_s)))
    projection_limits = _compute_projection_limits(records_by_condition)
    all_limits = _merge_ylim_overrides(
        {"trace": row_limits["trace"], "feature": row_limits["feature"], "projection": projection_limits},
        ylim_overrides=ylim_overrides,
    )

    fig = plt.figure(figsize=style["figure_size"], dpi=style["figure_dpi"])
    outer = fig.add_gridspec(
        nrows=5,
        ncols=len(condition_order),
        wspace=style["condition_wspace"],
        hspace=style["row_hspace"],
    )

    top_axes_pairs = []
    for ci, condition in enumerate(condition_order):
        condition_records = records_by_condition.get(condition, [])
        for ri, spec in enumerate(ROW_SPECS):
            pair_gs = outer[ri, ci].subgridspec(
                1,
                2,
                width_ratios=style["curve_width_ratios"],
                wspace=style["pair_wspace"],
            )
            ax_trace = fig.add_subplot(pair_gs[0, 0])
            ax_feature = fig.add_subplot(pair_gs[0, 1])
            _plot_trace_panel(
                ax_trace,
                condition_records,
                spec,
                color=colors[spec["key"]],
                display_window=tuple(map(float, display_window_rel_cue_s)),
                y_limits=all_limits["trace"][spec["key"]],
                style=style,
                reference_window=trace_reference_window_rel_cue_s,
            )
            _plot_feature_panel(
                ax_feature,
                condition_records,
                spec,
                color=colors[spec["key"]],
                y_limits=all_limits["feature"][spec["key"]],
                style=style,
            )
            if ci > 0:
                ax_trace.set_ylabel("")
            if ci < len(condition_order) - 1:
                ax_feature.tick_params(axis="y", labelleft=False)
            if ri < len(ROW_SPECS) - 1:
                ax_trace.set_xlabel("")
                ax_feature.set_xlabel("")
            else:
                ax_trace.set_xlabel("Time relative to go / no-go cue (s)", fontsize=style["axis_label_fontsize"])
                ax_feature.set_xlabel("Feature", fontsize=style["axis_label_fontsize"])
            if ri == 0:
                top_axes_pairs.append((ax_trace, ax_feature, condition_records, condition))

        pair_gs = outer[4, ci].subgridspec(
            1,
            2,
            width_ratios=style["projection_width_ratios"],
            wspace=style["pair_wspace"],
        )
        for pj, proj_spec in enumerate(PROJECTION_SPECS):
            ax_proj = fig.add_subplot(pair_gs[0, pj], projection="3d")
            _plot_projection_panel(
                ax_proj,
                condition_records,
                proj_spec,
                color=colors[proj_spec["color_key"]],
                limits=all_limits["projection"][proj_spec["title"]],
                style=style,
            )

    fig.canvas.draw()
    for ax_left, ax_right, condition_records, condition in top_axes_pairs:
        bbox_left = ax_left.get_position()
        bbox_right = ax_right.get_position()
        x_center = 0.5 * (bbox_left.x0 + bbox_right.x1)
        y_top = max(bbox_left.y1, bbox_right.y1)
        n_participants = len({str(record["participant_id"]) for record in condition_records})
        total_trials = int(sum(int(record["n_trials"]) for record in condition_records))
        fig.text(
            x_center,
            y_top + 0.02,
            f"{condition}\n n participants = {n_participants} | total trials = {total_trials}",
            ha="center",
            va="bottom",
            fontsize=style["condition_title_fontsize"],
        )

    fig.suptitle("Across-condition experimental summary", fontsize=style["condition_title_fontsize"] + 2, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))

    figure_path = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        figure_path = output_path
    return fig, {
        "figure_path": None if figure_path is None else str(figure_path),
        "n_subject_condition_records": len(records),
    }
