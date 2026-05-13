from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from experimental_across_condition_summary import (
    DEFAULT_ACROSS_CONDITION_STYLE,
    DEFAULT_ROW_COLORS,
    PROJECTION_SPECS,
    ROW_SPECS,
)
from parameter_set_batch_summary import (
    _interp_to_reference,
    collect_parameter_set_runs,
)


DEFAULT_SIM_ACROSS_CONDITION_STYLE = dict(DEFAULT_ACROSS_CONDITION_STYLE)
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mean_trace_linestyle", "-")
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mode_trace_color", "#111111")
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mode_trace_lw", 1.4)
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mode_trace_alpha", 1.0)
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mode_trace_linestyle", ":")
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mode_feature_marker", "X")
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mode_feature_marker_size", 90.0)
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mode_feature_marker_alpha", 1.0)
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mode_feature_marker_facecolor", "row")
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mode_feature_marker_edgecolor", "#111111")
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mode_feature_marker_edge_lw", 1.0)
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mode_projection_marker", "D")
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mode_projection_marker_size", 72.0)
DEFAULT_SIM_ACROSS_CONDITION_STYLE.setdefault("mode_projection_marker_alpha", 1.0)


def _resolve_style(style=None):
    resolved = dict(DEFAULT_SIM_ACROSS_CONDITION_STYLE)
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


def _parameter_set_id_from_name(path_like):
    name = Path(path_like).name
    if not name.startswith("parameter_set_"):
        raise ValueError(f"Could not parse parameter set id from {name!r}")
    if name.startswith("parameter_set_posterior_mode__"):
        raise ValueError(f"Folder {name!r} is a posterior-mode folder, not a numeric parameter set")
    return int(name.split("_")[-1])


def _mode_label_from_parameter_set_name(path_like):
    name = Path(path_like).name
    prefix = "parameter_set_posterior_mode__"
    if not name.startswith(prefix):
        return None
    return name[len(prefix) :]


def _resolve_summary_table_path(batch_root, summary_subdir_name):
    batch_root = Path(batch_root)
    summary_dir = batch_root / summary_subdir_name
    if not summary_dir.exists():
        raise FileNotFoundError(f"Missing summary subfolder: {summary_dir}")
    table_path = summary_dir / "parameter_set_feature_summary.pkl"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing exported summary table: {table_path}")
    return table_path


def _coerce_condition_batch_roots(condition_batch_roots, condition_order=None):
    if not condition_batch_roots:
        raise ValueError("condition_batch_roots must not be empty")
    normalized = {str(key): Path(value) for key, value in condition_batch_roots.items()}
    if condition_order is None:
        return normalized, list(normalized.keys())
    missing = [condition for condition in condition_order if str(condition) not in normalized]
    if missing:
        raise KeyError(f"Missing condition roots for: {missing}")
    return normalized, [str(condition) for condition in condition_order]


def _relative_event_times(event_times):
    event_times = dict(event_times or {})
    go_cue_s = float(event_times.get("go_nogo_cue_s", 0.0))
    ready_s = event_times.get("get_ready_cue_s")
    return {
        "get_ready_cue_s": None if ready_s is None else float(ready_s) - go_cue_s,
        "go_nogo_cue_s": 0.0,
    }


def _trace_time_and_signal(run, signal_key):
    cue_time_s = float(run["event_times"].get("go_nogo_cue_s", 0.0))
    if signal_key == "sync_trace":
        time = run.get("sync_time_abs")
        signal = run.get("sync_trace")
    else:
        time = run.get("t_s")
        signal = run.get(signal_key)
    if time is None or signal is None:
        return None, None
    time = np.asarray(time, dtype=float) - cue_time_s
    signal = np.asarray(signal, dtype=float)
    return time, signal


def _aggregate_parameter_set_trace(runs, signal_key):
    reference_time = None
    aligned_signals = []
    for run in runs:
        time_rel, signal = _trace_time_and_signal(run, signal_key)
        if time_rel is None or signal is None:
            continue
        if reference_time is None:
            reference_time = np.asarray(time_rel, dtype=float)
            aligned_signals.append(np.asarray(signal, dtype=float))
        else:
            aligned_signals.append(_interp_to_reference(time_rel, signal, reference_time))
    if reference_time is None or not aligned_signals:
        return None, None
    stacked = np.stack(aligned_signals, axis=0)
    return np.asarray(reference_time, dtype=float), np.nanmean(stacked, axis=0)


def _build_condition_records(
    condition_id,
    batch_root,
    summary_subdir_name,
    *,
    selected_parameter_set_ids=None,
    parameter_set_id_range=None,
    color_overrides=None,
):
    batch_root = Path(batch_root)
    table_path = _resolve_summary_table_path(batch_root, summary_subdir_name)
    export_df = _load_export_table(table_path)
    if export_df.empty:
        raise ValueError(f"Export table is empty: {table_path}")
    if "SIM_PARAM_GENERAL_batch_sim_index" not in export_df.columns:
        raise KeyError(f"{table_path} is missing SIM_PARAM_GENERAL_batch_sim_index")

    export_lookup = {}
    mode_export_lookup = {}
    for _, row in export_df.iterrows():
        is_mode = bool(row.get("SIM_SUMMARY_is_posterior_mode", False))
        mode_label = row.get("SIM_SUMMARY_mode_observation_label", np.nan)
        batch_index = row.get("SIM_PARAM_GENERAL_batch_sim_index", np.nan)
        if is_mode and pd.notna(mode_label):
            mode_export_lookup[str(mode_label)] = row.to_dict()
            continue
        if pd.isna(batch_index):
            continue
        export_lookup[int(batch_index)] = row.to_dict()

    grouped_runs = collect_parameter_set_runs(
        batch_root,
        color_overrides=color_overrides,
        selected_parameter_set_ids=selected_parameter_set_ids,
        parameter_set_id_range=parameter_set_id_range,
    )
    if not grouped_runs:
        raise FileNotFoundError(
            f"No simulation_output.h5 files found under parameter_set_* folders in {batch_root}"
        )

    records = []
    summary_rows = []
    for parameter_dir, runs in sorted(grouped_runs.items(), key=lambda item: item[0].name):
        mode_label = _mode_label_from_parameter_set_name(parameter_dir)
        is_mode = mode_label is not None
        if is_mode:
            parameter_set_id = None
            export_row = mode_export_lookup.get(str(mode_label))
            if export_row is None:
                raise KeyError(
                    f"Posterior-mode parameter set {parameter_dir.name} exists on disk but is missing from {table_path.name}"
                )
        else:
            parameter_set_id = _parameter_set_id_from_name(parameter_dir)
            export_row = export_lookup.get(parameter_set_id)
            if export_row is None:
                raise KeyError(
                    f"Parameter set {parameter_dir.name} exists on disk but is missing from {table_path.name}"
                )
        ref_run = runs[0]
        record = {
            "condition_id": str(condition_id),
            "parameter_set_id": None if parameter_set_id is None else int(parameter_set_id),
            "parameter_set_dir": str(parameter_dir),
            "n_repeats": int(len(runs)),
            "is_posterior_mode": bool(is_mode),
            "mode_observation_label": None if mode_label is None else str(mode_label),
            "event_times": _relative_event_times(ref_run.get("event_times", {})),
            "features": export_row,
        }
        for trace_key in ("cst_lf", "cst_alpha_mod", "cst_beta_mod", "sync_trace"):
            time_rel, mean_trace = _aggregate_parameter_set_trace(runs, trace_key)
            record[f"{trace_key}_time"] = time_rel
            record[trace_key] = mean_trace
        records.append(record)

        summary_row = dict(export_row)
        summary_row.update({
            "SIM_SUMMARY_condition_id": str(condition_id),
            "SIM_SUMMARY_parameter_set_id": np.nan if parameter_set_id is None else int(parameter_set_id),
            "SIM_SUMMARY_parameter_set_dir": str(parameter_dir),
            "SIM_SUMMARY_n_repeats": int(len(runs)),
            "SIM_SUMMARY_source_batch_root": str(batch_root),
            "SIM_SUMMARY_source_summary_subdir": str(summary_subdir_name),
            "SIM_SUMMARY_is_posterior_mode": bool(is_mode),
            "SIM_SUMMARY_mode_observation_label": None if mode_label is None else str(mode_label),
        })
        summary_rows.append(summary_row)

    if not records:
        raise FileNotFoundError(
            f"No parameter_set_* folders remained after filtering in {batch_root}"
        )
    return records, pd.DataFrame(summary_rows), table_path


def build_posterior_predictive_across_conditions_data(
    condition_batch_roots,
    *,
    summary_subdir_name="parameter_set_summaries",
    condition_order=None,
    selected_parameter_set_ids=None,
    parameter_set_id_range=None,
    color_overrides=None,
):
    """Collect posterior-predictive records and summary rows without plotting."""
    condition_batch_roots, condition_order = _coerce_condition_batch_roots(
        condition_batch_roots,
        condition_order=condition_order,
    )
    records_by_condition = {}
    summary_frames = []
    source_table_paths = {}
    for condition in condition_order:
        records, summary_df, table_path = _build_condition_records(
            condition,
            condition_batch_roots[condition],
            summary_subdir_name,
            selected_parameter_set_ids=selected_parameter_set_ids,
            parameter_set_id_range=parameter_set_id_range,
            color_overrides=color_overrides,
        )
        records_by_condition[condition] = records
        summary_frames.append(summary_df)
        source_table_paths[condition] = str(table_path)
    across_condition_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    return {
        "records_by_condition": records_by_condition,
        "across_condition_df": across_condition_df,
        "source_table_paths": source_table_paths,
        "condition_order": list(condition_order),
    }


def load_posterior_predictive_summary_table(summary_table_path):
    summary_table_path = Path(summary_table_path)
    if not summary_table_path.exists():
        raise FileNotFoundError(f"Missing posterior-predictive summary table: {summary_table_path}")
    return _load_export_table(summary_table_path)


def _pool_condition_mean_trace(records, trace_key):
    plotted = []
    for record in records:
        if record.get("is_posterior_mode", False):
            continue
        time = record.get(f"{trace_key}_time")
        trace = record.get(trace_key)
        if time is None or trace is None:
            continue
        time = np.asarray(time, dtype=float)
        trace = np.asarray(trace, dtype=float)
        if time.size == 0 or trace.size == 0:
            continue
        plotted.append((time, trace))
    if not plotted:
        return {"time": None, "mean": None, "n_parameter_sets": 0}
    common_time = plotted[0][0]
    mean_stack = []
    for time, trace in plotted:
        if time.shape == common_time.shape and np.allclose(time, common_time):
            mean_stack.append(trace)
        else:
            mean_stack.append(np.interp(common_time, time, trace, left=np.nan, right=np.nan))
    return {
        "time": np.asarray(common_time, dtype=float),
        "mean": np.nanmean(np.stack(mean_stack, axis=0), axis=0),
        "n_parameter_sets": int(len(mean_stack)),
    }


def build_posterior_predictive_condition_mean_traces(
    condition_batch_roots,
    *,
    summary_subdir_name="parameter_set_summaries",
    condition_order=None,
    selected_parameter_set_ids=None,
    parameter_set_id_range=None,
    color_overrides=None,
    output_path=None,
    load_if_available=True,
    force_recompute=False,
):
    """Build or load pooled posterior-predictive mean traces per condition."""
    if output_path is not None:
        output_path = Path(output_path)
        if load_if_available and output_path.exists() and not force_recompute:
            return pd.read_pickle(output_path)

    data = build_posterior_predictive_across_conditions_data(
        condition_batch_roots,
        summary_subdir_name=summary_subdir_name,
        condition_order=condition_order,
        selected_parameter_set_ids=selected_parameter_set_ids,
        parameter_set_id_range=parameter_set_id_range,
        color_overrides=color_overrides,
    )
    traces_by_condition = {}
    for condition, records in data["records_by_condition"].items():
        traces_by_condition[str(condition)] = {
            trace_key: _pool_condition_mean_trace(records, trace_key)
            for trace_key in ("cst_lf", "cst_alpha_mod", "cst_beta_mod", "sync_trace")
        }
    cache = {
        "condition_order": data["condition_order"],
        "source_table_paths": data["source_table_paths"],
        "selected_parameter_set_ids": selected_parameter_set_ids,
        "parameter_set_id_range": parameter_set_id_range,
        "summary_subdir_name": summary_subdir_name,
        "traces_by_condition": traces_by_condition,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(cache, output_path)
    return cache


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
                feature_values.append(
                    np.asarray(
                        [
                            feature_row.get(spec["burst_col"], np.nan),
                            feature_row.get(spec["trend_col"], np.nan),
                        ],
                        dtype=float,
                    )
                )
        limits["trace"][spec["key"]] = _auto_limits(
            np.concatenate(trace_values) if trace_values else np.asarray([0.0]),
            zero_floor=False,
        )
        limits["feature"][spec["key"]] = _auto_limits(
            np.concatenate(feature_values) if feature_values else np.asarray([0.0]),
            zero_floor=False,
        )
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


def _draw_time_markers(ax, event_times, style):
    ready_s = event_times.get("get_ready_cue_s")
    cue_s = event_times.get("go_nogo_cue_s")
    if ready_s is not None:
        ax.axvline(
            float(ready_s),
            color=style["event_color"],
            ls="--",
            lw=style["ready_lw"],
            alpha=style["ready_alpha"],
            zorder=1,
        )
    if cue_s is not None:
        ax.axvline(
            float(cue_s),
            color=style["event_color"],
            ls="-",
            lw=style["go_lw"],
            alpha=style["go_alpha"],
            zorder=1,
        )


def _plot_trace_panel(ax, records, spec, color, display_window, y_limits, style, reference_window=None):
    plotted = []
    mode_record = None
    reference_record = records[0] if records else None
    for record in records:
        if record.get("is_posterior_mode", False):
            mode_record = record
            continue
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
        plotted.append((time_zoom, trace_zoom))
        ax.plot(
            time_zoom,
            trace_zoom,
            color=color,
            lw=style["participant_trace_lw"],
            alpha=style["participant_trace_alpha"],
            zorder=2,
        )
    if plotted:
        common_time = plotted[0][0]
        mean_stack = []
        for time_zoom, trace_zoom in plotted:
            if time_zoom.shape == common_time.shape and np.allclose(time_zoom, common_time):
                mean_stack.append(trace_zoom)
            else:
                mean_stack.append(np.interp(common_time, time_zoom, trace_zoom, left=np.nan, right=np.nan))
        mean_trace = np.nanmean(np.stack(mean_stack, axis=0), axis=0)
        ax.plot(
            common_time,
            mean_trace,
            color=color,
            lw=style["mean_trace_lw"],
            alpha=1.0,
            linestyle=style["mean_trace_linestyle"],
            zorder=4,
        )
        if reference_record is not None:
            _draw_time_markers(ax, reference_record["event_times"], style)
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
    if mode_record is not None:
        mode_time = mode_record.get(f"{spec['key']}_time")
        mode_trace = mode_record.get(spec["key"])
        if mode_time is not None and mode_trace is not None:
            if reference_record is None:
                reference_record = mode_record
            mode_time = np.asarray(mode_time, dtype=float)
            mode_trace = np.asarray(mode_trace, dtype=float)
            mode_mask = (mode_time >= display_window[0]) & (mode_time <= display_window[1])
            if np.any(mode_mask):
                if not plotted and reference_record is not None:
                    _draw_time_markers(ax, reference_record["event_times"], style)
                ax.plot(
                    mode_time[mode_mask],
                    mode_trace[mode_mask],
                    color=style["mode_trace_color"],
                    lw=style["mode_trace_lw"],
                    alpha=style["mode_trace_alpha"],
                    linestyle=style["mode_trace_linestyle"],
                    zorder=5,
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
    base_records = [record for record in records if not record.get("is_posterior_mode", False)]
    mode_records = [record for record in records if record.get("is_posterior_mode", False)]
    burst_values = np.asarray([record["features"].get(spec["burst_col"], np.nan) for record in base_records], dtype=float)
    trend_values = np.asarray([record["features"].get(spec["trend_col"], np.nan) for record in base_records], dtype=float)
    datasets = [burst_values[np.isfinite(burst_values)], trend_values[np.isfinite(trend_values)]]
    positions = [1.0, 2.0]
    valid_positions = [pos for pos, vals in zip(positions, datasets) if vals.size > 0]
    valid_data = [vals for vals in datasets if vals.size > 0]
    if valid_data:
        violin = ax.violinplot(
            valid_data,
            positions=valid_positions,
            widths=style["violin_width"],
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
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
    if mode_records:
        mode_record = mode_records[0]
        mode_points = [
            mode_record["features"].get(spec["burst_col"], np.nan),
            mode_record["features"].get(spec["trend_col"], np.nan),
        ]
        for pos, mode_value in zip(positions, mode_points, strict=True):
            if not np.isfinite(float(mode_value)):
                continue
            mode_facecolor = style["mode_feature_marker_facecolor"]
            if mode_facecolor in (None, "row", "condition"):
                mode_facecolor = color
            ax.scatter(
                [pos],
                [float(mode_value)],
                s=style["mode_feature_marker_size"],
                alpha=style["mode_feature_marker_alpha"],
                facecolors=mode_facecolor,
                marker=style["mode_feature_marker"],
                edgecolors=style["mode_feature_marker_edgecolor"],
                linewidths=style["mode_feature_marker_edge_lw"],
                zorder=6,
            )
    ax.axhline(
        0.0,
        color=style["zero_line_color"],
        lw=style["zero_line_lw"],
        ls="--",
        alpha=style["zero_line_alpha"],
        zorder=1,
    )
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(*map(float, y_limits))
    ax.set_xticks(positions)
    ax.set_xticklabels(["Burst", "Trend"], rotation=0)
    ax.tick_params(axis="both", labelsize=style["tick_labelsize"])
    ax.grid(alpha=0.22, axis="y")


def _plot_projection_panel(ax, records, proj_spec, color, limits, style):
    x_col, y_col, z_col = proj_spec["columns"]
    base_records = [record for record in records if not record.get("is_posterior_mode", False)]
    mode_records = [record for record in records if record.get("is_posterior_mode", False)]
    xyz = np.asarray(
        [
            [
                record["features"].get(x_col, np.nan),
                record["features"].get(y_col, np.nan),
                record["features"].get(z_col, np.nan),
            ]
            for record in base_records
        ],
        dtype=float,
    )
    finite_mask = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite_mask]
    (xlim, ylim, zlim) = limits
    ax.view_init(elev=style["projection_elev"], azim=style["projection_azim"])
    if xyz.size:
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            np.full(xyz.shape[0], zlim[0]),
            s=style["projection_marker_size"],
            color=color,
            alpha=style["projection_alpha"],
            depthshade=False,
        )
        ax.scatter(
            xyz[:, 0],
            np.full(xyz.shape[0], ylim[1]),
            xyz[:, 2],
            s=style["projection_marker_size"],
            color=color,
            alpha=style["projection_alpha"],
            depthshade=False,
        )
        ax.scatter(
            np.full(xyz.shape[0], xlim[0]),
            xyz[:, 1],
            xyz[:, 2],
            s=style["projection_marker_size"],
            color=color,
            alpha=style["projection_alpha"],
            depthshade=False,
        )
    if mode_records:
        mode_xyz = np.asarray(
            [
                mode_records[0]["features"].get(x_col, np.nan),
                mode_records[0]["features"].get(y_col, np.nan),
                mode_records[0]["features"].get(z_col, np.nan),
            ],
            dtype=float,
        )
        if np.all(np.isfinite(mode_xyz)):
            ax.scatter(
                [mode_xyz[0]],
                [mode_xyz[1]],
                [mode_xyz[2]],
                s=style["mode_projection_marker_size"],
                color=style["mode_trace_color"],
                alpha=style["mode_projection_marker_alpha"],
                marker=style["mode_projection_marker"],
                edgecolors="white",
                linewidths=0.8,
                depthshade=False,
            )
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
        ((xlim[0], ylim[0], zlim[0]), (xlim[1], ylim[0], zlim[0])),
        ((xlim[0], ylim[1], zlim[0]), (xlim[1], ylim[1], zlim[0])),
        ((xlim[0], ylim[0], zlim[0]), (xlim[0], ylim[1], zlim[0])),
        ((xlim[1], ylim[0], zlim[0]), (xlim[1], ylim[1], zlim[0])),
        ((xlim[0], ylim[1], zlim[1]), (xlim[1], ylim[1], zlim[1])),
        ((xlim[0], ylim[1], zlim[0]), (xlim[0], ylim[1], zlim[1])),
        ((xlim[1], ylim[1], zlim[0]), (xlim[1], ylim[1], zlim[1])),
        ((xlim[0], ylim[0], zlim[1]), (xlim[0], ylim[1], zlim[1])),
        ((xlim[0], ylim[0], zlim[0]), (xlim[0], ylim[0], zlim[1])),
    }
    for (x0, y0, z0), (x1, y1, z1) in sorted(background_edge_segments):
        ax.plot([x0, x1], [y0, y1], [z0, z1], color=edge_color, lw=edge_lw, alpha=edge_alpha, zorder=6)


def plot_posterior_predictive_across_conditions_summary(
    condition_batch_roots,
    *,
    summary_subdir_name="parameter_set_summaries",
    output_path=None,
    condition_order=None,
    display_window_rel_cue_s=(-4.0, 1.0),
    selected_parameter_set_ids=None,
    parameter_set_id_range=None,
    row_colors=None,
    style=None,
    ylim_overrides=None,
    trace_reference_window_rel_cue_s=(-4.0, -2.0),
    export_summary_table=False,
    summary_table_output_dir=None,
    summary_table_csv_name="posterior_predictive_across_conditions_summary.csv",
    summary_table_pickle_name="posterior_predictive_across_conditions_summary.pkl",
    color_overrides=None,
):
    condition_batch_roots, condition_order = _coerce_condition_batch_roots(
        condition_batch_roots,
        condition_order=condition_order,
    )
    style = _resolve_style(style)
    colors = _resolve_colors(row_colors)

    records_by_condition = {}
    summary_frames = []
    source_table_paths = {}
    for condition in condition_order:
        records, summary_df, table_path = _build_condition_records(
            condition,
            condition_batch_roots[condition],
            summary_subdir_name,
            selected_parameter_set_ids=selected_parameter_set_ids,
            parameter_set_id_range=parameter_set_id_range,
            color_overrides=color_overrides,
        )
        records_by_condition[condition] = records
        summary_frames.append(summary_df)
        source_table_paths[condition] = str(table_path)

    across_condition_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()

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
        condition_records = records_by_condition[condition]
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
        base_records = [record for record in condition_records if not record.get("is_posterior_mode", False)]
        mode_records = [record for record in condition_records if record.get("is_posterior_mode", False)]
        n_parameter_sets = int(len(base_records))
        total_repeats = int(sum(int(record["n_repeats"]) for record in base_records))
        mode_suffix = ""
        if mode_records:
            mode_suffix = f"\nmode = {mode_records[0].get('mode_observation_label', 'present')}"
        fig.text(
            x_center,
            y_top + 0.02,
            f"{condition}\nn parameter sets = {n_parameter_sets} | total repeats = {total_repeats}{mode_suffix}",
            ha="center",
            va="bottom",
            fontsize=style["condition_title_fontsize"],
        )

    fig.suptitle(
        "Across-condition posterior-predictive simulation summary",
        fontsize=style["condition_title_fontsize"] + 2,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))

    figure_path = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        figure_path = output_path

    summary_table_paths = {}
    if export_summary_table:
        summary_table_output_dir = (
            Path(summary_table_output_dir)
            if summary_table_output_dir is not None
            else (Path(output_path).parent if output_path is not None else Path.cwd())
        )
        summary_table_output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = summary_table_output_dir / summary_table_csv_name
        pickle_path = summary_table_output_dir / summary_table_pickle_name
        across_condition_df.to_csv(csv_path, index=False)
        across_condition_df.to_pickle(pickle_path)
        summary_table_paths = {
            "csv_path": str(csv_path),
            "pickle_path": str(pickle_path),
        }

    summary_counts = {
        condition: {
            "n_parameter_sets": int(len([record for record in records if not record.get("is_posterior_mode", False)])),
            "n_total_repeats": int(sum(int(record["n_repeats"]) for record in records if not record.get("is_posterior_mode", False))),
            "n_posterior_mode_sets": int(len([record for record in records if record.get("is_posterior_mode", False)])),
            "posterior_mode_labels": [
                str(record.get("mode_observation_label"))
                for record in records
                if record.get("is_posterior_mode", False)
            ],
        }
        for condition, records in records_by_condition.items()
    }

    return fig, {
        "figure_path": None if figure_path is None else str(figure_path),
        "summary_table_paths": summary_table_paths,
        "source_table_paths": source_table_paths,
        "summary_counts_by_condition": summary_counts,
        "n_condition_records": int(len(across_condition_df)),
    }, across_condition_df
