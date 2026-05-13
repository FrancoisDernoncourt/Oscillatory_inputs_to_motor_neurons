from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


DEFAULT_CONDITION_ORDER = ("HAND_GO", "HAND_NOGO", "TA_NOGO")


DEFAULT_VIOLIN_STYLE = {
    "figure_dpi": 140,
    "figure_width_per_col": 4.4,
    "figure_height_per_row": 3.8,
    "ncols": 2,
    "panel_width_ratios": (1.8, 1.0),
    "sim_color": "#4C78A8",
    "sim_violin_alpha": 0.25,
    "sim_violin_edge_lw": 1.2,
    "sim_point_alpha": 0.10,
    "sim_point_size": 14,
    "sim_point_marker": "o",
    "sim_point_max": 450,
    "condition_colors": {
        "HAND_GO": "#2E8B57",
        "HAND_NOGO": "#E67E22",
        "TA_NOGO": "#7E57C2",
    },
    "exp_violin_alpha": 0.20,
    "exp_violin_edge_lw": 1.2,
    "exp_point_alpha": 0.85,
    "exp_point_size": 28,
    "exp_point_marker": "o",
    "point_jitter": 0.08,
    "mean_marker": "D",
    "mean_marker_size": 42,
    "mean_marker_edgecolor": "white",
    "mean_marker_alpha": 1.0,
    "condition_mean_line_lw": 1.4,
    "condition_mean_line_ls": "--",
    "condition_mean_line_alpha": 0.95,
    "grand_mean_color": "#222222",
    "grand_mean_lw": 1.8,
    "grand_mean_ls": "-",
    "grand_mean_alpha": 0.95,
    "zero_line_color": "#8A8A8A",
    "zero_line_alpha": 0.7,
    "zero_line_lw": 1.0,
    "title_fontsize": 11,
    "axis_label_fontsize": 9,
    "tick_labelsize": 8,
    "show_zero_line": True,
    "kde_fill_alpha": 0.16,
    "kde_line_alpha": 0.95,
    "kde_line_lw": 1.5,
    "kde_grid_alpha": 0.10,
}


DEFAULT_PCA_STYLE = {
    "figure_size": (9.2, 7.2),
    "figure_dpi": 150,
    "sim_scatter_alpha": 0.12,
    "sim_scatter_size": 14,
    "sim_scatter_marker": "o",
    "show_sim_points": True,
    "kde_alpha": 0.50,
    "n_contours": 5,
    "contour_mass_levels": None,
    "condition_colors": {
        "HAND_GO": "#2E8B57",
        "HAND_NOGO": "#E67E22",
        "TA_NOGO": "#7E57C2",
    },
    "exp_point_alpha": 0.95,
    "exp_point_size": 42,
    "exp_point_marker": "o",
    "exp_mean_marker": "D",
    "exp_mean_size": 72,
    "exp_mean_alpha": 1.0,
    "grand_mean_marker": "P",
    "grand_mean_size": 110,
    "grand_mean_alpha": 1.0,
    "grand_mean_color": "#111111",
    "sim_color": "#4C78A8",
    "title_fontsize": 12,
    "axis_label_fontsize": 10,
    "tick_labelsize": 9,
    "grid_alpha": 0.20,
}


def load_summary_table(path):
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        return pd.read_pickle(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def filter_simulated_rows(
    sim_df,
    *,
    min_units_kept=None,
    min_unit_fraction_kept=None,
):
    filtered = sim_df.copy()
    if min_units_kept is not None and "ANALYSIS_DIAGNOSIS_n_units_kept_min" in filtered.columns:
        kept = pd.to_numeric(filtered["ANALYSIS_DIAGNOSIS_n_units_kept_min"], errors="coerce")
        filtered = filtered.loc[kept >= float(min_units_kept)].copy()
    if min_unit_fraction_kept is not None and "ANALYSIS_DIAGNOSIS_unit_fraction_kept_mean" in filtered.columns:
        frac = pd.to_numeric(filtered["ANALYSIS_DIAGNOSIS_unit_fraction_kept_mean"], errors="coerce")
        filtered = filtered.loc[frac >= float(min_unit_fraction_kept)].copy()
    return filtered


def read_columns_from_export(path):
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        df = pd.read_pickle(path)
        return list(df.columns)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def discover_available_feature_groups(sim_table_path, exp_table_path):
    sim_cols = set(read_columns_from_export(sim_table_path))
    exp_cols = set(read_columns_from_export(exp_table_path))
    shared = sorted(col for col in sim_cols & exp_cols if col.startswith("OBS_"))
    groups = {
        "firing_statistics": [],
        "baseline": [],
        "spectrum": [],
        "burst": [],
        "trend": [],
        "window_features": [],
    }
    for col in shared:
        if col.startswith("OBS_FIRING_STATISTICS_"):
            groups["firing_statistics"].append(col)
        elif col.startswith("OBS_BASELINE_"):
            groups["baseline"].append(col)
        elif col.startswith("OBS_SPECTRUM_"):
            groups["spectrum"].append(col)
        elif col.startswith("OBS_BURST_FEATURES_"):
            groups["burst"].append(col)
        elif col.startswith("OBS_TREND_FEATURE_"):
            groups["trend"].append(col)
        elif col.startswith("OBS_WINDOW_"):
            groups["window_features"].append(col)
    return groups


def _format_window_token(token):
    if "_to_" not in token:
        return token
    start, end = token.split("_to_", 1)
    return f"[{_format_short_bound(start)}, {_format_short_bound(end)}] s"


def _format_short_bound(token):
    token = str(token)
    if token.startswith("m"):
        return f"-{token[1:]}"
    return token


def _pretty_signal_name(signal):
    mapping = {
        "CST_LF_raw": "CST low-pass",
        "ALPHA_raw": "Alpha band-pass",
        "ALPHA_mod_z": "Alpha modulation (z)",
        "BETA_raw": "Beta band-pass",
        "BETA_mod_z": "Beta modulation (z)",
        "SYNC": "Synchrony",
    }
    return mapping.get(signal, signal.replace("_", " "))


def _feature_unit(feature_name):
    if feature_name.startswith("OBS_FIRING_STATISTICS_firing_rate_"):
        return "spikes/s/MU"
    if feature_name.startswith("OBS_FIRING_STATISTICS_isi_cv_"):
        return "CV"
    if feature_name.endswith("_t_min_rel_cue_s") or feature_name.endswith("_t_max_rel_cue_s"):
        return "s rel cue"
    if "mod_z" in feature_name:
        return "z"
    if feature_name.startswith("OBS_SPECTRUM_"):
        return "Hz"
    if feature_name == "OBS_BURST_FEATURES_resid_cst_low_frequency" or feature_name == "OBS_TREND_FEATURE_cst_low_frequency":
        return "spikes/s/MU"
    if feature_name in {
        "OBS_BURST_FEATURES_resid_cst_alpha_modulation_hilbert_zscore",
        "OBS_BURST_FEATURES_resid_cst_beta_modulation_hilbert_zscore",
        "OBS_TREND_FEATURE_cst_alpha_modulation_hilbert_zscore",
        "OBS_TREND_FEATURE_cst_beta_modulation_hilbert_zscore",
    }:
        return "z"
    if feature_name in {
        "OBS_BURST_FEATURES_resid_sync_coincidence_curve",
        "OBS_TREND_FEATURE_sync_coincidence_curve",
    }:
        return "excess coincidence index"
    if "SYNC" in feature_name or "sync" in feature_name:
        return "excess coincidence index"
    if any(token in feature_name for token in ("CST_LF", "ALPHA_raw", "BETA_raw", "firing_rate")):
        return "spikes/s/MU"
    return ""


def format_feature_label(feature_name):
    if feature_name.startswith("OBS_WINDOW_"):
        body = feature_name[len("OBS_WINDOW_") :]
        window_token, rest = body.split("_", 1)
        stat_options = ("t_min_rel_cue_s", "t_max_rel_cue_s", "mean", "min", "max")
        stat = next((candidate for candidate in stat_options if rest.endswith(f"_{candidate}")), None)
        if stat is None:
            signal = rest
            stat = ""
        else:
            signal = rest[: -(len(stat) + 1)]
        return f"{_pretty_signal_name(signal)}\n{_format_window_token(window_token)} {stat}"
    if feature_name.startswith("OBS_BASELINE_"):
        body = feature_name[len("OBS_BASELINE_") :]
        stat_options = ("mean", "sd")
        stat = next((candidate for candidate in stat_options if body.endswith(f"_{candidate}")), None)
        if stat is None:
            signal = body
            stat = ""
        else:
            signal = body[: -(len(stat) + 1)]
        return f"{_pretty_signal_name(signal)}\nBaseline {stat}"
    if feature_name.startswith("OBS_SPECTRUM_"):
        body = feature_name[len("OBS_SPECTRUM_") :]
        signal, stat = body.split("_", 1)
        return f"{signal.title()} modulation spectrum\n{stat.replace('_', ' ')}"
    if feature_name.startswith("OBS_BURST_FEATURES_resid_"):
        burst_mapping = {
            "OBS_BURST_FEATURES_resid_cst_low_frequency": "CST low-pass\nBurst feature",
            "OBS_BURST_FEATURES_resid_cst_alpha_modulation_hilbert_zscore": "Alpha modulation (z)\nBurst feature",
            "OBS_BURST_FEATURES_resid_cst_beta_modulation_hilbert_zscore": "Beta modulation (z)\nBurst feature",
            "OBS_BURST_FEATURES_resid_sync_coincidence_curve": "Synchrony\nBurst feature",
        }
        return burst_mapping.get(feature_name, feature_name)
    if feature_name.startswith("OBS_TREND_FEATURE_"):
        trend_mapping = {
            "OBS_TREND_FEATURE_cst_low_frequency": "CST low-pass\nTrend feature",
            "OBS_TREND_FEATURE_cst_alpha_modulation_hilbert_zscore": "Alpha modulation (z)\nTrend feature",
            "OBS_TREND_FEATURE_cst_beta_modulation_hilbert_zscore": "Beta modulation (z)\nTrend feature",
            "OBS_TREND_FEATURE_sync_coincidence_curve": "Synchrony\nTrend feature",
        }
        return trend_mapping.get(feature_name, feature_name)
    if feature_name.startswith("OBS_FIRING_STATISTICS_"):
        body = feature_name[len("OBS_FIRING_STATISTICS_") :].replace("_", " ")
        return body
    return feature_name


def validate_selected_features(sim_df, exp_df, selected_features):
    if not selected_features:
        raise ValueError("selected_features is empty")
    missing = [feat for feat in selected_features if feat not in sim_df.columns or feat not in exp_df.columns]
    if missing:
        raise KeyError(f"Selected features not found in both tables: {missing}")
    numeric_features = []
    dropped = []
    for feat in selected_features:
        sim_numeric = pd.to_numeric(sim_df[feat], errors="coerce")
        exp_numeric = pd.to_numeric(exp_df[feat], errors="coerce")
        if np.isfinite(sim_numeric).any() and np.isfinite(exp_numeric).any():
            numeric_features.append(feat)
        else:
            dropped.append(feat)
    if not numeric_features:
        raise ValueError("No selected features have usable numeric values in both tables")
    return numeric_features, dropped


def _subsample(values, max_points):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size <= max_points:
        return finite
    rng = np.random.default_rng(12345)
    idx = rng.choice(finite.size, size=int(max_points), replace=False)
    return np.sort(finite[idx])


def _jitter(center, n, width):
    if n <= 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(12345 + int(center * 100))
    return center + rng.uniform(-width, width, size=n)


def _kde_density(values, grid):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.nanstd(values) <= 0:
        return np.zeros_like(grid, dtype=float)
    try:
        kde = gaussian_kde(values)
        return np.asarray(kde(grid), dtype=float)
    except Exception:
        return np.zeros_like(grid, dtype=float)


def _resolve_feature_ylim(sim_vals, exp_vals_by_condition, pad_frac=0.08):
    all_vals = [np.asarray(sim_vals, dtype=float)]
    all_vals.extend(np.asarray(v, dtype=float) for v in exp_vals_by_condition if len(v))
    merged = np.concatenate([v[np.isfinite(v)] for v in all_vals if np.isfinite(v).any()]) if any(np.isfinite(v).any() for v in all_vals) else np.asarray([], dtype=float)
    if merged.size == 0:
        return (-1.0, 1.0)
    lo = float(np.min(merged))
    hi = float(np.max(merged))
    if np.isclose(lo, hi):
        span = max(abs(lo), 1.0)
        return lo - 0.25 * span, hi + 0.25 * span
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def _resolve_condition_colors(style):
    return dict(DEFAULT_VIOLIN_STYLE["condition_colors"], **(style.get("condition_colors") or {}))


def _validate_experimental_display_mode(mode):
    valid_modes = {"participant_condition", "condition_mean", "grand_mean"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported experimental_display_mode {mode!r}. "
            f"Expected one of {sorted(valid_modes)}."
        )
    return mode


def plot_feature_violin_coverage(
    sim_df,
    exp_df,
    selected_features,
    *,
    condition_order=DEFAULT_CONDITION_ORDER,
    violin_style=None,
    experimental_display_mode="participant_condition",
    show_grand_mean_overlay=False,
):
    experimental_display_mode = _validate_experimental_display_mode(experimental_display_mode)
    style = dict(DEFAULT_VIOLIN_STYLE)
    if violin_style:
        style.update(violin_style)
    condition_colors = _resolve_condition_colors(style)

    n_features = len(selected_features)
    ncols = int(max(1, style["ncols"]))
    nrows = int(np.ceil(n_features / ncols))
    fig = plt.figure(
        figsize=(style["figure_width_per_col"] * ncols, style["figure_height_per_row"] * nrows),
        dpi=style["figure_dpi"],
    )
    outer = fig.add_gridspec(nrows, ncols, wspace=0.34, hspace=0.36)
    panel_axes = []

    for panel_index, feature in enumerate(selected_features):
        row = panel_index // ncols
        col = panel_index % ncols
        inner = outer[row, col].subgridspec(1, 2, width_ratios=style["panel_width_ratios"], wspace=0.08)
        ax = fig.add_subplot(inner[0, 0])
        ax_kde = fig.add_subplot(inner[0, 1], sharey=ax)
        panel_axes.append((ax, ax_kde))

        sim_vals = pd.to_numeric(sim_df[feature], errors="coerce").to_numpy(dtype=float)
        sim_vals = sim_vals[np.isfinite(sim_vals)]
        positions = [0.0] + [float(idx + 1) for idx in range(len(condition_order))]
        exp_vals_by_condition = []
        exp_means_by_condition = []
        grand_mean = np.nan

        violin_data = [sim_vals]
        violin_positions = [positions[0]]
        violin_colors = [style["sim_color"]]
        for idx, condition in enumerate(condition_order, start=1):
            cond_vals = pd.to_numeric(
                exp_df.loc[exp_df["DATASET_CONTEXT_condition_id"].astype(str) == str(condition), feature],
                errors="coerce",
            ).to_numpy(dtype=float)
            cond_vals = cond_vals[np.isfinite(cond_vals)]
            exp_vals_by_condition.append(cond_vals)
            exp_means_by_condition.append(float(np.mean(cond_vals)) if cond_vals.size else np.nan)
            if cond_vals.size and experimental_display_mode == "participant_condition":
                violin_data.append(cond_vals)
                violin_positions.append(float(idx))
                violin_colors.append(condition_colors.get(condition, "#777777"))
        all_exp_vals = np.concatenate([vals for vals in exp_vals_by_condition if len(vals)]) if any(
            len(vals) for vals in exp_vals_by_condition
        ) else np.array([], dtype=float)
        if all_exp_vals.size:
            grand_mean = float(np.mean(all_exp_vals))

        if violin_data:
            violin = ax.violinplot(
                violin_data,
                positions=violin_positions,
                widths=0.75,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            for body, color in zip(violin["bodies"], violin_colors):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(style["sim_violin_alpha"] if color == style["sim_color"] else style["exp_violin_alpha"])
                body.set_linewidth(style["sim_violin_edge_lw"] if color == style["sim_color"] else style["exp_violin_edge_lw"])

        sim_points = _subsample(sim_vals, style["sim_point_max"])
        if sim_points.size:
            ax.scatter(
                _jitter(positions[0], sim_points.size, style["point_jitter"]),
                sim_points,
                s=style["sim_point_size"],
                color=style["sim_color"],
                alpha=style["sim_point_alpha"],
                marker=style["sim_point_marker"],
                linewidths=0,
                zorder=3,
            )

        if experimental_display_mode == "participant_condition":
            for idx, condition in enumerate(condition_order, start=1):
                cond_vals = exp_vals_by_condition[idx - 1]
                if not cond_vals.size:
                    continue
                color = condition_colors.get(condition, "#777777")
                ax.scatter(
                    _jitter(float(idx), cond_vals.size, style["point_jitter"]),
                    cond_vals,
                    s=style["exp_point_size"],
                    color=color,
                    alpha=style["exp_point_alpha"],
                    marker=style["exp_point_marker"],
                    linewidths=0,
                    zorder=4,
                )
                ax.scatter(
                    [float(idx)],
                    [exp_means_by_condition[idx - 1]],
                    s=style["mean_marker_size"],
                    color=color,
                    marker=style["mean_marker"],
                    edgecolors=style["mean_marker_edgecolor"],
                    alpha=style["mean_marker_alpha"],
                    linewidths=0.9,
                    zorder=5,
                )
        elif experimental_display_mode == "condition_mean":
            for idx, condition in enumerate(condition_order, start=1):
                exp_mean = exp_means_by_condition[idx - 1]
                if not np.isfinite(exp_mean):
                    continue
                color = condition_colors.get(condition, "#777777")
                ax.scatter(
                    [float(idx)],
                    [exp_mean],
                    s=style["mean_marker_size"],
                    color=color,
                    marker=style["mean_marker"],
                    edgecolors=style["mean_marker_edgecolor"],
                    alpha=style["mean_marker_alpha"],
                    linewidths=0.9,
                    zorder=5,
                )

        if style.get("show_zero_line", True):
            ax.axhline(
                0.0,
                color=style["zero_line_color"],
                alpha=style["zero_line_alpha"],
                linewidth=style["zero_line_lw"],
                linestyle="--",
                zorder=1,
            )

        unit = _feature_unit(feature)
        ax.set_title(format_feature_label(feature), fontsize=style["title_fontsize"])
        ax.set_ylabel(unit or feature, fontsize=style["axis_label_fontsize"])
        ax.set_xticks(positions)
        ax.set_xticklabels(["Sim"] + list(condition_order), rotation=20)
        ax.tick_params(axis="both", labelsize=style["tick_labelsize"])
        ax.grid(axis="y", alpha=0.18)
        y_lim = _resolve_feature_ylim(sim_vals, exp_vals_by_condition)
        ax.set_ylim(y_lim)
        if (experimental_display_mode == "grand_mean" or show_grand_mean_overlay) and np.isfinite(grand_mean):
            ax.axhline(
                grand_mean,
                color=style["grand_mean_color"],
                linewidth=style["grand_mean_lw"],
                linestyle=style["grand_mean_ls"],
                alpha=style["grand_mean_alpha"],
                zorder=2,
            )

        kde_grid = np.linspace(y_lim[0], y_lim[1], 256)
        kde_sets = [("Sim", sim_vals, style["sim_color"])]
        if experimental_display_mode == "participant_condition":
            kde_sets.extend(
                [
                    (condition, values, condition_colors.get(condition, "#777777"))
                    for condition, values in zip(condition_order, exp_vals_by_condition)
                    if len(values)
                ]
            )
        max_density = 0.0
        kde_results = []
        for label, values, color in kde_sets:
            density = _kde_density(values, kde_grid)
            kde_results.append((label, density, color))
            if density.size:
                max_density = max(max_density, float(np.max(density)))
        for _, density, color in kde_results:
            ax_kde.fill_betweenx(
                kde_grid,
                0.0,
                density,
                color=color,
                alpha=style["kde_fill_alpha"],
                linewidth=0,
            )
            ax_kde.plot(
                density,
                kde_grid,
                color=color,
                linewidth=style["kde_line_lw"],
                alpha=style["kde_line_alpha"],
            )
        if experimental_display_mode == "condition_mean":
            for condition, exp_mean in zip(condition_order, exp_means_by_condition):
                if not np.isfinite(exp_mean):
                    continue
                color = condition_colors.get(condition, "#777777")
                ax_kde.axhline(
                    exp_mean,
                    color=color,
                    linewidth=style["condition_mean_line_lw"],
                    linestyle=style["condition_mean_line_ls"],
                    alpha=style["condition_mean_line_alpha"],
                )
        elif (experimental_display_mode == "grand_mean" or show_grand_mean_overlay) and np.isfinite(grand_mean):
            ax_kde.axhline(
                grand_mean,
                color=style["grand_mean_color"],
                linewidth=style["grand_mean_lw"],
                linestyle=style["grand_mean_ls"],
                alpha=style["grand_mean_alpha"],
            )
        ax_kde.set_xlabel("Density", fontsize=style["axis_label_fontsize"])
        ax_kde.tick_params(axis="x", labelsize=style["tick_labelsize"])
        ax_kde.tick_params(axis="y", left=False, labelleft=False)
        ax_kde.grid(axis="x", alpha=style["kde_grid_alpha"])
        ax_kde.spines["top"].set_visible(False)
        ax_kde.spines["right"].set_visible(False)
        ax_kde.spines["left"].set_visible(False)
        if max_density > 0:
            ax_kde.set_xlim(0.0, max_density * 1.08)
        else:
            ax_kde.set_xlim(0.0, 1.0)

    for panel_index in range(n_features, nrows * ncols):
        row = panel_index // ncols
        col = panel_index % ncols
        inner = outer[row, col].subgridspec(1, 2, width_ratios=style["panel_width_ratios"], wspace=0.08)
        fig.add_subplot(inner[0, 0]).axis("off")
        fig.add_subplot(inner[0, 1]).axis("off")

    legend_handles = [
        Patch(facecolor=style["sim_color"], edgecolor=style["sim_color"], alpha=style["sim_violin_alpha"], label="Simulated distribution"),
        Line2D([0], [0], marker=style["sim_point_marker"], color="none", markerfacecolor=style["sim_color"], markersize=6, alpha=style["sim_point_alpha"], label="Simulated parameter sets"),
        Line2D([0], [0], color=style["sim_color"], linewidth=style["kde_line_lw"], alpha=style["kde_line_alpha"], label="Marginal KDE"),
    ]
    if experimental_display_mode == "participant_condition":
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["mean_marker"],
                color="none",
                markerfacecolor="#666666",
                markeredgecolor=style["mean_marker_edgecolor"],
                markersize=7,
                label="Experimental condition mean",
            )
        )
        for condition in condition_order:
            legend_handles.append(
                Patch(
                    facecolor=condition_colors.get(condition, "#777777"),
                    edgecolor=condition_colors.get(condition, "#777777"),
                    alpha=style["exp_violin_alpha"],
                    label=f"{condition} participants",
                )
            )
    elif experimental_display_mode == "condition_mean":
        for condition in condition_order:
            color = condition_colors.get(condition, "#777777")
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=style["mean_marker"],
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=style["mean_marker_edgecolor"],
                    markersize=7,
                    linewidth=style["condition_mean_line_lw"],
                    linestyle=style["condition_mean_line_ls"],
                    alpha=style["condition_mean_line_alpha"],
                    label=f"{condition} mean",
                )
            )
    elif experimental_display_mode == "grand_mean":
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["grand_mean_color"],
                linewidth=style["grand_mean_lw"],
                linestyle=style["grand_mean_ls"],
                alpha=style["grand_mean_alpha"],
                label="Experimental grand mean",
            )
        )
    elif show_grand_mean_overlay:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["grand_mean_color"],
                linewidth=style["grand_mean_lw"],
                linestyle=style["grand_mean_ls"],
                alpha=style["grand_mean_alpha"],
                label="Experimental grand mean",
            )
        )
    fig.legend(handles=legend_handles, loc="upper center", ncol=min(len(legend_handles), 4 + len(condition_order)))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _prepare_pca_inputs(sim_df, exp_df, selected_features):
    sim_numeric = sim_df.loc[:, selected_features].apply(pd.to_numeric, errors="coerce")
    exp_numeric = exp_df.loc[:, selected_features].apply(pd.to_numeric, errors="coerce")

    valid_features = []
    for feature in selected_features:
        sim_col = sim_numeric[feature].to_numpy(dtype=float)
        if np.isfinite(sim_col).sum() < 3:
            continue
        if np.nanstd(sim_col) <= 0:
            continue
        valid_features.append(feature)
    if len(valid_features) < 2:
        raise ValueError("At least two selected features with finite variation are required for PCA")

    sim_numeric = sim_numeric.loc[:, valid_features]
    exp_numeric = exp_numeric.loc[:, valid_features]

    sim_medians = sim_numeric.median(axis=0, skipna=True)
    sim_imputed = sim_numeric.fillna(sim_medians)
    exp_imputed = exp_numeric.fillna(sim_medians)

    sim_mean = sim_imputed.mean(axis=0)
    sim_std = sim_imputed.std(axis=0, ddof=0).replace(0.0, 1.0)

    sim_z = (sim_imputed - sim_mean) / sim_std
    exp_z = (exp_imputed - sim_mean) / sim_std

    sim_matrix = sim_z.to_numpy(dtype=float)
    exp_matrix = exp_z.to_numpy(dtype=float)
    u, s, vt = np.linalg.svd(sim_matrix, full_matrices=False)
    components = vt[:2, :]
    sim_scores = sim_matrix @ components.T
    exp_scores = exp_matrix @ components.T

    variances = (s ** 2) / max(1, sim_matrix.shape[0] - 1)
    total_variance = float(np.sum(variances))
    explained = (variances[:2] / total_variance) if total_variance > 0 else np.asarray([np.nan, np.nan], dtype=float)
    return valid_features, sim_scores, exp_scores, explained


def _compute_density_mass_levels(density, mass_levels):
    density = np.asarray(density, dtype=float)
    flat = density.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0 or float(np.sum(flat)) <= 0.0:
        return None
    sorted_desc = np.sort(flat)[::-1]
    cumulative = np.cumsum(sorted_desc)
    cumulative /= cumulative[-1]
    thresholds = []
    for mass in mass_levels:
        idx = int(np.searchsorted(cumulative, float(mass), side="left"))
        idx = min(max(idx, 0), sorted_desc.size - 1)
        thresholds.append(float(sorted_desc[idx]))
    thresholds = np.asarray(thresholds, dtype=float)
    thresholds = np.unique(thresholds)
    thresholds.sort()
    return thresholds


def _resolve_contour_mass_levels(style):
    if style.get("contour_mass_levels") is not None:
        levels = [float(v) for v in style["contour_mass_levels"]]
    else:
        n = max(1, int(style.get("n_contours", 5)))
        levels = np.linspace(0.50, 0.95, n, dtype=float).tolist()
    levels = [v for v in levels if 0.0 < v < 1.0]
    if not levels:
        levels = [0.8]
    return levels


def plot_sim_vs_exp_pca(
    sim_df,
    exp_df,
    selected_features,
    *,
    condition_order=DEFAULT_CONDITION_ORDER,
    pca_style=None,
    experimental_display_mode="participant_condition",
    show_grand_mean_overlay=False,
):
    experimental_display_mode = _validate_experimental_display_mode(experimental_display_mode)
    style = dict(DEFAULT_PCA_STYLE)
    if pca_style:
        style.update(pca_style)
    condition_colors = dict(DEFAULT_PCA_STYLE["condition_colors"], **(style.get("condition_colors") or {}))

    used_features, sim_scores, exp_scores, explained = _prepare_pca_inputs(sim_df, exp_df, selected_features)

    fig, ax = plt.subplots(figsize=style["figure_size"], dpi=style["figure_dpi"])
    contour_mass_levels = _resolve_contour_mass_levels(style)
    if style.get("show_sim_points", True):
        ax.scatter(
            sim_scores[:, 0],
            sim_scores[:, 1],
            s=style["sim_scatter_size"],
            color=style["sim_color"],
            alpha=style["sim_scatter_alpha"],
            marker=style["sim_scatter_marker"],
            linewidths=0,
            zorder=2,
            label="Simulated parameter sets",
        )

    contour_handles = []
    if sim_scores.shape[0] >= 5:
        try:
            kde = gaussian_kde(sim_scores.T)
            x_min, x_max = np.min(sim_scores[:, 0]), np.max(sim_scores[:, 0])
            y_min, y_max = np.min(sim_scores[:, 1]), np.max(sim_scores[:, 1])
            x_pad = 0.15 * max(x_max - x_min, 1.0)
            y_pad = 0.15 * max(y_max - y_min, 1.0)
            grid_x = np.linspace(x_min - x_pad, x_max + x_pad, 160)
            grid_y = np.linspace(y_min - y_pad, y_max + y_pad, 160)
            xx, yy = np.meshgrid(grid_x, grid_y)
            density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            density_thresholds = _compute_density_mass_levels(density, contour_mass_levels)
            ax.contourf(
                xx,
                yy,
                density,
                levels=max(8, int(style.get("n_contours", 5)) + 3),
                cmap="Blues",
                alpha=style["kde_alpha"],
                zorder=1,
            )
            if density_thresholds is not None and density_thresholds.size:
                cs = ax.contour(
                    xx,
                    yy,
                    density,
                    levels=density_thresholds,
                    colors=[style["sim_color"]],
                    linewidths=0.9,
                    alpha=0.70,
                    zorder=1,
                )
                contour_handles = [
                    Line2D([0], [0], color=style["sim_color"], linewidth=1.0, alpha=0.70, label=f"Sim density {100.0 * mass:.0f}%")
                    for mass in contour_mass_levels[: len(cs.levels)]
                ]
        except Exception:
            pass

    exp_handles = []
    grand_mean_score = None
    if exp_scores.shape[0] > 0:
        grand_mean_score = np.mean(exp_scores, axis=0)
    for condition in condition_order:
        mask = exp_df["DATASET_CONTEXT_condition_id"].astype(str) == str(condition)
        if not np.any(mask.to_numpy(dtype=bool)):
            continue
        scores = exp_scores[mask.to_numpy(dtype=bool), :]
        if scores.size == 0:
            continue
        mean_score = np.mean(scores, axis=0)
        color = condition_colors.get(condition, "#777777")
        if experimental_display_mode == "participant_condition":
            ax.scatter(
                scores[:, 0],
                scores[:, 1],
                s=style["exp_point_size"],
                color=color,
                alpha=style["exp_point_alpha"],
                marker=style["exp_point_marker"],
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
                label=None,
            )
            ax.scatter(
                [mean_score[0]],
                [mean_score[1]],
                s=style["exp_mean_size"],
                color=color,
                alpha=style["exp_mean_alpha"],
                marker=style["exp_mean_marker"],
                edgecolors="white",
                linewidths=1.0,
                zorder=4,
            )
            exp_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=style["exp_point_marker"],
                    color="none",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markersize=max(6.0, np.sqrt(style["exp_point_size"])),
                    label=f"{condition} participants",
                )
            )
            exp_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=style["exp_mean_marker"],
                    color="none",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markersize=max(7.0, np.sqrt(style["exp_mean_size"])),
                    label=f"{condition} mean",
                )
            )
        elif experimental_display_mode == "condition_mean":
            ax.scatter(
                [mean_score[0]],
                [mean_score[1]],
                s=style["exp_mean_size"],
                color=color,
                alpha=style["exp_mean_alpha"],
                marker=style["exp_mean_marker"],
                edgecolors="white",
                linewidths=1.0,
                zorder=4,
            )
            exp_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=style["exp_mean_marker"],
                    color="none",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markersize=max(7.0, np.sqrt(style["exp_mean_size"])),
                    label=f"{condition} mean",
                )
            )

    if (experimental_display_mode == "grand_mean" or show_grand_mean_overlay) and grand_mean_score is not None:
        ax.scatter(
            [grand_mean_score[0]],
            [grand_mean_score[1]],
            s=style["grand_mean_size"],
            color=style["grand_mean_color"],
            alpha=style["grand_mean_alpha"],
            marker=style["grand_mean_marker"],
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
        )
        exp_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["grand_mean_marker"],
                color="none",
                markerfacecolor=style["grand_mean_color"],
                markeredgecolor="white",
                markersize=max(8.0, np.sqrt(style["grand_mean_size"])),
                label="Experimental grand mean",
            )
        )

    ax.set_title("Simulated vs experimental coverage in PCA space", fontsize=style["title_fontsize"])
    ax.set_xlabel(f"PC1 ({100.0 * explained[0]:.1f}% var)", fontsize=style["axis_label_fontsize"])
    ax.set_ylabel(f"PC2 ({100.0 * explained[1]:.1f}% var)", fontsize=style["axis_label_fontsize"])
    ax.tick_params(axis="both", labelsize=style["tick_labelsize"])
    ax.grid(alpha=style["grid_alpha"])
    legend_handles = []
    if style.get("show_sim_points", True):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=style["sim_scatter_marker"],
                color="none",
                markerfacecolor=style["sim_color"],
                markeredgecolor="none",
                alpha=style["sim_scatter_alpha"],
                markersize=max(5.5, np.sqrt(style["sim_scatter_size"])),
                label="Simulated parameter sets",
            )
        )
    legend_handles.extend(contour_handles)
    legend_handles.extend(exp_handles)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", frameon=True)
    fig.tight_layout()
    return fig, used_features


def build_coverage_summary_table(
    sim_df,
    exp_df,
    selected_features,
    *,
    condition_order=DEFAULT_CONDITION_ORDER,
):
    rows = []
    for feature in selected_features:
        sim_vals = pd.to_numeric(sim_df[feature], errors="coerce").to_numpy(dtype=float)
        sim_vals = sim_vals[np.isfinite(sim_vals)]
        if sim_vals.size == 0:
            continue
        for condition in condition_order:
            cond_vals = pd.to_numeric(
                exp_df.loc[exp_df["DATASET_CONTEXT_condition_id"].astype(str) == str(condition), feature],
                errors="coerce",
            ).to_numpy(dtype=float)
            cond_vals = cond_vals[np.isfinite(cond_vals)]
            if cond_vals.size == 0:
                continue
            exp_mean = float(np.mean(cond_vals))
            sim_fraction_within_exp_range = float(
                np.mean((sim_vals >= float(np.min(cond_vals))) & (sim_vals <= float(np.max(cond_vals))))
            )
            sim_percentile_of_exp_mean = float(np.mean(sim_vals <= exp_mean) * 100.0)
            rows.append(
                {
                    "feature": feature,
                    "feature_label": format_feature_label(feature),
                    "unit": _feature_unit(feature),
                    "condition_id": condition,
                    "n_sim": int(sim_vals.size),
                    "n_exp": int(cond_vals.size),
                    "sim_mean": float(np.mean(sim_vals)),
                    "sim_sd": float(np.std(sim_vals, ddof=0)),
                    "sim_q05": float(np.quantile(sim_vals, 0.05)),
                    "sim_q50": float(np.quantile(sim_vals, 0.50)),
                    "sim_q95": float(np.quantile(sim_vals, 0.95)),
                    "exp_mean": exp_mean,
                    "exp_sd": float(np.std(cond_vals, ddof=0)),
                    "exp_min": float(np.min(cond_vals)),
                    "exp_max": float(np.max(cond_vals)),
                    "sim_fraction_within_exp_range": sim_fraction_within_exp_range,
                    "sim_percentile_of_exp_mean": sim_percentile_of_exp_mean,
                }
            )
    return pd.DataFrame(rows)


def run_sim_exp_feature_coverage(
    simulated_table_path,
    experimental_table_path,
    *,
    selected_features,
    output_dir,
    condition_order=DEFAULT_CONDITION_ORDER,
    violin_style=None,
    pca_style=None,
    min_sim_units_kept=None,
    min_sim_unit_fraction_kept=None,
    experimental_display_mode="participant_condition",
    show_grand_mean_overlay=False,
):
    experimental_display_mode = _validate_experimental_display_mode(experimental_display_mode)
    sim_df = load_summary_table(simulated_table_path)
    exp_df = load_summary_table(experimental_table_path)
    sim_df_filtered = filter_simulated_rows(
        sim_df,
        min_units_kept=min_sim_units_kept,
        min_unit_fraction_kept=min_sim_unit_fraction_kept,
    )
    if sim_df_filtered.empty:
        raise ValueError("Simulated table is empty after applying kept-unit filters")
    features, dropped = validate_selected_features(sim_df_filtered, exp_df, selected_features)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    violin_fig = plot_feature_violin_coverage(
        sim_df_filtered,
        exp_df,
        features,
        condition_order=condition_order,
        violin_style=violin_style,
        experimental_display_mode=experimental_display_mode,
        show_grand_mean_overlay=show_grand_mean_overlay,
    )
    violin_path = output_dir / "sim_vs_exp_feature_violins.png"
    violin_fig.savefig(violin_path, bbox_inches="tight")

    pca_fig, used_pca_features = plot_sim_vs_exp_pca(
        sim_df_filtered,
        exp_df,
        features,
        condition_order=condition_order,
        pca_style=pca_style,
        experimental_display_mode=experimental_display_mode,
        show_grand_mean_overlay=show_grand_mean_overlay,
    )
    pca_path = output_dir / "sim_vs_exp_feature_pca.png"
    pca_fig.savefig(pca_path, bbox_inches="tight")

    summary_df = build_coverage_summary_table(
        sim_df_filtered,
        exp_df,
        features,
        condition_order=condition_order,
    )
    summary_path = output_dir / "sim_vs_exp_feature_coverage_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "sim_df": sim_df,
        "sim_df_filtered": sim_df_filtered,
        "exp_df": exp_df,
        "selected_features": features,
        "dropped_features": dropped,
        "used_pca_features": used_pca_features,
        "n_sim_before_filter": int(len(sim_df)),
        "n_sim_after_filter": int(len(sim_df_filtered)),
        "violin_fig": violin_fig,
        "pca_fig": pca_fig,
        "summary_df": summary_df,
        "experimental_display_mode": experimental_display_mode,
        "show_grand_mean_overlay": bool(show_grand_mean_overlay),
        "violin_path": violin_path,
        "pca_path": pca_path,
        "summary_csv_path": summary_path,
    }
