import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr


DEFAULT_GRID_STYLE = {
    "figure_dpi": 130,
    "cell_size": (3.0, 2.6),
    "scatter_s": 22,
    "scatter_alpha": 0.65,
    "scatter_color": "#1f77b4",
    "show_regression_line": False,
    "regression_lw": 1.2,
    "regression_alpha": 0.80,
    "regression_ls": "-",
    "font_family": "DejaVu Sans",
    "label_fontsize": 9,
    "annotation_fontsize": 9,
    "title_fontsize": 10,
    "tick_labelsize": 8,
    "grid_alpha": 0.20,
    "xtick_rotation": 45,
}


DEFAULT_PAIRPLOT_STYLE = {
    "figure_dpi": 130,
    "cell_size": (2.8, 2.8),
    "scatter_s": 18,
    "scatter_alpha": 0.55,
    "scatter_color": "#1f77b4",
    "font_family": "DejaVu Sans",
    "label_fontsize": 8,
    "hist_alpha": 0.75,
    "hist_color": "#1f77b4",
    "hist_edgecolor": "white",
    "kde_levels": 7,
    "kde_alpha": 0.45,
    "kde_cmap": "Blues",
    "title_fontsize": 12,
    "tick_labelsize": 8,
    "grid_alpha": 0.15,
}


DEFAULT_PCA_STYLE = {
    "figure_size": (8, 6.5),
    "figure_dpi": 140,
    "scatter_s": 26,
    "scatter_alpha": 0.65,
    "scatter_color": "#1f77b4",
    "font_family": "DejaVu Sans",
    "label_fontsize": 10,
    "tick_labelsize": 9,
    "kde_levels": 8,
    "kde_alpha": 0.45,
    "kde_cmap": "Blues",
    "grid_alpha": 0.2,
    "title_fontsize": 12,
}


SIM_PARAM_PREFIXES = (
    "SIM_PARAM_GENERAL_",
    "SIM_PARAM_BASELINE_INPUT_",
    "SIM_PARAM_BURST_INPUT_",
)

OBS_PREFIXES = (
    "OBS_FIRING_STATISTICS_",
    "OBS_BASELINE_",
    "OBS_BURST_FEATURES_",
)

ANALYSIS_PREFIXES = (
    "ANALYSIS_PARAM_",
)


def load_export_summary_table(path):
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        return pd.read_pickle(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported export table format: {path}")


def find_priors_batch_pickle(batch_root_dir):
    batch_root_dir = Path(batch_root_dir)
    matches = sorted(batch_root_dir.glob("*_priors_batch*.pkl"))
    if not matches:
        raise FileNotFoundError(f"No *_priors_batch*.pkl file found in {batch_root_dir}")
    return matches[0]


def load_priors_batch_metadata(priors_batch_path):
    with Path(priors_batch_path).open("rb") as f:
        return pickle.load(f)


def _raw_param_name(export_column_name):
    for prefix in SIM_PARAM_PREFIXES:
        if export_column_name.startswith(prefix):
            return export_column_name[len(prefix):]
    return export_column_name


def infer_parameter_sampling_modes(export_df, priors_payload=None):
    priors_payload = priors_payload or {}
    free_keys = set((priors_payload.get("free_parameter_bounds_prior") or {}).keys())
    parameter_sets_df = priors_payload.get("parameter_sets_df")
    if isinstance(parameter_sets_df, pd.DataFrame):
        set_df_cols = set(parameter_sets_df.columns)
    else:
        set_df_cols = set()
    metadata_cols = {"parameter_set_id", "repeat_index", "requested_random_seed", "subject", "intensity", "muscle"}
    specified_keys = {col for col in set_df_cols if col not in free_keys and col not in metadata_cols}

    sampling_modes = {}
    for column_name in export_df.columns:
        if not column_name.startswith("SIM_PARAM_"):
            continue
        raw_name = _raw_param_name(column_name)
        if raw_name in free_keys:
            sampling_modes[column_name] = "prior"
        elif raw_name in specified_keys:
            sampling_modes[column_name] = "specified"
        else:
            sampling_modes[column_name] = "fixed"
    return sampling_modes


def format_variable_label(name, multiline_prefix=True):
    prefixes = SIM_PARAM_PREFIXES + OBS_PREFIXES + ANALYSIS_PREFIXES
    for prefix in prefixes:
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            return f"{prefix[:-1]}\n{suffix}" if multiline_prefix else f"{prefix}{suffix}"
    return name


def build_variable_unit_map(unit_groups=None):
    unit_map = {}
    if not unit_groups:
        return unit_map
    for unit_label, variable_names in unit_groups.items():
        for variable_name in variable_names:
            unit_map[variable_name] = unit_label
    return unit_map


def _is_current_unit(unit_label):
    return unit_label in {"nA", "µA", "uA"}


def normalize_current_display_unit(display_current_unit):
    if display_current_unit in {"uA", "µA"}:
        return "µA"
    return "nA"


def displayed_unit_label(unit_label, display_current_unit="µA"):
    if _is_current_unit(unit_label):
        return normalize_current_display_unit(display_current_unit)
    return unit_label


def unit_display_scale(unit_label, display_current_unit="µA"):
    if _is_current_unit(unit_label):
        return 1e-3 if normalize_current_display_unit(display_current_unit) == "µA" else 1.0
    return 1.0


def _format_label_with_unit(name, variable_units=None, multiline_prefix=True, display_current_unit="µA"):
    base = format_variable_label(name, multiline_prefix=multiline_prefix)
    if variable_units is None:
        return base
    unit_label = variable_units.get(name)
    if not unit_label:
        return base
    return f"{base}\n({displayed_unit_label(unit_label, display_current_unit=display_current_unit)})"


def _darken_color(color, factor=0.65):
    rgba = np.array(plt.matplotlib.colors.to_rgba(color), dtype=float)
    rgba[:3] *= factor
    return tuple(rgba)


def _set_tick_fontfamily(ax, font_family):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(font_family)


def _to_scalar_or_category_series(series):
    values = []
    for value in series:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                try:
                    value = json.loads(stripped)
                except Exception:
                    pass
        if isinstance(value, (list, tuple, dict, np.ndarray)):
            values.append(str(value))
        else:
            values.append(value)
    converted = pd.Series(values, index=series.index)
    numeric = pd.to_numeric(converted, errors="coerce")
    numeric_fraction = float(np.mean(np.isfinite(numeric))) if len(numeric) else 0.0
    if numeric_fraction >= 0.95:
        return numeric, True
    return converted.astype(str), False


def _finite_xy(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _safe_pearsonr(x, y):
    x, y = _finite_xy(x, y)
    if x.size < 3 or y.size < 3:
        return np.nan
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return np.nan
    return float(pearsonr(x, y)[0])


def _scaled_numeric_series(series, unit_label=None, display_current_unit="µA"):
    numeric = pd.to_numeric(series, errors="coerce")
    factor = unit_display_scale(unit_label, display_current_unit=display_current_unit)
    return numeric * factor


def _compute_shared_limits(export_df, columns, variable_units, display_current_unit="µA"):
    limits = {}
    for column_name in columns:
        unit_label = variable_units.get(column_name)
        if not unit_label:
            continue
        numeric = _scaled_numeric_series(export_df[column_name], unit_label, display_current_unit=display_current_unit)
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            continue
        lo = float(finite.min())
        hi = float(finite.max())
        display_unit = displayed_unit_label(unit_label, display_current_unit=display_current_unit)
        if display_unit not in limits:
            limits[display_unit] = [lo, hi]
        else:
            limits[display_unit][0] = min(limits[display_unit][0], lo)
            limits[display_unit][1] = max(limits[display_unit][1], hi)
    return limits


def plot_parameter_observation_grid(
    export_df,
    parameter_columns,
    observation_columns,
    sampling_modes=None,
    style=None,
    observation_colors=None,
    unit_groups=None,
    display_current_unit="µA",
    share_axis_limits_by_unit=False,
    multiline_labels=True,
    output_path=None,
):
    style_resolved = dict(DEFAULT_GRID_STYLE)
    if style:
        style_resolved.update(style)
    sampling_modes = sampling_modes or {}
    variable_units = build_variable_unit_map(unit_groups)
    shared_x_limits = _compute_shared_limits(export_df, parameter_columns, variable_units, display_current_unit=display_current_unit) if share_axis_limits_by_unit else {}
    shared_y_limits = _compute_shared_limits(export_df, observation_columns, variable_units, display_current_unit=display_current_unit) if share_axis_limits_by_unit else {}

    n_rows = len(observation_columns)
    n_cols = len(parameter_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(style_resolved["cell_size"][0] * n_cols, style_resolved["cell_size"][1] * n_rows),
        dpi=style_resolved["figure_dpi"],
        squeeze=False,
    )

    for col_idx, param_col in enumerate(parameter_columns):
        x_unit = variable_units.get(param_col)
        x_series, is_numeric_x = _to_scalar_or_category_series(export_df[param_col])
        unique_categories = None
        plotted_x = x_series
        if not is_numeric_x:
            unique_categories = list(pd.unique(x_series))
            mapping = {value: idx for idx, value in enumerate(unique_categories)}
            plotted_x = x_series.map(mapping).astype(float)
        else:
            plotted_x = _scaled_numeric_series(x_series, x_unit, display_current_unit=display_current_unit)

        for row_idx, obs_col in enumerate(observation_columns):
            ax = axes[row_idx, col_idx]
            y_unit = variable_units.get(obs_col)
            y = _scaled_numeric_series(export_df[obs_col], y_unit, display_current_unit=display_current_unit)
            x = plotted_x if is_numeric_x else pd.to_numeric(plotted_x, errors="coerce")
            x_vals, y_vals = _finite_xy(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            obs_color = (observation_colors or {}).get(obs_col, style_resolved["scatter_color"])
            if x_vals.size:
                ax.scatter(
                    x_vals,
                    y_vals,
                    s=style_resolved["scatter_s"],
                    alpha=style_resolved["scatter_alpha"],
                    color=obs_color,
                    linewidths=0,
                )
                if style_resolved["show_regression_line"] and is_numeric_x and x_vals.size >= 2 and np.std(x_vals) > 0:
                    slope, intercept = np.polyfit(x_vals, y_vals, deg=1)
                    x_line = np.array([np.min(x_vals), np.max(x_vals)], dtype=float)
                    y_line = slope * x_line + intercept
                    ax.plot(
                        x_line,
                        y_line,
                        color=_darken_color(obs_color),
                        lw=style_resolved["regression_lw"],
                        alpha=style_resolved["regression_alpha"],
                        ls=style_resolved["regression_ls"],
                        zorder=5,
                    )

            r_value = _safe_pearsonr(np.asarray(x, dtype=float), np.asarray(y, dtype=float)) if is_numeric_x else np.nan
            if np.isfinite(r_value):
                ax.text(
                    0.03,
                    0.96,
                    f"r = {r_value:.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=style_resolved["annotation_fontsize"],
                    fontfamily=style_resolved["font_family"],
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )

            if row_idx == 0:
                mode = sampling_modes.get(param_col, "fixed")
                ax.set_title(
                    f"{_format_label_with_unit(param_col, variable_units=variable_units, multiline_prefix=multiline_labels, display_current_unit=display_current_unit)}\n[{mode}]",
                    fontsize=style_resolved["title_fontsize"],
                    fontfamily=style_resolved["font_family"],
                )
            if col_idx == 0:
                ax.set_ylabel(
                    _format_label_with_unit(obs_col, variable_units=variable_units, multiline_prefix=multiline_labels, display_current_unit=display_current_unit),
                    fontsize=style_resolved["label_fontsize"],
                    fontfamily=style_resolved["font_family"],
                )
            if row_idx == n_rows - 1:
                ax.set_xlabel(
                    _format_label_with_unit(param_col, variable_units=variable_units, multiline_prefix=multiline_labels, display_current_unit=display_current_unit),
                    fontsize=style_resolved["label_fontsize"],
                    fontfamily=style_resolved["font_family"],
                )
            if unique_categories is not None:
                ax.set_xticks(np.arange(len(unique_categories), dtype=float))
                ax.set_xticklabels(unique_categories, rotation=style_resolved["xtick_rotation"], ha="right")
            else:
                for label in ax.get_xticklabels():
                    label.set_rotation(style_resolved["xtick_rotation"])
                    label.set_ha("right")
            ax.tick_params(labelsize=style_resolved["tick_labelsize"])
            _set_tick_fontfamily(ax, style_resolved["font_family"])
            x_display_unit = displayed_unit_label(x_unit, display_current_unit=display_current_unit) if x_unit else None
            y_display_unit = displayed_unit_label(y_unit, display_current_unit=display_current_unit) if y_unit else None
            if is_numeric_x and x_display_unit in shared_x_limits:
                ax.set_xlim(shared_x_limits[x_display_unit][0], shared_x_limits[x_display_unit][1])
            if y_display_unit in shared_y_limits:
                ax.set_ylim(shared_y_limits[y_display_unit][0], shared_y_limits[y_display_unit][1])
            ax.grid(alpha=style_resolved["grid_alpha"])

    fig.suptitle(
        "Simulation parameters vs observed batch features",
        fontsize=style_resolved["title_fontsize"] + 2,
        fontfamily=style_resolved["font_family"],
    )
    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
    return fig


def _plot_2d_kde(ax, x, y, style):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 5 or y.size < 5 or np.std(x) <= 0 or np.std(y) <= 0:
        return
    kde = gaussian_kde(np.vstack([x, y]))
    x_grid = np.linspace(np.min(x), np.max(x), 80)
    y_grid = np.linspace(np.min(y), np.max(y), 80)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=style["kde_levels"], alpha=style["kde_alpha"], cmap=style["kde_cmap"])


def plot_observation_pairgrid(
    export_df,
    observation_columns,
    style=None,
    observation_colors=None,
    unit_groups=None,
    display_current_unit="µA",
    multiline_labels=True,
    show_kde=False,
    kde_only=False,
    output_path=None,
):
    style_resolved = dict(DEFAULT_PAIRPLOT_STYLE)
    if style:
        style_resolved.update(style)
    variable_units = build_variable_unit_map(unit_groups)

    n = len(observation_columns)
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(style_resolved["cell_size"][0] * n, style_resolved["cell_size"][1] * n),
        dpi=style_resolved["figure_dpi"],
        squeeze=False,
    )

    for i, y_col in enumerate(observation_columns):
        y = _scaled_numeric_series(export_df[y_col], variable_units.get(y_col), display_current_unit=display_current_unit)
        row_color = (observation_colors or {}).get(y_col, style_resolved["scatter_color"])
        for j, x_col in enumerate(observation_columns):
            ax = axes[i, j]
            x = _scaled_numeric_series(export_df[x_col], variable_units.get(x_col), display_current_unit=display_current_unit)
            if i == j:
                finite = x[np.isfinite(x)]
                if finite.size:
                    ax.hist(
                        finite,
                        bins="auto",
                        color=(observation_colors or {}).get(x_col, style_resolved["hist_color"]),
                        alpha=style_resolved["hist_alpha"],
                        edgecolor=style_resolved["hist_edgecolor"],
                    )
                ax.set_xlabel(
                    _format_label_with_unit(x_col, variable_units=variable_units, multiline_prefix=multiline_labels, display_current_unit=display_current_unit),
                    fontsize=style_resolved["label_fontsize"],
                    fontfamily=style_resolved["font_family"],
                )
                ax.set_ylabel("Count", fontsize=style_resolved["label_fontsize"], fontfamily=style_resolved["font_family"])
            else:
                x_vals, y_vals = _finite_xy(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
                if show_kde and x_vals.size:
                    _plot_2d_kde(ax, x_vals, y_vals, style_resolved)
                if (not kde_only) and x_vals.size:
                    ax.scatter(
                        x_vals,
                        y_vals,
                        s=style_resolved["scatter_s"],
                        alpha=style_resolved["scatter_alpha"],
                        color=row_color,
                        linewidths=0,
                    )
                ax.set_xlabel(
                    _format_label_with_unit(x_col, variable_units=variable_units, multiline_prefix=multiline_labels, display_current_unit=display_current_unit),
                    fontsize=style_resolved["label_fontsize"],
                    fontfamily=style_resolved["font_family"],
                )
                ax.set_ylabel(
                    _format_label_with_unit(y_col, variable_units=variable_units, multiline_prefix=multiline_labels, display_current_unit=display_current_unit),
                    fontsize=style_resolved["label_fontsize"],
                    fontfamily=style_resolved["font_family"],
                )
            ax.tick_params(labelsize=style_resolved["tick_labelsize"])
            _set_tick_fontfamily(ax, style_resolved["font_family"])
            ax.grid(alpha=style_resolved["grid_alpha"])
    fig.suptitle("Observation pairplot", fontsize=style_resolved["title_fontsize"], fontfamily=style_resolved["font_family"])
    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
    return fig


def compute_observation_pca(export_df, observation_columns):
    obs_df = export_df.loc[:, observation_columns].apply(pd.to_numeric, errors="coerce")
    obs_df = obs_df.dropna(axis=0, how="any")
    if obs_df.empty:
        raise ValueError("No complete rows available for PCA")

    variances = obs_df.var(axis=0, ddof=0)
    keep_cols = variances[variances > 0].index.tolist()
    if len(keep_cols) < 2:
        raise ValueError("At least two varying observation columns are required for PCA")
    if len(keep_cols) < len(observation_columns):
        dropped = [col for col in observation_columns if col not in keep_cols]
        warnings.warn(f"Dropping constant observation columns from PCA: {dropped}")

    x = obs_df[keep_cols].to_numpy(dtype=float)
    x = (x - np.mean(x, axis=0, keepdims=True)) / np.std(x, axis=0, ddof=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    scores = u[:, :2] * s[:2]
    explained_ratio = (s ** 2) / np.sum(s ** 2)
    return {
        "scores": scores,
        "explained_ratio": explained_ratio[:2],
        "used_columns": keep_cols,
        "row_index": obs_df.index.to_numpy(),
    }


def plot_observation_pca(
    export_df,
    observation_columns,
    style=None,
    show_kde=False,
    kde_only=False,
    output_path=None,
):
    style_resolved = dict(DEFAULT_PCA_STYLE)
    if style:
        style_resolved.update(style)
    pca = compute_observation_pca(export_df, observation_columns)
    scores = np.asarray(pca["scores"], dtype=float)
    pc1 = scores[:, 0]
    pc2 = scores[:, 1]

    fig, ax = plt.subplots(figsize=style_resolved["figure_size"], dpi=style_resolved["figure_dpi"])
    if show_kde:
        _plot_2d_kde(ax, pc1, pc2, {
            "kde_levels": style_resolved["kde_levels"],
            "kde_alpha": style_resolved["kde_alpha"],
            "kde_cmap": style_resolved["kde_cmap"],
        })
    if not kde_only:
        ax.scatter(
            pc1,
            pc2,
            s=style_resolved["scatter_s"],
            alpha=style_resolved["scatter_alpha"],
            color=style_resolved["scatter_color"],
            linewidths=0,
        )
    ax.set_xlabel(
        f"PC1 ({100 * pca['explained_ratio'][0]:.1f}% var)",
        fontsize=style_resolved["label_fontsize"],
        fontfamily=style_resolved["font_family"],
    )
    ax.set_ylabel(
        f"PC2 ({100 * pca['explained_ratio'][1]:.1f}% var)",
        fontsize=style_resolved["label_fontsize"],
        fontfamily=style_resolved["font_family"],
    )
    ax.set_title("Observation PCA", fontsize=style_resolved["title_fontsize"], fontfamily=style_resolved["font_family"])
    ax.tick_params(labelsize=style_resolved["tick_labelsize"])
    ax.grid(alpha=style_resolved["grid_alpha"])
    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
    return fig, pca
