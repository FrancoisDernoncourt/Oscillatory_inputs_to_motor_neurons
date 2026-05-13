from __future__ import annotations

import json
import math
import pickle
import random
import warnings
import colorsys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


OBS_PREFIXES = (
    "OBS_FIRING_STATISTICS_",
    "OBS_BASELINE_",
    "OBS_SPECTRUM_",
    "OBS_BURST_FEATURES_",
    "OBS_TREND_FEATURE_",
    "OBS_WINDOW_",
)

SIM_PARAM_PREFIXES = (
    "SIM_PARAM_GENERAL_",
    "SIM_PARAM_BASELINE_INPUT_",
    "SIM_PARAM_BURST_INPUT_",
    "SIM_PARAM_TREND_INPUT_",
)

EXPERIMENTAL_METADATA_COLUMNS = [
    "DATASET_CONTEXT_participant_id",
    "DATASET_CONTEXT_session_tag",
    "DATASET_CONTEXT_condition_id",
]

OBSERVATION_LEVELS = {
    "participant_condition",
    "condition_mean",
    "grand_mean",
}


def load_summary_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        return pd.read_pickle(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def read_columns_from_export(path: str | Path) -> list[str]:
    return list(load_summary_table(path).columns)


def discover_available_feature_groups(sim_df: pd.DataFrame, exp_df: pd.DataFrame) -> dict[str, list[str]]:
    sim_cols = set(sim_df.columns)
    exp_cols = set(exp_df.columns)
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


def discover_available_parameter_groups(sim_df: pd.DataFrame) -> dict[str, list[str]]:
    params = sorted(col for col in sim_df.columns if col.startswith("SIM_PARAM_"))
    groups = {
        "general": [],
        "baseline_input": [],
        "burst_input": [],
        "trend_input": [],
    }
    for col in params:
        if col.startswith("SIM_PARAM_GENERAL_"):
            groups["general"].append(col)
        elif col.startswith("SIM_PARAM_BASELINE_INPUT_"):
            groups["baseline_input"].append(col)
        elif col.startswith("SIM_PARAM_BURST_INPUT_"):
            groups["burst_input"].append(col)
        elif col.startswith("SIM_PARAM_TREND_INPUT_"):
            groups["trend_input"].append(col)
    return groups


def build_selection_templates(sim_df: pd.DataFrame, exp_df: pd.DataFrame) -> dict[str, str]:
    parameter_groups = discover_available_parameter_groups(sim_df)
    feature_groups = discover_available_feature_groups(sim_df, exp_df)

    parameter_lines = ["SELECTED_PARAMETERS = ["]
    for group_name, values in parameter_groups.items():
        if not values:
            continue
        parameter_lines.append(f"    # {group_name.replace('_', ' ').title()}")
        for value in values:
            parameter_lines.append(f'    # "{value}",')
        parameter_lines.append("")
    parameter_lines.append("]")

    feature_lines = ["SELECTED_FEATURES = ["]
    for group_name, values in feature_groups.items():
        if not values:
            continue
        feature_lines.append(f"    # {group_name.replace('_', ' ').title()}")
        for value in values:
            feature_lines.append(f'    # "{value}",')
        feature_lines.append("")
    feature_lines.append("]")

    return {
        "parameters": "\n".join(parameter_lines),
        "features": "\n".join(feature_lines),
    }


def resolve_priors_pickle_from_sim_table(sim_table_path: str | Path) -> Path:
    sim_table_path = Path(sim_table_path)
    batch_root = sim_table_path.parent.parent
    candidates = sorted(batch_root.glob("*priors_batch0*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find '*priors_batch0*.pkl' next to {sim_table_path.parent}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Found multiple priors_batch0 pickles under {batch_root}: {[str(path) for path in candidates]}"
        )
    return candidates[0]


def load_priors_bundle(priors_pickle_path: str | Path) -> dict[str, Any]:
    priors_pickle_path = Path(priors_pickle_path)
    with priors_pickle_path.open("rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict):
        raise TypeError(f"Unexpected priors pickle payload type: {type(bundle)}")
    return bundle


def _coerce_scalar_bound_pair(value: Any, parameter_name: str) -> tuple[float, float]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1 and arr.size == 2:
        low, high = float(arr[0]), float(arr[1])
    elif arr.ndim == 2 and arr.shape == (1, 2):
        low, high = float(arr[0, 0]), float(arr[0, 1])
    else:
        raise ValueError(
            f"Parameter {parameter_name!r} does not have scalar prior bounds. "
            f"Got shape {arr.shape}."
        )
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"Parameter {parameter_name!r} has non-finite prior bounds.")
    if high <= low:
        raise ValueError(f"Parameter {parameter_name!r} has invalid bounds [{low}, {high}].")
    return low, high


def extract_prior_bounds_for_parameters(
    priors_bundle: dict[str, Any],
    selected_parameters: list[str],
    parameter_bounds_overrides: dict[str, Any] | None = None,
) -> pd.DataFrame:
    free_bounds = dict(priors_bundle.get("free_parameter_bounds_prior", {}))
    specified_values = dict(priors_bundle.get("sample_at_specified_values", {}))
    parameter_bounds_overrides = dict(parameter_bounds_overrides or {})
    available_prior_keys = [*free_bounds.keys(), *specified_values.keys()]
    rows = []
    for parameter_name in selected_parameters:
        matched_prior_key = None
        if parameter_name in parameter_bounds_overrides:
            low, high = _coerce_scalar_bound_pair(parameter_bounds_overrides[parameter_name], parameter_name)
            source = "manual_override"
        else:
            candidate_keys: list[str] = []
            if parameter_name in free_bounds or parameter_name in specified_values:
                candidate_keys.append(parameter_name)
            candidate_keys.extend(
                sorted(
                    [
                        key
                        for key in available_prior_keys
                        if parameter_name.endswith(f"_{key}") or parameter_name == key
                    ],
                    key=len,
                    reverse=True,
                )
            )
            seen: set[str] = set()
            candidate_keys = [key for key in candidate_keys if not (key in seen or seen.add(key))]
            for candidate_key in candidate_keys:
                if candidate_key in free_bounds:
                    low, high = _coerce_scalar_bound_pair(free_bounds[candidate_key], parameter_name)
                    source = "free_parameter_bounds_prior"
                    matched_prior_key = candidate_key
                    break
                if candidate_key in specified_values:
                    values = np.asarray(specified_values[candidate_key], dtype=float).reshape(-1)
                    values = values[np.isfinite(values)]
                    if values.size < 2 or np.isclose(np.min(values), np.max(values)):
                        raise ValueError(
                            f"Parameter {parameter_name!r} only has one specified value in the priors pickle."
                        )
                    low, high = float(np.min(values)), float(np.max(values))
                    source = "sample_at_specified_values"
                    matched_prior_key = candidate_key
                    break
            else:
                raise KeyError(
                    f"Selected parameter {parameter_name!r} is missing from the priors pickle. "
                    f"Provide it in parameter_bounds_overrides if needed."
                )
            if matched_prior_key is not None and matched_prior_key != parameter_name:
                warnings.warn(
                    f"Resolved prior bounds for {parameter_name!r} from raw prior key "
                    f"{matched_prior_key!r}.",
                    stacklevel=2,
                )
        rows.append(
            {
                "parameter_name": parameter_name,
                "prior_key": matched_prior_key if matched_prior_key is not None else parameter_name,
                "source": source,
                "lower_bound": low,
                "upper_bound": high,
            }
        )
    return pd.DataFrame(rows)


def validate_selected_parameters(sim_df: pd.DataFrame, selected_parameters: list[str]) -> list[str]:
    if not selected_parameters:
        raise ValueError("selected_parameters is empty")
    missing = [param for param in selected_parameters if param not in sim_df.columns]
    if missing:
        raise KeyError(f"Selected parameters not found in simulated table: {missing}")
    usable = []
    for param in selected_parameters:
        numeric = pd.to_numeric(sim_df[param], errors="coerce")
        if np.isfinite(numeric).any():
            usable.append(param)
    if not usable:
        raise ValueError("No selected parameters contain usable numeric values in the simulated table")
    return usable


def validate_selected_features(
    sim_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    selected_features: list[str],
) -> list[str]:
    if not selected_features:
        raise ValueError("selected_features is empty")
    missing = [feat for feat in selected_features if feat not in sim_df.columns or feat not in exp_df.columns]
    if missing:
        raise KeyError(f"Selected features not found in both tables: {missing}")
    usable = []
    for feat in selected_features:
        sim_numeric = pd.to_numeric(sim_df[feat], errors="coerce")
        exp_numeric = pd.to_numeric(exp_df[feat], errors="coerce")
        if np.isfinite(sim_numeric).any() and np.isfinite(exp_numeric).any():
            usable.append(feat)
    if not usable:
        raise ValueError("No selected features have usable numeric values in both tables")
    return usable


def _normalize_condition_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    return str(value)


def aggregate_experimental_observations(
    exp_df: pd.DataFrame,
    *,
    selected_features: list[str],
    observation_level: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if observation_level not in OBSERVATION_LEVELS:
        raise ValueError(f"Unsupported observation_level: {observation_level}")

    work_df = exp_df.copy()
    for feature in selected_features:
        work_df[feature] = pd.to_numeric(work_df[feature], errors="coerce")

    missing_feature_mask = work_df[selected_features].isna().any(axis=1)
    skipped_rows = work_df.loc[missing_feature_mask, EXPERIMENTAL_METADATA_COLUMNS + selected_features].copy()
    valid_df = work_df.loc[~missing_feature_mask].copy()

    if valid_df.empty:
        raise ValueError("No experimental rows remain after dropping rows with missing selected features")

    if observation_level == "participant_condition":
        aggregated = valid_df[EXPERIMENTAL_METADATA_COLUMNS + selected_features].copy()
        aggregated["observation_label"] = aggregated.apply(
            lambda row: " | ".join(
                [
                    _normalize_condition_value(row["DATASET_CONTEXT_participant_id"]),
                    _normalize_condition_value(row["DATASET_CONTEXT_session_tag"]),
                    _normalize_condition_value(row["DATASET_CONTEXT_condition_id"]),
                ]
            ),
            axis=1,
        )
        aggregated["n_rows_aggregated"] = 1
        ordered_cols = ["observation_label", *EXPERIMENTAL_METADATA_COLUMNS, "n_rows_aggregated", *selected_features]
        return aggregated[ordered_cols].reset_index(drop=True), skipped_rows.reset_index(drop=True)

    if observation_level == "condition_mean":
        group_key = "DATASET_CONTEXT_condition_id"
        grouped = (
            valid_df.groupby(group_key, dropna=False, sort=True)[selected_features]
            .mean()
            .reset_index()
        )
        counts = valid_df.groupby(group_key, dropna=False, sort=True).size().reset_index(name="n_rows_aggregated")
        aggregated = grouped.merge(counts, on=group_key, how="left")
        aggregated.insert(0, "observation_label", aggregated[group_key].map(_normalize_condition_value))
        aggregated.insert(1, "DATASET_CONTEXT_participant_id", np.nan)
        aggregated.insert(2, "DATASET_CONTEXT_session_tag", np.nan)
        ordered_cols = ["observation_label", *EXPERIMENTAL_METADATA_COLUMNS, "n_rows_aggregated", *selected_features]
        return aggregated[ordered_cols].reset_index(drop=True), skipped_rows.reset_index(drop=True)

    aggregated = pd.DataFrame([{feature: float(valid_df[feature].mean()) for feature in selected_features}])
    aggregated.insert(0, "observation_label", "grand_mean")
    aggregated.insert(1, "DATASET_CONTEXT_participant_id", np.nan)
    aggregated.insert(2, "DATASET_CONTEXT_session_tag", np.nan)
    aggregated.insert(3, "DATASET_CONTEXT_condition_id", "ALL")
    aggregated["n_rows_aggregated"] = int(len(valid_df))
    ordered_cols = ["observation_label", *EXPERIMENTAL_METADATA_COLUMNS, "n_rows_aggregated", *selected_features]
    return aggregated[ordered_cols].reset_index(drop=True), skipped_rows.reset_index(drop=True)


@dataclass
class ParameterBoundsScaler:
    parameter_names: list[str]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        denom = self.upper_bounds - self.lower_bounds
        if np.any(denom <= 0):
            raise ValueError("Invalid parameter bounds for normalization")
        normalized = (values - self.lower_bounds) / denom
        return normalized

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return self.lower_bounds + values * (self.upper_bounds - self.lower_bounds)


@dataclass
class FeatureZScoreScaler:
    feature_names: list[str]
    means: np.ndarray
    stds: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return (values - self.means) / self.stds


def _fit_feature_zscore_scaler_from_rows(
    rows_df: pd.DataFrame,
    *,
    selected_features: list[str],
    source_label: str,
    min_rows: int = 1,
) -> tuple[FeatureZScoreScaler, pd.DataFrame, pd.DataFrame]:
    work_df = rows_df.copy()
    for feature_name in selected_features:
        work_df[feature_name] = pd.to_numeric(work_df[feature_name], errors="coerce")
    valid_mask = ~work_df[selected_features].isna().any(axis=1)
    rows_used = work_df.loc[valid_mask].copy()
    skipped_rows = work_df.loc[~valid_mask].copy()
    if len(rows_used) < int(min_rows):
        raise ValueError(
            f"Feature normalization source {source_label!r} has only {len(rows_used)} usable rows; "
            f"at least {int(min_rows)} are required."
        )
    values = rows_used[selected_features].to_numpy(dtype=float)
    feature_means = np.nanmean(values, axis=0)
    feature_stds = np.nanstd(values, axis=0)
    bad_mask = (~np.isfinite(feature_stds)) | (feature_stds <= 0)
    if np.any(bad_mask):
        bad_features = [selected_features[idx] for idx in np.where(bad_mask)[0]]
        raise ValueError(
            f"Some selected features have zero or non-finite standard deviation for "
            f"feature normalization source {source_label!r}: {bad_features}"
        )
    return (
        FeatureZScoreScaler(
            feature_names=list(selected_features),
            means=feature_means,
            stds=feature_stds,
        ),
        rows_used,
        skipped_rows,
    )


def build_feature_scaling_metadata(
    *,
    feature_scaler: FeatureZScoreScaler,
    source: str,
    selected_features: list[str],
    scaler_row_count: int,
    experimental_level: str | None = None,
    skipped_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    skipped_rows = pd.DataFrame() if skipped_rows is None else skipped_rows
    return {
        "feature_normalization_source": str(source),
        "feature_normalization_experimental_level": None if experimental_level is None else str(experimental_level),
        "scaler_row_count": int(scaler_row_count),
        "selected_features": list(selected_features),
        "feature_means": feature_scaler.means.tolist(),
        "feature_stds": feature_scaler.stds.tolist(),
        "feature_scaling_by_feature": [
            {
                "feature": str(feature_name),
                "mean": float(mean),
                "std": float(std),
            }
            for feature_name, mean, std in zip(selected_features, feature_scaler.means, feature_scaler.stds, strict=True)
        ],
        "n_skipped_feature_normalization_rows": int(len(skipped_rows)),
        "skipped_feature_normalization_rows": skipped_rows.to_dict(orient="records") if not skipped_rows.empty else [],
    }


def get_feature_scaling_metadata(dataset: "InferenceDatasetBundle") -> dict[str, Any]:
    metadata = getattr(dataset, "feature_scaling_metadata", None)
    if isinstance(metadata, dict) and metadata:
        return dict(metadata)
    return build_feature_scaling_metadata(
        feature_scaler=dataset.feature_scaler,
        source="simulated",
        selected_features=list(dataset.selected_features),
        scaler_row_count=int(len(dataset.sim_rows_used)),
        experimental_level=None,
        skipped_rows=pd.DataFrame(),
    )


@dataclass
class InferenceDatasetBundle:
    sim_table_path: Path
    exp_table_path: Path
    priors_pickle_path: Path
    sim_rows_used: pd.DataFrame
    exp_observations: pd.DataFrame
    skipped_experimental_rows: pd.DataFrame
    selected_parameters: list[str]
    selected_features: list[str]
    parameter_bounds: pd.DataFrame
    parameter_scaler: ParameterBoundsScaler
    feature_scaler: FeatureZScoreScaler
    theta_unit: np.ndarray
    x_sim_z: np.ndarray
    x_exp_z: np.ndarray
    feature_scaling_metadata: dict[str, Any] | None = None


def prepare_observation_dataset_for_inference(
    *,
    exp_table_path: str | Path,
    trained_dataset: InferenceDatasetBundle,
    observation_level: str,
) -> InferenceDatasetBundle:
    exp_table_path = Path(exp_table_path)
    exp_df = load_summary_table(exp_table_path)
    exp_observations, skipped_experimental_rows = aggregate_experimental_observations(
        exp_df,
        selected_features=trained_dataset.selected_features,
        observation_level=observation_level,
    )
    x_exp_raw = exp_observations[trained_dataset.selected_features].to_numpy(dtype=float)
    x_exp_z = trained_dataset.feature_scaler.transform(x_exp_raw)
    return InferenceDatasetBundle(
        sim_table_path=trained_dataset.sim_table_path,
        exp_table_path=exp_table_path,
        priors_pickle_path=trained_dataset.priors_pickle_path,
        sim_rows_used=trained_dataset.sim_rows_used,
        exp_observations=exp_observations,
        skipped_experimental_rows=skipped_experimental_rows,
        selected_parameters=list(trained_dataset.selected_parameters),
        selected_features=list(trained_dataset.selected_features),
        parameter_bounds=trained_dataset.parameter_bounds.copy(),
        parameter_scaler=trained_dataset.parameter_scaler,
        feature_scaler=trained_dataset.feature_scaler,
        theta_unit=trained_dataset.theta_unit,
        x_sim_z=trained_dataset.x_sim_z,
        x_exp_z=x_exp_z,
        feature_scaling_metadata=get_feature_scaling_metadata(trained_dataset),
    )


def prepare_inference_dataset(
    *,
    sim_table_path: str | Path,
    exp_table_path: str | Path,
    selected_parameters: list[str],
    selected_features: list[str],
    observation_level: str,
    parameter_bounds_overrides: dict[str, Any] | None = None,
    feature_normalization_source: str = "simulated",
    feature_normalization_experimental_level: str | None = None,
) -> InferenceDatasetBundle:
    sim_table_path = Path(sim_table_path)
    exp_table_path = Path(exp_table_path)

    sim_df = load_summary_table(sim_table_path)
    exp_df = load_summary_table(exp_table_path)
    selected_parameters = validate_selected_parameters(sim_df, selected_parameters)
    selected_features = validate_selected_features(sim_df, exp_df, selected_features)

    priors_pickle_path = resolve_priors_pickle_from_sim_table(sim_table_path)
    priors_bundle = load_priors_bundle(priors_pickle_path)
    parameter_bounds = extract_prior_bounds_for_parameters(
        priors_bundle,
        selected_parameters,
        parameter_bounds_overrides=parameter_bounds_overrides,
    )

    work_sim_df = sim_df.copy()
    for parameter_name in selected_parameters:
        work_sim_df[parameter_name] = pd.to_numeric(work_sim_df[parameter_name], errors="coerce")
    for feature_name in selected_features:
        work_sim_df[feature_name] = pd.to_numeric(work_sim_df[feature_name], errors="coerce")

    sim_required_columns = [*selected_parameters, *selected_features]
    valid_sim_mask = ~work_sim_df[sim_required_columns].isna().any(axis=1)
    sim_rows_used = work_sim_df.loc[valid_sim_mask].copy()
    if sim_rows_used.empty:
        raise ValueError("No simulated rows remain after dropping rows with missing selected parameters/features")

    lower = parameter_bounds["lower_bound"].to_numpy(dtype=float)
    upper = parameter_bounds["upper_bound"].to_numpy(dtype=float)
    theta_raw = sim_rows_used[selected_parameters].to_numpy(dtype=float)
    outside_mask = (theta_raw < lower) | (theta_raw > upper)
    if np.any(outside_mask):
        bad_param_indices = np.where(np.any(outside_mask, axis=0))[0].tolist()
        bad_params = [selected_parameters[idx] for idx in bad_param_indices]
        raise ValueError(
            f"Some simulated parameter values fall outside the loaded prior bounds: {bad_params}"
        )
    parameter_scaler = ParameterBoundsScaler(
        parameter_names=list(selected_parameters),
        lower_bounds=lower,
        upper_bounds=upper,
    )
    theta_unit = parameter_scaler.transform(theta_raw)

    exp_observations, skipped_experimental_rows = aggregate_experimental_observations(
        exp_df,
        selected_features=selected_features,
        observation_level=observation_level,
    )
    feature_normalization_source = str(feature_normalization_source).strip().lower()
    if feature_normalization_source == "simulated":
        feature_scaler, scaler_rows_used, skipped_feature_normalization_rows = _fit_feature_zscore_scaler_from_rows(
            sim_rows_used,
            selected_features=selected_features,
            source_label="simulated training rows",
            min_rows=1,
        )
        feature_scaling_metadata = build_feature_scaling_metadata(
            feature_scaler=feature_scaler,
            source="simulated",
            selected_features=selected_features,
            scaler_row_count=len(scaler_rows_used),
            experimental_level=None,
            skipped_rows=skipped_feature_normalization_rows,
        )
    elif feature_normalization_source == "experimental":
        if feature_normalization_experimental_level is None:
            raise ValueError(
                "feature_normalization_experimental_level must be set when "
                "feature_normalization_source='experimental'"
            )
        exp_scaling_observations, skipped_feature_normalization_rows = aggregate_experimental_observations(
            exp_df,
            selected_features=selected_features,
            observation_level=str(feature_normalization_experimental_level),
        )
        feature_scaler, scaler_rows_used, additional_skipped_rows = _fit_feature_zscore_scaler_from_rows(
            exp_scaling_observations,
            selected_features=selected_features,
            source_label=f"experimental observations at {feature_normalization_experimental_level!r}",
            min_rows=2,
        )
        if not additional_skipped_rows.empty:
            skipped_feature_normalization_rows = pd.concat(
                [skipped_feature_normalization_rows, additional_skipped_rows],
                ignore_index=True,
                sort=False,
            )
        feature_scaling_metadata = build_feature_scaling_metadata(
            feature_scaler=feature_scaler,
            source="experimental",
            selected_features=selected_features,
            scaler_row_count=len(scaler_rows_used),
            experimental_level=str(feature_normalization_experimental_level),
            skipped_rows=skipped_feature_normalization_rows,
        )
    else:
        raise ValueError(
            "feature_normalization_source must be either 'simulated' or 'experimental', "
            f"got {feature_normalization_source!r}"
        )

    x_sim_raw = sim_rows_used[selected_features].to_numpy(dtype=float)
    x_sim_z = feature_scaler.transform(x_sim_raw)
    x_exp_raw = exp_observations[selected_features].to_numpy(dtype=float)
    x_exp_z = feature_scaler.transform(x_exp_raw)

    return InferenceDatasetBundle(
        sim_table_path=sim_table_path,
        exp_table_path=exp_table_path,
        priors_pickle_path=priors_pickle_path,
        sim_rows_used=sim_rows_used,
        exp_observations=exp_observations,
        skipped_experimental_rows=skipped_experimental_rows,
        selected_parameters=list(selected_parameters),
        selected_features=list(selected_features),
        parameter_bounds=parameter_bounds,
        parameter_scaler=parameter_scaler,
        feature_scaler=feature_scaler,
        theta_unit=theta_unit,
        x_sim_z=x_sim_z,
        x_exp_z=x_exp_z,
        feature_scaling_metadata=feature_scaling_metadata,
    )


def _safe_torch_tensor(values: np.ndarray, *, device: str = "cpu"):
    import torch

    arr = np.asarray(values, dtype=np.float32)
    return torch.tensor(arr.tolist(), dtype=torch.float32, device=device)


@dataclass
class TrainedPosteriorBundle:
    inference: Any
    density_estimator: Any
    posterior: Any
    device: str
    density_estimator_name: str
    train_kwargs: dict[str, Any]
    num_simulations: int
    training_summary: dict[str, Any]


@dataclass
class EnsemblePosteriorBundle:
    networks: list[TrainedPosteriorBundle]
    network_seeds: list[int]
    ensemble_metadata: dict[str, Any]


def _is_ensemble_bundle(trained: Any) -> bool:
    return isinstance(trained, EnsemblePosteriorBundle)


def _iter_trained_networks(
    trained: TrainedPosteriorBundle | EnsemblePosteriorBundle,
) -> list[tuple[int, int | None, TrainedPosteriorBundle]]:
    if isinstance(trained, EnsemblePosteriorBundle):
        return [
            (
                int(network_id),
                int(trained.network_seeds[network_id]) if network_id < len(trained.network_seeds) else None,
                network,
            )
            for network_id, network in enumerate(trained.networks)
        ]
    if isinstance(trained, TrainedPosteriorBundle):
        seed = (trained.training_summary or {}).get("network_seed")
        return [(0, int(seed) if seed is not None else None, trained)]
    raise TypeError(f"Expected TrainedPosteriorBundle or EnsemblePosteriorBundle, got {type(trained).__name__}")


def allocate_ensemble_sample_counts(total_samples: int, n_networks: int) -> list[int]:
    total_samples = _coerce_positive_int_like(total_samples, name="total_samples")
    n_networks = _coerce_positive_int_like(n_networks, name="n_networks")
    base = total_samples // n_networks
    remainder = total_samples % n_networks
    counts = [base + (1 if idx < remainder else 0) for idx in range(n_networks)]
    if any(count <= 0 for count in counts):
        raise ValueError(
            f"total_samples={total_samples} is too small for n_networks={n_networks}; "
            "each network must receive at least one sample"
        )
    return counts


def _generate_ensemble_network_seeds(n_networks: int, base_seed: int) -> list[int]:
    n_networks = _coerce_positive_int_like(n_networks, name="n_networks")
    rng = np.random.default_rng(int(base_seed))
    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < n_networks:
        candidate = int(rng.integers(0, 2**31 - 1, dtype=np.int64))
        if candidate in seen:
            continue
        seen.add(candidate)
        seeds.append(candidate)
    return seeds


def _coerce_positive_int_like(value: Any, *, name: str) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise TypeError(f"{name} must be scalar-like, got {type(value).__name__} with length {len(value)}")
        value = value[0]
    if isinstance(value, np.ndarray):
        flat = np.asarray(value).reshape(-1)
        if flat.size != 1:
            raise TypeError(f"{name} must be scalar-like, got ndarray with shape {np.asarray(value).shape}")
        value = flat[0]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise TypeError(f"{name} must be scalar-like, got empty string")
        value = float(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise TypeError(f"{name} must be finite")
        if not float(value).is_integer():
            raise TypeError(f"{name} must be an integer-like value, got {value!r}")
        value = int(value)
    elif isinstance(value, (int, np.integer)):
        value = int(value)
    else:
        raise TypeError(f"{name} must be an integer-like scalar, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def train_amortized_posterior(
    dataset: InferenceDatasetBundle,
    *,
    density_estimator: str = "maf",
    device: str = "cpu",
    show_progress_bars: bool = True,
    train_kwargs: dict[str, Any] | None = None,
    random_seed: int | None = None,
) -> TrainedPosteriorBundle:
    import torch
    from sbi import utils as sbi_utils
    from sbi.inference import SNPE

    if random_seed is not None:
        seed_int = int(random_seed)
        random.seed(seed_int)
        np.random.seed(seed_int % (2**32 - 1))
        torch.manual_seed(seed_int)

    theta_tensor = _safe_torch_tensor(dataset.theta_unit, device=device)
    x_tensor = _safe_torch_tensor(dataset.x_sim_z, device=device)
    n_params = len(dataset.selected_parameters)
    prior = sbi_utils.BoxUniform(
        low=torch.zeros(n_params, dtype=torch.float32, device=device),
        high=torch.ones(n_params, dtype=torch.float32, device=device),
    )
    inference = SNPE(
        prior=prior,
        density_estimator=density_estimator,
        device=device,
        show_progress_bars=show_progress_bars,
    )
    inference = inference.append_simulations(theta_tensor, x_tensor)
    effective_train_kwargs = {
        "show_train_summary": False,
    }
    if train_kwargs:
        effective_train_kwargs.update(train_kwargs)
    density = inference.train(**effective_train_kwargs)
    posterior = inference.build_posterior(
        density,
        prior=prior,
        sample_with="direct",
    )
    training_summary = {
        key: _jsonable(val)
        for key, val in getattr(inference, "_summary", {}).items()
    }
    if random_seed is not None:
        training_summary["network_seed"] = int(random_seed)
    return TrainedPosteriorBundle(
        inference=inference,
        density_estimator=density,
        posterior=posterior,
        device=device,
        density_estimator_name=str(density_estimator),
        train_kwargs=effective_train_kwargs,
        num_simulations=int(dataset.theta_unit.shape[0]),
        training_summary=training_summary,
    )


def sample_posteriors_for_observations(
    trained: TrainedPosteriorBundle,
    dataset: InferenceDatasetBundle,
    *,
    num_samples: int = 1000,
    seed: int = 12345,
) -> pd.DataFrame:
    import torch

    num_samples = _coerce_positive_int_like(num_samples, name="num_samples")
    torch.manual_seed(int(seed))

    rows = []
    for obs_idx, observation in dataset.exp_observations.iterrows():
        x_obs = dataset.x_exp_z[obs_idx : obs_idx + 1, :]
        x_tensor = _safe_torch_tensor(x_obs, device=trained.device)
        posterior_samples = trained.posterior.sample((int(num_samples),), x=x_tensor)
        posterior_unit = np.asarray(posterior_samples.detach().cpu().tolist(), dtype=float)
        posterior_original = dataset.parameter_scaler.inverse_transform(posterior_unit)
        for sample_idx, sample_row in enumerate(posterior_original):
            row = {
                "observation_index": int(obs_idx),
                "observation_label": observation["observation_label"],
                "sample_index": int(sample_idx),
                "DATASET_CONTEXT_participant_id": observation.get("DATASET_CONTEXT_participant_id", np.nan),
                "DATASET_CONTEXT_session_tag": observation.get("DATASET_CONTEXT_session_tag", np.nan),
                "DATASET_CONTEXT_condition_id": observation.get("DATASET_CONTEXT_condition_id", np.nan),
            }
            for param_name, value in zip(dataset.selected_parameters, sample_row, strict=True):
                row[param_name] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def train_posterior_ensemble(
    dataset: InferenceDatasetBundle,
    *,
    n_networks: int = 1,
    ensemble_base_seed: int = 12345,
    density_estimator: str = "maf",
    device: str = "cpu",
    show_progress_bars: bool = True,
    train_kwargs: dict[str, Any] | None = None,
) -> TrainedPosteriorBundle | EnsemblePosteriorBundle:
    n_networks = _coerce_positive_int_like(n_networks, name="n_networks")
    network_seeds = _generate_ensemble_network_seeds(n_networks, int(ensemble_base_seed))
    networks: list[TrainedPosteriorBundle] = []
    for network_id, network_seed in enumerate(network_seeds):
        trained = train_amortized_posterior(
            dataset,
            density_estimator=density_estimator,
            device=device,
            show_progress_bars=show_progress_bars,
            train_kwargs=train_kwargs,
            random_seed=network_seed,
        )
        trained.training_summary = dict(trained.training_summary or {})
        trained.training_summary.update(
            {
                "network_id": int(network_id),
                "network_seed": int(network_seed),
                "ensemble_base_seed": int(ensemble_base_seed),
                "ensemble_n_networks": int(n_networks),
            }
        )
        networks.append(trained)
    if n_networks == 1:
        return networks[0]
    return EnsemblePosteriorBundle(
        networks=networks,
        network_seeds=network_seeds,
        ensemble_metadata={
            "n_networks": int(n_networks),
            "ensemble_base_seed": int(ensemble_base_seed),
            "sample_weighting": "equal",
            "density_estimator": str(density_estimator),
            "device": str(device),
            "train_kwargs": dict(networks[0].train_kwargs if networks else (train_kwargs or {})),
            "num_simulations": int(dataset.theta_unit.shape[0]),
        },
    )


def sample_ensemble_posteriors_for_observations(
    trained: TrainedPosteriorBundle | EnsemblePosteriorBundle,
    dataset: InferenceDatasetBundle,
    *,
    num_samples: int = 1000,
    seed: int = 12345,
) -> pd.DataFrame:
    networks = _iter_trained_networks(trained)
    sample_counts = allocate_ensemble_sample_counts(int(num_samples), len(networks))
    per_network_frames: list[pd.DataFrame] = []
    for network_pos, ((network_id, network_seed, network), sample_count) in enumerate(zip(networks, sample_counts, strict=True)):
        frame = sample_posteriors_for_observations(
            network,
            dataset,
            num_samples=int(sample_count),
            seed=int(seed) + int(network_pos),
        )
        if frame.empty:
            continue
        frame.insert(2, "network_id", int(network_id))
        frame.insert(3, "network_seed", np.nan if network_seed is None else int(network_seed))
        frame.insert(4, "network_sample_index", frame["sample_index"].astype(int))
        per_network_frames.append(frame)
    if not per_network_frames:
        return pd.DataFrame()
    pooled = pd.concat(per_network_frames, ignore_index=True)
    pooled["ensemble_sample_index"] = pooled.groupby("observation_label", sort=False).cumcount().astype(int)
    column_order = [
        "observation_index",
        "observation_label",
        "sample_index",
        "ensemble_sample_index",
        "network_id",
        "network_seed",
        "network_sample_index",
        *[col for col in pooled.columns if col not in {
            "observation_index",
            "observation_label",
            "sample_index",
            "ensemble_sample_index",
            "network_id",
            "network_seed",
            "network_sample_index",
        }],
    ]
    return pooled.reindex(columns=column_order)


def create_run_output_dir(root_dir: str | Path, *, observation_level: str) -> Path:
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{observation_level}__run_"
    existing_indices = []
    for child in root_dir.iterdir():
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        suffix = child.name[len(prefix) :]
        if suffix.isdigit():
            existing_indices.append(int(suffix))
    next_index = max(existing_indices, default=0) + 1
    run_dir = root_dir / f"{prefix}{next_index:04d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_trained_inference_bundle(
    *,
    output_dir: str | Path,
    dataset: InferenceDatasetBundle,
    trained: TrainedPosteriorBundle | EnsemblePosteriorBundle,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "trained_inference_bundle.pkl"
    with bundle_path.open("wb") as f:
        pickle.dump({"dataset": dataset, "trained": trained}, f)
    return bundle_path


def load_trained_inference_bundle(bundle_path: str | Path) -> tuple[InferenceDatasetBundle, TrainedPosteriorBundle | EnsemblePosteriorBundle]:
    bundle_path = Path(bundle_path)
    with bundle_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "dataset" not in payload or "trained" not in payload:
        raise TypeError(f"Unexpected trained inference bundle payload in {bundle_path}")
    dataset = payload["dataset"]
    trained = payload["trained"]
    if not isinstance(dataset, InferenceDatasetBundle):
        raise TypeError(f"Unexpected dataset payload type in {bundle_path}: {type(dataset)}")
    if not isinstance(trained, (TrainedPosteriorBundle, EnsemblePosteriorBundle)):
        raise TypeError(f"Unexpected trained payload type in {bundle_path}: {type(trained)}")
    return dataset, trained


def summarize_posterior_samples(
    posterior_samples_df: pd.DataFrame,
    *,
    parameter_names: list[str],
) -> pd.DataFrame:
    rows = []
    for observation_label, subset in posterior_samples_df.groupby("observation_label", sort=True):
        first = subset.iloc[0]
        for parameter_name in parameter_names:
            values = pd.to_numeric(subset[parameter_name], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            rows.append(
                {
                    "observation_label": observation_label,
                    "DATASET_CONTEXT_participant_id": first.get("DATASET_CONTEXT_participant_id", np.nan),
                    "DATASET_CONTEXT_session_tag": first.get("DATASET_CONTEXT_session_tag", np.nan),
                    "DATASET_CONTEXT_condition_id": first.get("DATASET_CONTEXT_condition_id", np.nan),
                    "parameter_name": parameter_name,
                    "posterior_mean": float(np.mean(finite)),
                    "posterior_median": float(np.median(finite)),
                    "posterior_q05": float(np.quantile(finite, 0.05)),
                    "posterior_q95": float(np.quantile(finite, 0.95)),
                }
            )
    return pd.DataFrame(rows)


def format_parameter_label(parameter_name: str) -> str:
    for prefix in SIM_PARAM_PREFIXES:
        if parameter_name.startswith(prefix):
            return prefix[:-1] + "\n" + parameter_name[len(prefix) :]
    return parameter_name


def format_feature_label(feature_name: str) -> str:
    for prefix in OBS_PREFIXES:
        if feature_name.startswith(prefix):
            return prefix[:-1] + "\n" + feature_name[len(prefix) :]
    return feature_name


def _resolve_name_color(name: str, color_map: dict[str, str] | None, fallback: str = "black") -> str:
    if not color_map:
        return fallback
    name_str = str(name)
    if name_str in color_map:
        return str(color_map[name_str])
    name_lower = name_str.lower()
    for key, color in color_map.items():
        if str(key).lower() in name_lower:
            return str(color)
    return fallback


def _safe_label_for_filename(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label))


def _compute_kde_1d(
    values: np.ndarray,
    *,
    grid: np.ndarray,
    bandwidth_scale: float = 1.0,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    grid = np.asarray(grid, dtype=float).reshape(-1)
    if values.size == 0 or grid.size == 0:
        return np.zeros_like(grid, dtype=float)
    if values.size == 1:
        spread = max(abs(values[0]) * 0.1, 1e-3)
    else:
        sample_std = float(np.std(values, ddof=1))
        iqr = float(np.subtract(*np.percentile(values, [75, 25])))
        sigma = min(sample_std, iqr / 1.349) if iqr > 0 else sample_std
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = max(sample_std, 1e-3)
        spread = 0.9 * sigma * (values.size ** (-1.0 / 5.0))
    bandwidth = max(float(bandwidth_scale) * spread, 1e-3)
    diffs = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * diffs * diffs).sum(axis=1)
    density /= values.size * bandwidth * np.sqrt(2.0 * np.pi)
    return density


def _coerce_color_to_rgb01(color: str) -> tuple[float, float, float]:
    return tuple(float(channel) for channel in to_rgb(color))


def _rgb01_to_hex(rgb: tuple[float, float, float]) -> str:
    clipped = [max(0, min(255, int(round(channel * 255.0)))) for channel in rgb]
    return "#{:02X}{:02X}{:02X}".format(*clipped)


def _srgb_to_linear(rgb: tuple[float, float, float]) -> np.ndarray:
    rgb_arr = np.asarray(rgb, dtype=float)
    return np.where(rgb_arr <= 0.04045, rgb_arr / 12.92, ((rgb_arr + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb_linear: np.ndarray) -> tuple[float, float, float]:
    rgb_linear = np.asarray(rgb_linear, dtype=float)
    srgb = np.where(rgb_linear <= 0.0031308, 12.92 * rgb_linear, 1.055 * np.power(np.maximum(rgb_linear, 0.0), 1.0 / 2.4) - 0.055)
    return tuple(float(channel) for channel in np.clip(srgb, 0.0, 1.0))


def _rgb01_to_xyz_d65(rgb: tuple[float, float, float]) -> np.ndarray:
    linear = _srgb_to_linear(rgb)
    matrix = np.asarray(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=float,
    )
    return matrix @ linear


def _xyz_d65_to_rgb01(xyz: np.ndarray) -> tuple[float, float, float]:
    matrix = np.asarray(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        dtype=float,
    )
    linear = matrix @ np.asarray(xyz, dtype=float)
    return _linear_to_srgb(linear)


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    white = np.asarray([0.95047, 1.00000, 1.08883], dtype=float)
    ratio = np.asarray(xyz, dtype=float) / white
    delta = 6.0 / 29.0
    f = np.where(ratio > delta**3, np.cbrt(ratio), ratio / (3.0 * delta**2) + 4.0 / 29.0)
    return np.asarray(
        [
            116.0 * f[1] - 16.0,
            500.0 * (f[0] - f[1]),
            200.0 * (f[1] - f[2]),
        ],
        dtype=float,
    )


def _lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=float)
    white = np.asarray([0.95047, 1.00000, 1.08883], dtype=float)
    fy = (lab[0] + 16.0) / 116.0
    fx = fy + lab[1] / 500.0
    fz = fy - lab[2] / 200.0
    delta = 6.0 / 29.0
    f = np.asarray([fx, fy, fz], dtype=float)
    ratio = np.where(f > delta, f**3, 3.0 * delta**2 * (f - 4.0 / 29.0))
    return ratio * white


def _blend_colors(color_a: str, color_b: str, *, color_space: str = "lab") -> str:
    rgb_a = _coerce_color_to_rgb01(color_a)
    rgb_b = _coerce_color_to_rgb01(color_b)
    color_space = str(color_space).strip().lower()
    if color_space == "rgb":
        rgb = tuple(0.5 * (a + b) for a, b in zip(rgb_a, rgb_b, strict=True))
        return _rgb01_to_hex(rgb)
    if color_space == "lab":
        lab_a = _xyz_to_lab(_rgb01_to_xyz_d65(rgb_a))
        lab_b = _xyz_to_lab(_rgb01_to_xyz_d65(rgb_b))
        lab = 0.5 * (lab_a + lab_b)
        return _rgb01_to_hex(_xyz_d65_to_rgb01(_lab_to_xyz(lab)))
    if color_space not in {"hsv", "hue"}:
        raise ValueError(f"Unsupported joint_color_blend_space: {color_space!r}")
    hsv_a = colorsys.rgb_to_hsv(*rgb_a)
    hsv_b = colorsys.rgb_to_hsv(*rgb_b)
    hue_a = float(hsv_a[0]) % 1.0
    hue_b = float(hsv_b[0]) % 1.0
    delta = ((hue_b - hue_a + 0.5) % 1.0) - 0.5
    blended_hue = (hue_a + 0.5 * delta) % 1.0
    blended_sat = 0.5 * (float(hsv_a[1]) + float(hsv_b[1]))
    blended_val = 0.5 * (float(hsv_a[2]) + float(hsv_b[2]))
    return _rgb01_to_hex(colorsys.hsv_to_rgb(blended_hue, blended_sat, blended_val))


def compute_posterior_modes(
    posterior_samples_df: pd.DataFrame,
    *,
    parameter_names: list[str],
    knn_neighbors: int = 25,
    min_neighbors: int = 5,
) -> pd.DataFrame:
    if posterior_samples_df.empty:
        raise ValueError("posterior_samples_df is empty")
    if "observation_label" not in posterior_samples_df.columns:
        raise KeyError("posterior_samples_df is missing 'observation_label'")

    rows: list[dict[str, Any]] = []
    metadata_columns = [
        column_name
        for column_name in posterior_samples_df.columns
        if column_name not in set(parameter_names)
    ]
    for observation_label, subset in posterior_samples_df.groupby("observation_label", sort=False):
        subset = subset.reset_index(drop=False).rename(columns={"index": "posterior_global_row_index"})
        sample_matrix = subset[parameter_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite_mask = np.all(np.isfinite(sample_matrix), axis=1)
        if not np.any(finite_mask):
            continue
        finite_subset = subset.loc[finite_mask].reset_index(drop=True)
        finite_samples = sample_matrix[finite_mask]
        means = np.nanmean(finite_samples, axis=0)
        stds = np.nanstd(finite_samples, axis=0)
        stds = np.where(np.isfinite(stds) & (stds > 0.0), stds, 1.0)
        standardized = (finite_samples - means) / stds
        if standardized.shape[0] == 1:
            best_index = 0
            density_score = float("inf")
        else:
            diffs = standardized[:, None, :] - standardized[None, :, :]
            distances = np.sqrt(np.sum(diffs * diffs, axis=2))
            np.fill_diagonal(distances, np.inf)
            max_k = max(1, standardized.shape[0] - 1)
            k = min(max_k, max(1, int(min_neighbors), min(int(knn_neighbors), max_k)))
            nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
            mean_knn_distance = np.nanmean(nearest, axis=1)
            density_scores = 1.0 / np.maximum(mean_knn_distance, 1e-12)
            best_index = int(np.nanargmax(density_scores))
            density_score = float(density_scores[best_index])
        best_row = finite_subset.iloc[best_index]
        mode_row = {
            "observation_label": str(observation_label),
            "posterior_density_score": density_score,
            "posterior_sample_row_index": int(best_index),
            "posterior_global_row_index": int(best_row["posterior_global_row_index"]),
        }
        for column_name in metadata_columns:
            if column_name == "observation_label":
                continue
            mode_row[column_name] = best_row.get(column_name)
        for parameter_name in parameter_names:
            mode_row[parameter_name] = float(best_row[parameter_name])
        rows.append(mode_row)
    columns = [
        "observation_label",
        "posterior_density_score",
        "posterior_sample_row_index",
        "posterior_global_row_index",
        *[c for c in metadata_columns if c != "observation_label"],
        *parameter_names,
    ]
    modes_df = pd.DataFrame(rows)
    if modes_df.empty:
        return pd.DataFrame(columns=columns)
    return modes_df.reindex(columns=columns)


def compute_network_posterior_modes(
    posterior_samples_df: pd.DataFrame,
    *,
    parameter_names: list[str],
    knn_neighbors: int = 25,
    min_neighbors: int = 5,
) -> pd.DataFrame:
    if posterior_samples_df.empty or "network_id" not in posterior_samples_df.columns:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for network_id, subset in posterior_samples_df.groupby("network_id", sort=True):
        modes = compute_posterior_modes(
            subset.copy(),
            parameter_names=parameter_names,
            knn_neighbors=knn_neighbors,
            min_neighbors=min_neighbors,
        )
        if modes.empty:
            continue
        modes = modes.drop(columns=[col for col in ("network_id", "network_seed") if col in modes.columns])
        modes.insert(1, "network_id", int(network_id))
        if "network_seed" in subset.columns:
            seeds = subset["network_seed"].dropna().unique()
            modes.insert(2, "network_seed", int(seeds[0]) if len(seeds) else np.nan)
        rows.append(modes)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fit_pca(sample_matrix: np.ndarray, *, standardize: bool = True) -> dict[str, np.ndarray]:
    sample_matrix = np.asarray(sample_matrix, dtype=float)
    if sample_matrix.ndim != 2:
        raise ValueError("sample_matrix must be 2D")
    if sample_matrix.shape[0] < 2:
        raise ValueError("At least two samples are required for PCA")
    if sample_matrix.shape[1] < 2:
        raise ValueError("At least two parameters are required for PCA")
    means = np.mean(sample_matrix, axis=0)
    if standardize:
        stds = np.std(sample_matrix, axis=0)
        if np.any(~np.isfinite(stds)) or np.any(stds <= 0):
            bad_indices = np.where((~np.isfinite(stds)) | (stds <= 0))[0].tolist()
            raise ValueError(f"Cannot standardize PCA input; zero or non-finite parameter std at columns {bad_indices}")
        pca_input = (sample_matrix - means) / stds
    else:
        stds = np.ones(sample_matrix.shape[1], dtype=float)
        pca_input = sample_matrix - means
    centered = pca_input - np.mean(pca_input, axis=0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt
    scores = centered @ components.T
    explained_variance = (singular_values**2) / max(1, sample_matrix.shape[0] - 1)
    total_variance = float(np.sum(explained_variance))
    if total_variance > 0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = np.zeros_like(explained_variance)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    return {
        "means": means,
        "stds": stds,
        "standardize": np.asarray([bool(standardize)], dtype=bool),
        "components": components,
        "scores": scores,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance_ratio": cumulative_variance_ratio,
    }


def _plot_posterior_pca_wall_projection(
    ax: plt.Axes,
    *,
    scores: np.ndarray,
    mode_score: np.ndarray | None = None,
    style: dict[str, Any],
) -> None:
    xyz = np.asarray(scores[:, :3], dtype=float)
    finite_mask = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite_mask]
    if xyz.size == 0:
        raise ValueError("No finite posterior PCA scores available for 3-wall projection")

    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pads = spans * float(style["projection_padding_fraction"])
    xlim = (float(mins[0] - pads[0]), float(maxs[0] + pads[0]))
    ylim = (float(mins[1] - pads[1]), float(maxs[1] + pads[1]))
    zlim = (float(mins[2] - pads[2]), float(maxs[2] + pads[2]))

    ax.view_init(elev=float(style["projection_elev"]), azim=float(style["projection_azim"]))
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        np.full(xyz.shape[0], zlim[0]),
        s=float(style["projection_marker_size"]),
        color=str(style["projection_color"]),
        alpha=float(style["projection_alpha"]),
        depthshade=False,
    )
    ax.scatter(
        xyz[:, 0],
        np.full(xyz.shape[0], ylim[1]),
        xyz[:, 2],
        s=float(style["projection_marker_size"]),
        color=str(style["projection_color"]),
        alpha=float(style["projection_alpha"]),
        depthshade=False,
    )
    ax.scatter(
        np.full(xyz.shape[0], xlim[0]),
        xyz[:, 1],
        xyz[:, 2],
        s=float(style["projection_marker_size"]),
        color=str(style["projection_color"]),
        alpha=float(style["projection_alpha"]),
        depthshade=False,
    )
    if mode_score is not None and np.all(np.isfinite(mode_score[:3])):
        ax.scatter(
            [float(mode_score[0])],
            [float(mode_score[1])],
            [float(zlim[0])],
            s=float(style["mode_marker_size"]),
            color=str(style["mode_marker_color"]),
            alpha=float(style["mode_marker_alpha"]),
            marker=str(style["mode_marker_symbol"]),
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
        )
        ax.scatter(
            [float(mode_score[0])],
            [float(ylim[1])],
            [float(mode_score[2])],
            s=float(style["mode_marker_size"]),
            color=str(style["mode_marker_color"]),
            alpha=float(style["mode_marker_alpha"]),
            marker=str(style["mode_marker_symbol"]),
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
        )
        ax.scatter(
            [float(xlim[0])],
            [float(mode_score[1])],
            [float(mode_score[2])],
            s=float(style["mode_marker_size"]),
            color=str(style["mode_marker_color"]),
            alpha=float(style["mode_marker_alpha"]),
            marker=str(style["mode_marker_symbol"]),
            edgecolors="white",
            linewidths=0.8,
            depthshade=False,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.xaxis.pane.set_alpha(float(style["projection_plane_alpha"]))
    ax.yaxis.pane.set_alpha(float(style["projection_plane_alpha"]))
    ax.zaxis.pane.set_alpha(float(style["projection_plane_alpha"]))
    ax.grid(alpha=float(style["projection_grid_alpha"]))
    ax.set_xlabel("PC1", fontsize=float(style["projection_label_fontsize"]))
    ax.set_ylabel("PC2", fontsize=float(style["projection_label_fontsize"]))
    ax.set_zlabel("PC3", fontsize=float(style["projection_label_fontsize"]))
    ax.tick_params(axis="both", labelsize=float(style["projection_tick_labelsize"]))
    edge_color = str(style["projection_edge_color"])
    edge_lw = float(style["projection_edge_lw"])
    edge_alpha = float(style["projection_edge_alpha"])
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


def plot_posterior_pca_columns(
    posterior_samples_df: pd.DataFrame,
    *,
    parameter_names: list[str],
    posterior_modes_df: pd.DataFrame | None = None,
    selected_labels: list[str] | None = None,
    parameter_colors: dict[str, str] | None = None,
    n_components_display: int | None = None,
    standardize_parameters: bool = True,
    figure_dpi: int = 150,
    left_column_width: float = 7.2,
    right_column_width: float = 5.8,
    base_height: float = 6.6,
    per_pc_height: float = 1.6,
    scatter_alpha: float = 0.14,
    scatter_size: float = 12.0,
    scatter_color: str = "#4C78A8",
    show_mode_marker: bool = True,
    mode_marker_color: str = "#000000",
    mode_marker_size: float = 90.0,
    mode_marker_alpha: float = 0.98,
    mode_marker_symbol: str = "X",
    loading_bar_edgecolor: str = "white",
    loading_bar_linewidth: float = 0.5,
    loading_tick_labelsize: float = 8.0,
    loading_title_fontsize: float = 10.0,
    loading_text_fontsize: float = 8.5,
    loading_text_loc: str = "upper right",
    share_loading_ylims: bool = True,
    projection_marker_size: float = 18.0,
    projection_alpha: float = 0.18,
    projection_color: str = "#4C78A8",
    projection_plane_alpha: float = 0.06,
    projection_grid_alpha: float = 0.18,
    projection_edge_color: str = "#555555",
    projection_edge_lw: float = 1.4,
    projection_edge_alpha: float = 0.9,
    projection_elev: float = 22.0,
    projection_azim: float = -54.0,
    projection_padding_fraction: float = 0.08,
    projection_label_fontsize: float = 9.0,
    projection_tick_labelsize: float = 8.0,
    output_dir: str | Path | None = None,
    filename_prefix: str = "posterior_pca",
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    if posterior_samples_df.empty:
        raise ValueError("posterior_samples_df is empty")
    if len(parameter_names) < 2:
        raise ValueError("At least two parameters are required for posterior PCA plots")
    available_labels = posterior_samples_df["observation_label"].dropna().astype(str).unique().tolist()
    if selected_labels is None:
        labels = available_labels
    else:
        missing = [label for label in selected_labels if label not in available_labels]
        if missing:
            raise KeyError(f"Requested observation labels not found in posterior samples: {missing}")
        labels = list(selected_labels)

    output_dir = None if output_dir is None else Path(output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    style = {
        "projection_marker_size": projection_marker_size,
        "projection_alpha": projection_alpha,
        "projection_color": projection_color,
        "projection_plane_alpha": projection_plane_alpha,
        "projection_grid_alpha": projection_grid_alpha,
        "projection_edge_color": projection_edge_color,
        "projection_edge_lw": projection_edge_lw,
        "projection_edge_alpha": projection_edge_alpha,
        "projection_elev": projection_elev,
        "projection_azim": projection_azim,
        "projection_padding_fraction": projection_padding_fraction,
        "projection_label_fontsize": projection_label_fontsize,
        "projection_tick_labelsize": projection_tick_labelsize,
        "mode_marker_color": mode_marker_color,
        "mode_marker_size": mode_marker_size,
        "mode_marker_alpha": mode_marker_alpha,
        "mode_marker_symbol": mode_marker_symbol,
    }
    resolved_parameter_colors = {
        str(parameter_name): str((parameter_colors or {}).get(parameter_name, scatter_color))
        for parameter_name in parameter_names
    }

    saved_paths: list[Path] = []
    summaries: dict[str, dict[str, Any]] = {}
    for label in labels:
        subset = posterior_samples_df.loc[posterior_samples_df["observation_label"].astype(str) == str(label)].copy()
        if subset.empty:
            raise ValueError(f"Posterior for observation_label={label!r} is empty")
        sample_matrix = subset[parameter_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite_mask = np.all(np.isfinite(sample_matrix), axis=1)
        if int(np.sum(finite_mask)) < 2:
            raise ValueError(f"Posterior for observation_label={label!r} has fewer than two finite samples")
        subset = subset.loc[finite_mask].reset_index(drop=True)
        sample_matrix = sample_matrix[finite_mask]
        pca = _fit_pca(sample_matrix, standardize=bool(standardize_parameters))
        available_components = int(pca["components"].shape[0])
        displayed_components = available_components if n_components_display is None else min(max(1, int(n_components_display)), available_components)
        mode_row = None
        mode_score = None
        if posterior_modes_df is not None and not posterior_modes_df.empty:
            mode_subset = posterior_modes_df.loc[
                posterior_modes_df["observation_label"].astype(str) == str(label)
            ].copy()
            if len(mode_subset) > 1:
                raise ValueError(f"Expected at most one mode row for observation_label={label!r}, found {len(mode_subset)}")
            if len(mode_subset) == 1:
                mode_row = mode_subset.iloc[0]
                mode_vector = pd.to_numeric(mode_row[parameter_names], errors="coerce").to_numpy(dtype=float)
                if np.all(np.isfinite(mode_vector)):
                    if bool(standardize_parameters):
                        mode_vector_for_pca = (mode_vector - pca["means"]) / pca["stds"]
                    else:
                        mode_vector_for_pca = mode_vector - pca["means"]
                    mode_score = mode_vector_for_pca @ pca["components"].T

        figure_height = max(float(base_height), float(base_height) + float(per_pc_height) * max(0, displayed_components - 3))
        fig = plt.figure(figsize=(float(left_column_width) + float(right_column_width), figure_height), dpi=figure_dpi)
        outer = GridSpec(1, 2, figure=fig, width_ratios=[float(left_column_width), float(right_column_width)], wspace=0.22)
        left_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.25)
        right_gs = GridSpecFromSubplotSpec(displayed_components, 1, subplot_spec=outer[0, 1], hspace=0.35)

        ax_scatter = fig.add_subplot(left_gs[0, 0])
        scores = pca["scores"]
        ax_scatter.scatter(
            scores[:, 0],
            scores[:, 1],
            color=scatter_color,
            alpha=float(scatter_alpha),
            s=float(scatter_size),
            linewidths=0,
        )
        if show_mode_marker and mode_score is not None and np.all(np.isfinite(mode_score[:2])):
            ax_scatter.scatter(
                [float(mode_score[0])],
                [float(mode_score[1])],
                color=mode_marker_color,
                alpha=float(mode_marker_alpha),
                s=float(mode_marker_size),
                marker=str(mode_marker_symbol),
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )
        ax_scatter.set_xlabel("PC1", fontsize=9)
        ax_scatter.set_ylabel("PC2", fontsize=9)
        space_label = "standardized parameter space" if bool(standardize_parameters) else "raw parameter space"
        ax_scatter.set_title(f"Posterior samples in PC space ({space_label})", loc="left", fontsize=10)
        ax_scatter.grid(alpha=0.18)

        if available_components >= 3:
            ax_projection = fig.add_subplot(left_gs[1, 0], projection="3d")
            _plot_posterior_pca_wall_projection(
                ax_projection,
                scores=scores,
                mode_score=mode_score,
                style=style,
            )
            ax_projection.set_title("PC1 / PC2 / PC3 wall projections", loc="left", fontsize=10)
        else:
            ax_projection = fig.add_subplot(left_gs[1, 0])
            ax_projection.scatter(
                scores[:, 0],
                scores[:, 1],
                color=scatter_color,
                alpha=float(scatter_alpha),
                s=float(scatter_size),
                linewidths=0,
            )
            if show_mode_marker and mode_score is not None and np.all(np.isfinite(mode_score[:2])):
                ax_projection.scatter(
                    [float(mode_score[0])],
                    [float(mode_score[1])],
                    color=mode_marker_color,
                    alpha=float(mode_marker_alpha),
                    s=float(mode_marker_size),
                    marker=str(mode_marker_symbol),
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=5,
                )
            ax_projection.set_xlabel("PC1", fontsize=9)
            ax_projection.set_ylabel("PC2", fontsize=9)
            ax_projection.set_title("PC1 / PC2 projection (only 2 PCs available)", loc="left", fontsize=10)
            ax_projection.grid(alpha=0.18)

        loading_axes: list[plt.Axes] = []
        max_abs_loading = float(np.nanmax(np.abs(pca["components"][:displayed_components, :]))) if displayed_components > 0 else 1.0
        shared_ylim = (-1.05 * max_abs_loading, 1.05 * max_abs_loading) if max_abs_loading > 0 else (-1.0, 1.0)
        text_loc = str(loading_text_loc).strip().lower()
        for pc_index in range(displayed_components):
            ax_load = fig.add_subplot(right_gs[pc_index, 0])
            loadings = pca["components"][pc_index, :]
            positions = np.arange(len(parameter_names))
            ax_load.bar(
                positions,
                loadings,
                width=0.82,
                color=[resolved_parameter_colors[name] for name in parameter_names],
                edgecolor=str(loading_bar_edgecolor),
                linewidth=float(loading_bar_linewidth),
            )
            ax_load.axhline(0.0, color="#666666", lw=1.0, ls="--", alpha=0.7)
            ax_load.set_ylabel("Coeff.", fontsize=9)
            ax_load.set_title(f"PC{pc_index + 1} loadings", loc="left", fontsize=float(loading_title_fontsize))
            if share_loading_ylims:
                ax_load.set_ylim(*shared_ylim)
            ax_load.set_xticks(positions)
            ax_load.set_xticklabels([format_parameter_label(name) for name in parameter_names], rotation=45, ha="right", fontsize=float(loading_tick_labelsize))
            ax_load.tick_params(axis="y", labelsize=float(loading_tick_labelsize))
            ax_load.grid(alpha=0.18, axis="y")
            text_lines = [
                f"Var: {100.0 * float(pca['explained_variance_ratio'][pc_index]):.1f}%",
                f"Cum: {100.0 * float(pca['cumulative_variance_ratio'][pc_index]):.1f}%",
            ]
            x_anchor = 0.97 if text_loc == "upper right" else 0.03
            ha = "right" if text_loc == "upper right" else "left"
            ax_load.text(
                x_anchor,
                0.95,
                "\n".join(text_lines),
                transform=ax_load.transAxes,
                ha=ha,
                va="top",
                fontsize=float(loading_text_fontsize),
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            loading_axes.append(ax_load)

        fig.suptitle(f"Posterior PCA\n{label}", fontsize=12)
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.93)
        safe_label = _safe_label_for_filename(str(label))
        if output_dir is not None:
            save_path = output_dir / f"{filename_prefix}__{safe_label}.png"
            fig.savefig(save_path, bbox_inches="tight")
            saved_paths.append(save_path)
        plt.show()
        plt.close(fig)
        summaries[str(label)] = {
            "n_samples": int(sample_matrix.shape[0]),
            "n_parameters": int(sample_matrix.shape[1]),
            "n_components_available": available_components,
            "n_components_displayed": displayed_components,
            "explained_variance_ratio": pca["explained_variance_ratio"].tolist(),
            "cumulative_variance_ratio": pca["cumulative_variance_ratio"].tolist(),
            "standardize_parameters": bool(standardize_parameters),
            "parameter_means": pca["means"].tolist(),
            "parameter_stds": pca["stds"].tolist(),
            "components": pca["components"].tolist(),
            "saved_path": str(saved_paths[-1]) if output_dir is not None else None,
        }
    return saved_paths, summaries


def _compute_empirical_percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values.copy()
    if values.size == 1:
        return np.asarray([50.0], dtype=float)
    ranks = _rankdata_average(values)
    return 100.0 * (ranks - 1.0) / float(values.size - 1)


def _coerce_percentile_target(target: Any, *, n_dims: int) -> np.ndarray:
    arr = np.asarray(target, dtype=float)
    if arr.ndim == 0:
        arr = np.repeat(float(arr), n_dims)
    else:
        arr = arr.reshape(-1)
        if arr.size != n_dims:
            raise ValueError(f"Percentile target must have length {n_dims}, got {arr.size}")
    if np.any(~np.isfinite(arr)):
        raise ValueError("Percentile target must be finite")
    if np.any((arr < 0.0) | (arr > 100.0)):
        raise ValueError("Percentile targets must lie in [0, 100]")
    return arr.astype(float, copy=False)


def _format_slice_percentile_label(target: np.ndarray) -> str:
    target = np.asarray(target, dtype=float).reshape(-1)
    rounded = [str(int(round(value))) for value in target]
    if target.size == 1:
        return f"p{rounded[0]}"
    return "p[" + ",".join(rounded) + "]"


def _resolve_observation_for_label(
    dataset: InferenceDatasetBundle,
    *,
    observation_label: str,
) -> tuple[int, pd.Series]:
    mask = dataset.exp_observations["observation_label"].astype(str) == str(observation_label)
    match_indices = np.flatnonzero(mask.to_numpy())
    if match_indices.size == 0:
        raise KeyError(f"observation_label={observation_label!r} not found in dataset.exp_observations")
    if match_indices.size > 1:
        raise ValueError(
            f"observation_label={observation_label!r} appears multiple times in dataset.exp_observations"
        )
    obs_idx = int(match_indices[0])
    return obs_idx, dataset.exp_observations.iloc[obs_idx]


def _resolve_reference_posterior_samples(
    trained: TrainedPosteriorBundle,
    dataset: InferenceDatasetBundle,
    *,
    observation_label: str,
    parameter_names: list[str],
    posterior_samples_df: pd.DataFrame,
    reference_posterior_samples: pd.DataFrame | np.ndarray | None,
    n_posterior_samples_for_reference: int,
    random_seed: int,
) -> tuple[pd.DataFrame, int, pd.Series]:
    obs_idx, observation_row = _resolve_observation_for_label(dataset, observation_label=observation_label)
    if reference_posterior_samples is None:
        subset = posterior_samples_df.loc[
            posterior_samples_df["observation_label"].astype(str) == str(observation_label)
        ].copy()
        if subset.empty:
            import torch

            torch.manual_seed(int(random_seed))
            x_obs = dataset.x_exp_z[obs_idx : obs_idx + 1, :]
            x_tensor = _safe_torch_tensor(x_obs, device=trained.device)
            posterior_samples = trained.posterior.sample((int(n_posterior_samples_for_reference),), x=x_tensor)
            posterior_unit = np.asarray(posterior_samples.detach().cpu().tolist(), dtype=float)
            posterior_original = dataset.parameter_scaler.inverse_transform(posterior_unit)
            rows = []
            for sample_idx, sample_row in enumerate(posterior_original):
                row = {"observation_label": str(observation_label), "sample_index": int(sample_idx)}
                for param_name, value in zip(parameter_names, sample_row, strict=True):
                    row[param_name] = float(value)
                rows.append(row)
            subset = pd.DataFrame(rows)
    elif isinstance(reference_posterior_samples, pd.DataFrame):
        subset = reference_posterior_samples.copy()
        if "observation_label" in subset.columns:
            subset = subset.loc[subset["observation_label"].astype(str) == str(observation_label)].copy()
    else:
        sample_matrix = np.asarray(reference_posterior_samples, dtype=float)
        if sample_matrix.ndim != 2 or sample_matrix.shape[1] != len(parameter_names):
            raise ValueError(
                "reference_posterior_samples ndarray must have shape "
                f"(n_samples, {len(parameter_names)})"
            )
        rows = []
        for sample_idx, sample_row in enumerate(sample_matrix):
            row = {"observation_label": str(observation_label), "sample_index": int(sample_idx)}
            for param_name, value in zip(parameter_names, sample_row, strict=True):
                row[param_name] = float(value)
            rows.append(row)
        subset = pd.DataFrame(rows)

    if subset.empty:
        raise ValueError(f"No reference posterior samples available for observation_label={observation_label!r}")
    missing = [name for name in parameter_names if name not in subset.columns]
    if missing:
        raise KeyError(f"Reference posterior samples are missing parameter columns: {missing}")
    finite_mask = np.all(
        np.isfinite(subset[parameter_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)),
        axis=1,
    )
    subset = subset.loc[finite_mask].reset_index(drop=True)
    if len(subset) < 2:
        raise ValueError(
            f"Need at least two finite reference posterior samples for observation_label={observation_label!r}"
        )
    if "sample_index" not in subset.columns:
        subset["sample_index"] = np.arange(len(subset), dtype=int)
    return subset, obs_idx, observation_row


def plot_conditional_posterior_slices(
    trained: TrainedPosteriorBundle | EnsemblePosteriorBundle,
    dataset: InferenceDatasetBundle,
    posterior_samples_df: pd.DataFrame,
    *,
    posterior_modes_df: pd.DataFrame | None = None,
    parameter_names: list[str],
    observation_label: str,
    constrained_param_names: list[str],
    slice_percentiles: list[Any],
    include_mode_slice: bool = True,
    reference_posterior_samples: pd.DataFrame | np.ndarray | None = None,
    n_posterior_samples_for_reference: int = 2000,
    n_conditional_samples: int = 600,
    conditional_mcmc_method: str = "slice_np_vectorized",
    conditional_warmup_steps: int = 200,
    conditional_num_chains: int = 8,
    conditional_thin: int = 1,
    conditional_num_workers: int = 1,
    conditional_mp_context: str = "spawn",
    conditional_init_strategy: str = "resample",
    conditional_show_progress: bool = True,
    random_seed: int = 12345,
    mismatch_warn_threshold: float = 8.0,
    output_path: str | Path | None = None,
    figure_dpi: int = 150,
    cell_size: float = 2.5,
    top_row_height: float = 2.5,
    annotation_column_width: float = 2.8,
    diagonal_kind: str = "kde",
    diagonal_kde_points: int = 256,
    diagonal_kde_bandwidth_scale: float = 1.0,
    hist_alpha: float = 0.65,
    hist_bins: int = 30,
    unconstrained_color: str = "#B8B8B8",
    unconstrained_scatter_alpha: float = 0.08,
    unconstrained_scatter_size: float = 10.0,
    conditional_scatter_alpha: float = 0.32,
    conditional_scatter_size: float = 12.0,
    slice_cmap: str = "tab10",
    correlation_fontsize: float = 8.0,
    correlation_loc: str = "upper left",
    correlation_bbox_alpha: float = 0.75,
    constrained_linewidth: float = 2.0,
    constrained_label_fontsize: float = 9.0,
    tick_labelsize: float = 8.0,
    block_title_fontsize: float = 10.0,
    block_text_fontsize: float = 8.5,
    parameter_colors: dict[str, str] | None = None,
    fallback_color: str = "#4C78A8",
    max_free_params: int = 6,
) -> dict[str, Any]:
    import torch
    from sbi.analysis.conditional_density import conditional_potential
    import sbi.inference.posteriors.mcmc_posterior as sbi_mcmc_posterior
    import sbi.utils.torchutils as sbi_torchutils
    from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior

    if posterior_samples_df.empty:
        raise ValueError("posterior_samples_df is empty")
    if len(parameter_names) < 2:
        raise ValueError("At least two parameters are required")
    available_labels = posterior_samples_df["observation_label"].dropna().astype(str).unique().tolist()
    if str(observation_label) not in available_labels:
        raise KeyError(f"observation_label={observation_label!r} not found in posterior_samples_df")
    if not constrained_param_names:
        raise ValueError("At least one constrained parameter must be provided")
    constrained_param_names = [str(name) for name in constrained_param_names]
    missing_constrained = [name for name in constrained_param_names if name not in parameter_names]
    if missing_constrained:
        raise KeyError(f"Constrained parameters not found in parameter_names: {missing_constrained}")
    free_parameter_names = [name for name in parameter_names if name not in set(constrained_param_names)]
    if not free_parameter_names:
        raise ValueError("At least one free parameter must remain after constraining parameters")
    if len(free_parameter_names) > int(max_free_params):
        raise ValueError(
            f"Conditional slice figure would be too large with {len(free_parameter_names)} free parameters "
            f"(max_free_params={int(max_free_params)})"
        )
    if not slice_percentiles and not include_mode_slice:
        raise ValueError("At least one slice must be requested via slice_percentiles or include_mode_slice=True")
    correlation_loc = str(correlation_loc).strip().lower()
    if correlation_loc not in {"upper left", "upper right"}:
        raise ValueError("correlation_loc must be 'upper left' or 'upper right'")
    diagonal_kind = str(diagonal_kind).strip().lower()
    if diagonal_kind not in {"hist", "kde"}:
        raise ValueError(f"Unsupported diagonal_kind: {diagonal_kind}")

    reference_df, obs_idx, observation_row = _resolve_reference_posterior_samples(
        trained,
        dataset,
        observation_label=observation_label,
        parameter_names=parameter_names,
        posterior_samples_df=posterior_samples_df,
        reference_posterior_samples=reference_posterior_samples,
        n_posterior_samples_for_reference=_coerce_positive_int_like(
            n_posterior_samples_for_reference,
            name="n_posterior_samples_for_reference",
        ),
        random_seed=int(random_seed),
    )
    reference_matrix = reference_df[parameter_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    constrained_indices = [parameter_names.index(name) for name in constrained_param_names]
    free_indices = [parameter_names.index(name) for name in free_parameter_names]

    percentile_ranks_by_param = {
        name: _compute_empirical_percentile_ranks(reference_matrix[:, parameter_names.index(name)])
        for name in constrained_param_names
    }
    constrained_percentile_rank_matrix = np.column_stack(
        [percentile_ranks_by_param[name] for name in constrained_param_names]
    )

    slice_records: list[dict[str, Any]] = []
    if include_mode_slice:
        if posterior_modes_df is None or posterior_modes_df.empty:
            warnings.warn(
                f"Requested mode slice for observation_label={observation_label!r}, but posterior_modes_df is unavailable.",
                stacklevel=2,
            )
        else:
            mode_subset = posterior_modes_df.loc[
                posterior_modes_df["observation_label"].astype(str) == str(observation_label)
            ].copy()
            if len(mode_subset) > 1:
                raise ValueError(
                    f"Expected at most one mode row for observation_label={observation_label!r}, found {len(mode_subset)}"
                )
            if len(mode_subset) == 1:
                mode_row = mode_subset.iloc[0]
                mode_values = pd.to_numeric(mode_row[parameter_names], errors="coerce").to_numpy(dtype=float)
                if np.all(np.isfinite(mode_values)):
                    distances = np.linalg.norm(reference_matrix - mode_values[None, :], axis=1)
                    selected_idx = int(np.argmin(distances))
                    selected_row = reference_df.iloc[selected_idx]
                    selected_ranks = constrained_percentile_rank_matrix[selected_idx]
                    slice_records.append(
                        {
                            "slice_label": "mode",
                            "slice_type": "mode",
                            "requested_percentile_target": None,
                            "selected_reference_index": selected_idx,
                            "selected_sample_index": int(selected_row.get("sample_index", selected_idx)),
                            "mismatch_score": float(np.min(distances)),
                            "mismatch_warning": False,
                            "selected_percentile_ranks": selected_ranks,
                            "selected_values": pd.to_numeric(selected_row[parameter_names], errors="coerce").to_numpy(dtype=float),
                        }
                    )

    for percentile_target in slice_percentiles:
        if len(constrained_param_names) == 1:
            target_vec = _coerce_percentile_target(percentile_target, n_dims=1)
            constrained_values = reference_matrix[:, constrained_indices[0]]
            target_value = float(np.percentile(constrained_values, target_vec[0]))
            selected_idx = int(np.argmin(np.abs(constrained_values - target_value)))
            mismatch_score = float(abs(constrained_percentile_rank_matrix[selected_idx, 0] - target_vec[0]))
        else:
            target_vec = _coerce_percentile_target(percentile_target, n_dims=len(constrained_param_names))
            distances = np.linalg.norm(constrained_percentile_rank_matrix - target_vec[None, :], axis=1)
            selected_idx = int(np.argmin(distances))
            mismatch_score = float(distances[selected_idx])
        selected_row = reference_df.iloc[selected_idx]
        mismatch_warning = bool(mismatch_score > float(mismatch_warn_threshold))
        if mismatch_warning:
            warnings.warn(
                f"Slice target {target_vec.tolist()} for observation_label={observation_label!r} is far from the "
                f"nearest actual posterior sample in constrained-percentile space (distance={mismatch_score:.2f}).",
                stacklevel=2,
            )
        slice_records.append(
            {
                "slice_label": _format_slice_percentile_label(target_vec),
                "slice_type": "percentile",
                "requested_percentile_target": target_vec,
                "selected_reference_index": int(selected_idx),
                "selected_sample_index": int(selected_row.get("sample_index", selected_idx)),
                "mismatch_score": mismatch_score,
                "mismatch_warning": mismatch_warning,
                "selected_percentile_ranks": constrained_percentile_rank_matrix[selected_idx],
                "selected_values": pd.to_numeric(selected_row[parameter_names], errors="coerce").to_numpy(dtype=float),
            }
        )

    if not slice_records:
        raise ValueError(f"No slices were constructed for observation_label={observation_label!r}")

    conditional_samples_by_slice: dict[str, pd.DataFrame] = {}
    selected_conditioning_rows: list[dict[str, Any]] = []
    slice_metadata_rows: list[dict[str, Any]] = []
    n_conditional_samples = _coerce_positive_int_like(n_conditional_samples, name="n_conditional_samples")
    network_records = _iter_trained_networks(trained)
    conditional_sample_counts = allocate_ensemble_sample_counts(int(n_conditional_samples), len(network_records))
    conditional_warmup_steps = _coerce_positive_int_like(conditional_warmup_steps, name="conditional_warmup_steps")
    conditional_num_chains = _coerce_positive_int_like(conditional_num_chains, name="conditional_num_chains")
    if int(conditional_thin) < 1:
        raise ValueError("conditional_thin must be >= 1")
    torch.manual_seed(int(random_seed))

    def _tensor2numpy_compat(x: Any) -> np.ndarray:
        return np.asarray(x.detach().cpu().tolist(), dtype=np.float32)

    def _torch_from_numpy_compat(arr: Any):
        np_arr = np.asarray(arr)
        dtype_map = {
            np.dtype("float32"): torch.float32,
            np.dtype("float64"): torch.float64,
            np.dtype("int32"): torch.int32,
            np.dtype("int64"): torch.int64,
            np.dtype("bool"): torch.bool,
        }
        return torch.tensor(
            np_arr.tolist(),
            dtype=dtype_map.get(np_arr.dtype, None),
        )

    original_mcmc_tensor2numpy = sbi_mcmc_posterior.tensor2numpy
    original_utils_tensor2numpy = sbi_torchutils.tensor2numpy
    original_torch_from_numpy = torch.from_numpy
    try:
        sbi_mcmc_posterior.tensor2numpy = _tensor2numpy_compat
        sbi_torchutils.tensor2numpy = _tensor2numpy_compat
        torch.from_numpy = _torch_from_numpy_compat
        try:
            from tqdm.auto import tqdm
        except Exception:
            tqdm = None
        progress_iter = (
            tqdm(
                total=len(slice_records) * len(network_records),
                desc="Conditional posterior slices",
                disable=not bool(conditional_show_progress),
            )
            if tqdm is not None
            else None
        )

        for slice_idx, record in enumerate(slice_records):
            condition_original = np.asarray(record["selected_values"], dtype=float).reshape(1, -1)
            condition_unit = dataset.parameter_scaler.transform(condition_original)
            per_network_conditional_frames: list[pd.DataFrame] = []
            for network_pos, ((network_id, network_seed, network), network_sample_count) in enumerate(
                zip(network_records, conditional_sample_counts, strict=True)
            ):
                posterior_obj = network.posterior
                x_tensor = _safe_torch_tensor(dataset.x_exp_z[obs_idx : obs_idx + 1, :], device=network.device)
                posterior_obj.set_default_x(x_tensor)
                if not hasattr(posterior_obj, "potential_fn") or not hasattr(posterior_obj, "theta_transform") or not hasattr(posterior_obj, "prior"):
                    raise AttributeError(
                        "trained.posterior does not expose potential_fn, theta_transform, and prior attributes required for conditional sampling"
                    )
                posterior_obj.potential_fn.set_x(x_tensor)
                condition_tensor = _safe_torch_tensor(condition_unit, device=network.device)
                conditioned_potential_fn, restricted_tf, restricted_prior = conditional_potential(
                    posterior_obj.potential_fn,
                    posterior_obj.theta_transform,
                    posterior_obj.prior,
                    condition_tensor,
                    free_indices,
                )
                conditioned_potential_fn.set_x(x_tensor)
                conditional_posterior = MCMCPosterior(
                    potential_fn=conditioned_potential_fn,
                    proposal=restricted_prior,
                    theta_transform=restricted_tf,
                    method=str(conditional_mcmc_method),
                    thin=int(conditional_thin),
                    warmup_steps=int(conditional_warmup_steps),
                    num_chains=int(conditional_num_chains),
                    init_strategy=str(conditional_init_strategy),
                    num_workers=int(conditional_num_workers),
                    mp_context=str(conditional_mp_context),
                    device=network.device,
                )
                conditional_posterior.set_default_x(x_tensor)
                torch.manual_seed(int(random_seed) + int(slice_idx) * 1009 + int(network_pos))
                free_samples_unit = conditional_posterior.sample(
                    (int(network_sample_count),),
                    show_progress_bars=False,
                )
                free_samples_unit_np = np.asarray(free_samples_unit.detach().cpu().tolist(), dtype=float)
                full_samples_unit = np.repeat(condition_unit, free_samples_unit_np.shape[0], axis=0)
                full_samples_unit[:, free_indices] = free_samples_unit_np
                full_samples_original = dataset.parameter_scaler.inverse_transform(full_samples_unit)
                conditional_df = pd.DataFrame(full_samples_original, columns=parameter_names)
                conditional_df.insert(0, "slice_label", str(record["slice_label"]))
                conditional_df.insert(1, "observation_label", str(observation_label))
                conditional_df.insert(2, "slice_index", int(slice_idx))
                conditional_df.insert(3, "network_id", int(network_id))
                conditional_df.insert(4, "network_seed", np.nan if network_seed is None else int(network_seed))
                conditional_df.insert(5, "network_sample_index", np.arange(len(conditional_df), dtype=int))
                per_network_conditional_frames.append(conditional_df)
                if progress_iter is not None:
                    progress_iter.update(1)
            conditional_df = pd.concat(per_network_conditional_frames, ignore_index=True)
            conditional_df.insert(6, "ensemble_sample_index", np.arange(len(conditional_df), dtype=int))
            conditional_samples_by_slice[str(record["slice_label"])] = conditional_df

            selected_conditioning_row = {
                "observation_label": str(observation_label),
                "slice_label": str(record["slice_label"]),
                "slice_type": str(record["slice_type"]),
                "selected_reference_index": int(record["selected_reference_index"]),
                "selected_sample_index": int(record["selected_sample_index"]),
            }
            for name, value in zip(parameter_names, condition_original.reshape(-1), strict=True):
                selected_conditioning_row[name] = float(value)
            selected_conditioning_rows.append(selected_conditioning_row)

            metadata_row = {
                "observation_label": str(observation_label),
                "slice_label": str(record["slice_label"]),
                "slice_type": str(record["slice_type"]),
                "n_networks": int(len(network_records)),
                "n_conditional_samples_total": int(len(conditional_df)),
                "n_conditional_samples_by_network": list(map(int, conditional_sample_counts)),
                "requested_percentile_target": (
                    None
                    if record["requested_percentile_target"] is None
                    else np.asarray(record["requested_percentile_target"], dtype=float).tolist()
                ),
                "selected_reference_index": int(record["selected_reference_index"]),
                "selected_sample_index": int(record["selected_sample_index"]),
                "mismatch_score": float(record["mismatch_score"]),
                "mismatch_warning": bool(record["mismatch_warning"]),
                "selected_percentile_ranks": np.asarray(record["selected_percentile_ranks"], dtype=float).tolist(),
            }
            for name, value in zip(constrained_param_names, condition_original.reshape(-1)[constrained_indices], strict=True):
                metadata_row[f"constrained_value__{name}"] = float(value)
            slice_metadata_rows.append(metadata_row)
        if progress_iter is not None:
            progress_iter.close()
    finally:
        sbi_mcmc_posterior.tensor2numpy = original_mcmc_tensor2numpy
        sbi_torchutils.tensor2numpy = original_utils_tensor2numpy
        torch.from_numpy = original_torch_from_numpy

    selected_conditioning_samples_df = pd.DataFrame(selected_conditioning_rows)
    slice_metadata_df = pd.DataFrame(slice_metadata_rows)

    slice_colors = [plt.get_cmap(str(slice_cmap))(idx % plt.get_cmap(str(slice_cmap)).N) for idx in range(len(slice_records))]
    parameter_bound_map: dict[str, tuple[float, float]] = {}
    if isinstance(dataset.parameter_bounds, pd.DataFrame) and {"parameter_name", "lower_bound", "upper_bound"}.issubset(dataset.parameter_bounds.columns):
        for _, row in dataset.parameter_bounds.iterrows():
            parameter_bound_map[str(row["parameter_name"])] = (float(row["lower_bound"]), float(row["upper_bound"]))
    reference_free = reference_df[free_parameter_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite_free_mask = np.all(np.isfinite(reference_free), axis=1)
    reference_free = reference_free[finite_free_mask]
    if reference_free.size == 0:
        raise ValueError(f"No finite free-parameter posterior samples available for observation_label={observation_label!r}")

    top_n = max(1, len(constrained_param_names))
    free_n = len(free_parameter_names)
    pairplot_grid_n = 1 if free_n == 1 else free_n
    figure_height = float(top_row_height) + len(slice_records) * float(cell_size) * pairplot_grid_n
    figure_width = float(cell_size) * (pairplot_grid_n + 1) + float(annotation_column_width)
    fig = plt.figure(figsize=(figure_width, figure_height), dpi=figure_dpi)
    outer = GridSpec(
        1 + len(slice_records),
        1,
        figure=fig,
        height_ratios=[float(top_row_height)] + [float(cell_size) * pairplot_grid_n for _ in slice_records],
        hspace=0.32,
    )

    top_gs = GridSpecFromSubplotSpec(1, top_n, subplot_spec=outer[0, 0], wspace=0.25)
    for idx, constrained_name in enumerate(constrained_param_names):
        ax = fig.add_subplot(top_gs[0, idx])
        values = pd.to_numeric(reference_df[constrained_name], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            if diagonal_kind == "hist":
                ax.hist(
                    finite,
                    bins=int(hist_bins),
                    color=unconstrained_color,
                    alpha=float(hist_alpha),
                    edgecolor="white",
                )
            else:
                if constrained_name in parameter_bound_map:
                    grid_low, grid_high = parameter_bound_map[constrained_name]
                else:
                    grid_low, grid_high = float(np.min(finite)), float(np.max(finite))
                    if np.isclose(grid_low, grid_high):
                        pad = max(abs(grid_low) * 0.05, 1e-3)
                        grid_low -= pad
                        grid_high += pad
                grid = np.linspace(grid_low, grid_high, int(diagonal_kde_points))
                density = _compute_kde_1d(
                    finite,
                    grid=grid,
                    bandwidth_scale=float(diagonal_kde_bandwidth_scale),
                )
                ax.plot(grid, density, color=unconstrained_color, alpha=1.0, linewidth=1.8)
                ax.fill_between(grid, 0.0, density, color=unconstrained_color, alpha=float(hist_alpha))
        for slice_idx, record in enumerate(slice_records):
            constrained_value = float(record["selected_values"][parameter_names.index(constrained_name)])
            ax.axvline(constrained_value, color=slice_colors[slice_idx], linewidth=float(constrained_linewidth), alpha=0.95)
        ax.set_title(format_parameter_label(constrained_name), fontsize=float(block_title_fontsize))
        ax.set_xlabel(format_parameter_label(constrained_name), fontsize=float(constrained_label_fontsize))
        ax.set_ylabel("Count" if diagonal_kind == "hist" else "Density", fontsize=float(constrained_label_fontsize))
        ax.tick_params(axis="both", labelsize=float(tick_labelsize))
        if constrained_name in parameter_bound_map:
            ax.set_xlim(*parameter_bound_map[constrained_name])
        ax.grid(alpha=0.16)

    legend_handles = [
        Line2D([0], [0], color=slice_colors[idx], lw=float(constrained_linewidth), label=str(record["slice_label"]))
        for idx, record in enumerate(slice_records)
    ]
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.99, 0.985), frameon=True, fontsize=float(block_text_fontsize))

    resolved_parameter_colors = {
        str(name): str((parameter_colors or {}).get(name, fallback_color))
        for name in parameter_names
    }
    for slice_idx, record in enumerate(slice_records):
        conditional_df = conditional_samples_by_slice[str(record["slice_label"])]
        slice_color = slice_colors[slice_idx]
        row_spec = GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=outer[1 + slice_idx, 0],
            width_ratios=[pairplot_grid_n * float(cell_size), float(annotation_column_width)],
            wspace=0.18,
        )
        if free_n == 1:
            pair_gs = GridSpecFromSubplotSpec(1, 1, subplot_spec=row_spec[0, 0])
            ax = fig.add_subplot(pair_gs[0, 0])
            unconstrained_values = pd.to_numeric(reference_df[free_parameter_names[0]], errors="coerce").to_numpy(dtype=float)
            conditional_values = pd.to_numeric(conditional_df[free_parameter_names[0]], errors="coerce").to_numpy(dtype=float)
            finite_unconstrained = unconstrained_values[np.isfinite(unconstrained_values)]
            finite_conditional = conditional_values[np.isfinite(conditional_values)]
            if finite_unconstrained.size:
                if diagonal_kind == "hist":
                    ax.hist(
                        finite_unconstrained,
                        bins=int(hist_bins),
                        color=unconstrained_color,
                        alpha=float(hist_alpha),
                        edgecolor="white",
                    )
                else:
                    grid_low, grid_high = (
                        parameter_bound_map.get(free_parameter_names[0], (float(np.min(finite_unconstrained)), float(np.max(finite_unconstrained))))
                    )
                    grid = np.linspace(grid_low, grid_high, int(diagonal_kde_points))
                    density = _compute_kde_1d(
                        finite_unconstrained,
                        grid=grid,
                        bandwidth_scale=float(diagonal_kde_bandwidth_scale),
                    )
                    ax.plot(grid, density, color=unconstrained_color, alpha=1.0, linewidth=1.8)
                    ax.fill_between(grid, 0.0, density, color=unconstrained_color, alpha=float(hist_alpha))
            if finite_conditional.size:
                if diagonal_kind == "hist":
                    ax.hist(
                        finite_conditional,
                        bins=int(hist_bins),
                        color=slice_color,
                        alpha=float(hist_alpha),
                        edgecolor="white",
                    )
                else:
                    grid_low, grid_high = (
                        parameter_bound_map.get(free_parameter_names[0], (float(np.min(finite_conditional)), float(np.max(finite_conditional))))
                    )
                    grid = np.linspace(grid_low, grid_high, int(diagonal_kde_points))
                    density = _compute_kde_1d(
                        finite_conditional,
                        grid=grid,
                        bandwidth_scale=float(diagonal_kde_bandwidth_scale),
                    )
                    ax.plot(grid, density, color=slice_color, alpha=1.0, linewidth=1.8)
                    ax.fill_between(grid, 0.0, density, color=slice_color, alpha=float(hist_alpha) * 0.9)
            ax.set_xlabel(format_parameter_label(free_parameter_names[0]), fontsize=9)
            ax.set_ylabel("Count" if diagonal_kind == "hist" else "Density", fontsize=9)
            if free_parameter_names[0] in parameter_bound_map:
                ax.set_xlim(*parameter_bound_map[free_parameter_names[0]])
            ax.tick_params(axis="both", labelsize=float(tick_labelsize))
            ax.grid(alpha=0.16)
        else:
            pair_gs = GridSpecFromSubplotSpec(free_n, free_n, subplot_spec=row_spec[0, 0], hspace=0.08, wspace=0.08)
            conditional_matrix = conditional_df[free_parameter_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            for i, row_param in enumerate(free_parameter_names):
                y_unconstrained = pd.to_numeric(reference_df[row_param], errors="coerce").to_numpy(dtype=float)
                y_conditional = conditional_matrix[:, i]
                for j, col_param in enumerate(free_parameter_names):
                    ax = fig.add_subplot(pair_gs[i, j])
                    x_unconstrained = pd.to_numeric(reference_df[col_param], errors="coerce").to_numpy(dtype=float)
                    x_conditional = conditional_matrix[:, j]
                    if i == j:
                        finite_unconstrained = x_unconstrained[np.isfinite(x_unconstrained)]
                        finite_conditional = x_conditional[np.isfinite(x_conditional)]
                        if finite_unconstrained.size:
                            if diagonal_kind == "hist":
                                ax.hist(
                                    finite_unconstrained,
                                    bins=int(hist_bins),
                                    color=unconstrained_color,
                                    alpha=float(hist_alpha),
                                    edgecolor="white",
                                )
                            else:
                                grid_low, grid_high = parameter_bound_map.get(col_param, (float(np.min(finite_unconstrained)), float(np.max(finite_unconstrained))))
                                grid = np.linspace(grid_low, grid_high, int(diagonal_kde_points))
                                density = _compute_kde_1d(
                                    finite_unconstrained,
                                    grid=grid,
                                    bandwidth_scale=float(diagonal_kde_bandwidth_scale),
                                )
                                ax.plot(grid, density, color=unconstrained_color, alpha=1.0, linewidth=1.6)
                                ax.fill_between(grid, 0.0, density, color=unconstrained_color, alpha=float(hist_alpha))
                        if finite_conditional.size:
                            if diagonal_kind == "hist":
                                ax.hist(
                                    finite_conditional,
                                    bins=int(hist_bins),
                                    color=slice_color,
                                    alpha=float(hist_alpha),
                                    edgecolor="white",
                                )
                            else:
                                grid_low, grid_high = parameter_bound_map.get(col_param, (float(np.min(finite_conditional)), float(np.max(finite_conditional))))
                                grid = np.linspace(grid_low, grid_high, int(diagonal_kde_points))
                                density = _compute_kde_1d(
                                    finite_conditional,
                                    grid=grid,
                                    bandwidth_scale=float(diagonal_kde_bandwidth_scale),
                                )
                                ax.plot(grid, density, color=slice_color, alpha=1.0, linewidth=1.8)
                                ax.fill_between(grid, 0.0, density, color=slice_color, alpha=float(hist_alpha) * 0.9)
                        ylabel = "Count" if diagonal_kind == "hist" else "Density"
                    else:
                        mask_un = np.isfinite(x_unconstrained) & np.isfinite(y_unconstrained)
                        if np.any(mask_un):
                            ax.scatter(
                                x_unconstrained[mask_un],
                                y_unconstrained[mask_un],
                                color=unconstrained_color,
                                alpha=float(unconstrained_scatter_alpha),
                                s=float(unconstrained_scatter_size),
                                linewidths=0,
                            )
                        mask_cond = np.isfinite(x_conditional) & np.isfinite(y_conditional)
                        if np.any(mask_cond):
                            ax.scatter(
                                x_conditional[mask_cond],
                                y_conditional[mask_cond],
                                color=slice_color,
                                alpha=float(conditional_scatter_alpha),
                                s=float(conditional_scatter_size),
                                linewidths=0,
                            )
                            corr_lines: list[str] = []
                            pearson_r = _pearson_corr_safe(x_conditional[mask_cond], y_conditional[mask_cond])
                            spearman_rho = _spearman_corr_safe(x_conditional[mask_cond], y_conditional[mask_cond])
                            if np.isfinite(pearson_r):
                                corr_lines.append(f"r = {pearson_r:.2f}")
                            if np.isfinite(spearman_rho):
                                corr_lines.append(f"\u03c1 = {spearman_rho:.2f}")
                            if corr_lines:
                                x_anchor = 0.03 if correlation_loc == "upper left" else 0.97
                                ha = "left" if correlation_loc == "upper left" else "right"
                                ax.text(
                                    x_anchor,
                                    0.97,
                                    "\n".join(corr_lines),
                                    transform=ax.transAxes,
                                    ha=ha,
                                    va="top",
                                    fontsize=float(correlation_fontsize),
                                    bbox={
                                        "boxstyle": "round,pad=0.2",
                                        "facecolor": "white",
                                        "alpha": float(correlation_bbox_alpha),
                                        "edgecolor": "none",
                                    },
                                    color=slice_color,
                                )
                        ylabel = format_parameter_label(row_param)
                    if j == 0:
                        ax.set_ylabel(ylabel, fontsize=9)
                    else:
                        ax.set_yticklabels([])
                    if i == free_n - 1:
                        ax.set_xlabel(format_parameter_label(col_param), fontsize=9)
                    else:
                        ax.set_xticklabels([])
                    if col_param in parameter_bound_map:
                        ax.set_xlim(*parameter_bound_map[col_param])
                    if i != j and row_param in parameter_bound_map:
                        ax.set_ylim(*parameter_bound_map[row_param])
                    ax.tick_params(axis="both", labelsize=float(tick_labelsize))
                    ax.grid(alpha=0.16)

        ax_info = fig.add_subplot(row_spec[0, 1])
        ax_info.axis("off")
        requested_target = record["requested_percentile_target"]
        target_text = "mode reference" if requested_target is None else np.asarray(requested_target, dtype=float).round(1).tolist()
        constrained_value_lines = [
            f"{format_parameter_label(name).replace(chr(10), ' ')} = {float(record['selected_values'][parameter_names.index(name)]):.3g}"
            for name in constrained_param_names
        ]
        info_lines = [
            f"Slice: {record['slice_label']}",
            f"Target: {target_text}",
            f"Sample idx: {record['selected_sample_index']}",
            f"Mismatch: {float(record['mismatch_score']):.2f}",
        ] + constrained_value_lines
        if bool(record["mismatch_warning"]):
            info_lines.append("Warning: target is jointly unnatural")
        ax_info.text(
            0.0,
            1.0,
            "\n".join(info_lines),
            ha="left",
            va="top",
            fontsize=float(block_text_fontsize),
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        ax_info.set_title(f"Conditional slice: {record['slice_label']}", loc="left", fontsize=float(block_title_fontsize))

    fig.suptitle(f"Conditional posterior slices\n{observation_label}", fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.05)

    saved_path = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        saved_path = output_path
    plt.show()

    return {
        "figure": fig,
        "saved_path": saved_path,
        "selected_conditioning_samples_df": selected_conditioning_samples_df,
        "conditional_samples_by_slice": conditional_samples_by_slice,
        "free_parameter_names": free_parameter_names,
        "constrained_parameter_names": constrained_param_names,
        "slice_metadata_df": slice_metadata_df,
        "reference_posterior_samples_df": reference_df,
        "observation_row": observation_row,
    }


def plot_posterior_pairplots(
    posterior_samples_df: pd.DataFrame,
    *,
    parameter_names: list[str],
    parameter_bounds: pd.DataFrame | dict[str, Any] | None = None,
    posterior_modes_df: pd.DataFrame | None = None,
    network_posterior_modes_df: pd.DataFrame | None = None,
    selected_labels: list[str] | None = None,
    figure_dpi: int = 140,
    cell_size: float = 2.8,
    scatter_alpha: float = 0.08,
    scatter_size: float = 8.0,
    diagonal_kind: str = "hist",
    diagonal_kde_points: int = 256,
    diagonal_kde_bandwidth_scale: float = 1.0,
    hist_alpha: float = 0.75,
    hist_bins: int = 40,
    color: str = "#4C78A8",
    parameter_colors: dict[str, str] | None = None,
    joint_color_blend_space: str = "lab",
    label_all_axes: bool = True,
    show_pair_correlations: bool = True,
    pair_correlation_fontsize: float = 8.0,
    pair_correlation_loc: str = "upper left",
    pair_correlation_bbox_alpha: float = 0.78,
    show_mode_marker: bool = True,
    mode_marker_color: str = "#000000",
    mode_marker_size: float = 90.0,
    mode_marker_alpha: float = 0.95,
    mode_marker_symbol: str = "X",
    show_mode_diagonal_line: bool = True,
    mode_diagonal_linewidth: float = 2.0,
    mode_diagonal_linestyle: str = "-",
    show_network_overlays: bool = True,
    network_density_color: str | None = None,
    network_density_alpha: float = 0.45,
    network_density_linewidth: float = 1.0,
    network_mode_color: str = "#666666",
    network_mode_alpha: float = 0.6,
    network_mode_size: float = 45.0,
    network_mode_diagonal_linewidth: float = 1.1,
    network_mode_diagonal_linestyle: str = ":",
    normalize_diagonal_density_ylim: bool = False,
    output_dir: str | Path | None = None,
) -> list[Path]:
    if posterior_samples_df.empty:
        raise ValueError("posterior_samples_df is empty")
    diagonal_kind = str(diagonal_kind).strip().lower()
    if diagonal_kind not in {"hist", "kde"}:
        raise ValueError(f"Unsupported diagonal_kind: {diagonal_kind}")
    available_labels = posterior_samples_df["observation_label"].dropna().astype(str).unique().tolist()
    if selected_labels is None:
        labels = available_labels
    else:
        missing = [label for label in selected_labels if label not in available_labels]
        if missing:
            raise KeyError(f"Requested observation labels not found in posterior samples: {missing}")
        labels = list(selected_labels)
    pair_correlation_loc = str(pair_correlation_loc).strip().lower()
    if pair_correlation_loc not in {"upper left", "upper right"}:
        raise ValueError("pair_correlation_loc must be 'upper left' or 'upper right'")

    saved_paths: list[Path] = []
    output_dir = None if output_dir is None else Path(output_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    parameter_bound_map: dict[str, tuple[float, float]] = {}
    if parameter_bounds is not None:
        if isinstance(parameter_bounds, pd.DataFrame):
            for _, row in parameter_bounds.iterrows():
                if {"parameter_name", "lower_bound", "upper_bound"}.issubset(parameter_bounds.columns):
                    parameter_bound_map[str(row["parameter_name"])] = (
                        float(row["lower_bound"]),
                        float(row["upper_bound"]),
                    )
        elif isinstance(parameter_bounds, dict):
            for key, value in parameter_bounds.items():
                low, high = _coerce_scalar_bound_pair(value, str(key))
                parameter_bound_map[str(key)] = (low, high)
        else:
            raise TypeError(
                f"parameter_bounds must be a DataFrame or dict, got {type(parameter_bounds).__name__}"
            )

    n_params = len(parameter_names)
    resolved_parameter_colors = {
        str(parameter_name): str((parameter_colors or {}).get(parameter_name, color))
        for parameter_name in parameter_names
    }
    has_multiple_networks = (
        "network_id" in posterior_samples_df.columns
        and posterior_samples_df["network_id"].dropna().nunique() > 1
    )
    if network_posterior_modes_df is None and has_multiple_networks:
        network_posterior_modes_df = compute_network_posterior_modes(
            posterior_samples_df,
            parameter_names=parameter_names,
        )
    for label in labels:
        subset = posterior_samples_df.loc[posterior_samples_df["observation_label"].astype(str) == str(label)].copy()
        diagonal_ymax_by_param: dict[str, float] = {}
        if bool(normalize_diagonal_density_ylim) and str(diagonal_kind) == "kde":
            all_diagonal_ymax: list[float] = []
            for parameter_name in parameter_names:
                values = pd.to_numeric(subset[parameter_name], errors="coerce").to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    continue
                if parameter_name in parameter_bound_map:
                    grid_low, grid_high = parameter_bound_map[parameter_name]
                else:
                    grid_low, grid_high = float(np.min(finite)), float(np.max(finite))
                    if np.isclose(grid_low, grid_high):
                        pad = max(abs(grid_low) * 0.05, 1e-3)
                        grid_low -= pad
                        grid_high += pad
                grid = np.linspace(grid_low, grid_high, int(diagonal_kde_points))
                y_max_values = [
                    float(
                        np.nanmax(
                            _compute_kde_1d(
                                finite,
                                grid=grid,
                                bandwidth_scale=float(diagonal_kde_bandwidth_scale),
                            )
                        )
                    )
                ]
                if has_multiple_networks:
                    for _network_id, network_subset in subset.groupby("network_id", sort=True):
                        network_values = pd.to_numeric(network_subset[parameter_name], errors="coerce").to_numpy(dtype=float)
                        network_finite = network_values[np.isfinite(network_values)]
                        if network_finite.size < 2:
                            continue
                        y_max_values.append(
                            float(
                                np.nanmax(
                                    _compute_kde_1d(
                                        network_finite,
                                        grid=grid,
                                        bandwidth_scale=float(diagonal_kde_bandwidth_scale),
                                    )
                                )
                            )
                        )
                finite_ymax = [value for value in y_max_values if np.isfinite(value) and value > 0.0]
                if finite_ymax:
                    all_diagonal_ymax.append(max(finite_ymax))
            if all_diagonal_ymax:
                shared_diagonal_ymax = 1.08 * max(all_diagonal_ymax)
                diagonal_ymax_by_param = {
                    parameter_name: shared_diagonal_ymax
                    for parameter_name in parameter_names
                }
        mode_row = None
        if posterior_modes_df is not None and not posterior_modes_df.empty:
            mode_subset = posterior_modes_df.loc[
                posterior_modes_df["observation_label"].astype(str) == str(label)
            ].copy()
            if len(mode_subset) > 1:
                raise ValueError(f"Expected at most one mode row for observation_label={label!r}, found {len(mode_subset)}")
            if len(mode_subset) == 1:
                mode_row = mode_subset.iloc[0]
        network_mode_subset = pd.DataFrame()
        if has_multiple_networks and network_posterior_modes_df is not None and not network_posterior_modes_df.empty:
            network_mode_subset = network_posterior_modes_df.loc[
                network_posterior_modes_df["observation_label"].astype(str) == str(label)
            ].copy()
        fig, axes = plt.subplots(
            n_params,
            n_params,
            figsize=(cell_size * n_params, cell_size * n_params),
            dpi=figure_dpi,
            squeeze=False,
        )
        for i, row_param in enumerate(parameter_names):
            y = pd.to_numeric(subset[row_param], errors="coerce").to_numpy(dtype=float)
            for j, col_param in enumerate(parameter_names):
                ax = axes[i, j]
                x = pd.to_numeric(subset[col_param], errors="coerce").to_numpy(dtype=float)
                diagonal_color = resolved_parameter_colors[col_param]
                if i == j:
                    finite = x[np.isfinite(x)]
                    if finite.size:
                        if diagonal_kind == "hist":
                            ax.hist(
                                finite,
                                bins=int(hist_bins),
                                color=diagonal_color,
                                alpha=hist_alpha,
                                edgecolor="white",
                            )
                        else:
                            if col_param in parameter_bound_map:
                                grid_low, grid_high = parameter_bound_map[col_param]
                            else:
                                grid_low, grid_high = float(np.min(finite)), float(np.max(finite))
                                if np.isclose(grid_low, grid_high):
                                    pad = max(abs(grid_low) * 0.05, 1e-3)
                                    grid_low -= pad
                                    grid_high += pad
                            grid = np.linspace(grid_low, grid_high, int(diagonal_kde_points))
                            density = _compute_kde_1d(
                                finite,
                                grid=grid,
                                bandwidth_scale=float(diagonal_kde_bandwidth_scale),
                            )
                            ax.plot(grid, density, color=diagonal_color, alpha=1.0, linewidth=1.8)
                            ax.fill_between(grid, 0.0, density, color=diagonal_color, alpha=float(hist_alpha))
                            if show_network_overlays and has_multiple_networks:
                                for _network_id, network_subset in subset.groupby("network_id", sort=True):
                                    network_values = pd.to_numeric(network_subset[col_param], errors="coerce").to_numpy(dtype=float)
                                    network_finite = network_values[np.isfinite(network_values)]
                                    if network_finite.size < 2:
                                        continue
                                    network_density = _compute_kde_1d(
                                        network_finite,
                                        grid=grid,
                                        bandwidth_scale=float(diagonal_kde_bandwidth_scale),
                                    )
                                    ax.plot(
                                        grid,
                                        network_density,
                                        color=str(network_density_color or diagonal_color),
                                        alpha=float(network_density_alpha),
                                        linewidth=float(network_density_linewidth),
                                        zorder=4,
                                    )
                        if show_network_overlays and not network_mode_subset.empty and show_mode_diagonal_line:
                            for _, network_mode_row in network_mode_subset.iterrows():
                                if pd.notna(network_mode_row.get(col_param)):
                                    ax.axvline(
                                        float(network_mode_row[col_param]),
                                        color=str(network_mode_color),
                                        alpha=float(network_mode_alpha),
                                        linewidth=float(network_mode_diagonal_linewidth),
                                        linestyle=str(network_mode_diagonal_linestyle),
                                        zorder=5,
                                    )
                        if (
                            show_mode_marker
                            and mode_row is not None
                            and pd.notna(mode_row.get(col_param))
                            and show_mode_diagonal_line
                        ):
                            mode_value = float(mode_row[col_param])
                            ax.axvline(
                                mode_value,
                                color=mode_marker_color,
                                alpha=float(mode_marker_alpha),
                                linewidth=float(mode_diagonal_linewidth),
                                linestyle=str(mode_diagonal_linestyle),
                                zorder=5,
                            )
                else:
                    scatter_color = _blend_colors(
                        resolved_parameter_colors[col_param],
                        resolved_parameter_colors[row_param],
                        color_space=joint_color_blend_space,
                    )
                    mask = np.isfinite(x) & np.isfinite(y)
                    if np.any(mask):
                        ax.scatter(
                            x[mask],
                            y[mask],
                            color=scatter_color,
                            alpha=float(scatter_alpha),
                            s=float(scatter_size),
                            linewidths=0,
                        )
                        if show_pair_correlations:
                            pearson_r = _pearson_corr_safe(x[mask], y[mask])
                            spearman_rho = _spearman_corr_safe(x[mask], y[mask])
                            corr_lines: list[str] = []
                            if np.isfinite(pearson_r):
                                corr_lines.append(f"r = {pearson_r:.2f}")
                            if np.isfinite(spearman_rho):
                                corr_lines.append(f"\u03c1 = {spearman_rho:.2f}")
                            if corr_lines:
                                x_anchor = 0.03 if pair_correlation_loc == "upper left" else 0.97
                                ha = "left" if pair_correlation_loc == "upper left" else "right"
                                ax.text(
                                    x_anchor,
                                    0.97,
                                    "\n".join(corr_lines),
                                    transform=ax.transAxes,
                                    ha=ha,
                                    va="top",
                                    fontsize=float(pair_correlation_fontsize),
                                    bbox={
                                        "boxstyle": "round,pad=0.2",
                                        "facecolor": "white",
                                        "alpha": float(pair_correlation_bbox_alpha),
                                        "edgecolor": "none",
                                    },
                                    zorder=4,
                                )
                    if (
                        show_network_overlays
                        and not network_mode_subset.empty
                    ):
                        for _, network_mode_row in network_mode_subset.iterrows():
                            if pd.notna(network_mode_row.get(col_param)) and pd.notna(network_mode_row.get(row_param)):
                                ax.scatter(
                                    [float(network_mode_row[col_param])],
                                    [float(network_mode_row[row_param])],
                                    color=str(network_mode_color),
                                    alpha=float(network_mode_alpha),
                                    s=float(network_mode_size),
                                    marker=str(mode_marker_symbol),
                                    linewidths=0.5,
                                    edgecolors="white",
                                    zorder=5,
                                )
                    if (
                        show_mode_marker
                        and mode_row is not None
                        and pd.notna(mode_row.get(col_param))
                        and pd.notna(mode_row.get(row_param))
                    ):
                        ax.scatter(
                            [float(mode_row[col_param])],
                            [float(mode_row[row_param])],
                            color=mode_marker_color,
                            alpha=float(mode_marker_alpha),
                            s=float(mode_marker_size),
                            marker=str(mode_marker_symbol),
                            linewidths=0.6,
                            edgecolors="white",
                            zorder=6,
                        )
                xlabel = format_parameter_label(col_param)
                if i == j:
                    ylabel = "Count" if diagonal_kind == "hist" else "Density"
                else:
                    ylabel = format_parameter_label(row_param)
                if label_all_axes or i == n_params - 1:
                    ax.set_xlabel(xlabel, fontsize=9)
                    ax.xaxis.label.set_color(resolved_parameter_colors[col_param])
                else:
                    ax.set_xticklabels([])
                if label_all_axes or j == 0:
                    ax.set_ylabel(ylabel, fontsize=9)
                    if i != j:
                        ax.yaxis.label.set_color(resolved_parameter_colors[row_param])
                else:
                    ax.set_yticklabels([])
                if col_param in parameter_bound_map:
                    ax.set_xlim(*parameter_bound_map[col_param])
                if i == j and col_param in diagonal_ymax_by_param:
                    ax.set_ylim(0.0, diagonal_ymax_by_param[col_param])
                if i != j and row_param in parameter_bound_map:
                    ax.set_ylim(*parameter_bound_map[row_param])
                ax.grid(alpha=0.16)
        fig.suptitle(f"Posterior pairplot\n{label}", fontsize=12)
        fig.tight_layout()
        if output_dir is not None:
            safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label))
            save_path = output_dir / f"posterior_pairplot__{safe_label}.png"
            fig.savefig(save_path, bbox_inches="tight")
            saved_paths.append(save_path)
        plt.show()
        plt.close(fig)
    return saved_paths


def plot_posterior_condition_diagnostics(
    posterior_samples_df: pd.DataFrame,
    *,
    parameter_names: list[str],
    selected_labels: list[str] | None = None,
    condition_colors: dict[str, str] | list[str] | tuple[str, ...] | None = None,
    condition_cmap: str = "tab10",
    parameter_colors: dict[str, str] | None = None,
    parameter_bounds: pd.DataFrame | dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    figure_dpi: int = 150,
    cell_width: float = 3.0,
    density_height: float = 2.8,
    heatmap_height: float = 2.8,
    kde_points: int = 256,
    kde_bandwidth_scale: float = 1.0,
    density_linewidth: float = 1.8,
    density_fill_alpha: float = 0.16,
    density_line_alpha: float = 0.95,
    ridge_density_scale: float = 0.78,
    p_gt_zero_text_fontsize: float = 7.5,
    p_gt_zero_text_x: float = 0.98,
    show_zero_reference_line: bool = True,
    zero_reference_line_color: str = "#333333",
    zero_reference_linewidth: float = 1.0,
    zero_reference_linestyle: str = ":",
    zero_reference_alpha: float = 0.8,
    p_delta_cmap: str = "coolwarm",
    p_delta_text_fontsize: float = 8.0,
    tick_labelsize: float = 8.0,
    parameter_label_fontsize: float = 9.0,
    title_fontsize: float = 12.0,
    legend_fontsize: float = 8.0,
    show_density_legend: bool = True,
    title_y: float = 0.99,
    legend_y: float = 0.93,
    figure_left: float = 0.06,
    figure_right: float = 0.90,
    figure_bottom: float = 0.16,
    figure_top: float = 0.80,
    subplot_wspace: float = 0.38,
    subplot_hspace: float = 0.78,
    colorbar_shrink: float = 0.78,
    colorbar_pad: float = 0.035,
    heatmap_colorbar_label: str = "P(row condition > column condition)",
) -> dict[str, Any]:
    if posterior_samples_df.empty:
        raise ValueError("posterior_samples_df is empty")
    if not parameter_names:
        raise ValueError("parameter_names is empty")
    if "observation_label" not in posterior_samples_df.columns:
        raise KeyError("posterior_samples_df must contain an 'observation_label' column")

    available_labels = posterior_samples_df["observation_label"].dropna().astype(str).unique().tolist()
    if selected_labels is None:
        labels = available_labels
    else:
        missing = [str(label) for label in selected_labels if str(label) not in available_labels]
        if missing:
            raise KeyError(f"Requested observation labels not found in posterior samples: {missing}")
        labels = [str(label) for label in selected_labels]
    labels = list(dict.fromkeys(labels))
    if len(labels) < 2:
        raise ValueError("At least two observation labels are required for posterior condition comparison")

    parameter_bound_map: dict[str, tuple[float, float]] = {}
    if parameter_bounds is not None:
        if isinstance(parameter_bounds, pd.DataFrame):
            for _, row in parameter_bounds.iterrows():
                if {"parameter_name", "lower_bound", "upper_bound"}.issubset(parameter_bounds.columns):
                    parameter_bound_map[str(row["parameter_name"])] = (
                        float(row["lower_bound"]),
                        float(row["upper_bound"]),
                    )
        elif isinstance(parameter_bounds, dict):
            for key, value in parameter_bounds.items():
                parameter_bound_map[str(key)] = _coerce_scalar_bound_pair(value, str(key))
        else:
            raise TypeError(
                f"parameter_bounds must be a DataFrame or dict, got {type(parameter_bounds).__name__}"
            )

    condition_color_map = _resolve_label_colors(
        labels,
        colors=condition_colors,
        cmap=str(condition_cmap),
    )
    resolved_parameter_colors = {
        str(parameter_name): _resolve_name_color(str(parameter_name), parameter_colors, fallback="black")
        for parameter_name in parameter_names
    }

    samples_by_label_param: dict[tuple[str, str], np.ndarray] = {}
    for label in labels:
        subset = posterior_samples_df.loc[posterior_samples_df["observation_label"].astype(str) == str(label)]
        for parameter_name in parameter_names:
            values = pd.to_numeric(subset[parameter_name], errors="coerce").to_numpy(dtype=float)
            samples_by_label_param[(label, parameter_name)] = values[np.isfinite(values)]

    density_records: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    density_x_limits_by_param: dict[str, tuple[float, float]] = {}
    shared_density_ymax = 0.0
    for parameter_name in parameter_names:
        finite_groups = [samples_by_label_param[(label, parameter_name)] for label in labels]
        finite_all = np.concatenate([values for values in finite_groups if values.size]) if any(values.size for values in finite_groups) else np.array([], dtype=float)
        if finite_all.size == 0:
            continue
        if parameter_name in parameter_bound_map:
            grid_low, grid_high = parameter_bound_map[parameter_name]
        else:
            grid_low, grid_high = float(np.min(finite_all)), float(np.max(finite_all))
            if np.isclose(grid_low, grid_high):
                pad = max(abs(grid_low) * 0.05, 1e-3)
                grid_low -= pad
                grid_high += pad
        density_x_limits_by_param[str(parameter_name)] = (float(grid_low), float(grid_high))
        grid = np.linspace(grid_low, grid_high, int(kde_points))
        for label in labels:
            values = samples_by_label_param[(label, parameter_name)]
            if values.size == 0:
                density = np.full_like(grid, np.nan, dtype=float)
            else:
                density = _compute_kde_1d(
                    values,
                    grid=grid,
                    bandwidth_scale=float(kde_bandwidth_scale),
                )
                finite_density = density[np.isfinite(density)]
                if finite_density.size:
                    shared_density_ymax = max(shared_density_ymax, float(np.max(finite_density)))
            density_records[(label, parameter_name)] = (grid, density)

    n_params = len(parameter_names)
    n_labels = len(labels)
    fig, axes = plt.subplots(
        2,
        n_params,
        figsize=(float(cell_width) * n_params, float(density_height) + float(heatmap_height)),
        dpi=figure_dpi,
        squeeze=False,
        gridspec_kw={"height_ratios": [float(density_height), float(heatmap_height)]},
    )
    p_delta_rows: list[dict[str, Any]] = []
    p_gt_zero_rows: list[dict[str, Any]] = []
    heatmap_images = []
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    ridge_offsets = np.arange(n_labels, dtype=float)
    ridge_scale = float(ridge_density_scale)
    ridge_density_denominator = shared_density_ymax if shared_density_ymax > 0 else 1.0
    ridge_ylim = (-0.35, max(float(n_labels - 1), 0.0) + ridge_scale + 0.35)

    for param_idx, parameter_name in enumerate(parameter_names):
        param_color = resolved_parameter_colors[str(parameter_name)]
        ax_density = axes[0, param_idx]
        x_limits = density_x_limits_by_param.get(str(parameter_name), parameter_bound_map.get(parameter_name))
        if x_limits is not None:
            ax_density.set_xlim(*x_limits)
        if (
            show_zero_reference_line
            and x_limits is not None
            and float(x_limits[0]) <= 0.0 <= float(x_limits[1])
        ):
            ax_density.axvline(
                0.0,
                color=str(zero_reference_line_color),
                linestyle=str(zero_reference_linestyle),
                linewidth=float(zero_reference_linewidth),
                alpha=float(zero_reference_alpha),
                zorder=0,
            )
        for label_idx, label in enumerate(labels):
            grid, density = density_records.get((label, parameter_name), (np.array([], dtype=float), np.array([], dtype=float)))
            condition_color = condition_color_map[label]
            ridge_offset = ridge_offsets[label_idx]
            values = samples_by_label_param[(label, parameter_name)]
            if values.size:
                p_gt_zero = float(np.mean(values > 0.0))
            else:
                p_gt_zero = float("nan")
            p_gt_zero_rows.append(
                {
                    "parameter_name": parameter_name,
                    "observation_label": label,
                    "p_gt_zero": p_gt_zero,
                    "n_samples": int(values.size),
                }
            )
            if grid.size:
                scaled_density = np.asarray(density, dtype=float) / ridge_density_denominator * ridge_scale
                scaled_density[~np.isfinite(scaled_density)] = np.nan
                ax_density.plot(
                    grid,
                    ridge_offset + scaled_density,
                    color=condition_color,
                    alpha=float(density_line_alpha),
                    linewidth=float(density_linewidth),
                    label=str(label),
                )
                ax_density.fill_between(
                    grid,
                    ridge_offset,
                    ridge_offset + scaled_density,
                    color=condition_color,
                    alpha=float(density_fill_alpha),
                    linewidth=0,
                )
            p_text = f"P(theta > 0)={p_gt_zero:.2f}" if np.isfinite(p_gt_zero) else "P(theta > 0)=NA"
            ax_density.text(
                float(p_gt_zero_text_x),
                ridge_offset + ridge_scale * 0.08,
                p_text,
                transform=ax_density.get_yaxis_transform(),
                ha="right",
                va="bottom",
                color=condition_color,
                fontsize=float(p_gt_zero_text_fontsize),
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 1.2},
            )
        ax_density.set_ylim(*ridge_ylim)
        ax_density.set_yticks(ridge_offsets)
        ax_density.set_yticklabels(labels, fontsize=float(tick_labelsize))
        for tick_label in ax_density.get_yticklabels():
            tick_label.set_color(condition_color_map.get(tick_label.get_text(), "black"))
        ax_density.set_title(format_parameter_label(parameter_name), color=param_color, fontsize=float(parameter_label_fontsize))
        ax_density.set_xlabel(format_parameter_label(parameter_name), color=param_color, fontsize=float(parameter_label_fontsize))
        ax_density.set_ylabel("Condition", fontsize=float(parameter_label_fontsize))
        ax_density.tick_params(axis="both", labelsize=float(tick_labelsize))
        ax_density.grid(alpha=0.18)

        p_delta_matrix = np.full((n_labels, n_labels), np.nan, dtype=float)
        for row_idx, row_label in enumerate(labels):
            row_values = samples_by_label_param[(row_label, parameter_name)]
            for col_idx, col_label in enumerate(labels):
                col_values = samples_by_label_param[(col_label, parameter_name)]
                p_delta = _empirical_probability_greater(row_values, col_values)
                p_delta_matrix[row_idx, col_idx] = p_delta
                p_delta_rows.append(
                    {
                        "parameter_name": parameter_name,
                        "row_condition": row_label,
                        "column_condition": col_label,
                        "p_delta": p_delta,
                    }
                )

        ax_heat = axes[1, param_idx]
        visible_mask = np.tril(np.ones_like(p_delta_matrix, dtype=bool), k=0)
        masked_matrix = np.ma.array(p_delta_matrix, mask=visible_mask)
        image = ax_heat.imshow(
            masked_matrix,
            cmap=str(p_delta_cmap),
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        heatmap_images.append(image)
        ax_heat.set_title("p_delta: P(row > column)", color=param_color, fontsize=float(parameter_label_fontsize))
        ax_heat.set_xticks(np.arange(n_labels))
        ax_heat.set_yticks(np.arange(n_labels))
        ax_heat.set_xticklabels(labels, rotation=45, ha="right", fontsize=float(tick_labelsize))
        ax_heat.set_yticklabels(labels, fontsize=float(tick_labelsize))
        for tick_label in ax_heat.get_xticklabels():
            tick_label.set_color(condition_color_map.get(tick_label.get_text(), "black"))
        for tick_label in ax_heat.get_yticklabels():
            tick_label.set_color(condition_color_map.get(tick_label.get_text(), "black"))
        for row_idx in range(n_labels):
            for col_idx in range(n_labels):
                if col_idx <= row_idx:
                    continue
                value = p_delta_matrix[row_idx, col_idx]
                if np.isfinite(value):
                    text_color = "white" if abs(value - 0.5) > 0.25 else "black"
                    ax_heat.text(
                        col_idx,
                        row_idx,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=float(p_delta_text_fontsize),
                        color=text_color,
                    )
        ax_heat.set_xlim(-0.5, n_labels - 0.5)
        ax_heat.set_ylim(n_labels - 0.5, -0.5)
        ax_heat.tick_params(axis="both", which="both", length=0)

    if show_density_legend:
        handles = [
            Line2D(
                [0],
                [0],
                color=condition_color_map[label],
                lw=float(density_linewidth),
                label=str(label),
            )
            for label in labels
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, float(legend_y)),
            ncol=min(len(handles), 4),
            fontsize=float(legend_fontsize),
            frameon=True,
        )
    if heatmap_images:
        colorbar = fig.colorbar(
            heatmap_images[-1],
            ax=axes[1, :].ravel().tolist(),
            shrink=float(colorbar_shrink),
            pad=float(colorbar_pad),
        )
        colorbar.set_label(str(heatmap_colorbar_label), fontsize=float(parameter_label_fontsize))
        colorbar.ax.tick_params(labelsize=float(tick_labelsize))

    fig.suptitle(
        "Posterior condition diagnostics\n"
        "Top: ridge densities with P(theta > 0) | Bottom: p_delta = P(row condition > column condition)",
        fontsize=float(title_fontsize),
        y=float(title_y),
    )
    fig.subplots_adjust(
        left=float(figure_left),
        right=float(figure_right),
        bottom=float(figure_bottom),
        top=float(figure_top),
        wspace=float(subplot_wspace),
        hspace=float(subplot_hspace),
    )

    saved_path = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        saved_path = output_path
    plt.show()

    return {
        "figure": fig,
        "saved_path": saved_path,
        "labels": labels,
        "condition_colors": condition_color_map,
        "p_delta_df": pd.DataFrame(p_delta_rows),
        "p_gt_zero_df": pd.DataFrame(p_gt_zero_rows),
    }


def plot_posterior_condition_comparison(
    posterior_samples_df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    return plot_posterior_condition_diagnostics(posterior_samples_df, **kwargs)


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values.copy()
    sorter = np.argsort(values, kind="mergesort")
    sorted_vals = values[sorter]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[sorter[start:end]] = avg_rank
        start = end
    return ranks


def _pearson_corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if x_std <= 0 or y_std <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    return _pearson_corr_safe(_rankdata_average(x), _rankdata_average(y))


def _format_corr_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.2f}"


def _qualitative_cmap_color(cmap_name: str, index: int) -> Any:
    cmap = plt.get_cmap(str(cmap_name))
    colors = getattr(cmap, "colors", None)
    if colors is not None and len(colors) > 0:
        return colors[int(index) % len(colors)]
    return cmap(int(index) % int(getattr(cmap, "N", 256)))


def _resolve_label_colors(
    labels: list[str],
    *,
    colors: dict[str, str] | list[str] | tuple[str, ...] | None = None,
    cmap: str = "tab10",
) -> dict[str, Any]:
    unique_labels = list(dict.fromkeys([str(label) for label in labels]))
    if isinstance(colors, dict):
        return {
            label: colors.get(label, _qualitative_cmap_color(cmap, idx))
            for idx, label in enumerate(unique_labels)
        }
    if colors is not None:
        color_list = [str(color) for color in colors]
        if not color_list:
            raise ValueError("colors must not be empty when provided as a sequence")
        return {
            label: color_list[idx % len(color_list)]
            for idx, label in enumerate(unique_labels)
        }
    return {
        label: _qualitative_cmap_color(cmap, idx)
        for idx, label in enumerate(unique_labels)
    }


def _empirical_probability_greater(row_values: np.ndarray, column_values: np.ndarray) -> float:
    row = np.asarray(row_values, dtype=float).reshape(-1)
    column = np.asarray(column_values, dtype=float).reshape(-1)
    row = row[np.isfinite(row)]
    column = column[np.isfinite(column)]
    if row.size == 0 or column.size == 0:
        return float("nan")
    sorted_column = np.sort(column)
    return float(np.mean(np.searchsorted(sorted_column, row, side="left") / float(sorted_column.size)))


def _add_relevance_annotation(
    ax: plt.Axes,
    *,
    n_relevant: int,
    r_relevant: float,
    r_all: float,
    rho_relevant: float,
    rho_all: float,
    relevant_color: str,
    all_color: str,
    fontsize: float,
    annotation_loc: str,
) -> None:
    text_props = {"fontsize": float(fontsize)}
    count_row = TextArea(f"n_rel = {int(n_relevant)}", textprops=text_props)
    r_row = HPacker(
        children=[
            TextArea(f"r={_format_corr_value(r_relevant)}", textprops={**text_props, "color": relevant_color}),
            TextArea(f" (all: r={_format_corr_value(r_all)})", textprops={**text_props, "color": all_color}),
        ],
        pad=0,
        sep=0,
        align="baseline",
    )
    rho_row = HPacker(
        children=[
            TextArea(f"\u03c1={_format_corr_value(rho_relevant)}", textprops={**text_props, "color": relevant_color}),
            TextArea(f" (all: \u03c1={_format_corr_value(rho_all)})", textprops={**text_props, "color": all_color}),
        ],
        pad=0,
        sep=0,
        align="baseline",
    )
    box = VPacker(children=[count_row, r_row, rho_row], pad=0, sep=1, align="left")
    box_alignment = (0, 1) if annotation_loc == "upper left" else (1, 1)
    anchor = (0.03, 0.97) if annotation_loc == "upper left" else (0.97, 0.97)
    annotation = AnnotationBbox(
        box,
        anchor,
        xycoords="axes fraction",
        box_alignment=box_alignment,
        frameon=True,
        bboxprops={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        zorder=3,
    )
    ax.add_artist(annotation)


def compute_feature_space_relevance_mask(
    *,
    x_sim_z: np.ndarray,
    x_exp_z: np.ndarray,
    k_nearest_per_x0: int = 200,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_sim_z = np.asarray(x_sim_z, dtype=float)
    x_exp_z = np.asarray(x_exp_z, dtype=float)
    if x_sim_z.ndim != 2:
        raise ValueError("x_sim_z must be a 2D array")
    if x_exp_z.ndim != 2:
        raise ValueError("x_exp_z must be a 2D array")
    if x_sim_z.shape[0] < 1:
        raise ValueError("x_sim_z is empty")
    if x_exp_z.shape[0] < 1:
        raise ValueError("x_exp_z is empty")
    if x_sim_z.shape[1] != x_exp_z.shape[1]:
        raise ValueError("x_sim_z and x_exp_z must have the same feature dimension")
    if int(k_nearest_per_x0) < 1:
        raise ValueError("k_nearest_per_x0 must be >= 1")
    effective_k = min(int(k_nearest_per_x0), int(x_sim_z.shape[0]))
    diffs = x_exp_z[:, None, :] - x_sim_z[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    relevant_mask = np.zeros(x_sim_z.shape[0], dtype=bool)
    nearest_indices_by_observation: list[list[int]] = []
    min_distance_per_sim = np.min(distances, axis=0)
    for obs_idx in range(distances.shape[0]):
        nearest_indices = np.argsort(distances[obs_idx], kind="mergesort")[:effective_k]
        relevant_mask[nearest_indices] = True
        nearest_indices_by_observation.append([int(idx) for idx in nearest_indices.tolist()])
    metadata = {
        "normalization": "dataset.feature_scaler",
        "k_requested": int(k_nearest_per_x0),
        "k_effective": int(effective_k),
        "n_simulations": int(x_sim_z.shape[0]),
        "n_observations": int(x_exp_z.shape[0]),
        "n_relevant_simulations": int(np.sum(relevant_mask)),
        "fraction_relevant": float(np.mean(relevant_mask)),
        "nearest_indices_by_observation": nearest_indices_by_observation,
        "min_distance_summary": {
            "min": float(np.min(min_distance_per_sim)),
            "median": float(np.median(min_distance_per_sim)),
            "max": float(np.max(min_distance_per_sim)),
        },
    }
    return relevant_mask, metadata


def plot_feature_parameter_relevance_grid(
    dataset: InferenceDatasetBundle,
    *,
    posterior_modes_df: pd.DataFrame | None = None,
    k_nearest_per_x0: int = 200,
    output_path: str | Path | None = None,
    figure_dpi: int = 140,
    cell_size: float = 2.4,
    all_points_color: str = "#CFCFCF",
    all_points_alpha: float = 0.45,
    all_points_size: float = 10.0,
    relevant_points_color: str = "#D62728",
    relevant_points_alpha: float = 0.9,
    relevant_points_size: float = 14.0,
    text_fontsize: float = 8.0,
    show_panel_annotations: bool = True,
    annotation_loc: str = "upper left",
    show_grid: bool = True,
    parameter_colors: dict[str, str] | None = None,
    feature_colors: dict[str, str] | None = None,
    axis_label_fallback_color: str = "black",
    show_experimental_guides: bool = False,
    experimental_feature_line_color: str = "#2CA02C",
    posterior_mode_line_color: str = "#9467BD",
    experimental_guide_colors: dict[str, str] | list[str] | tuple[str, ...] | None = None,
    experimental_guide_cmap: str = "tab10",
    experimental_guide_linewidth: float = 0.9,
    experimental_guide_linestyle: str = "--",
    experimental_guide_alpha: float = 0.78,
    show_experimental_guide_legend: bool = True,
    experimental_guide_legend_fontsize: float = 8.0,
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    if not dataset.selected_parameters:
        raise ValueError("dataset.selected_parameters is empty")
    if not dataset.selected_features:
        raise ValueError("dataset.selected_features is empty")
    if len(dataset.sim_rows_used) < 1:
        raise ValueError("dataset.sim_rows_used is empty")
    if len(dataset.exp_observations) < 1:
        raise ValueError("dataset.exp_observations is empty")

    relevance_mask, relevance_metadata = compute_feature_space_relevance_mask(
        x_sim_z=dataset.x_sim_z,
        x_exp_z=dataset.x_exp_z,
        k_nearest_per_x0=k_nearest_per_x0,
    )

    n_rows = len(dataset.selected_features)
    n_cols = len(dataset.selected_parameters)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(cell_size * n_cols, cell_size * n_rows),
        dpi=figure_dpi,
        squeeze=False,
    )

    annotation_loc = str(annotation_loc).strip().lower()
    if annotation_loc not in {"upper left", "upper right"}:
        raise ValueError("annotation_loc must be 'upper left' or 'upper right'")
    resolved_parameter_label_colors = {
        name: _resolve_name_color(name, parameter_colors, fallback=str(axis_label_fallback_color))
        for name in dataset.selected_parameters
    }
    resolved_feature_label_colors = {
        name: _resolve_name_color(name, feature_colors, fallback=str(axis_label_fallback_color))
        for name in dataset.selected_features
    }

    exp_guides_df = dataset.exp_observations.copy() if bool(show_experimental_guides) else pd.DataFrame()
    mode_guides_df = pd.DataFrame()
    if bool(show_experimental_guides) and not exp_guides_df.empty:
        exp_observation_labels = exp_guides_df["observation_label"].astype(str).tolist()
        if posterior_modes_df is not None and not posterior_modes_df.empty:
            if "observation_label" not in posterior_modes_df.columns:
                raise ValueError("posterior_modes_df must contain an 'observation_label' column")
            mode_guides_df = posterior_modes_df.copy()
            mode_guides_df["observation_label"] = mode_guides_df["observation_label"].astype(str)
            mode_guides_df = mode_guides_df.loc[mode_guides_df["observation_label"].isin(exp_observation_labels)].copy()
            duplicated_labels = mode_guides_df.loc[
                mode_guides_df["observation_label"].duplicated(),
                "observation_label",
            ].unique().tolist()
            if duplicated_labels:
                raise ValueError(
                    "posterior_modes_df must have at most one mode row per observation_label for relevance-grid guides; "
                    f"duplicates found for {duplicated_labels}"
                )
    else:
        exp_observation_labels = []

    guide_colors_by_label: dict[str, str] = {}
    if exp_observation_labels:
        unique_labels = list(dict.fromkeys(exp_observation_labels))
        if isinstance(experimental_guide_colors, dict):
            guide_colors_by_label = {
                label: str(experimental_guide_colors.get(label, _qualitative_cmap_color(str(experimental_guide_cmap), idx)))
                for idx, label in enumerate(unique_labels)
            }
        elif experimental_guide_colors is not None:
            color_list = [str(color) for color in experimental_guide_colors]
            if not color_list:
                raise ValueError("experimental_guide_colors must not be empty when provided as a sequence")
            guide_colors_by_label = {
                label: color_list[idx % len(color_list)]
                for idx, label in enumerate(unique_labels)
            }
        else:
            guide_colors_by_label = {
                label: _qualitative_cmap_color(str(experimental_guide_cmap), idx)
                for idx, label in enumerate(unique_labels)
            }

    guide_records: list[dict[str, Any]] = []
    if bool(show_experimental_guides) and not exp_guides_df.empty:
        mode_by_label = mode_guides_df.set_index("observation_label") if not mode_guides_df.empty else pd.DataFrame()
        for _, exp_row in exp_guides_df.iterrows():
            label = str(exp_row["observation_label"])
            mode_row = mode_by_label.loc[label] if not mode_by_label.empty and label in mode_by_label.index else None
            guide_records.append(
                {
                    "observation_label": label,
                    "color": guide_colors_by_label.get(label, str(experimental_feature_line_color)),
                    "exp_row": exp_row,
                    "mode_row": mode_row,
                }
            )

    for row_idx, feature_name in enumerate(dataset.selected_features):
        y_all = pd.to_numeric(dataset.sim_rows_used[feature_name], errors="coerce").to_numpy(dtype=float)
        for col_idx, parameter_name in enumerate(dataset.selected_parameters):
            ax = axes[row_idx, col_idx]
            x_all = pd.to_numeric(dataset.sim_rows_used[parameter_name], errors="coerce").to_numpy(dtype=float)
            finite_mask = np.isfinite(x_all) & np.isfinite(y_all)
            if np.any(finite_mask):
                ax.scatter(
                    x_all[finite_mask],
                    y_all[finite_mask],
                    color=all_points_color,
                    alpha=float(all_points_alpha),
                    s=float(all_points_size),
                    linewidths=0,
                    zorder=1,
                )
            relevant_finite_mask = finite_mask & relevance_mask
            if np.any(relevant_finite_mask):
                ax.scatter(
                    x_all[relevant_finite_mask],
                    y_all[relevant_finite_mask],
                    color=relevant_points_color,
                    alpha=float(relevant_points_alpha),
                    s=float(relevant_points_size),
                    linewidths=0,
                    zorder=2,
                )

            if show_panel_annotations:
                _add_relevance_annotation(
                    ax,
                    n_relevant=int(np.sum(relevant_finite_mask)),
                    r_relevant=_pearson_corr_safe(x_all[relevant_finite_mask], y_all[relevant_finite_mask]),
                    r_all=_pearson_corr_safe(x_all[finite_mask], y_all[finite_mask]),
                    rho_relevant=_spearman_corr_safe(x_all[relevant_finite_mask], y_all[relevant_finite_mask]),
                    rho_all=_spearman_corr_safe(x_all[finite_mask], y_all[finite_mask]),
                    relevant_color=relevant_points_color,
                    all_color=all_points_color,
                    fontsize=float(text_fontsize),
                    annotation_loc=annotation_loc,
                )

            if bool(show_experimental_guides):
                for guide_record in guide_records:
                    color = guide_record["color"]
                    exp_row = guide_record["exp_row"]
                    mode_row = guide_record["mode_row"]
                    feature_value = pd.to_numeric(pd.Series([exp_row.get(feature_name, np.nan)]), errors="coerce").iloc[0]
                    if np.isfinite(feature_value):
                        ax.axhline(
                            float(feature_value),
                            color=color,
                            lw=float(experimental_guide_linewidth),
                            ls=str(experimental_guide_linestyle),
                            alpha=float(experimental_guide_alpha),
                            zorder=2.5,
                        )
                    mode_value = (
                        pd.to_numeric(pd.Series([mode_row.get(parameter_name, np.nan)]), errors="coerce").iloc[0]
                        if mode_row is not None and parameter_name in mode_row.index
                        else np.nan
                    )
                    if np.isfinite(mode_value):
                        ax.axvline(
                            float(mode_value),
                            color=color,
                            lw=float(experimental_guide_linewidth),
                            ls=str(experimental_guide_linestyle),
                            alpha=float(experimental_guide_alpha),
                            zorder=2.6,
                        )

            ax.set_xlabel(format_parameter_label(parameter_name), fontsize=9)
            ax.xaxis.label.set_color(resolved_parameter_label_colors[parameter_name])
            ax.set_ylabel(format_feature_label(feature_name), fontsize=9)
            ax.yaxis.label.set_color(resolved_feature_label_colors[feature_name])
            ax.tick_params(axis="both", which="both", labelsize=8, length=3)
            if show_grid:
                ax.grid(alpha=0.18)

    fig.suptitle(
        "Feature-parameter relevance grid\n"
        f"k = {relevance_metadata['k_effective']} per observation | "
        f"relevant simulations = {relevance_metadata['n_relevant_simulations']} / {relevance_metadata['n_simulations']}",
        fontsize=12,
    )
    if bool(show_experimental_guides) and bool(show_experimental_guide_legend):
        handles = [
            Line2D(
                [0],
                [0],
                color=guide_record["color"],
                lw=float(experimental_guide_linewidth),
                ls=str(experimental_guide_linestyle),
                alpha=float(experimental_guide_alpha),
                label=str(guide_record["observation_label"]),
            )
            for guide_record in guide_records
        ]
        if not handles:
            handles = [
                Line2D(
                    [0],
                    [0],
                    color=str(experimental_feature_line_color),
                    lw=float(experimental_guide_linewidth),
                    ls=str(experimental_guide_linestyle),
                    alpha=float(experimental_guide_alpha),
                    label="Experimental guide",
                )
            ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=min(4, len(handles)),
            frameon=True,
            title="Observation guides: horizontal = feature, vertical = posterior mode",
            title_fontsize=float(experimental_guide_legend_fontsize),
            fontsize=float(experimental_guide_legend_fontsize),
        )
    fig.subplots_adjust(
        left=0.08,
        right=0.92,
        bottom=0.12,
        top=0.86 if bool(show_experimental_guides) and bool(show_experimental_guide_legend) else 0.90,
        wspace=0.30,
        hspace=0.35,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        relevance_metadata["output_path"] = str(output_path)
    if bool(show_experimental_guides):
        relevance_metadata["experimental_guides"] = {
            "n_observation_labels": int(len(exp_observation_labels)),
            "observation_labels": exp_observation_labels,
            "colors_by_observation_label": {str(key): str(value) for key, value in guide_colors_by_label.items()},
            "has_posterior_mode_guides": bool(not mode_guides_df.empty),
        }
    return fig, axes, relevance_metadata


def plot_feature_parameter_correlation_heatmaps(
    dataset: InferenceDatasetBundle,
    *,
    k_nearest_per_x0: int = 200,
    output_path: str | Path | None = None,
    figure_dpi: int = 140,
    cell_size: float = 0.55,
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    show_cell_values: bool = True,
    text_fontsize: float = 8.0,
    tick_labelsize: float = 8.0,
    parameter_colors: dict[str, str] | None = None,
    feature_colors: dict[str, str] | None = None,
    tick_label_fallback_color: str = "black",
    square_cells: bool = True,
    subplot_wspace: float = 0.35,
    subplot_hspace: float = 0.45,
    colorbar_shrink: float = 0.9,
    panels: str | list[str] | tuple[str, ...] = "all",
) -> tuple[plt.Figure, np.ndarray, dict[str, Any]]:
    if not dataset.selected_parameters:
        raise ValueError("dataset.selected_parameters is empty")
    if not dataset.selected_features:
        raise ValueError("dataset.selected_features is empty")
    if len(dataset.sim_rows_used) < 1:
        raise ValueError("dataset.sim_rows_used is empty")
    if len(dataset.exp_observations) < 1:
        raise ValueError("dataset.exp_observations is empty")

    relevance_mask, relevance_metadata = compute_feature_space_relevance_mask(
        x_sim_z=dataset.x_sim_z,
        x_exp_z=dataset.x_exp_z,
        k_nearest_per_x0=k_nearest_per_x0,
    )

    available_panels: dict[str, tuple[str, np.ndarray, Any]] = {
        "relevant_pearson": ("Relevant | Pearson r", relevance_mask, _pearson_corr_safe),
        "relevant_spearman": ("Relevant | Spearman \u03c1", relevance_mask, _spearman_corr_safe),
        "all_pearson": ("All | Pearson r", np.ones(len(relevance_mask), dtype=bool), _pearson_corr_safe),
        "all_spearman": ("All | Spearman \u03c1", np.ones(len(relevance_mask), dtype=bool), _spearman_corr_safe),
    }
    if isinstance(panels, str):
        requested_panels = list(available_panels) if panels.strip().lower() == "all" else [panels.strip().lower()]
    else:
        requested_panels = [str(panel).strip().lower() for panel in panels]
    if not requested_panels:
        raise ValueError("panels must not be empty")
    invalid_panels = [panel for panel in requested_panels if panel not in available_panels]
    if invalid_panels:
        raise ValueError(
            "Invalid panels requested: "
            f"{invalid_panels}. Valid values are {list(available_panels)} or 'all'."
        )
    corr_specs = [(panel_key, *available_panels[panel_key]) for panel_key in requested_panels]

    if len(corr_specs) == 1:
        n_fig_rows = 1
        n_fig_cols = 1
    elif len(corr_specs) == 2:
        n_fig_rows = 1
        n_fig_cols = 2
    else:
        n_fig_rows = 2
        n_fig_cols = int(math.ceil(len(corr_specs) / 2))

    fig, axes = plt.subplots(
        n_fig_rows,
        n_fig_cols,
        figsize=(
            max(4.5, float(cell_size) * len(dataset.selected_parameters) * n_fig_cols + 1.8 * n_fig_cols),
            max(4.5, float(cell_size) * len(dataset.selected_features) * n_fig_rows + 1.8 * n_fig_rows),
        ),
        dpi=figure_dpi,
        squeeze=False,
    )
    matrices: dict[str, list[list[float]]] = {}
    last_im = None

    for panel_idx, (panel_key, title, subset_mask, corr_fn) in enumerate(corr_specs):
        row_idx = panel_idx // n_fig_cols
        col_idx = panel_idx % n_fig_cols
        ax = axes[row_idx, col_idx]
        values: list[list[float]] = []
        for feature_name in dataset.selected_features:
            y_all = pd.to_numeric(dataset.sim_rows_used[feature_name], errors="coerce").to_numpy(dtype=float)
            row_values: list[float] = []
            for parameter_name in dataset.selected_parameters:
                x_all = pd.to_numeric(dataset.sim_rows_used[parameter_name], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(x_all) & np.isfinite(y_all) & subset_mask
                row_values.append(float(corr_fn(x_all[mask], y_all[mask])) if np.any(mask) else math.nan)
            values.append(row_values)
        matrix = np.asarray(values, dtype=float)
        matrices[panel_key] = values
        last_im = ax.imshow(
            matrix,
            aspect="equal" if bool(square_cells) else "auto",
            cmap=cmap,
            vmin=float(vmin),
            vmax=float(vmax),
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(len(dataset.selected_parameters)))
        ax.set_xticklabels(
            [format_parameter_label(name) for name in dataset.selected_parameters],
            rotation=45,
            ha="right",
            fontsize=float(tick_labelsize),
        )
        for tick_label, parameter_name in zip(ax.get_xticklabels(), dataset.selected_parameters, strict=True):
            tick_label.set_color(
                _resolve_name_color(
                    parameter_name,
                    parameter_colors,
                    fallback=str(tick_label_fallback_color),
                )
            )
        ax.set_yticks(np.arange(len(dataset.selected_features)))
        ax.set_yticklabels(
            [format_feature_label(name) for name in dataset.selected_features],
            fontsize=float(tick_labelsize),
        )
        for tick_label, feature_name in zip(ax.get_yticklabels(), dataset.selected_features, strict=True):
            tick_label.set_color(
                _resolve_name_color(
                    feature_name,
                    feature_colors,
                    fallback=str(tick_label_fallback_color),
                )
            )
        ax.tick_params(axis="both", which="both", length=3)
        if show_cell_values:
            for feature_idx in range(matrix.shape[0]):
                for parameter_idx in range(matrix.shape[1]):
                    value = matrix[feature_idx, parameter_idx]
                    label = "NA" if not np.isfinite(value) else f"{value:.2f}"
                    text_color = "white" if np.isfinite(value) and abs(value) >= 0.55 else "black"
                    ax.text(
                        parameter_idx,
                        feature_idx,
                        label,
                        ha="center",
                        va="center",
                        fontsize=float(text_fontsize),
                        color=text_color,
                    )
    for extra_idx in range(len(corr_specs), n_fig_rows * n_fig_cols):
        axes[extra_idx // n_fig_cols, extra_idx % n_fig_cols].set_visible(False)
    colorbar_ax = None
    if last_im is not None:
        colorbar_ax = fig.add_axes([0.915, 0.20, 0.018, 0.62])
        fig.colorbar(last_im, cax=colorbar_ax, shrink=float(colorbar_shrink), label="Correlation")
    fig.suptitle(
        "Feature-parameter correlation heatmaps\n"
        f"k = {relevance_metadata['k_effective']} per observation | relevant simulations = "
        f"{relevance_metadata['n_relevant_simulations']} / {relevance_metadata['n_simulations']}",
        fontsize=12,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.18,
        top=0.90,
        wspace=float(subplot_wspace),
        hspace=float(subplot_hspace),
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        relevance_metadata["output_path"] = str(output_path)
    relevance_metadata["matrices"] = matrices
    relevance_metadata["panels_displayed"] = requested_panels
    return fig, axes, relevance_metadata


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if pd.isna(value):
        return None
    return str(value)


def save_training_diagnostics(
    *,
    output_dir: str | Path,
    trained: TrainedPosteriorBundle | EnsemblePosteriorBundle,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    network_records = _iter_trained_networks(trained)
    summaries = [dict(network.training_summary or {}) for _, _, network in network_records]
    saved_paths: dict[str, Path] = {}
    if not summaries:
        return saved_paths

    n_networks = len(summaries)
    if n_networks == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6.0, 4.0), dpi=140, squeeze=False)
        axes_flat = [axes[0, 0]]
        summary_ax = None
    else:
        n_cols = min(3, n_networks)
        n_rows = int(math.ceil(n_networks / n_cols))
        fig = plt.figure(figsize=(6.0 * n_cols, 3.8 * (n_rows + 1)), dpi=140)
        grid = GridSpec(n_rows + 1, n_cols, figure=fig, hspace=0.38, wspace=0.25)
        axes_flat = [fig.add_subplot(grid[row, col]) for row in range(n_rows) for col in range(n_cols)]
        for ax in axes_flat[n_networks:]:
            ax.axis("off")
        axes_flat = axes_flat[:n_networks]
        summary_ax = fig.add_subplot(grid[n_rows, :])

    metadata_rows: list[dict[str, Any]] = []
    for idx, ((network_id, network_seed, _network), summary, ax) in enumerate(zip(network_records, summaries, axes_flat, strict=True)):
        training_loss = np.asarray(summary.get("training_loss", []), dtype=float).reshape(-1)
        validation_loss = np.asarray(summary.get("validation_loss", []), dtype=float).reshape(-1)
        if training_loss.size:
            ax.plot(np.arange(1, training_loss.size + 1), training_loss, label="training loss", color="#1f77b4")
        if validation_loss.size:
            ax.plot(np.arange(1, validation_loss.size + 1), validation_loss, label="validation loss", color="#d62728")
            if summary_ax is not None:
                summary_ax.plot(
                    np.arange(1, validation_loss.size + 1),
                    validation_loss,
                    label=f"network {network_id}",
                    alpha=0.75,
                )
        epochs = int(max(training_loss.size, validation_loss.size))
        final_train = float(training_loss[-1]) if training_loss.size else np.nan
        final_val = float(validation_loss[-1]) if validation_loss.size else np.nan
        metadata_rows.append(
            {
                "network_id": int(network_id),
                "network_seed": np.nan if network_seed is None else int(network_seed),
                "epochs": epochs,
                "final_training_loss": final_train,
                "final_validation_loss": final_val,
            }
        )
        title = f"Network {network_id}"
        if network_seed is not None:
            title += f" (seed {network_seed})"
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    if summary_ax is not None:
        summary_ax.set_title("Validation loss overlay")
        summary_ax.set_xlabel("Epoch")
        summary_ax.set_ylabel("Validation loss")
        summary_ax.grid(alpha=0.2)
        summary_ax.legend(fontsize=8, ncol=min(4, n_networks))

    fig.tight_layout()
    loss_curve_path = output_dir / "training_loss_curve.png"
    fig.savefig(loss_curve_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    saved_paths["training_loss_curve"] = loss_curve_path
    training_summary_path = output_dir / "training_network_summary.csv"
    pd.DataFrame(metadata_rows).to_csv(training_summary_path, index=False)
    saved_paths["training_network_summary_csv"] = training_summary_path
    return saved_paths


def save_inference_outputs(
    *,
    output_dir: str | Path,
    dataset: InferenceDatasetBundle,
    trained: TrainedPosteriorBundle | EnsemblePosteriorBundle,
    posterior_samples_df: pd.DataFrame,
    posterior_summary_df: pd.DataFrame,
    posterior_modes_df: pd.DataFrame | None = None,
    network_posterior_modes_df: pd.DataFrame | None = None,
    save_posterior_samples_pkl: bool = True,
    save_trained_bundle_pkl: bool = True,
    save_training_diagnostics_figures: bool = True,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prior_bounds_path = output_dir / "selected_parameter_prior_bounds.csv"
    dataset.parameter_bounds.to_csv(prior_bounds_path, index=False)

    posterior_samples_path = output_dir / "posterior_samples.pkl"
    if save_posterior_samples_pkl:
        posterior_samples_df.to_pickle(posterior_samples_path)

    posterior_summary_path = output_dir / "posterior_summary.csv"
    posterior_summary_df.to_csv(posterior_summary_path, index=False)

    posterior_modes_pickle_path = output_dir / "posterior_modes.pkl"
    posterior_modes_csv_path = output_dir / "posterior_modes.csv"
    if posterior_modes_df is not None and not posterior_modes_df.empty:
        posterior_modes_df.to_pickle(posterior_modes_pickle_path)
        posterior_modes_df.to_csv(posterior_modes_csv_path, index=False)

    network_modes_pickle_path = output_dir / "posterior_modes_by_network.pkl"
    network_modes_csv_path = output_dir / "posterior_modes_by_network.csv"
    if network_posterior_modes_df is not None and not network_posterior_modes_df.empty:
        network_posterior_modes_df.to_pickle(network_modes_pickle_path)
        network_posterior_modes_df.to_csv(network_modes_csv_path, index=False)

    feature_scaling_metadata = get_feature_scaling_metadata(dataset)
    preprocessing_metadata = {
        "selected_parameters": dataset.selected_parameters,
        "selected_features": dataset.selected_features,
        "parameter_bounds": dataset.parameter_bounds.to_dict(orient="records"),
        "feature_means": dataset.feature_scaler.means.tolist(),
        "feature_stds": dataset.feature_scaler.stds.tolist(),
        "feature_normalization_source": feature_scaling_metadata.get("feature_normalization_source", "simulated"),
        "feature_normalization_experimental_level": feature_scaling_metadata.get("feature_normalization_experimental_level"),
        "feature_scaler_row_count": feature_scaling_metadata.get("scaler_row_count"),
        "feature_normalization": feature_scaling_metadata,
    }
    preprocessing_path = output_dir / "preprocessing_metadata.json"
    preprocessing_path.write_text(json.dumps(_jsonable(preprocessing_metadata), indent=2), encoding="utf-8")

    network_records = _iter_trained_networks(trained)
    network_metadata = []
    for network_id, network_seed, network in network_records:
        summary = dict(network.training_summary or {})
        training_loss = np.asarray(summary.get("training_loss", []), dtype=float).reshape(-1)
        validation_loss = np.asarray(summary.get("validation_loss", []), dtype=float).reshape(-1)
        network_metadata.append(
            {
                "network_id": int(network_id),
                "network_seed": np.nan if network_seed is None else int(network_seed),
                "density_estimator": network.density_estimator_name,
                "device": network.device,
                "train_kwargs": network.train_kwargs,
                "num_simulations": network.num_simulations,
                "epochs": int(max(training_loss.size, validation_loss.size)),
                "final_training_loss": float(training_loss[-1]) if training_loss.size else np.nan,
                "final_validation_loss": float(validation_loss[-1]) if validation_loss.size else np.nan,
                "training_summary": summary,
            }
        )
    first_network = network_records[0][2]
    training_metadata = {
        "sim_table_path": dataset.sim_table_path,
        "exp_table_path": dataset.exp_table_path,
        "priors_pickle_path": dataset.priors_pickle_path,
        "num_sim_rows_used": int(len(dataset.sim_rows_used)),
        "num_exp_observations": int(len(dataset.exp_observations)),
        "num_skipped_experimental_rows": int(len(dataset.skipped_experimental_rows)),
        "feature_normalization": feature_scaling_metadata,
        "is_ensemble": bool(_is_ensemble_bundle(trained)),
        "n_networks": int(len(network_records)),
        "ensemble_metadata": dict(trained.ensemble_metadata) if isinstance(trained, EnsemblePosteriorBundle) else {},
        "density_estimator": first_network.density_estimator_name,
        "device": first_network.device,
        "train_kwargs": first_network.train_kwargs,
        "networks": network_metadata,
    }
    training_path = output_dir / "training_metadata.json"
    training_path.write_text(json.dumps(_jsonable(training_metadata), indent=2), encoding="utf-8")

    if not dataset.skipped_experimental_rows.empty:
        skipped_path = output_dir / "skipped_experimental_rows.csv"
        dataset.skipped_experimental_rows.to_csv(skipped_path, index=False)
    else:
        skipped_path = output_dir / "skipped_experimental_rows.csv"

    saved_paths = {
        "prior_bounds_csv": prior_bounds_path,
        "posterior_summary_csv": posterior_summary_path,
        "preprocessing_metadata_json": preprocessing_path,
        "training_metadata_json": training_path,
    }
    if save_posterior_samples_pkl:
        saved_paths["posterior_samples_pkl"] = posterior_samples_path
    if posterior_modes_df is not None and not posterior_modes_df.empty:
        saved_paths["posterior_modes_pkl"] = posterior_modes_pickle_path
        saved_paths["posterior_modes_csv"] = posterior_modes_csv_path
    if network_posterior_modes_df is not None and not network_posterior_modes_df.empty:
        saved_paths["posterior_modes_by_network_pkl"] = network_modes_pickle_path
        saved_paths["posterior_modes_by_network_csv"] = network_modes_csv_path
    if not dataset.skipped_experimental_rows.empty:
        saved_paths["skipped_experimental_rows_csv"] = skipped_path
    if save_trained_bundle_pkl:
        saved_paths["trained_inference_bundle_pkl"] = save_trained_inference_bundle(
            output_dir=output_dir,
            dataset=dataset,
            trained=trained,
        )
    if save_training_diagnostics_figures:
        saved_paths.update(
            save_training_diagnostics(
                output_dir=output_dir,
                trained=trained,
            )
        )
    return saved_paths


def prepare_inference_dataset_from_sim_rows(
    sim_rows_df: pd.DataFrame,
    *,
    selected_parameters: list[str],
    selected_features: list[str],
    parameter_bounds: pd.DataFrame,
    sim_table_path: str | Path | None = None,
    priors_pickle_path: str | Path | None = None,
) -> InferenceDatasetBundle:
    work_sim_df = sim_rows_df.copy()
    for parameter_name in selected_parameters:
        work_sim_df[parameter_name] = pd.to_numeric(work_sim_df[parameter_name], errors="coerce")
    for feature_name in selected_features:
        work_sim_df[feature_name] = pd.to_numeric(work_sim_df[feature_name], errors="coerce")

    sim_required_columns = [*selected_parameters, *selected_features]
    valid_sim_mask = ~work_sim_df[sim_required_columns].isna().any(axis=1)
    sim_rows_used = work_sim_df.loc[valid_sim_mask].copy()
    if sim_rows_used.empty:
        raise ValueError("No simulated rows remain after dropping rows with missing selected parameters/features")

    lower = parameter_bounds["lower_bound"].to_numpy(dtype=float)
    upper = parameter_bounds["upper_bound"].to_numpy(dtype=float)
    theta_raw = sim_rows_used[selected_parameters].to_numpy(dtype=float)
    outside_mask = (theta_raw < lower) | (theta_raw > upper)
    if np.any(outside_mask):
        bad_param_indices = np.where(np.any(outside_mask, axis=0))[0].tolist()
        bad_params = [selected_parameters[idx] for idx in bad_param_indices]
        raise ValueError(
            f"Some simulated parameter values fall outside the loaded prior bounds: {bad_params}"
        )
    parameter_scaler = ParameterBoundsScaler(
        parameter_names=list(selected_parameters),
        lower_bounds=lower,
        upper_bounds=upper,
    )
    theta_unit = parameter_scaler.transform(theta_raw)

    feature_scaler, scaler_rows_used, skipped_feature_normalization_rows = _fit_feature_zscore_scaler_from_rows(
        sim_rows_used,
        selected_features=selected_features,
        source_label="simulated training rows",
        min_rows=1,
    )
    feature_scaling_metadata = build_feature_scaling_metadata(
        feature_scaler=feature_scaler,
        source="simulated",
        selected_features=selected_features,
        scaler_row_count=len(scaler_rows_used),
        experimental_level=None,
        skipped_rows=skipped_feature_normalization_rows,
    )
    x_sim_raw = sim_rows_used[selected_features].to_numpy(dtype=float)
    x_sim_z = feature_scaler.transform(x_sim_raw)
    empty_exp = pd.DataFrame(columns=["observation_label", *EXPERIMENTAL_METADATA_COLUMNS, "n_rows_aggregated", *selected_features])
    return InferenceDatasetBundle(
        sim_table_path=Path("." if sim_table_path is None else sim_table_path),
        exp_table_path=Path("." if sim_table_path is None else sim_table_path),
        priors_pickle_path=Path("." if priors_pickle_path is None else priors_pickle_path),
        sim_rows_used=sim_rows_used,
        exp_observations=empty_exp,
        skipped_experimental_rows=pd.DataFrame(),
        selected_parameters=list(selected_parameters),
        selected_features=list(selected_features),
        parameter_bounds=parameter_bounds.copy(),
        parameter_scaler=parameter_scaler,
        feature_scaler=feature_scaler,
        theta_unit=theta_unit,
        x_sim_z=x_sim_z,
        x_exp_z=np.zeros((0, len(selected_features)), dtype=float),
        feature_scaling_metadata=feature_scaling_metadata,
    )


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy(dtype=float)
    return _pearson_corr(x, y)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.isclose(ss_tot, 0.0):
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray, *, low: float, high: float) -> float:
    span = float(high) - float(low)
    if not math.isfinite(span) or span <= 0:
        return float("nan")
    y_true = (np.asarray(y_true, dtype=float) - float(low)) / span
    y_pred = (np.asarray(y_pred, dtype=float) - float(low)) / span
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _rolling_mean_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window_fraction: float = 0.2,
    min_points: int = 10,
    stride_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    n = x.size
    window = max(int(math.ceil(float(window_fraction) * n)), int(min_points))
    window = min(window, n)
    stride = max(1, int(math.ceil(window * float(stride_fraction))))
    xs = []
    ys = []
    if window == n:
        return np.array([float(np.mean(x))]), np.array([float(np.mean(y))])
    for start in range(0, n - window + 1, stride):
        stop = start + window
        xs.append(float(np.mean(x[start:stop])))
        ys.append(float(np.mean(y[start:stop])))
    if (n - window) % stride != 0:
        xs.append(float(np.mean(x[-window:])))
        ys.append(float(np.mean(y[-window:])))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _moving_average_1d(values: np.ndarray, window_points: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    window_points = max(1, int(window_points))
    if window_points == 1:
        return values.copy()
    kernel = np.ones(window_points, dtype=float) / float(window_points)
    padded = np.pad(values, (window_points // 2, window_points - 1 - window_points // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _posterior_mode_from_samples(
    samples: np.ndarray,
    *,
    low: float,
    high: float,
    grid_points: int = 256,
    bandwidth_scale: float = 1.0,
) -> float:
    samples = np.asarray(samples, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan")
    grid = np.linspace(float(low), float(high), int(grid_points))
    density = _compute_kde_1d(samples, grid=grid, bandwidth_scale=float(bandwidth_scale))
    if density.size == 0 or not np.any(np.isfinite(density)):
        return float("nan")
    return float(grid[int(np.nanargmax(density))])


def _central_coverage_curve_from_samples(
    samples: np.ndarray,
    truth: float,
    *,
    mass_grid: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    mass_grid = np.asarray(mass_grid, dtype=float).reshape(-1)
    if samples.size == 0 or not math.isfinite(float(truth)):
        return np.full_like(mass_grid, np.nan, dtype=float)
    indicators = np.zeros_like(mass_grid, dtype=float)
    for idx, mass in enumerate(mass_grid):
        mass = float(np.clip(mass, 0.0, 1.0))
        alpha = 0.5 * (1.0 - mass)
        low = float(np.quantile(samples, alpha))
        high = float(np.quantile(samples, 1.0 - alpha))
        indicators[idx] = 1.0 if (truth >= low and truth <= high) else 0.0
    return indicators


def run_internal_posterior_validation(
    dataset: InferenceDatasetBundle,
    *,
    n_networks: int = 5,
    train_fraction: float = 0.9,
    posterior_num_samples: int = 1000,
    random_seed: int = 12345,
    density_estimator: str = "maf",
    device: str = "cpu",
    show_progress_bars: bool = True,
    train_kwargs: dict[str, Any] | None = None,
    mass_grid: np.ndarray | list[float] | None = None,
    mode_grid_points: int = 256,
    mode_bandwidth_scale: float = 1.0,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    posterior_num_samples = _coerce_positive_int_like(posterior_num_samples, name="posterior_num_samples")
    n_networks = _coerce_positive_int_like(n_networks, name="n_networks")
    if not (0.0 < float(train_fraction) < 1.0):
        raise ValueError("train_fraction must lie strictly between 0 and 1")
    if len(dataset.sim_rows_used) < 4:
        raise ValueError("At least 4 simulated rows are required for heldout validation")
    base_df = dataset.sim_rows_used.reset_index(drop=True).copy()
    n_total = len(base_df)
    n_train = max(1, min(n_total - 1, int(math.floor(float(train_fraction) * n_total))))
    n_holdout = n_total - n_train
    if n_holdout < 1:
        raise ValueError("Heldout validation requires at least one heldout row")
    mass_grid_arr = np.asarray(np.linspace(0.0, 1.0, 101) if mass_grid is None else mass_grid, dtype=float).reshape(-1)
    if mass_grid_arr.size < 2:
        raise ValueError("mass_grid must contain at least two points")
    parameter_bounds_map = {
        str(row["parameter_name"]): (float(row["lower_bound"]), float(row["upper_bound"]))
        for _, row in dataset.parameter_bounds.iterrows()
    }
    rng = np.random.default_rng(random_seed)
    prediction_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    accuracy_metric_rows: list[dict[str, Any]] = []
    calibration_metric_rows: list[dict[str, Any]] = []

    for network_index in range(n_networks):
        permutation = rng.permutation(n_total)
        train_idx = permutation[:n_train]
        holdout_idx = permutation[n_train:]
        train_df = base_df.iloc[train_idx].reset_index(drop=True).copy()
        holdout_df = base_df.iloc[holdout_idx].reset_index(drop=True).copy()
        train_dataset = prepare_inference_dataset_from_sim_rows(
            train_df,
            selected_parameters=dataset.selected_parameters,
            selected_features=dataset.selected_features,
            parameter_bounds=dataset.parameter_bounds,
            sim_table_path=dataset.sim_table_path,
            priors_pickle_path=dataset.priors_pickle_path,
        )
        trained = train_amortized_posterior(
            train_dataset,
            density_estimator=density_estimator,
            device=device,
            show_progress_bars=show_progress_bars,
            train_kwargs=train_kwargs,
        )
        holdout_x_raw = holdout_df[dataset.selected_features].to_numpy(dtype=float)
        holdout_x_z = train_dataset.feature_scaler.transform(holdout_x_raw)
        for holdout_local_index, holdout_row in holdout_df.iterrows():
            x_tensor = _safe_torch_tensor(holdout_x_z[holdout_local_index : holdout_local_index + 1, :], device=trained.device)
            posterior_samples = trained.posterior.sample((int(posterior_num_samples),), x=x_tensor)
            posterior_unit = np.asarray(posterior_samples.detach().cpu().tolist(), dtype=float)
            posterior_original = train_dataset.parameter_scaler.inverse_transform(posterior_unit)
            for param_idx, parameter_name in enumerate(dataset.selected_parameters):
                low, high = parameter_bounds_map[parameter_name]
                marginal_samples = posterior_original[:, param_idx]
                truth = float(holdout_row[parameter_name])
                pred_mode = _posterior_mode_from_samples(
                    marginal_samples,
                    low=low,
                    high=high,
                    grid_points=mode_grid_points,
                    bandwidth_scale=mode_bandwidth_scale,
                )
                prediction_rows.append(
                    {
                        "network_index": int(network_index),
                        "holdout_row_index": int(holdout_idx[holdout_local_index]),
                        "parameter_name": parameter_name,
                        "ground_truth": truth,
                        "posterior_mode": pred_mode,
                    }
                )
                coverage_curve = _central_coverage_curve_from_samples(
                    marginal_samples,
                    truth,
                    mass_grid=mass_grid_arr,
                )
                for mass_value, covered in zip(mass_grid_arr, coverage_curve, strict=True):
                    calibration_rows.append(
                        {
                            "network_index": int(network_index),
                            "holdout_row_index": int(holdout_idx[holdout_local_index]),
                            "parameter_name": parameter_name,
                            "mass": float(mass_value),
                            "covered": float(covered),
                        }
                    )

        predictions_df = pd.DataFrame(prediction_rows)
        network_predictions = predictions_df.loc[predictions_df["network_index"] == int(network_index)].copy()
        for parameter_name in dataset.selected_parameters:
            subset = network_predictions.loc[network_predictions["parameter_name"] == parameter_name].copy()
            y_true = subset["ground_truth"].to_numpy(dtype=float)
            y_pred = subset["posterior_mode"].to_numpy(dtype=float)
            low, high = parameter_bounds_map[parameter_name]
            accuracy_metric_rows.append(
                {
                    "network_index": int(network_index),
                    "parameter_name": parameter_name,
                    "pearson_r": _pearson_corr(y_true, y_pred),
                    "spearman_r": _spearman_corr(y_true, y_pred),
                    "r2": _r2_score(y_true, y_pred),
                    "normalized_rmse": _normalized_rmse(y_true, y_pred, low=low, high=high),
                    "n_holdout": int(len(subset)),
                }
            )

        calibration_df = pd.DataFrame(calibration_rows)
        network_calibration = calibration_df.loc[calibration_df["network_index"] == int(network_index)].copy()
        for parameter_name in dataset.selected_parameters:
            subset = network_calibration.loc[network_calibration["parameter_name"] == parameter_name].copy()
            empirical = subset.groupby("mass", sort=True)["covered"].mean().reset_index()
            ece = float(np.nanmean(np.abs(empirical["covered"].to_numpy(dtype=float) - empirical["mass"].to_numpy(dtype=float))))
            calibration_metric_rows.append(
                {
                    "network_index": int(network_index),
                    "parameter_name": parameter_name,
                    "ece": ece,
                }
            )

    predictions_df = pd.DataFrame(prediction_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    accuracy_metrics_df = pd.DataFrame(accuracy_metric_rows)
    calibration_metrics_df = pd.DataFrame(calibration_metric_rows)
    aggregated_calibration_df = (
        calibration_df.groupby(["network_index", "parameter_name", "mass"], sort=True)["covered"]
        .mean()
        .reset_index(name="empirical_coverage")
    )
    return {
        "predictions": predictions_df,
        "calibration_curves": aggregated_calibration_df,
        "accuracy_metrics": accuracy_metrics_df,
        "calibration_metrics": calibration_metrics_df,
        "config": {
            "n_networks": int(n_networks),
            "train_fraction": float(train_fraction),
            "n_total": int(n_total),
            "n_train": int(n_train),
            "n_holdout": int(n_holdout),
            "posterior_num_samples": int(posterior_num_samples),
            "random_seed": int(random_seed),
            "mass_grid": mass_grid_arr.tolist(),
            "mode_grid_points": int(mode_grid_points),
            "mode_bandwidth_scale": float(mode_bandwidth_scale),
            "feature_normalization_source": "simulated_train_split",
            "feature_normalization_note": (
                "Heldout validation fits feature z-score normalization on each simulated training split; "
                "experimental normalization is not used for simulated holdout rows."
            ),
        },
    }


def save_internal_validation_outputs(
    *,
    output_dir: str | Path,
    validation_results: dict[str, pd.DataFrame | dict[str, Any]],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}
    config_path = output_dir / "internal_validation_metadata.json"
    config_path.write_text(json.dumps(_jsonable(validation_results.get("config", {})), indent=2), encoding="utf-8")
    saved_paths["internal_validation_metadata_json"] = config_path
    for key, filename in [
        ("predictions", "heldout_predictions.csv"),
        ("calibration_curves", "heldout_calibration_curves.csv"),
        ("accuracy_metrics", "heldout_accuracy_metrics.csv"),
        ("calibration_metrics", "heldout_calibration_metrics.csv"),
    ]:
        value = validation_results.get(key)
        if isinstance(value, pd.DataFrame):
            path = output_dir / filename
            value.to_csv(path, index=False)
            saved_paths[filename.replace(".csv", "")] = path
    return saved_paths


def plot_internal_validation_accuracy(
    validation_results: dict[str, pd.DataFrame | dict[str, Any]],
    *,
    parameter_bounds: pd.DataFrame | dict[str, Any],
    parameter_names: list[str],
    output_dir: str | Path | None = None,
    figure_dpi: int = 140,
    cell_width: float = 3.1,
    cell_height: float = 2.8,
    scatter_alpha: float = 0.18,
    scatter_size: float = 10.0,
    overlay_scatter_alpha: float = 0.08,
    overlay_scatter_size: float = 8.0,
    moving_average_window_fraction: float = 0.2,
    moving_average_min_points: int = 10,
    moving_average_stride_fraction: float = 0.25,
    moving_average_lw: float = 2.0,
    moving_average_alpha: float = 0.9,
    overlay_curve_alpha: float = 0.2,
    identity_color: str = "#7f7f7f",
    identity_lw: float = 1.5,
    identity_ls: str = "--",
    scatter_color: str = "#1f77b4",
    overlay_color: str = "#1f77b4",
    text_fontsize: float = 8.0,
) -> Path | None:
    predictions_df = validation_results["predictions"]
    metrics_df = validation_results["accuracy_metrics"]
    n_networks = int(validation_results["config"]["n_networks"])
    if isinstance(parameter_bounds, pd.DataFrame):
        bound_map = {
            str(row["parameter_name"]): (float(row["lower_bound"]), float(row["upper_bound"]))
            for _, row in parameter_bounds.iterrows()
        }
    else:
        bound_map = {str(k): _coerce_scalar_bound_pair(v, str(k)) for k, v in parameter_bounds.items()}
    n_rows = len(parameter_names)
    n_cols = n_networks + 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(cell_width * n_cols, cell_height * n_rows),
        dpi=figure_dpi,
        squeeze=False,
    )
    for row_idx, parameter_name in enumerate(parameter_names):
        low, high = bound_map[parameter_name]
        param_predictions = predictions_df.loc[predictions_df["parameter_name"] == parameter_name].copy()
        param_metrics = metrics_df.loc[metrics_df["parameter_name"] == parameter_name].copy()
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.plot([low, high], [low, high], color=identity_color, linestyle=identity_ls, linewidth=identity_lw)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.grid(alpha=0.16)
            if col_idx < n_networks:
                subset = param_predictions.loc[param_predictions["network_index"] == col_idx].copy()
                ax.scatter(
                    subset["ground_truth"].to_numpy(dtype=float),
                    subset["posterior_mode"].to_numpy(dtype=float),
                    color=scatter_color,
                    alpha=float(scatter_alpha),
                    s=float(scatter_size),
                    linewidths=0,
                )
                xs, ys = _rolling_mean_curve(
                    subset["ground_truth"].to_numpy(dtype=float),
                    subset["posterior_mode"].to_numpy(dtype=float),
                    window_fraction=moving_average_window_fraction,
                    min_points=moving_average_min_points,
                    stride_fraction=moving_average_stride_fraction,
                )
                if xs.size:
                    ax.plot(xs, ys, color=scatter_color, linewidth=moving_average_lw, alpha=moving_average_alpha)
                metric_row = param_metrics.loc[param_metrics["network_index"] == col_idx].iloc[0]
                text = (
                    f"r={metric_row['pearson_r']:.2f}\n"
                    f"R2={metric_row['r2']:.2f}\n"
                    f"nRMSE={metric_row['normalized_rmse']:.2f}\n"
                    f"rho={metric_row['spearman_r']:.2f}"
                )
                ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=text_fontsize)
                if row_idx == 0:
                    ax.set_title(f"Net {col_idx + 1}", fontsize=11)
            else:
                for network_index in range(n_networks):
                    subset = param_predictions.loc[param_predictions["network_index"] == network_index].copy()
                    ax.scatter(
                        subset["ground_truth"].to_numpy(dtype=float),
                        subset["posterior_mode"].to_numpy(dtype=float),
                        color=overlay_color,
                        alpha=float(overlay_scatter_alpha),
                        s=float(overlay_scatter_size),
                        linewidths=0,
                    )
                    xs, ys = _rolling_mean_curve(
                        subset["ground_truth"].to_numpy(dtype=float),
                        subset["posterior_mode"].to_numpy(dtype=float),
                        window_fraction=moving_average_window_fraction,
                        min_points=moving_average_min_points,
                        stride_fraction=moving_average_stride_fraction,
                    )
                    if xs.size:
                        ax.plot(xs, ys, color=overlay_color, linewidth=moving_average_lw, alpha=overlay_curve_alpha)
                text = (
                    f"r={param_metrics['pearson_r'].mean():.2f}±{param_metrics['pearson_r'].std(ddof=0):.2f}\n"
                    f"R2={param_metrics['r2'].mean():.2f}±{param_metrics['r2'].std(ddof=0):.2f}\n"
                    f"nRMSE={param_metrics['normalized_rmse'].mean():.2f}±{param_metrics['normalized_rmse'].std(ddof=0):.2f}\n"
                    f"rho={param_metrics['spearman_r'].mean():.2f}±{param_metrics['spearman_r'].std(ddof=0):.2f}"
                )
                ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=text_fontsize)
                if row_idx == 0:
                    ax.set_title("Overlay", fontsize=11)
            ax.set_xlabel(f"Ground truth\n{format_parameter_label(parameter_name)}", fontsize=9)
            ax.set_ylabel(f"Posterior mode\n{format_parameter_label(parameter_name)}", fontsize=9)
    fig.suptitle("Heldout accuracy validation", fontsize=13)
    fig.tight_layout()
    save_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "heldout_accuracy_validation.png"
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return save_path


def plot_internal_validation_calibration(
    validation_results: dict[str, pd.DataFrame | dict[str, Any]],
    *,
    parameter_names: list[str],
    output_dir: str | Path | None = None,
    figure_dpi: int = 140,
    cell_width: float = 3.1,
    cell_height: float = 2.8,
    curve_lw: float = 2.0,
    curve_alpha: float = 0.9,
    overlay_curve_alpha: float = 0.2,
    smoothing_window_points: int = 1,
    identity_color: str = "#7f7f7f",
    identity_lw: float = 1.5,
    identity_ls: str = "--",
    curve_color: str = "#d62728",
    overlay_color: str = "#d62728",
    text_fontsize: float = 8.0,
) -> Path | None:
    calibration_curves_df = validation_results["calibration_curves"]
    metrics_df = validation_results["calibration_metrics"]
    n_networks = int(validation_results["config"]["n_networks"])
    n_rows = len(parameter_names)
    n_cols = n_networks + 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(cell_width * n_cols, cell_height * n_rows),
        dpi=figure_dpi,
        squeeze=False,
    )
    for row_idx, parameter_name in enumerate(parameter_names):
        param_curves = calibration_curves_df.loc[calibration_curves_df["parameter_name"] == parameter_name].copy()
        param_metrics = metrics_df.loc[metrics_df["parameter_name"] == parameter_name].copy()
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.plot([0.0, 1.0], [0.0, 1.0], color=identity_color, linestyle=identity_ls, linewidth=identity_lw)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.16)
            if col_idx < n_networks:
                subset = param_curves.loc[param_curves["network_index"] == col_idx].copy().sort_values("mass")
                x = subset["mass"].to_numpy(dtype=float)
                y = subset["empirical_coverage"].to_numpy(dtype=float)
                y_smooth = _moving_average_1d(y, int(smoothing_window_points))
                ax.plot(x, y_smooth, color=curve_color, linewidth=curve_lw, alpha=curve_alpha)
                metric_row = param_metrics.loc[param_metrics["network_index"] == col_idx].iloc[0]
                ax.text(0.03, 0.97, f"ECE={100.0 * metric_row['ece']:.1f}%", transform=ax.transAxes, va="top", ha="left", fontsize=text_fontsize)
                if row_idx == 0:
                    ax.set_title(f"Net {col_idx + 1}", fontsize=11)
            else:
                for network_index in range(n_networks):
                    subset = param_curves.loc[param_curves["network_index"] == network_index].copy().sort_values("mass")
                    x = subset["mass"].to_numpy(dtype=float)
                    y = subset["empirical_coverage"].to_numpy(dtype=float)
                    y_smooth = _moving_average_1d(y, int(smoothing_window_points))
                    ax.plot(x, y_smooth, color=overlay_color, linewidth=curve_lw, alpha=overlay_curve_alpha)
                ax.text(
                    0.03,
                    0.97,
                    f"ECE={100.0 * param_metrics['ece'].mean():.1f}±{100.0 * param_metrics['ece'].std(ddof=0):.1f}%",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=text_fontsize,
                )
                if row_idx == 0:
                    ax.set_title("Overlay", fontsize=11)
            ax.set_xlabel("Posterior mass", fontsize=9)
            ax.set_ylabel(f"Coverage\n{format_parameter_label(parameter_name)}", fontsize=9)
    fig.suptitle("Heldout calibration validation", fontsize=13)
    fig.tight_layout()
    save_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "heldout_calibration_validation.png"
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return save_path
