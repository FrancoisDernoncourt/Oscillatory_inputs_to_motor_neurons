"""Locations of the shared dataset used by the notebooks.

The data are not part of this repository (see README, "Data availability"). Place the dataset folder
next to the code (or change DATA_ROOT) and run the notebooks from the repository root. Every notebook
builds its input and output paths from the names below, so renaming or moving the dataset only
requires editing this file.
"""
from pathlib import Path

DATA_ROOT = Path("DATA_sim_and_exp")

CONDITIONS = ("TA_NOGO", "HAND_NOGO", "HAND_GO")  # TA_NOGO is called DF_NoGo in the paper

# Experimental data (exported by MATLAB, then analysed by analyze_experimental_hdf5.ipynb)
EXP_RAW_DIR = DATA_ROOT / "EXP_DATA_HDF5"
EXP_ANALYZED_DIR = DATA_ROOT / "EXP_DATA_ANALYZED_HDF5"
EXP_FEATURE_TABLE = EXP_ANALYZED_DIR / "_batch_summaries" / "experimental_batch_feature_summary.pkl"
EXP_TORQUE_CSV = EXP_ANALYZED_DIR / "TORQUE_experimental_data.csv"

# Simulation batches (run_simulation_batch.ipynb) and their parameter-set summary tables
STAGE1_BATCH_DIR = DATA_ROOT / "$_sim_batches_BASELINE_priors_V3"
STAGE2_BATCH_DIR = DATA_ROOT / "$_sim_batches_GONOGO_V3"
PPC_BATCH_DIRS = {
    condition: DATA_ROOT / f"$$$_sim_batches_posterior_predictive_checks_{condition}_sbi_with_all_features"
    for condition in CONDITIONS
}
FIGURE_EXAMPLE_BATCH_DIR = DATA_ROOT / "sim_example_for_figure"
SUMMARY_SUBDIR = "parameter_set_summaries"


def summary_table(batch_dir):
    """Parameter-set feature table written by summarize_parameter_set_batch.ipynb for a batch folder."""
    return Path(batch_dir) / SUMMARY_SUBDIR / "parameter_set_feature_summary.pkl"


def priors_pickle(batch_dir, batch_index=0):
    """Parameter sets drawn for a batch, saved by run_simulation_batch.ipynb."""
    batch_dir = Path(batch_dir)
    return batch_dir / f"{batch_dir.name}_priors_batch{batch_index}.pkl"


# Simulation-based inference outputs (run_sbi_sim_exp.ipynb)
STAGE1_SBI_DIR = DATA_ROOT / "$_sbi_output_BASELINE_grand_mean__run_0004"
STAGE2_SBI_ROOT = DATA_ROOT / "$_sbi_output_GONOGO_V3"
STAGE2_SBI_DIR = STAGE2_SBI_ROOT / "SBI_RUN_normalize_to_exp_FEATURES_all_windows"

# Posterior-predictive summaries and paper figures
PPC_RESULTS_DIR = DATA_ROOT / "$$$_posterior_predictive_checks_results_GO-NOGO_sbi_with_all_features"
COVERAGE_OUTPUT_DIR = DATA_ROOT / "sim_vs_exp_feature_coverage"
PAPER_FIGURES_DIR = Path("paper_figures")
