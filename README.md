# Brain Oscillations Extend Beyond Task-Relevant Motor Neuron Pools and Contribute to Shaping the Functional State of the Motor System

Code for the simulation and simulation-based inference analyses accompanying the manuscript:

- bioRxiv: [TBA]

This repository focuses on two simulation components from the paper:

1. **Figure 3 / SBI pipeline**: a Brian2 leaky integrate-and-fire motor-neuron pool model, shared simulation/experimental feature extraction, simulation-based inference, posterior diagnostics, and posterior-predictive checks.
2. **Figure 4 / simplified synchrony model**: a standalone phenomenological gamma-renewal model used to test how baseline motor-neuron synchronization affects the rapid build-up of motor-neuron output after a sudden increase in excitatory input.

The code was mostly AI-generated, with close human supervision, inspection, and verification.

## Installation

Create the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate oscillatory_inputs_mn
python -m ipykernel install --user --name oscillatory_inputs_mn
```

The main dependencies are Brian2, NumPy, SciPy, pandas, matplotlib, h5py, PyTorch, and `sbi`.

## Repository Structure

Core modules:

- `simulator.py`: Brian2 motor-neuron pool simulator. Public entrypoints are `SimulationParameters` and `run_simulation(params)`.
- `synchrony_analysis.py`: sliding-window pairwise coincidence synchrony index and related firing-rate diagnostic plotting.
- `shared_run_analysis.py`: shared feature schema and feature-extraction utilities used by both simulated and experimental data.
- `parameter_set_batch_summary.py`: simulation batch loading, per-parameter-set summaries, rate/ISI summaries, and exported simulation feature tables.
- `experimental_hdf5_analysis.py`: experimental trial HDF5 loading, sidecar analysis, and exported experimental feature tables.
- `sim_exp_feature_coverage.py`: prior-predictive / simulation-vs-experiment feature coverage plots.
- `sbi_sim_exp_inference.py`: SBI dataset preparation, posterior ensemble training, posterior sampling, validation, and posterior diagnostics.
- `posterior_predictive_across_condition_summary.py`: posterior-predictive summaries across experimental conditions.
- `experimental_across_condition_summary.py`: experimental across-condition summaries.

Main notebooks:

- `run_simulation_batch.ipynb`: generate training or posterior-predictive simulation batches.
- `summarize_parameter_set_batch.ipynb`: summarize simulation batches into feature tables.
- `analyze_experimental_hdf5.ipynb`: analyze private per-trial experimental HDF5 exports.
- `compare_sim_vs_exp_feature_coverage.ipynb`: compare simulated and experimental feature coverage.
- `run_sbi_sim_exp.ipynb`: train/sample SBI posteriors and generate posterior diagnostics.
- `summarize_posterior_predictive_sim_across_conditions.ipynb`: summarize posterior-predictive simulations.
- `paper_simulation_figures.ipynb`: assemble manuscript-facing simulation figures.
- `toy_direct_synchrony_cst_model.ipynb`: simplified Figure 4 synchrony/CST model.

Legacy and non-paper workflows are kept in `archive/`.

## Data Flow

The Figure 3 workflow is:

1. Sample simulator parameters and run repeated 6-s Go/No-Go trial simulations with `run_simulation_batch.ipynb`.
2. Save one HDF5 output per simulated trial, including spike trains, input components, CST diagnostics, synchrony diagnostics, and firing-rate diagnostics.
3. Aggregate stochastic repeats into parameter-set-level feature summaries with `parameter_set_batch_summary.py`.
4. Analyze private experimental per-trial HDF5 exports with `experimental_hdf5_analysis.py`.
5. Export simulated and experimental feature tables with aligned `OBS_*` feature columns.
6. Check prior-predictive feature coverage.
7. Train neural posterior estimators with `sbi_sim_exp_inference.py`.
8. Apply trained posteriors to experimental observations.
9. Run posterior-predictive simulations and compare predicted features/traces with experimental summaries.

The Figure 4 workflow is contained in `toy_direct_synchrony_cst_model.ipynb`. It simulates spike trains from a latent common gamma-renewal process, manipulates copy probability to control synchronization, applies a matched step/no-step design, integrates the response-only CST, and estimates the maximum response slope.

## Feature Schema

The public interfaces and output conventions to preserve are:

- `SimulationParameters`
- `run_simulation(params)`
- HDF5 groups such as `/simulation_parameters`, `/spike_trains`, `/input_components`, `/cst_diagnostics`, `/sync_diagnostics`, and `/firing_rate_diagnostics`
- exported table prefixes:
  - `SIM_PARAM_*`
  - `OBS_*`
  - `ANALYSIS_PARAM_*`
  - `ANALYSIS_DIAGNOSIS_*`

The second-stage SBI analysis uses 49 observable features extracted from the low-frequency CST modulation, alpha modulation, beta modulation, and synchrony profiles. These include burst and trend features, Baseline/Ready/Post-Cue window summaries, and post-cue timing features.

## Data Availability

Private experimental HDF5 exports and generated simulation/SBI outputs are not included in this repository. The notebooks assume local generated outputs unless their path variables are edited.

