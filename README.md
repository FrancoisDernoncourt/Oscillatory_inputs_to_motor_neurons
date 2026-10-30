# Brain Oscillations Extend Beyond Task-Relevant Motor Neuron Pools and Contribute to Shaping the Functional State of the Motor System

Code for the simulations and the simulation-based inference (SBI, Figure 3) of the manuscript:

- bioRxiv: https://www.biorxiv.org/content/10.64898/2026.05.16.725172v2

The code exactly as used for the preprint is kept under the tag [`biorxiv-v2`](https://github.com/FrancoisDernoncourt/Oscillatory_inputs_to_motor_neurons/tree/biorxiv-v2). The default branch contains the revised version; the differences are listed in [Changes since the preprint](#changes-since-the-preprint).

The code was mostly AI-generated, with close human supervision, inspection, and verification.

## What the code does

A pool of 40 leaky integrate-and-fire motor neurons (Brian2) receives a tonic excitatory drive, common low-frequency (0–5 Hz), alpha (8–13 Hz) and beta (13–30 Hz) inputs, and independent noise. Cue-related changes of the net excitatory drive and of the alpha and beta inputs are modelled as a transient burst and a linear trend. The same feature extraction is applied to simulated and experimental motor-unit spike trains (low-pass cumulative spike train, alpha and beta modulation of the CST, synchronisation index), and SBI estimates which inputs explain the experimental features:

- **Stage 1** infers 8 baseline parameters from baseline-only simulations, using the grand mean of all participant-condition rows.
- **Stage 2** infers the 6 cue-related parameters (burst and trend of each input) for each condition, from Go/No-Go simulations whose baseline parameters are drawn from the stage-1 posterior.
- **Posterior-predictive checks** re-simulate parameter sets drawn from both posteriors and compare them with the experiment.

## Installation

Tested with Python 3.12 on Windows 10, CPU only.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name oscillatory_inputs_mn
```

With conda: `conda env create -f environment.yml` (installs the same pinned packages).

`requirements.txt` pins the exact versions the code was tested with. Two pins matter in particular: the saved trained networks load with `sbi==0.27.0` (newer `sbi` releases removed the `SNPE` class used here), and Brian2 generates the simulation code with `sympy`, so changing its version can change results in the last digits.

Brian2 can run the simulations with two backends, set by `brian_codegen_target`:
- `"numpy"` works everywhere.
- `"cython"` (used for the preprint's simulations) is several times faster but needs a C++ compiler (on Windows, the Microsoft C++ Build Tools).

Both backends give the same results up to rounding (about 1e-11 on the saved features).

## Data

The data are not stored in this repository: experimental exports, simulation batches, trained networks and figure inputs will be shared on data.gouv.fr [link to be added]. Place the dataset folder `DATA_sim_and_exp/` at the repository root, or change `DATA_ROOT` in [data_paths.py](data_paths.py); every notebook builds its paths from that file. On Windows, keep the repository at a short location (e.g. `C:\code\`) or enable long paths: some output files are nested deeply, and Windows otherwise limits paths to 260 characters.

What you need depends on what you want to do (names as in `data_paths.py`):

| Goal | Folders needed | Size |
| --- | --- | --- |
| Redraw Figure 3 from saved results | `EXP_ANALYZED_DIR` (analysed trials, experimental feature table, torque traces), `PPC_BATCH_DIRS`, `STAGE2_SBI_DIR`, `PPC_RESULTS_DIR`, `FIGURE_EXAMPLE_BATCH_DIR` | about 4 GB |
| Rerun the inference on the saved simulations | `EXP_ANALYZED_DIR`, the summary tables and priors files of `STAGE1_BATCH_DIR` and `STAGE2_BATCH_DIR`, `STAGE1_SBI_DIR`, `STAGE2_SBI_DIR` | about 0.5 GB |
| Rerun everything | `EXP_RAW_DIR` (exported trials with motor-unit spike trains and the export manifests), `EXP_TORQUE_CSV`; the priors files of each batch to reuse the exact parameter values | about 0.5 GB, plus days of computing |

The full simulation batches (about 33 GB) are only needed to recompute the simulated features without re-simulating. Each simulation folder holds the run's `sim_parameters.json` (all parameters and its random seed) and an HDF5 file with the inputs and the CST and synchrony traces at 100 Hz; spike trains were not saved.

## Pipeline

Run the notebooks from the repository root, in this order. Each notebook has a configuration cell at the top; the defaults reproduce the preprint's runs. Times are for the `"numpy"` backend on a desktop PC.

1. **Experimental features**: `analyze_experimental_hdf5.ipynb` analyses every exported trial (about 6 s per trial, 3 h in total) and writes the experimental feature table. The export manifests are merged automatically.
2. **Stage-1 simulations**: `run_simulation_batch.ipynb` with `BATCH_PRESET = "stage1_training"` (600 parameter sets × 16 runs). The notebook summarises the batch into a feature table at the end; `summarize_parameter_set_batch.ipynb` does the same for an existing batch.
3. **Optional check**: `compare_sim_vs_exp_feature_coverage.ipynb` shows whether the simulated features cover the experimental ones.
4. **Stage-1 inference**: `run_sbi_sim_exp.ipynb` with `SBI_STAGE = "stage1"`. With `RUN_MODE = "reload"` it reuses the preprint's network; with `"train"` it trains a new one.
5. **Stage-2 simulations**: `run_simulation_batch.ipynb` with `"stage2_training"` (1,000 sets × 32 runs, baseline parameters drawn from the stage-1 posterior). The preprint's training set also includes `"stage2_training_part2"` (315 sets), run into the same folder.
6. **Stage-2 inference**: `run_sbi_sim_exp.ipynb` with `SBI_STAGE = "stage2"`.
7. **Posterior-predictive checks**: `run_simulation_batch.ipynb` with `"ppc_TA_NOGO"`, `"ppc_HAND_NOGO"` and `"ppc_HAND_GO"` (50 sets × 32 runs each, plus the posterior mode), then `summarize_posterior_predictive_sim_across_conditions.ipynb`.
8. **Figures**: `run_simulation_batch.ipynb` with `"figure_example"` (the example simulation of Fig. 3A), then `paper_simulation_figures.ipynb`.

`summarize_experimental_across_conditions.ipynb` gives an overview of the experimental data and is not needed for the figures.

Simulations take about 25 s each, so the stage-2 batch (38,432 simulations) takes about a day on 10 parallel workers. The posterior-predictive presets reproduce the preprint's runs exactly, parameter values and random seeds included. The uniform-prior draws of the stage-1 and stage-2 training batches were not seeded in the preprint; their exact values are in the batches' priors files and `sim_parameters.json` files, and new batches are seeded (`PRIOR_SAMPLING_SEED`).

## Computing resources

`run_simulation_batch.ipynb` chooses the number of parallel simulation workers from the memory available when it starts (about 1 GB per worker, half of the free memory at most) and leaves two CPUs free; set `sim_parallel_cpus` yourself to use more of a dedicated machine. Too many workers for the free memory can make the computer unresponsive. Before a new batch, the notebook also stops joblib workers left running by an interrupted run (`KILL_ORPHANED_WORKERS`); other Python programs are not affected. The optional conditional-posterior explorer of `run_sbi_sim_exp.ipynb` is off by default and uses `conditional_num_workers` processes.

## Notes on the model and the analyses

These points describe what the code does where the preprint's text is incomplete or differs.

- **Time reference.** Time zero is the imperative (Go/No-Go) cue; the ready cue is at −1 s. Simulated trials last 10 s and the first and last 2 s are discarded, so the analysed window is −5 to +1 s relative to the imperative cue (−4 to +2 s relative to the ready cue). Experimental trials span −5 to +2 s.
- **Afterhyperpolarisation.** The AHP reversal potential equals the resting potential (0 mV in the model's units, where rest is 0 mV and the spike threshold 10 mV). An earlier −13.3 mV setting in the code was never applied to the simulations.
- **SBI networks.** Stage 1 used a single masked autoregressive flow (batch size 200, learning rate 5×10⁻⁴); stage 2 used an ensemble of five (batch size 128, learning rate 2×10⁻⁴) whose posterior samples are pooled. The held-out validation of each stage trains five networks on random 90/10 splits.
- **Stage-1 filter settings.** The Butterworth filter orders depend on the cutoff frequency. Stage-1 simulations used orders 5 (alpha) and 6 (beta) both to generate the inputs and to analyse the CST; stage 2 and the experiment used 8 and 10. Simulations from the same stage-1 posterior give the same baseline alpha and beta CST amplitudes with either setting (differences below 0.04 SD of the training features).
- **Burst and trend features.** The "smoothing spline" fitted to each profile over −3 to +1 s, excluding the burst window (0 to 0.5 s), uses the default smoothing factor (number of points × variance), with which the fit has no interior knots: it is a least-squares polynomial of degree 4.
- **Active motor units.** Units with a mean discharge rate of at least 3 Hz are kept; in simulations the rate is computed over the whole analysed window, in the experiment over −4 to −3 s.
- **Stage-2 training set.** 1,201 parameter sets × 32 simulations = 38,432 simulations.

## Changes since the preprint

Code cleanup, with unchanged results (checked by re-running saved paper simulations and comparing all outputs bit for bit):

- Removed components that were never active in the paper's simulations: Renshaw cells and recurrent inhibition, the muscle-force model, force-target baseline generation and its optimisation, multi-pool scaffolding, the Brian2 step-current experiments, and unused parameters (including the unused `voltage_AHP`).
- Removed the simplified synchrony model of Figure 4 (`toy_direct_synchrony_cst_model.ipynb`), which is no longer part of the manuscript.
- Fixed two notebooks whose analysis cells had been overwritten (`run_sbi_sim_exp.ipynb`, `compare_sim_vs_exp_feature_coverage.ipynb`), and made the saved parameter files of the paper's batches loadable again (they refer to the simulator's former module name).
- Added `data_paths.py`, pinned requirements, and notebook settings that reproduce each of the preprint's runs.

Analysis changes, which alter the stage-2 results:

- **Polynomial degree of the burst/trend fits.** Degree 4 for every feature. The preprint used degree 3 for the simulated features but degree 4 for the experimental ones.
- **Burst window excluded from the fit.** Exactly 0 to 0.5 s, as described in the manuscript; the preprint excluded −0.05 to 0.55 s.
- **Baseline window of the stage-2 features.** −2 to −1 s, as described in the manuscript and as in the EEG and CST analyses; the preprint used −3 to −2 s. Stage 1 is unchanged.
- **Experimental features.** Recomputed from the exported trials with the current code. Synchrony and ISI-CV now exclude units discharging below 3 Hz, as the CST already did; the preprint's experimental analysis still included them in these two measures. The current export has 10 (Hand_Go), 15 (Hand_NoGo) and 15 (DF_NoGo) participants.

Stage 2 was therefore retrained and the posterior-predictive checks re-simulated; stage 1 and the stage-2 training simulations are unchanged.

## Repository structure

Modules:

- `simulator.py`: Brian2 motor-neuron pool simulator. Public entry points: `SimulationParameters` and `run_simulation(params)`.
- `synchrony_analysis.py`: sliding-window pairwise coincidence synchrony index.
- `shared_run_analysis.py`: feature schema and feature-extraction utilities shared by simulated and experimental data.
- `parameter_set_batch_summary.py`: loading of simulation batches, per-parameter-set summaries and simulated feature tables.
- `experimental_hdf5_analysis.py`: loading and analysis of experimental trial exports, and experimental feature tables.
- `sim_exp_feature_coverage.py`: simulation-vs-experiment feature coverage plots.
- `sbi_sim_exp_inference.py`: SBI dataset preparation, network training, posterior sampling, validation and diagnostics.
- `posterior_predictive_across_condition_summary.py`, `experimental_across_condition_summary.py`: summaries across conditions.
- `data_paths.py`: location of the dataset.

Notebooks:

- `analyze_experimental_hdf5.ipynb`: experimental features.
- `run_simulation_batch.ipynb`: simulation batches (stage 1, stage 2, posterior-predictive checks, figure example).
- `summarize_parameter_set_batch.ipynb`: simulated feature tables.
- `compare_sim_vs_exp_feature_coverage.ipynb`: does the simulated feature range cover the experiment.
- `run_sbi_sim_exp.ipynb`: training and sampling of the posteriors (stage 1 or 2), validation, diagnostics.
- `summarize_posterior_predictive_sim_across_conditions.ipynb`: posterior-predictive checks.
- `summarize_experimental_across_conditions.ipynb`: overview of the experimental data.
- `paper_simulation_figures.ipynb`: Figure 3 panels.

## Feature schema

Saved simulation files contain HDF5 groups such as `/simulation_parameters`, `/spike_trains`, `/input_components`, `/cst_diagnostics`, `/sync_diagnostics` and `/firing_rate_diagnostics`. Exported tables use the column prefixes `SIM_PARAM_*`, `OBS_*`, `ANALYSIS_PARAM_*` and `ANALYSIS_DIAGNOSIS_*`. The stage-2 inference uses 49 features from the low-frequency CST, alpha modulation, beta modulation and synchrony profiles: burst and trend features, Baseline/Ready/Post-Cue window summaries, and post-cue timing features.

## Contributing

Notebooks are committed without outputs. After cloning, run `pip install nbstripout` and `nbstripout --install` once, so that outputs are stripped automatically at each commit.

## License

MIT (see [LICENSE](LICENSE)).
