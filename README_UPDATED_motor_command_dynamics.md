# Motor command dynamics sandbox — current project state

This repository is a focused working copy for **single-pool motor neuron simulations**, **experimental trial analysis**, and **simulation-based inference (SBI)** of baseline and cue-related motor command features.

The current active workflow is **not** the older recurrent-inhibition / Renshaw-cell branch. The active path is now:

- **single motor-neuron pool**
- **no recurrent inhibition in the active simulation path**
- **optional force model** (available but not central for the current fitting workflow)
- **task-like cue structure** (baseline → get-ready → go/nogo cue → optional burst/trend modulation)
- **matched simulation/experimental analysis pipeline**
- **SBI used in two passes**:
  1. baseline fit
  2. cue-related fit

This README is meant as a practical re-entry guide for coding work in this repository.

---

## 1. Scientific goal

The project aims to infer which latent input processes to a motor-neuron pool can explain experimental observations extracted from TA motor-unit data during a Go/No-Go task.

The central strategy is:

1. simulate one motor-neuron pool with configurable common input structure
2. extract the same observable features from simulation and experiment
3. fit baseline parameters first
4. fit cue-related parameters second
5. inspect posterior geometry / degeneracy / compensation structure

The key observed feature families are:

- baseline firing-rate moments
- baseline ISI CV moments
- baseline CST low-frequency / alpha / beta metrics
- cue-related modulation of CST low-frequency activity
- cue-related modulation of CST alpha and beta envelopes
- cue-related synchrony trace and summary features

Force is still supported in the simulator, but for the current experimental-comparison workflow it is not the primary target.

---

## 2. Current model summary

### 2.1 Active simulator architecture

Current active simulator:
- **single motor-neuron pool**
- **no recurrent inhibition / no active Renshaw-cell path**
- motor-neuron intrinsic properties derived from soma size
- optional post hoc force model
- explicit low-frequency / alpha / beta common-input components
- optional cue-related burst / trend modulation
- built-in CST / synchrony / firing-rate diagnostics

### 2.2 Input model

The common input is built from three components:

1. **low-frequency component**
   - band-limited Gaussian process
   - configurable frequency band
   - configurable final amplitude / SD

2. **alpha component**
   - band-limited carrier
   - slow positive envelope
   - final component amplitude matched after envelope modulation
   - configurable alpha carrier band and envelope band

3. **beta component**
   - same logic as alpha
   - configurable beta carrier band and envelope band

Cue-related effects can modify:
- low-frequency component directly
- alpha envelope
- beta envelope

This makes it possible to represent both:
- slow cue-related drifts
- cue-centered transient increases / decreases in alpha / beta expression

### 2.3 Motor-neuron heterogeneity

The simulator now includes richer heterogeneity than earlier versions.

Current relevant ideas already implemented or explored:
- size distribution is no longer restricted to a Gaussian-like mean/sd parameterization
- intrinsic properties can inherit asymmetry/skew through the size distribution
- common-input weights can vary across neurons
- independent noise amplitude can be heterogeneous across neurons
- these changes were introduced mainly to better reproduce:
  - skewed firing-rate distributions
  - broader / more skewed ISI CV distributions

---

## 3. Task timing convention

The simulation uses a task-like timing structure.

Main public timing parameters include:
- `duration`
- `edges_ignore_duration`
- `get_ready_cue_time_s`
- `go_nogo_cue_delay_s`
- `burst_delay_after_go_nogo_cue_s`

Interpretation:
- usable baseline runs from usable start to the get-ready cue
- get-ready cue occurs first
- go/nogo cue occurs after `go_nogo_cue_delay_s`
- cue-related burst center occurs after the go/nogo cue

These event times are saved in the output and are used downstream for:
- baseline window definition
- cue-centered analysis
- burst-window feature extraction
- synchrony and CST summaries

---

## 4. Current simulation outputs

Main per-trial simulation output folder structure:

```text
<batch_root>/
    parameter_set_0000/
        simulation_output_0/
            sim_parameters.json
            simulation_output.h5
        simulation_output_1/
            ...
    parameter_set_0001/
        ...
```

Each `simulation_output_*` folder corresponds to one simulation repeat / one trial.

### 4.1 Important files per simulation
- `sim_parameters.json`
- `simulation_output.h5`

### 4.2 Important HDF5 groups
Current key groups in `simulation_output.h5` include:

- `/simulation_parameters`
- `/motoneurons_and_pools_indices`
- `/motoneurons_properties`
- `/spike_trains`
- `/input`
- `/driving_inputs`
- `/input_components`
- `/cst_diagnostics`
- `/sync_diagnostics`
- `/firing_rate_diagnostics`
- `/forces` (if force enabled)

### 4.3 Important CST diagnostic datasets
The CST analysis currently saves things such as:

- `cst_normalized`
- `cst_lf`
- `cst_alpha`
- `cst_alpha_envelope`
- `cst_alpha_envelope_z`
- `cst_alpha_envelope_pct`
- `cst_beta`
- `cst_beta_envelope`
- `cst_beta_envelope_z`
- `cst_beta_envelope_pct`

Important convention:
- CST is active-unit normalized
- alpha and beta traces are band-pass filtered CSTs
- alpha/beta envelopes are Hilbert envelopes
- baseline-relative z-score / percent modulation are computed relative to the pre-get-ready baseline

### 4.4 Synchrony diagnostics
Synchrony analysis is now implemented in the simulation-side workflow.

It is a sliding-window coincidence-based `sync_index`-style metric:
- default legacy-compatible direction mode
- optional symmetric direction mode
- analytic expected coincidence or circular-shift surrogate expectation
- cue-centered trace and summary features are saved

Key outputs include:
- `sync_trace`
- `sync_trace_zscore`
- `baseline_mean`
- `baseline_std`
- `post_peak`
- `post_peak_time_sec`
- `post_peak_width_sec`
- `post_integral_above_baseline`
- `sync_peak_rel_base`
- `sync_mean_rel_base`

---

## 5. Experimental data workflow

Experimental data are now exported from MATLAB into **Python-friendly per-trial HDF5 files**.

### 5.1 Export philosophy

MATLAB is used only to:
- read the reorganized MUedit + Trial_info + mask files
- reconstruct trials cleanly
- export one HDF5 per trial
- write a manifest

Python is then responsible for:
- CST analysis
- synchrony analysis
- firing-rate analysis
- feature extraction
- summary tables

### 5.2 Experimental HDF5 hierarchy

Export root:

```text
EXP_DATA_HDF5/
```

Hierarchy:

```text
EXP_DATA_HDF5/
    P01/
        ses-S001/
            TA_NOGO/
                P01_TA_NOGO_run_001_trial_001.h5
                ...
            HAND_NOGO/
                ...
            HAND_GO/
                ...
```

### 5.3 Experimental HDF5 content

Each per-trial experimental HDF5 contains:
- root provenance attributes
- `/trial_metadata`
- `/events`
- `/unit_metadata`
- `/motoneurons_and_pools_indices`
- `/spike_trains/MN/MN_####`

Important convention:
- spike times are in **seconds relative to READY**
- READY = `0 s`
- cue = `+1 s` relative to READY
- only units that fired in that trial are exported
- units are pooled into one flat TA pool
- grid provenance is stored only as metadata

### 5.4 Manifest files

The MATLAB export writes manifest files at the root of `EXP_DATA_HDF5`, e.g.:

- `experimental_trial_export_manifest_*.csv`
- `experimental_trial_export_manifest_*.mat`

These should be used for:
- file discovery
- QC
- skipped / excluded trial tracking

---

## 6. Current analysis architecture

The project now has a clearer separation between:

1. **per-trial raw-ish data**
2. **per-trial derived analyses**
3. **per-parameter-set summaries**
4. **posterior / SBI analysis**
5. **posterior-geometry / degeneracy exploration**

### 6.1 On simulation trials
Python computes:
- CST diagnostics
- synchrony diagnostics
- firing-rate diagnostics
- optional force diagnostics

### 6.2 On experimental trials
Python is intended to compute the same derived groups as for simulation:
- `/cst_diagnostics`
- `/sync_diagnostics`
- `/firing_rate_diagnostics`

The long-term goal is that simulation and experiment become aligned at the **derived-analysis level**, not at the raw-simulator-input level.

---

## 7. SBI workflow

The project currently uses a **two-stage SBI strategy**.

### 7.1 Pass 1: baseline inference
Infer baseline-related parameters such as:
- baseline input intensity
- independent noise amplitude
- common/independent input heterogeneity
- motor-neuron population shape / size-distribution parameters

Main targets:
- firing-rate moments
- ISI CV moments
- baseline CST summaries
- baseline synchrony summaries

### 7.2 Pass 2: cue-related inference
Given baseline-calibrated settings, infer cue-related parameters such as:
- low-frequency trends / burst amplitudes
- alpha cue-related modulation
- beta cue-related modulation
- envelope-related burst parameters

Main targets:
- cue-centered CST LF / alpha / beta features
- cue-centered synchrony features

### 7.3 Current conceptual issues
Current posterior-analysis work focuses on:
- degeneracy
- parameter tradeoffs
- posterior geometry
- weakly identified parameters
- effects of wide priors and prior-predictive coverage

---

## 8. Posterior geometry / degeneracy tools

This repository now also supports exploratory posterior-geometry analysis.

Current ideas/tools in active discussion or implementation include:
- posterior pairplots
- conditional posterior slices
- conditional pairplots after fixing selected parameters
- PCA of posterior samples for a given observation
- feature × parameter scatter-grid from training data
- relevance highlighting of training simulations close to experimental observations in feature space

Important interpretation:
- these tools are mainly for understanding **inverse geometry** and **parameter compensation**
- they are not direct forward Jacobian estimates of the simulator

---

## 9. Files / modules to inspect first

The exact filenames may evolve, but the following are the conceptual core:

### Simulation generation
- main simulator module
- main batch-launch notebook

### Batch summary / feature extraction
- parameter-set summary module
- exported-summary visualization notebook/module

### Experimental trial processing
- MATLAB export script (already used)
- Python experimental HDF5 reader/analyzer (current or upcoming)

### Posterior / SBI analysis
- SBI training/inference notebook or script
- posterior plotting / degeneracy utilities

If re-entering the project, first identify:
1. the active simulator file
2. the active batch launcher
3. the active exported-summary analysis path
4. the active SBI / posterior analysis notebook or module

---

## 10. Practical current mental model

The current end-to-end pipeline is:

### A. Simulations
1. generate parameter sets
2. run repeated simulations
3. save per-trial simulation HDF5 files
4. compute CST / synchrony / firing-rate diagnostics
5. aggregate per-parameter-set summary features

### B. Experiment
1. export trials from MATLAB to per-trial experimental HDF5
2. read those HDF5 files in Python
3. compute the same derived diagnostics as for simulation
4. aggregate subject × condition summary features

### C. Inference
1. compare simulation and experimental feature spaces
2. train SBI models
3. inspect posteriors
4. run posterior predictive checks
5. inspect degeneracy / conditional geometry / local structure

---

## 11. Coding guidance for future work

When adding new code, please try to preserve the following principles:

- do not revive the old recurrent-inhibition active path unless explicitly intended
- keep simulation and experiment aligned at the **analysis output** level
- prefer adding new diagnostics as separate groups/functions rather than mixing raw and derived data
- preserve backward-compatible plotting / analysis styles where useful
- keep notebook-facing APIs simple
- prefer explicit metadata and manifests for anything batch-related
- when exploring posterior geometry, distinguish:
  - global prior-predictive structure
  - local posterior-relevant structure
  - conditional posterior slices

---

## 12. Main current takeaway

This repository is now best understood as a **motor command dynamics sandbox** with:

- a single-pool motor-neuron simulator
- explicit low-frequency / alpha / beta input structure
- matched simulation + experiment analysis pipeline
- SBI-based fitting
- growing tools for posterior degeneracy / geometry exploration

The most important bridge in the project is now:
**simulation trial HDF5 ↔ experimental trial HDF5 ↔ common Python analysis ↔ common feature space ↔ SBI / posterior analysis**
