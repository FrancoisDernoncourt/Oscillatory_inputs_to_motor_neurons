import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import os
import sys
import json
import pandas as pd
from pathlib import Path
from brian2 import *
from scipy.signal import windows, butter, filtfilt, sosfiltfilt, hilbert, resample_poly
from scipy.special import ndtr, ndtri
from dataclasses import dataclass, field, asdict
from typing import Dict
from scipy.signal import welch
from scipy.optimize import minimize_scalar
import time
import Cython
import h5py
from h5py import string_dtype
from typing import List
import traceback
from threading import local
import logging
import warnings
from math import sqrt
from types import SimpleNamespace
from shared_run_analysis import (
    DEFAULT_MODULATION_SPECTRUM_CONFIG,
    DEFAULT_OBSERVATION_WINDOW_DEFS_REL_CUE_S,
    classify_active_units_by_rate,
    compute_observation_features_from_cst_sync,
    resolve_modulation_spectrum_config,
)

# # # IGNORE WARNINGS # # # 
# # # Ignore warnings from Brian2 which do not affect simulation behavior
# Silence the “TimedArray uses dt … not aligned” warning
logging.getLogger('brian2.input.timedarray').setLevel(logging.ERROR)
# Silence the “internal variable … exists in the namespace” warning
logging.getLogger('brian2.groups.group').setLevel(logging.ERROR)
# # #
# ignore only the “not compatible with tight_layout” UserWarning
warnings.filterwarnings(
    "ignore",
    message=r".*not compatible with tight_layout.*",
    category=UserWarning,
)


# # # DATA CLASS FOR SIMULATION's PARAMETERS
@dataclass
class SimulationParameters:
    # # # OUTPUT PARAMETERS
    output_folder_name: str = "simulation_batch_"
    make_unique_output_folder: bool = True
    output_plots: bool = True
    output_only_summary_input_cst_figure: bool = False
    output_plot_summary_input_cst_figure: bool = True
    output_plot_input_component_figures: bool = False
    output_plot_sync_diagnostic_figure: bool = False
    output_plot_firing_rate_figure: bool = False
    output_plot_spike_trains_figure: bool = False
    output_plot_mn_size_figure: bool = False
    output_plot_mn_properties_figure: bool = False
    output_plot_target_baseline_figure: bool = False
    output_plot_common_vs_independent_power_figure: bool = False
    output_plot_common_input_weight_figure: bool = False
    output_plot_independent_input_weight_figure: bool = False
    output_plot_force_tracking_figure: bool = False
    output_plot_force_properties_figures: bool = False
    output_plot_force_optimization_figures: bool = False
    output_plot_step_response_diagnostic_figure: bool = False
    enable_go_nogo_cue_diagnostics: bool = True
    save_common_input_power_csv: bool = False
    save_independent_input_power_csv: bool = False
    minimal_output: bool = False
    minimal_output_turn_off_figure_output: bool = True
    minimal_output_save_spike_trains: bool = False
    minimal_output_downsample_saved_traces: bool = True
    minimal_output_target_fsamp: int = 100
    brian_codegen_target: str = "numpy" # Brian2 runtime backend: "numpy" is slower but robust; "cython" is faster but depends on a working local C/Cython toolchain
    parameter_set_id: int = -1
    repeat_index: int = 0
    paired_repeat_seed: int | None = None
    step_index: int = -1
    is_zero_step_reference: bool = False
    parameter_set_folder_name: str = ""
    repeat_folder_name: str = ""
    step_folder_name: str = ""
    paired_reference_folder_name: str = ""

    # # # RANDOM SEED
    pre_specify_random_seed: bool = True
    requested_random_seed: int | None = None
    random_seed: int = field(init=False)

    # # # TIME PARAMETERS
    fsamp: int = 2000 # 2048 # in samples per second # 2000 matches experimental data from Novecento # 2048 from Quattrocento 
    # note that this is used to determine the time bins for input and output, but the actual integration time step is determined by 'defaultclock.dt' from Brian2 (0.1ms by default)
    duration: float = 6 # in seconds
    duration_with_ignored_window: float = field(init=False) # in seconds
    edges_ignore_duration: float = 1 # in seconds
    get_ready_cue_time_s: float = 3.0 # in seconds, relative to the start of the usable (non-ignored) simulation window
    go_nogo_cue_delay_s: float = 1.0 # in seconds, delay between the get-ready cue and the go/no-go cue
    burst_delay_after_go_nogo_cue_s: float = 0.3 # in seconds, delay between the go/no-go cue and the burst center
    task_event_times_s: Dict[str, float] = field(init=False)

    # # # NEURON NUMBERS AND SIZE
    nb_motoneurons: int = 100
    min_soma_diameter: float = 50 # 50 # in micrometers, for smallest motor neuron # Manuel et al. 2019 "Scaling of motor output, from Mouse to Humans"
    max_soma_diameter: float = 100 # 100 # in micrometers, for largest motor neuron # Manuel et al. 2019 "Scaling of motor output, from Mouse to Humans"
    tau_size_mode: str = "tau_size_ratio" # "tau_size_ratio" or "tau_size_um"
    tau_size_ratio: float = 0.2 # dimensionless truncated-exponential scale relative to (max_soma_diameter - min_soma_diameter)
    tau_size_um: float = 10.0 # absolute truncated-exponential scale in micrometers when tau_size_mode == "tau_size_um"
    # To calculate at initialization
    total_nb_motoneurons: int = field(init=False)

    # # # NEURON THRESHOLD AND EQUILIBRIUM POTENTIALS
    voltage_rest: Quantity = field(default_factory=lambda: 0 * mvolt) # arbitrary; 0 at rest
    voltage_thresh: Quantity = field(default_factory=lambda: 10 * mvolt) # arbitrary; 10 for generating a spike
    voltage_AHP: Quantity = field(default_factory=lambda: -13.3 * mvolt) # resting potential is typically -70mvolt, spike threshold is typically -55mvolt, and potassium reversal potential is typically -90mvolt.
    # To keep the relationships the same despite the arbitrary 0mvolt (resting voltage) and 10mvolt (spike generation threshold), voltage_AHP is set to -13.3 mvolt

    # # # COMMON INPUT PARAMETERS
    excitatory_input_baseline: float = 25*1e3 # in nA (nanoAmperes)
    lf_target_sd: float = 0.5*1e3 # in nA
    lf_band_hz: List[float] = field(default_factory=lambda: [0.0, 5.0]) # [low, high]
    alpha_target_sd_final: float = 0.0 # in nA
    alpha_band_hz: List[float] = field(default_factory=lambda: [8.0, 13.0]) # [low, high]
    alpha_envelope_band_hz: List[float] = field(default_factory=lambda: [1.0, 5.0]) # [low, high]
    beta_target_sd_final: float = 0.0 # in nA
    beta_band_hz: List[float] = field(default_factory=lambda: [13.0, 30.0]) # [low, high]
    beta_envelope_band_hz: List[float] = field(default_factory=lambda: [1.0, 5.0]) # [low, high]
    alpha_beta_envelope_variability_scale: float = 1.0 # 0 = stationary amplitude envelope, 1 = full envelope variability
    common_input_weight_sd: float = 0.0 # target SD of the per-MU common-input modulation weights (mean constrained to 1)
    common_input_weight_min: float = 0.0 # hard lower bound for common-input modulation weights
    common_input_weight_max: float = 5.0 # hard upper bound for common-input modulation weights
    independent_input_weight_sd: float = 0.0 # target SD of the per-MU independent-input scaling weights (mean constrained to 1)
    independent_input_weight_max: float = 10.0 # hard upper bound for independent-input scaling weights
    input_component_diag_zoom_duration_s: float = 3.0
    input_component_colors: Dict[str, str] = field(default_factory=lambda: {
        "lf": "#ff3300",
        "alpha": "#ffa600",
        "beta": "#008cff",
        "common": "#00BB3E",
        "cst": "#000000",
        "sync": "#7e2179",
        "envelope": "#444444",
        "trend": "#8c8c8c",
        "burst": "#FF0080",
        "cue": "#7a7a7a",
    })
    max_frequency_of_any_input: float = 80 # This can be necessary when setting the frequency content through another script
    enable_input_burst: bool = False # If True, add Gaussian burst kernels in nA to one or more component envelopes
    input_burst_center_ms: float = field(init=False) # in ms, center of the Gaussian burst envelope, derived from task timing
    input_burst_sigma_ms: float = 250.0 # in ms, standard deviation of the Gaussian burst envelope
    # LF is already a modulation-like drive component, so negative LF bursts are
    # allowed and interpreted as troughs in that slow modulation. Alpha/beta are
    # different: their burst kernels modulate non-negative oscillatory envelopes.
    lf_burst_peak_nA: float = 0.0 # additive Gaussian-kernel peak applied to the LF physical envelope, in nA
    alpha_burst_peak_nA: float = 0.0 # additive Gaussian-kernel peak applied to the alpha physical envelope, in nA
    beta_burst_peak_nA: float = 0.0 # additive Gaussian-kernel peak applied to the beta physical envelope, in nA
    lf_trend_start_rel_go_cue_s: float = -1.0 # trend onset relative to the go/no-go cue; before this time the LF trend is exactly zero
    alpha_trend_start_rel_go_cue_s: float = -1.0 # trend onset relative to the go/no-go cue; before this time the alpha-envelope trend is exactly zero
    beta_trend_start_rel_go_cue_s: float = -1.0 # trend onset relative to the go/no-go cue; before this time the beta-envelope trend is exactly zero
    lf_trend_slope_nA_per_s: float = 0.0 # additive LF trend slope in nA/s after trend onset
    alpha_trend_slope_nA_per_s: float = 0.0 # additive alpha-envelope trend slope in nA/s after trend onset
    beta_trend_slope_nA_per_s: float = 0.0 # additive beta-envelope trend slope in nA/s after trend onset
    input_burst_start_marker_n_sigma: float = 2.0 # burst start marker = center - n_sigma * sigma
    input_burst_zoom_start_n_sigma: float = 4.0 # zoom window start = center - n_sigma * sigma when plotting burst-centered outputs

    # # # STEP-RESPONSE EXPERIMENT
    enable_step_current: bool = False
    step_current_time_s: float = 3.0 # relative to the usable window start
    step_current_duration_s: float | None = None # None means the step persists until simulation end
    step_current_amplitude_nA: float = 0.0
    step_current_amplitude_percent_baseline: float | None = None
    step_current_mode: str = "absolute_nA" # "absolute_nA" or "percent_baseline"
    step_current_ramp_time_s: float = 0.0
    step_diagnostic_display_window_rel_s: List[float] = field(default_factory=lambda: [-1.5, 1.5])
    enable_step_sync_index_analysis: bool = True
    step_sync_analysis_window_rel_s: List[float] = field(default_factory=lambda: [-3.0, 3.0])
    step_response_cst_smoothing_method: str = "causal_exponential"
    step_response_cst_smoothing_tau_s: float = 0.020
    step_response_cst_hann_half_width_s: float = 0.080
    step_pre_window_rel_s: List[float] = field(default_factory=lambda: [-1.0, -0.2])
    step_late_post_window_rel_s: List[float] = field(default_factory=lambda: [0.6, 1.2])
    step_slope_candidate_min_s: float = 0.050
    step_slope_candidate_max_s: float = 0.300
    step_slope_candidate_step_s: float = 0.010
    step_slope_r2_min: float = 0.90
    step_slope_response_fraction_min: float = 0.25
    response_shape_fit_window_rel_s: List[float] = field(default_factory=lambda: [0.0, 0.5])
    response_shape_tau_min_ms: float = 1.0
    response_shape_tau_max_ms: float = 5000.0
    response_shape_min_r_inf: float = 1e-9
    xcorr_lag_window_rel_s: List[float] = field(default_factory=lambda: [0.0, 1.0])
    xcorr_max_lag_s: float = 0.25
    sync_predictor_pre_step_window_rel_s: List[float] = field(default_factory=lambda: [-0.250, 0.000])
    sync_predictor_local_step_window_rel_s: List[float] = field(default_factory=lambda: [-0.250, 0.250])
    step_diagnostic_plot_style: Dict[str, object] = field(default_factory=lambda: {
        "figsize": [12.0, 5.6],
        "dpi": 140,
        "xlim_rel_s": None,
        "input_ylim_uA": None,
        "cst_ylim": None,
        "show_cst_lf": False,
        "show_cst_hann": False,
        "show_cst_causal": True,
        "show_matched_no_step": True,
        "show_step_effect": True,
        "show_slope_fit": True,
        "show_response_shape_fit": True,
        "show_response_shape_fit_window": True,
        "show_xcorr_window": True,
        "show_sync_predictor_windows": True,
        "show_step_current_trace": False,
        "show_step_onset_line": True,
        "show_sync_subplot": True,
        "slope_fit_overlay_mode": "cst", # "cst" overlays on the step CST scale; "effect" overlays on the effect trace
        "total_input_color": "#16803a",
        "baseline_input_color": "#6f6f6f",
        "step_current_color": "#111111",
        "cst_color": "#ff3300",
        "cst_lf_color": "#ff3300",
        "cst_causal_color": "#ff3300",
        "matched_no_step_color": "#777777",
        "step_effect_color": "#111111",
        "slope_fit_color": "#0047ab",
        "response_shape_fit_color": "#008c72",
        "response_shape_fit_window_color": "#008c72",
        "xcorr_window_color": "#7b3294",
        "sync_predictor_pre_window_color": "#2ca25f",
        "sync_predictor_local_window_color": "#756bb1",
        "sync_trace_color": "#852680",
        "matched_no_step_sync_color": "#777777",
        "sync_effect_color": "#111111",
        "cst_hann_color": "#111111",
        "step_line_color": "#111111",
        "pre_window_color": "#37a055",
        "late_post_window_color": "#ff3300",
        "total_input_lw": 1.5,
        "baseline_input_lw": 1.2,
        "step_current_lw": 1.0,
        "cst_lw": 1.5,
        "cst_lf_lw": 1.0,
        "cst_causal_lw": 1.7,
        "matched_no_step_lw": 1.1,
        "step_effect_lw": 1.4,
        "slope_fit_lw": 1.5,
        "response_shape_fit_lw": 1.6,
        "sync_trace_lw": 1.4,
        "matched_no_step_sync_lw": 1.1,
        "sync_effect_lw": 1.2,
        "cst_hann_lw": 1.3,
        "step_line_lw": 1.1,
        "baseline_input_ls": "-",
        "step_current_ls": ":",
        "cst_lf_ls": "-",
        "matched_no_step_ls": "--",
        "step_effect_ls": "-",
        "slope_fit_ls": "-.",
        "response_shape_fit_ls": "--",
        "matched_no_step_sync_ls": "--",
        "sync_effect_ls": "-",
        "cst_hann_ls": "--",
        "step_line_ls": "--",
        "baseline_input_alpha": 0.95,
        "total_input_alpha": 1.0,
        "step_current_alpha": 1.0,
        "cst_alpha": 1.0,
        "cst_lf_alpha": 0.5,
        "cst_causal_alpha": 1.0,
        "matched_no_step_alpha": 0.85,
        "step_effect_alpha": 0.95,
        "slope_fit_alpha": 0.95,
        "response_shape_fit_alpha": 0.95,
        "sync_trace_alpha": 1.0,
        "matched_no_step_sync_alpha": 0.45,
        "sync_effect_alpha": 0.95,
        "cst_hann_alpha": 0.9,
        "pre_window_alpha": 0.12,
        "late_post_window_alpha": 0.10,
        "response_shape_fit_window_alpha": 0.08,
        "xcorr_window_alpha": 0.06,
        "sync_predictor_pre_window_alpha": 0.12,
        "sync_predictor_local_window_alpha": 0.08,
        "grid_alpha": 0.25,
        "legend_loc": "lower right",
        "annotation_fontsize": 8.3,
        "sync_annotation_fontsize": 7.3,
        "sync_annotation_loc": "upper left",
        "title": "Step-response diagnostic",
    })
    spike_train_show_task_cues_when_no_burst: bool = True
    spike_train_show_step_current_marker: bool = True
    enable_cst_hann_smoothing: bool = False
    cst_hann_smoothing_window_s: float = 0.4
    
    # ──── FREQUENCY FILTERING "MASTER PARAMETERS" ────
    scale_filter_order_to_frequency: bool = True
    filter_order_scaling_coeff: float = 0.5 # Multiplying the cutoff frequency to determine filter order. Used only if scale_filter_order_to_frequency == True
    default_freq_filter_order: int = 5 # Used if scale_filter_order_to_frequency = False
    lowest_freq_filter_order: int = 5 # # Used if scale_filter_order_to_frequency = True
    max_freq_filter_order: int = 100
    # ──── INPUTS DISTRIBUTION AND CORRELATION ACROSS POOLS

    # # # INDEPENDENT (NOISY) INPUT PARAMETERS
    low_pass_filter_of_MN_independent_input: float = 80 # in Hz
    independent_input_absolute_or_ratio: str = 'ratio' # 'ratio' # 'absolute' or 'ratio'
    independent_input_power: float = 3*np.sqrt(2) # 7.0*1e3
    # If independent_input_absolute_or_ratio == 'ratio', then independent_input_power is interpreted
    # relative to the total no-burst common-input power delivered to the pool.
    # If independent_input_absolute_or_ratio == 'absolute', then independent_input_power is the absolute value of the independent input std (in nA)

    # # # QUICK CST DIAGNOSTICS
    cst_analysis_window_start_s: float = 0.0 # relative to the usable (non-ignored) window
    cst_analysis_window_end_s: float = -1.0 # <= 0 means use the end of the usable window
    active_unit_rate_threshold_hz: float = 3.0
    cst_diag_zoom_duration_s: float = 3.0
    cst_modulation_mode: str = "zscore"  # "zscore" or "percent", baseline-referenced before the get-ready cue
    display_window_rel_cue_s: List[float] = field(default_factory=lambda: [-4.0, 1.0])
    firing_rate_isi_window_rel_cue_s: List[float] = field(default_factory=lambda: [-4.0, -1.0])
    cst_modulation_normalization_window_rel_cue_s: List[float] = field(default_factory=lambda: [-4.0, -3.0])
    sync_analysis_window_rel_cue_s: List[float] = field(default_factory=lambda: [-4.0, 1.0])
    observation_window_defs_rel_cue_s: Dict[str, List[float]] = field(
        default_factory=lambda: {
            alias: list(map(float, bounds))
            for alias, bounds in DEFAULT_OBSERVATION_WINDOW_DEFS_REL_CUE_S.items()
        }
    )
    obs_baseline_window_rel_cue_s: List[float] = field(default_factory=lambda: [-3.0, -1.0])
    modulation_spectrum_window_rel_cue_s: List[float] = field(default_factory=lambda: [-3.0, -1.0])
    modulation_spectrum_max_freq_hz: float = float(DEFAULT_MODULATION_SPECTRUM_CONFIG["max_freq_hz"])
    modulation_spectrum_interval_mass_pct: float = float(DEFAULT_MODULATION_SPECTRUM_CONFIG["interval_mass_pct"])
    feature_fitting_window_rel_cue_s: List[float] = field(default_factory=lambda: [-4.0, 1.0])
    feature_analysis_window_rel_cue_s: List[float] = field(default_factory=lambda: [0.0, 0.5])
    feature_buffer_s: float = 0.1
    summary_window_overlay_specs: Dict[str, Dict[str, float | str | bool]] = field(default_factory=lambda: {
        "firing_rate_isi_window_rel_cue_s": {"enabled": True, "label": "FR / ISI window", "color": "#37a055", "lw": 2.0, "alpha": 1.0, "y_axes": 0.02},
        "cst_modulation_normalization_window_rel_cue_s": {"enabled": True, "label": "CST normalization window", "color": "#ffa600", "lw": 2.0, "alpha": 0.95, "y_axes": 0.05},
        "sync_baseline_win_rel_cue_s": {"enabled": True, "label": "Sync baseline window", "color": "#7e2179", "lw": 2.0, "alpha": 1.0, "y_axes": 0.08},
        "feature_fitting_window": {"enabled": True, "label": "Spline fitting window", "color": "#E09ED8", "lw": 2.0, "alpha": 1.0, "y_axes": 0.15},
        "feature_analysis_window": {"enabled": True, "label": "Feature window", "color": "#E900CA", "lw": 2.0, "alpha": 1.0, "y_axes": 0.15},
    })
    # Coincidence-based sliding-window synchrony metric, kept configurable here so
    # the same analyzer logic can run directly at the end of a simulation.
    enable_sync_index_analysis: bool = False
    sync_bin_ms: float = 1.0
    sync_win_ms: float = 100.0
    sync_step_ms: float = 10.0
    sync_coinc_lag_ms: float = 6.0
    sync_direction_mode: str = "legacy_forward"  # legacy_forward or symmetric_mean
    sync_expectation_mode: str = "analytic"  # analytic or surrogate_circular_shift
    sync_n_surrogates: int = 100
    sync_surrogate_min_shift_ms: float = 1.0
    sync_baseline_win_rel_cue_s: List[float] = field(default_factory=lambda: [-3.0, -2.0])
    sync_post_win_rel_cue_s: List[float] = field(default_factory=lambda: [0.3, 0.6])
    sync_trace_display_mode: str = "absolute"  # "absolute" or "zscore"

    # # # MOTOR NEURONS ELECTROPHYSIOLOGICAL PROPERTIES CONSTANTS (Caillet et al 2022)
    # Resistance constants, used to generate the motor neuron resistance (in Ohms) according to their size
    resistance_constant: float = 9.6*(10**5)
    resistance_exponent: float = 2.4*(-1)
    # Rheobase constants, used to generate the motor neuron input current offset (in nA) according to their size
    rheobase_constant: float = 9.0*(10**-4)
    rheobase_exponent: float = 2.5
    rheobase_scaling: float = 6.0*1e2 # manually-tuned scaling
    # Capacitance constants, used to generate the motor neuron capacitance (in Farads) according to their size
    capacitance_constant: float = 1.2
    capacitance_exponent: float = 1
    # Afterhyperpolarization constants, used to generate the motor neuron AHP duration (in ms) according to their size + refractory period (in ms)
    AHP_duration_constant: float = 2.5 * (10**4)
    AHP_duration_exponent: float = 1.5 * (-1)
    # ^ these variables are used to create the variable 'motoneurons_AHP_conductance_decay_time_constant'
    # Caillet et al describe the relationship for the DURATION of the AHP, but for the equations I am using a time constant to control for the AHP conductance decay
    # I consider that the duration of the AHP correspond to the time it takes for the peak input at time 0 (x0) to decay to a tenth of its value x(t)<=X0/10
    # Thus, motoneurons_AHP_conductance_decay_time_constant = AHP_duration / ln(10)
    AHP_fast_conductance_delta_after_spiking: Quantity = field(default_factory=lambda: 10.0 * msiemens) # 1.5 * msiemens) # Hyperpolarizing conductance change after a spike
    AHP_slow_conductance_delta_after_spiking: Quantity = field(default_factory=lambda: 0.0 * msiemens) # 0.3 * msiemens) # Hyperpolarizing conductance change after a spike
    AHP_slow_conductance_decay_time_constant_multiplier: float = 50 # multiplier of the fast AHP time constant calculated per MN
    maximum_AHP_conductance_slow_conductance: float = 100.0 # will be converted to msiemens later
    refractory_period_absolute: float = 5 # in ms
    # Axonal conduction velocity constants, used to generate the MN-to-fiber velocity (in m/s) according to their size
    # Then, the delay (ms) is calculated from the axonal conduction velocity, assuming a 0.5m axon length => so correspond to the conduction speed from MN to muscle fiber (speed in m/s, so multiply speed by 2)
    axonal_conduction_velocity_constant: float = 4.0*2
    axonal_conduction_velocity_exponent: float = 0.7

    # # # MOTOR UNIT FORCE PARAMETERS
    # ----- FORCE MODEL (per motor unit) -----
    enable_force_model: bool = True
    save_per_mu_forces: bool = True            # if True, store per-MU force traces (large HDF5, but no per-MU plotting if false)
    force_kernel_duration_factor: float = 6.0   # twitch kernel length = factor * twitch_time # just to make sure that the kernel is not infinite, yet long enough to capture the twitch till it decays to ~0
    # Endpoint values at soma diameters (µm) defined by MN_min_soma_diameter / MN_max_soma_diameter
    tetanic_force_at_min_diam: float = 1.0
    tetanic_force_at_max_diam: float = 10.0
    twitch_peak_rel_at_min_diam: float = 0.20
    twitch_peak_rel_at_max_diam: float = 0.80
    twitch_time_at_min_diam: float = 0.10      # seconds (slow small units)
    twitch_time_at_max_diam: float = 0.02      # seconds (fast large units)
    # “Curvature” of the diameter→property mapping (1.0 = near-linear vs diameter,
    # >1 biases growth toward the large MUs, <1 toward small MUs)
    force_diameter_exponent: float = 2.0

    # # # FORCE TARGET AND VARYING BASELINE INPUT PARAMETERS
    # ---- Force target (for tracking + first-guess of baseline) ----
    force_target_mode: str = "constant"   # "constant" or "ramp_up"
    force_target_constant_percent: float = 30.0
    force_target_ramp_min_percent: float = 0.0
    force_target_ramp_max_percent: float = 50.0
    # ---- Time-varying baseline from target (first guess) ----
    variable_excitatory_input_baseline: bool = True # If True, override excitatory_input_baseline with a time-varying input calculated to match the force target
    excitatory_input_multiplier_of_rheobase: float = 3.0  # rehobase multiplier for first gues of time-varying input to match desired force target
    # Optional: save what we drove the model with
    save_timevarying_baseline: bool = True
    save_force_target_timeseries: bool = True
    # ----- Optimization of baseline excitatory input to match force target - associated parameters -----
    optimize_baseline: bool = False
    opt_learning_rate: float = 3.0*1e3              # Adam LR on baseline (nA units)
    opt_max_iters: int = 30
    opt_earlystop_patience: int = 5
    opt_rmse_improve_threshold_percent: float = 0.5  # % improvement vs best so far
    opt_sensitivity_calib_delta_nA: float = 1.0*1e3      # small +ΔnA to estimate d(%MVC)/d(nA)s
    opt_fix_noise_seed: int | None = 12345           # Used only when optimize_baseline=True. None = let noise vary; else fix for stable optimization iterations
    opt_gradient_smoothing_sigma_s: float = 0.35      # standard deviation of the Gaussian smoothing kernel of the driving input, in seconds. Higher values lead to more stable force output during optimization


    # # # POST INIT FUNCTION, EXECUTING AFTER INITIALIZATION
    def __post_init__(self):
        # Setting variables that are calculated post-initialization
        if self.requested_random_seed is not None:
            self.random_seed = int(self.requested_random_seed)
        elif self.pre_specify_random_seed:
            self.random_seed = 42
        else:
            # Avoid the legacy RandomState int32 bound on Windows/NumPy by drawing
            # the seed through Generator.integers and converting back to a Python int.
            self.random_seed = int(np.random.default_rng().integers(0, 2**32 - 1, dtype=np.uint32))
        self.duration_with_ignored_window = self.duration + (2 * self.edges_ignore_duration)
        self.total_nb_motoneurons = int(self.nb_motoneurons)
        #### PARAMETERS CHECK ####
        if self.duration <= 0:
            raise ValueError("duration must be strictly positive")
        if self.edges_ignore_duration < 0:
            raise ValueError("edges_ignore_duration must be non-negative")
        if self.total_nb_motoneurons < 1:
            raise ValueError("nb_motoneurons must be >= 1")
        if self.get_ready_cue_time_s < 0:
            raise ValueError("get_ready_cue_time_s must be non-negative")
        if self.go_nogo_cue_delay_s < 0:
            raise ValueError("go_nogo_cue_delay_s must be non-negative")
        if self.burst_delay_after_go_nogo_cue_s < 0:
            raise ValueError("burst_delay_after_go_nogo_cue_s must be non-negative")
        self.enable_input_burst = bool(self.enable_input_burst)
        self.enable_go_nogo_cue_diagnostics = bool(self.enable_go_nogo_cue_diagnostics)

        core_start_s = float(self.edges_ignore_duration)
        core_end_s = core_start_s + float(self.duration)
        get_ready_cue_s = core_start_s + float(self.get_ready_cue_time_s)
        go_nogo_cue_s = get_ready_cue_s + float(self.go_nogo_cue_delay_s)
        burst_center_s = go_nogo_cue_s + float(self.burst_delay_after_go_nogo_cue_s)

        uses_go_nogo_timed_inputs = bool(self.enable_input_burst) or any(
            abs(float(value)) > 1e-12
            for value in (
                self.lf_trend_slope_nA_per_s,
                self.alpha_trend_slope_nA_per_s,
                self.beta_trend_slope_nA_per_s,
            )
        )
        if (self.enable_go_nogo_cue_diagnostics or uses_go_nogo_timed_inputs) and get_ready_cue_s > core_end_s:
            raise ValueError("get_ready_cue_time_s places the get-ready cue after the usable simulation window")
        if (self.enable_go_nogo_cue_diagnostics or uses_go_nogo_timed_inputs) and go_nogo_cue_s > core_end_s:
            raise ValueError("go_nogo_cue_delay_s places the go/no-go cue after the usable simulation window")
        if self.enable_input_burst and burst_center_s > core_end_s:
            raise ValueError("burst_delay_after_go_nogo_cue_s places the burst center after the usable simulation window")

        self.input_burst_center_ms = burst_center_s * 1000.0
        self.task_event_times_s = {
            "simulation_start_s": 0.0,
            "usable_window_start_s": core_start_s,
            "baseline_segment_start_s": core_start_s,
            "get_ready_cue_s": get_ready_cue_s,
            "baseline_segment_end_s": get_ready_cue_s,
            "get_ready_segment_start_s": get_ready_cue_s,
            "go_nogo_cue_s": go_nogo_cue_s,
            "get_ready_segment_end_s": go_nogo_cue_s,
            "response_segment_start_s": go_nogo_cue_s,
            "burst_center_s": burst_center_s,
            "response_segment_end_s": core_end_s,
            "usable_window_end_s": core_end_s,
            "simulation_end_s": float(self.duration_with_ignored_window),
        }
        step_current_abs_s = core_start_s + float(self.step_current_time_s)
        self.task_event_times_s["step_current_s"] = step_current_abs_s
        if self.independent_input_absolute_or_ratio not in ('absolute', 'ratio'):
            raise ValueError(
                f"independent_input_absolute_or_ratio must be 'absolute' or 'ratio', not {self.independent_input_absolute_or_ratio!r}"
            )
        if self.cst_modulation_mode not in ("zscore", "percent"):
            raise ValueError(
                f"cst_modulation_mode must be 'zscore' or 'percent', not {self.cst_modulation_mode!r}"
            )
        self.firing_rate_isi_window_rel_cue_s = list(_normalize_rel_window(self.firing_rate_isi_window_rel_cue_s))
        self.display_window_rel_cue_s = list(_normalize_rel_window(self.display_window_rel_cue_s))
        self.cst_modulation_normalization_window_rel_cue_s = list(_normalize_rel_window(self.cst_modulation_normalization_window_rel_cue_s))
        self.sync_analysis_window_rel_cue_s = list(_normalize_rel_window(self.sync_analysis_window_rel_cue_s))
        self.obs_baseline_window_rel_cue_s = list(_normalize_rel_window(self.obs_baseline_window_rel_cue_s))
        self.modulation_spectrum_window_rel_cue_s = list(_normalize_rel_window(self.modulation_spectrum_window_rel_cue_s))
        self.feature_fitting_window_rel_cue_s = list(_normalize_rel_window(self.feature_fitting_window_rel_cue_s))
        self.feature_analysis_window_rel_cue_s = list(_normalize_rel_window(self.feature_analysis_window_rel_cue_s))
        self.sync_baseline_win_rel_cue_s = list(_normalize_rel_window(self.sync_baseline_win_rel_cue_s))
        self.sync_post_win_rel_cue_s = list(_normalize_rel_window(self.sync_post_win_rel_cue_s))
        self.step_diagnostic_display_window_rel_s = list(_normalize_rel_window(self.step_diagnostic_display_window_rel_s))
        self.step_sync_analysis_window_rel_s = list(_normalize_rel_window(self.step_sync_analysis_window_rel_s))
        self.step_pre_window_rel_s = list(_normalize_rel_window(self.step_pre_window_rel_s))
        self.step_late_post_window_rel_s = list(_normalize_rel_window(self.step_late_post_window_rel_s))
        self.response_shape_fit_window_rel_s = list(_normalize_rel_window(self.response_shape_fit_window_rel_s))
        self.xcorr_lag_window_rel_s = list(_normalize_rel_window(self.xcorr_lag_window_rel_s))
        self.sync_predictor_pre_step_window_rel_s = list(_normalize_rel_window(self.sync_predictor_pre_step_window_rel_s))
        self.sync_predictor_local_step_window_rel_s = list(_normalize_rel_window(self.sync_predictor_local_step_window_rel_s))
        self.observation_window_defs_rel_cue_s = {
            alias: list(bounds)
            for alias, bounds in _normalize_observation_window_defs(self.observation_window_defs_rel_cue_s).items()
        }
        if self.sync_direction_mode not in ("legacy_forward", "symmetric_mean"):
            raise ValueError(
                f"sync_direction_mode must be 'legacy_forward' or 'symmetric_mean', not {self.sync_direction_mode!r}"
            )
        if self.sync_expectation_mode not in ("analytic", "surrogate_circular_shift"):
            raise ValueError(
                f"sync_expectation_mode must be 'analytic' or 'surrogate_circular_shift', not {self.sync_expectation_mode!r}"
            )
        if self.sync_n_surrogates < 1:
            raise ValueError("sync_n_surrogates must be >= 1")
        if len(self.sync_baseline_win_rel_cue_s) != 2 or len(self.sync_post_win_rel_cue_s) != 2:
            raise ValueError("sync baseline/post windows must each contain exactly two values")
        if self.sync_trace_display_mode not in ("absolute", "zscore"):
            raise ValueError("sync_trace_display_mode must be 'absolute' or 'zscore'")
        if int(self.minimal_output_target_fsamp) <= 0:
            raise ValueError("minimal_output_target_fsamp must be strictly positive")
        self.minimal_output_target_fsamp = int(self.minimal_output_target_fsamp)
        self.parameter_set_id = int(self.parameter_set_id)
        self.repeat_index = int(self.repeat_index)
        if self.paired_repeat_seed is not None:
            self.paired_repeat_seed = int(self.paired_repeat_seed)
        self.step_index = int(self.step_index)
        self.is_zero_step_reference = bool(self.is_zero_step_reference)
        self.parameter_set_folder_name = str(self.parameter_set_folder_name)
        self.repeat_folder_name = str(self.repeat_folder_name)
        self.step_folder_name = str(self.step_folder_name)
        self.paired_reference_folder_name = str(self.paired_reference_folder_name)
        self.brian_codegen_target = str(self.brian_codegen_target).strip().lower()
        if self.brian_codegen_target not in {"numpy", "cython"}:
            raise ValueError("brian_codegen_target must be 'numpy' or 'cython'")

        # Public single-pool parameters are normalized here so notebooks can pass
        # simple scalars / tuples / lists without extra boilerplate.
        self.excitatory_input_baseline = float(np.asarray(self.excitatory_input_baseline, dtype=float).reshape(-1)[0])

        def _normalize_band(name, band_hz):
            arr = np.asarray(band_hz, dtype=float).reshape(-1)
            if arr.size != 2:
                raise ValueError(f"{name} must contain exactly two values: [low, high]")
            low_hz = float(arr[0])
            high_hz = float(arr[1])
            if low_hz < 0 or low_hz >= high_hz:
                raise ValueError(f"{name} must satisfy 0 <= low < high")
            if high_hz > self.max_frequency_of_any_input:
                warnings.warn(
                    f"Clamping {name} upper bound from {high_hz} Hz to {self.max_frequency_of_any_input} Hz."
                )
                high_hz = float(self.max_frequency_of_any_input)
            return [low_hz, high_hz]

        self.lf_target_sd = float(self.lf_target_sd)
        self.alpha_target_sd_final = float(self.alpha_target_sd_final)
        self.beta_target_sd_final = float(self.beta_target_sd_final)
        self.alpha_beta_envelope_variability_scale = float(self.alpha_beta_envelope_variability_scale)
        self.common_input_weight_sd = float(self.common_input_weight_sd)
        self.common_input_weight_min = float(self.common_input_weight_min)
        self.common_input_weight_max = float(self.common_input_weight_max)
        self.independent_input_weight_sd = float(self.independent_input_weight_sd)
        self.independent_input_weight_max = float(self.independent_input_weight_max)
        self.input_component_diag_zoom_duration_s = float(self.input_component_diag_zoom_duration_s)
        self.input_component_colors = dict(self.input_component_colors)
        self.cst_analysis_window_start_s = float(self.cst_analysis_window_start_s)
        self.cst_analysis_window_end_s = float(self.cst_analysis_window_end_s)
        self.active_unit_rate_threshold_hz = float(self.active_unit_rate_threshold_hz)
        if self.lf_target_sd < 0 or self.alpha_target_sd_final < 0 or self.beta_target_sd_final < 0:
            raise ValueError("lf_target_sd, alpha_target_sd_final, and beta_target_sd_final must be non-negative")
        if not (0.0 <= self.alpha_beta_envelope_variability_scale <= 1.0):
            raise ValueError("alpha_beta_envelope_variability_scale must be between 0 and 1")
        if self.input_component_diag_zoom_duration_s <= 0:
            raise ValueError("input_component_diag_zoom_duration_s must be strictly positive")
        if self.cst_analysis_window_start_s < 0:
            raise ValueError("cst_analysis_window_start_s must be non-negative")
        if self.cst_analysis_window_start_s >= self.duration:
            raise ValueError("cst_analysis_window_start_s must lie inside the usable simulation duration")
        if self.cst_analysis_window_end_s > 0 and self.cst_analysis_window_end_s <= self.cst_analysis_window_start_s:
            raise ValueError("cst_analysis_window_end_s must be larger than cst_analysis_window_start_s")
        if self.cst_analysis_window_end_s > self.duration:
            raise ValueError("cst_analysis_window_end_s must lie inside the usable simulation duration")
        if self.active_unit_rate_threshold_hz < 0:
            raise ValueError("active_unit_rate_threshold_hz must be non-negative")
        if self.common_input_weight_sd < 0:
            raise ValueError("common_input_weight_sd must be non-negative")
        if self.common_input_weight_min < 0:
            raise ValueError("common_input_weight_min must be non-negative")
        if self.common_input_weight_max <= self.common_input_weight_min:
            raise ValueError("common_input_weight_max must be larger than common_input_weight_min")
        if not (self.common_input_weight_min <= 1.0 <= self.common_input_weight_max):
            raise ValueError("common_input_weight_min/max must bracket 1.0 so the weight distribution can keep mean 1")
        if self.independent_input_weight_sd < 0:
            raise ValueError("independent_input_weight_sd must be non-negative")
        if self.independent_input_weight_max <= 1.0:
            raise ValueError("independent_input_weight_max must be larger than 1.0 so the weight distribution can keep mean 1")
        if not isinstance(self.input_component_colors, dict):
            raise ValueError("input_component_colors must be a dict mapping component names to colors")

        self.lf_band_hz = _normalize_band("lf_band_hz", self.lf_band_hz)
        self.alpha_band_hz = _normalize_band("alpha_band_hz", self.alpha_band_hz)
        self.alpha_envelope_band_hz = _normalize_band("alpha_envelope_band_hz", self.alpha_envelope_band_hz)
        self.beta_band_hz = _normalize_band("beta_band_hz", self.beta_band_hz)
        self.beta_envelope_band_hz = _normalize_band("beta_envelope_band_hz", self.beta_envelope_band_hz)
        if self.input_burst_sigma_ms <= 0:
            raise ValueError("input_burst_sigma_ms must be strictly positive")
        if self.input_burst_start_marker_n_sigma < 0:
            raise ValueError("input_burst_start_marker_n_sigma must be non-negative")
        if self.input_burst_zoom_start_n_sigma < 0:
            raise ValueError("input_burst_zoom_start_n_sigma must be non-negative")
        self.lf_burst_peak_nA = float(self.lf_burst_peak_nA)
        self.alpha_burst_peak_nA = float(self.alpha_burst_peak_nA)
        self.beta_burst_peak_nA = float(self.beta_burst_peak_nA)
        self.lf_trend_start_rel_go_cue_s = float(self.lf_trend_start_rel_go_cue_s)
        self.alpha_trend_start_rel_go_cue_s = float(self.alpha_trend_start_rel_go_cue_s)
        self.beta_trend_start_rel_go_cue_s = float(self.beta_trend_start_rel_go_cue_s)
        self.lf_trend_slope_nA_per_s = float(self.lf_trend_slope_nA_per_s)
        self.alpha_trend_slope_nA_per_s = float(self.alpha_trend_slope_nA_per_s)
        self.beta_trend_slope_nA_per_s = float(self.beta_trend_slope_nA_per_s)
        self.enable_step_current = bool(self.enable_step_current)
        self.step_current_time_s = float(self.step_current_time_s)
        self.step_current_amplitude_nA = float(self.step_current_amplitude_nA)
        self.step_current_mode = str(self.step_current_mode).strip().lower()
        self.step_current_ramp_time_s = float(self.step_current_ramp_time_s)
        if self.step_current_duration_s is not None:
            self.step_current_duration_s = float(self.step_current_duration_s)
        if self.step_current_amplitude_percent_baseline is not None:
            self.step_current_amplitude_percent_baseline = float(self.step_current_amplitude_percent_baseline)
        if self.step_current_mode not in {"absolute_na", "percent_baseline"}:
            raise ValueError("step_current_mode must be 'absolute_nA' or 'percent_baseline'")
        if self.step_current_mode == "absolute_na":
            self.step_current_mode = "absolute_nA"
        if self.step_current_time_s < 0 or step_current_abs_s > core_end_s:
            raise ValueError("step_current_time_s must place the step inside the usable simulation window")
        if self.step_current_duration_s is not None and self.step_current_duration_s < 0:
            raise ValueError("step_current_duration_s must be non-negative or None")
        if self.step_current_ramp_time_s < 0:
            raise ValueError("step_current_ramp_time_s must be non-negative")
        if self.enable_step_current and self.step_current_mode == "percent_baseline" and self.step_current_amplitude_percent_baseline is None:
            raise ValueError("step_current_amplitude_percent_baseline is required when step_current_mode='percent_baseline'")
        if self.enable_step_current and self.step_current_mode == "absolute_nA" and self.step_current_amplitude_percent_baseline is not None:
            raise ValueError(
                "step_current_amplitude_percent_baseline must be None when step_current_mode='absolute_nA'. "
                "Use step_current_mode='percent_baseline' to specify a percent-baseline step."
            )
        if not isinstance(self.step_diagnostic_plot_style, dict):
            raise ValueError("step_diagnostic_plot_style must be a dict")
        self.step_diagnostic_plot_style = dict(self.step_diagnostic_plot_style)
        self.enable_step_sync_index_analysis = bool(self.enable_step_sync_index_analysis)
        self.step_response_cst_smoothing_method = str(self.step_response_cst_smoothing_method).strip().lower()
        if self.step_response_cst_smoothing_method not in {"causal_exponential", "causal_hann"}:
            raise ValueError("step_response_cst_smoothing_method must be 'causal_exponential' or 'causal_hann'")
        self.step_response_cst_smoothing_tau_s = float(self.step_response_cst_smoothing_tau_s)
        if self.step_response_cst_smoothing_tau_s <= 0:
            raise ValueError("step_response_cst_smoothing_tau_s must be strictly positive")
        self.step_response_cst_hann_half_width_s = float(self.step_response_cst_hann_half_width_s)
        if self.step_response_cst_hann_half_width_s <= 0:
            raise ValueError("step_response_cst_hann_half_width_s must be strictly positive")
        self.step_slope_candidate_min_s = float(self.step_slope_candidate_min_s)
        self.step_slope_candidate_max_s = float(self.step_slope_candidate_max_s)
        self.step_slope_candidate_step_s = float(self.step_slope_candidate_step_s)
        self.step_slope_r2_min = float(self.step_slope_r2_min)
        self.step_slope_response_fraction_min = float(self.step_slope_response_fraction_min)
        self.response_shape_tau_min_ms = float(self.response_shape_tau_min_ms)
        self.response_shape_tau_max_ms = float(self.response_shape_tau_max_ms)
        self.response_shape_min_r_inf = float(self.response_shape_min_r_inf)
        self.xcorr_max_lag_s = float(self.xcorr_max_lag_s)
        if self.step_slope_candidate_min_s <= 0 or self.step_slope_candidate_max_s <= 0:
            raise ValueError("step slope candidate bounds must be strictly positive")
        if self.step_slope_candidate_min_s > self.step_slope_candidate_max_s:
            raise ValueError("step_slope_candidate_min_s must be <= step_slope_candidate_max_s")
        if self.step_slope_candidate_step_s <= 0:
            raise ValueError("step_slope_candidate_step_s must be strictly positive")
        if not (0.0 <= self.step_slope_r2_min <= 1.0):
            raise ValueError("step_slope_r2_min must be between 0 and 1")
        if self.step_slope_response_fraction_min < 0:
            raise ValueError("step_slope_response_fraction_min must be non-negative")
        if self.response_shape_tau_min_ms <= 0 or self.response_shape_tau_max_ms <= 0:
            raise ValueError("response_shape tau bounds must be strictly positive")
        if self.response_shape_tau_min_ms >= self.response_shape_tau_max_ms:
            raise ValueError("response_shape_tau_min_ms must be < response_shape_tau_max_ms")
        if self.response_shape_min_r_inf < 0:
            raise ValueError("response_shape_min_r_inf must be non-negative")
        if self.xcorr_max_lag_s < 0:
            raise ValueError("xcorr_max_lag_s must be non-negative")
        self.spike_train_show_task_cues_when_no_burst = bool(self.spike_train_show_task_cues_when_no_burst)
        self.spike_train_show_step_current_marker = bool(self.spike_train_show_step_current_marker)
        self.enable_cst_hann_smoothing = bool(self.enable_cst_hann_smoothing)
        self.cst_hann_smoothing_window_s = float(self.cst_hann_smoothing_window_s)
        if self.cst_hann_smoothing_window_s <= 0:
            raise ValueError("cst_hann_smoothing_window_s must be strictly positive")
        if self.enable_input_burst:
            if np.allclose(
                [self.lf_burst_peak_nA, self.alpha_burst_peak_nA, self.beta_burst_peak_nA],
                0.0,
            ):
                warnings.warn("enable_input_burst=True but all component burst peaks are zero.")
            if (burst_center_s < 0) or (burst_center_s > self.duration_with_ignored_window):
                raise ValueError(
                    "input_burst_center_ms must lie within the simulated time window when enable_input_burst=True"
                )
        if self.minimal_output and self.minimal_output_downsample_saved_traces:
            highest_saved_band_hz = max(
                float(self.lf_band_hz[1]),
                float(self.alpha_band_hz[1]),
                float(self.beta_band_hz[1]),
            )
            nyquist_hz = 0.5 * float(self.minimal_output_target_fsamp)
            if nyquist_hz <= highest_saved_band_hz:
                warnings.warn(
                    "minimal_output_target_fsamp may be too low for the saved band-limited traces: "
                    f"Nyquist={nyquist_hz:.1f} Hz, highest saved band upper bound={highest_saved_band_hz:.1f} Hz."
                )
######################## END OF SimulationParameters CLASS

######################################
### FETCH THREAD-LOCAL STORAGE FOR "CURRENT" PARAMETERS
### So that they can be used in the helper functions without explicitely passing them as arguments
######################################
_params_context = local()
def _set_filter_params(params: SimulationParameters):
    """Call this at the top of run_simulation to make params visible to the filters."""
    _params_context.params = params
def _get_filter_params() -> SimulationParameters:
    try:
        return _params_context.params
    except AttributeError:
        raise RuntimeError("No filter params set; forgot to call _set_filter_params?")


def _baseline_by_pool_nA(params: SimulationParameters) -> np.ndarray:
    return np.array([float(params.excitatory_input_baseline)], dtype=float)


def resolve_step_current_amplitude_nA(params: SimulationParameters) -> float:
    if not bool(params.enable_step_current):
        return 0.0
    mode = str(params.step_current_mode)
    if mode == "absolute_nA":
        return float(params.step_current_amplitude_nA)
    if mode == "percent_baseline":
        percent = float(params.step_current_amplitude_percent_baseline)
        return float(params.excitatory_input_baseline) * percent / 100.0
    raise ValueError(f"Unsupported step_current_mode {mode!r}")


def build_step_current_trace(params: SimulationParameters, n_samples: int) -> dict:
    t_s = np.arange(int(n_samples), dtype=float) / float(params.fsamp)
    step_abs_s = float(params.edges_ignore_duration) + float(params.step_current_time_s)
    step_amp_nA = resolve_step_current_amplitude_nA(params)
    trace = np.zeros(int(n_samples), dtype=float)
    if bool(params.enable_step_current):
        duration_s = params.step_current_duration_s
        step_end_s = float(params.duration_with_ignored_window) if duration_s is None else step_abs_s + float(duration_s)
        active_mask = (t_s >= step_abs_s) & (t_s < step_end_s)
        if float(params.step_current_ramp_time_s) <= 0.0:
            trace[active_mask] = step_amp_nA
        else:
            ramp_s = float(params.step_current_ramp_time_s)
            ramp_mask = active_mask & (t_s < step_abs_s + ramp_s)
            plateau_mask = active_mask & ~ramp_mask
            if np.any(ramp_mask):
                trace[ramp_mask] = step_amp_nA * np.clip((t_s[ramp_mask] - step_abs_s) / ramp_s, 0.0, 1.0)
            trace[plateau_mask] = step_amp_nA
    else:
        step_end_s = step_abs_s

    percent = np.nan
    if abs(float(params.excitatory_input_baseline)) > 1e-12:
        percent = 100.0 * step_amp_nA / float(params.excitatory_input_baseline)
    return {
        "time_s": t_s,
        "step_current_trace_nA": trace,
        "step_current_amplitude_resolved_nA": float(step_amp_nA),
        "step_current_amplitude_resolved_percent_baseline": float(percent),
        "step_current_time_abs_s": float(step_abs_s),
        "step_current_time_usable_rel_s": float(params.step_current_time_s),
        "step_current_end_abs_s": float(step_end_s),
    }


def _baseline_trace_for_step(params: SimulationParameters, baseline_ts_nA, n_samples: int) -> np.ndarray:
    if baseline_ts_nA is None:
        return np.full(int(n_samples), float(params.excitatory_input_baseline), dtype=float)
    arr = np.asarray(baseline_ts_nA, dtype=float).reshape(-1)
    if arr.size < int(n_samples):
        raise ValueError("baseline time series is shorter than the simulation input trace")
    return arr[:int(n_samples)].copy()


def _safe_nanmean(values) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.nanmean(finite)) if finite.size else np.nan


def _safe_nanstd(values) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.nanstd(finite)) if finite.size else np.nan


def _safe_nanmedian(values) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.nanmedian(finite)) if finite.size else np.nan


def _figures_enabled(params: SimulationParameters) -> bool:
    return bool(params.output_plots) and not (
        bool(params.minimal_output) and bool(params.minimal_output_turn_off_figure_output)
    )


def _copy_value_for_save(value):
    if value is None or isinstance(value, (str, bytes, int, float, bool, np.integer, np.floating, np.bool_)):
        return value
    if isinstance(value, dict):
        return {k: _copy_value_for_save(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_value_for_save(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_value_for_save(v) for v in value)
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    try:
        return np.asarray(value).copy()
    except Exception:
        return value


def _is_trace_like_array(value, source_n_samples=None):
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    if arr.ndim == 0 or arr.shape[-1] <= 1:
        return False
    if source_n_samples is None:
        return True
    return int(arr.shape[-1]) == int(source_n_samples)


def _resample_array_last_axis(data, source_fsamp_hz, target_fsamp_hz):
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.shape[-1] <= 1:
        return np.array(arr, copy=True)
    source_fsamp_hz = float(source_fsamp_hz)
    target_fsamp_hz = float(target_fsamp_hz)
    if target_fsamp_hz >= source_fsamp_hz - 1e-12:
        return np.array(arr, copy=True)
    source_i = int(round(source_fsamp_hz))
    target_i = int(round(target_fsamp_hz))
    gcd = int(np.gcd(source_i, target_i))
    up = target_i // gcd
    down = source_i // gcd
    resampled = resample_poly(arr, up=up, down=down, axis=-1)
    expected_len = max(1, int(round(arr.shape[-1] * target_fsamp_hz / source_fsamp_hz)))
    if resampled.shape[-1] > expected_len:
        resampled = resampled[..., :expected_len]
    elif resampled.shape[-1] < expected_len:
        pad_width = [(0, 0)] * resampled.ndim
        pad_width[-1] = (0, expected_len - resampled.shape[-1])
        resampled = np.pad(resampled, pad_width, mode="edge")
    return np.asarray(resampled, dtype=arr.dtype if np.issubdtype(arr.dtype, np.number) else float)


def _resample_uniform_time_and_traces(time_s, trace_dict, target_fsamp_hz):
    time_s = np.asarray(time_s, dtype=float)
    if time_s.ndim != 1 or time_s.size <= 1:
        return time_s.copy(), {k: _copy_value_for_save(v) for k, v in trace_dict.items()}
    current_fsamp_hz = 1.0 / float(np.median(np.diff(time_s)))
    if target_fsamp_hz >= current_fsamp_hz - 1e-12:
        return time_s.copy(), {k: _copy_value_for_save(v) for k, v in trace_dict.items()}
    dt_s = 1.0 / float(target_fsamp_hz)
    new_time = np.arange(time_s[0], time_s[-1] + 0.5 * dt_s, dt_s, dtype=float)
    resampled = {}
    for key, value in trace_dict.items():
        if value is None:
            resampled[key] = None
            continue
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.shape[-1] != time_s.size:
            resampled[key] = _copy_value_for_save(value)
            continue
        if arr.ndim == 1:
            resampled[key] = np.interp(new_time, time_s, np.asarray(arr, dtype=float))
        else:
            flat = np.reshape(np.asarray(arr, dtype=float), (-1, arr.shape[-1]))
            interp_flat = np.stack(
                [np.interp(new_time, time_s, row) for row in flat],
                axis=0,
            )
            resampled[key] = interp_flat.reshape(arr.shape[:-1] + (new_time.size,))
    return new_time, resampled


def _compact_trace_dict(data_dict, source_fsamp_hz, target_fsamp_hz, source_n_samples=None, drop_keys=None):
    compact = {}
    drop_keys = set(drop_keys or [])
    for key, value in (data_dict or {}).items():
        if key in drop_keys:
            continue
        if value is None:
            compact[key] = None
            continue
        if _is_trace_like_array(value, source_n_samples=source_n_samples):
            compact[key] = _resample_array_last_axis(value, source_fsamp_hz, target_fsamp_hz)
        else:
            compact[key] = _copy_value_for_save(value)
    return compact


def compute_firing_rate_diagnostics(spike_trains_MN, active_unit_ids_for_isi_cv=None):
    n_units = len(spike_trains_MN)
    mean_firing_rate_hz = np.zeros(n_units, dtype=float)
    std_firing_rate_hz = np.zeros(n_units, dtype=float)
    max_firing_rate_hz = np.zeros(n_units, dtype=float)
    min_firing_rate_hz = np.zeros(n_units, dtype=float)
    isi_cv = np.full(n_units, np.nan, dtype=float)
    active_mask = np.ones(n_units, dtype=bool)
    if active_unit_ids_for_isi_cv is not None:
        active_mask[:] = False
        active_ids = np.asarray(active_unit_ids_for_isi_cv, dtype=int).reshape(-1)
        active_ids = active_ids[(active_ids >= 0) & (active_ids < n_units)]
        active_mask[active_ids] = True

    for unit_i, spike_times in enumerate(spike_trains_MN):
        times = np.asarray(spike_times, dtype=float)
        if times.size < 2:
            continue
        isis = np.diff(times)
        if np.any(isis <= 0):
            continue
        inst_rates = 1.0 / isis
        mean_firing_rate_hz[unit_i] = float(np.mean(inst_rates))
        std_firing_rate_hz[unit_i] = float(np.std(inst_rates))
        max_firing_rate_hz[unit_i] = float(np.max(inst_rates))
        min_firing_rate_hz[unit_i] = float(np.min(inst_rates))
        if not active_mask[unit_i]:
            continue
        mean_isi = float(np.mean(isis))
        std_isi = float(np.std(isis))
        raw_cv = std_isi / mean_isi if mean_isi > 0 else np.nan
        isi_cv[unit_i] = raw_cv if (np.isfinite(raw_cv) and raw_cv >= 0.01) else 0.0

    return {
        "mn_mean_firing_rate_hz": mean_firing_rate_hz,
        "mn_std_firing_rate_hz": std_firing_rate_hz,
        "mn_max_firing_rate_hz": max_firing_rate_hz,
        "mn_min_firing_rate_hz": min_firing_rate_hz,
        "mn_isi_cv": isi_cv,
        "mn_is_active_for_thresholded_analyses": active_mask.astype(float),
        "n_units_total": np.asarray([float(n_units)], dtype=float),
        "n_units_kept": np.asarray([float(np.count_nonzero(active_mask))], dtype=float),
    }


def _window_unit_rate_and_cv(spike_trains_MN, active_unit_ids, window_start_s, window_end_s):
    active_ids = np.asarray(active_unit_ids, dtype=int).reshape(-1)
    duration_s = float(window_end_s) - float(window_start_s)
    rates = np.full(active_ids.size, np.nan, dtype=float)
    cvs = np.full(active_ids.size, np.nan, dtype=float)
    if duration_s <= 0:
        return rates, cvs
    for out_i, unit_i in enumerate(active_ids):
        if unit_i < 0 or unit_i >= len(spike_trains_MN):
            continue
        times = np.asarray(spike_trains_MN[int(unit_i)], dtype=float)
        times = times[(times >= float(window_start_s)) & (times < float(window_end_s))]
        rates[out_i] = float(times.size) / duration_s
        if times.size >= 2:
            isis = np.diff(times)
            mean_isi = float(np.mean(isis))
            if mean_isi > 0:
                cvs[out_i] = float(np.std(isis) / mean_isi)
    return rates, cvs


def _window_stats_from_trace(t_s, trace, window_start_s, window_end_s):
    t_s = np.asarray(t_s, dtype=float)
    trace = np.asarray(trace, dtype=float)
    mask = (t_s >= float(window_start_s)) & (t_s < float(window_end_s))
    if not np.any(mask):
        return np.nan, np.nan, mask
    values = trace[mask]
    return _safe_nanmean(values), _safe_nanstd(values), mask


def _window_summary_from_trace(t_s, trace, window_start_s, window_end_s):
    t_s = np.asarray(t_s, dtype=float)
    trace = np.asarray(trace, dtype=float)
    mask = (t_s >= float(window_start_s)) & (t_s < float(window_end_s))
    if not np.any(mask):
        return {"mean": np.nan, "median": np.nan, "sd": np.nan, "mask": mask}
    values = trace[mask]
    return {
        "mean": _safe_nanmean(values),
        "median": _safe_nanmedian(values),
        "sd": _safe_nanstd(values),
        "mask": mask,
    }


def _empty_sync_predictor_updates(source, pre_window_rel_s=None, local_window_rel_s=None):
    pre_window = np.asarray(pre_window_rel_s if pre_window_rel_s is not None else [-0.250, 0.000], dtype=float)
    local_window = np.asarray(local_window_rel_s if local_window_rel_s is not None else [-0.250, 0.250], dtype=float)
    return {
        "sync_predictor_source": str(source),
        "sync_predictor_pre_step_window_rel_s": pre_window,
        "sync_predictor_local_step_window_rel_s": local_window,
        "pre_step_sync_mean_baseline_matched": np.asarray([np.nan], dtype=float),
        "pre_step_sync_sd_baseline_matched": np.asarray([np.nan], dtype=float),
        "local_step_sync_mean_baseline_matched": np.asarray([np.nan], dtype=float),
        "local_step_sync_sd_baseline_matched": np.asarray([np.nan], dtype=float),
    }


def _compute_sync_predictor_updates(sync_time_rel_s, sync_trace, *, source, pre_window_rel_s, local_window_rel_s):
    pre_window = np.asarray(pre_window_rel_s, dtype=float).reshape(-1)
    local_window = np.asarray(local_window_rel_s, dtype=float).reshape(-1)
    if pre_window.size != 2:
        pre_window = np.asarray([-0.250, 0.000], dtype=float)
    if local_window.size != 2:
        local_window = np.asarray([-0.250, 0.250], dtype=float)
    sync_time_rel_s = np.asarray(sync_time_rel_s, dtype=float)
    sync_trace = np.asarray(sync_trace, dtype=float)
    if sync_time_rel_s.shape != sync_trace.shape or sync_time_rel_s.size == 0:
        return _empty_sync_predictor_updates(source, pre_window, local_window)
    pre = _window_summary_from_trace(sync_time_rel_s, sync_trace, pre_window[0], pre_window[1])
    local = _window_summary_from_trace(sync_time_rel_s, sync_trace, local_window[0], local_window[1])
    return {
        "sync_predictor_source": str(source),
        "sync_predictor_pre_step_window_rel_s": pre_window,
        "sync_predictor_local_step_window_rel_s": local_window,
        "pre_step_sync_mean_baseline_matched": np.asarray([pre["mean"]], dtype=float),
        "pre_step_sync_sd_baseline_matched": np.asarray([pre["sd"]], dtype=float),
        "local_step_sync_mean_baseline_matched": np.asarray([local["mean"]], dtype=float),
        "local_step_sync_sd_baseline_matched": np.asarray([local["sd"]], dtype=float),
    }


def compute_step_response_diagnostics(
        *,
        params,
        spike_trains_MN,
        active_unit_selection,
        cst_diagnostics,
        sync_diagnostics,
        input_components):
    step_trace_info = build_step_current_trace(params, len(input_components["common_input_with_burst"]))
    step_amp_nA = float(step_trace_info["step_current_amplitude_resolved_nA"])
    step_abs_s = float(step_trace_info["step_current_time_abs_s"])
    pre_rel = list(params.step_pre_window_rel_s)
    late_post_rel = list(params.step_late_post_window_rel_s)
    shape_fit_rel = list(params.response_shape_fit_window_rel_s)
    xcorr_rel = list(params.xcorr_lag_window_rel_s)
    sync_pred_pre_rel = list(params.sync_predictor_pre_step_window_rel_s)
    sync_pred_local_rel = list(params.sync_predictor_local_step_window_rel_s)
    pre_win = tuple(float(step_abs_s + rel) for rel in pre_rel)
    late_post_win = tuple(float(step_abs_s + rel) for rel in late_post_rel)
    slope_candidate_win = (
        float(step_abs_s),
        float(step_abs_s + params.step_slope_candidate_max_s),
    )

    rows = {
        "status": "ok",
        "diagnostics_stage": "stage1_raw_only",
        "step_response_smoothing_method": str(getattr(params, "step_response_cst_smoothing_method", "causal_exponential")),
        "step_response_cst_smoothing_method": str(getattr(params, "step_response_cst_smoothing_method", "causal_exponential")),
        "step_current_enabled": np.asarray([float(bool(params.enable_step_current))], dtype=float),
        "baseline_tonic_input_nA": np.asarray([float(params.excitatory_input_baseline)], dtype=float),
        "lf_input_amplitude_nA": np.asarray([float(params.lf_target_sd)], dtype=float),
        "alpha_input_amplitude_nA": np.asarray([float(params.alpha_target_sd_final)], dtype=float),
        "beta_input_amplitude_nA": np.asarray([float(params.beta_target_sd_final)], dtype=float),
        "step_current_amplitude_nA": np.asarray([step_amp_nA], dtype=float),
        "step_current_amplitude_percent_baseline": np.asarray([step_trace_info["step_current_amplitude_resolved_percent_baseline"]], dtype=float),
        "step_current_time_abs_s": np.asarray([step_abs_s], dtype=float),
        "step_current_time_usable_rel_s": np.asarray([step_trace_info["step_current_time_usable_rel_s"]], dtype=float),
        "step_current_duration_s": np.asarray([np.nan if params.step_current_duration_s is None else float(params.step_current_duration_s)], dtype=float),
        "step_pre_window_rel_s": np.asarray(pre_rel, dtype=float),
        "step_late_post_window_rel_s": np.asarray(late_post_rel, dtype=float),
        "step_pre_window_abs_s": np.asarray(pre_win, dtype=float),
        "step_late_post_window_abs_s": np.asarray(late_post_win, dtype=float),
        "step_slope_candidate_window_abs_s": np.asarray(slope_candidate_win, dtype=float),
        "response_shape_fit_window_rel_s": np.asarray(shape_fit_rel, dtype=float),
        "response_shape_tau_min_ms": np.asarray([float(params.response_shape_tau_min_ms)], dtype=float),
        "response_shape_tau_max_ms": np.asarray([float(params.response_shape_tau_max_ms)], dtype=float),
        "response_shape_min_r_inf": np.asarray([float(params.response_shape_min_r_inf)], dtype=float),
        "xcorr_window_rel_s": np.asarray(xcorr_rel, dtype=float),
        "xcorr_max_lag_s": np.asarray([float(params.xcorr_max_lag_s)], dtype=float),
        "sync_predictor_pre_step_window_rel_s": np.asarray(sync_pred_pre_rel, dtype=float),
        "sync_predictor_local_step_window_rel_s": np.asarray(sync_pred_local_rel, dtype=float),
        "sync_predictor_source": "stage1_not_finalized",
        "step_response_cst_smoothing_tau_s": np.asarray([float(getattr(params, "step_response_cst_smoothing_tau_s", np.nan))], dtype=float),
        "step_response_cst_hann_half_width_s": np.asarray([float(getattr(params, "step_response_cst_hann_half_width_s", np.nan))], dtype=float),
        "step_slope_candidate_min_s": np.asarray([float(getattr(params, "step_slope_candidate_min_s", np.nan))], dtype=float),
        "step_slope_candidate_max_s": np.asarray([float(getattr(params, "step_slope_candidate_max_s", np.nan))], dtype=float),
        "step_slope_candidate_step_s": np.asarray([float(getattr(params, "step_slope_candidate_step_s", np.nan))], dtype=float),
        "step_slope_r2_min": np.asarray([float(getattr(params, "step_slope_r2_min", np.nan))], dtype=float),
        "step_slope_response_fraction_min": np.asarray([float(getattr(params, "step_slope_response_fraction_min", np.nan))], dtype=float),
        "zero_step_condition": np.asarray([bool(abs(step_amp_nA) < 1e-12)], dtype=bool),
        "is_zero_step_reference": np.asarray([bool(getattr(params, "is_zero_step_reference", abs(step_amp_nA) < 1e-12))], dtype=bool),
        "matched_step_effect_available": np.asarray([False], dtype=bool),
    }
    rows.update(_empty_sync_predictor_updates("stage1_not_finalized", sync_pred_pre_rel, sync_pred_local_rel))

    active_ids = np.asarray(active_unit_selection.get("active_unit_ids", []), dtype=int)
    pre_rates, pre_cvs = _window_unit_rate_and_cv(spike_trains_MN, active_ids, *pre_win)
    post_rates, post_cvs = _window_unit_rate_and_cv(spike_trains_MN, active_ids, *late_post_win)
    rows.update({
        "active_unit_ids": active_ids,
        "pre_step_unit_firing_rate_hz": pre_rates,
        "post_step_unit_firing_rate_hz": post_rates,
        "pre_step_unit_isi_cv": pre_cvs,
        "post_step_unit_isi_cv": post_cvs,
        "pre_step_mean_firing_rate_hz": np.asarray([_safe_nanmean(pre_rates)], dtype=float),
        "pre_step_sd_firing_rate_hz": np.asarray([_safe_nanstd(pre_rates)], dtype=float),
        "pre_step_isi_cv_mean": np.asarray([_safe_nanmean(pre_cvs)], dtype=float),
        "pre_step_isi_cv_sd": np.asarray([_safe_nanstd(pre_cvs)], dtype=float),
        "post_step_mean_firing_rate_hz": np.asarray([_safe_nanmean(post_rates)], dtype=float),
        "post_step_sd_firing_rate_hz": np.asarray([_safe_nanstd(post_rates)], dtype=float),
        "post_step_isi_cv_mean": np.asarray([_safe_nanmean(post_cvs)], dtype=float),
        "post_step_isi_cv_sd": np.asarray([_safe_nanstd(post_cvs)], dtype=float),
    })

    if cst_diagnostics is None or str(cst_diagnostics.get("status", "")).lower() != "ok":
        rows["status"] = "no_cst_diagnostics"
        return rows

    cst_time_s = np.asarray(cst_diagnostics["time_s"], dtype=float)
    cst_rel_s = cst_time_s - step_abs_s
    cst_normalized = np.asarray(cst_diagnostics["cst_normalized"], dtype=float)
    smoothing_method = str(params.step_response_cst_smoothing_method).strip().lower()
    cst_smooth_alpha = np.nan
    cst_smooth_kernel = None
    if smoothing_method == "causal_exponential":
        cst_smooth, cst_smooth_alpha = apply_causal_exponential_smoothing(
            cst_normalized,
            fsamp=params.fsamp,
            tau_s=float(params.step_response_cst_smoothing_tau_s),
        )
    elif smoothing_method == "causal_hann":
        cst_smooth, cst_smooth_kernel = apply_causal_hann_smoothing(
            cst_normalized,
            fsamp=params.fsamp,
            half_width_s=float(params.step_response_cst_hann_half_width_s),
        )
    else:
        raise ValueError(f"Unsupported step_response_cst_smoothing_method: {smoothing_method}")
    pre_smooth = _window_summary_from_trace(cst_time_s, cst_smooth, *pre_win)
    post_smooth = _window_summary_from_trace(cst_time_s, cst_smooth, *late_post_win)
    raw_delta_smooth = (
        float(post_smooth["median"] - pre_smooth["median"])
        if np.isfinite(pre_smooth["median"]) and np.isfinite(post_smooth["median"])
        else np.nan
    )
    raw_gain_smooth = np.nan if abs(step_amp_nA) < 1e-12 else raw_delta_smooth / step_amp_nA
    raw_gain_smooth_per_uA = raw_gain_smooth * 1000.0 if np.isfinite(raw_gain_smooth) else np.nan
    rows.update({
        "time_s": cst_time_s,
        "time_rel_step_s": cst_rel_s,
        "t_step_rel_s": cst_rel_s,
        "cst_smooth_step_response": cst_smooth,
        "cst_smooth_causal_exp": cst_smooth,
        "cst_smooth_causal_exp_alpha": np.asarray([float(cst_smooth_alpha)], dtype=float),
        "step_pre_mask": np.asarray(pre_smooth["mask"], dtype=float),
        "step_late_post_mask": np.asarray(post_smooth["mask"], dtype=float),
        "pre_step_cst_smooth_mean": np.asarray([pre_smooth["mean"]], dtype=float),
        "pre_step_cst_smooth_median": np.asarray([pre_smooth["median"]], dtype=float),
        "pre_step_cst_smooth_sd": np.asarray([pre_smooth["sd"]], dtype=float),
        "post_step_cst_smooth_mean": np.asarray([post_smooth["mean"]], dtype=float),
        "post_step_cst_smooth_median": np.asarray([post_smooth["median"]], dtype=float),
        "post_step_cst_smooth_sd": np.asarray([post_smooth["sd"]], dtype=float),
        "late_post_step_cst_smooth_mean": np.asarray([post_smooth["mean"]], dtype=float),
        "late_post_step_cst_smooth_median": np.asarray([post_smooth["median"]], dtype=float),
        "late_post_step_cst_smooth_sd": np.asarray([post_smooth["sd"]], dtype=float),
        "delta_cst_smooth_raw": np.asarray([raw_delta_smooth], dtype=float),
        "response_gain_cst_smooth_raw_per_nA": np.asarray([raw_gain_smooth], dtype=float),
        "response_gain_cst_smooth_raw_per_uA": np.asarray([raw_gain_smooth_per_uA], dtype=float),
        "response_gain_cst_per_nA": np.asarray([raw_gain_smooth], dtype=float),
        "response_gain_cst_per_uA": np.asarray([raw_gain_smooth_per_uA], dtype=float),
    })
    if cst_smooth_kernel is not None:
        rows.update({
            "cst_smooth_causal_hann": cst_smooth,
            "cst_smooth_causal_hann_kernel": np.asarray(cst_smooth_kernel, dtype=float),
            "cst_smooth_causal_hann_half_width_s": np.asarray([float(params.step_response_cst_hann_half_width_s)], dtype=float),
            "cst_smooth_causal_hann_half_width_samples": np.asarray([int(cst_smooth_kernel.size - 1)], dtype=int),
            "cst_smooth_causal_hann_kernel_samples": np.asarray([int(cst_smooth_kernel.size)], dtype=int),
        })
    cst_lf = np.asarray(cst_diagnostics["cst_lf"], dtype=float)
    pre_cst_lf_mean, pre_cst_lf_sd, pre_lf_mask = _window_stats_from_trace(cst_time_s, cst_lf, *pre_win)
    post_cst_lf_mean, post_cst_lf_sd, post_lf_mask = _window_stats_from_trace(cst_time_s, cst_lf, *late_post_win)
    delta_cst_lf = float(post_cst_lf_mean - pre_cst_lf_mean) if np.isfinite(pre_cst_lf_mean) and np.isfinite(post_cst_lf_mean) else np.nan
    delta_fr = float(rows["post_step_mean_firing_rate_hz"][0] - rows["pre_step_mean_firing_rate_hz"][0])
    gain_cst = np.nan if abs(step_amp_nA) < 1e-12 else delta_cst_lf / step_amp_nA
    gain_fr = np.nan if abs(step_amp_nA) < 1e-12 else delta_fr / step_amp_nA
    response_values = cst_lf[post_lf_mask]
    if response_values.size and np.isfinite(delta_cst_lf):
        overshoot = (
            float(np.nanmax(response_values) - post_cst_lf_mean)
            if delta_cst_lf >= 0
            else float(post_cst_lf_mean - np.nanmin(response_values))
        )
    else:
        overshoot = np.nan

    rows.update({
        "time_s": cst_time_s,
        "time_rel_step_s": cst_rel_s,
        "step_pre_mask_cst_lf": pre_lf_mask.astype(float),
        "step_late_post_mask_cst_lf": post_lf_mask.astype(float),
        "cst_lf": cst_lf,
        "pre_step_cst_lf_mean": np.asarray([pre_cst_lf_mean], dtype=float),
        "pre_step_cst_lf_sd": np.asarray([pre_cst_lf_sd], dtype=float),
        "post_step_cst_lf_mean": np.asarray([post_cst_lf_mean], dtype=float),
        "post_step_cst_lf_sd": np.asarray([post_cst_lf_sd], dtype=float),
        "delta_cst_lf": np.asarray([delta_cst_lf], dtype=float),
        "delta_firing_rate": np.asarray([delta_fr], dtype=float),
        "response_gain_cst_lf_per_nA": np.asarray([gain_cst], dtype=float),
        "response_gain_fr_per_nA": np.asarray([gain_fr], dtype=float),
        "overshoot_cst_lf": np.asarray([overshoot], dtype=float),
    })

    if "cst_hann" in cst_diagnostics:
        cst_hann = np.asarray(cst_diagnostics["cst_hann"], dtype=float)
        pre_hann_mean, pre_hann_sd, hann_pre_mask = _window_stats_from_trace(cst_time_s, cst_hann, *pre_win)
        post_hann_mean, post_hann_sd, hann_post_mask = _window_stats_from_trace(cst_time_s, cst_hann, *late_post_win)
        delta_cst_hann = (
            float(post_hann_mean - pre_hann_mean)
            if np.isfinite(pre_hann_mean) and np.isfinite(post_hann_mean)
            else np.nan
        )
        gain_cst_hann = np.nan if abs(step_amp_nA) < 1e-12 else delta_cst_hann / step_amp_nA
        hann_response_values = cst_hann[hann_post_mask]
        if hann_response_values.size and np.isfinite(delta_cst_hann):
            overshoot_hann = (
                float(np.nanmax(hann_response_values) - post_hann_mean)
                if delta_cst_hann >= 0
                else float(post_hann_mean - np.nanmin(hann_response_values))
            )
        else:
            overshoot_hann = np.nan
        rows.update({
            "cst_hann": cst_hann,
            "step_pre_mask_cst_hann": hann_pre_mask.astype(float),
            "step_late_post_mask_cst_hann": hann_post_mask.astype(float),
            "pre_step_cst_hann_mean": np.asarray([pre_hann_mean], dtype=float),
            "pre_step_cst_hann_sd": np.asarray([pre_hann_sd], dtype=float),
            "post_step_cst_hann_mean": np.asarray([post_hann_mean], dtype=float),
            "post_step_cst_hann_sd": np.asarray([post_hann_sd], dtype=float),
            "delta_cst_hann": np.asarray([delta_cst_hann], dtype=float),
            "response_gain_cst_hann_per_nA": np.asarray([gain_cst_hann], dtype=float),
            "overshoot_cst_hann": np.asarray([overshoot_hann], dtype=float),
        })

    for prefix, key in (("alpha", "cst_alpha_envelope"), ("beta", "cst_beta_envelope")):
        if key in cst_diagnostics:
            trace = np.asarray(cst_diagnostics[key], dtype=float)
            pre_mean, pre_sd, _ = _window_stats_from_trace(cst_time_s, trace, *pre_win)
            post_mean, post_sd, _ = _window_stats_from_trace(cst_time_s, trace, *late_post_win)
            rows[f"pre_step_cst_{prefix}_envelope_mean"] = np.asarray([pre_mean], dtype=float)
            rows[f"pre_step_cst_{prefix}_envelope_sd"] = np.asarray([pre_sd], dtype=float)
            rows[f"post_step_cst_{prefix}_envelope_mean"] = np.asarray([post_mean], dtype=float)
            rows[f"post_step_cst_{prefix}_envelope_sd"] = np.asarray([post_sd], dtype=float)

    if sync_diagnostics is not None and "sync_trace" in sync_diagnostics:
        sync_trace = np.asarray(sync_diagnostics["sync_trace"], dtype=float)
        if "t_sync_absolute_sec" in sync_diagnostics:
            sync_time_s = np.asarray(sync_diagnostics["t_sync_absolute_sec"], dtype=float)
        else:
            cue_time_s = float(sync_diagnostics.get("cue_time_sec", step_abs_s))
            sync_time_s = cue_time_s + np.asarray(sync_diagnostics["t_sync"], dtype=float)
        pre_sync_mean, pre_sync_sd, sync_pre_mask = _window_stats_from_trace(sync_time_s, sync_trace, *pre_win)
        post_sync_mean, post_sync_sd, sync_late_post_mask = _window_stats_from_trace(sync_time_s, sync_trace, *late_post_win)
        rows.update({
            "sync_time_s": sync_time_s,
            "sync_time_rel_step_s": sync_time_s - step_abs_s,
            "sync_trace": sync_trace,
            "step_sync_reference_event": "step_current",
            "step_sync_analysis_window_rel_s": np.asarray(getattr(params, "step_sync_analysis_window_rel_s", params.step_diagnostic_display_window_rel_s), dtype=float),
            "sync_step_pre_mask": sync_pre_mask.astype(float),
            "sync_step_late_post_mask": sync_late_post_mask.astype(float),
            "pre_step_sync_mean": np.asarray([pre_sync_mean], dtype=float),
            "pre_step_sync_sd": np.asarray([pre_sync_sd], dtype=float),
            "post_step_sync_mean": np.asarray([post_sync_mean], dtype=float),
            "post_step_sync_sd": np.asarray([post_sync_sd], dtype=float),
            "baseline_sync_mean": np.asarray([pre_sync_mean], dtype=float),
            "baseline_sync_sd": np.asarray([pre_sync_sd], dtype=float),
            "post_step_sync_mean_actual_step": np.asarray([post_sync_mean], dtype=float),
            "post_step_sync_sd_actual_step": np.asarray([post_sync_sd], dtype=float),
        })
    else:
        for key in (
            "pre_step_sync_mean", "pre_step_sync_sd", "post_step_sync_mean",
            "post_step_sync_sd", "baseline_sync_mean", "baseline_sync_sd",
            "post_step_sync_mean_actual_step", "post_step_sync_sd_actual_step"
        ):
            rows[key] = np.asarray([np.nan], dtype=float)
    return rows


def build_compact_save_payload(
        params,
        common_input_MN,
        input_components,
        driving_inputs,
        forces,
        cst_diagnostics,
        sync_diagnostics):
    save_trace_fsamp_hz = float(params.fsamp)
    sync_saved_fsamp_hz = None
    common_input_to_save = {k: _copy_value_for_save(v) for k, v in (common_input_MN or {}).items()}
    input_components_to_save = {k: _copy_value_for_save(v) for k, v in (input_components or {}).items()}
    driving_inputs_to_save = {k: _copy_value_for_save(v) for k, v in (driving_inputs or {}).items()}
    forces_to_save = None if forces is None else {k: _copy_value_for_save(v) for k, v in forces.items()}
    cst_to_save = None if cst_diagnostics is None else {k: _copy_value_for_save(v) for k, v in cst_diagnostics.items()}
    sync_to_save = None if sync_diagnostics is None else {k: _copy_value_for_save(v) for k, v in sync_diagnostics.items()}

    if not params.minimal_output:
        return {
            "common_input_MN": common_input_to_save,
            "input_components": input_components_to_save,
            "driving_inputs": driving_inputs_to_save,
            "forces": forces_to_save,
            "cst_diagnostics": cst_to_save,
            "sync_diagnostics": sync_to_save,
            "saved_trace_fsamp_hz": save_trace_fsamp_hz,
            "sync_saved_fsamp_hz": sync_saved_fsamp_hz,
        }

    if cst_to_save is not None:
        for key in ("cst_normalized", "time_s", "analysis_mask", "zoom_mask"):
            cst_to_save.pop(key, None)

    if params.minimal_output_downsample_saved_traces:
        target_fsamp_hz = float(params.minimal_output_target_fsamp)
        source_n_samples = None
        if common_input_to_save:
            first_common = next(iter(common_input_to_save.values()))
            source_n_samples = int(np.asarray(first_common).shape[-1])
        common_input_to_save = _compact_trace_dict(
            common_input_to_save,
            source_fsamp_hz=params.fsamp,
            target_fsamp_hz=target_fsamp_hz,
            source_n_samples=source_n_samples,
        )
        input_components_to_save = _compact_trace_dict(
            input_components_to_save,
            source_fsamp_hz=params.fsamp,
            target_fsamp_hz=target_fsamp_hz,
            source_n_samples=source_n_samples,
        )
        driving_inputs_to_save = _compact_trace_dict(
            driving_inputs_to_save,
            source_fsamp_hz=params.fsamp,
            target_fsamp_hz=target_fsamp_hz,
            source_n_samples=source_n_samples,
        )
        if forces_to_save is not None:
            compact_forces = {}
            for key, value in forces_to_save.items():
                if key == "per_pool" and isinstance(value, dict):
                    compact_forces[key] = {
                        sub_key: (
                            _compact_trace_dict(
                                {sub_key: sub_value},
                                source_fsamp_hz=params.fsamp,
                                target_fsamp_hz=target_fsamp_hz,
                                source_n_samples=source_n_samples,
                            )[sub_key]
                            if sub_key in ("force", "force_percent")
                            else _copy_value_for_save(sub_value)
                        )
                        for sub_key, sub_value in value.items()
                    }
                elif _is_trace_like_array(value, source_n_samples=source_n_samples):
                    compact_forces[key] = _resample_array_last_axis(value, params.fsamp, target_fsamp_hz)
                else:
                    compact_forces[key] = _copy_value_for_save(value)
            forces_to_save = compact_forces
        if cst_to_save is not None:
            cst_to_save = _compact_trace_dict(
                cst_to_save,
                source_fsamp_hz=params.fsamp,
                target_fsamp_hz=target_fsamp_hz,
                source_n_samples=source_n_samples,
            )
        save_trace_fsamp_hz = target_fsamp_hz

        if sync_to_save is not None and sync_to_save.get("t_sync") is not None:
            sync_time = np.asarray(sync_to_save["t_sync"], dtype=float)
            trace_keys = {
                key: value
                for key, value in sync_to_save.items()
                if key != "t_sync"
            }
            sync_time_new, trace_keys_new = _resample_uniform_time_and_traces(
                sync_time,
                trace_keys,
                target_fsamp_hz=target_fsamp_hz,
            )
            sync_to_save = {"t_sync": sync_time_new, **trace_keys_new}
            if sync_time_new.size > 1:
                sync_saved_fsamp_hz = 1.0 / float(np.median(np.diff(sync_time_new)))
        elif sync_to_save is not None and sync_to_save.get("t_sync") is not None and np.asarray(sync_to_save["t_sync"]).size > 1:
            sync_saved_fsamp_hz = 1.0 / float(np.median(np.diff(np.asarray(sync_to_save["t_sync"], dtype=float))))

    if sync_saved_fsamp_hz is None and sync_to_save is not None and sync_to_save.get("t_sync") is not None:
        sync_time = np.asarray(sync_to_save["t_sync"], dtype=float)
        if sync_time.size > 1:
            sync_saved_fsamp_hz = 1.0 / float(np.median(np.diff(sync_time)))

    return {
        "common_input_MN": common_input_to_save,
        "input_components": input_components_to_save,
        "driving_inputs": driving_inputs_to_save,
        "forces": forces_to_save,
        "cst_diagnostics": cst_to_save,
        "sync_diagnostics": sync_to_save,
        "saved_trace_fsamp_hz": save_trace_fsamp_hz,
        "sync_saved_fsamp_hz": sync_saved_fsamp_hz,
    }

######################################
### HELPER FUNCTIONS
######################################
def lerp(a, b, t):
    return a + t * (b - a)
# ───────────────────
# Filtering functions
def butter_lowpass(cutoff, fs, order):
    """
    Return an SOS filter for a lowpass Butterworth of the given order.
    """
    nyq   = 0.5 * fs
    Wn    = cutoff / nyq
    # design in SOS form
    sos   = butter(order, Wn,
                   btype='low',
                   analog=False,
                   output='sos')
    return sos
def butter_highpass(cutoff, fs, order):
    """
    Return an SOS filter for a highpass Butterworth of the given order.
    """
    nyq   = 0.5 * fs
    Wn    = cutoff / nyq
    sos   = butter(order, Wn,
                   btype='high',
                   analog=False,
                   output='sos')
    return sos
# public filter‐wrappers
def lowpass_filter(data, cutoff, fs):
    p = _get_filter_params()
    order = p.default_freq_filter_order
    if p.scale_filter_order_to_frequency:
        if cutoff > p.lowest_freq_filter_order:
            order = cutoff * p.filter_order_scaling_coeff
    if ~np.isfinite(order):
        order = p.default_freq_filter_order
    elif order > p.max_freq_filter_order:
        order = p.max_freq_filter_order
    elif order < p.lowest_freq_filter_order:
        order = p.lowest_freq_filter_order
    order = np.floor(order).astype(int)
    sos = butter_lowpass(cutoff, fs, order=order)
    return sosfiltfilt(sos, data)
def highpass_filter(data, cutoff, fs, order=None):
    p = _get_filter_params()
    order = p.default_freq_filter_order
    if p.scale_filter_order_to_frequency:
        if cutoff > p.lowest_freq_filter_order:
            order = cutoff * p.filter_order_scaling_coeff
    if ~np.isfinite(order):
        order = p.default_freq_filter_order
    elif order > p.max_freq_filter_order:
        order = p.max_freq_filter_order
    elif order < p.lowest_freq_filter_order:
        order = p.lowest_freq_filter_order
    order = np.floor(order).astype(int)
    sos = butter_highpass(cutoff, fs, order=order)
    return sosfiltfilt(sos, data)
# ───────────────────────
def filter_artifact_removal(fsamp, edges_ignore_duration):
    duration_to_remove = edges_ignore_duration # in second
    Wind_s = duration_to_remove * 2
    artifact_removal_window = windows.hann(round(fsamp * Wind_s))
    artifact_removal_window = artifact_removal_window[:int(np.round(len(artifact_removal_window)/2))]
    nb_samples_artifact_removal_window = len(artifact_removal_window)
    return artifact_removal_window, nb_samples_artifact_removal_window
def scale_to_band_std(signal, fs, band, desired_std):
    """
    Scale `signal` so that its *power* (variance) in [fmin,fmax] ⟶ (desired_std)^2.
    Uses Welch's PSD + frequency‐bin masking.  Avoids filtfilt entirely.

    Parameters
    ----------
    signal : 1D array
    fs : float
        Sampling rate (Hz)
    band : (fmin, fmax)
    desired_std : float
        Target standard deviation within [fmin,fmax]
    nperseg : int
        segment length for Welch.  If signal_length < nperseg, welch auto‐truncates.

    Returns
    -------
    scaled_signal : ndarray, same shape as `signal`
    """
    fmin, fmax = band
    if fmin < 0 or fmax >= fs/2 or fmin >= fmax:
        raise ValueError("band must be [fmin, fmax] with 0≤fmin<fmax<fs/2")

    # 1) full‐PSD
    freqs, Pxx = welch(signal, fs=fs, window='hann', nperseg=fs, scaling='spectrum')

    # 2) mask to only [fmin, fmax]
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        raise ValueError(f"No PSD bins in [{fmin},{fmax}] Hz; choose a smaller nperseg or adjust band.")

    current_power = np.trapz(Pxx[mask], freqs[mask])    # = ∫_{fmin}^{fmax} PSD(f) df
    if current_power <= 0:
        raise ValueError(f"No power in the {fmin}–{fmax} Hz band to scale.")

    # 3) we want variance in [fmin,fmax] = (desired_std)^2
    desired_power = desired_std**2
    scale_factor = np.sqrt(desired_power / current_power)

    # 4) apply to entire time-series
    return signal * scale_factor
def Generate_filtered_gaussian_noise_input(fsamp, 
        duration_with_ignored_window, edges_ignore_duration, artifact_removal_window,
        low_pass_filter_cutoff, input_mean, scaling_std, high_pass_filter_cutoff=0):
    # Generate random input
    temp_input = np.random.normal(0, 1, int(duration_with_ignored_window * fsamp))
    # Apply artifact removal window to the end of the signal
    end_ignore_start = int(duration_with_ignored_window * fsamp) - int(np.round(edges_ignore_duration * fsamp))
    temp_input[end_ignore_start:] = temp_input[end_ignore_start:] * np.flip(artifact_removal_window)
    # Apply artifact removal window to the beginning of the signal
    beginning_ignore_end = int(np.round(edges_ignore_duration * fsamp))
    temp_input[:beginning_ignore_end] = temp_input[:beginning_ignore_end] * artifact_removal_window
    # Apply low-pass filter
    temp_input = lowpass_filter(temp_input, low_pass_filter_cutoff, fsamp)
    # Apply high-pass filter
    if high_pass_filter_cutoff >= 1:
        temp_input = highpass_filter(temp_input, high_pass_filter_cutoff, fsamp)
    # Normalize the signal
    temp_input = temp_input - np.mean(temp_input)
    temp_input = temp_input / np.std(temp_input)
    # Scale and add mean
    temp_input = temp_input * scaling_std
    temp_input = temp_input + input_mean
    
    return temp_input
def standardize_signal(x, eps=1e-12, name="signal"):
    """
    Mean-center and scale to unit standard deviation.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    sd = float(np.std(x))
    if sd < eps:
        raise ValueError(f"Cannot standardize {name}: near-zero standard deviation ({sd:.3e})")
    return x / sd
def standardize_signal_with_reference(x, reference_mask=None, eps=1e-12, name="signal"):
    """
    Mean-center and scale using statistics computed on a reference segment.
    The same affine transform is then applied to the whole signal.
    """
    x = np.asarray(x, dtype=float)
    if reference_mask is None:
        ref = x
    else:
        reference_mask = np.asarray(reference_mask, dtype=bool)
        if reference_mask.shape != x.shape:
            raise ValueError(f"{name} reference_mask must have the same shape as the signal")
        if not np.any(reference_mask):
            raise ValueError(f"{name} reference_mask is empty")
        ref = x[reference_mask]
    ref_mean = float(np.mean(ref))
    ref_sd = float(np.std(ref))
    if ref_sd < eps:
        raise ValueError(f"Cannot standardize {name}: near-zero reference standard deviation ({ref_sd:.3e})")
    return (x - ref_mean) / ref_sd, ref_mean, ref_sd
def _resolve_filter_order(reference_cutoff_hz):
    """
    Keep filter-order behavior consistent with the rest of the simulator while
    exposing the logic in one explicit helper.
    """
    p = _get_filter_params()
    order = p.default_freq_filter_order
    if p.scale_filter_order_to_frequency and reference_cutoff_hz > p.lowest_freq_filter_order:
        order = reference_cutoff_hz * p.filter_order_scaling_coeff
    if not np.isfinite(order):
        order = p.default_freq_filter_order
    elif order > p.max_freq_filter_order:
        order = p.max_freq_filter_order
    elif order < p.lowest_freq_filter_order:
        order = p.lowest_freq_filter_order
    return int(np.floor(order))
def generate_band_limited_noise(n_samples, dt, band_hz, rng, padding_s=2.0, name="band_limited_noise"):
    """
    Generate padded Gaussian noise, filter it with zero-phase SOS filtering, crop
    back to the requested length, then standardize it.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be strictly positive")

    fs = 1.0 / float(dt)
    low_hz = float(band_hz[0])
    high_hz = float(band_hz[1])
    nyquist = fs / 2.0
    if low_hz < 0 or low_hz >= high_hz or high_hz >= nyquist:
        raise ValueError(
            f"{name} band must satisfy 0 <= low < high < Nyquist ({nyquist:.3f} Hz), got {band_hz}"
        )

    pad_samples = int(np.ceil(float(padding_s) * fs))
    n_total = n_samples + 2 * pad_samples
    raw = rng.normal(0.0, 1.0, n_total)

    lowpass_threshold_hz = max(1e-9, 0.05 * high_hz)
    use_lowpass = low_hz <= lowpass_threshold_hz
    reference_cutoff_hz = high_hz if use_lowpass else low_hz
    order = _resolve_filter_order(reference_cutoff_hz)
    if use_lowpass:
        sos = butter(order, high_hz, btype='lowpass', fs=fs, output='sos')
    else:
        sos = butter(order, [low_hz, high_hz], btype='bandpass', fs=fs, output='sos')
    filtered = sosfiltfilt(sos, raw)
    cropped = filtered[pad_samples:pad_samples + n_samples]
    return standardize_signal(cropped, name=name)
def generate_low_frequency_component(
        n_samples, dt, target_sd, band_hz, rng,
        padding_s=2.0, baseline_mask=None, burst_kernel_nA=None, trend_kernel_nA=None):
    lf_raw_unit = generate_band_limited_noise(
        n_samples=n_samples,
        dt=dt,
        band_hz=band_hz,
        rng=rng,
        padding_s=padding_s,
        name="lf_raw_unit",
    )
    lf_carrier_unit, baseline_mean_no_burst, baseline_sd_no_burst = standardize_signal_with_reference(
        lf_raw_unit,
        reference_mask=baseline_mask,
        name="lf_raw_unit",
    )
    if burst_kernel_nA is None:
        burst_kernel_nA = np.zeros(n_samples, dtype=float)
    else:
        burst_kernel_nA = np.asarray(burst_kernel_nA, dtype=float)
    if trend_kernel_nA is None:
        trend_kernel_nA = np.zeros(n_samples, dtype=float)
    else:
        trend_kernel_nA = np.asarray(trend_kernel_nA, dtype=float)
    # LF is already the slow modulation itself, so its burst kernel is added
    # directly to the LF component rather than scaling it as an envelope.
    lf_final = float(target_sd) * lf_carrier_unit
    lf_final_with_trend = lf_final + trend_kernel_nA
    lf_final_with_burst = lf_final_with_trend + burst_kernel_nA
    base_envelope_nA = np.zeros(n_samples, dtype=float)
    effective_envelope_nA = np.zeros(n_samples, dtype=float)
    return {
        "lf_raw_unit": lf_raw_unit,
        "lf_carrier_unit": lf_carrier_unit,
        "lf_envelope_nA": base_envelope_nA,
        "lf_effective_envelope_nA": effective_envelope_nA,
        "lf_burst_kernel_nA": burst_kernel_nA,
        "lf_trend_kernel_nA": trend_kernel_nA,
        "lf_final": lf_final,
        "lf_final_with_trend": lf_final_with_trend,
        "lf_final_with_burst": lf_final_with_burst,
        "lf_trend_only_delta_nA": trend_kernel_nA,
        "lf_burst_only_delta_nA": burst_kernel_nA,
        "lf_reference_mean": np.array([baseline_mean_no_burst], dtype=float),
        "lf_reference_sd": np.array([baseline_sd_no_burst], dtype=float),
    }
def generate_enveloped_band_component(
        n_samples, dt,
        carrier_band_hz, envelope_band_hz,
        target_sd_final,
        rng,
        padding_s=2.0,
        component_name="alpha",
        baseline_mask=None,
        burst_kernel_nA=None,
        trend_kernel_nA=None,
        envelope_variability_scale=1.0):
    carrier_key = f"{component_name}_carrier_unit"
    env_driver_key = f"{component_name}_env_driver_unit"
    env_multiplier_key = f"{component_name}_env_multiplier"
    env_multiplier_with_burst_key = f"{component_name}_env_multiplier_with_burst"
    burst_kernel_key = f"{component_name}_burst_kernel_nA"
    trend_kernel_key = f"{component_name}_trend_kernel_nA"
    envelope_nA_key = f"{component_name}_envelope_nA"
    effective_envelope_nA_key = f"{component_name}_effective_envelope_nA"
    mod_unscaled_key = f"{component_name}_modulated_unscaled"
    mod_unit_key = f"{component_name}_modulated_unit"
    final_key = f"{component_name}_final"
    final_with_trend_key = f"{component_name}_final_with_trend"
    final_with_burst_key = f"{component_name}_final_with_burst"
    reference_mean_key = f"{component_name}_reference_mean"
    reference_sd_key = f"{component_name}_reference_sd"
    envelope_variability_scale = float(envelope_variability_scale)
    if not (0.0 <= envelope_variability_scale <= 1.0):
        raise ValueError(f"{component_name} envelope_variability_scale must be between 0 and 1")

    carrier_unit = generate_band_limited_noise(
        n_samples=n_samples,
        dt=dt,
        band_hz=carrier_band_hz,
        rng=rng,
        padding_s=padding_s,
        name=carrier_key,
    )
    env_driver_unit = generate_band_limited_noise(
        n_samples=n_samples,
        dt=dt,
        band_hz=envelope_band_hz,
        rng=rng,
        padding_s=padding_s,
        name=env_driver_key,
    )
    env_multiplier_raw = np.exp(env_driver_unit)
    if baseline_mask is None:
        env_multiplier_full = env_multiplier_raw / np.mean(env_multiplier_raw)
    else:
        env_multiplier_full = env_multiplier_raw / np.mean(env_multiplier_raw[np.asarray(baseline_mask, dtype=bool)])
    env_multiplier = 1.0 + envelope_variability_scale * (env_multiplier_full - 1.0)
    modulated_unscaled = env_multiplier * carrier_unit
    modulated_unit, reference_mean, reference_sd = standardize_signal_with_reference(
        modulated_unscaled,
        reference_mask=baseline_mask,
        name=mod_unit_key,
    )
    envelope_scale_nA = 0.0 if float(target_sd_final) == 0.0 else float(target_sd_final) / float(reference_sd)
    envelope_nA = envelope_scale_nA * env_multiplier
    if burst_kernel_nA is None:
        burst_kernel_nA = np.zeros(n_samples, dtype=float)
    else:
        burst_kernel_nA = np.asarray(burst_kernel_nA, dtype=float)
    if trend_kernel_nA is None:
        trend_kernel_nA = np.zeros(n_samples, dtype=float)
    else:
        trend_kernel_nA = np.asarray(trend_kernel_nA, dtype=float)
    effective_envelope_trend_only_nA = np.maximum(envelope_nA + trend_kernel_nA, 0.0)
    effective_envelope_nA = np.maximum(effective_envelope_trend_only_nA + burst_kernel_nA, 0.0)
    env_multiplier_with_burst = np.divide(
        effective_envelope_nA,
        envelope_scale_nA,
        out=np.zeros_like(effective_envelope_nA),
        where=np.abs(envelope_scale_nA) > 1e-12,
    )
    modulated_unscaled_with_trend = effective_envelope_trend_only_nA * carrier_unit
    modulated_unscaled_with_burst = effective_envelope_nA * carrier_unit
    final_base = envelope_nA * carrier_unit
    final_with_trend = modulated_unscaled_with_trend
    final_with_burst = modulated_unscaled_with_burst
    return {
        carrier_key: carrier_unit,
        env_driver_key: env_driver_unit,
        env_multiplier_key: env_multiplier,
        env_multiplier_with_burst_key: env_multiplier_with_burst,
        f"{component_name}_env_multiplier_full_variability": env_multiplier_full,
        f"{component_name}_env_variability_scale": np.asarray([envelope_variability_scale], dtype=float),
        burst_kernel_key: burst_kernel_nA,
        trend_kernel_key: trend_kernel_nA,
        envelope_nA_key: envelope_nA,
        effective_envelope_nA_key: effective_envelope_nA,
        mod_unscaled_key: modulated_unscaled,
        mod_unit_key: modulated_unit,
        final_key: final_base,
        final_with_trend_key: final_with_trend,
        final_with_burst_key: final_with_burst,
        f"{component_name}_trend_only_delta_nA": final_with_trend - final_base,
        f"{component_name}_burst_only_delta_nA": final_with_burst - final_with_trend,
        reference_mean_key: np.array([reference_mean], dtype=float),
        reference_sd_key: np.array([reference_sd], dtype=float),
    }
def compute_psd_welch(signal, fsamp):
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("compute_psd_welch expects a 1D signal")
    nperseg = min(len(signal), max(32, int(fsamp)))
    freqs, psd = welch(
        signal,
        fs=fsamp,
        window='hann',
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling='density',
        detrend='constant',
    )
    return freqs, psd
def get_input_component_colors(params):
    colors = {
        "lf": "#2ca02c",
        "alpha": "#1f77b4",
        "beta": "#9467bd",
        "common": "#111111",
        "cst": "#ff7f0e",
        "envelope": "#444444",
        "trend": "#8c8c8c",
        "burst": "#FF0080",
        "cue": "#7a7a7a",
    }
    user_colors = getattr(params, "input_component_colors", None)
    if isinstance(user_colors, dict):
        colors.update({str(k): str(v) for k, v in user_colors.items()})
    return colors
def resolve_diagnostic_zoom_window(params, t_s, window_start_s, window_end_s, burst_center_s=None):
    t_s = np.asarray(t_s, dtype=float)
    if window_start_s >= window_end_s:
        raise ValueError("Diagnostic zoom window bounds are empty or inverted")
    default_zoom_duration_s = max(0.1, window_end_s - window_start_s)
    requested_zoom_duration_s = getattr(params, "input_component_diag_zoom_duration_s", default_zoom_duration_s)
    zoom_duration_s = min(float(requested_zoom_duration_s), default_zoom_duration_s)
    burst_start_s = None
    if burst_center_s is not None:
        burst_center_s = float(burst_center_s)
        burst_sigma_s = float(params.input_burst_sigma_ms) / 1000.0
        burst_start_s = burst_center_s - float(params.input_burst_start_marker_n_sigma) * burst_sigma_s
        zoom_start_s = burst_center_s - float(params.input_burst_zoom_start_n_sigma) * burst_sigma_s
    else:
        zoom_start_s = 0.5 * (window_start_s + window_end_s) - zoom_duration_s / 2.0
    zoom_start_s = max(window_start_s, zoom_start_s)
    zoom_end_s = min(window_end_s, zoom_start_s + zoom_duration_s)
    zoom_start_s = max(window_start_s, zoom_end_s - zoom_duration_s)
    zoom_mask = (t_s >= zoom_start_s) & (t_s <= zoom_end_s)
    return {
        "zoom_start_s": float(zoom_start_s),
        "zoom_end_s": float(zoom_end_s),
        "zoom_mask": zoom_mask,
        "burst_center_s": None if burst_center_s is None else float(burst_center_s),
        "burst_start_s": None if burst_start_s is None else float(burst_start_s),
        "burst_end_s": None if burst_center_s is None else float(burst_center_s + float(params.input_burst_start_marker_n_sigma) * float(params.input_burst_sigma_ms) / 1000.0),
    }
def _draw_diagnostic_markers(
        ax,
        task_event_times_s=None,
        burst_center_s=None,
        burst_start_s=None,
        burst_end_s=None,
        colors=None,
        time_offset_s=0.0,
        show_task_cue_markers=True,
        show_step_current_marker=False):
    task_event_times_s = task_event_times_s or {}
    colors = colors or {}
    cue_color = colors.get("cue", "0.5")
    burst_color = colors.get("burst", "#FF0080")
    step_color = colors.get("step", "#111111")
    get_ready_cue_s = task_event_times_s.get("get_ready_cue_s")
    go_nogo_cue_s = task_event_times_s.get("go_nogo_cue_s")
    step_current_s = task_event_times_s.get("step_current_s")
    if burst_start_s is not None and burst_end_s is not None:
        ax.axvspan(float(burst_start_s) - time_offset_s, float(burst_end_s) - time_offset_s, color=burst_color, alpha=0.10, zorder=0)
    if show_task_cue_markers and get_ready_cue_s is not None:
        ax.axvline(float(get_ready_cue_s) - time_offset_s, color=cue_color, lw=1.0, ls="--", alpha=0.8, zorder=1)
    if show_task_cue_markers and go_nogo_cue_s is not None:
        ax.axvline(float(go_nogo_cue_s) - time_offset_s, color=cue_color, lw=1.1, ls="-", alpha=0.9, zorder=1)
    if burst_center_s is not None:
        ax.axvline(float(burst_center_s) - time_offset_s, color=burst_color, lw=1.1, ls="--", zorder=1)
    if show_step_current_marker and step_current_s is not None:
        ax.axvline(float(step_current_s) - time_offset_s, color=step_color, lw=1.2, ls="--", alpha=0.95, zorder=1)
def _normalize_for_diagnostic_plot(signal):
    signal = np.asarray(signal, dtype=float)
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        return np.zeros_like(signal), 1.0
    peak = float(np.max(np.abs(finite)))
    if peak < 1e-12:
        return np.zeros_like(signal), 0.0
    return signal / peak, peak
def _normalized_envelope_pair(envelope_signal):
    envelope_signal = np.asarray(envelope_signal, dtype=float)
    finite = envelope_signal[np.isfinite(envelope_signal)]
    if finite.size == 0:
        zeros = np.zeros_like(envelope_signal)
        return zeros, zeros
    peak = float(np.max(finite))
    if peak < 1e-12:
        zeros = np.zeros_like(envelope_signal)
        return zeros, zeros
    env = envelope_signal / peak
    return env, -env
def apply_zero_phase_component_filter(signal, fsamp, band_hz, name="filtered_signal"):
    """
    Zero-phase low-pass or band-pass filter using the simulator's filter-order
    convention.
    """
    signal = np.asarray(signal, dtype=float)
    low_hz = float(band_hz[0])
    high_hz = float(band_hz[1])
    if low_hz < 0 or low_hz >= high_hz or high_hz >= (fsamp / 2.0):
        raise ValueError(f"{name} band must satisfy 0 <= low < high < Nyquist, got {band_hz}")
    lowpass_threshold_hz = max(1e-9, 0.05 * high_hz)
    use_lowpass = low_hz <= lowpass_threshold_hz
    reference_cutoff_hz = high_hz if use_lowpass else low_hz
    order = _resolve_filter_order(reference_cutoff_hz)
    if use_lowpass:
        sos = butter(order, high_hz, btype='lowpass', fs=fsamp, output='sos')
    else:
        sos = butter(order, [low_hz, high_hz], btype='bandpass', fs=fsamp, output='sos')
    return sosfiltfilt(sos, signal)


def build_normalized_hann_kernel(window_s, fsamp):
    """Return a unit-area Hann FIR kernel for smoothing rate-like traces."""
    window_s = float(window_s)
    fsamp = float(fsamp)
    if window_s <= 0:
        raise ValueError("Hann smoothing window_s must be strictly positive")
    if fsamp <= 0:
        raise ValueError("fsamp must be strictly positive")
    n_samples = max(3, int(round(window_s * fsamp)))
    kernel = windows.hann(n_samples, sym=True).astype(float)
    kernel_sum = float(np.sum(kernel))
    if not np.isfinite(kernel_sum) or kernel_sum <= 0:
        raise ValueError(
            f"Could not build a valid Hann smoothing kernel for window_s={window_s}, fsamp={fsamp}"
        )
    return kernel / kernel_sum


def apply_hann_filtfilt_smoothing(signal, fsamp, window_s):
    """Smooth a signal with a normalized Hann FIR kernel using zero-phase filtfilt."""
    signal = np.asarray(signal, dtype=float)
    kernel = build_normalized_hann_kernel(window_s, fsamp)
    if signal.ndim != 1:
        raise ValueError("Hann smoothing currently expects a 1D signal")
    if signal.size == 0:
        return signal.copy(), kernel
    smoothed = filtfilt(kernel, [1.0], signal, method="gust")
    return np.asarray(smoothed, dtype=float), kernel


def build_normalized_causal_hann_kernel(half_width_s, fsamp):
    """Return a causal half-Hann kernel indexed by lag, with lag 0 at the current sample."""
    half_width_s = float(half_width_s)
    fsamp = float(fsamp)
    if half_width_s <= 0:
        raise ValueError("causal Hann half_width_s must be strictly positive")
    if fsamp <= 0:
        raise ValueError("fsamp must be strictly positive")
    n_samples = max(2, int(round(half_width_s * fsamp)) + 1)
    lags = np.arange(n_samples, dtype=float)
    if n_samples == 2:
        kernel = np.asarray([1.0, 0.0], dtype=float)
    else:
        kernel = 0.5 * (1.0 + np.cos(np.pi * lags / float(n_samples - 1)))
    kernel_sum = float(np.sum(kernel))
    if not np.isfinite(kernel_sum) or kernel_sum <= 0:
        raise ValueError(
            f"Could not build a valid causal Hann kernel for half_width_s={half_width_s}, fsamp={fsamp}"
        )
    return kernel / kernel_sum


def apply_causal_hann_smoothing(signal, fsamp, half_width_s):
    """Smooth a 1D trace using only current and past samples with a normalized half-Hann kernel."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("causal Hann smoothing expects a 1D signal")
    kernel = build_normalized_causal_hann_kernel(half_width_s, fsamp)
    if signal.size == 0:
        return signal.copy(), kernel
    finite = np.isfinite(signal)
    filled = np.where(finite, signal, 0.0)
    numerator = np.convolve(filled, kernel, mode="full")[: signal.size]
    denominator = np.convolve(finite.astype(float), kernel, mode="full")[: signal.size]
    out = np.full(signal.shape, np.nan, dtype=float)
    valid = denominator > 1e-12
    out[valid] = numerator[valid] / denominator[valid]
    return out, kernel


def apply_causal_exponential_smoothing(signal, fsamp, tau_s):
    """Smooth a 1D trace with a causal one-pole exponential filter."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("causal exponential smoothing expects a 1D signal")
    fsamp = float(fsamp)
    tau_s = float(tau_s)
    if fsamp <= 0:
        raise ValueError("fsamp must be strictly positive")
    if tau_s <= 0:
        raise ValueError("tau_s must be strictly positive")
    if signal.size == 0:
        return signal.copy(), np.nan
    alpha = 1.0 - float(np.exp(-(1.0 / fsamp) / tau_s))
    y = np.empty_like(signal, dtype=float)
    y[0] = signal[0]
    for idx in range(1, signal.size):
        x_val = signal[idx]
        if not np.isfinite(x_val):
            x_val = y[idx - 1]
        y[idx] = alpha * x_val + (1.0 - alpha) * y[idx - 1]
    return y, alpha
def resolve_cst_analysis_window(params):
    """
    Return the CST analysis window in absolute simulation seconds.
    The public parameters are relative to the usable (non-ignored) window.
    """
    usable_start_s = float(params.edges_ignore_duration)
    usable_end_s = float(params.duration_with_ignored_window - params.edges_ignore_duration)
    analysis_start_s = usable_start_s + float(params.cst_analysis_window_start_s)
    if params.cst_analysis_window_end_s <= 0:
        analysis_end_s = usable_end_s
    else:
        analysis_end_s = usable_start_s + float(params.cst_analysis_window_end_s)
    if analysis_start_s >= analysis_end_s:
        raise ValueError("Resolved CST analysis window is empty or inverted")
    if analysis_start_s < usable_start_s or analysis_end_s > usable_end_s:
        raise ValueError("Resolved CST analysis window must stay inside the usable simulation window")
    return analysis_start_s, analysis_end_s
def compute_baseline_normalized_envelope(
        envelope,
        baseline_mask,
        *,
        name,
        logger=None,
        eps_sd=1e-9,
        eps_mean=1e-9):
    """
    Normalize an already-computed Hilbert envelope relative to the baseline period
    only. The z-score and percent traces are both returned so plotting/output can
    choose either representation later without recomputing.
    """
    logger = logger or logging.getLogger(__name__)
    envelope = np.asarray(envelope, dtype=float)
    baseline_mask = np.asarray(baseline_mask, dtype=bool)
    if baseline_mask.shape != envelope.shape:
        raise ValueError(f"{name} baseline_mask must have the same shape as the envelope")
    if not np.any(baseline_mask):
        raise ValueError(f"{name} baseline window is empty")

    baseline_values = envelope[baseline_mask]
    baseline_mean = float(np.mean(baseline_values))
    baseline_sd = float(np.std(baseline_values))

    if baseline_sd < eps_sd:
        logger.warning(
            "%s baseline SD is near zero (%.3e); returning zeros for z-scored modulation",
            name, baseline_sd,
        )
        envelope_z = np.zeros_like(envelope, dtype=float)
    else:
        envelope_z = (envelope - baseline_mean) / baseline_sd

    if abs(baseline_mean) < eps_mean:
        logger.warning(
            "%s baseline mean is near zero (%.3e); returning NaNs for percent modulation",
            name, baseline_mean,
        )
        envelope_pct = np.full_like(envelope, np.nan, dtype=float)
    else:
        envelope_pct = 100.0 * (envelope - baseline_mean) / baseline_mean

    return {
        "baseline_mean": baseline_mean,
        "baseline_sd": baseline_sd,
        "envelope_z": envelope_z,
        "envelope_pct": envelope_pct,
    }
def _normalize_rel_window(bounds):
    arr = np.asarray(bounds, dtype=float).reshape(-1)
    if arr.size != 2 or not np.isfinite(arr).all() or not (arr[0] < arr[1]):
        raise ValueError(f"Invalid cue-relative window: {bounds}")
    return (float(arr[0]), float(arr[1]))


def _normalize_observation_window_defs(window_defs):
    merged = {
        alias: tuple(map(float, bounds))
        for alias, bounds in DEFAULT_OBSERVATION_WINDOW_DEFS_REL_CUE_S.items()
    }
    if window_defs:
        for alias, bounds in dict(window_defs).items():
            merged[str(alias)] = _normalize_rel_window(bounds)
    return {alias: _normalize_rel_window(bounds) for alias, bounds in merged.items()}


def _crop_spike_trains_to_rel_window(spike_trains, cue_time_abs_s, window_rel_cue_s):
    window_start_s = float(cue_time_abs_s) + float(window_rel_cue_s[0])
    window_end_s = float(cue_time_abs_s) + float(window_rel_cue_s[1])
    return [
        np.asarray(spike_train, dtype=float)[
            (np.asarray(spike_train, dtype=float) >= window_start_s)
            & (np.asarray(spike_train, dtype=float) < window_end_s)
        ]
        for spike_train in spike_trains
    ]


def _recompute_cst_modulation_normalization(cst_diagnostics, cue_time_abs_s, params):
    if cst_diagnostics is None or str(cst_diagnostics.get("status", "")).lower() != "ok":
        return cst_diagnostics
    t_s = np.asarray(cst_diagnostics.get("time_s"), dtype=float)
    if t_s.size == 0:
        return cst_diagnostics
    normalization_window_rel_cue_s = _normalize_rel_window(params.cst_modulation_normalization_window_rel_cue_s)
    normalization_start_s = float(cue_time_abs_s) + normalization_window_rel_cue_s[0]
    normalization_end_s = float(cue_time_abs_s) + normalization_window_rel_cue_s[1]
    normalization_mask = (t_s >= normalization_start_s) & (t_s < normalization_end_s)
    if not np.any(normalization_mask):
        raise ValueError(
            "CST modulation normalization window does not contain any samples after clipping. "
            f"Requested cue-relative normalization window={normalization_window_rel_cue_s}."
        )
    alpha_mod = compute_baseline_normalized_envelope(
        np.asarray(cst_diagnostics["cst_alpha_envelope"], dtype=float),
        normalization_mask,
        name="sim_cst_alpha_envelope",
        logger=None,
    )
    beta_mod = compute_baseline_normalized_envelope(
        np.asarray(cst_diagnostics["cst_beta_envelope"], dtype=float),
        normalization_mask,
        name="sim_cst_beta_envelope",
        logger=None,
    )
    cst_diagnostics = dict(cst_diagnostics)
    cst_diagnostics["normalization_window_start_s"] = float(normalization_start_s)
    cst_diagnostics["normalization_window_end_s"] = float(normalization_end_s)
    cst_diagnostics["cst_alpha_baseline_mean"] = np.array([alpha_mod["baseline_mean"]], dtype=float)
    cst_diagnostics["cst_alpha_baseline_sd"] = np.array([alpha_mod["baseline_sd"]], dtype=float)
    cst_diagnostics["cst_alpha_envelope_z"] = np.asarray(alpha_mod["envelope_z"], dtype=float)
    cst_diagnostics["cst_alpha_envelope_pct"] = np.asarray(alpha_mod["envelope_pct"], dtype=float)
    cst_diagnostics["cst_beta_baseline_mean"] = np.array([beta_mod["baseline_mean"]], dtype=float)
    cst_diagnostics["cst_beta_baseline_sd"] = np.array([beta_mod["baseline_sd"]], dtype=float)
    cst_diagnostics["cst_beta_envelope_z"] = np.asarray(beta_mod["envelope_z"], dtype=float)
    cst_diagnostics["cst_beta_envelope_pct"] = np.asarray(beta_mod["envelope_pct"], dtype=float)
    return cst_diagnostics


def _build_sim_analysis_summary_metadata(params):
    return {
        "analysis_version": "sim_analysis_v1",
        "normalization_win_rel_cue_s": _normalize_rel_window(
            params.observation_window_defs_rel_cue_s.get("normalization_win", (-4.0, -3.0))
        ),
        "baseline_win_rel_cue_s": _normalize_rel_window(
            params.observation_window_defs_rel_cue_s.get("baseline_win", (-3.0, -2.0))
        ),
        "baseline_extended_win_rel_cue_s": _normalize_rel_window(
            params.observation_window_defs_rel_cue_s.get("baseline_extended_win", (-2.0, -1.0))
        ),
        "ready_win_rel_cue_s": _normalize_rel_window(
            params.observation_window_defs_rel_cue_s.get("ready_win", (-1.0, 0.0))
        ),
        "post_win_rel_cue_s": _normalize_rel_window(
            params.observation_window_defs_rel_cue_s.get("post_win", (0.0, 1.0))
        ),
        "obs_baseline_window_rel_cue_s": _normalize_rel_window(params.obs_baseline_window_rel_cue_s),
        "cst_modulation_normalization_window_rel_cue_s": _normalize_rel_window(
            params.cst_modulation_normalization_window_rel_cue_s
        ),
        "firing_rate_isi_window_rel_cue_s": _normalize_rel_window(params.firing_rate_isi_window_rel_cue_s),
        "sync_analysis_window_rel_cue_s": _normalize_rel_window(params.sync_analysis_window_rel_cue_s),
        "sync_baseline_win_rel_cue_s": _normalize_rel_window(params.sync_baseline_win_rel_cue_s),
        "sync_post_win_rel_cue_s": _normalize_rel_window(params.sync_post_win_rel_cue_s),
        "modulation_spectrum_window_rel_cue_s": _normalize_rel_window(params.modulation_spectrum_window_rel_cue_s),
        "modulation_spectrum_max_freq_hz": float(params.modulation_spectrum_max_freq_hz),
        "modulation_spectrum_interval_mass_pct": float(params.modulation_spectrum_interval_mass_pct),
        "lf_band_hz": tuple(map(float, np.asarray(params.lf_band_hz, dtype=float).reshape(-1).tolist())),
        "alpha_band_hz": tuple(map(float, np.asarray(params.alpha_band_hz, dtype=float).reshape(-1).tolist())),
        "beta_band_hz": tuple(map(float, np.asarray(params.beta_band_hz, dtype=float).reshape(-1).tolist())),
        "active_unit_rate_threshold_hz": float(params.active_unit_rate_threshold_hz),
        "sync_win_ms": float(params.sync_win_ms),
        "sync_step_ms": float(params.sync_step_ms),
        "sync_coinc_lag_ms": float(params.sync_coinc_lag_ms),
        "sync_direction_mode": str(params.sync_direction_mode),
        "sync_expectation_mode": str(params.sync_expectation_mode),
    }


def compute_cst_diagnostics(
        spike_trains_MN,
        params,
        burst_center_s=None,
        logger=None,
        active_unit_selection=None):
    """
    Build an active-unit-normalized CST and its LF / alpha / beta components from
    the simulated MN spike trains.
    """
    logger = logger or logging.getLogger(__name__)
    analysis_start_s, analysis_end_s = resolve_cst_analysis_window(params)
    window_duration_s = analysis_end_s - analysis_start_s
    if window_duration_s <= 0:
        raise ValueError("CST analysis window must have strictly positive duration")

    if active_unit_selection is None:
        active_unit_selection = classify_active_units_by_rate(
            spike_trains_MN,
            analysis_start_s=analysis_start_s,
            analysis_end_s=analysis_end_s,
            rate_threshold_hz=params.active_unit_rate_threshold_hz,
        )
    active_ids = np.asarray(active_unit_selection.get("active_unit_ids", []), dtype=int).tolist()
    excluded_ids = np.asarray(active_unit_selection.get("excluded_unit_ids", []), dtype=int).tolist()
    active_rates_hz = np.asarray(active_unit_selection.get("active_unit_rates_hz", []), dtype=float).tolist()

    logger.info(
        "CST diagnostics: analysis window %.3f-%.3f s | active threshold %.2f Hz | retained %d/%d MNs",
        analysis_start_s, analysis_end_s,
        params.active_unit_rate_threshold_hz,
        len(active_ids), len(spike_trains_MN),
    )

    if len(active_ids) == 0:
        return {
            "status": "no_active_units",
            "analysis_window_start_s": analysis_start_s,
            "analysis_window_end_s": analysis_end_s,
            "active_unit_ids": np.array([], dtype=int),
            "excluded_unit_ids": np.asarray(excluded_ids, dtype=int),
            "active_unit_rates_hz": np.array([], dtype=float),
        }

    n_samples = int(np.round(params.duration_with_ignored_window * params.fsamp))
    cst_counts = np.zeros(n_samples, dtype=float)
    for unit_i in active_ids:
        spike_idx = np.floor(np.asarray(spike_trains_MN[unit_i], dtype=float) * params.fsamp).astype(int)
        spike_idx = spike_idx[(spike_idx >= 0) & (spike_idx < n_samples)]
        if spike_idx.size > 0:
            np.add.at(cst_counts, spike_idx, 1.0)

    cst_normalized = (cst_counts / float(len(active_ids))) * float(params.fsamp)
    cst_lf = apply_zero_phase_component_filter(cst_normalized, params.fsamp, params.lf_band_hz, name="cst_lf")
    cst_hann = None
    cst_hann_kernel = None
    if bool(getattr(params, "enable_cst_hann_smoothing", False)):
        cst_hann, cst_hann_kernel = apply_hann_filtfilt_smoothing(
            cst_normalized,
            fsamp=params.fsamp,
            window_s=params.cst_hann_smoothing_window_s,
        )
    cst_alpha = apply_zero_phase_component_filter(cst_normalized, params.fsamp, params.alpha_band_hz, name="cst_alpha")
    cst_beta = apply_zero_phase_component_filter(cst_normalized, params.fsamp, params.beta_band_hz, name="cst_beta")
    cst_alpha_envelope = np.abs(hilbert(cst_alpha))
    cst_beta_envelope = np.abs(hilbert(cst_beta))

    t_s = np.arange(n_samples, dtype=float) / float(params.fsamp)
    analysis_mask = (t_s >= analysis_start_s) & (t_s <= analysis_end_s)
    if bool(getattr(params, "enable_step_current", False)) and not bool(getattr(params, "enable_go_nogo_cue_diagnostics", True)):
        step_abs_s = float(params.task_event_times_s["step_current_s"])
        step_pre_rel = _normalize_rel_window(params.step_pre_window_rel_s)
        baseline_start_s = float(step_abs_s + step_pre_rel[0])
        baseline_end_s = float(step_abs_s + step_pre_rel[1])
        baseline_reference_event = "step_current"
        empty_message = "CST step-pre baseline window is empty: check step_pre_window_rel_s and the CST analysis window"
    else:
        baseline_start_s = float(analysis_start_s)
        baseline_end_s = float(params.task_event_times_s["get_ready_cue_s"])
        baseline_reference_event = "get_ready_cue"
        empty_message = "CST baseline window is empty: the analysis window must start before the get-ready cue"
    baseline_mask = (t_s >= max(analysis_start_s, baseline_start_s)) & (t_s < min(analysis_end_s, baseline_end_s))
    if baseline_end_s <= baseline_start_s or not np.any(baseline_mask):
        raise ValueError(empty_message)
    alpha_mod = compute_baseline_normalized_envelope(
        cst_alpha_envelope,
        baseline_mask,
        name="cst_alpha_envelope",
        logger=logger,
    )
    beta_mod = compute_baseline_normalized_envelope(
        cst_beta_envelope,
        baseline_mask,
        name="cst_beta_envelope",
        logger=logger,
    )
    zoom_info = resolve_diagnostic_zoom_window(
        params,
        t_s,
        window_start_s=analysis_start_s,
        window_end_s=analysis_end_s,
        burst_center_s=burst_center_s if burst_center_s is not None and analysis_start_s <= burst_center_s <= analysis_end_s else None,
    )

    diagnostics = {
        "status": "ok",
        "time_s": t_s,
        "analysis_mask": analysis_mask,
        "zoom_mask": zoom_info["zoom_mask"],
        "zoom_start_s": zoom_info["zoom_start_s"],
        "zoom_end_s": zoom_info["zoom_end_s"],
        "burst_center_s": zoom_info["burst_center_s"],
        "burst_start_s": zoom_info["burst_start_s"],
        "analysis_window_start_s": float(analysis_start_s),
        "analysis_window_end_s": float(analysis_end_s),
        "baseline_reference_event": baseline_reference_event,
        "baseline_window_start_s": float(baseline_start_s),
        "baseline_window_end_s": float(baseline_end_s),
        "active_unit_rate_threshold_hz": np.asarray([float(params.active_unit_rate_threshold_hz)], dtype=float),
        "n_units_total": np.asarray([float(len(spike_trains_MN))], dtype=float),
        "n_units_kept": np.asarray([float(len(active_ids))], dtype=float),
        "unit_fraction_kept": np.asarray([float(len(active_ids) / len(spike_trains_MN))], dtype=float),
        "active_unit_ids": np.asarray(active_ids, dtype=int),
        "excluded_unit_ids": np.asarray(excluded_ids, dtype=int),
        "active_unit_rates_hz": np.asarray(active_rates_hz, dtype=float),
        "cst_normalized": cst_normalized,
        "cst_lf": cst_lf,
        "cst_alpha": cst_alpha,
        "cst_alpha_envelope": cst_alpha_envelope,
        "cst_alpha_baseline_mean": np.array([alpha_mod["baseline_mean"]], dtype=float),
        "cst_alpha_baseline_sd": np.array([alpha_mod["baseline_sd"]], dtype=float),
        "cst_alpha_envelope_z": np.asarray(alpha_mod["envelope_z"], dtype=float),
        "cst_alpha_envelope_pct": np.asarray(alpha_mod["envelope_pct"], dtype=float),
        "cst_beta": cst_beta,
        "cst_beta_envelope": cst_beta_envelope,
        "cst_beta_baseline_mean": np.array([beta_mod["baseline_mean"]], dtype=float),
        "cst_beta_baseline_sd": np.array([beta_mod["baseline_sd"]], dtype=float),
        "cst_beta_envelope_z": np.asarray(beta_mod["envelope_z"], dtype=float),
        "cst_beta_envelope_pct": np.asarray(beta_mod["envelope_pct"], dtype=float),
    }
    if cst_hann is not None and cst_hann_kernel is not None:
        diagnostics.update({
            "cst_hann": np.asarray(cst_hann, dtype=float),
            "cst_hann_kernel": np.asarray(cst_hann_kernel, dtype=float),
            "cst_hann_smoothing_window_s": np.asarray([float(params.cst_hann_smoothing_window_s)], dtype=float),
            "cst_hann_smoothing_window_samples": np.asarray([int(cst_hann_kernel.size)], dtype=int),
            "cst_hann_smoothing_method": "hann_fir_filtfilt",
        })
    return diagnostics
def plot_input_component_diagnostics(
        save_dir, figure_name, component_name,
        t_s, zoom_mask, band_hz, target_sd,
        final_signal,
        time_series_specs,
        task_event_times_s=None,
        burst_center_s=None,
        burst_start_s=None,
        burst_end_s=None,
        band_color="tab:red",
        colors=None,
        show_task_cue_markers=True,
        show_step_current_marker=False):
    """
    Plot time-domain diagnostics for one input component plus a full-trace PSD of
    the final no-burst component.
    """
    n_rows = len(time_series_specs) + 1
    fig = plt.figure(figsize=(11, 2.8 * n_rows), dpi=120)
    gs = fig.add_gridspec(n_rows, 1, hspace=0.35)

    for row_i, spec in enumerate(time_series_specs):
        ax = fig.add_subplot(gs[row_i, 0])
        ax.plot(t_s[zoom_mask], np.asarray(spec["signal"], dtype=float)[zoom_mask], color=spec["color"], lw=1.2)
        ax.set_ylabel(spec["ylabel"])
        ax.set_title(spec["title"], fontsize=10)
        ax.grid(alpha=0.25)
        _draw_diagnostic_markers(
            ax,
            task_event_times_s=task_event_times_s,
            burst_center_s=burst_center_s,
            burst_start_s=burst_start_s,
            burst_end_s=burst_end_s,
            colors=colors,
            show_task_cue_markers=show_task_cue_markers,
            show_step_current_marker=show_step_current_marker,
        )

    ax_psd = fig.add_subplot(gs[-1, 0])
    freqs, psd = compute_psd_welch(final_signal, fsamp=1.0 / float(np.mean(np.diff(t_s))))
    ax_psd.plot(freqs, psd, color="black", lw=1.3)
    ax_psd.axvspan(band_hz[0], band_hz[1], color=band_color, alpha=0.15)
    ax_psd.set_xlim(left=0, right=max(40.0, band_hz[1] * 2.0))
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power")
    ax_psd.set_title(f"{component_name} final PSD", fontsize=10)
    ax_psd.grid(alpha=0.25)

    fig.suptitle(
        f"{component_name} input diagnostics | band={band_hz[0]:.2f}-{band_hz[1]:.2f} Hz | target SD={target_sd:.1f} nA",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, figure_name), bbox_inches="tight")
    plt.close(fig)


def _resolve_summary_window_overlay_specs(overlay_specs=None):
    default_specs = {
        "firing_rate_isi_window_rel_cue_s": {"enabled": True, "label": "FR / ISI window", "color": "#37a055", "lw": 2.0, "alpha": 1.0, "y_axes": 0.02},
        "cst_modulation_normalization_window_rel_cue_s": {"enabled": True, "label": "CST normalization window", "color": "#ffa600", "lw": 2.0, "alpha": 0.95, "y_axes": 0.05},
        "sync_baseline_win_rel_cue_s": {"enabled": True, "label": "Sync baseline window", "color": "#7e2179", "lw": 2.0, "alpha": 1.0, "y_axes": 0.08},
        "feature_fitting_window": {"enabled": True, "label": "Spline fitting window", "color": "#E09ED8", "lw": 2.0, "alpha": 1.0, "y_axes": 0.15},
        "feature_analysis_window": {"enabled": True, "label": "Feature window", "color": "#E900CA", "lw": 2.0, "alpha": 1.0, "y_axes": 0.15},
    }
    if overlay_specs:
        for key, value in overlay_specs.items():
            base = dict(default_specs.get(key, {}))
            base.update(value or {})
            default_specs[key] = base
    return default_specs


def _summary_window_segments(params):
    fitting_start_s, fitting_end_s = map(float, params.feature_fitting_window_rel_cue_s)
    analysis_start_s, analysis_end_s = map(float, params.feature_analysis_window_rel_cue_s)
    exclusion_start_s = analysis_start_s - float(params.feature_buffer_s)
    exclusion_end_s = analysis_end_s + float(params.feature_buffer_s)
    return {
        "firing_rate_isi_window_rel_cue_s": [tuple(map(float, params.firing_rate_isi_window_rel_cue_s))],
        "cst_modulation_normalization_window_rel_cue_s": [tuple(map(float, params.cst_modulation_normalization_window_rel_cue_s))],
        "sync_baseline_win_rel_cue_s": [tuple(map(float, params.sync_baseline_win_rel_cue_s))],
        "feature_fitting_window": [
            (fitting_start_s, min(exclusion_start_s, fitting_end_s)),
            (max(exclusion_end_s, fitting_start_s), fitting_end_s),
        ],
        "feature_analysis_window": [(analysis_start_s, analysis_end_s)],
    }


def _draw_summary_window_overlays(ax, display_window_rel_cue_s, params):
    overlay_specs = _resolve_summary_window_overlay_specs(getattr(params, "summary_window_overlay_specs", None))
    segments_by_key = _summary_window_segments(params)
    trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for key, spec in overlay_specs.items():
        if not bool(spec.get("enabled", True)):
            continue
        for segment in segments_by_key.get(key, []):
            seg_start_s, seg_end_s = map(float, segment)
            clipped_start_s = max(float(display_window_rel_cue_s[0]), seg_start_s)
            clipped_end_s = min(float(display_window_rel_cue_s[1]), seg_end_s)
            if clipped_start_s >= clipped_end_s:
                continue
            ax.plot(
                [clipped_start_s, clipped_end_s],
                [float(spec.get("y_axes", 0.02)), float(spec.get("y_axes", 0.02))],
                color=str(spec.get("color", "#111111")),
                lw=float(spec.get("lw", 2.0)),
                alpha=float(spec.get("alpha", 0.95)),
                solid_capstyle="butt",
                transform=trans,
                clip_on=False,
                zorder=7,
            )


def plot_step_response_diagnostic(save_dir, input_components, cst_diagnostics, step_response_diagnostics, params):
    if step_response_diagnostics is None or str(step_response_diagnostics.get("status", "")).lower() != "ok":
        return
    style = dict(getattr(params, "step_diagnostic_plot_style", {}) or {})

    def _style_float(key, default):
        try:
            value = style.get(key, default)
            return default if value is None else float(value)
        except (TypeError, ValueError):
            return default

    def _style_lim(key):
        value = style.get(key)
        if value is None:
            return None
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != 2 or not np.all(np.isfinite(arr)) or arr[0] >= arr[1]:
            warnings.warn(f"Ignoring invalid step_diagnostic_plot_style[{key!r}]={value!r}; expected [min, max].")
            return None
        return float(arr[0]), float(arr[1])

    figsize = style.get("figsize", [12.0, 5.6])
    figsize_arr = np.asarray(figsize, dtype=float).reshape(-1)
    if figsize_arr.size != 2 or np.any(figsize_arr <= 0):
        figsize_arr = np.asarray([12.0, 5.6], dtype=float)

    step_abs_s = float(np.asarray(step_response_diagnostics["step_current_time_abs_s"]).reshape(-1)[0])
    display_rel = _style_lim("xlim_rel_s") or tuple(map(float, params.step_diagnostic_display_window_rel_s))
    t_input_s = np.arange(len(input_components["common_input_with_burst"]), dtype=float) / float(params.fsamp)
    input_rel_s = t_input_s - step_abs_s
    display_mask = (input_rel_s >= display_rel[0]) & (input_rel_s <= display_rel[1])
    if not np.any(display_mask):
        return
    baseline_plus_step = np.asarray(input_components["baseline_plus_step_trace_nA"], dtype=float)
    common_input = np.asarray(input_components["common_input_with_burst"], dtype=float)
    total_input = baseline_plus_step + common_input

    cst_time = np.asarray(cst_diagnostics["time_s"], dtype=float)
    cst_rel_s = cst_time - step_abs_s
    cst_mask = (cst_rel_s >= display_rel[0]) & (cst_rel_s <= display_rel[1])
    cst_lf = np.asarray(cst_diagnostics["cst_lf"], dtype=float) if "cst_lf" in cst_diagnostics else None
    cst_hann = np.asarray(cst_diagnostics["cst_hann"], dtype=float) if "cst_hann" in cst_diagnostics else None
    cst_causal = np.asarray(
        step_response_diagnostics.get(
            "cst_smooth_step_response",
            step_response_diagnostics.get(
                "cst_smooth_causal_exp",
                cst_lf if cst_lf is not None else np.full_like(cst_time, np.nan),
            ),
        ),
        dtype=float,
    )
    matched_no_step = (
        np.asarray(
            step_response_diagnostics.get(
                "matched_no_step_cst_smooth_step_response",
                step_response_diagnostics.get("matched_no_step_cst_smooth_causal_exp"),
            ),
            dtype=float,
        )
        if "matched_no_step_cst_smooth_step_response" in step_response_diagnostics
        or "matched_no_step_cst_smooth_causal_exp" in step_response_diagnostics
        else None
    )
    step_effect = (
        np.asarray(
            step_response_diagnostics.get(
                "cst_step_effect_sign_corrected",
                step_response_diagnostics.get("cst_step_effect"),
            ),
            dtype=float,
        )
        if "cst_step_effect" in step_response_diagnostics or "cst_step_effect_sign_corrected" in step_response_diagnostics
        else None
    )
    sync_rel_s = np.asarray(step_response_diagnostics.get("sync_time_rel_step_s", []), dtype=float)
    sync_trace = np.asarray(step_response_diagnostics.get("sync_trace", []), dtype=float)
    matched_no_step_sync = (
        np.asarray(step_response_diagnostics["matched_no_step_sync_trace"], dtype=float)
        if "matched_no_step_sync_trace" in step_response_diagnostics
        else None
    )
    sync_effect = (
        np.asarray(step_response_diagnostics["sync_step_effect"], dtype=float)
        if "sync_step_effect" in step_response_diagnostics
        else None
    )
    show_sync_subplot = bool(style.get("show_sync_subplot", True)) and sync_rel_s.size and sync_trace.size

    if show_sync_subplot:
        fig, (ax_input, ax_sync) = plt.subplots(
            2,
            1,
            figsize=tuple(figsize_arr),
            dpi=int(_style_float("dpi", 140)),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.15], "hspace": 0.08},
        )
    else:
        fig, ax_input = plt.subplots(figsize=tuple(figsize_arr), dpi=int(_style_float("dpi", 140)))
        ax_sync = None
    ax_cst = ax_input.twinx()
    baseline_line, = ax_input.plot(
        input_rel_s[display_mask],
        baseline_plus_step[display_mask] / 1000.0,
        color=style.get("baseline_input_color", "#6f6f6f"),
        lw=_style_float("baseline_input_lw", 1.2),
        ls=style.get("baseline_input_ls", "-"),
        alpha=_style_float("baseline_input_alpha", 0.95),
        zorder=2,
        label="Baseline tonic + step",
    )
    input_line, = ax_input.plot(
        input_rel_s[display_mask],
        total_input[display_mask] / 1000.0,
        color=style.get("total_input_color", "#16803a"),
        lw=_style_float("total_input_lw", 1.5),
        alpha=_style_float("total_input_alpha", 1.0),
        zorder=3,
        label="Baseline + common + step input",
    )
    step_current_line = None
    if bool(style.get("show_step_current_trace", False)):
        step_current_line, = ax_input.plot(
            input_rel_s[display_mask],
            np.asarray(input_components["step_current_trace_nA"], dtype=float)[display_mask] / 1000.0,
            color=style.get("step_current_color", "#111111"),
            lw=_style_float("step_current_lw", 1.0),
            ls=style.get("step_current_ls", ":"),
            alpha=_style_float("step_current_alpha", 1.0),
            zorder=4,
            label="Step current",
        )
    cst_lf_line = None
    if bool(style.get("show_cst_lf", False)) and cst_lf is not None and np.any(cst_mask):
        cst_lf_line, = ax_cst.plot(
            cst_rel_s[cst_mask],
            cst_lf[cst_mask],
            color=style.get("cst_lf_color", style.get("cst_color", "#ff3300")),
            lw=_style_float("cst_lf_lw", 1.0),
            ls=style.get("cst_lf_ls", "-"),
            alpha=_style_float("cst_lf_alpha", 0.5),
            label="Low-pass CST",
        )
    cst_line = None
    show_cst_causal = bool(style.get("show_cst_causal", True))
    if show_cst_causal and np.any(cst_mask):
        cst_line, = ax_cst.plot(
            cst_rel_s[cst_mask],
            cst_causal[cst_mask],
            color=style.get("cst_causal_color", style.get("cst_color", "#ff3300")),
            lw=_style_float("cst_causal_lw", _style_float("cst_lw", 1.5)),
            alpha=_style_float("cst_causal_alpha", _style_float("cst_alpha", 1.0)),
            label="Causal-exp CST",
        )
    matched_line = None
    if bool(style.get("show_matched_no_step", True)) and matched_no_step is not None and np.any(cst_mask):
        matched_line, = ax_cst.plot(
            cst_rel_s[cst_mask],
            matched_no_step[cst_mask],
            color=style.get("matched_no_step_color", "#777777"),
            lw=_style_float("matched_no_step_lw", 1.1),
            ls=style.get("matched_no_step_ls", "--"),
            alpha=_style_float("matched_no_step_alpha", 0.85),
            label="Matched no-step CST",
        )
    effect_line = None
    if bool(style.get("show_step_effect", True)) and step_effect is not None and np.any(cst_mask):
        effect_line, = ax_cst.plot(
            cst_rel_s[cst_mask],
            step_effect[cst_mask],
            color=style.get("step_effect_color", "#111111"),
            lw=_style_float("step_effect_lw", 1.4),
            ls=style.get("step_effect_ls", "-"),
            alpha=_style_float("step_effect_alpha", 0.95),
            label="Step effect (sign-corrected)",
        )
    cst_hann_line = None
    show_cst_hann = bool(style.get("show_cst_hann", False))
    if show_cst_hann and cst_hann is not None and np.any(cst_mask):
        cst_hann_line, = ax_cst.plot(
            cst_rel_s[cst_mask],
            cst_hann[cst_mask],
            color=style.get("cst_hann_color", "#111111"),
            lw=_style_float("cst_hann_lw", 1.3),
            ls=style.get("cst_hann_ls", "--"),
            alpha=_style_float("cst_hann_alpha", 0.9),
            label="Hann-smoothed CST",
        )

    def _metric(key):
        value = step_response_diagnostics.get(key, np.asarray([np.nan]))
        return float(np.asarray(value, dtype=float).reshape(-1)[0])

    def _metric_alias(*keys):
        for key in keys:
            if key in step_response_diagnostics:
                value = _metric(key)
                if np.isfinite(value):
                    return value
        return np.nan

    def _status(key, default=""):
        value = step_response_diagnostics.get(key, default)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = value.reshape(-1)[0]
            if isinstance(value, bytes):
                return value.decode("utf-8")
        return str(value)

    def _metric_per_uA(key_uA, key_nA):
        value = _metric(key_uA)
        if np.isfinite(value):
            return value
        per_nA = _metric(key_nA)
        return per_nA * 1000.0 if np.isfinite(per_nA) else np.nan

    slope_line = None
    if bool(style.get("show_slope_fit", True)) and step_effect is not None:
        slope_end = _metric_alias("linear_selected_window_end_s", "selected_slope_window_end_s")
        slope = _metric_alias("linear_initial_slope_cst_per_s", "initial_slope_cst_per_s")
        intercept = _metric_alias("linear_selected_intercept", "selected_slope_intercept")
        if np.isfinite(slope_end) and np.isfinite(slope) and np.isfinite(intercept) and slope_end > 0:
            slope_t = np.asarray([0.0, slope_end], dtype=float)
            effect_fit_y = intercept + slope * slope_t
            overlay_mode = str(style.get("slope_fit_overlay_mode", "cst")).strip().lower()
            if overlay_mode == "cst" and matched_no_step is not None:
                no_step_fit_base = np.interp(slope_t, cst_rel_s, matched_no_step)
                step_sign = _metric("step_sign")
                if not np.isfinite(step_sign) or abs(step_sign) < 1e-12:
                    step_amp = _metric("step_current_amplitude_nA")
                    step_sign = float(np.sign(step_amp)) if np.isfinite(step_amp) else 1.0
                slope_y = no_step_fit_base + step_sign * effect_fit_y
                slope_label = "Selected slope fit on CST"
            else:
                slope_y = effect_fit_y
                slope_label = "Selected slope fit"
            slope_line, = ax_cst.plot(
                slope_t,
                slope_y,
                color=style.get("slope_fit_color", "#0047ab"),
                lw=_style_float("slope_fit_lw", 1.5),
                ls=style.get("slope_fit_ls", "-."),
                alpha=_style_float("slope_fit_alpha", 0.95),
                label=slope_label,
            )

    response_shape_line = None
    if bool(style.get("show_response_shape_fit", True)) and step_effect is not None:
        shape_t = np.asarray(step_response_diagnostics.get("response_shape_fitted_curve_time_rel_s", []), dtype=float)
        shape_y = np.asarray(step_response_diagnostics.get("response_shape_fitted_curve", []), dtype=float)
        shape_mask = (
            (shape_t >= display_rel[0])
            & (shape_t <= display_rel[1])
            & np.isfinite(shape_t)
            & np.isfinite(shape_y)
        )
        if shape_t.shape == shape_y.shape and np.any(shape_mask):
            response_shape_line, = ax_cst.plot(
                shape_t[shape_mask],
                shape_y[shape_mask],
                color=style.get("response_shape_fit_color", "#008c72"),
                lw=_style_float("response_shape_fit_lw", 1.6),
                ls=style.get("response_shape_fit_ls", "--"),
                alpha=_style_float("response_shape_fit_alpha", 0.95),
                label="Response-shape fit",
            )

    pre_win = np.asarray(step_response_diagnostics["step_pre_window_rel_s"], dtype=float)
    late_post_win = np.asarray(step_response_diagnostics["step_late_post_window_rel_s"], dtype=float)
    shape_fit_win = np.asarray(step_response_diagnostics.get("response_shape_fit_window_rel_s", [0.0, 0.5]), dtype=float).reshape(-1)
    xcorr_win = np.asarray(step_response_diagnostics.get("xcorr_window_rel_s", [0.0, 1.0]), dtype=float).reshape(-1)
    sync_pred_pre_win = np.asarray(step_response_diagnostics.get("sync_predictor_pre_step_window_rel_s", [-0.250, 0.000]), dtype=float).reshape(-1)
    sync_pred_local_win = np.asarray(step_response_diagnostics.get("sync_predictor_local_step_window_rel_s", [-0.250, 0.250]), dtype=float).reshape(-1)
    pre_window_color = style.get("pre_window_color", "#37a055")
    late_post_window_color = style.get("late_post_window_color", "#ff3300")
    pre_window_alpha = _style_float("pre_window_alpha", 0.12)
    late_post_window_alpha = _style_float("late_post_window_alpha", 0.10)
    ax_input.axvspan(pre_win[0], pre_win[1], color=pre_window_color, alpha=pre_window_alpha, label="Pre-step window")
    ax_input.axvspan(late_post_win[0], late_post_win[1], color=late_post_window_color, alpha=late_post_window_alpha, label="Late post-step window")
    if bool(style.get("show_response_shape_fit_window", True)) and shape_fit_win.size == 2:
        ax_input.axvspan(
            shape_fit_win[0],
            shape_fit_win[1],
            color=style.get("response_shape_fit_window_color", "#008c72"),
            alpha=_style_float("response_shape_fit_window_alpha", 0.08),
            label="Shape-fit window",
        )
    if bool(style.get("show_xcorr_window", True)) and xcorr_win.size == 2:
        ax_input.axvspan(
            xcorr_win[0],
            xcorr_win[1],
            color=style.get("xcorr_window_color", "#7b3294"),
            alpha=_style_float("xcorr_window_alpha", 0.06),
            label="Xcorr window",
        )
    step_onset_line = None
    if bool(style.get("show_step_onset_line", True)):
        step_onset_line = ax_input.axvline(
            0.0,
            color=style.get("step_line_color", "#111111"),
            lw=_style_float("step_line_lw", 1.1),
            ls=style.get("step_line_ls", "--"),
            label="Step onset",
        )
    if show_sync_subplot and ax_sync is not None:
        sync_mask = (sync_rel_s >= display_rel[0]) & (sync_rel_s <= display_rel[1]) & np.isfinite(sync_trace)
        sync_handles = []
        ax_sync.axvspan(pre_win[0], pre_win[1], color=pre_window_color, alpha=pre_window_alpha)
        ax_sync.axvspan(late_post_win[0], late_post_win[1], color=late_post_window_color, alpha=late_post_window_alpha)
        if bool(style.get("show_sync_predictor_windows", True)):
            if sync_pred_local_win.size == 2:
                ax_sync.axvspan(
                    sync_pred_local_win[0],
                    sync_pred_local_win[1],
                    color=style.get("sync_predictor_local_window_color", "#756bb1"),
                    alpha=_style_float("sync_predictor_local_window_alpha", 0.08),
                    zorder=0,
                )
            if sync_pred_pre_win.size == 2:
                ax_sync.axvspan(
                    sync_pred_pre_win[0],
                    sync_pred_pre_win[1],
                    color=style.get("sync_predictor_pre_window_color", "#2ca25f"),
                    alpha=_style_float("sync_predictor_pre_window_alpha", 0.12),
                    zorder=0,
                )
        if step_onset_line is not None or bool(style.get("show_step_onset_line", True)):
            ax_sync.axvline(
                0.0,
                color=style.get("step_line_color", "#111111"),
                lw=_style_float("step_line_lw", 1.1),
                ls=style.get("step_line_ls", "--"),
                alpha=0.75,
            )
        if matched_no_step_sync is not None and matched_no_step_sync.shape == sync_rel_s.shape and np.any(sync_mask):
            line, = ax_sync.plot(
                sync_rel_s[sync_mask],
                matched_no_step_sync[sync_mask],
                color=style.get("matched_no_step_sync_color", "#777777"),
                lw=_style_float("matched_no_step_sync_lw", 1.1),
                ls=style.get("matched_no_step_sync_ls", "--"),
                alpha=_style_float("matched_no_step_sync_alpha", 0.45),
                zorder=1,
                label="Matched no-step sync",
            )
            sync_handles.append(line)
        if np.any(sync_mask):
            line, = ax_sync.plot(
                sync_rel_s[sync_mask],
                sync_trace[sync_mask],
                color=style.get("sync_trace_color", "#852680"),
                lw=_style_float("sync_trace_lw", 1.4),
                alpha=_style_float("sync_trace_alpha", 1.0),
                zorder=3,
                label="Step sync",
            )
            sync_handles.append(line)
        if sync_effect is not None and sync_effect.shape == sync_rel_s.shape and np.any(sync_mask):
            line, = ax_sync.plot(
                sync_rel_s[sync_mask],
                sync_effect[sync_mask],
                color=style.get("sync_effect_color", "#111111"),
                lw=_style_float("sync_effect_lw", 1.2),
                ls=style.get("sync_effect_ls", "-"),
                alpha=_style_float("sync_effect_alpha", 0.95),
                zorder=2,
                label="Sync residual",
            )
            sync_handles.append(line)
            ax_sync.axhline(0.0, color="0.55", lw=0.8, ls=":", zorder=0)
        ax_sync.set_ylabel("Synchrony")
        ax_sync.grid(alpha=_style_float("grid_alpha", 0.25))
        def _fmt_rel_window(win, unit="s"):
            win = np.asarray(win, dtype=float).reshape(-1)
            if win.size != 2 or not np.all(np.isfinite(win)):
                return "[?,?]"
            if unit == "ms":
                return f"[{win[0] * 1000:.0f},{win[1] * 1000:.0f}]ms"
            return f"[{win[0]:.2g},{win[1]:.2g}]s"

        sync_annotation = "\n".join([
            f"Sync baseline {_fmt_rel_window(pre_win)}: {_metric('baseline_sync_mean'):.3f} +/- {_metric('baseline_sync_sd'):.3f}",
            f"Sync post actual {_fmt_rel_window(late_post_win)}: {_metric('post_step_sync_mean_actual_step'):.3f} +/- {_metric('post_step_sync_sd_actual_step'):.3f}",
            f"Sync pre matched {_fmt_rel_window(sync_pred_pre_win, 'ms')}: {_metric('pre_step_sync_mean_baseline_matched'):.3f} +/- {_metric('pre_step_sync_sd_baseline_matched'):.3f}",
            f"Sync local matched {_fmt_rel_window(sync_pred_local_win, 'ms')}: {_metric('local_step_sync_mean_baseline_matched'):.3f} +/- {_metric('local_step_sync_sd_baseline_matched'):.3f}",
            f"source={_status('sync_predictor_source', '')}",
        ])
        sync_loc = str(style.get("sync_annotation_loc", "upper left")).strip().lower()
        if sync_loc in {"upper right", "right"}:
            sync_xy = (0.99, 0.98)
            sync_ha = "right"
            sync_va = "top"
        elif sync_loc in {"lower left", "bottom left"}:
            sync_xy = (0.01, 0.02)
            sync_ha = "left"
            sync_va = "bottom"
        elif sync_loc in {"lower right", "bottom right"}:
            sync_xy = (0.99, 0.02)
            sync_ha = "right"
            sync_va = "bottom"
        else:
            sync_xy = (0.01, 0.98)
            sync_ha = "left"
            sync_va = "top"
        ax_sync.text(
            sync_xy[0],
            sync_xy[1],
            sync_annotation,
            transform=ax_sync.transAxes,
            ha=sync_ha,
            va=sync_va,
            fontsize=_style_float("sync_annotation_fontsize", _style_float("annotation_fontsize", 7.3)),
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.65", alpha=0.88),
            zorder=100,
        )
        if sync_handles:
            ax_sync.legend(handles=sync_handles, loc="upper right", fontsize=_style_float("legend_fontsize", 8.0))

    annotation = "\n".join([
        f"baseline={_metric('baseline_tonic_input_nA')/1000.0:.2f} uA | LF={_metric('lf_input_amplitude_nA')/1000.0:.2f} uA",
        f"alpha={_metric('alpha_input_amplitude_nA')/1000.0:.2f} uA | beta={_metric('beta_input_amplitude_nA')/1000.0:.2f} uA",
        f"step={_metric('step_current_amplitude_nA')/1000.0:.2f} uA ({_metric('step_current_amplitude_percent_baseline'):.1f}%)",
        f"FR pre/post={_metric('pre_step_mean_firing_rate_hz'):.2f}/{_metric('post_step_mean_firing_rate_hz'):.2f} Hz",
        f"ISI CV pre/post={_metric('pre_step_isi_cv_mean'):.2f}/{_metric('post_step_isi_cv_mean'):.2f}",
        f"dEffect={_metric('delta_cst_effect'):.2f} sp/s/MN | raw dCST={_metric('delta_cst_smooth_raw'):.2f} sp/s/MN",
        f"gain={_metric_per_uA('response_gain_cst_per_uA', 'response_gain_cst_per_nA'):.3g} sp/s/MN/uA",
        f"linear slope={_metric_alias('linear_initial_slope_cst_per_s', 'initial_slope_cst_per_s'):.2f} sp/s/MN/s | norm={_metric_alias('linear_initial_slope_normalized_per_nA', 'initial_slope_normalized_per_nA'):.3g} per nA",
        f"linear end={_metric_alias('linear_selected_window_end_s', 'selected_slope_window_end_s'):.3f} s | R2={_metric_alias('linear_selected_window_r2', 'selected_slope_window_r2'):.2f} | {_status('linear_slope_selection_status', _status('slope_selection_status', ''))}",
        f"shape tau={_metric('response_shape_tau_ms'):.1f} ms | response at 50/150 ms={_metric('response_shape_fraction_50ms'):.2f}/{_metric('response_shape_fraction_150ms'):.2f} of full gain",
        f"shape t50/t90={_metric('response_shape_fitted_t50_ms'):.1f}/{_metric('response_shape_fitted_t90_ms'):.1f} ms | R2={_metric('response_shape_fit_r2'):.2f} | {_status('response_shape_fit_status')}",
        f"xcorr lag={_metric('xcorr_lag_s') * 1000.0:.1f} ms | corr={_metric('xcorr_peak_corr'):.2f} | sign-corr={_metric('xcorr_lag_sign_corrected_s') * 1000.0:.1f} ms",
        f"matched={'yes' if _metric('matched_step_effect_available') > 0.5 else 'no'} | stage={step_response_diagnostics.get('diagnostics_stage', 'unknown')}",
    ])
    annotation_ax = ax_cst
    annotation_ax.patch.set_alpha(0.0)
    annotation_ax.text(
        0.01,
        0.99,
        annotation,
        transform=annotation_ax.transAxes,
        va="top",
        ha="left",
        fontsize=_style_float("annotation_fontsize", 8.3),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.65", alpha=0.92),
        zorder=100,
        clip_on=False,
    )
    if show_sync_subplot and ax_sync is not None:
        ax_input.set_xlabel("")
        ax_sync.set_xlabel("Time relative to step onset (s)")
    else:
        ax_input.set_xlabel("Time relative to step onset (s)")
    ax_input.set_ylabel("Input current (uA)")
    ax_cst.set_ylabel("CST / CST effect (spikes/s/MU)")
    ax_input.set_title(str(style.get("title", "Step-response diagnostic")))
    ax_input.grid(alpha=_style_float("grid_alpha", 0.25))
    ax_input.set_xlim(display_rel)
    input_ylim = _style_lim("input_ylim_uA")
    if input_ylim is not None:
        ax_input.set_ylim(input_ylim)
    cst_ylim = _style_lim("cst_ylim")
    if cst_ylim is not None:
        ax_cst.set_ylim(cst_ylim)
    handles = [input_line, baseline_line]
    if step_current_line is not None:
        handles.append(step_current_line)
    if cst_lf_line is not None:
        handles.append(cst_lf_line)
    if cst_line is not None:
        handles.append(cst_line)
    if matched_line is not None:
        handles.append(matched_line)
    if effect_line is not None:
        handles.append(effect_line)
    if slope_line is not None:
        handles.append(slope_line)
    if response_shape_line is not None:
        handles.append(response_shape_line)
    if cst_hann_line is not None:
        handles.append(cst_hann_line)
    handles.extend([
        Patch(facecolor=pre_window_color, alpha=pre_window_alpha, label="Pre-step window"),
        Patch(facecolor=late_post_window_color, alpha=late_post_window_alpha, label="Late post-step window"),
    ])
    if bool(style.get("show_response_shape_fit_window", True)) and shape_fit_win.size == 2:
        handles.append(Patch(
            facecolor=style.get("response_shape_fit_window_color", "#008c72"),
            alpha=_style_float("response_shape_fit_window_alpha", 0.08),
            label="Shape-fit window",
        ))
    if bool(style.get("show_xcorr_window", True)) and xcorr_win.size == 2:
        handles.append(Patch(
            facecolor=style.get("xcorr_window_color", "#7b3294"),
            alpha=_style_float("xcorr_window_alpha", 0.06),
            label="Xcorr window",
        ))
    if step_onset_line is not None:
        handles.append(Line2D([0], [0], color=style.get("step_line_color", "#111111"), lw=_style_float("step_line_lw", 1.1), ls=style.get("step_line_ls", "--"), label="Step onset"))
    ax_input.legend(handles=handles, loc=style.get("legend_loc", "lower right"), fontsize=_style_float("legend_fontsize", 8.0))
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "step_response_diagnostic.png"), bbox_inches="tight")
    plt.close(fig)


def _decode_h5_value(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{\"" and stripped[-1] in "]}\"":
            try:
                return json.loads(stripped)
            except Exception:
                return value
    return value


def _read_h5_group_to_dict(group):
    out = {}
    if group is None:
        return out
    for key, value in group.attrs.items():
        out[key] = _decode_h5_value(value)
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            arr = value[()]
            if isinstance(arr, bytes):
                arr = arr.decode("utf-8")
            out[key] = arr
    return out


def _scalar_from_mapping(mapping, key, default=np.nan):
    if key not in mapping:
        return default
    try:
        arr = np.asarray(mapping[key]).reshape(-1)
        if arr.size == 0:
            return default
        value = arr[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, np.generic):
            value = value.item()
        return value
    except Exception:
        return mapping.get(key, default)


def _float_from_mapping(mapping, key, default=np.nan):
    try:
        return float(_scalar_from_mapping(mapping, key, default))
    except Exception:
        return float(default)


def _bool_from_mapping(mapping, key, default=False):
    value = _scalar_from_mapping(mapping, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _trace_from_mapping(mapping, preferred_key, fallback_key=None):
    if preferred_key in mapping:
        return np.asarray(mapping[preferred_key], dtype=float)
    if fallback_key is not None and fallback_key in mapping:
        return np.asarray(mapping[fallback_key], dtype=float)
    raise KeyError(preferred_key)


def _write_step_response_updates(h5_path, updates):
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "a") as f:
        grp = f.require_group("step_response_diagnostics")
        for key, value in dict(updates).items():
            if value is None:
                continue
            if isinstance(value, (str, bytes)):
                grp.attrs[key] = value.decode("utf-8") if isinstance(value, bytes) else value
                if key in grp:
                    del grp[key]
                continue
            arr = np.asarray(value)
            if arr.dtype.kind in {"U", "S", "O"}:
                try:
                    grp.attrs[key] = str(value)
                    if key in grp:
                        del grp[key]
                    continue
                except Exception:
                    pass
            if key in grp:
                del grp[key]
            grp.create_dataset(key, data=arr)


def _try_write_step_response_updates(h5_path, updates, *, context="step-response update"):
    try:
        _write_step_response_updates(h5_path, updates)
        return True
    except Exception as exc:
        warnings.warn(f"Could not write {context} to {h5_path}: {exc}")
        return False


def _write_mapping_to_h5_group(group, mapping):
    for key, value in dict(mapping).items():
        if value is None:
            continue
        if isinstance(value, (str, bytes)):
            group.attrs[key] = value.decode("utf-8") if isinstance(value, bytes) else value
            if key in group:
                del group[key]
            continue
        if isinstance(value, (list, tuple)) and any(isinstance(item, (str, bytes, Path)) for item in value):
            group.attrs[key] = json.dumps([str(item) for item in value])
            if key in group:
                del group[key]
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in {"U", "S", "O"}:
            group.attrs[key] = json.dumps(value.tolist() if hasattr(value, "tolist") else str(value))
            if key in group:
                del group[key]
            continue
        if key in group:
            del group[key]
        group.create_dataset(key, data=arr)


def _linear_fit_slope_r2(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3 or np.unique(x).size < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = intercept + slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot <= 1e-18 else 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r2)


def _response_shape_fraction(t_ms, tau_ms):
    t_ms = np.asarray(t_ms, dtype=float)
    tau_ms = float(tau_ms)
    out = np.full_like(t_ms, np.nan, dtype=float)
    valid = np.isfinite(t_ms) & (t_ms >= 0.0) & np.isfinite(tau_ms) & (tau_ms > 0.0)
    out[valid] = (t_ms[valid] / (t_ms[valid] + tau_ms)) ** 2
    return out


def _response_shape_threshold_time_ms(tau_ms, fraction):
    tau_ms = float(tau_ms)
    fraction = float(fraction)
    if not (np.isfinite(tau_ms) and tau_ms > 0.0 and 0.0 < fraction < 1.0):
        return np.nan
    root = sqrt(fraction)
    return float(tau_ms * root / (1.0 - root))


def _empty_response_shape_updates(status, fit_window_rel_s=None):
    fit_window = np.asarray(fit_window_rel_s if fit_window_rel_s is not None else [0.0, 0.5], dtype=float)
    return {
        "response_shape_tau_ms": np.asarray([np.nan], dtype=float),
        "response_shape_gamma": np.asarray([np.nan], dtype=float),
        "response_shape_fraction_50ms": np.asarray([np.nan], dtype=float),
        "response_shape_fraction_150ms": np.asarray([np.nan], dtype=float),
        "response_shape_fitted_t50_ms": np.asarray([np.nan], dtype=float),
        "response_shape_fitted_t90_ms": np.asarray([np.nan], dtype=float),
        "response_shape_fit_r2": np.asarray([np.nan], dtype=float),
        "response_shape_fit_rmse": np.asarray([np.nan], dtype=float),
        "response_shape_fit_status": str(status),
        "response_shape_R_inf": np.asarray([np.nan], dtype=float),
        "response_shape_fit_window_rel_s": fit_window,
        "response_shape_fitted_curve_time_rel_s": np.asarray([], dtype=float),
        "response_shape_fitted_curve": np.asarray([], dtype=float),
    }


def _fit_response_shape_metrics(
        t_rel_s,
        effect_sign_corrected,
        *,
        r_inf,
        fit_window_rel_s,
        tau_min_ms,
        tau_max_ms,
        min_r_inf):
    fit_window = np.asarray(fit_window_rel_s, dtype=float).reshape(-1)
    if fit_window.size != 2:
        fit_window = np.asarray([0.0, 0.5], dtype=float)
    updates = _empty_response_shape_updates("not_fit", fit_window)
    updates["response_shape_R_inf"] = np.asarray([r_inf], dtype=float)
    if not (np.isfinite(r_inf) and r_inf > float(min_r_inf)):
        updates["response_shape_fit_status"] = "invalid_r_inf"
        return updates

    t_rel_s = np.asarray(t_rel_s, dtype=float)
    effect_sign_corrected = np.asarray(effect_sign_corrected, dtype=float)
    fit_mask = (
        (t_rel_s >= max(0.0, float(fit_window[0])))
        & (t_rel_s <= float(fit_window[1]))
        & np.isfinite(t_rel_s)
        & np.isfinite(effect_sign_corrected)
    )
    if np.count_nonzero(fit_mask) < 3:
        updates["response_shape_fit_status"] = "insufficient_fit_samples"
        return updates
    t_fit_s = t_rel_s[fit_mask]
    t_fit_ms = t_fit_s * 1000.0
    y = effect_sign_corrected[fit_mask] / float(r_inf)
    valid = np.isfinite(t_fit_ms) & np.isfinite(y) & (t_fit_ms >= 0.0)
    t_fit_s = t_fit_s[valid]
    t_fit_ms = t_fit_ms[valid]
    y = y[valid]
    if y.size < 3:
        updates["response_shape_fit_status"] = "insufficient_finite_samples"
        return updates

    tau_min_ms = float(tau_min_ms)
    tau_max_ms = float(tau_max_ms)
    if not (np.isfinite(tau_min_ms) and np.isfinite(tau_max_ms) and 0.0 < tau_min_ms < tau_max_ms):
        updates["response_shape_fit_status"] = "invalid_tau_bounds"
        return updates

    def _objective(tau_ms):
        fitted = _response_shape_fraction(t_fit_ms, tau_ms)
        residual = y - fitted
        return float(np.nanmean(residual ** 2))

    try:
        result = minimize_scalar(_objective, bounds=(tau_min_ms, tau_max_ms), method="bounded")
    except Exception:
        updates["response_shape_fit_status"] = "fit_error"
        return updates
    if not (getattr(result, "success", False) and np.isfinite(result.x)):
        updates["response_shape_fit_status"] = "fit_failed"
        return updates

    tau_ms = float(result.x)
    fitted = _response_shape_fraction(t_fit_ms, tau_ms)
    residual = y - fitted
    rmse = float(np.sqrt(np.nanmean(residual ** 2)))
    ss_res = float(np.nansum(residual ** 2))
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    r2 = np.nan if ss_tot <= 1e-18 else 1.0 - ss_res / ss_tot
    frac_50 = float(_response_shape_fraction(np.asarray([50.0]), tau_ms)[0])
    frac_150 = float(_response_shape_fraction(np.asarray([150.0]), tau_ms)[0])
    updates.update({
        "response_shape_tau_ms": np.asarray([tau_ms], dtype=float),
        "response_shape_gamma": np.asarray([1.0 / sqrt(tau_ms)], dtype=float),
        "response_shape_fraction_50ms": np.asarray([frac_50], dtype=float),
        "response_shape_fraction_150ms": np.asarray([frac_150], dtype=float),
        "response_shape_fitted_t50_ms": np.asarray([_response_shape_threshold_time_ms(tau_ms, 0.5)], dtype=float),
        "response_shape_fitted_t90_ms": np.asarray([_response_shape_threshold_time_ms(tau_ms, 0.9)], dtype=float),
        "response_shape_fit_r2": np.asarray([r2], dtype=float),
        "response_shape_fit_rmse": np.asarray([rmse], dtype=float),
        "response_shape_fit_status": "ok",
        "response_shape_fitted_curve_time_rel_s": np.asarray(t_fit_s, dtype=float),
        "response_shape_fitted_curve": np.asarray(fitted * float(r_inf), dtype=float),
    })
    return updates


def _empty_xcorr_updates(status, window_rel_s=None, max_lag_s=np.nan):
    window = np.asarray(window_rel_s if window_rel_s is not None else [0.0, 1.0], dtype=float)
    return {
        "xcorr_lag_s": np.asarray([np.nan], dtype=float),
        "xcorr_peak_corr": np.asarray([np.nan], dtype=float),
        "xcorr_lag_sign_convention": (
            "negative means step CST leads/advanced relative to no-step; "
            "positive means step CST lags/delayed relative to no-step"
        ),
        "xcorr_lag_sign_corrected_s": np.asarray([np.nan], dtype=float),
        "xcorr_window_rel_s": window,
        "xcorr_max_lag_s": np.asarray([float(max_lag_s)], dtype=float),
        "xcorr_lags_s": np.asarray([], dtype=float),
        "xcorr_values": np.asarray([], dtype=float),
        "xcorr_status": str(status),
    }


def _compute_xcorr_lag_metrics(t_rel_s, step_cst, zero_cst, *, step_sign, window_rel_s, max_lag_s):
    t_rel_s = np.asarray(t_rel_s, dtype=float)
    step_cst = np.asarray(step_cst, dtype=float)
    zero_cst = np.asarray(zero_cst, dtype=float)
    window = np.asarray(window_rel_s, dtype=float).reshape(-1)
    if window.size != 2:
        window = np.asarray([0.0, 1.0], dtype=float)
    max_lag_s = float(max_lag_s)
    updates = _empty_xcorr_updates("not_computed", window, max_lag_s)
    if t_rel_s.size < 3 or step_cst.shape != t_rel_s.shape or zero_cst.shape != t_rel_s.shape:
        updates["xcorr_status"] = "invalid_trace_shape"
        return updates
    if not (np.isfinite(max_lag_s) and max_lag_s >= 0.0):
        updates["xcorr_status"] = "invalid_max_lag"
        return updates
    mask = (
        (t_rel_s >= float(window[0]))
        & (t_rel_s <= float(window[1]))
        & np.isfinite(t_rel_s)
        & np.isfinite(step_cst)
        & np.isfinite(zero_cst)
    )
    if np.count_nonzero(mask) < 3:
        updates["xcorr_status"] = "insufficient_window_samples"
        return updates
    t = t_rel_s[mask]
    step = step_cst[mask]
    zero = zero_cst[mask]
    dt_values = np.diff(t)
    dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0)]
    if dt_values.size == 0:
        updates["xcorr_status"] = "invalid_time_grid"
        return updates
    dt = float(np.median(dt_values))
    max_lag_samples = int(np.floor(max_lag_s / dt))
    lag_samples = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=int)
    corr_values = np.full(lag_samples.shape, np.nan, dtype=float)
    all_indices = np.arange(step.size, dtype=int)
    for out_i, lag_i in enumerate(lag_samples):
        zero_indices = all_indices - int(lag_i)
        valid = (zero_indices >= 0) & (zero_indices < zero.size)
        if np.count_nonzero(valid) < 3:
            continue
        a = step[valid]
        b = zero[zero_indices[valid]]
        finite = np.isfinite(a) & np.isfinite(b)
        if np.count_nonzero(finite) < 3:
            continue
        a = a[finite] - float(np.mean(a[finite]))
        b = b[finite] - float(np.mean(b[finite]))
        denom = float(np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))
        if denom <= 1e-18:
            continue
        corr_values[out_i] = float(np.sum(a * b) / denom)
    if not np.any(np.isfinite(corr_values)):
        updates["xcorr_status"] = "no_finite_correlations"
        updates["xcorr_lags_s"] = lag_samples.astype(float) * dt
        updates["xcorr_values"] = corr_values
        return updates
    best_i = int(np.nanargmax(corr_values))
    lag_s = float(lag_samples[best_i] * dt)
    peak_corr = float(corr_values[best_i])
    sign_corrected = -float(step_sign) * lag_s if np.isfinite(step_sign) else np.nan
    updates.update({
        "xcorr_lag_s": np.asarray([lag_s], dtype=float),
        "xcorr_peak_corr": np.asarray([peak_corr], dtype=float),
        "xcorr_lag_sign_corrected_s": np.asarray([sign_corrected], dtype=float),
        "xcorr_lags_s": lag_samples.astype(float) * dt,
        "xcorr_values": corr_values,
        "xcorr_status": "ok",
    })
    return updates


def _unmatched_step_response_finalization_updates(step_diag=None, status="matched_no_step_unavailable"):
    step_diag = {} if step_diag is None else step_diag
    fit_window = np.asarray(step_diag.get("response_shape_fit_window_rel_s", [0.0, 0.5]), dtype=float)
    xcorr_window = np.asarray(step_diag.get("xcorr_window_rel_s", [0.0, 1.0]), dtype=float)
    sync_pred_pre_window = np.asarray(step_diag.get("sync_predictor_pre_step_window_rel_s", [-0.250, 0.000]), dtype=float)
    sync_pred_local_window = np.asarray(step_diag.get("sync_predictor_local_step_window_rel_s", [-0.250, 0.250]), dtype=float)
    max_lag_s = _float_from_mapping(step_diag, "xcorr_max_lag_s", 0.25)
    updates = {
        "matched_step_effect_available": np.asarray([False], dtype=bool),
        "response_has_expected_sign": np.asarray([False], dtype=bool),
        "response_magnitude_sufficient": np.asarray([False], dtype=bool),
        "linear_initial_slope_cst_per_s": np.asarray([np.nan], dtype=float),
        "linear_initial_slope_normalized_per_nA": np.asarray([np.nan], dtype=float),
        "linear_initial_slope_normalized_per_uA": np.asarray([np.nan], dtype=float),
        "linear_selected_intercept": np.asarray([np.nan], dtype=float),
        "linear_selected_window_end_s": np.asarray([np.nan], dtype=float),
        "linear_selected_window_r2": np.asarray([np.nan], dtype=float),
        "linear_selected_window_response_fraction": np.asarray([np.nan], dtype=float),
        "linear_slope_selection_status": str(status),
        "linear_slope_fit_valid": np.asarray([False], dtype=bool),
        # Transitional aliases for old readers.
        "initial_slope_cst_per_s": np.asarray([np.nan], dtype=float),
        "initial_slope_normalized_per_nA": np.asarray([np.nan], dtype=float),
        "initial_slope_normalized_per_uA": np.asarray([np.nan], dtype=float),
        "selected_slope_intercept": np.asarray([np.nan], dtype=float),
        "selected_slope_window_end_s": np.asarray([np.nan], dtype=float),
        "selected_slope_window_r2": np.asarray([np.nan], dtype=float),
        "selected_slope_window_response_fraction": np.asarray([np.nan], dtype=float),
        "slope_selection_status": str(status),
        "slope_fit_valid": np.asarray([False], dtype=bool),
    }
    updates.update(_empty_response_shape_updates(status, fit_window))
    updates.update(_empty_xcorr_updates(status, xcorr_window, max_lag_s))
    sync_source = "missing_matched_no_step" if "zero_step_reference" not in str(status) else "self_no_step_missing_sync"
    updates.update(_empty_sync_predictor_updates(sync_source, sync_pred_pre_window, sync_pred_local_window))
    return updates


def _compute_matched_step_effect_metrics(step_diag, zero_diag):
    t_rel = np.asarray(step_diag.get("t_step_rel_s", step_diag.get("time_rel_step_s")), dtype=float)
    step_cst = _trace_from_mapping(step_diag, "cst_smooth_step_response", "cst_smooth_causal_exp")
    zero_t_rel = np.asarray(zero_diag.get("t_step_rel_s", zero_diag.get("time_rel_step_s")), dtype=float)
    zero_cst = _trace_from_mapping(zero_diag, "cst_smooth_step_response", "cst_smooth_causal_exp")
    if zero_t_rel.shape != t_rel.shape or not np.allclose(zero_t_rel, t_rel, atol=1e-9, rtol=0):
        zero_cst = np.interp(t_rel, zero_t_rel, zero_cst, left=np.nan, right=np.nan)
        alignment_status = "interpolated_no_step_trace"
    else:
        alignment_status = "matched_sample_grid"

    step_amp = _float_from_mapping(step_diag, "step_current_amplitude_nA", np.nan)
    step_sign = float(np.sign(step_amp)) if np.isfinite(step_amp) and abs(step_amp) > 1e-12 else 0.0
    effect = step_cst - zero_cst
    effect_sc = step_sign * effect

    pre_rel = np.asarray(step_diag.get("step_pre_window_rel_s", [-1.0, -0.2]), dtype=float).reshape(-1)
    late_rel = np.asarray(step_diag.get("step_late_post_window_rel_s", [0.6, 1.2]), dtype=float).reshape(-1)
    if pre_rel.size != 2:
        pre_rel = np.asarray([-1.0, -0.2], dtype=float)
    if late_rel.size != 2:
        late_rel = np.asarray([0.6, 1.2], dtype=float)

    pre = _window_summary_from_trace(t_rel, effect, pre_rel[0], pre_rel[1])
    late = _window_summary_from_trace(t_rel, effect, late_rel[0], late_rel[1])
    pre_sc = _window_summary_from_trace(t_rel, effect_sc, pre_rel[0], pre_rel[1])
    late_sc = _window_summary_from_trace(t_rel, effect_sc, late_rel[0], late_rel[1])
    delta_effect = (
        float(late["median"] - pre["median"])
        if np.isfinite(late["median"]) and np.isfinite(pre["median"])
        else np.nan
    )
    r_inf = float(late_sc["median"]) if np.isfinite(late_sc["median"]) else np.nan
    gain = np.nan if abs(step_amp) < 1e-12 else delta_effect / step_amp
    gain_sc = np.nan if abs(step_amp) < 1e-12 else step_sign * delta_effect / abs(step_amp)
    gain_per_uA = gain * 1000.0 if np.isfinite(gain) else np.nan
    gain_sc_per_uA = gain_sc * 1000.0 if np.isfinite(gain_sc) else np.nan
    response_has_expected_sign = bool(np.isfinite(r_inf) and r_inf > 0)
    response_magnitude_sufficient = bool(np.isfinite(r_inf) and r_inf > 1e-9)

    r2_min = _float_from_mapping(step_diag, "step_slope_r2_min", 0.90)
    frac_min = _float_from_mapping(step_diag, "step_slope_response_fraction_min", 0.25)
    cand_min = _float_from_mapping(step_diag, "step_slope_candidate_min_s", 0.050)
    cand_max = _float_from_mapping(step_diag, "step_slope_candidate_max_s", 0.300)
    cand_step = _float_from_mapping(step_diag, "step_slope_candidate_step_s", 0.010)
    candidate_ends = np.arange(cand_min, cand_max + 0.5 * cand_step, cand_step, dtype=float)
    candidates = []
    for end_s in candidate_ends:
        mask = (t_rel >= 0.0) & (t_rel <= float(end_s)) & np.isfinite(effect_sc)
        slope, intercept, r2 = _linear_fit_slope_r2(t_rel[mask], effect_sc[mask])
        endpoint = intercept + slope * float(end_s) if np.isfinite(slope) and np.isfinite(intercept) else np.nan
        frac = endpoint / r_inf if np.isfinite(endpoint) and np.isfinite(r_inf) and abs(r_inf) > 1e-12 else np.nan
        candidates.append((float(end_s), slope, intercept, r2, frac))

    selected = None
    status = "insufficient_response"
    if response_magnitude_sufficient:
        for candidate in candidates:
            end_s, slope, intercept, r2, frac = candidate
            if (
                np.isfinite(slope) and slope > 0
                and np.isfinite(r2) and r2 >= r2_min
                and np.isfinite(frac) and frac >= frac_min
            ):
                selected = candidate
                status = "passed"
                break
        if selected is None:
            valid = [
                candidate for candidate in candidates
                if np.isfinite(candidate[1]) and candidate[1] > 0 and np.isfinite(candidate[3])
            ]
            if valid:
                selected = max(valid, key=lambda item: item[3])
                status = "fallback_best_valid"
    if selected is None:
        selected = (np.nan, np.nan, np.nan, np.nan, np.nan)
    selected_end, selected_slope, selected_intercept, selected_r2, selected_frac = selected

    updates = {
        "diagnostics_stage": "finalized_matched",
        "matched_step_effect_available": np.asarray([True], dtype=bool),
        "matched_trace_alignment_status": alignment_status,
        "cst_step_effect": effect,
        "cst_step_effect_sign_corrected": effect_sc,
        "matched_no_step_cst_smooth_step_response": zero_cst,
        "matched_no_step_cst_smooth_causal_exp": zero_cst,
        "step_sign": np.asarray([step_sign], dtype=float),
        "pre_step_effect_mean": np.asarray([pre["mean"]], dtype=float),
        "pre_step_effect_median": np.asarray([pre["median"]], dtype=float),
        "pre_step_effect_sd": np.asarray([pre["sd"]], dtype=float),
        "late_post_step_effect_mean": np.asarray([late["mean"]], dtype=float),
        "late_post_step_effect_median": np.asarray([late["median"]], dtype=float),
        "late_post_step_effect_sd": np.asarray([late["sd"]], dtype=float),
        "late_post_step_effect_sign_corrected_median": np.asarray([late_sc["median"]], dtype=float),
        "delta_cst_effect": np.asarray([delta_effect], dtype=float),
        "response_gain_cst_per_nA": np.asarray([gain], dtype=float),
        "response_gain_cst_per_uA": np.asarray([gain_per_uA], dtype=float),
        "response_gain_sign_corrected_cst_per_nA": np.asarray([gain_sc], dtype=float),
        "response_gain_sign_corrected_cst_per_uA": np.asarray([gain_sc_per_uA], dtype=float),
        "response_has_expected_sign": np.asarray([response_has_expected_sign], dtype=bool),
        "response_magnitude_sufficient": np.asarray([response_magnitude_sufficient], dtype=bool),
        "baseline_sync_mean": np.asarray([_float_from_mapping(step_diag, "pre_step_sync_mean", np.nan)], dtype=float),
        "baseline_sync_sd": np.asarray([_float_from_mapping(step_diag, "pre_step_sync_sd", np.nan)], dtype=float),
        "post_step_sync_mean_actual_step": np.asarray([_float_from_mapping(step_diag, "post_step_sync_mean", np.nan)], dtype=float),
        "post_step_sync_sd_actual_step": np.asarray([_float_from_mapping(step_diag, "post_step_sync_sd", np.nan)], dtype=float),
        "linear_initial_slope_cst_per_s": np.asarray([selected_slope], dtype=float),
        "linear_initial_slope_normalized_per_nA": np.asarray([np.nan if abs(step_amp) < 1e-12 else selected_slope / abs(step_amp)], dtype=float),
        "linear_initial_slope_normalized_per_uA": np.asarray([np.nan if abs(step_amp) < 1e-12 else 1000.0 * selected_slope / abs(step_amp)], dtype=float),
        "linear_selected_intercept": np.asarray([selected_intercept], dtype=float),
        "linear_selected_window_end_s": np.asarray([selected_end], dtype=float),
        "linear_selected_window_r2": np.asarray([selected_r2], dtype=float),
        "linear_selected_window_response_fraction": np.asarray([selected_frac], dtype=float),
        "linear_slope_selection_status": status,
        "linear_slope_fit_valid": np.asarray([status in {"passed", "fallback_best_valid"}], dtype=bool),
        # Transitional aliases for previously saved/loaded code paths.
        "initial_slope_cst_per_s": np.asarray([selected_slope], dtype=float),
        "initial_slope_normalized_per_nA": np.asarray([np.nan if abs(step_amp) < 1e-12 else selected_slope / abs(step_amp)], dtype=float),
        "initial_slope_normalized_per_uA": np.asarray([np.nan if abs(step_amp) < 1e-12 else 1000.0 * selected_slope / abs(step_amp)], dtype=float),
        "selected_slope_intercept": np.asarray([selected_intercept], dtype=float),
        "selected_slope_window_end_s": np.asarray([selected_end], dtype=float),
        "selected_slope_window_r2": np.asarray([selected_r2], dtype=float),
        "selected_slope_window_response_fraction": np.asarray([selected_frac], dtype=float),
        "slope_selection_status": status,
        "slope_fit_valid": np.asarray([status in {"passed", "fallback_best_valid"}], dtype=bool),
    }
    updates.update(_fit_response_shape_metrics(
        t_rel,
        effect_sc,
        r_inf=r_inf,
        fit_window_rel_s=np.asarray(step_diag.get("response_shape_fit_window_rel_s", [0.0, 0.5]), dtype=float),
        tau_min_ms=_float_from_mapping(step_diag, "response_shape_tau_min_ms", 1.0),
        tau_max_ms=_float_from_mapping(step_diag, "response_shape_tau_max_ms", 5000.0),
        min_r_inf=_float_from_mapping(step_diag, "response_shape_min_r_inf", 1e-9),
    ))
    updates.update(_compute_xcorr_lag_metrics(
        t_rel,
        step_cst,
        zero_cst,
        step_sign=step_sign,
        window_rel_s=np.asarray(step_diag.get("xcorr_window_rel_s", [0.0, 1.0]), dtype=float),
        max_lag_s=_float_from_mapping(step_diag, "xcorr_max_lag_s", 0.25),
    ))
    sync_pred_pre_window = np.asarray(step_diag.get("sync_predictor_pre_step_window_rel_s", [-0.250, 0.000]), dtype=float)
    sync_pred_local_window = np.asarray(step_diag.get("sync_predictor_local_step_window_rel_s", [-0.250, 0.250]), dtype=float)
    if "sync_trace" in zero_diag:
        zero_sync_t = np.asarray(zero_diag.get("sync_time_rel_step_s"), dtype=float)
        zero_sync_trace = np.asarray(zero_diag["sync_trace"], dtype=float)
        updates.update(_compute_sync_predictor_updates(
            zero_sync_t,
            zero_sync_trace,
            source="matched_no_step",
            pre_window_rel_s=sync_pred_pre_window,
            local_window_rel_s=sync_pred_local_window,
        ))
    else:
        updates.update(_empty_sync_predictor_updates(
            "missing_matched_no_step",
            sync_pred_pre_window,
            sync_pred_local_window,
        ))
    if "sync_trace" in step_diag and "sync_trace" in zero_diag:
        try:
            sync_t = np.asarray(step_diag.get("sync_time_rel_step_s"), dtype=float)
            sync_trace = np.asarray(step_diag["sync_trace"], dtype=float)
            zero_sync_t = np.asarray(zero_diag.get("sync_time_rel_step_s"), dtype=float)
            zero_sync_trace = np.asarray(zero_diag["sync_trace"], dtype=float)
            if sync_t.size and sync_trace.size and zero_sync_t.size and zero_sync_trace.size:
                if zero_sync_t.shape != sync_t.shape or not np.allclose(zero_sync_t, sync_t, atol=1e-9, rtol=0):
                    zero_sync_on_step_grid = np.interp(sync_t, zero_sync_t, zero_sync_trace, left=np.nan, right=np.nan)
                    sync_alignment_status = "interpolated_no_step_sync_trace"
                else:
                    zero_sync_on_step_grid = zero_sync_trace
                    sync_alignment_status = "matched_sync_sample_grid"
                sync_effect = sync_trace - zero_sync_on_step_grid
                updates.update({
                    "matched_no_step_sync_trace": zero_sync_on_step_grid,
                    "sync_step_effect": sync_effect,
                    "sync_step_effect_sign_corrected": step_sign * sync_effect,
                    "matched_sync_trace_alignment_status": sync_alignment_status,
                })
        except Exception as exc:
            updates["matched_sync_trace_alignment_status"] = f"sync_match_error_{type(exc).__name__}"
    return updates


def _load_step_response_record(h5_path):
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        params = _read_h5_group_to_dict(f.get("simulation_parameters"))
        step_diag = _read_h5_group_to_dict(f.get("step_response_diagnostics"))
    amp = _float_from_mapping(step_diag, "step_current_amplitude_nA", _float_from_mapping(params, "step_current_amplitude_nA", np.nan))
    zero_flag = _bool_from_mapping(step_diag, "is_zero_step_reference", False) or _bool_from_mapping(params, "is_zero_step_reference", False)
    zero_flag = bool(zero_flag or (np.isfinite(amp) and abs(amp) < 1e-12))
    return {
        "h5_path": h5_path,
        "params": params,
        "step_diag": step_diag,
        "step_current_amplitude_nA": amp,
        "is_zero_step_reference": zero_flag,
        "parameter_set_id": _float_from_mapping(params, "parameter_set_id", np.nan),
        "repeat_index": _float_from_mapping(params, "repeat_index", np.nan),
        "requested_random_seed": _float_from_mapping(params, "requested_random_seed", _float_from_mapping(params, "random_seed", np.nan)),
    }


def _build_plot_proxy_params(params_dict, input_components=None):
    style = params_dict.get("step_diagnostic_plot_style", {})
    if not isinstance(style, dict):
        style = {}
    fsamp = float(params_dict.get("fsamp", 1.0))
    if input_components is not None and "_saved_fsamp_Hz" in input_components:
        fsamp = float(input_components["_saved_fsamp_Hz"])
    return SimpleNamespace(
        fsamp=fsamp,
        step_diagnostic_plot_style=style,
        step_diagnostic_display_window_rel_s=params_dict.get("step_diagnostic_display_window_rel_s", [-1.5, 1.5]),
    )


def _read_plot_groups_from_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        params = _read_h5_group_to_dict(f.get("simulation_parameters"))
        input_components = _read_h5_group_to_dict(f.get("input_components"))
        cst_diagnostics = _read_h5_group_to_dict(f.get("cst_diagnostics"))
        step_response_diagnostics = _read_h5_group_to_dict(f.get("step_response_diagnostics"))
        if "input_components" in f:
            input_components["_saved_fsamp_Hz"] = float(f["input_components"].attrs.get("saved_fsamp_Hz", params.get("fsamp", 1.0)))
    return params, input_components, cst_diagnostics, step_response_diagnostics


def _regenerate_step_response_figure_from_h5(h5_path, plot_style=None):
    params_dict, input_components, cst_diagnostics, step_diag = _read_plot_groups_from_h5(h5_path)
    if plot_style is not None:
        params_dict["step_diagnostic_plot_style"] = dict(plot_style)
    params_proxy = _build_plot_proxy_params(params_dict, input_components=input_components)
    plot_step_response_diagnostic(
        save_dir=str(Path(h5_path).parent),
        input_components=input_components,
        cst_diagnostics=cst_diagnostics,
        step_response_diagnostics=step_diag,
        params=params_proxy,
    )


def finalize_step_response_repeat(h5_paths, *, regenerate_figure=True, plot_style=None):
    h5_paths = list(h5_paths)
    records = []
    for h5_path in h5_paths:
        try:
            record = _load_step_response_record(h5_path)
            if "cst_smooth_step_response" not in record["step_diag"] and "cst_smooth_causal_exp" not in record["step_diag"]:
                updates = _unmatched_step_response_finalization_updates(
                    record.get("step_diag", {}),
                    status="missing_stage1_step_response_cst",
                )
                updates["finalization_status"] = "missing_stage1_step_response_cst"
                _try_write_step_response_updates(h5_path, updates, context="missing-stage1 finalization status")
                continue
            records.append(record)
        except Exception as exc:
            warnings.warn(f"Could not load step-response record from {h5_path}: {exc}")
    if not records:
        return {"status": "no_valid_step_response_records", "n_paths": len(h5_paths)}

    zero_records = [record for record in records if record["is_zero_step_reference"]]
    if not zero_records:
        for record in records:
            updates = _unmatched_step_response_finalization_updates(
                record.get("step_diag", {}),
                status="missing_zero_step_reference",
            )
            updates["finalization_status"] = "missing_zero_step_reference"
            _try_write_step_response_updates(record["h5_path"], updates, context="missing-zero finalization status")
        return {
            "status": "missing_zero_step_reference",
            "n_paths": len(records),
            "n_finalized": 0,
            "n_zero_references": 0,
        }

    zero_record = sorted(zero_records, key=lambda item: str(item["h5_path"]))[0]
    zero_updates = _unmatched_step_response_finalization_updates(
        zero_record.get("step_diag", {}),
        status="zero_step_reference",
    )
    zero_updates.update({
        "diagnostics_stage": "finalized_zero_reference",
        "matched_step_effect_available": np.asarray([False], dtype=bool),
        "zero_step_condition": np.asarray([True], dtype=bool),
        "response_gain_cst_per_nA": np.asarray([np.nan], dtype=float),
        "response_gain_cst_per_uA": np.asarray([np.nan], dtype=float),
        "response_gain_sign_corrected_cst_per_nA": np.asarray([np.nan], dtype=float),
        "response_gain_sign_corrected_cst_per_uA": np.asarray([np.nan], dtype=float),
        "baseline_sync_mean": np.asarray([_float_from_mapping(zero_record["step_diag"], "pre_step_sync_mean", np.nan)], dtype=float),
        "baseline_sync_sd": np.asarray([_float_from_mapping(zero_record["step_diag"], "pre_step_sync_sd", np.nan)], dtype=float),
        "post_step_sync_mean_actual_step": np.asarray([_float_from_mapping(zero_record["step_diag"], "post_step_sync_mean", np.nan)], dtype=float),
        "post_step_sync_sd_actual_step": np.asarray([_float_from_mapping(zero_record["step_diag"], "post_step_sync_sd", np.nan)], dtype=float),
        "finalization_status": "zero_step_reference",
    })
    if "sync_trace" in zero_record["step_diag"]:
        zero_updates.update(_compute_sync_predictor_updates(
            np.asarray(zero_record["step_diag"].get("sync_time_rel_step_s"), dtype=float),
            np.asarray(zero_record["step_diag"]["sync_trace"], dtype=float),
            source="self_no_step",
            pre_window_rel_s=np.asarray(zero_record["step_diag"].get("sync_predictor_pre_step_window_rel_s", [-0.250, 0.000]), dtype=float),
            local_window_rel_s=np.asarray(zero_record["step_diag"].get("sync_predictor_local_step_window_rel_s", [-0.250, 0.250]), dtype=float),
        ))
    if not _try_write_step_response_updates(zero_record["h5_path"], zero_updates, context="zero-step finalization"):
        return {
            "status": "zero_reference_write_failed",
            "parameter_set_id": zero_record["parameter_set_id"],
            "repeat_index": zero_record["repeat_index"],
            "requested_random_seed": zero_record["requested_random_seed"],
            "n_paths": len(records),
            "n_zero_references": len(zero_records),
            "zero_reference_h5_path": str(zero_record["h5_path"]),
            "n_finalized": 0,
            "n_failed": len(records) - 1,
        }
    if regenerate_figure:
        try:
            _regenerate_step_response_figure_from_h5(zero_record["h5_path"], plot_style=plot_style)
        except Exception as exc:
            warnings.warn(f"Could not regenerate zero-step diagnostic figure {zero_record['h5_path']}: {exc}")

    finalized = 0
    failed = 0
    for record in records:
        if record["h5_path"] == zero_record["h5_path"]:
            continue
        try:
            updates = _compute_matched_step_effect_metrics(record["step_diag"], zero_record["step_diag"])
            updates["matched_no_step_h5_path"] = str(zero_record["h5_path"])
            updates["finalization_status"] = "finalized_matched"
            if not _try_write_step_response_updates(record["h5_path"], updates, context="matched finalization"):
                raise OSError("matched finalization write failed")
            finalized += 1
            if regenerate_figure:
                try:
                    _regenerate_step_response_figure_from_h5(record["h5_path"], plot_style=plot_style)
                except Exception as exc:
                    warnings.warn(f"Could not regenerate step diagnostic figure {record['h5_path']}: {exc}")
        except Exception as exc:
            failed += 1
            updates = _unmatched_step_response_finalization_updates(
                record.get("step_diag", {}),
                status="finalization_error",
            )
            updates.update({
                "finalization_status": "finalization_error",
                "finalization_error_message": str(exc),
            })
            _try_write_step_response_updates(record["h5_path"], updates, context="finalization-error status")
            warnings.warn(f"Could not finalize matched step response for {record['h5_path']}: {exc}")

    first = records[0]
    return {
        "status": "ok" if failed == 0 else "partial_failure",
        "parameter_set_id": first["parameter_set_id"],
        "repeat_index": first["repeat_index"],
        "requested_random_seed": first["requested_random_seed"],
        "n_paths": len(records),
        "n_zero_references": len(zero_records),
        "zero_reference_h5_path": str(zero_record["h5_path"]),
        "n_finalized": finalized,
        "n_failed": failed,
    }


def _coerce_finalization_paths(run_table_or_paths):
    if isinstance(run_table_or_paths, pd.DataFrame):
        if "simulation_output_h5" in run_table_or_paths.columns:
            return run_table_or_paths["simulation_output_h5"].dropna().astype(str).tolist()
        if "h5_path" in run_table_or_paths.columns:
            return run_table_or_paths["h5_path"].dropna().astype(str).tolist()
        raise ValueError("run table must contain 'simulation_output_h5' or 'h5_path'")
    return [str(path) for path in run_table_or_paths]


def _path_column_from_run_table(run_table):
    if "simulation_output_h5" in run_table.columns:
        return "simulation_output_h5"
    if "h5_path" in run_table.columns:
        return "h5_path"
    raise ValueError("run table must contain 'simulation_output_h5' or 'h5_path'")


def _group_finalization_paths_from_table(run_table, group_columns):
    if not isinstance(run_table, pd.DataFrame):
        return None
    path_col = _path_column_from_run_table(run_table)
    if not all(col in run_table.columns for col in group_columns):
        return None
    table = run_table.dropna(subset=[path_col]).copy()
    if table.empty:
        return []
    groups = []
    for _, group_df in table.groupby(list(group_columns), dropna=False, sort=True):
        groups.append(group_df[path_col].dropna().astype(str).tolist())
    return groups


def _safe_finalize_step_response_repeat(paths, *, regenerate_figures=True, plot_style=None):
    try:
        return finalize_step_response_repeat(paths, regenerate_figure=regenerate_figures, plot_style=plot_style)
    except Exception as exc:
        warnings.warn(f"Step-response repeat finalization crashed for {len(paths)} paths: {exc}")
        return {
            "status": "repeat_finalization_exception",
            "n_paths": len(paths),
            "n_finalized": 0,
            "n_failed": len(paths),
            "finalization_error_message": str(exc),
        }


def finalize_step_response_batch(run_table_or_paths, *, n_jobs=1, regenerate_figures=True, plot_style=None):
    items = _group_finalization_paths_from_table(
        run_table_or_paths,
        ["parameter_set_id", "repeat_index", "requested_random_seed"],
    )
    if items is None:
        paths = _coerce_finalization_paths(run_table_or_paths)
        if not paths:
            return pd.DataFrame()
        records = []
        for path in paths:
            try:
                records.append(_load_step_response_record(path))
            except Exception as exc:
                warnings.warn(f"Skipping {path}: {exc}")
        if not records:
            return pd.DataFrame()
        groups = {}
        for record in records:
            key = (
                record["parameter_set_id"],
                record["repeat_index"],
                record["requested_random_seed"],
            )
            groups.setdefault(key, []).append(record["h5_path"])
        items = list(groups.values())
    if not items:
        return pd.DataFrame()
    n_jobs = int(n_jobs)
    if n_jobs <= 1:
        summaries = [
            _safe_finalize_step_response_repeat(paths, regenerate_figures=regenerate_figures, plot_style=plot_style)
            for paths in items
        ]
    else:
        from joblib import Parallel, delayed
        summaries = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(_safe_finalize_step_response_repeat)(
                paths,
                regenerate_figures=regenerate_figures,
                plot_style=plot_style,
            )
            for paths in items
        )
    return pd.DataFrame(summaries)


def _format_step_folder_name_for_output(step_amplitude_nA: float) -> str:
    amplitude = float(step_amplitude_nA)
    abs_amplitude = abs(amplitude)
    prefix = "m" if amplitude < 0 else "p"
    if np.isclose(abs_amplitude, round(abs_amplitude), atol=1e-9, rtol=0.0):
        label = f"{int(round(abs_amplitude)):05d}"
    else:
        label = f"{abs_amplitude:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"step_{prefix}{label}_nA"


def _parameter_set_dir_from_h5_path(h5_path):
    path = Path(h5_path).resolve()
    for parent in path.parents:
        if parent.name.startswith("parameter_set_"):
            return parent
    return path.parent


def _extract_step_trace_for_mean(step_diag, key, ref_t):
    if key not in step_diag:
        return None, "missing"
    trace = np.asarray(step_diag[key], dtype=float)
    t = np.asarray(step_diag.get("t_step_rel_s", step_diag.get("time_rel_step_s")), dtype=float)
    ref_t = np.asarray(ref_t, dtype=float)
    if trace.shape != t.shape or t.size == 0:
        return None, "invalid_shape"
    if t.shape == ref_t.shape and np.allclose(t, ref_t, atol=1e-9, rtol=0):
        return trace, "matched"
    return np.interp(ref_t, t, trace, left=np.nan, right=np.nan), "interpolated"


def _extract_sync_trace_for_mean(step_diag, key, ref_t):
    if key not in step_diag:
        return None, "missing"
    trace = np.asarray(step_diag[key], dtype=float)
    t = np.asarray(step_diag.get("sync_time_rel_step_s", []), dtype=float)
    ref_t = np.asarray(ref_t, dtype=float)
    if trace.shape != t.shape or t.size == 0:
        return None, "invalid_shape"
    if t.shape == ref_t.shape and np.allclose(t, ref_t, atol=1e-9, rtol=0):
        return trace, "matched"
    return np.interp(ref_t, t, trace, left=np.nan, right=np.nan), "interpolated"


def _nanmean_stack(traces):
    traces = [np.asarray(trace, dtype=float) for trace in traces if trace is not None]
    if not traces:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.stack(traces, axis=0), axis=0)


def _nanstd_stack(traces):
    traces = [np.asarray(trace, dtype=float) for trace in traces if trace is not None]
    if not traces:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(np.stack(traces, axis=0), axis=0)


def _mean_scalar_from_records(records, key):
    values = [_float_from_mapping(record["step_diag"], key, np.nan) for record in records]
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def _copy_first_available(step_diag, keys, default=np.nan):
    for key in keys:
        if key in step_diag:
            return step_diag[key]
    return default


def _build_mean_step_diag_for_records(records, ref_record):
    ref_diag = ref_record["step_diag"]
    ref_t = np.asarray(ref_diag.get("t_step_rel_s", ref_diag.get("time_rel_step_s")), dtype=float)
    if ref_t.size == 0:
        raise ValueError("Reference record has no step-relative CST time vector")
    ref_sync_t = np.asarray(ref_diag.get("sync_time_rel_step_s", []), dtype=float)

    trace_keys = [
        "cst_smooth_step_response",
        "matched_no_step_cst_smooth_step_response",
        "cst_step_effect",
        "cst_step_effect_sign_corrected",
    ]
    sync_keys = [
        "sync_trace",
        "matched_no_step_sync_trace",
        "sync_step_effect",
        "sync_step_effect_sign_corrected",
    ]
    mean_diag = {
        "status": "ok",
        "diagnostics_stage": "mean_trace_input",
        "t_step_rel_s": ref_t,
        "time_rel_step_s": ref_t,
        "time_s": ref_t + _float_from_mapping(ref_diag, "step_current_time_abs_s", 0.0),
    }
    interpolation_status = []
    trace_arrays_by_key = {}
    for key in trace_keys:
        traces = []
        for record in records:
            trace, status = _extract_step_trace_for_mean(record["step_diag"], key, ref_t)
            if trace is not None:
                traces.append(trace)
            if status == "interpolated":
                interpolation_status.append(f"{key}:{record['h5_path']}")
        mean_trace = _nanmean_stack(traces)
        sd_trace = _nanstd_stack(traces)
        if mean_trace is not None:
            mean_diag[key] = mean_trace
            mean_diag[f"{key}_across_repeat_sd"] = sd_trace
            trace_arrays_by_key[key] = traces
    if "cst_smooth_step_response" not in mean_diag and "cst_smooth_causal_exp" in ref_diag:
        traces = []
        for record in records:
            trace, status = _extract_step_trace_for_mean(record["step_diag"], "cst_smooth_causal_exp", ref_t)
            if trace is not None:
                traces.append(trace)
            if status == "interpolated":
                interpolation_status.append(f"cst_smooth_causal_exp:{record['h5_path']}")
        mean_trace = _nanmean_stack(traces)
        if mean_trace is not None:
            mean_diag["cst_smooth_step_response"] = mean_trace
            mean_diag["cst_smooth_causal_exp"] = mean_trace
            trace_arrays_by_key["cst_smooth_step_response"] = traces

    if ref_sync_t.size:
        mean_diag["sync_time_rel_step_s"] = ref_sync_t
        for key in sync_keys:
            traces = []
            for record in records:
                trace, status = _extract_sync_trace_for_mean(record["step_diag"], key, ref_sync_t)
                if trace is not None:
                    traces.append(trace)
                if status == "interpolated":
                    interpolation_status.append(f"{key}:{record['h5_path']}")
            mean_trace = _nanmean_stack(traces)
            sd_trace = _nanstd_stack(traces)
            if mean_trace is not None:
                mean_diag[key] = mean_trace
                mean_diag[f"{key}_across_repeat_sd"] = sd_trace
                trace_arrays_by_key[key] = traces

    scalar_keys = [
        "step_current_amplitude_nA",
        "step_current_amplitude_percent_baseline",
        "step_current_time_abs_s",
        "step_current_time_usable_rel_s",
        "step_current_duration_s",
        "baseline_tonic_input_nA",
        "lf_input_amplitude_nA",
        "alpha_input_amplitude_nA",
        "beta_input_amplitude_nA",
        "step_response_cst_smoothing_tau_s",
        "step_response_cst_hann_half_width_s",
        "step_slope_candidate_min_s",
        "step_slope_candidate_max_s",
        "step_slope_candidate_step_s",
        "step_slope_r2_min",
        "step_slope_response_fraction_min",
        "response_shape_tau_min_ms",
        "response_shape_tau_max_ms",
        "response_shape_min_r_inf",
        "xcorr_max_lag_s",
    ]
    for key in scalar_keys:
        mean_diag[key] = np.asarray([_mean_scalar_from_records(records, key)], dtype=float)
    for key in [
        "step_pre_window_rel_s",
        "step_late_post_window_rel_s",
        "response_shape_fit_window_rel_s",
        "xcorr_window_rel_s",
        "sync_predictor_pre_step_window_rel_s",
        "sync_predictor_local_step_window_rel_s",
    ]:
        mean_diag[key] = np.asarray(_copy_first_available(ref_diag, [key], []), dtype=float)

    for key in [
        "pre_step_mean_firing_rate_hz",
        "pre_step_sd_firing_rate_hz",
        "pre_step_isi_cv_mean",
        "pre_step_isi_cv_sd",
        "post_step_mean_firing_rate_hz",
        "post_step_sd_firing_rate_hz",
        "post_step_isi_cv_mean",
        "post_step_isi_cv_sd",
    ]:
        mean_diag[key] = np.asarray([_mean_scalar_from_records(records, key)], dtype=float)

    if "sync_time_rel_step_s" in mean_diag and "sync_trace" in mean_diag:
        sync_t = np.asarray(mean_diag["sync_time_rel_step_s"], dtype=float)
        sync_trace = np.asarray(mean_diag["sync_trace"], dtype=float)
        pre_rel = np.asarray(mean_diag.get("step_pre_window_rel_s", [-1.0, -0.2]), dtype=float).reshape(-1)
        late_rel = np.asarray(mean_diag.get("step_late_post_window_rel_s", [0.6, 1.2]), dtype=float).reshape(-1)
        if pre_rel.size != 2:
            pre_rel = np.asarray([-1.0, -0.2], dtype=float)
        if late_rel.size != 2:
            late_rel = np.asarray([0.6, 1.2], dtype=float)
        pre_sync = _window_summary_from_trace(sync_t, sync_trace, pre_rel[0], pre_rel[1])
        post_sync = _window_summary_from_trace(sync_t, sync_trace, late_rel[0], late_rel[1])
        mean_diag.update({
            "pre_step_sync_mean": np.asarray([pre_sync["mean"]], dtype=float),
            "pre_step_sync_sd": np.asarray([pre_sync["sd"]], dtype=float),
            "post_step_sync_mean": np.asarray([post_sync["mean"]], dtype=float),
            "post_step_sync_sd": np.asarray([post_sync["sd"]], dtype=float),
            "baseline_sync_mean": np.asarray([pre_sync["mean"]], dtype=float),
            "baseline_sync_sd": np.asarray([pre_sync["sd"]], dtype=float),
            "post_step_sync_mean_actual_step": np.asarray([post_sync["mean"]], dtype=float),
            "post_step_sync_sd_actual_step": np.asarray([post_sync["sd"]], dtype=float),
        })
    else:
        for key in [
            "pre_step_sync_mean",
            "pre_step_sync_sd",
            "post_step_sync_mean",
            "post_step_sync_sd",
            "baseline_sync_mean",
            "baseline_sync_sd",
            "post_step_sync_mean_actual_step",
            "post_step_sync_sd_actual_step",
        ]:
            mean_diag[key] = np.asarray([_mean_scalar_from_records(records, key)], dtype=float)

    mean_diag["mean_trace_interpolation_status"] = "interpolated_some_traces" if interpolation_status else "matched_sample_grid"
    mean_diag["interpolated_trace_sources"] = interpolation_status
    return mean_diag, trace_arrays_by_key


def _write_mean_trace_hdf5(output_path, diagnostics, repeat_metadata):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        diag_grp = f.create_group("step_response_diagnostics")
        _write_mapping_to_h5_group(diag_grp, diagnostics)
        meta_grp = f.create_group("repeat_metadata")
        _write_mapping_to_h5_group(meta_grp, repeat_metadata)
    return output_path


def plot_step_response_mean_trace_diagnostic(
        save_dir,
        mean_step_response_diagnostics,
        individual_traces_by_key=None,
        *,
        plot_style=None,
        figure_name="averaged_step_response_diagnostic.png"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    style = dict(plot_style or {})
    individual_traces_by_key = dict(individual_traces_by_key or {})

    def _style_float(key, default):
        try:
            value = style.get(key, default)
            return default if value is None else float(value)
        except (TypeError, ValueError):
            return default

    def _style_lim(key):
        value = style.get(key)
        if value is None:
            return None
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != 2 or not np.all(np.isfinite(arr)) or arr[0] >= arr[1]:
            return None
        return float(arr[0]), float(arr[1])

    t_rel = np.asarray(mean_step_response_diagnostics.get("t_step_rel_s", []), dtype=float)
    if t_rel.size == 0:
        return None
    display_rel = _style_lim("xlim_rel_s")
    if display_rel is None:
        display_window = mean_step_response_diagnostics.get("step_diagnostic_display_window_rel_s", [-1.5, 1.5])
        display_arr = np.asarray(display_window, dtype=float).reshape(-1)
        display_rel = (float(display_arr[0]), float(display_arr[1])) if display_arr.size == 2 else (float(np.nanmin(t_rel)), float(np.nanmax(t_rel)))
    mask = (t_rel >= display_rel[0]) & (t_rel <= display_rel[1])
    if not np.any(mask):
        return None

    figsize = np.asarray(style.get("figsize", [11.5, 6.8]), dtype=float).reshape(-1)
    if figsize.size != 2 or np.any(figsize <= 0):
        figsize = np.asarray([11.5, 6.8], dtype=float)
    fig, (ax_cst, ax_sync) = plt.subplots(
        2,
        1,
        figsize=tuple(figsize),
        dpi=int(_style_float("dpi", 140)),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25], "hspace": _style_float("hspace", 0.12)},
    )

    individual_alpha = _style_float("individual_trace_alpha", 0.16)
    individual_lw = _style_float("individual_trace_lw", 0.8)
    mean_lw = _style_float("mean_trace_lw", 2.2)
    effect_color = style.get("step_effect_color", "#111111")
    sync_trace_mode = str(style.get("sync_trace_mode", "raw")).strip().lower()
    if sync_trace_mode not in {"raw", "effect", "residual"}:
        sync_trace_mode = "raw"
    sync_key = "sync_trace" if sync_trace_mode == "raw" else "sync_step_effect"
    sync_color = style.get(
        "sync_trace_color" if sync_trace_mode == "raw" else "sync_effect_color",
        "#852680" if sync_trace_mode == "raw" else "#111111",
    )
    sync_label = "Mean sync" if sync_trace_mode == "raw" else "Mean sync residual"
    sync_ylabel = "Synchrony" if sync_trace_mode == "raw" else "Sync residual"
    show_individuals = bool(style.get("show_individual_traces", True))
    if show_individuals:
        for trace in individual_traces_by_key.get("cst_step_effect_sign_corrected", []):
            trace = np.asarray(trace, dtype=float)
            if trace.shape == t_rel.shape:
                ax_cst.plot(t_rel[mask], trace[mask], color=effect_color, lw=individual_lw, alpha=individual_alpha, zorder=1)
    mean_effect = np.asarray(mean_step_response_diagnostics.get("cst_step_effect_sign_corrected", []), dtype=float)
    if mean_effect.shape == t_rel.shape:
        ax_cst.plot(t_rel[mask], mean_effect[mask], color=effect_color, lw=mean_lw, alpha=_style_float("mean_trace_alpha", 1.0), zorder=4, label="Mean CST effect")

    pre_win = np.asarray(mean_step_response_diagnostics.get("step_pre_window_rel_s", [-1.0, -0.2]), dtype=float).reshape(-1)
    late_win = np.asarray(mean_step_response_diagnostics.get("step_late_post_window_rel_s", [0.6, 1.2]), dtype=float).reshape(-1)
    if pre_win.size == 2:
        ax_cst.axvspan(pre_win[0], pre_win[1], color=style.get("pre_window_color", "#6babff"), alpha=_style_float("pre_window_alpha", 0.12))
        ax_sync.axvspan(pre_win[0], pre_win[1], color=style.get("pre_window_color", "#6babff"), alpha=_style_float("pre_window_alpha", 0.12))
    if late_win.size == 2:
        ax_cst.axvspan(late_win[0], late_win[1], color=style.get("late_post_window_color", "#ffdb3d"), alpha=_style_float("late_post_window_alpha", 0.12))
        ax_sync.axvspan(late_win[0], late_win[1], color=style.get("late_post_window_color", "#ffdb3d"), alpha=_style_float("late_post_window_alpha", 0.12))
    ax_cst.axvline(0.0, color=style.get("step_line_color", "#555555"), lw=_style_float("step_line_lw", 1.0), ls=style.get("step_line_ls", "--"))
    ax_cst.axhline(0.0, color="0.55", lw=0.8, ls=":")

    slope_end = _float_from_mapping(mean_step_response_diagnostics, "linear_selected_window_end_s", np.nan)
    slope = _float_from_mapping(mean_step_response_diagnostics, "linear_initial_slope_cst_per_s", np.nan)
    intercept = _float_from_mapping(mean_step_response_diagnostics, "linear_selected_intercept", np.nan)
    if bool(style.get("show_slope_fit", True)) and np.isfinite(slope_end) and np.isfinite(slope) and np.isfinite(intercept):
        slope_t = np.asarray([0.0, slope_end], dtype=float)
        ax_cst.plot(
            slope_t,
            intercept + slope * slope_t,
            color=style.get("slope_fit_color", "#000000"),
            lw=_style_float("slope_fit_lw", 2.0),
            alpha=_style_float("slope_fit_alpha", 0.9),
            label="Mean linear fit",
        )
    shape_t = np.asarray(mean_step_response_diagnostics.get("response_shape_fitted_curve_time_rel_s", []), dtype=float)
    shape_y = np.asarray(mean_step_response_diagnostics.get("response_shape_fitted_curve", []), dtype=float)
    if bool(style.get("show_response_shape_fit", True)) and shape_t.shape == shape_y.shape and shape_t.size:
        shape_mask = (shape_t >= display_rel[0]) & (shape_t <= display_rel[1]) & np.isfinite(shape_y)
        ax_cst.plot(
            shape_t[shape_mask],
            shape_y[shape_mask],
            color=style.get("response_shape_fit_color", "#008c72"),
            lw=_style_float("response_shape_fit_lw", 1.8),
            ls=style.get("response_shape_fit_ls", "--"),
            alpha=_style_float("response_shape_fit_alpha", 0.95),
            label="Mean response-shape fit",
        )

    sync_t = np.asarray(mean_step_response_diagnostics.get("sync_time_rel_step_s", []), dtype=float)
    sync_mask = (sync_t >= display_rel[0]) & (sync_t <= display_rel[1]) if sync_t.size else np.asarray([], dtype=bool)
    if show_individuals and sync_t.size:
        for trace in individual_traces_by_key.get(sync_key, []):
            trace = np.asarray(trace, dtype=float)
            if trace.shape == sync_t.shape:
                ax_sync.plot(sync_t[sync_mask], trace[sync_mask], color=sync_color, lw=individual_lw, alpha=individual_alpha, zorder=1)
    mean_sync_trace = np.asarray(mean_step_response_diagnostics.get(sync_key, []), dtype=float)
    if sync_t.size and mean_sync_trace.shape == sync_t.shape:
        ax_sync.plot(sync_t[sync_mask], mean_sync_trace[sync_mask], color=sync_color, lw=mean_lw, alpha=1.0, zorder=4, label=sync_label)
    ax_sync.axvline(0.0, color=style.get("step_line_color", "#555555"), lw=_style_float("step_line_lw", 1.0), ls=style.get("step_line_ls", "--"))
    show_sync_zero_line = bool(style.get("show_sync_zero_line", sync_trace_mode != "raw"))
    if show_sync_zero_line:
        ax_sync.axhline(0.0, color="0.55", lw=0.8, ls=":")

    def _metric(key):
        return _float_from_mapping(mean_step_response_diagnostics, key, np.nan)

    def _fmt_window(values, scale=1.0, suffix="s"):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != 2 or not np.all(np.isfinite(arr)):
            return "[NA]"
        return f"[{arr[0] * scale:.2g},{arr[1] * scale:.2g}]{suffix}"

    annotation = "\n".join([
        f"step={_metric('step_current_amplitude_nA')/1000.0:.2f} uA | n repeats={int(_metric('n_repeats_used')) if np.isfinite(_metric('n_repeats_used')) else 'NA'}",
        f"dEffect={_metric('delta_cst_effect'):.2f} sp/s/MN | gain={_metric('response_gain_cst_per_uA'):.3g} sp/s/MN/uA",
        f"linear slope={_metric('linear_initial_slope_cst_per_s'):.2f} sp/s/MN/s | end={_metric('linear_selected_window_end_s'):.3f} s | R2={_metric('linear_selected_window_r2'):.2f}",
        f"shape tau={_metric('response_shape_tau_ms'):.1f} ms | f50/f150={_metric('response_shape_fraction_50ms'):.2f}/{_metric('response_shape_fraction_150ms'):.2f}",
        f"shape t50/t90={_metric('response_shape_fitted_t50_ms'):.1f}/{_metric('response_shape_fitted_t90_ms'):.1f} ms",
        f"xcorr lag={_metric('xcorr_lag_s') * 1000.0:.1f} ms | corr={_metric('xcorr_peak_corr'):.2f}",
        f"stage={mean_step_response_diagnostics.get('diagnostics_stage', 'finalized_mean_trace')}",
    ])
    ax_cst.text(
        0.01,
        0.98,
        annotation,
        transform=ax_cst.transAxes,
        va="top",
        ha="left",
        fontsize=_style_float("annotation_fontsize", 8.0),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.65", alpha=0.92),
        zorder=100,
    )
    sync_pred_pre_win = np.asarray(mean_step_response_diagnostics.get("sync_predictor_pre_step_window_rel_s", [-0.25, 0.0]), dtype=float)
    sync_pred_local_win = np.asarray(mean_step_response_diagnostics.get("sync_predictor_local_step_window_rel_s", [-0.25, 0.25]), dtype=float)
    sync_annotation = "\n".join([
        f"baseline {_fmt_window(pre_win)}: {_metric('baseline_sync_mean'):.3f} +/- {_metric('baseline_sync_sd'):.3f}",
        f"post actual {_fmt_window(late_win)}: {_metric('post_step_sync_mean_actual_step'):.3f} +/- {_metric('post_step_sync_sd_actual_step'):.3f}",
        f"pre matched {_fmt_window(sync_pred_pre_win, 1000.0, 'ms')}: {_metric('pre_step_sync_mean_baseline_matched'):.3f} +/- {_metric('pre_step_sync_sd_baseline_matched'):.3f}",
        f"local matched {_fmt_window(sync_pred_local_win, 1000.0, 'ms')}: {_metric('local_step_sync_mean_baseline_matched'):.3f} +/- {_metric('local_step_sync_sd_baseline_matched'):.3f}",
        f"source={mean_step_response_diagnostics.get('sync_predictor_source', 'unknown')}",
    ])
    ax_sync.text(
        0.01,
        0.98,
        sync_annotation,
        transform=ax_sync.transAxes,
        va="top",
        ha="left",
        fontsize=_style_float("sync_annotation_fontsize", _style_float("annotation_fontsize", 7.0)),
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.65", alpha=0.88),
        zorder=100,
    )
    ax_cst.set_title(str(style.get("title", "Mean-trace step-response diagnostic")))
    ax_cst.set_ylabel("CST effect\n(spikes/s/MN)")
    ax_sync.set_ylabel(sync_ylabel)
    ax_sync.set_xlabel("Time relative to step onset (s)")
    ax_cst.set_xlim(display_rel)
    cst_ylim = _style_lim("cst_ylim")
    if cst_ylim is not None:
        ax_cst.set_ylim(cst_ylim)
    sync_ylim = _style_lim("sync_ylim")
    if sync_ylim is not None:
        ax_sync.set_ylim(sync_ylim)
    ax_cst.grid(alpha=_style_float("grid_alpha", 0.25))
    ax_sync.grid(alpha=_style_float("grid_alpha", 0.25))
    ax_cst.legend(loc=str(style.get("legend_loc", "upper right")), fontsize=_style_float("legend_fontsize", 8.0))
    ax_sync.legend(loc=str(style.get("sync_legend_loc", "upper right")), fontsize=_style_float("legend_fontsize", 8.0))
    fig.savefig(save_dir / figure_name, dpi=int(_style_float("dpi", 140)), bbox_inches="tight")
    pdf_name = str(Path(figure_name).with_suffix(".pdf"))
    fig.savefig(save_dir / pdf_name, bbox_inches="tight")
    plt.close(fig)
    return save_dir / figure_name


def finalize_step_response_parameter_set_mean_traces(h5_paths, *, regenerate_figure=True, plot_style=None):
    h5_paths = list(h5_paths)
    records = []
    skipped = []
    for path in h5_paths:
        try:
            record = _load_step_response_record(path)
            records.append(record)
        except Exception as exc:
            skipped.append({"path": str(path), "status": f"load_error_{type(exc).__name__}", "message": str(exc)})
    if not records:
        return pd.DataFrame([{"status": "no_valid_records", "n_paths": len(h5_paths)}])

    parameter_set_id = records[0]["parameter_set_id"]
    parameter_set_dir = _parameter_set_dir_from_h5_path(records[0]["h5_path"])
    step_groups = {}
    for record in records:
        amp = float(record["step_current_amplitude_nA"])
        step_groups.setdefault(amp, []).append(record)
    nonzero_steps = sorted([amp for amp in step_groups if np.isfinite(amp) and not np.isclose(amp, 0.0)])
    summaries = []
    for step_amp in nonzero_steps:
        step_records_all = step_groups[step_amp]
        valid_records = []
        step_skipped = []
        for record in step_records_all:
            diag = record["step_diag"]
            matched = _bool_from_mapping(diag, "matched_step_effect_available", False)
            if not matched:
                step_skipped.append({"path": str(record["h5_path"]), "status": "matched_step_effect_unavailable"})
                continue
            if "cst_step_effect_sign_corrected" not in diag:
                step_skipped.append({"path": str(record["h5_path"]), "status": "missing_cst_step_effect_sign_corrected"})
                continue
            valid_records.append(record)
        step_folder = _format_step_folder_name_for_output(step_amp)
        output_dir = parameter_set_dir / "averaged_repeats" / step_folder
        output_h5 = output_dir / "averaged_step_response_diagnostics.h5"
        if not valid_records:
            summaries.append({
                "status": "no_valid_finalized_repeats",
                "parameter_set_id": parameter_set_id,
                "step_current_amplitude_nA": step_amp,
                "n_repeats_available": len(step_records_all),
                "n_repeats_used": 0,
                "output_h5_path": str(output_h5),
            })
            continue
        ref_record = valid_records[0]
        mean_step_diag, individual_traces = _build_mean_step_diag_for_records(valid_records, ref_record)
        zero_mean_diag = {
            "t_step_rel_s": np.asarray(mean_step_diag["t_step_rel_s"], dtype=float),
            "time_rel_step_s": np.asarray(mean_step_diag["time_rel_step_s"], dtype=float),
            "cst_smooth_step_response": np.asarray(mean_step_diag["matched_no_step_cst_smooth_step_response"], dtype=float),
        }
        if "sync_time_rel_step_s" in mean_step_diag and "matched_no_step_sync_trace" in mean_step_diag:
            zero_mean_diag["sync_time_rel_step_s"] = np.asarray(mean_step_diag["sync_time_rel_step_s"], dtype=float)
            zero_mean_diag["sync_trace"] = np.asarray(mean_step_diag["matched_no_step_sync_trace"], dtype=float)
        metric_updates = _compute_matched_step_effect_metrics(mean_step_diag, zero_mean_diag)
        mean_step_diag.update(metric_updates)
        mean_step_diag.update({
            "status": "ok",
            "diagnostics_stage": "finalized_mean_trace",
            "metric_source": "mean_trace_across_repeats",
            "parameter_set_id": np.asarray([parameter_set_id], dtype=float),
            "step_current_amplitude_nA": np.asarray([step_amp], dtype=float),
            "n_repeats_available": np.asarray([len(step_records_all)], dtype=int),
            "n_repeats_used": np.asarray([len(valid_records)], dtype=int),
            "n_repeats_skipped": np.asarray([len(step_skipped)], dtype=int),
        })
        used_repeat_indices = [_float_from_mapping(record["params"], "repeat_index", np.nan) for record in valid_records]
        used_repeat_seeds = [_float_from_mapping(record["params"], "requested_random_seed", np.nan) for record in valid_records]
        repeat_metadata = {
            "parameter_set_id": np.asarray([parameter_set_id], dtype=float),
            "step_current_amplitude_nA": np.asarray([step_amp], dtype=float),
            "n_repeats_available": np.asarray([len(step_records_all)], dtype=int),
            "n_repeats_used": np.asarray([len(valid_records)], dtype=int),
            "used_repeat_indices": np.asarray(used_repeat_indices, dtype=float),
            "used_repeat_seeds": np.asarray(used_repeat_seeds, dtype=float),
            "used_h5_paths": [str(record["h5_path"]) for record in valid_records],
            "skipped_repeat_records": json.dumps(step_skipped + skipped),
        }
        _write_mean_trace_hdf5(output_h5, mean_step_diag, repeat_metadata)
        figure_path = None
        if regenerate_figure:
            try:
                figure_path = plot_step_response_mean_trace_diagnostic(
                    output_dir,
                    mean_step_diag,
                    individual_traces,
                    plot_style=plot_style,
                )
            except Exception as exc:
                warnings.warn(f"Could not plot mean-trace diagnostic for parameter_set_id={parameter_set_id}, step={step_amp}: {exc}")
        summaries.append({
            "status": "ok",
            "parameter_set_id": parameter_set_id,
            "step_current_amplitude_nA": step_amp,
            "n_repeats_available": len(step_records_all),
            "n_repeats_used": len(valid_records),
            "n_repeats_skipped": len(step_skipped),
            "output_h5_path": str(output_h5),
            "figure_path": "" if figure_path is None else str(figure_path),
            "delta_cst_effect": _float_from_mapping(mean_step_diag, "delta_cst_effect", np.nan),
            "response_gain_cst_per_uA": _float_from_mapping(mean_step_diag, "response_gain_cst_per_uA", np.nan),
            "linear_initial_slope_cst_per_s": _float_from_mapping(mean_step_diag, "linear_initial_slope_cst_per_s", np.nan),
            "response_shape_tau_ms": _float_from_mapping(mean_step_diag, "response_shape_tau_ms", np.nan),
            "xcorr_lag_s": _float_from_mapping(mean_step_diag, "xcorr_lag_s", np.nan),
        })
    if not summaries:
        summaries.append({
            "status": "no_nonzero_steps",
            "parameter_set_id": parameter_set_id,
            "n_paths": len(records),
        })
    return pd.DataFrame(summaries)


def finalize_step_response_batch_mean_traces(run_table_or_paths, *, n_jobs=1, regenerate_figures=True, plot_style=None):
    items = _group_finalization_paths_from_table(run_table_or_paths, ["parameter_set_id"])
    if items is None:
        paths = _coerce_finalization_paths(run_table_or_paths)
        if not paths:
            return pd.DataFrame()
        records = []
        for path in paths:
            try:
                records.append(_load_step_response_record(path))
            except Exception as exc:
                warnings.warn(f"Skipping {path}: {exc}")
        if not records:
            return pd.DataFrame()
        groups = {}
        for record in records:
            groups.setdefault(record["parameter_set_id"], []).append(record["h5_path"])
        items = list(groups.values())
    if not items:
        return pd.DataFrame()
    n_jobs = int(n_jobs)
    if n_jobs <= 1:
        frames = [
            finalize_step_response_parameter_set_mean_traces(paths, regenerate_figure=regenerate_figures, plot_style=plot_style)
            for paths in items
        ]
    else:
        from joblib import Parallel, delayed
        frames = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(finalize_step_response_parameter_set_mean_traces)(
                paths,
                regenerate_figure=regenerate_figures,
                plot_style=plot_style,
            )
            for paths in items
        )
    frames = [frame for frame in frames if frame is not None and len(frame)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_input_summary_diagnostics(
        save_dir,
        figure_name,
        t_s,
        zoom_mask,
        input_components,
        cst_diagnostics,
        sync_diagnostics,
        params,
        burst_center_s=None,
        burst_start_s=None,
        burst_end_s=None):
    colors = get_input_component_colors(params)
    fig, axes = plt.subplots(4, 3, figsize=(19, 11), dpi=120, sharex=True)
    cue_time_s = float(params.task_event_times_s.get("go_nogo_cue_s", 0.0))
    full_t_rel = np.asarray(t_s, dtype=float) - cue_time_s
    display_window_rel_cue_s = tuple(map(float, params.display_window_rel_cue_s))
    zoom_mask = np.asarray(zoom_mask, dtype=bool) & (full_t_rel >= display_window_rel_cue_s[0]) & (full_t_rel <= display_window_rel_cue_s[1])
    zoom_t = full_t_rel[zoom_mask]
    task_event_times_s = params.task_event_times_s
    cst_modulation_mode = str(getattr(params, "cst_modulation_mode", "zscore")).lower()
    if cst_modulation_mode not in ("zscore", "percent"):
        raise ValueError(f"Unsupported cst_modulation_mode {cst_modulation_mode!r}")

    def _auto_limits(data, pad_frac=0.08, zero_floor=False):
        data = np.asarray(data, dtype=float)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return (0.0, 1.0) if zero_floor else (-1.0, 1.0)
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        if zero_floor:
            y_min = min(0.0, y_min)
        span = y_max - y_min
        pad = max(0.1, span * pad_frac) if span > 1e-12 else max(0.1, abs(y_max) * pad_frac, 0.1)
        return y_min - (0.0 if zero_floor else pad), y_max + pad

    def _plot_input_panel(ax, signal, envelope_nA, burst_kernel_nA, trend_kernel_nA, color, title, show_envelope=True):
        signal = np.asarray(signal, dtype=float)[zoom_mask]
        envelope_nA = np.asarray(envelope_nA, dtype=float)[zoom_mask]
        burst_kernel_nA = np.asarray(burst_kernel_nA, dtype=float)[zoom_mask]
        trend_kernel_nA = np.asarray(trend_kernel_nA, dtype=float)[zoom_mask]
        signal_uA = signal / 1000.0
        env_pos_uA = envelope_nA / 1000.0
        env_neg_uA = -envelope_nA / 1000.0
        modulation_uA = (burst_kernel_nA + trend_kernel_nA) / 1000.0
        ax.plot(zoom_t, signal_uA, color=color, lw=1.2, zorder=3)
        if show_envelope:
            ax.plot(zoom_t, env_pos_uA, color=colors["envelope"], lw=1.0, alpha=0.9, zorder=2)
            ax.plot(zoom_t, env_neg_uA, color=colors["envelope"], lw=1.0, alpha=0.9, zorder=2)
        if np.any(np.abs(burst_kernel_nA + trend_kernel_nA) > 1e-12):
            ax.plot(zoom_t, modulation_uA, color=colors["burst"], lw=1.1, ls="--", alpha=0.95, zorder=5)
        ylim_data = [signal_uA]
        if show_envelope:
            ylim_data.extend([env_pos_uA, env_neg_uA])
        if np.any(np.abs(burst_kernel_nA + trend_kernel_nA) > 1e-12):
            ylim_data.append(modulation_uA)
        ymin, ymax = _auto_limits(np.concatenate(ylim_data) if ylim_data else np.zeros(1), zero_floor=False)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("Input (µA)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        _draw_diagnostic_markers(ax, task_event_times_s=task_event_times_s, burst_center_s=burst_center_s, burst_start_s=burst_start_s, burst_end_s=burst_end_s, colors=colors, time_offset_s=cue_time_s)
        _draw_summary_window_overlays(ax, display_window_rel_cue_s, params)

    def _plot_cst_panel(ax, traces, title):
        all_values = []
        for trace in traces:
            values = np.asarray(trace["signal"], dtype=float)[zoom_mask]
            all_values.append(values)
            ax.plot(zoom_t, values, color=trace["color"], lw=trace.get("lw", 1.2), ls=trace.get("ls", "-"), alpha=trace.get("alpha", 1.0), zorder=trace.get("zorder", 3))
        ymin, ymax = _auto_limits(np.concatenate(all_values) if all_values else np.zeros(1), zero_floor=False)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("spikes/s/MU")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        _draw_diagnostic_markers(ax, task_event_times_s=task_event_times_s, burst_center_s=burst_center_s, burst_start_s=burst_start_s, burst_end_s=burst_end_s, colors=colors, time_offset_s=cue_time_s)
        _draw_summary_window_overlays(ax, display_window_rel_cue_s, params)

    def _plot_modulation_panel(ax, signal, color, title):
        values = np.asarray(signal, dtype=float)[zoom_mask]
        ymin, ymax = _auto_limits(values, zero_floor=False)
        ax.plot(zoom_t, values, color=color, lw=1.3, zorder=3)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(
            "Modulation relative to baseline (z)"
            if cst_modulation_mode == "zscore"
            else "Modulation relative to baseline (%)"
        )
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        _draw_diagnostic_markers(ax, task_event_times_s=task_event_times_s, burst_center_s=burst_center_s, burst_start_s=burst_start_s, burst_end_s=burst_end_s, colors=colors, time_offset_s=cue_time_s)
        _draw_summary_window_overlays(ax, display_window_rel_cue_s, params)

    _plot_input_panel(
        axes[0, 0],
        signal=input_components["lf_final_with_burst"],
        envelope_nA=input_components["lf_effective_envelope_nA"],
        burst_kernel_nA=input_components["lf_burst_kernel_nA"],
        trend_kernel_nA=input_components.get("lf_trend_kernel_nA", np.zeros_like(input_components["lf_final_with_burst"])),
        color=colors["lf"],
        title="LF input",
        show_envelope=False,
    )
    _plot_input_panel(
        axes[1, 0],
        signal=input_components["alpha_final_with_burst"],
        envelope_nA=input_components["alpha_effective_envelope_nA"],
        burst_kernel_nA=input_components["alpha_burst_kernel_nA"],
        trend_kernel_nA=input_components.get("alpha_trend_kernel_nA", np.zeros_like(input_components["alpha_final_with_burst"])),
        color=colors["alpha"],
        title="Alpha input",
    )
    _plot_input_panel(
        axes[2, 0],
        signal=input_components["beta_final_with_burst"],
        envelope_nA=input_components["beta_effective_envelope_nA"],
        burst_kernel_nA=input_components["beta_burst_kernel_nA"],
        trend_kernel_nA=input_components.get("beta_trend_kernel_nA", np.zeros_like(input_components["beta_final_with_burst"])),
        color=colors["beta"],
        title="Beta input",
    )
    common_signal = np.asarray(input_components["common_input_with_burst"], dtype=float)[zoom_mask]
    common_signal_uA = common_signal / 1000.0
    axes[3, 0].plot(zoom_t, common_signal_uA, color=colors["common"], lw=1.3, zorder=3)
    ymin, ymax = _auto_limits(common_signal_uA, zero_floor=False)
    axes[3, 0].set_ylim(ymin, ymax)
    axes[3, 0].set_ylabel("Input (µA)")
    axes[3, 0].set_title("Delivered common input", fontsize=10)
    axes[3, 0].grid(alpha=0.25)
    _draw_diagnostic_markers(axes[3, 0], task_event_times_s=task_event_times_s, burst_center_s=burst_center_s, burst_start_s=burst_start_s, burst_end_s=burst_end_s, colors=colors, time_offset_s=cue_time_s)
    _draw_summary_window_overlays(axes[3, 0], display_window_rel_cue_s, params)

    cst_zero = np.zeros_like(t_s, dtype=float)
    cst_lf = np.asarray(cst_diagnostics.get("cst_lf", cst_zero), dtype=float)
    cst_alpha = np.asarray(cst_diagnostics.get("cst_alpha", cst_zero), dtype=float)
    cst_beta = np.asarray(cst_diagnostics.get("cst_beta", cst_zero), dtype=float)
    cst_alpha_mod = np.asarray(
        cst_diagnostics.get(
            "cst_alpha_envelope_z" if cst_modulation_mode == "zscore" else "cst_alpha_envelope_pct",
            cst_zero,
        ),
        dtype=float,
    )
    cst_beta_mod = np.asarray(
        cst_diagnostics.get(
            "cst_beta_envelope_z" if cst_modulation_mode == "zscore" else "cst_beta_envelope_pct",
            cst_zero,
        ),
        dtype=float,
    )

    _plot_cst_panel(
        axes[0, 1],
        traces=[
            {"signal": cst_lf, "color": colors["lf"], "lw": 1.3},
        ],
        title=f"CST low-pass ({params.lf_band_hz[0]:.2f}-{params.lf_band_hz[1]:.2f} Hz)",
    )
    _plot_cst_panel(
        axes[1, 1],
        traces=[
            {"signal": cst_alpha, "color": colors["alpha"], "lw": 1.3},
        ],
        title=f"CST band-pass alpha ({params.alpha_band_hz[0]:.1f}-{params.alpha_band_hz[1]:.1f} Hz)",
    )
    _plot_cst_panel(
        axes[2, 1],
        traces=[
            {"signal": cst_beta, "color": colors["beta"], "lw": 1.3},
        ],
        title=f"CST band-pass beta ({params.beta_band_hz[0]:.1f}-{params.beta_band_hz[1]:.1f} Hz)",
    )
    axes[3, 1].axis("off")

    sync_display_mode = str(getattr(params, "sync_trace_display_mode", "absolute")).lower()
    if sync_display_mode not in ("absolute", "zscore"):
        raise ValueError(f"Unsupported sync_trace_display_mode {sync_display_mode!r}")
    axes[0, 2].axis("off")
    axes[0, 2].legend(
        handles=[
            Line2D([0], [0], color=colors["cue"], lw=1.0, ls="--", label="Get-ready cue"),
            Line2D([0], [0], color=colors["cue"], lw=1.1, ls="-", label="Go cue"),
            Patch(facecolor=colors["burst"], edgecolor="none", alpha=0.10, label="Burst epoch"),
            Line2D([0], [0], color=colors["burst"], lw=1.1, ls="--", label="Burst center"),
        ],
        loc="center",
        frameon=True,
        title="Shared markers",
    )
    _plot_modulation_panel(
        axes[1, 2],
        signal=cst_alpha_mod,
        color=colors["alpha"],
        title=f"Alpha modulation ({'z-score' if cst_modulation_mode == 'zscore' else '% baseline'} of Hilbert transform)",
    )
    _plot_modulation_panel(
        axes[2, 2],
        signal=cst_beta_mod,
        color=colors["beta"],
        title=f"Beta modulation ({'z-score' if cst_modulation_mode == 'zscore' else '% baseline'} of Hilbert transform)",
    )
    if sync_diagnostics is None:
        axes[3, 2].axis("off")
    else:
        sync_time_rel = np.asarray(sync_diagnostics.get("t_sync", np.zeros(0)), dtype=float)
        go_cue_s = float(task_event_times_s.get("go_nogo_cue_s", 0.0))
        sync_time_abs = sync_time_rel + go_cue_s
        sync_trace = np.asarray(
            sync_diagnostics.get(
                "sync_trace" if sync_display_mode == "absolute" else "sync_trace_zscore",
                np.zeros_like(sync_time_rel),
            ),
            dtype=float,
        )
        sync_time_rel_plot = sync_time_abs - cue_time_s
        sync_mask = (sync_time_rel_plot >= display_window_rel_cue_s[0]) & (sync_time_rel_plot <= display_window_rel_cue_s[1])
        if np.any(sync_mask):
            sync_vals = sync_trace[sync_mask]
            ymin, ymax = _auto_limits(sync_vals, zero_floor=False)
            axes[3, 2].plot(sync_time_rel_plot[sync_mask], sync_vals, color=colors.get("sync", colors["cst"]), lw=1.3, zorder=3)
            axes[3, 2].set_ylim(ymin, ymax)
            axes[3, 2].set_ylabel(
                "Excess coincidence index" if sync_display_mode == "absolute" else "Excess coincidence index (z)"
            )
            axes[3, 2].set_title(
                f"Sliding synchrony ({'absolute' if sync_display_mode == 'absolute' else 'baseline-zscored'})",
                fontsize=10,
            )
            axes[3, 2].grid(alpha=0.25)
            _draw_diagnostic_markers(axes[3, 2], task_event_times_s=task_event_times_s, burst_center_s=burst_center_s, burst_start_s=burst_start_s, burst_end_s=burst_end_s, colors=colors, time_offset_s=cue_time_s)
            _draw_summary_window_overlays(axes[3, 2], display_window_rel_cue_s, params)
        else:
            axes[3, 2].axis("off")

    axes[3, 0].set_xlabel("Time relative to go/no-go cue (s)")
    axes[2, 1].set_xlabel("Time relative to go/no-go cue (s)")
    axes[3, 2].set_xlabel("Time relative to go/no-go cue (s)")
    for row_axes in axes:
        for ax in row_axes:
            if ax.axison:
                ax.set_xlim(*display_window_rel_cue_s)
    fig.suptitle("Input and CST summary diagnostics", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, figure_name), bbox_inches="tight")
    plt.close(fig)
def generate_component_burst_kernels(
        fsamp,
        duration_with_ignored_window,
        burst_center_ms,
        burst_sigma_ms,
        lf_burst_peak_nA=0.0,
        alpha_burst_peak_nA=0.0,
        beta_burst_peak_nA=0.0,
        burst_start_marker_n_sigma=2.0):
    """
    Build Gaussian burst kernels in nA that are additively applied to the physical
    LF / alpha / beta component envelopes.
    """
    n_samples = int(np.round(duration_with_ignored_window * fsamp))
    t_s = np.arange(n_samples, dtype=float) / float(fsamp)
    burst_center_s = float(burst_center_ms) / 1000.0
    burst_sigma_s = float(burst_sigma_ms) / 1000.0
    gaussian_kernel = np.exp(-0.5 * ((t_s - burst_center_s) / burst_sigma_s) ** 2)
    burst_start_s = burst_center_s - float(burst_start_marker_n_sigma) * burst_sigma_s
    return {
        "gaussian_kernel": gaussian_kernel,
        "lf_burst_kernel_nA": float(lf_burst_peak_nA) * gaussian_kernel,
        "alpha_burst_kernel_nA": float(alpha_burst_peak_nA) * gaussian_kernel,
        "beta_burst_kernel_nA": float(beta_burst_peak_nA) * gaussian_kernel,
        "center_s": float(burst_center_s),
        "start_s": float(burst_start_s),
    }
def generate_component_trend_kernels(
        fsamp,
        duration_with_ignored_window,
        go_nogo_cue_s,
        lf_trend_start_rel_go_cue_s=-1.0,
        alpha_trend_start_rel_go_cue_s=-1.0,
        beta_trend_start_rel_go_cue_s=-1.0,
        lf_trend_slope_nA_per_s=0.0,
        alpha_trend_slope_nA_per_s=0.0,
        beta_trend_slope_nA_per_s=0.0):
    """
    Build one-sided linear trend kernels in nA that start at the requested time
    relative to the go/no-go cue and remain exactly zero before that onset.
    """
    n_samples = int(np.round(duration_with_ignored_window * fsamp))
    t_s = np.arange(n_samples, dtype=float) / float(fsamp)

    def _build_kernel(start_rel_go_cue_s, slope_nA_per_s):
        start_s = float(go_nogo_cue_s) + float(start_rel_go_cue_s)
        return float(slope_nA_per_s) * np.maximum(t_s - start_s, 0.0), float(start_s)

    lf_trend_kernel_nA, lf_start_s = _build_kernel(lf_trend_start_rel_go_cue_s, lf_trend_slope_nA_per_s)
    alpha_trend_kernel_nA, alpha_start_s = _build_kernel(alpha_trend_start_rel_go_cue_s, alpha_trend_slope_nA_per_s)
    beta_trend_kernel_nA, beta_start_s = _build_kernel(beta_trend_start_rel_go_cue_s, beta_trend_slope_nA_per_s)
    return {
        "lf_trend_kernel_nA": lf_trend_kernel_nA,
        "alpha_trend_kernel_nA": alpha_trend_kernel_nA,
        "beta_trend_kernel_nA": beta_trend_kernel_nA,
        "lf_start_s": lf_start_s,
        "alpha_start_s": alpha_start_s,
        "beta_start_s": beta_start_s,
    }
def sample_block(n_pre, n_post, p_block, dist, params, ranks=None, binary=False, diam_um=None):
    """
    Sample an n_pre x n_post submatrix with mean targeting p_block.
    New 'size_powerlaw' mode: per-row mean scales with (diam / d_ref)^exponent,
    then renormalized so the *average* over rows is p_block.
    """
    if binary:
        A = np.zeros((n_pre, n_post), dtype=int)
    else:
        A = np.zeros((n_pre, n_post), dtype=float)

    # -----------------------------
    # NEW: size_powerlaw (uses diam_um)
    # -----------------------------
    if dist == "size_powerlaw":
        if diam_um is None:
            raise ValueError("size_powerlaw requires diam_um (length n_pre)")

        # Params (with sensible defaults)
        exp = float(params.get("exponent", 1.0))
        std = float(params.get("std", 0.0))
        std_is_prct = bool(params.get("std_is_prct", True))
        dref_mode = params.get("d_ref_mode", "min_block")  # 'min_block' | 'mean_block' | 'global_value'
        if dref_mode == "min_block":
            d_ref = float(np.min(diam_um))
        elif dref_mode == "mean_block":
            d_ref = float(np.mean(diam_um))
        elif dref_mode == "global_value":
            d_ref = float(params["d_ref_value"])  # e.g., 50.0  (µm)
        else:
            d_ref = float(np.min(diam_um))

        # Per-row multiplier and renormalization so <mu_row> = p_block
        diam_um = np.asarray(diam_um, float)
        mult = (diam_um / d_ref) ** exp                 # shape: (n_pre,)
        alpha = mult / np.mean(mult)                    # avg(alpha) = 1
        mu_row = p_block * alpha                        # desired per-row mean

        # For binary case, keep probabilities within (0,1) and re-normalize once
        if binary:
            mu_row = np.clip(mu_row, 1e-6, 1 - 1e-6)
            mean_mu = float(np.mean(mu_row))
            if mean_mu > 0:
                mu_row = mu_row * (p_block / mean_mu)
                mu_row = np.clip(mu_row, 1e-6, 1 - 1e-6)

        # Sample
        for u in range(n_pre):
            mu = float(mu_row[u])
            sigma = (mu * std) if std_is_prct else std
            if binary:
                # Bernoulli with per-edge p ~ TruncNormal(mu,sigma) in [0,1]
                for v in range(n_post):
                    p_samp = np.clip(np.random.randn() * sigma + mu, 0.0, 1.0)
                    A[u, v] = (np.random.rand() < p_samp).astype(int)
            else:
                # Real-valued: truncated normal ≥ 0 around mu
                for v in range(n_post):
                    raw = np.random.randn() * sigma + mu
                    A[u, v] = max(raw, 0.0)
        return A

    # -----------------------------
    # existing modes kept as-is
    # -----------------------------
    if dist == "binarize":
        if binary:
            A[:] = (np.random.rand(n_pre, n_post) < p_block).astype(int)
        else:
            A[:] = np.random.rand(n_pre, n_post) * p_block

    elif dist == "gaussian":
        for u in range(n_pre):
            for v in range(n_post):
                μ = p_block
                σ = p_block * params.get("std", 0.0) if params['std_is_prct'] else params.get("std", 0.0)
                if binary:
                    p_samp = np.clip(np.random.randn() * σ + μ, 0, 1)
                    A[u, v] = (np.random.rand() < p_samp).astype(int)
                else:
                    raw = np.random.randn() * σ + μ
                    A[u, v] = max(raw, 0.0)

    elif dist == "size_gaussian" and (ranks is not None):
        R = params["ratio_large_small"]
        for u in range(n_pre):
            r = ranks[u]
            μ0 = p_block
            σ = p_block * params.get("std", 0.0) if params['std_is_prct'] else params.get("std", 0.0)
            μ_s = 2 * μ0 / (1 + R)
            μ_l = R * μ_s
            μ_ij = μ_s + r * (μ_l - μ_s)
            for v in range(n_post):
                if binary:
                    p_samp = np.clip(np.random.randn() * σ + μ_ij, 0, 1)
                    A[u, v] = (np.random.rand() < p_samp).astype(int)
                else:
                    raw = np.random.randn() * σ + μ_ij
                    A[u, v] = max(raw, 0.0)
    else:
        # fallback to Gaussian logic
        for u in range(n_pre):
            for v in range(n_post):
                μ = p_block
                σ = p_block * params.get("std", 0.0) if params['std_is_prct'] else params.get("std", 0.0)
                if binary:
                    p_samp = np.clip(np.random.randn() * σ + μ, 0, 1)
                    A[u, v] = (np.random.rand() < p_samp).astype(int)
                else:
                    raw = np.random.randn() * σ + μ
                    A[u, v] = max(raw, 0.0)

    return A

def plot_connectivity_matrix(
            mat, title,
            pool_labels_pre, pool_labels_post,
            n_pre_pool, n_post_pool,
            cmap='viridis',
            add_colorbar=False,
            savepath=None
        ):
            n_pre, n_post = mat.shape
            n_pre_blocks  = len(pool_labels_pre)
            n_post_blocks = len(pool_labels_post)

            # compute block means & stds (unchanged) …
            means = np.zeros((n_pre_blocks, n_post_blocks))
            stds  = np.zeros((n_pre_blocks, n_post_blocks))
            for i in range(n_pre_blocks):
                for j in range(n_post_blocks):
                    r = slice(i*n_pre_pool, (i+1)*n_pre_pool)
                    c = slice(j*n_post_pool, (j+1)*n_post_pool)
                    blk = mat[r,c]
                    means[i,j] = blk.mean()
                    stds[i,j]  = blk.std()

            fig, ax = plt.subplots(figsize=(10,10))
            im = ax.imshow(mat, cmap=cmap, aspect='equal')

            # grid lines at pool boundaries
            for y in np.arange(n_pre_pool, n_pre, n_pre_pool):
                ax.axhline(y-0.5, color='white', lw=1)
            for x in np.arange(n_post_pool, n_post, n_post_pool):
                ax.axvline(x-0.5, color='white', lw=1)

            # ### 1) show every cell index tick ###
            ax.set_xticks(np.arange(n_post))
            ax.set_xticklabels([str(i) for i in range(n_post)], rotation=90, fontsize=6)
            ax.set_yticks(np.arange(n_pre))
            ax.set_yticklabels([str(i) for i in range(n_pre)], fontsize=6)
            ax.invert_yaxis()

            # ### 2) overlay pool labels at group centers ###
            # x‐axis (pool‐pair labels)
            post_centers = [j*n_post_pool + n_post_pool/2 for j in range(n_post_blocks)]
            for j, lbl in enumerate(pool_labels_post):
                ax.text(
                    post_centers[j], -0.7,   # x at center, y just above top row
                    lbl,
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    rotation=90,
                    clip_on=False
                )

            # y‐axis (pre‐pool labels)
            pre_centers = [i*n_pre_pool + n_pre_pool/2 for i in range(n_pre_blocks)]
            for i, lbl in enumerate(pool_labels_pre):
                ax.text(
                    -0.7, pre_centers[i],   # x just before first column, y at center
                    lbl,
                    ha='right', va='center',
                    fontsize=10, fontweight='bold',
                    clip_on=False
                )

            # title + subtitle
            μ, σ = mat.mean(), mat.std()
            subtitle = "\n".join(f"{pool_labels_pre[i]}→{pool_labels_post[j]}: {means[i,j]:.2f}±{stds[i,j]:.2f}"
                                for i in range(n_pre_blocks)
                                for j in range(n_post_blocks))
            ax.set_title(f"{title}\nOverall μ={μ:.2f}, σ={σ:.2f}\n{subtitle}", pad=50)

            ax.set_xlabel(" ".join(pool_labels_post[0].split()[:1]).capitalize() + " cell index (receiving)")
            ax.set_ylabel(" ".join(pool_labels_pre[0].split()[:1]).capitalize() + " cell index (delivering)")

            if add_colorbar:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("# of disynaptic connections", rotation=90, labelpad=20)

            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath)
            plt.show()
### Force model helper functions
def _affine_powerlaw_from_diameter(d_um, dmin_um, dmax_um, y_at_dmin, y_at_dmax, exponent=1.0):
    """
    y(d) = c0 + c1 * d**exponent, with c0,c1 set so y(dmin)=y_at_dmin, y(dmax)=y_at_dmax
    Hits endpoints exactly while keeping power-law curvature.
    """
    dmin_p = (dmin_um ** exponent)
    dmax_p = (dmax_um ** exponent)
    c1 = 0.0 if abs(dmax_p - dmin_p) < 1e-12 else (y_at_dmax - y_at_dmin) / (dmax_p - dmin_p)
    c0 = y_at_dmin - c1 * dmin_p
    return c0 + c1 * (d_um ** exponent)
def assign_motor_unit_force_params(params, motoneurons_soma_diameters_um):
    """Return dict with arrays: tetanic_max_force, twitch_peak_force_relative, twitch_time (s)."""
    dmin = float(params.min_soma_diameter)
    dmax = float(params.max_soma_diameter)
    expn = float(params.force_diameter_exponent)
    d_um = np.asarray(motoneurons_soma_diameters_um, dtype=float)

    tetanic = _affine_powerlaw_from_diameter(
        d_um, dmin, dmax,
        params.tetanic_force_at_min_diam,
        params.tetanic_force_at_max_diam, exponent=expn)

    twitch_peak_rel = _affine_powerlaw_from_diameter(
        d_um, dmin, dmax,
        params.twitch_peak_rel_at_min_diam,
        params.twitch_peak_rel_at_max_diam, exponent=expn)

    twitch_time = _affine_powerlaw_from_diameter(
        d_um, dmin, dmax,
        params.twitch_time_at_min_diam,
        params.twitch_time_at_max_diam, exponent=expn)

    twitch_peak_rel = np.clip(twitch_peak_rel, 0.0, 1.0)
    twitch_time = np.maximum(twitch_time, 1e-4)  # ≥0.1 ms

    return dict(
        tetanic_max_force=tetanic.astype(float),
        twitch_peak_force_relative=twitch_peak_rel.astype(float),
        twitch_time=twitch_time.astype(float),
    )
def smoothstep01(x):
    """C¹ smooth clamp: 0 for x≤0; 1 for x≥1; 3x²−2x³ in-between."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask1 = x >= 1.0
    maskm = (~mask1) & (x > 0.0)
    xm = x[maskm]
    out[mask1] = 1.0
    out[maskm] = 3.0 * xm * xm - 2.0 * xm * xm * xm
    return out
def _twitch_kernel(fsamp, tau_s, duration_factor=6.0):
    """
    Fuglevand twitch (unit peak at t=τ): h(t) = (t/τ) * exp(1 − t/τ), t≥0.
    Discretized with dt=1/fsamp and truncated at duration_factor*τ.
    """
    dt = 1.0 / float(fsamp)
    L = int(np.ceil(duration_factor * float(tau_s) / dt))
    t = np.arange(L, dtype=float) * dt
    with np.errstate(divide='ignore', invalid='ignore'):
        h = (t / tau_s) * np.exp(1.0 - t / tau_s)
    h[0] = 0.0
    return h
def compute_mu_and_pool_force_traces(
    spike_trains_MN,                 # list of 1D arrays (seconds) per MU (post-delay & post-trim)
    mu_force_params,                 # dict from assign_motor_unit_force_params
    fsamp,                           # Hz (params.fsamp)
    duration_s,                      # seconds (params.duration_with_ignored_window)
    kernel_duration_factor=6.0,
    save_per_mu=False,
    idx_of_MN_by_pool=None,          # <-- NEW: dict like {"pool_0": idx_array, ...}
    nb_pools=None                    # <-- NEW
):
    """
    Computes (a) total pool force across all MUs and (b) per-pool forces.
    Returns a dict with:
      - pool_max_tetanic_force        (total, scalar)
      - pool_force                    (total, 1D)
      - pool_force_percent            (total, 1D)
      - per_pool: {
            'names'        : list[str] length nb_pools
            'max_tetanic'  : (nb_pools,)
            'force'        : (nb_pools, T)
            'force_percent': (nb_pools, T)
        }
      - mu_force (optional): (n_mu, T)
    """
    n_mu = len(spike_trains_MN)
    T = int(np.round(duration_s * fsamp))

    tetanic = np.asarray(mu_force_params["tetanic_max_force"], dtype=float)
    peakrel = np.asarray(mu_force_params["twitch_peak_force_relative"], dtype=float)
    taus    = np.asarray(mu_force_params["twitch_time"], dtype=float)

    mu_force = np.zeros((n_mu, T), dtype=float) if save_per_mu else None
    total_force = np.zeros(T, dtype=float)

    # ---- Per-pool bookkeeping ----
    if (idx_of_MN_by_pool is not None) and (nb_pools is not None):
        pool_names = [f"pool_{i}" for i in range(nb_pools)]
        per_pool_force = np.zeros((nb_pools, T), dtype=float)
        per_pool_max_tetanic = np.zeros(nb_pools, dtype=float)

        # Precompute a quick MU->pool map for fast accumulation
        mu_to_pool = np.empty(n_mu, dtype=int)
        for p_i, pname in enumerate(pool_names):
            inds = np.asarray(idx_of_MN_by_pool[pname], dtype=int)
            per_pool_max_tetanic[p_i] = float(np.sum(tetanic[inds]))
            mu_to_pool[inds] = p_i
    else:
        pool_names = []
        per_pool_force = None
        per_pool_max_tetanic = None
        mu_to_pool = None

    # ---- Stamp twitches & accumulate ----
    for i in range(n_mu):
        spikes_s = np.asarray(spike_trains_MN[i], dtype=float)
        spikes_idx = np.clip(np.floor(spikes_s * fsamp).astype(int), 0, T-1)

        h = _twitch_kernel(fsamp, taus[i], duration_factor=kernel_duration_factor) * peakrel[i]
        L = len(h)

        # relative force trace from twitches
        f_rel = np.zeros(T + L, dtype=float)
        for s in spikes_idx:
            f_rel[s:s+L] += h
        f_rel = f_rel[:T]

        # clamp & scale to absolute
        y_rel = smoothstep01(f_rel)         # ∈ [0,1]
        f_abs = tetanic[i] * y_rel          # absolute force for MU i

        if save_per_mu:
            mu_force[i, :] = f_abs

        total_force += f_abs

        # per-pool accumulation
        if mu_to_pool is not None:
            p = mu_to_pool[i]
            per_pool_force[p, :] += f_abs

    # ---- Percent scale(s) ----
    total_max_tetanic = float(np.sum(tetanic))
    total_force_percent = 100.0 * total_force / max(total_max_tetanic, 1e-12)

    per_pool_percent = None
    if per_pool_force is not None:
        per_pool_percent = np.zeros_like(per_pool_force)
        for p in range(nb_pools):
            denom = max(per_pool_max_tetanic[p], 1e-12)
            per_pool_percent[p, :] = 100.0 * per_pool_force[p, :] / denom

    out = dict(
        pool_max_tetanic_force=total_max_tetanic,        # keep legacy key name
        pool_force=total_force,                           # keep legacy key name
        pool_force_percent=total_force_percent,           # keep legacy key name
        per_pool=dict(
            names=pool_names,
            max_tetanic=per_pool_max_tetanic if per_pool_max_tetanic is not None else np.array([]),
            force=per_pool_force if per_pool_force is not None else np.empty((0, T)),
            force_percent=per_pool_percent if per_pool_percent is not None else np.empty((0, T)),
        )
    )
    if save_per_mu:
        out["mu_force"] = mu_force
    return out
def build_force_target_percent_ts(fsamp,
                                  duration_with_ignored_window,
                                  edges_ignore_duration,
                                  mode="constant",
                                  constant_percent=0.0,
                                  ramp_min_percent=0.0,
                                  ramp_max_percent=50.0):
    """
    Return target %MVC over the *full* duration (including ignored edges).
    - constant: flat value for whole trace
    - ramp_up:  leading edge fixed at min, linear ramp across the core, trailing edge fixed at max
    """
    T = int(np.round(float(duration_with_ignored_window) * float(fsamp)))
    E = int(np.round(float(edges_ignore_duration) * float(fsamp)))
    core_len = max(T - 2*E, 0)

    target = np.zeros(T, dtype=float)

    if mode == "constant":
        val = float(constant_percent)
        target[:] = val
    elif mode == "ramp_up":
        vmin = float(ramp_min_percent)
        vmax = float(ramp_max_percent)
        target[:E]  = vmin
        target[-E:] = vmax
        if core_len > 0:
            target[E:E+core_len] = np.linspace(vmin, vmax, core_len)
    else:
        # Unknown -> zeros
        pass

    return target
def generate_timevarying_baseline_from_target_helper(fsamp,
                                              duration_with_ignored_window,
                                              edges_ignore_duration,
                                              target_mode,
                                              target_constant_percent,
                                              target_ramp_min_percent,
                                              target_ramp_max_percent,
                                              soma_diameters_um,            # shape (nMU,)
                                              motoneurons_rheobases_nA,     # shape (nMU,) in nA
                                              multiplier=2.0,
                                              generate_figure=False,
                                              savepath=None):
    """
    1) Build target %MVC time series over the full duration.
    2) Map each target % to a diameter along [dmin..dmax].
    3) Interpolate rheobase(d) from your MU rheobases vs diameter.
    4) Baseline(t) = multiplier * rheobase(d_target(t))  [in nA]

    Returns:
      baseline_ts_nA : (T,) float, same for all pools/MUs (for now)
      target_percent_ts : (T,) float
    """
    # Target % timeseries
    target_percent_ts = build_force_target_percent_ts(
        fsamp=fsamp,
        duration_with_ignored_window=duration_with_ignored_window,
        edges_ignore_duration=edges_ignore_duration,
        mode=target_mode,
        constant_percent=target_constant_percent,
        ramp_min_percent=target_ramp_min_percent,
        ramp_max_percent=target_ramp_max_percent
    )

    T = target_percent_ts.size
    d = np.asarray(soma_diameters_um, float)
    rb = np.asarray(motoneurons_rheobases_nA, float)

    # Sort by diameter to interpolate smoothly
    order = np.argsort(d)
    d_sorted = d[order]
    rb_sorted = rb[order]
    # start rheobase at 0
    rb_sorted -= rb_sorted[0]
    dmin, dmax = float(d_sorted[0]), float(d_sorted[-1])
    # Desired diameter at each time from target %
    p = np.clip(target_percent_ts / 100.0, 0.0, 1.0)
    d_desired = dmin + p * (dmax - dmin)
    # Interpolate rheobase(d)
    rb_desired = np.interp(d_desired, d_sorted, rb_sorted)

    baseline_ts_nA = float(multiplier) * rb_desired  # nA

    # Optional quick viz
    if generate_figure:
        import matplotlib.pyplot as plt
        t = np.arange(T) / float(fsamp)
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=120, sharex=True)
        ax[0].plot(t, target_percent_ts, lw=2, color="tab:orange")
        ax[0].set_ylabel("Target (%MVC)")
        ax[0].set_title("Target and baseline (first guess)")
        ax[1].plot(t, baseline_ts_nA, lw=2, color="tab:blue")
        ax[1].set_ylabel("Baseline (nA)")
        ax[1].set_xlabel("Time (s)")
        fig.tight_layout()
        if savepath is not None:
            plt.savefig(os.path.join(savepath, "Target_and_Baseline_FirstGuess.png"),
                        bbox_inches="tight")
        plt.close(fig)

    return baseline_ts_nA.astype(float), target_percent_ts.astype(float)
def plot_pool_force_tracking(fig_dir, fsamp, pool_force_percent, target_percent_ts, mask, rmse_val,
                             common_input_by_pool=None, baseline_input_by_pool=None,
                             per_pool_force_percent=None, pool_names=None,
                             zoom_duration_s=3.0, zoom_center_s=None,
                             burst_center_s=None, burst_start_s=None,
                             burst_trace_nA=None, burst_zoom_start_s=None,
                             task_event_times_s=None):
    """
    Overview figure with:
      - left column: full-duration force (fixed 0-100 %MVC) + common input (µA) on a second y-axis
      - right column: zoomed common input, force, and force derivative stacked vertically
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    def _as_2d(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        return x

    force_color = "tab:blue"
    target_color = "#08306B"
    input_color = "tab:red"
    neutral_color = "0.25"
    neutral_fill = "0.7"

    def _stack_common_inputs(common_input_obj):
        if common_input_obj is None:
            return None
        if isinstance(common_input_obj, dict):
            keys = sorted(common_input_obj.keys())
            arr = np.stack([np.asarray(common_input_obj[k], dtype=float) for k in keys], axis=0)
        else:
            arr = _as_2d(common_input_obj)
        return arr / 1000.0  # nA -> µA

    def _stack_baseline_inputs(baseline_obj, nb_rows, n_samples):
        if baseline_obj is None:
            return None
        arr = np.asarray(baseline_obj, dtype=float)
        if arr.ndim == 0:
            arr = np.full((nb_rows, n_samples), float(arr), dtype=float)
        elif arr.ndim == 1:
            if arr.size == n_samples:
                arr = np.repeat(arr[None, :], nb_rows, axis=0)
            elif arr.size >= nb_rows:
                arr = arr[:nb_rows]
                arr = np.repeat(arr[:, None], n_samples, axis=1)
            elif arr.size == 1:
                arr = np.full((nb_rows, n_samples), float(arr.item()), dtype=float)
            else:
                raise ValueError(
                    "baseline_input_by_pool 1D input must have length 1, at least nb_pools, or n_samples"
                )
        elif arr.ndim == 2:
            if arr.shape == (1, n_samples) and nb_rows > 1:
                arr = np.repeat(arr, nb_rows, axis=0)
            elif arr.shape == (nb_rows, 1):
                arr = np.repeat(arr, n_samples, axis=1)
            elif arr.shape == (1, 1):
                arr = np.full((nb_rows, n_samples), float(arr[0, 0]), dtype=float)
            elif arr.shape[0] >= nb_rows:
                arr = arr[:nb_rows]
                if arr.shape[1] == 1:
                    arr = np.repeat(arr, n_samples, axis=1)
                elif arr.shape[1] != n_samples:
                    raise ValueError(
                        "baseline_input_by_pool 2D input columns must match 1 or n_samples"
                    )
            else:
                raise ValueError(
                    "baseline_input_by_pool 2D input must have one row or at least one row per pool"
                )
        else:
            raise ValueError("baseline_input_by_pool must be scalar, 1D, or 2D")
        return arr / 1000.0  # nA -> µA

    def _auto_limits(data, zero_floor=False, pad_frac=0.08, min_span=1.0):
        data = np.asarray(data, dtype=float)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return (0.0, 1.0) if zero_floor else (-1.0, 1.0)
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        if zero_floor:
            y_min = 0.0
        span = y_max - y_min
        if span < 1e-12:
            pad = max(min_span * 0.5, abs(y_max) * pad_frac, 0.5)
        else:
            pad = span * pad_frac
        return y_min - (0.0 if zero_floor else pad), y_max + pad

    total_force_percent = np.asarray(pool_force_percent, dtype=float)
    if total_force_percent.ndim != 1:
        total_force_percent = total_force_percent.squeeze()
    target_percent_ts = np.asarray(target_percent_ts, dtype=float)
    per_pool_force_percent = (
        _as_2d(per_pool_force_percent)
        if per_pool_force_percent is not None and np.size(per_pool_force_percent) > 0
        else _as_2d(total_force_percent)
    )
    nb_pools = per_pool_force_percent.shape[0]
    common_input_uA = _stack_common_inputs(common_input_by_pool)

    lengths = [total_force_percent.size, per_pool_force_percent.shape[1], target_percent_ts.size]
    if common_input_uA is not None:
        lengths.append(common_input_uA.shape[1])
    T = int(min(lengths))
    total_force_percent = total_force_percent[:T]
    per_pool_force_percent = per_pool_force_percent[:, :T]
    target_percent_ts = target_percent_ts[:T]

    if common_input_uA is None:
        common_input_uA = np.zeros((nb_pools, T), dtype=float)
    else:
        common_input_uA = common_input_uA[:, :T]
        if common_input_uA.shape[0] == 1 and nb_pools > 1:
            common_input_uA = np.repeat(common_input_uA, nb_pools, axis=0)
        elif common_input_uA.shape[0] != nb_pools:
            raise ValueError("common_input_by_pool must have one row or one row per pool")

    baseline_input_uA = _stack_baseline_inputs(baseline_input_by_pool, nb_pools, T)
    if baseline_input_uA is None:
        baseline_input_uA = np.zeros((nb_pools, T), dtype=float)
    else:
        baseline_input_uA = baseline_input_uA[:, :T]

    net_input_uA = common_input_uA + baseline_input_uA
    burst_trace_uA = None
    if burst_trace_nA is not None:
        burst_trace_uA = np.asarray(burst_trace_nA, dtype=float).squeeze()[:T] / 1000.0

    mask = np.asarray(mask, dtype=bool)[:T]
    if mask.size != T:
        mask = np.ones(T, dtype=bool)
    if pool_names is None or len(pool_names) != nb_pools:
        pool_names = [f"pool_{i}" for i in range(nb_pools)]

    t = np.arange(T) / float(fsamp)
    core_idx = np.flatnonzero(mask)
    if core_idx.size:
        core_start_s = t[core_idx[0]]
        core_end_s = t[core_idx[-1]]
    else:
        core_start_s = 0.0
        core_end_s = t[-1] if T else 0.0
    zoom_duration_s = min(float(zoom_duration_s), max(core_end_s - core_start_s, 1.0 / fsamp))
    if burst_zoom_start_s is not None:
        zoom_start_s = max(core_start_s, float(burst_zoom_start_s))
        zoom_end_s = min(core_end_s, zoom_start_s + zoom_duration_s)
        zoom_start_s = max(core_start_s, zoom_end_s - zoom_duration_s)
    else:
        if zoom_center_s is None:
            zoom_center_s = burst_center_s if burst_center_s is not None else 0.5 * (core_start_s + core_end_s)
        zoom_start_s = max(core_start_s, zoom_center_s - 0.5 * zoom_duration_s)
        zoom_end_s = min(core_end_s, zoom_start_s + zoom_duration_s)
        zoom_start_s = max(core_start_s, zoom_end_s - zoom_duration_s)
    zoom_mask = (t >= zoom_start_s) & (t <= zoom_end_s)
    if not np.any(zoom_mask):
        zoom_mask[max(0, T // 2 - 1):min(T, T // 2 + 2)] = True

    force_derivative = np.gradient(per_pool_force_percent, 1.0 / float(fsamp), axis=1)

    fig = plt.figure(figsize=(17, 9), dpi=140)
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        width_ratios=[1.7, 1.0],
        height_ratios=[1, 1, 1],
        wspace=0.28, hspace=0.18
    )

    ax_force_main = fig.add_subplot(gs[:, 0])
    ax_input_main = ax_force_main.twinx()
    ax_zoom_input = fig.add_subplot(gs[0, 1])
    ax_zoom_force = fig.add_subplot(gs[1, 1], sharex=ax_zoom_input)
    ax_zoom_deriv = fig.add_subplot(gs[2, 1], sharex=ax_zoom_input)

    def _add_burst_markers(ax):
        _draw_diagnostic_markers(
            ax,
            task_event_times_s=task_event_times_s,
            burst_center_s=burst_center_s,
            burst_start_s=burst_start_s,
        )

    ax_force_main.fill_between(
        t, 0, 100, where=~mask, color=neutral_fill, alpha=0.15, label="Ignored edges", zorder=0
    )
    ax_force_main.plot(
        t, target_percent_ts, color=target_color, lw=2.0, ls="--", alpha=0.4,
        label="Target (%MVC)", zorder=4
    )
    ax_force_main.plot(
        t, total_force_percent, color=force_color, lw=2.4, alpha=0.95,
        label="Total force (%MVC)", zorder=5
    )
    for p_i, (pname, force_trace) in enumerate(zip(pool_names, per_pool_force_percent)):
        alpha = 0.45 if nb_pools > 1 else 0.0
        if nb_pools > 1:
            ax_force_main.plot(
                t, force_trace, color=force_color, lw=1.2, alpha=alpha,
                label=f"{pname} force (%MVC)", zorder=3
            )

    ax_force_main.set_xlim(t[0], t[-1] if T else 1.0)
    ax_force_main.set_ylim(0, 100)
    ax_force_main.set_xlabel("Time (s)")
    ax_force_main.set_ylabel("Force (% MVC)")
    ax_force_main.set_title(f"Force overview and zoom window (RMSE = {rmse_val:.2f}% MVC)")
    ax_force_main.grid(alpha=0.25)

    input_handles = []
    if net_input_uA is not None:
        for p_i, (pname, input_trace) in enumerate(zip(pool_names, net_input_uA)):
            alpha = 0.95 if p_i == 0 else max(0.35, 0.75 - 0.2 * p_i)
            (line,) = ax_input_main.plot(t, input_trace, color=input_color, lw=1.3, alpha=alpha, ls="-", zorder=2,
                                         label=f"{pname} net input (µA)")
            line.set_label(f"{pname} net input (uA)")
            input_handles.append(line)
        _, input_ymax = _auto_limits(net_input_uA, zero_floor=True, min_span=0.5)
        ax_input_main.set_ylim(0, input_ymax)
    else:
        ax_input_main.set_ylim(0, 1.0)
    ax_input_main.set_ylabel("Common input (µA)")

    ax_input_main.set_ylabel("Net common input (uA)")
    force_zoom_vals = np.concatenate([per_pool_force_percent[:, zoom_mask], target_percent_ts[zoom_mask][None, :]], axis=0)
    force_y0, force_y1 = _auto_limits(force_zoom_vals, zero_floor=False, min_span=2.0)
    force_rect = Rectangle((zoom_start_s, force_y0), zoom_end_s - zoom_start_s, force_y1 - force_y0,
                           fill=False, ec=neutral_color, lw=1.8, ls="--", zorder=30)
    ax_force_main.add_patch(force_rect)

    if net_input_uA is not None:
        input_y0, input_y1 = _auto_limits(net_input_uA[:, zoom_mask], zero_floor=False, min_span=0.5)
        input_rect_y0 = max(0.0, input_y0)
        input_rect = Rectangle((zoom_start_s, input_rect_y0), zoom_end_s - zoom_start_s, input_y1 - input_rect_y0,
                               fill=False, ec="0.35", lw=1.6, ls=":", zorder=30)
        ax_input_main.add_patch(input_rect)

    handles, labels = ax_force_main.get_legend_handles_labels()
    handles += input_handles
    labels += [h.get_label() for h in input_handles]
    ax_force_main.legend(handles, labels, loc="upper left", fontsize=9, ncol=2, frameon=True)
    _add_burst_markers(ax_force_main)

    if net_input_uA is not None:
        for p_i, (pname, input_trace) in enumerate(zip(pool_names, net_input_uA)):
            alpha = 0.95 if p_i == 0 else max(0.35, 0.75 - 0.2 * p_i)
            ax_zoom_input.plot(t[zoom_mask], input_trace[zoom_mask], color=input_color, lw=1.5, alpha=alpha, label=pname, zorder=3)
        ax_zoom_input.set_ylim(*_auto_limits(net_input_uA[:, zoom_mask], zero_floor=False, min_span=0.5))
    else:
        ax_zoom_input.text(0.5, 0.5, "No common input saved", transform=ax_zoom_input.transAxes,
                           ha="center", va="center")
    ax_zoom_input.set_ylabel("Input (µA)")
    ax_zoom_input.set_ylabel("Input (uA)")
    ax_zoom_input.set_title("Zoomed common input")
    ax_zoom_input.grid(alpha=0.25)
    if burst_trace_uA is not None:
        input_zoom_y0, input_zoom_y1 = ax_zoom_input.get_ylim()
        burst_overlay_center = 0.5 * (input_zoom_y0 + input_zoom_y1)
        burst_overlay = burst_overlay_center + burst_trace_uA[zoom_mask]
        ax_zoom_input.plot(
            t[zoom_mask], burst_overlay, color="#7f0000", lw=2.0, alpha=0.9,
            label="Burst only (centered)", zorder=6
        )

    ax_zoom_force.plot(t[zoom_mask], target_percent_ts[zoom_mask], color=target_color, lw=2.0, ls="--", alpha=0.4, label="Target", zorder=4)
    ax_zoom_force.plot(t[zoom_mask], total_force_percent[zoom_mask], color=force_color, lw=2.1, alpha=0.95, label="Total", zorder=5)
    for p_i, (pname, force_trace) in enumerate(zip(pool_names, per_pool_force_percent)):
        if nb_pools > 1:
            alpha = max(0.35, 0.75 - 0.2 * p_i)
            ax_zoom_force.plot(t[zoom_mask], force_trace[zoom_mask], color=force_color, lw=1.4, alpha=alpha, label=pname, zorder=3)
    ax_zoom_force.set_ylim(force_y0, force_y1)
    ax_zoom_force.set_ylabel("Force (%MVC)")
    ax_zoom_force.set_title("Zoomed force")
    ax_zoom_force.grid(alpha=0.25)

    for p_i, (pname, deriv_trace) in enumerate(zip(pool_names, force_derivative)):
        alpha = 0.95 if p_i == 0 else max(0.35, 0.75 - 0.2 * p_i)
        ax_zoom_deriv.plot(t[zoom_mask], deriv_trace[zoom_mask], color=force_color, lw=1.4, alpha=alpha, label=pname, zorder=3)
    deriv_y0, deriv_y1 = _auto_limits(force_derivative[:, zoom_mask], zero_floor=False, min_span=5.0)
    ax_zoom_deriv.set_ylim(deriv_y0, deriv_y1)
    ax_zoom_deriv.axhline(0, color=neutral_color, lw=0.8, alpha=0.5, zorder=1)
    ax_zoom_deriv.set_xlabel("Time (s)")
    ax_zoom_deriv.set_ylabel("dForce/dt\n(%MVC/s)")
    ax_zoom_deriv.set_title("Zoomed force derivative")
    ax_zoom_deriv.grid(alpha=0.25)

    for ax in (ax_zoom_input, ax_zoom_force, ax_zoom_deriv):
        ax.set_xlim(zoom_start_s, zoom_end_s)
        _add_burst_markers(ax)

    if nb_pools > 1:
        ax_zoom_input.legend(loc="upper right", fontsize=8, frameon=True)

    fig.tight_layout()
    out = os.path.join(fig_dir, "fig_force_tracking.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
### Plotting helpers (for force)
def _color_by_size(d_um):
    """Return colors for each MU based on soma diameter (small→large)."""
    d_um = np.asarray(d_um, float)
    order = np.argsort(d_um)  # small→large
    cmap = get_cmap("viridis")
    norm = Normalize(vmin=d_um.min(), vmax=d_um.max())
    colors = np.zeros((len(d_um), 4))
    for rank, idx in enumerate(order):
        colors[idx] = cmap(norm(d_um[idx]))
    return colors, order, cmap, norm
def _add_third_axis(ax, offset=60, spine_name="right2"):
    """
    Add a 3rd y-axis on the right, offset outward by `offset` points.
    Returns the new axis.
    """
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", offset))
    # Matplotlib <3.8 needs this hack:
    ax3.spines["right"].set_visible(True)
    return ax3
def plot_force_params_overview(fig_dir, soma_d_um, tetanic, peakrel, tau_s):
    """
    Figure with two subplots:
      left: params vs MU index (sorted by size)
      right: params vs soma diameter (µm)
    Uses 3 y-axes so raw units remain readable.
    """
    idx_sorted = np.argsort(soma_d_um)
    n = len(soma_d_um)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), dpi=120)

    # ---- Left: x = MU index (sorted by size)
    x = np.arange(n)

    # primary y: tetanic
    axL.plot(x, tetanic[idx_sorted], label="Tetanic max force", color="tab:blue", lw=2)
    axL.set_xlabel("MU index (small → large)")
    axL.set_ylabel("Tetanic force (a.u.)", color="tab:blue")
    axL.tick_params(axis='y', labelcolor="tab:blue")

    # second y (shared right): twitch_time
    axL2 = axL.twinx()
    axL2.plot(x, tau_s[idx_sorted], label="Twitch time τ", color="tab:orange", lw=2, ls="--")
    axL2.set_ylabel("Twitch time τ (s)", color="tab:orange")
    axL2.tick_params(axis='y', labelcolor="tab:orange")

    # third y (offset): peakrel
    axL3 = _add_third_axis(axL)
    axL3.plot(x, peakrel[idx_sorted], label="Twitch peak (rel.)", color="tab:green", lw=2, ls="-.")
    axL3.set_ylabel("Twitch peak (fraction of tetanic)", color="tab:green")
    axL3.tick_params(axis='y', labelcolor="tab:green")

    axL.set_title("Force parameters vs MU index")

    # ---- Right: x = soma diameter
    x2 = soma_d_um[idx_sorted]
    axR.plot(x2, tetanic[idx_sorted], color="tab:blue", lw=2, label="Tetanic max force")
    axR.set_xlabel("Soma diameter (µm)")
    axR.set_ylabel("Tetanic force (a.u.)", color="tab:blue")
    axR.tick_params(axis='y', labelcolor="tab:blue")

    axR2 = axR.twinx()
    axR2.plot(x2, tau_s[idx_sorted], color="tab:orange", lw=2, ls="--", label="Twitch time τ")
    axR2.set_ylabel("Twitch time τ (s)", color="tab:orange")
    axR2.tick_params(axis='y', labelcolor="tab:orange")

    axR3 = _add_third_axis(axR)
    axR3.plot(x2, peakrel[idx_sorted], color="tab:green", lw=2, ls="-.", label="Twitch peak (rel.)")
    axR3.set_ylabel("Twitch peak (fraction)", color="tab:green")
    axR3.tick_params(axis='y', labelcolor="tab:green")

    fig.suptitle("Motor-unit force parameters", y=1.03, fontsize=13)
    fig.tight_layout()
    out = os.path.join(fig_dir, "fig_force_params.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
def plot_all_twitch_kernels(fig_dir, fsamp, peakrel, taus_s, soma_d_um, duration_factor=6.0):
    """
    Overlays all scaled twitch kernels (peakrel * h(t;τ)) for each MU.
    Colored from smallest→largest MU.
    """
    colors, order, cmap, norm = _color_by_size(soma_d_um)
    tmax = float(np.max(taus_s) * duration_factor)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

    for i in order:  # small→large
        h = _twitch_kernel(fsamp, taus_s[i], duration_factor=duration_factor) * peakrel[i]
        t = np.arange(len(h)) / float(fsamp)
        ax.plot(t, h, lw=1.2, color=colors[i], alpha=0.9)

    ax.set_xlim(0, tmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Twitch kernel × peak_rel (fraction of tetanic)")
    ax.set_title("All twitch kernels (colored by MU size)")
    # Colorbar to map diameter
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Soma diameter (µm)")
    out = os.path.join(fig_dir, "fig_twitch_kernels.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
def plot_cumulative_mu_forces(fig_dir, fsamp, mu_force, soma_d_um):
    """
    Plots cumulative (stacked) MU forces as lines: line k is sum of MUs 0..k (small→large).
    Colors follow the MU at each cumulative level.
    """
    colors, order, cmap, norm = _color_by_size(soma_d_um)
    mu_sorted = mu_force[order, :]
    cum = np.cumsum(mu_sorted, axis=0)  # shape: (n_mu, T)

    T = cum.shape[1]
    t = np.arange(T) / float(fsamp)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    n_mu = cum.shape[0]
    for k in range(n_mu):
        ax.plot(t, cum[k, :], lw=0.9, color=colors[order[k]], alpha=0.9)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative force (a.u.)")
    ax.set_title("Cumulative MU forces (small→large)")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Soma diameter (µm)")
    out = os.path.join(fig_dir, "fig_cumulative_forces.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
def plot_pool_forces_by_pool(fig_dir, fsamp, per_pool_force_percent, pool_names, mask, edges_ignore_duration):
    """
    Quick overlay of %MVC per pool (one line per pool).
    - Y axis is scaled to the data within the visible x-window (ignoring edges).
    - Displays Pearson correlation between the first two pools over the visible window.
    """
    if per_pool_force_percent.size == 0:
        return None

    T = per_pool_force_percent.shape[1]
    t = np.arange(T) / float(fsamp)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    for p_i, pname in enumerate(pool_names):
        ax.plot(t, per_pool_force_percent[p_i, :], lw=1.6, alpha=0.9, label=f"{pname} (%)")

    # Limit x to ignore edges
    ax.set_xlim(edges_ignore_duration, t[-1] - edges_ignore_duration)

    # Determine y-limits from ONLY the visible x-range
    x0, x1 = ax.get_xlim()
    in_view = (t >= x0) & (t <= x1)
    visible_data = per_pool_force_percent[:, in_view]
    if visible_data.size:
        y_min = np.nanmin(visible_data)
        y_max = np.nanmax(visible_data)
        if np.isfinite(y_min) and np.isfinite(y_max):
            if y_max - y_min < 1e-9:  # avoid zero range
                y_min -= 1.0
                y_max += 1.0
            pad = 0.05 * (y_max - y_min)
            ax.set_ylim(y_min - pad, y_max + pad)

    # Optional: shade ignored edges (uncomment if desired)
    # ax.fill_between(t, *ax.get_ylim(), where=~in_view, color='gray', alpha=0.12)

    # Correlation between the first two pools over the visible window
    if per_pool_force_percent.shape[0] >= 2:
        a = per_pool_force_percent[0, in_view]
        b = per_pool_force_percent[1, in_view]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() > 2:
            r = float(np.corrcoef(a[valid], b[valid])[0, 1])
            corr_label = (f"Pearson r ({pool_names[0]} vs {pool_names[1]}): {r:.3f}"
                          if len(pool_names) >= 2 else f"Pearson r: {r:.3f}")
        else:
            corr_label = "Pearson r: n/a"
        ax.text(0.01, 0.99, corr_label,
                transform=ax.transAxes, ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("% MVC")
    ax.set_title("Force by pool (each pool's % of MVC)\n(Ignoring edges for x and y scaling)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = os.path.join(fig_dir, "fig_pool_forces_by_pool.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


### Other helper functions
def _ensure_logging(): # Define logger to keep track of each simulation's progress despite parallelization
    """Make sure each process has a file+console logger attached."""
    root = logging.getLogger()
    if not root.handlers:
        # file handler
        fh = logging.FileHandler("simulations_progress_log.log", mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(processName)s %(levelname)s: %(message)s"))
        fh.setLevel(logging.INFO)
        root.addHandler(fh)
        # console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(processName)s: %(message)s"))
        root.addHandler(ch)
        root.setLevel(logging.INFO)
def make_unique_output_dir(parent_folder, prefix="simulation_output_"):
    os.makedirs(parent_folder, exist_ok=True)
    i = 0
    while True:
        dirname = f"{parent_folder}//{prefix}{i}"
        try:
            # this call is atomic: if two processes hit the same i,
            # only one will succeed, the other will get FileExistsError
            os.mkdir(dirname)
            return dirname, i
        except FileExistsError:
            i += 1
def wrap_motoneuron_properties(motoneuron_soma_diameter,
                               motoneurons_resistance, motoneurons_input_weight,
                               motoneurons_capacitance, motoneurons_membrane_conductance,
                               motoneurons_membrane_time_constant,
                               motoneurons_AHP_duration, motoneurons_AHP_conductance_decay_time_constant,
                               motoneurons_refractory_periods, motoneurons_rheobases,
                               motoneurons_spike_transmission_delays):
    return {
        'soma_diameter': motoneuron_soma_diameter,
        'resistance': motoneurons_resistance,
        'input_weight': motoneurons_input_weight,
        'capacitance': motoneurons_capacitance,
        'membrane_conductance': motoneurons_membrane_conductance,
        'membrane_time_constant': motoneurons_membrane_time_constant,
        'AHP_duration': motoneurons_AHP_duration,
        'AHP_conductance_decay_time_constant': motoneurons_AHP_conductance_decay_time_constant,
        'refractory_period': motoneurons_refractory_periods,
        'rheobase': motoneurons_rheobases,
        'spike_transmission_delay': motoneurons_spike_transmission_delays
    }

######################################
### FUNCTIONS TO SET UP THE SIMULATION
######################################

# # # SET BRIAN2 EQUATIONS
def set_brian2_equations():
    MN_equations = Equations('''
        dv/dt = (
            - g_leak*(v - voltage_rest)           # leak current from membrane conductance
            - (g_ahp_f + g_ahp_s)*(v - voltage_AHP) # AHP current from AHP conductance (based on potassium reversal potential) => g_ahp_f is FAST (regular AHP), g_ahp_s is SLOW (ensuring spike frequency adaptation and saturation)
            + input_weight*(input_MN_timedarray_amp(t,i) + I_syn)  # excitatory + inhibitory synaptic currents (weighted by input_weight, i.e. normalized resistance)
        )/C_m : volt (unless refractory)

        dI_syn/dt = -I_syn/tau_syn  : amp  # Input from RC decays exponentially
        dg_ahp_f/dt = -g_ahp_f/tau_ahp_f       : siemens  # AHP conductance decays exponentially
        dg_ahp_s/dt = -g_ahp_s/tau_ahp_s    : siemens

        g_leak            : siemens
        C_m               : farad                    
        input_weight      : 1
        voltage_rest      : volt
        voltage_AHP       : volt
        refractory_period : second
        tau_syn           : second  # synaptic input from RC time constant (can be either the membrane time constant or an arbitrarily defined time constant)
        tau_ahp_f         : second   # fast AHP component (size-dependent, "base" AHP tau)
        tau_ahp_s         : second   # slow AHP component
        g_ahp_s_max       : siemens  # cap on slow AHP conductance (no cap on fast AHP)
    ''')
    RC_equations = Equations('''
        dv/dt = (input_RC_timedarray_volt(t,i)-v)/tau: volt (unless refractory)
        tau : second
        ''')
    return MN_equations, RC_equations

# # # CREATE MOTOR NEURON SIZES AND ASSIGN THEM TO THEIR POOLS
def _sample_truncated_exponential_0_to_L(n_samples, tau, upper_bound, eps=1e-12):
    upper_bound = float(upper_bound)
    tau = float(tau)
    if upper_bound <= 0.0 or tau <= eps:
        return np.zeros(int(n_samples), dtype=float)
    u = np.random.uniform(0.0, 1.0, size=int(n_samples))
    return -tau * np.log1p(u * np.expm1(-upper_bound / tau))


def _truncated_exponential_pdf_0_to_L(x, tau, upper_bound, eps=1e-12):
    x = np.asarray(x, dtype=float)
    upper_bound = float(upper_bound)
    tau = float(tau)
    pdf = np.zeros_like(x, dtype=float)
    if upper_bound <= 0.0 or tau <= eps:
        return pdf
    mask = (x >= 0.0) & (x <= upper_bound)
    norm = -np.expm1(-upper_bound / tau)
    if norm <= eps:
        pdf[mask] = 1.0 / upper_bound
        return pdf
    pdf[mask] = np.exp(-x[mask] / tau) / (tau * norm)
    return pdf


def _truncated_lognormal_moments(mu_ln, sigma_ln, lower_bound, upper_bound):
    sigma_ln = float(sigma_ln)
    lower_bound = float(lower_bound)
    upper_bound = float(upper_bound)
    if upper_bound <= 0.0:
        raise ValueError("upper_bound must be strictly positive for truncated lognormal weights")
    if lower_bound < 0.0:
        raise ValueError("lower_bound must be non-negative for truncated lognormal weights")
    if sigma_ln <= 1e-12:
        value = float(np.clip(np.exp(mu_ln), lower_bound, upper_bound))
        return value, value * value

    upper_log = np.log(upper_bound)
    if lower_bound <= 0.0:
        lower_log = -np.inf
        cdf_lower = 0.0
    else:
        lower_log = np.log(lower_bound)
        cdf_lower = float(ndtr((lower_log - mu_ln) / sigma_ln))
    cdf_upper = float(ndtr((upper_log - mu_ln) / sigma_ln))
    norm = max(cdf_upper - cdf_lower, 1e-15)

    def _moment(order):
        shift = order * sigma_ln * sigma_ln
        upper_term = float(ndtr((upper_log - mu_ln - shift) / sigma_ln))
        if lower_bound <= 0.0:
            lower_term = 0.0
        else:
            lower_term = float(ndtr((lower_log - mu_ln - shift) / sigma_ln))
        raw = np.exp(order * mu_ln + 0.5 * (order * sigma_ln) ** 2) * (upper_term - lower_term)
        return float(raw / norm)

    return _moment(1), _moment(2)


def _solve_truncated_lognormal_mu_for_mean(target_mean, sigma_ln, lower_bound, upper_bound):
    sigma_ln = float(sigma_ln)
    if sigma_ln <= 1e-12:
        return float(np.log(max(target_mean, 1e-12)))

    def _mean_for_mu(mu_value):
        mean_value, _ = _truncated_lognormal_moments(mu_value, sigma_ln, lower_bound, upper_bound)
        return mean_value

    step = max(1.0, 2.0 * sigma_ln)
    mu_lo = float(np.log(max(target_mean, 1e-12))) - step
    mu_hi = float(np.log(max(target_mean, 1e-12))) + step
    while _mean_for_mu(mu_lo) > target_mean:
        mu_lo -= step
        step *= 1.5
    step = max(1.0, 2.0 * sigma_ln)
    while _mean_for_mu(mu_hi) < target_mean:
        mu_hi += step
        step *= 1.5
    for _ in range(90):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        if _mean_for_mu(mu_mid) < target_mean:
            mu_lo = mu_mid
        else:
            mu_hi = mu_mid
    return 0.5 * (mu_lo + mu_hi)


def _solve_truncated_lognormal_params_for_mean_sd(target_mean, target_sd, lower_bound, upper_bound):
    target_mean = float(target_mean)
    target_sd = float(target_sd)
    lower_bound = float(lower_bound)
    upper_bound = float(upper_bound)
    if target_sd <= 1e-12:
        return {
            "mu_ln": 0.0,
            "sigma_ln": 0.0,
            "mean": float(target_mean),
            "sd": 0.0,
        }

    def _stats_for_sigma(sigma_value):
        mu_value = _solve_truncated_lognormal_mu_for_mean(target_mean, sigma_value, lower_bound, upper_bound)
        mean_value, second_moment = _truncated_lognormal_moments(mu_value, sigma_value, lower_bound, upper_bound)
        variance = max(second_moment - mean_value * mean_value, 0.0)
        return mu_value, mean_value, float(np.sqrt(variance))

    sigma_lo = 1e-6
    sigma_hi = 0.2
    _, _, sd_hi = _stats_for_sigma(sigma_hi)
    while sd_hi < target_sd and sigma_hi < 64.0:
        sigma_hi *= 2.0
        _, _, sd_hi = _stats_for_sigma(sigma_hi)
    if sd_hi < target_sd - 1e-6:
        raise ValueError(
            f"Requested common_input_weight_sd={target_sd:.4g} is too large for bounds "
            f"[{lower_bound:.4g}, {upper_bound:.4g}] while keeping mean 1."
        )
    for _ in range(80):
        sigma_mid = 0.5 * (sigma_lo + sigma_hi)
        _, _, sd_mid = _stats_for_sigma(sigma_mid)
        if sd_mid < target_sd:
            sigma_lo = sigma_mid
        else:
            sigma_hi = sigma_mid
    sigma_final = 0.5 * (sigma_lo + sigma_hi)
    mu_final, mean_final, sd_final = _stats_for_sigma(sigma_final)
    return {
        "mu_ln": float(mu_final),
        "sigma_ln": float(sigma_final),
        "mean": float(mean_final),
        "sd": float(sd_final),
    }


def _truncated_lognormal_pdf(x, mu_ln, sigma_ln, lower_bound, upper_bound):
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x, dtype=float)
    if sigma_ln <= 1e-12:
        return pdf
    lower_bound = float(lower_bound)
    upper_bound = float(upper_bound)
    if lower_bound <= 0.0:
        cdf_lower = 0.0
    else:
        cdf_lower = float(ndtr((np.log(lower_bound) - mu_ln) / sigma_ln))
    cdf_upper = float(ndtr((np.log(upper_bound) - mu_ln) / sigma_ln))
    norm = max(cdf_upper - cdf_lower, 1e-15)
    mask = (x >= lower_bound) & (x <= upper_bound) & (x > 0.0)
    if not np.any(mask):
        return pdf
    x_mask = x[mask]
    pdf[mask] = (
        np.exp(-0.5 * ((np.log(x_mask) - mu_ln) / sigma_ln) ** 2)
        / (x_mask * sigma_ln * np.sqrt(2.0 * np.pi) * norm)
    )
    return pdf


def sample_mean_one_truncated_lognormal_weights(
    n_units,
    *,
    target_sd,
    min_weight,
    max_weight,
):
    n_units = int(n_units)
    if n_units <= 0:
        raise ValueError("n_units must be strictly positive")
    target_sd = float(target_sd)
    min_weight = float(min_weight)
    max_weight = float(max_weight)
    if target_sd <= 1e-12:
        weights = np.ones(n_units, dtype=float)
        return {
            "weights": weights,
            "mu_ln": 0.0,
            "sigma_ln": 0.0,
            "mean": 1.0,
            "sd": 0.0,
            "min_weight": min_weight,
            "max_weight": max_weight,
        }

    solved = _solve_truncated_lognormal_params_for_mean_sd(
        target_mean=1.0,
        target_sd=target_sd,
        lower_bound=min_weight,
        upper_bound=max_weight,
    )
    if min_weight <= 0.0:
        lower_cdf = 0.0
    else:
        lower_cdf = float(ndtr((np.log(min_weight) - solved["mu_ln"]) / solved["sigma_ln"]))
    upper_cdf = float(ndtr((np.log(max_weight) - solved["mu_ln"]) / solved["sigma_ln"]))
    u = np.random.uniform(lower_cdf, upper_cdf, size=n_units)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    weights = np.exp(solved["mu_ln"] + solved["sigma_ln"] * ndtri(u))
    return {
        "weights": np.asarray(weights, dtype=float),
        "mu_ln": solved["mu_ln"],
        "sigma_ln": solved["sigma_ln"],
        "mean": solved["mean"],
        "sd": solved["sd"],
        "min_weight": min_weight,
        "max_weight": max_weight,
    }


def sample_common_input_modulation_weights(
    n_units,
    *,
    target_sd,
    min_weight,
    max_weight,
):
    return sample_mean_one_truncated_lognormal_weights(
        n_units,
        target_sd=target_sd,
        min_weight=min_weight,
        max_weight=max_weight,
    )


def plot_truncated_lognormal_weight_distribution(
    weights,
    *,
    mu_ln,
    sigma_ln,
    min_weight,
    max_weight,
    target_sd,
    xlabel,
    title_prefix,
    filename,
    savepath=None,
):
    weights = np.asarray(weights, dtype=float).reshape(-1)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8), dpi=120)
    hist_weights = np.ones_like(weights) * (100.0 / max(len(weights), 1))
    _, bins, _ = ax.hist(
        weights,
        bins=min(24, max(8, int(np.sqrt(max(len(weights), 1))))),
        weights=hist_weights,
        color="gray",
        alpha=0.75,
        edgecolor="white",
        label="Sampled weights",
    )
    if sigma_ln > 1e-12:
        x_pdf = np.linspace(float(min_weight), float(max_weight), 500)
        pdf = _truncated_lognormal_pdf(
            x_pdf,
            mu_ln=float(mu_ln),
            sigma_ln=float(sigma_ln),
            lower_bound=float(min_weight),
            upper_bound=float(max_weight),
        )
        typical_bin_width = np.mean(np.diff(bins)) if len(bins) > 1 else max(float(max_weight) - float(min_weight), 1.0)
        ax.plot(
            x_pdf,
            pdf * (100.0 * typical_bin_width),
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="Analytical truncated-lognormal PDF",
        )
    ax.axvline(float(min_weight), color="C1", linewidth=1.8, label="Min weight")
    ax.axvline(float(max_weight), color="C3", linewidth=1.8, label="Max weight")
    ax.axvline(1.0, color="C2", linewidth=1.5, linestyle=":", label="Mean = 1")
    ax.set_xlabel(str(xlabel))
    ax.set_ylabel("Proportion (% of MNs)")
    ax.set_title(
        f"{str(title_prefix)}\n"
        f"target SD={float(target_sd):.3g}, sampled SD={float(np.std(weights)):.3g}"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(os.path.join(savepath, str(filename)), bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_common_input_weight_distribution(
    weights,
    *,
    mu_ln,
    sigma_ln,
    min_weight,
    max_weight,
    target_sd,
    savepath=None,
):
    plot_truncated_lognormal_weight_distribution(
        weights,
        mu_ln=mu_ln,
        sigma_ln=sigma_ln,
        min_weight=min_weight,
        max_weight=max_weight,
        target_sd=target_sd,
        xlabel="Common-input modulation weight",
        title_prefix="Distribution of per-MU common-input modulation weights",
        filename="Common_input_modulation_weights.png",
        savepath=savepath,
    )


def plot_independent_input_weight_distribution(
    weights,
    *,
    mu_ln,
    sigma_ln,
    min_weight,
    max_weight,
    target_sd,
    savepath=None,
):
    plot_truncated_lognormal_weight_distribution(
        weights,
        mu_ln=mu_ln,
        sigma_ln=sigma_ln,
        min_weight=min_weight,
        max_weight=max_weight,
        target_sd=target_sd,
        xlabel="Independent-input weight",
        title_prefix="Distribution of per-MU independent-input weights",
        filename="Independent_input_weights.png",
        savepath=savepath,
    )


def _resolve_tau_size_um(min_soma_diameter, max_soma_diameter, tau_size_mode, tau_size_ratio, tau_size_um):
    diameter_span = float(max_soma_diameter) - float(min_soma_diameter)
    mode = str(tau_size_mode)
    if mode == "tau_size_um":
        resolved_tau = float(tau_size_um)
    elif mode == "tau_size_ratio":
        resolved_tau = float(tau_size_ratio) * max(diameter_span, 0.0)
    else:
        raise ValueError(f"Unknown tau_size_mode '{tau_size_mode}'. Expected 'tau_size_ratio' or 'tau_size_um'.")
    return max(resolved_tau, 0.0), diameter_span


def generate_motor_neurons(
        min_soma_diameter, max_soma_diameter,
        nb_pools, nb_motoneurons_per_pool, total_nb_motoneurons,
        tau_size_mode, tau_size_ratio, tau_size_um,
        generate_figure=False, savepath=None):
    tau_size, diameter_span = _resolve_tau_size_um(
        min_soma_diameter=min_soma_diameter,
        max_soma_diameter=max_soma_diameter,
        tau_size_mode=tau_size_mode,
        tau_size_ratio=tau_size_ratio,
        tau_size_um=tau_size_um,
    )
    sampled_offsets = _sample_truncated_exponential_0_to_L(
        n_samples=total_nb_motoneurons,
        tau=tau_size,
        upper_bound=diameter_span,
    )
    motoneuron_soma_diameters = np.sort(float(min_soma_diameter) + sampled_offsets)
    if diameter_span > 0.0:
        motoneuron_normalized_soma_diameters = np.clip(
            (motoneuron_soma_diameters - float(min_soma_diameter)) / diameter_span,
            0.0,
            1.0,
        )
    else:
        motoneuron_normalized_soma_diameters = np.zeros(total_nb_motoneurons, dtype=float)
    pool_list_by_MN, idx_of_MN_by_pool = [], {}
    for pooli in range(nb_pools):
        poolname_temp = f"pool_{pooli}"
        start = pooli * nb_motoneurons_per_pool
        stop = (pooli + 1) * nb_motoneurons_per_pool
        idx_of_MN_by_pool[poolname_temp] = np.arange(start, stop)
        pool_list_by_MN.extend([poolname_temp] * nb_motoneurons_per_pool)
    
    if generate_figure:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        tau_label = f"{tau_size:.2f} µm"
        if str(tau_size_mode) == "tau_size_ratio":
            tau_label += f" (ratio={float(tau_size_ratio):.3g})"
        else:
            tau_label += f" (absolute; ratio={float(tau_size_ratio):.3g})"
        # 1) Histogram of raw soma diameters
        ax = axes[0]
        weights = np.ones_like(motoneuron_soma_diameters) * (100.0 / len(motoneuron_soma_diameters))
        _, bins, _ = ax.hist(
            motoneuron_soma_diameters,
            density=False,
            weights=weights,
            edgecolor='white',
            color='gray',
            alpha=0.75,
            label='Sampled diameters'
        )
        if diameter_span > 0.0 and tau_size > 1e-12:
            x_pdf = np.linspace(float(min_soma_diameter), float(max_soma_diameter), 400)
            pdf = _truncated_exponential_pdf_0_to_L(
                x_pdf - float(min_soma_diameter),
                tau=tau_size,
                upper_bound=diameter_span,
            )
            typical_bin_width = np.mean(np.diff(bins)) if len(bins) > 1 else diameter_span
            ax.plot(
                x_pdf,
                pdf * (100.0 * typical_bin_width),
                color='C0',
                linewidth=2.0,
                label='Analytical truncated-exponential PDF',
            )
        ymin, ymax = ax.get_ylim()
        ax.vlines(min_soma_diameter, ymin, ymax, color='C1', label='Min soma diameter', linewidth=2)
        ax.vlines(max_soma_diameter, ymin, ymax, color='C3', label='Max soma diameter', linewidth=2)
        ax.set_xlabel("Soma diameter (μm)")
        ax.set_ylabel("Proportion (% of MNs)")
        ax.set_title(f"Distribution of MN soma diameters\nτ = {tau_label}")
        ax.legend(loc='upper right')
        # 2) Histogram of normalized soma diameters
        ax = axes[1]
        _, bins_norm, _ = ax.hist(
            motoneuron_normalized_soma_diameters,
            density=False,
            weights=weights,
            edgecolor='white',
            color='gray',
            alpha=0.5,
            label='Sampled normalized diameters'
        )
        if diameter_span > 0.0 and tau_size > 1e-12:
            x_norm = np.linspace(0.0, 1.0, 400)
            pdf_norm = _truncated_exponential_pdf_0_to_L(
                x_norm * diameter_span,
                tau=tau_size,
                upper_bound=diameter_span,
            ) * diameter_span
            typical_bin_width_norm = np.mean(np.diff(bins_norm)) if len(bins_norm) > 1 else 1.0
            ax.plot(
                x_norm,
                pdf_norm * (100.0 * typical_bin_width_norm),
                color='C0',
                linewidth=2.0,
                label='Analytical normalized PDF',
            )
        ymin, ymax = ax.get_ylim()
        ax.vlines(0, ymin, ymax, color='C1', label='Min normalized', linewidth=2)
        ax.vlines(1, ymin, ymax, color='C3', label='Max normalized', linewidth=2)
        ax.set_xlabel("Normalized soma diameter")
        ax.set_ylabel("Proportion (% of MNs)")
        ax.set_title(f"Normalized MN size distribution\nτ = {tau_label}")
        ax.legend(loc='upper right')
        # 3) Soma diameter vs index
        ax = axes[2]
        ax.plot(motoneuron_soma_diameters, color='gray', label='Sorted sampled diameters')
        mean_diameter = np.mean(motoneuron_soma_diameters)
        ax.hlines(mean_diameter, 0, len(motoneuron_soma_diameters)-1, color='C2', linestyle='--',
                label=f"Mean = {mean_diameter:.1f} μm")
        ax.set_xlabel("MN index")
        ax.set_ylabel("Soma diameter (μm)")
        ax.set_title("MN soma diameter vs. index")
        ax.legend(loc='upper right')
        # Set layout and save
        plt.tight_layout()
        if savepath is not None:
            save_file = os.path.join(savepath, 'MN_sizes.png')
            fig.savefig(save_file)
        plt.show()

    return (motoneuron_soma_diameters, motoneuron_normalized_soma_diameters,
            pool_list_by_MN, idx_of_MN_by_pool)

# # # CREATE MOTOR NEURON ELECTROPHYSIOLOGICAL PROPERTIES
def generate_motor_neuron_electrophysiological_properties(
        total_nb_motoneurons, motoneuron_soma_diameters,
        resistance_constant, resistance_exponent,
        capacitance_constant, capacitance_exponent,
        AHP_duration_constant, AHP_duration_exponent,
        rheobase_constant, rheobase_exponent, rheobase_scaling,
        refractory_period_absolute,
        axonal_conduction_velocity_constant, axonal_conduction_velocity_exponent,
        generate_figure=False, savepath=None):
    motoneurons_resistance = np.zeros(total_nb_motoneurons)
    motoneurons_input_weight = np.zeros(total_nb_motoneurons)
    motoneurons_capacitance = np.zeros(total_nb_motoneurons)
    motoneurons_membrane_conductance = np.zeros(total_nb_motoneurons)
    motoneurons_membrane_time_constant = np.zeros(total_nb_motoneurons)
    motoneurons_AHP_duration = np.zeros(total_nb_motoneurons)
    motoneurons_AHP_conductance_decay_time_constant = np.zeros(total_nb_motoneurons)
    motoneurons_refractory_periods = np.zeros(total_nb_motoneurons)
    motoneurons_rheobases = np.zeros(total_nb_motoneurons)
    motoneurons_spike_transmission_delays = np.zeros(total_nb_motoneurons)
    for mni in range(total_nb_motoneurons):
        motoneurons_resistance[mni] = resistance_constant*(motoneuron_soma_diameters[mni]**resistance_exponent)
        motoneurons_input_weight[mni] = motoneurons_resistance[mni] / motoneurons_resistance[0] # normalized value so that smallest MN has weight of 1
        motoneurons_capacitance[mni] = capacitance_constant*(motoneuron_soma_diameters[mni]**capacitance_exponent)
        motoneurons_membrane_conductance[mni] = 1/motoneurons_resistance[mni]
        motoneurons_membrane_time_constant[mni] = (motoneurons_resistance[mni]*ohm) * (motoneurons_capacitance[mni]*farad) / 100 # using the right units should make it be in seconds already. Diving by 100 because some mismatch of scale somewhere
        motoneurons_AHP_duration[mni] = AHP_duration_constant*(motoneuron_soma_diameters[mni]**AHP_duration_exponent)
        motoneurons_AHP_conductance_decay_time_constant[mni] = motoneurons_AHP_duration[mni] / np.log(10) # Time constant defined so that the AHP duration corresponds to the duration to reach 1/10th of the hyperpolarizing increase in conductance caused by a spike
        motoneurons_refractory_periods[mni] = refractory_period_absolute
        motoneurons_rheobases[mni] = rheobase_constant*(motoneuron_soma_diameters[mni]**rheobase_exponent)*rheobase_scaling
        motoneurons_spike_transmission_delays[mni] = 0.5/(axonal_conduction_velocity_constant*(motoneuron_soma_diameters[mni]**axonal_conduction_velocity_exponent))  # in s # The delay (s) is calculated from the axonal conduction velocity, assuming a 0.5m axon length => so correspond to the conduction speed from MN to muscle fiber (speed in m/s, so multiply speed by 2 -> numerator is 0.5 meter)
    
    motoneuron_properties_dict = wrap_motoneuron_properties(motoneuron_soma_diameters,
        motoneurons_resistance, motoneurons_input_weight,
        motoneurons_capacitance, motoneurons_membrane_conductance,
        motoneurons_membrane_time_constant,
        motoneurons_AHP_duration, motoneurons_AHP_conductance_decay_time_constant,
        motoneurons_refractory_periods, motoneurons_rheobases,
        motoneurons_spike_transmission_delays)
    
    if generate_figure:
        # Create a 5×1 grid of subplots
        fig, axes = plt.subplots(6, 1, figsize=(8, 22), sharex=True)
        # 1) Resistance & Input weight (dual axis)
        ax1 = axes[0]
        curve1, = ax1.plot(motoneurons_resistance, label="Resistance (Ω)", color='C1', linewidth=2)
        ax2 = ax1.twinx()
        curve2, = ax2.plot(
            motoneurons_input_weight, 
            label="Normalized input resistance", 
            color='C5', 
            linewidth=2, 
            linestyle=":"
        )
        ax1.set_ylabel("Resistance (Ω)", color='C1')
        ax2.set_ylabel("Normalized input resistance", color='C5')
        ax1.tick_params(axis='y', labelcolor='C1')
        ax2.tick_params(axis='y', labelcolor='C5')
        ax1.legend([curve1, curve2], [curve1.get_label(), curve2.get_label()], loc='best')
        ax1.set_title("Motoneuron Resistance & Input Weight")
        # 2) Capacitance
        ax3 = axes[1]
        ax3.plot(motoneurons_capacitance, label="Capacitance (F)", color='C2')
        ax3.set_ylabel("Capacitance (F)")
        ax3.legend(loc='best')
        ax3.set_title("Motoneuron Capacitance")
        # 3) Membrane conductance
        ax4 = axes[2]
        ax4.plot(motoneurons_membrane_conductance, label="Membrane conductance (mS)", color='C3')
        ax4.set_ylabel("Conductance (mS)")
        ax4.legend(loc='best')
        ax4.set_title("Motoneuron Membrane Conductance")
        # 5) Membrane time constant (determines RC IPSP effect)
        ax5 = axes[3]
        ln1, = ax5.plot(motoneurons_membrane_time_constant, label="Membrane time constant (ms)", color='black', linestyle='--')
        ax5.set_ylabel("Time (ms)")
        ax5.set_title("Membrane time constant\n(can be used for RC's IPSPs decay rate)")
        # 6) AHP & Refractory
        ax6 = axes[4]
        ln1, = ax6.plot(motoneurons_AHP_duration, label="AHP duration (ms)", color='blue', linestyle='--')
        ln2, = ax6.plot(motoneurons_AHP_conductance_decay_time_constant, label="AHP time constant (ms)", color='blue')
        ln3, = ax6.plot(motoneurons_refractory_periods, label="Refractory period (ms)", color='C6')
        ax6.set_ylabel("Time (ms)")
        ax6.legend([ln1, ln2, ln3], [ln1.get_label(), ln2.get_label(), ln3.get_label()], loc='best')
        ax6.set_title("AHP & Refractory Properties")
        # 7) Rheobase
        ax7 = axes[5]
        ax7.plot(motoneurons_rheobases, label="Rheobase (nA)", color='C7')
        ax7.set_xlabel("MN index")
        ax7.set_ylabel("Rheobase (nA)")
        ax7.legend(loc='best')
        ax7.set_title("Motoneuron Rheobase")
        # Adjust layout and save
        plt.tight_layout()
        if savepath is not None:
            new_filename = f'MN_electrophysiological_properties.png'
            save_file_path = os.path.join(savepath, new_filename)
            plt.savefig(save_file_path)
        plt.show()

    return (motoneurons_resistance, motoneurons_input_weight, motoneurons_capacitance, motoneurons_membrane_conductance, motoneurons_membrane_time_constant,
            motoneurons_AHP_conductance_decay_time_constant, motoneurons_refractory_periods, motoneurons_rheobases, motoneurons_spike_transmission_delays,
            motoneuron_properties_dict)

# # # GENERATE AND DISTRIBUTE COMMON INPUT(S)
def generate_and_distribute_common_inputs(fsamp,
    nb_pools, nb_motoneurons_per_pool,
    duration_with_ignored_window, edges_ignore_duration,
    lf_target_sd, lf_band_hz,
    alpha_target_sd_final, alpha_band_hz, alpha_envelope_band_hz,
    beta_target_sd_final, beta_band_hz, beta_envelope_band_hz,
    alpha_beta_envelope_variability_scale=1.0,
    common_input_weight_sd=0.0,
    common_input_weight_min=0.0,
    common_input_weight_max=5.0,
    lf_trend_start_rel_go_cue_s=-1.0,
    alpha_trend_start_rel_go_cue_s=-1.0,
    beta_trend_start_rel_go_cue_s=-1.0,
    lf_trend_slope_nA_per_s=0.0,
    alpha_trend_slope_nA_per_s=0.0,
    beta_trend_slope_nA_per_s=0.0,
    enable_input_burst=False,
    input_burst_center_ms=5000.0,
    input_burst_sigma_ms=250.0,
    lf_burst_peak_nA=0.0,
    alpha_burst_peak_nA=0.0,
    beta_burst_peak_nA=0.0,
    input_burst_start_marker_n_sigma=2.0,
    generate_figure=False, savepath=None, save_power_csv=False
):
    """
    Generate the single-pool common input from three explicit no-burst components:
    low-frequency + enveloped alpha + enveloped beta.
    """
    if nb_pools != 1:
        raise ValueError("The simplified common-input generator expects nb_pools == 1")
    alpha_beta_envelope_variability_scale = float(alpha_beta_envelope_variability_scale)
    if not (0.0 <= alpha_beta_envelope_variability_scale <= 1.0):
        raise ValueError("alpha_beta_envelope_variability_scale must be between 0 and 1")

    rng = np.random
    n_samples = int(np.round(duration_with_ignored_window * fsamp))
    dt = 1.0 / float(fsamp)
    component_padding_s = max(float(edges_ignore_duration), 2.0)
    t_s = np.arange(n_samples, dtype=float) / float(fsamp)
    baseline_mask = t_s < float(edges_ignore_duration + max(0.0, (float(input_burst_center_ms) / 1000.0 - edges_ignore_duration)))
    filter_params = _get_filter_params()
    if bool(getattr(filter_params, "enable_step_current", False)) and not bool(getattr(filter_params, "enable_go_nogo_cue_diagnostics", True)):
        step_abs_s = float(filter_params.task_event_times_s["step_current_s"])
        step_pre_rel = _normalize_rel_window(filter_params.step_pre_window_rel_s)
        reference_start_s = float(step_abs_s + step_pre_rel[0])
        reference_end_s = float(step_abs_s + step_pre_rel[1])
        baseline_mask = (t_s >= reference_start_s) & (t_s < reference_end_s)
    else:
        task_baseline_end_s = float(filter_params.task_event_times_s["baseline_segment_end_s"])
        baseline_mask = t_s < task_baseline_end_s
    go_nogo_cue_s = float(filter_params.task_event_times_s["go_nogo_cue_s"])

    input_burst_info = None
    lf_burst_kernel_nA = np.zeros(n_samples, dtype=float)
    alpha_burst_kernel_nA = np.zeros(n_samples, dtype=float)
    beta_burst_kernel_nA = np.zeros(n_samples, dtype=float)
    if enable_input_burst:
        input_burst_info = generate_component_burst_kernels(
            fsamp=fsamp,
            duration_with_ignored_window=duration_with_ignored_window,
            burst_center_ms=input_burst_center_ms,
            burst_sigma_ms=input_burst_sigma_ms,
            lf_burst_peak_nA=lf_burst_peak_nA,
            alpha_burst_peak_nA=alpha_burst_peak_nA,
            beta_burst_peak_nA=beta_burst_peak_nA,
            burst_start_marker_n_sigma=input_burst_start_marker_n_sigma,
        )
        lf_burst_kernel_nA = input_burst_info["lf_burst_kernel_nA"]
        alpha_burst_kernel_nA = input_burst_info["alpha_burst_kernel_nA"]
        beta_burst_kernel_nA = input_burst_info["beta_burst_kernel_nA"]

    input_trend_info = generate_component_trend_kernels(
        fsamp=fsamp,
        duration_with_ignored_window=duration_with_ignored_window,
        go_nogo_cue_s=go_nogo_cue_s,
        lf_trend_start_rel_go_cue_s=lf_trend_start_rel_go_cue_s,
        alpha_trend_start_rel_go_cue_s=alpha_trend_start_rel_go_cue_s,
        beta_trend_start_rel_go_cue_s=beta_trend_start_rel_go_cue_s,
        lf_trend_slope_nA_per_s=lf_trend_slope_nA_per_s,
        alpha_trend_slope_nA_per_s=alpha_trend_slope_nA_per_s,
        beta_trend_slope_nA_per_s=beta_trend_slope_nA_per_s,
    )
    lf_trend_kernel_nA = input_trend_info["lf_trend_kernel_nA"]
    alpha_trend_kernel_nA = input_trend_info["alpha_trend_kernel_nA"]
    beta_trend_kernel_nA = input_trend_info["beta_trend_kernel_nA"]

    lf_components = generate_low_frequency_component(
        n_samples=n_samples,
        dt=dt,
        target_sd=lf_target_sd,
        band_hz=lf_band_hz,
        rng=rng,
        padding_s=component_padding_s,
        baseline_mask=baseline_mask,
        burst_kernel_nA=lf_burst_kernel_nA,
        trend_kernel_nA=lf_trend_kernel_nA,
    )
    alpha_components = generate_enveloped_band_component(
        n_samples=n_samples,
        dt=dt,
        carrier_band_hz=alpha_band_hz,
        envelope_band_hz=alpha_envelope_band_hz,
        target_sd_final=alpha_target_sd_final,
        rng=rng,
        padding_s=component_padding_s,
        component_name="alpha",
        baseline_mask=baseline_mask,
        burst_kernel_nA=alpha_burst_kernel_nA,
        trend_kernel_nA=alpha_trend_kernel_nA,
        envelope_variability_scale=alpha_beta_envelope_variability_scale,
    )
    beta_components = generate_enveloped_band_component(
        n_samples=n_samples,
        dt=dt,
        carrier_band_hz=beta_band_hz,
        envelope_band_hz=beta_envelope_band_hz,
        target_sd_final=beta_target_sd_final,
        rng=rng,
        padding_s=component_padding_s,
        component_name="beta",
        baseline_mask=baseline_mask,
        burst_kernel_nA=beta_burst_kernel_nA,
        trend_kernel_nA=beta_trend_kernel_nA,
        envelope_variability_scale=alpha_beta_envelope_variability_scale,
    )

    common_input_no_burst = (
        lf_components["lf_final"]
        + alpha_components["alpha_final"]
        + beta_components["beta_final"]
    )
    delivered_common_input = (
        lf_components["lf_final_with_burst"]
        + alpha_components["alpha_final_with_burst"]
        + beta_components["beta_final_with_burst"]
    )
    common_input_trend_trace_nA = (
        lf_components["lf_trend_only_delta_nA"]
        + alpha_components["alpha_trend_only_delta_nA"]
        + beta_components["beta_trend_only_delta_nA"]
    )
    common_input_burst_trace_nA = (
        lf_components["lf_burst_only_delta_nA"]
        + alpha_components["alpha_burst_only_delta_nA"]
        + beta_components["beta_burst_only_delta_nA"]
    )

    common_input_weight_info = sample_common_input_modulation_weights(
        nb_pools * nb_motoneurons_per_pool,
        target_sd=common_input_weight_sd,
        min_weight=common_input_weight_min,
        max_weight=common_input_weight_max,
    )
    common_input_weights = np.asarray(common_input_weight_info["weights"], dtype=float)

    MN_excit_input = {0: delivered_common_input}
    corr_mat = np.eye(nb_pools)

    total_nb_mn = nb_pools * nb_motoneurons_per_pool
    inputs_to_mn_weight_matrix = np.zeros((total_nb_mn, nb_pools))
    inputs_to_mn_weight_matrix[:, 0] = common_input_weights

    freqs_common_final, psd_common_final = compute_psd_welch(delivered_common_input, fsamp=fsamp)
    freqs_common_no_burst, psd_common_no_burst = compute_psd_welch(common_input_no_burst, fsamp=fsamp)
    total_power = {0: float(np.trapz(psd_common_final, freqs_common_final))}
    common_input_reference_power = {0: float(np.trapz(psd_common_no_burst, freqs_common_no_burst))}
    power_per_frequency_band = {
        "frequencies": freqs_common_final,
        "power": psd_common_final,
    }

    input_components = {
        **lf_components,
        **alpha_components,
        **beta_components,
        "common_input_no_burst": np.asarray(common_input_no_burst, dtype=float),
        "common_input_with_burst": np.asarray(delivered_common_input, dtype=float),
        "common_input_trend_trace_nA": np.asarray(common_input_trend_trace_nA, dtype=float),
        "common_input_burst_trace_nA": np.asarray(common_input_burst_trace_nA, dtype=float),
        "common_input_modulation_weights": common_input_weights,
        "common_input_modulation_weight_mu_ln": np.asarray([common_input_weight_info["mu_ln"]], dtype=float),
        "common_input_modulation_weight_sigma_ln": np.asarray([common_input_weight_info["sigma_ln"]], dtype=float),
        "common_input_modulation_weight_target_sd": np.asarray([common_input_weight_sd], dtype=float),
        "common_input_modulation_weight_min": np.asarray([common_input_weight_min], dtype=float),
        "common_input_modulation_weight_max": np.asarray([common_input_weight_max], dtype=float),
        "alpha_beta_envelope_variability_scale": np.asarray([alpha_beta_envelope_variability_scale], dtype=float),
    }
    if input_burst_info is not None:
        input_burst_info["trace_nA"] = np.asarray(common_input_burst_trace_nA, dtype=float)
        input_burst_info["envelope"] = np.asarray(input_burst_info["gaussian_kernel"], dtype=float)

    if savepath is not None and save_power_csv:
        pd.DataFrame(power_per_frequency_band).to_csv(
            os.path.join(savepath, 'power_per_frequency_band_common_input.csv'),
            index=False,
        )
    if generate_figure:
        plot_common_input_weight_distribution(
            common_input_weights,
            mu_ln=common_input_weight_info["mu_ln"],
            sigma_ln=common_input_weight_info["sigma_ln"],
            min_weight=common_input_weight_min,
            max_weight=common_input_weight_max,
            target_sd=common_input_weight_sd,
            savepath=savepath,
        )

    return (
        MN_excit_input,
        corr_mat,
        inputs_to_mn_weight_matrix,
        total_power,
        common_input_reference_power,
        power_per_frequency_band,
        input_burst_info,
        input_components,
    )

# # # GENERATE FORCE-TARGET FOLLOWING TIME-VARYING BASELINE EXCITATORY INPUT
# === Build time-varying baseline (optional) and keep target % for later plotting ===
def generate_timevarying_baseline_from_target_main(params, motoneurons_properties_dict):
    baseline_ts_nA = None
    target_percent_ts = None
    if getattr(params, "variable_excitatory_input_baseline", False):
        # Pull what we need from your property dict / params
        soma_d_um = np.asarray(motoneurons_properties_dict["soma_diameter"], float)
        rheobase_nA = np.asarray(motoneurons_properties_dict["rheobase"], float)

        baseline_ts_nA, target_percent_ts = generate_timevarying_baseline_from_target_helper(
            fsamp=params.fsamp,
            duration_with_ignored_window=params.duration_with_ignored_window,
            edges_ignore_duration=params.edges_ignore_duration,
            target_mode=params.force_target_mode,                              # "constant" or "ramp_up"
            target_constant_percent=params.force_target_constant_percent,
            target_ramp_min_percent=params.force_target_ramp_min_percent,
            target_ramp_max_percent=params.force_target_ramp_max_percent,
            soma_diameters_um=soma_d_um,
            motoneurons_rheobases_nA=rheobase_nA,
            multiplier=params.excitatory_input_multiplier_of_rheobase,
            generate_figure=_figures_enabled(params) and params.output_plot_target_baseline_figure,
            savepath=params.output_dir if hasattr(params, "output_dir") else None
        )
    else:
        # Still useful to have a target for plotting; uses defaults
        target_percent_ts = build_force_target_percent_ts(
            fsamp=params.fsamp,
            duration_with_ignored_window=params.duration_with_ignored_window,
            edges_ignore_duration=params.edges_ignore_duration,
            mode=params.force_target_mode,
            constant_percent=params.force_target_constant_percent,
            ramp_min_percent=params.force_target_ramp_min_percent,
            ramp_max_percent=params.force_target_ramp_max_percent
        )
    return baseline_ts_nA, target_percent_ts

# # # GENERATE INDEPENDENT INPUTS AND CREATE BRIAN2 TIME ARRAYS
def generate_independent_inputs(fsamp,
        nb_pools, total_nb_motoneurons, total_nb_renshaw_cells,
        low_pass_filter_of_MN_independent_input,
        low_pass_filter_of_RC_independent_input,
        ref_common_input_power,
        MN_independent_input_absolute_or_ratio,
        MN_independent_input_power,
        independent_input_weight_sd,
        independent_input_weight_max,
        RC_independent_input_std,
        MN_excit_input,
        inputs_to_mn_weight_matrix,
        excitatory_input_baseline, # list of size <= nb_pools
        duration_with_ignored_window,
        edges_ignore_duration,
        motoneurons_rheobases,
        generate_figure=False, generate_weight_figure=False, savepath=None,
        save_power_csv=False,
        excitatory_input_baseline_timeseries=None   # <--- (n_samples,) in nA; same for all pools/MUs
    ):
    # logger = logging.getLogger(__name__)
    # Motor neuron - generate independent input
    artifact_removal_window, _ = filter_artifact_removal(fsamp, edges_ignore_duration) # Get artifact removal window
    MN_independent_input = []
    independent_input_weight_info = sample_mean_one_truncated_lognormal_weights(
        total_nb_motoneurons,
        target_sd=independent_input_weight_sd,
        min_weight=0.0,
        max_weight=independent_input_weight_max,
    )
    independent_input_weights = np.asarray(independent_input_weight_info["weights"], dtype=float)
    nperseg = min(fsamp, len(MN_excit_input[0]))  # only first pool
    for mni in range(total_nb_motoneurons):
        temp_input = Generate_filtered_gaussian_noise_input(fsamp, 
            duration_with_ignored_window, edges_ignore_duration, artifact_removal_window,
            low_pass_filter_cutoff=low_pass_filter_of_MN_independent_input,
            input_mean=0, scaling_std=1)
        pooli = np.round(mni // (total_nb_motoneurons/nb_pools)).astype(int)
        # Scale it to the desired power
        temp_input_power = np.mean(temp_input**2)
        if MN_independent_input_absolute_or_ratio == 'absolute':
            independent_input_scaling_factor = MN_independent_input_power
        elif MN_independent_input_absolute_or_ratio == 'ratio':
            desired_independent_power = MN_independent_input_power * ref_common_input_power[pooli]
            independent_input_scaling_factor = np.sqrt(desired_independent_power / temp_input_power)
        temp_input *= (independent_input_scaling_factor * independent_input_weights[mni])
        # Keep the first independent input as a separate variable for plotting purpose:
        MN_independent_input.append(temp_input)
        if generate_figure and mni==0:
            independent_input_for_plotting = temp_input.copy()
        # # Some logs for debugging
        #     logger.info(f"Pool # = {pooli}")
        #     logger.info(f"inital temp_input_power = {temp_input_power:.2f}")
        #     logger.info(f"ref_common_input_power = {ref_common_input_power[0]:.2f}")
        #     logger.info(f"desired_independent_power = {desired_independent_power:.2f}")
        #     logger.info(f"Scaling factor = {independent_input_scaling_factor:.2f}")
        #     logger.info(f"Transformed temp input power = {np.mean(temp_input**2):.2f}")
        #     logger.info(f"independent_input_for_plotting power = {np.mean(independent_input_for_plotting**2):.2f}")
        #     logger.info(f"Excitatory input shape = {MN_excit_input[pooli].shape}")
        #     # logger.info(f"Excitatory input power = {np.mean((np.array(MN_excit_input[pooli]).flatten() * inputs_to_mn_weight_matrix[mni, pooli])**2)}")
        #     logger.info(f"Excitatory input power = {np.mean(np.array(MN_excit_input[pooli])**2)}")
        #     logger.info(f"Excitatory input power multiplied by weight matrix = {np.mean((np.array(MN_excit_input[pooli]) * inputs_to_mn_weight_matrix[mni, pooli])**2)}")
    # Get mean power of the independent input
    max_freq_lim = 150
    psd_independent_input = []
    for mni in range(len(MN_independent_input)):
        freqs, psd_temp = welch( # freqs stay the same each time so no need to have a dict containing them for each pool
            MN_independent_input[mni],
            fs=fsamp,
            window='hann',
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling='density',
            detrend='constant')
        psd_independent_input.append(psd_temp)
    power_per_frequency_band_independent = {"frequencies": freqs,
                                            "power": np.mean(np.array(psd_independent_input), axis = 0)}
    total_power_independent = np.trapz(power_per_frequency_band_independent["power"], freqs)
    power_per_frequency_band_independent = {"frequencies": freqs[np.array(range(max_freq_lim)).astype(int)],
                                            "power": power_per_frequency_band_independent["power"][np.array(range(max_freq_lim)).astype(int)]} 
    # Save power_per_frequency_band as csv file
    if savepath is not None and save_power_csv:
        power_per_frequency_band_df = pd.DataFrame(power_per_frequency_band_independent)
        power_per_frequency_band_df.to_csv(os.path.join(savepath, 'power_per_frequency_band_independent_input.csv'), index=False) 
    if generate_weight_figure:
        plot_independent_input_weight_distribution(
            independent_input_weights,
            mu_ln=independent_input_weight_info["mu_ln"],
            sigma_ln=independent_input_weight_info["sigma_ln"],
            min_weight=0.0,
            max_weight=independent_input_weight_max,
            target_sd=independent_input_weight_sd,
            savepath=savepath,
        )
    n_samples = len(MN_excit_input[0])
    temp_timed_array = np.zeros((n_samples, total_nb_motoneurons))
    for mni in range(total_nb_motoneurons):
        pooli = np.round(mni // (total_nb_motoneurons/nb_pools)).astype(int)
        temp_common_plus_baseline = np.array(
                MN_excit_input[pooli]) * inputs_to_mn_weight_matrix[mni, pooli] # the weights are relative to the distribution of common input to the different pools, not the size of the motor neurons!
        # add excitatory baseline (potentially time-varying if following a force target)
        if excitatory_input_baseline_timeseries is None:
            # No force target => per-pool constant baseline (nA)
            temp_common_plus_baseline += excitatory_input_baseline[pooli]
        else:
            # With force target => same time-varying baseline for every MU (nA)
            # ensure shapes match
            if excitatory_input_baseline_timeseries.ndim != 1:
                raise ValueError("excitatory_input_baseline_timeseries must be 1D (n_samples,)")
            if excitatory_input_baseline_timeseries.shape[0] != temp_timed_array.shape[0]:
                raise ValueError("baseline TS length != number of samples")
            temp_common_plus_baseline += excitatory_input_baseline_timeseries
        # LF bursts are allowed to create negative troughs in the LF component, but
        # the final tonic+common pool drive should stay non-negative.
        temp_common_plus_baseline = np.maximum(temp_common_plus_baseline, 0.0)
        temp_timed_array[:, mni] += temp_common_plus_baseline
        # add and independent input
        temp_timed_array[:, mni] += MN_independent_input[mni]
        # Rheobase = clip value to 0 if it is below a given value (in nA)
        temp_timed_array[:, mni] = np.clip(
            temp_timed_array[:, mni]-motoneurons_rheobases[mni],
            a_min=0, a_max=np.inf)
    input_MN_timedarray_amp = TimedArray(temp_timed_array * nA, dt=(1/fsamp)*second) # in nano Ampere

    # Renshaw cell - generate independent input and fill time array
    RC_independent_input = []
    temp_timed_array = np.zeros((n_samples, total_nb_renshaw_cells))
    for renshawi in range(total_nb_renshaw_cells):
        RC_independent_input.append(Generate_filtered_gaussian_noise_input(fsamp, 
            duration_with_ignored_window, edges_ignore_duration, artifact_removal_window,
            low_pass_filter_cutoff=low_pass_filter_of_RC_independent_input,
            input_mean=0, scaling_std=RC_independent_input_std))
    input_RC_timedarray_volt = TimedArray(temp_timed_array * mvolt, dt=(1/fsamp)*second)

    # Sanity check plot
    if generate_figure:
        plt.figure(figsize=(30,5))
        plt.plot(input_MN_timedarray_amp.values[:,
            np.linspace(0, total_nb_motoneurons-1, 10, dtype=int)],
            alpha=0.2, color = 'C0')
        test = np.mean(input_MN_timedarray_amp.values[:,np.arange(1,total_nb_motoneurons)],axis=1)
        plt.plot(test, color='darkblue')
        plt.xlabel("Time (samples)")
        plt.ylabel("Input (Amperes)")
        if savepath is not None:
            new_filename = f'Input_TimedArray_SanityCheck.png'
            save_file_path = os.path.join(savepath, new_filename)
            plt.savefig(save_file_path)
        plt.show()

        plt.figure()
        duration_to_plot = 3 # in seconds
        nb_samples_to_plot = int(np.round(duration_to_plot*fsamp))
        independent_input_for_plotting_power = np.mean(independent_input_for_plotting**2)
        common_input_for_plotting_power = np.mean(MN_excit_input[0]**2)
        if nb_samples_to_plot > len(independent_input_for_plotting):
            nb_samples_to_plot = len(independent_input_for_plotting)
        plt.plot(np.arange(nb_samples_to_plot)/fsamp,
            MN_excit_input[0][:nb_samples_to_plot] * inputs_to_mn_weight_matrix[0, 0],
            label=f"Common input delivered to MN#0\nPower={common_input_for_plotting_power:.2f}",
            color = 'red')
        plt.plot(np.arange(nb_samples_to_plot)/fsamp,
            independent_input_for_plotting[:nb_samples_to_plot],
            label=f"Independent input delivered to MN#0\nPower={independent_input_for_plotting_power:.2f}",
            color='blue', alpha=0.5, linewidth=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Input (nA)")
        plt.title(f"Recalculated ratio of independent VS common input = {independent_input_for_plotting_power/common_input_for_plotting_power:.2f}")
        plt.legend()
        if savepath is not None:
            new_filename = f'Input_common_VS_independent_MN0.png'
            save_file_path = os.path.join(savepath, new_filename)
            plt.savefig(save_file_path)
        plt.show()

    independent_input_payload = {
        "weights": independent_input_weights,
        "mu_ln": independent_input_weight_info["mu_ln"],
        "sigma_ln": independent_input_weight_info["sigma_ln"],
        "target_sd": float(independent_input_weight_sd),
        "max_weight": float(independent_input_weight_max),
    }
    return (
        input_MN_timedarray_amp,
        input_RC_timedarray_volt,
        total_power_independent,
        power_per_frequency_band_independent,
        independent_input_payload,
    )

# # # CREATE CONNECTIVITY BETWEEN MOTOR NEURONS AND RENSHAW CELLS
def create_connectivity(total_nb_motoneurons, total_nb_renshaw_cells,
        nb_pools, nb_motoneurons_per_pool, RC_pair_indices, nb_RCs_per_pool_pair,
        disynpatic_inhib_connections_desired_MN_MN,
        split_MN_RC_ratio, motoneuron_normalized_soma_diameters, motoneuron_soma_diameters,
        distribution_type, distribution_params, disynaptic_inhib_received_arbitrary_adjustment,
        distribution_binary_weights,
        generate_figure=False, savepath=None):
    # --- Allocate global adjacency matrices ---
    MN_to_Renshaw_connectivity_matrix = np.zeros(
        (total_nb_motoneurons, total_nb_renshaw_cells), dtype=float
    )
    Renshaw_to_MNs_connectivity_matrix = np.zeros(
        (total_nb_renshaw_cells, total_nb_motoneurons), dtype=float
    )
    # --- Fill in each pool‐pair block independently ---
    motoneuron_size_ranks   = np.asarray(motoneuron_normalized_soma_diameters)
    motoneuron_soma_diameters = np.asarray(motoneuron_soma_diameters)
    for i in range(nb_pools):
        mn_pre = np.arange(i*nb_motoneurons_per_pool,
                        (i+1)*nb_motoneurons_per_pool)
        for j in range(nb_pools):
            mn_post = np.arange(j*nb_motoneurons_per_pool,
                                (j+1)*nb_motoneurons_per_pool)
            rc_inds  = RC_pair_indices[(i, j)]
            nRC      = len(rc_inds)
            # desired *mean disynaptic count* for this block
            mean_disyn = disynpatic_inhib_connections_desired_MN_MN[i, j]
            # solve p_mn_rc * p_rc_mn = mean_disyn / nRC
            base_conn = mean_disyn / nRC
            p_mn_rc   = base_conn ** split_MN_RC_ratio
            p_rc_mn   = base_conn ** (1.0 - split_MN_RC_ratio)
            # sample MN→RC sub-matrix and write into global
            subA = sample_block(
                n_pre=len(mn_pre),
                n_post=nRC,
                p_block=p_mn_rc,
                dist=distribution_type,
                params=distribution_params,
                ranks=motoneuron_size_ranks[mn_pre],
                binary=distribution_binary_weights,
                diam_um=motoneuron_soma_diameters[mn_pre],
            )
            MN_to_Renshaw_connectivity_matrix[np.ix_(mn_pre, rc_inds)] = subA
            # sample RC→MN sub‐matrix and write into global
            subB = sample_block(
                n_pre=nRC,
                n_post=len(mn_post),
                p_block=p_rc_mn,
                dist="gaussian" if distribution_type=="size_powerlaw" else distribution_type,
                params=distribution_params,
                ranks=None,
                binary=distribution_binary_weights
            )
            Renshaw_to_MNs_connectivity_matrix[np.ix_(rc_inds, mn_post)] = subB
    # --- Modify the connectivity from RCs to MNs from the point of view of each MN
    if disynaptic_inhib_received_arbitrary_adjustment > 0:
        for mni in range(total_nb_motoneurons):
            Renshaw_to_MNs_connectivity_matrix[:, mni] += np.random.normal(loc=0, scale=disynaptic_inhib_received_arbitrary_adjustment, size=total_nb_renshaw_cells)
        # Make sure the weights are non-negative (no excitation from Renshaw cells!)
        Renshaw_to_MNs_connectivity_matrix = np.maximum(Renshaw_to_MNs_connectivity_matrix, 0)
    # --- Compute disynaptic MN→MN counts (or binary for 'binarize') ---
    MN_to_MN_counts = MN_to_Renshaw_connectivity_matrix.dot(
        Renshaw_to_MNs_connectivity_matrix)
    # np.fill_diagonal(MN_to_MN_counts, 0)
    if distribution_type == 'binarize':
        # interpret exactly as mean *probability* of ≥1 path
        MN_to_MN_connectivity_matrix = (MN_to_MN_counts > 0).astype(int)
    else:
        # keep raw counts as your “weights”
        MN_to_MN_connectivity_matrix = MN_to_MN_counts
    
    ### Plotting, if requested
    if generate_figure:
        mn_pool_labels = [f"MN pool {i}" for i in range(nb_pools)]
        rc_pair_list   = [(i,j) for i in range(nb_pools) for j in range(nb_pools)]
        rc_pool_pair_labels = [f"RC pool pair {i}-{j}" for (i,j) in rc_pair_list]
        plot_connectivity_matrix(
            MN_to_Renshaw_connectivity_matrix,
            "Monosynaptic MN → RC",
            mn_pool_labels,
            rc_pool_pair_labels,
            n_pre_pool=nb_motoneurons_per_pool,
            n_post_pool=nb_RCs_per_pool_pair,
            cmap='autumn',
            add_colorbar=False,
            savepath=f'{savepath}/Connectivity_MN_to_RC.png'
        )
        plot_connectivity_matrix(
            Renshaw_to_MNs_connectivity_matrix,
            "Monosynaptic RC → MN",
            rc_pool_pair_labels,
            mn_pool_labels,
            n_pre_pool=nb_RCs_per_pool_pair,
            n_post_pool=nb_motoneurons_per_pool,
            cmap='winter',
            add_colorbar=False,
            savepath=f'{savepath}/Connectivity_RC_to_MN.png'
        )
        plot_connectivity_matrix(
            MN_to_MN_connectivity_matrix,
            "Disynaptic MN → MN via RCs",
            mn_pool_labels,
            mn_pool_labels,
            n_pre_pool=nb_motoneurons_per_pool,
            n_post_pool=nb_motoneurons_per_pool,
            cmap='viridis',
            add_colorbar=True,
            savepath=f'{savepath}/Connectivity_MN_to_MN.png'
        )
        # ─── Histograms per pool→pool ─────────────────── #
        nP = nb_pools
        counts_MN_RC = {}
        counts_RC_MN = {}
        counts_MN_MN = {}
        for i in range(nP):
            mn_pre  = slice(i*nb_motoneurons_per_pool, (i+1)*nb_motoneurons_per_pool)
            for j in range(nP):
                mn_post = slice(j*nb_motoneurons_per_pool, (j+1)*nb_motoneurons_per_pool)
                rc_inds = RC_pair_indices[(i,j)]
                counts_MN_RC[(i,j)] = (
                    MN_to_Renshaw_connectivity_matrix[np.ix_(range(*mn_pre.indices(total_nb_motoneurons)), rc_inds)]
                    .sum(axis=1)
                )
                counts_RC_MN[(i,j)] = (
                    Renshaw_to_MNs_connectivity_matrix[np.ix_(rc_inds, range(*mn_post.indices(total_nb_motoneurons)))]
                    .sum(axis=0)
                )
                counts_MN_MN[(i,j)] = (
                    MN_to_MN_connectivity_matrix[mn_pre, mn_post]
                    .sum(axis=1)
                )
        fig = plt.figure(figsize=(4*nP, 4*nP))
        outer = gridspec.GridSpec(nP, nP, wspace=0.4, hspace=0.6)
        for i in range(nP):
            for j in range(nP):
                cell = outer[i,j]
                inner = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=cell,
                                                        height_ratios=[1,1], hspace=0.2)
                # top
                ax1 = fig.add_subplot(inner[0])
                ax1.hist(counts_MN_RC[(i,j)], bins='auto', alpha=0.6, label='MN→RC', color='C1')
                ax1.hist(counts_RC_MN[(i,j)], bins='auto', alpha=0.6, label='RC→MN', color='C0')
                ax1.set_ylabel("Count")
                ax1.set_title(f"Pools {i}→{j}")
                ax1.legend(fontsize=8, loc="upper left")
                ax1.text(0.95,0.7,
                        f"μ₁={counts_MN_RC[(i,j)].mean():.1f}±{counts_MN_RC[(i,j)].std():.1f}\n"
                        f"μ₂={counts_RC_MN[(i,j)].mean():.1f}±{counts_RC_MN[(i,j)].std():.1f}",
                        transform=ax1.transAxes, ha='right', va='top', fontsize=7)
                # bottom
                ax2 = fig.add_subplot(inner[1])
                ax2.hist(counts_MN_MN[(i,j)], bins='auto', alpha=0.6, color='green', label='MN→MN')
                ax2.set_xlabel("Number of synapses")
                ax2.set_ylabel("Count")
                ax2.legend(fontsize=8, loc="upper left")
                ax2.text(0.95,0.7,
                        f"μ₃={counts_MN_MN[(i,j)].mean():.1f}±{counts_MN_MN[(i,j)].std():.1f}",
                        transform=ax2.transAxes, ha='right', va='top', fontsize=7)
        plt.suptitle("Number of synapses per pool, per cell type")
        plt.tight_layout()
        if savepath is not None:
            new_filename = f'Connectivity_histogram.png'
            save_file_path = os.path.join(savepath, new_filename)
            plt.savefig(save_file_path)
        plt.show()
    
    return MN_to_MN_connectivity_matrix, MN_to_Renshaw_connectivity_matrix, Renshaw_to_MNs_connectivity_matrix

# # # CREATE Brian2 NEURONGROUPS AND SYNAPSES OBJECTS
def create_neurongroups_and_synapses_objects(
        total_nb_motoneurons, total_nb_renshaw_cells,
        MN_equations, RC_equations, voltage_rest, voltage_thresh,
        motoneurons_membrane_conductance, motoneurons_capacitance,
        motoneurons_input_weight, synaptic_IPSP_decay_time_constant_per_MN,
        motoneurons_AHP_conductance_decay_time_constant, # fast AHP (per MN, sie-dependent)
        AHP_slow_conductance_decay_time_constant_multiplier,
        AHP_fast_conductance_delta_after_spiking, AHP_slow_conductance_delta_after_spiking,
        maximum_AHP_conductance_slow_conductance,
        motoneurons_refractory_periods,
        MN_to_Renshaw_excit, Renshaw_to_MN_inhib,
        scale_initial_IPSP_to_be_same_integral_regardless_of_synaptic_tau,
        tau_Renshaw, MN_RC_synpatic_delay, refractory_period_RC,
        MN_to_Renshaw_connectivity_matrix, Renshaw_to_MNs_connectivity_matrix):
    logger = logging.getLogger(__name__)
    # # ----------------- BRIAN2 COMMON NAMESPACE FOR VARIABLES
    common_brian2_namespace = {
        'voltage_thresh': voltage_thresh,
        'voltage_rest':   voltage_rest,
        'refractory_period_RC': refractory_period_RC,
        'AHP_fast_conductance_delta_after_spiking': AHP_fast_conductance_delta_after_spiking,
        'AHP_slow_conductance_delta_after_spiking': AHP_slow_conductance_delta_after_spiking,
        'MN_to_Renshaw_excit': MN_to_Renshaw_excit,
        'Renshaw_to_MN_IPSP_integral': Renshaw_to_MN_inhib * second # Should be in amp * s = Coulomb (total charge). Renshaw_to_MN_inhib is already in amp. 
    }
    # ----------------- NEURON GROUPS ----------- 
    # MOTOR NEURONS
    motoneurons = NeuronGroup(
        total_nb_motoneurons, 
        MN_equations, 
        threshold='v > voltage_thresh', 
        reset='''
        v = voltage_rest
        g_ahp_f += AHP_fast_conductance_delta_after_spiking
        g_ahp_s += AHP_slow_conductance_delta_after_spiking * (1 - g_ahp_s/g_ahp_s_max) # the (1 - ...) term ensures conductance saturation
        ''',
        refractory='refractory_period',
        method='euler',
        namespace=common_brian2_namespace
    )
    motoneurons.v = voltage_rest # in mV #
    motoneurons.g_leak = motoneurons_membrane_conductance * msiemens # in milisiemens
    motoneurons.C_m = motoneurons_capacitance * ufarad # in microfarads
    motoneurons.refractory_period = motoneurons_refractory_periods * ms  # in milliseconds
    motoneurons.input_weight = motoneurons_input_weight # dimensionless unit
    motoneurons.tau_syn = synaptic_IPSP_decay_time_constant_per_MN # already in millisecond
    # AHP
    motoneurons.tau_ahp_f = motoneurons_AHP_conductance_decay_time_constant * ms # in millisecond
    motoneurons.tau_ahp_s = motoneurons_AHP_conductance_decay_time_constant * AHP_slow_conductance_decay_time_constant_multiplier * ms # in millisecond
    motoneurons.g_ahp_s_max = maximum_AHP_conductance_slow_conductance * msiemens # in milisiemens
    # Initialize AHP with a conductance of 0
    motoneurons.g_ahp_f = 0*siemens
    motoneurons.g_ahp_s = 0*siemens
    # logger.info(f"motoneurons tau syn = {motoneurons.tau_syn}")
    # logger.info(f"motoneurons input weights = {motoneurons.input_weight}")
    # logger.info(f"IPSP integrals = {common_brian2_namespace['Renshaw_to_MN_IPSP_integral']}")
    # RENSHAW CELLS
    rc_group_size = max(int(total_nb_renshaw_cells), 1)
    renshaw_cells = NeuronGroup(
        rc_group_size,
        RC_equations,
        threshold='v > voltage_thresh', 
        reset='v = voltage_rest',
        refractory='refractory_period_RC',
        method='euler',
        namespace=common_brian2_namespace
    )
    renshaw_cells.v = voltage_rest  # Initialize membrane potential
    renshaw_cells.tau = tau_Renshaw
    # ----------------- SYNAPSES ----------- 
    # Connect motor neurons to Renshaw cells
    synapses_MN_to_Renshaw = Synapses(motoneurons, renshaw_cells, 'w : 1',
                            on_pre='v += MN_to_Renshaw_excit*w',
                            delay = MN_RC_synpatic_delay,
                            namespace=common_brian2_namespace)
    pre_indices, post_indices = np.nonzero(MN_to_Renshaw_connectivity_matrix)
    weights_to_assign = MN_to_Renshaw_connectivity_matrix[pre_indices,post_indices]
    if int(total_nb_renshaw_cells) > 0 and len(pre_indices)>0 and len(post_indices)>0:
        synapses_MN_to_Renshaw.connect(i=pre_indices, j=post_indices)
        synapses_MN_to_Renshaw.w = weights_to_assign
    else:
        synapses_MN_to_Renshaw.active = False
    # Connect Renshaw cells to motor neurons
    # # ----------------- CHANGE Renshaw_to_MN_inhib IF IT IS USED AS A TARGET RELATIVE TO THE SYNAPTIC TIME CONSTANT
    if scale_initial_IPSP_to_be_same_integral_regardless_of_synaptic_tau:
        on_pre_action = 'I_syn -= (Renshaw_to_MN_IPSP_integral * w) / tau_syn'
        logger.info(f"      Defined IPSP is an integral over hyperpolarizing current (hyperpolarizing charge, in Coulomb) = {common_brian2_namespace['Renshaw_to_MN_IPSP_integral']}")
    else:
        on_pre_action = 'I_syn -= (Renshaw_to_MN_IPSP_integral * w) / (1*second)'   # Re-interpret the initial IPSP amplitude as an area over 1s
        logger.info(f"      Defined IPSP is the initial hyperpolarizing current induced by the Renshaw cell's IPSP (hyperpolarizing current, in Amp) = {common_brian2_namespace['Renshaw_to_MN_IPSP_integral']/second}")
    synapses_Renshaw_to_MN = Synapses(renshaw_cells, motoneurons, 
                                '''
                                w: 1
                                Renshaw_to_MN_IPSP_integral: amp*second     # the total ∫I(t)dt you want = total charge (Coulomb)
                                ''', 
                                on_pre=on_pre_action, # pick up each post‐cell's tau_syn
                                delay = MN_RC_synpatic_delay,
                                namespace=common_brian2_namespace)
    pre_indices, post_indices = np.nonzero(Renshaw_to_MNs_connectivity_matrix)
    weights_to_assign = Renshaw_to_MNs_connectivity_matrix[pre_indices, post_indices]
    if int(total_nb_renshaw_cells) > 0 and len(pre_indices)>0 and len(post_indices)>0:
        synapses_Renshaw_to_MN.connect(i=pre_indices, j=post_indices)
        synapses_Renshaw_to_MN.w = weights_to_assign
        synapses_Renshaw_to_MN.Renshaw_to_MN_IPSP_integral = common_brian2_namespace['Renshaw_to_MN_IPSP_integral']
    else:
        synapses_Renshaw_to_MN.active = False
        
    return motoneurons, renshaw_cells, synapses_MN_to_Renshaw, synapses_Renshaw_to_MN

# # # GET SPIKE TRAINS
def get_spike_trains(spike_monitor_MN, spike_monitor_RC,
    spike_transmission_delay, total_nb_motoneurons, total_nb_renshaws,
    edges_ignore_duration, duration_with_ignored_window, motoneurons_soma_diameters,
    generate_figure=False, savepath=None, task_event_times_s=None, burst_start_s=None, burst_center_s=None,
    show_task_cue_markers=True, show_step_current_marker=False):
    """
    Pull out and (optionally) plot the post‐delay spike trains from your monitors.

    Parameters
    ----------
    spike_monitor_MN : Brian2 SpikeMonitor recording the motoneurons
    spike_monitor_RC : Brian2 SpikeMonitor recording the Renshaw cells
    spike_transmission_delay : array_like, length total_nb_motoneurons
        delay (in seconds) to add to each MN spike train before returning.
    total_nb_motoneurons : int
        How many MNs in total you expect (so that even silent ones get an empty list).
    generate_figure : bool
    savepath : str or None
    """
    # 1) Extract raw trains from the monitors
    raw_MN = spike_monitor_MN.spike_trains()   # { neuron_index: array_of_times_in_s, ... }
    raw_RC = spike_monitor_RC.spike_trains()

    # helper to strip units only if needed
    def to_seconds(qt):
        # If it's a Brian Quantity, divide by second
        if isinstance(qt, Quantity):
            return np.asarray(qt/second, dtype=float)
        # Otherwise assume it's already a float array in seconds
        return np.asarray(qt, dtype=float)

    # build MN list
    spike_trains_MN = []
    for m in range(total_nb_motoneurons):
        qt = raw_MN.get(m, np.array([],float))
        times_s = to_seconds(qt)
        # add your float delays (in seconds)
        times_s = times_s + float(spike_transmission_delay[m])
        spike_trains_MN.append(times_s)

    # build RC list
    # max_r = max(raw_RC.keys())+1 if raw_RC else 0 # Get nb of Renshaw cells directly from the monitor
    spike_trains_RC = []
    for r in range(total_nb_renshaws):
        qt = raw_RC.get(r, np.array([],float))
        times_s = to_seconds(qt)
        spike_trains_RC.append(times_s)

    # (Optional) Plot them
    if generate_figure:
        fig, ax1 = plt.subplots(1, 1, figsize=(20, 6), sharex=True)
        t_min, t_max = 0.0, duration_with_ignored_window

        ax1.axvspan(t_min, edges_ignore_duration, color='grey', alpha=0.3)
        ax1.axvspan(t_max - edges_ignore_duration, t_max, color='grey', alpha=0.3)
        ax1.axvline(edges_ignore_duration, color='black', linestyle='--', linewidth=1)
        ax1.axvline(t_max - edges_ignore_duration, color='black', linestyle='--', linewidth=1)
        _draw_diagnostic_markers(
            ax1,
            task_event_times_s=task_event_times_s,
            burst_center_s=burst_center_s,
            burst_start_s=burst_start_s,
            show_task_cue_markers=show_task_cue_markers,
            show_step_current_marker=show_step_current_marker,
        )

        # prepare colormap for MN sizes
        cmap = plt.get_cmap('viridis')
        norm = matplotlib.colors.Normalize(
            vmin=motoneurons_soma_diameters.min(),
            vmax=motoneurons_soma_diameters.max()
        )
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # for the colorbar

        # panel 1: MN rasters, colored by soma diameter
        ax1.set_title("Motoneuron Spike Trains")
        for mni, times in enumerate(spike_trains_MN):
            c = cmap(norm(motoneurons_soma_diameters[mni]))
            ax1.eventplot(times, lineoffsets=mni, colors=[c], alpha=0.6)
        ax1.set_ylabel("MN index")
        cbar = fig.colorbar(sm, ax=ax1, pad=0.02)
        cbar.set_label("MN soma diameter (µm)")

        ax1.set_xlabel("Time (s)")

        plt.tight_layout()
        if savepath is not None:
            fn = os.path.join(savepath, "Spike_Trains.png")
            plt.savefig(fn)
        plt.show()

    return spike_trains_MN, spike_trains_RC

# # # SAVE OUTPUT AS HDF5 FILE WITH H5PY
def save_output_hdf5(directory_name,
                     params,                    # SimulationParameters dataclass
                     motoneurons_and_pools_idx, # dict of dicts, see generate_motor_neurons() (saving {pool_list_by_MN, idx_of_MN_by_pool})
                     motoneurons_properties,    # dict of np.ndarrays
                     connectivity_matrix_MN_to_RC,
                     connectivity_matrix_RC_to_MN,
                     connectivity_matrix_MN_to_MN,
                     spike_trains_MN,           # list of 1D np.ndarrays
                     spike_trains_RC,           # list of 1D np.ndarrays
                     common_input_MN,           # dict of 1D np.arrays (one per pool)
                     common_input_power_total, common_input_power_spectrum, # floats
                     independent_input_power_total=None, independent_input_power_spectrum=None, # dict with keys "frequencies" and "power"  
                     forces=None,
                     driving_inputs=None,
                     input_components=None,
                     cst_diagnostics=None,
                     sync_diagnostics=None,
                     firing_rate_diagnostics=None,
                     step_response_diagnostics=None,
                     analysis_summary_metadata=None,
                     observation_features=None,
                     save_spike_trains=True,
                     saved_trace_fsamp_hz=None,
                     sync_saved_fsamp_hz=None): # Inputs to save (useful when input is iteratively optimized to match force target)            
    
    output_file = os.path.join(directory_name, "simulation_output.h5")
    with h5py.File(output_file, "w") as f:
        # 1) simulation parameters
        sim_grp = f.create_group("simulation_parameters")
        param_dict = asdict(params)
        param_dict.pop('RC_pair_indices', None) # Remove this value from the parameter list (it's a tuple and not readily writable)
        for k, v in param_dict.items():
            # numpy arrays → datasets
            if isinstance(v, np.ndarray):
                sim_grp.create_dataset(k, data=v)
            # Brian2 quantities → two entries: numeric + unit
            elif isinstance(v, Quantity):
                sim_grp.create_dataset(k + "_value", data=v.magnitude)
                sim_grp.attrs[k + "_unit"] = str(v.units)
            # “simple” scalars OK as attrs
            elif isinstance(v, (int, float, bool, str)):
                sim_grp.attrs[k] = v
            # anything else (lists, tuples, dicts, etc.) → JSON‐dumped string
            else:
                sim_grp.attrs[k] = json.dumps(v)
        
        # 2) motoneuron and pool indices
        mn_pool_grp = f.create_group("motoneurons_and_pools_indices")
        # (a) pool_list_by_MN -- a string array
        pool_list = motoneurons_and_pools_idx["pool_list_by_MN"]
        # make a numpy array of dtype "variable‐length UTF‐8 string"
        str_dt = string_dtype(encoding="utf-8")
        mn_pool_grp.create_dataset(
            "pool_list_by_MN",
            data=np.array(pool_list, dtype=str_dt),
            dtype=str_dt)
        # (b) idx_of_MN_by_pool -- subgroup of integer arrays
        by_pool_grp = mn_pool_grp.create_group("idx_of_MN_by_pool")
        for poolname, idx_array in motoneurons_and_pools_idx["idx_of_MN_by_pool"].items():
            # poolname is something like "pool_0", "pool_1", ...
            by_pool_grp.create_dataset(poolname, data=idx_array.astype(int))
        # # Example to read back:
        # with h5py.File(..., "r") as f:
        #     grp = f["motoneurons_and_pools_indices"]
        #     pool_list = grp["pool_list_by_MN"][()]        # array of bytes→ decode to str if you like
        #     by_pool   = grp["idx_of_MN_by_pool"]
        #     idx0      = by_pool["pool_0"][()]             # MN indices in pool_0

        # 3) motoneuron properties
        mn_grp = f.create_group("motoneurons_properties")
        for k, arr in motoneurons_properties.items():
            mn_grp.create_dataset(k, data=arr)

        # 2) connectivity
        conn = f.create_group("connectivity")
        conn.create_dataset("MN_to_RC", data=connectivity_matrix_MN_to_RC)
        conn.create_dataset("RC_to_MN", data=connectivity_matrix_RC_to_MN)
        conn.create_dataset("MN_to_MN", data=connectivity_matrix_MN_to_MN)

        # 5) spike trains
        if save_spike_trains:
            spikes = f.create_group("spike_trains")
            mn_spikes = spikes.create_group("MN")
            for i, tr in enumerate(spike_trains_MN):
                mn_spikes.create_dataset(f"MN_{i}", data=tr)
            rc_spikes = spikes.create_group("RC")
            for i, tr in enumerate(spike_trains_RC):
                rc_spikes.create_dataset(f"RC_{i}", data=tr)

        # 6) Synaptic input
        input_grp = f.create_group("input")
        if saved_trace_fsamp_hz is None:
            saved_trace_fsamp_hz = float(params.fsamp)
        input_grp.attrs["saved_fsamp_Hz"] = float(saved_trace_fsamp_hz)
        # Common input
        common_input_list = []
        for pooli in sorted(common_input_MN):
            common_input_list.append(common_input_MN[pooli]) # now this is a 2D numeric array: (n_pools, n_timepoints)
        common_input_array = np.stack(common_input_list, axis=0)
        input_grp.create_dataset("common_input", data=common_input_array)
        # Power spectrums
        #       # Total
        if independent_input_power_total is None:
            independent_input_power_total = np.array([np.nan])
        if independent_input_power_spectrum is None:
            independent_input_power_spectrum = {
                "frequencies": np.array([np.nan]),
                "power": np.array([np.nan])
            }
        input_grp.attrs["total_power_common_input"] = common_input_power_total[0] # only from first pool
        input_grp.attrs["total_power_independent_input"] = independent_input_power_total
        #       # Power spectrum (per frequency)
        for input_type_i in ["common_input", "independent_input"]:
            if input_type_i == "common_input":
                power_spectrum_to_use = common_input_power_spectrum
                input_grp.create_dataset("frequencies", data=power_spectrum_to_use["frequencies"])
            else: # if input_type_i == "independent_input"
                power_spectrum_to_use = independent_input_power_spectrum
            input_grp.create_dataset(f"power_spectrum_{input_type_i}", data=power_spectrum_to_use["power"]) # only the power spectrum of the first common input is saved (for the independent input, it is the average over all MNs)

        if input_components:
            comp_grp = f.create_group("input_components")
            comp_grp.attrs["saved_fsamp_Hz"] = float(saved_trace_fsamp_hz)
            keys_to_save = [
                "lf_final",
                "lf_final_with_trend",
                "lf_final_with_burst",
                "lf_burst_kernel_nA",
                "lf_trend_kernel_nA",
                "lf_effective_envelope_nA",
                "alpha_final",
                "alpha_final_with_trend",
                "alpha_final_with_burst",
                "alpha_env_multiplier",
                "alpha_env_multiplier_with_burst",
                "alpha_env_multiplier_full_variability",
                "alpha_burst_kernel_nA",
                "alpha_trend_kernel_nA",
                "alpha_effective_envelope_nA",
                "alpha_env_variability_scale",
                "beta_final",
                "beta_final_with_trend",
                "beta_final_with_burst",
                "beta_env_multiplier",
                "beta_env_multiplier_with_burst",
                "beta_env_multiplier_full_variability",
                "beta_burst_kernel_nA",
                "beta_trend_kernel_nA",
                "beta_effective_envelope_nA",
                "beta_env_variability_scale",
                "common_input_no_burst",
                "common_input_with_burst",
                "common_input_with_step",
                "common_input_trend_trace_nA",
                "common_input_burst_trace_nA",
                "step_current_trace_nA",
                "baseline_tonic_trace_nA",
                "baseline_plus_step_trace_nA",
                "total_pool_input_with_step_trace_nA",
                "alpha_beta_envelope_variability_scale",
            ]
            for key in keys_to_save:
                if key in input_components:
                    comp_grp.create_dataset(key, data=np.asarray(input_components[key], dtype=float))

        if cst_diagnostics:
            cst_grp = f.create_group("cst_diagnostics")
            cst_grp.attrs["saved_fsamp_Hz"] = float(saved_trace_fsamp_hz)
            for key, value in cst_diagnostics.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    cst_grp.attrs[key] = value
                else:
                    cst_grp.create_dataset(key, data=np.asarray(value))

        if sync_diagnostics:
            sync_grp = f.create_group("sync_diagnostics")
            if sync_saved_fsamp_hz is not None:
                sync_grp.attrs["saved_fsamp_Hz"] = float(sync_saved_fsamp_hz)
            for key, value in sync_diagnostics.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    sync_grp.attrs[key] = value
                else:
                    sync_grp.create_dataset(key, data=np.asarray(value))

        if firing_rate_diagnostics:
            fr_diag_grp = f.create_group("firing_rate_diagnostics")
            for key, value in firing_rate_diagnostics.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    fr_diag_grp.attrs[key] = value
                else:
                    fr_diag_grp.create_dataset(key, data=np.asarray(value))

        if step_response_diagnostics:
            step_grp = f.create_group("step_response_diagnostics")
            for key, value in step_response_diagnostics.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    step_grp.attrs[key] = value
                else:
                    step_grp.create_dataset(key, data=np.asarray(value))

        if analysis_summary_metadata is not None or observation_features is not None:
            analysis_grp = f.create_group("analysis")
            if analysis_summary_metadata is not None:
                summary_grp = analysis_grp.create_group("summary_metadata")
                for key, value in dict(analysis_summary_metadata).items():
                    if value is None:
                        continue
                    if isinstance(value, str):
                        summary_grp.attrs[key] = value
                    else:
                        summary_grp.create_dataset(key, data=np.asarray(value))
            if observation_features is not None:
                obs_grp = analysis_grp.create_group("observation_features")
                for key, value in dict(observation_features).items():
                    if value is None:
                        continue
                    obs_grp.create_dataset(key, data=np.asarray([value], dtype=float))

        # 7) forces
        if forces is not None:
            fr = f.create_group("forces")
            fr.attrs["fsamp_Hz"] = float(saved_trace_fsamp_hz)

            # --- keep legacy datasets (total) so old code keeps working
            fr.create_dataset("pool_max_tetanic_force", data=forces["pool_max_tetanic_force"])
            fr.create_dataset("pool_force", data=forces["pool_force"])
            fr.create_dataset("pool_force_percent", data=forces["pool_force_percent"])

            # --- NEW: per-pool datasets
            per_pool = forces.get("per_pool", {})
            # names as a variable-length UTF-8 string array
            if per_pool:
                str_dt = string_dtype(encoding="utf-8")
                fr.create_dataset("per_pool_names",
                                  data=np.array(per_pool["names"], dtype=str_dt),
                                  dtype=str_dt)
                fr.create_dataset("per_pool_max_tetanic_force", data=per_pool["max_tetanic"])
                fr.create_dataset("per_pool_force", data=per_pool["force"])
                fr.create_dataset("per_pool_force_percent", data=per_pool["force_percent"])

            if "mu_force" in forces:
                fr.create_dataset("mu_force", data=forces["mu_force"])


        # 8) driving inputs used in the simulation and for optimization
        g = f.create_group("driving_inputs")
        g.attrs["saved_fsamp_Hz"] = float(saved_trace_fsamp_hz)
        for k, v in driving_inputs.items():
            if v is not None:
                g.create_dataset(k, data=np.asarray(v))
            else:
                g.create_dataset(k, data=np.array([np.nan]))

    return output_file


def compute_step_sync_diagnostics(params, spike_trains_MN, active_unit_selection, logger=None):
    """Compute a step-current-referenced synchrony trace without touching Go/No-Go sync diagnostics."""
    if not (
        bool(getattr(params, "enable_step_current", False))
        and bool(getattr(params, "enable_step_sync_index_analysis", False))
    ):
        return None
    logger = logger or logging.getLogger(__name__)
    try:
        from analyzer_with_force import run_sliding_sync_index_analysis as _run_sync_index_analysis

        step_abs_s = float(params.task_event_times_s["step_current_s"])
        sync_active_ids = np.asarray(active_unit_selection["active_unit_ids"], dtype=int)
        sync_spike_trains = [spike_trains_MN[int(unit_i)] for unit_i in sync_active_ids]
        if len(sync_spike_trains) < 2:
            raise ValueError("At least two active units are required for step synchrony analysis")
        analysis_window_rel_s = _normalize_rel_window(params.step_sync_analysis_window_rel_s)
        analysis_start_s = max(0.0, float(step_abs_s + analysis_window_rel_s[0]))
        analysis_end_s = min(float(params.duration_with_ignored_window), float(step_abs_s + analysis_window_rel_s[1]))
        if analysis_end_s <= analysis_start_s:
            raise ValueError("step_sync_analysis_window_rel_s gives an empty analysis window")
        out = _run_sync_index_analysis(
            sync_spike_trains,
            cue_time_sec=step_abs_s,
            baseline_win=params.step_pre_window_rel_s,
            post_win=params.step_late_post_window_rel_s,
            generate_figure=False,
            display_mode=params.sync_trace_display_mode,
            bin_ms=params.sync_bin_ms,
            sync_win_ms=params.sync_win_ms,
            sync_step_ms=params.sync_step_ms,
            coinc_lag_ms=params.sync_coinc_lag_ms,
            direction_mode=params.sync_direction_mode,
            expectation_mode=params.sync_expectation_mode,
            n_surrogates=params.sync_n_surrogates,
            surrogate_min_shift_ms=params.sync_surrogate_min_shift_ms,
            duration_s=float(params.duration_with_ignored_window),
            analysis_start_s=analysis_start_s,
            analysis_end_s=analysis_end_s,
            unit_ids=sync_active_ids,
        )
        out["sync_reference_event"] = "step_current"
        out["cue_time_sec"] = float(step_abs_s)
        out["t_sync_absolute_sec"] = float(step_abs_s) + np.asarray(out.get("t_sync", []), dtype=float)
        out["step_sync_analysis_window_rel_s"] = np.asarray(analysis_window_rel_s, dtype=float)
        return out
    except Exception as exc:
        logger.warning(f"Could not compute step-referenced sync_index diagnostics: {exc}")
        return None


######################################
### RUN SIMULATION
######################################

def run_simulation(params=None):
    """
    Runs one simulation.  `params` may be:
      • None                             → use every default
      • a SimulationParameters object    → used directly
    """
    # # # SET PARAMETERS
    if params is None:
        params = SimulationParameters()
    elif isinstance(params, SimulationParameters):
        params = params
    else:
        raise ValueError("run_simulation() expects a SimulationParameters object, or None")
    nb_pools = 1
    total_nb_renshaw_cells = 0
    nb_motoneurons_per_pool = params.nb_motoneurons
    baseline_by_pool = _baseline_by_pool_nA(params)
    figures_enabled = _figures_enabled(params)
    go_nogo_cue_diagnostics_enabled = bool(getattr(params, "enable_go_nogo_cue_diagnostics", True))

    # Create new folder and get simulation index number
    if params.make_unique_output_folder:
        directory_name, sim_index = make_unique_output_dir(parent_folder=params.output_folder_name)
    else:
        directory_name = params.output_folder_name
        sim_index = os.path.basename(os.path.normpath(directory_name))
    # Initialize
    _ensure_logging() # ensure logging is configured for _this_ process
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing simulation {sim_index}...")
    _set_filter_params(params)
    prefs.codegen.target = params.brian_codegen_target
    start_scope()  # Re-initialize Brian
    start_time = time.time()
    np.random.seed(params.random_seed)
    # write JSON with all parameters
    param_dict = asdict(params)
    param_dict.pop('RC_pair_indices', None) # Remove this value from the parameter list (it's a tuple and not readily writable in a json file)
    with open(f"{directory_name}/sim_parameters.json","w") as fp:
        json.dump(param_dict, fp, indent=2, default=str)

    # # # SET EQUATIONS
    (MN_equations, RC_equations) = set_brian2_equations()

    # # # GENERATE MOTOR NEURONS
    (motoneuron_soma_diameters, motoneuron_normalized_soma_diameters,
     pool_list_by_MN, idx_of_MN_by_pool) = generate_motor_neurons(
            min_soma_diameter=params.min_soma_diameter, max_soma_diameter=params.max_soma_diameter,
            nb_pools=nb_pools, nb_motoneurons_per_pool=nb_motoneurons_per_pool, total_nb_motoneurons=params.total_nb_motoneurons,
            tau_size_mode=params.tau_size_mode, tau_size_ratio=params.tau_size_ratio, tau_size_um=params.tau_size_um,
            generate_figure=figures_enabled and params.output_plot_mn_size_figure, savepath=directory_name)
    
    # # # GENERATE MOTOR NEURONS ELECTROPHYSIOLOGICAL PROPERTIES
    (motoneurons_resistance, motoneurons_input_weight,
     motoneurons_capacitance, motoneurons_membrane_conductance,
     motoneurons_membrane_time_constant,
     motoneurons_AHP_conductance_decay_time_constant, motoneurons_refractory_periods,
     motoneurons_rheobases, motoneurons_spike_transmission_delays,
     motoneurons_properties_dict) = generate_motor_neuron_electrophysiological_properties(
        total_nb_motoneurons=params.total_nb_motoneurons, motoneuron_soma_diameters=motoneuron_soma_diameters,
        resistance_constant=params.resistance_constant, resistance_exponent=params.resistance_exponent,
        capacitance_constant=params.capacitance_constant, capacitance_exponent=params.capacitance_exponent,
        AHP_duration_constant=params.AHP_duration_constant, AHP_duration_exponent=params.AHP_duration_exponent,
        rheobase_constant=params.rheobase_constant, rheobase_exponent=params.rheobase_exponent, rheobase_scaling=params.rheobase_scaling,
        refractory_period_absolute=params.refractory_period_absolute,
        axonal_conduction_velocity_constant=params.axonal_conduction_velocity_constant, axonal_conduction_velocity_exponent=params.axonal_conduction_velocity_exponent,
        generate_figure=figures_enabled and params.output_plot_mn_properties_figure, savepath=directory_name)
    
    # # # GENERATE AND DISTRIBUTE COMMON INPUT
    (MN_excit_input, corr_mat, inputs_to_mn_weight_matrix, total_power, common_input_reference_power,
     power_per_frequency_band, input_burst_info, input_components) = generate_and_distribute_common_inputs(
        fsamp=params.fsamp,
        nb_pools=nb_pools, nb_motoneurons_per_pool=nb_motoneurons_per_pool,
        duration_with_ignored_window=params.duration_with_ignored_window, edges_ignore_duration=params.edges_ignore_duration,
        lf_target_sd=params.lf_target_sd,
        lf_band_hz=params.lf_band_hz,
        alpha_target_sd_final=params.alpha_target_sd_final,
        alpha_band_hz=params.alpha_band_hz,
        alpha_envelope_band_hz=params.alpha_envelope_band_hz,
        beta_target_sd_final=params.beta_target_sd_final,
        beta_band_hz=params.beta_band_hz,
        beta_envelope_band_hz=params.beta_envelope_band_hz,
        alpha_beta_envelope_variability_scale=params.alpha_beta_envelope_variability_scale,
        common_input_weight_sd=params.common_input_weight_sd,
        common_input_weight_min=params.common_input_weight_min,
        common_input_weight_max=params.common_input_weight_max,
        lf_trend_start_rel_go_cue_s=params.lf_trend_start_rel_go_cue_s,
        alpha_trend_start_rel_go_cue_s=params.alpha_trend_start_rel_go_cue_s,
        beta_trend_start_rel_go_cue_s=params.beta_trend_start_rel_go_cue_s,
        lf_trend_slope_nA_per_s=params.lf_trend_slope_nA_per_s,
        alpha_trend_slope_nA_per_s=params.alpha_trend_slope_nA_per_s,
        beta_trend_slope_nA_per_s=params.beta_trend_slope_nA_per_s,
        enable_input_burst=params.enable_input_burst,
        input_burst_center_ms=params.input_burst_center_ms,
        input_burst_sigma_ms=params.input_burst_sigma_ms,
        lf_burst_peak_nA=params.lf_burst_peak_nA,
        alpha_burst_peak_nA=params.alpha_burst_peak_nA,
        beta_burst_peak_nA=params.beta_burst_peak_nA,
        input_burst_start_marker_n_sigma=params.input_burst_start_marker_n_sigma,
        generate_figure=figures_enabled and params.output_plot_common_input_weight_figure,
        savepath=directory_name,
        save_power_csv=params.save_common_input_power_csv)
    if "common_input_modulation_weights" in input_components:
        motoneurons_properties_dict["common_input_modulation_weight"] = np.asarray(
            input_components["common_input_modulation_weights"],
            dtype=float,
        )
    if "independent_input_weights" in input_components:
        motoneurons_properties_dict["independent_input_weight"] = np.asarray(
            input_components["independent_input_weights"],
            dtype=float,
        )
    
    # # # CREATE BASELINE TIMESERIES FOR EXCITATORY INPUT AND FORCE TARGET
    (baseline_ts_nA, target_percent_ts) = generate_timevarying_baseline_from_target_main(params=params, motoneurons_properties_dict=motoneurons_properties_dict)
    expected_n_samples = len(MN_excit_input[0])
    target_percent_ts = np.asarray(target_percent_ts, dtype=float)[:expected_n_samples]
    if baseline_ts_nA is not None:
        baseline_ts_nA = np.asarray(baseline_ts_nA, dtype=float)[:expected_n_samples]
    step_current_info = build_step_current_trace(params, expected_n_samples)
    step_current_trace_nA = np.asarray(step_current_info["step_current_trace_nA"], dtype=float)
    initial_baseline_tonic_trace_nA = _baseline_trace_for_step(params, baseline_ts_nA, expected_n_samples)
    input_components["step_current_trace_nA"] = step_current_trace_nA
    input_components["baseline_tonic_trace_nA"] = initial_baseline_tonic_trace_nA.copy()
    input_components["baseline_plus_step_trace_nA"] = initial_baseline_tonic_trace_nA + step_current_trace_nA
    input_components["common_input_with_step"] = np.asarray(input_components["common_input_with_burst"], dtype=float) + step_current_trace_nA
    input_components["total_pool_input_with_step_trace_nA"] = (
        np.asarray(input_components["common_input_with_burst"], dtype=float)
        + input_components["baseline_plus_step_trace_nA"]
    )
    
    # Recurrent inhibition has been removed from this simplified branch.
    MN_to_MN_connectivity_matrix = np.zeros((params.total_nb_motoneurons, params.total_nb_motoneurons), dtype=float)
    MN_to_Renshaw_connectivity_matrix = np.zeros((params.total_nb_motoneurons, 0), dtype=float)
    Renshaw_to_MNs_connectivity_matrix = np.zeros((0, params.total_nb_motoneurons), dtype=float)
    synaptic_IPSP_decay_time_constant_per_MN = motoneurons_membrane_time_constant * ms

    # # # THE REST OF THE INITIALIZATION IS WRAPPED INTO THE FUNCTION BELOW TO ALLOW RE-INSTANTIATION FOR THE OPTIMIZATION PROCEDURE
    def simulate_once(baseline_ts_nA, iter_label=0, make_plots=False):
        """
        Build time arrays (using *the same* MN_excit_input and weights),
        create groups & synapses, run Brian2 once, return pool force %.
        """
        # Keep the noise realization fixed only across baseline-optimization
        # iterations. Ordinary batch simulations should keep the per-run seed
        # drawn at initialization so distinct simulations stay distinct.
        if params.optimize_baseline and params.opt_fix_noise_seed is not None:
            np.random.seed(int(params.opt_fix_noise_seed))

        # Important: reset Brian2 state for a fresh run
        start_scope()
        rc_group_size_for_sim = max(total_nb_renshaw_cells, 1)

        baseline_tonic_trace_nA = _baseline_trace_for_step(params, baseline_ts_nA, expected_n_samples)
        baseline_plus_step_trace_nA = baseline_tonic_trace_nA + step_current_trace_nA
        # # # ADD INDEPENDENT INPUTS AND CREATE Brian2 TIMED ARRAY OBJECTS
        # --- TimedArrays (MN and RC) using *this* baseline_ts_nA ---
        (
            input_MN_timedarray_amp,
            input_RC_timedarray_volt,
            total_power_independent_iter,
            power_per_frequency_band_independent_iter,
            independent_input_payload_iter,
        ) = generate_independent_inputs(
            fsamp=params.fsamp,
            nb_pools=nb_pools, total_nb_motoneurons=params.total_nb_motoneurons, total_nb_renshaw_cells=rc_group_size_for_sim,
            low_pass_filter_of_MN_independent_input=params.low_pass_filter_of_MN_independent_input,
            low_pass_filter_of_RC_independent_input=50,
            ref_common_input_power=common_input_reference_power,    # full-band no-burst common-input power
            MN_independent_input_absolute_or_ratio=params.independent_input_absolute_or_ratio,
            MN_independent_input_power=params.independent_input_power,
            independent_input_weight_sd=params.independent_input_weight_sd,
            independent_input_weight_max=params.independent_input_weight_max,
            RC_independent_input_std=0.0,
            MN_excit_input=MN_excit_input,
            inputs_to_mn_weight_matrix=inputs_to_mn_weight_matrix,
            excitatory_input_baseline=baseline_by_pool,
            duration_with_ignored_window=params.duration_with_ignored_window,
            edges_ignore_duration=params.edges_ignore_duration,
            motoneurons_rheobases=motoneurons_rheobases,
            generate_figure=False,
            generate_weight_figure=figures_enabled and params.output_plot_independent_input_weight_figure,
            savepath=directory_name,
            save_power_csv=params.save_independent_input_power_csv,
            excitatory_input_baseline_timeseries=baseline_plus_step_trace_nA
        )
        if independent_input_payload_iter:
            input_components["independent_input_weights"] = np.asarray(independent_input_payload_iter["weights"], dtype=float)
            input_components["independent_input_weight_mu_ln"] = np.asarray([independent_input_payload_iter["mu_ln"]], dtype=float)
            input_components["independent_input_weight_sigma_ln"] = np.asarray([independent_input_payload_iter["sigma_ln"]], dtype=float)
            input_components["independent_input_weight_target_sd"] = np.asarray([independent_input_payload_iter["target_sd"]], dtype=float)
            input_components["independent_input_weight_max"] = np.asarray([independent_input_payload_iter["max_weight"]], dtype=float)
        # # # IF DESIRED, DISPLAY POWER OF COMMON AND INDEPENDENT INPUTS
        if figures_enabled and params.output_plot_common_vs_independent_power_figure:
            plt.figure(figsize=(10,6))
            scaling_power = 1/1e6
            plt.fill_between(
                    power_per_frequency_band_independent_iter['frequencies'],
                    power_per_frequency_band_independent_iter['power']*scaling_power,
                    y2=0, color='blue', alpha=0.3)
            plt.fill_between(
                    power_per_frequency_band['frequencies'],
                    power_per_frequency_band['power']*scaling_power,
                    y2=0, color='red', alpha=0.3)
            plt.plot(power_per_frequency_band['frequencies'],
                    power_per_frequency_band['power']*scaling_power,
                    color='red', label=f'Common input power\ntotal={total_power[0]*scaling_power:.2f} a.u.')
            plt.plot(power_per_frequency_band_independent_iter['frequencies'],
                power_per_frequency_band_independent_iter['power']*scaling_power,
                color='blue', label=f'Independent input power\ntotal={total_power_independent_iter*scaling_power:.2f} a.u.')
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power (a.u.)")
            plt.xlim(0, 100)
            plt.legend(loc="upper right")
            plt.savefig(f"{directory_name}/Power_of_common_and_independent_inputs.png")

        # # # CREATE Brian2 NEURONGROUPS AND SYNAPSES OBJECTS
        # set synaptic_IPSP_decay_time_constant_per_MN
        # --- Groups & synapses (reuse your already-built arrays & equations) ---
        (motoneurons, renshaw_cells,
        synapses_MN_to_Renshaw, synapses_Renshaw_to_MN) = create_neurongroups_and_synapses_objects(
            total_nb_motoneurons=params.total_nb_motoneurons, total_nb_renshaw_cells=rc_group_size_for_sim,
            MN_equations=MN_equations, RC_equations=RC_equations, voltage_rest=params.voltage_rest, voltage_thresh=params.voltage_thresh,
            motoneurons_membrane_conductance=motoneurons_membrane_conductance, motoneurons_capacitance=motoneurons_capacitance,
            motoneurons_input_weight=motoneurons_input_weight, synaptic_IPSP_decay_time_constant_per_MN=synaptic_IPSP_decay_time_constant_per_MN,
            # AHP parameters
            motoneurons_AHP_conductance_decay_time_constant=motoneurons_AHP_conductance_decay_time_constant,
            AHP_slow_conductance_decay_time_constant_multiplier=params.AHP_slow_conductance_decay_time_constant_multiplier,
            AHP_fast_conductance_delta_after_spiking=params.AHP_fast_conductance_delta_after_spiking,
            AHP_slow_conductance_delta_after_spiking=params.AHP_slow_conductance_delta_after_spiking,
            maximum_AHP_conductance_slow_conductance=params.maximum_AHP_conductance_slow_conductance,
            # 
            motoneurons_refractory_periods=motoneurons_refractory_periods,
            MN_to_Renshaw_excit=0.0*mvolt, Renshaw_to_MN_inhib=0.0*nA,
            scale_initial_IPSP_to_be_same_integral_regardless_of_synaptic_tau=False,
            tau_Renshaw=8*ms, MN_RC_synpatic_delay=0*ms, refractory_period_RC=5*ms,
            MN_to_Renshaw_connectivity_matrix=MN_to_Renshaw_connectivity_matrix,
            Renshaw_to_MNs_connectivity_matrix=Renshaw_to_MNs_connectivity_matrix
        )

        # --- Run ---
        motoneurons.v = params.voltage_rest
        renshaw_cells.v = params.voltage_rest
        monitor_spikes_motoneurons = SpikeMonitor(motoneurons, record=True)
        monitor_spikes_renshaw_cells = SpikeMonitor(renshaw_cells, record=True)

        logger.info(f"Starting simulation iter {iter_label} (T={params.duration_with_ignored_window:.1f}s)")
        run(params.duration_with_ignored_window * second)
        logger.info(f"...iteration {iter_label} finished.")

        # --- Extract spikes ---
        spike_trains_MN, spike_trains_RC = get_spike_trains(
            spike_monitor_MN=monitor_spikes_motoneurons, spike_monitor_RC=monitor_spikes_renshaw_cells, 
            spike_transmission_delay=motoneurons_spike_transmission_delays, 
            total_nb_motoneurons=params.total_nb_motoneurons, total_nb_renshaws=total_nb_renshaw_cells,
            edges_ignore_duration=params.edges_ignore_duration,
            duration_with_ignored_window=params.duration_with_ignored_window,
            motoneurons_soma_diameters=motoneuron_soma_diameters,
            generate_figure=figures_enabled and params.output_plot_spike_trains_figure,
            savepath=directory_name,
            task_event_times_s=params.task_event_times_s,
            burst_start_s=params.task_event_times_s.get("burst_center_s", 0.0) - params.input_burst_start_marker_n_sigma * (params.input_burst_sigma_ms / 1000.0) if params.enable_input_burst else None,
            burst_center_s=params.task_event_times_s.get("burst_center_s") if params.enable_input_burst else None,
            show_task_cue_markers=bool(params.enable_input_burst or params.spike_train_show_task_cues_when_no_burst),
            show_step_current_marker=bool(params.enable_step_current and params.spike_train_show_step_current_marker),
        )

        # --- Forces (post-hoc) ---
        # Compute total & per-pool forces
        if params.enable_force_model:
            # 1) per-MU force parameters from soma diameter
            mu_force_params = assign_motor_unit_force_params(
                params,
                motoneurons_properties_dict["soma_diameter"]  # (µm)
            )
            # persist MU params alongside other properties
            motoneurons_properties_dict["tetanic_max_force"] = mu_force_params["tetanic_max_force"]
            motoneurons_properties_dict["twitch_peak_force_relative"] = mu_force_params["twitch_peak_force_relative"]
            motoneurons_properties_dict["twitch_time"] = mu_force_params["twitch_time"]

            # 2) compute total + per-pool forces
            forces = compute_mu_and_pool_force_traces(
                spike_trains_MN=spike_trains_MN,
                mu_force_params=mu_force_params,
                fsamp=params.fsamp,
                duration_s=params.duration_with_ignored_window,
                kernel_duration_factor=params.force_kernel_duration_factor,
                save_per_mu=params.save_per_mu_forces,
                idx_of_MN_by_pool=idx_of_MN_by_pool,   # <-- NEW
                nb_pools=nb_pools                # <-- NEW
            )

            out = {
                "spike_trains_MN": spike_trains_MN,
                "spike_trains_RC": spike_trains_RC,
                # keep the same key as before so the rest of your code continues to work
                "pool_force_percent": forces["pool_force_percent"],
                "forces": forces,
            }
        else:
            out = {
                "spike_trains_MN": spike_trains_MN,
                "spike_trains_RC": spike_trains_RC,
                "pool_force_percent": np.zeros(int(params.fsamp * params.duration_with_ignored_window)),
                "forces": dict(
                    pool_max_tetanic_force=0.0,
                    pool_force=np.zeros(int(params.fsamp * params.duration_with_ignored_window)),
                    pool_force_percent=np.zeros(int(params.fsamp * params.duration_with_ignored_window)),
                    per_pool=dict(names=[], max_tetanic=np.array([]),
                                  force=np.empty((0, int(params.fsamp * params.duration_with_ignored_window))),
                                  force_percent=np.empty((0, int(params.fsamp * params.duration_with_ignored_window)))))
            }
        return out
    ### END OF "simulate_once()" FUNCTION

    # ------------- OPTIMIZATION (or single run) -------------
    # Build a mask to exclude ignored edges from RMSE
    def _core_mask(n_samples, fs, edges):
        T = int(n_samples); E = int(round(edges * fs))
        m = np.zeros(T, bool); 
        if T > 2*E: m[E:T-E] = True
        return m
    core_mask = _core_mask(len(target_percent_ts), params.fsamp, params.edges_ignore_duration)

    # Keep copies for diagnostics
    if baseline_ts_nA is not None: # avoid errors later
        baseline_initial = baseline_ts_nA.copy()
    else:
        baseline_initial = np.zeros(int(params.fsamp * params.duration_with_ignored_window))
    rmse_history = []
    force_initial = None
    force_final   = None
    final_forces_dict = None

    def _rmse_core(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        n = min(a.size, b.size, core_mask.size)
        e = a[:n] - b[:n]
        e = e[core_mask[:n]]
        return float(np.sqrt(np.mean(e*e)))

    if params.optimize_baseline:
        # ---- Iter 0: initial baseline
        logger.info("   Optimizing input to match force target - iter 0 (first guess)")
        out0 = simulate_once(baseline_ts_nA, iter_label=0)
        force0 = out0["pool_force_percent"]
        rmse0 = _rmse_core(force0, target_percent_ts)
        rmse_history.append(rmse0)

        # ---- Sensitivity calibration: +ΔnA everywhere
        delta = float(params.opt_sensitivity_calib_delta_nA)
        out_cal = simulate_once(baseline_ts_nA + delta, iter_label=0)
        slope_percent_per_nA = np.clip(
            np.mean(out_cal["pool_force_percent"][core_mask] - force0[core_mask]) / max(delta, 1e-9),
            1e-4, 1e6
        )
        logger.info(f"Calibrated d(%MVC)/d(nA) ≈ {slope_percent_per_nA:.3f}")

        # ---- Adam loop
        m = np.zeros_like(baseline_ts_nA)  # Adam moments
        v = np.zeros_like(baseline_ts_nA)
        b1, b2, eps = 0.9, 0.999, 1e-8
        t_adam = 0

        best_rmse = rmse0
        best_baseline = baseline_ts_nA.copy()
        patience = 0

        for it in range(1, params.opt_max_iters + 1):
            logger.info(f"   Optimizing input to match force target - iter {it}")
            out = simulate_once(baseline_ts_nA, iter_label=it)
            force = out["pool_force_percent"]
            rmse  = _rmse_core(force, target_percent_ts)
            rmse_history.append(rmse)
            logger.info(f"[iter {it:02d}] RMSE (core) = {rmse:.3f} %MVC")

            # gradient approximation: e * slope
            grad = (force - target_percent_ts) * slope_percent_per_nA

            # optional smoothing of grad in time (in seconds)
            if getattr(params, "opt_gradient_smoothing_sigma_s", 0.0) and params.opt_gradient_smoothing_sigma_s > 0:
                sig = params.opt_gradient_smoothing_sigma_s * params.fsamp
                from scipy.ndimage import gaussian_filter1d
                grad = gaussian_filter1d(grad, sigma=sig, mode="nearest")

            # Adam update
            t_adam += 1
            m = b1*m + (1-b1)*grad
            v = b2*v + (1-b2)*(grad*grad)
            mhat = m / (1 - b1**t_adam)
            vhat = v / (1 - b2**t_adam)
            step = - params.opt_learning_rate * mhat / (np.sqrt(vhat) + eps)

            baseline_ts_nA = baseline_ts_nA + step

            # early stopping bookkeeping
            improve_pct = 100.0 * (best_rmse - rmse) / max(best_rmse, 1e-9)
            if rmse < best_rmse - 1e-9:
                best_rmse = rmse
                best_baseline = baseline_ts_nA.copy()
                patience = 0
            else:
                patience += 1

            # require minimum improvement size
            if improve_pct < params.opt_rmse_improve_threshold_percent:
                # didn’t improve enough this iter → counts toward patience
                pass
            else:
                patience = 0

            if patience >= params.opt_earlystop_patience:
                logger.info(f"Early stopping at iter {it} (ΔRMSE < {params.opt_rmse_improve_threshold_percent:.1f}% for {params.opt_earlystop_patience} iters).")
                break

        # Finalize with best baseline
        baseline_ts_nA = best_baseline.copy()
        logger.info(f"   Optimizing input to match force target - final iteration (repeat best from history)")
        outF = simulate_once(baseline_ts_nA, iter_label=-1)
        force_final = outF["pool_force_percent"]
        final_forces_dict = outF["forces"]
        force_initial = force0.copy()
        rmse  = _rmse_core(force_final, target_percent_ts)
        logger.info(f"      Final RMSE = {rmse:.3f} %MVC")

    else:
        # Single pass (no optimization)
        outF = simulate_once(baseline_ts_nA, iter_label=0)
        force_final = outF["pool_force_percent"]
        final_forces_dict = outF["forces"]
        force_initial = force_final.copy()
        rmse  = _rmse_core(force_final, target_percent_ts)

    final_baseline_tonic_trace_nA = _baseline_trace_for_step(params, baseline_ts_nA, expected_n_samples)
    input_components["baseline_tonic_trace_nA"] = final_baseline_tonic_trace_nA.copy()
    input_components["baseline_plus_step_trace_nA"] = final_baseline_tonic_trace_nA + step_current_trace_nA
    input_components["common_input_with_step"] = np.asarray(input_components["common_input_with_burst"], dtype=float) + step_current_trace_nA
    input_components["total_pool_input_with_step_trace_nA"] = (
        np.asarray(input_components["common_input_with_burst"], dtype=float)
        + input_components["baseline_plus_step_trace_nA"]
    )

    # driving inputs used in the simulation and for optimization
    driving_inputs = {}
    driving_inputs["target_percent_ts"] = target_percent_ts
    driving_inputs["final_baseline_ts_nA"] = baseline_ts_nA
    driving_inputs["step_current_trace_nA"] = step_current_trace_nA
    driving_inputs["baseline_plus_step_trace_nA"] = input_components["baseline_plus_step_trace_nA"]
    for key, value in params.task_event_times_s.items():
        driving_inputs[f"task_{key}"] = np.array([float(value)], dtype=float)
    for key in (
        "lf_burst_kernel_nA",
        "alpha_burst_kernel_nA",
        "beta_burst_kernel_nA",
        "lf_trend_kernel_nA",
        "alpha_trend_kernel_nA",
        "beta_trend_kernel_nA",
        "lf_effective_envelope_nA",
        "alpha_effective_envelope_nA",
        "beta_effective_envelope_nA",
        "common_input_trend_trace_nA",
        "common_input_burst_trace_nA",
    ):
        if key in input_components:
            driving_inputs[key] = np.asarray(input_components[key], dtype=float)
    if input_burst_info is not None:
        driving_inputs["input_burst_trace_nA"] = input_burst_info["trace_nA"]
        driving_inputs["input_burst_envelope"] = input_burst_info["envelope"]
        driving_inputs["input_burst_center_s"] = np.array([input_burst_info["center_s"]], dtype=float)
        driving_inputs["input_burst_start_s"] = np.array([input_burst_info["start_s"]], dtype=float)
        driving_inputs["input_burst_zoom_start_s"] = np.array([
            input_burst_info["center_s"] - params.input_burst_zoom_start_n_sigma * (params.input_burst_sigma_ms / 1000.0)
        ], dtype=float)
    if params.optimize_baseline:
        driving_inputs["initial_baseline_ts_nA"] = baseline_initial
        driving_inputs["rmse_history"] = np.asarray(rmse_history, float)

    ########### SAVING HDF5 (final outputs) AND PLOTTING
    spike_trains_MN = outF["spike_trains_MN"]
    spike_trains_RC = outF["spike_trains_RC"]
    cst_analysis_start_s, cst_analysis_end_s = resolve_cst_analysis_window(params)
    active_unit_selection = classify_active_units_by_rate(
        spike_trains_MN,
        analysis_start_s=cst_analysis_start_s,
        analysis_end_s=cst_analysis_end_s,
        rate_threshold_hz=params.active_unit_rate_threshold_hz,
    )
    cst_diagnostics = compute_cst_diagnostics(
        spike_trains_MN=spike_trains_MN,
        params=params,
        burst_center_s=input_burst_info["center_s"] if input_burst_info is not None else None,
        logger=logger,
        active_unit_selection=active_unit_selection,
    )
    cue_time_abs_s = float(params.task_event_times_s["go_nogo_cue_s"])
    if go_nogo_cue_diagnostics_enabled:
        cst_diagnostics = _recompute_cst_modulation_normalization(
            cst_diagnostics,
            cue_time_abs_s=cue_time_abs_s,
            params=params,
        )
    if cst_diagnostics.get("status") == "ok":
        logger.info(
            "CST diagnostics retained MNs: %s | excluded: %s",
            cst_diagnostics["active_unit_ids"].tolist(),
            cst_diagnostics["excluded_unit_ids"].tolist(),
        )
    else:
        logger.warning("CST diagnostics skipped: %s", cst_diagnostics.get("status", "unknown"))
    firing_rate_diagnostics = None
    if go_nogo_cue_diagnostics_enabled:
        firing_rate_spike_trains = _crop_spike_trains_to_rel_window(
            spike_trains_MN,
            cue_time_abs_s=cue_time_abs_s,
            window_rel_cue_s=params.firing_rate_isi_window_rel_cue_s,
        )
        firing_rate_diagnostics = compute_firing_rate_diagnostics(
            firing_rate_spike_trains,
            active_unit_ids_for_isi_cv=active_unit_selection["active_unit_ids"],
        )
    sync_diagnostics = None
    if go_nogo_cue_diagnostics_enabled and params.enable_sync_index_analysis:
        try:
            from analyzer_with_force import run_sliding_sync_index_analysis as _run_sync_index_analysis

            sync_analysis_window_rel_cue_s = _normalize_rel_window(params.sync_analysis_window_rel_cue_s)
            sync_active_ids = np.asarray(active_unit_selection["active_unit_ids"], dtype=int)
            sync_spike_trains = [spike_trains_MN[int(unit_i)] for unit_i in sync_active_ids]
            if len(sync_spike_trains) < 2:
                raise ValueError("At least two active units are required for synchrony analysis")
            sync_diagnostics = _run_sync_index_analysis(
                sync_spike_trains,
                cue_time_sec=cue_time_abs_s,
                baseline_win=params.sync_baseline_win_rel_cue_s,
                post_win=params.sync_post_win_rel_cue_s,
                generate_figure=False,
                display_mode=params.sync_trace_display_mode,
                bin_ms=params.sync_bin_ms,
                sync_win_ms=params.sync_win_ms,
                sync_step_ms=params.sync_step_ms,
                coinc_lag_ms=params.sync_coinc_lag_ms,
                direction_mode=params.sync_direction_mode,
                expectation_mode=params.sync_expectation_mode,
                n_surrogates=params.sync_n_surrogates,
                surrogate_min_shift_ms=params.sync_surrogate_min_shift_ms,
                duration_s=float(params.duration_with_ignored_window),
                analysis_start_s=float(cue_time_abs_s + sync_analysis_window_rel_cue_s[0]),
                analysis_end_s=float(cue_time_abs_s + sync_analysis_window_rel_cue_s[1]),
                unit_ids=sync_active_ids,
            )
            logger.info(
                "Sliding sync_index calculated | baseline=%.4f | peak=%.4f at %.4f s",
                float(sync_diagnostics["baseline_mean"]),
                float(sync_diagnostics["post_peak"]),
                float(sync_diagnostics["post_peak_time_sec"]),
            )
        except Exception as exc:
            logger.warning(f"Could not compute sliding sync_index diagnostics: {exc}")
    step_sync_diagnostics = None
    if params.enable_step_current:
        step_sync_diagnostics = compute_step_sync_diagnostics(
            params=params,
            spike_trains_MN=spike_trains_MN,
            active_unit_selection=active_unit_selection,
            logger=logger,
        )
    step_response_diagnostics = None
    if params.enable_step_current:
        try:
            step_response_diagnostics = compute_step_response_diagnostics(
                params=params,
                spike_trains_MN=spike_trains_MN,
                active_unit_selection=active_unit_selection,
                cst_diagnostics=cst_diagnostics,
                sync_diagnostics=step_sync_diagnostics,
                input_components=input_components,
            )
            logger.info(
                "Step-response diagnostics calculated | dCST=%.4f | gain=%.6f per nA",
                float(np.asarray(step_response_diagnostics.get("delta_cst_lf", [np.nan])).reshape(-1)[0]),
                float(np.asarray(step_response_diagnostics.get("response_gain_cst_per_nA", [np.nan])).reshape(-1)[0]),
            )
        except Exception as exc:
            logger.warning(f"Could not compute step-response diagnostics: {exc}")
            step_response_diagnostics = {
                "status": "error",
                "error_message": str(exc),
            }
    observation_features = None
    analysis_summary_metadata = None
    if go_nogo_cue_diagnostics_enabled:
        modulation_spectrum_config = resolve_modulation_spectrum_config(
            {
                "window_rel_cue_s": params.modulation_spectrum_window_rel_cue_s,
                "max_freq_hz": params.modulation_spectrum_max_freq_hz,
                "interval_mass_pct": params.modulation_spectrum_interval_mass_pct,
            }
        )
        observation_features = compute_observation_features_from_cst_sync(
            cst_diagnostics,
            sync_diagnostics,
            cue_time_abs_s=cue_time_abs_s,
            observation_window_defs_rel_cue_s=_normalize_observation_window_defs(params.observation_window_defs_rel_cue_s),
            obs_baseline_window_rel_cue_s=_normalize_rel_window(params.obs_baseline_window_rel_cue_s),
            modulation_spectrum_config=modulation_spectrum_config,
        )
        analysis_summary_metadata = _build_sim_analysis_summary_metadata(params)
        if cst_diagnostics is not None:
            for key in ("n_units_total", "n_units_kept", "unit_fraction_kept", "active_unit_ids", "excluded_unit_ids"):
                if key in cst_diagnostics and cst_diagnostics[key] is not None:
                    analysis_summary_metadata[key] = np.asarray(cst_diagnostics[key])
    compact_payload = build_compact_save_payload(
        params=params,
        common_input_MN=MN_excit_input,
        input_components=input_components,
        driving_inputs=driving_inputs,
        forces=final_forces_dict,
        cst_diagnostics=cst_diagnostics,
        sync_diagnostics=sync_diagnostics,
    )
    # ---------- SAVE (final outputs only) ----------
    output_savefile = save_output_hdf5(
        directory_name=directory_name,
        params=params,
        motoneurons_and_pools_idx={"pool_list_by_MN": pool_list_by_MN, "idx_of_MN_by_pool": idx_of_MN_by_pool},
        motoneurons_properties=motoneurons_properties_dict,
        connectivity_matrix_MN_to_RC=MN_to_Renshaw_connectivity_matrix,
        connectivity_matrix_RC_to_MN=Renshaw_to_MNs_connectivity_matrix,
        connectivity_matrix_MN_to_MN=MN_to_MN_connectivity_matrix,
        spike_trains_MN=spike_trains_MN, spike_trains_RC=spike_trains_RC,
        common_input_MN=compact_payload["common_input_MN"],
        common_input_power_total=total_power,
        common_input_power_spectrum=power_per_frequency_band,
        independent_input_power_total=None,       # could pass from last simulate_once if you modify it to return
        independent_input_power_spectrum=None,
        forces=compact_payload["forces"],                 # <-- final (best, if optimized) forces
        driving_inputs=compact_payload["driving_inputs"],
        input_components=compact_payload["input_components"],
        cst_diagnostics=compact_payload["cst_diagnostics"],
        sync_diagnostics=compact_payload["sync_diagnostics"],
        firing_rate_diagnostics=firing_rate_diagnostics,
        step_response_diagnostics=step_response_diagnostics,
        analysis_summary_metadata=analysis_summary_metadata,
        observation_features=observation_features,
        save_spike_trains=(not params.minimal_output) or bool(params.minimal_output_save_spike_trains),
        saved_trace_fsamp_hz=compact_payload["saved_trace_fsamp_hz"],
        sync_saved_fsamp_hz=compact_payload["sync_saved_fsamp_hz"],
    )
    logger.info(f"Data of simulation {sim_index} saved successfully to '{output_savefile}'.")

    # ---------- PLOTS ----------
    if figures_enabled:
        fig_dir = directory_name
        plot_summary_input_cst = go_nogo_cue_diagnostics_enabled and bool(params.output_plot_summary_input_cst_figure)
        plot_detailed_input_cst = bool(params.output_plot_input_component_figures) and not bool(params.output_only_summary_input_cst_figure)
        plot_sync_diag = go_nogo_cue_diagnostics_enabled and bool(params.output_plot_sync_diagnostic_figure)
        plot_firing_rate = bool(params.output_plot_firing_rate_figure)
        plot_force_tracking = bool(params.output_plot_force_tracking_figure)
        plot_force_properties = bool(params.output_plot_force_properties_figures)
        plot_force_optimization = bool(params.output_plot_force_optimization_figures)
        plot_step_response = bool(params.output_plot_step_response_diagnostic_figure)

        t_input_s = np.arange(len(input_components["common_input_no_burst"]), dtype=float) / float(params.fsamp)
        usable_start_s = float(params.edges_ignore_duration)
        usable_end_s = float(params.duration_with_ignored_window - params.edges_ignore_duration)
        diag_colors = get_input_component_colors(params)
        diag_zoom = resolve_diagnostic_zoom_window(
            params,
            t_input_s,
            window_start_s=usable_start_s,
            window_end_s=usable_end_s,
            burst_center_s=input_burst_info["center_s"] if input_burst_info is not None else None,
        )
        input_component_zoom_mask = diag_zoom["zoom_mask"]
        burst_center_s = diag_zoom["burst_center_s"]
        burst_start_s = diag_zoom["burst_start_s"]
        burst_end_s = diag_zoom["burst_end_s"]
        show_go_nogo_plot_markers = bool(go_nogo_cue_diagnostics_enabled or params.enable_input_burst)
        show_step_plot_marker = bool(params.enable_step_current and params.spike_train_show_step_current_marker)

        if plot_detailed_input_cst:
            plot_input_component_diagnostics(
                save_dir=fig_dir,
                figure_name="fig_input_component_lf.png",
                component_name="Low-frequency",
                t_s=t_input_s,
                zoom_mask=input_component_zoom_mask,
                band_hz=params.lf_band_hz,
                target_sd=params.lf_target_sd,
                final_signal=input_components["lf_final"],
                time_series_specs=[
                    {
                        "signal": input_components["lf_raw_unit"],
                        "color": "tab:gray",
                        "ylabel": "Unitless",
                        "title": "LF raw unit signal",
                    },
                    {
                        "signal": input_components["lf_final"],
                        "color": diag_colors["lf"],
                        "ylabel": "nA",
                        "title": "LF final signal",
                    },
                ],
                task_event_times_s=params.task_event_times_s,
                burst_center_s=burst_center_s,
                burst_start_s=burst_start_s,
                burst_end_s=burst_end_s,
                band_color=diag_colors["lf"],
                colors=diag_colors,
                show_task_cue_markers=show_go_nogo_plot_markers,
                show_step_current_marker=show_step_plot_marker,
            )
            plot_input_component_diagnostics(
                save_dir=fig_dir,
                figure_name="fig_input_component_alpha.png",
                component_name="Alpha",
                t_s=t_input_s,
                zoom_mask=input_component_zoom_mask,
                band_hz=params.alpha_band_hz,
                target_sd=params.alpha_target_sd_final,
                final_signal=input_components["alpha_final"],
                time_series_specs=[
                    {
                        "signal": input_components["alpha_carrier_unit"],
                        "color": diag_colors["alpha"],
                        "ylabel": "Unitless",
                        "title": "Alpha carrier (unit SD)",
                    },
                    {
                        "signal": input_components["alpha_env_multiplier"],
                        "color": diag_colors["envelope"],
                        "ylabel": "Multiplier",
                        "title": f"Alpha envelope multiplier (scale={params.alpha_beta_envelope_variability_scale:.2f}, no burst)",
                    },
                    {
                        "signal": input_components["alpha_final"],
                        "color": diag_colors["alpha"],
                        "ylabel": "nA",
                        "title": "Alpha final signal",
                    },
                ],
                task_event_times_s=params.task_event_times_s,
                burst_center_s=burst_center_s,
                burst_start_s=burst_start_s,
                burst_end_s=burst_end_s,
                band_color=diag_colors["alpha"],
                colors=diag_colors,
                show_task_cue_markers=show_go_nogo_plot_markers,
                show_step_current_marker=show_step_plot_marker,
            )
            plot_input_component_diagnostics(
                save_dir=fig_dir,
                figure_name="fig_input_component_beta.png",
                component_name="Beta",
                t_s=t_input_s,
                zoom_mask=input_component_zoom_mask,
                band_hz=params.beta_band_hz,
                target_sd=params.beta_target_sd_final,
                final_signal=input_components["beta_final"],
                time_series_specs=[
                    {
                        "signal": input_components["beta_carrier_unit"],
                        "color": diag_colors["beta"],
                        "ylabel": "Unitless",
                        "title": "Beta carrier (unit SD)",
                    },
                    {
                        "signal": input_components["beta_env_multiplier"],
                        "color": diag_colors["envelope"],
                        "ylabel": "Multiplier",
                        "title": f"Beta envelope multiplier (scale={params.alpha_beta_envelope_variability_scale:.2f}, no burst)",
                    },
                    {
                        "signal": input_components["beta_final"],
                        "color": diag_colors["beta"],
                        "ylabel": "nA",
                        "title": "Beta final signal",
                    },
                ],
                task_event_times_s=params.task_event_times_s,
                burst_center_s=burst_center_s,
                burst_start_s=burst_start_s,
                burst_end_s=burst_end_s,
                band_color=diag_colors["beta"],
                colors=diag_colors,
                show_task_cue_markers=show_go_nogo_plot_markers,
                show_step_current_marker=show_step_plot_marker,
            )

            if cst_diagnostics.get("status") == "ok":
                cst_time = cst_diagnostics["time_s"]
                cst_zoom_mask = (cst_time >= diag_zoom["zoom_start_s"]) & (cst_time <= diag_zoom["zoom_end_s"])
                plot_input_component_diagnostics(
                    save_dir=fig_dir,
                    figure_name="fig_cst_component_lf.png",
                    component_name="CST low-frequency",
                    t_s=cst_time,
                    zoom_mask=cst_zoom_mask,
                    band_hz=params.lf_band_hz,
                    target_sd=float(np.std(cst_diagnostics["cst_lf"])),
                    final_signal=cst_diagnostics["cst_lf"],
                    time_series_specs=[
                        {
                            "signal": cst_diagnostics["cst_normalized"],
                            "color": "tab:gray",
                            "ylabel": "spikes/s/MU",
                            "title": "Active-unit-normalized CST",
                        },
                        {
                            "signal": cst_diagnostics["cst_lf"],
                            "color": diag_colors["lf"],
                            "ylabel": "spikes/s/MU",
                            "title": "LF component of CST",
                        },
                    ],
                    task_event_times_s=params.task_event_times_s,
                    burst_center_s=burst_center_s,
                    burst_start_s=burst_start_s,
                    burst_end_s=burst_end_s,
                    band_color=diag_colors["lf"],
                    colors=diag_colors,
                    show_task_cue_markers=show_go_nogo_plot_markers,
                    show_step_current_marker=show_step_plot_marker,
                )
                plot_input_component_diagnostics(
                    save_dir=fig_dir,
                    figure_name="fig_cst_component_alpha.png",
                    component_name="CST alpha",
                    t_s=cst_time,
                    zoom_mask=cst_zoom_mask,
                    band_hz=params.alpha_band_hz,
                    target_sd=float(np.std(cst_diagnostics["cst_alpha"])),
                    final_signal=cst_diagnostics["cst_alpha"],
                    time_series_specs=[
                        {
                            "signal": cst_diagnostics["cst_normalized"],
                            "color": "tab:gray",
                            "ylabel": "spikes/s/MU",
                            "title": "Active-unit-normalized CST",
                        },
                        {
                            "signal": cst_diagnostics["cst_alpha"],
                            "color": diag_colors["alpha"],
                            "ylabel": "spikes/s/MU",
                            "title": "Alpha component of CST",
                        },
                        {
                            "signal": cst_diagnostics["cst_alpha_envelope"],
                            "color": diag_colors["envelope"],
                            "ylabel": "spikes/s/MU",
                            "title": "Alpha envelope (Hilbert amplitude)",
                        },
                    ],
                    task_event_times_s=params.task_event_times_s,
                    burst_center_s=burst_center_s,
                    burst_start_s=burst_start_s,
                    burst_end_s=burst_end_s,
                    band_color=diag_colors["alpha"],
                    colors=diag_colors,
                    show_task_cue_markers=show_go_nogo_plot_markers,
                    show_step_current_marker=show_step_plot_marker,
                )
                plot_input_component_diagnostics(
                    save_dir=fig_dir,
                    figure_name="fig_cst_component_beta.png",
                    component_name="CST beta",
                    t_s=cst_time,
                    zoom_mask=cst_zoom_mask,
                    band_hz=params.beta_band_hz,
                    target_sd=float(np.std(cst_diagnostics["cst_beta"])),
                    final_signal=cst_diagnostics["cst_beta"],
                    time_series_specs=[
                        {
                            "signal": cst_diagnostics["cst_normalized"],
                            "color": "tab:gray",
                            "ylabel": "spikes/s/MU",
                            "title": "Active-unit-normalized CST",
                        },
                        {
                            "signal": cst_diagnostics["cst_beta"],
                            "color": diag_colors["beta"],
                            "ylabel": "spikes/s/MU",
                            "title": "Beta component of CST",
                        },
                        {
                            "signal": cst_diagnostics["cst_beta_envelope"],
                            "color": diag_colors["envelope"],
                            "ylabel": "spikes/s/MU",
                            "title": "Beta envelope (Hilbert amplitude)",
                        },
                    ],
                    task_event_times_s=params.task_event_times_s,
                    burst_center_s=burst_center_s,
                    burst_start_s=burst_start_s,
                    burst_end_s=burst_end_s,
                    band_color=diag_colors["beta"],
                    colors=diag_colors,
                    show_task_cue_markers=show_go_nogo_plot_markers,
                    show_step_current_marker=show_step_plot_marker,
                )
        if plot_summary_input_cst:
            plot_input_summary_diagnostics(
                save_dir=fig_dir,
                figure_name="fig_input_component_summary.png",
                t_s=t_input_s,
                zoom_mask=input_component_zoom_mask,
                input_components=input_components,
                cst_diagnostics=cst_diagnostics,
                sync_diagnostics=sync_diagnostics,
                params=params,
                burst_center_s=burst_center_s,
                burst_start_s=burst_start_s,
                burst_end_s=burst_end_s,
            )
        if plot_sync_diag and params.enable_sync_index_analysis and sync_diagnostics is not None:
            try:
                from analyzer_with_force import plot_sync_trace_diagnostic as _plot_sync_trace_diagnostic
                _plot_sync_trace_diagnostic(
                    sync_diagnostics,
                    savepath=fig_dir,
                    figure_name="fig_sync_index_diagnostic.png",
                    display_mode=params.sync_trace_display_mode,
                )
            except Exception as exc:
                logger.warning(f"Could not generate sync_index diagnostic figure from simulator output: {exc}")

        if plot_step_response and params.enable_step_current and step_response_diagnostics is not None:
            try:
                plot_step_response_diagnostic(
                    save_dir=fig_dir,
                    input_components=input_components,
                    cst_diagnostics=cst_diagnostics,
                    step_response_diagnostics=step_response_diagnostics,
                    params=params,
                )
            except Exception as exc:
                logger.warning(f"Could not generate step-response diagnostic figure: {exc}")

        # Reuse the analyzer's firing-rate summary figure to avoid a separate analysis pass.
        if plot_firing_rate:
            try:
                from analyzer_with_force import get_firing_rate as _plot_firing_rate_summary

                _plot_firing_rate_summary(
                    spike_trains_MN={f"MN_{i}": np.asarray(spk, dtype=float) for i, spk in enumerate(spike_trains_MN)},
                    spike_trains_RC={f"RC_{i}": np.asarray(spk, dtype=float) for i, spk in enumerate(spike_trains_RC)},
                    generate_figure=True,
                    savepath=fig_dir,
                    figure_name="SIMULATION_Firing_rate_and_ISI.png",
                )
            except Exception as exc:
                logger.warning(f"Could not generate firing-rate summary figure from simulator output: {exc}")

        if params.enable_force_model:
            # your existing force parameter/twitch plots (optional now)
            soma_d_um = np.asarray(motoneurons_properties_dict["soma_diameter"], float)
            if plot_force_properties and "tetanic_max_force" in motoneurons_properties_dict:  # present if you assigned earlier
                tetanic = np.asarray(motoneurons_properties_dict["tetanic_max_force"], float)
                peakrel = np.asarray(motoneurons_properties_dict["twitch_peak_force_relative"], float)
                tau_s   = np.asarray(motoneurons_properties_dict["twitch_time"], float)
                plot_force_params_overview(fig_dir, soma_d_um, tetanic, peakrel, tau_s)
                plot_all_twitch_kernels(fig_dir, params.fsamp, peakrel, tau_s, soma_d_um,
                                        duration_factor=params.force_kernel_duration_factor)

            # pool force vs target (final)
            if plot_force_tracking:
                plot_pool_force_tracking(
                    fig_dir,
                    params.fsamp,
                    final_forces_dict["pool_force_percent"],
                    target_percent_ts,
                    core_mask,
                    rmse,
                    common_input_by_pool=MN_excit_input,
                    baseline_input_by_pool=(
                        baseline_ts_nA
                        if baseline_ts_nA is not None
                        else np.asarray(params.excitatory_input_baseline, dtype=float)
                    ),
                    per_pool_force_percent=final_forces_dict["per_pool"]["force_percent"],
                    pool_names=final_forces_dict["per_pool"]["names"],
                    burst_center_s=input_burst_info["center_s"] if input_burst_info is not None else None,
                    burst_start_s=input_burst_info["start_s"] if input_burst_info is not None else None,
                    burst_trace_nA=input_burst_info["trace_nA"] if input_burst_info is not None else None,
                    burst_zoom_start_s=(
                        input_burst_info["center_s"] - params.input_burst_zoom_start_n_sigma * (params.input_burst_sigma_ms / 1000.0)
                        if input_burst_info is not None else None
                    ),
                    task_event_times_s=params.task_event_times_s,
                )


            # Initial VS final (after optimization, if present) force and driving input
            if plot_force_optimization and params.optimize_baseline:
                # 1) RMSE curve
                def _plot_opt_progress_rmse(fig_dir, rmse_hist):
                    if not rmse_hist: return
                    t = np.arange(len(rmse_hist))
                    plt.figure(figsize=(7,4), dpi=120)
                    plt.plot(t, rmse_hist, marker='o')
                    plt.xlabel("Iteration"); plt.ylabel("RMSE (%MVC, core)"); plt.grid(alpha=0.3)
                    plt.title("Optimization progress")
                    plt.tight_layout()
                    plt.savefig(os.path.join(fig_dir, "fig_opt_progress_rmse.png"), bbox_inches="tight")
                    plt.close()
                _plot_opt_progress_rmse(fig_dir, rmse_history)

            # 2) initial vs final tracking
            if plot_force_optimization:
                plt.figure(figsize=(11,5), dpi=120)
                tt = np.arange(len(target_percent_ts)) / float(params.fsamp)
                plt.plot(tt, target_percent_ts, color="tab:orange", lw=2.0, label="Target (%)")
                plt.plot(tt, force_initial,   color="tab:blue",   lw=1.2, alpha=0.7, label="Initial force (%)")
                plt.plot(tt, force_final,     color="tab:green",  lw=1.6, alpha=0.9, label="Final force (%)")
                plt.xlabel("Time (s)"); plt.ylabel("% MVC"); plt.grid(alpha=0.25)
                plt.fill_between(tt, np.min(target_percent_ts)-(np.max(target_percent_ts)*0.1), np.max(target_percent_ts)*1.1,
                                 where=~core_mask, color='gray', alpha=0.3, label="Ignored edges")
                plt.title(f"Pool force: initial vs final (RMSE = {rmse:.2f}% MVC)")
                plt.legend(loc="upper right")
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "fig_tracking_initial_vs_final.png"), bbox_inches="tight")
                plt.close()

                # 3) baseline TS initial vs final
                plt.figure(figsize=(11,4), dpi=120)
                tt = np.arange(len(baseline_initial)) / float(params.fsamp)
                plt.plot(tt, baseline_initial, color="tab:blue",  alpha=0.7, label="Initial baseline (nA)")
                if baseline_ts_nA is not None:
                    plt.plot(tt, baseline_ts_nA,   color="tab:green", alpha=0.9, label="Final baseline (nA)")
                plt.xlabel("Time (s)"); plt.ylabel("Baseline (nA)"); plt.grid(alpha=0.25)
                plt.title("Baseline time-series: initial vs final")
                plt.legend(loc="upper right")
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "fig_baseline_initial_vs_final.png"), bbox_inches="tight")
                plt.close()


    return output_savefile
