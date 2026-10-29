import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import json
import pandas as pd
from brian2 import *
from scipy.signal import windows, butter, sosfiltfilt, hilbert, resample_poly
from scipy.special import ndtr, ndtri
from dataclasses import dataclass, field, asdict
from typing import Dict
from scipy.signal import welch
import time
import h5py
from h5py import string_dtype
from typing import List
from threading import local
import logging
import warnings
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
    output_plot_common_vs_independent_power_figure: bool = False
    output_plot_common_input_weight_figure: bool = False
    output_plot_independent_input_weight_figure: bool = False
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
    parameter_set_folder_name: str = ""

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
    # The AHP reversal potential is the resting potential (set in create_motoneuron_group)

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

    spike_train_show_task_cues_when_no_burst: bool = True # show the task-cue markers on the spike-train figure even without an input burst
    
    # ──── FREQUENCY FILTERING "MASTER PARAMETERS" ────
    scale_filter_order_to_frequency: bool = True
    filter_order_scaling_coeff: float = 0.5 # Multiplying the cutoff frequency to determine filter order. Used only if scale_filter_order_to_frequency == True
    default_freq_filter_order: int = 5 # Used if scale_filter_order_to_frequency = False
    lowest_freq_filter_order: int = 5 # # Used if scale_filter_order_to_frequency = True
    max_freq_filter_order: int = 100

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
    feature_buffer_s: float = 0.0
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

        core_start_s = float(self.edges_ignore_duration)
        core_end_s = core_start_s + float(self.duration)
        get_ready_cue_s = core_start_s + float(self.get_ready_cue_time_s)
        go_nogo_cue_s = get_ready_cue_s + float(self.go_nogo_cue_delay_s)
        burst_center_s = go_nogo_cue_s + float(self.burst_delay_after_go_nogo_cue_s)

        if get_ready_cue_s > core_end_s:
            raise ValueError("get_ready_cue_time_s places the get-ready cue after the usable simulation window")
        if go_nogo_cue_s > core_end_s:
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
        self.parameter_set_folder_name = str(self.parameter_set_folder_name)
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
        self.spike_train_show_task_cues_when_no_burst = bool(self.spike_train_show_task_cues_when_no_burst)
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


def build_compact_save_payload(
        params,
        common_input,
        input_components,
        driving_inputs,
        cst_diagnostics,
        sync_diagnostics):
    save_trace_fsamp_hz = float(params.fsamp)
    sync_saved_fsamp_hz = None
    common_input_to_save = _copy_value_for_save(common_input)
    input_components_to_save = {k: _copy_value_for_save(v) for k, v in (input_components or {}).items()}
    driving_inputs_to_save = {k: _copy_value_for_save(v) for k, v in (driving_inputs or {}).items()}
    cst_to_save = None if cst_diagnostics is None else {k: _copy_value_for_save(v) for k, v in cst_diagnostics.items()}
    sync_to_save = None if sync_diagnostics is None else {k: _copy_value_for_save(v) for k, v in sync_diagnostics.items()}

    if not params.minimal_output:
        return {
            "common_input": common_input_to_save,
            "input_components": input_components_to_save,
            "driving_inputs": driving_inputs_to_save,
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
        source_n_samples = int(np.asarray(common_input_to_save).shape[-1])
        common_input_to_save = _resample_array_last_axis(common_input_to_save, params.fsamp, target_fsamp_hz)
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
        "common_input": common_input_to_save,
        "input_components": input_components_to_save,
        "driving_inputs": driving_inputs_to_save,
        "cst_diagnostics": cst_to_save,
        "sync_diagnostics": sync_to_save,
        "saved_trace_fsamp_hz": save_trace_fsamp_hz,
        "sync_saved_fsamp_hz": sync_saved_fsamp_hz,
    }

######################################
### HELPER FUNCTIONS
######################################
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
# ───────────────────────
def filter_artifact_removal(fsamp, edges_ignore_duration):
    duration_to_remove = edges_ignore_duration # in second
    Wind_s = duration_to_remove * 2
    artifact_removal_window = windows.hann(round(fsamp * Wind_s))
    artifact_removal_window = artifact_removal_window[:int(np.round(len(artifact_removal_window)/2))]
    nb_samples_artifact_removal_window = len(artifact_removal_window)
    return artifact_removal_window, nb_samples_artifact_removal_window
def Generate_filtered_gaussian_noise_input(fsamp, 
        duration_with_ignored_window, edges_ignore_duration, artifact_removal_window,
        low_pass_filter_cutoff, input_mean, scaling_std):
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
        show_task_cue_markers=True):
    task_event_times_s = task_event_times_s or {}
    colors = colors or {}
    cue_color = colors.get("cue", "0.5")
    burst_color = colors.get("burst", "#FF0080")
    get_ready_cue_s = task_event_times_s.get("get_ready_cue_s")
    go_nogo_cue_s = task_event_times_s.get("go_nogo_cue_s")
    if burst_start_s is not None and burst_end_s is not None:
        ax.axvspan(float(burst_start_s) - time_offset_s, float(burst_end_s) - time_offset_s, color=burst_color, alpha=0.10, zorder=0)
    if show_task_cue_markers and get_ready_cue_s is not None:
        ax.axvline(float(get_ready_cue_s) - time_offset_s, color=cue_color, lw=1.0, ls="--", alpha=0.8, zorder=1)
    if show_task_cue_markers and go_nogo_cue_s is not None:
        ax.axvline(float(go_nogo_cue_s) - time_offset_s, color=cue_color, lw=1.1, ls="-", alpha=0.9, zorder=1)
    if burst_center_s is not None:
        ax.axvline(float(burst_center_s) - time_offset_s, color=burst_color, lw=1.1, ls="--", zorder=1)
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
    cst_alpha = apply_zero_phase_component_filter(cst_normalized, params.fsamp, params.alpha_band_hz, name="cst_alpha")
    cst_beta = apply_zero_phase_component_filter(cst_normalized, params.fsamp, params.beta_band_hz, name="cst_beta")
    cst_alpha_envelope = np.abs(hilbert(cst_alpha))
    cst_beta_envelope = np.abs(hilbert(cst_beta))

    t_s = np.arange(n_samples, dtype=float) / float(params.fsamp)
    analysis_mask = (t_s >= analysis_start_s) & (t_s <= analysis_end_s)
    baseline_start_s = float(analysis_start_s)
    baseline_end_s = float(params.task_event_times_s["get_ready_cue_s"])
    baseline_reference_event = "get_ready_cue"
    baseline_mask = (t_s >= max(analysis_start_s, baseline_start_s)) & (t_s < min(analysis_end_s, baseline_end_s))
    if baseline_end_s <= baseline_start_s or not np.any(baseline_mask):
        raise ValueError("CST baseline window is empty: the analysis window must start before the get-ready cue")
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
        show_task_cue_markers=True):
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
            - (g_ahp_f + g_ahp_s)*(v - voltage_AHP) # AHP current from AHP conductance (reversal potential voltage_AHP = resting potential) => g_ahp_f is FAST (regular AHP), g_ahp_s is SLOW (zero increment in the paper's simulations)
            + input_weight*input_MN_timedarray_amp(t,i)  # excitatory synaptic current (weighted by input_weight, i.e. normalized resistance)
        )/C_m : volt (unless refractory)

        dg_ahp_f/dt = -g_ahp_f/tau_ahp_f       : siemens  # AHP conductance decays exponentially
        dg_ahp_s/dt = -g_ahp_s/tau_ahp_s    : siemens

        g_leak            : siemens
        C_m               : farad                    
        input_weight      : 1
        voltage_rest      : volt
        voltage_AHP       : volt
        refractory_period : second
        tau_ahp_f         : second   # fast AHP component (size-dependent, "base" AHP tau)
        tau_ahp_s         : second   # slow AHP component
        g_ahp_s_max       : siemens  # cap on slow AHP conductance (no cap on fast AHP)
    ''')
    return MN_equations

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
        total_nb_motoneurons,
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
    # Single motor-neuron pool; the pool labels are kept in the saved output for compatibility
    pool_list_by_MN = ["pool_0"] * total_nb_motoneurons
    idx_of_MN_by_pool = {"pool_0": np.arange(total_nb_motoneurons)}
    
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
        # 5) Membrane time constant
        ax5 = axes[3]
        ln1, = ax5.plot(motoneurons_membrane_time_constant, label="Membrane time constant (ms)", color='black', linestyle='--')
        ax5.set_ylabel("Time (ms)")
        ax5.set_title("Membrane time constant")
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
    total_nb_motoneurons,
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
    alpha_beta_envelope_variability_scale = float(alpha_beta_envelope_variability_scale)
    if not (0.0 <= alpha_beta_envelope_variability_scale <= 1.0):
        raise ValueError("alpha_beta_envelope_variability_scale must be between 0 and 1")

    rng = np.random
    n_samples = int(np.round(duration_with_ignored_window * fsamp))
    dt = 1.0 / float(fsamp)
    component_padding_s = max(float(edges_ignore_duration), 2.0)
    t_s = np.arange(n_samples, dtype=float) / float(fsamp)
    filter_params = _get_filter_params()
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
        total_nb_motoneurons,
        target_sd=common_input_weight_sd,
        min_weight=common_input_weight_min,
        max_weight=common_input_weight_max,
    )
    common_input_weights = np.asarray(common_input_weight_info["weights"], dtype=float)

    freqs_common_final, psd_common_final = compute_psd_welch(delivered_common_input, fsamp=fsamp)
    freqs_common_no_burst, psd_common_no_burst = compute_psd_welch(common_input_no_burst, fsamp=fsamp)
    total_power = float(np.trapezoid(psd_common_final, freqs_common_final))
    common_input_reference_power = float(np.trapezoid(psd_common_no_burst, freqs_common_no_burst))
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
        delivered_common_input,
        common_input_weights,
        total_power,
        common_input_reference_power,
        power_per_frequency_band,
        input_burst_info,
        input_components,
    )

# # # GENERATE INDEPENDENT INPUTS AND CREATE BRIAN2 TIME ARRAYS
def generate_independent_inputs(fsamp,
        total_nb_motoneurons,
        low_pass_filter_of_MN_independent_input,
        ref_common_input_power,
        MN_independent_input_absolute_or_ratio,
        MN_independent_input_power,
        independent_input_weight_sd,
        independent_input_weight_max,
        common_input,
        common_input_weights,
        excitatory_input_baseline, # in nA
        duration_with_ignored_window,
        edges_ignore_duration,
        motoneurons_rheobases,
        generate_figure=False, generate_weight_figure=False, savepath=None,
        save_power_csv=False
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
    nperseg = min(fsamp, len(common_input))
    for mni in range(total_nb_motoneurons):
        temp_input = Generate_filtered_gaussian_noise_input(fsamp, 
            duration_with_ignored_window, edges_ignore_duration, artifact_removal_window,
            low_pass_filter_cutoff=low_pass_filter_of_MN_independent_input,
            input_mean=0, scaling_std=1)
        # Scale it to the desired power
        temp_input_power = np.mean(temp_input**2)
        if MN_independent_input_absolute_or_ratio == 'absolute':
            independent_input_scaling_factor = MN_independent_input_power
        elif MN_independent_input_absolute_or_ratio == 'ratio':
            desired_independent_power = MN_independent_input_power * ref_common_input_power
            independent_input_scaling_factor = np.sqrt(desired_independent_power / temp_input_power)
        temp_input *= (independent_input_scaling_factor * independent_input_weights[mni])
        # Keep the first independent input as a separate variable for plotting purpose:
        MN_independent_input.append(temp_input)
        if generate_figure and mni==0:
            independent_input_for_plotting = temp_input.copy()
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
    total_power_independent = np.trapezoid(power_per_frequency_band_independent["power"], freqs)
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
    n_samples = len(common_input)
    temp_timed_array = np.zeros((n_samples, total_nb_motoneurons))
    for mni in range(total_nb_motoneurons):
        temp_common_plus_baseline = np.array(
                common_input) * common_input_weights[mni] # the weights modulate the common input per MN, independently of the size of the motor neurons!
        # add constant excitatory baseline (nA)
        temp_common_plus_baseline += excitatory_input_baseline
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
        common_input_for_plotting_power = np.mean(common_input**2)
        if nb_samples_to_plot > len(independent_input_for_plotting):
            nb_samples_to_plot = len(independent_input_for_plotting)
        plt.plot(np.arange(nb_samples_to_plot)/fsamp,
            common_input[:nb_samples_to_plot] * common_input_weights[0],
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
        total_power_independent,
        power_per_frequency_band_independent,
        independent_input_payload,
    )

# # # CREATE Brian2 NEURONGROUP OF MOTOR NEURONS
def create_motoneuron_group(
        total_nb_motoneurons,
        MN_equations, voltage_rest, voltage_thresh,
        motoneurons_membrane_conductance, motoneurons_capacitance,
        motoneurons_input_weight,
        motoneurons_AHP_conductance_decay_time_constant, # fast AHP (per MN, sie-dependent)
        AHP_slow_conductance_decay_time_constant_multiplier,
        AHP_fast_conductance_delta_after_spiking, AHP_slow_conductance_delta_after_spiking,
        maximum_AHP_conductance_slow_conductance,
        motoneurons_refractory_periods):
    # # ----------------- BRIAN2 NAMESPACE FOR VARIABLES
    common_brian2_namespace = {
        'voltage_thresh': voltage_thresh,
        'voltage_rest':   voltage_rest,
        'AHP_fast_conductance_delta_after_spiking': AHP_fast_conductance_delta_after_spiking,
        'AHP_slow_conductance_delta_after_spiking': AHP_slow_conductance_delta_after_spiking,
    }
    # ----------------- NEURON GROUP -----------
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
    # The AHP reversal potential equals the resting potential, as in all of the paper's simulations;
    # an earlier -13.3 mV setting was never applied (both per-neuron variables were left at 0 mV = rest).
    motoneurons.voltage_rest = voltage_rest
    motoneurons.voltage_AHP = voltage_rest
    motoneurons.v = voltage_rest # in mV #
    motoneurons.g_leak = motoneurons_membrane_conductance * msiemens # in milisiemens
    motoneurons.C_m = motoneurons_capacitance * ufarad # in microfarads
    motoneurons.refractory_period = motoneurons_refractory_periods * ms  # in milliseconds
    motoneurons.input_weight = motoneurons_input_weight # dimensionless unit
    # AHP
    motoneurons.tau_ahp_f = motoneurons_AHP_conductance_decay_time_constant * ms # in millisecond
    motoneurons.tau_ahp_s = motoneurons_AHP_conductance_decay_time_constant * AHP_slow_conductance_decay_time_constant_multiplier * ms # in millisecond
    motoneurons.g_ahp_s_max = maximum_AHP_conductance_slow_conductance * msiemens # in milisiemens
    # Initialize AHP with a conductance of 0
    motoneurons.g_ahp_f = 0*siemens
    motoneurons.g_ahp_s = 0*siemens
    # logger.info(f"motoneurons input weights = {motoneurons.input_weight}")
    return motoneurons

# # # GET SPIKE TRAINS
def get_spike_trains(spike_monitor_MN,
    spike_transmission_delay, total_nb_motoneurons,
    edges_ignore_duration, duration_with_ignored_window, motoneurons_soma_diameters,
    generate_figure=False, savepath=None, task_event_times_s=None, burst_start_s=None, burst_center_s=None,
    show_task_cue_markers=True):
    """
    Pull out and (optionally) plot the post‐delay spike trains from your monitor.

    Parameters
    ----------
    spike_monitor_MN : Brian2 SpikeMonitor recording the motoneurons
    spike_transmission_delay : array_like, length total_nb_motoneurons
        delay (in seconds) to add to each MN spike train before returning.
    total_nb_motoneurons : int
        How many MNs in total you expect (so that even silent ones get an empty list).
    generate_figure : bool
    savepath : str or None
    """
    # 1) Extract raw trains from the monitors
    raw_MN = spike_monitor_MN.spike_trains()   # { neuron_index: array_of_times_in_s, ... }

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

    return spike_trains_MN

# # # SAVE OUTPUT AS HDF5 FILE WITH H5PY
def save_output_hdf5(directory_name,
                     params,                    # SimulationParameters dataclass
                     motoneurons_and_pools_idx, # dict of dicts, see generate_motor_neurons() (saving {pool_list_by_MN, idx_of_MN_by_pool})
                     motoneurons_properties,    # dict of np.ndarrays
                     spike_trains_MN,           # list of 1D np.ndarrays
                     common_input,              # 1D np.array
                     common_input_power_total, common_input_power_spectrum, # floats
                     independent_input_power_total=None, independent_input_power_spectrum=None, # dict with keys "frequencies" and "power"  
                     driving_inputs=None,
                     input_components=None,
                     cst_diagnostics=None,
                     sync_diagnostics=None,
                     firing_rate_diagnostics=None,
                     analysis_summary_metadata=None,
                     observation_features=None,
                     save_spike_trains=True,
                     saved_trace_fsamp_hz=None,
                     sync_saved_fsamp_hz=None):
    
    output_file = os.path.join(directory_name, "simulation_output.h5")
    with h5py.File(output_file, "w") as f:
        # 1) simulation parameters
        sim_grp = f.create_group("simulation_parameters")
        param_dict = asdict(params)
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

        # 5) spike trains
        if save_spike_trains:
            spikes = f.create_group("spike_trains")
            mn_spikes = spikes.create_group("MN")
            for i, tr in enumerate(spike_trains_MN):
                mn_spikes.create_dataset(f"MN_{i}", data=tr)

        # 6) Synaptic input
        input_grp = f.create_group("input")
        if saved_trace_fsamp_hz is None:
            saved_trace_fsamp_hz = float(params.fsamp)
        input_grp.attrs["saved_fsamp_Hz"] = float(saved_trace_fsamp_hz)
        # Common input, saved as a 2D numeric array of shape (1, n_timepoints) (one row for the single pool)
        common_input_array = np.stack([common_input], axis=0)
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
        input_grp.attrs["total_power_common_input"] = common_input_power_total
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
                "common_input_trend_trace_nA",
                "common_input_burst_trace_nA",
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

        # 7) driving inputs used in the simulation
        g = f.create_group("driving_inputs")
        g.attrs["saved_fsamp_Hz"] = float(saved_trace_fsamp_hz)
        for k, v in driving_inputs.items():
            if v is not None:
                g.create_dataset(k, data=np.asarray(v))
            else:
                g.create_dataset(k, data=np.array([np.nan]))

    return output_file


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
    figures_enabled = _figures_enabled(params)

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
    with open(f"{directory_name}/sim_parameters.json","w") as fp:
        json.dump(param_dict, fp, indent=2, default=str)

    # # # SET EQUATIONS
    MN_equations = set_brian2_equations()

    # # # GENERATE MOTOR NEURONS
    (motoneuron_soma_diameters, motoneuron_normalized_soma_diameters,
     pool_list_by_MN, idx_of_MN_by_pool) = generate_motor_neurons(
            min_soma_diameter=params.min_soma_diameter, max_soma_diameter=params.max_soma_diameter,
            total_nb_motoneurons=params.total_nb_motoneurons,
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
    (common_input, common_input_weights, total_power, common_input_reference_power,
     power_per_frequency_band, input_burst_info, input_components) = generate_and_distribute_common_inputs(
        fsamp=params.fsamp,
        total_nb_motoneurons=params.total_nb_motoneurons,
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
    
    # Important: reset Brian2 state for a fresh run
    start_scope()

    # # # ADD INDEPENDENT INPUTS AND CREATE Brian2 TIMED ARRAY OBJECTS
    (
        input_MN_timedarray_amp,
        total_power_independent,
        power_per_frequency_band_independent,
        independent_input_payload,
    ) = generate_independent_inputs(
        fsamp=params.fsamp,
        total_nb_motoneurons=params.total_nb_motoneurons,
        low_pass_filter_of_MN_independent_input=params.low_pass_filter_of_MN_independent_input,
        ref_common_input_power=common_input_reference_power,    # full-band no-burst common-input power
        MN_independent_input_absolute_or_ratio=params.independent_input_absolute_or_ratio,
        MN_independent_input_power=params.independent_input_power,
        independent_input_weight_sd=params.independent_input_weight_sd,
        independent_input_weight_max=params.independent_input_weight_max,
        common_input=common_input,
        common_input_weights=common_input_weights,
        excitatory_input_baseline=params.excitatory_input_baseline,
        duration_with_ignored_window=params.duration_with_ignored_window,
        edges_ignore_duration=params.edges_ignore_duration,
        motoneurons_rheobases=motoneurons_rheobases,
        generate_figure=False,
        generate_weight_figure=figures_enabled and params.output_plot_independent_input_weight_figure,
        savepath=directory_name,
        save_power_csv=params.save_independent_input_power_csv,
    )
    if independent_input_payload:
        input_components["independent_input_weights"] = np.asarray(independent_input_payload["weights"], dtype=float)
        input_components["independent_input_weight_mu_ln"] = np.asarray([independent_input_payload["mu_ln"]], dtype=float)
        input_components["independent_input_weight_sigma_ln"] = np.asarray([independent_input_payload["sigma_ln"]], dtype=float)
        input_components["independent_input_weight_target_sd"] = np.asarray([independent_input_payload["target_sd"]], dtype=float)
        input_components["independent_input_weight_max"] = np.asarray([independent_input_payload["max_weight"]], dtype=float)
    # # # IF DESIRED, DISPLAY POWER OF COMMON AND INDEPENDENT INPUTS
    if figures_enabled and params.output_plot_common_vs_independent_power_figure:
        plt.figure(figsize=(10,6))
        scaling_power = 1/1e6
        plt.fill_between(
                power_per_frequency_band_independent['frequencies'],
                power_per_frequency_band_independent['power']*scaling_power,
                y2=0, color='blue', alpha=0.3)
        plt.fill_between(
                power_per_frequency_band['frequencies'],
                power_per_frequency_band['power']*scaling_power,
                y2=0, color='red', alpha=0.3)
        plt.plot(power_per_frequency_band['frequencies'],
                power_per_frequency_band['power']*scaling_power,
                color='red', label=f'Common input power\ntotal={total_power*scaling_power:.2f} a.u.')
        plt.plot(power_per_frequency_band_independent['frequencies'],
            power_per_frequency_band_independent['power']*scaling_power,
            color='blue', label=f'Independent input power\ntotal={total_power_independent*scaling_power:.2f} a.u.')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (a.u.)")
        plt.xlim(0, 100)
        plt.legend(loc="upper right")
        plt.savefig(f"{directory_name}/Power_of_common_and_independent_inputs.png")

    # # # CREATE Brian2 NEURONGROUP
    # --- Group (reuse your already-built arrays & equations) ---
    motoneurons = create_motoneuron_group(
        total_nb_motoneurons=params.total_nb_motoneurons,
        MN_equations=MN_equations, voltage_rest=params.voltage_rest, voltage_thresh=params.voltage_thresh,
        motoneurons_membrane_conductance=motoneurons_membrane_conductance, motoneurons_capacitance=motoneurons_capacitance,
        motoneurons_input_weight=motoneurons_input_weight,
        # AHP parameters
        motoneurons_AHP_conductance_decay_time_constant=motoneurons_AHP_conductance_decay_time_constant,
        AHP_slow_conductance_decay_time_constant_multiplier=params.AHP_slow_conductance_decay_time_constant_multiplier,
        AHP_fast_conductance_delta_after_spiking=params.AHP_fast_conductance_delta_after_spiking,
        AHP_slow_conductance_delta_after_spiking=params.AHP_slow_conductance_delta_after_spiking,
        maximum_AHP_conductance_slow_conductance=params.maximum_AHP_conductance_slow_conductance,
        #
        motoneurons_refractory_periods=motoneurons_refractory_periods,
    )

    # --- Run ---
    motoneurons.v = params.voltage_rest
    monitor_spikes_motoneurons = SpikeMonitor(motoneurons, record=True)

    logger.info(f"Starting simulation (T={params.duration_with_ignored_window:.1f}s)")
    run(params.duration_with_ignored_window * second)
    logger.info("...simulation finished.")

    # --- Extract spikes ---
    spike_trains_MN = get_spike_trains(
        spike_monitor_MN=monitor_spikes_motoneurons,
        spike_transmission_delay=motoneurons_spike_transmission_delays,
        total_nb_motoneurons=params.total_nb_motoneurons,
        edges_ignore_duration=params.edges_ignore_duration,
        duration_with_ignored_window=params.duration_with_ignored_window,
        motoneurons_soma_diameters=motoneuron_soma_diameters,
        generate_figure=figures_enabled and params.output_plot_spike_trains_figure,
        savepath=directory_name,
        task_event_times_s=params.task_event_times_s,
        burst_start_s=params.task_event_times_s.get("burst_center_s", 0.0) - params.input_burst_start_marker_n_sigma * (params.input_burst_sigma_ms / 1000.0) if params.enable_input_burst else None,
        burst_center_s=params.task_event_times_s.get("burst_center_s") if params.enable_input_burst else None,
        show_task_cue_markers=bool(params.enable_input_burst or params.spike_train_show_task_cues_when_no_burst),
    )

    # driving inputs used in the simulation
    driving_inputs = {}
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

    ########### SAVING HDF5 (final outputs) AND PLOTTING
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
    if params.enable_sync_index_analysis:
        try:
            from synchrony_analysis import run_sliding_sync_index_analysis as _run_sync_index_analysis

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
        common_input=common_input,
        input_components=input_components,
        driving_inputs=driving_inputs,
        cst_diagnostics=cst_diagnostics,
        sync_diagnostics=sync_diagnostics,
    )
    # ---------- SAVE (final outputs only) ----------
    output_savefile = save_output_hdf5(
        directory_name=directory_name,
        params=params,
        motoneurons_and_pools_idx={"pool_list_by_MN": pool_list_by_MN, "idx_of_MN_by_pool": idx_of_MN_by_pool},
        motoneurons_properties=motoneurons_properties_dict,
        spike_trains_MN=spike_trains_MN,
        common_input=compact_payload["common_input"],
        common_input_power_total=total_power,
        common_input_power_spectrum=power_per_frequency_band,
        independent_input_power_total=None,
        independent_input_power_spectrum=None,
        driving_inputs=compact_payload["driving_inputs"],
        input_components=compact_payload["input_components"],
        cst_diagnostics=compact_payload["cst_diagnostics"],
        sync_diagnostics=compact_payload["sync_diagnostics"],
        firing_rate_diagnostics=firing_rate_diagnostics,
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
        plot_summary_input_cst = bool(params.output_plot_summary_input_cst_figure)
        plot_detailed_input_cst = bool(params.output_plot_input_component_figures) and not bool(params.output_only_summary_input_cst_figure)
        plot_sync_diag = bool(params.output_plot_sync_diagnostic_figure)
        plot_firing_rate = bool(params.output_plot_firing_rate_figure)

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
                from synchrony_analysis import plot_sync_trace_diagnostic as _plot_sync_trace_diagnostic
                _plot_sync_trace_diagnostic(
                    sync_diagnostics,
                    savepath=fig_dir,
                    figure_name="fig_sync_index_diagnostic.png",
                    display_mode=params.sync_trace_display_mode,
                )
            except Exception as exc:
                logger.warning(f"Could not generate sync_index diagnostic figure from simulator output: {exc}")

        # Reuse the analyzer's firing-rate summary figure to avoid a separate analysis pass.
        if plot_firing_rate:
            try:
                from synchrony_analysis import get_firing_rate as _plot_firing_rate_summary

                _plot_firing_rate_summary(
                    spike_trains_MN={f"MN_{i}": np.asarray(spk, dtype=float) for i, spk in enumerate(spike_trains_MN)},
                    generate_figure=True,
                    savepath=fig_dir,
                    figure_name="SIMULATION_Firing_rate_and_ISI.png",
                )
            except Exception as exc:
                logger.warning(f"Could not generate firing-rate summary figure from simulator output: {exc}")

    return output_savefile
