from dendro.sdk import InputFile, OutputFile
from pydantic import BaseModel, Field
from typing import List, Union, Optional
from enum import Enum


# ------------------------------
# Recording Models
# ------------------------------
class RecordingContext(BaseModel):
    electrical_series_path: str = Field(description='Path to the electrical series in the NWB file')


# ------------------------------
# Preprocessing Models
# ------------------------------
class HighpassFilter(BaseModel):
    freq_min: float = Field(default=300.0, description="Minimum frequency for the highpass filter")
    margin_ms: float = Field(default=5.0, description="Margin in milliseconds")


class PhaseShift(BaseModel):
    margin_ms: float = Field(default=100.0, description="Margin in milliseconds for phase shift")


class DetectBadChannels(BaseModel):
    method: str = Field(default="coherence+psd", description="Method to detect bad channels")
    dead_channel_threshold: float = Field(default=-0.5, description="Threshold for dead channel")
    noisy_channel_threshold: float = Field(default=1.0, description="Threshold for noisy channel")
    outside_channel_threshold: float = Field(default=-0.3, description="Threshold for outside channel")
    n_neighbors: int = Field(default=11, description="Number of neighbors")
    seed: int = Field(default=0, description="Seed value")


class CommonReference(BaseModel):
    reference: str = Field(default="global", description="Type of reference")
    operator: str = Field(default="median", description="Operator used for common reference")


class HighpassSpatialFilter(BaseModel):
    n_channel_pad: int = Field(default=60, description="Number of channels to pad")
    n_channel_taper: Union[int, None] = Field(default=None, description="Number of channels to taper")
    direction: str = Field(default="y", description="Direction for the spatial filter")
    apply_agc: bool = Field(default=True, description="Whether to apply automatic gain control")
    agc_window_length_s: float = Field(default=0.01, description="Window length in seconds for AGC")
    highpass_butter_order: int = Field(default=3, description="Order for the Butterworth filter")
    highpass_butter_wn: float = Field(default=0.01, description="Natural frequency for the Butterworth filter")


# ---------------------------------------------------------------
# Motion Correction Models
# ---------------------------------------------------------------
class MCDetectKwargs(BaseModel):
    method: str = Field(default="locally_exclusive", description="")
    peak_sign: str = Field(default="neg", description="")
    detect_threshold: float = Field(default=8.0, description="")
    exclude_sweep_ms: float = Field(default=0.1, description="")
    radius_um: float = Field(default=50.0, description="")


class MCLocalizeCenterOfMass(BaseModel):
    radius_um: float = Field(default=75.0, description="Radius in um for channel sparsity.")
    feature: str = Field(default="ptp", description="'ptp', 'mean', 'energy' or 'peak_voltage'. Feature to consider for computation")


class MCLocalizeMonopolarTriangulation(BaseModel):
    radius_um: float = Field(default=75.0, description="For channel sparsity.")
    max_distance_um: float = Field(default=150.0, description="Boundary for distance estimation.")
    optimizer: str = Field(default="minimize_with_log_penality", description="")
    enforce_decrease: bool = Field(default=True, description="Enforce spatial decreasingness for PTP vectors")
    feature: str = Field(default="ptp", description="'ptp', 'energy' or 'peak_voltage'. The available features to consider for estimating the position via monopolar triangulation are peak-to-peak amplitudes (ptp, default), energy ('energy', as L2 norm) or voltages at the center of the waveform (peak_voltage)")


class MCLocalizeGridConvolution(BaseModel):
    radius_um: float = Field(default=40.0, description="Radius in um for channel sparsity.")
    upsampling_um: float = Field(default=5.0, description="Upsampling resolution for the grid of templates.")
    sigma_um: List[float] = Field(default=[5.0, 25.0, 5], description="Spatial decays of the fake templates.")
    sigma_ms: float = Field(default=0.25, description="The temporal decay of the fake templates.")
    margin_um: float = Field(default=30.0, description="The margin for the grid of fake templates.")
    percentile: float = Field(default=10.0, description="The percentage in [0, 100] of the best scalar products kept to estimate the position.")
    sparsity_threshold: float = Field(default=0.01, description="The sparsity threshold (in [0, 1]) below which weights should be considered as 0.")


class MCEstimateMotionDecentralized(BaseModel):
    method: str = Field(default="decentralized", description="")
    direction: str = Field(default="y", description="")
    bin_duration_s: float = Field(default=2.0, description="")
    rigid: bool = Field(default=False, description="")
    bin_um: float = Field(default=5.0, description="")
    margin_um: float = Field(default=0.0, description="")
    win_shape: str = Field(default="gaussian", description="")
    win_step_um: float = Field(default=100.0, description="")
    win_sigma_um: float = Field(default=200.0, description="")
    histogram_depth_smooth_um: float = Field(default=5.0, description="")
    histogram_time_smooth_s: Optional[float] = Field(default=None, description="")
    pairwise_displacement_method: str = Field(default="conv", description="")
    max_displacement_um: float = Field(default=100.0, description="")
    weight_scale: str = Field(default="linear", description="")
    error_sigma: float = Field(default=0.2, description="")
    conv_engine: Optional[str] = Field(default=None, description="")
    torch_device: Optional[str] = Field(default=None, description="")
    batch_size: int = Field(default=1, description="")
    corr_threshold: float = Field(default=0.0, description="")
    time_horizon_s: Optional[float] = Field(default=None, description="")
    convergence_method: str = Field(default="lsmr", description="")
    soft_weights: bool = Field(default=False, description="")
    normalized_xcorr: bool = Field(default=True, description="")
    centered_xcorr: bool = Field(default=True, description="")
    temporal_prior: bool = Field(default=True, description="")
    spatial_prior: bool = Field(default=False, description="")
    force_spatial_median_continuity: bool = Field(default=False, description="")
    reference_displacement: str = Field(default="median", description="")
    reference_displacement_time_s: float = Field(default=0, description="")
    robust_regression_sigma: int = Field(default=2, description="")
    weight_with_amplitude: bool = Field(default=False, description="")


class MCEstimateMotionIterativeTemplate(BaseModel):
    bin_duration_s: float = Field(default=2.0, description="")
    rigid: bool = Field(default=False, description="")
    win_step_um: float = Field(default=50.0, description="")
    win_sigma_um: float = Field(default=150.0, description="")
    margin_um: float = Field(default=0.0, description="")
    win_shape: str = Field(default="rect", description="")


class MCInterpolateMotionKwargs(BaseModel):
    direction: int = Field(default=1, description="0 | 1 | 2. Dimension along which channel_locations are shifted (0 - x, 1 - y, 2 - z).")
    border_mode: str = Field(default="remove_channels", description="'remove_channels' | 'force_extrapolate' | 'force_zeros'. Control how channels are handled on border.")
    spatial_interpolation_method: str = Field(default="idw", description="The spatial interpolation method used to interpolate the channel locations.")
    sigma_um: float = Field(default=20.0, description="Used in the 'kriging' formula")
    p: int = Field(default=1, description="Used in the 'kriging' formula")
    num_closest: int = Field(default=3, description="Number of closest channels used by 'idw' method for interpolation.")


class MCNonrigidAccurate(BaseModel):
    detect_kwargs: MCDetectKwargs = Field(default=MCDetectKwargs(), description="")
    localize_peaks_kwargs: MCLocalizeMonopolarTriangulation = Field(default=MCLocalizeMonopolarTriangulation(), description="")
    estimate_motion_kwargs: MCEstimateMotionDecentralized = Field(default=MCEstimateMotionDecentralized(), description="")
    interpolate_motion_kwargs: MCInterpolateMotionKwargs = Field(default=MCInterpolateMotionKwargs(), description="")


class MCRigidFast(BaseModel):
    detect_kwargs: MCDetectKwargs = Field(default=MCDetectKwargs(), description="")
    localize_peaks_kwargs: MCLocalizeCenterOfMass = Field(default=MCLocalizeCenterOfMass(), description="")
    estimate_motion_kwargs: MCEstimateMotionDecentralized = Field(default=MCEstimateMotionDecentralized(bin_duration_s=10.0, rigid=True), description="")
    interpolate_motion_kwargs: MCInterpolateMotionKwargs = Field(default=MCInterpolateMotionKwargs(), description="")


class MCKilosortLike(BaseModel):
    detect_kwargs: MCDetectKwargs = Field(default=MCDetectKwargs(), description="")
    localize_peaks_kwargs: MCLocalizeGridConvolution = Field(default=MCLocalizeGridConvolution(), description="")
    estimate_motion_kwargs: MCEstimateMotionIterativeTemplate = Field(default=MCEstimateMotionIterativeTemplate(), description="")
    interpolate_motion_kwargs: MCInterpolateMotionKwargs = Field(default=MCInterpolateMotionKwargs(border_mode="force_extrapolate", spatial_interpolation_method="kriging"), description="")


class MotionCorrection(BaseModel):
    compute: bool = Field(default=True, description="Whether to compute motion correction")
    apply: bool = Field(default=False, description="Whether to apply motion correction")
    preset: str = Field(
        default="nonrigid_accurate",
        description="Preset for motion correction",
        json_schema_extra={'options': ["nonrigid_accurate", "rigid_fast", "kilosort_like"]},
    )
    motion_kwargs: Union[MCNonrigidAccurate, MCRigidFast, MCKilosortLike] = Field(default=MCNonrigidAccurate(), description="Motion correction parameters")


# ---------------------------------------------------------------
# Preprocessing Context
# ---------------------------------------------------------------
class PreprocessingContext(BaseModel):
    preprocessing_strategy: str = Field(default="cmr", description="Strategy for preprocessing")
    highpass_filter: HighpassFilter = Field(default=HighpassFilter(), description="Highpass filter")
    phase_shift: PhaseShift = Field(default=PhaseShift(), description="Phase shift")
    detect_bad_channels: DetectBadChannels = Field(default=DetectBadChannels(), description="Detect bad channels")
    common_reference: CommonReference = Field(default=CommonReference(), description="Common reference")
    highpass_spatial_filter: HighpassSpatialFilter = Field(default=HighpassSpatialFilter(), description="Highpass spatial filter")
    motion_correction: MotionCorrection = Field(default=MotionCorrection(), description="Motion correction")
    remove_out_channels: bool = Field(default=False, description="Flag to remove out channels")
    remove_bad_channels: bool = Field(default=False, description="Flag to remove bad channels")
    max_bad_channel_fraction_to_remove: float = Field(default=1.1, description="Maximum fraction of bad channels to remove")


# ------------------------------
# Sorter Models
# ------------------------------
class Kilosort25SortingContext(BaseModel):
    detect_threshold: float = Field(default=6, description="Threshold for spike detection")
    projection_threshold: List[int] = Field(default=[10, 4], description="Threshold on projections")
    preclust_threshold: float = Field(default=8, description="Threshold crossings for pre-clustering (in PCA projection space)")
    car: bool = Field(default=True, description="Enable or disable common reference")
    minFR: float = Field(default=0.1, description="Minimum spike rate (Hz), if a cluster falls below this for too long it gets removed")
    minfr_goodchannels: float = Field(default=0.1, description="Minimum firing rate on a 'good' channel")
    nblocks: int = Field(default=5, description="blocks for registration. 0 turns it off, 1 does rigid registration. Replaces 'datashift' option.")
    sig: float = Field(default=20, description="spatial smoothness constant for registration")
    freq_min: float = Field(default=150, description="High-pass filter cutoff frequency")
    sigmaMask: float = Field(default=30, description="Spatial constant in um for computing residual variance of spike")
    lam: float = Field(default=10.0, description="The importance of the amplitude penalty (like in Kilosort1: 0 means not used, 10 is average, 50 is a lot)")
    nPCs: int = Field(default=3, description="Number of PCA dimensions")
    ntbuff: int = Field(default=64, description="Samples of symmetrical buffer for whitening and spike detection")
    nfilt_factor: int = Field(default=4, description="Max number of clusters per good channel (even temporary ones) 4")
    # NT: int = Field(default=-1, description='Batch size (if -1 it is automatically computed)')
    AUCsplit: float = Field(default=0.9, description="Threshold on the area under the curve (AUC) criterion for performing a split in the final step")
    do_correction: bool = Field(default=True, description="If True drift registration is applied")
    wave_length: float = Field(default=61, description="size of the waveform extracted around each detected peak, (Default 61, maximum 81)")
    keep_good_only: bool = Field(default=False, description="If True only 'good' units are returned")
    skip_kilosort_preprocessing: bool = Field(default=False, description="Can optionaly skip the internal kilosort preprocessing")
    # scaleproc: int = Field(default=-1, description="int16 scaling of whitened data, if -1 set to 200.")


class SpikeSortingContext(BaseModel):
    sorter_name: str = Field(default="kilosort2_5", description="Name of the sorter to use.")
    sorter_kwargs: Kilosort25SortingContext = Field(default=Kilosort25SortingContext(), description="Sorter specific kwargs.")


# ------------------------------
# Postprocessing Models
# ------------------------------
class PostprocessingContext(BaseModel):
    pass


# ------------------------------
# Curation Models
# ------------------------------
class CurationContext(BaseModel):
    pass


# ------------------------------
# Pipeline Models
# ------------------------------
class PipelineContext(BaseModel):
    input: InputFile = Field(description='Input NWB file')
    output: OutputFile = Field(description='Output NWB file')
    lazy_read_input: bool = Field(default=True, description='Lazy read input file')
    stub_test: bool = Field(default=False, description='Stub test')
    recording_context: RecordingContext = Field(description='Recording context')
    run_preprocessing: bool = Field(default=True, description='Run preprocessing')
    preprocessing_context: PreprocessingContext = Field(default=PreprocessingContext(), description='Preprocessing context')
    run_spikesorting: bool = Field(default=True, description='Run spike sorting')
    spikesorting_context: SpikeSortingContext = Field(default=SpikeSortingContext(), description='Sorting context')
    run_postprocessing: bool = Field(default=True, description='Run postprocessing')
    # postprocessing_context: PostprocessingContext = Field(default=PostprocessingContext(), description='Postprocessing context')
    # curation_context: CurationContext = Field(default=CurationContext(), description='Curation context')


# # ------------------------------
# # Sorter Models
# # ------------------------------
# class SorterName(str, Enum):
#     ironclust = "ironclust"
#     kilosort2 = "kilosort2"
#     kilosort25 = "kilosort25"
#     kilosort3 = "kilosort3"
#     spykingcircus = "spykingcircus"


# class SorterKwargs:
#     sorter_name: SorterName = Field(description="Name of the sorter to use.")


# # ------------------------------
# # Postprocessing Models
# # ------------------------------
# class PresenceRatio:
#     bin_duration_s: float = Field(default=60, description="Duration of the bin in seconds.")


# class SNR:
#     peak_sign: str = Field(default="neg", description="Sign of the peak.")
#     peak_mode: str = Field(default="extremum", description="Mode of the peak.")
#     random_chunk_kwargs_dict: Optional[dict] = Field(default=None, description="Random chunk arguments.")


# class ISIViolation:
#     isi_threshold_ms: float = Field(default=1.5, description="ISI threshold in milliseconds.")
#     min_isi_ms: float = Field(default=0., description="Minimum ISI in milliseconds.")


# class RPViolation:
#     refractory_period_ms: float = Field(default=1., description="Refractory period in milliseconds.")
#     censored_period_ms: float = Field(default=0.0, description="Censored period in milliseconds.")


# class SlidingRPViolation:
#     min_spikes: int = Field(default=0, description="Contamination is set to np.nan if the unit has less than this many spikes across all segments.")
#     bin_size_ms: float = Field(default=0.25, description="The size of binning for the autocorrelogram in ms, by default 0.25.")
#     window_size_s: float = Field(default=1, description="Window in seconds to compute correlogram, by default 1.")
#     exclude_ref_period_below_ms: float = Field(default=0.5, description="Refractory periods below this value are excluded, by default 0.5")
#     max_ref_period_ms: float = Field(default=10, description="Maximum refractory period to test in ms, by default 10 ms.")
#     contamination_values: Optional[list] = Field(default=None, description="The contamination values to test, by default np.arange(0.5, 35, 0.5) %")


# class PeakSign(str, Enum):
#     neg = "neg"
#     pos = "pos"
#     both = "both"


# class AmplitudeCutoff:
#     peak_sign: PeakSign = Field(default="neg", description="The sign of the peaks.")
#     num_histogram_bins: int = Field(default=100, description="The number of bins to use to compute the amplitude histogram.")
#     histogram_smoothing_value: int = Field(default=3, description="Controls the smoothing applied to the amplitude histogram.")
#     amplitudes_bins_min_ratio: int = Field(default=5, description="The minimum ratio between number of amplitudes for a unit and the number of bins. If the ratio is less than this threshold, the amplitude_cutoff for the unit is set to NaN.")


# class AmplitudeMedian:
#     peak_sign: PeakSign = Field(default="neg", description="The sign of the peaks.")


# class NearestNeighbor:
#     max_spikes: int = Field(default=10000, description="The number of spikes to use, per cluster. Note that the calculation can be very slow when this number is >20000.")
#     min_spikes: int = Field(default=10, description="Minimum number of spikes.")
#     n_neighbors: int = Field(default=4, description="The number of neighbors to use.")


# class NNIsolation(NearestNeighbor):
#     n_components: int = Field(default=10, description="The number of PC components to use to project the snippets to.")
#     radius_um: int = Field(default=100, description="The radius, in um, that channels need to be within the peak channel to be included.")


# class QMParams:
#     presence_ratio: PresenceRatio
#     snr: SNR
#     isi_violation: ISIViolation
#     rp_violation: RPViolation
#     sliding_rp_violation: SlidingRPViolation
#     amplitude_cutoff: AmplitudeCutoff
#     amplitude_median: AmplitudeMedian
#     nearest_neighbor: NearestNeighbor
#     nn_isolation: NNIsolation
#     nn_noise_overlap: NNIsolation


# class QualityMetrics:
#     qm_params: QMParams = Field(description="Quality metric parameters.")
#     metric_names: List[str] = Field(description="List of metric names to compute.")
#     n_jobs: int = Field(default=1, description="Number of jobs.")


# class Sparsity:
#     method: str = Field("radius", description="Method for determining sparsity.")
#     radius_um: int = Field(100, description="Radius in micrometers for sparsity.")


# class Waveforms:
#     ms_before: float = Field(3.0, description="Milliseconds before")
#     ms_after: float = Field(4.0, description="Milliseconds after")
#     max_spikes_per_unit: int = Field(500, description="Maximum spikes per unit")
#     return_scaled: bool = Field(True, description="Flag to determine if results should be scaled")
#     dtype: Optional[str] = Field(None, description="Data type for the waveforms")
#     precompute_template: Tuple[str, str] = Field(("average", "std"), description="Precomputation template method")
#     use_relative_path: bool = Field(True, description="Use relative paths")


# class SpikeAmplitudes:
#     peak_sign: str = Field("neg", description="Sign of the peak")
#     return_scaled: bool = Field(True, description="Flag to determine if amplitudes should be scaled")
#     outputs: str = Field("concatenated", description="Output format for the spike amplitudes")


# class Similarity:
#     method: str = Field("cosine_similarity", description="Method to compute similarity")


# class Correlograms:
#     window_ms: float = Field(100.0, description="Size of the window in milliseconds")
#     bin_ms: float = Field(2.0, description="Size of the bin in milliseconds")


# class ISIS:
#     window_ms: float = Field(100.0, description="Size of the window in milliseconds")
#     bin_ms: float = Field(5.0, description="Size of the bin in milliseconds")


# class Locations:
#     method: str = Field("monopolar_triangulation", description="Method to determine locations")


# class TemplateMetrics:
#     upsampling_factor: int = Field(10, description="Upsampling factor")
#     sparsity: Optional[str] = Field(None, description="Sparsity method")


# class PrincipalComponents:
#     n_components: int = Field(5, description="Number of principal components")
#     mode: str = Field("by_channel_local", description="Mode of principal component analysis")
#     whiten: bool = Field(True, description="Whiten the components")


# class PostprocessingKwargs:
#     sparsity: Sparsity
#     waveforms_deduplicate: Waveforms
#     waveforms: Waveforms
#     spike_amplitudes: SpikeAmplitudes
#     similarity: Similarity
#     correlograms: Correlograms
#     isis: ISIS
#     locations: Locations
#     template_metrics: TemplateMetrics
#     principal_components: PrincipalComponents
#     quality_metrics: QualityMetrics


# # ------------------------------
# # Curation Models
# # ------------------------------
# class CurationKwargs:
#     duplicate_threshold: float = Field(0.9, description="Threshold for duplicate units")
#     isi_violations_ratio_threshold: float = Field(0.5, description="Threshold for ISI violations ratio")
#     presence_ratio_threshold: float = Field(0.8, description="Threshold for presence ratio")
#     amplitude_cutoff_threshold: float = Field(0.1, description="Threshold for amplitude cutoff")


# # ------------------------------
# # Visualization Models
# # ------------------------------
# class Timeseries:
#     n_snippets_per_segment: int = Field(2, description="Number of snippets per segment")
#     snippet_duration_s: float = Field(0.5, description="Duration of the snippet in seconds")
#     skip: bool = Field(False, description="Flag to skip")


# class Detection:
#     method: str = Field("locally_exclusive", description="Method for detection")
#     peak_sign: str = Field("neg", description="Sign of the peak")
#     detect_threshold: int = Field(5, description="Detection threshold")
#     exclude_sweep_ms: float = Field(0.1, description="Exclude sweep in milliseconds")


# class Localization:
#     ms_before: float = Field(0.1, description="Milliseconds before")
#     ms_after: float = Field(0.3, description="Milliseconds after")
#     local_radius_um: float = Field(100.0, description="Local radius in micrometers")


# class Drift:
#     detection: Detection
#     localization: Localization
#     n_skip: int = Field(30, description="Number of skips")
#     alpha: float = Field(0.15, description="Alpha value")
#     vmin: int = Field(-200, description="Minimum value")
#     vmax: int = Field(0, description="Maximum value")
#     cmap: str = Field("Greys_r", description="Colormap")
#     figsize: Tuple[int, int] = Field((10, 10), description="Figure size")


# class VisualizationKwargs:
#     timeseries: Timeseries
#     drift: Drift
