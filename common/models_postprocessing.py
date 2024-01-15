from pydantic import BaseModel, Field


class PresenceRatio(BaseModel):
    bin_duration_s: float = Field(default=60, description="Duration of the bin in seconds.")


class SNR(BaseModel):
    peak_sign: str = Field(
        default="neg",
        description="Sign of the peaks.",
        json_schema_extra={'options': ["neg", "pos", "both"]},
    )
    peak_mode: str = Field(default="extremum", description="Mode of the peak.")


class ISIViolation(BaseModel):
    isi_threshold_ms: float = Field(default=1.5, description="ISI threshold in milliseconds.")
    min_isi_ms: float = Field(default=0.0, description="Minimum ISI in milliseconds.")


class RPViolation(BaseModel):
    refractory_period_ms: float = Field(default=1.0, description="Refractory period in milliseconds.")
    censored_period_ms: float = Field(default=0.0, description="Censored period in milliseconds.")


class SlidingRPViolation(BaseModel):
    bin_size_ms: float = Field(
        default=0.25, description="The size of binning for the autocorrelogram in ms, by default 0.25."
    )
    window_size_s: float = Field(default=1, description="Window in seconds to compute correlogram, by default 1.")
    exclude_ref_period_below_ms: float = Field(
        default=0.5, description="Refractory periods below this value are excluded, by default 0.5"
    )
    max_ref_period_ms: float = Field(
        default=10, description="Maximum refractory period to test in ms, by default 10 ms."
    )


class AmplitudeCutoff(BaseModel):
    peak_sign: str = Field(
        default="neg",
        description="Sign of the peaks.",
        json_schema_extra={'options': ["neg", "pos", "both"]},
    )
    num_histogram_bins: int = Field(
        default=100, description="The number of bins to use to compute the amplitude histogram."
    )
    histogram_smoothing_value: int = Field(
        default=3, description="Controls the smoothing applied to the amplitude histogram."
    )
    amplitudes_bins_min_ratio: int = Field(
        default=5,
        description="The minimum ratio between number of amplitudes for a unit and the number of bins. If the ratio is less than this threshold, the amplitude_cutoff for the unit is set to NaN.",
    )


class AmplitudeMedian(BaseModel):
    peak_sign: str = Field(
        default="neg",
        description="Sign of the peaks.",
        json_schema_extra={'options': ["neg", "pos", "both"]},
    )


class NearestNeighbor(BaseModel):
    max_spikes: int = Field(
        default=10000,
        description="The number of spikes to use, per cluster. Note that the calculation can be very slow when this number is >20000.",
    )
    min_spikes: int = Field(default=10, description="Minimum number of spikes.")
    n_neighbors: int = Field(default=4, description="The number of neighbors to use.")


class NNIsolation(NearestNeighbor):
    n_components: int = Field(default=10, description="The number of PC components to use to project the snippets to.")
    radius_um: int = Field(
        default=100, description="The radius, in um, that channels need to be within the peak channel to be included."
    )


class QMParams(BaseModel):
    presence_ratio: PresenceRatio = Field(default=PresenceRatio(), description="Presence ratio.")
    snr: SNR = Field(default=SNR(), description="Signal to noise ratio.")
    isi_violation: ISIViolation = Field(default=ISIViolation(), description="ISI violation.")
    rp_violation: RPViolation = Field(default=RPViolation(), description="Refractory period violation.")
    sliding_rp_violation: SlidingRPViolation = Field(
        default=SlidingRPViolation(), description="Sliding refractory period violation."
    )
    amplitude_cutoff: AmplitudeCutoff = Field(default=AmplitudeCutoff(), description="Amplitude cutoff.")
    amplitude_median: AmplitudeMedian = Field(default=AmplitudeMedian(), description="Amplitude median.")
    nearest_neighbor: NearestNeighbor = Field(default=NearestNeighbor(), description="Nearest neighbor.")
    nn_isolation: NNIsolation = Field(default=NNIsolation(), description="Nearest neighbor isolation.")
    nn_noise_overlap: NNIsolation = Field(default=NNIsolation(), description="Nearest neighbor noise overlap.")


class QualityMetrics(BaseModel):
    presence_ratio: bool = Field(default=True, description="Presence ratio.")
    snr: bool = Field(default=True, description="Signal to noise ratio.")
    isi_violation: bool = Field(default=True, description="ISI violation.")
    rp_violation: bool = Field(default=True, description="Refractory period violation.")
    sliding_rp_violation: bool = Field(default=True, description="Sliding refractory period violation.")
    amplitude_cutoff: bool = Field(default=True, description="Amplitude cutoff.")
    amplitude_median: bool = Field(default=True, description="Amplitude median.")
    nearest_neighbor: bool = Field(default=True, description="Nearest neighbor.")
    nn_isolation: bool = Field(default=True, description="Nearest neighbor isolation.")
    nn_noise_overlap: bool = Field(default=True, description="Nearest neighbor noise overlap.")
    qm_params: QMParams = Field(default=QMParams(), description="Quality metric parameters.")
    n_jobs: int = Field(default=1, description="Number of jobs.")


class Sparsity(BaseModel):
    method: str = Field(default="radius", description="Method for determining sparsity.")
    radius_um: int = Field(default=100, description="Radius in micrometers for sparsity.")


class WaveformsRaw(BaseModel):
    ms_before: float = Field(default=1.0, description="Milliseconds before")
    ms_after: float = Field(default=2.0, description="Milliseconds after")
    max_spikes_per_unit: int = Field(default=100, description="Maximum spikes per unit")
    return_scaled: bool = Field(default=True, description="Flag to determine if results should be scaled")
    use_relative_path: bool = Field(default=True, description="Use relative paths")


class Waveforms(BaseModel):
    ms_before: float = Field(default=3.0, description="Milliseconds before")
    ms_after: float = Field(default=4.0, description="Milliseconds after")
    max_spikes_per_unit: int = Field(default=500, description="Maximum spikes per unit")
    return_scaled: bool = Field(default=True, description="Flag to determine if results should be scaled")
    use_relative_path: bool = Field(default=True, description="Use relative paths")


class SpikeAmplitudes(BaseModel):
    peak_sign: str = Field(
        default="neg",
        description="Sign of the peaks.",
        json_schema_extra={'options': ["neg", "pos", "both"]},
    )
    return_scaled: bool = Field(default=True, description="Flag to determine if amplitudes should be scaled")
    outputs: str = Field(default="concatenated", description="Output format for the spike amplitudes")


class Similarity(BaseModel):
    method: str = Field(default="cosine_similarity", description="Method to compute similarity")


class Correlograms(BaseModel):
    window_ms: float = Field(default=100.0, description="Size of the window in milliseconds")
    bin_ms: float = Field(default=2.0, description="Size of the bin in milliseconds")


class ISIS(BaseModel):
    window_ms: float = Field(default=100.0, description="Size of the window in milliseconds")
    bin_ms: float = Field(default=5.0, description="Size of the bin in milliseconds")


class Locations(BaseModel):
    method: str = Field(default="monopolar_triangulation", description="Method to determine locations")


class TemplateMetrics(BaseModel):
    upsampling_factor: int = Field(default=10, description="Upsampling factor")


class PrincipalComponents(BaseModel):
    n_components: int = Field(default=5, description="Number of principal components")
    mode: str = Field(default="by_channel_local", description="Mode of principal component analysis")
    whiten: bool = Field(default=True, description="Whiten the components")


class PostprocessingContext(BaseModel):
    sparsity: Sparsity = Field(default=Sparsity(), description="Sparsity")
    waveforms_raw: WaveformsRaw = Field(default=WaveformsRaw(), description="Waveforms raw")
    waveforms: Waveforms = Field(default=Waveforms(), description="Waveforms")
    spike_amplitudes: SpikeAmplitudes = Field(default=SpikeAmplitudes(), description="Spike amplitudes")
    similarity: Similarity = Field(default=Similarity(), description="Similarity")
    correlograms: Correlograms = Field(default=Correlograms(), description="Correlograms")
    isis: ISIS = Field(default=ISIS(), description="ISIS")
    locations: Locations = Field(default=Locations(), description="Locations")
    template_metrics: TemplateMetrics = Field(default=TemplateMetrics(), description="Template metrics")
    principal_components: PrincipalComponents = Field(default=PrincipalComponents(), description="Principal components")
    quality_metrics: QualityMetrics = Field(default=QualityMetrics(), description="Quality metrics")
    duplicate_threshold: float = Field(default=0.9, description="Duplicate threshold")
