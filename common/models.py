from dendro.sdk import InputFile, OutputFile
from pydantic import BaseModel, Field
from typing import List, Union

from .models_preprocessing import PreprocessingContext
from .models_postprocessing import PostprocessingContext


class RecordingContext(BaseModel):
    electrical_series_path: str = Field(description='Path to the electrical series in the NWB file')


class JobKwargs(BaseModel):
    n_jobs: float = Field(default=0.8, description='Number of jobs, must be a positive number between 0 and 1, or -1 for all processors.')
    chunk_duration: str = Field(default='1s', description='Chunk duration.')
    progress_bar: bool = Field(default=False, description='Show progress bar.')


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
    AUCsplit: float = Field(default=0.9, description="Threshold on the area under the curve (AUC) criterion for performing a split in the final step")
    do_correction: bool = Field(default=True, description="If True drift registration is applied")
    wave_length: float = Field(default=61, description="size of the waveform extracted around each detected peak, (Default 61, maximum 81)")
    keep_good_only: bool = Field(default=False, description="If True only 'good' units are returned")
    skip_kilosort_preprocessing: bool = Field(default=False, description="Can optionaly skip the internal kilosort preprocessing")


class Kilosort3SortingContext(BaseModel):
    detect_threshold: float = Field(default=6, description="Threshold for spike detection")
    projection_threshold: List[int] = Field(default=[9, 9], description="Threshold on projections")
    preclust_threshold: float = Field(default=8, description="Threshold crossings for pre-clustering (in PCA projection space)")
    car: bool = Field(default=True, description="Enable or disable common reference")
    minFR: float = Field(default=0.2, description="Minimum spike rate (Hz), if a cluster falls below this for too long it gets removed")
    minfr_goodchannels: float = Field(default=0.2, description="Minimum firing rate on a 'good' channel")
    nblocks: int = Field(default=5, description="blocks for registration. 0 turns it off, 1 does rigid registration. Replaces 'datashift' option.")
    sig: float = Field(default=20, description="spatial smoothness constant for registration")
    freq_min: float = Field(default=300, description="High-pass filter cutoff frequency")
    sigmaMask: float = Field(default=30, description="Spatial constant in um for computing residual variance of spike")
    lam: float = Field(default=20.0, description="The importance of the amplitude penalty (like in Kilosort1: 0 means not used, 10 is average, 50 is a lot)")
    nPCs: int = Field(default=3, description="Number of PCA dimensions")
    ntbuff: int = Field(default=64, description="Samples of symmetrical buffer for whitening and spike detection")
    nfilt_factor: int = Field(default=4, description="Max number of clusters per good channel (even temporary ones) 4")
    AUCsplit: float = Field(default=0.8, description="Threshold on the area under the curve (AUC) criterion for performing a split in the final step")
    do_correction: bool = Field(default=True, description="If True drift registration is applied")
    wave_length: float = Field(default=61, description="size of the waveform extracted around each detected peak, (Default 61, maximum 81)")
    keep_good_only: bool = Field(default=False, description="If True only 'good' units are returned")
    skip_kilosort_preprocessing: bool = Field(default=False, description="Can optionaly skip the internal kilosort preprocessing")


# ------------------------------
# Pipeline Models
# ------------------------------
class PipelineContext(BaseModel):
    input: InputFile = Field(description='Input NWB file')
    output: OutputFile = Field(description='Output NWB file')
    lazy_read_input: bool = Field(default=True, description='Lazy read input file')
    stub_test: bool = Field(default=False, description='Stub test')
    job_kwargs: JobKwargs = Field(default=JobKwargs(), description='Job kwargs')
    recording_context: RecordingContext = Field(description='Recording context')
    run_preprocessing: bool = Field(default=True, description='Run preprocessing')
    preprocessing_context: PreprocessingContext = Field(default=PreprocessingContext(), description='Preprocessing context')
    run_spikesorting: bool = Field(default=True, description='Run spike sorting')
    spikesorting_context: Union[
        Kilosort25SortingContext,
        Kilosort3SortingContext,
    ] = Field(description='Sorting context')
    run_postprocessing: bool = Field(default=True, description='Run postprocessing')
    postprocessing_context: PostprocessingContext = Field(default=PostprocessingContext(), description='Postprocessing context')
    # curation_context: CurationContext = Field(default=CurationContext(), description='Curation context')




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
