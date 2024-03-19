from dendro.sdk import InputFile, OutputFile
from pydantic import BaseModel, Field
from typing import Union

from .models_preprocessing import PreprocessingContext
from .models_postprocessing import PostprocessingContext
from .models_curation import CurationContext
from .models_sorting import (
    Kilosort25SortingContext,
    Kilosort3SortingContext,
    MountainSort5SortingContext,
    SpykingCircusModel,
)


class RecordingContext(BaseModel):
    electrical_series_path: str = Field(description='Path to the electrical series in the NWB file')
    lazy_read_input: bool = Field(default=True, description='Lazy read input file')
    write_recording: bool = Field(default=False, description='Write recording')
    stub_test: bool = Field(default=False, description='Stub test')
    stub_test_duration_sec: float = Field(default=300, description='Stub test duration in seconds')


class JobKwargs(BaseModel):
    n_jobs: float = Field(default=0.8, description='Number of jobs, must be a positive number between 0 and 1, or -1 for all processors.')
    chunk_duration: str = Field(default='1s', description='Chunk duration.')
    progress_bar: bool = Field(default=False, description='Show progress bar.')


# ------------------------------
# Pipeline Models
# ------------------------------
class PipelinePreprocessingContext(BaseModel):
    input: InputFile = Field(description='Input NWB file')
    output: OutputFile = Field(description='Output NWB file')
    job_kwargs: JobKwargs = Field(default=JobKwargs(), description='Job kwargs')
    recording_context: RecordingContext = Field(description='Recording context')
    preprocessing_context: PreprocessingContext = Field(default=PreprocessingContext())


class PipelineFullContext(BaseModel):
    input: InputFile = Field(description='Input NWB file')
    output: OutputFile = Field(description='Output NWB file')
    job_kwargs: JobKwargs = Field(default=JobKwargs(), description='Job kwargs')
    recording_context: RecordingContext = Field(description='Recording context')
    run_preprocessing: bool = Field(default=True, description='Run preprocessing')
    preprocessing_context: PreprocessingContext = Field(default=PreprocessingContext(), description='Preprocessing context')
    run_spikesorting: bool = Field(default=True, description='Run spike sorting')
    sorter_name: str = Field(
        default='mountainsort5',
        description="Name of the sorter to use.",
        json_schema_extra={'options': ["kilosort2_5", "kilosort3", "mountainsort5"]}
    )
    spikesorting_context: Union[
        Kilosort25SortingContext,
        Kilosort3SortingContext,
        MountainSort5SortingContext,
        SpykingCircusModel,
    ] = Field(description='Sorting context', union_mode="left_to_right")
    run_postprocessing: bool = Field(default=True, description='Run postprocessing')
    postprocessing_context: PostprocessingContext = Field(default=PostprocessingContext(), description='Postprocessing context')
    run_curation: bool = Field(default=True, description='Run curation')
    curation_context: CurationContext = Field(default=CurationContext(), description='Curation context')
    run_visualization: bool = Field(default=True, description='Run visualization')
    # visualization_context: VisualizationContext = Field(default=VisualizationContext(), description='Visualization context')


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
