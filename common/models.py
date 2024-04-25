from dendro.sdk import InputFile, OutputFile
from pydantic import BaseModel, Field
from typing import Union

from .models_preprocessing import PreprocessingContext
from .models_postprocessing import PostprocessingContext
from .models_curation import CurationContext
from .models_visualization import VisualizationContext
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
    stub_test_num_channels: int = Field(default=-1, description='Stub test number of channels')


class JobKwargs(BaseModel):
    n_jobs: float = Field(default=0.8, description='Number of jobs, must be a positive number between 0 and 1, or -1 for all processors.')
    chunk_duration: str = Field(default='1s', description='Chunk duration.')
    progress_bar: bool = Field(default=False, description='Show progress bar.')
    mp_context: str = Field(
        default="spawn",
        description='Context for multiprocessing. It can be "fork" or "spawn".',
        json_schema_extra={'options': ["fork", "spawn"]},
    )


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
    visualization_context: VisualizationContext = Field(default=VisualizationContext(), description='Visualization context')
