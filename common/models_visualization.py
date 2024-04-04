from pydantic import BaseModel, Field


class TracesParams(BaseModel):
    n_snippets_per_segment: int = Field(default=2, description="Number of snippets per segment to visualize.")
    snippet_duration_s: float = Field(default=0.5, description="Duration of each snippet in seconds.")
    skip: bool = Field(default=False, description="Skip traces visualization.")


class DetectionParams(BaseModel):
    peak_sign: str = Field(
        default="neg",
        description="Peak sign for peak detection.",
        json_schema_extra={"options": ["neg", "pos", "both"]},
    )
    detect_threshold: float = Field(default=5.0, description="Threshold for peak detection.")
    exclude_sweep_ms: float = Field(default=0.1, description="Exclude sweep in ms around peak detection.")


class LocalizationParams(BaseModel):
    ms_before: float = Field(default=0.1, description="Time before peak in ms.")
    ms_after: float = Field(default=0.3, description="Time after peak in ms.")
    radius_um: float = Field(default=100.0, description="Radius in um for sparsifying waveforms before localization.")


class DriftParams(BaseModel):
    detection: DetectionParams = Field(
        default=DetectionParams(),
        description="Detection parameters (only used if spike localization was not performed in postprocessing)",
    )
    localization: LocalizationParams = Field(
        default=LocalizationParams(),
        description="Localization parameters (only used if spike localization was not performed in postprocessing)",
    )
    decimation_factor: int = Field(
        default=30,
        description="The decimation factor for drift visualization. E.g. 30 means that 1 out of 30 spikes is plotted.",
    )
    alpha: float = Field(default=0.15, description="Alpha for scatter plot.")
    vmin: float = Field(default=-200, description="Min value for colormap.")
    vmax: float = Field(default=0, description="Max value for colormap.")
    cmap: str = Field(default="Greys_r", description="Matplotlib colormap for drift visualization.")
    # figsize: Union[list, tuple] = Field(default=(10, 10), description="Figure size for drift visualization.")


class SortingSummaryVisualizationParams(BaseModel):
    unit_table_properties: str = Field(
        default="default_qc", description="Comma-separated list of properties to show in the unit table."
    )
    curation: bool = Field(default=True, description="Whether to show curation buttons.")
    label_choices: str = Field(
        default="SUA, MUA, noise", description="Comma-separated list of labels to choose from (if `curation=True`)"
    )
    label: str = Field(default="Sorting summary from SI pipelines", description="Label for the sorting summary.")


class RecordingVisualizationParams(BaseModel):
    timeseries: TracesParams = Field(default=TracesParams(), description="Traces visualization parameters.")
    drift: DriftParams = Field(default=DriftParams(), description="Drift visualization parameters.")
    label: str = Field(default="Recording visualization from SI pipelines", description="Label for the recording.")


class VisualizationContext(BaseModel):
    recording: RecordingVisualizationParams = Field(
        default=RecordingVisualizationParams(), description="Recording visualization parameters."
    )
    sorting_summary: SortingSummaryVisualizationParams = Field(
        default=SortingSummaryVisualizationParams(), description="Sorting summary visualization parameters."
    )
