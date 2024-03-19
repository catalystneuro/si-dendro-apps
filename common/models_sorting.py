from pydantic import BaseModel, Field
from typing import List, Union


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


class MountainSort5SortingContext(BaseModel):
    scheme: str = Field(
        default='2',
        description="Sorting scheme",
        json_schema_extra={'options': ["1", "2", "3"]}
    )
    detect_threshold: float = Field(default=5.5, description="Threshold for spike detection")
    detect_sign: int = Field(default=-1, description="Sign of the peak")
    detect_time_radius_msec: float = Field(default=0.5, description="Time radius in milliseconds")
    snippet_T1: int = Field(default=20, description="Snippet T1")
    snippet_T2: int = Field(default=20, description="Snippet T2")
    npca_per_channel: int = Field(default=3, description="Number of PCA per channel")
    npca_per_subdivision: int = Field(default=10, description="Number of PCA per subdivision")
    snippet_mask_radius: int = Field(default=250, description="Snippet mask radius")
    scheme1_detect_channel_radius: int = Field(default=150, description="Scheme 1 detect channel radius")
    scheme2_phase1_detect_channel_radius: int = Field(default=200, description="Scheme 2 phase 1 detect channel radius")
    scheme2_detect_channel_radius: int = Field(default=50, description="Scheme 2 detect channel radius")
    scheme2_max_num_snippets_per_training_batch: int = Field(default=200, description="Scheme 2 max number of snippets per training batch")
    scheme2_training_duration_sec: int = Field(default=300, description="Scheme 2 training duration in seconds")
    scheme2_training_recording_sampling_mode: str = Field(default='uniform', description="Scheme 2 training recording sampling mode")
    scheme3_block_duration_sec: int = Field(default=1800, description="Scheme 3 block duration in seconds")
    freq_min: int = Field(default=300, description="High-pass filter cutoff frequency")
    freq_max: int = Field(default=6000, description="Low-pass filter cutoff frequency")
    filter: bool = Field(default=True, description="Enable or disable filter")
    whiten: bool = Field(default=True, description="Enable or disable whiten")


class SpykingCircusModel(BaseModel):
    detect_sign: int = Field(default=-1, description="Sign of the peak")
    adjacency_radius: int = Field(default=100, description="Adjacency radius")
    detect_threshold: float = Field(default=6, description="Threshold for spike detection")
    template_width_ms: int = Field(default=3, description="Template width in milliseconds")
    filter: bool = Field(default=True, description="Enable or disable filter")
    merge_spikes: bool = Field(default=True, description="Enable or disable merge spikes")
    auto_merge: float = Field(default=0.75, description="Auto merge")
    num_workers: Union[int, None] = Field(default=None, description="Number of workers")
    whitening_max_elts: int = Field(default=1000, description="Whitening max elements")
    clustering_max_elts: int = Field(default=10000, description="Clustering max elements")
