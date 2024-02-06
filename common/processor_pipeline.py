from spikeinterface_pipelines import pipeline as si_pipeline
from spikeinterface.extractors import NwbRecordingExtractor
from pathlib import Path
import os
import pynwb
import h5py
import logging

from .models import PipelineContext
from .nwb_utils import create_sorting_out_nwb_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def run_pipeline(context: PipelineContext):
    """
    Runs the spikeinterface pipeline.

    Args:
        context (PipelineContext): Pipeline context model.
    """
    # Create folders
    results_folder = Path("./results/")
    results_folder.mkdir(exist_ok=True, parents=True)
    scratch_folder = Path("./scratch/")
    scratch_folder.mkdir(exist_ok=True, parents=True)

    # Create SI recording from InputFile
    logger.info('Opening remote input file')
    download = not context.lazy_read_input
    ff = context.input.get_file(download=download)

    logger.info('Creating input recording')
    recording = NwbRecordingExtractor(
        file=ff,
        electrical_series_location=context.recording_context.electrical_series_path,
        # file_path=context.input.get_url(),
        # stream_mode="remfile"
    )

    if context.stub_test:
        logger.info('Running in stub test mode')
        stub_test_num_frames = context.stub_test_duration_sec * recording.get_sampling_frequency()
        n_frames = int(min(stub_test_num_frames, recording.get_num_frames()))
        recording = recording.frame_slice(start_frame=0, end_frame=n_frames)

    logger.info(recording)

    if context.write_recording:
        logger.info('Writing binary recording')
        recording = recording.save(folder=scratch_folder / "recording")

    # Job kwargs
    job_kwargs = context.job_kwargs.model_dump()

    # Preprocessing params
    run_preprocessing = context.run_preprocessing
    preprocessing_params = context.preprocessing_context.model_dump()
    motion_correction_preset = preprocessing_params['motion_correction']['preset']
    nonrigid_accurate_kwargs = preprocessing_params['motion_correction'].pop('motion_kwargs_nonrigid_accurate')
    rigid_fast_kwargs = preprocessing_params['motion_correction'].pop('motion_kwargs_rigid_fast')
    kilosort_like_kwargs = preprocessing_params['motion_correction'].pop('motion_kwargs_kilosort_like')
    if motion_correction_preset == 'nonrigid_accurate':
        preprocessing_params['motion_correction']['motion_kwargs'] = nonrigid_accurate_kwargs
    elif motion_correction_preset == 'rigid_fast':
        preprocessing_params['motion_correction']['motion_kwargs'] = rigid_fast_kwargs
    elif motion_correction_preset == 'kilosort_like':
        preprocessing_params['motion_correction']['motion_kwargs'] = kilosort_like_kwargs

    # Spikesorting params
    run_spikesorting = context.run_spikesorting
    spikesorting_params = context.spikesorting_context.model_dump()

    # Postprocessing params
    run_postprocessing = context.run_postprocessing
    postprocessing_params = context.postprocessing_context.model_dump()
    qm_list = list()
    if postprocessing_params['quality_metrics'].pop('presence_ratio'):
        qm_list.append('presence_ratio')
    if postprocessing_params['quality_metrics'].pop('snr'):
        qm_list.append('snr')
    if postprocessing_params['quality_metrics'].pop('isi_violation'):
        qm_list.append('isi_violation')
    if postprocessing_params['quality_metrics'].pop('rp_violation'):
        qm_list.append('rp_violation')
    if postprocessing_params['quality_metrics'].pop('sliding_rp_violation'):
        qm_list.append('sliding_rp_violation')
    if postprocessing_params['quality_metrics'].pop('amplitude_cutoff'):
        qm_list.append('amplitude_cutoff')
    if postprocessing_params['quality_metrics'].pop('amplitude_median'):
        qm_list.append('amplitude_median')
    if postprocessing_params['quality_metrics'].pop('nearest_neighbor'):
        qm_list.append('nearest_neighbor')
    if postprocessing_params['quality_metrics'].pop('nn_isolation'):
        qm_list.append('nn_isolation')
    if postprocessing_params['quality_metrics'].pop('nn_noise_overlap'):
        qm_list.append('nn_noise_overlap')
    postprocessing_params['quality_metrics']['metric_names'] = qm_list

    # Run pipeline
    logger.info('Running pipeline')
    _, sorting, _ = si_pipeline.run_pipeline(
        recording=recording,
        scratch_folder="./scratch/",
        results_folder="./results/",
        job_kwargs=job_kwargs,
        run_preprocessing=run_preprocessing,
        preprocessing_params=preprocessing_params,
        run_spikesorting=run_spikesorting,
        spikesorting_params=spikesorting_params,
        run_postprocessing=run_postprocessing,
        postprocessing_params=postprocessing_params,
    )

    # Upload output file
    if sorting:
        logger.info('Writing output NWB file')
        h5_file = h5py.File(ff, 'r')
        with pynwb.NWBHDF5IO(file=h5_file, mode='r', load_namespaces=True) as io:
            nwbfile_rec = io.read()

            if not os.path.exists('output'):
                os.mkdir('output')
            sorting_out_fname = 'output/sorting.nwb'

            create_sorting_out_nwb_file(
                nwbfile_original=nwbfile_rec,
                sorting=sorting,
                sorting_out_fname=sorting_out_fname
            )

        logger.info('Uploading output NWB file')
        context.output.upload(sorting_out_fname)
