from spikeinterface_pipelines import pipeline as si_pipeline
from spikeinterface.extractors import NwbRecordingExtractor
from neuroconv.tools.spikeinterface import write_waveforms, write_sorting
from pathlib import Path
import os
import pynwb
import h5py
import logging

from .models import PipelineFullContext
from .nwb_utils import create_base_nwb_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def run_pipeline(context: PipelineFullContext):
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
    download = not context.recording_context.lazy_read_input
    ff = context.input.get_file(download=download)

    logger.info('Creating input recording')
    recording = NwbRecordingExtractor(
        file=ff,
        electrical_series_path=context.recording_context.electrical_series_path,
        # file_path=context.input.get_url(),
        # stream_mode="remfile"
    )

    if context.recording_context.stub_test:
        logger.info('Running in stub test mode')
        stub_test_num_frames = context.recording_context.stub_test_duration_sec * recording.get_sampling_frequency()
        n_frames = int(min(stub_test_num_frames, recording.get_num_frames()))
        recording = recording.frame_slice(start_frame=0, end_frame=n_frames)

    logger.info(recording)

    if context.recording_context.write_recording:
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
    spikesorting_params = dict()
    spikesorting_params["sorter_name"] = context.sorter_name
    spikesorting_params["sorter_kwargs"] = context.spikesorting_context.model_dump()

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

    # Curation params
    run_curation = context.run_curation
    curation_params = context.curation_context.model_dump()

    # Visualization params
    run_visualization = context.run_visualization

    # Run pipeline
    logger.info('Running pipeline')
    recording_preprocessed, sorting, waveform_extractor, sorting_curated, visualization_output = si_pipeline.run_pipeline(
        recording=recording,
        scratch_folder="./scratch/",
        results_folder="./results/",
        job_kwargs=job_kwargs,
        preprocessing_params=preprocessing_params,
        spikesorting_params=spikesorting_params,
        postprocessing_params=postprocessing_params,
        curation_params=curation_params,
        # visualization_params=,
        run_preprocessing=run_preprocessing,
        run_spikesorting=run_spikesorting,
        run_postprocessing=run_postprocessing,
        run_curation=run_curation,
        run_visualization=run_visualization
    )

    # Upload output file
    if sorting:
        logger.info('Writing output NWB file')
        if not os.path.exists('output'):
            os.mkdir('output')

        output_fname = 'output/output.nwb'

        h5_file = h5py.File(ff, 'r')
        with pynwb.NWBHDF5IO(file=h5_file, mode='r', load_namespaces=True) as io:
            nwbfile_rec = io.read()

            nwbfile_base = create_base_nwb_file(nwbfile_original=nwbfile_rec)

        if waveform_extractor is not None:
            write_waveforms(
                waveform_extractor=waveform_extractor,
                nwbfile_path=Path(output_fname).resolve(),
                nwbfile=nwbfile_base,
                write_as="processing",
                # metadata=metadata_dict,
            )
        else:
            write_sorting(
                sorting=sorting,
                nwbfile_path=Path(output_fname).resolve(),
                nwbfile=nwbfile_base,
                write_as="processing",
                # metadata=metadata_dict,
            )

        # h5_file = h5py.File(ff, 'r')
        # with pynwb.NWBHDF5IO(file=h5_file, mode='r', load_namespaces=True) as io:
        #     nwbfile_rec = io.read()

        #     if not os.path.exists('output'):
        #         os.mkdir('output')

        #     output_fname = 'output/output.nwb'

        #     create_sorting_out_nwb_file(
        #         nwbfile_original=nwbfile_rec,
        #         recording=recording_preprocessed,
        #         sorting=sorting_curated if run_curation else sorting,
        #         output_fname=output_fname
        #     )

        logger.info('Uploading output NWB file')
        context.output.upload(output_fname)
