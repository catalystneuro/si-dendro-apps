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
        # Limit the number of frames to the stub test duration
        stub_test_num_frames = context.recording_context.stub_test_duration_sec * recording.get_sampling_frequency()
        n_frames = int(min(stub_test_num_frames, recording.get_num_frames()))
        recording = recording.frame_slice(start_frame=0, end_frame=n_frames)
        # Limit the number of channels to the stub test range
        channel_ids = recording.channel_ids
        stub_test_channels_ids = [channel_ids[int(a)] for a in context.recording_context.stub_test_channels.split(',')]
        recording = recording.channel_slice(channel_ids=stub_test_channels_ids)

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
    qm_names = [
        'presence_ratio',
        'snr',
        'isi_violation',
        'rp_violation',
        'sliding_rp_violation',
        'amplitude_cutoff',
        'amplitude_median',
        'nearest_neighbor',
        'nn_isolation',
        'nn_noise_overlap'
    ]
    for qm_name in qm_names:
        if postprocessing_params['quality_metrics'].pop(qm_name):
            qm_list.append(qm_name)
    postprocessing_params['quality_metrics']['metric_names'] = qm_list

    # Curation params
    run_curation = context.run_curation
    curation_params = context.curation_context.model_dump()

    # Visualization params
    run_visualization = context.run_visualization
    visualization_params = context.visualization_context.model_dump()
    utp = visualization_params['sorting_summary'].pop('unit_table_properties', '')
    visualization_params['sorting_summary']['unit_table_properties'] = [a.strip() for a in utp.split(',')]
    lcs = visualization_params['sorting_summary'].pop('label_choices', '')
    visualization_params['sorting_summary']['label_choices'] = [a.strip() for a in lcs.split(',')]

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
        visualization_params=visualization_params,
        run_preprocessing=run_preprocessing,
        run_spikesorting=run_spikesorting,
        run_postprocessing=run_postprocessing,
        run_curation=run_curation,
        run_visualization=run_visualization
    )

    # FOR TESTING ONLY, REMOVE LATER --------------------------------
    # dump visualization output dict to json file, using json
    if run_visualization:
        import json
        logger.info('Writing visualization output')
        if not os.path.exists('output'):
            os.mkdir('output')
        with open('output/visualization_output.json', 'w') as f:
            json.dump(visualization_output, f)
    # ---------------------------------------------------------------

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

        logger.info('Uploading output NWB file')
        context.output.upload(output_fname)
