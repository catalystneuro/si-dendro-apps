from dendro.sdk import ProcessorBase
from spikeinterface_pipelines import pipeline as si_pipeline
from spikeinterface.extractors import NwbRecordingExtractor
import os
import pynwb
import h5py
import logging

from models import PipelineContext
from nwb_utils import create_sorting_out_nwb_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineProcessor(ProcessorBase):
    name = 'spikeinterface_pipeline_ks25'
    label = 'SpikeInterface Pipeline - Kilosort 2.5'
    description = "SpikeInterface Pipeline Processor for Kilosort 2.5"
    tags = ['spike_sorting', 'spike_interface', 'electrophysiology', 'pipeline']
    attributes = {
        'wip': True
    }

    @staticmethod
    def run(context: PipelineContext):

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
            n_frames = int(min(300_000, recording.get_num_frames()))
            recording = recording.frame_slice(start_frame=0, end_frame=n_frames)

        logger.info(recording)

        # TODO - run pipeline
        job_kwargs = {
            'n_jobs': -1,
            'chunk_duration': '1s',
            'progress_bar': False
        }

        run_preprocessing = context.run_preprocessing
        preprocessing_params = context.preprocessing_context.model_dump()

        run_spikesorting = context.run_spikesorting
        spikesorting_params = context.spikesorting_context.model_dump()

        run_postprocessing = context.run_postprocessing
        # postprocessing_params = context.postprocessing_context.model_dump()

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
            # postprocessing_params=postprocessing_params,
        )

        # TODO - upload output file
        logger.info('Writing output NWB file')
        h5_file = h5py.File(ff, 'r')
        with pynwb.NWBHDF5IO(file=h5_file, mode='r', load_namespaces=True) as io:
        # with pynwb.NWBHDF5IO(file=input.get_h5py_file(), mode='r', load_namespaces=True) as io:
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
        context.output.set(sorting_out_fname)
