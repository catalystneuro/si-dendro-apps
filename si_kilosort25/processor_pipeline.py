from dendro.sdk import ProcessorBase
from spikeinterface_pipelines import pipeline as si_pipeline
from spikeinterface.extractors import NwbRecordingExtractor
import os
import pynwb
import h5py

from models import PipelineContext
from nwb_utils import NwbRecording, create_sorting_out_nwb_file


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
        # input = context.input
        # recording = NwbRecording(
        #     file=input.get_h5py_file(),
        #     electrical_series_path=context.recording_context.electrical_series_path
        # )
        print('Opening remote input file')
        download = not context.lazy_read_input
        ff = context.input.get_file(download=download)

        print('Creating input recording')
        recording = NwbRecordingExtractor(
            file=ff,
            electrical_series_location=context.recording_context.electrical_series_path,
            # file_path=context.input.get_url(),
            # stream_mode="remfile"
        )

        ############### FOR TESTING -- REMOVE LATER  ############
        print(recording)

        from spikeinterface.sorters import Kilosort2_5Sorter
        Kilosort2_5Sorter.set_kilosort2_5_path(kilosort2_5_path="/mnt/shared_storage/Github/Kilosort")
        #######################################################

        # TODO - run pipeline
        job_kwargs = {
            'n_jobs': -1,
            'chunk_duration': '1s',
            'progress_bar': False
        }
        print('Running pipeline')
        _, sorting, _ = si_pipeline.run_pipeline(
            recording=recording,
            scratch_folder="./scratch/",
            results_folder="./results/",
            job_kwargs=job_kwargs,
            preprocessing_params=context.preprocessing_context.model_dump(),
            spikesorting_params=context.sorting_context.model_dump(),
            # postprocessing_params=context.postprocessing_params,
            # run_preprocessing=context.run_preprocessing,
        )

        # TODO - upload output file
        print('Writing output NWB file')
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

        print('Uploading output NWB file')
        context.output.set(sorting_out_fname)
