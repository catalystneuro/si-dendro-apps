from dendro.sdk import ProcessorBase
from spikeinterface_pipelines import pipeline as si_pipeline
import os
import pynwb

from models import PipelineContext
from nwb_utils import NwbRecording, create_sorting_out_nwb_file


class PipelineProcessor(ProcessorBase):
    name = 'spikeinterface_pipeline'
    label = 'SpikeInterface Pipeline'
    description = "SpikeInterface Pipeline Processor"
    tags = ['spike_interface', 'electrophysiology', 'preprocessing', 'spike_sorter', 'postprocessing']
    attributes = {
        'wip': True
    }

    @staticmethod
    def run(context: PipelineContext):

        # Create SI recording from InputFile
        input = context.input_file
        recording = NwbRecording(
            file=input.get_h5py_file(),
            electrical_series_path=context.recording_context.electrical_series_path
        )

        # TODO - run pipeline
        _, sorting = si_pipeline.pipeline(
            recording=recording,
            results_path="./results/",
            preprocessing_params=context.preprocessing_params,
            sorting_params=context.sorting_context,
            # postprocessing_params=context.postprocessing_params,
            # run_preprocessing=context.run_preprocessing,
        )

        # TODO - upload output file
        print('Writing output NWB file')
        with pynwb.NWBHDF5IO(file=input.get_h5py_file(), mode='r', load_namespaces=True) as io:
            nwbfile_rec = io.read()

            if not os.path.exists('output'):
                os.mkdir('output')
            sorting_out_fname = 'output/sorting.nwb'

            create_sorting_out_nwb_file(
                nwbfile_rec=nwbfile_rec,
                sorting=sorting,
                sorting_out_fname=sorting_out_fname
            )

        print('Uploading output NWB file')
        context.output.set(sorting_out_fname)
