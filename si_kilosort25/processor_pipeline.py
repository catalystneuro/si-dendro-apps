from dendro.sdk import ProcessorBase
from spikeinterface_pipelines import pipeline as si_pipeline

from .models import PipelineContext
from nwb_recording import NwbRecording


class PipelineProcessor(ProcessorBase):
    name = 'spikeinterface_pipeline'
    label = 'SpikeInterface Pipeline'
    help = "SpikeInterface Pipeline Processor"
    tags = ['spike_interface', 'electrophysiology', 'preprocessing', 'sorting', 'postprocessing']

    @staticmethod
    def run(context: PipelineContext):

        # Create SI recording from InputFile
        input = context.input_file
        recording = NwbRecording(
            file=input.get_h5py_file(),
            electrical_series_path=context.recording_context.electrical_series_path
        )

        # TODO - run pipeline
        si_pipeline.pipeline(
            recording=recording,
            results_path="./results/",
            preprocessing_params=context.preprocessing_params,
            sorting_params=context.sorting_context,
            # postprocessing_params=context.postprocessing_params,
            # run_preprocessing=context.run_preprocessing,
        )

        # TODO - upload output file
        # print('Writing output NWB file')
        # with pynwb.NWBHDF5IO(file=f, mode='r', load_namespaces=True) as io:
        #     nwbfile_rec = io.read()

        #     if not os.path.exists('output'):
        #         os.mkdir('output')
        #     sorting_out_fname = 'output/sorting.nwb'

        #     create_sorting_out_nwb_file(nwbfile_rec=nwbfile_rec, sorting=sorting, sorting_out_fname=sorting_out_fname)
        # print_elapsed_time()

        # print('Uploading output NWB file')
        # output.set(sorting_out_fname)
        # print_elapsed_time()
