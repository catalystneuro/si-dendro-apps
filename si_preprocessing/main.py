#!/usr/bin/env python3

from dendro.sdk import App, ProcessorBase
from common.models import PipelineFullContext, PipelinePreprocessingContext
from common.processor_pipeline import run_pipeline


app_name = 'si_preprocessing'

app = App(
    name=app_name,
    description="Spike Interface Pipeline - Preprocessing",
    app_image=f"ghcr.io/catalystneuro/dendro_{app_name}",
    app_executable="/app/main.py"
)


class PipelineProcessor(ProcessorBase):
    name = 'spikeinterface_pipeline_preprocessing'
    label = 'SpikeInterface Pipeline - Preprocessing'
    description = "SpikeInterface Pipeline Processor for Preprocessing tasks"
    tags = ['spike_interface', 'preprocessing', 'electrophysiology', 'pipeline']
    attributes = {
        'wip': True
    }

    @staticmethod
    def run(context: PipelinePreprocessingContext):
        context_preprocessing = context.model_dump()
        context_preprocessing['preprocessing_context']['add_preprocessed_to_output_nwb'] = True
        context_full = PipelineFullContext(
            run_preprocessing=True,
            run_spikesorting=False,
            run_postprocessing=False,
            run_curation=False,
            run_visualization=False,
            **context_preprocessing
        )
        run_pipeline(context_full)


app.add_processor(PipelineProcessor)


if __name__ == '__main__':
    app.run()
