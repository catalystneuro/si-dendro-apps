#!/usr/bin/env python3

from dendro.sdk import App, ProcessorBase
from pydantic import Field
from common.models import (
    Kilosort3SortingContext,
    PipelineContext as CommonPipelineContext
)
from common.processor_pipeline import run_pipeline


app_name = 'si_kilosort3'

app = App(
    name=app_name,
    description="Spike Interface Pipeline - Kilosort3",
    app_image=f"ghcr.io/catalystneuro/dendro_{app_name}",
    app_executable="/app/main.py"
)


# We need to overwrite this with the specific sorter, to generate the correct forms
class PipelineContext(CommonPipelineContext):
    sorter_name: str = 'kilosort3'
    spikesorting_context: Kilosort3SortingContext = Field(default=Kilosort3SortingContext())


class PipelineProcessor(ProcessorBase):
    name = 'spikeinterface_pipeline_ks3'
    label = 'SpikeInterface Pipeline - Kilosort 3'
    description = "SpikeInterface Pipeline Processor for Kilosort 3"
    tags = ['spike_sorting', 'spike_sorter', 'spike_interface', 'electrophysiology', 'pipeline']
    attributes = {
        'wip': True
    }

    @staticmethod
    def run(context: PipelineContext):
        run_pipeline(context)


app.add_processor(PipelineProcessor)


if __name__ == '__main__':
    app.run()
