#!/usr/bin/env python3

from dendro.sdk import App, ProcessorBase
from pydantic import Field
from common.models import (
    Kilosort25SortingContext,
    PipelineFullContext
)
from common.processor_pipeline import run_pipeline


app_name = 'si_kilosort25'

app = App(
    name=app_name,
    description="Spike Interface Pipeline - Kilosort2.5",
    app_image=f"ghcr.io/catalystneuro/dendro_{app_name}",
    app_executable="/app/main.py"
)


# We need to overwrite this with the specific sorter, to generate the correct forms
class PipelineContext(PipelineFullContext):
    spikesorting_context: Kilosort25SortingContext = Field(default=Kilosort25SortingContext())


class PipelineProcessor(ProcessorBase):
    name = 'spikeinterface_pipeline_ks25'
    label = 'SpikeInterface Pipeline - Kilosort 2.5'
    description = "SpikeInterface Pipeline Processor for Kilosort 2.5"
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
