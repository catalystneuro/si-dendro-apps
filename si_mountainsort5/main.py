#!/usr/bin/env python3

from dendro.sdk import App, ProcessorBase
from pydantic import Field
from common.models import (
    MountainSort5SortingContext,
    PipelineFullContext
)
from common.processor_pipeline import run_pipeline


app_name = 'si_mountainsort5'

app = App(
    name=app_name,
    description="Spike Interface Pipeline - Mountainsort 5",
    app_image=f"ghcr.io/catalystneuro/dendro_{app_name}",
    app_executable="/app/main.py"
)


# We need to overwrite this with the specific sorter, to generate the correct forms
class PipelineContext(PipelineFullContext):
    spikesorting_context: MountainSort5SortingContext = Field(default=MountainSort5SortingContext())


class PipelineProcessor(ProcessorBase):
    name = 'spikeinterface_pipeline_mountainsort5'
    label = 'SpikeInterface Pipeline - Mountainsort 5'
    description = "SpikeInterface Pipeline Processor for Mountainsort 5"
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
