#!/usr/bin/env python3

from typing import Optional
import logging
from dendro.sdk import App, ProcessorBase, BaseModel, Field, InputFile, OutputFile
from common.models_preprocessing import PreprocessingContext


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app_name = 'si_preprocessing_dev'

app = App(
    name=app_name,
    description="Spike Interface Pipeline - Preprocessing Dev",
    app_image=f"ghcr.io/magland/dendro_{app_name}",
    app_executable="/app/main.py"
)


class SIPreprocessingDevContext(BaseModel):
    input: InputFile = Field(description="Input NWB file")
    output: OutputFile = Field(description="Output SI .json file")
    electrical_series_path: str = Field(description="Path to the electrical series in the NWB file")
    preprocessing_context: PreprocessingContext = Field(default=PreprocessingContext(), description="Preprocessing context")
    start_time_sec: Optional[float] = Field(default=None, description="Start time in seconds, or None for the beginning")
    end_time_sec: Optional[float] = Field(default=None, description="End time in seconds, or None for the end")


class SIPreprocessingDevProcessor(ProcessorBase):
    name = 'si-preprocessing-dev.preprocessing'
    label = 'SpikeInterface Pipeline - Preprocessing Dev'
    description = "SpikeInterface Pipeline Processor for Preprocessing tasks Dev"
    tags = ['spike_interface', 'preprocessing', 'electrophysiology', 'pipeline']
    attributes = {
        'wip': True
    }

    @staticmethod
    def run(context: SIPreprocessingDevContext):
        from pathlib import Path
        from spikeinterface_pipelines.preprocessing import preprocess, PreprocessingParams
        from nwbdendroextractors import NwbRecordingExtractor

        scratch_folder = Path("./scratch/")
        results_folder = Path("./results/")
        scratch_folder.mkdir(exist_ok=True, parents=True)
        results_folder.mkdir(exist_ok=True, parents=True)
        results_folder_preprocessing = results_folder / "preprocessing"

        # Create SI recording from InputFile
        logger.info('Opening remote input file')
        uri = context.input.get_project_file_uri()

        logger.info('Creating input recording')
        recording = NwbRecordingExtractor(
            file_path=uri,
            electrical_series_path=context.electrical_series_path,
            stream_mode='dendro'
        )

        preprocessing_params_dict = context.preprocessing_context.model_dump()
        preprocessing_params = PreprocessingParams(**preprocessing_params_dict)

        if preprocessing_params.motion_correction.strategy != 'skip':
            raise Exception('You cannot run motion correction within this processor')

        recording_preprocessed = preprocess(
            recording=recording,
            preprocessing_params=preprocessing_params,
            scratch_folder=scratch_folder,
            results_folder=results_folder_preprocessing,
        )

        recording_preprocessed.dump_to_json('recording_preprocessed.json')
        context.output.upload('recording_preprocessed.json')


app.add_processor(SIPreprocessingDevProcessor)


if __name__ == '__main__':
    app.run()
