#!/usr/bin/env python3

import logging
from dendro.sdk import App, ProcessorBase, BaseModel, Field, InputFile, OutputFile
from common.models_preprocessing import MCKilosortLike, MCNonrigidAccurate, MCRigidFast, MotionCorrection, PreprocessingContext


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
        from spikeinterface_pipelines.preprocessing import PreprocessingParams
        from dendroextractors.nwb.nwbdendroextractors import NwbDendroRecordingExtractor
        import spikeinterface as si
        import numpy as np
        import spikeinterface.preprocessing as spre

        scratch_folder = Path("./scratch/")
        results_folder = Path("./results/")
        scratch_folder.mkdir(exist_ok=True, parents=True)
        results_folder.mkdir(exist_ok=True, parents=True)
        results_folder_preprocessing = results_folder / "preprocessing"

        # Create SI recording from InputFile
        logger.info('Opening remote input file')
        uri = context.input.get_project_file_uri()

        logger.info('Creating input recording')
        recording = NwbDendroRecordingExtractor(
            file_path=uri,
            electrical_series_path=context.electrical_series_path,
            stream_mode='dendro'
        )

        preprocessing_params_dict = context.preprocessing_context.model_dump()
        preprocessing_params = PreprocessingParams(**preprocessing_params_dict)

        if preprocessing_params.motion_correction.strategy != 'skip':
            raise Exception('You cannot run motion correction within this processor')

        def preprocess(
            recording: si.BaseRecording,
            preprocessing_params: PreprocessingParams = PreprocessingParams(),
            scratch_folder: Path = Path("./scratch/"),
            results_folder: Path = Path("./results/preprocessing/"),
        ) -> si.BaseRecording:
            """
            Apply preprocessing to recording.

            Parameters
            ----------
            recording: si.BaseRecording
                The input recording
            preprocessing_params: PreprocessingParams
                Preprocessing parameters
            scratch_folder: Path
                Path to the scratch folder
            results_folder: Path
                Path to the results folder

            Returns
            -------
            si.BaseRecording | None
                Preprocessed recording. If more than `max_bad_channel_fraction_to_remove` channels are detected as bad,
                returns None.
            """
            logger.info("[Preprocessing] \tRunning Preprocessing stage")
            logger.info(f"[Preprocessing] \tDuration: {np.round(recording.get_total_duration(), 2)} s")

            # Phase shift correction
            print('PHASE SHIFT')
            if "inter_sample_shift" in recording.get_property_keys():
                logger.info("[Preprocessing] \tPhase shift")
                recording = spre.phase_shift(recording, **preprocessing_params.phase_shift.model_dump())
            else:
                logger.info("[Preprocessing] \tSkipping phase shift: 'inter_sample_shift' property not found")

            # Highpass filter
            print('HIGH PASS FILTER')
            recording_hp_full = spre.highpass_filter(recording, **preprocessing_params.highpass_filter.model_dump())

            # Detect and remove bad channels
            print('DETECT BAD CHANNELS')
            _, channel_labels = spre.detect_bad_channels(
                recording_hp_full, **preprocessing_params.detect_bad_channels.model_dump(),
                num_random_chunks=3,
                chunk_duration_s=0.1
            )
            dead_channel_mask = channel_labels == "dead"
            noise_channel_mask = channel_labels == "noise"
            out_channel_mask = channel_labels == "out"
            logger.info(
                f"[Preprocessing] \tBad channel detection found: {np.sum(dead_channel_mask)} dead, {np.sum(noise_channel_mask)} noise, {np.sum(out_channel_mask)} out channels"
            )
            dead_channel_ids = recording_hp_full.channel_ids[dead_channel_mask]
            noise_channel_ids = recording_hp_full.channel_ids[noise_channel_mask]
            out_channel_ids = recording_hp_full.channel_ids[out_channel_mask]
            all_bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids, out_channel_ids))

            max_bad_channel_fraction_to_remove = preprocessing_params.max_bad_channel_fraction_to_remove
            if len(all_bad_channel_ids) >= int(max_bad_channel_fraction_to_remove * recording.get_num_channels()):
                logger.info(
                    f"[Preprocessing] \tMore than {max_bad_channel_fraction_to_remove * 100}% bad channels ({len(all_bad_channel_ids)}). "
                )
                logger.info("[Preprocessing] \tSkipping further processing for this recording.")
                return recording_hp_full

            if preprocessing_params.remove_out_channels:
                logger.info(f"[Preprocessing] \tRemoving {len(out_channel_ids)} out channels")
                recording_rm_out = recording_hp_full.remove_channels(out_channel_ids)
            else:
                recording_rm_out = recording_hp_full

            bad_channel_ids = np.concatenate((dead_channel_ids, noise_channel_ids))

            # Denoise: CMR or destripe
            print('DE-NOISE')
            if preprocessing_params.preprocessing_strategy == "cmr":
                recording_processed = spre.common_reference(
                    recording_rm_out, **preprocessing_params.common_reference.model_dump()
                )
            else:
                recording_interp = spre.interpolate_bad_channels(recording_rm_out, bad_channel_ids)
                recording_processed = spre.highpass_spatial_filter(
                    recording_interp, **preprocessing_params.highpass_spatial_filter.model_dump()
                )

            print('REMOVE BAD CHANNELS')
            if preprocessing_params.remove_bad_channels:
                logger.info(
                    f"[Preprocessing] \tRemoving {len(bad_channel_ids)} channels after {preprocessing_params.preprocessing_strategy} preprocessing"
                )
                recording_processed = recording_processed.remove_channels(bad_channel_ids)

            # # Motion correction
            # if preprocessing_params.motion_correction.strategy != "skip":
            #     preset = preprocessing_params.motion_correction.preset
            #     if preset == "nonrigid_accurate":
            #         motion_correction_kwargs = MCNonrigidAccurate(**preprocessing_params.motion_correction.motion_kwargs.model_dump())
            #     elif preset == "rigid_fast":
            #         motion_correction_kwargs = MCRigidFast(**preprocessing_params.motion_correction.motion_kwargs.model_dump())
            #     elif preset == "kilosort_like":
            #         motion_correction_kwargs = MCKilosortLike(**preprocessing_params.motion_correction.motion_kwargs.model_dump())
            #     logger.info(f"[Preprocessing] \tComputing motion correction with preset: {preset}")
            #     motion_folder = results_folder / "motion_correction"
            #     recording_corrected = spre.correct_motion(
            #         recording_processed,
            #         preset=preset,
            #         folder=motion_folder,
            #         verbose=False,
            #         detect_kwargs=motion_correction_kwargs.detect_kwargs.model_dump(),
            #         localize_peaks_kwargs=motion_correction_kwargs.localize_peaks_kwargs.model_dump(),
            #         estimate_motion_kwargs=motion_correction_kwargs.estimate_motion_kwargs.model_dump(),
            #         interpolate_motion_kwargs=motion_correction_kwargs.interpolate_motion_kwargs.model_dump(),
            #     )
            #     if preprocessing_params.motion_correction.strategy == "apply":
            #         logger.info("[Preprocessing] \tApplying motion correction")
            #         recording_processed = recording_corrected

            print('Returning recording_processed')
            assert isinstance(recording_processed, si.BaseRecording), "recording_processed is not a si.BaseRecording"
            return recording_processed

        print('Preprocessing...')
        recording_preprocessed = preprocess(
            recording=recording,
            preprocessing_params=preprocessing_params,
            scratch_folder=scratch_folder,
            results_folder=results_folder_preprocessing,
        )

        print('Saving output...')
        recording_preprocessed.dump_to_json('recording_preprocessed.json')
        print('Uplading output...')
        context.output.upload('recording_preprocessed.json')
        print('Done')


class SIMotionCorrectionDevContext(BaseModel):
    input: InputFile = Field(description="Input SI .json file")
    output: OutputFile = Field(description="Output SI .json file")
    motion_correction_context: MotionCorrection = Field(default=MotionCorrection(), description="Motion correction context")
    n_jobs: int = Field(default=4, description="Number of jobs for writing the .dat file")
    chunk_duration: str = Field(default='2s', description="Chunk duration for writing the .dat file")


class SIMotionCorrectionDevProcessor(ProcessorBase):
    name = 'si-preprocessing-dev.motion_correction'
    label = 'SpikeInterface Pipeline - Motion Correction Dev'
    description = "SpikeInterface Pipeline Processor for Motion Correction tasks Dev"
    tags = ['spike_interface', 'motion_correction', 'electrophysiology', 'pipeline']
    attributes = {
        'wip': True
    }

    @staticmethod
    def run(context: SIMotionCorrectionDevContext):
        import spikeinterface as si
        import spikeinterface.preprocessing as spre

        context.input.download('recording.json')

        print('Loading recording...')
        recording1 = si.load_extractor('recording.json')
        assert isinstance(recording1, si.BaseRecording), "Recording is not a BaseRecording"

        print('Writing recording to .dat file...')
        si.BinaryRecordingExtractor.write_recording(
            recording=recording1,
            file_paths=['recording.dat'],
            dtype='float32',
            n_jobs=context.n_jobs,
            chunk_duration=context.chunk_duration,
            mp_context='spawn'
        )
        print('Loading recording from .dat file...')
        recording = si.BinaryRecordingExtractor(
            file_paths=['recording.dat'],
            sampling_frequency=recording1.get_sampling_frequency(),
            channel_ids=recording1.get_channel_ids(),
            num_channels=recording1.get_num_channels(),
            dtype='float32'
        )

        motion_correction = context.motion_correction_context

        if motion_correction.strategy == 'apply':
            raise Exception('You cannot run motion correction with strategy "apply": within this processor')
        if motion_correction.strategy == 'skip':
            raise Exception('You cannot run motion correction with strategy "skip": within this processor')
        assert motion_correction.strategy == 'compute', f"Unknown motion correction strategy: {motion_correction.strategy}"

        preset = motion_correction.preset
        if preset == "nonrigid_accurate":
            motion_correction_kwargs = MCNonrigidAccurate(**motion_correction.motion_kwargs_nonrigid_accurate.model_dump())
        elif preset == "rigid_fast":
            motion_correction_kwargs = MCRigidFast(**motion_correction.motion_kwargs_rigid_fast.model_dump())
        elif preset == "kilosort_like":
            motion_correction_kwargs = MCKilosortLike(**motion_correction.motion_kwargs_kilosort_like.model_dump())
        else:
            raise Exception(f"Unknown motion correction preset: {preset}")
        logger.info(f"[Preprocessing] \tComputing motion correction with preset: {preset}")
        motion_folder = "motion_correction"
        recording_corrected = spre.correct_motion(
            recording,
            preset=preset,
            folder=motion_folder,
            verbose=False,
            detect_kwargs=motion_correction_kwargs.detect_kwargs.model_dump(),
            localize_peaks_kwargs=motion_correction_kwargs.localize_peaks_kwargs.model_dump(),
            estimate_motion_kwargs=motion_correction_kwargs.estimate_motion_kwargs.model_dump(),
            interpolate_motion_kwargs=motion_correction_kwargs.interpolate_motion_kwargs.model_dump(),
        )
        assert isinstance(recording_corrected, si.BaseRecording), "recording_corrected is not a si.BaseRecording"

        print('Saving output...')
        recording_corrected.dump_to_json('recording_preprocessed.json')
        print('Uplading output...')
        context.output.upload('recording_preprocessed.json')
        print('Done')


app.add_processor(SIPreprocessingDevProcessor)
app.add_processor(SIMotionCorrectionDevProcessor)

if __name__ == '__main__':
    app.run()
