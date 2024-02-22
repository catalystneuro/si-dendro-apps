import dendro.client as den


def main():
    project = den.load_project('f774623e')
    den.submit_job(
        project=project,
        processor_name='si-preprocessing-dev.preprocessing',
        input_files=[
            den.SubmitJobInputFile(
                name='input',
                file_name='imported/000409/sub-CSH-ZAD-001/sub-CSH-ZAD-001_ses-3e7ae7c0-fe8b-487c-9354-036236fa1010-chunking-327680-16_behavior+ecephys.nwb'
            )
        ],
        output_files=[
            den.SubmitJobOutputFile(
                name='output',
                file_name='generated/test2.json'
            )
        ],
        parameters=[
            den.SubmitJobParameter(
                name='electrical_series_path',
                value='acquisition/ElectricalSeriesAp'
            ),
            den.SubmitJobParameter(
                name='preprocessing_context.motion_correction.strategy',
                value='skip'
            ),
            den.SubmitJobParameter(
                name='start_time_sec',
                value=0
            ),
            den.SubmitJobParameter(
                name='end_time_sec',
                value=120
            )
        ],
        required_resources=den.DendroJobRequiredResources(
            numCpus=2,
            numGpus=0,
            memoryGb=8,
            timeSec=3600
        ),
        run_method='local'
    )

    den.submit_job(
        project=project,
        processor_name='dandi-vis-1.ecephys_summary',
        input_files=[
            den.SubmitJobInputFile(
                name='input',
                file_name='generated/test2.json'
            )
        ],
        output_files=[
            den.SubmitJobOutputFile(
                name='output',
                file_name='generated/test2_ecephys_summary.nh5'
            )
        ],
        parameters=[
        ],
        required_resources=den.DendroJobRequiredResources(
            numCpus=2,
            numGpus=0,
            memoryGb=8,
            timeSec=3600
        ),
        run_method='local'
    )


if __name__ == "__main__":
    main()
