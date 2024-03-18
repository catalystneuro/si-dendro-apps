from typing import Union, List, Optional
# from neuroconv.tools.spikeinterface import write_sorting
from pynwb import NWBFile
from pynwb.file import Subject
from uuid import uuid4
import numpy as np
import h5py
import spikeinterface as si
import pynwb


class NwbRecording(si.BaseRecording):
    def __init__(
        self,
        file,  # file-like object
        electrical_series_path: str
    ) -> None:
        h5_file = h5py.File(file, 'r')

        electrical_series: h5py.Group = h5_file[electrical_series_path]
        electrical_series_data = electrical_series['data']
        dtype = electrical_series_data.dtype

        # Get sampling frequency
        if 'starting_time' in electrical_series.keys():
            # t_start = electrical_series['starting_time'][()]
            sampling_frequency = electrical_series['starting_time'].attrs['rate']
        elif 'timestamps' in electrical_series.keys():
            # t_start = electrical_series['timestamps'][0]
            sampling_frequency = 1 / np.median(np.diff(electrical_series['timestamps'][:1000]))

        # Get channel ids
        electrode_indices = electrical_series['electrodes'][:]
        electrodes_table = h5_file['/general/extracellular_ephys/electrodes']
        channel_ids = [electrodes_table['id'][i] for i in electrode_indices]

        super().__init__(
            channel_ids=channel_ids,
            sampling_frequency=sampling_frequency,
            dtype=dtype
        )

        # Set electrode locations
        if 'rel_x' in electrodes_table:
            channel_loc_x = [electrodes_table['rel_x'][i] for i in electrode_indices]
            channel_loc_y = [electrodes_table['rel_y'][i] for i in electrode_indices]
            if 'rel_z' in electrodes_table:
                channel_loc_z = [electrodes_table['rel_z'][i] for i in electrode_indices]
            else:
                channel_loc_z = None
        elif 'x' in electrodes_table:
            channel_loc_x = [electrodes_table['x'][i] for i in electrode_indices]
            channel_loc_y = [electrodes_table['y'][i] for i in electrode_indices]
            if 'z' in electrodes_table:
                channel_loc_z = [electrodes_table['z'][i] for i in electrode_indices]
            else:
                channel_loc_z = None
        else:
            channel_loc_x = None
            channel_loc_y = None
            channel_loc_z = None
        if channel_loc_x is not None:
            ndim = 2 if channel_loc_z is None else 3
            locations = np.zeros((len(electrode_indices), ndim), dtype=float)
            for i, electrode_index in enumerate(electrode_indices):
                locations[i, 0] = channel_loc_x[electrode_index]
                locations[i, 1] = channel_loc_y[electrode_index]
                if channel_loc_z is not None:
                    locations[i, 2] = channel_loc_z[electrode_index]
            self.set_dummy_probe_from_locations(locations)

        # Extractors channel groups must be integers, but Nwb electrodes group_name can be strings
        if "group_name" in electrodes_table:
            unique_electrode_group_names = list(np.unique(electrodes_table["group_name"][:]))
            print(unique_electrode_group_names)

            groups = []
            for electrode_index in electrode_indices:
                group_name = electrodes_table["group_name"][electrode_index]
                group_id = unique_electrode_group_names.index(group_name)
                groups.append(group_id)
            self.set_channel_groups(groups)

        recording_segment = NwbRecordingSegment(
            electrical_series_data=electrical_series_data,
            sampling_frequency=sampling_frequency
        )
        self.add_recording_segment(recording_segment)


class NwbRecordingSegment(si.BaseRecordingSegment):
    def __init__(self, electrical_series_data: h5py.Dataset, sampling_frequency: float) -> None:
        self._electrical_series_data = electrical_series_data
        super().__init__(sampling_frequency=sampling_frequency)

    def get_num_samples(self) -> int:
        return self._electrical_series_data.shape[0]

    def get_traces(self, start_frame: int, end_frame: int, channel_indices: Union[List[int], None] = None) -> np.ndarray:
        if channel_indices is None:
            return self._electrical_series_data[start_frame:end_frame, :]
        else:
            return self._electrical_series_data[start_frame:end_frame, channel_indices]


def create_sorting_out_nwb_file(
    nwbfile_original,
    recording: Optional[si.BaseRecording] = None,
    sorting: Optional[si.BaseSorting] = None,
    output_fname: Optional[str] = None
):
    nwbfile = NWBFile(
        session_description=nwbfile_original.session_description + " - spike sorting results.",
        identifier=str(uuid4()),
        session_start_time=nwbfile_original.session_start_time,
        session_id=nwbfile_original.session_id,
        experimenter=nwbfile_original.experimenter,
        lab=nwbfile_original.lab,
        institution=nwbfile_original.institution,
        experiment_description=nwbfile_original.experiment_description,
        related_publications=nwbfile_original.related_publications,
        subject=Subject(
            subject_id=nwbfile_original.subject.subject_id,
            age=nwbfile_original.subject.age,
            description=nwbfile_original.subject.description,
            species=nwbfile_original.subject.species,
            sex=nwbfile_original.subject.sex,
        )
    )

    # Add sorting
    if sorting is not None:
        for ii, unit_id in enumerate(sorting.get_unit_ids()):
            st = sorting.get_unit_spike_train(unit_id) / sorting.get_sampling_frequency()
            nwbfile.add_unit(
                id=ii + 1,  # must be an int
                spike_times=st
            )

    # Write the nwb file
    with pynwb.NWBHDF5IO(path=output_fname, mode='w') as io:
        io.write(container=nwbfile, cache_spec=True)

    # write_sorting(
    #     sorting=sorting,
    #     nwbfile=nwbfile,
    #     nwbfile_path=sorting_out_fname,
    #     overwrite=True
    # )


def create_base_nwb_file(nwbfile_original: NWBFile) -> NWBFile:
    return NWBFile(
        session_description=nwbfile_original.session_description + " - spike sorting results.",
        identifier=str(uuid4()),
        session_start_time=nwbfile_original.session_start_time,
        session_id=nwbfile_original.session_id,
        experimenter=nwbfile_original.experimenter,
        lab=nwbfile_original.lab,
        institution=nwbfile_original.institution,
        experiment_description=nwbfile_original.experiment_description,
        related_publications=nwbfile_original.related_publications,
        subject=Subject(
            subject_id=nwbfile_original.subject.subject_id,
            age=nwbfile_original.subject.age,
            description=nwbfile_original.subject.description,
            species=nwbfile_original.subject.species,
            sex=nwbfile_original.subject.sex,
        )
    )
