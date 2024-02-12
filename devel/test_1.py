import h5py
import pynwb
import dendro.client as prc
import remfile
from nwbdendroextractors import NwbRecordingExtractor


# Load project
project = prc.load_project('f774623e')

# Lazy load
nwb_file = project.get_file('imported/000409/sub-CSH-ZAD-001/sub-CSH-ZAD-001_ses-3e7ae7c0-fe8b-487c-9354-036236fa1010-chunking-327680-16_behavior+ecephys.nwb')
if nwb_file is None:
    raise Exception('File not found')
# nwb_remf = remfile.File(nwb_file)
# io = pynwb.NWBHDF5IO(file=h5py.File(nwb_remf, 'r'), mode='r')
# nwb = io.read()

# Explore the NWB file
# print(nwb)

x = NwbRecordingExtractor(file_path=nwb_file.get_url(), stream_mode='remfile', electrical_series_path='acquisition/ElectricalSeriesAp', use_pynwb=True)

print(x.get_num_channels())
print(x.get_channel_locations())
