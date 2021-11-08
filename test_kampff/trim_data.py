import h5py
import numpy as np
import os

input_path = r"C:\Users\Ryan\Documents\SpikeSorting\deepinterpolation\data\PAIRED_KAMPFF\c14\c14_npx_raw.bin"

npx_channels = 384
npx_recording = np.memmap(input_path, mode='r', dtype=np.int16, order='C')
npx_samples = int(len(npx_recording)/npx_channels)
npx_recording = npx_recording.reshape((npx_channels, npx_samples), order='F')

num_samples = 120000
npx_recording = npx_recording[:, :num_samples]

output_path = f"{os.path.splitext(input_path)[0]}_nSamples{num_samples}.h5"
with h5py.File(output_path, "w") as f:
    f['/data'] = npx_recording  # NOTE: int data type
