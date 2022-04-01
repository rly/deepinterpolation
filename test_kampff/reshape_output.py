import h5py
import numpy as np
import os

output_path = r"/global/cscratch1/sd/rly/deepinterpolation/output/c14.h5"

f = h5py.File(output_path, 'r')
output = np.squeeze(f['/data'][:])
num_samples = output.shape[0]
num_channels = output.shape[1]
di_output = np.zeros((num_channels, num_samples))

# recombine even and odd
even = np.arange(0, num_channels, 2)
odd = even + 1
di_output[even, :] = output[:, even, 0].T
di_output[odd, :] = output[:, odd, 1].T

new_output_path = f"{os.path.splitext(output_path)[0]}_reshape.h5"
with h5py.File(new_output_path, "w") as f:
    f['/data'] = di_output  # NOTE: float data type
