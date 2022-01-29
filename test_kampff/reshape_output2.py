import h5py
import numpy as np
import os

input_path = '/global/cscratch1/sd/rly/deepinterpolation/output/c14.h5'  # TODO take in a CLI argument
memory_per_chunk_in_bytes = 4e9

with h5py.File(input_path, 'r') as f:
    num_samples = f['/data'].shape[0]
    num_channels = f['/data'].shape[1]

# convert to bits, divide by bits per sample (64), divide by num channels,
# divide by even-odd
num_samples_per_chunk = int(memory_per_chunk_in_bytes*8/64/num_channels/2)
num_chunks = int(np.ceil(num_samples/num_samples_per_chunk))

# recombine even and odd
even = np.arange(0, num_channels, 2)
odd = even + 1

output_path = f'{os.path.splitext(input_path)[0]}_reshaped2.dat'

if os.path.exists(output_path):
    ValueError('Output file exists, first delete it or specify new file')

with open(output_path, 'w+') as binary_f:
    with h5py.File(input_path, 'r') as f:
        for chunk_ind in range(num_chunks):
            samples_this_chunk = np.min([(chunk_ind+1)*num_samples_per_chunk, num_samples]) - chunk_ind*num_samples_per_chunk
            di_output = np.zeros((num_channels, samples_this_chunk))
            output = f['/data'][chunk_ind*num_samples_per_chunk:chunk_ind*num_samples_per_chunk+samples_this_chunk,:,:]
            di_output[even, :] = output[:, even, 0].T
            di_output[odd, :] = output[:, odd, 1].T
            # save to int16
            di_output.T.astype('int16').tofile(binary_f)
            print(f'{chunk_ind/num_chunks*100}% complete \n')
