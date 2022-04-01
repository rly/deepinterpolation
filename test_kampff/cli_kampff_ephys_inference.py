import h5py
import numpy as np
import os

from argparse import ArgumentParser

from deepinterpolation.cli.inference import Inference


def main():
    parser = ArgumentParser(description="Run DI on a Kampff dataset")
    parser.add_argument("dataset_index", type=int, help="index of the line in kampff_data.csv with the dataset name")
    parser.add_argument("data_dir", type=str, help="directory containing data")
    parser.add_argument("output_dir", type=str, help="directory to store output")
    args = parser.parse_args()
    print(args)

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # necessary for interactive mode on cori
    # os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # disable GPU for inference

    f = open("kampff_data.csv", "r")
    lines = f.read().splitlines()
    dataset_name = lines[args.dataset_index]

    generator_param = {}
    inference_param = {}

    # We are reusing the data generator for training here. Some parameters like
    # steps_per_epoch are irrelevant but currently needs to be provided
    generator_param["name"] = "KampffEphysGenerator"
    generator_param["pre_frame"] = 30
    generator_param["post_frame"] = 30
    generator_param["pre_post_omission"] = 1

    generator_param["data_path"] = f"{args.data_dir}/{dataset_name}/{dataset_name}_npx_raw.bin"
    # Note the CLI has changed train_path to data_path to take into account
    # the use of generators for inference

    generator_param["batch_size"] = 100
    generator_param["start_frame"] = 0  # 0 to start at earliest frame
    generator_param["end_frame"] = -1  # -1 to go until the end
    generator_param["randomize"] = False  # should be False for inference

    inference_param["name"] = "core_inferrence"

    # Replace this path to where you stored your model
    inference_param["model_source"] = {
        "local_path": "sample_data/2020_02_29_15_28_unet_single_ephys_1024_mean_squared_error-1050.h5"
    }

    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Replace this path to where you want to store your output file
    inference_param["output_file"] = f"{args.output_dir}/{dataset_name}.h5"

    # This option is to add blank frames at the onset and end of the output
    # movie if some output frames are missing input frames to go through
    # the model. This could be present at the start and end of the movie.
    inference_param["output_padding"] = False

    args = {
        "generator_params": generator_param,
        "inference_params": inference_param,
        "output_full_args": True
    }

    inference_obj = Inference(input_data=args, args=[])
    inference_obj.run()
    print("Done inference!")

    reshape(inference_param["output_file"])
    print("Done reshaping!")


def reshape(path):
    memory_per_chunk_in_bytes = 4e9

    with h5py.File(path, 'r') as f:
        num_samples = f['/data'].shape[0]
        num_channels = f['/data'].shape[1]

    # convert to bits, divide by bits per sample (64), divide by num channels,
    # divide by even-odd
    num_samples_per_chunk = int(memory_per_chunk_in_bytes*8/64/num_channels/2)
    num_chunks = int(np.ceil(num_samples/num_samples_per_chunk))

    # recombine even and odd
    even = np.arange(0, num_channels, 2)
    odd = even + 1

    output_path = f'{os.path.splitext(path)[0]}_reshaped_int.dat'
    if os.path.exists(output_path):
        ValueError('Output file exists. Delete it or specify new file.')

    with open(output_path, 'w+') as binary_f:
        with h5py.File(path, 'r') as f:
            for chunk_ind in range(num_chunks):
                samples_this_chunk = np.min([(chunk_ind+1)*num_samples_per_chunk, num_samples]) - chunk_ind*num_samples_per_chunk
                di_output = np.zeros((num_channels, samples_this_chunk))
                output = f['/data'][chunk_ind*num_samples_per_chunk:chunk_ind*num_samples_per_chunk+samples_this_chunk,:,:]
                di_output[even, :] = output[:, even, 0].T
                di_output[odd, :] = output[:, odd, 1].T
                # save to int16
                di_output.T.astype('int16').tofile(binary_f)
                print(f'{chunk_ind/num_chunks*100}% complete \n')


if __name__ == "__main__":
    main()

