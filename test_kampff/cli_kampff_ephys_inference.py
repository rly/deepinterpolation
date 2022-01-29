import os
from deepinterpolation.cli.inference import Inference

if __name__ == '__main__':
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # necessary for interactive mode on cori
    # os.environ["CUDA_VISIBLE_DEVICES"] = '-1'  # disable GPU for inference
    dataset_name = "c1"

    generator_param = {}
    inference_param = {}

    # We are reusing the data generator for training here. Some parameters like
    # steps_per_epoch are irrelevant but currently needs to be provided
    generator_param["name"] = "KampffEphysGenerator"
    generator_param["pre_frame"] = 30
    generator_param["post_frame"] = 30
    generator_param["pre_post_omission"] = 1

    generator_param["data_path"] = f"/global/cscratch1/sd/rly/deepinterpolation/data/{dataset_name}/{dataset_name}_npx_raw.bin"
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
    output_dir = f"/global/cscratch1/sd/rly/deepinterpolation/output/{os.environ['SLURM_JOB_NAME']}.{os.environ['SLURM_JOBID']}/"
    os.makedirs(output_dir, exist_ok=True)

    # Replace this path to where you want to store your output file
    inference_param["output_file"] = f"{output_dir}/{dataset_name}.h5"

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
