import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import pathlib

if __name__ == '__main__':
    generator_param = {}
    inferrence_param = {}

    # We are reusing the data generator for training here. Some parameters
    # like steps_per_epoch are irrelevant but currently needs to be provided
    generator_param["type"] = "generator"
    generator_param["name"] = "KampffEphysGenerator"
    generator_param["pre_post_frame"] = 30
    generator_param["pre_post_omission"] = 1
    # No steps necessary for inference as epochs are not relevant.
    # -1 deactivate it.
    generator_param["steps_per_epoch"] = -1

    generator_param["train_path"] = "/global/cscratch1/sd/rly/deepinterpolation/data/c14/c14_npx_raw.bin"

    generator_param["batch_size"] = 100
    generator_param["start_frame"] = 100
    generator_param["end_frame"] = -1  # -1 to go until the end.

    # This is important to keep the order and avoid the
    # randomization used during training
    generator_param["randomize"] = 0

    inferrence_param["type"] = "inferrence"
    inferrence_param["name"] = "core_inferrence"

    # Replace this path to where you stored your model
    inferrence_param[
        "model_path"
    ] = "sample_data/2020_02_29_15_28_unet_single_ephys_1024_mean_squared_error-1050.h5"

    # Replace this path to where you want to store your output file
    inferrence_param[
        "output_file"
    ] = "/global/cscratch1/sd/rly/deepinterpolation/output/c14.h5"

    jobdir = "/global/cscratch1/sd/rly/deepinterpolation/jobs/"

    try:
        os.mkdir(jobdir)
    except Exception:
        print("folder already exists")

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_infer = os.path.join(jobdir, "inferrence.json")
    json_obj = JsonSaver(inferrence_param)
    json_obj.save_json(path_infer)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    inferrence_obj = ClassLoader(path_infer)
    inferrence_class = inferrence_obj.find_and_build()(path_infer,
                                                       data_generator)

    # Except this to be slow on a laptop without GPU. Inference needs
    # parallelization to be effective.
    inferrence_class.run()
