import os
import sys
from pbstools import PythonJob
from shutil import copyfile
import datetime
import json
import csv
import ophysextractor
from ophysextractor.datasets.lims_ophys_experiment import LimsOphysExperiment
import pandas as pd

folder_path = r"/allen/programs/braintv/workgroups/neuralcoding/2p_data/single_plane/2021-02-17-transfer_traning_oephys"
raw_path_folder = r"/allen/programs/braintv/workgroups/mct-t300/CalibrationTF/rawdata"
list_exp_id = [
    103954,
    103980,
    103871,
    103879,
    103913,
    103922,
    103930,
    104003,
    103948,
    103951,
    103960,
    103976,
    103997,
    103993,
    103984,
    103987,
    103972,
]
raw_model = r"/allen/programs/braintv/workgroups/ophysdev/OPhysCore/Deep2p/unet_single_1024_mean_absolute_error_Ai93_2019_09_11_23_32/2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450.h5"

python_file = os.path.join(
    "/home/jeromel/Documents/Projects/Deep2P/repos/neuronsextractor/examples",
    "2021-02-17-fine_tune_tif_charact.py",
)

for indiv_id in list_exp_id:
    path_tif = os.path.join(raw_path_folder, str(indiv_id) + "_2.tif")

    local_output_path = os.path.join(folder_path, str(indiv_id))

    try:
        os.mkdir(local_output_path)
    except:
        print("folder already exists")

    now = datetime.datetime.now()
    run_uid = now.strftime("%Y_%m_%d_%H_%M_%S_%f")

    jobdir = local_output_path

    output_terminal = os.path.join(
        jobdir, run_uid + os.path.basename(python_file) + "_running_terminal.txt"
    )

    job_settings = {
        "queue": "braintv",
        "mem": "250g",
        "walltime": "200:00:00",
        "ppn": 16,
        "gpus": 1,
    }

    job_settings.update(
        {
            "outfile": os.path.join(jobdir, "$PBS_JOBID.out"),
            "errfile": os.path.join(jobdir, "$PBS_JOBID.err"),
            "email": "jeromel@alleninstitute.org",
            "email_options": "a",
        }
    )

    arg_to_pass = (
        " --file_h5_path "
        + path_h5
        + " --raw_model_path "
        + raw_model
        + " --output_path "
        + local_output_path
    )

    PythonJob(
        python_file,
        python_executable="/allen/programs/braintv/workgroups/nc-ophys/Jeromel/conda/tf20-env/bin/python",  # "/home/jeromel/.conda/envs/deep_work_gpu/bin/python",
        conda_env="/allen/programs/braintv/workgroups/nc-ophys/Jeromel/conda/tf20-env",  # "deep_work_gpu",
        jobname="tf_" + os.path.basename(python_file),
        python_args=arg_to_pass + " > " + output_terminal,
        **job_settings
    ).run(dryrun=False)