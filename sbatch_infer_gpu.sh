#!/bin/bash
#SBATCH -q regular
#SBATCH -A m3513
#SBATCH -t 3:59:59
#SBATCH -n 1
#SBATCH -o /global/cscratch1/sd/rly/deepinterpolation/test_kampff/infer.%j.log
#SBATCH -e /global/cscratch1/sd/rly/deepinterpolation/test_kampff/infer.%j.log
#SBATCH -J kampff_c14
#SBATCH -C gpu
#SBATCH -c 10
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 1

conda activate /global/cscratch1/sd/rly/env/di

python test_kampff/cli_kampff_ephys_inference.py
