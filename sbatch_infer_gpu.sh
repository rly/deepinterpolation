#!/bin/bash
#SBATCH --qos regular
#SBATCH --constraint gpu
#SBATCH --account m3513
#SBATCH --job-name c1
#SBATCH --output /global/cscratch1/sd/rly/deepinterpolation/sbatch_out/%x.%j.log
#SBATCH --error /global/cscratch1/sd/rly/deepinterpolation/sbatch_out/%x.%j.log
#SBATCH --time 3:59:59
#SBATCH --nodes 1
#SBATCH --gpus 1

out_sh="/global/cscratch1/sd/rly/deepinterpolation/sbatch_out/$SLURM_JOB_NAME.$SLURM_JOBID.sh"
me=`basename $0`
printf "sbatch ${me}\n\n" >> ${out_sh}
printenv >> ${out_sh}
printf "\n" >> ${out_sh}
cat ${me} >> ${out_sh}

out_py="/global/cscratch1/sd/rly/deepinterpolation/sbatch_out/$SLURM_JOB_NAME.$SLURM_JOBID.py"
cat test_kampff/cli_kampff_ephys_inference.py >> ${out_py}

conda activate /global/cscratch1/sd/rly/env/di

# TODO parameterize
python test_kampff/cli_kampff_ephys_inference.py
