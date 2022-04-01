#!/bin/bash
#SBATCH --qos regular
#SBATCH --constraint gpu
#SBATCH --account m3513
#SBATCH --array 4,5,6,7,9,15,22,23
#SBATCH --job-name di_kampff
#SBATCH --output /global/cscratch1/sd/rly/deepinterpolation/sbatch_out/%A_%a.log
#SBATCH --error /global/cscratch1/sd/rly/deepinterpolation/sbatch_out/%A_%a.log
#SBATCH --time 11:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1

# NOTE that the 0-42 is based on all of the number of lines in kampff_data.csv

out_sh="$CSCRATCH/deepinterpolation/sbatch_out/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sh"
#me=`basename $0`
#printf "sbatch ${me}\n\n" >> ${out_sh}
printenv >> ${out_sh}
printf "\n" >> ${out_sh}
printf "SLURM_ARRAY_JOB_ID ${SLURM_ARRAY_JOB_ID}" >> ${out_sh}
printf "\n" >> ${out_sh}
printf "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID}" >> ${out_sh}
#cat ${me} >> ${out_sh}

out_py="$CSCRATCH/deepinterpolation/sbatch_out/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.py"
cat test_kampff/cli_kampff_ephys_inference.py >> ${out_py}

conda activate /global/cscratch1/sd/rly/env/di

python test_kampff/cli_kampff_ephys_inference.py $SLURM_ARRAY_TASK_ID $CSCRATCH/deepinterpolation/data $CSCRATCH/deepinterpolation/output/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
