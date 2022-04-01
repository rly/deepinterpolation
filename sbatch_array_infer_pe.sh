#!/bin/bash
#SBATCH --qos regular
#SBATCH --constraint gpu
#SBATCH --account m3513_g
#SBATCH --array 26-29
#SBATCH --job-name di_kampff
#SBATCH --output /pscratch/sd/r/rly/deepinterpolation/sbatch_out/%A_%a.log
#SBATCH --error /pscratch/sd/r/rly/deepinterpolation/sbatch_out/%A_%a.log
#SBATCH --time 05:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1

# NOTE that the 0-42 is based on all of the number of lines in kampff_data.csv

out_sh="$PSCRATCH/deepinterpolation/sbatch_out/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sh"
#me=`basename $0`
#printf "sbatch ${me}\n\n" >> ${out_sh}
printenv >> ${out_sh}
printf "\n" >> ${out_sh}
printf "SLURM_ARRAY_JOB_ID ${SLURM_ARRAY_JOB_ID}" >> ${out_sh}
printf "\n" >> ${out_sh}
printf "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID}" >> ${out_sh}
#cat ${me} >> ${out_sh}

out_py="$PSCRATCH/deepinterpolation/sbatch_out/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.py"
cat test_kampff/cli_kampff_ephys_inference.py >> ${out_py}

conda activate $PSCRATCH/env/di

python test_kampff/cli_kampff_ephys_inference.py $SLURM_ARRAY_TASK_ID $PSCRATCH/deepinterpolation/data $PSCRATCH/deepinterpolation/output/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
