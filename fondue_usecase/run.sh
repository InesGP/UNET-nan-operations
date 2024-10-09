#!/bin/bash
# --nodes=2
#SBATCH --job-name=t1
#SBATCH --cpus-per-task=3
#SBATCH --time=3:0:00
#SBATCH --mem-per-cpu=3G
#SBATCH --account=ACCOUNT
#SBATCH --mail-type=ALL
#SBATCH --mail-user=EMAIL

# Dynamically set the output file based on the argument
#LOGFILE="slurm/t${3}_$1_${SLURM_ARRAY_TASK_ID}.out"
#exec > $LOGFILE 2>&1

#echo TEST
subject=$(echo ${1} | cut -d'/' -f3)
export SUBJECT=$subject

START_TIME=$(date +%s)
module load apptainer

srun apptainer exec --writable-tmpfs \
--bind /scratch/vinuyans/skips/raw_data:/raw_data \
--bind /scratch/vinuyans/skips/embeddings:/embeddings \
--bind /scratch/vinuyans/fondue:/FONDUE/ \
--bind /scratch/vinuyans/fondue/archs/FONDUE_LT_nan.py:/FONDUE/archs/FONDUE_LT.py \
--env THRESHOLD=$3 \
--env FILE_ROOT=$2 \
--env NAN_OPS_ENABLED=True \
--env EPSILON=$4 \
fondue_ieee.sif python /FONDUE/eval.py --in_name $1 --no_cuda > slurm/t${3}_${subject}_${SLURM_ARRAY_TASK_ID}.out 2>&1
