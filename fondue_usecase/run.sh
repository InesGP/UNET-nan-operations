#!/bin/bash
# --nodes=2
#SBATCH --job-name=unpool_t1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:0:0
#SBATCH --mem-per-cpu=6G
#SBATCH --account=rrg-glatard
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inesgp99@gmail.com

#vinuyansivakolunthu@gmail.com

# Dynamically set the output file based on the argument
#LOGFILE="slurm/t${3}_$1_${SLURM_ARRAY_TASK_ID}.out"
#exec > $LOGFILE 2>&1

#echo TEST
subject=$(echo ${1} | cut -d'/' -f3)
export SUBJECT=$subject

START_TIME=$(date +%s)
module load apptainer/1.2

#--bind /scratch/ine5/fs_subject_ieee:/data \
#--bind /scratch/vinuyans/fondue/archs/FONDUE_copy_LT.py:/FONDUE/archs/FONDUE_LT.py \
#--bind /scratch/vinuyans/skips/raw_data:/raw_data \
#--bind /scratch/vinuyans/fondue/archs/FONDUE_LT_orig.py:/FONDUE/archs/FONDUE_LT.py \

srun apptainer exec --writable-tmpfs \
--bind /scratch/ine5/fs_subject_ieee:/data \
--bind /scratch/vinuyans/skips/embeddings:/embeddings \
--bind /scratch/vinuyans/fondue:/FONDUE/ \
--bind /scratch/vinuyans/fondue/archs/${5}:/FONDUE/archs/FONDUE_LT.py \
--env THRESHOLD=$3 \
--env FILE_ROOT=$2 \
--env NAN_OPS_ENABLED=True \
--env EPSILON=$4 \
--env POOL_THRESH=$6 \
fondue_ieee.sif python /FONDUE/eval.py --in_name $1 --no_cuda > slurm/debug_${3}_${subject}_${SLURM_ARRAY_TASK_ID}.out 2>&1
