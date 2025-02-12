#!/bin/bash

# List of NIfTI files
# List of NIfTI files
files=(
    # "/data/sub-0003002/norm.nii.gz"
   "/data/sub-0025011/norm.nii.gz"
#    "/data/sub-0025248/norm.nii.gz"
#    "/data/sub-0025350/norm.nii.gz"
#    "/data/sub-0025531/norm.nii.gz"
)

# files=(
#     # "/raw_data/sub-0003002_ses-1_run-1_T1w.nii.gz"
#     "/raw_data/sub-0025011_ses-1_run-1_T1w.nii.gz"
#     "/raw_data/sub-0025248_ses-1_run-1_T1w.nii.gz"
#     "/raw_data/sub-0025350_ses-1_run-1_T1w.nii.gz"
#     "/raw_data/sub-0025531_ses-1_run-1_T1w.nii.gz"
# )

# $1 0.5
# $2 1e-7
# $3 FONDUE_LT.py
# $4 1


# 127
# Loop through each file and submit a job
for i in "${!files[@]}"; do
     sbatch --array=60-70 run.sh ${files[$i]} /scratch/vinuyans/skips/$1 $1 $2 $3 $4
done

# bash run-processed.sh 0.5 1e-7 FONDUE_LT_proc2.py 1

# OLD bash run-processed.sh 1.0 1e-7 FONDUE_LT_proc2.py