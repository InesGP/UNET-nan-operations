#!/bin/bash

# List of NIfTI files
# List of NIfTI files
# files=(
#     "/data/sub-0003002/norm.nii.gz"
# #    "/data/sub-0025011/norm.nii.gz"
# #    "/data/sub-0025248/norm.nii.gz"
# #    "/data/sub-0025350/norm.nii.gz"
# #    "/data/sub-0025531/norm.nii.gz"
# )

files=(
    "/raw_data/sub-0003002_ses-1_run-1_T1w.nii.gz"
)

# 127
# Loop through each file and submit a job
for i in "${!files[@]}"; do
     sbatch --array=60 run.sh ${files[$i]} /scratch/vinuyans/skips/$1 $1 $2
done
