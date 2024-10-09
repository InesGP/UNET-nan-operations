#!/bin/bash
#SBATCH --job-name=10subs_coronal
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks-per-node=5
#SBATCH --nodes=1
#SBATCH --time=5:0:0
#SBATCH --array=0-255
#SBATCH --account=ACCOUNT
#SBATCH --output=slurm/%x_%a.out
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=EMAIL


module load apptainer/1.2

parallel "{} > allsubs_skipconv/thresh05/10subs/coronal_{#}_${SLURM_ARRAY_TASK_ID}.log 2>&1" :::: allsub_axial.txt

# NaN Fastsurfer command: 
# apptainer exec --nv --env BRAIN_VIEW='coronal' --env BRAIN_SLICE_INDEX=${SLURM_ARRAY_TASK_ID} --env THRESHOLD=1 --env NANCONV_THRESHOLD=1 \
# --env NAN_ACTIVE=false --env MULTI_MAXVAL=1 --env NAN_PROB=false --env NAN_PROBVAL=1 -B license.txt:/fs_license/license.txt \
# -B ~/fs_subject_ieee/sub-0025350/mri/:/data -B nan_ops.py:/fastsurfer/FastSurferCNN/models/nan_ops.py \
# -B results/cpptorch/0025350/:/output -B inference.py:/fastsurfer/FastSurferCNN/inference.py \
# -B networks.py:/fastsurfer/FastSurferCNN/models/networks.py -B sub_module.py:/fastsurfer/FastSurferCNN/models/sub_module.py \
# -B run_prediction.py:/fastsurfer/FastSurferCNN/run_prediction.py \
# nan_fastsurfer2.sif time python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/norm.mgz --sd /output --device cpu

# Default Fastsurfer command: 
# apptainer exec --nv --env BRAIN_VIEW='axial' --env BRAIN_SLICE_INDEX=$SLURM_ARRAY_TASK_ID --env NANCONV_THRESHOLD=ieee \
# --env NAN_ACTIVE=false --env MULTI_MAXVAL=1 --env NAN_PROB=true --env NAN_PROBVAL=1 \
# -B ~/fs_subject_ieee/sub-0025531/mri/:/data -B nan_ops.py:/fastsurfer/FastSurferCNN/models/nan_ops.py \
# -B results/cpptorch/0025531/ieee/:/output -B inference.py:/fastsurfer/FastSurferCNN/inference.py \
# -B networks.py:/fastsurfer/FastSurferCNN/models/networks.py -B run_prediction.py:/fastsurfer/FastSurferCNN/run_prediction.py \
# verrou_fastsurfer_2_1_1_perturbation.sif \
# time python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/norm.mgz --sd /output --device cpu

#OLD POTENTIAL OUT OF DATE COMMANDS FOR DIFFERENT EXPERIMENTS ex: single brain slice, etc.

# threshold_vals=(
#     0
#     0.2
#     0.5
#     1
#     0.8
#     'ieee'
# )

# echo "${threshold_vals[(${SLURM_ARRAY_TASK_ID} - 1)]}"


#RUN C++ NAN TORCH
# apptainer exec --nv --env EMBEDDING_LAYER='inp_block' \
# --env BRAIN_VIEW='sagittal' --env BRAIN_SLICE_INDEX=$SLURM_ARRAY_TASK_ID \
# --env THRESHOLD=1 \
# --env NANCONV_THRESHOLD=1 --env NAN_ACTIVE=false \
# --env MULTI_MAXVAL=1 --env NAN_PROB=false --env NAN_PROBVAL=1 \
# -B license.txt:/fs_license/license.txt \
# -B /home/ine5/projects/def-glatard/ine5/fs_subject_ieee/sub-0025531/mri/:/data \
# -B nan_ops.py:/fastsurfer/FastSurferCNN/models/nan_ops.py \
# -B /scratch/ine5/fastsurfer_embeddings/results/test/:/output \
# -B inf_model.py:/fastsurfer/FastSurferCNN/inference.py \
# -B networks.py:/fastsurfer/FastSurferCNN/models/networks.py \
# -B sub_module_unpool.py:/fastsurfer/FastSurferCNN/models/sub_module.py \
# -B dataset.py:/fastsurfer/FastSurferCNN/data_loader/dataset.py \
# -B run_prediction.py:/fastsurfer/FastSurferCNN/run_prediction.py /scratch/ine5/nan_fastsurfer2.sif \
# time python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/norm.mgz --sd /output --device cpu

# time /fastsurfer/run_fastsurfer.sh --t1 /data/norm.mgz --sid "norm" --sd /output --fs_license /fs_license/license.txt --seg_only --parallel --device cpu
# for actual brain segmentation post-model
# time /fastsurfer/run_fastsurfer.sh --t1 /data/norm.mgz --sid "norm" --sd /output --fs_license /fs_license/license.txt --seg_only --parallel --device cpu



#RUN FULL MODEL CORONAL
# apptainer exec --nv --env EMBEDDING_LAYER='inp_block' \
# --env BRAIN_VIEW='axial' --env BRAIN_SLICE_INDEX=$SLURM_ARRAY_TASK_ID \
# --env NANCONV_THRESHOLD=1 --env NAN_ACTIVE=true \
# --env MULTI_MAXVAL=1 --env NAN_PROB=true --env NAN_PROBVAL=1 \
# -B /home/ine5/projects/def-glatard/ine5/fs_subject_ieee/sub-0025531/mri/:/data \
# -B nan_ops.py:/fastsurfer/FastSurferCNN/models/nan_ops.py \
# -B ../../results/fullbrain/sub-0025531/thresh1/:/output \
# -B inf_model.py:/fastsurfer/FastSurferCNN/inference.py \
# -B networks.py:/fastsurfer/FastSurferCNN/models/networks.py \
# -B sub_module.py:/fastsurfer/FastSurferCNN/models/sub_module.py \
# -B dataset.py:/fastsurfer/FastSurferCNN/data_loader/dataset.py \
# -B run_prediction.py:/fastsurfer/FastSurferCNN/run_prediction.py /scratch/rprasann/test_fastsurfer_1.1.0/verrou_fastsurfer_2_1_1_perturbation.sif \
# time python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/norm.mgz --sd /output --device cpu

# # #ORIGINAL BRAIN SCAN
# apptainer exec --nv --env EMBEDDING_LAYER='inp_block' \
# --env BRAIN_VIEW='axial' --env NANCONV_THRESHOLD='ieee' \
# -B ~/CoRR/fs_crop/:/data \
# -B ../../results/fullbrain/ieee/sub-0025531.nii.gz/axial:/output \
# -B networks_unpool.py:/fastsurfer/FastSurferCNN/models/networks.py \
# -B inference_original.py:/fastsurfer/FastSurferCNN/inference.py \
# -B run_prediction.py:/fastsurfer/FastSurferCNN/run_prediction.py \
# /scratch/rprasann/test_fastsurfer_1.1.0/verrou_fastsurfer_2_1_1_perturbation.sif \
# time python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/sub-0025531.nii.gz --sd /output --device cpu

## COUNT SKIP CONVS ACROSS THRESHOLD 0.8
# apptainer exec --nv --env EMBEDDING_LAYER='inp_block' --env NAN_ACTIVE=true \
# --env BRAIN_VIEW='coronal' --env BRAIN_SLICE_INDEX=$SLURM_ARRAY_TASK_ID \
# --env MULTI_MAXVAL=1 --env NANCONV_THRESHOLD=1 --env NAN_PROB=true --env NAN_PROBVAL=1 \
# -B /home/ine5/projects/def-glatard/ine5/fs_subject_ieee/sub-0025531/mri/:/data \
# -B ../../results/sub-0025531_cerebellum/:/output \
# -B inference.py:/fastsurfer/FastSurferCNN/inference.py \
# -B nan_ops.py:/fastsurfer/FastSurferCNN/models/nan_ops.py \
# -B networks_unpool.py:/fastsurfer/FastSurferCNN/models/networks.py \
# -B sub_module_unpool.py:/fastsurfer/FastSurferCNN/models/sub_module.py \
# -B run_prediction2.py:/fastsurfer/FastSurferCNN/run_prediction.py /scratch/rprasann/test_fastsurfer_1.1.0/verrou_fastsurfer_2_1_1_perturbation.sif \
# python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/norm.mgz --sid sub-0025531 --sd /output --device cpu


# ## GET REGISTERED IMAGE
# apptainer exec -B sub-0025531/:/root \
# ~/projects/def-glatard/ine5/freesurfer2.sif \
# mri_convert /root/mri/norm.mgz \
# --apply_transform /root/mri/transforms/talairach.m3z \
# -oc 0 0 0 /root/mri/sub-0025531.mgz



# #ORIGINAL BRAIN SCAN
# apptainer exec --nv --env EMBEDDING_LAYER='inp_block' \
# --env BRAIN_VIEW='sagittal' --env NANCONV_THRESHOLD='ieee' \
# -B sub-0025531/mri/:/data \
# -B license.txt:/fs_license/license.txt \
# -B ../../results/fullbrain/sub-0025531/ieee4:/output \
# /scratch/rprasann/test_fastsurfer_1.1.0/verrou_fastsurfer_2_1_1_perturbation.sif \
# time /fastsurfer/run_fastsurfer.sh --t1 /data/norm.mgz --sid "norm" --sd /output --fs_license /fs_license/license.txt --seg_only --parallel --device cpu


max_vals=(
    1
    2
    3
    4
)

# # NAN AGGRESSIVITY RUNS
# apptainer exec --nv --env EMBEDDING_LAYER='inp_block' --env NAN_ACTIVE=true \
# --env BRAIN_VIEW='axial' --env BRAIN_SLICE_INDEX=$SLURM_ARRAY_TASK_ID --env MULTI_MAXVAL=2 --env NANCONV_THRESHOLD=1 \
# -B /home/ine5/projects/def-glatard/ine5/fs_subject_ieee/sub-0025531/mri/:/data \
# -B ../../results/sub-0025531_cerebellum/:/output \
# -B inference.py:/fastsurfer/FastSurferCNN/inference.py \
# -B sub_module.py:/fastsurfer/FastSurferCNN/models/sub_module.py \
# -B run_prediction.py:/fastsurfer/FastSurferCNN/run_prediction.py /scratch/rprasann/test_fastsurfer_1.1.0/verrou_fastsurfer_2_1_1_perturbation.sif \
# python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/norm.mgz --sid sub-0025531 --sd /output --device cpu

# apptainer exec --nv --env EMBEDDING_LAYER='inp_block' \
# --env BRAIN_VIEW='sagittal' --env BRAIN_SLICE_INDEX=$SLURM_ARRAY_TASK_ID \
# --env NANCONV_THRESHOLD=1 --env NAN_ACTIVE=true \
# --env MULTI_MAXVAL=1 --env NAN_PROB=false --env NAN_PROBVAL=1 \
# -B license.txt:/fs_license/license.txt \
# -B /home/ine5/projects/def-glatard/ine5/fs_subject_ieee/sub-0025531/mri/:/data \
# -B nan_ops.py:/fastsurfer/FastSurferCNN/models/nan_ops.py \
# -B /scratch/ine5/fastsurfer_embeddings/results/sub-0025531_cerebellum/thresh1_mean/:/output \
# -B inf_model.py:/fastsurfer/FastSurferCNN/inference.py \
# -B networks.py:/fastsurfer/FastSurferCNN/models/networks.py \
# -B sub_module.py:/fastsurfer/FastSurferCNN/models/sub_module.py \
# -B dataset.py:/fastsurfer/FastSurferCNN/data_loader/dataset.py \
# -B run_prediction.py:/fastsurfer/FastSurferCNN/run_prediction.py /scratch/rprasann/test_fastsurfer_1.1.0/verrou_fastsurfer_2_1_1_perturbation.sif \
# time /fastsurfer/run_fastsurfer.sh --t1 /data/norm.mgz --sid "norm" --sd /output --fs_license /fs_license/license.txt --seg_only --parallel --device cpu

# time python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/norm.mgz --sd /output --device cpu
