#!/bin/bash
#SBATCH --job-name=nanunpool_allsubs_t1_axial
#SBATCH --mem-per-cpu=5G
#SBATCH --ntasks-per-node=5
#SBATCH --nodes=1
#SBATCH --time=6:0:0
#SBATCH --array=0-255
#SBATCH --account=rrg-glatard
#SBATCH --output=slurm/%x_%a.out
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=X
#
module load apptainer/1.2


#NAN UNPOOLING; single subject and brain plane
## apptainer exec --nv --env EMBEDDING_LAYER='inp_block' --env NAN_ACTIVE=true \
# --env BRAIN_VIEW='axial' --env BRAIN_SLICE_INDEX=$SLURM_ARRAY_TASK_ID \
# --env MULTI_MAXVAL=1 --env NANCONV_THRESHOLD=0.4 --env NAN_PROB=true --env NAN_PROBVAL=1 \
# -B /home/ine5/projects/def-glatard/ine5/fs_subject_ieee/sub-0025248/mri/:/data \
# -B /scratch/ine5/fastsurfer_embeddings/results/nanunpool/0025248/thresh04/:/output \
# -B nan_ops.py:/fastsurfer/FastSurferCNN/models/nan_ops.py \
# -B inference.py:/fastsurfer/FastSurferCNN/inference.py \
# -B networks_unpool.py:/fastsurfer/FastSurferCNN/models/networks.py \
# -B sub_module_unpool.py:/fastsurfer/FastSurferCNN/models/sub_module.py \
# -B run_prediction2.py:/fastsurfer/FastSurferCNN/run_prediction.py /scratch/rprasann/test_fastsurfer_1.1.0/verrou_fastsurfer_2_1_1_perturbation.sif \
# python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/norm.mgz --sid sub-0025248 --sd /output --device cpu

parallel "{} > allsubs_unpool_skipconv/thresh1/5subs/axial_{#}_${SLURM_ARRAY_TASK_ID}.log 2>&1" :::: /scratch/ine5/fastsurfer_embeddings/fastsurfer-embeddings/embedding_project/allsub_views.txt

# RUNNING ADDITIONAL SUBJECTS
# parallel "{} > allsubs_skipconv/thresh04/15subs/sagittal_{#}_${SLURM_ARRAY_TASK_ID}.log 2>&1" :::: /scratch/ine5/fastsurfer_embeddings/fastsurfer-embeddings/embedding_project/allsub_sagittal.txt

#RUN C++ NAN TORCH
# apptainer shell --nv --env EMBEDDING_LAYER='inp_block' \
# --env BRAIN_VIEW='axial' --env BRAIN_SLICE_INDEX=$SLURM_ARRAY_TASK_ID \
# --env THRESHOLD=1 \
# --env NANCONV_THRESHOLD=1 --env NAN_ACTIVE=false \
# --env MULTI_MAXVAL=1 --env NAN_PROB=false --env NAN_PROBVAL=1 \
# -B license.txt:/fs_license/license.txt \
# -B /home/ine5/projects/def-glatard/ine5/fs_subject_ieee/sub-0025531/mri/:/data \
# -B skipconv.py:/fastsurfer/FastSurferCNN/models/skipconv.py \
# -B /scratch/ine5/fastsurfer_embeddings/results/test/:/output \
# -B inf_model.py:/fastsurfer/FastSurferCNN/inference.py \
# -B networks_unpool.py:/fastsurfer/FastSurferCNN/models/networks.py \
# -B sub_module.py:/fastsurfer/FastSurferCNN/models/sub_module.py \
# -B dataset.py:/fastsurfer/FastSurferCNN/data_loader/dataset.py \
# -B run_prediction.py:/fastsurfer/FastSurferCNN/run_prediction.py /scratch/ine5/nan_fastsurfer2.sif \
# time python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/norm.mgz --sd /output --device cpu




# GET UNSTABLE EMBEDDINGS FOR SINGLE SUBJECT AND BRAIN PLANE
# apptainer exec --nv --env EMBEDDING_LAYER='inp_block' \
# --env BRAIN_VIEW='sagittal' --env BRAIN_SLICE_INDEX=126 \
# -B ~/CoRR/fs_crop/:/data \
# -B ../../results/sub-0025531_pool_sanitycheck/:/output \
# -B inference.py:/fastsurfer/FastSurferCNN/inference.py \
# -B networks_unpool.py:/fastsurfer/FastSurferCNN/models/networks.py \
# -B sub_module_unpool.py:/fastsurfer/FastSurferCNN/models/sub_module.py \
# -B run_prediction.py:/fastsurfer/FastSurferCNN/run_prediction.py /scratch/rprasann/test_fastsurfer_1.1.0/verrou_fastsurfer_2_1_1_perturbation.sif \
# /bin/bash -c "cd /home/valgrind-3.21.0+verrou-dev && echo 'python3' | xargs which | xargs ldd | grep libm | cut -d ' ' -f 3 | xargs readlink -f | xargs echo '*' > /tmp/libm.ex && \
# VERROU_LIBM_ROUNDING_MODE=random VERROU_ROUNDING_MODE=random LD_PRELOAD=/home/valgrind-3.21.0+verrou-dev/verrou/Interlibmath/interlibmath.so \
# valgrind --tool=verrou --rounding-mode=random --exclude=/tmp/libm.ex --trace-children=yes time \
# python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/sub-0025531.nii.gz --sd /output --device cpu"


#GET IEEE EMBEDDINGS
## apptainer exec --nv --env EMBEDDING_LAYER='inp_block' \
# --env BRAIN_VIEW='sagittal' --env BRAIN_SLICE_INDEX=126 \
# --env NANCONV_THRESHOLD='ieee' \
# -B ~/CoRR/fs_crop/:/data \
# -B ../../results/sub-0025531_sagittal2:/bottleneck \
# -B ../../results/sub-0025531_indices:/indices \
# -B ../../results/sub-0025531_comparison/:/output \
# -B ../../results/sub-0025531_skip/:/skip \
# -B inference.py:/fastsurfer/FastSurferCNN/inference.py \
# -B networks.py:/fastsurfer/FastSurferCNN/models/networks.py \
# -B sub_module_ieee.py:/fastsurfer/FastSurferCNN/models/sub_module.py \
# -B dataset.py:/fastsurfer/FastSurferCNN/data_loader/dataset.py \
# -B run_prediction.py:/fastsurfer/FastSurferCNN/run_prediction.py /scratch/rprasann/test_fastsurfer_1.1.0/verrou_fastsurfer_2_1_1_perturbation.sif \
# python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/sub-0025531.nii.gz --sd /output --device cpu


# threshold_vals=(
#     0
#     0.2
#     0.5
#     1
#     0.8
#     'ieee'
# )

# RUN FULL MODEL WITH SEG APARC
# time /fastsurfer/run_fastsurfer.sh --t1 /data/norm.mgz --sid "sub-0025531" --sd /output --fs_license /fs_license/license.txt --seg_only --parallel --device cpu

# #RUN FULL MODEL WITH SEG
# time python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/norm.mgz --sd /output --device cpu

##GET FULL RECON_ALL OUTPUT
# apptainer exec  --writable-tmpfs -B 0025531_fs/:/root \
# ~/projects/def-glatard/ine5/freesurfer2.sif \
# recon-all -sd /root -log logging.log -s sub-0025531 -autorecon1 -autorecon2 -autorecon3

# OR

# recon-all -sd /root -log logging.log -s sub-0025531 -normalization -skullstrip -gcareg -canorm -careg -calabel \
# -normalization2 -maskbfs -segmentation -fill -tessellate -smooth1 -inflate1 -qsphere -fix -white -smooth2 \
# -inflate2 -curvHK -curvstats -sphere -surfreg -jacobian_white -avgcurv -cortparc -pial -cortribbon -parcstats \
# -cortparc2 -parcstats2 -cortparc3 -parcstats3 -pctsurfcon -hyporelabel -aparc2aseg

# ORIGINAL BRAIN SCAN
# apptainer exec --nv --env EMBEDDING_LAYER='inp_block' \
# --env BRAIN_VIEW='sagittal' --env NANCONV_THRESHOLD='ieee' \
# -B ~/CoRR/fs_crop/:/data \
# -B license.txt:/fs_license/license.txt \
# -B ../../results/fullbrain/sub-0025531/ieee:/output \
# /scratch/rprasann/test_fastsurfer_1.1.0/verrou_fastsurfer_2_1_1_perturbation.sif \
# time /fastsurfer/run_fastsurfer.sh --t1 /data/sub-0025531.nii.gz --sid "sub-0025531" --sd /output --fs_license /fs_license/license.txt --seg_only --parallel --device cpu



