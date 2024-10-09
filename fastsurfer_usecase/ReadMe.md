# Instructions

## General Comments
* [`FastSurfer_Embeddings.ipynb`](https://github.com/InesGP/UNET-nan-operations/blob/main/fastsurfer_usecase/FastSurfer_Embeddings.ipynb) contains the work analyzing the numerical unstability in the intermediate layers of the FastSurfer CNN
* [`Final_Analysis.ipynb`](https://github.com/InesGP/UNET-nan-operations/blob/main/fastsurfer_usecase/FastSurfer_Embeddings.ipynb) contains the results of experiments with NaN Pooling and Convolution on FastSurfer
* [`NaN_Pooling.ipynb`](https://github.com/InesGP/UNET-nan-operations/blob/main/NaN_Pooling.ipynb) contains work done on developing NaN Pooling and Convolution and testing it, primarily on FastSurfer

## Data
* Subjects were extracted from [CoRR dataset](https://fcon_1000.projects.nitrc.org/indi/CoRR/html/), available [here](https://datasets.datalad.org/?dir=/corr/RawDataBIDS) on Datalad
* Since we used FastSurfer for whole brain segmentation, we processed the subjects to prepare them for that data processing task
    * Using FreeSurfer v7.3.1, we processed the raw data with:
      ```
      recon-all -verbose -sd /root -subjid sub-01 -log logging.log -status status.log \
    	-motioncor -talairach -nuintensitycor -normalization -skullstrip -gcareg -canorm -careg
      ```
* We then ran FastSurfer on the subjects with the command: \
      ```
      python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/SUBJECT.mgz --sd /output --device cpu
      ```
## Scripts
* To study NaN Pooling and Convolution's performance on FastSurfer we cloned FastSurfer v2.1.1 from the [FastSurfer repository](https://github.com/Deep-MI/FastSurfer)
* We ran our experiments on an Apptainer/Singularity image on Compute Canada
  * We provide a Dockerfile [here](https://github.com/InesGP/UNET-nan-operations/blob/main/fastsurfer_usecase/Dockerfile), and an image of FastSurfer built with NaN Pooling and Convolution is available on Dockerhub [here](https://hub.docker.com/repository/docker/inesgp/torch_nan_fastsurfer/general) (latest version recommended)
* Throughout our experiments, we extracted intermediate layers from the model to study their numerical study as well as test NaN Pooling and Convolution
* This usually slowed down the model dramatically, so we ran the model on individual brain slices
  * To do so we replaced the files `inference.py`, `run_prediction.py`, `networks.py` and `sub_module.py` usually by overwriting the existing copies in the Apptainer/Singularity image; if the goal is to replicate our work, the same will need to be done
* A sample command to run FastSurfer in Apptainer with Slurm is:
    ```
    apptainer exec \
    --env BRAIN_VIEW='sagittal' --env BRAIN_SLICE_INDEX=${SLURM_ARRAY_TASK_ID} --env THRESHOLD=1 \
    --env NANCONV_THRESHOLD=1 --env NAN_ACTIVE=false --env MULTI_MAXVAL=1 --env NAN_PROB=false \
    --env NAN_PROBVAL=1 -B license.txt:/fs_license/license.txt \
    -B data/sub-01/:/data \
    -B nan_ops.py:/fastsurfer/FastSurferCNN/models/nan_ops.py \
    -B results/:/output
    -B inference.py:/fastsurfer/FastSurferCNN/inference.py \
    -B networks.py:/fastsurfer/FastSurferCNN/models/networks.py \
    -B sub_module.py:/fastsurfer/FastSurferCNN/models/sub_module.py \
    -B run_prediction.py:/fastsurfer/FastSurferCNN/run_prediction.py \
    nan_fastsurfer.sif \
    python /fastsurfer/FastSurferCNN/run_prediction.py --t1 /data/sub-01/norm.mgz --sd /output --device cpu
    ```
 * A sample script to run FastSurfer in parallel across all slices of 5 subjects' brain with Slurm is available in [`run_brainslices.sh`](https://github.com/InesGP/UNET-nan-operations/blob/main/fastsurfer_usecase/run_brainslices.sh)
