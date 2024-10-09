# Instructions

## General Comments
* [`FONDUE_Analysis.ipynb`](https://github.com/InesGP/UNET-nan-operations/blob/main/fondue_usecase/FONDUE_Analysis.ipynb) contains preliminary analysis for FONDUE with NaN Pooling and Convolution
* [`Final_Analysis.ipynb`](https://github.com/InesGP/UNET-nan-operations/blob/main/Final_Analysis.ipynb) will contain final analysis

## Data
* Subjects were extracted from [CoRR dataset](https://fcon_1000.projects.nitrc.org/indi/CoRR/html/), available [here](https://datasets.datalad.org/?dir=/corr/RawDataBIDS) on Datalad
* As FONDUE is a denoising model, we fed the subjects as is to the model
* We then ran FONDUE on the subjects with the command: \
      ```
      python /FONDUE/fondue_eval_simpleitk.py --in_name SUBJECT_ID --no_cuda
      ```

## Scripts
* We provide a Dockerfile [here](https://github.com/InesGP/UNET-nan-operations/blob/main/fondue_usecase/Dockerfile) or an image available on Dockerhub **(ADD)** to build the environment necessary to run the model
* We then clone FONDUE v1.1 from its [repository](https://github.com/waadgo/FONDUE) and mount it to our environment to test NaN Pooling and Convolution on this use case
* To test NaN Pooling and Convolution we substituted `fondue_eval_simpleitk.py` and `FONDUE_LT.py` with our modified versions
* A sample command to replicate our experiment is:
  ```
  apptainer exec --writable-tmpfs \
  --bind data:/data \
  --bind ~/fondue:/FONDUE/ \
  --bind ~/fondue/archs/FONDUE_LT_nan.py:/FONDUE/archs/FONDUE_LT.py \
  --env THRESHOLD=1 \
  --env NAN_OPS_ENABLED=True \
  --env EPSILON=1e-7 \
  fondue_nan.sif python /FONDUE/fondue_eval_simpleitk.py --in_name SUBJECT_ID --no_cuda 
  ```
* A sample script to run FONDUE in parallel across all slices of 5 subjects' brain with Slurm is available at [`run-all.sh`](https://github.com/InesGP/UNET-nan-operations/blob/main/fondue_usecase/run-all.sh)
