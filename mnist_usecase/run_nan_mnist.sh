#!/bin/bash
#SBATCH --job-name=nan_reducedprec_bfloat16_mnist_test
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=1
#SBATCH --time=2:0:0
#SBATCH --array=0-3
#SBATCH --account=rrg-glatard
#SBATCH --output=%x_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inesgp99@gmail.com


export OMP_NUM_THREADS=1
export NUMPEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TASK_ID=$SLURM_ARRAY_TASK_ID

# NAN Inference
#time apptainer exec --env THRESHOLD=0.5 --env EPSILON=1e-7 -B ../mnist/:/mnist mnist.sif python3 /mnist/nan_mnist_test.py --load-model --model-path /mnist/mnist_cnn_pool.pt 

threshold_vals=(
1
0.85
0.75
0.65
0.5
'ieee'
)

echo ${threshold_vals[(${SLURM_ARRAY_TASK_ID})]}
#Normal Test
#time apptainer exec --env THRESHOLD=${threshold_vals[(${SLURM_ARRAY_TASK_ID})]} --env POOL_THRESH=1 --env EPSILON=1e-7 -B ../mnist/:/mnist mnist.sif python3 /mnist/nan_mnist_test.py --load-model --model-path /mnist/mnist_cnn_pool2.pt

#Cross Fold Validation for Testing
export THRESHOLD=${threshold_vals[(${SLURM_ARRAY_TASK_ID})]}
export POOL_THRESH=1 
export EPSILON=1e-7
source /home/ine5/projects/rrg-glatard/ine5/venv/bin/activate
python3 nan_mnist_test_kfold.py --load-model --model-path mnist_cnn_pool2.pt



