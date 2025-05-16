#!/bin/bash
#SBATCH --job-name=reduced_prec_gpu_float32_random_idx_nanconv_basic_runtime
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --time=UNLIMITED
#SBATCH --array=1-10
#SBATCH --account=rrg-glatard
#SBATCH --output=%x_%a.out

source /home/inesgp/torch_env/bin/activate

# python nanconv_runtime.py
python reduced_prec_nanconv_runtime.py