#!/bin/bash
#SBATCH --job-name=nan_xception_baseline
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=0:40:0
#SBATCH --account=rrg-glatard
#SBATCH --output=%x.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inesgp99@gmail.com

source /home/ine5/projects/rrg-glatard/ine5/venv/bin/activate

time python xception_baseline.py

