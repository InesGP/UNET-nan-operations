#!/bin/bash
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=64
#SBATCH --time=0:20:0
#SBATCH --array=0-3
#SBATCH --account=def-glatard
#SBATCH --output=%x_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inesgp99@gmail.com

threshold_vals=(
1
0.75
0.5
0.25
)

epsilon_vals=(
1e-7
1e-5
1e-3
1e-2
)

echo "NAN CONV THRESHOLD"
# echo ${threshold_vals[(${SLURM_ARRAY_TASK_ID})]}
echo $1
echo "EPSILON"
echo ${epsilon_vals[(${SLURM_ARRAY_TASK_ID})]}

export EPSILON=${epsilon_vals[(${SLURM_ARRAY_TASK_ID})]}
# export NAN_CONV_THRESH=${threshold_vals[(${SLURM_ARRAY_TASK_ID})]}
export NAN_CONV_THRESH=$1

source /home/ine5/projects/rrg-glatard/ine5/venv/bin/activate

parallel < subjects.txt


# Don't go below 0.25, can test 0.75
# For NAN_CONV_THRESH = 0.5 don't drop below EPSILON = 1e-3 (1e-2 fails)
# NAN_CONV_THRESH = 0.4, EPSILON = 1e-6, 1e-5, 1e-4, 1e-3 1e-2=(fails)
# NAN_CONV_THRESH = 0.3, EPSILON = 1e-6, 1e-5, 1e-4, 1e-3 1e-2=(fails)

#DO: NANCONV= 1, 0.75, 0.5, 0.3 with 1e-7 1e-5 1e-3
