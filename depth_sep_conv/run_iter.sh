#!/bin/bash

for thresh in 1 0.75 0.5 0.25; do
# for thresh in 0.5; do
     sbatch --job-name=nan_xception_test_batch15_$thresh run_xception.sh $thresh
     # sbatch --job-name=nan_xception_test_batch2_$thresh run_xception.sh $thresh
done

for thresh in 1 0.75 0.5 0.25; do
     sbatch --job-name=nan_xception_test_batch16_$thresh run_xception2.sh $thresh
done