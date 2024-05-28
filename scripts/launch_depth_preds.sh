#!/bin/bash -l

#$ -pe omp 3
#$ -P ivc-ml
#$ -l h_rt=18:00:00
#$ -l gpus=1
#$ -l gpu_memory=16G
#$ -N depth
#$ -j y
#$ -m ea
#$ -o outputs/depth.out

cd /projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/
conda activate depthanything
python compute_depth.py
