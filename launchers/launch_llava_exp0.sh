#!/bin/bash -l

#$ -pe omp 3
#$ -P ivc-ml
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_memory=48G
#$ -N llava_exp0
#$ -j y
#$ -m ea
#$ -o outputs/llava_exp0.out

cd /projectnb/ivc-ml/array/research/robotics/dreamworlds/
conda activate llava
python -m accelerate.commands.launch --num_processes=0 main.py exp_name=llava_exp0