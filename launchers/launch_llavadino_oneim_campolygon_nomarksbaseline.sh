#!/bin/bash -l

#$ -pe omp 4
#$ -P ivc-ml
#$ -l h_rt=30:00:00
#$ -l gpus=2
#$ -l gpu_memory=48G
#$ -l gpu_type=L40S|L40|A6000
#$ -N llavadino_oneim_campolygon_nomarksbaseline
#$ -j y
#$ -m ea
#$ -o outputs/llavadino_oneim_campolygon_nomarksbaseline.out

cd /projectnb/ivc-ml/array/research/robotics/dreamworlds/
conda activate llava_dino
python -m accelerate.commands.launch --num_processes=1 main.py exp_name=llavadino_oneim_campolygon_nomarksbaseline
