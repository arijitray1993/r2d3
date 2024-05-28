#!/bin/bash -l

#$ -pe omp 4
#$ -P ivc-ml
#$ -l h_rt=24:00:00
#$ -l gpus=2
#$ -l gpu_memory=48G
#$ -l gpu_type=L40S|L40|A6000|RTX8000
#$ -N llava_incomplete_oneim_campos_nopolygon_mark_nodepth
#$ -j y
#$ -m ea
#$ -o outputs/llava_incomplete_oneim_campos_nopolygon_mark_nodepth.out

cd /projectnb/ivc-ml/array/research/robotics/dreamworlds/
conda activate llava_dino
python -m accelerate.commands.launch --num_processes=1 main.py exp_name=llava_incomplete_oneim_campos_nopolygon_mark_nodepth
