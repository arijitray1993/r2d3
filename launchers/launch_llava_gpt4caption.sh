#!/bin/bash -l

#$ -pe omp 4
#$ -P ivc-ml
#$ -l h_rt=24:00:00
#$ -l gpus=2
#$ -l gpu_memory=48G
#$ -N llava_gpt4caption
#$ -j y
#$ -m ea
#$ -o outputs/llava_gpt4caption.out

cd /projectnb/ivc-ml/array/research/robotics/dreamworlds/
conda activate llava_dino
python -m accelerate.commands.launch --num_processes=2 main.py exp_name=llava_gpt4caption