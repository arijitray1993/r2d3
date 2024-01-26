#!/bin/bash -l

#$ -pe omp 3
#$ -P ivc-ml
#$ -l h_rt=8:00:00
#$ -l gpus=1
#$ -l gpu_memory=48G
#$ -N blip2_exp0
#$ -j y
#$ -m ea
#$ -o outputs/blip2_exp0.out


python -m accelerate.commands.launch --num_processes=2 main.py exp_name=blip2_exp0
python -m accelerate.commands.launch --num_processes=2 main.py exp_name=instructblip_exp0
python -m accelerate.commands.launch --num_processes=0 main.py exp_name=llava_exp0