#!/bin/bash -l

#$ -pe omp 4
#$ -P ivc-ml
#$ -l h_rt=1:00:00
#$ -l gpus=1
#$ -l gpu_memory=48G
#$ -N llava_incomplete_im_caption_segframes_polygon
#$ -j y
#$ -m ea
#$ -o outputs/llava_incomplete_im_caption_segframes_polygon.out

cd /projectnb/ivc-ml/array/research/robotics/dreamworlds/
conda activate llava
python -m accelerate.commands.launch --num_processes=2 main.py exp_name=llava_incomplete_im_caption_segframes_polygon
