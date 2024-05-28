#!/bin/bash -l

#$ -pe omp 4
#$ -P ivc-ml
#$ -l h_rt=2:00:00
#$ -l gpus=1
#$ -l gpu_memory=48G
#$ -l gpu_type=RTX8000|L40|A6000
#$ -N llava_incomplete_im_caption_allcornerims_lite_sparse_larger
#$ -j y
#$ -m ea
#$ -o outputs/llava_incomplete_im_caption_allcornerims_lite_sparse_larger.out

cd /projectnb/ivc-ml/array/research/robotics/dreamworlds/
conda activate llava
python -m accelerate.commands.launch --num_processes=2 main.py exp_name=llava_incomplete_im_caption_allcornerims_lite_sparse_larger
