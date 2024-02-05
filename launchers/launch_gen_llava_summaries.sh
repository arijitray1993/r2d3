#!/bin/bash -l

#$ -pe omp 3
#$ -P ivc-ml
#$ -l h_rt=8:00:00
#$ -l gpus=1
#$ -l gpu_memory=48G
#$ -N llava_summaries
#$ -j y
#$ -m ea
#$ -o outputs/llava_summaries.out

cd /projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/
conda activate llava

python generate_llava_descriptions.py