#!/bin/bash -l

#$ -pe omp 1
#$ -P ivc-ml
#$ -l h_rt=20:00:00
#$ -N gpt4
#$ -j y
#$ -m ea
#$ -o gpt4.out

cd /projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/
conda activate llava
python generate_gpt4_descriptions.py