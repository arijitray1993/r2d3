#!/bin/bash -l

#$ -pe omp 3
#$ -P ivc-ml
#$ -l h_rt=6:00:00
#$ -l gpus=1
#$ -l gpu_memory=48G
#$ -N precomput_topdown_feats
#$ -j y
#$ -m ea
#$ -o outputs/precomput_topdown_feats.out

cd /projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/
conda activate llava_dino
python precompute_top_clip_feats.py