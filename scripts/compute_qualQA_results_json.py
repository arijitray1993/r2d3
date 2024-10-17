import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import models.eval_funcs as eval_funcs
import yaml
import wandb

import utils.ai2thor_utils as ai2thor_utils
import random
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
import torch
import base64
import requests
import pdb
from custom_datasets.dataloaders import AllMLMBench
import tqdm
import numpy as np
import wandb
from collections import defaultdict



if __name__=="__main__":
    
    #json_file = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/GPT4_zero_shot_exps/responses_randomobjpoint_updated.json"
    json_file1 = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/checkpoints/llava_mixdata_reasoning_MLMBench/output.json"
    json_file2 = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/checkpoints/zeroshot_llava_MLMBench/output.json"

    log = False
    if log:
        exp_name = "bothwrong_compare_"+json_file1.split('/')[-2]+"_"+json_file2.split('/')[-2]
        wandb.login()
        run = wandb.init(project=exp_name)
        logger = run

        table = wandb.Table(columns=["prompt", "gt_answer", "answer1", "answer2", "image"])

    data1 = json.load(open(json_file1))
    data2 = json.load(open(json_file2))

    args = {
        'split': "val",
        'mode': "val",
        'num_data_points': 5000,
    }
    
    dataset = AllMLMBench(args, None, None)
    
    all_correct = defaultdict(list)
    count = 0
    for entry1, entry2, entry3 in tqdm.tqdm(zip(data1, data2, dataset)):
        images, prompt, text_label, answer, dataname = entry3

        gt_question, gt_answers, pred_answers1, data_name, correct1 = entry1
        gt_question, gt_answers, pred_answers2, data_name, correct2 = entry2
        
        if 'Human: Answer in natural language.' not in pred_answers2:
            count += 1
            all_correct[dataname].append(correct1)

        if not correct2 and not correct1:
            if log:    
                table.add_data(gt_question, gt_answers, pred_answers1, pred_answers2, wandb.Image(images[0]))

    print(count)
    # print acc for all_correct
    for key, value in all_correct.items():
        print(key, np.mean(value))

    if log:
        logger.log({"compare": table})