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
from custom_datasets.dataloaders import AllMLMBench, ProcTHOR_reasoning
import tqdm
import numpy as np
import wandb
from collections import defaultdict



if __name__=="__main__":
    
    #json_file = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/GPT4_zero_shot_exps/responses_randomobjpoint_updated.json"
    json_file1 = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/checkpoints/zeroshot_llava_MLMBench/output.json"
    json_file2 = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/checkpoints/llava_mixdata_IT_3dcapqa_complexreasoning_MLMBench/output.json"

    log = True
    if log:
        exp_name = "compare_"+json_file1.split('/')[-2]+"_"+json_file2.split('/')[-2]
        wandb.login()
        run = wandb.init(project=exp_name)
        logger = run
        if "MLMBench" in json_file1:
            table = wandb.Table(columns=["prompt", "gt_answer", "answer1", "answer2", "image1", "image2"])
        else:
            table = wandb.Table(columns=["prompt", "gt_answer", "answer1", "answer2", "image1", "image2"])

    data1 = json.load(open(json_file1))
    data2 = json.load(open(json_file2))

    if "MLMBench" in json_file1:
        args = {
            'split': "val",
            'mode': "val",
            'num_data_points': 5000,
        }
        dataset = AllMLMBench(args, None, None)
        
    else:
        args={
            'split': "val",
            'mode': "val",
            'prompt_mode': "text_choice",
            'complex_only': True,
            'add_complex': True, 
            'add_perspective': True
        }
        dataset = ProcTHOR_reasoning(args, None, None)
    
    all_correct = defaultdict(list)
    count = 0
    for entry1, entry2, entry3 in tqdm.tqdm(zip(data1, data2, dataset)):
        if "MLMBench" in json_file1:
            images, prompt, text_label, answer, dataname = entry3
        else:
            image_filenames, images, prompt, text_label, answer, answer_choices, dataname = entry3

        try:
            gt_question, gt_answers, pred_answers1, data_name, correct1 = entry1
            gt_question, gt_answers, pred_answers2, data_name, correct2 = entry2
        except:
            gt_question, gt_answers, pred_answers1, data_name= entry1
            gt_question, gt_answers, pred_answers2, data_name= entry2

            format_answer1 = pred_answers1.split("\n")[0].split("###")[-1].strip().lower()
            gt_answer = gt_answers[0].lower().strip()

            correct1 = gt_answer in format_answer1 or format_answer1 in gt_answer
            
            format_answer2 = pred_answers2.split("\n")[0].split("###")[-1].strip().lower()
            
            correct2 = gt_answer in format_answer2 or format_answer2 in gt_answer

            
        #if 'Human: Answer in natural language.' not in pred_answers2:
        #    count += 1
        #    all_correct[dataname].append(correct1)

        if correct2 and not correct1:
            if log:   
                if "MLMBench" in json_file1:
                    if len(images) == 1:
                        table.add_data(prompt, gt_answers, pred_answers1, pred_answers2, wandb.Image(images[0]), wandb.Image(images[0]))
                    else:
                        table.add_data(prompt, gt_answers, pred_answers1, pred_answers2, wandb.Image(images[0]), wandb.Image(images[1]))
                else:
                    if len(images) == 1:
                        image_a = Image.open(image_filenames[0])
                        table.add_data(gt_question, gt_answers, pred_answers1, pred_answers2, wandb.Image(image_a), wandb.Image(image_a))
                    else: 
                        image_a = Image.open(image_filenames[0])
                        image_b = Image.open(image_filenames[1])
                        table.add_data(gt_question, gt_answers, pred_answers1, pred_answers2, wandb.Image(image_a), wandb.Image(image_b))

    print(count)
    # print acc for all_correct
    for key, value in all_correct.items():
        print(key, np.mean(value))

    if log:
        logger.log({"compare": table})