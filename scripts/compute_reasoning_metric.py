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



def compute_metrics_from_json_gpt_reconqa(json_file, logger):

    data = json.load(open(json_file))
    print("Number of samples: ", len(data))
    # initialize metrics
    args = {}
    args['logger'] = logger
    args['exp_name'] = json_file.split('/')[-2]
    accs = eval_funcs.ReasoningAccuracy(args)

    for entry in data:
        # pdb.set_trace()
        
        response_text = entry['response']

        entry_dict = {
            'dataset': [entry['dataset'],],
            'prompts': [entry['prompt'],],
            'answers': [entry['answer'],],
            'answer_choices': [entry['answer_choices'],],
        }
        # compute metrics
        accs.update(response_text, entry_dict)
        

    # Compute the metrics
    accs.compute()


def compute_metrics_from_json(json_file, logger):

    data = json.load(open(json_file))
    print("Number of samples: ", len(data))
    # initialize metrics
    args = {}
    args['logger'] = logger
    args['exp_name'] = json_file.split('/')[-2]
    accs = eval_funcs.ReasoningAccuracy(args)

    for entry in data:
        # pdb.set_trace()
        gt_question, gt_answers, pred_answers, data_name = entry
        
        response_text = pred_answers

        entry_dict = {
            'dataset': [data_name,],
            'prompts': [gt_question,],
            'answers': gt_answers,
            'answer_choices': entry['answer_choices'],
        }
        # compute metrics
        accs.update(response_text, entry_dict)
        

    # Compute the metrics
    accs.compute()


if __name__=="__main__":
    
    json_file = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/GPT4_zero_shot_exps/GPT4_procthorreasoning_response_updated.json"

    exp_name = json_file.split('/')[-2]
    wandb.login()
    run = wandb.init(project=exp_name)
    logger = run
    
    #compute_metrics_from_json(json_file, logger)
    compute_metrics_from_json_gpt_reconqa(json_file, logger)

    # compute_image_caption_metrics(json_file, logger, eval_caption_sim=False)
    
    