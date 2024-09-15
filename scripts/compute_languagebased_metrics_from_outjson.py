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


def compute_metrics_from_json_gpt(json_file, logger):

    data = json.load(open(json_file))
    print("Number of samples: ", len(data))
    # initialize metrics
    args = {}
    args['logger'] = logger
    accs = eval_funcs.HouseNatLanguageSemSimSelected(args)

    for entry in data:
        prompt = entry['prompt']
        text_labels = [entry["text_label"],]
        image_path = entry["image_path"]
        response_text = entry["response"]
        # objs_present = entry["objs_present"]

        gt_dict = {}
        gt_dict['text_labels'] = text_labels
        
        # compute metrics
        accs.update(response_text, gt_dict)
        
    # Compute the metrics
    accs.compute()

def compute_metrics_from_json_gpt_simpl(json_file, logger):

    data = json.load(open(json_file))
    print("Number of samples: ", len(data))
    # initialize metrics
    args = {}
    args['logger'] = logger
    accs = eval_funcs.HouseNatLanguageSemSimSelectedGPT4(args)

    for ind, entry in enumerate(data):
        if ind>500:
            break
        # pdb.set_trace()
        prompt = entry['prompt']
        text_labels = [entry["text_label"],]
        image_path = entry["image_path"]
        response_text = entry["response"]
        objs_present = entry["objs_present"]

        gt_dict = {}
        gt_dict['text_labels'] = text_labels
        gt_dict['objs_present'] = objs_present
        
        # compute metrics
        accs.update(response_text, gt_dict)
        

    # Compute the metrics
    accs.compute()



def compute_metrics_from_json(json_file, logger):

    data = json.load(open(json_file))
    print("Number of samples: ", len(data))
    # initialize metrics
    args = {}
    args['logger'] = logger
    accs = eval_funcs.HouseNatLanguageSemSimSelected(args)

    for entry in data:

        output, gt_house_json, gt_house_dict, gt_text_labels = entry
        
        text_labels = [gt_text_labels,]

        gt_dict = {}
        gt_dict['text_labels'] = text_labels
        gt_dict['objs_present'] = gt_house_dict['objs_present']
        
        # compute metrics
        accs.update(output, gt_dict)
        
    # Compute the metrics
    accs.compute()


if __name__=="__main__":
    
    json_file = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/GPT4_zero_shot_exps/responses_randomobjpoint_updated.json"

    exp_name = json_file.split('/')[-2]
    wandb.login()
    run = wandb.init(project=exp_name)
    logger = run
    
    #compute_metrics_from_json(json_file, logger)
    compute_metrics_from_json_gpt_simpl(json_file, logger)

    # compute_image_caption_metrics(json_file, logger, eval_caption_sim=False)
    
    