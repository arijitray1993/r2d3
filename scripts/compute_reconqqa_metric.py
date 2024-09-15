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
    accs = eval_funcs.ReconQAAcc(args)

    for entry in data:
        # pdb.set_trace()
        
        response_text = entry['response']
        # compute metrics
        accs.update(response_text, entry)
        

    # Compute the metrics
    accs.compute()



if __name__=="__main__":
    
    json_file = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/GPT4_zero_shot_exps/GPT4_responses_recon_qa.json"

    exp_name = json_file.split('/')[-2]
    wandb.login()
    run = wandb.init(project=exp_name)
    logger = run
    
    #compute_metrics_from_json(json_file, logger)
    compute_metrics_from_json_gpt_reconqa(json_file, logger)

    # compute_image_caption_metrics(json_file, logger, eval_caption_sim=False)
    
    