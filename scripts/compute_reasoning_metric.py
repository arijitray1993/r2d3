from collections import defaultdict
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

def get_qa_type(question):
    question_type = "other"
    
    if "how did the camera" in question.lower() or "is the camera moving" in question.lower():
        question_type = "action_sequence"

    if ("need to go" in question.lower()):
        question_type = "goal_aim"

    if "any of the objects in the initial" in question.lower():
        question_type = "obj_movement"

    if "if i" in question.lower():
        question_type = "action_consequence"

    if 'if i move to the' in question.lower() or "for someone at the" in question.lower():
        question_type = "perspective"

    
    return question_type



def compute_metrics_from_json_gpt_reconqa(json_file, logger):

    data = json.load(open(json_file))
    print("Number of samples: ", len(data))
    # initialize metrics
    args = {}
    args['logger'] = logger
    args['exp_name'] = json_file.split('/')[-1].split('.')[0]
    accs = eval_funcs.ReasoningAccuracy(args)

    question_type_count = defaultdict(int)
    for entry in data:
        # pdb.set_trace()
        
        response_text = entry['response']

        question = entry['prompt']
        question_type = get_qa_type(question)
        question_type_count[question_type] += 1

        entry_dict = {
            'dataset': ["procthor_reasoning_"+question_type,],
            'prompts': [entry['prompt'],],
            'answers': [entry['answer'],],
            'answer_choices': [entry['answer_choices'],],
        }
        # compute metrics
        accs.update([response_text,], entry_dict)
        
    

    # Compute the metrics
    # accs.compute()

    print("====================")
    print(question_type_count)
    print("Number of samples: ", len(data))


def compute_metrics_from_json(json_file, logger):

    data = json.load(open(json_file))
    print("Number of samples: ", len(data))
    # initialize metrics
    args = {}
    args['logger'] = logger
    args['exp_name'] = json_file.split('/')[-2]
    accs = eval_funcs.ReasoningAccuracy(args)

    question_type_count = defaultdict(int)
    for entry in data:
        # pdb.set_trace()
        gt_question, gt_answers, pred_answers, data_name = entry
        question_type = get_qa_type(gt_question[0])
        
        question_type_count[question_type] += 1

        response_text = pred_answers

        entry_dict = {
            'dataset': ["procthor_reasoning_"+question_type,],
            'prompts': gt_question,
            'answers': gt_answers,
        }
        # compute metrics
        #pdb.set_trace()
        accs.update([response_text,], entry_dict)
        
    print("====================")
    print(question_type_count)
    print("Number of samples: ", len(data))
    # Compute the metrics
    accs.compute()

def compute_metric_robopoint(jsonl_file, logger):

    with open(jsonl_file) as f:
        data = [json.loads(line) for line in f]

    gt_qa_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/3d_reasoning_qas/llava_complex_spatial_reasoning_val.json"))
    
    args = {}
    args['logger'] = logger
    args['exp_name'] = json_file.split('/')[-1]
    accs = eval_funcs.ReasoningAccuracy(args)


    # format data in the eay our eval function wants
    for entry, gt_entry in zip(data, gt_qa_data):
        prompt = entry['prompt'].strip().lower()
        answer = entry['text'].strip().lower()
        question_type = get_qa_type(prompt)
        gt_answer = gt_entry['conversations'][1]['value'].strip().lower()

        entry_dict = {
            'dataset': ["procthor_reasoning_"+question_type,],
            'prompts': [prompt,],
            'answers': [gt_answer,],
        }

        accs.update([answer,], entry_dict)
    
    accs.compute()


    


if __name__=="__main__":
    
    # json_file = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/GPT4_zero_shot_exps/GPT4_complexreasoning_response.json"
    json_file = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/checkpoints/llava_mixdata_IT_VSR25/output.json"
    # json_file = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/checkpoints/robopoint/results.jsonl"
    exp_name = json_file.split('/')[-2].split('.')[0]
    wandb.login()
    run = wandb.init(project=exp_name)
    logger = run
    
    # compute_metric_robopoint(json_file, logger)
    compute_metrics_from_json(json_file, logger)
    # compute_metrics_from_json_gpt_reconqa(json_file, logger)

    # compute_image_caption_metrics(json_file, logger, eval_caption_sim=False)
    
    