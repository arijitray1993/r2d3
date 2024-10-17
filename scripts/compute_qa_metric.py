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
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS



def compute_qa_metrics_from_json_gpt(json_file):
    exp_name = json_file.split('/')[-2]
    wandb.login()
    run = wandb.init(project=exp_name)
    logger = run
    

    data = json.load(open(json_file))
    print("Number of samples: ", len(data))
    # initialize metrics
    args = {}
    args['logger'] = logger
    args['exp_name'] = json_file.split('/')[-2]
    accs = eval_funcs.QAAccuracy(args)

    for entry in data:
        # pdb.set_trace()
        
        response_text = entry['response']

        entry_dict = {
            'dataset': [entry['dataset'],],
            'prompts': [entry['prompt'],],
            'answers': [entry['answer'],],
        }
        # compute metrics
        accs.update(response_text, entry_dict)
        

    # Compute the metrics
    accs.compute()

def compute_qa_metrics_from_json(json_file):
    exp_name = json_file.split('/')[-2]
    wandb.login()
    run = wandb.init(project=exp_name)
    logger = run
    

    data = json.load(open(json_file))
    print("Number of samples: ", len(data))
    # initialize metrics
    args = {}
    args['logger'] = logger
    args['exp_name'] = json_file.split('/')[-2]
    args['log_table'] = False
    accs = eval_funcs.QAAccuracy(args)

    for entry in data:
        # pdb.set_trace()
        gt_question, gt_answers, pred_answers, data_name, correct = entry
        response_text = pred_answers

        entry_dict = {
            'dataset': [data_name,],
            'prompts': [gt_question,],
            'answers': gt_answers,
        }
        # compute metrics
        accs.update(response_text, entry_dict)
        

    # Compute the metrics
    accs.compute()


def compute_stats_from_pred_json(json_file):
    data = json.load(open(json_file))

    correct_spatial_words = []
    wrong_spatial_words = []
    for entry in data:
        # pdb.set_trace()
        gt_question, gt_answers, pred_answers, data_name, correct = entry

        gt_question = gt_question.split("Answer in natural language.")[1]

        gt_question = gt_question.replace("?", "")

        if "BLINK_Spatial_Relation" in data_name:
            if correct:
                correct_spatial_words.extend(gt_question.split(" "))
            else:
                wrong_spatial_words.extend(gt_question.split(" "))

            
    common_prompts = [
        "Which point is closer to the camera",
        "Considering the relative positions",
        "Which object is",
        "Estimate the real world distances between objects in this image",
        "highlighted by",
        "red box",
        "blue box",
        "green box",
        "yellow box",
        "choose",
        "Assistant",
        "circled",
        "marked",
        "following options",
        "choose between",
        "one of the",
        "The images are frames from a video. The first image is from the beginning of the video and the second image is from the end. Is the camera moving",
        "when shooting the video",
        "Select between A and B",

    ]

    stop_words = []
    for prompt in common_prompts:
        stop_words.extend(prompt.split(" "))

    stop_words += ["a", "A", "the", "on", "in", "and", "of", "with", "is", "what", "considering", "relative", "distance", "distances", "which", "how", "or", "yes", "no"]

    
    # get the wrds that have the highest wrong to correct ratio
    correct_spatial_words = [word for word in correct_spatial_words if word not in stop_words]
    wrong_spatial_words = [word for word in wrong_spatial_words if word not in stop_words]

    wrong_word_counts = defaultdict(int)
    correct_word_counts = defaultdict(int)
    for word in wrong_spatial_words:
        wrong_word_counts[word] += 1

    for word in correct_spatial_words:
        correct_word_counts[word] += 1
    
    wrong_correct_ratio = {}
    for word in wrong_word_counts:
        wrong_correct_ratio[word] = wrong_word_counts[word]/(correct_word_counts[word]+1)

    sorted_wrong_correct_ratio = sorted(wrong_correct_ratio.items(), key=lambda x: x[1], reverse=True)

    wrong_word_str = ""
    wrong_words = [word for word, count in sorted_wrong_correct_ratio[:30]]
    correct_words = [word for word, count in sorted_wrong_correct_ratio[-30:]]
    
    print("Top 30 wrong words: ", wrong_words)
    print("Top 30 correct words: ", correct_words)

    wrong_word_str = " ".join(wrong_words)


    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stop_words,
                min_font_size = 10).generate(wrong_word_str)
    
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    
    plt.savefig("qa_wordcloud_correct.png")

    '''
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stop_words,
                min_font_size = 10).generate(wrong_spatial_words)       

    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    plt.savefig("qa_wordcloud_wrong.png")
    '''

if __name__=="__main__":
    
    #json_file = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/GPT4_zero_shot_exps/GPT4_responses_mlmBench_fixed.json"
    json_file = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/checkpoints/llava_mixdata_reasoning_randompointrecon_MLMBench/output.json"

    
    #compute_metrics_from_json(json_file, logger)
    compute_qa_metrics_from_json(json_file)

    # compute_image_caption_metrics(json_file, logger, eval_caption_sim=False)
    
    # compute_stats_from_pred_json(json_file)