import json
from collections import defaultdict
import numpy as np
import os
import sys
sys.path.append("../")
from custom_datasets.dataloaders import CVBench, BLINK
import yaml
import pdb
import tqdm
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

import re



if __name__=="__main__":

    spatial_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_spatial_qas_v2_train.json'

    sp_data = json.load(open(spatial_qa_json_path))

    data = []
    for house_ind, cam_pos, cam_rot, qa_entries in sp_data:
        data.extend(qa_entries)

    
    answer_choice_histogram = defaultdict(lambda: defaultdict(int))
    question_word_histogram = defaultdict(lambda: defaultdict(int))
    for question, image_order, answer_choices in data:
        
        if "how many" in question.lower():
            question_type = "count"

        if "is closer" in question.lower():
            question_type = "obj_depth"
        
        if "considering the relative positions" in question.lower():
            question_type = "sp_rel"

        if 'estimate the real world distances' in question.lower():
            question_type = "relative_distance"

        correct_answer = answer_choices[0]
        correct_answer = correct_answer.split("(")[0].strip().lower()

        answer_choice_histogram[question_type][correct_answer] += 1

        question_words = question.split(" ")
        for word in question_words:
            question_word_histogram[question_type][word] += 1

    # print the top 10 answer words for each question type
    for question_type, answer_choice_hist in answer_choice_histogram.items():
        print(question_type)
        sorted_answer_choice_hist = sorted(answer_choice_hist.items(), key=lambda x: x[1], reverse=True)
        for answer_choice, count in sorted_answer_choice_hist[:10]:
            print(answer_choice, count)
    
    # make a histogram of the top answer words for each question type in a separate plot
    fig, ax = plt.subplots(len(answer_choice_histogram), 1, figsize=(20, 20))
    for i, (question_type, answer_choice_hist) in enumerate(answer_choice_histogram.items()):
        sorted_answer_choice_hist = sorted(answer_choice_hist.items(), key=lambda x: x[1], reverse=True)
        answer_choices = [x[0] for x in sorted_answer_choice_hist]
        counts = [x[1] for x in sorted_answer_choice_hist]
        # rotate the x-axis labels so they don't overlap
        ax[i].bar(answer_choices, counts, label=question_type)
        plt.xticks(rotation=90)
    
        # add the question type labels to each plot
        ax[i].set_title(question_type)
    
    plt.savefig("answer_choice_histogram_r2d3.pdf")


    
    args = {
        'split': "val",
        'mode': "val",
        'num_data_points': 5000,
    }

    blink_data = BLINK(args, None, None)

    cvbench = CVBench(args, None, None)

    blink_answer_choice_histogram = defaultdict(lambda: defaultdict(int))
    cvbench_answer_choice_histogram = defaultdict(lambda: defaultdict(int))
    for entry in blink_data:
        images, prompt, text_label, answer, question_type = entry
        
        blink_answer_choice_histogram[question_type][answer] += 1
    
    for entry in cvbench:
        images, prompt, text_label, answer, question_type = entry
        
        cvbench_answer_choice_histogram[question_type][answer] += 1
    
    print(blink_answer_choice_histogram)
    print(cvbench_answer_choice_histogram)

    
    # make a histogram of the top answer words for each question type in a separate plot for blink and cvbench
    fig, ax = plt.subplots(len(blink_answer_choice_histogram), 1, figsize=(20, 10))
    for i, (question_type, answer_choice_hist) in enumerate(blink_answer_choice_histogram.items()):
        sorted_answer_choice_hist = sorted(answer_choice_hist.items(), key=lambda x: x[1], reverse=True)
        answer_choices = [x[0] for x in sorted_answer_choice_hist[:10]]
        counts = [x[1] for x in sorted_answer_choice_hist[:10]]
        ax[i].bar(answer_choices, counts, label=question_type)
    plt.savefig("blink_answer_choice_histogram.png")

    fig, ax = plt.subplots(len(cvbench_answer_choice_histogram), 1, figsize=(20, 10))
    for i, (question_type, answer_choice_hist) in enumerate(cvbench_answer_choice_histogram.items()):
        sorted_answer_choice_hist = sorted(answer_choice_hist.items(), key=lambda x: x[1], reverse=True)
        answer_choices = [x[0] for x in sorted_answer_choice_hist[:10]]
        counts = [x[1] for x in sorted_answer_choice_hist[:10]]
        ax[i].bar(answer_choices, counts, label=question_type)

    plt.savefig("cvbench_answer_choice_histogram.png")
    