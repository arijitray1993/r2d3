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
import random
import pandas as pd
import plotly.express as px


import re

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


if __name__=="__main__":
    train= False
    if train:
        complex_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_train_v2.json' # remove v2 for prev version.
        complex_qa_json_path_split2 = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_train_v2_split2.json'
        complex_data = json.load(open(complex_qa_json_path)) + json.load(open(complex_qa_json_path_split2))
        complex_data = random.sample(complex_data, 6900)

        perspective_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/perspective_qas.json'
        perspective_data = json.load(open(perspective_qa_json_path))
        perspective_data = random.sample(perspective_data, 300) 
    
        print("num images in complex data:", len(complex_data))
        print("num images in perspective data:", len(perspective_data))
        all_complex_data = []

        for house_ind, cam_pos, cam_rot, qa_entries in complex_data:
            for question, im_order, answers in qa_entries:
                question = question.replace("turn look straight", "look straight")

                if answers[0] == "rotated left and rotated right" or answers[0] == "rotated right and rotated left": # bug fix
                    new_answers = ["did not move", random.choice(["rotated left", "rotated right"])]
                    answers = new_answers

                all_complex_data.append((question, im_order, answers))
    else:
        complex_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_val_v2.json' # remove v2 for prev version.
        complex_data = json.load(open(complex_qa_json_path))

        perspective_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/perspective_qas.json'
        perspective_data = json.load(open(perspective_qa_json_path))

        print("num images in complex data:", len(complex_data))
        
        all_complex_data = []
        all_action_consequence_data = []
        for house_ind, cam_pos, cam_rot, qa_entries in complex_data[int(len(complex_data)*0.1):]:
            for question, im_order, answers in qa_entries:
                question = question.replace("turn look straight", "look straight")

                if answers[0] == "rotated left and rotated right" or answers[0] == "rotated right and rotated left": # bug fix
                    new_answers = ["did not move", random.choice(["rotated left", "rotated right"])]
                    answers = new_answers
                
                qa_type = get_qa_type(question)
                #if qa_type == "action_consequence":
                #    all_action_consequence_data.append((question, im_order, answers))
                #else: 
                all_complex_data.append((question, im_order, answers))
        
        perspective_count = 0
        pers_im_count = 0
        for _,_,_, qa_entries in perspective_data[int(len(perspective_data)*0.1):]:
            pers_im_count += 1
            for question, im_order, answers in qa_entries:
                question = question.replace("turned towards the", "facing 90 degrees to the")
                question = question.replace("turned right", "turned right by 90 degrees")
                question = question.replace("turned left", "turned left by 90 degrees")

                all_complex_data.append((question, im_order, answers))
                perspective_count += 1
                if perspective_count >= 779:
                    break
            if perspective_count >= 779:
                break

    data = all_complex_data
    
    answer_choice_histogram = defaultdict(lambda: defaultdict(int))
    question_word_histogram = defaultdict(lambda: defaultdict(int))
    for question, image_order, answer_choices in data:
        
        question_type = get_qa_type(question)

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
    fig, ax = plt.subplots(len(answer_choice_histogram), 1, figsize=(20, 40))
    for i, (question_type, answer_choice_hist) in enumerate(answer_choice_histogram.items()):
        sorted_answer_choice_hist = sorted(answer_choice_hist.items(), key=lambda x: x[1], reverse=True)
        answer_choices = [x[0][:30] for x in sorted_answer_choice_hist[:20]]
        counts = [x[1] for x in sorted_answer_choice_hist[:20]]
        # rotate the x-axis labels so they don't overlap
        ax[i].bar(answer_choices, counts, label=question_type)
        ax[i].set_yscale('log')
        #ax[i].xtick_labels(rotation=45)
    
        # add the question type labels to each plot
        ax[i].set_title(question_type)
    for axis in ax:
        axis.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig("dynamic_answer_choice_histogram_r2d3.png")



    ################ SUNBURST PLOT ################

    # Sample questions
    questions = [q for q, _, _ in data]

    # Tokenize questions and extract the first 5 words
    tokenized_questions = [q.split()[:5] for q in questions]

    # Build the hierarchy for the sunburst plot
    def build_hierarchy(tokenized):
        hierarchy = defaultdict(int)
        for ind, tokens in enumerate(tokenized):
            key = tuple(tokens)
            hierarchy[key] = ind
        return hierarchy

    hierarchy = build_hierarchy(tokenized_questions)

    # Prepare data for Plotly
    data = []
    for words, count in hierarchy.items():
        for i in range(len(words)):
            data.append({
                "id": " ".join(words[:i+1]),
                "parent": " ".join(words[:i]) if i > 0 else "",
                "value": count if i == len(words) - 1 else None
            })

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Fill root node
    df.loc[df['parent'] == "", 'parent'] = "Root"

    # Sunburst plot
    fig = px.sunburst(
        df,
        names="id",
        parents="parent",
        values="value",
        title="Word Distribution in Questions (First 5 Words)"
    )

    fig.write_image("dynamic_sunburst_plot_r2d3.png") 
        