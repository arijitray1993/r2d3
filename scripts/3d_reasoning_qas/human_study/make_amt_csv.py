import json
import os
import sys
import random
import tqdm
from collections import defaultdict
import pdb
import shutil
import numpy as np
import csv

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


if __name__ == "__main__":
    '''
    required format for llava
    [
        {
            "id": "997bb945-628d-4724-b370-b84de974a19f",
            "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
            "conversations": [
            {
                "from": "human",
                "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
            },
            {
                "from": "gpt",
                "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
            },
            ]
        },
        ...
    ]
    '''
    
    complex_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_val_v2.json' # remove v2 for prev version.
    complex_data = json.load(open(complex_qa_json_path))

    perspective_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/perspective_qas.json'
    perspective_data = json.load(open(perspective_qa_json_path))
    
    all_complex_data = []
    qa_type_data = defaultdict(list)
    for house_ind, cam_pos, cam_rot, qa_entries in complex_data[int(len(complex_data)*0.1):]:
        for question, im_order, answers in qa_entries:
            question = question.replace("turn look straight", "look straight")

            if answers[0] == "rotated left and rotated right" or answers[0] == "rotated right and rotated left": # bug fix
                new_answers = ["did not move", random.choice(["rotated left", "rotated right"])]
                answers = new_answers
            
            qa_type = get_qa_type(question)

            question = question.replace("frame", "image") 
            
            if "in the first frame" in answers[0] or "in the first frame" in answers[1]:
                new_answers = (answers[0].replace("in the first frame", ""), answers[1].replace("in the first frame", ""))
                answers = new_answers
            #if qa_type == "action_consequence":
            #    all_action_consequence_data.append((question, im_order, answers))
            #else: 
            qa_type_data[qa_type].append((question, im_order, answers))
    
    for qa_type, qa_data in qa_type_data.items():
        qas = random.sample(qa_data, 50)
        all_complex_data.extend(qas)

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
            if perspective_count >= 50:
                break
        if perspective_count >= 50:
            break

    print("total number of questions:", len(all_complex_data))
    # make csv with headers as img1 path, img2 path, question, answer1, answer2, correct
    csv_data = [["img1", "img2", "question", "answer1", "answer2", "correct"],] 
    public_im_folder = "/net/cs-nfs/home/grad2/array/public_html/research/r2d3/multi_qa_ims/human_study/"
    if not os.path.exists(public_im_folder):
        os.makedirs(public_im_folder)

    for question, im_order, answers in all_complex_data:
        
        im_paths = []
        for im in im_order:
            public_im_name = "_".join(im.split("/")[-3:])
            full_public_im_path = os.path.join(public_im_folder, public_im_name)

            shutil.copy(im, full_public_im_path)

            html_im_name = "https://cs-people.bu.edu/array/"+full_public_im_path.split("/net/cs-nfs/home/grad2/array/public_html/")[-1]

            im_paths.append(html_im_name)
        
        if len(im_paths) == 1:
            im_paths.append("")

        if random.random()<0.5:
            csv_data.append([im_paths[0], im_paths[1], question, answers[0], answers[1], "1"])
        else:
            csv_data.append([im_paths[0], im_paths[1], question, answers[1], answers[0], "2"])

    with open("amt_data.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

        