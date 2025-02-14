
import json
import os
import sys
import random
import tqdm
from collections import defaultdict
import shutil
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


if __name__ == "__main__":
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

            if "i need to go" in question.lower():
                new_answers = []
                for ans in answers:
                    if "by" in ans:
                        new_ans = "slightly "+ans.split("by")[0]
                    else:
                        new_ans = ans
                    new_answers.append(new_ans)
                answers = tuple(new_answers)

            qa_type_data[qa_type].append((question, im_order, answers))
    
    for qa_type, qa_data in qa_type_data.items():
        qas = random.sample(qa_data, 20)
        all_complex_data.extend(qas)

    
    pers_im_count = 0
    for _,_,_, qa_entries in perspective_data[int(len(perspective_data)*0.1):]:
        pers_im_count += 1
        perspective_count = 0
        random.shuffle(qa_entries)
        for question, im_order, answers in qa_entries:
            question = question.replace("turned towards the", "facing 90 degrees to the")
            question = question.replace("turned right", "turned right by 90 degrees")
            question = question.replace("turned left", "turned left by 90 degrees")

            all_complex_data.append((question, im_order, answers))
            perspective_count += 2
            if perspective_count >= 2:
                break
        if pers_im_count >= 10:
            break

    print("total number of questions:", len(all_complex_data))

    random.shuffle(all_complex_data)

    public_im_folder = "images/"
    html_str = f"<html><head><h1>Examples of SAT Dynamic QAs</h1></head><body>"
    for question, im_order, answers in all_complex_data:
        qa_type = get_qa_type(question)

        html_str += f"<p><b>QA Type: </b> {qa_type}</p>"
        html_str += f"<p>{question}</p>"
        for im in im_order:
            public_im_name = "_".join(im.split("/")[-3:])
            full_public_im_path = os.path.join(public_im_folder, public_im_name)

            shutil.copy(im, full_public_im_path)
            html_im_url = full_public_im_path
            html_str += f"<img src='{html_im_url}' style='width: 400px; height: 400px;'>"
        
        html_str += "<p>"
        
        if random.random() > 0.5:
            html_str += f"<p><b>Choose one of the following options:</b> '{answers[0]}' or '{answers[1]}'</p>"
        else:
            html_str += f"<p><b>Choose one of the following options:</b> '{answers[1]}' or '{answers[0]}'</p>"
        
        html_str += f"<p><b>Correct Answer: </b> {answers[0]}</p>"
       

        html_str += "<p>"
        html_str += "<hr>"
    html_str += "</body></html>"

    with open("anon_html.html", "w") as f:
        f.write(html_str)
        
