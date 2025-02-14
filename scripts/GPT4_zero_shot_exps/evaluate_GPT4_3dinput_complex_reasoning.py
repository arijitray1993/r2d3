import base64
import requests
import tqdm
import json
import os
import pdb
from collections import defaultdict
import random
from PIL import Image
import cv2
import re
import tqdm

import sys
sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/")
sys.path.append('../../')
from dataloaders import ProcTHOR_reasoning, get_qa_type


# OpenAI API Key
api_key_file = "/projectnb/ivc-ml/array/research/robotics/openai"
with open(api_key_file, "r") as f:
    api_key = f.read().strip()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_caption(image_path, prompt, api_key):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response


if __name__=="__main__":

    # Load the dataset
    
    args= {
        "split": "val",
        "mode": "val",
        "prompt_mode": "text_choice",
        "complex_only": True,
        "add_complex": True, 
        "add_perspective": True
    } 

    input_3d = True

    # dataset = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/satdynamicqa_3dinfo.json"))
    dataset = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/satdynamicqa_3dinfo_v2.json"))

    all_responses = []
    all_responses = json.load(open("GPT4_3dinput_complexreasoning_response.json"))

    for ind, entry in enumerate(tqdm.tqdm(dataset)):
        house_ind, cam_pos, cam_rot, question, image_paths, answer_choices, all_obj_descs = entry
        
        correct_answer = answer_choices[0]
        if len(image_paths)==1:
            image_path =  image_paths[0]
        else:
            # join the images into one
            image1 = cv2.imread(image_paths[0])
            image2 = cv2.imread(image_paths[1])
            image = cv2.hconcat([image1, image2])
            cv2.imwrite("temp.jpg", image)
            image_path = "temp.jpg"
          
        all_obj_descs = ", ".join(all_obj_descs)

        qatype = get_qa_type(question)

        answer_choices_format = " or ".join([f'"{ans}"' for ans in answer_choices])

        if input_3d:
            prompt = f"This is the list of objects and their 3D locations in x, y, z in the image shown. Assume camera is at 0, 0, 0 and pointing parallel to the floor. x increases towards the right, y increases with height and z increases with depth. {all_obj_descs}. If there are two images, the 3D locations are of the first (left) image shown. Based on this and using other information from the image, try to answer the question: {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}\n"
        else:
            prompt = f"{question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}\n"

        text_label = f"{prompt} {correct_answer}"
        datatype = f"complexreasoning_{qatype}"
        # pdb.set_trace()

        response = get_caption(image_path, prompt, api_key)

        try:
            response_text = response.json()['choices'][0]['message']['content']
        except:
            response_text = "No response"
          
        if ind % 20 == 0:
          print(prompt, response_text)
        # Save the response
        all_responses.append({
            "prompt": prompt,
            "text_label": text_label,
            "image_path": image_paths,
            "response": response_text,
            "answer": correct_answer,
            "answer_choices": answer_choices,
            "dataset": datatype
        })
        # pdb.set_trace()

        # Save the responses
        if ind % 10 == 0:
            with open("GPT4_3dinput_complexreasoning_response.json", "w") as f:
                json.dump(all_responses, f)

        if ind>=1000:
            break
      