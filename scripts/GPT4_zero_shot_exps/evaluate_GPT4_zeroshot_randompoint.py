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
from dataloaders import ProcTHOR_image_camposition_marked

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
    args = {
      'split': "val",
      'mode': "val",
      'include_children': False,
      'use_angle': True,
      'use_attributes': True,
      'no_polygon': True,
      'use_incontext': True,
      'incontext_pointmark': True,
      'randomize_point': True,
      'normalize_rotation': True
    }

    dataset = ProcTHOR_image_camposition_marked(args, tokenizer=None, image_processor=None)

    # pdb.set_trace()

    all_responses = []
    for entry in tqdm.tqdm(iter(dataset)):
        # pdb.set_trace()

        image_path, img, caption, prompt, text_labels, program_text, house_json, objs_present = entry
        
        image_path =  image_path[0]

        prompt = prompt.split("## HUMAN: <image> ")[-1].split(" \n ASSISTANT: ")[0]

        # pdb.set_trace()
        response = get_caption(image_path, prompt, api_key)

        response_text = response.json()['choices'][0]['message']['content']

        print(response_text)

        # pdb.set_trace()
        # Save the response
        all_responses.append({
          "prompt": prompt,
          "text_label": text_labels,
          
          "image_path": image_path,
          "response": response_text,
          "objs_present": objs_present
        })
        # pdb.set_trace()

        # Save the responses
        with open("responses_randomobjpoint.json", "w") as f:
          json.dump(all_responses, f)
        