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
from dataloaders import ProcTHOR_reasoning


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
        'prompt_mode': 'text_choice'
    }

    dataset = ProcTHOR_reasoning(args, tokenizer=None, image_processor=None)

    all_responses = []
    for ind, entry in enumerate(tqdm.tqdm(dataset)):
        image_paths, images, prompt, text_label, correct_answer, answer_choices, datatype = entry
        
        image_path =  image_paths[0]

        prompt = prompt.split("Human: Answer in natural language.")[-1].split("###Assistant")[0]

        pdb.set_trace()

        response = get_caption(image_path, prompt, api_key)

        response_text = response.json()['choices'][0]['message']['content']
        # print(response_text)
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
        with open("GPT4_procthorreasoning_response.json", "w") as f:
            json.dump(all_responses, f)

        if ind>=2000:
            break
      