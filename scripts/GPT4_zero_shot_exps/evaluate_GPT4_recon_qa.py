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
import ast
import yaml
import numpy as np

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

    data = json.load(open("all_recon_qas_randompoint.json"))

    all_responses = []
    for entry in tqdm.tqdm(data):
        full_prompt = entry["prompts"]
        image_path = entry["image_path"]
        answers = entry['answers']
        answer_choices = entry['answer_choices']
        
        response = get_caption(image_path[0], full_prompt, api_key)

        response_text = response.json()['choices'][0]['message']['content']
        
        reposnse_text = response_text.strip().lower()
        response_text = response_text.replace("(", "[")
        response_text = response_text.replace(")", "]")

        print("Prompt: ", full_prompt)
        print("Answer: ", answers)

        print("GPT4 response", response_text)

        # pdb.set_trace()
        # Save the response
        all_responses.append({
            "prompts": full_prompt,
            "image_path": image_path,
            "response": response_text,
            "answers": str(answers),
            "answer_choices": answer_choices
        })
        # pdb.set_trace()

        # Save the responses
        with open("GPT4_responses_recon_qa.json", "w") as f:
            json.dump(all_responses, f)

        
    