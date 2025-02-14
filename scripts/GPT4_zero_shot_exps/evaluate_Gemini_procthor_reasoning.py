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

import google.generativeai as genai
import os

api_key_file = "/projectnb/ivc-ml/array/research/robotics/gemini"
with open(api_key_file, "r") as f:
  api_key = f.read().strip()


genai.configure(api_key=api_key)


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_caption(image_paths, prompt, model):
  # Choose a Gemini model.

  response = model.generate_content([prompt, *image_paths])

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

    dataset = ProcTHOR_reasoning(args, tokenizer=None, image_processor=None)

    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    all_responses = []
    for ind, entry in enumerate(tqdm.tqdm(dataset)):
        image_paths, images, prompt, text_label, correct_answer, answer_choices, datatype = entry

        prompt = prompt.split("Human: Answer in natural language.")[-1].split("###Assistant")[0]

        

        response = get_caption(image_paths, prompt, model)

        # pdb.set_trace()

        response_text = response.text
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
        if ind % 100 == 0:
          with open("gemini_complexreasoning_response.json", "w") as f:
              json.dump(all_responses, f)

        if ind>=4000:
            break
      