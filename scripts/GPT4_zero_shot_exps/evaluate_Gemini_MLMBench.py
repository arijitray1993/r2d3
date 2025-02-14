import base64
from io import BytesIO
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
import numpy as np

import sys
sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/")
sys.path.append('../../')
from dataloaders import AllMLMBench


import google.generativeai as genai
import os

api_key_file = "/projectnb/ivc-ml/array/research/robotics/gemini"
with open(api_key_file, "r") as f:
  api_key = f.read().strip()


genai.configure(api_key=api_key)



def get_caption(images, prompt, model):
  # Choose a Gemini model.

  response = model.generate_content([prompt, *images])

  return response



if __name__=="__main__":

    # Load the dataset
    args = {
        'split': "val",
        'mode': "val",
        'num_data_points': 3000,
    }

    dataset = AllMLMBench(args, tokenizer=None, image_processor=None)

    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    all_responses = []
    with open("GPT4_responses_mlmBench.json", "r") as f:
      all_responses = json.load(f)

    count = 0
    for entry in tqdm.tqdm(dataset):
        count += 1
        if count < len(all_responses):
          continue
        images, prompt, text_label, correct_answer, datatype = entry

        prompt = prompt.split("Human: Answer in natural language.")[-1].split("###Assistant")[0]
        prompt += " Please answer just one of the options and no other text."
        # pdb.set_trace()
        try:
          response = get_caption(images, prompt, model)
          response_text = response.text
        except:
          print("skipping")
          response_text = "n/a"
        
        print("Prompt: ", prompt)
        print(response_text)

        # Save the response
        all_responses.append({
          "prompt": prompt,
          "text_label": text_label,
          "image_path": "",
          "response": response_text,
          "answer": correct_answer,
          "answer_choices": [correct_answer,],
          "dataset": datatype
        })
        # pdb.set_trace()

        # Save the responses
        with open("Gemini_responses_mlmBench.json", "w") as f:
            json.dump(all_responses, f)