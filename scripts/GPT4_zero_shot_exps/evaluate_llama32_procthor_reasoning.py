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
from llamaapi import LlamaAPI

# API Key


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_caption(image_path, prompt, api_key):
    # Getting the base64 string
    api_request_json = {
        "model": "llama3.2-11b-vision",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        
        "stream": False,
        "function_call": "get_current_weather",
    }

    # Execute the Request
    response = llama.run(api_request_json)

  return response


if __name__=="__main__":

    # Load the dataset
    args = {
        'split': "val",
        'mode': "val",
        'prompt_mode': 'text_choice',
        'complex_only': True,
        'add_complex': True
    }

    dataset = ProcTHOR_reasoning(args, tokenizer=None, image_processor=None)

    all_responses = []
    for ind, entry in enumerate(tqdm.tqdm(dataset)):
        image_paths, images, prompt, text_label, correct_answer, answer_choices, datatype = entry
        
        if len(image_paths)==1:
            image_path =  image_paths[0]
        else:
          # join the images into one
          image1 = cv2.imread(image_paths[0])
          image2 = cv2.imread(image_paths[1])
          image = cv2.hconcat([image1, image2])
          cv2.imwrite("temp.jpg", image)
          image_path = "temp.jpg"
          

        prompt = prompt.split("Human: Answer in natural language.")[-1].split("###Assistant")[0]

        # pdb.set_trace()

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
        with open("GPT4_complexreasoning_response.json", "w") as f:
            json.dump(all_responses, f)

        if ind>=4000:
            break