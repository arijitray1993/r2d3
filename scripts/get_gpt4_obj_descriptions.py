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

# OpenAI API Key
api_key = "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"

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
    "model": "gpt-4-vision-preview",
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
    "max_tokens": 200
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  return response


def get_object_class_from_asset(obj):
    pattern = r'[0-9]'
    obj_name = obj.replace("_", " ")
    obj_name = re.sub(pattern, '', obj_name)

    obj_name = obj_name.replace("RoboTHOR", "")

    obj_name = obj_name.replace("jokkmokk", " ")

    if "Countertop" in obj_name:
        shape = obj_name.split(" ")[-1]
        if shape =="I":
            obj_name = "I-shaped countertop"
        elif shape =="L":
            obj_name = "L-shaped countertop"
        elif shape =="C":
            obj_name = "C-shaped countertop"

    obj_name = obj_name.strip()

    return obj_name

if __name__=="__main__":

    im_folder_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/all_obj_vis"

    all_object_images = os.listdir(im_folder_path)

    done_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/asset_descriptions.json", "r"))

    imfile_to_caption = {}
    for d in done_data:
        imfile_to_caption[d[0]] = d[3]

    all_object_desc = []
    for obj_im in tqdm.tqdm(all_object_images):
        
        asset_name = obj_im.split(".")[0]

        object_class = get_object_class_from_asset(asset_name)

        image_file = os.path.join(im_folder_path, obj_im)

        if image_file in imfile_to_caption:
          if 'sorry' not in imfile_to_caption[image_file].lower():
            continue

        prompt = f"The image shows a {object_class}. Can you please output a very few word description of the object in the image. Just say a few attributes of the object like the color and/or material followed by the object name (like a white wooden door). Even if you cannot quite tell, please try it to the best of your ability using the fact that this is a {object_class}. Please output only the description and no additional text or context."

        description_caption = get_caption(image_file, prompt, api_key)

        # pdb.set_trace()

        try:
            caption = description_caption.json()['choices'][0]['message']['content']
        except:
            continue

        if 'sorry' in description_caption.json()['choices'][0]['message']['content'].lower():
          print(image_file, object_class, "couldn't be described.")
          caption  =  object_class
        
        # pdb.set_trace()

        all_object_desc.append((image_file, asset_name, object_class, caption))

        with open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/asset_descriptions_missingobjs.json", "w") as f:
            json.dump(all_object_desc, f)
        


