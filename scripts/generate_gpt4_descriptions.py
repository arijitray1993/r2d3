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
from stitching import Stitcher
import re

# OpenAI API Key
api_key = ""

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
    "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  return response


def create_panorama(images):
    stitcher = Stitcher()
    stitcher = Stitcher(detector="sift", confidence_threshold=0.2)

    panorama = stitcher.stitch(images)

    return panorama


if __name__=="__main__":

    json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_train_childrenadded_all_14k.json"))

    image_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/sparsepanorama/images"

    # have only the json data we have top down for
    image_program_data = []
    for room_data in json_data:
        all_imgs = room_data[4]
        if len(all_imgs) < 2:
          continue
        if not os.path.exists(all_imgs[0]):
          continue

        image_program_data.append(room_data)
      
    print("total number of rooms after filtering: ", len(image_program_data))

    
    all_house_caption_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GPT4V_room_descriptions.json"))

    done_apartments = []
    for caption_data in all_house_caption_data:
      apartment_ind = caption_data[0]
      done_apartments.append(apartment_ind)

    print("total number of apartments done: ", len(done_apartments))

    # pdb.set_trace()
    for ind, room_data in enumerate(tqdm.tqdm(image_program_data)):

      program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name = room_data

      apartment_ind = all_imgs[0].split("/")[-2]
      if apartment_ind in done_apartments:
        continue
      
      
      # the image with the most objects
      max_objects = 0
      max_ind = 0
      for c_ind, img in enumerate(all_imgs):
          if len(all_objs[c_ind]) > max_objects:
              max_objects = len(all_objs[c_ind])
              max_ind = c_ind
      
      image_file = all_imgs[max_ind]
      # pdb.set_trace()

      objs = []
      pattern = r'[0-9]'
      for obj in all_objs[max_ind]:
          obj_name = re.sub(pattern, '', obj)

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

          objs.append(obj_name)
      
      objs = list(set(objs))

      # print(objs)
      # pdb.set_trace()

      # prompt = "Can you describe the functionality and vibes of the room very briefly? To give you a hint in case the image is not clear, these are the objects present in the view: " + ", ".join(objs) + ". Please avoid using phrases with an absolute location like 'on the left side' or 'on the right side' since these can change with perspective. Instead try to use the relative locations to a few other objects in the room."

      prompt = "We need to make the AI generate a 3D model of the room as close to this image as possible. How would you instruct the AI agent to do this? These are some of the objects present in the view: " + ", ".join(objs) + ". Avoid using absolute locations like left-side or right-side since these can change depending on the perspective. Try to use the relative locations to a few other objects in the room to convey where things should roughly be placed. Just output a brief description without any other text."

      description_caption = get_caption(image_file, prompt, api_key)

      # pdb.set_trace()

      try:
        caption = description_caption.json()['choices'][0]['message']['content']
      except:
        continue
      # pdb.set_trace()

      all_house_caption_data.append((apartment_ind, image_file, caption))

      with open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GPT4V_room_descriptions.json", "w") as f:
        json.dump(all_house_caption_data, f)
      
      #description_caption = description_caption.json()['choices'][0]['message']['content']

      #pdb.set_trace()


