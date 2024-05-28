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
import yaml
import sys
sys.path.append("../")
from utils.ai2thor_utils import generate_program_from_roomjson
from PIL import Image, ImageDraw, ImageFont

# OpenAI API Key
api_key = "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"


def add_red_dot_with_text(image, position, text):
    # Load the image
    draw = ImageDraw.Draw(image)

    # Coordinates and radius of the dot
    x, y = position
    radius = 10  # You can adjust the size of the dot

    # Draw the red dot
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='red')

    # Load a font (optional, comment out if not needed)
    try:
        font = ImageFont.truetype("arial.ttf", 8)  # Adjust font and size as needed
    except IOError:
        font = ImageFont.load_default()

    # Calculate text width and height to center it
    text_width = draw.textlength(text, font=font)
    text_x = x - text_width / 2
    text_y = y

    # Draw the text
    draw.text((text_x, text_y), text, fill='white', font=font)

    return image



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


def get_obj_class_from_assetname(obj):
  
  pattern = r'[0-9]'
  
  obj_name = re.sub(pattern, '', obj)
  obj_name = obj_name.replace("_", " ")
  
  obj_name = obj_name.replace("RoboTHOR", "")

  obj_name = obj_name.replace("jokkmokk", " ")
  obj_name = obj_name.replace("ikea", " ")
  

  if "Countertop" in obj_name:
    if "I" in obj_name:
      obj_name = "I-shaped countertop"
    elif "L" in obj_name:
      obj_name = "L-shaped countertop"
    elif "C" in obj_name:
      obj_name = "C-shaped countertop"

  if 'bnyd salt' in obj_name:
     obj_name = "salt shaker"
    
  if "Wall Decor" in obj_name:
    obj_name = "wall decor photo"

  
  obj_name = obj_name.strip()

  return obj_name

def create_panorama(images):
    stitcher = Stitcher()
    stitcher = Stitcher(detector="sift", confidence_threshold=0.2)

    panorama = stitcher.stitch(images)

    return panorama


if __name__=="__main__":

    json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/final_data_neurips.json"))

    # have only the json data we have top down for
    image_program_data = []
    for room_data in json_data:
        all_imgs = room_data[4]
        if len(all_imgs) < 1:
          continue
        if not os.path.exists(all_imgs[0]):
          continue

        top_down_im = "/".join(all_imgs[0].split("/")[:-1]) + "/top_down.png"

        if not os.path.exists(top_down_im):
          continue

        image_program_data.append(room_data)
      
    print("total number of rooms after filtering: ", len(image_program_data))

    out_json = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GPT4V_room_descriptions_topdown.json"
    if os.path.exists(out_json):
      all_house_caption_data = json.load(open(out_json))
    else:
      all_house_caption_data = []

    done_apartments = []
    for caption_data in all_house_caption_data:
      apartment_ind = caption_data[0]
      done_apartments.append(apartment_ind)

    print("total number of apartments done: ", len(done_apartments))

    pdb.set_trace()
    for ind, room_data in enumerate(tqdm.tqdm(image_program_data)):

      program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name = room_data

      program_text = generate_program_from_roomjson(house_json, include_children=True)

      apartment_ind = all_imgs[0].split("/")[-2]

      if apartment_ind in done_apartments:
        continue

      # pdb.set_trace()

      room = house_json['rooms'][0]
      polygon = room['floorPolygon']
      polygon_coords = [(round(point['x'], 2)*100, round(point['z'], 2)*100) for point in polygon]
      
      min_x = min([point[0] for point in polygon_coords])
      max_x = max([point[0] for point in polygon_coords])
      min_z = min([point[1] for point in polygon_coords])
      max_z = max([point[1] for point in polygon_coords])

      normalized_polygon_coordinates = [(point[0] - min_x, point[1] - min_z) for point in polygon_coords]



      apartment_ind = all_imgs[0].split("/")[-2]
      #if apartment_ind in done_apartments:
      #  continue

      # get obj x,z coordinates from room json. 
      # get obj0, obj1, ... until we run out of objects in dict
      cfg_dict = yaml.load(program_text, Loader=yaml.FullLoader)
      i = 0
      objects = {}
      while(True):
          if f'obj_{i}' in cfg_dict:
              obj = cfg_dict[f'obj_{i}']
              # print(obj[1])
              
              obj_asset = obj[0]
              obj_pos = obj[1]
              obj_rotation = obj[2]
              obj_id = f'obj_{i}'
              
              objects[obj_id] = (obj_asset, obj_pos, obj_rotation, [])
              i += 1
          else:
              break
      # print(f"made {i} objects"

      
      # get child0, child1, ... until we run out of children in dict
      i = 0
      children = []
      while(True):
          if f'child_{i}' in cfg_dict:
              child = cfg_dict[f'child_{i}']
              
              asset = child[0]
              position = list(child[1])
              rotation = list(child[2])
              id = f'child_{i}'
              parent_id = child[3]
              
              children.append((asset, position, rotation, parent_id))
              i += 1
          else:
              break
      
      par_id_to_child = defaultdict(list)
      for child in children:
          asset, position, rotation, parent_id = child
          par_id_to_child[parent_id].append((asset, position, rotation, parent_id))

      for parent_id, children in par_id_to_child.items():
          if parent_id in objects:
              obj_asset, obj_pos, obj_rotation, empty = objects[parent_id]
              objects[parent_id] = (obj_asset, obj_pos, obj_rotation, children)
          else:
              print(f"parent {parent_id} not in objects")
      
      # pdb.set_trace()
      obj_prompt = " As a hint, here are the object classes and their positions on the floor mentioned as (x,y). x is the horizonal direction increasing left to right and y is vertical increasing bottom to top. "
      obj_prompt += "This means that lower values of x means the object is towards the left and higher values of x means the object is towards the right. Similarly, lower values of y means the object is towards the bottom and higher values of y means the object is towards the top."

      image_file = "/".join(all_imgs[0].split("/")[:-1]) + "/top_down.png"
      # image = Image.open(image_file).convert("RGB")

      count = 0
      for obj_id, obj in objects.items():
          
        obj_asset, obj_position, obj_rotation, children = obj

        obj_children = []
        for child in children:
            asset, _, _, _ = child
            child_name = get_obj_class_from_assetname(asset)
            obj_children.append(child_name)
        
        
        obj_class = get_obj_class_from_assetname(obj_asset)

        position = (int(obj_position[0] - min_x), int(obj_position[2] - min_z))

        
        if len(obj_children) > 0:
          obj_entry_prompt = f"\n{obj_class} at {position} position with children: {', '.join(obj_children)}"
        else:
          obj_entry_prompt = f"\n{obj_class} at {position} position"
        
        obj_prompt += obj_entry_prompt
        count += 1

      obj_prompt += "\n "
      # print(objs)
      # pdb.set_trace()

      # prompt = "Can you describe the functionality and vibes of the room very briefly? To give you a hint in case the image is not clear, these are the objects present in the view: " + ", ".join(objs) + ". Please avoid using phrases with an absolute location like 'on the left side' or 'on the right side' since these can change with perspective. Instead try to use the relative locations to a few other objects in the room."

      prompt = "We have a top down image of a room and we need to make the AI generate a 3D model of the room using just text. How would you instruct the AI agent to do this?"
      prompt += obj_prompt
      prompt += "Briefly describe the locations of the objects in the top down view. Please try to use the generic object class name instead of the exact asset name and describe the attributes of the object in natural language. No need to exhaustively list all minor objects and just use a very brief description. You are not allowed to use any numeric locations of any objects, just use the relative positions."
      prompt += "Also describe the functionality and vibes of the room very briefly. Just output the brief instructions and do not include any other text."
      

      #image_file_marked = f"/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/images/train/{apartment_ind}/top_down_obj_marked.png"
      #image.save(image_file_marked)

      # pdb.set_trace()

      description_caption = get_caption(image_file, prompt, api_key)

      # pdb.set_trace()

      try:
        caption = description_caption.json()['choices'][0]['message']['content']
      except:
        print("failed caption")
        continue
      # pdb.set_trace()

      all_house_caption_data.append((apartment_ind, image_file, caption))

      with open(out_json, "w") as f:
        json.dump(all_house_caption_data, f)
      
      #description_caption = description_caption.json()['choices'][0]['message']['content']

      #pdb.set_trace()


