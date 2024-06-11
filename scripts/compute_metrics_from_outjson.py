import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import models.eval_funcs as eval_funcs
import yaml
import wandb

import utils.ai2thor_utils as ai2thor_utils
import random
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
import torch
import base64
import requests
import pdb

api_key = "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"


def compute_metrics_from_json(json_file, logger):

    data = json.load(open(json_file))
    print("Number of samples: ", len(data))
    # initialize metrics
    args = {}
    args['logger'] = logger
    obj_accs = eval_funcs.HouseSemanticSimilarity(args)
    obj_distance_accs = eval_funcs.HouseObjectDistancesAccuracy(args)
    selected_obj_accs = eval_funcs.HouseSelectedObjAccuracy(args)
    selected_obj_dists = eval_funcs.HouseSelectedObjectDistancesAccuracy(args)
    obj_attr_accs = eval_funcs.AttributeObjectMetrics(args)

    for output, house_json, gt_house_dict, gt_text_labels in data:

        # pdb.set_trace()
        gt = {}
        gt['text_labels'] = [gt_text_labels,]

        if "camera_pos" in gt_house_dict:
            gt['camera_pos'] = gt_house_dict['camera_pos']
          
        if "polygon" in gt_house_dict:
            gt['polygon'] = gt_house_dict['polygon']

        if 'objs_present' in gt_house_dict:
            gt['objs_present'] = gt_house_dict['objs_present']

        # compute metrics
        #obj_accs.update(output, gt)
        #obj_distance_accs.update(output, gt)
        obj_attr_accs.update(output, gt)
        #if 'objs_present' in gt_house_dict:
        #  selected_obj_accs.update(output, gt)
        #  selected_obj_dists.update(output, gt)

    # Compute the metrics
    #obj_accs.compute()
    #obj_distance_accs.compute()
    obj_attr_accs.compute()
    #if 'objs_present' in gt_house_dict:
    #  selected_obj_accs.compute()
    #  selected_obj_dists.compute()


def get_json_from_text(output):
    # get pred house JSON and gt house JSON
    if "#room" in output:
        room_response = output.split(": #room \n")[-1]
    else:
        room_response = output.split(": \n")[-1]

    room_json_text = "\n".join(room_response.split("\n")[:-1])
    room_json_text = room_json_text.split("###")[0]
    room_json_text = room_json_text.replace("(", "[")
    room_json_text = room_json_text.replace(")", "]")
    
    return room_json_text

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


def compute_image_caption_metrics(json_file, logger, eval_caption_sim=False):
    import prior
    from ai2thor.controller import Controller    
    data = json.load(open(json_file))

    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    all_im_sims = []
    for output, house_json, gt_house_dict, gt_text_labels in data:
        pred_room_text = get_json_from_text(output)
        gt_room_text = get_json_from_text(gt_text_labels)

        house_json = ai2thor_utils.make_house_from_cfg(pred_room_text).house_json
        gt_house_json = ai2thor_utils.make_house_from_cfg(gt_room_text).house_json

        # render the image from pred and JSON at the same camera coordinates and top down for both
        controller = Controller(width=800, height=800, quality="High WebGL", scene="Procedural", gridSize=0.25,) # renderInstanceSegmentation=True)
        controller.step(action="CreateHouse", house=house_json)

        controller_gt = Controller(width=800, height=800, quality="High WebGL", scene="Procedural", gridSize=0.25,)
        controller_gt.step(action="CreateHouse", house=gt_house_json)

        try:
            event = controller.step(action="GetReachablePositions")
        except:
            print("Cannot get reachable positions, continuing")
            controller.stop()
            continue
        reachable_positions = event.metadata["actionReturn"]

        try:
            event_gt = controller_gt.step(action="GetReachablePositions")
        except:
            print("Cannot get reachable positions, continuing")
            controller.stop()
            continue
        reachable_positions_gt = event_gt.metadata["actionReturn"]

        # get reachable positions in pred that are also in gt
        reachable_positions = [(pos['x'], pos['z']) for pos in reachable_positions]
        reachable_positions_gt = [(pos['x'], pos['z']) for pos in reachable_positions_gt]
        # common_reachable_positions = list(set(reachable_positions) & set(reachable_positions_gt))

        if len(reachable_positions) == 0 or len(reachable_positions_gt) == 0:
            print("No reachable positions, continuing")
            controller.stop()
            continue

        # pdb.set_trace()
        # see if we have camera positions in gt_house_dict, use that, else use random
        if "camera_pos" in gt_house_dict:
          cam_position_entries = gt_house_dict["camera_pos"]
          cam_positions = [(entry[0]/100, entry[1]/100) for entry in cam_position_entries]
          camera_angles = [entry[2] for entry in cam_position_entries]
        else:
          cam_positions = random.sample(reachable_positions_gt, min(5, len(reachable_positions_gt)))
          camera_angles = [0, 90, 180, 270]

        image_similarities = []
        for pos in cam_positions:
            
          closest_pos = min(reachable_positions, key=lambda x: (x[0]-pos[0])**2 + (x[1]-pos[1])**2)
          closest_pos_gt = min(reachable_positions_gt, key=lambda x: (x[0]-pos[0])**2 + (x[1]-pos[1])**2)

          c_pos = {"x": round(closest_pos[0]*4)/4.0, "y": 0.9, "z": round(closest_pos[1]*4)/4.0}
          c_pos_gt = {"x": round(closest_pos_gt[0]*4)/4.0, "y": 0.9, "z": round(closest_pos_gt[1]*4)/4.0}

          print("Original pos: ", pos)
          print("Closest pos: ", c_pos)
          print("Closest pos GT: ", c_pos_gt)
          
          for angle in camera_angles:

            try:
              controller.step(action="Teleport", position=c_pos, rotation=angle)
              controller_gt.step(action="Teleport", position=c_pos_gt, rotation=angle)
            except:
              print("failed teleport")
              # pdb.set_trace()
              continue
                
            if controller.last_event.metadata["lastActionSuccess"] == False:
                print("failed teleport")
                # pdb.set_trace()
                continue
            if controller_gt.last_event.metadata["lastActionSuccess"] == False:
                print("failed teleport")
                # pdb.set_trace()
                continue

            # render the images
            image = controller.last_event.frame
            image_gt = controller_gt.last_event.frame

            # pdb.set_trace()
            # measure image similarity
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                pooled_output = outputs.pooler_output 

            inputs_gt = processor(images=image_gt, return_tensors="pt")
            with torch.no_grad():
                outputs_gt = model(**inputs_gt)
                pooled_output_gt = outputs_gt.pooler_output

            image_similarity = torch.nn.functional.cosine_similarity(pooled_output, pooled_output_gt, dim=-1)

            image_similarities.append(image_similarity)

            if eval_caption_sim:
              # generate captions for the images
              prompt = "Can you describe this image using the the relative locations of the objects to a few other objects in the room? Just output a very brief description without any other text."
              response = get_caption(image, prompt, api_key)
              caption = response.json()['choices'][0]['message']['content']

              response_gt = get_caption(image_gt, prompt, api_key)
              caption_gt = response_gt.json()['choices'][0]['message']['content']

        controller.stop()
        controller_gt.stop()
      
        im_sim = torch.mean(torch.stack(image_similarities))
        all_im_sims.append(im_sim.item())

    logger.log({"image_similarity": torch.mean(torch.tensor(all_im_sims)).item()})
    # print("Image similarity: ", torch.mean(torch.tensor(all_im_sims)).item())


if __name__=="__main__":
    
    json_file = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/checkpoints/llava_incomplete_oneim_campolygonangle_nomarksbaseline/output.json"

    exp_name = json_file.split('/')[-2]
    wandb.login()
    run = wandb.init(project=exp_name)
    logger = run
    
    compute_metrics_from_json(json_file, logger)

    # compute_image_caption_metrics(json_file, logger, eval_caption_sim=False)
    
    