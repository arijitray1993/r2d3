import base64
import requests
import json
import os
import pdb
from collections import defaultdict
import random
from PIL import Image
import cv2
import re
import tqdm

import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

import numpy as np


import sys


def get_current_state(controller):
    nav_visible_objects = controller.step("GetVisibleObjects", maxDistance=5).metadata["actionReturn"]
    nav_all_visible_objects = controller.step("GetVisibleObjects", maxDistance=20).metadata["actionReturn"]
    objid2assetid = {}
    for obj in controller.last_event.metadata['objects']:
        objid2assetid[obj['objectId']] = obj['assetId']
    nav_visible_objects = [obj for obj in nav_visible_objects if objid2assetid[obj]!=""] # these are the visible object asset ids in the scene
    
    bboxes = controller.last_event.instance_detections2D
    vis_obj_to_size = {}
    for obj_id in bboxes:
        vis_obj_to_size[obj_id] = (bboxes[obj_id][2] - bboxes[obj_id][0])*(bboxes[obj_id][3] - bboxes[obj_id][1])


    objid2info = {}
    objdesc2cnt = defaultdict(int)
    for obj_entry in controller.last_event.metadata['objects']:
        obj_name = obj_entry['name']
        obj_type = obj_entry['objectType']
        asset_id = obj_entry['assetId']
        obj_id = obj_entry['objectId']
        
        distance = obj_entry['distance']
        pos = np.array([obj_entry['position']['x'], obj_entry['position']['y'], obj_entry['position']['z']])
        rotation = obj_entry['rotation']
        desc = assetid2desc.get(asset_id, obj_type)
        moveable = obj_entry['moveable'] or obj_entry['pickupable']
        
        asset_size_xy = vis_obj_to_size.get(obj_entry['objectId'], 0)
        asset_pos_box = bboxes.get(obj_entry['objectId'], None)
        if asset_pos_box is not None:
            asset_pos_xy = [(asset_pos_box[0]+asset_pos_box[2])/2, (asset_pos_box[1]+asset_pos_box[3])/2]
        else:
            continue

        parent = obj_entry.get('parentReceptacles')
        if parent is not None:
            if len(parent) > 0:
                parent = parent[-1]
                
        
        is_receptacle = obj_entry['receptacle']
        objid2info[obj_id] = (obj_name, obj_type, distance, pos, rotation, desc, moveable, parent, asset_size_xy, is_receptacle, asset_pos_xy, asset_pos_box)
        if obj_id in nav_all_visible_objects:
            objdesc2cnt[obj_type] += 1

    
    visible_objs = []
    for objid in nav_all_visible_objects:
        if objid in objid2info:
            visible_objs.append(objid)

    # pdb.set_trace()
    return nav_visible_objects, objid2info, objdesc2cnt, visible_objs


if __name__=="__main__":

    # Load the dataset
    complex_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_val_v2.json'
    complex_data = json.load(open(complex_qa_json_path))

    perspective_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/perspective_qas_val.json'
    perspective_data = json.load(open(perspective_qa_json_path))

    data = []

    num_complex=0
    num_perspective=0
            
    for house_ind, cam_pos, cam_rot, qa_entries in complex_data[int(len(complex_data)*0.1):]:
        for question, im_order, answers in qa_entries:
            question = question.replace("turn look straight", "look straight")

            if answers[0] == "rotated left and rotated right" or answers[0] == "rotated right and rotated left": # bug fix
                new_answers = ["did not move", random.choice(["rotated left", "rotated right"])]
                answers = new_answers
            
            if "how did the camera likely move" in question.lower():
                question = question.replace("How did the camera likely move when shooting the video", "How did the camera rotate from the first image to the second?") # fix this bug in generation later.
            
            data.append((house_ind, cam_pos, cam_rot, question, im_order, answers))
            num_complex += 1

    
    for _,_,_, qa_entries in perspective_data[int(len(perspective_data)*0.1):]:
        for question, im_order, answers in qa_entries:
            question = question.replace("turned towards the", "facing 90 degrees to the")
            question = question.replace("turned right", "turned right by 90 degrees")
            question = question.replace("turned left", "turned left by 90 degrees")

            data.append((house_ind, cam_pos, cam_rot, question, im_order, answers))
            num_perspective += 1
            if num_perspective > 1000:
                break
        if num_perspective > 1000:
            break
    print(" Complex: ", num_complex, " Perspective: ", num_perspective)


    asset_id_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json", "r"))
    assetid2desc = {}
    for asset in asset_id_desc:
        entries = asset_id_desc[asset]
        captions = []
        for im, obj, desc in entries:
            desc = desc.strip().lower().replace(".", "")
            captions.append(desc)
        assetid2desc[asset] = random.choice(captions)

    qa_3dinfo_data = []

    qa_3dinfo_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/satdynamicqa_3dinfo.json"))

    prior_dataset = prior.load_dataset("procthor-10k")["val"]

    for house_ind, cam_pos, cam_rot, question, im_order, answers in tqdm.tqdm(data[-1000:]):
        
        house_json = prior_dataset[house_ind]
        try:
            controller = Controller(scene=house_json, width=100, height=100, quality="Low", renderInstanceSegmentation=True, platform=CloudRendering)
            controller.step(action="Teleport", position=cam_pos, rotation=cam_rot)
        except:
            print("cannot load house")
            continue

        cam_pos = (cam_pos['x'], cam_pos['y'], cam_pos['z'])

        nav_visible_obj_assets, objid2info, objdesc2cnt, visible_obj_assets = get_current_state(controller)

        all_obj_descs = []
        for obj_ind, obj in enumerate(visible_obj_assets):
            obj_desc = objid2info[obj][5]
            obj_pos = objid2info[obj][3]
            obj_count = objdesc2cnt[objid2info[obj][1]]

            if obj_desc in ["Floor", "Wall", "Ceiling"]:
                continue
            
            obj_pos_norm = (obj_pos[0] - cam_pos[0], obj_pos[1] - cam_pos[1], obj_pos[2] - cam_pos[2])
            obj_pos_norm_rot = (obj_pos_norm[0]*np.cos(np.deg2rad(cam_rot)) - obj_pos_norm[2]*np.sin(np.deg2rad(cam_rot)), obj_pos_norm[0]*np.sin(np.deg2rad(cam_rot)) + obj_pos_norm[2]*np.cos(np.deg2rad(cam_rot)))

            desc_text = f"{obj_desc} at {int(obj_pos_norm_rot[0]*100)}, {int(obj_pos_norm[1]*100)}, {int(obj_pos_norm_rot[1]*100)}"

            all_obj_descs.append(desc_text)
        
        qa_3dinfo_data.append((house_ind, cam_pos, cam_rot, question, im_order, answers, all_obj_descs))

        if len(qa_3dinfo_data) % 100 == 0:
            json.dump(qa_3dinfo_data, open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/satdynamicqa_3dinfo_v2.json", "w"))

        controller.stop()
    json.dump(qa_3dinfo_data, open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/satdynamicqa_3dinfo_v2.json", "w"))
