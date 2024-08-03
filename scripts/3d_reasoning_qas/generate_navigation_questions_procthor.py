from collections import defaultdict
import json
import prior
from ai2thor.controller import Controller
from PIL import Image
import random
from pprint import pprint
import pdb
import math
from PIL import ImageDraw, ImageFont

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import ast
import tqdm
import copy
import os
import numpy as np
import yaml
import sys
import shutil
sys.path.append("../../")
from utils.ai2thor_utils import generate_program_from_roomjson, generate_room_programs_from_house_json, make_house_from_cfg

from ai2thor.util.metrics import (
    get_shortest_path_to_object_type
)


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


def get_current_state(controller):
    nav_visible_objects = controller.step("GetVisibleObjects", maxDistance=6).metadata["actionReturn"]
    nav_visible_obj_assets = [objid2assetid[obj] for obj in nav_visible_objects] # these are the visible object asset ids in the scene
    nav_visible_obj_assets = [asset for asset in nav_visible_obj_assets if asset!=""]

    assetid2info = {}
    objdesc2cnt = defaultdict(int)
    for obj_entry in controller.last_event.metadata['objects']:
        obj_name = obj_entry['name']
        obj_type = obj_entry['objectType']
        asset_id = obj_entry['assetId']
        
        distance = obj_entry['distance']
        pos = np.array([obj_entry['position']['x'], obj_entry['position']['y'], obj_entry['position']['z']])
        rotation = obj_entry['rotation']
        desc = assetid2desc.get(asset_id, obj_type)
        moveable = obj_entry['moveable'] or obj_entry['pickupable']
        asset_size = obj_entry['axisAlignedBoundingBox']['size']['x']*obj_entry['axisAlignedBoundingBox']['size']['y']*obj_entry['axisAlignedBoundingBox']['size']['z']

        parent = obj_entry.get('parentReceptacles')
        if parent is not None:
            if len(parent) > 0:
                parent = parent[-1]
                if parent== "Floor":
                    parent = "Floor"
                else:
                    parent = objid2assetid[parent]
        
        is_receptacle = obj_entry['receptacle']
        assetid2info[asset_id] = (obj_name, obj_type, distance, pos, rotation, desc, moveable, parent, asset_size, is_receptacle)
        objdesc2cnt[desc] += 1

    moveable_visible_objs = []
    for objid in nav_visible_objects:
        assetid = objid2assetid[objid]
        if assetid in assetid2info:
            if assetid2info[assetid][6] and assetid2info[assetid][8]>0.003:
                moveable_visible_objs.append(objid)

    return nav_visible_obj_assets, assetid2info, objdesc2cnt, moveable_visible_objs


if __name__ == "__main__":
    
    asset_id_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json", "r"))
    qa_im_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/multi_qa_images/navigation/'
    qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas.json'

    if not os.path.exists(qa_im_path):
        os.makedirs(qa_im_path)

    assetid2desc = {}
    for asset in asset_id_desc:
        entries = asset_id_desc[asset]
        captions = []
        for im, obj, desc in entries:
            captions.append(desc)
        assetid2desc[asset] = random.choice(captions)

 

    dataset = prior.load_dataset("procthor-10k")

    
    all_im_qas = []
    for house_ind, house in enumerate(dataset["train"]):
        house_json = house
        try:
            controller = Controller(scene=house, width=300, height=300, quality="Low") # quality="Ultra", renderInstanceSegmentation=True, visibilityDistance=30)
        except:
            print("Cannot render environment, continuing")
            # pdb.set_trace()
            continue
        
        # get camera position with max objects visible

        try:
            event = controller.step(action="GetReachablePositions")
        except:
            print("Cannot get reachable positions, continuing")
            controller.stop()
            continue
        reachable_positions = event.metadata["actionReturn"]
        
        random_positions = []
        for cam_pos in random.sample(reachable_positions, 10):
            
            cam_rot = random.choice(range(360))
            try:
                controller.step(action="Teleport", position=cam_pos, rotation=cam_rot)
            except:
                print("Cannot teleport, continuing")
                continue

            nav_visible_objects = controller.step(
                "GetVisibleObjects",
                maxDistance=6,
            ).metadata["actionReturn"]

            random_positions.append((cam_pos, cam_rot, len(nav_visible_objects)))
        
        if len(random_positions)==0:
            print("No objects visible, continuing")
            controller.stop()
            continue

        random_positions = sorted(random_positions, key=lambda x: x[2], reverse=True)
        
        controller.stop()

        if not os.path.exists(qa_im_path + f"{house_ind}"):
            os.makedirs(qa_im_path + f"{house_ind}")

        sample_count = 0
        for cam_pos, cam_rot, _ in random_positions[:1]:
            qa_pair_choices = []

            controller = Controller(scene=house_json, width=800, height=800, quality="Ultra")
            try:
                controller.step(action="Teleport", position=cam_pos, rotation=cam_rot)
            except:
                print("Cannot teleport, continuing")
                controller.stop()
                continue

            objid2assetid = {}
            for obj in controller.last_event.metadata['objects']:
                objid2assetid[obj['objectId']] = obj['assetId']

            # get visible objects
            nav_visible_obj_assets, assetid2info, objdesc2cnt, moveable_visible_objs = get_current_state(controller)

            img_view = Image.fromarray(controller.last_event.frame)

            new_path_init = qa_im_path + f"{house_ind}/{sample_count}_0.jpg"
            img_view.save(new_path_init)

            choices = ["nav_only",]
            qa_choice = random.choice(choices)
            if qa_choice == "nav_only":
                actions = []
                wrong_action_seq = []
                image_seq = [new_path_init,]
                
                step_i = 1
                while step_i < random.choice(range(3, 10)):
                    if random.random() < 0.66:
                        # choose a random angle and direction to move.
                        angle = random.choice(range(20, 60, 10))
                        direction = random.choice(["RotateLeft", "RotateRight"])
                        text_direction = "left" if direction=="RotateLeft" else "right"
                        wrong_text_direction = "right" if direction=="RotateLeft" else "left"
                        # pdb.set_trace()
                        try:
                            state = controller.step(action=direction, degrees=angle)
                        except:
                            step_i += 1
                            continue
                        if state.metadata["lastActionSuccess"]:
                            actions.append(f"rotated {text_direction} by {int(angle)} degrees")
                            wrong_action_seq.append(
                                random.choice([f"rotated {wrong_text_direction} by {int(angle)} degrees", "moved forward by 0.25 meters", "moved backward by 0.25 meters"])
                            )
                            img_view = Image.fromarray(controller.last_event.frame)
                            step_img_path = qa_im_path + f"{house_ind}/{sample_count}_{step_i}.jpg"
                            img_view.save(step_img_path)
                            image_seq.append(step_img_path)

                        else:
                            question = f"Can I rotate {text_direction} by {int(angle)} degrees at this position?"
                            explantion = state.metadata["errorMessage"]
                            blocking_obj_name = state.metadata["errorMessage"].split(" is blocking")[0].strip()
                            blocking_obj_name = assetid2info[objid2assetid[blocking_obj_name]][5]
                            explantion = f"{blocking_obj_name} is blocking the rotation."
                            answer_choices = ["No because: " + explantion, "yes"]
                            qa_pair_choices.append((question, image_seq, answer_choices))
                    else:
                        try:
                            state = controller.step(action="MoveAhead", moveMagnitude=0.25)
                        except:
                            step_i += 1
                            continue
                        if state.metadata["lastActionSuccess"]:
                            actions.append("moved forward by 0.25 meters")
                            wrong_action_seq.append(random.choice(["moved backward by 0.25 meters", f"rotated left by {random.choice(range(50, 160, 10))} degrees", f"rotated right by {random.choice(range(50, 160, 10))} degrees"]))
                            img_view = Image.fromarray(controller.last_event.frame)
                            step_img_path = qa_im_path + f"{house_ind}/{sample_count}_{step_i}.jpg"
                            img_view.save(step_img_path)
                            image_seq.append(step_img_path)
                        else:
                            question = f"Can I move forward by 0.25 meters at this position?"
                            explantion = state.metadata["errorMessage"]
                            blocking_obj_name = state.metadata["errorMessage"].split(" is blocking")[0].strip()
                            blocking_obj_name = assetid2info[objid2assetid[blocking_obj_name]][5]
                            explantion = f"{blocking_obj_name} is blocking the movement."
                            answer_choices = ["No because: " + explantion, "yes"]
                            qa_pair_choices.append((question, image_seq, answer_choices))
                    
                    step_i += 1
                

                action = " and then ".join(actions)
                wrong_action = " and then ".join(wrong_action_seq)
                
                # questions about opbject movement
                question = "Did any of the objects in the initial frame that you can still see in the subsequent frames move from their original positions?"
                answer_choices = ["No", "Yes"] # first choice is always correct, randomize later in dataloader
                qa_pair_choices.append((question, image_seq, answer_choices))
                
                # question about the actions taken in the sequence of images
                question = "How did I likely move for the given image frames in order?"
                answer_choices = [action, wrong_action]
                # pdb.set_trace()
                qa_pair_choices.append((question, image_seq, answer_choices))

                # check objects we moved closer to. 
                obj_distance_changes = []
                move_closer_assets = []
                move_farther_assets = []
                for obj_entry in controller.last_event.metadata['objects']:
                    asset_id = obj_entry['assetId']
                    asset_size = obj_entry['axisAlignedBoundingBox']['size']['x']*obj_entry['axisAlignedBoundingBox']['size']['y']*obj_entry['axisAlignedBoundingBox']['size']['z']

                    if asset_id in nav_visible_obj_assets and asset_size>0.006 and objdesc2cnt[assetid2info[asset_id][5]]==1:
                        distance = obj_entry['distance']
                        og_distance = assetid2info[asset_id][2]  
                        obj_distance_changes.append((asset_id, distance, og_distance))
                        if distance < og_distance:
                            move_closer_assets.append(asset_id)
                        else:
                            move_farther_assets.append(asset_id)
                
                for asset_id, distance, og_distance in obj_distance_changes:
                    if distance < og_distance:
                        if random.random() < 0.5:
                            question = f"If I {action}, would {assetid2info[asset_id][5]} move closer to the camera?"
                            answer_choices = ["yes", "no"]
                            image_order = image_seq[:1]
                            qa_pair_choices.append((question, image_order, answer_choices))
                        else:
                            question = f"If I {action}, would {assetid2info[asset_id][5]} move further from the camera?"    
                            answer_choices = ["no", "yes"]
                            image_order = image_seq[:1]
                            qa_pair_choices.append((question, image_order, answer_choices)) 

                        if len(move_farther_assets) > 0:
                            question = f"Which of the following objects will move closer to the camera if I {action}?"
                            answer_choices = [assetid2info[asset_id][5]] + [assetid2info[asset_id][5] for asset_id in move_farther_assets]
                            image_order = image_seq[:1]
                            qa_pair_choices.append((question, image_order, answer_choices))
                    elif distance > og_distance:
                        if random.random() < 0.5:
                            question = f"if I {action}, would {assetid2info[asset_id][5]} move closer to the camera?"
                            answer_choices = ["no", "yes"]
                            image_order = image_seq[:1]
                            qa_pair_choices.append((question, image_order, answer_choices))
                        else:
                            question = f"If I {action}, would {assetid2info[asset_id][5]} move further from the camera?"
                            answer_choices = ["yes", "no"]
                            image_order = image_seq[:1]
                            qa_pair_choices.append((question, image_order, answer_choices))

                        if len(move_closer_assets) > 0:
                            question = f"Which of the following objects will move further from the camera if I {action}?"
                            answer_choices = [assetid2info[asset_id][5]] + [assetid2info[asset_id][5] for asset_id in move_closer_assets]
                            image_order = image_seq[:1]
                            qa_pair_choices.append((question, image_order, answer_choices))
                    else:
                        question = f"If I {action}, would {assetid2info[asset_id][5]} move closer to the camera?"
                        answer_choices = ["no, it would stay the same", "yes"]
                        image_order = image_seq[:1]
                        qa_pair_choices.append((question, image_order, answer_choices))

                # pdb.set_trace()

            elif qa_choice == "obj_only":
                # choose a random object and move it to a random position on a receptacle
                obj_choices = random.sample(moveable_visible_objs, 2)
                    
                #receptacle_choice = random.choice(receptacle_objects)
                #spawn_coordinates = controller.step(action="GetSpawnCoordinatesAboveReceptacle", objectId=receptacle_choice, anywhere=True).metadata["actionReturn"]
                #pdb.set_trace()
                #spawn_choice = random.choice(spawn_coordinates)
                
                #move obj1 closer to obj2
                obj1 = obj_choices[0]
                obj2 = obj_choices[1]

                obj1_pos = assetid2info[objid2assetid[obj1]][3]
                obj2_pos = assetid2info[objid2assetid[obj2]][3]

                spawn_choice = obj1_pos + 0.5*(obj2_pos - obj1_pos)

                spawn_choice = {"x": spawn_choice[0], "y": spawn_choice[1], "z": spawn_choice[2]}
                
                event = controller.step(action="PlaceObjectAtPoint", objectId=obj1, position=spawn_choice)

                if event.metadata["lastActionSuccess"]:
                    img_view = Image.fromarray(controller.last_event.frame)
                    new_path_final = qa_im_path + f"{house_ind}/{sample_count}_final.jpg"
                    img_view.save(new_path_final)

                    question = "Did any of the objects in the initial frame that you can still see in the final frame move from their original positions?"
                    answer_choices = ["Yes", "No"] # first choice is always correct, randomize later in dataloader
                    image_order = (new_path_init, new_path_final)
                    qa_pair_choices.append((question, image_order, answer_choices))

                    question = f"Which object moved in the two frames?"
                    answer_choices = [
                        assetid2info[objid2assetid[obj1]][5],
                        assetid2info[objid2assetid[obj2]][5],
                        "None of the objects"
                    ]
                    image_order = (new_path_init, new_path_final)
                    qa_pair_choices.append((question, image_order, answer_choices))

                    if random.random() < 0.5:
                        question = f"Did {assetid2info[objid2assetid[obj1]][5]} move closer to {assetid2info[objid2assetid[obj2]][5]} from the first frame to the second frame?"
                        answer_choices = ["yes", "no"]
                        image_order = (new_path_init, new_path_final)
                        qa_pair_choices.append((question, image_order, answer_choices))
                    else:
                        question = f"Did {assetid2info[objid2assetid[obj1]][5]} move further from {assetid2info[objid2assetid[obj2]][5]} from the first frame to the second frame?"
                        answer_choices = ["yes", "no"]
                        image_order = (new_path_final, new_path_init)
                        qa_pair_choices.append((question, image_order, answer_choices))

                    spawn_pos = np.array([spawn_choice["x"], spawn_choice["y"], spawn_choice["z"]])
                    cam_pos_arr = np.array([cam_pos["x"], cam_pos["y"], cam_pos["z"]])
                    # pdb.set_trace()
                    distance_cam_updated = np.linalg.norm(spawn_pos - cam_pos_arr)
                    
                    distance_cam_initial = np.linalg.norm(np.array(obj1_pos) - np.array(cam_pos_arr))

                    question = f"If I move {assetid2info[objid2assetid[obj1]][5]} closer to {assetid2info[objid2assetid[obj2]][5]}, will it be closer to the camera than it was in the initial frame?"
                    if distance_cam_updated < distance_cam_initial:
                        answer_choices = ["yes", "no"]
                    else:
                        answer_choices = ["no", "yes"]
                    image_order = (new_path_init,)
                    qa_pair_choices.append((question, image_order, answer_choices))
            elif qa_choice == "obj_movement":
                
                pass        
                #else:
                #    question = f"Can we place {assetid2info[objid2assetid[obj1]][5]} midway to {assetid2info[objid2assetid[obj2]][5]}?"
                #    explantion = event.metadata["errorMessage"]
                #    answer_choices = ["yes", "No because: " + explantion]

                #    image_order = (new_path_init, "")
                #    qa_pair_choices.append((question, image_order, answer_choices))



            # pdb.set_trace()
            if len(qa_pair_choices) > 0:
                all_im_qas.append((house_ind, cam_pos, cam_rot, qa_pair_choices))
            sample_count += 1
            controller.stop()
        
        
        json.dump(all_im_qas, open(qa_json_path, "w"))

        if house_ind > 3:
            break

    
    # view in html
    html_str = f"<html><head></head><body>"
    public_im_folder = "/net/cs-nfs/home/grad2/array/public_html/research/r2d3/multi_qa_ims/navigation/"
    for house_ind, _, _, qa_pairs in all_im_qas:
        if not os.path.exists(public_im_folder + f"{house_ind}"):
            os.makedirs(public_im_folder + f"{house_ind}")
            
        for sample_count, qa_pair in enumerate(qa_pairs):
            question, image_order, answer_choices = qa_pair
            html_str += f"<p>{question}</p>"
            for im in image_order:
                if im == "":
                    continue
                public_path_im = public_im_folder + "/".join(im.split("/")[-2:])
                shutil.copyfile(im, public_path_im)
                html_im_url = "https://cs-people.bu.edu/array/"+public_path_im.split("/net/cs-nfs/home/grad2/array/public_html/")[-1]
                html_str += f"<img src='{html_im_url}' style='width: 300px; height: 300px;'>"
            html_str += "<p>"
            for ans in answer_choices:
                html_str += f"<p>{ans}</p>"
            html_str += "<p>"
            html_str += "<hr>"
    html_str += "</body></html>"
    with open("multi_im_qa_navigation.html", "w") as f:
        f.write(html_str)

        


        
         
       