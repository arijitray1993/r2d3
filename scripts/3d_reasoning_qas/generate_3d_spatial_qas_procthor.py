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
from ai2thor.platform import CloudRendering
import shutil
import sys
sys.path.append("../../")
from utils.ai2thor_utils import generate_program_from_roomjson, generate_room_programs_from_house_json, make_house_from_cfg

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
    nav_visible_objects = controller.step("GetVisibleObjects", maxDistance=2).metadata["actionReturn"]
    nav_visible_obj_assets = [objid2assetid[obj] for obj in nav_visible_objects] # these are the visible object asset ids in the scene
    nav_visible_obj_assets = [asset for asset in nav_visible_obj_assets if asset!=""]
    
    bboxes = controller.last_event.instance_detections2D
    vis_obj_to_size = {}
    for obj_id in bboxes:
        vis_obj_to_size[obj_id] = (bboxes[obj_id][2] - bboxes[obj_id][0])*(bboxes[obj_id][3] - bboxes[obj_id][1])


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
        
        asset_size_xy = vis_obj_to_size.get(obj_entry['objectId'], 0)
        asset_pos_xy = bboxes.get(obj_entry['objectId'], None)

        parent = obj_entry.get('parentReceptacles')
        if parent is not None:
            if len(parent) > 0:
                parent = parent[-1]
                if parent== "Floor":
                    parent = "Floor"
                else:
                    parent = objid2assetid[parent]
        
        is_receptacle = obj_entry['receptacle']
        assetid2info[asset_id] = (obj_name, obj_type, distance, pos, rotation, desc, moveable, parent, asset_size_xy, is_receptacle, asset_pos_xy)
        objdesc2cnt[obj_type] += 1

    moveable_visible_objs = []
    for objid in nav_visible_objects:
        assetid = objid2assetid[objid]
        if assetid in assetid2info:
            if assetid2info[assetid][6] and assetid2info[assetid][8]>400:
                moveable_visible_objs.append(assetid)

    return nav_visible_obj_assets, assetid2info, objdesc2cnt, moveable_visible_objs


if __name__ == "__main__":
    
    qa_im_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/multi_qa_images/spatial/'
    qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_spatial_qas.json'
    html_vis_file = '3d_spatial_qas.html'

    asset_id_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json", "r"))

    if not os.path.exists(qa_im_path):
        os.makedirs(qa_im_path)
    #assetid2desc = {}
    #for image_file, asset_name, object_class, caption in asset_id_desc:
    #    assetid2desc[asset_name] = caption

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
                maxDistance=2,
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

            controller = Controller(scene=house_json, width=800, height=800, quality="Ultra", renderInstanceSegmentation=True,)
            try:
                controller.step(action="Teleport", position=cam_pos, rotation=cam_rot)
            except:
                print("Cannot teleport, continuing")
                controller.stop()
                continue

            objid2assetid = {}
            for obj in controller.last_event.metadata['objects']:
                objid2assetid[obj['objectId']] = obj['assetId']

            img_view = Image.fromarray(controller.last_event.frame)

            new_path_init = qa_im_path + f"{house_ind}/{sample_count}_0.jpg"
            img_view.save(new_path_init)

            # get visible objects
            nav_visible_obj_assets, assetid2info, objdesc2cnt, moveable_visible_obj_assets = get_current_state(controller)
            
            # choose random object
            if len(moveable_visible_obj_assets) < 2:
                print("No moveable objects, continuing")
                controller.stop()
                continue
            obj1 = random.choice(moveable_visible_obj_assets) # these are asset_ids that are visible and moveable

            # pdb.set_trace()
            obj1_cnt = objdesc2cnt[assetid2info[obj1][1]]

            # Now let's ask questions about the object
            obj_name = assetid2info[obj1][0]

            obj_desc = assetid2info[obj1][5]
            obj1_type = assetid2info[obj1][1]

            new_pos = cam_pos
            new_rot = cam_rot
            
            xz_rotation_matrix = np.array([[np.cos(math.radians(new_rot)), -np.sin(math.radians(new_rot))], [np.sin(math.radians(new_rot)), np.cos(math.radians(new_rot))]])

            # if count >1 then we need to make the exact object clear
            if obj1_cnt > 1:
                # get the distances to camera
                obj1_distances = []
                for asset in assetid2info:
                    if assetid2info[asset][1].lower().strip() == assetid2info[obj1][1].lower().strip():
                        obj1_distances.append(assetid2info[asset][2])
                obj1_distance = assetid2info[obj1][2]
                obj1_distances.sort()
                obj1_distance_rank = obj1_distances.index(obj1_distance) + 1
                if obj1_distance_rank == 1:
                    obj_desc = f"{obj_desc} (closest {obj1_type} to camera)"
                elif obj1_distance_rank == 2:
                    obj_desc = f"{obj_desc} (second closest {obj1_type} to camera)"
                elif obj1_distance_rank == 3:
                    obj_desc = f"{obj_desc} (third closest {obj1_type} to camera)"
                else:
                    obj_desc = f"{obj_desc} ({obj1_distance_rank}th closest {obj1_type} to camera)"
                
            for another_obj in random.sample(nav_visible_obj_assets, min(3, len(nav_visible_obj_assets))):
                if another_obj == obj1:
                    continue

                # if parent is obj1 continue
                if assetid2info[another_obj][7] == obj1:
                    continue

                # if another obj is parent continue
                if assetid2info[obj1][7] == another_obj:
                    continue
                
                another_obj_desc = assetid2info[another_obj][5]

                another_obj_cnt = objdesc2cnt[assetid2info[another_obj][1]]
                if another_obj_cnt > 1:
                    # get the distances to camera
                    another_obj_distances = []
                    for asset in assetid2info:
                        if assetid2info[asset][1].lower().strip() == assetid2info[another_obj][1].lower().strip():
                            another_obj_distances.append(assetid2info[asset][2])
                    another_obj_distance = assetid2info[another_obj][2]
                    another_obj_distances.sort()
                    another_obj_distance_rank = another_obj_distances.index(another_obj_distance) + 1
                    if another_obj_distance_rank == 1:
                        another_obj_desc = f"{another_obj_desc} (closest {assetid2info[another_obj][1]} to camera)"
                    elif another_obj_distance_rank == 2:
                        another_obj_desc = f"{another_obj_desc} (second closest {assetid2info[another_obj][1]} to camera)"
                    elif another_obj_distance_rank == 3:
                        another_obj_desc = f"{another_obj_desc} (third closest {assetid2info[another_obj][1]} to camera)"
                    else:
                        another_obj_desc = f"{another_obj_desc} ({another_obj_distance_rank}th closest {assetid2info[another_obj][1]} to camera)"


                another_obj_pos = assetid2info[another_obj][3]

                ##### simple which object is behind the other object questions#####

                # obj 1  and another obj distance
                obj1_distance = assetid2info[obj1][2]
                another_obj_distance = assetid2info[another_obj][2]

                if obj1_distance > another_obj_distance:
                    question = f"Which object is closer to the camera, {obj_desc} or {another_obj_desc}?"
                    answer = f"{another_obj_desc}"
                    incorrect_answer = f"{obj_desc}"
                    answer_choices = (answer, incorrect_answer)
                    image_order = (new_path_init,)
                    qa_pair_choices.append((question, image_order, answer_choices))


                #######  which angle to push obj1 to move closer to other object #############
                # compute angle between camera, obj1 and another obj
                cam_to_obj1_vector = np.array([assetid2info[obj1][3][0] - cam_pos['x'], assetid2info[obj1][3][2] - cam_pos['z']])
                obj1_to_anotherobj_vec = np.array([another_obj_pos[0] - assetid2info[obj1][3][0], another_obj_pos[2] - assetid2info[obj1][3][2]])

                angle = np.arccos(np.dot(cam_to_obj1_vector, obj1_to_anotherobj_vec)/(np.linalg.norm(cam_to_obj1_vector)*np.linalg.norm(obj1_to_anotherobj_vec)))
                angle = int(math.degrees(angle))
                # angle = 180 - angle

                # rotate using rotation matrix to figure out whether another obj is left to right to camera
                obj1_rotated_pos = np.dot(xz_rotation_matrix, np.array([assetid2info[obj1][3][0], assetid2info[obj1][3][2]]))
                another_obj_rotated_pos = np.dot(xz_rotation_matrix, np.array([another_obj_pos[0], another_obj_pos[2]]))

                if another_obj_rotated_pos[0] > obj1_rotated_pos[0]:
                    direction = "right"
                    wrong_direction = "left"
                else:
                    direction = "left"
                    wrong_direction = "right"

                if angle > 90:
                    push_dir = "pull"
                    wrong_push_dir = "push"
                else:
                    push_dir = "push"
                    wrong_push_dir = "pull"

                incorrect_angle_delta = random.choice([90, 120, 150, 100, 85, 75])
                
                question = f"Should we push or pull {obj_desc} to the left or right to move it closer to {another_obj_desc}?", 
                answer_choices = (
                        f"{push_dir} towards the {direction}",
                        f"{wrong_push_dir} towards the {direction}",
                        f"{push_dir} towards the {wrong_direction}",
                    ) 
                image_order = (new_path_init,)
                qa_pair_choices.append((question, image_order, answer_choices))

                controller.stop()
            if len(qa_pair_choices) > 0:
                all_im_qas.append((house_ind, cam_pos, cam_rot, qa_pair_choices))
            json.dump(all_im_qas, open(qa_json_path, "w"))
        
        if house_ind > 10:
            break

    
    # view in html
    if html_vis_file is not None:
        html_str = f"<html><head></head><body>"
        public_im_folder = "/net/cs-nfs/home/grad2/array/public_html/research/r2d3/multi_qa_ims/spatial/"
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
                    html_str += f"<img src='{html_im_url}' style='width: 500px; height: 500px;'>"
                html_str += "<p>"
                for ans in answer_choices:
                    html_str += f"<p>{ans}</p>"
                html_str += "<p>"
                html_str += "<hr>"
        html_str += "</body></html>"
        with open(html_vis_file, "w") as f:
            f.write(html_str)

        


        