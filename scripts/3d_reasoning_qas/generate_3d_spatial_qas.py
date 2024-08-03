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


if __name__ == "__main__":
    
    image_program_json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/final_data_neurips.json", "r"))

    image_save_folder = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/images/train"

    asset_id_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json", "r"))

    #assetid2desc = {}
    #for image_file, asset_name, object_class, caption in asset_id_desc:
    #    assetid2desc[asset_name] = caption

    assetid2desc = {}
    for asset in asset_id_desc:
        entries = asset_id_desc[asset]
        captions = []
        for im, obj, desc in entries:
            captions.append(desc)
        assetid2desc[asset] = captions


    gen_blink_style_depth = False
    gen_force_question = True
    qa_im_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/qa_images/'
    if not os.path.exists(qa_im_path):
        os.makedirs(qa_im_path)
    qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/physics_qas.json'

    all_im_qas = []
    maximum_distance = 30
    for ind, (program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name) in enumerate(tqdm.tqdm(image_program_json_data)):
        
        #if ind<98:
        #    continue

        if len(all_imgs) == 0:
            continue

        for light in house_json['proceduralParameters']['lights']:
            light['intensity'] = 1.4
            light['type'] = 'point'
            light['range'] = 20
            light['color'] = {"r": 1, "g": 1, "b": 1}

        apartment_id = all_imgs[0].split("/")[-2]
        if not os.path.exists(qa_im_path + apartment_id):
            os.makedirs(qa_im_path + apartment_id)

        try:
            controller = Controller(scene=house_json, width=300, height=300, quality="Low", platform=CloudRendering) # quality="Ultra", renderInstanceSegmentation=True, visibilityDistance=30)
        except:
            print("Cannot render environment, continuing")
            # pdb.set_trace()
            continue
        
        # get camera position with max objects visible
        vis_obj_count = 0
        max_vis_cam_pos_ind = None
        for cam_pos_ind in cam_ind_to_position:
            cam_pos, cam_rot = cam_ind_to_position[cam_pos_ind]
            try:
                controller.step(action="Teleport", position=cam_pos, rotation=cam_rot)
            except:
                print("Cannot teleport, continuing")
                continue

            nav_visible_objects = controller.step(
                "GetVisibleObjects",
                maxDistance=maximum_distance,
            ).metadata["actionReturn"]

            if len(nav_visible_objects) > vis_obj_count:
                vis_obj_count = len(nav_visible_objects)
                max_vis_cam_pos_ind = cam_pos_ind
        
        if max_vis_cam_pos_ind is None:
            print("No objects visible, continuing")
            controller.stop()
            continue

        cam_pos, cam_rot = cam_ind_to_position[max_vis_cam_pos_ind]
        
        controller.stop()

    
        for sample_i in range(2):
            controller = Controller(scene=house_json, width=800, height=800, quality="Ultra", platform=CloudRendering)
            try:
                controller.step(action="Teleport", position=cam_pos, rotation=cam_rot)
            except:
                print("Cannot teleport, continuing")
                controller.stop()
                continue

            try:
                event = controller.step(action="GetReachablePositions")
            except:
                print("Cannot get reachable positions, continuing")
                controller.stop()
                continue
            reachable_positions = event.metadata["actionReturn"]
            
            cam_pos_ind = int(max_vis_cam_pos_ind)
            
            controller.step(
                action="RandomizeLighting",
                brightness=(1.4, 1.55),
                randomizeColor=False,
                synchronized=False
            )

            objid2assetid = {}
            for obj in controller.last_event.metadata['objects']:
                objid2assetid[obj['objectId']] = obj['assetId']

            img_view = Image.fromarray(controller.last_event.frame)
            new_path = qa_im_path + f"{apartment_id}/{sample_i}_physics.jpg"
            if not os.path.exists(new_path):
                img_view.save(new_path)

            # get visible objects
            nav_visible_objects = controller.step(
                "GetVisibleObjects",
                maxDistance=maximum_distance,
            ).metadata["actionReturn"]

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
                desc = random.choice(assetid2desc.get(asset_id, [obj_type,])).lower().strip()
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
                assetid2info[asset_id] = (obj_name, obj_type, distance, pos, rotation, desc, moveable, parent, asset_size)
                objdesc2cnt[obj_type] += 1

            moveable_visible_obj_assets = []
            for asset in nav_visible_obj_assets:
                if asset in assetid2info:
                    if assetid2info[asset][6] and assetid2info[asset][8]>0.003:
                        moveable_visible_obj_assets.append(asset)

            
            qa_pair_choices = []
            
            # choose random object
            if len(moveable_visible_obj_assets) < 2:
                print("No moveable objects, continuing")
                continue
            obj1 = random.choice(moveable_visible_obj_assets) # these are asset_ids that are visible and moveable

            obj1_cnt = objdesc2cnt[assetid2info[obj1][1]]

            new_pos = cam_pos
            new_rot = cam_rot

            xz_rotation_matrix = np.array([[np.cos(math.radians(new_rot['y'])), -np.sin(math.radians(new_rot['y']))], [np.sin(math.radians(new_rot['y'])), np.cos(math.radians(new_rot['y']))]])

            # Now let's ask questions about the object
            obj_name = assetid2info[obj1][0]

            obj_desc = assetid2info[obj1][5]
            obj1_type = assetid2info[obj1][1]

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
                else:
                    direction = "left"

                if angle > 90:
                    push_dir = "pull"
                else:
                    push_dir = "push"

                incorrect_angle_delta =random.choice([90, 120, 150, 100, 85, 75])
                qa_pair_choices.append((
                    f"What is the angle formed between the camera, {obj_desc}, and {another_obj_desc}?", 
                    (f"{angle} degrees to the {direction} from the forward direction when looking at it", f"{angle-incorrect_angle_delta} degrees to the {direction} from the forward direction when looking at it", f"{angle+incorrect_angle_delta} degrees to the {direction} from the forward direction when looking at it"),
                    "open, 3D, geometric reasoning"
                ))

                incorrect_angle_delta = random.choice([90, 120, 150, 100, 85, 75])
                qa_pair_choices.append((
                    f"Which direction should we push/pull {obj_desc} when facing it to move it closer to {another_obj_desc}?", 
                    (
                        f"{push_dir} at an angle {angle} degrees from the forward direction towards the {direction}",
                        f"{push_dir} at an angle {angle-incorrect_angle_delta} degrees from the forward direction towards the {direction}",
                        f"{push_dir} at an angle {angle+incorrect_angle_delta} degrees from the forward direction towards the {direction}",
                    ), 
                    "open, 3D reasoning, precise"
                ))

                # pdb.set_trace()

                ####### Distance between the two objects ############
                distance = np.linalg.norm(np.array(assetid2info[obj1][3]) - np.array(another_obj_pos))
                distance = int(distance*100)

                incorrect_distance_delta = random.choice([150, 250, 300, 400])

                qa_pair_choices.append((
                    f"What is the approximate distance between {obj_desc} and {another_obj_desc}?", 
                    (f"{distance} cm", f"{distance-incorrect_distance_delta} cm", f"{distance+incorrect_distance_delta} cm"),
                    "open, 3D, geometric reasoning"
                ))
            
                # pdb.set_trace()
                all_im_qas.append((apartment_id, new_path, new_pos, new_rot, qa_pair_choices))
                controller.stop()
            
            json.dump(all_im_qas, open(qa_json_path, "w"))
