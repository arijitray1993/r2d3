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
sys.path.append("../")
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

    all_im_qas = []
    maximum_distance = 30
    for ind, (program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name) in enumerate(tqdm.tqdm(image_program_json_data)):
        
        if len(all_imgs) == 0:
            continue

        if ind < 8000:
            continue

        apartment_id = all_imgs[0].split("/")[-2]

        try:
            controller = Controller(scene=house_json, width=800, height=800, visiblityDistance=30) #, renderInstanceSegmentation=True, visibilityDistance=30)
        except:
            print("Cannot render environment, continuing")
            # pdb.set_trace()
            continue
        
        max_objects = 0
        for im_ind, objlist in enumerate(all_objs):
            obj_count = len(objlist)
            if obj_count > max_objects:
                max_objects = obj_count
                cam_pos_ind = im_ind

        if max_objects == 0:
            controller.stop()
            continue

        im_corner = all_imgs[cam_pos_ind].split("/")[-1].split(".")[0]
        
        cam_pos, cam_rot = cam_ind_to_position[im_corner]
        

        try:
            controller.step(action="Teleport", position=cam_pos, rotation=cam_rot)
        except:
            print("Cannot teleport, continuing")
            controller.stop()
            continue

        controller.step(
            action="RandomizeLighting",
            brightness=(1.2, 1.7),
            randomizeColor=True,
            hue=(0, 0.7),
            saturation=(0.4, 0.7),
            synchronized=False
        )

        objid2assetid = {}
        for obj in controller.last_event.metadata['objects']:
            objid2assetid[obj['objectId']] = (obj['assetId'], obj['objectType'])

        img_view = Image.fromarray(controller.last_event.frame)
        

        ## spatial relations and relative depth

        # get the distances from camera
        nav_visible_objects = controller.step(
            "GetVisibleObjects",
            maxDistance=maximum_distance,
        ).metadata["actionReturn"]

        nav_visible_obj_name = [objid2assetid[obj] for obj in nav_visible_objects] 

        objid2dist = {}
        for obj_entry in controller.last_event.metadata['objects']:
            obj_name = obj_entry['name']
            asset_id = obj_entry['assetId']
            distance = obj_entry['distance']
            objid2dist[asset_id] = distance

        objectvis2distance = {}
        for asset_id, objtype in nav_visible_obj_name:
            obj_distance = objid2dist[asset_id]
            objectvis2distance[asset_id] = (obj_distance, objtype)

        vis_obj_names = list(objectvis2distance.keys())
        vis_obj_names = [vis_name for vis_name in vis_obj_names if vis_name !='']
        # pdb.set_trace()
        # get the locations in seg frame
        seg_frame_path = all_seg_frames[cam_pos_ind]
        seg_frame = Image.open(seg_frame_path).convert("RGB")
        seg_frame = np.array(seg_frame)
        all_colors = seg_frame.reshape(-1, 3)
        unique_colors = np.unique(all_colors, axis=0)
        obj_ids_present = [color_to_objid.get(str(tuple(color))) for color in unique_colors]
        obj_ids_present = [obj_id for obj_id in obj_ids_present if obj_id is not None]

        objid_to_color = {}
        for color in color_to_objid:
            objid_to_color[color_to_objid[color]] = color

        # only obj_id in program_text based colors are accurate, ignore other colors.  

        program_json = yaml.load(program_text, Loader=yaml.FullLoader)
        assetids_to_programobjids = {}
        i = 0
        while(True):
            if f'obj_{i}' in program_json:
                assetids_to_programobjids[program_json[f'obj_{i}'][0]] = f'obj_{i}'
                i += 1
            else:
                break
        i = 0
        while(True):
            if f'child_{i}' in program_json:
                assetids_to_programobjids[program_json[f'child_{i}'][0]] = f'child_{i}'
                i += 1
            else:
                break
        
        i = 0
        while(True):
            if f'window_{i}' in program_json:
                assetids_to_programobjids[program_json[f'window_{i}'][0]] = f'window_{i}'
                i += 1
            else:
                break

        for di in range(min(2, len(vis_obj_names)//2)):
            obj1, obj2 = random.sample(vis_obj_names, 2) # these are asset_ids
            
            programobj1 = assetids_to_programobjids[obj1]
            programobj2 = assetids_to_programobjids[obj2]

            # mark the two objects in the image with red dots 
            obj1_color = list(ast.literal_eval(objid_to_color[programobj1]))

            color_mask = np.all(seg_frame == obj1_color, axis=-1)
            y1, x1 = np.where(color_mask)
            if len(y1) == 0:
                continue
            if len(x1) == 0:
                continue
            x1 = x1.mean()
            y1 = y1.mean()

            obj2_color = list(ast.literal_eval(objid_to_color[programobj2]))
            color_mask = np.all(seg_frame == obj2_color, axis=-1)
            y2, x2 = np.where(color_mask)
            if len(y2) == 0:
                continue
            if len(x2) == 0:
                continue
            x2 = x2.mean()
            y2 = y2.mean()

            img_view_marked = add_red_dot_with_text(img_view.copy(), (x1, y1), "1")
            img_view_marked = add_red_dot_with_text(img_view_marked, (x2, y2), "2")

            obj1_distance, obj1_type = objectvis2distance[obj1]
            obj2_distance, obj2_type = objectvis2distance[obj2]

            # make a qa pair
            questions = []
            if obj1_distance < obj2_distance:
                question = f"Is the point marked A closer to the camera than the point marked B?"
                answer = "yes"
                questions.append((question, answer))
                question = f"Two points are circled on the image, labeled by A and B. Which point is closer to the camera?"
                answer = f"Point A with object {obj1_type}"
                questions.append((question, answer))
                question = f"Is the point marked B closer to the camera than the point marked A?"
                answer = "no"
                questions.append((question, answer))
            else:
                question = f"Is the point marked B closer to the camera than the point marked A?"
                answer = "yes"
                questions.append((question, answer))
                question = f"Two points are circled on the image, labeled by A and B. Which point is closer to the camera?"
                answer = f"Point B with object {obj2_type}"
                questions.append((question, answer))
                question = f"Is the point marked A closer to the camera than the point marked B?"
                answer = "no"
                questions.append((question, answer))

            im_name = f"{apartment_id}_{ind}_qaview_{di}.jpg"
            img_view_marked.save(os.path.join(image_save_folder, im_name))
            all_im_qas.append((apartment_id, im_name, questions))

            # pdb.set_trace()




        
        '''
        # move and object view
        #   forward
        obj_classes_before = [objid2assetid[obj][1] for obj in nav_visible_objects]

        distances = random.sample(range(1, 5), 3)

        for distance in distances:
            controller.step(action="MoveAhead", moveMagnitude=distance)
            
            nav_visible_objects = controller.step(
                "GetVisibleObjects",
                maxDistance=maximum_distance,
            ).metadata["actionReturn"]

            obj_classes_after = [objid2assetid[obj][1] for obj in nav_visible_objects]
        '''
            




            

        #   rotate


        # move and collision
        #   forward


        #   rotate


        # apply force at direction 


        # pickup and throw at direction


        controller.stop()
    
        json.dump(all_im_qas, open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/depth_reasoning_qas_split3.json", "w"))

       