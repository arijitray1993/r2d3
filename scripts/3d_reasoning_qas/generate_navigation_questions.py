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
    asset_id_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/asset_descriptions_all.json", "r"))
    qa_im_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/qa_images/'
    qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/physics_qas.json'


    assetid2desc = {}
    for image_file, asset_name, object_class, caption in asset_id_desc:
        assetid2desc[asset_name] = caption

    if not os.path.exists(qa_im_path):
        os.makedirs(qa_im_path)
    
    all_im_qas = []
    maximum_distance = 30
    for ind, (program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name) in enumerate(tqdm.tqdm(image_program_json_data)):
        
        if ind<58:
            continue

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
            controller = Controller(scene=house_json, width=300, height=300, quality="Low") # quality="Ultra", renderInstanceSegmentation=True, visibilityDistance=30)
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
            controller = Controller(scene=house_json, width=800, height=800, quality="Ultra")
            try:
                controller.step(action="Teleport", position=cam_pos, rotation=cam_rot)
            except:
                print("Cannot teleport, continuing")
                controller.stop()
                continue
            
            polygon =  house_json['rooms'][0]['floorPolygon']
            corner_positions = []
            for point in polygon:
                corner_positions.append((point['x'], point['z']))

            shape_polygon = Polygon(corner_positions) 

            format_polygon_coords_num = []
            for point in list(shape_polygon.exterior.coords):
                format_polygon_coords_num.append((int(round(point[0],2)*100), int(round(point[1],2)*100)))
            
            format_polygon_coords = str(format_polygon_coords_num)  

            cam_pos_ind = int(max_vis_cam_pos_ind)
            im_corner = all_imgs[cam_pos_ind].split("/")[-1].split(".")[0]

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
                assetid2info[asset_id] = (obj_name, obj_type, distance, pos, rotation, desc, moveable, parent, asset_size)
                objdesc2cnt[obj_type] += 1

            moveable_visible_obj_assets = []
            for asset in nav_visible_obj_assets:
                if asset in assetid2info:
                    if assetid2info[asset][6] and assetid2info[asset][8]>0.003:
                        moveable_visible_obj_assets.append(asset)

            qa_pair_choices = []
            
            # choose a random angle and direction to move. 
            
            
            
            all_im_qas.append((apartment_id, new_path, new_pos, new_rot, qa_pair_choices))
        json.dump(all_im_qas, open(qa_json_path, "w"))


        


        
         
       