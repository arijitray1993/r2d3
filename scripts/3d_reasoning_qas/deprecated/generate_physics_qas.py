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
            
            # choose random object
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
                    if assetid2info[asset][1] == assetid2info[obj1][1]:
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
            

            #### choose random force angle ########
            force_angle = random.choice(range(0, 360, 20))
            force_magnitude = random.choice(range(200, 1150, 50))
            if 315 < force_angle < 0: 
                force_direction = f"{360-force_angle} degrees left to the forward direction"   
            elif 0 < force_angle < 45:
                force_direction = f"{force_angle} degrees right to the forward direction"
            elif 45 < force_angle < 135:
                force_direction = f"{force_angle} degrees right to the forward direction"
            elif 135 < force_angle < 180:
                force_direction = f"pull {180-force_angle} degrees left to the forward direction"
            elif 180 < force_angle < 225:
                force_direction = f"pull {360-force_angle} degrees right to the forward direction"
            elif 225 < force_angle < 315:
                force_direction = f"{360-force_angle} degrees left to the forward direction"
            

            # apply force at direction
            event = controller.step(action="DirectionalPush", objectId=obj_name, moveMagnitude=force_magnitude, pushAngle=force_angle, forceAction=True)

            if not event.metadata["lastActionSuccess"]:
                print("Force application failed, continuing")
                controller.stop()
                continue

            # get updated object location
            updated_obj_to_info = {}
            for updated_obj in controller.last_event.metadata['objects']:
                updated_obj_asset = updated_obj['assetId']
                if updated_obj_asset == "":
                    continue
                if updated_obj_asset not in assetid2info:
                    continue
                updated_obj_pos = np.array([updated_obj['position']['x'], updated_obj['position']['y'], updated_obj['position']['z']])

                # check if different from original object pos
                if np.all(updated_obj_pos != assetid2info[updated_obj_asset][3]): 
                    up_obj_parent = updated_obj.get('parentReceptacles')
                    # pdb.set_trace()
                    if up_obj_parent is not None:
                        if len(up_obj_parent) > 0:
                            up_obj_parent = up_obj_parent[-1]
                            if up_obj_parent== "Floor":
                                up_obj_parent = "Floor"
                            else:
                                up_obj_parent = objid2assetid[up_obj_parent]
                                up_obj_parent = assetid2info[up_obj_parent][1]
                    if updated_obj_pos[0] < 0 or updated_obj_pos[1] < 0 or updated_obj_pos[2] < 0:
                        continue
                    if not shape_polygon.contains(Point(updated_obj_pos[0], updated_obj_pos[2])):
                        continue
                    updated_obj_to_info[updated_obj_asset] = (updated_obj_pos, updated_obj['rotation'], updated_obj['distance'], up_obj_parent)

            if obj1 not in updated_obj_to_info:
                print("Object did not move, or some error, continuing")
                controller.stop()
                continue
            # check if current object is visible
            nav_visible_objects = controller.step(
                "GetVisibleObjects",
                maxDistance=maximum_distance,
            ).metadata["actionReturn"]

            obj1_visible = "yes" if obj_name in nav_visible_objects else "no"
            # check distance traveled  
            distance_traveled = math.sqrt(sum((updated_obj_to_info[obj1][0] - assetid2info[obj1][3])**2))
            distance = int(distance_traveled*100)

            if distance_traveled < 3:
                print("Object did not move, continuing")
                controller.stop()
                continue

            # compute angle between camera, original and updated object
            cam_to_obj1_vector = np.array([assetid2info[obj1][3][0] - cam_pos['x'], assetid2info[obj1][3][2] - cam_pos['z']])
            obj1_to_updatedobj_vec = np.array([updated_obj_to_info[obj1][0][0] - assetid2info[obj1][3][0], updated_obj_to_info[obj1][0][2] - assetid2info[obj1][3][2]])

            angle = np.arccos(np.dot(cam_to_obj1_vector, obj1_to_updatedobj_vec)/(np.linalg.norm(cam_to_obj1_vector)*np.linalg.norm(obj1_to_updatedobj_vec)))
            angle = math.degrees(angle)

            # rotate using rotation matrix to figure out whether another obj is left to right to camera
            obj1_rotated_pos = np.dot(xz_rotation_matrix, np.array([assetid2info[obj1][3][0], assetid2info[obj1][3][2]]))
            updated_obj_rotated_pos = np.dot(xz_rotation_matrix, np.array([updated_obj_to_info[obj1][0][0], updated_obj_to_info[obj1][0][2]]))

            if updated_obj_rotated_pos[0] > obj1_rotated_pos[0]:
                direction = "right"
            else:
                direction = "left"

            qa_pair_choices.append((f"Is there a {obj_desc} visible in this picture?", f"yes", "baseline, semantics"))

            qa_pair_choices.append((f"Is {obj_desc} likely to be visible in this frame if pushed {distance} cm to the {direction} at an angle of {angle} from the forward direction if facing the object?", obj1_visible, "binary, 3D reasoning, coarse"))

            current_parent = assetid2info[obj1][7]
            if current_parent is not None:
                if current_parent != "Floor":
                    current_parent = assetid2info[current_parent][1]
                qa_pair_choices.append((f"Where is the {obj_desc} in this picture?", f"It is on the {current_parent}", "baseline, semantics"))
            # check parent of object
            if updated_obj_to_info[obj1][3] is not None:
                if assetid2info[obj1][7]!=updated_obj_to_info[obj1][3]:
                    # check where is object right now
                    if current_parent is not None:
                        qa_pair_choices.append((f"If {obj_desc} is pushed {distance} cm to the {direction} at an angle of {angle} from the forward direction if facing the object, where does it end up?", f"The object will likely go from {current_parent} to the {updated_obj_to_info[obj1][3]}", "open, 3D + physics, 2 step complex"))
                    else:
                        qa_pair_choices.append((f"If {obj_desc} is pushed {distance} cm to the {direction} at an angle of {angle} from the forward direction if facing the object, where does it end up?", f"The object will likely end up on the {updated_obj_to_info[obj1][3]}", "open, 3D + physics, 2 step complex"))
                else:
                    qa_pair_choices.append((f"If {obj_desc} is pushed {distance} cm to the {direction} at an angle of {angle} from the forward direction if facing the object, where does it end up?", f"The object will likely stay on the {current_parent}", "open, 3D + physics, 2 step complex"))


            # check if object farther or closer to camera
            dist_to_cam_closer = int((assetid2info[obj1][2] - updated_obj_to_info[obj1][2])*100)
            if dist_to_cam_closer < 0:
                qa_pair_choices.append((f"If {obj_desc} is pushed {distance} cm to the {direction} at an angle of {angle} from the forward direction if facing the object, does it end up closer to the camera?", "no", "binary"))
                qa_pair_choices.append((f"How much farther is {obj_desc} from the camera if pushed {distance} cm to the {direction} at an angle of {angle} from the forward direction if facing the object?", f"{int((updated_obj_to_info[obj1][2] - assetid2info[obj1][2])*100)} cm", "open, 3D reasoning, precise"))
            elif dist_to_cam_closer > 0:
                qa_pair_choices.append((f"If {obj_desc} is pushed {distance} cm to the {direction} at an angle of {angle} from the forward direction if facing the object, does it end up closer to the camera?", "yes", "binary"))
                qa_pair_choices.append((f"How much closer is {obj_desc} from the camera if pushed {distance} cm to the {direction} at an angle of {angle} from the forward direction if facing the object?", f"{int((assetid2info[obj1][2] - updated_obj_to_info[obj1][2])*100)} cm", "open, 3D reasoning, precise"))


            # generate 3D precise location question
            og_location = (np.array(assetid2info[obj1][3])*100).astype(np.int32)
            updated_location = (np.array(updated_obj_to_info[obj1][0])*100).astype(np.int32)

            camera_pos_form = (np.array([cam_pos['x'], cam_pos['z']])*100).astype(np.int32)
            camera_rot_form = int(cam_rot['y'])

            qa_entry = f"Say, we are looking at the scene from position (x,z) {list(camera_pos_form)} with rotation {camera_rot_form}. The 3D position of the {obj_desc} is (x,y,z) {list(og_location)} here. \
{obj_desc} is pushed {distance} cm to the {direction} at an angle of {angle} from the forward direction while facing the object. Where does the object likely end up in 3D space?"

            qa_pair_choices.append((qa_entry, f"The object likely is at the 3D location {list(updated_location)}", "open, 3D reasoning, precise"))


            if len(updated_obj_to_info) > 1:
                moved_objs = []
                for other_obj in updated_obj_to_info:
                    if other_obj == obj1:
                        continue
                    
                    og_otherobj_loc = assetid2info[other_obj][3]
                    updated_otherobj_loc = updated_obj_to_info[other_obj][0]

                    distance_traveled = math.sqrt(sum((updated_otherobj_loc - og_otherobj_loc)**2))

                    distance_traveled = int(distance_traveled*100)

                    if distance_traveled > 1:
                        moved_objs.append((other_obj, og_otherobj_loc, updated_otherobj_loc))
                unmoved_objs = []
                for other_obj in assetid2info:
                    if other_obj not in updated_obj_to_info:
                        unmoved_objs.append(other_obj)

                for other_obj, og_otherobj_loc, updated_otherobj_loc in moved_objs:
                    other_obj_desc = assetid2info[other_obj][5]

                    og_otherobj_obj1_dist = math.sqrt(sum((assetid2info[obj1][3] - og_otherobj_loc)**2))

                    qa_entry = f"If {obj_desc} is pushed {og_otherobj_obj1_dist} cm at {force_direction}, which object might it collide with?"

                    qa_pair_choices.append((qa_entry, f"{other_obj_desc}", "open, 3D reasoning, precise"))
            


            pdb.set_trace()
            controller.stop()
            print(random.sample(qa_pair_choices, 3))
            all_im_qas.append((apartment_id, new_path, new_pos, new_rot, qa_pair_choices))
        json.dump(all_im_qas, open(qa_json_path, "w"))


        


        
         
       