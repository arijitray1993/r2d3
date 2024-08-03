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

    image_save_folder = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/images/train"

    asset_id_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/asset_descriptions_all.json", "r"))

    assetid2desc = {}
    for image_file, asset_name, object_class, caption in asset_id_desc:
        assetid2desc[asset_name] = caption


    gen_blink_style_depth = False
    gen_force_question = True
    qa_im_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/qa_images/'
    if not os.path.exists(qa_im_path):
        os.makedirs(qa_im_path)
    qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/physics_qas.json'

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

        if gen_force_question:
            for sample_i in range(2):
                controller = Controller(scene=house_json, width=800, height=800, quality="Ultra")
                try:
                    controller.step(action="Teleport", position=cam_pos, rotation=cam_rot)
                except:
                    print("Cannot teleport, continuing")
                    controller.stop()
                    continue

                #try:
                #    event = controller.step(action="GetReachablePositions")
                #except:
                #    print("Cannot get reachable positions, continuing")
                #    controller.stop()
                #    continue
                #reachable_positions = event.metadata["actionReturn"]
                
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


                '''
                # let's move closer to the object
                obj_pos = assetid2info[obj1][3]
                obj_pos = {"x": obj_pos[0], "y": obj_pos[1], "z": obj_pos[2]}

                # calculate rotation to face object
                cam_pos = controller.last_event.metadata['agent']['position']

                # teleport to nearer object and face it
                new_pos = {"x": (obj_pos['x'] + cam_pos['x'])*2/3, "y": cam_pos['y'], "z": (obj_pos['z'] + cam_pos['z'])*2/3}
                closest_reachable_positions = sorted(reachable_positions, key=lambda x: (x['x']-new_pos['x'])**2 + (x['z']-new_pos['z'])**2)
                new_pos = closest_reachable_positions[0]

                # make them snap 0.25's
                new_pos['x'] = round(new_pos['x']*4)/4.0
                new_pos['z'] = round(new_pos['z']*4)/4.0

                cam_to_obj_vector = np.array([obj_pos['x'] - new_pos['x'], obj_pos['z'] - new_pos['z']])
                cam_to_obj_angle = math.degrees(math.atan2(cam_to_obj_vector[1], cam_to_obj_vector[0]))
                if cam_to_obj_angle < 0:
                    cam_to_obj_angle = 360 + cam_to_obj_angle
                
                cam_to_obj_angle = cam_to_obj_angle + 90
                new_rot = {"x": 0, "y": cam_to_obj_angle, "z": 0}
                controller.step(action="Teleport", position=new_pos, rotation=new_rot)

                new_path = qa_im_path + f"{apartment_id}/{sample_i}_physics.jpg"

                # save the image
                img_view = Image.fromarray(controller.last_event.frame)
                img_view.save(new_path)
                '''
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
                    
                
                #######  which angle to push to move closer to some other object that is not obj1 #############
                if len(moveable_visible_obj_assets) > 1:
                    another_obj = random.choice(nav_visible_obj_assets)
                    while another_obj == obj1:
                        another_obj = random.choice(nav_visible_obj_assets)
                    
                    another_obj_desc = assetid2info[another_obj][5]

                    another_obj_cnt = objdesc2cnt[assetid2info[another_obj][1]]
                    if another_obj_cnt > 1:
                        # get the distances to camera
                        another_obj_distances = []
                        for asset in assetid2info:
                            if assetid2info[asset][1] == assetid2info[another_obj][1]:
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

                    # compute angle between camera, obj1 and another obj
                    cam_to_obj1_vector = np.array([assetid2info[obj1][3][0] - cam_pos['x'], assetid2info[obj1][3][2] - cam_pos['z']])
                    obj1_to_anotherobj_vec = np.array([another_obj_pos[0] - assetid2info[obj1][3][0], another_obj_pos[2] - assetid2info[obj1][3][2]])

                    angle = np.arccos(np.dot(cam_to_obj1_vector, obj1_to_anotherobj_vec)/(np.linalg.norm(cam_to_obj1_vector)*np.linalg.norm(obj1_to_anotherobj_vec)))
                    angle = math.degrees(angle)
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

                    qa_pair_choices.append((f"Which direction should we push/pull {obj_desc} when facing it to move it closer to {another_obj_desc}?", f"{push_dir} at an angle {angle} degrees from the forward direction towards the {direction}", "open, 3D, complex reasoning"))
 
                    pdb.set_trace()
                

                #### choose random force angle ########
                force_angle = random.choice(range(0, 360, 20))
                force_magnitude = random.choice(range(200, 1150, 50))
                
                direction = str(force_angle)
                if 350 < force_angle or force_angle < 10:
                    direction = "the forward"
                elif 80 < force_angle < 100:
                    direction = "the right"
                elif 170 < force_angle < 190:
                    direction = "the backward (pull)"
                elif 260 < force_angle < 280:
                    direction = "the left"
                else:
                    if force_angle > 180:
                        direction = f"{360 - force_angle} degrees counterclockwise from the forward"
                    else:
                        direction = f"{force_angle} degrees clockwise from the forward"

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

                qa_pair_choices.append((f"Is there a {obj_desc} visible in this picture?", f"yes", "baseline, semantics"))

                qa_pair_choices.append((f"Is {obj_desc} likely to be visible in this frame after applying a force of {force_magnitude} Newtons in {direction} direction?", obj1_visible, "binary, physics, coarse location"))

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
                            qa_pair_choices.append((f"If we apply a force of {force_magnitude} Newtons in {direction} direction to {obj_desc}, where does it end up?", f"The object will likely go from {current_parent} to the {updated_obj_to_info[obj1][3]}", "open, physics, complex reasoning"))
                        else:
                            qa_pair_choices.append((f"If we apply a force of {force_magnitude} Newtons in {direction} direction to {obj_desc}, where does it end up?", f"The object will likely end up on the {updated_obj_to_info[obj1][3]}", "open, physics, complex reasoning"))
                    else:
                        qa_pair_choices.append((f"If we apply a force of {force_magnitude} Newtons in {direction} direction to {obj_desc}, where does it end up?", f"The object will likely stay on the {current_parent}", "open, physics, complex reasoning"))

                # check distance traveled  
                distance_traveled = math.sqrt(sum((updated_obj_to_info[obj1][0] - assetid2info[obj1][3])**2))

                qa_pair_choices.append(
                    (f"If we apply a force of {force_magnitude} Newtons in the {direction} direction to {obj_desc}, how far does the object move?", 
                    f"It will likely move {int(distance_traveled*100)} cm and be on the {updated_obj_to_info[obj1][3]}", "open, distance")
                    )
                
                # check if object farther or closer to camera
                dist_to_cam_closer = int((assetid2info[obj1][2] - updated_obj_to_info[obj1][2])*100)
                if dist_to_cam_closer < 0:
                    qa_pair_choices.append((f"If we apply a force of {force_magnitude} Newtons in the {direction} direction to {obj_desc}, does it end up closer to the camera?", "no", "binary"))
                    qa_pair_choices.append((f"How much farther is {obj_desc} from the camera after applying a force of {force_magnitude} Newtons in the {direction} direction?", f"{int((updated_obj_to_info[obj1][2] - assetid2info[obj1][2])*100)} cm", "open, physics, precise relative"))
                elif dist_to_cam_closer > 0:
                    qa_pair_choices.append((f"If we apply a force of {force_magnitude} Newtons in the {direction} direction to {obj_desc}, does it end up closer to the camera?", "yes", "binary"))
                    qa_pair_choices.append((f"How much closer is {obj_desc} to the camera after applying a force of {force_magnitude} Newtons in the {direction} direction?", f"{int((assetid2info[obj1][2] - updated_obj_to_info[obj1][2])*100)} cm", "open, physics, precise relative"))


                # generate 3D precise location question
                og_location = (np.array(assetid2info[obj1][3])*100).astype(np.int32)
                updated_location = (np.array(updated_obj_to_info[obj1][0])*100).astype(np.int32)

                camera_pos_form = (np.array([cam_pos['x'], cam_pos['z']])*100).astype(np.int32)
                camera_rot_form = int(cam_rot['y'])

                qa_entry = f"Say, we are looking at the scene from position (x,z) {list(camera_pos_form)} with rotation {camera_rot_form}. The 3D position of the {obj_desc} is (x,y,z) {list(og_location)} here. \
We apply a force of {force_magnitude} Newtons in the {direction} direction. Where does the object likely end up in 3D space?"

                qa_pair_choices.append((qa_entry, f"The object likely is at the 3D location {list(updated_location)}", "open, physics, precise location"))


                # pick another object that moved that is not obj1
                if len(updated_obj_to_info) > 1:
                    for other_obj in updated_obj_to_info:
                        if other_obj == obj1:
                            continue
                        other_obj_desc = assetid2info[other_obj][5]

                        og_otherobj_loc = assetid2info[other_obj][3]
                        updated_otherobj_loc = updated_obj_to_info[other_obj][0]

                        qa_entry = f"If we apply a force of {force_magnitude} Newtons in the {direction} direction to {obj_desc}, how far will the {other_obj_desc} likely travel?"

                        distance_traveled = math.sqrt(sum((updated_otherobj_loc - og_otherobj_loc)**2))

                        distance_traveled = int(distance_traveled*100)

                        if distance_traveled > 0:
                            qa_pair_choices.append((qa_entry, f"The {other_obj_desc} will likely move {distance_traveled} cm because it collided.", "open, physics, complex reasoning"))
                        else:
                            qa_pair_choices.append((qa_entry, f"The {other_obj_desc} will likely not move", "open, physics, complex reasoning"))

                pdb.set_trace()
                controller.stop()
                print(random.sample(qa_pair_choices, 3))
                all_im_qas.append((apartment_id, new_path, new_pos, new_rot, qa_pair_choices))
            json.dump(all_im_qas, open(qa_json_path, "w"))


        


        if gen_navigation_question:
            pass


        if gen_blink_style_depth:
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

                pdb.set_trace()
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
                    question = f"Is "
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


            controller.stop()

            json.dump(all_im_qas, open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/depth_reasoning_qas.json", "w"))

       