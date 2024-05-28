import json
import prior
from ai2thor.controller import Controller
from PIL import Image
import random
from pprint import pprint
import pdb
import math

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import tqdm
import copy
import os

import sys
sys.path.append("../")
from utils.ai2thor_utils import generate_program_from_roomjson, generate_room_programs_from_house_json, make_house_from_cfg


def get_top_down_frame(controller):
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)


def render_room_program_images(program_json_data_path, image_save_folder="", save_path="", load_progress=True, done_path=""):

        # pdb.set_trace()
        program_json_data = json.load(open(program_json_data_path))
        
        if load_progress:
            image_program_json_data = json.load(open(done_path))

            done_inds = []
            for program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name in image_program_json_data:
                if len(all_imgs) > 0:
                    apartment_ind = all_imgs[0].split("/")[-2].split("_")[-1]
                    done_inds.append(int(apartment_ind))
            
            if save_path == done_path:
                print("Warning, might overwrite the same file, exiting")
                return
        # else:
        image_program_json_data = []

        for ind, (program_text, house_json, og_house_json) in enumerate(tqdm.tqdm(program_json_data)):
            
            if ind>=200:
                break

            # if we have to continue from a certain index
            if load_progress:
                # if ind < max_ind: # len(image_program_json_data):
                #    continue
                #if ind < len(image_program_json_data):
                #    continue

                if ind <= max(done_inds):
                    continue
                    
            # pdb.set_trace()
            if ind in [5, 11, 23, 32]: # everything crashes unexpectedly for these, to do look into later
                continue
            # pdb.set_trace()
            # render the json
            
            try:
                controller = Controller(scene=house_json, width=800, height=800, renderInstanceSegmentation=True, visibilityDistance=5)
                # pdb.set_trace()
            except:
                print("Cannot render environment, continuing")
                # pdb.set_trace()
                continue

            ## place the camera at the room corners and take pictures of room.
            # 1. get reachable positions 
            try:
                event = controller.step(action="GetReachablePositions")
            except:
                print("Cannot get reachable positions, continuing")
                controller.stop()
                continue
            reachable_positions = event.metadata["actionReturn"]
            try:
                print("number of reachable positions: ", len(reachable_positions))
            except:
                print("no reachable positions, continuing")
                controller.stop()
                continue

            # 2. get the corner x,z coordinates from house json room polygon
            corner_positions = []
            room = house_json['rooms'][0]
            polygon = room['floorPolygon']
            for point in polygon:
                corner_positions.append((point['x'], point['z']))
            
            cam_ind_to_position = {}
            # 3. get the interior angle for each corner
            # hacky way: get the neihboring two polygon points, compute the centroid
            # , check if centroid is inside the polygon, if inside, compute the angle of the vector from the corner to the centroid to the x-axis
            # if outside, compute the 180-angle. 

            shape_polygon = Polygon(corner_positions)

            # make a mapping to obj id to obj name
            obj_id_to_name = {}
            for obj in house_json['objects']:
                obj_id_to_name[obj['id']] = obj['assetId']  
            
            all_imgs = []
            all_objs = []
            all_seg_frames = []
            if not os.path.exists(f"{image_save_folder}/example_{ind}"):
                os.makedirs(f"{image_save_folder}/example_{ind}")
            for corner_ind, corner in enumerate(corner_positions):
                prev_ind = (corner_ind-1)%len(corner_positions)
                next_ind = (corner_ind+1)%len(corner_positions)
                
                PREV_POINT = corner_positions[prev_ind]
                NEXT_POINT = corner_positions[next_ind] 

                centroid_x = (PREV_POINT[0] + NEXT_POINT[0] + corner[0])/3
                centroid_z = (PREV_POINT[1] + NEXT_POINT[1] + corner[1])/3

                # check if centroid is inside polygon
                in_polygon = shape_polygon.contains(Point(centroid_x, centroid_z))
                
                # compute angle from corner-centroid vector to x-axis
                x1, z1 = corner
                x2, z2 = centroid_x, centroid_z
                pi = 3.1415
                try:
                    angle_pos_y = 180*math.atan((x2-x1)/(z2-z1))/pi
                except:
                    print("not really a corner")
                    controller.stop()
                    continue
                
                if x2 > x1 and z2 > z1:
                    quadrant = "First Quadrant"
                elif x2 < x1 and z2 > z1:
                    quadrant = "Second Quadrant"
                elif x2 < x1 and z2 < z1:
                    quadrant = "Third Quadrant"
                elif x2 > x1 and z2 < z1:
                    quadrant = "Fourth Quadrant"

                if angle_pos_y < 0:
                    if quadrant == "Second Quadrant":
                        angle = 360 + angle_pos_y
                    elif quadrant == "Fourth Quadrant":
                        angle = 180 + angle_pos_y
                else:
                    if quadrant == "First Quadrant":
                        angle = angle_pos_y
                    elif quadrant == "Third Quadrant":
                        angle = 180 + angle_pos_y
                
                if not in_polygon:
                    angle = angle + 180
                
                print("corner", corner, "angle", angle)

                # 4. find the closest reachable position and teleport the agent there and place camera at angle
                closest_reachable_positions = sorted(reachable_positions, key=lambda x: (x['x']-corner[0])**2 + (x['z']-corner[1])**2)
                position = random.choice(closest_reachable_positions[:10])

                position['x'] = round(position['x']*4)/4.0
                position['z'] = round(position['z']*4)/4.0
                position['y'] = 0.9
                rotation = { "x": 0.0, "y": angle, "z": 0.0} 

                # check if position is in polygon
                if not shape_polygon.contains(Point(position['x'], position['z'])):
                    print("not in polygon")
                    # controller.stop()
                    continue             
                
                try:
                    event = controller.step(action="Teleport", position=position, rotation=rotation)
                except:
                    print("teleport failed")
                    # controller.stop()
                    continue
                
                if not event.metadata['lastActionSuccess']:
                    print("teleport failed")
                    continue 


                img = Image.fromarray(controller.last_event.frame)
                img.save(f"{image_save_folder}/example_{ind}/{corner_ind}.png")

                cam_ind_to_position[corner_ind] = (position, rotation)
                all_imgs.append(f"{image_save_folder}/example_{ind}/{corner_ind}.png")

                ''' this suddenly stopped working in ai2thor
                objects = []
                for key in controller.last_event.instance_detections2D:
                    objects.append((key.split("|")[0], controller.last_event.instance_detections2D[key]))
                
                all_objs.append(objects)
                '''
                ## a hacky way to get objects in the room
                segmentation_frame = controller.last_event.instance_segmentation_frame
                all_colors = segmentation_frame.reshape(-1, 3)
                unique_colors = set([tuple(color) for color in all_colors])
                color_to_obj = event.color_to_object_id

                color_to_objid = {} # a new one to save since keys can't be tuple when saving json
                for keys in color_to_obj:
                    color_to_objid[str(keys)] = color_to_obj[keys]

                # pdb.set_trace()

                objs = []
                for color in unique_colors:
                    if color in color_to_obj:
                        objs.append(color_to_obj[color])

                # now these objs are interestingly the id and not the asset name - which is useless. 
                # so we need to get the asset name from the id
                obj_asset_names = []
                for obj in objs:
                    if obj in obj_id_to_name:
                        obj_name = " ".join(obj_id_to_name[obj].split("_")[:-1])
                        obj_asset_names.append(obj_name)
                
                all_objs.append(obj_asset_names)

                # save segmentation frame
                seg_img = Image.fromarray(segmentation_frame)
                seg_img.save(f"{image_save_folder}/example_{ind}/{corner_ind}_seg.png")

                all_seg_frames.append(f"{image_save_folder}/example_{ind}/{corner_ind}_seg.png")
                # pdb.set_trace()


            image_program_json_data.append((program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name))
            # save image of top down view
            '''
            try:
                top_down_frame = get_top_down_frame(controller)
                top_down_frame.save(f"vis/ai2thor/example_top_down_{ind}.png")
            except:
                print("couldnt get top down frame")
            '''
            #pdb.set_trace()
            json.dump(image_program_json_data, open(save_path, "w"))
            controller.stop()


def generate_house_programs_train():
    dataset = prior.load_dataset("procthor-10k")

    all_room_json_programs = []

    for ind, entry in enumerate(dataset["train"]):
        house_json = entry

        room_yamls_programs = generate_room_programs_from_house_json(house_json)

        all_room_json_programs.extend(room_yamls_programs)

    json.dump(all_room_json_programs, open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_10k_room_json_programs_train_windowsadded_format.json", "w"))

def generate_house_programs_val():
    dataset = prior.load_dataset("procthor-10k")

    all_room_json_programs = []

    for ind, entry in enumerate(dataset["val"]):
        house_json = entry

        room_yaml_programs = generate_room_programs_from_house_json(house_json)

        all_room_json_programs.extend(room_yaml_programs)

    json.dump(all_room_json_programs, open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_10k_room_json_programs_val_windowsadded.json", "w"))


if __name__=="__main__":

    # generate_house_programs_train()
    #generate_house_programs_val()

    im_folder_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/images/val"
    program_json_data_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_10k_room_json_programs_val_windowsadded.json"

    save_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_val_childrenadded.json"

    if not os.path.exists(im_folder_path):
        os.makedirs(im_folder_path)
    render_room_program_images(program_json_data_path, image_save_folder=im_folder_path, save_path=save_path, load_progress=False)
