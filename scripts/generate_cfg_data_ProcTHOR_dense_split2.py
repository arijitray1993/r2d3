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


def render_room_program_images(program_json_data_path, image_save_folder="", load_progress=True):

        # pdb.set_trace()
        program_json_data = json.load(open(program_json_data_path))
        #except:

        done_inds = [int(im_ind.split(".")[0].split("_")[1]) for im_ind in os.listdir(image_save_folder)]
        done_inds = list(set(done_inds))
    
        if load_progress:
            image_program_json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_densemultiview_split2.json"))
        else:
            image_program_json_data = []
        
        for ind, (program_text, house_json, og_house_json) in enumerate(tqdm.tqdm(program_json_data)):
            
            if ind<3000:
                continue

            # if we have to continue from a certain index
            if load_progress:
                # if ind < max_ind: # len(image_program_json_data):
                #    continue
                if ind < 3000 + len(image_program_json_data):
                    continue

                if ind in done_inds:
                    continue
            
            # pdb.set_trace()
            if ind in [5, 11, 23, 32]: # everything crashes unexpectedly for these, to do look into later
                continue
            # pdb.set_trace()
            # render the json
            
            try:
                controller = Controller(scene=house_json, width=800, height=800)
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

            cam_ind_to_position = {}
            
            obj_id_to_name = {}
            for obj in house_json['objects']:
                obj_id_to_name[obj['id']] = obj['assetId']  
            
            all_imgs = []
            all_objs = []
            all_seg_frames = []

            randomly_sampled_positions = random.sample(reachable_positions, min(200, len(reachable_positions)))

            print("number of randomly sampled positions: ", len(randomly_sampled_positions))

            if not os.path.exists(f"{image_save_folder}/apartment_{ind}"):
                os.makedirs(f"{image_save_folder}/apartment_{ind}")

            position_count = 0
            for corner_ind, position in enumerate(randomly_sampled_positions):
                for angle in random.sample(range(0, 360, 20), 6):
                    position['x'] = round(position['x']*4)/4.0
                    position['z'] = round(position['z']*4)/4.0
                    position['y'] = 0.9
                    rotation = { "x": 0.0, "y": angle, "z": 0.0} 

                    
                    try:
                        event = controller.step(action="Teleport", position=position, rotation=rotation)
                    except:
                        print("teleport failed")
                        continue
                        
                    if not event.metadata['lastActionSuccess']:
                        print("teleport failed")
                        continue 

                    img = Image.fromarray(controller.last_event.frame)
                    img.save(f"{image_save_folder}/apartment_{ind}/example_{position_count}.png")

                    cam_ind_to_position[position_count] = (position, rotation)
                    all_imgs.append(f"{image_save_folder}/apartment_{ind}/example_{position_count}.png")

                    position_count+=1
                if position_count > 600:
                    break

            image_program_json_data.append((program_text, house_json, og_house_json, cam_ind_to_position, all_imgs))
            # save image of top down view
            '''
            try:
                top_down_frame = get_top_down_frame(controller)
                top_down_frame.save(f"vis/ai2thor/example_top_down_{ind}.png")
            except:
                print("couldnt get top down frame")
            '''
            #pdb.set_trace()
            json.dump(image_program_json_data, open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_densemultiview_split2.json", "w"))
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
    # generate_house_programs_val()

    im_folder_path = "//projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/denseviews/images/"
    if not os.path.exists(im_folder_path):
        os.makedirs(im_folder_path)
    program_json_data_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_10k_room_json_programs_train_windowsadded_format.json"

    render_room_program_images(program_json_data_path, image_save_folder=im_folder_path, load_progress=True)
