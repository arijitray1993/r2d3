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

    pose["fieldOfView"] = 30
    pose["position"]["y"] += 0.7 * max_bound
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

    #controller.step('ToggleMapView')
    #top_down_seg_frame = controller.last_event.instance_segmentation_frame
    
    return Image.fromarray(top_down_frame)# , Image.fromarray(top_down_seg_frame)


if __name__ == "__main__":
    
    image_program_json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_train_childrenadded_all_14k.json", "r"))

    image_save_folder = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/images/train"

    all_top_down_ims = []
    for ind, (program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name) in enumerate(tqdm.tqdm(image_program_json_data)):
        
        if len(all_imgs) == 0:
            continue

        apartment_id = all_imgs[0].split("/")[-2]

        # pdb.set_trace()
        if os.path.exists(os.path.join(image_save_folder, apartment_id, "top_down.png")):
            continue

        try:
            controller = Controller(scene=house_json, width=800, height=800) #, renderInstanceSegmentation=True, visibilityDistance=30)
        except:
            print("Cannot render environment, continuing")
            # pdb.set_trace()
            continue
        
        
        
        im_name = all_imgs[0]

        apartment_id = im_name.split("/")[-2]
        top_down_im_name = os.path.join(image_save_folder, apartment_id, "top_down.png")
        # pdb.set_trace()
        try:
            # top_down_frame, top_down_seg_frame = get_top_down_frame(controller)
            top_down_frame = get_top_down_frame(controller)
            top_down_frame.save(top_down_im_name)
            # top_down_seg_frame.save(top_down_im_name.replace(".png", "_seg.png"))
        except:
            print("Cannot get top down frame, continuing")
            controller.stop()
            continue
        # pdb.set_trace()

        # all_top_down_ims.append((apartment_id, top_down_im_name))
        controller.stop()

        # json.dump(all_top_down_ims, open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/top_down_im_names.json", "w"))
