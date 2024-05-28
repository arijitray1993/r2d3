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



if __name__ == "__main__":
    
    image_program_json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/final_data_neurips.json", "r"))

    image_save_folder = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/images/train"

    all_top_down_ims = []
    maximum_distance = 30
    for ind, (program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name) in enumerate(tqdm.tqdm(image_program_json_data)):
        
        if len(all_imgs) == 0:
            continue

        if ind<2000:
            continue

        apartment_id = all_imgs[0].split("/")[-2]

        try:
            controller = Controller(scene=house_json, width=800, height=800, visiblityDistance=30) #, renderInstanceSegmentation=True, visibilityDistance=30)
        except:
            print("Cannot render environment, continuing")
            # pdb.set_trace()
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

        
        nav_visible_objects = controller.step(
                "GetVisibleObjects",
                maxDistance=maximum_distance,
            ).metadata["actionReturn"]
        
        ## spatial relations and relative depth

        objid2dist = {}
        objid2name = {}
        for obj_entry in controller.last_event.metadata['objects']:
            obj_id = obj_entry['name']
            distance = obj_entry['distance']
            objid2dist[obj_id] = distance
            objid2name[obj_id] = obj_entry['objectType']

        
        pdb.set_trace()




        
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





        pdb.set_trace()

        controller.stop()

       