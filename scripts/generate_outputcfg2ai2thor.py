import prior
from ai2thor.controller import Controller
from PIL import Image
import random
from pprint import pprint
import json
import pdb
import math
import os
import copy
import tqdm
from ai2thor.platform import CloudRendering

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import yaml
import ast

import sys
sys.path.append("../")
import utils.ai2thor_utils as ai2thor_utils


def generate_output_houses(json_path):

    output_json = json.load(open(json_path, 'r'))
    # this is list of raw text, each for a room gen 

    response_rooms = [out_text.split("\n Answer: \n")[-1] for out_text in output_json]
    
    for room_ind, room_response in enumerate(response_rooms):
        room_cfg_text = "\n".join(room_response.split("\n")[:-1])
        room_cfg_text = room_cfg_text.replace("(", "[")
        room_cfg_text = room_cfg_text.replace(")", "]")

        # pdb.set_trace()
        try:
            house_json = ai2thor_utils.make_house_from_cfg(room_cfg_text)
        except:
            print("no room json found")
            continue

        try:
            controller = Controller(scene=house_json.house_json, width=800,height=800)
        except:
            print("Cannot render environment")
            continue

        try:
            controller.step(dict(action='Initialize'))
        except:
            print("Cannot initialize environment")
            continue

        img = Image.fromarray(controller.last_event.frame)

        exp_path = "/".join(json_path.split("/")[:-1])
        os.makedirs(f"{exp_path}/vis", exist_ok=True)

        im_path = os.path.join(exp_path, "vis", f"example_{room_ind}.png")

        img.save(im_path)



if __name__=="__main__":

    generate_output_houses("/projectnb/ivc-ml/array/research/robotics/dreamworlds/checkpoints/llava_imonly_lsunbedroom/output.json")

    playground = False
    if playground:

        ### Example GT config.
        cfg = """
polygon: [[30, 0, 30], [30, 0, 45], [45, 0, 45], [45, 0, 30]]
floor_material: TexturesCom_WoodFine0038_1_seamless_S_dark
wall_material: ['PureWhite', 'PureWhite', 'PureWhite', 'PureWhite']

obj_0: ('Toilet_1', (43, 2, 31), (0, 269, 0))

obj_1: ('Sink_19', (31, 2, 44), (0, 88, 0))

obj_2: ('bin_7', (31, 1, 37), (0, 88, 0))
    """

        ### example generated config
        cfg="""
polygon: [[0, 0, 0], [0, 0, 14], [22, 0, 14], [22, 0, 7], [14, 0, 7], [14, 0, 0]]
floor_material: MediumWoodSmooth
wall_material: ['EggshellDrywall', 'EggshellDrywall', 'EggshellDrywall', 'EggshellDrywall', 'EggshellDrywall', 'EggshellDrywall']

obj_0: ('Countertop_C_8x6', (11, 1, 4), (0, 269, 0))

obj_1: ('Fridge_10', (20, 3, 11), (0, 269, 0))

obj_2: ('Dining_Table_201_1', (8, 1, 13), (0, 0, 0))

obj_3: ('Chair_303_1', (8, 1, 11), (0, 19, 0))
    """


        house_json = ai2thor_utils.make_house_from_cfg(cfg)

        #pdb.set_trace()
        #try:
        controller = Controller(scene=house_json.house_json, width=800,height=800) #), renderInstanceSegmentation=True)
        #except:
        #    print("Cannot render environment")
        #    exit()

        img = Image.fromarray(controller.last_event.frame)
        img.save(f"vis/example_.png")

        pdb.set_trace()