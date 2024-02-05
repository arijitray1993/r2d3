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
        room_json_text = "\n".join(room_response.split("\n")[:-1])
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")

        # pdb.set_trace()
        try:
            house_json = ai2thor_utils.make_house_from_cfg(room_json_text)
        except:
            print("no room json found")
            continue
        controller = Controller(scene=house_json.house_json, width=800,height=800)
        
        controller.step(dict(action='Initialize'))

        img = Image.fromarray(controller.last_event.frame)
        img.save(f"vis/example_{room_ind}.png")


if __name__=="__main__":

    pass