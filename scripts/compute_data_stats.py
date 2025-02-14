import json
from collections import defaultdict
import numpy as np
import os
import sys
sys.path.append("../")
from utils.ai2thor_utils import generate_program_from_roomjson
import yaml
import pdb
import tqdm
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS

import re
def get_object_class_from_asset(obj):
    pattern = r'[0-9]'
    obj_name = obj.replace("_", " ")
    obj_name = re.sub(pattern, '', obj_name)

    obj_name = obj_name.replace("RoboTHOR", "")

    obj_name = obj_name.replace("jokkmokk", " ")

    if "Countertop" in obj_name:
        shape = obj_name.split(" ")[-1]
        if shape =="I":
            obj_name = "I-shaped countertop"
        elif shape =="L":
            obj_name = "L-shaped countertop"
        elif shape =="C":
            obj_name = "C-shaped countertop"

    obj_name = obj_name.strip()

    return obj_name


if __name__=="__main__":

    data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/final_data_neurips.json"))

    asset_desc_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/asset_descriptions_all.json"))

    print("Number of rooms: ", len(data))

    avg_num_major_obj = []
    avg_num_total_obj = []
    avg_num_children = []
    avg_num_images = []
    all_room_types = []
    avg_num_corners = []

    all_objs = defaultdict(int)
    all_obj_assets = {}
    all_window_assets = {}
    all_wall_floor_materials = {}
    for entry in tqdm.tqdm(data):

        program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs_im, all_seg_frames, color_to_objid, obj_id_to_name = entry

        num_images = len(all_imgs)
        # num_objects = len(all_objs)

        
        # pdb.set_trace()
        num_corners = len(house_json['rooms'][0]['floorPolygon'])
        avg_num_corners.append(num_corners)
        program_text = generate_program_from_roomjson(house_json, include_children=True)
    
        
        cfg_dict = yaml.load(program_text, Loader=yaml.FullLoader)
        all_children = []
        all_child_classes = []
        i = 0
        while(True):
            if f'child_{i}' in cfg_dict:
                child = cfg_dict[f'child_{i}']
                ch_assetId = child[0]
                all_children.append(ch_assetId)
                all_child_classes.append(get_object_class_from_asset(ch_assetId))
                i += 1
            else:
                break
        
        all_objects = []
        all_obj_classes = []
        i = 0
        while(True):
            if f'obj_{i}' in cfg_dict:
                obj = cfg_dict[f'obj_{i}']
                obj_asset = obj[0]
                all_objects.append(obj_asset)
                all_obj_classes.append(get_object_class_from_asset(obj_asset))
                i += 1
            else:
                break
        
        window_assets = []
        i=0
        while(True):
            if f'window_{i}' in cfg_dict:
                window = cfg_dict[f'window_{i}']
                window_asset = window[0]
                window_assets.append(window_asset)
                i += 1
            else:
                break
        
        wall_material = cfg_dict['wall_material']
        floor_material = cfg_dict['floor_material']
        for wall_m in wall_material:
            all_wall_floor_materials[wall_m] = 1
        all_wall_floor_materials[floor_material] = 1
        
        for obj in all_objects:
                all_obj_assets[obj] = 1
        
        #for obj in all_children:
        #        all_obj_assets[obj] = 1

        for obj_cls in all_obj_classes:
            all_objs[obj_cls] += 1
        
        for obj_cls in all_child_classes:
            all_objs[obj_cls] += 1

        for window in window_assets:
            all_window_assets[window] = 1



        num_children = len(all_children)
        num_objects = len(all_objects)
        total_num_objects = num_objects + num_children

        # estimate room type

        if 'toilet' in program_text.lower():
            room_type = 'bathroom'
        elif 'fridge' in program_text.lower():
            room_type = 'kitchen'
        elif 'couch' in program_text.lower():
            room_type = 'living_room'
        elif 'bed' in program_text.lower():
            room_type = 'bedroom'
        else:
            room_type = 'other'

        avg_num_major_obj.append(num_objects)
        avg_num_total_obj.append(total_num_objects)
        avg_num_children.append(num_children)
        avg_num_images.append(num_images)
        all_room_types.append(room_type)

    print("Average number of major objects: ", np.mean(avg_num_major_obj))
    print("Average number of total objects: ", np.mean(avg_num_total_obj))
    print("Average number of children: ", np.mean(avg_num_children))
    print("Average number of images: ", np.mean(avg_num_images))

    # print room type histogram
    room_type_counts = {}
    for room_type in all_room_types:
        if room_type in room_type_counts:
            room_type_counts[room_type] += 1
        else:
            room_type_counts[room_type] = 1

    print("Room type counts: ", room_type_counts)    

    print("Average number of corners: ", np.mean(avg_num_corners))  

    print("number of object assets: ", len(all_obj_assets))

    print("number of object classes: ", len(all_objs))

    print("number of window assets: ", len(all_window_assets))

    print("number of wall/floor materials: ", len(all_wall_floor_materials))

    # make a histogram of object classes using pyplot
    obj_counts = []
    obj_classes = []
    for obj, count in all_objs.items():
        obj_counts.append(count)
        obj_classes.append(obj)
    
    sorted_indices = np.argsort(obj_counts)[::-1]
    obj_counts = np.array(obj_counts)
    obj_classes = np.array(obj_classes)
    obj_counts = obj_counts[sorted_indices]
    obj_classes = obj_classes[sorted_indices]

    plt.figure(figsize=(30, 30))
    plt.bar(obj_classes, obj_counts)
    plt.xticks(rotation=90, fontsize=7)
    plt.set_yscale('log')
    plt.savefig("object_class_histogram.pdf")


    asset_words = []
    for entry in asset_desc_data:
        asset_desc = entry[-1].lower()
        
        desc_words = asset_desc.split(" ")

        asset_words.extend(desc_words)

    asset_words = " ".join(asset_words)

    # make a word cloud
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = ["a", "A", "the", "on", "in", "and", "of", "with"],
                min_font_size = 10).generate(asset_words)
 
    # plot the WordCloud image                       
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    
    plt.savefig("asset_wordcloud.pdf")
