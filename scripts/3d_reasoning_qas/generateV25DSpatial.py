from collections import defaultdict
import json
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import os
import random
import tqdm
import shutil
from PIL import ImageDraw, ImageFont
from generateGQASpatial import generateQAfromAnn, add_red_dot_with_text, add_box_dot_with_color
import csv


if __name__=="__main__":

    qa_json_path = "/projectnb/ivc-ml/array/data/VRD25/data/qa_data.json"

    # make ann data to feed into the generateQA function

    # load the ann data
    vrd_path = "/projectnb/ivc-ml/array/data/VRD25/data"
    image_path = "/projectnb/ivc-ml/array/data/VRD25/images"
    
    # read object class descriptiosn csv as list
    obj_desc = {}
    with open(os.path.join(vrd_path, "oidv6-class-descriptions.csv"), 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            obj_desc[row[0]] = row[1]


    # read object csv file as list
    object_data = []
    with open(os.path.join(vrd_path, "within_image_objects_train.csv"), 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            object_data.append(row)

    vrd_data = []
    with open(os.path.join(vrd_path, "within_image_vrd_train.csv"), 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            vrd_data.append(row)

    imid2obj = defaultdict(dict)
    for entry in object_data[1:]:
        # image_id object_id	entity	xmin	xmax	ymin	ymax are entries
        obj_name= obj_desc[entry[2]]
        imid2obj[entry[0]][object_id] = [obj_name, entry[3], entry[4], entry[5], entry[6]])
        

    obj_data= defaultdict(list)
    all_obj_count = defaultdict(lambda: defaultdict(int))
    for entry in vrd_data:
        # image_id_1 object_id_1	image_id_2	object_id_2	distance	occlusion	raw_distance	raw_occlusion

        image_path = os.path.join(vrd_path, "images", entry[0]+".jpg")
        if not os.path.exists(image_path):
            continue

        obj_details = imid2obj[entry[0]][entry[1]]

        obj_desc = obj_details[0]
        obj_pos_x = obj_details[1]
        obj_pos_y = obj_details[3]
        obj_height = obj_details[4] - obj_details[3]
        obj_height = obj_details[2] - obj_details[1]

        depth = entry[4]
        
        obj_data[image_path].append([obj_desc, obj_pos_x, obj_pos_y, obj_width, obj_height, depth])
        
        all_obj_count[image_path][obj_desc] += 1

    # generate the QA
    im_qas_pairs = []
    data_count = 0
    for image_path, obj_list in obj_data.items():
        obj_count = all_obj_count[image_path]
        image = Image.open(image_path)
        imid = image_path.split("/")[-1].split(".")[0]
        mark_AB = random.choice([True, False])

        qa_pairs = generateQAfromAnn(obj_list, obj_count, image, imid, image_path, mark_AB=mark_AB)

        if len(qa_pairs) > 0:
            im_qas_pairs.append([image_file, qa_pairs])
            

        if len(qa_pairs) > 0:
            data_count += 1
            if data_count %10 == 0:
                json.dump(im_qas_pairs, open(qa_json_path, 'w'))