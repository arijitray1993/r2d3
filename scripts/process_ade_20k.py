from collections import defaultdict
import json
from PIL import Image
import numpy as np
import os
import pdb
import re
from sentence_transformers import SentenceTransformer
import torch
import tqdm

if __name__=="__main__":

    # get all images in validation/home_or_hotel

    image_path = "/projectnb/ivc-ml/array/data/ADE20K_2021_17_01/images/ADE/validation/home_or_hotel"

    
    print("getting all ade objects")
    all_possible_objects = set()
    obj_to_color = dict()
    for folders in os.listdir(image_path):
        # check if folder
        if "." in folders:
            continue
        folder_path = os.path.join(image_path, folders)
        for image in os.listdir(folder_path):
            if "_seg" not in image:
                continue
            
            imfile = os.path.join(folder_path, image)
            img = Image.open(imfile)
            seg = np.array(img)

            # load the corresponding json
            json_path = imfile.replace("_seg.png", ".json")
            im_json = json.load(open(json_path, "r"))

            # Convert seg_im to object class masks
            R = seg[:,:,0]
            G = seg[:,:,1]
            B = seg[:,:,2]
            ObjectClassMasks = (R/10).astype(np.int32)*256 + (G.astype(np.int32))

            # get class id to object in json
            obj_id_to_name = {}
            for obj_entry in im_json['annotation']['object']:
                obj_id_to_name[obj_entry['name_ndx']] = obj_entry['name']
            
            for unique_value in np.unique(ObjectClassMasks):
                inds = np.where(ObjectClassMasks == unique_value)
                rgb_value = (int(R[inds[0][0], inds[1][0]]), int(G[inds[0][0], inds[1][0]]), int(B[inds[0][0], inds[1][0]]))

                if unique_value in obj_id_to_name:
                    obj_name = obj_id_to_name[unique_value]
                    obj_to_color[obj_name] = rgb_value
                    all_possible_objects.add(obj_name)
                    # pdb.set_trace()

            # all_possible_objects.update(set(obj_id_to_name.values()))
    
    # get all ai2thor objects
    print("getting all ai2thor objects")
    ai2thor_objs = json.load(open("/projectnb/ivc-ml/array/research/robotics/ProcTHOR/all_objects.json"))

    all_ai2_objs = defaultdict(list)
    for ai2_obj in ai2thor_objs:
        obj_name = " ".join(ai2_obj.split("_")[:-1])
        obj_name = re.sub(r'[0-9]', '', obj_name)

        if 'RoboTHOR' in obj_name:
            obj_name = obj_name.split(" ")[1]

        obj_name = obj_name.lower().strip()
        all_ai2_objs[obj_name].append(ai2_obj)

    ai2assetname_to_objname = defaultdict(list)
    for obj_name in all_ai2_objs:
        for asset_name in all_ai2_objs[obj_name]:
            ai2assetname_to_objname[asset_name].append(obj_name)
    
    
    model = SentenceTransformer('sentence-transformers/sentence-t5-base')
    # map closest all_possible_objects to ai2thor objects
    print("mapping closest ade objects to ai2thor objects")

    ai2obj_to_embed = {}
    for ai2_obj in all_ai2_objs:
        ai2_obj_emb = model.encode([ai2_obj])
        ai2obj_to_embed[ai2_obj] = ai2_obj_emb

    adeobj_to_embed = {}
    for obj in all_possible_objects:
        ade_obj_emb = model.encode([obj])
        adeobj_to_embed[obj] = ade_obj_emb

    # get closest ade object for each ai2thor object
    closest_ai2thor_to_ade = defaultdict(list)
    for ai2_obj in all_ai2_objs:

        ai2_obj_emb = ai2obj_to_embed[ai2_obj]
        closest = []
        for obj in all_possible_objects:  
            ade_obj_emb = adeobj_to_embed[obj]
            sim = np.mean((ade_obj_emb - ai2_obj_emb)**2)
            closest.append((obj, sim.item()))
        

        closest.sort(key=lambda x: x[1])
        closest_ai2thor_to_ade[ai2_obj] = closest[:10]

    pdb.set_trace()

    # save to json
    with open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/ade_to_ai2thor.json", "w") as f:
        json.dump([closest_ai2thor_to_ade, ai2assetname_to_objname, obj_to_color], f)
    
    pdb.set_trace()