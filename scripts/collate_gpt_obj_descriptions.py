import json
import pdb
from PIL import Image
import tqdm

obj_desc_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/asset_descriptions.json"))

obj_desc_missing_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/asset_descriptions_missingobjs.json"))

obj_desc_filtered = []
for image_file, asset_name, object_class, caption in obj_desc_data:
    if "sorry" not in caption.lower():
        obj_desc_filtered.append((image_file, asset_name, object_class, caption))

all_obj_desc = obj_desc_filtered + obj_desc_missing_data

print(len(all_obj_desc))

all_new_data = []
for entry in tqdm.tqdm(all_obj_desc):
    image_file, asset_name, object_class, caption = entry

    if caption == object_class:
        Image.open(image_file).save("test.png")
        print(asset_name)
        print(object_class)
        
        # request user input
        caption = input("Input caption: ")
        
    
    all_new_data.append((image_file, asset_name, object_class, caption))

json.dump(all_new_data, open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/asset_descriptions_all.json", "w"))