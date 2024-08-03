import json
import csv
import os
import shutil
import pdb

asset_file = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/asset_descriptions_all.json"))

print(len(asset_file))

all_data_csv = [
    ['image_url1', 'asset_name1', 'object_class1', 'caption1', 'image_url2', 'asset_name2', 'object_class2', 'caption2', 'image_url3', 'asset_name3', 'object_class3', 'caption3', 'image_url4', 'asset_name4', 'object_class4', 'caption4'],
]

batch = 4
csv_entry = []
for image_file, asset_name, object_class, caption in asset_file:
    
    # copy image to public html folder
    # src = os.path.join("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/all_obj_vis", image_file)

    # dst = os.path.join("/net/cs-nfs/home/grad2/array/public_html/research/r2d3/asset_ims", image_file)

    # shutil.copy(src, dst)

    html_im_path = "https://cs-people.bu.edu/array/research/r2d3/asset_ims/"+ image_file.split("/")[-1]

    csv_entry.extend([html_im_path, asset_name, object_class, caption])
    # pdb.set_trace()
    if len(csv_entry) == batch*4:
        all_data_csv.append(csv_entry)
        csv_entry = []

# pdb.set_trace()

# write to csv
with open(f"asset_descriptions_mturk.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(all_data_csv)


