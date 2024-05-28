import json
import tqdm
import os

json_files = [
    "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/top_down_im_names.json",
    "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/top_down_im_names_split2.json",
    "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/top_down_im_names_split3.json",
    "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/top_down_im_names_split4.json",
    "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/top_down_im_names_split5.json",
#    "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/top_down_im_names_missinginds.json",
]

# join into one json file
all_data = []
for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        all_data.extend(data)

# sort by inds
all_data = sorted(all_data, key=lambda x: x[0])

print("Len of data:", len(all_data))


# find missing data
image_program_json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_train_childrenadded_all.json", "r"))

missing_inds = []
for ind, (program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_ims, color_to_objid, obj_id_to_name) in enumerate(tqdm.tqdm(image_program_json_data)):
    if len(all_imgs) == 0:
        continue
    im_name = all_imgs[0]
    top_down_im_name = im_name.replace(".png", "_topdown.png")
    if not os.path.exists(top_down_im_name):
        missing_inds.append(ind)

with open("top_down_missing_inds.json", "w") as f:
    json.dump(missing_inds, f)

with open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/top_down_im_names_all.json", "w") as f:
    json.dump(all_data, f)
