import json

data_files = [
    "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/depth_reasoning_qas.json",
    "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/depth_reasoning_qas_split2.json",
    "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/depth_reasoning_qas_split3.json" 
]

all_data = []
done_ap_im = {}
for data_file in data_files:
    data = json.load(open(data_file))

    for entry in data:
        ap_id, im_name, ques = entry

        if ap_id + "_" + im_name in done_ap_im:
            continue

        all_data.append(entry)

        done_ap_im[ap_id + "_" + im_name] = True

print(len(all_data))
json.dump(all_data, open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/depth_reasoning_qas_all.json", "w"))