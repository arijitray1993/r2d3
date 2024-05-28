import json
import os
import tqdm

if __name__=="__main__":

    data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_train_childrenadded_all_14k.json"))

    caption_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GPT4V_room_descriptions.json"))
    apartment_folder = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/images/train"

    apid2caption = {}
    for apartment_ind, im, cap in caption_data:
        apid2caption[apartment_ind] = cap
    
    final_data= []

    for program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name in tqdm.tqdm(data):
        
        if len(all_imgs) < 1:
            continue

        apartment_ind = all_imgs[0].split("/")[-2]

        if apartment_ind not in apid2caption:
            continue
            
        top_down_im_path = os.path.join(apartment_folder, apartment_ind, "top_down.png")

        if not os.path.exists(top_down_im_path):
            continue

        top_town_feats_path = os.path.join(apartment_folder, apartment_ind, "top_down_feats.pt")

        if not os.path.exists(top_town_feats_path):
            continue

        final_data.append([program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name])

    print("Total number of rooms after filtering: ", len(final_data))
    json.dump(final_data, open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/final_data_neurips.json", "w"))



