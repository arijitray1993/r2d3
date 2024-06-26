from transformers import AutoTokenizer, BitsAndBytesConfig
import sys
sys.path.append('../models/LLaVA')
from llava.model import LlavaLlamaForCausalLM
import torch
import json

import os
import requests
from PIL import Image
from io import BytesIO
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer
import tqdm
import yaml

import pdb

def caption_image(image_file, prompt):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    # pdb.set_trace()

    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
      output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2, 
                                  max_new_tokens=700, use_cache=True, stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]
    return output


if __name__=="__main__":
    # run choice
    caption_individual = False
    generate_summary = False
    caption_individual_top_down = True

    model_path = "4bit/llava-v1.5-13b-3GB"
    kwargs = {"device_map": "auto"}
    kwargs['load_in_4bit'] = True
    kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda')
    image_processor = vision_tower.image_processor

    if caption_individual:
        # load the ai2thor apartments, images, and objects dataset
        image_program_json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_train_childrenadded_all.json", "r"))

        all_house_caption_data = []
        start_ind = len(all_house_caption_data)
        for ind, (program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_ims, color_to_objid, obj_id_to_name) in enumerate(tqdm.tqdm(image_program_json_data)):
            if ind < start_ind:
                continue
            all_room_captions = []
            for img, seg_im, objs in zip(all_imgs, all_seg_ims, all_objs):
                
                # use llava to generate a caption for the image given the objects in it
                # first we need to design the right prompt
                object_names = []
                for obj in objs:
                    obj_format = " ".join(obj.split(" ")[:-1])
                    if obj_format == "":
                        continue
                    object_names.append(obj_format)

                if len(object_names) < 2:
                    continue

                object_names = list(set(object_names))

                obj_prompt = f"The image in view contains these objects: {', '.join(object_names)}. So please only use these objects in your description. Avoid using words like RoboTHOR and only use the generic object names."

                prompt = f"Can you please write a caption describing how the room looks like - the objects and how many, the shape, wall color and material, floor material, and how the room feels like based on this image? {obj_prompt} "

                caption = caption_image(os.path.join("/projectnb/ivc-ml/array/research/robotics/ProcTHOR", img), prompt)

                # pdb.set_trace()

                # print(prompt, caption)
                all_room_captions.append((img, seg_im, objs, caption))
                # pdb.set_trace()
        
            all_house_caption_data.append((ind, all_room_captions))

            json.dump(all_house_caption_data, open("../custom_datasets/procThor/all_room_json_programs_ai2_train_room_captions_childrenadded_gtobjonly_new.json", "w"))

    if caption_individual_top_down:
        json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_train_childrenadded_all.json"))

        # have only the json data we have top down for
        image_program_data = []
        for room_data in json_data:
            all_imgs = room_data[4]
            if len(all_imgs) < 1:
                continue
            if not os.path.exists(all_imgs[0]):
                continue
            if not os.path.exists(all_imgs[0].replace(".png", "_topdown.png")):
                continue
            if not os.path.exists(all_imgs[0].replace(".png", "_topdown_seg.png")):
                continue
            image_program_data.append(room_data)

        all_house_caption_data = []
        for ind, room_data in enumerate(tqdm.tqdm(image_program_data)):
            program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name = room_data

            top_down_img = all_imgs[0].replace(".png", "_topdown.png")

            unique_objs = []
            for obj_list in all_objs:
                
                unique_objs.extend(obj_list)

            unique_objs = list(set(unique_objs))

            house_dict = yaml.load(program_text, Loader=yaml.FullLoader)

            floor_material = house_dict['floor_material']
            wall_material = house_dict['wall_material'][0]

            obj_prompt = f"The image in view contains these objects: {', '.join(unique_objs)}. Please only use these objects in your description. Avoid using words like RoboTHOR and only use the generic object names."

            prompt = f"This is a top-down image. Can you please write a caption describing how the room looks like - the objects and how many, the wall material and color, and the floor material based on the top-down image? {obj_prompt} "

            caption = caption_image(top_down_img, prompt)

            all_house_caption_data.append((ind, top_down_img, house_json, caption))
        
            json.dump(all_house_caption_data, open("../custom_datasets/procThor/all_room_json_programs_ai2_train_room_captions_topdown.json", "w"))


    if generate_summary:
        # load the captions data
        room_image_caption_data = json.load(open("../custom_datasets/procThor/all_room_json_programs_ai2_train_room_captions_gtobjonly.json", "r"))

        # generate a summary caption for each room based on the captions of the various views. 
        all_room_summaries = []
        for program_text, house_json, cam_ind_to_position, all_imgs, all_objs, all_room_captions in tqdm.tqdm(room_image_caption_data):
            
            prompt = f"Can you generate a short summary caption of the room based on these descriptions of images taken from various viewpoints in the room?"
            all_view_point_captions = "\n".join(all_room_captions)

            prompt = prompt + all_view_point_captions + "\n Answer: "

            caption = caption_image(os.path.join("/projectnb/ivc-ml/array/research/robotics/ProcTHOR", all_imgs[0]), prompt)

            # pdb.set_trace()

            all_room_summaries.append((program_text, house_json, caption))

            json.dump(all_room_summaries, open("../custom_datasets/procThor/all_room_json_programs_ai2_train_room_summaries_gtobjonly.json", "w"))
