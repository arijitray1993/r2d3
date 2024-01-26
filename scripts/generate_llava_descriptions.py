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
                                  max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]
    return output


if __name__=="__main__":
    # run choice
    caption_individual = False
    generate_summary = True


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
        image_program_json_data = json.load(open("../custom_datasets/procThor/all_room_json_programs_ai2_train_room_descriptions.json", "r"))

        all_house_caption_data = []
        for program_text, house_json, cam_ind_to_position, all_imgs, all_objs in tqdm.tqdm(image_program_json_data):
            
            all_room_captions = []
            for img, objs in zip(all_imgs, all_objs):
                
                # use llava to generate a caption for the image given the objects in it
                # first we need to design the right prompt

                object_names = [obj[0] for obj in objs]

                obj_prompt = f"The image in view contains only these objects: {', '.join(object_names)}. So please only use these objects in your description if you need to and not any other objects."

                prompt = f"Can you please write a caption describing how the room looks like and feels like based on this image? {obj_prompt} "


                caption = caption_image(os.path.join("/projectnb/ivc-ml/array/research/robotics/ProcTHOR", img), prompt)

                # print(prompt, caption)
                all_room_captions.append(caption)
        
            all_house_caption_data.append((program_text, house_json, cam_ind_to_position, all_imgs, all_objs, all_room_captions))

            json.dump(all_house_caption_data, open("../custom_datasets/procThor/all_room_json_programs_ai2_train_room_captions_gtobjonly.json", "w"))

    if generate_summary:
        # load the captions data
        room_image_caption_data = json.load(open("../custom_datasets/procThor/all_room_json_programs_ai2_train_room_captions_gtobjonly.json", "r"))

        # generate a summary caption for each room based on the captions of the various views. 
        all_room_summaries = []
        for program_text, house_json, cam_ind_to_position, all_imgs, all_objs, all_room_captions in room_image_caption_data:
            
            prompt = f"Can you generate a short summary caption of the room based on these descriptions of images taken from various viewpoints in the room?"
            all_view_point_captions = "\n".join(all_room_captions)

            prompt = prompt + all_view_point_captions + "\n Answer: "

            caption = caption_image(os.path.join("/projectnb/ivc-ml/array/research/robotics/ProcTHOR", all_imgs[0]), prompt)

            all_room_summaries.append((program_text, house_json, caption))
        
        json.dump(all_room_summaries, open("../custom_datasets/procThor/all_room_json_programs_ai2_train_room_summaries_gtobjonly.json", "w"))
