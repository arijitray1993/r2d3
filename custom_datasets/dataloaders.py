import json
import os
import pdb  # noqa
import random
from collections import defaultdict
from itertools import combinations
import pickle as pkl
 
import torch
import tqdm  # noqa
from PIL import Image
from torch.utils.data import Dataset
import time
import torchvision

from torch.utils.data import WeightedRandomSampler
from transformers import Blip2Processor, InstructBlipProcessor # , CodeLlamaTokenizer
from transformers import AutoProcessor

import sys
sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/models/LLaVA")
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.mm_utils import expand2square
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
import csv
from utils.ai2thor_utils import generate_program_from_roomjson

import numpy as np


class ProcTHOR_image_caption(Dataset):
    def __init__(self, args):
        self.args = args
        if args['model'] == "blip2":
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.batch_decode = self.processor.batch_decode
        elif args['model'] == "instructblip":
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.batch_decode = self.processor.batch_decode
        elif args['model'] == "llava":
            self.tokenizer, _, self.image_processor, self.context_len = load_pretrained_model(
                model_path=args['model_path'],
                model_base=None,
                model_name=get_model_name_from_path(args['model_path'])
            )
            self.batch_decode = self.tokenizer.batch_decode
        elif args['model'] == "codellama":
            self.tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
            self.batch_decode = self.tokenizer.batch_decode

        self.json_data = json.load(
            open(
                "/projectnb/ivc-ml/array/research/robotics/dreamworlds/"
                + "custom_datasets/procThor/all_room_json_programs_ai2_train_room_captions_gtobjonly.json", "r"
                )
            )
    
        self.image_data_path = "/projectnb/ivc-ml/array/research/robotics/ProcTHOR/"
        # only use the json data we have images for
        self.data = []
        for program_text, house_json, cam_ind_to_position, all_imgs, all_objs, all_room_captions in self.json_data:
            if not os.path.exists(os.path.join("/projectnb/ivc-ml/array/research/robotics/ProcTHOR/", all_imgs[0])):
                continue
            self.data.append((program_text, house_json, cam_ind_to_position, all_imgs, all_objs, all_room_captions))

        print("Total number of data points: ", len(self.data))

        # split into train and test
        if args["split"] == "train":
            self.data = self.data[:int(len(self.data) * 0.9)]
            self.split = "train"
        elif args["split"] == "val":
            self.data = self.data[int(len(self.data) * 0.9):]
            self.split = "val"


    def __getitem__(self, idx):
        program_text, house_json, cam_ind_to_position, all_imgs, all_objs, all_room_captions = self.data[idx]

        program_text = generate_program_from_roomjson(house_json)

        # choose an image at random to feed in and the caption from another image at random
        try:
            image_path, caption_image_path = random.sample(all_imgs, 2)
        except:
            caption_image_path = all_imgs[0]
            image_path = caption_image_path
        image_path = os.path.join(self.image_data_path, image_path)

        image = Image.open(image_path).convert("RGB")
        
        caption = all_room_captions[all_imgs.index(caption_image_path)]
        # if caption too long take onl first 2 sentences
        caption = ". ".join(caption.split(". ")[:2])
        # pdb.set_trace()
        num_corner = len(all_imgs) # since we have an image from each corner.
        if self.args.get('language_only'):
            prefix_text = f"## \n Write code for an interactive room with {num_corner} corners with the description: {caption}. \n Answer: "
        else:
            prefix_text = f"## <image> \n Write code for an interactive room with {num_corner} corners that looks like the image with the description: {caption}. \n Answer: "
        
        if self.split == "train":
            prompt = prefix_text + program_text
            text_labels = prompt
        else:
            prompt = prefix_text
            text_labels = prefix_text + program_text


        return image, caption, prompt, text_labels, [image_path,], caption_image_path

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        images, captions, prompts, text_labels, image_paths, caption_image_paths = zip(*batch)

        if self.args['model'] == 'llava':
            input_ids = []
            attention_mask = []
            for prompt in prompts:
                # pdb.set_trace()
                input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                input_ids.append(input_id)
                attention_mask.append(torch.ones_like(input_id))
            input_ids = torch.stack(input_ids, dim=0)
            attention_mask = torch.stack(attention_mask, dim=0)
            
            new_images = []
            for image in images:
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)
            
            pixel_values = torch.stack(new_images, dim=0)

            # pixel_values = self.image_processor(images, return_tensors="pt")['pixel_values']
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids,
                "pixel_values": pixel_values,
                "program_texts": prompts,
                "text_labels": text_labels,
                "image_lists": image_paths,
                "caption_images": caption_image_paths,
            }

        return return_dict


class ProcTHOR_Program_image(Dataset):
    def __init__(self, args):
        self.args = args
        if args['model'] == "blip2":
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.batch_decode = self.processor.batch_decode
        elif args['model'] == "instructblip":
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.batch_decode = self.processor.batch_decode
        elif args['model'] == "llava":
            self.tokenizer, _, self.image_processor, self.context_len = load_pretrained_model(
                model_path=args['model_path'],
                model_base=None,
                model_name=get_model_name_from_path(args['model_path'])
            )
            self.batch_decode = self.tokenizer.batch_decode
        elif args['model'] == "codellama":
            self.tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
            self.batch_decode = self.tokenizer.batch_decode

        self.json_data = json.load(
            open(
                "/projectnb/ivc-ml/array/research/robotics/dreamworlds/"
                + "custom_datasets/procThor/all_room_json_programs_ai2_train_room_captions_gtobjonly.json", "r"
                )
            )

        self.image_data_path = "/projectnb/ivc-ml/array/research/robotics/ProcTHOR/"
        # only use the json data we have images for
        self.data = []
        for program_text, house_json, cam_ind_to_position, all_imgs, all_objs, all_room_captions in self.json_data:
            if not os.path.exists(os.path.join("/projectnb/ivc-ml/array/research/robotics/ProcTHOR/", all_imgs[0])):
                continue
            self.data.append((program_text, house_json, cam_ind_to_position, all_imgs, all_objs, all_room_captions))

        print("Total number of data points: ", len(self.data))

        # split into train and test
        if args["split"] == "train":
            self.data = self.data[:int(len(self.data) * 0.9)]
            self.split = "train"
        elif args["split"] == "val":
            self.data = self.data[int(len(self.data) * 0.9):]
            self.split = "val"

        self.tile_images = args['tile_images']

    def __getitem__(self, idx):
        program_text, house_json, cam_ind_to_position, all_imgs, all_objs, all_room_captions = self.data[idx]

        # generate compact representation to fit in context length
        program_text = generate_program_from_roomjson(house_json)
        # pdb.set_trace()
        
        if self.tile_images:
            images = [Image.open(os.path.join(self.image_data_path, image_path)) for image_path in all_imgs]
        
            # Calculate total width and maximum height
            total_width = sum(image.width for image in images)
            max_height = max(image.height for image in images)

            # Create a new blank image with the correct size
            new_image = Image.new('RGB', (total_width, max_height))

            # Paste images into the new image
            x_offset = 0
            for image in images:
                new_image.paste(image, (x_offset, 0))
                x_offset += image.width
            
            image = new_image
        else:
            image_path = os.path.join(self.image_data_path, all_imgs[0])
            image = Image.open(image_path).convert("RGB")

        if self.args['model'] == "codellama":
            env_desc = ""
            prompt_text = f"## Write code: Can you write yaml for a 3D environment {env_desc} \n Answer: "
        else:
            prompt_text = "## Write code: <image> \n Can you write yaml for a 3D environment that looks like these images? \n Answer: "

        if self.split == "train":
            program_text = prompt_text + program_text
            text_labels = program_text
        else:
            text_labels = prompt_text + program_text
            program_text = prompt_text
            
        return image, program_text, text_labels, [os.path.join(self.image_data_path, image_path) for image_path in all_imgs]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        images, program_texts, text_labels, image_lists = zip(*batch)
        # pdb.set_trace()
        if self.args['model'] == 'llava':
            input_ids = []
            attention_mask = []
            for prompt in program_texts:
                input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                input_ids.append(input_id)
                attention_mask.append(torch.ones_like(input_id))
            input_ids = torch.stack(input_ids, dim=0)
            attention_mask = torch.stack(attention_mask, dim=0)
            new_images = []
            for image in images:
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)
            
            pixel_values = torch.stack(new_images, dim=0)

            # pixel_values = self.image_processor(images, return_tensors="pt")['pixel_values']
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids,
                "pixel_values": pixel_values,
                "program_texts": program_texts,
                "text_labels": text_labels,
                "image_lists": image_lists,
            }
        elif self.args['model'] == 'codellama':
            inputs = self.tokenizer(program_texts, return_tensors="pt", padding=True)
            return_dict = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": inputs.input_ids,
                "program_texts": program_texts,
                "text_labels": text_labels,
                "image_lists": image_lists,
            }
        else:
            inputs = self.processor(images=images, text=program_texts, return_tensors="pt", padding=True).to(torch.float16)
            # pdb.set_trace()
            return_dict = {
                "qformer_input_ids": inputs.qformer_input_ids,
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": inputs.input_ids,
                "pixel_values": inputs.pixel_values,
                "program_texts": program_texts,
                "text_labels": text_labels,
                "image_lists": image_lists,
            }

        return return_dict
    
