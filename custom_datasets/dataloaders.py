import json
import os
import pdb  # noqa
import random
from collections import defaultdict
import itertools
from itertools import combinations
import functools
import pickle as pkl
import requests
 
import torch
import tqdm  # noqa
from PIL import Image
from torch.utils.data import Dataset
import time
import torchvision

from torch.utils.data import WeightedRandomSampler
from transformers import Blip2Processor, InstructBlipProcessor # , CodeLlamaTokenizer
from transformers import AutoProcessor

from shapely.geometry.polygon import Polygon

from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import yaml
import numpy as np
import h5py
import math

import ast
import cv2
import wandb
from numpy.random import choice
import sys
# sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/models/LLaVA")
sys.path.append("/projectnb/ivc-ml/array/research/robotics/LLaVA")
# sys.path.append("models/LLaVA_modified/LLaVA")
#NOLINT

try:
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
except:
    pass
import csv
from utils.ai2thor_utils import generate_program_from_roomjson, format_program, generate_attribute_program_from_roomjson

import numpy as np

from custom_datasets.embodied_ai_datasets import *
from custom_datasets.d3_datasets import *

try:
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
except:
    pass


def stich_image(images):
    total_width = sum(image.width for image in images)
    max_height = max(image.height for image in images)

    # Create a new blank image with the correct size
    new_image = Image.new('RGB', (total_width, max_height))

    # Paste images into the new image
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width

    return new_image

def add_red_dot_with_text(image, position, text):
    if position[0] is None:
        return image
    # Load the image
    draw = ImageDraw.Draw(image)

    # Coordinates and radius of the dot
    x, y = position
    radius = 10  # You can adjust the size of the dot

    # Draw the red dot
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='red')

    # Load a font (optional, comment out if not needed)
    try:
        font = ImageFont.truetype("arial.ttf", 8)  # Adjust font and size as needed
    except IOError:
        font = ImageFont.load_default()

    # Calculate text width and height to center it
    text_width = draw.textlength(text, font=font)
    text_x = x - text_width / 2
    text_y = y

    # Draw the text
    draw.text((text_x, text_y), text, fill='white', font=font)

    return image

def convert_panoptic_to_ade_colors(segmentation_frame, color_to_objid, obj_id_to_name, ai2assetname_to_objname, closest_ai2thor_to_ade, ade_obj_to_color):
    all_colors = segmentation_frame.reshape(-1, 3)
    unique_colors = set([tuple(color) for color in all_colors])

    for color in unique_colors:
        if str(color) in color_to_objid:
            ai2_objid = color_to_objid[str(color)]
            if "exterior" in ai2_objid:
                closest_ade_obj = "wall"
            elif "room" in ai2_objid:
                closest_ade_obj = "floor, flooring"
            elif "Ceiling" in ai2_objid or "ceiling" in ai2_objid:
                closest_ade_obj = "ceiling"
            elif ai2_objid.isnumeric():
                closest_ade_obj = "wall"
            elif "window" in ai2_objid:
                closest_ade_obj = "windowpane, window"
            else:
                ai2_objid_format = ai2_objid.split("___")[0]
                try:
                    ai2_assetname = obj_id_to_name[ai2_objid_format]
                    obj_name = ai2assetname_to_objname[ai2_assetname]
                    closest_ade_obj, distance = closest_ai2thor_to_ade[obj_name[0]][0]
                except:
                    print("error in finding closest ade object")
                    print(ai2_objid)
                    print(ai2_objid_format)
                    print(ai2_assetname)
                    print(obj_name)
        else:
            print("Error in finding color to obj mapping")
            print(color)
            print(color_to_objid)
            closest_ade_obj = "wall"

        try:
            new_color = ade_obj_to_color[closest_ade_obj]
        except:
            print(closest_ade_obj)

        # repeat new colors the number of times color appears
        new_color = np.array(new_color)
        new_color = np.tile(new_color, (np.sum(np.all(all_colors == color, axis=1)), 1))
    
        all_colors[np.all(all_colors == color, axis=1)] = new_color
    
    new_segmentation_frame = all_colors.reshape(segmentation_frame.shape)

    return new_segmentation_frame


def color_instance_specific(segmentation_frames, color_to_obj_id):

    new_segmentation_frames = []

    # get all the objids and wall ids present in all the segmentation frames
    all_obj_ids = set()

    for seg_frame in segmentation_frames:
        seg_frame = np.array(seg_frame)
        unique_colors = set([tuple(color) for color in seg_frame.reshape(-1, 3)])
        for color in unique_colors:
            if str(color) in color_to_obj_id:
                obj_id = color_to_obj_id[str(color)]
                all_obj_ids.add(obj_id)

    # assign a unique color to each object id
    obj_id_to_color = {}
    for obj_id in all_obj_ids:
        obj_id_to_color[obj_id] = np.random.randint(0, 255, 3)

    # assign the new colors to the segmentation frames
    for seg_frame in segmentation_frames:
        seg_frame = np.array(seg_frame)
        new_seg_frame = np.zeros_like(seg_frame)
        unique_colors = set([tuple(color) for color in seg_frame.reshape(-1, 3)])
        for color in unique_colors:
            if str(color) in color_to_obj_id:
                obj_id = color_to_obj_id[str(color)]
                new_color = obj_id_to_color[obj_id]
                new_seg_frame[np.all(seg_frame == color, axis=2)] = new_color
        new_seg_frame = Image.fromarray(new_seg_frame)
        new_segmentation_frames.append(new_seg_frame)

    return new_segmentation_frames

def mark_objects_instance_specific(segmentation_frames, image_frames):
    # mark the center of each object based on the color in the instance segmentation frame
    marked_images = []
    for seg_frame, image_frame in zip(segmentation_frames, image_frames):
        seg_frame = np.array(seg_frame)
        image_frame = np.array(image_frame)
        unique_colors = set([tuple(color) for color in seg_frame.reshape(-1, 3)])
        for color in unique_colors:
            mask = np.all(seg_frame == color, axis=2)
            if np.sum(mask) == 0:
                continue
            center = np.mean(np.argwhere(mask), axis=0)
            center = tuple(center.astype(int))
            image_frame = cv2.circle(image_frame, center, 5, (0, 0, 255), -1)
        marked_images.append(image_frame)


def normalize_coords(program_text, cam_pos, cam_rot, attr=False):

    # convert prgram text to json using yaml
    room_json = yaml.load(program_text, Loader=yaml.FullLoader)

    room_polygon = room_json['polygon']
    room_polygon_norm = [(point[0] - cam_pos[0], point[2] - cam_pos[2]) for point in room_polygon]
    room_polygon_norm = [(point[0]*np.cos(np.deg2rad(cam_rot)) - point[1]*np.sin(np.deg2rad(cam_rot)), point[0]*np.sin(np.deg2rad(cam_rot)) + point[1]*np.cos(np.deg2rad(cam_rot))) for point in room_polygon_norm]
    room_polygon = [(int(point[0]), 0,  int(point[1])) for point in room_polygon_norm]

    i = 0
    while(True):
        if f'obj_{i}' not in room_json:
            break
        if attr:
            obj_entry = room_json[f'obj_{i}']
            gt_object = obj_entry.split("at location")[0].strip()
            location = ast.literal_eval(obj_entry.split("at location")[-1].split("with rotation")[0].strip())
            rotation = ast.literal_eval(obj_entry.split("with rotation")[-1].strip())
        else:
            gt_object, location, rotation = room_json[f'obj_{i}']
        norm_location = (location[0] - cam_pos[0], location[1], location[2] - cam_pos[2])
        norm_location = (norm_location[0]*np.cos(np.deg2rad(cam_rot)) - norm_location[2]*np.sin(np.deg2rad(cam_rot)), norm_location[0]*np.sin(np.deg2rad(cam_rot)) + norm_location[2]*np.cos(np.deg2rad(cam_rot)))
        # pdb.set_trace()
        norm_rotation = rotation[1] - cam_rot
        if norm_rotation < 0:
            norm_rotation += 360
        room_json[f'obj_{i}'] = (gt_object, (int(norm_location[0]), int(location[1]), int(norm_location[1])), norm_rotation)
        i += 1

    i = 0
    while(True):
        if f'child_{i}' not in room_json:
            break
        if attr:
            child_entry = room_json[f'child_{i}']
            child_obj = child_entry.split("at location")[0].strip()
            location = ast.literal_eval(child_entry.split("at location")[-1].split("with rotation")[0].strip())
            rotation = ast.literal_eval(child_entry.split("with rotation")[-1].strip())
        else:
            child_obj, location, rotation = room_json[f'child_{i}']
        norm_location = (location[0] - cam_pos[0], location[1], location[2] - cam_pos[2])
        norm_location = (norm_location[0]*np.cos(np.deg2rad(cam_rot)) - norm_location[2]*np.sin(np.deg2rad(cam_rot)), norm_location[0]*np.sin(np.deg2rad(cam_rot)) + norm_location[2]*np.cos(np.deg2rad(cam_rot)))
        norm_rotation = rotation[1] - cam_rot
        if norm_rotation < 0:
            norm_rotation += 360
        room_json[f'child_{i}'] = (child_obj, (int(norm_location[0]), int(location[1]), int(norm_location[1])), norm_rotation)
        i += 1
    
    i = 0
    while(True):
        if f'window_{i}' not in room_json:
            break

        window_token, window_position, window_polygon, window_wall = room_json[f'window_{i}']
        window_position_norm = (window_position[0] - cam_pos[0], window_position[1], window_position[2] - cam_pos[2])
        window_position_norm = (window_position_norm[0]*np.cos(np.deg2rad(cam_rot)) - window_position_norm[2]*np.sin(np.deg2rad(cam_rot)), window_position_norm[0]*np.sin(np.deg2rad(cam_rot)) + window_position_norm[2]*np.cos(np.deg2rad(cam_rot)))
        room_json[f'window_{i}'] = (window_token, (int(window_position_norm[0]), int(window_position[1]), int(window_position_norm[1])), window_polygon, window_wall)
        i += 1
    
    # make room_json back to program_text
    program_text = program_text = f"""
polygon: {room_polygon}
floor_material: '{room_json['floor_material']}'
wall_material: {room_json['wall_material']}
"""

    i = 0
    while(True):
        if f'obj_{i}' not in room_json:
            break
        gt_object, location, rotation = room_json[f'obj_{i}']
        if attr:
            program_text += f"""\nobj_{i}: {gt_object} at location {location} with rotation {rotation}"""
        else:
            program_text += f"""\nobj_{i}: [{gt_object}, {location}, {rotation}]"""
        i += 1
    
    i = 0
    while(True):
        if f'child_{i}' not in room_json:
            break
        child_obj, location, rotation = room_json[f'child_{i}']
        if attr:
            program_text += f"""\nchild_{i}: {child_obj} at location {location} with rotation {rotation}"""
        else:
            program_text += f"""\nchild_{i}: [{child_obj}, {location}, {rotation}]"""
        i += 1
    
    i = 0
    while(True):
        if f'window_{i}' not in room_json:
            break
        window_token, window_position, window_polygon, window_wall = room_json[f'window_{i}']
        if attr:
            program_text += f"""\nwindow_{i}: {window_token} at location {window_position} with polygon {window_polygon} on wall {window_wall}"""
        else:
            program_text += f"""\nwindow_{i}: [{window_token}, {window_position}, {window_polygon}, {window_wall}]"""
        i += 1
    

    return program_text


def interleave_iterators(*iterators):
    finished = [False for x in range(len(iterators))]
    stop_cond = functools.reduce(lambda x,y:not x or not y,finished)
    while stop_cond:
        for i,it in enumerate(iterators):
            try:
                yield next(it)
            except StopIteration:
                finished[i] = True
        stop_cond = functools.reduce(lambda x,y:not x or not y,finished)

def format_prompt(image, question, answer, answer_choices, model_choice, mode, **kwargs):
    
    image_order = [image,]
    
    correct_answer = answer_choices[0]
        
    ans_choice_order = answer_choices
    ans_choice_order = ['"'+ans+'"' for ans in ans_choice_order]
    random.shuffle(ans_choice_order)
    answer_choices_format = " or ".join(ans_choice_order)
    
    image_prompt_format = "<image>"*len(image_order)

    if model_choice=="llava":
        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        if mode == "train":
            prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}.###Assistant: \n {correct_answer} \n###"
            text_label = prompt
        else:
            prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}.###Assistant: \n" 
            text_label = prompt + correct_answer + " \n###"
    
    return image_order, images, prompt, text_label, correct_answer, answer_choices, f"procthor_reasoning_{question_type}"


def get_inputs_for_model(imgs, prompts, tokenizer=None, image_processor=None, model_choice=None):

    if model_choice == "llava_ov":
        inputs = image_processor(text=prompts, images=imgs, return_tensors='pt').to(torch.float16)
        return inputs['pixel_values'], inputs['input_ids'], inputs['attention_mask'], inputs['image_sizes']

    if model_choice == "llava":
        new_images = []
        for image_b in imgs:
            for image in image_b:
                image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
                # pdb.set_trace()
                image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)

        pixel_values = torch.stack(new_images, dim=0)

        input_ids = []
        attention_mask = []
        for prompt in prompts:
            # pdb.set_trace()
            input_id = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids.append(input_id)
            attention_mask.append(torch.ones_like(input_id))
        
        # pad with zeros
        # pdb.set_trace()
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return pixel_values, input_ids, attention_mask

class MixLLavaProcthor(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.batch_decode = self.tokenizer.batch_decode

        self.procthor_data = ProcTHOR_image_camposition_marked(args, tokenizer, image_processor)
        self.llava_data = LLaVAInstructTune(args, tokenizer, image_processor)
        self.proc_len = len(self.procthor_data)
        self.llava_len = len(self.llava_data)
        
        # self.data = interleave_iterators(procthor_data, llava_data)

        print("combined data ...")

        print("Total number of data points: ", self.proc_len + self.llava_len)

    def __getitem__(self, idx):
        if random.random() < 0.7:
            return self.procthor_data[idx%self.proc_len]
        else:
            return self.llava_data[idx%self.llava_len]
        
    
    def __len__(self):
        return max(self.proc_len, self.llava_len)

    def collate_fn(self, batch):
        image_paths, imgs, captions, prompts, text_labels, program_texts, house_jsons, objs_present = zip(*batch)

        new_images = []
        for image_b in imgs:
            for image in image_b:
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                # pdb.set_trace()
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)

        pixel_values = torch.stack(new_images, dim=0)

        input_ids = []
        attention_mask = []
        for prompt in prompts:
            # pdb.set_trace()
            input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids.append(input_id)
            attention_mask.append(torch.ones_like(input_id))
        
        # pad with zeros
        # pdb.set_trace()
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        # pdb.set_trace()
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "labels": input_ids,
            "prompts": prompts,
            "text_labels": text_labels,
            "program_texts": program_texts,
            "house_json": house_jsons,
            "image_lists": image_paths,
            'objs_present': objs_present,
        }

        return return_dict
        
class MixLLaVAProcthorReasoning(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode
        
        if args.get("instructBLIP") or args.get("BLIP2"):
            self.batch_decode = self.image_processor.batch_decode

        self.all_mix = []
        self.all_lens = []

        self.procthor_data = ProcTHOR_reasoning(args, tokenizer, image_processor)
        #self.all_mix.append(self.procthor_data)
        self.all_lens.append(len(self.procthor_data))

        self.llava_data = LLaVAInstructTune(args, tokenizer, image_processor)
        #self.all_mix.append(self.llava_data)
        self.all_lens.append(len(self.llava_data))

        if args.get("mix_recon"):
            self.procthor_recon = ProcTHOR_image_camposition_marked(args, tokenizer, image_processor)
            #self.all_mix.append(self.procthor_recon)
            self.all_lens.append(len(self.procthor_recon))
        
        if args.get("mix_recon_qa"):
            self.procthor_recon_qa = ProcTHOR_recon_qa(args, tokenizer, image_processor)
            #self.all_mix.append(self.procthor_recon_qa)
            self.all_lens.append(len(self.procthor_recon_qa))
        
        if args.get("mix_3dcapqa"):
            self.procthor_3dcapqa = ProcTHOR_3DCaptions(args, tokenizer, image_processor)
            #self.all_mix.append(self.procthor_3dcapqa)
            self.all_lens.append(len(self.procthor_3dcapqa))

        print("combined data ...")

        print("Total number of data points: ", sum(self.all_lens))
    
    def __getitem__(self, idx):
        ran_num = random.random()

        if self.args.get("mix_recon"):
            if ran_num < 0.4:
                return self.procthor_data[idx%self.all_lens[0]]
            elif ran_num < 0.7:
                return self.llava_data[idx%self.all_lens[1]]
            else:
                return self.procthor_recon[idx%self.all_lens[2]]
        elif self.args.get("mix_recon_qa"):
            if ran_num < 0.4:
                return self.procthor_data[idx%self.all_lens[0]]
            elif ran_num < 0.7:
                return self.llava_data[idx%self.all_lens[1]]
            else:
                return self.procthor_recon_qa[idx%self.all_lens[2]]
        elif self.args.get("mix_3dcapqa"):
            if ran_num < 0.3:
                return self.llava_data[idx%self.all_lens[1]]
            else:
                return self.procthor_3dcapqa[idx%self.all_lens[2]]
        else:
            if ran_num < 0.6:
                return self.procthor_data[idx%self.all_lens[0]]
            else:
                return self.llava_data[idx%self.all_lens[1]]
        
    def __len__(self):
        return max(self.all_lens)
    
    def collate_fn(self, batch):
        
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        if self.args.get("instructBLIP"):
            # only works with batch size 1 for now change later. 
            inputs = self.image_processor(images=images_batch[0], text=prompts[0], return_tensors="pt")
            return_dict = {
                "input_ids": inputs["input_ids"][:, :510],
                "attention_mask": inputs["attention_mask"][:, :510],
                'qformer_input_ids': inputs["qformer_input_ids"][:, :510],
                'qformer_attention_mask': inputs["qformer_attention_mask"][:, :510],
                'pixel_values': inputs["pixel_values"],
                "labels": inputs["input_ids"][:, :510],
                "prompts": prompts,
                "text_labels": text_labels,
                "dataset": datanames,
                "answers": answers,
                "answer_choices": answer_choices,
            }
        else:
            new_images = []
            for image_b in images_batch:
                for image in image_b:
                    image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                    # pdb.set_trace()
                    image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    new_images.append(image)

            pixel_values = torch.stack(new_images, dim=0)

            input_ids = []
            attention_mask = []
            for prompt in prompts:
                # pdb.set_trace()
                input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                input_ids.append(input_id)
                attention_mask.append(torch.ones_like(input_id))
            
            # pad with zeros
            # pdb.set_trace()
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
            
            return_dict = {
                "input_ids": input_ids[:, :800],
                "attention_mask": attention_mask[:, :800],
                'pixel_values': pixel_values,
                "labels": input_ids[:, :800],
                "prompts": prompts,
                "text_labels": text_labels,
                "datanames": datanames,
                "answers": answers,
            }

        return return_dict



class CustomMix(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode
        
        if args.get("instructBLIP") or args.get("BLIP2"):
            self.batch_decode = self.image_processor.batch_decode

        self.all_mix = []
        self.all_lens = []
        self.weights = []

        mix_datas = args.get("mix_datas")

        if "llavaIT" in mix_datas:
            self.llava_data = LLaVAInstructTune(args, tokenizer, image_processor)
            self.all_mix.append(self.llava_data)
            self.weights.append(mix_datas["llavaIT"])
            self.all_lens.append(len(self.llava_data))

        if "thor_reasoning" in mix_datas:
            self.procthor_data = ProcTHOR_reasoning(args, tokenizer, image_processor)
            self.all_mix.append(self.procthor_data)
            self.weights.append(mix_datas["thor_reasoning"])
            self.all_lens.append(len(self.procthor_data))

        if "thor_recon" in mix_datas:
            self.procthor_recon = ProcTHOR_image_camposition_marked(args, tokenizer, image_processor)
            self.all_mix.append(self.procthor_recon)
            self.weights.append(mix_datas["thor_recon"])
            self.all_lens.append(len(self.procthor_recon))
        
        if "thor_recon_qa" in mix_datas:
            self.procthor_recon_qa = ProcTHOR_recon_qa(args, tokenizer, image_processor)
            self.all_mix.append(self.procthor_recon_qa)
            self.weights.append(mix_datas["thor_recon_qa"])
            self.all_lens.append(len(self.procthor_recon_qa))
        
        if "thor_3dcapqa" in mix_datas:
            self.procthor_3dcapqa = ProcTHOR_3DCaptions(args, tokenizer, image_processor)
            self.all_mix.append(self.procthor_3dcapqa)
            self.weights.append(mix_datas["thor_3dcapqa"])
            self.all_lens.append(len(self.procthor_3dcapqa))

        if "gqa_spatial" in mix_datas:
            self.gqa_spatial = GQASpatial(args, tokenizer, image_processor)
            self.all_mix.append(self.gqa_spatial)
            self.weights.append(mix_datas["gqa_spatial"])
            self.all_lens.append(len(self.gqa_spatial))
        
        if "VSR_VRD25D" in mix_datas:
            self.vsr_vrd25d = VSR_VRD25D(args, tokenizer, image_processor)
            self.all_mix.append(self.vsr_vrd25d)
            self.weights.append(mix_datas["VSR_VRD25D"])
            self.all_lens.append(len(self.vsr_vrd25d))

        if 'spoc_easyobjnav' in mix_datas:
            self.spoc_easyobjnav = SPOC_data(args, tokenizer, image_processor)
            self.all_mix.append(self.spoc_easyobjnav)
            self.weights.append(mix_datas["spoc_easyobjnav"])
            self.all_lens.append(len(self.spoc_easyobjnav))
        
        if 'robopoint' in mix_datas:
            self.robopoint = RoboPointDataset(args, tokenizer, image_processor)
            self.all_mix.append(self.robopoint)
            self.weights.append(mix_datas["robopoint"])
            self.all_lens.append(len(self.robopoint))
        
        print("combined data ...")

        print("Total number of data points: ", sum(self.all_lens))
    
    def __getitem__(self, idx):
        
        data = random.choices(population=self.all_mix, k=1, weights=self.weights)[0]
        return data[idx%len(data)]

    def __len__(self):
        return max(self.all_lens)
    
    def collate_fn(self, batch):
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        if self.args.get("instructBLIP"):
            # only works with batch size 1 for now change later. 
            inputs = self.image_processor(images=images_batch[0], text=prompts[0], return_tensors="pt")
            return_dict = {
                "input_ids": inputs["input_ids"][:, :510],
                "attention_mask": inputs["attention_mask"][:, :510],
                'qformer_input_ids': inputs["qformer_input_ids"][:, :510],
                'qformer_attention_mask': inputs["qformer_attention_mask"][:, :510],
                'pixel_values': inputs["pixel_values"],
                "labels": inputs["input_ids"][:, :510],
                "prompts": prompts,
                "text_labels": text_labels,
                "dataset": datanames,
                "answers": answers,
                "answer_choices": answer_choices,
            }
        elif self.args.get("llava_ov"):
            
            pixel_values, input_ids, attention_mask, image_sizes =  get_inputs_for_model(images_batch, prompts, None, self.image_processor, model_choice="llava_ov")
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "image_sizes": image_sizes,
            }
        else:
            new_images = []
            for image_b in images_batch:
                for image in image_b:
                    image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                    # pdb.set_trace()
                    image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    new_images.append(image)

            pixel_values = torch.stack(new_images, dim=0)

            input_ids = []
            attention_mask = []
            for prompt in prompts:
                # pdb.set_trace()
                input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                input_ids.append(input_id)
                attention_mask.append(torch.ones_like(input_id))
            
            # pad with zeros
            # pdb.set_trace()
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
            
            return_dict = {
                'image_paths': image_paths,
                "input_ids": input_ids[:, :800],
                "attention_mask": attention_mask[:, :800],
                'pixel_values': pixel_values,
                "labels": input_ids[:, :800],
                "prompts": prompts,
                "text_labels": text_labels,
                "datanames": datanames,
                "answers": answers,
            }

        return return_dict


class VSR_VRD25D(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.batch_decode = self.tokenizer.batch_decode

        data_files = {"train": "train.jsonl",}
        dataset = load_dataset("cambridgeltl/vsr_random", data_files=data_files)
        self.coco_path = "/projectnb/ivc-ml/array/data/COCO/images/"

        vrd_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/vrd_qa_data.json"

        self.data = []
        for entry in dataset['train']:
            image_path = entry["image_link"]
            image_path = image_path.split("/")[-2:]
            image_path = os.path.join(self.coco_path, *image_path)

            caption = entry["caption"].lower()
            relation = entry["relation"].lower()
            label = entry["label"]

            entities = caption.split(relation)

            subject, object = entities[0], entities[1]
            subject = subject.strip().lower().replace("is", "").replace("the", "").replace("are", "")
            object = object.strip().lower().replace("is", "").replace("the", "").replace("are", "")

            question = f"Is {subject} {relation} the {object}?"
            answer = "yes" if label == 1 else "no"
            wrong_answer = "no" if label == 1 else "yes"

            self.data.append((image_path, question, [answer, wrong_answer]))

        vrd25data = json.load(open(vrd_path))
        v25_data = []
        for img, qa_entries in vrd25data:
            for question, answers in qa_entries:
                v25_data.append((img, question, answers))
        
        print("Total number of data points in VSR: ", len(self.data))
        print("Total number of data points in VRD25D: ", len(v25_data))

        self.data += random.sample(v25_data, 100000)

        random.shuffle(self.data)

        if args.get("split") != "train": # we never use this for val reporting, only valtrain. 
            self.data = self.data[int(len(self.data)*0.9):]

        print("Total number of data points in VSR_VRD25D: ", len(self.data))
    
    def __getitem__(self, idx):
        im_file, question, answer = self.data[idx]
        
        correct_answer = answer[0]

        ans_choice_order = ['"'+ans+'"' for ans in answer]
        random.shuffle(ans_choice_order)
        answer_choices_format = " or ".join(ans_choice_order) 

        #if im_file is not None:
        img = [Image.open(im_file).convert("RGB"),]

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        if self.args['mode'] == "train":
            prompt = f"{prefix}###Human: <im_start><image><im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}. ###Assistant: \n {correct_answer} \n###"
            text_labels = prompt
        else:
            prompt = f"{prefix}###Human: <im_start><image><im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}. ###Assistant: \n "
            text_labels = prompt + correct_answer + " \n###"        

        
        return [im_file,], img, prompt, text_labels, correct_answer, answer, "vsr25d_spatial"
    
    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)
        new_images = []
        for image_b in images_batch:
            for image in image_b:
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                # pdb.set_trace()
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)

        pixel_values = torch.stack(new_images, dim=0)

        input_ids = []
        attention_mask = []
        for prompt in prompts:
            # pdb.set_trace()
            input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids.append(input_id)
            attention_mask.append(torch.ones_like(input_id))
        
        # pad with zeros
        # pdb.set_trace()
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        # pdb.set_trace()
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "labels": input_ids,
            "prompts": prompts,
            "text_labels": text_labels,
            "dataset": datanames,
            "answers": answers,
            "answer_choices": answer_choices,
        }

        return return_dict

class GQASpatial_OG_QA(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        gqa_qa = json.load(open("/projectnb/ivc-ml/array/data/GQA/val_balanced_questions.json"))
        gqa_im_path = "/projectnb/ivc-ml/array/data/GQA/images/"

        qa_data = []
        for qaid in gqa_qa:
            entry = gqa_qa[qaid]
            if entry['types']['semantic'] == 'rel':
                img_id = entry["imageId"]
                question = entry["question"]
                answer = entry["answer"]
                image_path = os.path.join(gqa_im_path, f"{img_id}.jpg")

                qa_data.append((image_path, question, answer))
        
        self.data = qa_data[:args.get("num_data_points", 10000)]
    
    def __getitem__(self, idx):
        im_file, question, answer = self.data[idx]

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        prompt = f"{prefix}###Human: <im_start><image><im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. ###Assistant: \n "
        text_labels = prompt + answer + " \n###" 

        img = [Image.open(im_file).convert("RGB"),]       

        return [im_file,], img, prompt, text_labels, answer, [answer,], "gqa_spatial_ogqa"


    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        pixel_values, input_ids, attention_mask =  get_inputs_for_model(images_batch, prompts, self.tokenizer, self.image_processor, model_choice="llava")
        
        # pdb.set_trace()
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "labels": input_ids,
            "prompts": prompts,
            "text_labels": text_labels,
            "dataset": datanames,
            "answers": answers,
            "answer_choices": answer_choices,
        }

        return return_dict



class CocoSpatialDataset(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        """
        Initializes the CocoDataset with the root directory of the dataset and the annotation file.
        
        dataset_root: The root directory where the dataset is stored.
        annotation_file: The path to the COCO annotations file (JSON format).
        autoload: Whether to load the dataset automatically. Defaults to False.
        """
        self.dataset_root = "/projectnb/ivc-ml/array/data/COCO/images/"
        self.annotation_file = "https://github.com/kahnchana/locvlm/releases/download/v1.0/coco_spatial.json"
        self.image_id_list = None
        self.coco_data = None
        self.categories = None
        self.annotations = None
        autoload = True
        if autoload:
            self.load_dataset()

        # make the spatial qas
        self.data = []
        for image_id in self.image_id_list:
            image, annotation = self.get_image_annotations(image_id)
            image = image.convert("RGB")

            spatial_eval_data = self.generate_spatial_questions(image, annotation)
            ab_spatial_eval_data = self.generate_up_down_questions(image, annotation)

            self.data.append((image, spatial_eval_data['questions'][0], spatial_eval_data['answers'][0][0], ["left", "right"]))
            self.data.append((image, ab_spatial_eval_data['questions'][0], ab_spatial_eval_data['answers'][0][0], ["above", "below"]))

        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, index):
        image, question, answer, answer_choices = self.data[index]

        random.shuffle(answer_choices)

        answer_choice_format = " or ".join([f'"{ans}"' for ans in answer_choices])

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        prompt = f"{prefix}###Human: <im_start><image><im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choice_format}. ###Assistant: \n "
        text_labels = prompt + answer + " \n###" 

        img = [image,]       

        return ["",], img, prompt, text_labels, answer, [answer,], "coco_spatial"
    
    def collate_fn(self, batch):
        
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        pixel_values, input_ids, attention_mask =  get_inputs_for_model(images_batch, prompts, self.tokenizer, self.image_processor, model_choice="llava")
        
        # pdb.set_trace()
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "labels": input_ids,
            "prompts": prompts,
            "text_labels": text_labels,
            "dataset": datanames,
            "answers": answers,
            "answer_choices": answer_choices,
        }

        return return_dict

    @property
    def qa_pair_count(self):
        """The total number of question-answer pairs in the dataset."""
        return sum([len(x['good_pairs']) for x in self.annotations.values()])

    def load_dataset(self):
        """
        Loads the COCO dataset annotations from the specified JSON file.
        """
        if self.annotation_file.startswith('https://'):
            response = requests.get(self.annotation_file)
            if response.status_code == 200:
                self.coco_data = response.json()  # This automatically parses the JSON content
            else:
                raise Exception(f"Failed to download JSON file. Status code: {response.status_code}")
        else:
            with open(self.annotation_file, 'r') as f:
                self.coco_data = json.load(f)
        # Create a mapping from category ID to category name
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.annotations = self.coco_data['data']
        self.image_id_list = list(self.annotations.keys())

    def get_image_annotations(self, image_id):
        """
        Retrieves the image and its annotations given an image ID.
        
        image_id: The ID of the image to retrieve.

        Return: 
            A tuple (image, annotations) where `image` is a PIL image object,
            `annotations` is a list of bounding boxes and category IDs for the given image.
        """
        # Find the image information by image ID
        datum = self.annotations[image_id]
        
        image_path = f'{self.dataset_root}/val2014/{datum["file_name"]}'
        image = Image.open(image_path)
        
        # Get the annotations for the given image ID
        annotations = datum['annotations']
        good_pairs = datum['good_pairs'] if 'good_pairs' in datum else None
        data = {'annotation': annotations, 'good_pairs': good_pairs}
        
        return image, data

    def visualize_image(self, image, annotations, font_path=None, font_size=25):
        """
        Visualizes the image by drawing bounding boxes and category labels on it.
        
        :param image: The PIL image object to visualize.
        :param annotations: A list of annotations with bounding boxes and category IDs.
        :param font_path: The path to the font file for rendering text. Defaults to "arial.ttf" if available.
        :param font_size: The font size to use for the text.
        """
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)

        if isinstance(annotations, dict):
            annotations = annotations['annotation']

        # Load the font for text labels
        try:
            font = ImageFont.truetype(font_path or "arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Loop through each annotation and draw the bounding box and label
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, width, height = bbox
            category_id = ann['category_id']
            category_name = self.categories[category_id]

            # Draw the bounding box
            draw.rectangle([x, y, x + width, y + height], outline='green', width=3)

            # Draw the category label
            text_position = (x, y)
            draw.text(text_position, category_name, fill="red", font=font)

        return vis_image

    def generate_spatial_questions(self, image, annotation):
        flipped_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        object_list = [self.categories[x['category_id']] for x in annotation['annotation']]
        object_pairs = annotation['good_pairs']

        question_list = []
        answer_list = []
        for (obj_left, obj_right) in object_pairs:
            name_left = object_list[obj_left]
            name_right = object_list[obj_right]
            question = f"Which side of the {name_left} is the {name_right}?"
            # correct and wrong answers respectively
            answers = [
                f"The {name_right} is on the right side of the {name_left}.",
                f"The {name_right} is on the left side of the {name_left}.",
            ]
            question_list.append(question)
            answer_list.append(answers)
        
        return {
            'image': image,
            'image_flipped': flipped_image,
            'questions': question_list,
            'answers': answer_list
        }

    def generate_up_down_questions(self, image, annotation):
        flipped_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        object_list = [self.categories[x['category_id']] for x in annotation['annotation']]
        object_pairs = annotation['good_pairs']

        question_list = []
        answer_list = []
        for (obj_up, obj_down) in object_pairs:
            name_down = object_list[obj_down]
            name_up = object_list[obj_up]
            question = f"Where is the {name_down} relative to the {name_up}?"
            # Correct and wrong answers respectively.
            answers = [
                f"The {name_down} is below the {name_up}.",
                f"The {name_down} is above the {name_up}.",
            ]
            question_list.append(question)
            answer_list.append(answers)

            question_reversed = f"Where is the {name_up} relative to the {name_down}?"
            # Correct and wrong answers respectively.
            answers = [
                f"The {name_up} is above the {name_down}.",
                f"The {name_up} is below the {name_down}.",
            ]

            question_list.append(question_reversed)
            answer_list.append(answers)

        return {
            'image': image,
            'image_flipped': flipped_image,
            'questions': question_list,
            'answers': answer_list
        }


    def generate_object_questions(self, annotation):
        object_list = [self.categories[x['category_id']] for x in annotation['annotation']]
        question_list = []
        answer_list = []
        for obj_name in object_list:
            if obj_name.startswith(tuple("aeiou")):
                question = f"Is there an {obj_name} in the image?"
                # correct and wrong answers respectively
                answer = [
                    f"Yes, there is a {obj_name} in the image.",
                    f"No, there is no {obj_name} in the image.",
                ]
            else:
                question = f"Is there a {obj_name} in the image?"
                # correct and wrong answers respectively
                answer = [
                    f"Yes, there is a {obj_name} in the image.",
                    f"No, there is no {obj_name} in the image.",
                ]
            
            question_list.append(question)
            answer_list.append(answer)
        
        return {
            'questions': question_list,
            'answers': answer_list
        }


class GQASpatial(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode
        
        if args.get("instructBLIP") or args.get("BLIP2"):
            self.batch_decode = self.image_processor.batch_decode

        json_data = json.load(open('/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GQA_spatial_qas_train.json'))

        self.gqa_im_path = "/projectnb/ivc-ml/array/data/GQA/images/"

        self.data = []
        for img, qa_entries in json_data:
            for question, answers in qa_entries:
                self.data.append((img, question, answers))

        if args.get("split") == "train":
            self.data = self.data[:int(len(self.data)*0.9)]
        else:
            self.data = self.data[int(len(self.data)*0.9):]
        
        print("Total number of data points in GQA spatial: ", len(self.data))
    
    def __getitem__(self, idx):
        im_file, question, answer = self.data[idx]
        
        correct_answer = answer[0]

        ans_choice_order = ['"'+ans+'"' for ans in answer]
        random.shuffle(ans_choice_order)
        answer_choices_format = " or ".join(ans_choice_order) 

        #if im_file is not None:
        im_file = im_file.replace(".jpg", "_marked.jpg")
        img = [Image.open(im_file).convert("RGB"),]

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        if self.args['mode'] == "train":
            prompt = f"{prefix}###Human: <im_start><image><im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}. ###Assistant: \n {correct_answer} \n###"
            text_labels = prompt
        else:
            prompt = f"{prefix}###Human: <im_start><image><im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}. ###Assistant: \n "
            text_labels = prompt + correct_answer + " \n###"        

        return [im_file,], img, prompt, text_labels, correct_answer, answer, "gqa_spatial"

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        if self.args.get("instructBLIP"):
            # only works with batch size 1 for now change later. 
            inputs = self.image_processor(images=images_batch[0], text=prompts[0], return_tensors="pt")
            return_dict = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                'qformer_input_ids': inputs["qformer_input_ids"],
                'qformer_attention_mask': inputs["qformer_attention_mask"],
                'pixel_values': inputs["pixel_values"],
                "labels": inputs["input_ids"],
                "prompts": prompts,
                "text_labels": text_labels,
                "dataset": datanames,
                "answers": answers,
                "answer_choices": answer_choices,
            }
        else:
            new_images = []
            for image_b in images_batch:
                for image in image_b:
                    image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                    # pdb.set_trace()
                    image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    new_images.append(image)

            pixel_values = torch.stack(new_images, dim=0)

            input_ids = []
            attention_mask = []
            for prompt in prompts:
                # pdb.set_trace()
                input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                input_ids.append(input_id)
                attention_mask.append(torch.ones_like(input_id))
            
            # pad with zeros
            # pdb.set_trace()
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
            
            # pdb.set_trace()
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "dataset": datanames,
                "answers": answers,
                "answer_choices": answer_choices,
            }

        return return_dict


class LLaVAInstructTune(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode
        
        if args.get("llava_ov"):
            self.batch_decode = self.image_processor.batch_decode

        json_data = json.load(open("/projectnb/ivc-ml/array/data/llava_data/llava_v1_5_mix665k.json"))
        
        self.image_path = "/projectnb/ivc-ml/array/data/llava_data/image_data"
        
        self.data = []

        for entry in tqdm.tqdm(random.sample(json_data, args.get('num_data_points'))):
            
            im_path = entry.get('image')

            # pdb.set_trace()
            if im_path is None:
                continue

            if not os.path.exists(os.path.join(self.image_path, im_path)):
                continue
            
            if len(entry['conversations'])%2!=0:
                continue

            for question, answer in zip(entry['conversations'][::2], entry['conversations'][1::2]):
                self.data.append((os.path.join(self.image_path, im_path), question['value'], answer['value']))

        
        if args.get("split") == "train":
            self.data = self.data[:int(len(self.data)*0.9)]
        else:
            self.data = self.data[int(len(self.data)*0.9):]
        
        print("Total number of data points in instructtune: ", len(self.data))

    def __getitem__(self, idx):
        im_file, question, answer = self.data[idx]
        
        #if im_file is not None:
        img = [Image.open(im_file).convert("RGB"),]
        #else:
        #    img = Image.new("RGB", (224, 224), (255, 255, 255)) 

        if "<image>" in question:
            question = question.replace("<image>", "")

        
        if self.args.get("instructBLIP") or self.args.get("BLIP2"):
            if self.args['mode'] == "train":
                prompt = f"Question: {question} Answer: " + answer + "###"
                text_labels = prompt
            else:
                if self.args['prompt_mode'] == "zero_shot":
                    prompt = f"{question}"
                    text_labels = prompt + answer + "###"
                else:
                    prompt = f"Question: {question} Answer: "
                    text_labels = prompt + answer + "###"
        elif self.args.get("llava_ov"):
            """
            <|im_start|>user <image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant \nThere is a red stop sign in the image.<|im_end|><|im_start|>user <image>\nWhat about this image? How many cats do you see?<|im_end|><|im_start|>assistant\n'
            """
            image_prompt_format = "<|im_start|>user  <image>\n <|im_end|>"*len(img)
            image_prompt_format = image_prompt_format[:-len("<|im_end|>")]
            if self.args['mode'] == "train":
                prompt = f"{image_prompt_format}{question} <|im_end|><|im_start|>assistant: \n {answer} <|im_end|>"
                text_labels = prompt
            else:
                prompt = f"{image_prompt_format}{question} <|im_end|><|im_start|>assistant: \n"
                text_labels = prompt + answer + " <|im_end|>"
            
        else:
        
            image_prompt_format = "<image>"*len(img)
            
            prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
            if self.args['mode'] == "train":
                prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} ###Assistant: \n {answer} \n###"
                text_labels = prompt
            else:
                prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} ###Assistant: \n "
                text_labels = prompt + answer + " \n###"

        caption = prompt
        
        program_text = ""
        house_json = {}
        objs_present = []
        # pdb.set_trace()
        if self.args.get("qa_format"):
            return [im_file,], img, prompt, text_labels, answer, [answer,], "llava_instructtune"
        else:
            return [im_file,], img, caption, prompt, text_labels, program_text, house_json, objs_present
    
    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        image_paths, imgs, captions, prompts, text_labels, program_texts, house_jsons, objs_present = zip(*batch)
        
        if self.args.get("llava_ov"):
            pixel_values, input_ids, attention_mask, image_sizes =  get_inputs_for_model(imgs, prompts, None, self.image_processor, model_choice="llava_ov")
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "image_sizes": image_sizes,
            }
        else:
            pixel_values, input_ids, attention_mask =  get_inputs_for_model(imgs, prompts, self.tokenizer, self.image_processor, model_choice="llava")
            # pdb.set_trace()
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
            }

        return return_dict

def get_qa_type(question):
    question_type = "other"
    
    if "how did the camera" in question.lower() or "is the camera moving" in question.lower():
        question_type = "action_sequence"

    if ("need to go" in question.lower()):
        question_type = "goal_aim"

    if "any of the objects in the initial" in question.lower():
        question_type = "obj_movement"

    if "if i" in question.lower():
        question_type = "action_consequence"

    if 'if i move to the' in question.lower() or "for someone at the" in question.lower():
        question_type = "perspective"

    return question_type


class RealSATDynamic(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        self.data = json.load(open("/projectnb/ivc-ml/array/data/SAT/realDynamic/SATDynamicReal.json"))
        
    
    def __getitem__(self, idx):
        images, question, answer, distractor, datatype = self.data[idx]

        correct_answer = answer
        answer_choices = [answer, distractor]
        random.shuffle(answer_choices)
        
        answer_choices_format = " or ".join([f'"{ans}"' for ans in answer_choices])

        image_prompt_format = "<image>"*len(["im" for im in images if im!=""])

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}.###Assistant: \n" 
        text_label = prompt + correct_answer + " \n###"

        imgs = [Image.open(im_file).convert("RGB") for im_file in images if im_file!=""]

        return images, imgs, prompt, text_label, correct_answer, answer_choices, "realsat_"+datatype

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        pixel_values, input_ids, attention_mask =  get_inputs_for_model(images_batch, prompts, self.tokenizer, self.image_processor, model_choice="llava")

        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "labels": input_ids,
            "prompts": prompts,
            "text_labels": text_labels,
            "dataset": datanames,
            "answers": answers,
            "answer_choices": answer_choices,
        }

        return return_dict


class ProcTHOR_reasoning(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode
        
        if args.get("instructBLIP") or args.get("BLIP2"):
            self.batch_decode = self.image_processor.batch_decode

        #nav_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas.json'

        if args.get("split") == "train":
            spatial_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_spatial_qas_v2_train.json'           
        else:
            spatial_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_spatial_qas_new_val.json'
        
        
        if args.get("add_complex"):
            print("Adding complex data")
            if args.get("split") == "train":
                complex_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_train_v2.json' # remove v2 for prev version.
                complex_qa_json_path_split2 = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_train_v2_split2.json'
                complex_data = json.load(open(complex_qa_json_path)) + json.load(open(complex_qa_json_path_split2))
                
                # complex_data = random.sample(complex_data, args.get("num_complex", 6900)) # just to keep proportions similar to other kinds of spatial data
                camera_move_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_cameramove_qas_train.json"
                camera_move_data = json.load(open(camera_move_path))

            else:
                complex_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_val_v2.json'
                complex_data = json.load(open(complex_qa_json_path))
            
            print("Length of complex data: ", len(complex_data))

        if args.get("add_perspective"):
            print("Adding perspective data")
            if args.get("split") == "train":
                perspective_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/perspective_qas.json'
                perspective_data = json.load(open(perspective_qa_json_path))
                # perspective_data = random.sample(perspective_data, args.get("num_perspective", 4500)) # just to keep proportions similar to other kinds of spatial data
            else:
                perspective_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/perspective_qas_val.json'
                perspective_data = json.load(open(perspective_qa_json_path))
            
            print("Length of perspective data: ", len(perspective_data))

        #nav_data = json.load(open(nav_qa_json_path))
        spatial_data = json.load(open(spatial_qa_json_path))
        print("Length of spatial data: ", len(spatial_data))

        self.data = []
        if args.get("split") == "train":
            print("Adding training data")
            #for house_ind, cam_pos, cam_rot, qa_entries in nav_data[:int(len(nav_data)*0.8)]:
            #    self.data.extend(qa_entries)
            if not args.get("complex_only"):
                for house_ind, cam_pos, cam_rot, qa_entries in spatial_data:
                    self.data.extend(qa_entries)
            print("Length of basic data: ", len(self.data))

            if args.get("add_complex"):
                for house_ind, cam_pos, cam_rot, qa_entries in complex_data:
                    for question, im_order, answers in qa_entries:
                        question = question.replace("turn look straight", "look straight")

                        if answers[0] == "rotated left and rotated right" or answers[0] == "rotated right and rotated left": # bug fix
                            new_answers = ["did not move", random.choice(["rotated left", "rotated right"])]
                            answers = new_answers
                        
                        if "how did the camera likely move" in question.lower():
                            question = question.replace("How did the camera likely move when shooting the video", "How did the camera rotate from the first image to the second image") # fix this bug in generation later.
                            if self.args.get("skip rotate"):
                                continue
                            # if random.random()<0.7:
                            #    continue # too many of these QAs 

                        self.data.append((question, im_order, answers))
                for _,_,_, qa_entries in camera_move_data:
                    
                    self.data.extend(qa_entries)

                print("Length after adding complex data: ", len(self.data))
                if args.get("add_perspective"):
                    for _,_,_, qa_entries in perspective_data:
                        for question, im_order, answers in qa_entries:
                            if random.random() > args.get("perspective_prob", 0.05):
                                continue # too many of these QAs.

                            question = question.replace("turned towards the", "facing 90 degrees to the")
                            question = question.replace("turned right", "turned right by 90 degrees")
                            question = question.replace("turned left", "turned left by 90 degrees")

                            self.data.append((question, im_order, answers))
                        # pdb.set_trace()
                    print("Length after adding perspective data: ", len(self.data))
        elif args.get("split") == "valtrain":
            #for house_ind, cam_pos, cam_rot, qa_entries in nav_data[int(len(nav_data)*0.8):int(len(nav_data)*0.9)]:
            #    self.data.extend(qa_entries)
            if not args.get("complex_only"):
                for house_ind, cam_pos, cam_rot, qa_entries in spatial_data[:int(len(spatial_data)*0.1)]:
                    self.data.extend(qa_entries)

            if args.get("add_complex"):
                for house_ind, cam_pos, cam_rot, qa_entries in complex_data[:int(len(complex_data)*0.1)]:
                    for question, im_order, answers in qa_entries:
                        question = question.replace("turn look straight", "look straight")

                        if answers[0] == "rotated left and rotated right" or answers[0] == "rotated right and rotated left": # bug fix
                            new_answers = ["did not move", random.choice(["rotated left", "rotated right"])]
                            answers = new_answers

                        if "how did the camera likely move" in question.lower():
                            question = question.replace("How did the camera likely move when shooting the video", "How did the camera rotate from the first image to the second?") # fix this bug in generation later.

                        self.data.append((question, im_order, answers))
                
                if args.get("add_perspective"):
                    for _,_,_, qa_entries in perspective_data[:int(len(perspective_data)*0.1)]:
                        for question, im_order, answers in qa_entries:
                            question = question.replace("turned towards the", "facing 90 degrees to the")
                            question = question.replace("turned right", "turned right by 90 degrees")
                            question = question.replace("turned left", "turned left by 90 degrees")

                            self.data.append((question, im_order, answers))

        elif args.get("split") == "val":
            #for house_ind, cam_pos, cam_rot, qa_entries in nav_data[int(len(nav_data)*0.9):]:
            #    self.data.extend(qa_entries)
            num_basic=0
            num_complex=0
            num_perspective=0
            if not args.get("complex_only"):
                for house_ind, cam_pos, cam_rot, qa_entries in spatial_data[int(len(spatial_data)*0.1):]:
                    self.data.extend(qa_entries)
                    num_basic += len(qa_entries)
            
            if args.get("add_complex"):
                for house_ind, cam_pos, cam_rot, qa_entries in complex_data[int(len(complex_data)*0.1):]:
                    for question, im_order, answers in qa_entries:
                        question = question.replace("turn look straight", "look straight")

                        if answers[0] == "rotated left and rotated right" or answers[0] == "rotated right and rotated left": # bug fix
                            new_answers = ["did not move", random.choice(["rotated left", "rotated right"])]
                            answers = new_answers
                        
                        if "how did the camera likely move" in question.lower():
                            question = question.replace("How did the camera likely move when shooting the video", "How did the camera rotate from the first image to the second?") # fix this bug in generation later.
                        
                        self.data.append((question, im_order, answers))
                        num_complex += 1

                if args.get("add_perspective"):
                    for _,_,_, qa_entries in perspective_data[int(len(perspective_data)*0.1):]:
                        for question, im_order, answers in qa_entries:
                            question = question.replace("turned towards the", "facing 90 degrees to the")
                            question = question.replace("turned right", "turned right by 90 degrees")
                            question = question.replace("turned left", "turned left by 90 degrees")

                            self.data.append((question, im_order, answers))
                            num_perspective += 1
                            if num_perspective > 1000:
                                break
                        if num_perspective > 1000:
                            break
            print("Basic: ", num_basic, " Complex: ", num_complex, " Perspective: ", num_perspective)
        
        if args.get("subsample_data"):
            print("Subsampling data")
            self.data = random.sample(self.data, args.get("subsample_data"))


        if args.get("split") != "val":
            random.shuffle(self.data)
        print("Total number of data points: ", len(self.data))


    def __getitem__(self, idx):
        qa_entry = self.data[idx]
        
        question, image_order, answer_choices = qa_entry

        
        corrected_answer_choices = []
        for answer in answer_choices:
            if "in the first frame" in answer: # a small bug, todo fix in data gemeration later.
                answer = answer.replace("in the first frame", "")
            corrected_answer_choices.append(answer)
        answer_choices = corrected_answer_choices

        # judge the question type
        question_type = get_qa_type(question)

        correct_answer = answer_choices[0]
        
        ans_choice_order = answer_choices
        ans_choice_order = ['"'+ans+'"' for ans in ans_choice_order]
        random.shuffle(ans_choice_order)
        answer_choices_format = " or ".join(ans_choice_order)
        
        image_prompt_format = "<image>"*len(image_order)

        if self.args.get("instructBLIP") or self.args.get("BLIP2"):
            if self.args['mode'] == "train":
                prompt = f"Question: {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}. Answer: " + correct_answer + "###"
                text_label = prompt
            else:
                if self.args['prompt_mode'] == "zero_shot":
                    prompt = f"{question} Choose between the following options: {answer_choices_format}?"
                    text_label = prompt + correct_answer + "###"
                else:
                    prompt = f"Question: {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}. Answer: "
                    text_label = prompt + correct_answer + "###"
        elif self.args.get("llava_ov"):
            
            """
            <|im_start|>user <image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant \nThere is a red stop sign in the image.<|im_end|><|im_start|>user <image>\nWhat about this image? How many cats do you see?<|im_end|><|im_start|>assistant\n'
            """
            image_prompt_format = "<|im_start|>user:  <image>\n <|im_end|>"*len(image_order)
            image_prompt_format = image_prompt_format[:-len("<|im_end|>")]
            if self.args['mode'] == "train":
                prompt = f"{image_prompt_format}{question} Choose between the following options: {answer_choices_format} <|im_end|><|im_start|>assistant: \n {correct_answer} <|im_end|>"
                text_label = prompt
            else:
                prompt = f"{image_prompt_format}{question} Choose between the following options: {answer_choices_format} <|im_end|><|im_start|>assistant: \n"
                text_label = prompt + correct_answer + " <|im_end|>"
        elif self.args.get("molmo"):
            prefix = ""
            if self.args['mode'] == "train":
                prompt = f"Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}.###Assistant: \n {correct_answer} \n###"
                text_label = prompt
            else:
                prompt = f"Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}.###Assistant: \n" 
                text_label = prompt + correct_answer + " \n###"
                
        else:
            if self.args['prompt_mode'] == "zero_shot":
                '''
                A chat between a curious human and an artificial intelligence assistant. 
                The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <im_start><image><im_end>
                Human: Describe the image and color details.###Assistant:
                The image features a wooden pier extending out into a large body of water, possibly a lake or a bay. The pier is surrounded by a serene environment, with trees and mountains in the background. The water appears to be calm and clear, making it an ideal spot for relaxation or leisurely activities. The scene is captured in black and white, giving it a timeless and classic feel.
                '''
                prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
                prompt = f"{prefix}.###Human: <im_start>{image_prompt_format}<im_end> \nHuman: {question} Choose between the following options: {answer_choices_format}.###Assistant: \n "
                text_label = prompt + correct_answer + " \n###"

            elif self.args['prompt_mode'] == "finetune_choice_format":
                prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
                ans_choices_format = ""
                num_to_letter = {
                    0: "A",
                    1: "B",
                    2: "C",
                    3: "D",
                    4: "E",
                    5: "F",
                }
                for i, ans in enumerate(answer_choices):
                    ans_choices_format += f"{num_to_letter[i]}) {ans} "

                if self.args['mode'] == "train":
                    prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: {question} Choose between the following options: {answer_choices_format}.###Assistant: \n {correct_answer} \n###"
                    text_label = prompt
                else:
                    prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: {question} Choose between the following options: {answer_choices_format}.###Assistant: \n" 
                    text_label = prompt + correct_answer + " \n###"
            else:
                prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
                if self.args['mode'] == "train":
                    if self.args.get("knowledge_injection_mode"):
                        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. ###Assistant: \n {correct_answer} \n###"
                        text_label = prompt
                    else:
                        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}.###Assistant: \n {correct_answer} \n###"
                        text_label = prompt
                else:
                    prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {answer_choices_format}.###Assistant: \n" 
                    text_label = prompt + correct_answer + " \n###"

        images = [Image.open(img).convert("RGB") for img in image_order]

        return image_order, images, prompt, text_label, correct_answer, answer_choices, f"procthor_reasoning_{question_type}"

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        if self.args.get("llava_ov"):
            
            pixel_values, input_ids, attention_mask, image_sizes =  get_inputs_for_model(images_batch, prompts, None, self.image_processor, model_choice="llava_ov")
            
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "image_sizes": image_sizes,
                "dataset": datanames,
                "answers": answers,
                "answer_choices": answer_choices,
            }
        elif self.args.get("molmo"):
            imgs = []
            for img in images_batch:
                for im in img:
                    imgs.append(im)
            inputs = self.molmo_processor.process(
                images=imgs,
                text=prompts,
            )
        else:
            pixel_values, input_ids, attention_mask =  get_inputs_for_model(images_batch, prompts, self.tokenizer, self.image_processor, model_choice="llava")
            # pdb.set_trace()
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "dataset": datanames,
                "answers": answers,
                "answer_choices": answer_choices,
            }

        return return_dict


class RoboPointDataset(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode        

        llava_format_data = json.load(open("/projectnb/ivc-ml/array/data/Robopoint/robopoint_1432k.json"))
        
        self.image_path = "/projectnb/ivc-ml/array/data/Robopoint/images"
        self.data = []
        for entry in tqdm.tqdm(llava_format_data):
            im_path = entry.get('image')

            # pdb.set_trace()
            if im_path is None:
                continue

            if not os.path.exists(os.path.join(self.image_path, im_path)):
                continue
            
            if len(entry['conversations'])%2!=0:
                continue

            for question, answer in zip(entry['conversations'][::2], entry['conversations'][1::2]):
                self.data.append(([os.path.join(self.image_path, im_path),], question['value'], answer['value']))

        print("length of robopoint data: ", len(self.data))

    def __getitem__(self, idx):
        image_order, question, correct_answer = self.data[idx]

        image_prompt_format = "<image>"*len(image_order)

        question = question.replace("<image>", "")
        
        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        if self.args['mode'] == "train":
            prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. ###Assistant: \n {correct_answer} \n###"
            text_label = prompt
        else:
            prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. ###Assistant: \n" 
            text_label = prompt + correct_answer + " \n###"

        images = [Image.open(img).convert("RGB") for img in image_order]

        return image_order, images, prompt, text_label, correct_answer, [correct_answer,], f"robopoint"
    
    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        pixel_values, input_ids, attention_mask =  get_inputs_for_model(images_batch, prompts, self.tokenizer, self.image_processor, model_choice="llava")
        # pdb.set_trace()
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "labels": input_ids,
            "prompts": prompts,
            "text_labels": text_labels,
            "dataset": datanames,
            "answers": answers,
            "answer_choices": answer_choices,
        }

        return return_dict


class ProcTHOR_recon_qa(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode
        
        if args.get("instructBLIP") or args.get("BLIP2"):
            self.batch_decode = self.image_processor.batch_decode

        if args.get("split") == "train":
            if args.get("random_point"):
                json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/GPT4_zero_shot_exps/all_recon_qas_train.json"))
            else:
                json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/GPT4_zero_shot_exps/all_recon_qas_nomark_train.json"))
        else:
            if args.get("random_point"):
                json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/GPT4_zero_shot_exps/all_recon_qas_randompoint.json"))
            else:
                json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/GPT4_zero_shot_exps/all_recon_qas_nomark.json"))    

        self.data = json_data
        print("Total number of data points: ", len(self.data))
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        prompt = entry['prompts']
        answer = entry['answers']
        image_path = entry['image_path']
        answer_choices = entry['answer_choices']

        if self.args.get("instructBLIP") or self.args.get("BLIP2"):
            if self.args['mode'] == "train":
                prompt = "Question: " + prompt + " Answer: " + answer
                text_label = prompt
            else:
                prompt = "Question: " + prompt + " Answer: "
                text_label =  "Question: " + prompt + " " + answer
        else:
            prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
            if self.args['mode'] == "train":
                prompt = f"{prefix}###Human: <im_start><image><im_end> \nHuman: Answer in natural language. {prompt}.###Assistant: \n {answer} \n###"
                text_label = prompt
            else:
                prompt = f"{prefix}###Human: <im_start><image><im_end> \nHuman: Answer in natural language. {prompt}.###Assistant: \n" 
                text_label = prompt + answer + " \n###"


        images = [Image.open(image_path[0]).convert("RGB"),]

        #pdb.set_trace()
        return image_path, images, prompt, text_label, answer, answer_choices, "procthor_recon_qa"

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        if self.args.get("instructBLIP"):
            # only works with batch size 1 for now change later. 
            inputs = self.image_processor(images=images_batch[0], text=prompts[0], return_tensors="pt")
            return_dict = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                'qformer_input_ids': inputs["qformer_input_ids"],
                'qformer_attention_mask': inputs["qformer_attention_mask"],
                'pixel_values': inputs["pixel_values"],
                "labels": inputs["input_ids"],
                "prompts": prompts,
                "text_labels": text_labels,
                "dataset": datanames,
                "answers": answers,
                "answer_choices": answer_choices,
            }
        elif self.args.get('BLIP2'):
            inputs = self.image_processor(images=images_batch[0], text=prompts[0], return_tensors="pt")
            return_dict = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                'pixel_values': inputs["pixel_values"],
                "labels": inputs["input_ids"],
                "prompts": prompts,
                "text_labels": text_labels,
                "dataset": datanames,
                "answers": answers,
                "answer_choices": answer_choices,
            }
        else:
            new_images = []
            for image_b in images_batch:
                for image in image_b:
                    image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                    # pdb.set_trace()
                    image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    new_images.append(image)

            pixel_values = torch.stack(new_images, dim=0)

            input_ids = []
            attention_mask = []
            for prompt in prompts:
                # pdb.set_trace()
                input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                input_ids.append(input_id)
                attention_mask.append(torch.ones_like(input_id))
            
            # pad with zeros
            # pdb.set_trace()
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
            
            # pdb.set_trace()
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "dataset": datanames,
                "answers": answers,
                "answer_choices": answer_choices,
            }

        return return_dict


class AnyImageCaption(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
       
        self.batch_decode = self.tokenizer.batch_decode

        self.image_data_path = args['image_data_path']
        
        self.data = []
        for image_path in os.listdir(self.image_data_path):
            if ".jpeg" in image_path or ".jpg" in image_path or ".png" in image_path:
                self.data.append(os.path.join(self.image_data_path, image_path))
            # self.data.append(os.path.join(self.image_data_path, image_path))

        self.polygons = {
            'bedroom_1': [(0, 0, 0), (0, 0, 600), (600, 0, 600), (600, 0, 0)],
            'bedroom_2': [(0, 0, 0), (0, 0, 600), (400, 0, 600), (400, 0, 0)],
            'bedroom_3': [(0, 0, 0), (0, 0, 400), (600, 0, 400), (600, 0, 0)],
            'living_room_1': [(0, 0, 0), (0, 0, 400), (600, 0, 400), (600, 0, 0)],
            'living_room_2': [(0, 0, 0), (0, 0, 400), (600, 0, 400), (600, 0, 0)],
            'living_room_3': [(0, 0, 0), (0, 0, 400), (600, 0, 400), (600, 0, 0)],
            'toilet_1': [(0, 0, 0), (0, 0, 400), (600, 0, 400), (400, 0, 0)],
            'toilet_2': [(0, 0, 0), (0, 0, 400), (600, 0, 400), (600, 0, 0)],
            'dalle_livingroom': [(0, 0, 0), (0, 0, 400), (600, 0, 400), (600, 0, 0)],
        }

        self.cams = {
            'bedroom_1': (300, 300),
            'bedroom_2': (30, 30),
            'bedroom_3': (300, 300),
            'living_room_1': (300, 300),
            'living_room_2': (300, 300),
            'living_room_3': (300, 300),
            'toilet_1': (300, 300),
            'toilet_2': (20, 20),
            'dalle_livingroom': (300, 300),
        }
        self.args = args

    def __getitem__(self, idx):
       
        image_path = self.data[idx]
        
        image = Image.open(image_path).convert("RGB")

        room_type = image_path.split("/")[-1].split(".")[0]
        format_polygon_coords = str(self.polygons[room_type])
        camera_prompt = f"Image taken from (x,z) {self.cams[room_type]}."
        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        prompt = f"Answer in a structured JSON format. <image> The room polygon is (x,z) {format_polygon_coords}. {camera_prompt} Plausible 3D coordinates (x, y,z) for the rest of the room: \n"
        
        return [image_path, ], image, prompt
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        im_files, images, prompts = zip(*batch)

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
            "text_labels": prompts,
            "image_lists": im_files,
        }

        return return_dict

def calc_camera_intrinsics(fov_y, frame_height, frame_width):
    # this functionality is now here to avoid a circularity or duplication issue
    focal_length = 0.5 * frame_height / math.tan(math.radians(fov_y / 2))
    f_x = f_y = focal_length

    c_x = frame_width / 2
    c_y = frame_height / 2
    return round(f_x, 2), round(f_y, 2), round(c_x, 2), round(c_y, 2)

class ProcTHOR_3DCaptions(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        split = args.get("split")
        self.split = split

        if split in ["train", "valtrain"]:
            json_data1 = json.load(open(f"/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_captions_train.json"))
            json_data2 = json.load(open(f"/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_captions_train_split2.json"))
            json_data = json_data1 + json_data2
        else:
            json_data = json.load(open(f"/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_captions_{split}.json"))

        self.data = []
        for frame_path, visible_obj_descs, coordinate_text, field_of_view in json_data:
            for question, answer in visible_obj_descs:
                self.data.append((frame_path, question, answer, coordinate_text, field_of_view))
        
        print("Total number of data points: ", len(self.data))

        if args["split"] == "train":
            self.data = self.data[:int(len(self.data)*0.8)]
        elif args["split"] == "valtrain":
            self.data = self.data[int(len(self.data)*0.8):int(len(self.data)*0.9)]
        elif args["split"] == "val":
            self.data = self.data[:]

    def __getitem__(self, idx):
        frame_path, question, answer, coordinate_text, field_of_view = self.data[idx]
        marked_frame = frame_path.replace("_0.jpg", "_0_marked.jpg")

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        prefix += "###Human: <im_start><image><im_end> \nHuman: Answer with precise 3D coordinate points."
        prefix += "Assume camera is at origin. Camera looks at positive Z, X points to right and Y points upwards. "
        
        if self.args.get("random_point"):
            im_file_path = marked_frame
            img = Image.open(im_file_path).convert("RGB")
            prompt = f"{prefix} {coordinate_text} "
        else:
            im_file_path = frame_path
            img = Image.open(im_file_path).convert("RGB")
            prompt = f"{prefix} "

        if self.args.get("use_cam_intrinsic"):
            fx, fy, cx, cy = calc_camera_intrinsics(field_of_view, img.size[1], img.size[0])
            cam_instrinsic_prompt = f"Camera intrinsic parameters are: focal length x: {fx}, focal length y: {fy}, center point x: {cx}, center point y: {cy}. The image resolution is: {img.size[0]}x{img.size[1]}."
            prompt = f"{prompt} {cam_instrinsic_prompt} "

        if self.split == "train":
            prompt = f"{prompt} {question} ###Assistant: \n {answer} \n###"
            text_label = prompt
        else:
            prompt = f"{prompt} {question} ###Assistant: \n"
            text_label = prompt + answer + " \n###"

        if self.args.get("llava_ov"):
            
            """
            <|im_start|>user <image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant \nThere is a red stop sign in the image.<|im_end|><|im_start|>user <image>\nWhat about this image? How many cats do you see?<|im_end|><|im_start|>assistant\n'
            """
            image_prompt_format = "<|im_start|>user  <image>\n <|im_end|>"*1
            image_prompt_format = image_prompt_format[:-len("<|im_end|>")]
            
            prompt = f"{image_prompt_format}{cam_instrinsic_prompt} {question} <|im_end|><|im_start|>assistant: \n"
            text_label = prompt + answer + " <|im_end|>"

        return [im_file_path,], [img,], prompt, text_label, answer, [answer,], f"procthor_3dcapqa"
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        image_paths, images_batch, prompts, text_labels, answers, answer_choices, datanames = zip(*batch)

        if self.args.get("llava_ov"):
            pixel_values, input_ids, attention_mask, image_sizes =  get_inputs_for_model(images_batch, prompts, None, self.image_processor, model_choice="llava_ov")
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "image_sizes": image_sizes,
                "dataset": datanames,
                "answers": answers,
                "answer_choices": answer_choices,
            }
        else:
            pixel_values, input_ids, attention_mask =  get_inputs_for_model(images_batch, prompts, self.tokenizer, self.image_processor, model_choice="llava")
            # pdb.set_trace()
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "dataset": datanames,
                "answers": answers,
                "answer_choices": answer_choices,
            }

        return return_dict



class ProcTHOR_image_camposition_marked(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        if args.get("BLIP2"):
            self.batch_decode = self.image_processor.batch_decode

        json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/final_data_neurips.json"))

        #if self.args.get("use_topdown"):
        topdown_caption_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GPT4V_room_descriptions_topdown.json"))
        # else:
        caption_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GPT4V_room_descriptions.json"))

        # asset_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/asset_descriptions_all.json"))
        self.asset_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json"))
        # contains, im_file, obj_class, caption

        #pdb.set_trace()
        #self.asset_desc = {}
        #for asset_id in asset_desc:
        #    asset_entry = random.choice(asset_desc[asset_id])
        #    self.asset_desc[asset_id] = (asset_entry[1], asset_entry[2])

        # self.asset_desc = {}
        #for image_file, asset_name, object_class, caption in asset_desc:
        #    self.asset_desc[asset_name] = (object_class, caption)


        apid_to_topdown_caption = {}
        for apid, image_file, caption in topdown_caption_data:
            apid_to_topdown_caption[apid] = caption

        self.apartment_ind_to_caption = {}
        for apartment_ind, image_file, caption in caption_data:
            #pdb.set_trace()
            # if apartment_ind in caption_data:
            self.apartment_ind_to_caption[apartment_ind] = (image_file, apid_to_topdown_caption.get(apartment_ind, ""))
        # pdb.set_trace()
        self.data = []
        for entry in json_data:
            if len(entry[4]) < 1:
                continue
            if not os.path.exists(entry[4][0]):
                continue

            apartment_ind = entry[4][0].split("/")[-2]
            if apartment_ind not in self.apartment_ind_to_caption:
                continue

            self.data.append(entry)
        
        print("Total number of data points: ", len(self.data))

        if args["split"] == "train":
            self.data = self.data[:11000]
            self.split = "train"
        elif args["split"] == "valtrain":
            self.data = self.data[11000:11100]
            self.split = "val"
        elif args["split"] == "val":
            self.data = self.data[-550:]
            self.split = "test"
    
        print("Total number of data points in split: ", len(self.data))
        
    def __getitem__(self, idx):
        
        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

        room_data = self.data[idx]

        program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name = room_data

        if self.args.get("use_attributes"):
            program_text = generate_attribute_program_from_roomjson(house_json, include_children=self.args.get('include_children'), asset_desc=self.asset_desc)
        else:
            program_text = generate_program_from_roomjson(house_json, include_children=self.args.get('include_children'))

        apartment_id = all_imgs[0].split("/")[-2]
        image_path, caption = self.apartment_ind_to_caption[apartment_id]

        corner_ind = image_path.split("/")[-1].split(".")[0]

        # pdb.set_trace()
        camera_poses = cam_ind_to_position[corner_ind]

        pos, rot = camera_poses

        cam_pos = (int(round(pos['x'],2)*100), 95, int(round(pos['z'],2)*100))
        cam_angle = int(rot['y'])
        
        
        if self.args.get("normalize_rotation"):
            program_text = normalize_coords(program_text, cam_pos, cam_angle, attr=self.args.get("use_attributes"))
            cam_pos = (0, 95, 0)
            cam_angle = 0
            program_text = program_text.replace("(", "[")
            program_text = program_text.replace(")", "]")
            program_text = program_text.split("window_")[0].strip() # window positions seem confusing, fix later. 

        program_json = yaml.load(program_text, Loader=yaml.FullLoader)
        format_polygon_coords = str(program_json['polygon'])

        image_ind = all_imgs.index(image_path)
        
        seg_frame = Image.open(all_seg_frames[image_ind]).convert("RGB")
        

        seg_frame = np.array(seg_frame)
        
        # get image mean coordinate of the color in the seg frame
        all_colors = seg_frame.reshape(-1, 3)
        unique_colors = np.unique(all_colors, axis=0)
        obj_ids_present = [color_to_objid.get(str(tuple(color))) for color in unique_colors]
        obj_ids_present = [obj_id for obj_id in obj_ids_present if obj_id is not None]

        cfg_dict = yaml.load(program_text, Loader=yaml.FullLoader)
        i = 0
        max_dist_to_camera = 0
        max_dist_obj_pos = None
        max_dist_obj_name = None
        objs_to_choose = []
        while(True):
            if f'obj_{i}' in cfg_dict:
                if f'obj_{i}' not in obj_ids_present:
                    i += 1
                    continue
                obj = cfg_dict[f'obj_{i}']
                # print(obj[1])
                if self.args.get("use_attributes"):
                    obj_pos = obj.split("at location ")[-1].split(" with")[0]
                    obj_pos = ast.literal_eval(obj_pos)
                else:
                    obj_pos = obj[1]
                dist_to_camera = np.sqrt((obj_pos[0] - cam_pos[0])**2 + (obj_pos[2] - cam_pos[1])**2)

                if self.args.get("randomize_point") or self.args.get("use_multimarks"):
                    objs_to_choose.append((f'obj_{i}', obj_pos))
                else:
                    if dist_to_camera > max_dist_to_camera:
                        max_dist_to_camera = dist_to_camera
                        max_dist_obj_pos = obj_pos
                        max_dist_obj_name = f'obj_{i}'
                i += 1
            else:
                break
        
        if self.args.get("randomize_point"):
            if len(objs_to_choose) > 0:
                max_dist_obj_name, max_dist_obj_pos = random.choice(objs_to_choose)
        
        if self.args.get("use_multimarks"):
            if len(objs_to_choose) > 0:
                marked_points = random.sample(objs_to_choose, min(5, len(objs_to_choose)))
                max_dist_obj_name = marked_points[0][0]
        
        objid_to_color = {}
        for color in color_to_objid:
            objid_to_color[color_to_objid[color]] = color

        # pdb.set_trace()

        # mark furthest object in the image if available
        x,y = None, None
        all_points = []
        if max_dist_obj_name is not None:
            if self.args.get("use_multimarks"):
                all_points = []
                for obj_name, obj_pos in marked_points:
                    color = list(ast.literal_eval(objid_to_color[obj_name]))
                    color_mask = np.all(seg_frame == color, axis=-1)
                    y, x = np.where(color_mask)
                    x = x.mean()
                    y = y.mean()
                    all_points.append((x,y))
            else:
                farthest_obj_color = list(ast.literal_eval(objid_to_color[max_dist_obj_name]))

                color_mask = np.all(seg_frame == farthest_obj_color, axis=-1)
                y, x = np.where(color_mask)
                x = x.mean()
                y = y.mean()

        # mark the position in the image
        if self.args.get("use_depth"):
            depth_path = image_path.replace(".png", "_depth.pt")
            depth = torch.load(depth_path)
            # pdb.set_trace()
            depth = depth.squeeze(0).numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = (depth * 255).astype(np.uint8)
            img = Image.fromarray(depth[0,:,:],  'L')
            
            # stich og image and depth side by side
            og_img = Image.open(image_path).convert("RGB")
            
            img = stich_image([og_img, img])
            
            img = [add_red_dot_with_text(img, (x,y), "1"),]
            new_im_file = "/".join(image_path.split("/")[:-1])+f"/{corner_ind}_depth_marked.png"
            # pdb.set_trace()
        elif self.args.get("captiononly_baseline"):
            img = [Image.open(image_path).convert("RGB"),] # still have to pass in some pixel value even thogh it will be ignored. Engineering issue in batch collate, fix later. 
            new_im_file = ""
        elif self.args.get("use_no_mark_baseline"):
            img = [Image.open(image_path).convert("RGB"),]

            new_im_file = ""

        elif self.args.get("use_depth_stiched"):
            depth_path = image_path.replace(".png", "_depth.pt")
            depth = torch.load(depth_path)

            depth = depth.squeeze().numpy()

            d_min = 1
            d_max = 500
            prediction = np.clip(depth, d_min, d_max)
            prediction = (d_min/prediction) - (prediction-d_min)/((d_max-d_min)*d_max/d_min)

            depth = ((1-prediction) * 255).astype(np.uint8)
            img = Image.fromarray(depth[:,:],  'L')

            # stich og image and depth side by side
            og_img = Image.open(image_path).convert("RGB")
            
            img = stich_image([og_img, img.convert("RGB")])
            
            img = [add_red_dot_with_text(img, (x,y), "1"),]
            new_im_file = "/".join(image_path.split("/")[:-1])+f"/{corner_ind}_depth_marked_inverse.png"
        
        elif self.args.get("use_depth_greyoverlay"):
            depth_path = image_path.replace(".png", "_depth.pt")
            depth = torch.load(depth_path)

            depth = depth.squeeze().numpy()

            d_min = 1
            d_max = 500
            prediction = np.clip(depth, d_min, d_max)
            prediction = (d_min/prediction) - (prediction-d_min)/((d_max-d_min)*d_max/d_min)

            depth = ((1-prediction) * 255).astype(np.uint8)
            # pdb.set_trace()
            img = Image.fromarray(depth[:,:],  'L')

            # stich og image and depth side by side
            og_img = Image.open(image_path).convert("RGB")
            # pdb.set_trace()
            img = Image.blend(og_img, img.convert("RGB"), 0.2)
            
            img = [add_red_dot_with_text(img, (x,y), "1"),]
            new_im_file = "/".join(image_path.split("/")[:-1])+f"/{corner_ind}_depth_greyoverlay.png"
        elif self.args.get("use_depth_greyoverlay_nopoint"):
            depth_path = image_path.replace(".png", "_depth.pt")
            depth = torch.load(depth_path)

            depth = depth.squeeze().numpy()

            d_min = 1
            d_max = 500
            prediction = np.clip(depth, d_min, d_max)
            prediction = (d_min/prediction) - (prediction-d_min)/((d_max-d_min)*d_max/d_min)

            depth = ((1-prediction) * 255).astype(np.uint8)
            # pdb.set_trace()
            img = Image.fromarray(depth[:,:],  'L')

            # stich og image and depth side by side
            og_img = Image.open(image_path).convert("RGB")
            # pdb.set_trace()
            img = Image.blend(og_img, img.convert("RGB"), 0.2)
            
            img = [img,]
            new_im_file = "/".join(image_path.split("/")[:-1])+f"/{corner_ind}_depth_greyoverlay.png"

        elif self.args.get("use_depth_yuv"):
            depth_path = image_path.replace(".png", "_depth.pt")
            depth = torch.load(depth_path)
            depth = depth.squeeze(0).numpy()
            d_min = 1
            d_max = 500 # changed from 1000
            prediction = np.clip(depth, d_min, d_max)
            prediction = (d_min/prediction) - (prediction-d_min)/((d_max-d_min)*d_max/d_min)

            og_img_yuv = np.array(Image.open(image_path).convert('YCbCr'))
            
            img_d = og_img_yuv
            img_d[:,:,0] = og_img_yuv[:,:,0]*(1 - 0.85*prediction)

            img_rgb = cv2.cvtColor(img_d, cv2.COLOR_YCrCb2BGR)
            img = Image.fromarray(img_rgb[:,:,::-1])
            img = [add_red_dot_with_text(img, (x,y), "1"),]
            new_im_file = "/".join(image_path.split("/")[:-1])+f"/{corner_ind}_depth_marked_yuv.png"
        elif self.args.get("use_depth_yuv_no_point"):
            depth_path = image_path.replace(".png", "_depth.pt")
            depth = torch.load(depth_path)
            depth = depth.squeeze(0).numpy()
            d_min = 1
            d_max = 500
            prediction = np.clip(depth, d_min, d_max)
            prediction = (d_min/prediction) - (prediction-d_min)/((d_max-d_min)*d_max/d_min)

            og_img_yuv = np.array(Image.open(image_path).convert('YCbCr'))
            
            img_d = og_img_yuv
            img_d[:,:,0] = og_img_yuv[:,:,0]*(1 - 0.85*prediction)

            img_rgb = cv2.cvtColor(img_d, cv2.COLOR_YCrCb2BGR)
            img = [Image.fromarray(img_rgb[:,:,::-1]),]
            
            new_im_file = "/".join(image_path.split("/")[:-1])+f"/{corner_ind}_depth_marked_yuv_nopoint.png"

            # pdb.set_trace()
        elif self.args.get("use_depth_points"):
            depth_path = image_path.replace(".png", "_depth.pt")
            depth = torch.load(depth_path)

            depth = depth.squeeze().numpy()

            d_min = 1
            d_max = 500
            prediction = np.clip(depth, d_min, d_max)
            prediction = (d_min/prediction) - (prediction-d_min)/((d_max-d_min)*d_max/d_min)

            sussamp = 2
            def samp(x):
                return np.random.binomial(1, x)

            depth_mask = np.vectorize(samp)(prediction[::sussamp,::sussamp])
            depth_mask = np.repeat(np.repeat(depth_mask, sussamp, axis=0), sussamp, axis=1)

            og_img_yuv = np.array(Image.open(image_path).convert('YCbCr'))

            img_d = np.array(og_img_yuv)
            img_d[:,:,0] = og_img_yuv[:,:,0]*(1 - 0.3*depth_mask)

            img_rgb = cv2.cvtColor(img_d, cv2.COLOR_YCrCb2BGR)

            img = Image.fromarray(img_rgb[:,:,::-1])
            img = [add_red_dot_with_text(img, (x,y), "1"),]
            new_im_file = "/".join(image_path.split("/")[:-1])+f"/{corner_ind}_depth_marked_points.png"
            # pdb.set_trace()
        elif self.args.get("use_depth_twoim"):
            img = Image.open(image_path).convert("RGB")
            img = add_red_dot_with_text(img, (x,y), "1")

            depth_path = image_path.replace(".png", "_depth.pt")
            depth = torch.load(depth_path)

            depth = depth.squeeze().numpy()

            d_min = 1
            d_max = 500
            prediction = np.clip(depth, d_min, d_max)
            prediction = (d_min/prediction) - (prediction-d_min)/((d_max-d_min)*d_max/d_min)

            depth = ((1-prediction) * 255).astype(np.uint8)
            # pdb.set_trace()
            d_img = Image.fromarray(depth[:,:],  'L')

            img = [img, d_img]

            new_im_file = "/".join(image_path.split("/")[:-1])+f"/{corner_ind}_marked_2.png"
        elif self.args.get("use_multimarks"):
            img = Image.open(image_path).convert("RGB")
            for mi, (x,y) in enumerate(all_points):
                img = add_red_dot_with_text(img, (x,y), str(mi))
            img = [img,]
            new_im_file = "/".join(image_path.split("/")[:-1])+f"/{corner_ind}_marked_multi.png"
            
        else:
            img = Image.open(image_path).convert("RGB")
            
            img = [add_red_dot_with_text(img, (x,y), "1"),]

            new_im_file = "/".join(image_path.split("/")[:-1])+f"/{corner_ind}_marked.png"
            if self.args.get("randomize_point"):
                new_im_file = "/".join(image_path.split("/")[:-1])+f"/{corner_ind}_marked_random.png"

        if self.args.get("recon_qa_mode"):
            if new_im_file != "":
                new_im_file = new_im_file.replace("marked", "marked_recon")
                img[0].save(new_im_file)
            else:
                new_im_file = image_path

        if self.args['split']!= "train":
            if new_im_file != "":
                img[0].save(new_im_file)
            else:
                new_im_file = image_path

        if max_dist_obj_name is not None:
            if self.args.get("captiononly_baseline"):
                camera_prompt = ""
            elif self.args.get("use_multimarks"):
                camera_prompt = f"Image taken from (x, y, z) {cam_pos}."
                if self.args.get("use_angle"):
                    camera_prompt = f"Image taken from (x,z) {cam_pos} with angle around y as {cam_angle}."
                for mi, (_, obj_pos) in enumerate(marked_points):
                    camera_prompt += f" The red {mi} mark in the image is at 3D coordinate (x, y, z) {obj_pos}."
            else:
                camera_prompt = f"Image taken from (x, y, z) {cam_pos} looking inside the polygon. The red circular 1 mark in the image is at 3D coordinate (x, y, z) {max_dist_obj_pos}. "
                if self.args.get("use_angle"):
                    camera_prompt = f"Image taken from (x, y, z) {cam_pos} with angle around y as {cam_angle} looking inside the polygon. The camera field of view is 90. The red circular 1 mark in the image is at 3D coordinate (x, y, z) {max_dist_obj_pos}. "
                if self.args.get("use_no_mark_baseline") or self.args.get("use_depth_yuv_no_point"):
                    camera_prompt = f"Image taken from (x, y, z) {cam_pos}."
                    if self.args.get("use_angle"):
                        camera_prompt = f"Image taken from (x, y, z) {cam_pos} with angle around y as {cam_angle}."

                if self.args.get("use_depth_greyoverlay_nopoint"):
                    camera_prompt = f"The image has been overlayed with depth with darker colors meaning more depth (farther away from camera, which means higher z). Image taken from (x, y, z) {cam_pos} with rotation angle clockwise around y as {cam_angle}. "

        else:
            if self.args.get("captiononly_baseline"):
                camera_prompt = ""
                new_im_file = image_path
                img = []
            else:
                new_im_file = image_path
                img = [Image.open(image_path).convert("RGB"),]
                camera_prompt = f"Image taken from (x,z) {cam_pos}. "
                if self.args.get("use_angle"):
                    camera_prompt = f"Image taken from (x,z) {cam_pos} with angle around y as {cam_angle}. "
                if self.args.get("use_no_mark_baseline") or self.args.get("use_depth_yuv_no_point"):
                    camera_prompt = f"Image taken from (x,z) {cam_pos}."
                    if self.args.get("use_angle"):
                        camera_prompt = f"Image taken from (x,z) {cam_pos} with angle around y as {cam_angle}."
                
                if self.args.get("use_depth_greyoverlay_nopoint"):
                    camera_prompt = f"The image has been overlayed with depth with darker colors meaning more depth. Image taken from (x, y, z) {cam_pos} with rotation angle clockwise around y as {cam_angle}. "


        if self.args.get("use_caption"):
            prefix_options = [
                f"## <image> Describe the image: {caption} If the room polygon is (x,z) {format_polygon_coords}. {camera_prompt} Plausible 3D coordinates (x, y,z) for the rest of the room: \n",
            ]
            if self.args['mode'] == "val":
                prefix_options = [
                    f"## <image> If the room polygon is (x,z) {format_polygon_coords}. {camera_prompt} Plausible 3D coordinates (x, y,z) for the rest of the room: \n",
                ]
        elif self.args.get("captiononly_baseline"):
            prefix_options = [
                f"## The room polygon is (x,z) {format_polygon_coords}. {caption} Plausible 3D coordinates (x, y,z) for the room: \n",
            ]
        elif self.args.get("use_incontext"):

            if self.args.get("incontext_language"):
                prefix_options = [
                    f"{prefix}###Human: <im_start><image><im_end> \nHuman: Imagine that in a 3D cubical space looking from the top down, x goes from left to right, y comes out towards the camera out of the plane, and z goes from bottom to top. \
The rotations are always specified as around the y axis (axis coming towards camera) with respect to positive z-axis rotating clockwise in the top down view. \
For instance, 90 degrees rotation means pointing towards the positive x axis (towards the right when looking from top down) and so on. \
The image was taken by placing the camera at (x,z) {cam_pos} with rotation as {cam_angle} degrees. \
Please try to estimate a rich description of the room- the floor, walls and the objects present. Each object should be in a separate line along with the 3D position. The 3D locations x, y, z should follow the above conventions and the rotation as well.\
Please generate the description in the following format: \n \
Floor material:  material/color of the floor eg dark wood or white etc \n \
Wall material: material/color of the wall eg white or tiles etc \n \
Object 1: Object description at location (x,y,z) with rotation D degrees. eg, brown wooden bed at location (10,20,15) with rotation 80 degrees. \n \
And so on. \n \
Remember that each object is in a separate line. \
Can you please do this for the image shown? Please just output the description in the above format and no other text. \n \
\n Answer: \n",
                ]
            elif self.args.get("incontext_pointmark"):
                prefix_options = [
                    f"{prefix}###Human: <im_start><image><im_end> \nHuman: In this image, x increases from left to right, y increases from bottom to top, and z increases towards the direction you are looking at (depth). \
You are at (x, y, z) of ({cam_pos[0]}, 95, {cam_pos[2]}). \
The red dot marked in the image is at {max_dist_obj_pos}. Using this information, can you roughly estimate the locations of all the objects in the room? Just try to the best of your abilities. Objects to the left of the dot will have a lower x. Objects closer to the camera from the dot will have a lower z. And, objects lower in height from the camera height will have a lower y.  \
Please output each object along with the location on each line with no other text in the following format: \n Object name at location (x, y, z). eg, Brown bed at (10, 15, 20)",
                ]
            elif self.args.get("incontext_pointmark_GPT"):
                prefix_options = [
                    f"In this image, you (the camera taking the photo) is at an (x, y, z) of ({cam_pos[0]}, 95, {cam_pos[2]}) respectively. x decreases towards the left and increases towards the right, y decreases towards the bottom and increases towards the top, and z increases from 0 towards the direction you are looking at (the depth direction). The scale is in centimeters, which means a value of 100 means 100 cm. \
The red dot marked in the image is at an (x, y, z) of {max_dist_obj_pos}. Using this information, can you roughly estimate the locations of all the objects in the room? Just try to the best of your abilities. Objects to the left of the dot will have a lower x. Objects closer to the camera from the dot will have a lower z. And, objects lower in height from the dot will have a lower y.  \
Please output each object along with the location on each line with no other text in the following format: \n Object name at location (x, y, z). eg, Brown bed at location (10, 15, 20)",
                ]
            elif self.args.get("incontext_nomark_GPT"):
                prefix_options = [
                    f"In this image, you (the camera taking the photo) is at an (x, y, z) of ({cam_pos[0]}, 95, {cam_pos[2]}) respectively. x decreases towards the left and increases towards the right, y decreases towards the bottom and increases towards the top, and z increases from 0 towards the direction you are looking at (the depth direction). The scale is in centimeters, which means a value of 100 means 100 cm. \
Using this information, can you roughly estimate the locations of all the objects in the room? Just try to the best of your abilities. Objects to the left of the camera will have a lower x. Objects closer to the camera from the dot will have a lower z. And, objects lower in height from the camera will have a lower y.  \
Please output each object along with the location on each line with no other text in the following format: \n Object name at location (x, y, z). eg, Brown bed at location (10, 15, 20)",
                ]
            elif self.args.get("incontext_GPT"):
                prefix_options = [
                    f"In this image, x increases from left to right, y increases from bottom to top, and z increases towards the direction you are looking at (depth). \
You are at (x, y, z) ({cam_pos[0]}, 95, {cam_pos[2]}). Using this information, can you roughly estimate the locations of all the objects in the room? Just try to the best of your abilities. Objects to the left of the camera will have a lower x. Objects closer to the camera have a lower z. And, objects lower in height from the camera height will have a lower y.  \
Please output each object along with the location on each line with no other text in the following format: \n Object name at location (x, y, z). eg, Brown bed at location (10, 15, 20)",
                ]
            
        else:
            if self.args.get("no_camera_prompt"):
                prefix_options = [
                    f"## Answer in a structured JSON format. <image> The room polygon is (x,z) {format_polygon_coords}. Plausible 3D coordinates (x, y,z) for the rest of the room: \n",
                ]
            elif self.args.get("no_polygon"):
                prefix_options = [
                    f"## Answer in a structured JSON format. <image> {camera_prompt} Plausible 3D coordinates (x, y,z) for the rest of the room: \n",
                ]
            else:
                prefix_options = [
                    f"{prefix}###Human: <im_start><image><im_end> \nHuman: Answer in a structured JSON format. The room polygon is (x,z) {format_polygon_coords}. {camera_prompt} Plausible 3D coordinates (x, y,z) for the rest of the room: \n",
                ]

        if self.args['mode'] == "train":
            prompt = random.choice(prefix_options) + "\n Answer: \n" + program_text + " \n###"
            text_labels = prompt
        else:    
            prompt = random.choice(prefix_options) + "\n Answer: \n"
            text_labels = prompt + program_text + " \n###"
            if self.args.get('input_polygon_test'):
                polygon_line = program_text.split("\n")[1]
                prompt +=  "\n"+polygon_line + " \n"
        

        if self.args.get("BLIP2"):
            prompt = prompt.replace("A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <im_start><image><im_end> \nHuman: ", "Question: ")
            text_labels = text_labels.replace("A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <im_start><image><im_end> \nHuman: ", "Question: ")

        # pdb.set_trace()
        image_ind = all_imgs.index(image_path)
        objs_present = all_objs[image_ind]

        # pdb.set_trace()
        if self.args.get("qa_format"):
            return [new_im_file,], img, prompt, text_labels, program_text, obj_ids_present, f"procthor_recon"
        else:
            return [new_im_file,], img, caption, prompt, text_labels, program_text, house_json, obj_ids_present
    
    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        image_paths, imgs, captions, prompts, text_labels, program_texts, house_jsons, objs_present = zip(*batch)

        if self.args.get("BLIP2"):
            inputs = self.image_processor(images=imgs[0], text=prompts[0], return_tensors="pt")
            return_dict = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                'pixel_values': inputs["pixel_values"],
                "labels": inputs["input_ids"],
                # 'qformer_input_ids': inputs["qformer_input_ids"],
                # 'qformer_attention_mask': inputs["qformer_attention_mask"],
                "prompts": prompts,
                "text_labels": text_labels,
                "program_texts": program_texts,
                "house_json": house_jsons,
                "image_lists": image_paths,
                'objs_present': objs_present,
            }
        else:
            new_images = []
            for image_b in imgs:
                for image in image_b:
                    image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                    # pdb.set_trace()
                    image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    new_images.append(image)

            pixel_values = torch.stack(new_images, dim=0)

            input_ids = []
            attention_mask = []
            for prompt in prompts:
                # pdb.set_trace()
                input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                input_ids.append(input_id)
                attention_mask.append(torch.ones_like(input_id))
            input_ids = torch.stack(input_ids, dim=0)
            attention_mask = torch.stack(attention_mask, dim=0)
            
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "program_texts": program_texts,
                "house_json": house_jsons,
                "image_lists": image_paths,
                'objs_present': objs_present,
            }
        # pdb.set_trace()
        return return_dict
    

#### all image qa real datasets

class GQA(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.batch_decode = self.tokenizer.batch_decode
        
        gqa_path = "/projectnb/ivc-ml/array/data/GQA/" 

        if args['split'] == "valtrain":
            data = json.load(open(os.path.join(gqa_path, "val_all_questions.json")))
        elif args['split'] == "val":
            data = json.load(open(os.path.join(gqa_path, "val_all_questions.json")))
        elif args['split'] == "train":
            data = json.load(open(os.path.join(gqa_path, "train_all_questions.json")))

        self.data = []
        for qid in data:
            question = data[qid]['question']
            answer = data[qid]['answer']
            image_id = data[qid]['imageId']
            im_file = os.path.join(gqa_path, "images", f"{image_id}.jpg")
            self.data.append((question, answer, im_file))

        self.data = self.data[:args['num_data_points']]

        print("Split: ", args['split'])
        print("Total number of data points ", len(self.data))


    def __getitem__(self, idx):
        question, answer, im_file = self.data[idx]

        image = Image.open(im_file).convert("RGB")

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

        image_prompt_format = "<image>"

        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. ###Assistant: \n "
        text_label = prompt + answer

        return [image,], prompt, text_label, answer, "GQA"

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        
        images, prompts, text_labels, answers, datanames = zip(*batch)

        new_images = []
        for image_b in images:
            for image in image_b:
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                # pdb.set_trace()
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)

        pixel_values = torch.stack(new_images, dim=0)

        input_ids = []
        attention_mask = []
        for prompt in prompts:
            # pdb.set_trace()
            input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids.append(input_id)
            attention_mask.append(torch.ones_like(input_id))
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)

        # pdb.set_trace()
        return_dict = {
            "input_ids": input_ids,
            'labels': input_ids,
            'text_labels': text_labels,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "prompts": prompts,
            'answers': answers,
            "dataset": datanames
        }
        # pdb.set_trace()
        return return_dict


class VQAV2(Dataset):
    def __init__(self, args, tokenizer, image_processor):

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.batch_decode = self.tokenizer.batch_decode

        self.vqa_path = "/projectnb/ivc-ml/array/data/VQA/VQAV2"

        vqa_anno = json.load(open(os.path.join(self.vqa_path, "v2_mscoco_val2014_annotations.json")))
        vqa_ques = json.load(open(os.path.join(self.vqa_path, "v2_OpenEnded_mscoco_val2014_questions.json")))

        self.data = []
        for anno_entry, ques_entry in zip(vqa_anno['annotations'], vqa_ques['questions']):
            assert anno_entry['question_id'] == ques_entry['question_id']
            question = ques_entry['question']
            answer = anno_entry['multiple_choice_answer']
            image_id = anno_entry['image_id']
            im_file = os.path.join(self.vqa_path, "val2014", f"COCO_val2014_{str(image_id).zfill(12)}.jpg")
            self.data.append((question, answer, im_file))
        
        self.data = self.data[:args['num_data_points']]

    def __getitem__(self, idx):
        question, answer, im_file = self.data[idx]

        image = Image.open(im_file).convert("RGB")

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

        image_prompt_format = "<image>"

        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. ###Assistant: \n "

        text_label = prompt + answer

        return [image,], prompt, text_label, answer, "vqav2"

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        images, prompts, text_labels, answers, datanames = zip(*batch)

        new_images = []
        for image_b in images:
            for image in image_b:
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                # pdb.set_trace()
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)

        pixel_values = torch.stack(new_images, dim=0)

        input_ids = []
        attention_mask = []
        for prompt in prompts:
            # pdb.set_trace()
            input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids.append(input_id)
            attention_mask.append(torch.ones_like(input_id))
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)

        # pdb.set_trace()
        return_dict = {
            "input_ids": input_ids,
            'labels': input_ids,
            'text_labels': text_labels,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "prompts": prompts,
            'answers': answers,
            "dataset": datanames
        }
        # pdb.set_trace()
        return return_dict
        
class OKVQA(Dataset):
    def __init__(self, args, tokenizer, image_processor):

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.batch_decode = self.tokenizer.batch_decode

        self.okvqa_path = "/projectnb/ivc-ml/array/data/VQA/OKVQA"

        okvqa_anno = json.load(open(os.path.join(self.okvqa_path, "mscoco_val2014_annotations.json")))
        okvqa_ques = json.load(open(os.path.join(self.okvqa_path, "OpenEnded_mscoco_val2014_questions.json")))

        self.data = []
        for anno_entry, ques_entry in zip(okvqa_anno['annotations'], okvqa_ques['questions']):
            assert anno_entry['question_id'] == ques_entry['question_id']
            question = ques_entry['question']
            answer = anno_entry['answers'][0]['answer']
            image_id = anno_entry['image_id']
            im_file = os.path.join(self.okvqa_path, "val2014", f"COCO_val2014_{str(image_id).zfill(12)}.jpg")
            self.data.append((question, answer, im_file))
        
        self.data = self.data[:args['num_data_points']]

    def __getitem__(self, idx):
        question, answer, im_file = self.data[idx]

        image = Image.open(im_file).convert("RGB")

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

        image_prompt_format = "<image>"

        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. ###Assistant: \n "

        text_label = prompt + answer

        return [image,], prompt, text_label, answer, "okvqa"

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        images, prompts, text_labels, answers, datanames = zip(*batch)

        new_images = []
        for image_b in images:
            for image in image_b:
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                # pdb.set_trace()
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)

        pixel_values = torch.stack(new_images, dim=0)

        input_ids = []
        attention_mask = []
        for prompt in prompts:
            # pdb.set_trace()
            input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids.append(input_id)
            attention_mask.append(torch.ones_like(input_id))
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)


        # pdb.set_trace()
        return_dict = {
            "input_ids": input_ids,
            'labels': input_ids,
            'text_labels': text_labels,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "prompts": prompts,
            'answers': answers,
            'dataset': datanames
        }
        # pdb.set_trace()
        return return_dict


class AllVQA(Dataset):
    def __init__(self, args, tokenizer, image_processor):

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.batch_decode = self.tokenizer.batch_decode

        self.gqa = GQA(args, tokenizer, image_processor)
        self.vqav2 = VQAV2(args, tokenizer, image_processor)
        self.okvqa = OKVQA(args, tokenizer, image_processor)
        
    
    def __getitem__(self, idx):
        if idx < len(self.gqa):
            return self.gqa[idx]
        elif idx < len(self.gqa) + len(self.vqav2):
            return self.vqav2[idx - len(self.gqa)]
        else:
            return self.okvqa[idx - len(self.gqa) - len(self.vqav2)]
    
    def __len__(self):
        return len(self.gqa) + len(self.vqav2) + len(self.okvqa)
    
    def collate_fn(self, batch):
        return self.gqa.collate_fn(batch)

class MMBench(Dataset):
    def __init__(self, args):
        
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def collate_fn(self, batch):
        pass

class SeedBench(Dataset):
    pass


class CVBench(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        cv_bench = load_dataset("nyu-visionx/CV-Bench")
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        if args.get("instructBLIP") or args.get("BLIP2"):
            self.batch_decode = self.image_processor.batch_decode

        self.data = cv_bench['test'].shuffle(seed=42)

        # random.shuffle(self.data)
        self.data = self.data[:args['num_data_points']]
        
        self.choice_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7} 

    def __getitem__(self, idx):
        '''
        {'idx': 0,
        'type': '2D',
        'task': 'Count',
        'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=256x256>,
        'question': 'How many organs are in the image?',
        'choices': ['3', '2', '1', '0'],
        'answer': '(C)',
        'prompt': 'How many organs are in the image? Select from the following choices.\n(A) 3\n(B) 2\n(C) 1\n(D) 0',
        'filename': 'img/2D/count/ade20k_10.png',
        'source': 'ADE20K',
        'source_dataset': 'ADE20K Validation Set',
        'source_filename': 'ADE_val_00000248.jpg',
        'target_class': None,
        'target_size': None,
        'bbox': None}
        '''
        image = self.data['image'][idx]
        question = self.data['question'][idx]

        choices = self.data['choices'][idx]
        choice_format = ", ".join(choices[:-1]) + ", or "+choices[-1]


        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        image_prompt_format = "<image>"

        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Choose between the following options: {choice_format}.###Assistant: \n "

        answer = self.data['answer'][idx]
        answer = answer.replace("(", "").replace(")", "")
        answer = choices[self.choice_to_number[answer]]

        type_task = self.data['type'][idx] + "_" + self.data['task'][idx]
        
        

        if self.args.get("instructBLIP") or self.args.get("BLIP2"):
            if self.args.get("zero_shot_mode"):
                prompt = f"{question} Choose between the following options: {choice_format}?"
            else:
                prompt = f"Question: {question} Answer the question using a single word or phrase. Choose between the following options: {choice_format}. Answer: "
        elif self.args.get("llava_ov"):
            
            """
            <|im_start|>user <image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant \nThere is a red stop sign in the image.<|im_end|><|im_start|>user <image>\nWhat about this image? How many cats do you see?<|im_end|><|im_start|>assistant\n'
            """
            image_prompt_format = "<|im_start|>user  <image>\n <|im_end|>"*1
            image_prompt_format = image_prompt_format[:-len("<|im_end|>")]
            
            prompt = f"{image_prompt_format}{question} Choose between the following options: {choice_format} <|im_end|><|im_start|>assistant: \n"
            text_label = prompt + answer
        else:
            if self.args.get("zero_shot_mode"):
                prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {choice_format}.###Assistant: \n "
            elif self.args.get("zero_shot_choice_mode"):
                prompt = self.data['prompt'][idx]
                prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: {prompt} ###Assistant: \n "
                answer = self.data['answer'][idx]
            text_label = prompt + answer
        

        
        # pdb.set_trace()
        return [image,], prompt, text_label, answer, f"cvbench_{type_task}"
        
    def __len__(self):
        return len(self.data['prompt'])
        
    def collate_fn(self, batch):
        images, prompts, text_labels, answers, datanames = zip(*batch)

        if self.args.get("llava_ov"):
            pixel_values, input_ids, attention_mask, image_sizes =  get_inputs_for_model(images, prompts, None, self.image_processor, model_choice="llava_ov")
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "image_sizes": image_sizes,
                "dataset": datanames,
                "answers": answers,
            }
        else:
            pixel_values, input_ids, attention_mask =  get_inputs_for_model(images, prompts, self.tokenizer, self.image_processor, model_choice="llava")
            # pdb.set_trace()
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "dataset": datanames,
                "answers": answers,
            }

        return return_dict


class BLINK(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode
        
        if args.get("instructBLIP") or args.get("BLIP2"):
            self.batch_decode = self.image_processor.batch_decode

        self.args = args

        dataset_name = 'BLINK-Benchmark/BLINK'

        SUBTASK_NAME = ['Multi-view_Reasoning', 'Relative_Depth', 'Spatial_Relation'] # , 'Object_Localization',]
        #SUBTASK_NAME = ['Relative_Depth', 'Spatial_Relation'] # , 'Object_Localization',]

        self.data = []
        for subtask in SUBTASK_NAME:
            count = 0
            for entry in load_dataset(dataset_name, subtask)['val']:
                self.data.append((entry, subtask))
                count += 1
                if count >= args['num_data_points']/3:
                    break

        self.choice_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7} 


    def __getitem__(self, idx):
        entry, subtask = self.data[idx]
        question = entry['prompt'].split("?")[0]+"?"
        
        
        # question = question.replace("The images are frames from a video. The video is shooting a static scene. The camera is either moving clockwise (left) or counter-clockwise (right) around the object.", "")
        
        answer = entry['answer']
        answer = answer.replace("(", "").replace(")", "")
        answer = entry['choices'][self.choice_to_number[answer]]

        if "The video is shooting a static scene. The camera is either moving clockwise" in question:
            answer_choices = ["moved "+x for x in entry['choices']]
            answer = "moved "+answer
            choice_format = ", ".join(answer_choices[:-1]) + ", or "+answer_choices[-1]
        else:    
            choice_format = ", ".join(entry['choices'][:-1]) + ", or "+entry['choices'][-1]

        images = []
        image_1 = entry['image_1']
        images.append(image_1)
        if entry['image_2'] is not None:
            image_2 = entry['image_2']
            images.append(image_2)

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        image_prompt_format = "<image>"*len(images)

        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Choose between the following options: {choice_format}.###Assistant: \n "

        if self.args.get("zero_shot_mode"):
            prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Answer the question using a single word or phrase. Choose between the following options: {choice_format}.###Assistant: \n "
        elif self.args.get("zero_shot_choice_mode"):
            prompt = entry['prompt']
            prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: {prompt} ###Assistant: \n "
            answer = entry['answer']

        text_label = prompt + answer

        if self.args.get("instructBLIP") or self.args.get("BLIP2"):
            if self.args.get("zero_shot_mode"):
                prompt = f"{question} Choose between the following options: {choice_format}?"
            else:
                prompt = f"Question: {question} Answer the question using a single word or phrase. Choose between the following options: {choice_format}. Answer: "
            text_label = prompt + answer
        elif self.args.get("llava_ov"):
            
            """
            <|im_start|>user <image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant \nThere is a red stop sign in the image.<|im_end|><|im_start|>user <image>\nWhat about this image? How many cats do you see?<|im_end|><|im_start|>assistant\n'
            """
            image_prompt_format = "<|im_start|>user  <image>\n <|im_end|>"*len(images)
            image_prompt_format = image_prompt_format[:-len("<|im_end|>")]
            
            prompt = f"{image_prompt_format}{question} Choose between the following options: {choice_format} <|im_end|><|im_start|>assistant: \n"
            text_label = prompt + answer
        

        # pdb.set_trace()
        return images, prompt, text_label, answer, "BLINK_"+subtask

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        images, prompts, text_labels, answers, datanames = zip(*batch)

        if self.args.get("llava_ov"):
            pixel_values, input_ids, attention_mask, image_sizes =  get_inputs_for_model(images, prompts, None, self.image_processor, model_choice="llava_ov")
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "image_sizes": image_sizes,
                "dataset": datanames,
                "answers": answers,
            }
        else:
            pixel_values, input_ids, attention_mask =  get_inputs_for_model(images, prompts, self.tokenizer, self.image_processor, model_choice="llava")
            # pdb.set_trace()
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'pixel_values': pixel_values,
                "labels": input_ids,
                "prompts": prompts,
                "text_labels": text_labels,
                "dataset": datanames,
                "answers": answers,
            }

        return return_dict


class AllMLMBench(Dataset):
    def __init__(self, args, tokenizer, image_processor):

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        if args.get("instructBLIP") or args.get("BLIP2"):
            self.batch_decode = self.image_processor.batch_decode

        self.all_data = [
            BLINK(args, tokenizer, image_processor),
            CVBench(args, tokenizer, image_processor),
        ]

    def __getitem__(self, idx):
        if idx < len(self.all_data[0]):
            return self.all_data[0][idx]
        else:
            return self.all_data[1][idx - len(self.all_data[0])]

    def __len__(self):
        total_len = 0
        for data in self.all_data:
            total_len += len(data)
        return total_len

    def collate_fn(self, batch):
        return self.all_data[0].collate_fn(batch)


