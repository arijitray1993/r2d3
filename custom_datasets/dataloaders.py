import json
import os
import pdb  # noqa
import random
from collections import defaultdict
import itertools
from itertools import combinations
import functools
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

from shapely.geometry.polygon import Polygon

from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import yaml

import ast
import cv2

import sys
# sys.path.append("/projectnb/ivc-ml/array/research/robotics/dreamworlds/models/LLaVA")
sys.path.append("/projectnb/ivc-ml/array/research/robotics/LLaVA")
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
from utils.ai2thor_utils import generate_program_from_roomjson, format_program, generate_attribute_program_from_roomjson

import numpy as np

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
    room_polygon_norm = [(point[0] - cam_pos[0], point[2] - cam_pos[1]) for point in room_polygon]
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
        norm_location = (location[0] - cam_pos[0], location[1], location[2] - cam_pos[1])
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
        norm_location = (location[0] - cam_pos[0], location[1], location[2] - cam_pos[1])
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
        window_position_norm = (window_position[0] - cam_pos[0], window_position[1], window_position[2] - cam_pos[1])
        window_position_norm = (window_position[0]*np.cos(np.deg2rad(cam_rot)) - window_position[2]*np.sin(np.deg2rad(cam_rot)), window_position[0]*np.sin(np.deg2rad(cam_rot)) + window_position[2]*np.cos(np.deg2rad(cam_rot)))
        room_json[f'window_{i}'] = (window_token, (int(window_position[0]), int(window_position[1]), int(window_position[2])), window_polygon, window_wall)
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
        self.batch_decode = self.tokenizer.batch_decode

        self.procthor_data = ProcTHOR_reasoning(args, tokenizer, image_processor)
        self.llava_data = LLaVAInstructTune(args, tokenizer, image_processor)
        self.proc_len = len(self.procthor_data)
        self.llava_len = len(self.llava_data)

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
        
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "labels": input_ids,
            "prompts": prompts,
            "text_labels": text_labels,
            "datanames": datanames,
            "answers": answers,
        }

        return return_dict


class LLaVAInstructTune(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
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
        }

        return return_dict


class ProcTHOR_reasoning(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

        nav_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas.json'
        spatial_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_spatial_qas.json'

        nav_data = json.load(open(nav_qa_json_path))
        spatial_data = json.load(open(spatial_qa_json_path))

        self.data = []
        if args.get("split") == "train":
            for house_ind, cam_pos, cam_rot, qa_entries in nav_data[:int(len(nav_data)*0.8)]:
                self.data.extend(qa_entries)
            for house_ind, cam_pos, cam_rot, qa_entries in spatial_data[:int(len(spatial_data)*0.8)]:
                self.data.extend(qa_entries)
        elif args.get("split") == "valtrain":
            for house_ind, cam_pos, cam_rot, qa_entries in nav_data[int(len(nav_data)*0.8):int(len(nav_data)*0.9)]:
                self.data.extend(qa_entries)
            for house_ind, cam_pos, cam_rot, qa_entries in spatial_data[int(len(spatial_data)*0.8):int(len(spatial_data)*0.9)]:
                self.data.extend(qa_entries)
        elif args.get("split") == "val":
            for house_ind, cam_pos, cam_rot, qa_entries in nav_data[int(len(nav_data)*0.8):]:
                self.data.extend(qa_entries)
            for house_ind, cam_pos, cam_rot, qa_entries in spatial_data[int(len(spatial_data)*0):]:
                self.data.extend(qa_entries)
        
        random.shuffle(self.data)
        print("Total number of data points: ", len(self.data))


    def __getitem__(self, idx):
        qa_entry = self.data[idx]
        
        question, image_order, answer_choices = qa_entry

        # judge the question type
        question_type = "other"
        if "how many" in question.lower():
            question_type = "count"
        
        if "did i likely move" in question.lower():
            question_type = "action_sequence"

        if ("if i" in question.lower() and "the camera?" in question.lower()):
            question_type = "obj_action_movement"

        if "would i be moving more" in question.lower():
            question_type = "relative_obj_action_movement"

        if "object is closer to the camera" in question.lower():
            question_type = "obj_depth"
        
        if "consider the relative distances" in question.lower():
            question_type = "obj_positions"


        correct_answer = answer_choices[0]
        
        hard_distractor = None
        if question_type == "action_sequence":
            hard_distractor = answer_choices[2]
        
        
        if hard_distractor is not None:
            ans_choice_order = ['"'+correct_answer+'"', '"'+answer_choices[1]+'"', '"'+hard_distractor+'"']
            random.shuffle(ans_choice_order)
            answer_choices_format = " or ".join(ans_choice_order)
            question_type+="_hard"
        else:
            ans_choice_order = answer_choices
            ans_choice_order = ['"'+ans+'"' for ans in ans_choice_order]
            random.shuffle(ans_choice_order)
            answer_choices_format = " or ".join(ans_choice_order)
        
        image_prompt_format = "<image>"*len(image_order)

        if self.args['mode'] == "zero_shot":
            '''
            A chat between a curious human and an artificial intelligence assistant. 
            The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <im_start><image><im_end>
            Human: Describe the image and color details.###Assistant:
            The image features a wooden pier extending out into a large body of water, possibly a lake or a bay. The pier is surrounded by a serene environment, with trees and mountains in the background. The water appears to be calm and clear, making it an ideal spot for relaxation or leisurely activities. The scene is captured in black and white, giving it a timeless and classic feel.
            '''
            prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
            prompt = f"{prefix}.###Human: <im_start>{image_prompt_format}<im_end> \nHuman: {question} Choose between the following options: {answer_choices_format}.###Assistant: \n "
            text_label = prompt + correct_answer + " \n###"

        else:
            prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
            if self.args['mode'] == "train":
                prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Choose between the following options: {answer_choices_format}.###Assistant: \n {correct_answer} \n###"
                text_label = prompt
            else:
                prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Choose between the following options: {answer_choices_format}.###Assistant: \n" 
                text_label = prompt + correct_answer + " \n###"

        images = [Image.open(img).convert("RGB") for img in image_order]

        # pdb.set_trace()

        return image_order, images, prompt, text_label, correct_answer, answer_choices, f"procthor_reasoning_{question_type}"

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


class ProcTHOR_image_camposition_marked(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        if tokenizer is not None:
            self.batch_decode = self.tokenizer.batch_decode

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

        cam_pos = (int(round(pos['x'],2)*100), int(round(pos['z'],2)*100))
        cam_angle = int(rot['y'])
        
        
        if self.args.get("normalize_rotation"):
            program_text = normalize_coords(program_text, cam_pos, cam_angle, attr=self.args.get("use_attributes"))
            cam_pos = (0, 95, 0)
            cam_angle = 0

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

        # mark furthest object in the image
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


            if self.args['split']!= "train":
                if new_im_file != "":
                    img[0].save(new_im_file)
                else:
                    new_im_file = image_path

            if self.args.get("captiononly_baseline"):
                camera_prompt = ""
            elif self.args.get("use_multimarks"):
                camera_prompt = f"Image taken from (x,z) {cam_pos}."
                if self.args.get("use_angle"):
                    camera_prompt = f"Image taken from (x,z) {cam_pos} with angle around y as {cam_angle}."
                for mi, (_, obj_pos) in enumerate(marked_points):
                    camera_prompt += f" The red {mi} mark in the image is at 3D coordinate (x, y, z) {obj_pos}."
            else:
                camera_prompt = f"Image taken from (x,z) {cam_pos} looking inside the polygon. The red circular 1 mark in the image is at 3D coordinate (x, y, z) {max_dist_obj_pos}. "
                if self.args.get("use_angle"):
                    camera_prompt = f"Image taken from (x,z) {cam_pos} with angle around y as {cam_angle} looking inside the polygon. The red circular 1 mark in the image is at 3D coordinate (x, y, z) {max_dist_obj_pos}. "
                if self.args.get("use_no_mark_baseline") or self.args.get("use_depth_yuv_no_point"):
                    camera_prompt = f"Image taken from (x,z) {cam_pos}."
                    if self.args.get("use_angle"):
                        camera_prompt = f"Image taken from (x,z) {cam_pos} with angle around y as {cam_angle}."

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
                camera_prompt = f"Image taken from (x,z) {cam_pos}. No object in the image is present in the yaml file."
                if self.args.get("use_angle"):
                    camera_prompt = f"Image taken from (x,z) {cam_pos} with angle around y as {cam_angle}. No object in the image is present in the yaml file."
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
You are at ({cam_pos[0]}, 95, {cam_pos[1]}). \
The red dot marked in the image is at {max_dist_obj_pos}. Using this information, can you roughly estimate the locations of all the objects in the room? Just try to the best of your abilities. Objects to the left of the dot will have a lower x. Objects closer to the camera from the dot will have a lower z. And, objects lower in height from the camera height will have a lower y.  \
Please output each object along with the location on each line with no other text in the following format: \n Object name at location (x, y, z). eg, Brown bed at (10, 15, 20) \
\n Answer: \n",
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
            

        # pdb.set_trace()
        image_ind = all_imgs.index(image_path)
        objs_present = all_objs[image_ind]

        # pdb.set_trace()
        return [new_im_file,], img, caption, prompt, text_labels, program_text, house_json, objs_present
    
    def __len__(self):
        return len(self.data)

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
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        
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
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        self.batch_decode = self.tokenizer.batch_decode

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
        
        text_label = prompt + answer
        return [image,], prompt, text_label, answer, f"cvbench_{type_task}"
        
    def __len__(self):
        return len(self.data['prompt'])
        
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
            "attention_mask": attention_mask,
            'pixel_values': pixel_values,
            "prompts": prompts,
            'answers': answers,
            'dataset': datanames,
            'labels': input_ids,

            "text_labels": text_labels
        }
        # pdb.set_trace()
        return return_dict


class BLINK(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.batch_decode = self.tokenizer.batch_decode

        dataset_name = 'BLINK-Benchmark/BLINK'

        SUBTASK_NAME = ['Multi-view_Reasoning', 'Relative_Depth', 'Spatial_Relation'] # , 'Object_Localization',]

        self.data = []
        for subtask in SUBTASK_NAME:
            count = 0
            for entry in load_dataset(dataset_name, subtask)['val']:
                self.data.append((entry, subtask))
                count += 1
                if count >= args['num_data_points']/len(SUBTASK_NAME):
                    break

        self.choice_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7} 


    def __getitem__(self, idx):
        entry, subtask = self.data[idx]
        question = entry['prompt'].split("?")[0]+"?"
        # prompt = entry['prompt']
        

        answer = entry['answer']
        answer = answer.replace("(", "").replace(")", "")
        answer = entry['choices'][self.choice_to_number[answer]]

        images = []
        image_1 = entry['image_1']
        images.append(image_1)
        if entry['image_2'] is not None:
            image_2 = entry['image_2']
            images.append(image_2)

        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
        image_prompt_format = "<image>"*len(images)

        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {question} Choose between the following options: {choice_format}.###Assistant: \n "
        text_label = prompt + answer
        
        return images, prompt, text_label, answer, "BLINK_"+subtask

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        images, prompts, text_labels, answers, subtasks = zip(*batch)

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
            'dataset': subtasks,
        }
        # pdb.set_trace()
        return return_dict


class AllMLMBench(Dataset):
    def __init__(self, args, tokenizer, image_processor):

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.batch_decode = self.tokenizer.batch_decode

        self.all_data =[
            CVBench(args, tokenizer, image_processor),
            BLINK(args, tokenizer, image_processor),
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

