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
from utils.ai2thor_utils import generate_program_from_roomjson, format_program

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


class LSUNBedrooms(Dataset):
    def __init__(self, args):
        self.args = args

        self.tokenizer, _, self.image_processor, self.context_len = load_pretrained_model(
                model_path=args['model_path'],
                model_base=None,
                model_name=get_model_name_from_path(args['model_path'])
        )
        self.batch_decode = self.tokenizer.batch_decode

        self.dataset = load_dataset("pcuenq/lsun-bedrooms")[args["split"]]


    def __getitem__(self, idx):
        bedroom_im = self.dataset[idx]["image"]

        num_corner = 4
        prefix_text = f"## <image> . The room polygon is [(961, 0), (961, 576), (1345, 576), (1345, 0)]. The 3D specifications: \n"
    
        return [bedroom_im, ], prefix_text
    
    def __len__(self):
        return len(self.dataset)
    
    def collate_fn(self, batch):
        images, prompts = zip(*batch)

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
            image = image[0]
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
            "text_labels": prompts,
            "image_lists": images,
        }

        return return_dict


class ADE20K(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args

        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        self.batch_decode = self.tokenizer.batch_decode
        
        self.ade_image_folder_path = "/projectnb/ivc-ml/array/data/ADE20K_2021_17_01/images/ADE/validation/home_or_hotel"

        self.seg_images = []
        for folder in os.listdir(self.ade_image_folder_path):
            for im_file in os.listdir(os.path.join(self.ade_image_folder_path, folder)):
                self.seg_images.append(os.path.join(folder, im_file))

        # keep only the _seg images
        self.seg_images = [x for x in self.seg_images if "_seg" in x]

        self.ade_color_to_obj = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/ade_color_to_obj.json"))
        _, _, self.obj_to_color = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/ade_to_ai2thor.json"))
        # pdb.set_trace()

    def __getitem__(self, idx):
        seg_image = self.seg_images[idx]

        segmentation_frame = Image.open(os.path.join(self.ade_image_folder_path, seg_image)).convert("RGB")
        '''
        segmentation_frame = np.array(segmentation_frame)
        
        all_colors = segmentation_frame.reshape(-1, 3)
        unique_colors = set([tuple(color) for color in all_colors])

        # pdb.set_trace()
        for color in unique_colors:
            if str(color) in self.ade_color_to_obj:
                ade_obj_name = self.ade_color_to_obj[str(color)]
            else: 
                print(str(color))
                print("couldnt find object")
                ade_obj_name = "wall"
            try:
                new_color = self.obj_to_color[ade_obj_name]
            except:
                print("couldnt find mapping to color used in training")
                new_color = self.obj_to_color["wall"]
            
            # repeat new colors the number of times color appears
            new_color = np.array(new_color)
            new_color = np.tile(new_color, (np.sum(np.all(all_colors == color, axis=1)), 1))

            # add a bit of random gaussian noise
            if self.args.get('add_noise'):
                if self.args['split']=="train":
                    # new_color = new_color + np.random.normal(0, random.choice([2,4,6,8,10]), new_color.shape)
                    new_color = new_color + np.random.normal(0, 4, new_color.shape)
                else:
                    new_color = new_color + np.random.normal(0, 4, new_color.shape)
                
            all_colors[np.all(all_colors == color, axis=1)] = new_color
        
        image = all_colors.reshape(segmentation_frame.shape)
        image = Image.fromarray(image)
        # pdb.set_trace()
        '''
        image = segmentation_frame 
        caption = ""

        prefix_text = f"## <image> . The room polygon is [(961, 0), (961, 576), (1345, 576), (1345, 0)]. The 3D specifications: \n"
    
        return image, prefix_text, [os.path.join(self.ade_image_folder_path, seg_image),]

    def __len__(self):
        return len(self.seg_images)
    
    def collate_fn(self, batch):
        images, prompts, image_list = zip(*batch)

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
            #if self.args.get('add_noise'):
            #    image = image + np.random.normal(0, 4, image.shape)
            # pdb.set_trace()
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            new_images.append(image)
        
        pixel_values = torch.stack(new_images, dim=0)

        # pixel_values = self.image_processor(images, return_tensors="pt")['pixel_values']
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
            "pixel_values": pixel_values,
            "text_labels": prompts,
            "image_lists": image_list,
        }

        return return_dict


class AnyImageCaption(Dataset):
    def __init__(self, args):
        self.tokenizer, _, self.image_processor, self.context_len = load_pretrained_model(
                model_path=args['model_path'],
                model_base=None,
                model_name=get_model_name_from_path(args['model_path'])
        )
        self.batch_decode = self.tokenizer.batch_decode

        self.image_data_path = args['image_data_path']
        if args['caption_data_path'] is not None:
            self.caption_data_path = args['caption_data_path']

        self.data = []
        for image_path in os.listdir(self.image_data_path):
            if args['caption_data_path'] is not None:
                caption_path = os.path.join(self.caption_data_path, image_path.split(".")[0] + ".txt")
                if not os.path.exists(caption_path):
                    continue
                with open(caption_path, "r") as f:
                    caption = f.read()
                
                self.data.append((image_path, caption))
            else:
                self.data.append(image_path)

        self.args = args

    def __getitem__(self, idx):
        if self.args['caption_data_path'] is not None:
            image_path, caption = self.data[idx]
        else:
            image_path = self.data[idx]
            caption = ""

        image_path = os.path.join(self.image_data_path, image_path)
        image = Image.open(image_path).convert("RGB")

        num_corner = 4
        if caption == "":
            prefix_text = f"## <image> \n Write code for an interactive room with {num_corner} corners that looks like the image. \n Answer: "
        else:
            prefix_text = f"## <image> \n Write code for an interactive room with {num_corner} corners that looks like the image with the description: {caption}. \n Answer: "
       
        return image, caption, prefix_text
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        images, captions, prompts = zip(*batch)

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
            "text_labels": captions,
        }

        return return_dict



class ProcTHOR_failure_refine(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        self.batch_decode = self.tokenizer.batch_decode

        caption_data = json.load(
            open(
                "/projectnb/ivc-ml/array/research/robotics/dreamworlds/"
                + "custom_datasets/procThor/all_room_json_programs_ai2_train_room_captions_gtobjonly_new.json", "r"
                )
            )

        # get the data only for which we have captions
        json_program_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_train.json"))
        json_data = []
        for ind, all_room_captions in caption_data:
            json_data.append((json_program_data[ind], all_room_captions))

        # pdb.set_trace()
        self.failure_data = json.load(open(self.args.get('failure_data_path')))

        self.image_data_path = "/projectnb/ivc-ml/array/research/robotics/ProcTHOR/"
        # only use the json data we have images for
        self.data = []
        self.summary_data = []
        for ind, (room_data, room_captions) in enumerate(json_data):
            all_imgs = room_data[4]
            if len(all_imgs) < 2:
                continue
            if len(room_captions) < 2:
                continue
            if not os.path.exists(all_imgs[0]):
                continue
            self.data.append((room_data, room_captions))
        
        print("Total number of data points: ", len(self.data))

        # split into train and test
        if args["split"] == "train":
            self.data = self.data[:int(len(self.data) * 0.8)]
            self.split = "train"
        elif args["split"] == "valtrain":
            self.data = self.data[int(len(self.data) * 0.8):int(len(self.data) * 0.9)]
            self.split = "val"
        elif args["split"] == "val":
            self.data = self.data[int(len(self.data) * 0.9):]
            self.split = "test"

        print(f"Total number of data in {args['split']}: ", len(self.data))

        self.closest_ai2thor_to_ade, self.ai2assetname_to_objname, self.obj_to_color = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/ade_to_ai2thor.json"))
    
    def __getitem__(self, idx):
        # randomly choose og data or failure data
        if random.random() < 0.25:
            room_data, all_room_captions = self.data[idx]
            program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name = room_data
            program_text = generate_program_from_roomjson(house_json)

            image_path = random.sample(all_seg_frames, 1)
            caption_image  = random.choice(all_room_captions)
            image = Image.open(image_path[0]).convert("RGB")
            # segmentation_frame = np.array(image)
            # new_segmentation_frame = convert_panoptic_to_ade_colors(segmentation_frame, color_to_objid, obj_id_to_name, self.ai2assetname_to_objname, self.closest_ai2thor_to_ade, self.obj_to_color)

            caption = caption_image[-1]

            num_corner = len(all_imgs) # since we have an image from each corner.
            
            prefix_options = [
                f"## <image> An interactive room like the image can be rendered as: \n",
                f"## <image> Can you render a room with {num_corner} corners that looks like the image? Surely: \n",
                f"## <image> A room with {num_corner} corners with description: {caption}. The yaml is: \n",
                f"## <image> Can you render a room with that looks like the image? Surely: \n",
                f"## <image> {caption}. Generate such a room with {num_corner} corners as: \n",
                f"## <image> {caption}. The yaml for the room with {num_corner} corners: \n ",
                f"## <image> {caption}. We can generate it as: \n",
                f"## <image> {caption}. How can I render it? Like this: \n",
            ]

            if self.args['mode'] in ["val", "valtrain"]:
                prefix_text = prefix_options[8]
            else:
                prefix_text = random.choice(prefix_options)
        
            if self.args['mode'] == "train":
                prompt = prefix_text + program_text + " \n###"
                text_labels = prompt
            else:
                prompt = prefix_text
                text_labels = prefix_text + program_text + " \n###"

            # pdb.set_trace()
            
            return [image,], caption, prompt, text_labels, [image_path[0],], house_json

        else:
            failure_sentence, input_im, current_im = self.failure_data[idx%len(self.failure_data)]
            image1 = Image.open(input_im).convert("RGB")

            # convert from panoptic to ade
            # segmentation_frame = Image.open(current_im).convert("RGB")
            # new_segmentation_frame = convert_panoptic_to_ade_colors(segmentation_frame, color_to_objid, obj_id_to_name, self.ai2assetname_to_objname, self.closest_ai2thor_to_ade, self.obj_to_color)

            image2 = Image.open(current_im).convert("RGB")
            prompt = f"# {failure_sentence} \n###"

            if self.args['mode'] == "train":
                text_labels = prompt
            else:
                text_labels = prompt
                prompt = prompt.split("refined program would be: ")[0] 
            # pdb.set_trace()
            return [image1, image2], failure_sentence, prompt, text_labels, [input_im, current_im], 0

    def __len__(self,):
        return len(self.data)

    def collate_fn(self, batch):
        
        images_batch, captions, prompts, text_labels, image_file_list, house_jsons = zip(*batch)

        input_ids = []
        attention_mask = []
        for prompt in prompts:
            input_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            # chop off at 1400 len
            if len(input_id) > 1300:
                input_id = input_id[:1300]
            input_ids.append(input_id)
            attention_mask.append   (torch.ones_like(input_id))
        input_ids = torch.stack(input_ids, dim=0)   

        attention_mask = torch.stack(attention_mask, dim=0)

        new_images = []
        for images in images_batch:
            for image in images:
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)

        pixel_values = torch.stack(new_images, dim=0)

        # pdb.set_trace()
        return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids,
                "pixel_values": pixel_values,
                "program_texts": prompts,
                "text_labels": text_labels,
                "image_lists": image_file_list,
                'house_json': house_jsons,
            }
        
        return return_dict


class ProcTHOR_multiview(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        self.batch_decode = self.tokenizer.batch_decode
        
        multiview_json_data_split_1 = json.load(
            open(
                "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_densemultiview.json", "r"
            )
        )

        multiview_json_data_split_2 = json.load(
            open(
                "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_densemultiview_split2.json", "r"
            )
        )

        self.data = []
        for program_text, house_json, og_house_json, cam_ind_to_position, all_imgs in multiview_json_data_split_1 + multiview_json_data_split_2:
            if len(all_imgs) < 2:
                continue
            self.data.append((program_text, house_json, og_house_json, cam_ind_to_position, all_imgs))

        # self.data = multiview_json_data_split_1 + multiview_json_data_split_2

        self.image_data_path = "/projectnb/ivc-ml/array/research/robotics/ProcTHOR/denseviews/images"

        if args["split"] == "train":
            self.data = self.data[:int(len(self.data) * 0.9)]
            self.split = "train"
        elif args["split"] == "valtrain":
            self.data = self.data[int(len(self.data) * 0.8):int(len(self.data) * 0.9)]
            self.split = "val"
        elif args["split"] == "val":
            self.data = self.data[int(len(self.data) * 0.9):]
            self.split = "test"

        print("Total number of data points: ", len(self.data))

    def __getitem__(self, idx):
        program_text, house_json, og_house_json, cam_ind_to_position, all_imgs = self.data[idx]

        program_text = generate_program_from_roomjson(house_json, include_children=self.args.get('include_children'))

        if self.args.mode == "train":    
            num_images_to_use = random.choice([10, 15, 20, 40, 50, 100, 75, 25, 80, 90])
            image_choices = random.sample(all_imgs, min(len(all_imgs), num_images_to_use))
        else:
            num_images_to_use = 10
            image_choices = random.sample(all_imgs, min(len(all_imgs), num_images_to_use))

        camera_poses = []
        for image_path in image_choices:
            position_ind = image_path.split("_")[-1].split(".")[0]
            position, rotation = cam_ind_to_position[position_ind]
            camera_poses.append((position['x'], position['z'], rotation['y']))

        all_images = [Image.open(image_path).convert("RGB") for image_path in image_choices]

        room = house_json['rooms'][0]
        polygon = room['floorPolygon']
        polygon_coords = [(point['x'], point['z']) for point in polygon]
        room_polygon = Polygon(polygon_coords)
        simplified_polygon = room_polygon.simplify(0.001)
        format_polygon_coords = []
        for point in list(simplified_polygon.exterior.coords):
            format_polygon_coords.append((int(round(point[0],2)*100), int(round(point[1],2)*100)))
        
        format_polygon_coords = str(format_polygon_coords[:-1])

        prefix = f"## {'<image>'*len(image_choices)} The room polygon is {format_polygon_coords}. List the 3D specifications: #room \n"

        if self.args['mode'] == "train":
            prompt = prefix + program_text + " \n###"
            text_labels = prompt
        else:
            prompt = prefix
            text_labels = prefix + program_text + " \n###"

        # pdb.set_trace()
        return all_images, prompt, text_labels, image_choices, house_json, camera_poses

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        images_batch, prompts, text_labels, image_paths, house_jsons, camera_poses = zip(*batch)

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
        for images in images_batch:
            for image in images:
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                # pdb.set_trace()
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)

        pixel_values = torch.stack(new_images, dim=0)

        camera_pos_inp = torch.tensor(camera_poses, dtype=torch.float32)

        # pdb.set_trace()
        # pixel_values = self.image_processor(images, return_tensors="pt")['pixel_values']
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
            "pixel_values": pixel_values,
            "program_texts": prompts,
            "text_labels": text_labels,
            "image_lists": image_paths,
            'house_json': house_jsons,
            'camera_pos': camera_pos_inp,
        }
        # pdb.set_trace()
        return return_dict
        



class ProcTHOR_caption_only(Dataset):
    def __init__(self, args, tokenizer, image_processor=None):
        self.args = args
        self.tokenizer = tokenizer
        self.batch_decode = self.tokenizer.batch_decode

        json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/final_data_neurips.json"))

        if self.args.get("use_topdown"):
            caption_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GPT4V_room_descriptions_topdown.json"))
        else:
            caption_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GPT4V_room_descriptions.json"))

        self.apartment_ind_to_caption = {}
        for apartment_ind, image_file, caption in caption_data:
            self.apartment_ind_to_caption[apartment_ind] = (image_file, caption)

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
        

        if args["split"] == "train":
            self.data = self.data[:11000]
            self.split = "train"
        elif args["split"] == "valtrain":
            self.data = self.data[11000:11100]
            self.split = "val"
        elif args["split"] == "val":
            self.data = self.data[-100:]
            self.split = "test"
    
        print("Total number of data points: ", len(self.data))

    def __getitem__(self, idx):
        room_data = self.data[idx]

        program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name = room_data
        program_text = generate_program_from_roomjson(house_json, include_children=self.args.get('include_children'))

        apartment_id = all_imgs[0].split("/")[-2]
        image_path, caption = self.apartment_ind_to_caption[apartment_id]

        # image = Image.open(image_path).convert("RGB")

        room = house_json['rooms'][0]
        polygon = room['floorPolygon']
        polygon_coords = [(point['x'], point['z']) for point in polygon]
        # pdb.set_trace()
        room_polygon = Polygon(polygon_coords)
        
        format_polygon_coords_num = []
        for point in list(room_polygon.exterior.coords):
            format_polygon_coords_num.append((int(round(point[0],2)*100), int(round(point[1],2)*100)))
        
        format_polygon_coords = str(format_polygon_coords_num[:-1])


        prefix = [f"## Render a room with a layout polygon {format_polygon_coords}. {caption}. List the exact 3D specifications: \n",
                f"## Generate a room with a layout polygon {format_polygon_coords}. {caption}. The 3D specifications for such a room: \n",
                f"## {caption}. The room polygon is {format_polygon_coords}. The 3D specifications: \n",
            ]
        
        if self.args['mode'] == "train":
            prompt = random.choice(prefix) + program_text + " \n###"
            text_labels = prompt
        else:
            prompt = random.choice(prefix)
            text_labels = prompt + program_text + " \n###"
        
        return caption, prompt, text_labels, program_text, house_json
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):

        captions, prompts, text_labels, program_texts, house_jsons = zip(*batch)

        if self.args.get("codellama"):
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        else:
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
            "labels": input_ids,
            "prompts": prompts,
            "text_labels": text_labels,
            "program_texts": program_texts,
            "house_json": house_jsons,
            "caption": captions,
        }

        return return_dict

class ProcTHOR_image_only(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        self.batch_decode = self.tokenizer.batch_decode
        

        json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/final_data_neurips.json"))

        caption_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GPT4V_room_descriptions.json"))

        self.apartment_ind_to_caption = {}
        for apartment_ind, image_file, caption in caption_data:
            self.apartment_ind_to_caption[apartment_ind] = (image_file, caption)

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

        if args["split"] == "train":
            self.data = self.data[:11000]
            self.split = "train"
        elif args["split"] == "valtrain":
            self.data = self.data[11000:11100]
            self.split = "val"
        elif args["split"] == "val":
            self.data = self.data[-100:]
            self.split = "test"
    
        print("Total number of data points: ", len(self.data))
        
    def __getitem__(self, idx):
        room_data = self.data[idx]

        program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name = room_data
        program_text = generate_program_from_roomjson(house_json, include_children=self.args.get('include_children'))

        apartment_id = all_imgs[0].split("/")[-2]
        image_path, caption = self.apartment_ind_to_caption[apartment_id]

        room = house_json['rooms'][0]
        polygon = room['floorPolygon']
        polygon_coords = [(point['x'], point['z']) for point in polygon]
        # pdb.set_trace()
        room_polygon = Polygon(polygon_coords)

        format_polygon_coords_num = []
        for point in list(room_polygon.exterior.coords):
            format_polygon_coords_num.append((int(round(point[0],2)*100), int(round(point[1],2)*100)))
        
        format_polygon_coords = str(format_polygon_coords_num)

        corner_ind = image_path.split("/")[-1].split(".")[0]

        # pdb.set_trace()
        camera_poses = cam_ind_to_position[corner_ind]

        pos, rot = camera_poses

        cam_pos = (int(round(pos['x'],2)*100), int(round(pos['z'],2)*100))
        cam_angle = int(rot['y'])
        
        prefix_options = [
            f"## <image> The room polygon is {format_polygon_coords}. The image was taken from camera position {cam_pos} with an angle {cam_angle}. A plausible 3d specification for the room: \n",
            f"## <image> The room polygon is {format_polygon_coords}. The image was taken from camera position {cam_pos} with an angle {cam_angle}. The 3D specifications could be: \n",
            f"## <image> The room polygon is {format_polygon_coords}. The image was taken from camera position {cam_pos} with an angle {cam_angle}. The 3D specifications might be: \n",
        ]

        if self.args['mode'] == "train":
            prompt = random.choice(prefix_options) + program_text + " \n###"
            text_labels = prompt
        else:
            prompt = random.choice(prefix_options)
            text_labels = prompt + program_text + " \n###"

        image_ind = all_imgs.index(image_path)
        objs_present = all_objs[image_ind]


        # pdb.set_trace()
        return image_path, caption, prompt, text_labels, program_text, house_json, objs_present
    
    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        image_paths, captions, prompts, text_labels, program_texts, house_jsons, objs_present = zip(*batch)

        new_images = []
        for image in image_paths:
            image = Image.open(image).convert("RGB")
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
            "image_paths": image_paths,
            'objs_present': objs_present,
        }

        return return_dict

class Mix_Procthor_ImageCamPos_VLMBench(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def collate_fn(self, batch):
        pass


class ProcTHOR_image_camposition_marked(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        self.batch_decode = self.tokenizer.batch_decode

        json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/final_data_neurips.json"))

        #if self.args.get("use_topdown"):
        # topdown_caption_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GPT4V_room_descriptions_topdown.json"))
        # else:
        caption_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GPT4V_room_descriptions.json"))

        self.apartment_ind_to_caption = {}
        for apartment_ind, image_file, caption in caption_data:
            #pdb.set_trace()
            # if apartment_ind in caption_data:
            self.apartment_ind_to_caption[apartment_ind] = (image_file, caption)

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
            self.data = self.data[-100:]
            self.split = "test"
    
        print("Total number of data points in split: ", len(self.data))
        
    def __getitem__(self, idx):
        room_data = self.data[idx]

        program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name = room_data
        program_text = generate_program_from_roomjson(house_json, include_children=self.args.get('include_children'))

        apartment_id = all_imgs[0].split("/")[-2]
        image_path, caption = self.apartment_ind_to_caption[apartment_id]

        room = house_json['rooms'][0]
        polygon = room['floorPolygon']
        polygon_coords = [(point['x'], point['z']) for point in polygon]
        # pdb.set_trace()
        room_polygon = Polygon(polygon_coords)

        format_polygon_coords_num = []
        for point in list(room_polygon.exterior.coords):
            format_polygon_coords_num.append((int(round(point[0],2)*100), int(round(point[1],2)*100)))
        
        format_polygon_coords = str(format_polygon_coords_num)

        corner_ind = image_path.split("/")[-1].split(".")[0]

        # pdb.set_trace()
        camera_poses = cam_ind_to_position[corner_ind]

        pos, rot = camera_poses

        cam_pos = (int(round(pos['x'],2)*100), int(round(pos['z'],2)*100))
        cam_angle = int(rot['y'])
        
        
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
                obj_pos = obj[1]
                dist_to_camera = np.sqrt((obj_pos[0] - cam_pos[0])**2 + (obj_pos[2] - cam_pos[1])**2)

                if self.args.get("randomize_point"):
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
        
        
        objid_to_color = {}
        for color in color_to_objid:
            objid_to_color[color_to_objid[color]] = color

        # pdb.set_trace()

        # mark furthest object in the image
        if max_dist_obj_name is not None:
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
            else:
                camera_prompt = f"Image taken from (x,z) {cam_pos} looking inside the polygon. The red circular 1 mark in the image is at 3D coordinate (x, y, z) {max_dist_obj_pos}. "
                if self.args.get("use_no_mark_baseline") or self.args.get("use_depth_yuv_no_point"):
                    camera_prompt = f"Image taken from (x,z) {cam_pos}."
            
        else:
            if self.args.get("captiononly_baseline"):
                camera_prompt = ""
                new_im_file = image_path
                img = []
            else:
                new_im_file = image_path
                img = [Image.open(image_path).convert("RGB"),]
                camera_prompt = f"Image taken from (x,z) {cam_pos}. No object in the image is present in the yaml file."
                if self.args.get("use_no_mark_baseline") or self.args.get("use_depth_yuv_no_point"):
                    camera_prompt = f"Image taken from (x,z) {cam_pos}."


        if self.args.get("use_caption"):
            prefix_options = [
                f"## <image> Describe the image: {caption} If the room polygon is (x,z) {format_polygon_coords}. {camera_prompt} Plausible 3D coordinates (x, y,z) for the rest of the room: \n",
            ]
        elif self.args.get("captiononly_baseline"):
            prefix_options = [
                f"## The room polygon is (x,z) {format_polygon_coords}. {caption} Plausible 3D coordinates (x, y,z) for the room: \n",
            ]
        else:
            if self.args.get("no_camera_prompt"):
                prefix_options = [
                    f"## <image> The room polygon is (x,z) {format_polygon_coords}. Plausible 3D coordinates (x, y,z) for the rest of the room: \n",
                ]
            elif self.args.get("no_polygon"):
                prefix_options = [
                    f"## <image> {camera_prompt} Plausible 3D coordinates (x, y,z) for the rest of the room: \n",
                ]
            else:
                prefix_options = [
                    f"## <image> The room polygon is (x,z) {format_polygon_coords}. {camera_prompt} Plausible 3D coordinates (x, y,z) for the rest of the room: \n",
                ]

        if self.args['mode'] == "train":
            prompt = random.choice(prefix_options) + program_text + " \n###"
            text_labels = prompt
        else:
            prompt = random.choice(prefix_options)
            text_labels = prompt + program_text + " \n###"

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

        return return_dict
    


class ProcTHOR_image_caption(Dataset):
    def __init__(self, args, tokenizer, image_processor):
        self.args = args
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor 
        self.batch_decode = self.tokenizer.batch_decode

        # get the data only for which we have captions
        #if args.get('use_14k'):
        # json_program_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_train_childrenadded_all_14k.json"))
        json_program_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/final_data_neurips.json"))
        caption_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/GPT4V_room_descriptions.json"))

        self.apartment_ind_to_caption = {}
        for apartment_id, im, cap in caption_data:
            self.apartment_ind_to_caption[apartment_id] = (im, cap)

        #else:
        #    json_program_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/procthor_roomjson_programs_imgs_train_childrenadded_all.json"))
        # use generate_cfg_data_ProcTHOR.py and all generate_cfg_data_ProcTHOR_split*.py and then collate_room_json_data.py to generate this.
        
        if self.args.get('use_top_down') or self.args.get('use_top_down_seg') or self.args.get('use_top_down_normal') or self.args.get('top_down_pretrained'):
            json_data = json_program_data
        elif self.args.get('use_all_data'):
            json_data = json_program_data
        else:
            json_data = json_program_data

        self.image_data_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/images"
        
        # have only the json data we have top down for
        topdown_data = []
        for entry in json_data:
            # pdb.set_trace()
            all_imgs = entry[4]
            if len(all_imgs) < 1:
                continue
            if not os.path.exists(all_imgs[0]):
                continue
            
            topdown_im = "/".join(all_imgs[0].split("/")[:-1]) + "/top_down.png"
            topdown_feat = "/".join(all_imgs[0].split("/")[:-1]) + "/top_down_feats.pt"

            if not os.path.exists(topdown_im) or not os.path.exists(topdown_feat):
                continue
            
            topdown_data.append(entry)

        self.data = topdown_data

        print("Total number of data points: ", len(self.data))

        # split into train and test
        if args["split"] == "train":
            self.data = self.data[:11000]
            self.split = "train"
        elif args["split"] == "valtrain":
            self.data = self.data[11000:11100]
            self.split = "val"
        elif args["split"] == "val":
            self.data = self.data[-100:]
            self.split = "test"

        print(f"Total number of data in {args['split']}: ", len(self.data))

        # self.closest_ai2thor_to_ade, self.ai2assetname_to_objname, self.obj_to_color = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/ade_to_ai2thor.json"))

    def __getitem__(self, idx):
        room_data = self.data[idx]
        
        program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name = room_data
        
        cap_im, caption = self.apartment_ind_to_caption[all_imgs[0].split("/")[-2]]

        camera_poses = []
        program_text = generate_program_from_roomjson(house_json, include_children=self.args.get('include_children'))
        # program_text = format_program(program_text)

        # choose an image at random to feed in and the caption from another image at random
        objs_present = []
        top_feats_gt = None
        if self.args.get('use_seg_im'):
            selected_paths = random.sample(list(zip(all_seg_frames, all_objs)), self.args.get('num_images'))
            image_path = [x[0] for x in selected_paths]
            objs_present = [x[1] for x in selected_paths]
            if self.args.get('im_only'):
                caption_image = ["", ""]
            
            images = [Image.open(im_path).convert("RGB") for im_path in image_path]
        elif self.args.get('use_seg_real_im'):
            image_path = random.choice(all_room_captions)
            seg_image_path = image_path[1]
            real_image_path = image_path[0]
            image_path = [seg_image_path, real_image_path]
            
            images = [Image.open(im_path).convert("RGB") for im_path in image_path]
        elif self.args.get('use_top_down'):
            image_path_topdown = all_imgs[0].replace(".png", "_topdown.png")
            image_path_topdown_seg = all_imgs[0].replace(".png", "_topdown_seg.png")
            image_path = [image_path_topdown, image_path_topdown_seg]
            caption_image  = ["", td_caption_data[-1]]
            images = [Image.open(im_path).convert("RGB") for im_path in image_path]
        elif self.args.get('use_top_down_seg'):
            image_path_topdown_seg = all_imgs[0].replace(".png", "_topdown_seg.png")
            image_path = [image_path_topdown_seg, ]
            caption_image  = ["", td_caption_data[-1]]
            images = [Image.open(im_path).convert("RGB") for im_path in image_path]
        elif self.args.get('use_top_down_normal'):
            image_path_topdown = all_imgs[0].replace(".png", "_topdown.png")
            seg_image_path = random.choice(all_room_captions)[1]
            image_path = [image_path_topdown, seg_image_path]
            caption_image = ["", td_caption_data[-1]] 
            images = [Image.open(im_path).convert("RGB") for im_path in image_path]
        elif self.args.get('use_top_down_loss'):
            image_topdown_feats_path = "/".join(all_imgs[0].split("/")[:-1])+ "/top_down_feats.pt" # use precompute_top_clip_feats.py in scripts/

            top_feats_gt = torch.load(image_topdown_feats_path)
            
            image_path_inds = [all_imgs.index(cap_im),]  # random.sample(range(0, len(all_imgs)), 1) # random.choice(range(1, len(all_imgs) + 1)))
            image_path = [all_imgs[ind] for ind in image_path_inds]
            objs_visible = [all_objs[ind] for ind in image_path_inds]

            images = [Image.open(im_path).convert("RGB") for im_path in image_path]
            camera_poses = []
            for im_path in image_path:
                corner_ind = im_path.split("/")[-1].split(".")[0]
                # pdb.set_trace()
                camera_poses.append(cam_ind_to_position[corner_ind])
            
            for obj in objs_visible:
                objs_present.extend(obj)
            caption = ""

        elif self.args.get('all_corner_seg_ims'):
            seg_images = all_seg_frames
            caption_image = ["", ""]
            if self.args.get("tile_images"):
                images = [Image.open(os.path.join(self.image_data_path, im_path)) for im_path in seg_images]

                if self.args.get('recolor_instance_wise'):
                    images = color_instance_specific(images, color_to_objid)
                
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

                images = [new_image,]
                # pdb.set_trace()
                image_path = seg_images
                camera_poses = []
                for im_path in seg_images:
                    corner_ind = im_path.split("_")[2]
                    camera_poses.append(cam_ind_to_position[corner_ind])
            else:
                image_path = seg_images
                images = [Image.open(im_path).convert("RGB") for im_path in image_path]
        elif self.args.get('all_corner_ims'):
            caption_image = ["", ""] #td_caption_data[-1]]
            if self.args.get("tile_images"):
                images = [Image.open(os.path.join(self.image_data_path, im_path)) for im_path in all_imgs]

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

                images = [new_image,]
                image_path = all_imgs
                
            else:
                # if self.args.get('mode') == "train":
                image_path_inds = random.sample(range(0, len(all_imgs)), 1) # random.choice(range(1, len(all_imgs) + 1)))
                image_path = [all_imgs[ind] for ind in image_path_inds]
                objs_visible = [all_objs[ind] for ind in image_path_inds]
                # else:
                #    image_path = all_imgs
                #    objs_visible = all_objs
                images = [Image.open(im_path).convert("RGB") for im_path in image_path]
                camera_poses = []
                for im_path in image_path:
                    corner_ind = im_path.split("/")[-1].split(".")[0]
                    # pdb.set_trace()
                    camera_poses.append(cam_ind_to_position[corner_ind])
                
                for obj in objs_visible:
                    objs_present.extend(obj)
        elif self.args.get('all_corner_ims_priv'):
            caption_image = ["", td_caption_data[-1]]
            images = [Image.open(os.path.join(self.image_data_path, im_path)) for im_path in all_imgs]

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

            top_down_img = Image.open(all_imgs[0].replace(".png", "_topdown.png"))
            if random.random() < 0.5:
                images = [new_image, top_down_img]
            else:
                images = [new_image,]
            image_path = all_imgs
        else:
            image_path = random.sample(all_imgs, self.args.get('num_images'))
            # image_path = [x[0] for x in image_path]
            caption_image = random.choice(all_room_captions)

            images = [Image.open(im_path).convert("RGB") for im_path in image_path]

        if self.args.get('use_seg_im'):
            # convert the seg im colors to ade compatible colors. 
            all_images = []
            for image in images:
                segmentation_frame = np.array(image)
                
                if self.args.get('use_panoptic'):
                    new_segmentation_frame = segmentation_frame
                else:
                    # might be bugs here, just use panoptic, works fine. 
                    new_segmentation_frame = convert_panoptic_to_ade_colors(segmentation_frame, color_to_objid, obj_id_to_name, self.ai2assetname_to_objname, self.closest_ai2thor_to_ade, self.obj_to_color)

                image = Image.fromarray(new_segmentation_frame)
                if self.args.get('use_depth_im'):
                    depth_im_path = image_path[0].replace("_seg", "_depth")
                    depth_im = Image.open(depth_im_path)
                    # overlay the depth image on the seg image
                    depth_im = depth_im.convert("RGB")
                    depth_im = depth_im.resize(image.size)
                    image = Image.blend(image, depth_im, self.args.get('depth_alpha'))

                if self.args.get('blend_real_im'):
                    real_im_path = image_path[0].replace("_seg", "")
                    real_im = Image.open(real_im_path).convert("RGB")
                    real_im = real_im.resize(image.size)
                    image = Image.blend(image, real_im, self.args.get('real_alpha'))
                all_images.append(image)
        else:
            all_images = images

        # compute area of the room
        room = house_json['rooms'][0]
        polygon = room['floorPolygon']
        polygon_coords = [(point['x'], point['z']) for point in polygon]
        # pdb.set_trace()
        room_polygon = Polygon(polygon_coords)
        simplified_polygon = room_polygon # .simplify(0.001)
        num_corner = len(simplified_polygon.exterior.coords) - 1
        area = round(room_polygon.area, 2)

        if self.args.get("polygon_guided"):
            
            format_polygon_coords_num = []
            for point in list(simplified_polygon.exterior.coords):
                format_polygon_coords_num.append((int(round(point[0],2)*100), int(round(point[1],2)*100)))
            
            format_polygon_coords = str(format_polygon_coords_num[:-1])
            # pdb.set_trace()

            prefix_options = [
                f"## {'<image>'*len(images)} The room polygon is {format_polygon_coords}. List the exact 3D specifications: \n",
                f"## {'<image>'*len(images)} The room polygon is {format_polygon_coords}. The 3D specifications for such a room: \n",
                f"## {'<image>'*len(images)} {caption}. The room polygon is {format_polygon_coords}. The 3D specifications: \n",
                f"## {'<image>'*len(images)} {caption}. The 3D yaml for a room like this: \n",
            ]
            # pdb.set_trace()
            if self.args['mode'] in ["val", "valtrain"]:
                #if self.args.get('caption_only'):
                #    prefix_text = prefix_options[3]
                #elif self.args.get('im_only'):
                #    prefix_text = prefix_options[0]
                #else:
                prefix_text = prefix_options[0]
            else:
                prefix_text = random.choice(prefix_options)
        elif self.args.get("polygon_camera_guided"):
            format_polygon_coords = []
            for point in list(simplified_polygon.exterior.coords):
                format_polygon_coords.append((int(round(point[0],2)*100), int(round(point[1],2)*100)))
            
            format_polygon_coords = str(format_polygon_coords[:-1])
            camera_prompt = ""
            #camera_prompt = "The image contains multiple room views. The camera position for each of the room views is specified in posiiton and rotation. The position is the 2D x,y coordinate in top down view. The rotation angle 0-360 is along the perpendicular axis coming out of the top down view: \n"
            #for pos, rot in camera_poses:
            #    camera_prompt += f"Position: {int(round(pos['x'], 2)*100)}, {int(round(pos['z'], 2)*100)}, Rotation: {int(rot['y'])} degrees \n "
            #camera_prompt += "."
            
            prefix_options = [
                f"## {'<image>'*len(images)} The room polygon is {format_polygon_coords}. {camera_prompt} List the 3D specifications: #room \n",
                f"## {'<image>'*len(images)} The room polygon is {format_polygon_coords}. {camera_prompt} The 3D specifications: #room \n",
                f"## {'<image>'*len(images)} The room polygon is {format_polygon_coords}. {camera_prompt} The 3D specifications: #room \n",
                f"## {caption} The room polygon is {format_polygon_coords}. Can you list the 3D specifications? Surely: #room \n",
                f"## {caption}. The 3D specifications for the room with polygon {format_polygon_coords} is: #room \n",
                f"## {'<image>'*len(images)} The room polygon is {format_polygon_coords}. {camera_prompt} 3D specifications for a room like this: #room \n",
            ]
            # pdb.set_trace()
            if self.args['mode'] in ["val", "valtrain"]:
                if self.args.get('caption_only'):
                    prefix_text = prefix_options[3]
                elif self.args.get('im_only'):
                    prefix_text = prefix_options[0]
                else:
                    prefix_text = prefix_options[2]
            else:
                prefix_text = random.choice(prefix_options)

        else:
            prefix_options = [
                f"## {'<image>'*len(images)} List the exact 3D specifications: \n",
                f"## {'<image>'*len(images)} {caption}. The exact 3D specifications for such a room with {num_corner} corners: \n",
                f"## {'<image>'*len(images)} {caption}. The 3D specifications: \n",
                f"## {'<image>'*len(images)} Can you list the 3D specifications of such a room with {num_corner} corners? Surely: \n",
                f"## {'<image>'*len(images)} {caption}. The 3D specifications for the room with {num_corner} corners: \n ",
                f"## {'<image>'*len(images)} 3D specifications for a room like this as: \n",
            ]

            if self.args['mode'] in ["val", "valtrain"]:
                prefix_text = prefix_options[2]
            else:
                prefix_text = random.choice(prefix_options)

        if self.args['mode'] == "train":
            prompt = prefix_text + program_text + " \n###"
            text_labels = prompt
        else:
            prompt = prefix_text
            text_labels = prefix_text + program_text + " \n###"
        # pdb.set_trace()
        return all_images, caption, prompt, text_labels, image_path, house_json, objs_present, camera_poses, format_polygon_coords_num, top_feats_gt

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        images_batch, captions, prompts, text_labels, image_paths, house_jsons, objs_present, camera_poses, polygon_coords, top_feats_gts = zip(*batch)
        # pdb.set_trace()
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
            for images in images_batch:
                for image in images:
                    image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
                    # pdb.set_trace()
                    image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    new_images.append(image)

            pixel_values = torch.stack(new_images, dim=0)

            camera_pos_inp = []
            for pos, rot in camera_poses[0]:
                camera_pos_inp.append([int(round(pos['x'],2)*100), int(round(pos['z'],2)*100), int(rot['y'])])
            
            camera_pos_inp = torch.tensor(camera_pos_inp, dtype=torch.float32)

            # pdb.set_trace()
            # pixel_values = self.image_processor(images, return_tensors="pt")['pixel_values']
            return_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids,
                "pixel_values": pixel_values,
                "program_texts": prompts,
                "text_labels": text_labels,
                "image_lists": image_paths,
                'house_json': house_jsons,
                'objs_present': objs_present,
                'camera_pos': torch.tensor(camera_pos_inp),
                'polygon': torch.tensor(polygon_coords),
            }
            if top_feats_gts[0] is not None:
                return_dict['gt_im_features'] = torch.stack(top_feats_gts)[:,:,0,:]
        # pdb.set_trace()
        return return_dict

class Structured3D(Dataset):
    def __init__(self, args):

        all_scene_ids = [str(index).rjust(5, '0') for index in range(0, 200)]

        if args['mode'] == "train":
            all_scene_ids = all_scene_ids[:int(len(all_scene_ids)*0.9)]
        else:
            all_scene_ids = all_scene_ids[int(len(all_scene_ids)*0.9):]

        all_room_data = []
        for scene_id in all_scene_ids:
            scene_path = os.path.join(args.data_path, scene_id, "2D_rendering", f"all_room_data_{scene_id}.json")
            room_data = json.load(open(scene_path))
            for room_id in room_data:
                scene_id, room_id, panorama, polygon, objs, obj_features = room_data[room_id]

                # create program text from polygon, floor material, wall material, and objs
                program_text = generate_program_from_polygon_objs(polygon, objs)

                all_room_data.append(room_data[room_id])
        
        self.data = all_room_data

    def __getitem__(self, idx):

        
        pass

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        pass

    

            


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

        num_corner = len(all_imgs)
        if self.args['model'] == "codellama":
            env_desc = ""
            prompt_text = f"## Write code: Can you write yaml for a 3D environment {env_desc} \n Answer: "
        else:
            if self.args.get('prompt_style') == "multi_images":
                prompt_text = f"## {'<image>'*num_corner} \n Can you write yaml for a interactive room with {num_corner} corners that looks like these images? \n Answer: "
            else:
                prompt_text = f"## Write code: <image> \n Can you write yaml for a interactive room with {num_corner} corners that looks like these images? \n Answer: "

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
        #pdb.set_trace()
        return return_dict
    
