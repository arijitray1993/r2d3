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
from utils.ai2thor_utils import generate_program_from_roomjson, format_program

import numpy as np


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
        prefix_text = f"## <image> \n Write code for an interactive room with {num_corner} corners that looks like the image. \n Answer: "
    
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

        caption = ""

        prefix_text = f"## <image> Can you render a room with that looks like the image? Surely: \n"
    
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


class ProcTHOR_image_caption(Dataset):
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

        if self.args.get('summary_only'):
            summary_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/all_room_json_programs_ai2_train_room_summaries_gtobjonly.json"))
            assert len(self.json_data) == len(summary_data)

        self.image_data_path = "/projectnb/ivc-ml/array/research/robotics/ProcTHOR/"
        # only use the json data we have images for
        self.data = []
        self.summary_data = []
        for ind, (room_data, room_captions) in enumerate(json_data):
            all_imgs= room_data[4]
            if len(all_imgs) < 2:
                continue
            if len(room_captions) < 2:
                continue
            if not os.path.exists(all_imgs[0]):
                continue
            self.data.append((room_data, room_captions))
            if self.args.get('summary_only'):
                self.summary_data.append(summary_data[ind])

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
        room_data, all_room_captions = self.data[idx]

        program_text, house_json, og_house_json, cam_ind_to_position, all_imgs, all_objs, all_seg_frames, color_to_objid, obj_id_to_name = room_data

        program_text = generate_program_from_roomjson(house_json)
        # program_text = format_program(program_text)

        # choose an image at random to feed in and the caption from another image at random
        if self.args.get('use_seg_im'):
            image_path = random.sample(all_seg_frames, self.args.get('num_images'))
            caption_image  = random.choice(all_room_captions)

        else:
            image_path, caption_image = random.sample(all_room_captions, 2)
            
        if self.args.get('num_images') == 1:
            image = Image.open(image_path[0]).convert("RGB")
        else:
            images = [Image.open(im_path) for im_path in image_path]
        
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

        if self.args.get('use_seg_im'):
            # convert the seg im colors to ade compatible colors. 
            segmentation_frame = np.array(image)
            
            if self.args.get('use_panoptic'):
                new_segmentation_frame = segmentation_frame
            else:
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
                                obj_name = self.ai2assetname_to_objname[ai2_assetname]
                                closest_ade_obj, distance = self.closest_ai2thor_to_ade[obj_name[0]][0]
                            except:
                                print(ai2_objid)
                                print(ai2_objid_format)
                                print(ai2_assetname)
                                print(obj_name)
                    else:
                        print(color)
                        print(color_to_objid)
                        closest_ade_obj = "wall"

                    try:
                        new_color = self.obj_to_color[closest_ade_obj]
                    except:
                        print(closest_ade_obj)

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
                
                new_segmentation_frame = all_colors.reshape(segmentation_frame.shape)
            
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

                # pdb.set_trace()
            # pdb.set_trace()
            # pdb.set_trace()
        
        caption = caption_image[-1]
        # pdb.set_trace()
        # if caption too long take only first 2 sentences
        # pdb.set_trace()
        # caption = ". ".join(caption.split(". ")[:2])
        # pdb.set_trace()
        num_corner = len(all_imgs) # since we have an image from each corner.
        # compute area of the room
        room = house_json['rooms'][0]
        polygon = room['floorPolygon']
        polygon_coords = [(point['x'], point['z']) for point in polygon]
        room_polygon = Polygon(polygon_coords)
        area = round(room_polygon.area, 2)
        # pdb.set_trace()

        if self.args.get('language_only'):
            prefix_text = f"## \n Write yaml for an interactive room with {num_corner} corners with the description: {caption}. \n Answer: "
        elif self.args.get('summary_only'):
            _, _, summary = self.summary_data[idx]
            prefix_text = f"## \n Write yaml for an interactive room with {num_corner} corners with the description: {summary}. \n Answer: "
        else:
            prefix_options = [
                f"## <image> An interactive room with {num_corner} corners with {area} area that looks like the image can be rendered as: \n",
                f"## <image> An interactive room like the image can be rendered as: \n",
                f"## <image> Can you render a room with {num_corner} corners that looks like the image? Surely: \n",
                f"## <image> A room with {num_corner} corners like the image with description: {caption}. The yaml is: \n",
                f"## <image> A room with {num_corner} corners with description: {caption}. The yaml is: \n",
                f"## <image> Can you render a room with that looks like the image? Surely: \n",
                f"## <image> {caption}. Generate such a room with {num_corner} corners as: \n",
                f"## <image> {caption}. The yaml for the room with {num_corner} corners: \n ",
                f"## <image> {caption}. We can generate it as: \n",
                f"## <image> {caption}. How can I render it? Like this: \n",
                f"## <image> {caption}. A room that looks like this can be rendered as: \n",
            ]

            if self.args['split'] in ["val", "valtrain"]:
                prefix_text = prefix_options[8]
            else:
                prefix_text = random.choice(prefix_options)
        
        if self.split == "train":
            prompt = prefix_text + program_text + " \n###"
            text_labels = prompt
        else:
            prompt = prefix_text
            text_labels = prefix_text + program_text + " \n###"

        # pdb.set_trace()
        return image, caption, prompt, text_labels, [image_path[0],], house_json

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        images, captions, prompts, text_labels, image_paths, house_jsons = zip(*batch)

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
                "program_texts": prompts,
                "text_labels": text_labels,
                "image_lists": image_paths,
                'house_json': house_jsons,
            }
        # pdb.set_trace()
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
    
