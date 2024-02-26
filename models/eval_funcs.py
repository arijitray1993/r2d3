from collections import defaultdict
import json
import os
from utils.ai2thor_utils import make_house_from_cfg
import pdb 
from PIL import Image
import numpy as np
import yaml
from shapely.geometry import Polygon
import shapely

class GenHouseIms:

    def __init__(self,):
        self.im_sim_scores = []

    def update(self, output):

        # generate house from output
        house = make_house_from_cfg(output)

        # get image from house


class HouseSemanticSimilarity:
    def __init__(self, args):
            
        self.sim_scores = []

        self.house_jsons = []
        self.gt_house_jsons = []

        self.polygon_accuracy = []
        self.floor_material_accuracy = []
        self.wall_material_accuracy = []
        self.object_class_accuracy = []
        self.object_location_error = []
        self.object_count_error = []
        self.num_corner_error = []
        self.object_finegrain_accuracy = []
        self.window_accuracy= []

        self.logger = args['logger']

    def update(self, output, gt):
        # compute text sim
        # convert output program text to gen config
        # pdb.set_trace()
        room_response = output.split(": \n")[-1]
        room_json_text = "\n".join(room_response.split("\n")[:-1])
        room_json_text = room_json_text.split("###")[0]
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")

        try:
            pred_house_dict = yaml.load(room_json_text, Loader=yaml.FullLoader)
        except:
            pred_house_dict = {}
        
        self.house_jsons.append(pred_house_dict)
        
        # convert gt program text to gen config
        gt_house_text = gt['text_labels'][0]
        gt_house_text = gt_house_text.split(": \n")[-1]
        gt_house_text = gt_house_text.replace("(", "[")
        gt_house_text = gt_house_text.replace(")", "]")
        # pdb.set_trace()

        try:
            gt_house_dict = yaml.load(gt_house_text, Loader=yaml.FullLoader)
        except:
            gt_house_dict = {}

        self.gt_house_jsons.append(gt_house_dict)

    def compute(self):
        
        for pred_house, gt_house in zip(self.house_jsons, self.gt_house_jsons):
            if len(pred_house) == 0 or len(gt_house) == 0:
                continue
            # check accuracy of room polygon i.e number of sides match
            pred_polygon = pred_house['polygon']
            gt_polygon = gt_house['polygon']

            self.polygon_accuracy.append(len(pred_polygon) == len(gt_polygon))
            self.num_corner_error.append(abs(len(pred_polygon) - len(gt_polygon)))

            # check accuracy of wall materials. wall materials are lists
            pred_materials = np.array(pred_house['wall_material'])
            gt_materials = np.array(gt_house['wall_material'])

            polygon_len = min(len(pred_materials), len(gt_materials))
            self.wall_material_accuracy.append(np.mean(pred_materials[:polygon_len] == gt_materials[:polygon_len]))

            # check accuracy of floor materials
            self.floor_material_accuracy.append(pred_house['floor_material'] == gt_house['floor_material'])

            # check accuracy of object classes
            # make counts of objects and locations
            gt_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in gt_house:
                    break
                gt_object, location, rotation = gt_house[f'obj_{i}']
                gt_objs[gt_object].append((location, rotation))
                i+=1
            
            pred_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in pred_house:
                    break
                try:
                    pred_object, location, rotation = pred_house[f'obj_{i}']
                except:
                    print("error object")
                    i+=1
                    continue
                pred_objs[pred_object].append((location, rotation))
                i+=1

            pred_obj_classes = defaultdict(list)
            for obj in pred_objs:
                obj_class = obj.split("_")[0]
                pred_obj_classes[obj_class].append(pred_objs[obj])
            
            for obj in gt_objs:
                obj_class = obj.split("_")[0]
                if obj_class in pred_obj_classes:
                    self.object_class_accuracy.append(1)
                else:
                    self.object_class_accuracy.append(0)
            

            for obj in gt_objs:
                if obj in pred_objs:
                    self.object_finegrain_accuracy.append(1)
                else:
                    self.object_finegrain_accuracy.append(0)
            
            for obj in gt_objs:
                if obj in pred_objs:
                    count_diff = abs(len(gt_objs[obj]) - len(pred_objs[obj]))
                    self.object_count_error.append(count_diff)

        if len(self.polygon_accuracy) == 0:
            return
        # compute mean of all accuracies
        polygon_acc = np.mean(self.polygon_accuracy)
        wall_material_acc = np.mean(self.wall_material_accuracy)
        floor_material_acc = np.mean(self.floor_material_accuracy)
        object_class_acc = np.mean(self.object_class_accuracy)
        num_corner_err = np.mean(self.num_corner_error)
        object_count_err = np.mean(self.object_count_error)
        object_finegrain_acc = np.mean(self.object_finegrain_accuracy)

        # log
        self.logger.log({"PolygonAcc": polygon_acc})
        self.logger.log({"WallMaterialAcc": wall_material_acc})
        self.logger.log({"FloorMaterialAcc": floor_material_acc})
        self.logger.log({"ObjectClassAcc": object_class_acc})
        self.logger.log({"NumCornerError": num_corner_err})
        self.logger.log({"ObjectCountError": object_count_err})
        self.logger.log({"ObjectFinegrainAcc": object_finegrain_acc})


        

class HouseJsonSimilarity:
    def __init__(self, args):
        self.sim_scores = []
        self.house_jsons = []
        self.gt_images = []

        self.exp_folder = os.path.join("checkpoints", args['exp_name'])

    def update(self, output, gt):
        # compute text sim
        self.house_jsons.append((output, 0))
        self.gt_images.append(gt['image_lists'][0])

    def compute(self):
        # pdb.set_trace()
        # save the json for now in a file
        with open(os.path.join(self.exp_folder, 'output.json'), 'w') as f:
            json.dump(self.house_jsons, f)
        
        # save the gt images one by one in vis
        os.makedirs(os.path.join(self.exp_folder, "vis"), exist_ok=True)
        
        for i, image_list in enumerate(self.gt_images):
            # check if image is PIL or file path
            if isinstance(image_list[0], str):
                images = [Image.open(image_path) for image_path in image_list]
            else:
                images = image_list
            
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

            image.save(os.path.join(self.exp_folder, f"vis/gt_images_{i}.png"))
        
            
