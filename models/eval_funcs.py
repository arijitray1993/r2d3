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
from scipy.optimize import linear_sum_assignment
import re
import ast

class GenHouseIms:

    def __init__(self,):
        self.im_sim_scores = []

    def update(self, output):

        # generate house from output
        house = make_house_from_cfg(output)

        # get image from house

def compute_location_error(pred_locations, gt_locations, max_dimension):
    
    # first we create a cost matrix that calculates distance between pred dist and gt dist
    k_gt = len(gt_locations)
    k_pred = len(pred_locations)
    cost_matrix = np.zeros((k_gt, k_pred))
    for i in range(k_gt):
        for j in range(k_pred):
            cost_matrix[i, j] = np.linalg.norm(pred_locations[j] - gt_locations[i])
    #print(cost_matrix)
    # then we use linear_sum_assignment to find the best matching between gt and pred
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    #compute the mean of the least mapped distances
    mean_dist = 0
    accuracy = []
    for i in range(len(row_ind)):
        if cost_matrix[row_ind[i], col_ind[i]]/max_dimension < 0.1:
            accuracy.append(1)
        else:
            accuracy.append(0)

        mean_dist += cost_matrix[row_ind[i], col_ind[i]]

    mean_dist = mean_dist / len(row_ind)

    #print(cluster_to_label_map)

    return mean_dist, np.mean(accuracy)


def get_object_class_from_asset(obj):
    pattern = r'[0-9]'
    obj_name = obj.replace("_", " ")
    obj_name = re.sub(pattern, '', obj_name)

    obj_name = obj_name.replace("RoboTHOR", "")

    obj_name = obj_name.replace("jokkmokk", " ")

    if "Countertop" in obj_name:
        shape = obj_name.split(" ")[-1]
        if shape =="I":
            obj_name = "I-shaped countertop"
        elif shape =="L":
            obj_name = "L-shaped countertop"
        elif shape =="C":
            obj_name = "C-shaped countertop"

    obj_name = obj_name.strip()

    return obj_name

class HouseObjectDistancesAccuracy:
    def __init__(self, args):
        self.house_jsons = []
        self.gt_house_jsons = []
        self.selected_objs = []
        self.object_location_error = []
        self.object_location_accuracy = []
        self.logger = args['logger']
    
    def update(self, output, gt):
        if "#room" in output:
            room_response = output.split(": #room \n")[-1]
        else:
            room_response = output.split(": \n")[-1]

        room_json_text = "\n".join(room_response.split("\n")[:-1])
        room_json_text = room_json_text.split("###")[0]
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")
        
        try:
            pred_house_dict = yaml.load(room_json_text, Loader=yaml.FullLoader)
        except:
            pred_house_dict = {}
        if pred_house_dict is None:
            print("none house dict")
            print(room_json_text)
        
        self.house_jsons.append(pred_house_dict)
        
        # convert gt program text to gen config
        gt_house_text = gt['text_labels'][0]
        if "#room" in gt_house_text:
            gt_house_text = gt_house_text.split(": #room \n")[-1]
        else:
            gt_house_text = gt_house_text.split(": \n")[-1]
        gt_house_text = gt_house_text.replace("(", "[")
        gt_house_text = gt_house_text.replace(")", "]")
        # pdb.set_trace()

        try:
            gt_house_dict = yaml.load(gt_house_text, Loader=yaml.FullLoader)
        except:
            gt_house_dict = {}

        if gt_house_dict is None:
            print("none house dict GT!")
            print(gt_house_text)
        self.gt_house_jsons.append(gt_house_dict)
        # pdb.set_trace()
        # self.selected_objs.append(gt['objs_present'][0])

    def compute(self):
        for pred_house, gt_house in zip(self.house_jsons, self.gt_house_jsons):   

            house_obj_loc_error = []
            house_obj_loc_acc = []
            if pred_house is None or gt_house is None:
                continue
            if len(pred_house) == 0 or len(gt_house) == 0:
                continue

            pred_polygon = pred_house['polygon']
            gt_polygon = gt_house['polygon']

            max_coordinate = np.max(gt_polygon)
            min_coordinate = np.min(gt_polygon)
            max_dimension = max_coordinate - min_coordinate

            gt_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in gt_house:
                    break
                gt_object, location, rotation = gt_house[f'obj_{i}']
                gt_objs[gt_object].append(location)
                i += 1
            
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
                pred_objs[pred_object].append(location)
                i+=1
            
            pred_obj_classes = defaultdict(list)
            
            for obj in pred_objs:
                obj_name = get_object_class_from_asset(obj)
                pred_obj_classes[obj_name].extend(pred_objs[obj])
            
            pred_locations = {}
            gt_locations = defaultdict(list)
            for obj in gt_objs:
                obj_name = get_object_class_from_asset(obj)
                if obj_name in pred_obj_classes:
                    pred_locations[obj_name] = pred_obj_classes[obj_name]
                    gt_locations[obj_name].extend(gt_objs[obj])
                
            
            # compute distance of each object to other objects
            obj_pair_dist = defaultdict(dict)
            for obj_1 in gt_locations:
                for obj_2 in gt_locations:
                    if obj_1 == obj_2:
                        continue
                    loc_1 = gt_locations[obj_1]
                    loc_2 = gt_locations[obj_2]
                    
                    all_dist = []
                    for l1 in loc_1:
                        for l2 in loc_2:
                            dist = np.linalg.norm(np.array(l1) - np.array(l2))
                            all_dist.append(dist)

                    obj_pair_dist[obj_1][obj_2] = all_dist
            
            pred_obj_pair_dist = defaultdict(dict)
            for obj_1 in pred_locations:
                for obj_2 in pred_locations:
                    if obj_1 == obj_2:
                        continue
                    loc_1 = pred_locations[obj_1]
                    loc_2 = pred_locations[obj_2]
                    
                    all_dist = []
                    for l1 in loc_1:
                        for l2 in loc_2:
                            dist = np.linalg.norm(np.array(l1) - np.array(l2))
                            all_dist.append(dist)
                        
                    pred_obj_pair_dist[obj_1][obj_2] = all_dist
            
            # compute distance error
            for obj_1 in obj_pair_dist:
                for obj_2 in obj_pair_dist[obj_1]:
                    gt_dists = obj_pair_dist[obj_1][obj_2]

                    if obj_1 not in pred_obj_pair_dist or obj_2 not in pred_obj_pair_dist[obj_1]:
                        continue
                    pred_dists = pred_obj_pair_dist[obj_1][obj_2]
                    
                    # hungarian match min distances
                    dist_error, dist_acc = compute_location_error(pred_dists, gt_dists, max_dimension)
                    house_obj_loc_error.append(dist_error/max_dimension)
                    house_obj_loc_acc.append(dist_acc)
                    
            # pdb.set_trace()\
            if len(house_obj_loc_error) == 0:
                continue
            self.object_location_error.append(np.mean(house_obj_loc_error))
            self.object_location_accuracy.append(np.mean(house_obj_loc_acc))
        # pdb.set_trace()
        if len(self.object_location_accuracy) == 0:
            mean_obj_loc_acc = 0
            mean_obj_loc_err = 1
        else:
            mean_obj_loc_acc = np.mean(self.object_location_accuracy)
            mean_obj_loc_err = np.mean(self.object_location_error)
        
        self.logger.log({"ObjectDistancesError": mean_obj_loc_err})
        self.logger.log({"ObjectDistancesAcc": mean_obj_loc_acc})



class HouseSelectedObjectDistancesAccuracy:
    def __init__(self, args):
        self.house_jsons = []
        self.gt_house_jsons = []
        self.selected_objs = []
        self.object_location_error = []
        self.object_location_accuracy = []
        self.logger = args['logger']
        self.selected_objs = []
    
    def update(self, output, gt):
        if "#room" in output:
            room_response = output.split(": #room \n")[-1]
        else:
            room_response = output.split(": \n")[-1]

        room_json_text = "\n".join(room_response.split("\n")[:-1])
        room_json_text = room_json_text.split("###")[0]
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")
        
        try:
            pred_house_dict = yaml.load(room_json_text, Loader=yaml.FullLoader)
        except:
            pred_house_dict = {}
        if pred_house_dict is None:
            print("none house dict")
            print(room_json_text)
        
        self.house_jsons.append(pred_house_dict)
        
        # convert gt program text to gen config
        gt_house_text = gt['text_labels'][0]
        if "#room" in gt_house_text:
            gt_house_text = gt_house_text.split(": #room \n")[-1]
        else:
            gt_house_text = gt_house_text.split(": \n")[-1]
        gt_house_text = gt_house_text.replace("(", "[")
        gt_house_text = gt_house_text.replace(")", "]")
        # pdb.set_trace()

        try:
            gt_house_dict = yaml.load(gt_house_text, Loader=yaml.FullLoader)
        except:
            gt_house_dict = {}

        if gt_house_dict is None:
            print("none house dict GT!")
            print(gt_house_text)
        self.gt_house_jsons.append(gt_house_dict)
        
        self.selected_objs.append(gt['objs_present'][0])

    def compute(self):
        for pred_house, gt_house, selected_obj in zip(self.house_jsons, self.gt_house_jsons, self.selected_objs):   

            house_obj_loc_error = []
            house_obj_loc_acc = []
            if pred_house is None or gt_house is None:
                continue
            if len(pred_house) == 0 or len(gt_house) == 0:
                continue

            pred_polygon = pred_house['polygon']
            gt_polygon = gt_house['polygon']

            max_coordinate = np.max(gt_polygon)
            min_coordinate = np.min(gt_polygon)
            max_dimension = max_coordinate - min_coordinate

            gt_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in gt_house:
                    break
                gt_object, location, rotation = gt_house[f'obj_{i}']
                gt_objs[gt_object].append(location)
                i += 1
            
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
                pred_objs[pred_object].append(location)
                i+=1
            
            pred_obj_classes = defaultdict(list)
            
            for obj in pred_objs:
                obj_name = get_object_class_from_asset(obj)
                pred_obj_classes[obj_name].extend(pred_objs[obj])
            
            pred_locations = {}
            gt_locations = defaultdict(list)
            for obj in gt_objs:
                obj_name = get_object_class_from_asset(obj)
                if obj_name in pred_obj_classes:
                    pred_locations[obj_name] = pred_obj_classes[obj_name]
                    gt_locations[obj_name].extend(gt_objs[obj])
                
            
            # compute distance of each object to other objects
            obj_pair_dist = defaultdict(dict)
            for obj_1 in gt_locations:
                for obj_2 in gt_locations:
                    if obj_1 == obj_2:
                        continue
                    loc_1 = gt_locations[obj_1]
                    loc_2 = gt_locations[obj_2]
                    
                    all_dist = []
                    for l1 in loc_1:
                        for l2 in loc_2:
                            dist = np.linalg.norm(np.array(l1) - np.array(l2))
                            all_dist.append(dist)

                    obj_pair_dist[obj_1][obj_2] = all_dist
            
            pred_obj_pair_dist = defaultdict(dict)
            for obj_1 in pred_locations:
                for obj_2 in pred_locations:
                    if obj_1 == obj_2:
                        continue
                    loc_1 = pred_locations[obj_1]
                    loc_2 = pred_locations[obj_2]
                    
                    all_dist = []
                    for l1 in loc_1:
                        for l2 in loc_2:
                            dist = np.linalg.norm(np.array(l1) - np.array(l2))
                            all_dist.append(dist)
                        
                    pred_obj_pair_dist[obj_1][obj_2] = all_dist
            
            # compute distance error
            for obj_1 in obj_pair_dist:
                for obj_2 in obj_pair_dist[obj_1]:
                    gt_dists = obj_pair_dist[obj_1][obj_2]

                    if obj_1 not in pred_obj_pair_dist or obj_2 not in pred_obj_pair_dist[obj_1]:
                        continue
                        
                    if obj_1 in selected_obj and obj_2 in selected_obj:
                        pred_dists = pred_obj_pair_dist[obj_1][obj_2]
                        
                        # hungarian match min distances
                        dist_error, dist_acc = compute_location_error(pred_dists, gt_dists, max_dimension)
                        house_obj_loc_error.append(dist_error/max_dimension)
                        house_obj_loc_acc.append(dist_acc)
                    
            # pdb.set_trace()\
            if len(house_obj_loc_error) == 0:
                continue
            self.object_location_error.append(np.mean(house_obj_loc_error))
            self.object_location_accuracy.append(np.mean(house_obj_loc_acc))
        # pdb.set_trace()
        if len(self.object_location_accuracy) == 0:
            mean_obj_loc_acc = 0
            mean_obj_loc_err = 1
        else:
            mean_obj_loc_acc = np.mean(self.object_location_accuracy)
            mean_obj_loc_err = np.mean(self.object_location_error)
        
        self.logger.log({"SelectedObjectDistancesError": mean_obj_loc_err})
        self.logger.log({"SelectedObjectDistancesAcc": mean_obj_loc_acc})



class HouseSelectedObjAccuracy:
    def __init__(self, args):
        self.house_jsons = []
        self.gt_house_jsons = []
        self.selected_objs = []
        self.object_location_error = []
        self.object_location_accuracy = []
        self.object_class_accuracy = []
        self.logger = args['logger']
        pass

    def update(self, output, gt):
        if "#room" in output:
            room_response = output.split(": #room \n")[-1]
        else:
            room_response = output.split(": \n")[-1]

        room_json_text = "\n".join(room_response.split("\n")[:-1])
        room_json_text = room_json_text.split("###")[0]
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")

        try:
            pred_house_dict = yaml.load(room_json_text, Loader=yaml.FullLoader)
        except:
            pred_house_dict = {}
        if pred_house_dict is None:
            print("none house dict")
            print(room_json_text)
        
        self.house_jsons.append(pred_house_dict)
        
        # convert gt program text to gen config
        gt_house_text = gt['text_labels'][0]
        if "#room" in gt_house_text:
            gt_house_text = gt_house_text.split(": #room \n")[-1]
        else:
            gt_house_text = gt_house_text.split(": \n")[-1]
        gt_house_text = gt_house_text.replace("(", "[")
        gt_house_text = gt_house_text.replace(")", "]")
        # pdb.set_trace()

        try:
            gt_house_dict = yaml.load(gt_house_text, Loader=yaml.FullLoader)
        except:
            gt_house_dict = {}

        if gt_house_dict is None:
            print("none house dict GT!")
            print(gt_house_text)
        self.gt_house_jsons.append(gt_house_dict)
        
        self.selected_objs.append(gt['objs_present'][0])
        
    
    def compute(self):
        
        for pred_house, gt_house, selected_objs in zip(self.house_jsons, self.gt_house_jsons, self.selected_objs):
            if pred_house is None or gt_house is None:
                    continue
            if len(pred_house) == 0 or len(gt_house) == 0:
                continue
            # check accuracy of room polygon i.e number of sides match
            pred_polygon = pred_house['polygon']
            gt_polygon = gt_house['polygon']

            max_coordinate = np.max(gt_polygon)
            min_coordinate = np.min(gt_polygon)
            max_dimension = max_coordinate - min_coordinate

            # check accuracy of object classes
            # make counts of objects and locations
            gt_objs = defaultdict(list)
            i=0
            while(True):
                if f'obj_{i}' not in gt_house:
                    break
                gt_object, location, rotation = gt_house[f'obj_{i}']
                gt_objs[gt_object].append(location)
                i += 1
            
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
                pred_objs[pred_object].append(location)
                i += 1

            pred_obj_classes = defaultdict(list)
            for obj in pred_objs:
                obj_class = get_object_class_from_asset(obj)
                pred_obj_classes[obj_class].extend(pred_objs[obj])
            
            if len(selected_objs)>0:
                selected_obj_classes = [get_object_class_from_asset(obj) for obj in selected_objs]
            else:
                selected_obj_classes = []
            
            pred_locations = {}
            gt_locations = defaultdict(list)
            for obj in gt_objs:
                obj_class = get_object_class_from_asset(obj)
                if obj_class in pred_obj_classes:
                    if obj_class in selected_obj_classes:
                        self.object_class_accuracy.append(1)
                    pred_locations[obj_class] = pred_obj_classes[obj_class]
                    gt_locations[obj_class].extend(gt_objs[obj])
                else:
                    if obj_class in selected_obj_classes:
                        self.object_class_accuracy.append(0)

            # compute hungarian matching of predicted object locations to gt object locations
            for obj in gt_locations:
                pred_loc = pred_locations[obj]
                gt_loc = gt_locations[obj]
               
                pred_loc = np.array(pred_loc).astype(np.float32)
                gt_loc = np.array(gt_loc).astype(np.float32)
                
                mean_dist, accuracy = compute_location_error(pred_loc, gt_loc, max_dimension)
                if obj in selected_obj_classes:
                    self.object_location_accuracy.append(accuracy)
                    self.object_location_error.append(mean_dist/max_dimension)
            
        if len(self.object_location_accuracy) == 0:
            return
        # compute mean of all accuracies
        object_class_acc = np.mean(self.object_class_accuracy)
        object_location_err = np.mean(self.object_location_error)
        object_location_acc = np.mean(self.object_location_accuracy)

        self.logger.log({"SelObjectClassAcc": object_class_acc})
        self.logger.log({"SelObjectLocationError": object_location_err})
        self.logger.log({"SelObjectLocationAcc": object_location_acc})



class HouseSemanticSimilarity:
    def __init__(self, args):
            
        self.sim_scores = []

        self.house_jsons = []
        self.gt_house_jsons = []
        self.selected_objs = []

        self.polygon_accuracy = []
        self.floor_material_accuracy = []
        self.wall_material_accuracy = []
        self.object_class_accuracy = []
        self.object_location_error = []
        self.object_location_accuracy = []
        self.objectsel_location_accuracy = []
        self.object_count_error = []
        self.num_corner_error = []
        self.object_finegrain_accuracy = []
        self.window_accuracy= []
        self.ignore_loc = []

        self.logger = args['logger']

    def update(self, output, gt):
        # compute text sim
        # convert output program text to gen config
        # pdb.set_trace()
        if "#room" in output:
            room_response = output.split(": #room \n")[-1]
        else:
            room_response = output.split(": \n")[-1]
        # pdb.set_trace()
        room_json_text = "\n".join(room_response.split("\n")[:-1])
        room_json_text = room_json_text.split("###")[0]
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")
        
        try:
            pred_house_dict = yaml.load(room_json_text, Loader=yaml.FullLoader)
        except:
            pred_house_dict = {}
        
        if pred_house_dict is None:
            print("none house dict")
            print(room_json_text)
        self.house_jsons.append(pred_house_dict)
        
        # convert gt program text to gen config
        og_gt_house_text = gt['text_labels'][0]
        if "#room" in og_gt_house_text:
            gt_house_text = og_gt_house_text.split(": #room \n")[-1]
        else:
            gt_house_text = og_gt_house_text.split(": \n")[-1]
        gt_house_text = gt_house_text.replace("(", "[")
        gt_house_text = gt_house_text.replace(")", "]")
        # pdb.set_trace()

        try:
            gt_house_dict = yaml.load(gt_house_text, Loader=yaml.FullLoader)
        except:
            gt_house_dict = {}

        if gt_house_dict is None:
            print("none house dict GT!")
            print(gt_house_text)
        self.gt_house_jsons.append(gt_house_dict)

        if "The red circular 1 mark in the image is at 3D coordinate (x, y, z)" in og_gt_house_text:
            ignore_loc = og_gt_house_text.split("The red circular 1 mark in the image is at 3D coordinate (x, y, z)")[-1].split(".")[0]
            ignore_loc = ignore_loc.strip()
            ignore_loc = ast.literal_eval(ignore_loc)
            self.ignore_loc.append(ignore_loc)
        # pdb.set_trace()
        # self.selected_objs.append(gt['objs_present'][0][0])

    def compute(self):
        
        for ignore_ind, (pred_house, gt_house) in enumerate(zip(self.house_jsons, self.gt_house_jsons)):
            # pdb.set_trace()
            try:
                if pred_house is None or gt_house is None:
                    continue
                if len(pred_house) == 0 or len(gt_house) == 0:
                    continue
                # check accuracy of room polygon i.e number of sides match
                pred_polygon = pred_house['polygon']
                gt_polygon = gt_house['polygon']

                max_coordinate = np.max(gt_polygon)
                min_coordinate = np.min(gt_polygon)
                max_dimension = max_coordinate - min_coordinate

                # convert to shapely polygon and simplify
                pred_polygon_simple = Polygon(pred_polygon).simplify(0.01)
                gt_polygon_simple = Polygon(gt_polygon).simplify(0.01)


                self.polygon_accuracy.append(len(pred_polygon_simple.exterior.coords) == len(gt_polygon_simple.exterior.coords))
                self.num_corner_error.append(abs(len(pred_polygon_simple.exterior.coords) - len(gt_polygon_simple.exterior.coords)))

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
                    gt_objs[gt_object].append(location)
                    i += 1
                
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
                    pred_objs[pred_object].append(location)
                    i += 1

                pred_obj_classes = defaultdict(list)
                for obj in pred_objs:
                    obj_class = get_object_class_from_asset(obj)
                    pred_obj_classes[obj_class].extend(pred_objs[obj])
                
                pred_locations = {}
                gt_locations = defaultdict(list)
                for obj in gt_objs:
                    obj_class = get_object_class_from_asset(obj)
                    if obj_class in pred_obj_classes:
                        self.object_class_accuracy.append(1)
                        pred_locations[obj_class] = pred_obj_classes[obj_class]
                        gt_locations[obj_class].extend(gt_objs[obj])
                    else:
                        self.object_class_accuracy.append(0)

                # compute hungarian matching of predicted object locations to gt object locations
                # if len(selected_objs)>0:
                #    selected_obj_classes = [obj.split(" ")[0] for obj in selected_objs]
                if len(self.ignore_loc)>0:
                    ignore_locs = self.ignore_loc[ignore_ind]
                else:
                    ignore_locs = -1
                for obj in gt_locations:
                    pred_loc = pred_locations[obj]
                    gt_loc = gt_locations[obj]
                    if gt_loc[0] == ignore_locs:
                        continue
                    try:
                        pred_loc = np.array(pred_loc).astype(np.float32)
                        gt_loc = np.array(gt_loc).astype(np.float32)
                    except:
                         self.object_location_accuracy.append(0)
                         self.object_location_error.append(1)
                         continue
                    mean_dist, accuracy = compute_location_error(pred_loc, gt_loc, max_dimension)
                    self.object_location_accuracy.append(accuracy)
                    self.object_location_error.append(mean_dist/max_dimension)

                    # if len(selected_objs)>0:
                    #     if obj in selected_obj_classes:
                    #         self.objectsel_location_accuracy.append(accuracy)


                for obj in gt_objs:
                    if obj in pred_objs:
                        self.object_finegrain_accuracy.append(1)
                    else:
                        self.object_finegrain_accuracy.append(0)
                
                for obj in gt_objs:
                    if obj in pred_objs:
                        count_diff = abs(len(gt_objs[obj]) - len(pred_objs[obj]))
                        self.object_count_error.append(count_diff)
                    else:
                        self.object_count_error.append(len(gt_objs[obj]))
            except:
                print("some error in JSON")
                continue

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
        object_location_err = np.mean(self.object_location_error)
        object_location_acc = np.mean(self.object_location_accuracy)
        #if len(self.objectsel_location_accuracy)>0:
        #    objectsel_location_acc = np.mean(self.objectsel_location_accuracy)
        #else:
        #    objectsel_location_acc = -1

        # log
        self.logger.log({"PolygonAcc": polygon_acc})
        self.logger.log({"WallMaterialAcc": wall_material_acc})
        self.logger.log({"FloorMaterialAcc": floor_material_acc})
        self.logger.log({"ObjectClassAcc": object_class_acc})
        self.logger.log({"NumCornerError": num_corner_err})
        self.logger.log({"ObjectCountError": object_count_err})
        self.logger.log({"ObjectFinegrainAcc": object_finegrain_acc})
        self.logger.log({"ObjectLocationError": object_location_err})
        self.logger.log({"ObjectLocationAcc": object_location_acc})
        # self.logger.log({"ObjectSelLocationAcc": objectsel_location_acc})


class HouseJsonSimilarity:
    def __init__(self, args):
        self.sim_scores = []
        self.house_jsons = []
        self.gt_images = []

        self.exp_folder = os.path.join("checkpoints", args['exp_name'])

    def update(self, output, gt):
        # compute text sim
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

        if 'objs_present' in gt:
            gt_house_dict['objs_present'] = gt['objs_present']
        
        if 'camera_pos' in gt:
            gt_house_dict['camera_pos'] = gt['camera_pos'].tolist()

        if 'polygon' in gt:
            gt_house_dict['polygon'] = gt['polygon'].tolist()
            

        if 'house_json' in gt:
            self.house_jsons.append((output, gt['house_json'][0], gt_house_dict, gt['text_labels'][0]))    
        else:
            self.house_jsons.append((output, 0, {}, gt['text_labels'][0]))
        
        if 'image_lists' in gt:
            self.gt_images.append(gt['image_lists'][0])

    def compute(self):
        # pdb.set_trace()
        # save the json for now in a file
        with open(os.path.join(self.exp_folder, 'output.json'), 'w') as f:
            json.dump(self.house_jsons, f)
        
        # save the gt images one by one in vis
        os.makedirs(os.path.join(self.exp_folder, "vis"), exist_ok=True)
        
        for i, image_list in enumerate(self.gt_images):
            images = [Image.open(image_path) for image_path in image_list]

            for corner_ind, image in enumerate(images):
                image.save(os.path.join(self.exp_folder, f"vis/gt_images_{corner_ind}_{i}.png"))

