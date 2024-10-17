from collections import defaultdict
import prior
from ai2thor.controller import Controller
from PIL import Image
import random
from pprint import pprint
import json
import pdb
import math
import os
import copy
import tqdm
from ai2thor.platform import CloudRendering

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import yaml
import ast
from transformers import AutoTokenizer, AutoModel

import sys
sys.path.append("../")
import utils.ai2thor_utils as ai2thor_utils
from matplotlib import pyplot as plt


def get_top_down_frame(controller):
    # Setup the top-down camera
    '''
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    # pdb.set_trace()
    top_down_frame = event.third_party_camera_frames[-1]
    '''

    controller.step('ToggleMapView')
    top_down_frame = controller.last_event.frame
    return Image.fromarray(top_down_frame)


def generate_output_houses(json_path, generate_explanation=False, use_language=False, assetdesc_to_obj=None, attrtokenizer=None, attrmodel=None):

    output_json = json.load(open(json_path, 'r'))
    image_save_folder = json_path.replace("output.json", "gen_images")
    format_results_folder = json_path.replace("output.json", "format_results")
    os.makedirs(image_save_folder, exist_ok=True)
    os.makedirs(format_results_folder, exist_ok=True)
    # this is list of raw text, each for a room gen 

    exp_path = "/".join(json_path.split("/")[:-1])

    # pdb.set_trace()

    response_rooms = [out_text[0].split(": \n")[-1] for out_text in output_json]

    gt_rooms = [out_text[1] for out_text in output_json]
    
    failure_explanations = []
    for room_ind, room_response in enumerate(response_rooms):
        print(f"-------------Processing room {room_ind}---------------")
        room_json_text = "\n".join(room_response.split("\n")[:-1])
        room_json_text = room_json_text.replace("(", "[")
        room_json_text = room_json_text.replace(")", "]")
        # pdb.set_trace()
        print(room_json_text)

        #try:
        if use_language:
            house_json = ai2thor_utils.make_house_from_cfg_language(room_json_text, attrmodel, attrtokenizer, assetdesc_to_obj)
        else:
            house_json = ai2thor_utils.make_house_from_cfg(room_json_text)
        #except:
        #    print("not a valid house json")
        #    continue
        house_json = house_json.house_json
        try:
            controller = Controller(width=800, height=800, quality="High WebGL", scene="Procedural", gridSize=0.1) # renderInstanceSegmentation=True)
            controller.step(action="CreateHouse", house=house_json)
        except:
            print("not a valid house json")

        controller.step(
            action="RandomizeLighting",
            brightness=(1.5, 2),
            randomizeColor=False,
            synchronized=False
        )
        try:
            event = controller.step(action="GetReachablePositions")
        except:
            print("Cannot get reachable positions, continuing")
            controller.stop()
            continue
        reachable_positions = event.metadata["actionReturn"]

        if reachable_positions is None:
            print("No reachable positions, continuing")
            # pdb.set_trace()
            controller.stop()
            continue

        # 2. get the corner x,z coordinates from house json room polygon
        corner_positions = []
        room = house_json['rooms'][0]
        polygon = room['floorPolygon']
        for point in polygon:
            corner_positions.append((point['x'], point['z']))
        
        cam_ind_to_position = {}
        # 3. get the interior angle for each corner
        # hacky way: get the neihboring two polygon points, compute the centroid
        # , check if centroid is inside the polygon, if inside, compute the angle of the vector from the corner to the centroid to the x-axis
        # if outside, compute the 180-angle. 

        shape_polygon = Polygon(corner_positions)

        # make a mapping to obj id to obj name
        obj_id_to_name = {}
        for obj in house_json['objects']:
            obj_id_to_name[obj['id']] = obj['assetId']  
        
        all_imgs = []
        all_objs = []
        all_seg_frames = []
        for corner_ind, corner in enumerate(corner_positions):
            prev_ind = (corner_ind-1)%len(corner_positions)
            next_ind = (corner_ind+1)%len(corner_positions)
            
            PREV_POINT = corner_positions[prev_ind]
            NEXT_POINT = corner_positions[next_ind] 

            centroid_x = (PREV_POINT[0] + NEXT_POINT[0] + corner[0])/3
            centroid_z = (PREV_POINT[1] + NEXT_POINT[1] + corner[1])/3

            # check if centroid is inside polygon
            in_polygon = shape_polygon.contains(Point(centroid_x, centroid_z))
            
            # compute angle from corner-centroid vector to x-axis
            x1, z1 = corner
            x2, z2 = centroid_x, centroid_z
            pi = 3.1415
            try:
                angle_pos_y = 180*math.atan((x2-x1)/(z2-z1))/pi
            except:
                print("not really a corner")
                controller.stop()
                continue
            
            if x2 > x1 and z2 > z1:
                quadrant = "First Quadrant"
            elif x2 < x1 and z2 > z1:
                quadrant = "Second Quadrant"
            elif x2 < x1 and z2 < z1:
                quadrant = "Third Quadrant"
            elif x2 > x1 and z2 < z1:
                quadrant = "Fourth Quadrant"

            if angle_pos_y < 0:
                if quadrant == "Second Quadrant":
                    angle = 360 + angle_pos_y
                elif quadrant == "Fourth Quadrant":
                    angle = 180 + angle_pos_y
            else:
                if quadrant == "First Quadrant":
                    angle = angle_pos_y
                elif quadrant == "Third Quadrant":
                    angle = 180 + angle_pos_y
            
            if not in_polygon:
                angle = angle + 180
            
            print("corner", corner, "angle", angle)

            # 4. find the closest reachable position and teleport the agent there and place camera at angle
            closest_reachable_positions = sorted(reachable_positions, key=lambda x: (x['x']-corner[0])**2 + (x['z']-corner[1])**2)
            

            
            # pdb.set_trace()
            num_tries = 0
            success = 0
            while(success == 0 and num_tries < 10):
                try:
                    position = random.choice(closest_reachable_positions[:10])
                    position['x'] = round(position['x']*4)/4.0
                    position['z'] = round(position['z']*4)/4.0
                    position['y'] = 0.9
                    rotation = { "x": 0.0, "y": angle, "z": 0.0} 

                    # check if position is in polygon
                    if not shape_polygon.contains(Point(position['x'], position['z'])):
                        print("not in polygon")
                        num_tries+=1
                        success = 0
                        continue      

                    event = controller.step(action="Teleport", position=position, rotation=rotation)
                    # pdb.set_trace()
                    if event.metadata["lastActionSuccess"]:
                        success = 1
                    else:
                        num_tries+=1
                        success = 0
                except:
                    print("teleport failed")
                    num_tries+=1
                    success = 0
            
            # pdb.set_trace()

            img = Image.fromarray(controller.last_event.frame)
            img.save(f"{image_save_folder}/example_{room_ind}_{corner_ind}.png")

        try:
            top_down_frame = get_top_down_frame(controller)
            top_down_frame.save(f"{image_save_folder}/example_top_down_{room_ind}.png")
        except:
            print("couldnt get top down frame")
            # controller.stop()
            top_down_frame = Image.new('RGB', (800, 800), color = (255, 255, 255))
            #continue
        
        gt_house_json = gt_rooms[room_ind]
        if gt_house_json != 0:
            controller_gt = Controller(scene=gt_house_json, width=800,height=800)
            try:
                top_down_frame_gt = get_top_down_frame(controller_gt)
                top_down_frame_gt.save(f"{image_save_folder}/example_top_down_gt_{room_ind}.png")
            except:
                print("couldnt get top down frame gt")
                # controller_gt.stop()
                top_down_frame_gt = Image.new('RGB', (800, 800), color = (255, 255, 255))
                #continue    
            controller_gt.stop()
        else:
            top_down_frame_gt = top_down_frame
        # plot input im/segframe, language and output corner ims in one im
        input_im = exp_path+f"/vis/gt_images_0_{room_ind}.png"
        out_text = output_json[room_ind]
        caption = out_text[-1].split(": \n")[0]
        # pdb.set_trace()

        # add a \n in caption after every 50 chaarcters
        caption = "\n".join([caption[i:i+50] for i in range(0, len(caption), 50)])

        # plot input_im, caption, top down frame and corner_ims
        plt.figure(figsize=(30, 30))
        plt.subplot(3, 3, 1)
        if os.path.exists(input_im):
            plt.imshow(Image.open(input_im))
        plt.title(caption)
        plt.axis('off')
        plt.subplot(3, 3, 2)
        plt.imshow(top_down_frame)
        plt.title("Top Down View")
        plt.axis('off')
        plt.subplot(3, 3, 3)
        plt.imshow(top_down_frame_gt)
        plt.title("GT Top Down View")
        plt.axis('off')
        for corner_ind, corner in enumerate(corner_positions[:6]):
            if os.path.exists(f"{image_save_folder}/example_{room_ind}_{corner_ind}.png"):
                plt.subplot(3, 3, 4+corner_ind)
                plt.imshow(Image.open(f"{image_save_folder}/example_{room_ind}_{corner_ind}.png"))
                plt.title(f"Corner {corner_ind}")
                plt.axis('off')
        
        plt.axis('off')
        plt.savefig(f"{format_results_folder}/example_{room_ind}.png")
        plt.close()

        # pdb.set_trace()
        if generate_explanation:
            gt_program_text = ai2thor_utils.generate_program_from_roomjson(gt_house_json)
            failure_sentence = f"To generate a room that looks like this image: <image>, the predicted program was {room_json_text}. This generated a room with top-down view like this image: <image>. The refined program would be: {gt_program_text}"
            failure_explanations.append((failure_sentence, input_im, f"{image_save_folder}/example_top_down_{room_ind}.png"))
            # pdb.set_trace()


        # pdb.set_trace()
        #img = Image.fromarray(controller.last_event.frame)
        #img.save(f"vis/example_{room_ind}.png")

        controller.stop()
        #controller_gt.stop()

    if generate_explanation:
        with open(f"{exp_path}/failure_explanations.json", "w") as f:
            json.dump(failure_explanations, f)

if __name__=="__main__":
    use_language = True
    
    if use_language:
        assetdesc_to_obj = defaultdict(list)
        asset_descs = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json"))
        
        for asset in asset_descs:
            entries = asset_descs[asset]
            for im, obj, desc in entries:
                assetdesc_to_obj[desc].append((obj, asset))

            
        attrtokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        attrmodel = AutoModel.from_pretrained('intfloat/e5-base-v2').cuda()

        generate_output_houses("/projectnb/ivc-ml/array/research/robotics/dreamworlds/checkpoints/llava_incomplete_oneim_attr_campolygonangle_randompointorient/output.json", generate_explanation=False, use_language=True, assetdesc_to_obj=assetdesc_to_obj, attrtokenizer=attrtokenizer, attrmodel=attrmodel)

    else:
        generate_output_houses("/projectnb/ivc-ml/array/research/robotics/dreamworlds/checkpoints/llava_incomplete_oneim_caption_campolygon_orientlanguage/output.json", generate_explanation=False)
    