# run a bunch of apartment images through llava along with the scene graph info to get descriptions. 

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


if __name__ == "__main__":

    playground = False

    generate_ai2thor_json_images = True

    if generate_ai2thor_json_images:
        
        program_json_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/ProcTHOR/data/all_room_json_programs_ai2_train.json", "r"))

        #load progress so far
        load_progress = True
        #try:
        if load_progress:
            image_program_json_data = json.load(open("../custom_datasets/procThor/all_room_json_programs_ai2_train_room_descriptions.json", "r"))
        else:
            image_program_json_data = []

        print("length of already done data", len(image_program_json_data))

        # list of images done
        done_images = []
        for program_text, house_json, cam_ind_to_position, all_imgs, all_objs in image_program_json_data:
            done_images.extend(all_imgs)

        #except:

        for ind, (program_text, house_json) in enumerate(tqdm.tqdm(program_json_data)):

            # if we have to continue from a certain index
            if ind < 100: #len(image_program_json_data):
                continue

            if ind in [5, 11, 23, 32]: # everything crashes unexpectedly for these, to do look into later
                continue

            if f"vis/ai2thor/example_{ind}_0.png" in done_images:
                print("already done")
                continue

            # render the json
            try:
                controller = Controller(scene=house_json, width=800,height=800, renderInstanceSegmentation=True)
            except:
                print("Cannot render environment, continuing")
                # pdb.set_trace()
                continue

            ## place the camera at the room corners and take pictures of room.
            # 1. get reachable positions 
            try:
                event = controller.step(action="GetReachablePositions")
            except:
                print("Cannot get reachable positions, continuing")
                continue
            reachable_positions = event.metadata["actionReturn"]
            try:
                print("number of reachable positions: ", len(reachable_positions))
            except:
                print("no reachable positions, continuing")
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
            
            all_imgs = []
            all_objects = []
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
                position = random.choice(closest_reachable_positions[:10])

                position['x'] = round(position['x']*4)/4.0
                position['z'] = round(position['z']*4)/4.0
                position['y'] = 0.9
                rotation = { "x": 0.0, "y": angle, "z": 0.0} 

                # check if position is in polygon
                if not shape_polygon.contains(Point(position['x'], position['z'])):
                    print("not in polygon")
                    continue             
                
                try:
                    event = controller.step(action="Teleport", position=position, rotation=rotation)
                except:
                    print("teleport failed")
                    continue
                
                #top_down_frame = get_top_down_frame(controller)
                #top_down_frame.save(f"vis/ai2thor/example_top_down_{ind}.png")
                # pdb.set_trace()
                # img = Image.fromarray(controller.last_event.frame)
                # img.save(f"vis/ai2thor/example_{ind}_{corner_ind}.png")

                cam_ind_to_position[corner_ind] = (position, rotation)
                all_imgs.append(f"vis/ai2thor/example_{ind}_{corner_ind}.png")

                # let's get the objects etc in the scene to create description
                objects = []
                for key in controller.last_event.instance_detections2D:
                    objects.append((key.split("|")[0], controller.last_event.instance_detections2D[key]))
                
                all_objects.append(objects)
                # pdb.set_trace()
                # get description from llava model.
                # todo


            image_program_json_data.append((program_text, house_json, cam_ind_to_position, all_imgs, all_objects))
            

            json.dump(image_program_json_data, open("../custom_datasets/procThor/all_room_json_programs_ai2_train_room_descriptions.json", "w"))
            controller.stop()


    if playground:
        #dataset = prior.load_dataset("procthor-10k")

        #house = dataset["train"][0]

        print("=======HOUSE params ========")
        #print(house)

        #change color of walls to green
        #for ind, wall in enumerate(house['walls']):
        #    wall['color'] = {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0}
        #    house['walls'][ind] = wall
        #    wall['material']['color'] = {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0}
        #    wall['material']['name'] = 'LivingRoomFireplacebrickTex'

        #print("updated walls")
        #print(house)

        #json.dump(house, open("house_gt.json", "w"))
        #exit()
        #print("=======SCENE params ========")

        house = json.load(open("example_house.json", "r"))

        controller = Controller(scene=house, width=500,height=500, renderInstanceSegmentation=True)

        #event = controller.step(action="GetReachablePositions")
        #reachable_positions = event.metadata["actionReturn"]
        #position = reachable_positions[0]
        #rotation = random.choice(range(360))
        #event = controller.step(action="Teleport", position=position, rotation=rotation)

        img = Image.fromarray(controller.last_event.frame)
        img.save("vis/example_0.png")

        # save image of top down view
        top_down_frame = get_top_down_frame(controller)
        top_down_frame.save("vis/example_top_down.png")

        '''
        event = controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]

        #position = random.choice(reachable_positions)
        position = reachable_positions[0]

        for i in range(1):
            rotation = random.choice(range(360))
            print("Teleporting the agent to", position, " with rotation", rotation)

            event = controller.step(action="Teleport", position=position, rotation=rotation)
            
            seg_frame = controller.last_event.instance_segmentation_frame

            print(seg_frame)

            # meta_data = controller.last_event.metadata

            print("=======instance objects ========") 
            #pprint(meta_data)

            for key in controller.last_event.instance_detections2D:
                print(key.split("|")[0], controller.last_event.instance_detections2D[key])

            img = Image.fromarray(event.frame)
            img.save("vis/example-%d.png" % i)

            seg_img = Image.fromarray(seg_frame)
            seg_img.save("vis/example-seg-%d.png" % i)
        '''

