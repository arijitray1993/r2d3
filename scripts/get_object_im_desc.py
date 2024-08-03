from ai2thor.controller import Controller
import json
import pdb
import sys
import prior
sys.path.append("../utils")
import numpy as np
from PIL import Image
import ai2thor_utils as utils
import tqdm
import os

if __name__=="__main__":
    dataset = prior.load_dataset("procthor-10k")

    all_objects = []
    for house in dataset["train"]:
        for obj in house['objects']:
            all_objects.append(obj['assetId'])

    empty_house_json = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/empty_house.json"
    
    all_objects = list(set(all_objects))
    for asset_id in tqdm.tqdm(all_objects):
        #if os.path.exists(f"all_obj_vis/{asset_id}.png"):
        #    continue
        with open(empty_house_json, "r") as f:
            house = json.load(f)
        house = utils.House()
        floor_polygon = [[0, 0, 0], [0, 0, 600], [600, 0, 600], [600, 0, 0]]
        house.make_floorplan_walls({
            'polygon': floor_polygon,
            'floor_material': "PureWhite",
            'wall_material': ["PureWhite", "PureWhite", "PureWhite", "PureWhite"]
        })
        house = house.house_json
        #house.add_object({
        #        'assetId': asset_id,
        #        'position': [300, 50, 300],
        #        'rotation': [0, 0, 0],
        #        'id': 'obj_0'
        #    })
        #house["proceduralParameters"]["skyboxColor"] = {
        #    "r": 255,
        #    "g": 255,
        #    "b": 255,
        #}
        skybox_color=(0, 0, 0)
        instance_id="asset_0"
        house["objects"] = [
            {
                "assetId": asset_id,
                "id": instance_id,
                "kinematic": True,
                "position": {"x": 3, "y": 0.25, "z": 3},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "layer": "Procedural2",
                "material": None,
            }
        ]
        house["proceduralParameters"]["skyboxColor"] = {
            "r": skybox_color[0],
            "g": skybox_color[1],
            "b": skybox_color[2],
        }

        controller = Controller(scene="Procedural", makeAgentsVisible=False, width=500, height=500, quality="High WebGL")
        controller.step(action="CreateHouse", house=house)

        
        # pdb.set_trace()
        
        controller.step("BBoxDistance", objectId0=instance_id, objectId1=instance_id)
        for obj_entry in controller.step("AdvancePhysicsStep").metadata["objects"]:
            if obj_entry["name"] == "asset_0":
                obj= obj_entry
                break
        # pdb.set_trace()
        if obj["objectOrientedBoundingBox"] is None:
            # pdb.set_trace()
            controller.stop()
            continue
        obj_center_arr = np.array(obj["objectOrientedBoundingBox"]["cornerPoints"]).mean(0)

        evt = controller.step(action="LookAtObjectCenter", objectId=instance_id)

        #for rotation in rotations:
        #    controller.step(
        #        action="RotateObject",
        #        angleAxisRotation={
        #            "axis": {
        #                "x": rotation[0],
        #                "y": rotation[1],
        #                "z": rotation[2],
        #            },
        #            "degrees": rotation[3],
        #        },
        #    )
        for obj_entry in controller.last_event.metadata["objects"]:
            if obj_entry["name"] == "asset_0":
                obj= obj_entry
        
        delta = obj_center_arr - np.array(
            obj["objectOrientedBoundingBox"]["cornerPoints"]
        ).mean(0)

        cur_pos = obj["position"]
        target_pos = {
            "x": cur_pos["x"] + delta[0],
            "y": cur_pos["y"] + delta[1]+0.5,
            "z": cur_pos["z"] + delta[2],
        }

        target_pos['x']-=1
        target_pos['z']+=1

        controller.step(
            action="TeleportObject",
            objectId=instance_id,
            position=target_pos,
            rotation=obj["rotation"],
            forceAction=True,
            forceKinematic=True,
        )

        controller.step(
            action="RandomizeLighting",
            brightness=(1.5, 1.7),
            randomizeColor=False,
            synchronized=False
        )

        # pdb.set_trace()
        
        im = Image.fromarray(controller.last_event.frame)
        
        # pdb.set_trace()
        im.save(f"all_obj_vis/{asset_id}.png")
        
        controller.stop()
        
