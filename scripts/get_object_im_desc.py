from ai2thor.controller import Controller
import json
import pdb
import sys
sys.path.append("../utils")
import ai2thor_utils as utils

if __name__=="__main__":
    dataset = prior.load_dataset("procthor-10k")

    all_objects = []
    for house in dataset["train"]:
        for obj in house['objects']:
            all_objects.append(obj['assetId'])

    empty_house_json = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/empty_house.json"
    
    for asset_id in object_assetid_list:
        with open(empty_house_json, "r") as f:
            house = json.load(f)
        #house = utils.House()
        #floor_polygon = [[0, 0, 0], [0, 0, 600], [600, 0, 600], [600, 0, 0]]
        #house.make_floorplan_walls({
        #    'polygon': floor_polygon,
        #    'floor_material': "PureWhite",
        #    'wall_material': ["PureWhite", "PureWhite", "PureWhite", "PureWhite"]
        #})

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
        
        #controller = Controller(scene="Procedural", makeAgentsVisible=False, width=900, height=900) # renderInstanceSegmentation=True)
        #evt = controller.step(action="CreateHouse", house=house.house_json)

        evt = Controller(scene="Procedural", makeAgentsVisible=False, width=900, height=900)
        controller.step(action="CreateHouse", house=house)
        controller.step(action="LookAtObjectCenter", objectId="obj_0")

        
        pdb.set_trace()
        
        # angles = [r[-1] for r in rotations]

        
