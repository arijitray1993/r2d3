from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import json

import yaml
import ast

def tokenize_polygon(polygon):
    # get the tokens for the x,y,z coordinates of the polygon
    tokens = []
    for point in polygon:
        token_x = int(point['x']/8*31)
        token_y = int(point['y']/8*31)
        token_z = int(point['z']/8*31)
        tokens.append((token_x, token_y, token_z))
    
    return tokens

def get_token_from_coordinate(x,y,z):
    # 32 tokens represent 0 to 8. Convert 0-8 for x, y, z to 0-31 tokens

    # get the number from the token
    token_x = int(x/8*31)
    token_y = int(y/8*31)
    token_z = int(z/8*31)

    return token_x, token_y, token_z

def get_token_from_rotation(x,y,z):
    # 360 tokens represent 0 to 360. Convert 0-360 for x, y, z to 0-359 tokens

    # get the number from the token
    token_x = int(x/360*359)
    token_y = int(y/360*359)
    token_z = int(z/360*359)

    return token_x, token_y, token_z

def get_coordinate_from_token(token):
    # 32 tokens represent 0 to 8. Convert 0-31 to 0-8

    # get the number from the token
    num = int(token)

    # now get corrdinate
    coordinate = num/31*8

    return coordinate

def get_rotation_from_token(token):
    # 360 tokens represent 0 to 360. Convert 0-359 to 0-360

    # get the number from the token
    num = int(token)

    # now get rotation
    rotation = num/359*360

    return rotation

def get_rotation_from_tokens(token_x, token_y, token_z):
    # get the x,y,z coordinates from the tokens
    x = get_rotation_from_token(token_x)
    y = get_rotation_from_token(token_y)
    z = get_rotation_from_token(token_z)

    return {"x": x, "y": y, "z": z}

def get_xyz_from_tokens(token_x, token_y, token_z):

    # get the x,y,z coordinates from the tokens
    x = get_coordinate_from_token(token_x)
    y = get_coordinate_from_token(token_y)
    z = get_coordinate_from_token(token_z)

    return {"x": x, "y": y, "z": z}


def generate_program_from_roomjson(house_json):
    # for each room, we need polygon, floor material, wall materials, and objects at location and rotation

    
    room_id = house_json['rooms'][0]['id']
    # get polygon
    polygon = house_json['rooms'][0]['floorPolygon']

    # get floor material
    floor_material = house_json['rooms'][0]['floorMaterial']['name']

    # get wall materials
    wall_materials = []
    for wall in house_json['walls']:
        if wall['roomId'] == room_id:
            wall_materials.append(wall['material']['name'])
    
    # tokenize polygon, floor material, and wall materials
    polygon_tokens = tokenize_polygon(polygon)
    floor_material_token = floor_material
    wall_material_tokens = wall_materials

    # get objects
    shapely_polygon = Polygon([(room_point['x'], room_point['z']) for room_point in polygon]) # this is room polygon
    objects = []
    for obj in house_json['objects']:
        # use shapely polygon to check if the object is inside the room
        point = Point(obj['position']['x'], obj['position']['z'])
        if shapely_polygon.contains(point): # check if object point in room polygon
            objects.append(obj)
        
    
    # make a program
    program_text = f"""
polygon: {polygon_tokens}
floor_material: {floor_material_token}
wall_material: {wall_material_tokens}
"""
    for ind, obj in enumerate(objects):
        obj_token = obj['assetId']
        obj_position = get_token_from_coordinate(obj['position']['x'], obj['position']['y'], obj['position']['z'])
        obj_rotation = get_token_from_rotation(obj['rotation']['x'], obj['rotation']['y'], obj['rotation']['z'])
        obj_id = obj['id']

        obj_entry_dict = (obj_token, obj_position, obj_rotation)
        
        program_text += f"""
obj_{ind}: {obj_entry_dict}
"""
    return program_text


class House:

    def __init__(self, args={}, house_json=""):
        if house_json == "":
            self.house_json = self.generate_house_template()
        else:
            self.house_json = house_json
        
        #self.object_choices = json.load(open("all_objects.json", "r"))
        #self.wall_material_choices = json.load(open("all_wall_materials.json", "r"))
        #self.room_type_choices = json.load(open("all_room_types.json", "r"))
        #self.floor_material_choices = json.load(open("all_floor_materials.json", "r"))

    # utils
    def get_object_from_token(self, token):
        # get the object from the token
        obj = self.object_choices[int(token.split("_")[1])]
        return obj

    def get_wall_material_from_token(self, token):
        # get the wall material from the token
        wall_material = self.wall_material_choices[int(token.split("_")[1])]
        return wall_material

    def get_floor_material_from_token(self, token):
        # get the floor material from the token
        floor_material = self.floor_material_choices[int(token.split("_")[1])]
        return floor_material
    
    def get_token_from_object(self, obj):
        # get the token from the object
        token = f"object_{self.object_choices.index(obj)}"
        return token

    def get_token_from_wall_material(self, wall_material):
        token = f"wallmaterial_{self.wall_material_choices.index(wall_material)}"
        return token

    def get_token_from_floor_material(self, floor_material):
        token = f"floormaterial_{self.floor_material_choices.index(floor_material)}"
        return token

    # house generation functions
    def generate_house_template(self,):
        # generate emtpy house json structure
        house_json = {
            "doors": [],
            "metadata": {
                "agent": {
                    "horizon": 30,
                    "position": {
                        "x": 3.5,
                        "y": 0.95,
                        "z": 2
                    },
                    "rotation": {
                        "x": 0,
                        "y": 90,
                        "z": 0
                    },
                    "standing": True
                },
                "roomSpecId": "kitchen",
                "schema": "1.0.0",
                "warnings": {},
                "agentPoses": {
                    "arm": {
                        "horizon": 30,
                        "position": {
                        "x": 3.5,
                        "y": 0.95,
                        "z": 2
                        },
                        "rotation": {
                        "x": 0,
                        "y": 90,
                        "z": 0
                        },
                        "standing": True
                    },
                    "default": {
                        "horizon": 30,
                        "position": {
                        "x": 3.5,
                        "y": 0.95,
                        "z": 2
                        },
                        "rotation": {
                        "x": 0,
                        "y": 90,
                        "z": 0
                        },
                        "standing": True
                    },
                    "locobot": {
                        "horizon": 30,
                        "position": {
                        "x": 3.5,
                        "y": 0.95,
                        "z": 2
                        },
                        "rotation": {
                        "x": 0,
                        "y": 90,
                        "z": 0
                        }
                    },
                    "stretch": {
                        "horizon": 30,
                        "position": {
                        "x": 3.5,
                        "y": 0.95,
                        "z": 2
                        },
                        "rotation": {
                        "x": 0,
                        "y": 90,
                        "z": 0
                        },
                        "standing": True
                    }
                }
            },
            "objects": [],
            "proceduralParameters": {
                "ceilingColor": {
                "b": 0.3058823529411765,
                "g": 0.3843137254901961,
                "r": 0.42745098039215684
                },
                "ceilingMaterial": {
                "name": "PureWhite",
                "color": {
                    "b": 0.3058823529411765,
                    "g": 0.3843137254901961,
                    "r": 0.42745098039215684
                }
                },
                "floorColliderThickness": 1,
                "lights": [
                {
                    "id": "DirectionalLight",
                    "indirectMultiplier": 1,
                    "intensity": 1,
                    "position": {
                    "x": 0.84,
                    "y": 0.1855,
                    "z": -1.09
                    },
                    "rgb": {
                    "r": 1,
                    "g": 1,
                    "b": 1
                    },
                    "rotation": {
                    "x": 66,
                    "y": 75,
                    "z": 0
                    },
                    "shadow": {
                    "type": "Soft",
                    "strength": 1,
                    "normalBias": 0,
                    "bias": 0,
                    "nearPlane": 0.2,
                    "resolution": "FromQualitySettings"
                    },
                    "type": "directional"
                },
                {
                    "id": "light_2",
                    "intensity": 0.45,
                    "position": {
                    "x": 3.9899999999999998,
                    "y": 4.351928794929904,
                    "z": 2.49375
                    },
                    "range": 15,
                    "rgb": {
                    "r": 1,
                    "g": 0.855,
                    "b": 0.722
                    },
                    "shadow": {
                    "type": "Soft",
                    "strength": 1,
                    "normalBias": 0,
                    "bias": 0.05,
                    "nearPlane": 0.2,
                    "resolution": "FromQualitySettings"
                    },
                    "type": "point"
                }
                ],
                "receptacleHeight": 0.7,
                "reflections": [],
                "skyboxId": "SkyAlbany"
            },
            "rooms": [
                {
                "ceilings": [],
                "children": [],
                "floorMaterial": {
                    "name": ""
                },
                "floorPolygon": [
                    
                ],
                "id": "",
                "roomType": ""
                }
            ],
            "walls": [],
            "windows": []
        }

        return house_json

    def make_floorplan_walls(self, args):
        '''
        expects keys:
        - polygon: list of x,y,z coordinates that join to make a polygon
        - floor_material
        - wall_material: list of wall materials for each wall (number of edges in polygon)
        '''
        polygon = args['polygon']
        floor_material = args['floor_material']

        polygon = [get_xyz_from_tokens(token_x, token_y, token_z) for token_x, token_y, token_z in polygon]
        
        self.house_json['rooms'][0]['floorPolygon'] = polygon
        self.house_json['rooms'][0]['floorMaterial']['name'] = floor_material
        self.house_json['rooms'][0]['id'] = "room_0"

        # make walls from polygon
        walls = []
        for i in range(len(polygon)):
            wall_height_point_1 = polygon[i].copy()
            wall_height_point_1['y'] = 4.551928794929904
            wall_height_point_2 = polygon[(i+1)%len(polygon)].copy()
            wall_height_point_2['y'] = 4.551928794929904

            wall = {
                "id": i,
                "color": {
                    "r": 1,
                    "g": 1,
                    "b": 1,
                    "a": 1
                },
                "material": {
                    "name": args['wall_material'][i],
                    "color": {
                        "r": 1,
                        "g": 1,
                        "b": 1,
                        "a": 1
                    },
                },
                "roomId": "room_0",
                "polygon": [
                    polygon[i],
                    polygon[(i+1)%len(polygon)],
                    wall_height_point_1,
                    wall_height_point_2
                ]
            }
            walls.append(wall)
            wall_exterior = wall.copy()
            wall_exterior['roomId'] = "exterior"
            wall_exterior['id'] = f"exterior_{i}"
            walls.append(wall_exterior)

        self.house_json['walls'] = walls

        # fix the agent position based on the floor ploygon to make sure it is inside the room
        self.house_json['metadata']['agent']['position']['x'] = polygon[0]['x']+1
        self.house_json['metadata']['agent']['position']['z'] = polygon[0]['z']+1
        for entry in self.house_json['metadata']['agentPoses']:
            self.house_json['metadata']['agentPoses'][entry]['position']['x'] = polygon[0]['x']+1
            self.house_json['metadata']['agentPoses'][entry]['position']['z'] = polygon[0]['z']+1

        # fix the light position based on the floor ploygon to make sure it is inside the room
        self.house_json['proceduralParameters']['lights'][1]['position']['x'] = polygon[0]['x']+1
        self.house_json['proceduralParameters']['lights'][1]['position']['z'] = polygon[0]['z']+1


    def add_object(self, args):
        '''
        expects keys:
        - assetId: id of the object, used to identify the object from a list of possible choices.
        - position: x,y,z coordinates of the object
        - rotation: x,y,z angles 0-360 - what does this exactly mean? not sure
        '''
        obj = {
            "assetId": args['assetId'],
            "position": get_xyz_from_tokens(*args['position']),
            "rotation": get_rotation_from_tokens(*args['rotation']),
            "id": args['id'],
        }
        self.house_json['objects'].append(obj)

    def add_object_children(self, args):
        '''
        expects keys:
        - children: list of objects that are children of other objects
            each children object should have:
            - parent_id: id of the parent object. we will attach the child object to this parent object
            - assetId: id of the object, used to identify the object from a list of possible choices.
            - position: x,y,z coordinates of the object
            - rotation: x,y,z angles 0-360 - what does this exactly mean? not sure
        '''
        child = args
        child_obj = {
            "assetId": child['assetId'],
            "position": get_xyz_from_tokens(*child['position']),
            "rotation": get_rotation_from_tokens(*child['rotation']),
            "id": child['id'],
            "parent_id": child['parent_id']
        }
        
        # search for the house object with the parent id
        for obj in self.house_json['objects']:
            if obj['id'] == child_obj['parent_id']:
                if 'children' not in obj:
                    obj['children'] = []

                child_obj_to_add = child_obj.copy()
                del child_obj_to_add['parent_id']
                obj['children'].append(child_obj_to_add)
                break
    
def make_house_from_cfg(cfg):
    '''
    cfg is a string like:
    polygon: [[0, 0, 0], [0, 0, 15], [15, 0, 15], [15, 0, 0]] # note there are no parantheses, only square brackets
    floor_material: LightWoodCounters3
    wall_material: ['PureWhite', 'PureWhite', 'PureWhite', 'PureWhite']

    obj_0: ['Toilet_2', [1, 1, 1], [0, 0, 0]]

    obj_1: ['Sink_26', [5, 1, 1], [0, 0, 0]]
    '''

    # get dictionary from yaml-like string
    cfg_dict = yaml.load(cfg, Loader=yaml.FullLoader)

    floor_polygon = cfg_dict['polygon']
    floor_material = cfg_dict['floor_material']
    wall_material = cfg_dict['wall_material']

    # make house
    house = House()
    house.make_floorplan_walls({
        'polygon': floor_polygon,
        'floor_material': floor_material,
        'wall_material': wall_material
    })

    # get obj0, obj1, ... until we run out of objects in dict
    i = 0
    while(True):
        if f'obj_{i}' in cfg_dict:
            obj = cfg_dict[f'obj_{i}']
            # print(obj[1])
            house.add_object({
                'assetId': obj[0],
                'position': list(obj[1]),
                'rotation': list(obj[2]),
                'id': f'obj_{i}'
            })
            i += 1
        else:
            break
    
    return house