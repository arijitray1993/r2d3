from collections import defaultdict
import os
import h5py
import cv2
import numpy as np
import random
import json
import ast


def convert_byte_to_string(bytes_to_decode, max_len=None):
    if max_len is None:
        max_len = bytes_to_decode.shape[-1]
    return (bytes_to_decode.view(f"S{max_len}")[0]).decode()


def read_video_frames(video_path):
    """Reads frames from an MP4 video file."""

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def make_desc_for_task(task_name, task_templ):
    if task_name == "ObjectNavAffordance":
        task_desc = f"I need to go to the {task_templ['synsets'][0].split('.')[0]} that is best used to {task_templ['affordance'].lower().strip()} Which direction should I turn to face the object? Make sure to not run into obstacles."
    elif task_name == "PickupType":
        task_desc = f"I need to pickup {task_templ['synsets'][0].split('.')[0]}. Which direction should I turn to face the object?"
    elif task_name == "EasyObjectNavType":
        obj_name = task_templ['synsets'][0].split('.')[0]
        obj_name = obj_name.replace("_", " ")
        task_desc = f"I need to go to the {obj_name}. Which direction should I turn to face the object?" # Make sure to not run into obstacles."

    return task_desc


def get_current_state(controller):
    objid2assetid = {}
    for obj in controller.last_event.metadata['objects']:
        objid2assetid[obj['objectId']] = obj['assetId']
    nav_visible_objects = controller.step("GetVisibleObjects", maxDistance=5).metadata["actionReturn"]
    nav_visible_objects = [obj for obj in nav_visible_objects if objid2assetid[obj]!=""] # these are the visible object asset ids in the scene
    
    bboxes = controller.last_event.instance_detections2D
    vis_obj_to_size = {}
    for obj_id in bboxes:
        vis_obj_to_size[obj_id] = (bboxes[obj_id][2] - bboxes[obj_id][0])*(bboxes[obj_id][3] - bboxes[obj_id][1])


    objid2info = {}
    objdesc2cnt = defaultdict(int)
    for obj_entry in controller.last_event.metadata['objects']:
        obj_name = obj_entry['name']
        obj_type = obj_entry['objectType']
        asset_id = obj_entry['assetId']
        obj_id = obj_entry['objectId']
        
        distance = obj_entry['distance']
        pos = np.array([obj_entry['position']['x'], obj_entry['position']['y'], obj_entry['position']['z']])
        rotation = obj_entry['rotation']
        desc = assetid2desc.get(asset_id, obj_type)
        moveable = obj_entry['moveable'] or obj_entry['pickupable']
        
        asset_size_xy = vis_obj_to_size.get(obj_entry['objectId'], 0)
        asset_pos_box = bboxes.get(obj_entry['objectId'], None)
        if asset_pos_box is not None:
            asset_pos_xy = [(asset_pos_box[0]+asset_pos_box[2])/2, (asset_pos_box[1]+asset_pos_box[3])/2]
        else:
            asset_pos_xy = None

        parent = obj_entry.get('parentReceptacles')
        if parent is not None:
            if len(parent) > 0:
                parent = parent[-1]
                if parent== "Floor":
                    parent = "Floor"
                else:
                    parent = objid2assetid[parent]
        
        is_receptacle = obj_entry['receptacle']
        objid2info[obj_id] = (obj_name, obj_type, distance, pos, rotation, desc, moveable, parent, asset_size_xy, is_receptacle, asset_pos_xy)
        objdesc2cnt[obj_type] += 1

    
    moveable_visible_objs = []
    for objid in nav_visible_objects:
        if objid2info[objid][6] and objid2info[objid][8]>1600:
            moveable_visible_objs.append(objid)

    # pdb.set_trace()
    return nav_visible_objects, objid2info, objdesc2cnt, moveable_visible_objs


if __name__=="__main__":

    data_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/SPOC_data"
    split = "train"

    tasks = os.listdir(data_path)
    action_map = {
        'm': "go straight",
        'l': "left",
        'r': "right",
        'ls': "slight left",
        'rs': "slight right",
        'b': "move backward",
        'end': "no action, goal achieved",
    }
    action_map = {
        'move_ahead': "go straight",
        'rotate_left': "left",
        'rotate_right': "right",
        'rotate_left_small': "slight left",
        'rotate_right_small': "slight right",
    }
    selected_tasks = ["EasyObjectNavType"]#, "PickupType"] # "ObjectNavDescription",]
    action_choices = list(action_map.values())


    dataset = prior.load_dataset("procthor-10k")
    
    for task in tasks:
        if task not in selected_tasks:
            continue
        train_ids = os.listdir(os.path.join(data_path, task, split))
        
        #load train h5py
        for train_id in train_ids:
            h5_file = h5py.File(os.path.join(data_path, task, split, train_id, "hdf5_sensors.hdf5"), 'r')
            
            episodes = h5_file.keys()

            for episode_id in episodes:
                video_file = os.path.join(data_path, task, split, train_id, f"raw_navigation_camera__{episode_id}.mp4")
                episode = h5_file[episode_id]

                # these are the keys:
                # 'house_index', 'hypothetical_task_success', 'last_action_is_random', 'last_action_str', 'last_action_success', 
                # 'last_agent_location', 'minimum_l2_target_distance', 'minimum_visible_target_alignment', 'room_current_seen', 
                # 'rooms_seen', 'task_relevant_object_bbox', 'templated_task_spec', 'visible_target_4m_count'

                # convert the actions into strings -  they are stored in bytes. 
                actions_taken = []
                for action in episode['last_action_str']:
                    actions_taken.append(convert_byte_to_string(action))

                random_action = []
                for action in episode['last_action_is_random']:
                    random_action.append(action[0])
                
                last_action_success = []
                for action in episode['last_action_success']:
                    last_action_success.append(action[0])

                agent_locations = episode['last_agent_location']

                # Whether target object is visible or not for each frame
                object_bbox_cols = np.array(episode['nav_task_relevant_object_bbox']['max_cols'])
                obj_visible = [col[0]!=-1 for col in object_bbox_cols]

                # make a description for the task
                task_templ = ast.literal_eval(convert_byte_to_string(episode['templated_task_spec'][0]))
                task_desc = make_desc_for_task(task, task_templ)


                # instantiate a controller in the house
                house_ind = episode['house_index'][0]
                house_json = dataset[split][house_ind]
                
                try:
                    controller = Controller(scene=house_json, width=512, height=512, quality="Ultra", platform=CloudRendering,  renderInstanceSegmentation=True)
                except:
                    print("Cannot render environment, continuing")
                    continue


                for index, (action, rand_action, last_action_succ, obj_vis, agent_loc) in enumerate(zip(actions_taken, random_action, last_action_success, obj_visible, agent_locations)):
                    
                    if (not rand_action) and last_action_succ and index > 0:
                        if action in action_map:
                            action_word = action_map[action]
                        else:
                            action_word = action.replace("_", " ")

                        # teleport the agent to the last location
                        controller.step("Teleport", x=agent_loc[0], y=agent_loc[1], z=agent_loc[2], rotation=dict(x=0, y=agent_loc[3], z=0))

                        if "straight" in action_word:
                            continue
                        else:
                            # Get the closest object to controller
                            nav_visible_objects, objid2info, objdesc2cnt, moveable_visible_objs = get_current_state(controller)
                            
                            closest_obj = None
                            closest_dist = 1000
                            for obj in objid2info:
                                if objid2info[obj][2] < closest_dist:
                                    closest_dist = objid2info[obj][2]
                                    closest_obj = obj
                            
                            if closest_obj is None:
                                continue

                            action_so_far = f"go straight near {closest_obj}"

                            
                        


    split = "train"
    asset_id_desc = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/mturk_clean_assrt_desc/assetid_to_info.json", "r"))
    qa_json_path = f'/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/route_planning_qas.json'
    vis = True
    stats = False
    generate = False
    load_progress = False

    if generate:
        if not os.path.exists(qa_im_path):
            os.makedirs(qa_im_path)

        assetid2desc = {}
        for asset in asset_id_desc:
            entries = asset_id_desc[asset]
            captions = []
            for im, obj, desc in entries:
                desc = desc.strip().lower().replace(".", "")
                captions.append(desc)
            assetid2desc[asset] = random.choice(captions)


        dataset = prior.load_dataset("procthor-10k")

        all_im_qas = []
        
        if load_progress:
            all_im_qas = json.load(open(qa_json_path, "r"))

        for house_ind, house in enumerate(tqdm.tqdm(dataset[split])):
            
            house_json = house