import json
import os
import pdb  # noqa
import random
from collections import defaultdict
import csv
import numpy as np
import scipy
import scipy.spatial
import cv2
from PIL import Image

class_names = [
    "cabinet", "refrigerator", "shelf", "stove", "bed", # 0..5
    "sink", "washer", "toilet", "bathtub", "oven", # 5..10
    "dishwasher", "fireplace", "stool", "chair", "table", # 10..15
    "tv_monitor", "sofa", # 15..17
]

def extract_gt(gt_fn):
    ### copied from https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py
    """extract original label data

    Args:
        gt_fn: str (file name of "annotation.json")
            after loading, we got a dict with keys
                'data', 'stats', 'comment', 'confirm', 'skipped'
            ['data']: a list of dict for bboxes, each dict has keys:
                'uid', 'label', 'modelId', 'children', 'objectId',
                'segments', 'hierarchy', 'isInGroup', 'labelType', 'attributes'
                'label': str
                'segments': dict for boxes
                    'centroid': list of float (x, y, z)?
                    'axesLengths': list of float (x, y, z)?
                    'normalizedAxes': list of float len()=9
                'uid'
            'comments':
            'stats': ...
    Returns:
        skipped: bool
            skipped or not
        boxes_corners: (n, 8, 3) box corners
            **world-coordinate**
        centers: (n, 3)
            **world-coordinate**
        sizes: (n, 3) full-sizes (no halving!)
        labels: list of str
        uids: list of str
    """
    gt = json.load(open(gt_fn, "r"))
    skipped = gt['skipped']
    if len(gt) == 0:
        boxes_corners = np.zeros((0, 8, 3))
        centers = np.zeros((0, 3))
        sizes = np.zeros((0, 3))
        labels, uids = [], []
        return skipped, boxes_corners, centers, sizes, labels, uids

    boxes_corners = []
    centers = []
    sizes = []
    labels = []
    uids = []
    for data in gt['data']:
        l = data["label"]
        for delimiter in [" ", "-", "/"]:
            l = l.replace(delimiter, "_")
        if l not in class_names:
            print("unknown category: %s" % l)
            continue

        rotmat = np.array(data["segments"]["obbAligned"]["normalizedAxes"]).reshape(
            3, 3
        )
        center = np.array(data["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
        size = np.array(data["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)
        box3d = compute_box_3d(size.reshape(3).tolist(), center, rotmat)

        '''
            Box corner order that we return is of the format below:
                6 -------- 7
               /|         /|
              5 -------- 4 .
              | |        | |
              . 2 -------- 3
              |/         |/
              1 -------- 0 
        '''

        boxes_corners.append(box3d.reshape(1, 8, 3))
        size = np.array(get_size(box3d)).reshape(1, 3)
        center = np.mean(box3d, axis=0).reshape(1, 3)

        # boxes_corners.append(box3d.reshape(1, 8, 3))
        centers.append(center)
        sizes.append(size)
        # labels.append(l)
        labels.append(data["label"])
        uids.append(data["uid"])
    centers = np.concatenate(centers, axis=0)
    sizes = np.concatenate(sizes, axis=0)
    boxes_corners = np.concatenate(boxes_corners, axis=0)
    return skipped, boxes_corners, centers, sizes, labels, uids

def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix

def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])

def decide_pose(pose):
    """
    Args:
        pose: np.array (4, 4)
    Returns:
        index: int (0, 1, 2, 3)
        for upright, left, upside-down and right
    """
    # pose style
    z_vec = pose[2, :3]
    z_orien = np.array(
        [
            [0.0, -1.0, 0.0],  # upright
            [-1.0, 0.0, 0.0],  # left
            [0.0, 1.0, 0.0],  # upside-down
            [1.0, 0.0, 0.0],
        ]  # right
    )
    corr = np.matmul(z_orien, z_vec)
    corr_max = np.argmax(corr)
    return corr_max


def rotate_pose(im, rot_index):
    """
    Args:
        im: (m, n)
    """
    h, w, d = im.shape
    if d == 3:
        if rot_index == 0:
            new_im = im
        elif rot_index == 1:
            new_im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif rot_index == 2:
            new_im = cv2.rotate(im, cv2.ROTATE_180)
        elif rot_index == 3:
            new_im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return new_im


def get_size(box):
    """
    Args:
        box: 8x3
    Returns:
        size: [dx, dy, dz]
    """
    distance = scipy.spatial.distance.cdist(box[0:1, :], box[1:5, :])
    l = distance[0, 2]
    w = distance[0, 0]
    h = distance[0, 3]
    return [l, w, h]


def compute_box_3d(size, center, rotmat):
    """Compute corners of a single box from rotation matrix
    Args:
        size: list of float [dx, dy, dz]
        center: np.array [x, y, z]
        rotmat: np.array (3, 3)
    Returns:
        corners: (8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    l, h, w = [i / 2 for i in size]
    center = np.reshape(center, (-1, 3))
    center = center.reshape(3)
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [h, -h, -h, h, h, -h, -h, h]
    z_corners = [w, w, w, w, -w, -w, -w, -w]
    corners_3d = np.dot(
        np.transpose(rotmat), np.vstack([x_corners, y_corners, z_corners])
    )
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return np.transpose(corners_3d)

def rotate_pc(pc, rotmat):
    """Rotation points w.r.t. rotmat
    Args:
        pc: np.array (n, 3)
        rotmat: np.array (4, 4)
    Returns:
        pc: (n, 3)
    """
    pc_4 = np.ones([pc.shape[0], 4])
    pc_4[:, 0:3] = pc
    pc_4 = np.dot(pc_4, rotmat.T)

    return pc_4[:, 0:3]

def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has six columns:
        * Columns 1-3: rotation (axis-angle representation in radians)
        * Columns 4-6: translation (usually in meters)

    Returns:
        Rt: rotation matrix
    """
    # line=[float(x) for x in traj_str.split()]
    # ts = line[0];
    # R = cv2.Rodrigues(np.array(line[1:4]))[0];
    # t = np.array(line[4:7]);
    # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
    tokens = traj_str.split()
    #assert len(tokens) == 7
    #ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[0]), float(tokens[1]), float(tokens[2])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[3]), float(tokens[4]), float(tokens[5])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return Rt


def transform_to_new_camera_coords(x, y, z, p, q, r, a1, a2, a3):
    # Step 1: Translate the point relative to the camera position
    translated_point = np.array([x - p, y - q, z - r])

    # Step 2: Compute rotation matrices for each Euler angle
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(a1), -np.sin(a1)],
        [0, np.sin(a1), np.cos(a1)]
    ])

    R_y = np.array([
        [np.cos(a2), 0, np.sin(a2)],
        [0, 1, 0],
        [-np.sin(a2), 0, np.cos(a2)]
    ])

    R_z = np.array([
        [np.cos(a3), -np.sin(a3), 0],
        [np.sin(a3), np.cos(a3), 0],
        [0, 0, 1]
    ])

    # Step 3: Combine the rotation matrices
    R = R_z @ R_y @ R_x  # Use @ for matrix multiplication

    # Step 4: Apply the rotation to the translated point
    transformed_point = R @ translated_point

    return transformed_point


def compute_visible_objects(intrinsics, camera_pos, obj_bbox_ann):
    # instrinsics are reprensted by a list of these values: 
    #  width height focal_length_x focal_length_y principal_point_x principal_point_y

    # camera_pos is represented by a list containing rotation along x, y, z in radians and translation along x, y, z

    # obj_bbox_ann is a json file containing the following
    #    |-- data[]: list of bounding box data
    #    |  |-- label: object name of bounding box
    #    |  |-- axesLengths[x, y, z]: size of the origin bounding-box before transforming
    #    |  |-- centroid[]: the translation matrix（1*3）of bounding-box
    #    |  |-- normalizedAxes[]: the rotation matrix（3*3）of bounding-box 

    # use compute_box_3d(axeslengths, center, rotmat) to get the corner points of the bounding box

    # retunrns a list of objects that are visible in the image along with their 3D bounding boxes normalized to the camera coordinate system
    visible_objs = []
    obj_3d_data = extract_gt(obj_bbox_ann)
    if obj_3d_data[0]:
        return visible_objs
    
    for boxes_corner, center, size, label, uid in zip(*obj_3d_data[1:]): 
        
        #center = np.array(center).astype(np.float32)
        camera_pos = np.array(camera_pos).astype(np.float32)
        intrinsics = np.array(intrinsics).astype(np.float32)
        
        # cam_extrinsic_mat = TrajStringToMatrix(" ".join([str(x) for x in camera_pos]))
        
        center = np.array(center).astype(np.float32)
        # cam_extrinsic_mat = np.array(cam_extrinsic_mat).astype(np.float32)

        # transformed_center = rotate_pc(center, cam_extrinsic_mat).squeeze()

        # pdb.set_trace()
        # use the camera intrinsics and camera pose to project the 3D bounding box to the image plane
        # let's just do centroid for now   
        transformed_center = transform_to_new_camera_coords(center[0], center[1], center[2], camera_pos[3], camera_pos[4], camera_pos[5], camera_pos[0], camera_pos[1], camera_pos[2])

        #if z is negative, the object is behind the camera
        if transformed_center[1] < 0:
            continue

        # project the transformed center to the camera image plane
        focal_length_x = intrinsics[2]
        focal_length_z = intrinsics[3]
        principal_point_x = intrinsics[4]
        principal_point_z = intrinsics[5]

        cam_obj_x = (transformed_center[0] * focal_length_x / transformed_center[1]) + principal_point_x
        cam_obj_y = (transformed_center[2] * focal_length_z / transformed_center[1]) + principal_point_z

        im_width = intrinsics[0]
        im_height = intrinsics[1]

        if cam_obj_x < 10 or cam_obj_x >= im_width-10 or cam_obj_y < 10 or cam_obj_y >= im_height-10:
            continue

        cam_obj_x = cam_obj_x / im_width
        cam_obj_y = cam_obj_y / im_height
            
        visible_objs.append((label, cam_obj_x, cam_obj_y, transformed_center))

    return visible_objs


if __name__== "__main__":
    data_folder = "/projectnb/ivc-ml/array/data/ARKitScenes"
    objd_data =  os.path.join(data_folder, "data_files", "3dod", "Validation")

    data = []
    scans = os.listdir(objd_data)
    for scan in scans:
        scan_path = os.path.join(objd_data, scan)
        frames_path = os.path.join(scan_path, f"{scan}_frames", "lowres_wide")
        intrinsics_path = os.path.join(scan_path, f"{scan}_frames", "lowres_wide_intrinsics")
        
        frames = random.sample(os.listdir(frames_path), min(700, len(os.listdir(frames_path))))
        traj_file = os.path.join(scan_path, f"{scan}_frames", f"lowres_wide.traj")
        #open the space delinmited file
        trajs = {}
        with open(traj_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                time_stamp = round(float(line[0]), 3)
                trajs[time_stamp] = line[1:]
        
        obj_bbox_ann = os.path.join(scan_path, f"{scan}_3dod_annotation.json")

        for frame in frames:
            frame_path = os.path.join(frames_path, frame)
            
            intrinsics = os.path.join(intrinsics_path, frame.replace(".png", ".pincam"))
            intrinsics = open(intrinsics, "r").read().strip().split(" ")
            try:
                camera_pos = trajs[float(frame.split("_")[1].split(".png")[0])]
            except:
                continue

            visible_objs = compute_visible_objects(intrinsics, camera_pos, obj_bbox_ann)
            
            unique_visible_objs = set([x[0] for x in visible_objs])

            if len(unique_visible_objs) > 2:
                pose = TrajStringToMatrix(" ".join([str(x) for x in camera_pos]))
                orientation_index = decide_pose(pose)  # index 2, means image should be rotated 180-deg
                im = rotate_pose(np.array(Image.open(frame_path)), orientation_index)
                im = Image.fromarray(im)

                data.append((frame_path, intrinsics, camera_pos, visible_objs))

                pdb.set_trace()
    
    # compute precise 3d captioning data



    # compute precise 3D questioning data


