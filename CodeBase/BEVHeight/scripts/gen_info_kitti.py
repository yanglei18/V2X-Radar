import os
import csv
import math
import random
import argparse

import pickle
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

name2nuscenceclass = {
    "Car": "vehicle.car",
    "Van": "vehicle.car",
    "Truck": "vehicle.truck",
    "Bus": "vehicle.bus.rigid",
    "Cyclist": "vehicle.bicycle",
    "Pedestrian": "human.pedestrian.adult",
}

def alpha2roty(alpha, pos):
    ry = alpha + np.arctan2(pos[0], pos[2])
    if ry > np.pi:
        ry -= 2 * np.pi
    if ry < -np.pi:
        ry += 2 * np.pi
    return ry

def clip2pi(ry):
    if ry > 2 * np.pi:
        ry -= 2 * np.pi
    if ry < - 2* np.pi:
        ry += 2 * np.pi
    return ry

def equation_plane(points): 
    x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
    x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
    x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return np.array([a, b, c, d])

def get_denorm(rotation_matrix, translation):
    lidar2cam = np.eye(4)
    lidar2cam[:3, :3] = rotation_matrix
    lidar2cam[:3, 3] = translation.flatten()
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(lidar2cam, ground_points_lidar.T).T
    denorm = -1 * equation_plane(ground_points_cam)
    return denorm

def cam2velo(r_velo2cam, t_velo2cam):
    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = r_velo2cam
    Tr_velo2cam[:3 ,3] = t_velo2cam.flatten()
    Tr_cam2velo = np.linalg.inv(Tr_velo2cam)
    r_cam2velo = Tr_cam2velo[:3, :3]
    t_cam2velo = Tr_cam2velo[:3, 3]
    return r_cam2velo, t_cam2velo

def load_calib_kitti(calib_file):
    with open(calib_file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        P2, R0_rect, Tr_velo2cam = None, None, None
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                P2 = row[1:]
                P2 = [float(i) for i in P2]
                P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
                continue
            elif row[0] == 'R0_rect:':
                R0_rect = row[1:]
                R0_rect = [float(i) for i in R0_rect]
                R0_rect = np.array(R0_rect, dtype=np.float32).reshape(3, 3)
                continue
            elif row[0] == 'Tr_velo_to_cam:':
                Tr_velo2cam = row[1:]
                Tr_velo2cam = [float(i) for i in Tr_velo2cam]
                Tr_velo2cam = np.array(Tr_velo2cam, dtype=np.float32).reshape(3, 4)
                break
        if R0_rect is not None:
            Tr_velo2cam = np.matmul(R0_rect, Tr_velo2cam)

        r_velo2cam, t_velo2cam = Tr_velo2cam[:3, :3], Tr_velo2cam[:3, 3]
    return P2, r_velo2cam, t_velo2cam

def get_annos(label_path, r_cam2velo, t_cam2velo):
    Tr_cam2velo = np.eye(4)
    Tr_cam2velo[:3, :3], Tr_cam2velo[:3, 3] = r_cam2velo, t_cam2velo
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']
    annos = []
    with open(label_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            if row["type"] in name2nuscenceclass.keys():
                alpha = float(row["alpha"])
                pos = np.array((float(row['lx']), float(row['ly']), float(row['lz'])), dtype=np.float32)
                ry = float(row["ry"])
                if alpha > np.pi:
                    alpha -= 2 * np.pi
                    ry = alpha2roty(alpha, pos)
                alpha = clip2pi(alpha)
                ry = clip2pi(ry)
                rotation =  0.5 * np.pi - ry
                
                name = name2nuscenceclass[row["type"]]
                dim = [float(row['dl']), float(row['dw']), float(row['dh'])]
                box2d = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
                truncated_state = float(row["truncated"])
                occluded_state = float(row["occluded"])
                if sum(dim) == 0:
                    continue
                loc_cam = np.array([float(row['lx']), float(row['ly']), float(row['lz']), 1.0]).reshape(4, 1)
                loc_lidar = np.matmul(Tr_cam2velo, loc_cam).squeeze(-1)[:3]
                loc_lidar[2] += 0.5 * float(row['dh'])
                anno = {"dim": dim, "loc": loc_lidar, "rotation": rotation, "name": name, "box2d": box2d, "truncated_state": truncated_state, "occluded_state": occluded_state}
                annos.append(anno)
    return annos

def generate_info_kitti(kitti_root, split='train'):
    src_dir = os.path.join(kitti_root, "training") if split in ["train", "val"] else os.path.join(kitti_root, "training")
    label_path = os.path.join(src_dir, "label_2")
    calib_path = os.path.join(src_dir, "calib")
    split_txt = os.path.join(kitti_root, "ImageSets",  split + ".txt")
    idx_list = [x.strip() for x in open(split_txt).readlines()]
    infos = list()
    for idx in tqdm(range(len(idx_list))):
        index = idx_list[idx]
        sample_token = "training/" + index if split in ["train", "val"] else "training/" + index
        label_file = os.path.join(label_path, index + ".txt")
        calib_file = os.path.join(calib_path, index + ".txt")
        
        info = dict()
        cam_info = dict()
        info['sample_token'] = sample_token
        info['timestamp'] = int(index)
        info['scene_token'] = index
        
        cam_names = ['CAM_FRONT']
        lidar_names = ['LIDAR_TOP']
        cam_infos, lidar_infos = dict(), dict()
        for cam_name in cam_names:
            cam_info = dict()
            cam_info['sample_token'] = sample_token
            cam_info['timestamp'] = int(index)
            cam_info['is_key_frame'] = True
            cam_info['height'] = 384
            cam_info['width'] = 1280
            cam_info['filename'] = os.path.join("training", "image_2", index + ".png") if split in ["train", "val"] else os.path.join("training", "image_2", index + ".png")
            ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "token": index, "timestamp": int(index)}
            cam_info['ego_pose'] = ego_pose
            
            P2, r_velo2cam, t_velo2cam = load_calib_kitti(calib_file)
            r_cam2velo, t_cam2velo = cam2velo(r_velo2cam, t_velo2cam)
            Tr_cam2velo = np.eye(4)
            Tr_cam2velo[:3, :3], Tr_cam2velo[:3, 3] = r_cam2velo, t_cam2velo
            denorm = get_denorm(r_velo2cam, t_velo2cam)

            calibrated_sensor = {"token": index, "sensor_token": index, "translation": t_cam2velo.flatten(), "rotation_matrix": r_cam2velo, "camera_intrinsic": P2}
            cam_info['calibrated_sensor'] = calibrated_sensor
            cam_info['denorm'] = denorm
            cam_infos[cam_name] = cam_info
            
        for lidar_name in lidar_names:
            calibrated_sensor = {"token": index, "sensor_token": index, "translation": t_velo2cam.flatten(), "rotation_matrix": r_velo2cam}
            lidar_info = dict(
                filename=os.path.join("training", "velodyne", index + ".bin") if split in ["train", "val"] else os.path.join("training", "velodyne", index + ".bin"),
                calibrated_sensor=calibrated_sensor,
            )
            lidar_infos[lidar_name] = lidar_info

        info['cam_infos'] = cam_infos
        info['lidar_infos'] = lidar_infos
        info['sweeps'] = list()
        ann_infos = list()
        annos = get_annos(label_file, r_cam2velo, t_cam2velo) if split in ["train", "val"] else list()
        for anno in annos:
            category_name = anno["name"]
            translation = anno["loc"]
            size = anno["dim"]
            yaw_lidar = anno["rotation"]
            rot_mat = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                                [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                                [0, 0, 1]])    
            rotation = Quaternion(matrix=rot_mat)
            ann_info = dict()
            ann_info["category_name"] = category_name
            ann_info["translation"] = translation
            ann_info["rotation"] = rotation
            ann_info["size"] = size
            ann_info["prev"] = ""
            ann_info["next"] = ""
            ann_info["sample_token"] = sample_token
            ann_info["instance_token"] = index
            ann_info["token"] = index
            ann_info["visibility_token"] = str(anno["occluded_state"])
            ann_info["num_lidar_pts"] = 3
            ann_info["num_radar_pts"] = 0            
            ann_info['velocity'] = np.zeros(3)
            ann_infos.append(ann_info)
        info['ann_infos'] = ann_infos
        infos.append(info)
    return infos

def main():
    parser = argparse.ArgumentParser(description="Create Dataset Infos in KITTI format ...")
    parser.add_argument("--data_root", type=str,
                        default="../../data/kitti",
                        help="Path to Dataset root in KITTI format")
    args = parser.parse_args()

    kitti_root = args.data_root
    prefix = kitti_root.split('/')[1]
    train_infos = generate_info_kitti(kitti_root, split='train')
    val_infos = generate_info_kitti(kitti_root, split='val')
    test_infos = generate_info_kitti(kitti_root, split='test')
    
    with open(os.path.join(kitti_root, prefix + "_12hz_infos_train.pkl"), 'wb') as fid:        
        pickle.dump(train_infos, fid)
    with open(os.path.join(kitti_root, prefix + "_12hz_infos_val.pkl"), 'wb') as fid:        
        pickle.dump(val_infos, fid)
    with open(os.path.join(kitti_root, prefix + "_12hz_infos_test.pkl"), 'wb') as fid:        
        pickle.dump(test_infos, fid)

if __name__ == '__main__':
    main()
