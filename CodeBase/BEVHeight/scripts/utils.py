import copy
import csv
import os
import math

import numpy as np

def lidar_to_rect(pts_lidar, Rlidar_camera):
    """
    :param pts_lidar: (N, 3)
    :return pts_rect: (N, 3)
    """
    pts_lidar_hom = np.hstack((pts_lidar, np.ones((pts_lidar.shape[0], 1), dtype=np.float32)))
    pts_rect = np.dot(pts_lidar_hom, Rlidar_camera.T)
    # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
    return pts_rect


#box3d_lidar_to_camera
def boxes3d_lidar_to_camera(boxes3d_lidar, Rlidar_camera):
    """
    :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param calib:
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    """
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    xyz_lidar = boxes3d_lidar_copy[:, 0:3]
    l, w, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = lidar_to_rect(xyz_lidar, Rlidar_camera)
    r = -r - np.pi/2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)

#boxes3d_to_corners3d_camera
def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)

#boxes3d_camera_to_imageboxes
def boxes3d_kitti_camera_to_imageboxes(boxes3d, C_matric, image_shape=None):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d).reshape(-1, 3)
    corners3d_hom = np.hstack((corners3d, np.ones((corners3d.shape[0], 1), dtype=np.float32)))
    corners3d_2d_hom = np.dot(corners3d_hom , C_matric.T)
    pts_img = (corners3d_2d_hom[:, 0:2].T / corners3d_2d_hom[:, 2]).T  # (N, 2)
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
    max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d_image

def compute_box_3d(dim, yaw, loc):
    l, w, h = dim
    liadr_r = np.matrix(
        [[math.cos(yaw), -math.sin(yaw), 0], 
         [math.sin(yaw), math.cos(yaw), 0], 
         [0, 0, 1]]
    )
    corners_3d = np.matrix([
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2],
        ])
    corners_3d = liadr_r * corners_3d + np.matrix(loc).T
    return corners_3d.T

def compute_box_3d_camera(dim, yaw, loc, extrinsic_matrix):
    corners_3d = compute_box_3d(dim, yaw, loc)
    corners_3d = np.concatenate((corners_3d, np.ones((corners_3d.shape[0], 1))), axis=1)
    corners_3d = corners_3d.T
    corners_3d_camera = np.dot(extrinsic_matrix, corners_3d).T.reshape([-1, 3])
    return np.array(corners_3d_camera)

def compute_box_3d_image(dim, yaw, loc, extrinsic_matrix, camera_matrix):
    corners_3d = compute_box_3d(dim, yaw, loc)
    corners_3d = np.concatenate((corners_3d, np.ones((corners_3d.shape[0], 1))), axis=1)
    corners_3d = corners_3d.T
    corners_image = np.dot(np.dot(camera_matrix, extrinsic_matrix), corners_3d).T.reshape([-1, 3])
    corners_image[:, 0] = corners_image[:, 0] / corners_image[:, 2]
    corners_image[:, 1] = corners_image[:, 1] / corners_image[:, 2]
    corners_image = corners_image[:, :3].astype(np.int32)
    return corners_image

def compute_box_3d_from_camera(dim, location, rotation_y):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)

def bbox_cam2lidar(dim, loc_cam, rot_y, Tr_cam2velo):
    l, w, h = dim
    corners_cam = compute_box_3d_from_camera(dim, loc_cam, rot_y)                
    corners_cam = np.concatenate((corners_cam, np.ones((corners_cam.shape[0], 1))), axis=1)
    corners_cam = corners_cam.T
    corners_lidar = np.dot(Tr_cam2velo, corners_cam).T.reshape([-1, 4])
    x0, y0 = corners_lidar[0, 0], corners_lidar[0, 1]
    x3, y3 = corners_lidar[3, 0], corners_lidar[3, 1]
    dx, dy = x0 - x3, y0 - y3
    yaw = math.atan2(dy, dx)
    loc_cam = np.array([loc_cam[0], loc_cam[1], loc_cam[2], 1.0]).reshape(4, 1)
    loc_lidar = np.dot(Tr_cam2velo, loc_cam).T.reshape([4,])[:3]
    return loc_lidar, yaw, dim

def bbox_lidar2cam(dim, loc, yaw, Tr_velo2cam):
    l, w, h = dim
    loc_lidar = np.array([loc[0], loc[1], loc[2], 1.0]).reshape(4, 1)
    loc_cam = np.dot(Tr_velo2cam, loc_lidar).T.reshape([4,])
   
    boxes_camera = compute_box_3d_camera(dim, yaw, loc, Tr_velo2cam[:3, :])
    x0, z0 = boxes_camera[0, 0], boxes_camera[0, 2]
    x3, z3 = boxes_camera[3, 0], boxes_camera[3, 2]
    dx, dz = x0 - x3, z0 - z3
    rot_y = math.atan2(-dz, dx)
    alpha = rot_y - math.atan2(loc_cam[0], loc_cam[2])
    return alpha, rot_y, dim, loc_cam

def convert_annotation(file_in, file_out, Rlidar_camera, C_matric, image_shape):
    in_file = open(file_in, 'r')
    out_file = open(file_out, 'w')
    lines = in_file.readlines()
    for line in lines:
        line = line.strip().split(' ')
        cls_type = line[0]
        truncation = int(line[1])
        occlusion = int(line[2])
        l = float(line[3])
        w = float(line[4])
        h = float(line[5])
        x = float(line[6])
        y = float(line[7])
        z = float(line[8])
        rotation_y = float(line[9])
        tracker_id = int(line[10])
        dim, loc, yaw = [l, w, h], [x, y, z], rotation_y
        Tr_velo2cam = np.eye(4)
        Tr_velo2cam[:3, :] = Rlidar_camera
        alpha, rot_y, dim, loc_cam = bbox_lidar2cam(dim, loc, yaw, Tr_velo2cam)        
            
        corners_image = compute_box_3d_image(dim, yaw, loc, Rlidar_camera, C_matric[:3,:3])
        min_value, max_value = np.amin(corners_image, axis=0), np.amax(corners_image, axis=0)
        xmin, ymin, xmax, ymax = min_value[0,0], min_value[0,1], max_value[0,0], max_value[0,1]
        boxes2d_image = [xmin, ymin, xmax, ymax]
        if image_shape is not None:
            boxes2d_image[0] = np.clip(boxes2d_image[0], a_min=0, a_max=image_shape[0] - 1)
            boxes2d_image[1] = np.clip(boxes2d_image[1], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[2] = np.clip(boxes2d_image[2], a_min=0, a_max=image_shape[0] - 1)
            boxes2d_image[3] = np.clip(boxes2d_image[3], a_min=0, a_max=image_shape[1] - 1)
        # print(min_value, max_value, xmin, ymin, xmax, ymax, boxes2d_image)
        if boxes2d_image[0]==boxes2d_image[2] or boxes2d_image[1]==boxes2d_image[3]:
            # print(min_value, max_value, boxes2d_image, file_in)
            boxes2d_image = [0,0,200,200]
        
        out_file.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(cls_type.capitalize(), truncation, occlusion, alpha, boxes2d_image[0] , boxes2d_image[1],
                                                                boxes2d_image[2], boxes2d_image[3], h, w, l, loc_cam[0], loc_cam[1], loc_cam[2], rot_y, tracker_id) + '\n')

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

def cam2velo(r_velo2cam, t_velo2cam):
    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = r_velo2cam
    Tr_velo2cam[:3 ,3] = t_velo2cam.flatten()
    Tr_cam2velo = np.linalg.inv(Tr_velo2cam)
    r_cam2velo = Tr_cam2velo[:3, :3]
    t_cam2velo = Tr_cam2velo[:3, 3]
    return r_cam2velo, t_cam2velo

class KITTIDataset:
    def __init__(self, kitti_root, camera_view="CAM_FRONT", split="val"):
        super(KITTIDataset, self).__init__()
        self.kitti_root = kitti_root
        self.split = split
        self.camera_view = camera_view
        self.image_1_dir = os.path.join(kitti_root, "training", "image_1")
        self.image_2_dir = os.path.join(kitti_root, "training", "image_2")
        self.image_3_dir = os.path.join(kitti_root, "training", "image_3")
        self.label_dir = os.path.join(kitti_root, "training", "label_2")
        
        self.calib_dir = os.path.join(kitti_root, "training", "calib")
        self.lidar_dir = os.path.join(kitti_root, "training", "velodyne")
        self.radar_dir = os.path.join(kitti_root, "training", "radar")
        
        image_files = [] 
        for label_name in os.listdir(self.label_dir):
            base_name = label_name.split('.')[0]
            image_files.append(base_name + ".jpg")
            
        self.image_files = image_files
        self.lidar_files = [i.replace(".jpg", ".bin") for i in self.image_files]
        self.radar_files = [i.replace(".jpg", ".bin") for i in self.image_files]
        
        self.label_files = [i.replace(".jpg", ".txt") for i in self.image_files]
        self.num_samples = len(self.image_files)

    def __len__(self):
        return self.num_samples
    
    def load_calib_kitti(self, idx):
        calib_file = os.path.join(self.calib_dir, self.label_files[idx])
        with open(calib_file, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            P2, R0_rect, Tr_velo2cam = None, None, None
            for line, row in enumerate(reader):
                if row[0] == 'P1:':
                    P1 = row[1:]
                    P1 = [float(i) for i in P1]
                    P1 = np.array(P1, dtype=np.float32).reshape(3, 4)
                    continue
                if row[0] == 'P2:':
                    P2 = row[1:]
                    P2 = [float(i) for i in P2]
                    P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
                    continue
                if row[0] == 'P3:':
                    P3 = row[1:]
                    P3 = [float(i) for i in P3]
                    P3 = np.array(P3, dtype=np.float32).reshape(3, 4)
                    continue
                elif row[0] == 'R0_rect:':
                    R0_rect = row[1:]
                    R0_rect = [float(i) for i in R0_rect]
                    R0_rect = np.array(R0_rect, dtype=np.float32).reshape(3, 3)
                    continue
                elif row[0] == 'Tr_velo_to_cam1:':
                    Tr_velo2cam1 = row[1:]
                    Tr_velo2cam1 = [float(i) for i in Tr_velo2cam1]
                    Tr_velo2cam1 = np.array(Tr_velo2cam1, dtype=np.float32).reshape(3, 4)
                    continue
                elif row[0] == 'Tr_velo_to_cam:':
                    Tr_velo2cam2 = row[1:]
                    Tr_velo2cam2 = [float(i) for i in Tr_velo2cam2]
                    Tr_velo2cam2 = np.array(Tr_velo2cam2, dtype=np.float32).reshape(3, 4)
                    continue
                elif row[0] == 'Tr_velo_to_cam3:':
                    Tr_velo2cam3 = row[1:]
                    Tr_velo2cam3 = [float(i) for i in Tr_velo2cam3]
                    Tr_velo2cam3 = np.array(Tr_velo2cam3, dtype=np.float32).reshape(3, 4)
                    continue
            if R0_rect is not None:
                Tr_velo2cam1 = np.matmul(R0_rect, Tr_velo2cam1)
                Tr_velo2cam2 = np.matmul(R0_rect, Tr_velo2cam2)
                Tr_velo2cam3 = np.matmul(R0_rect, Tr_velo2cam3)
            sensor_params = {
                "CAM_LEFT": (P1, Tr_velo2cam1[:3, :3], Tr_velo2cam1[:3, 3]),
                "CAM_FRONT": (P2, Tr_velo2cam2[:3, :3], Tr_velo2cam2[:3, 3]), 
                "CAM_RIGHT": (P3, Tr_velo2cam3[:3, :3], Tr_velo2cam3[:3, 3])
            }
        return sensor_params
    
    def load_annotations(self, idx, label_dir):
        file_name = self.label_files[idx]
        label_path = os.path.join(label_dir, file_name)
        calib_FRONT = self.load_calib_kitti(idx)["CAM_FRONT"]
        calib_LEFT = self.load_calib_kitti(idx)["CAM_LEFT"]
        calib_RIGHT = self.load_calib_kitti(idx)["CAM_RIGHT"]

        P, r_velo2cam, t_velo2cam = calib_FRONT
        r_cam2velo, t_cam2velo = cam2velo(r_velo2cam, t_velo2cam)
        Tr_cam2velo = np.eye(4)
        Tr_cam2velo[:3, :3], Tr_cam2velo[:3, 3] = r_cam2velo, t_cam2velo
        
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw', 'dl', 'lx', 'ly', 'lz', 'ry']
        annos = []
        with open(label_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
            for line, row in enumerate(reader):
                alpha = float(row["alpha"])
                loc = np.array((float(row['lx']), float(row['ly']), float(row['lz'])),dtype=np.float32)
                dim = np.array((float(row['dl']), float(row['dw']), float(row['dh'])), dtype=np.float32)
                ry = float(row["ry"])
                loc_lidar, yaw, dim = bbox_cam2lidar(dim, loc, ry, Tr_cam2velo)
                truncated_state = float(row["truncated"])
                occluded_state = float(row["occluded"])
                
                anno = {"row": row,
                        "dim": dim,
                        "loc": loc_lidar,  # bottom point
                        "yaw": yaw, 
                        "class": row["type"], 
                        "label": row["type"],
                        "alpha": alpha, 
                        "truncated": truncated_state, 
                        "occluded": occluded_state,
                        "CAM_FRONT": calib_FRONT,
                        "CAM_LEFT": calib_LEFT,
                        "CAM_RIGHT": calib_RIGHT}
                annos.append(anno)
        return annos
    
    def __getitem__(self, idx):
        # load default parameter here
        original_idx = self.label_files[idx].replace(".txt", "")
        annos = self.load_annotations(idx, self.label_dir)
        sensor_params = self.load_calib_kitti(idx)
        image_1_path = os.path.join(self.image_1_dir, self.image_files[idx]) 
        image_2_path = os.path.join(self.image_2_dir, self.image_files[idx]) 
        image_3_path = os.path.join(self.image_3_dir, self.image_files[idx]) 
        radar_path = os.path.join(self.radar_dir, self.radar_files[idx]) 
        lidar_path = os.path.join(self.lidar_dir, self.lidar_files[idx])
        
        data_path = {
            "original_idx": original_idx,
            "image_1_path": image_1_path,
            "image_2_path": image_2_path,
            "image_3_path": image_3_path,
            "radar_path": radar_path,
            "lidar_path": lidar_path
        } 
        return annos, sensor_params, data_path
    