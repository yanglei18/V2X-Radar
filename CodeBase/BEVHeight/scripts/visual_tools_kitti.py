import os
import math
import cv2
import csv
import argparse
import numpy as np

color_map = {"Car":(0, 255, 0), "Bus":(0, 255, 255), "Pedestrian":(255, 255, 0), "Cyclist":(0, 0, 255), "Van":(0, 255, 255), "Truck":(0, 255, 255)}

def parse_option():
    parser = argparse.ArgumentParser('Visualize InfrastructureSide dataset', add_help=False)
    parser.add_argument('--kitti_root', type=str, default='data/kitti', help='root path to KITTI Dataset')
    parser.add_argument('--pred_path', type=str, default=None, help='path to predictions')
    parser.add_argument('--demo_path', type=str, default='kitt_demo', help='')    
    args = parser.parse_args()
    return args

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

def get_denorm(Tr_velo_to_cam):
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(Tr_velo_to_cam, ground_points_lidar.T).T
    denorm = -1 * equation_plane(ground_points_cam)
    return denorm

def read_bin(path):
    points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])
    return points[:, :3] 

def lidar2camera_projection(image, points, sensor_params):
    rmat, tvec, K, dist = sensor_params["rmat"], sensor_params["tvec"], sensor_params["K"], sensor_params["dist"]
    
    image = cv2.undistort(image, K, dist)
    image_points = lidar_to_image(points, rmat, tvec, K)
    coor, depth = image_points[:, :2], image_points[:, 2]
    height, width, _ = image.shape
    kept = (image_points[:, 0] >= 1) & (image_points[:, 0] < width-1) & (
            image_points[:, 1] >= 1) & (image_points[:, 1] < height-1) & (depth > 1) & (depth < 150)
    coor, depth = coor[kept], depth[kept]
    color_map = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    for id in range(coor.shape[0]):
        dis = (depth[id] - 1) / 100 * 256
        dis = min(int(dis), 255)
        color = tuple(color_map[dis, 0].astype(np.uint8))
        image = cv2.circle(image, (coor[id][0], coor[id][1]), 2, (int(color[0]), int(color[1]), int(color[2])), -1)
    return image

def get_bev_image(points, c=(255, 0, 0), r=1, range_list = [(-80.6, 80.6), (-64.8, 64.8), (-2.0, -2.0), 0.12]):
    points_filter = PointCloudFilter(side_range=range_list[0], fwd_range=range_list[1], res=range_list[-1])
    bev_image = points_filter.get_bev_image(points, c, r)
    return bev_image

def lidar_to_image(points_cloud, rmat, tvec, camera_matrix):
    points_cloud = points_cloud.copy()
    points_cloud = np.concatenate((points_cloud, np.ones((points_cloud.shape[0], 1))), axis=1)
    points_cloud = points_cloud.T
    extrinsic_matrix = np.concatenate([rmat, tvec.reshape(3, 1)], axis=1)
    image_points = np.dot(np.dot(camera_matrix, extrinsic_matrix), points_cloud).T.reshape([-1, 3])
    image_points[:, 0] = image_points[:, 0] / image_points[:, 2]
    image_points[:, 1] = image_points[:, 1] / image_points[:, 2]
    image_points = image_points[:, :3].astype(np.int32)
    return image_points

def read_radar_bin(path):
    points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 5])
    return points[:, :3] 

def compute_box_3d_camera(dim, location, rotation_y, denorm):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    
    denorm = denorm[:3]
    denorm_norm = denorm / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)
    ori_denorm = np.array([0.0, -1.0, 0.0])
    theta = -1 * math.acos(np.dot(denorm_norm, ori_denorm))
    n_vector = np.cross(denorm, ori_denorm)
    n_vector_norm = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    rotation_matrix, j = cv2.Rodrigues(theta * n_vector_norm)
    corners_3d = np.dot(rotation_matrix, corners_3d)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    
    return corners_3d.transpose(1, 0)

def project_to_image(pts_3d, P):
    pts_3d_homo = np.concatenate(
        [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d

def draw_box_3d(image, corners, c=(0, 255, 0)):
    face_idx = [[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    for ind_f in [3, 2, 1, 0]:
        f = face_idx[ind_f]
        for j in [0, 1, 2, 3]:
            cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
                (int(corners[f[(j+1)%4], 0]), int(corners[f[(j+1)%4], 1])), c, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                (int(corners[f[2], 0]), int(corners[f[2], 1])), c, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                (int(corners[f[3], 0]), int(corners[f[3], 1])), c, 1, lineType=cv2.LINE_AA)
    return image

def bbox2image_projection(image, annos, P2, denorm, c=(0,255,0)):
    for anno in annos:
        category, truncated, occulated = anno["class"], anno["truncated"], anno["occluded"]
        loc, dim, rot_y = anno["loc"], anno["dim"], anno["rot_y"]
        box_3d = compute_box_3d_camera(dim, loc, rot_y, denorm)
        box_2d = project_to_image(box_3d, P2)
        # if category not in color_map.keys(): continue
        image = draw_box_3d(image, box_2d, c=c)
    return image

def bbox2lidar_projection(bev_image, annos, sensor_params, denorm, c=(0, 255, 0)):
    rmat, tvec = sensor_params["rmat"], sensor_params["tvec"]
    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = rmat
    Tr_velo2cam[:3, 3] = tvec
    Tr_cam2velo = np.linalg.inv(Tr_velo2cam)
    
    range_list = [(-80.6, 80.6), (-64.8, 64.8), (-2.0, -2.0), 0.12]
    points_filter = PointCloudFilter(side_range=range_list[0], fwd_range=range_list[1], res=range_list[-1])
    for anno in annos:
        category, truncated, occulated = anno["class"], anno["truncated"], anno["occluded"]
        loc, dim, rot_y = anno["loc"], anno["dim"], anno["rot_y"]
        box_3d_camera = compute_box_3d_camera(dim, loc, rot_y, denorm)
        box_3d_camera_extend = np.concatenate((box_3d_camera, np.ones((box_3d_camera.shape[0], 1))), axis=1)
        box_3d_camera_lidar = np.matmul(Tr_cam2velo, box_3d_camera_extend.T).T[:, :3]
        box_3d = box_3d_camera_lidar
        x_img, y_img = points_filter.pcl2xy_plane(box_3d[:, 0], box_3d[:, 1])
        for i in np.arange(4):
            cv2.line(bev_image, (x_img[0], y_img[0]), (x_img[1], y_img[1]), c, 3)
            cv2.line(bev_image, (x_img[0], y_img[0]), (x_img[3], y_img[3]), c, 3)
            cv2.line(bev_image, (x_img[1], y_img[1]), (x_img[2], y_img[2]), c, 3)
            cv2.line(bev_image, (x_img[2], y_img[2]), (x_img[3], y_img[3]), c, 3)
    return bev_image

class PointCloudFilter(object):
    def __init__(self,
                 side_range=(-39.68, 39.68),
                 fwd_range=(0, 69.12),
                 height_range=(-2., -2.),
                 res=0.10
                ):
        self.res = res
        self.side_range = side_range
        self.fwd_range = fwd_range
        self.height_range = height_range

    def set_range_patameters(self, side_range, fwd_range, height_range):
        self.side_range = side_range
        self.fwd_range = fwd_range
        self.height_range = height_range

    def read_bin(self, path):
        """
        Helper function to read one frame of lidar pointcloud in .bin format.
        :param path: where pointcloud is stored in .bin format.
        :return: (x, y, z, intensity) of pointcloud, N x 4.
        """
        points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])
        x_points, y_points, z_points, indices = self.get_pcl_range(points)
        filtered_points = np.concatenate((x_points[:,np.newaxis], y_points[:,np.newaxis], z_points[:,np.newaxis]), axis = 1)
        return filtered_points

    def scale_to_255(self, value, minimum, maximum, dtype=np.uint8):
        """
        Scales an array of values from specified min, max range to 0-255.
        Optionally specify the data type of the output (default is uint8).
        """
        if minimum!= maximum:
            return (((value - minimum) / float(maximum - minimum))
                    * 255).astype(dtype)
        else:
            return self.get_meshgrid()

    def get_pcl_range(self, points):
        """
        Get the pointcloud wihtin side_range and fwd_range.
        :param points: np.float, N x 4. each column is [x, y, z, intensity].
        :return: [x, y, z, intensity] of filtered points and corresponding
                 indices.
        """
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]
        indices = []
        for i in range(points.shape[0]):
            if points[i, 0] > self.fwd_range[0] and points[i, 0] < self.fwd_range[1]:
                if points[i, 1]  > self.side_range[0] and points[i, 1] < self.side_range[1]:
                    indices.append(i)

        indices = np.array(indices)
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        x_points = x_points[indices]
        y_points = y_points[indices]
        z_points = z_points[indices]
        return x_points, y_points, z_points, indices

    def clip_height(self, z_points):
        """
        Clip the height between (min, max).
        :param z_points: z_points from get_pcl_range
        :return: clipped height between (min,max).
        """
        height = np.clip(
            a=z_points, a_max=self.height_range[1], a_min=self.height_range[0]
        )
        return height

    def get_meshgrid(self):
        """
        Create mesh grids (size: res x res) in the x-y plane of the lidar
        :return: np.array: uint8, x-y plane mesh grids based on resolution.
        """
        x_max = 1 + int((self.side_range[1] - self.side_range[0]) / self.res)
        y_max = 1 + int((self.fwd_range[1] - self.fwd_range[0]) / self.res)
        img = np.ones([y_max, x_max], dtype=np.uint8) * 255
        return img

    def pcl2xy_plane(self, x_points, y_points):
        """
        Convert the lidar coordinate to x-y plane coordinate.
        :param x_points: x of points in lidar coordinate.
        :param y_points: y of points in lidar coordinate.
        :return: corresponding pixel position based on resolution.
        """
        x_img = (-y_points / self.res).astype(np.int32) # x axis is -y in lidar
        y_img = (-x_points / self.res).astype(np.int32) # y axis is -x in lidar
        # shift pixels to have minimum be (0,0)
        x_img -= int(np.floor(self.side_range[0] / self.res))
        y_img += int(np.ceil(self.fwd_range[1] / self.res))
        return x_img, y_img

    def pcl_2_bev(self, points, c, r=1):
        """
        Creates an 2D birds eye view representation of the pointcloud.
        :param points: np.float, N x 4. input pointcloud matrix,
                       each column is [x, y, z, intensity]
        :return: np.array, representing an image of the BEV.
        """
        # rescale the height values - to be between the range 0-255
        x_points, y_points, z_points, _ = self.get_pcl_range(points)
        x_img, y_img = self.pcl2xy_plane(x_points, y_points)
        bev_img = self.get_meshgrid()
        
        bev_img[y_img, x_img] = 255
        bev_img = cv2.merge([bev_img, bev_img, bev_img])
        
        if r > 1:
            for i in range(y_img.shape[0]):
                bev_img = cv2.circle(bev_img, (x_img[i], y_img[i]), r, c, -1)
        else:
            bev_img[y_img, x_img] = c
        return bev_img

    def get_bev_image(self, points_cloud, c, r=1):
        bev_image = self.pcl_2_bev(points_cloud, c, r)
        return bev_image

class KITTIDataset:
    def __init__(self, kitti_root, pred_path=None):
        super(KITTIDataset, self).__init__()
        self.kitti_root = kitti_root
        self.image_dir = os.path.join(kitti_root, "training", "image_2")
        self.image3_dir = os.path.join(kitti_root, "training", "image_3")
        self.calib_dir = os.path.join(kitti_root, "training", "calib")
        self.lidar_dir = os.path.join(kitti_root, "training", "velodyne")
        self.radar_dir = os.path.join(kitti_root, "training", "radar")
        self.label_dir = os.path.join(kitti_root, "training", "label_2")
        self.pred_path = pred_path
        
        image_files = [] 
        for label_name in os.listdir(self.label_dir) if self.pred_path is None else os.listdir(self.pred_path):
            base_name = label_name.split('.')[0]
            image_files.append(base_name + ".jpg")
            
        self.image_files = image_files
        self.image_files = image_files
        self.lidar_files = [i.replace(".jpg", ".bin") for i in self.image_files]
        self.radar_files = [i.replace(".jpg", ".bin") for i in self.image_files]
        
        self.label_files = [i.replace(".jpg", ".txt") for i in self.image_files]
        self.num_samples = len(self.image_files)

    def __len__(self):
        return self.num_samples
    
    def load_calib(self, idx):
        file_name = self.label_files[idx]
        with open(os.path.join(self.calib_dir, file_name), 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            P2, Tr_velo_to_cam, Tr_radar_to_velo = None, None, None
            for line, row in enumerate(reader):
                if row[0] == 'P2:':
                    P2 = row[1:]
                    P2 = [float(i) for i in P2]
                    P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
                    continue
                elif row[0] == 'Tr_velo_to_cam:':
                    Tr_velo_to_cam = row[1:]
                    Tr_velo_to_cam = [float(i) for i in Tr_velo_to_cam]
                    Tr_velo_to_cam = np.array(Tr_velo_to_cam, dtype=np.float32).reshape(3, 4)
                    continue
                elif row[0] == 'Tr_radar_to_velo:':
                    Tr_radar_to_velo = row[1:]
                    Tr_radar_to_velo = [float(i) for i in Tr_radar_to_velo]
                    Tr_radar_to_velo = np.array(Tr_radar_to_velo, dtype=np.float32).reshape(3, 4)
                    continue
        return P2, Tr_velo_to_cam, Tr_radar_to_velo     
    
    def load_annotations(self, idx, pred=False):
        annotations = []
        file_name = self.label_files[idx]
        file_name = os.path.join(self.label_dir, file_name) if not pred else os.path.join(self.pred_path, file_name)
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']
        with open(file_name, 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
            for line, row in enumerate(reader):
                annotations.append({
                    "class": row["type"],
                    "label": row["type"],
                    "truncated": float(row["truncated"]),
                    "occluded": float(row["occluded"]),
                    "alpha": float(row["alpha"]),
                    "dim": [float(row['dh']), float(row['dw']), float(row['dl'])],
                    "loc": [float(row['lx']), float(row['ly']), float(row['lz'])],
                    "rot_y": float(row["ry"])
                })
        return annotations
    
    def __getitem__(self, idx):
        # load default parameter here
        original_idx = self.label_files[idx].replace(".txt", "")
                
        P2, Tr_velo_to_cam, Tr_radar_to_velo = self.load_calib(idx)
        image = cv2.imread(os.path.join(self.image_dir, self.image_files[idx]))
        image = cv2.resize(image, (1536, 864))
        points_rslidar = read_bin(os.path.join(self.lidar_dir, self.lidar_files[idx]))
        radar_file = os.path.join(self.radar_dir, self.radar_files[idx])
        points_radar = read_radar_bin(radar_file) if os.path.exists(radar_file) else None
        
        P2[:3, :3] = P2[:3, :3] * 0.8
        P2[2, 2] = 1.0
        sensor_params = {
            "rmat": Tr_velo_to_cam[:3, :3],
            "tvec": Tr_velo_to_cam[:3, 3],
            "K": P2[:3, :3],
            "dist": np.array([0.0, 0.0,	0.0, 0.0, 0.0]),
        }
        denorm = get_denorm(Tr_velo_to_cam)
        image = lidar2camera_projection(image, points_rslidar, sensor_params)
        
        annos = self.load_annotations(idx)
        image = bbox2image_projection(image, annos, P2, denorm, c=(0,0,255))
        bev_image = get_bev_image(points_rslidar, c=(255, 0, 0), r=2)
        if points_radar is not None:
            points_radar_extend = np.concatenate((points_radar, np.ones((points_radar.shape[0], 1))), axis=1)
            radar2lidar = np.eye(4)
            radar2lidar[:3, :4] = Tr_radar_to_velo
            points_radar_rslidar = np.matmul(radar2lidar, points_radar_extend.T).T[:, :3]
            bev_image_radar = get_bev_image(points_radar_rslidar, c=(0, 0, 255), r=3)
            bev_image = cv2.addWeighted(bev_image_radar, 0.5, bev_image, 0.5, 0)
        bev_image = bbox2lidar_projection(bev_image, annos, sensor_params, denorm, c=(0,0,255))

        if self.pred_path is not None:
            annos_pred = self.load_annotations(idx, pred=True)
            image = bbox2image_projection(image, annos_pred, P2, denorm)
            bev_image = bbox2lidar_projection(bev_image, annos_pred, sensor_params, denorm)
        
        return image, bev_image, original_idx

if __name__ == "__main__":
    args = parse_option()
    dataset = KITTIDataset(args.kitti_root, args.pred_path)
    os.makedirs(args.demo_path, exist_ok=True)
    
    for i in range(len(dataset)):
        image, bev_image, original_idx = dataset[i]
        cv2.imwrite(os.path.join(args.demo_path, original_idx + ".jpg"), image)
        # cv2.imwrite(os.path.join(args.demo_path, original_idx + "_bev.jpg"), bev_image)
        