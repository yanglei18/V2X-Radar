import os
import cv2
import shutil
import copy
import numpy as np

from tqdm import tqdm
from pypcd import pypcd

ARBE_PCD_BINARY_TEMPLATE = """VERSION 0.7
FIELDS x y z intensity dopple
SIZE 4 4 4 4 4 
TYPE F F F F F
COUNT 1 1 1 1 1
WIDTH {}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {}
DATA binary
"""


def to_arbe_binary(pcdpath, points):
    f = open(pcdpath, 'wb')
    header = copy.deepcopy(ARBE_PCD_BINARY_TEMPLATE).format(points.shape[0], points.shape[0])
    f.write(header.encode())
    import struct
    for id in range(points.shape[0]):
        h = struct.pack('<fffff', points[id,0], points[id,1], points[id,2], points[id,3], points[id,4])
        f.write(h)
    f.close()
    

def read_bin(path):
    points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 5])
    return points[:, :5] 

def read_pcd(pcd_file_path):
    pc = pypcd.PointCloud.from_path(pcd_file_path)
    # Get data from pcd (x, y, z, intensity, dopple)
    np_x = np.array(pc.pc_data['x'], dtype=np.float32)
    np_y = np.array(pc.pc_data['y'], dtype=np.float32)
    np_z = np.array(pc.pc_data['z'], dtype=np.float32)
    np_i = np.array(pc.pc_data['intensity'], dtype=np.float32) / 256
    dopple = np.array(pc.pc_data['dopple'], dtype=np.float32) / 256
    # Stack all data
    points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i, dopple)))
    return points_32

def transform_points(points, radar2velo):
    i_dop_infos = points[:, 3:]
    points = points[:, :3]
    Tr_radar2velo = radar2velo
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    points = np.dot(points, Tr_radar2velo.T)
    points = np.concatenate((points[:,:3], i_dop_infos), axis=1)
    return points


if __name__ == "__main__":
    src_root = "../../data/v2x-radar/mini/v2x-radar-c"
    dest_root = "../../data/v2x-radar/mini/v2x-radar-c-radar_velo"
    print("hello world ...")
    
    for split in ["train", "validate"]:
        for seq in os.listdir(os.path.join(src_root, split)):     
            for agent in ["-1", "142"]:
                src_path = os.path.join(src_root, split, seq, agent)
                dest_path = os.path.join(dest_root, split, seq, agent)
                os.makedirs(dest_path, exist_ok=True)
                for filename in tqdm(os.listdir(src_path)):
                    if "camera0" not in filename: continue
                    frame_id = int(filename.split('_')[0])
                    shutil.copy2(os.path.join(src_path, "{:05d}_camera0.png".format(frame_id)), os.path.join(dest_path, "{:05d}_camera0.png".format(frame_id)))
                    shutil.copy2(os.path.join(src_path, "{:05d}_camera1.png".format(frame_id)), os.path.join(dest_path, "{:05d}_camera1.png".format(frame_id)))
                    shutil.copy2(os.path.join(src_path, "{:05d}_camera2.png".format(frame_id)), os.path.join(dest_path, "{:05d}_camera2.png".format(frame_id)))
                    shutil.copy2(os.path.join(src_path, "{:05d}.yaml".format(frame_id)), os.path.join(dest_path, "{:05d}.yaml".format(frame_id)))
                    
                    radarfile = os.path.join(src_path, "{:05d}_radar.pcd".format(frame_id))
                    save_radarfile = os.path.join(dest_path, "{:05d}.pcd".format(frame_id))
                    
                    if agent == "-1":
                        Tr_radar2velo = np.array([[0.984074,-0.0154492,0.177085,0.0437389],
                                                  [0.0157072,0.999877,-5.48299e-05,-0.000294566],
                                                  [-0.177062,0.00283546,0.984196,-0.432688],
                                                  [0,0,0,1]])
                    else:
                        Tr_radar2velo = np.array([[-0.013970759076831379, 0.9985046529494882, 0.05285145153199537, 2.2198703343289825],
                                                  [-0.9998950756585999, -0.014153495799834579, 0.0030848387844398294, 0.02787126097813525],
                                                  [0.003828258677135485, -0.05280280858880308, 0.9985976205863071, -1.2864166043457155],
                                                  [0,0,0,1]])
                    points = read_pcd(radarfile)
                    points = transform_points(points, Tr_radar2velo)
                    
                    to_arbe_binary(save_radarfile, points)
                    