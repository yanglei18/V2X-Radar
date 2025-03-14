import os
import csv
import numpy as np

from tqdm import tqdm
    
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
                continue
            elif row[0] == 'Tr_radar_to_velo:':
                Tr_radar2velo = row[1:]
                Tr_radar2velo = [float(i) for i in Tr_radar2velo]
                Tr_radar2velo = np.array(Tr_radar2velo, dtype=np.float32).reshape(3, 4)
                continue
    return Tr_radar2velo, R0_rect


def read_bin(path):
    points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)
    return points

def transform_points(points, radar2velo):
    i_dop_infos = points[:, 3:]
    points = points[:, :3]
    Tr_radar2velo = np.eye(4).astype(np.float32)
    Tr_radar2velo[:3, :] = radar2velo
    points = np.concatenate((points, np.ones((points.shape[0], 1), dtype=np.float32)), axis=1)

    points = np.matmul(Tr_radar2velo, points.T).T
    points = np.concatenate((points[:,:3], i_dop_infos), axis=1)
    return points

if __name__ == "__main__":
    print("hello world ...")
    root_path = "data/v2x-radar-v-kitti"
    radar_folder = os.path.join(root_path, "training", "radar")
    calib_folder = os.path.join(root_path, "training", "calib")
    radar_velo_folder = os.path.join(root_path, "training", "radar_velo")
    
    os.makedirs(radar_velo_folder, exist_ok=True)
    for filename in tqdm(os.listdir(radar_folder)):
        filename = filename.split(".")[0]
        
        radar_file = os.path.join(radar_folder, filename + ".bin")
        radar_velo_file = os.path.join(radar_velo_folder, filename + ".bin")
        calib_file = os.path.join(calib_folder, filename + ".txt")
        
        Tr_radar2velo, R0_rect = load_calib_kitti(calib_file)
        print("Tr_radar2velo: ", Tr_radar2velo.dtype)
        points = read_bin(radar_file)
        print("01: ", points.shape)

        points = transform_points(points, Tr_radar2velo)
        print("02: ", points.shape)

        points.tofile(radar_velo_file)

        print(os.path.getsize(radar_velo_file))

