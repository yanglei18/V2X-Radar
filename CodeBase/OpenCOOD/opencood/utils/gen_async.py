import os
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    src_root = "data/v2x-radar/v2x-radar-c"
    dst_root = "data/v2x-radar/v2x-radar-c-async"
    os.makedirs(dst_root, exist_ok=True)
    print("begin to generate async data...")
    
    for split in ["train", "validate"]:
        for seq in os.listdir(os.path.join(src_root, split)):
            # first copy the vehicle side data     
            shutil.copytree(os.path.join(src_root, split, seq, "142"), os.path.join(dst_root, split, seq, "142"))
            # then copy the road side data, async data
            road_side_src_path = os.path.join(src_root, split, seq, "-1")
            road_side_dst_path = os.path.join(dst_root, split, seq, "-1")
            os.makedirs(road_side_dst_path, exist_ok=True)
            for filename in tqdm(os.listdir(road_side_src_path)):
                assert "camera0" in filename, "camera0 not in filename"
                frame_id = int(filename.split('_')[0])
                src_frame_id = frame_id - 20 if frame_id - 20 > 0 else 0
                shutil.copy2(os.path.join(road_side_src_path, "{:05d}_camera0.jpg".format(src_frame_id)), os.path.join(road_side_dst_path, "{:05d}_camera0.jpg".format(frame_id)))
                shutil.copy2(os.path.join(road_side_src_path, "{:05d}_camera1.jpg".format(src_frame_id)), os.path.join(road_side_dst_path, "{:05d}_camera1.jpg".format(frame_id)))
                shutil.copy2(os.path.join(road_side_src_path, "{:05d}_camera2.jpg".format(src_frame_id)), os.path.join(road_side_dst_path, "{:05d}_camera2.jpg".format(frame_id)))
                shutil.copy2(os.path.join(road_side_src_path, "{:05d}_radar.pcd".format(src_frame_id)), os.path.join(road_side_dst_path, "{:05d}_radar.pcd".format(frame_id)))
                shutil.copy2(os.path.join(road_side_src_path, "{:05d}.pcd".format(src_frame_id)), os.path.join(road_side_dst_path, "{:05d}.pcd".format(frame_id)))
                shutil.copy2(os.path.join(road_side_src_path, "{:05d}.yaml".format(frame_id)), os.path.join(road_side_dst_path, "{:05d}.yaml".format(frame_id)))
