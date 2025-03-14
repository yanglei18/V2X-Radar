import os
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    src_root = "datasets/v2x-radar-v2xset-coop-radar-data"
    dest_root = "datasets/v2x-radar-v2xset-coop-radar-data-async"
    print("hello world ...")
    
    for split in ["train", "validate"]:
        for seq in os.listdir(os.path.join(src_root, split)):     
            shutil.copytree(os.path.join(src_root, split, seq, "142"), os.path.join(dest_root, split, seq, "142"))
            src_path = os.path.join(src_root, split, seq, "-1")
            dest_path = os.path.join(dest_root, split, seq, "-1")
            os.makedirs(dest_path, exist_ok=True)
            for filename in tqdm(os.listdir(src_path)):
                if "camera0" not in filename: continue
                
                frame_id = int(filename.split('_')[0])
                src_frame_id = frame_id - 20 if frame_id - 20 > 0 else 0
                shutil.copy2(os.path.join(src_path, "{:05d}_camera0.jpg".format(src_frame_id)), os.path.join(dest_path, "{:05d}_camera0.jpg".format(frame_id)))
                shutil.copy2(os.path.join(src_path, "{:05d}_camera1.jpg".format(src_frame_id)), os.path.join(dest_path, "{:05d}_camera1.jpg".format(frame_id)))
                shutil.copy2(os.path.join(src_path, "{:05d}_camera2.jpg".format(src_frame_id)), os.path.join(dest_path, "{:05d}_camera2.jpg".format(frame_id)))
                # shutil.copy2(os.path.join(src_path, "{:05d}_radar.pcd".format(src_frame_id)), os.path.join(dest_path, "{:05d}_radar.pcd".format(frame_id)))
                shutil.copy2(os.path.join(src_path, "{:05d}.pcd".format(src_frame_id)), os.path.join(dest_path, "{:05d}.pcd".format(frame_id)))
                shutil.copy2(os.path.join(src_path, "{:05d}.yaml".format(frame_id)), os.path.join(dest_path, "{:05d}.yaml".format(frame_id)))
                
                
                
                

                
