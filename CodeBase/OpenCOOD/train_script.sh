python -m torch.distributed.launch  --master_port 5179 --nproc_per_node=8 --use_env opencood/tools/train_ddp.py -y opencood/hypes_yaml/v2x-radar/CameraOnly/camera_coalign.yaml

# python opencood/tools/inference_modify.py --model_dir  opencood/logs/HeterBaseline_v2xset_lidar_fcooper_2024_11_02_13_49_04

