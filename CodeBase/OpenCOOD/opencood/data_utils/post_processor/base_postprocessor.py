# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Template for AnchorGenerator
"""

import numpy as np
import torch
import cv2

from opencood.utils import box_utils
from opencood.utils import common_utils
from opencood.utils.transformation_utils import x1_to_x2, x_to_world

class BasePostprocessor(object):
    """
    Template for Anchor generator.

    Parameters
    ----------
    anchor_params : dict
        The dictionary containing all anchor-related parameters.
    train : bool
        Indicate train or test mode.

    Attributes
    ----------
    bbx_dict : dictionary
        Contain all objects information across the cav, key: id, value: bbx
        coordinates (1, 7)
    """

    def __init__(self, anchor_params, params, train=True):
        self.params = anchor_params
        self.all_params = params
        self.bbx_dict = {}
        self.train = train

    def generate_anchor_box(self):
        # needs to be overloaded
        return None

    def generate_label(self, *argv):
        return None

    def generate_gt_bbx(self, data_dict):
        """
        The base postprocessor will generate 3d groundtruth bounding box.

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.
            To generate gt boxes, transformation_matrix should be clean

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        Returns
        -------
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor, shape (N, 8, 3).
        """
        gt_box3d_list = []
        # used to avoid repetitive bounding box
        object_id_list = []

        for cav_id, cav_content in data_dict.items():
            # used to project gt bounding box to ego space
            # object_bbx_center is clean.
            transformation_matrix = cav_content['transformation_matrix_clean']

            object_bbx_center = cav_content['object_bbx_center']
            object_bbx_mask = cav_content['object_bbx_mask']
            object_ids = cav_content['object_ids']
            object_bbx_center = object_bbx_center[object_bbx_mask == 1]

            # convert center to corner
            object_bbx_corner = \
                box_utils.boxes_to_corners_3d(object_bbx_center,
                                              self.params['order'])
            projected_object_bbx_corner = \
                box_utils.project_box3d(object_bbx_corner.float(),
                                        transformation_matrix)
            gt_box3d_list.append(projected_object_bbx_corner)
            # append the corresponding ids
            object_id_list += object_ids

        # gt bbx 3d
        gt_box3d_list = torch.vstack(gt_box3d_list)
        # some of the bbx may be repetitive, use the id list to filter
        gt_box3d_selected_indices = \
            [object_id_list.index(x) for x in set(object_id_list)]
        gt_box3d_tensor = gt_box3d_list[gt_box3d_selected_indices]

        # filter the gt_box to make sure all bbx are in the range. with z dim
        gt_box3d_np = gt_box3d_tensor.cpu().numpy()
        gt_box3d_np = box_utils.mask_boxes_outside_range_numpy(gt_box3d_np,
                                                    self.params['gt_range'],
                                                    order=None)
        gt_box3d_tensor = torch.from_numpy(gt_box3d_np).to(device=gt_box3d_list.device)

        return gt_box3d_tensor

    def generate_gt_bbx_by_iou(self, data_dict):
        """
        This function is only used by DAIR-V2X + late fusion dataset

        DAIR-V2X + late fusion dataset's label are from veh-side and inf-side
        and do not have unique object id.

        So we will filter the same object by IoU

        The base postprocessor will generate 3d groundtruth bounding box.

        For early and intermediate fusion,
            data_dict only contains ego.

        For late fusion,
            data_dcit contains all cavs, so we need transformation matrix.
            To generate gt boxes, transformation_matrix should be clean

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        Returns
        -------
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor, shape (N, 8, 3).
        """
        gt_box3d_list = []

        for cav_id, cav_content in data_dict.items():
            # used to project gt bounding box to ego space
            # object_bbx_center is clean.
            transformation_matrix = cav_content['transformation_matrix_clean']

            object_bbx_center = cav_content['object_bbx_center']
            object_bbx_mask = cav_content['object_bbx_mask']
            object_ids = cav_content['object_ids']
            object_bbx_center = object_bbx_center[object_bbx_mask == 1]

            # convert center to corner
            object_bbx_corner = \
                box_utils.boxes_to_corners_3d(object_bbx_center,
                                              self.params['order'])
            projected_object_bbx_corner = \
                box_utils.project_box3d(object_bbx_corner.float(),
                                        transformation_matrix)
            gt_box3d_list.append(projected_object_bbx_corner)

        # if only ego agent
        if len(data_dict) == 1:
            gt_box3d_tensor = torch.vstack(gt_box3d_list)
        # both veh-side and inf-side label
        else:
            veh_corners_np = gt_box3d_list[0].cpu().numpy()
            inf_corners_np = gt_box3d_list[1].cpu().numpy()
            inf_polygon_list = list(common_utils.convert_format(inf_corners_np))
            veh_polygon_list = list(common_utils.convert_format(veh_corners_np))
            iou_thresh = 0.05 


            gt_from_inf = []
            for i in range(len(inf_polygon_list)):
                inf_polygon = inf_polygon_list[i]
                ious = common_utils.compute_iou(inf_polygon, veh_polygon_list)
                if (ious > iou_thresh).any():
                    continue
                gt_from_inf.append(inf_corners_np[i])
            
            if len(gt_from_inf):
                gt_from_inf = np.stack(gt_from_inf)
                gt_box3d = np.vstack([veh_corners_np, gt_from_inf])
            else:
                gt_box3d = veh_corners_np

            gt_box3d_tensor = torch.from_numpy(gt_box3d).to(device=gt_box3d_list[0].device)

        # mask_boxes_outside_range_numpy has filtering of z-dim
        # gt_box3d_np = gt_box3d_tensor.cpu().numpy()
        # gt_box3d_np = box_utils.mask_boxes_outside_range_numpy(gt_box3d_np,
        #                                             self.params['gt_range'],
        #                                             self.params['order'])
        # gt_box3d_tensor = torch.from_numpy(gt_box3d_np).to(device=gt_box3d_list[0].device)

        # need discussion. not filter z-dim.
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(gt_box3d_tensor, self.params['gt_range'])
        gt_box3d_tensor = gt_box3d_tensor[mask, :, :]


        return gt_box3d_tensor

    def generate_object_center_radar_hfov(self, cav_contents, reference_lidar_pose, enlarge_z=False):
        tmp_object_dict = {}
        for cav_content in cav_contents:
            # 取出该CAV的所有目标GT
            vehicles_dict = cav_content['params']['vehicles']

            for obj_id, obj_gt in vehicles_dict.items():
                # ---------- 1. 读取目标位姿 ----------
                location = obj_gt['location']        # [x, y, z]
                rotation = obj_gt['angle']           # [roll, pitch, yaw]

                # center是box中心偏移（有的标注没有center）
                center = obj_gt.get('center', [0, 0, 0])

                # ---------- 2. 构造 object_pose (CAV坐标系下) ----------
                object_pose = [
                    location[0] + center[0],
                    location[1] + center[1],
                    location[2] + center[2],
                    rotation[0], rotation[1], rotation[2]
                ]

                # ---------- 4. world -> radar ----------
                object_pose2cav = x1_to_x2(object_pose, cav_content['params']['lidar_pose'])
                radar_extrinsic = np.linalg.inv(cav_content['params']['radar']['extrinsic'])
                object_pose_in_radar = np.dot(radar_extrinsic, object_pose2cav)

                # ---------- 5. 提取 radar 坐标系下位置 ----------
                x_radar = object_pose_in_radar[0, 3]
                y_radar = object_pose_in_radar[1, 3]

                # ---------- 6. 判断是否Infra yaml中lidar_pose为[0,0,0,0,0,0] ----------
                lidar_pose = cav_content['params']['lidar_pose']
                t_world = x_to_world(lidar_pose)

                is_Infra = (t_world[0, 3] == 0) and (t_world[1, 3] == 0)

                # ---------- 7. 计算雷达水平角 ----------
                # Infra: x轴指向前方，使用 arctan2(y, x)
                # Vehicle: y轴指向前方，使用 arctan2(x, y)
                if is_Infra:
                    angle_deg = np.degrees(np.arctan2(y_radar, x_radar))
                else:
                    angle_deg = np.degrees(np.arctan2(x_radar, y_radar))

                # ---------- 8. Radar HFOV过滤 ----------
                # 水平角在阈值内即可（角度检查本身已能过滤掉后方目标）
                if is_Infra:
                    hfov_thresh = 56
                else:
                    hfov_thresh = 50

                if abs(angle_deg) <= hfov_thresh:
                    # ---------- 9. 边缘区域距离过滤 ----------
                    # 在 [hfov_thresh-10, hfov_thresh] 范围内且距离超过25m的目标滤除
                    distance_radar = np.sqrt(x_radar**2 + y_radar**2)
                    # 检查是否在边缘区域且距离过远
                    if (abs(angle_deg) >= hfov_thresh - 15) and (distance_radar > 25.0):
                        continue  # 滤除此目标
                    
                    tmp_object_dict[obj_id] = obj_gt

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range'] \
            if self.train else self.params['gt_range']

        box_utils.project_world_objects(tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'],
                                        enlarge_z)

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)
        return object_np, mask, object_ids

    def generate_object_center_radar_bycamhfov(self, cav_contents, reference_lidar_pose, enlarge_z=False):
        """
        根据相机水平视场角(HFOV)过滤目标
        使用固定的半视场角40度（总视场角80度）
        目标需要落在data_aug_conf中image_list指定的任意一个相机的视场角内
        """
        tmp_object_dict = {}

        # 确定数据集类型和image_list
        if 'v2x-r' in self.all_params['root_dir'].split('/'): 
            dataset_type = 'v2x-r'
            image_list = [0]
            hfov_thresh_dict = {"rsu": 100.0/2, "vehicle": 100.0/2}
        if 'v2x-radar' in self.all_params['root_dir'].split('/'): 
            dataset_type = 'v2x-radar'
            image_list = [1]
            hfov_thresh_dict = {"rsu": 48.77/2, "vehicle": 68.40/2}
        
        for cav_content in cav_contents:
            # 取出该CAV的所有目标GT
            vehicles_dict = cav_content['params']['vehicles']
            params = cav_content['params']
            is_rsu = cav_content['params']['RSU']
            if is_rsu: hfov_thresh = hfov_thresh_dict["rsu"]
            else: hfov_thresh = hfov_thresh_dict["vehicle"]
                
            # ========== 优化：预先计算所有相机的变换矩阵（避免在循环内重复计算） ==========
            lidar_to_camera_matrices = {}  # {cam_idx: lidar_to_camera_matrix}
            
            for cam_idx in image_list:
                cam_key = f"camera{cam_idx}"
                
                # 获取相机外参
                camera_coords = np.array(params[cam_key]['cords']).astype(np.float32)
                lidar_pose_clean = params.get('lidar_pose_clean', params['lidar_pose'])
                camera_to_lidar = x1_to_x2(camera_coords, lidar_pose_clean).astype(np.float32)
                
                # v2x-r数据集需要坐标转换
                if dataset_type == "v2x-r":
                    camera_to_lidar = camera_to_lidar @ np.array([
                        [0, 0, 1, 0], 
                        [1, 0, 0, 0], 
                        [0, -1, 0, 0], 
                        [0, 0, 0, 1]
                    ], dtype=np.float32)
                
                # 计算lidar_to_camera（只计算一次，后续复用）
                lidar_to_camera = np.linalg.inv(camera_to_lidar)
                lidar_to_camera_matrices[cam_idx] = lidar_to_camera

            for obj_id, obj_gt in vehicles_dict.items():
                # ---------- 1. 读取目标位姿 ----------
                location = obj_gt['location']        # [x, y, z]
                rotation = obj_gt['angle']           # [roll, pitch, yaw]

                # center是box中心偏移（有的标注没有center）
                center = obj_gt.get('center', [0, 0, 0])

                # ---------- 2. 构造 object_pose (CAV坐标系下) ----------
                object_pose = [
                    location[0] + center[0],
                    location[1] + center[1],
                    location[2] + center[2],
                    rotation[0], rotation[1], rotation[2]
                ]

                # ---------- 3. world -> lidar (CAV坐标系) ----------
                object_pose2cav = x1_to_x2(object_pose, cav_content['params']['lidar_pose'])
                
                # 提取目标在CAV坐标系下的位置
                object_pos_cav = object_pose2cav[:3, 3]  # [x, y, z]
                
                # ---------- 4. 检查目标是否在任意一个相机的HFOV内 ----------
                in_camera_fov = False
                
                # 使用预先计算的变换矩阵
                object_pos_hom = np.concatenate([object_pos_cav, [1.0]])
                
                for cam_idx, lidar_to_camera in lidar_to_camera_matrices.items():
                    # 将目标位置转换到相机坐标系（使用预先计算的矩阵）
                    object_pos_cam = (lidar_to_camera @ object_pos_hom)[:3]  # [x, y, z] in camera coord
                    
                    # 检查深度是否有效（z > 0表示在相机前方）
                    if object_pos_cam[2] <= 0:
                        continue
                    
                    # 计算水平角（在相机坐标系中）
                    # 相机坐标系：x向右，y向下，z向前
                    # 水平角 = arctan2(x, z)
                    x_cam = object_pos_cam[0]
                    z_cam = object_pos_cam[2]
                    horizontal_angle_deg = np.degrees(np.arctan2(x_cam, z_cam))
                    
                    # 检查是否在HFOV内（使用固定的40度半视场角）
                    if abs(horizontal_angle_deg) <= hfov_thresh:
                        in_camera_fov = True
                        break  # 只要在任意一个相机内即可
                
                # 如果目标在任意一个相机的视场角内，保留
                if in_camera_fov:
                    tmp_object_dict[obj_id] = obj_gt

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range'] \
            if self.train else self.params['gt_range']

        box_utils.project_world_objects(tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'],
                                        enlarge_z)

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)
        return object_np, mask, object_ids

    def generate_object_center_radar_filter_by_points(self, cav_contents, reference_lidar_pose, enlarge_bbox=1.3, enlarge_z=False):
        tmp_object_dict = {}
        
        # Process each CAV and filter boxes by radar points in their own coordinate system
        for cav_content in cav_contents:
            vehicles_dict = cav_content['params']['vehicles']
            
            # Get radar points if available
            if 'radar_np' in cav_content and cav_content['radar_np'] is not None:
                radar_np = cav_content['radar_np'].copy()
                # First transform radar points from radar coordinate to lidar coordinate
                if 'radar' in cav_content['params'].keys(): radar_coords = np.array(cav_content['params']['radar']['cords']).astype(np.float32)
                if 'radar_pose' in cav_content['params'].keys(): radar_coords = np.array(cav_content['params']['radar_pose']).astype(np.float32)
                lidar_pose = cav_content['params']['lidar_pose']
                radar_to_lidar_matrix = x1_to_x2(radar_coords, lidar_pose)  # T_lidar_radar
                # Transform radar points to lidar coordinate system
                radar_xyz_hom = np.concatenate([radar_np[:, :3], np.ones((radar_np.shape[0], 1))], axis=1)
                radar_xyz_lidar = (radar_to_lidar_matrix @ radar_xyz_hom.T).T[:, :3]

                # Get boxes in this CAV's coordinate system (use lidar_pose as reference)
                cav_boxes_dict = {}
                box_utils.project_world_objects(vehicles_dict,
                                                cav_boxes_dict,
                                                lidar_pose,
                                                self.params['gt_range'] if not self.train else self.params['anchor_args']['cav_lidar_range'],
                                                self.params['order'],
                                                enlarge_z)
                
                # Convert boxes from center format to format for points_in_boxes_cpu
                if len(cav_boxes_dict) == 0:
                    continue
                # Extract boxes: [N, 7] format [x, y, z, dx, dy, dz, heading]
                boxes_center = np.array([bbx[0] for bbx in cav_boxes_dict.values()])  # [N, 7]
                if boxes_center.ndim == 1:
                    boxes_center = boxes_center.reshape(1, -1)
                object_ids_list = list(cav_boxes_dict.keys())
                
                # Enlarge boxes by 1.3x for point filtering
                boxes_for_points = boxes_center.copy()
                boxes_for_points[:, 3:6] = boxes_for_points[:, 3:6]*enlarge_bbox  # Enlarge l, w, h
                
                # Convert to lwh format for points_in_boxes_cpu
                if self.params['order'] == 'hwl':
                    boxes_for_points = boxes_for_points[:, [0, 1, 2, 5, 4, 3, 6]]
                else:
                    pass  # Already in lwh format

                # Filter boxes by points
                from packages.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
                point_indices = points_in_boxes_cpu(radar_xyz_lidar, boxes_for_points)  # [N, num_points]
                valid_mask = point_indices.sum(axis=1) > 0
                
                # Get final valid object IDs (point filtered)
                for idx, obj_id in enumerate(object_ids_list):
                    if valid_mask[idx]:
                        tmp_object_dict[obj_id] = vehicles_dict[obj_id]

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range'] \
            if self.train else self.params['gt_range']

        box_utils.project_world_objects(tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'],
                                        enlarge_z)

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)
        return object_np, mask, object_ids

    def generate_object_center_radar_hfov_filter_by_points(self, cav_contents, reference_lidar_pose, enlarge_bbox=1.3, enlarge_z=False):
        tmp_object_dict = {}
        
        # Process each CAV and filter boxes by radar points in their own coordinate system
        for cav_content in cav_contents:
            vehicles_dict = cav_content['params']['vehicles']
            
            # Get radar points if available
            if 'radar_np' in cav_content and cav_content['radar_np'] is not None:
                radar_np = cav_content['radar_np'].copy()
                # First transform radar points from radar coordinate to lidar coordinate
                if 'radar' in cav_content['params'].keys(): radar_coords = np.array(cav_content['params']['radar']['cords']).astype(np.float32)
                if 'radar_pose' in cav_content['params'].keys(): radar_coords = np.array(cav_content['params']['radar_pose']).astype(np.float32)
                lidar_pose = cav_content['params']['lidar_pose']
                radar_to_lidar_matrix = x1_to_x2(radar_coords, lidar_pose)  # T_lidar_radar
                # Transform radar points to lidar coordinate system
                radar_xyz_hom = np.concatenate([radar_np[:, :3], np.ones((radar_np.shape[0], 1))], axis=1)
                radar_xyz_lidar = (radar_to_lidar_matrix @ radar_xyz_hom.T).T[:, :3]

                # Get boxes in this CAV's coordinate system (use lidar_pose as reference)
                cav_boxes_dict = {}
                box_utils.project_world_objects(vehicles_dict,
                                                cav_boxes_dict,
                                                lidar_pose,
                                                self.params['gt_range'] if not self.train else self.params['anchor_args']['cav_lidar_range'],
                                                self.params['order'],
                                                enlarge_z)
                
                # Convert boxes from center format to format for points_in_boxes_cpu
                if len(cav_boxes_dict) == 0:
                    continue
                # Extract boxes: [N, 7] format [x, y, z, dx, dy, dz, heading]
                boxes_center = np.array([bbx[0] for bbx in cav_boxes_dict.values()])  # [N, 7]
                if boxes_center.ndim == 1:
                    boxes_center = boxes_center.reshape(1, -1)
                object_ids_list = list(cav_boxes_dict.keys())
                
                # Enlarge boxes by 1.3x for point filtering
                boxes_for_points = boxes_center.copy()
                boxes_for_points[:, 3:6] = boxes_for_points[:, 3:6]*enlarge_bbox  # Enlarge l, w, h
                
                # Convert to lwh format for points_in_boxes_cpu
                if self.params['order'] == 'hwl':
                    boxes_for_points = boxes_for_points[:, [0, 1, 2, 5, 4, 3, 6]]
                else:
                    pass  # Already in lwh format

                # Filter by angle (similar to generate_object_center_radar_hfov)
                valid_object_ids_angle = []
                for i, obj_id in enumerate(object_ids_list):
                    obj_gt = vehicles_dict[obj_id]
                    location = obj_gt['location']
                    rotation = obj_gt['angle']
                    center = obj_gt.get('center', [0, 0, 0])
                    
                    object_pose = [
                        location[0] + center[0],
                        location[1] + center[1],
                        location[2] + center[2],
                        rotation[0], rotation[1], rotation[2]
                    ]
                    
                    # world -> radar
                    object_pose2cav = x1_to_x2(object_pose, lidar_pose)
                    radar_extrinsic = np.linalg.inv(cav_content['params']['radar']['extrinsic'])
                    object_pose_in_radar = np.dot(radar_extrinsic, object_pose2cav)
                    
                    x_radar = object_pose_in_radar[0, 3]
                    y_radar = object_pose_in_radar[1, 3]
                    
                    # Check if Infra
                    t_world = x_to_world(lidar_pose)
                    is_Infra = (t_world[0, 3] == 0) and (t_world[1, 3] == 0)
                    
                    # Calculate angle
                    if is_Infra:
                        angle_deg = np.degrees(np.arctan2(y_radar, x_radar))
                        hfov_thresh = 56-15
                    else:
                        angle_deg = np.degrees(np.arctan2(x_radar, y_radar))
                        hfov_thresh = 50-15
                    
                    # Filter by angle
                    if abs(angle_deg) <= hfov_thresh:
                        valid_object_ids_angle.append(i)
                
                if len(valid_object_ids_angle) == 0:
                    continue
                
                # Filter boxes by points (only for angle-valid boxes)
                boxes_for_points_filtered = boxes_for_points[valid_object_ids_angle]
                from packages.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
                point_indices = points_in_boxes_cpu(radar_xyz_lidar, boxes_for_points_filtered)  # [N, num_points]
                valid_mask = point_indices.sum(axis=1) > 0
                
                # Get final valid object IDs (both angle and point filtered)
                for idx, valid_idx in enumerate(valid_object_ids_angle):
                    if valid_mask[idx]:
                        tmp_object_dict[object_ids_list[valid_idx]] = vehicles_dict[object_ids_list[valid_idx]]

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range'] \
            if self.train else self.params['gt_range']

        box_utils.project_world_objects(tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'],
                                        enlarge_z)

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)
        return object_np, mask, object_ids

    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose,
                               enlarge_z=False):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        enlarge_z :
            if True, enlarge the z axis range to include more object

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        tmp_object_dict = {}
        for cav_content in cav_contents:
            tmp_object_dict.update(cav_content['params']['vehicles'])

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range'] \
            if self.train else self.params['gt_range']

        box_utils.project_world_objects(tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'],
                                        enlarge_z)

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)
        return object_np, mask, object_ids

    def generate_object_center_v2x(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            In fact, only the ego vehile needs to generate object center

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        # from opencood.data_utils.datasets import GT_RANGE

        assert len(cav_contents) == 1
        
        """
        In old version, we only let ego agent return gt box.
        Other agent return empty.

        But it's not suitable for late fusion.
        Also, we should filter out boxes that don't have any lidar point hits.

        Thankfully, 'lidar_np' is in cav_contents[0].keys()
        """


        gt_boxes = cav_contents[0]['params']['vehicles'] # notice [N,10], 10 includes [x,y,z,dx,dy,dz,w,a,b,c]
        object_ids = cav_contents[0]['params']['object_ids']
        lidar_np = cav_contents[0]['lidar_np']
        
        tmp_object_dict = {"gt_boxes": gt_boxes, "object_ids":object_ids}

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range'] # v2x we don't use GT_RANGE.

        box_utils.project_world_objects_v2x(tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'],
                                        lidar_np=lidar_np)

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []


        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)

        return object_np, mask, object_ids

    def generate_object_center_dairv2x(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        # tmp_object_dict = {}
        tmp_object_list = []
        cav_content = cav_contents[0]
        tmp_object_list = cav_content['params']['vehicles'] #世界坐标系下

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range']


        box_utils.project_world_objects_dairv2x(tmp_object_list,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'])

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)

        return object_np, mask, object_ids

    def generate_object_center_dairv2x_single(self,
                               cav_contents,
                               suffix=""):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        # tmp_object_dict = {}
        tmp_object_list = []
        cav_content = cav_contents[0]
        tmp_object_list = cav_content['params'][f'vehicles{suffix}'] # ego 坐标系下

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range']


        box_utils.load_single_objects_dairv2x(tmp_object_list,
                                        output_dict,
                                        filter_range,
                                        self.params['order'])

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)

        return object_np, mask, object_ids

    def generate_object_center_dairv2x_single_hetero(self,
                               cav_contents,
                               reference_lidar_pose,
                               suffix,
                               ):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        # tmp_object_dict = {}
        tmp_object_list = []
        cav_content = cav_contents[0]
        tmp_object_list = cav_content['params'][f'vehicles{suffix}'] # ego 坐标系下

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range']

        cav_coor = cav_content['params']['lidar_pose'] # T_world_cav
        ego_coor = reference_lidar_pose # T_world_ego
        T_ego_cav = x1_to_x2(cav_coor, ego_coor) # T_ego_cav

        box_utils.load_single_objects_dairv2x_hetero(tmp_object_list,
                                        output_dict,
                                        filter_range,
                                        T_ego_cav,
                                        self.params['order'])

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)

        return object_np, mask, object_ids

    def generate_visible_object_center(self,
                               cav_contents,
                               reference_lidar_pose,
                               enlarge_z=False):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        visibility_map : np.ndarray, uint8
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        enlarge_z :
            if True, enlarge the z axis range to include more object

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        tmp_object_dict = {}
        for cav_content in cav_contents:
            tmp_object_dict.update(cav_content['params']['vehicles'])

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range'] # if self.train else GT_RANGE_OPV2V
        inf_filter_range = [-1e5, -1e5, -1e5, 1e5, 1e5, 1e5]
        visibility_map = np.asarray(cv2.cvtColor(cav_contents[0]["bev_visibility.png"], cv2.COLOR_BGR2GRAY))
        ego_lidar_pose = cav_contents[0]["params"]["lidar_pose_clean"]

        # 1-time filter: in ego coordinate, use visibility map to filter.
        box_utils.project_world_visible_objects(tmp_object_dict,
                                        output_dict,
                                        ego_lidar_pose,
                                        inf_filter_range,
                                        self.params['order'],
                                        visibility_map,
                                        enlarge_z)

        updated_tmp_object_dict = {}
        for k, v in tmp_object_dict.items():
            if k in output_dict:
                updated_tmp_object_dict[k] = v # not visible
        output_dict = {}

        # 2-time filter: use reference_lidar_pose
        box_utils.project_world_objects(updated_tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'],
                                        enlarge_z)

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)

        return object_np, mask, object_ids

    def generate_object_center_v2xset_camera(self,
                               cav_contents,
                               reference_lidar_pose,
                               enlarge_z=False):

        tmp_object_dict = {}
        for cav_content in cav_contents:
            tmp_object_dict.update(cav_content['params']['vehicles'])

        output_dict = {}
        # Use cav_lidar_range from config instead of hardcoded 90m range
        # This allows RSU cameras to utilize their long-range perception capability
        # (e.g., 200m+ range) even if targets are far from ego car
        filter_range = self.params['anchor_args']['cav_lidar_range'] \
            if self.train else self.params['gt_range']

        box_utils.project_world_objects(tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'],
                                        enlarge_z)

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)
        return object_np, mask, object_ids