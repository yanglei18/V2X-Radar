#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project LiDAR and Radar point clouds to camera images and generate videos

This script processes all timestamps in a scenario directory and generates two videos:
    - BEV video: BEV view with all CAVs' LiDAR and Radar point clouds
    - Camera grid video: 5 rows x 3 columns grid image showing camera projections
"""
import yaml
import numpy as np
import cv2
import os
import sys
import open3d as o3d
from pathlib import Path
import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from opencood.utils import pcd_utils

# ==================== Constants ====================
LIDAR_POINT_RADIUS = 2
RADAR_POINT_RADIUS = 10
VERBOSE = False  # Global verbose flag for controlling output messages
MAX_POINTS_PER_CAV = 50000  # Maximum points to draw per CAV for performance
SHOW_LEGEND = False  # Disable legend for faster rendering

# Color configuration
LIDAR_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
RADAR_COLORS = ['cyan', 'magenta', 'magenta', 'pink', 'lightblue', 'lightgreen']
GT_BOX_COLOR_EGO_BGR = (0, 255, 0)      # Green for ego (filled)
GT_BOX_COLOR_OTHER_BGR = (0, 165, 255)  # Orange for other CAVs (outline only) - BGR format
GT_BOX_COLOR_EGO_MPL = 'green'          # Green for ego (filled)
GT_BOX_COLOR_OTHER_MPL = 'orange'       # Orange for other CAVs (outline only)


# ==================== Utility Functions ====================
def load_yaml(yaml_path):
    """Load YAML file, handling numpy objects if present"""
    with open(yaml_path, 'r') as f:
        try:
            # Try safe_load first (faster and safer)
            return yaml.safe_load(f)
        except yaml.YAMLError:
            # If safe_load fails (e.g., due to numpy objects), use load with FullLoader
            f.seek(0)  # Reset file pointer
            return yaml.load(f, Loader=yaml.FullLoader)


def get_lidar_pose(params):
    """Get LiDAR pose from parameters"""
    if "true_ego_pose" in params:
        return np.array(params["true_ego_pose"]).astype(np.float32)
    return np.array(params["lidar_pose"]).astype(np.float32)


def transform_points_to_ego(points, cav_lidar_pose, ego_lidar_pose):
    """Transform points from CAV coordinate system to ego coordinate system"""
    if np.array_equal(cav_lidar_pose, ego_lidar_pose):
        return points
    
    T_cav_to_ego = x1_to_x2(cav_lidar_pose, ego_lidar_pose)
    xyz = points[:, :3]
    xyz_hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    xyz_transformed = (T_cav_to_ego @ xyz_hom.T).T[:, :3]
    return np.hstack([xyz_transformed, points[:, 3:]]).astype(np.float32)


def load_lidar_point_cloud(pcd_path):
    """
    Load point cloud file using pcd_utils.read_lidar (same as in opv2v_basedataset)
    
    Returns xyz and intensity separately for compatibility.
    Note: pcd_utils.read_lidar already filters out nan values.
    
    Returns
    -------
    xyz : np.ndarray
        Point cloud coordinates, shape [N, 3]
    intensity : np.ndarray
        Intensity values, shape [N, 1]
    """
    try:
        # Use pcd_utils.read_lidar which handles nan filtering automatically
        pcd_np, time = pcd_utils.read_lidar(pcd_path)
        
        if len(pcd_np) == 0:
            return None, None
        
        # pcd_np is [N, 4] format: [x, y, z, intensity]
        xyz = pcd_np[:, :3]  # [N, 3]
        intensity = pcd_np[:, 3:4]  # [N, 1] - keep as 2D array
        
        return xyz, intensity
    except Exception as e:
        print(f"    Error loading point cloud from {pcd_path}: {e}")
        return None, None


def load_radar_point_cloud(pcd_path):
    """
    Load point cloud file using pcd_utils.read_lidar (same as in opv2v_basedataset)
    
    Returns xyz and intensity separately for compatibility.
    Note: pcd_utils.read_lidar already filters out nan values.
    
    Returns
    -------
    xyz : np.ndarray
        Point cloud coordinates, shape [N, 3]
    intensity : np.ndarray
        Intensity values, shape [N, 1]
    """
    try:
        # Use pcd_utils.read_lidar which handles nan filtering automatically
        pcd_np, time = pcd_utils.read_radar(pcd_path)
        
        if len(pcd_np) == 0:
            return None, None
        
        # pcd_np is [N, 4] format: [x, y, z, intensity]
        xyz = pcd_np[:, :3]  # [N, 3]
        dopple = pcd_np[:, 4:5]  # [N, 1] - keep as 2D array
        
        return xyz, dopple
    except Exception as e:
        print(f"    Error loading point cloud from {pcd_path}: {e}")
        return None, None


# ==================== Coordinate Transformation Functions ====================
def x_to_world(pose):
    """
    Convert pose to transformation matrix in world coordinate system T_world_x
    
    Args:
        pose: [x, y, z, roll, yaw, pitch] (degrees)
    
    Returns:
        [4, 4] transformation matrix
    """
    x, y, z, roll, yaw, pitch = pose[:]
    
    c_y, s_y = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    c_r, s_r = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    c_p, s_p = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    
    matrix = np.identity(4)
    matrix[0, 3], matrix[1, 3], matrix[2, 3] = x, y, z
    
    # Rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    
    return matrix


def x1_to_x2(x1_pose, x2_pose):
    """
    Compute transformation matrix from x1 coordinate system to x2 coordinate system T_x2_x1
    
    Args:
        x1_pose: Pose of x1 in world coordinate system
        x2_pose: Pose of x2 in world coordinate system
    
    Returns:
        [4, 4] transformation matrix
    """
    T_x1_to_world = x_to_world(x1_pose)
    T_x2_to_world = x_to_world(x2_pose)
    T_world_to_x2 = np.linalg.inv(T_x2_to_world)
    return T_world_to_x2 @ T_x1_to_world

def get_radar_to_lidar(params):
    """Get transformation matrix from radar to lidar"""
    if 'radar' in params.keys(): 
        radar_coords = np.array(params["radar"]["cords"]).astype(np.float32)
    else: radar_coords = np.array(params["radar_pose"]).astype(np.float32)
    lidar_pose = get_lidar_pose(params)
    radar_to_lidar = x1_to_x2(radar_coords, lidar_pose).astype(np.float32)
    return radar_to_lidar


def get_ext_int(params, camera_id, dataset_type):
    """Get camera extrinsic and intrinsic matrices"""
    camera_key = f"camera{camera_id}" if f"camera{camera_id}" in params else "camera0"
    camera_coords = np.array(params[camera_key]["cords"]).astype(np.float32)
    lidar_pose = get_lidar_pose(params)
    
    camera_to_lidar = x1_to_x2(camera_coords, lidar_pose).astype(np.float32)
    camera_intrinsic = np.array(params[camera_key]["intrinsic"]).astype(np.float32)
    if dataset_type == "v2x-r":
        camera_to_lidar = camera_to_lidar @ np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)  # UE4 coord to opencv coord
    
    return camera_to_lidar, camera_intrinsic


# ==================== Filter Functions ====================
def filter_lidar_points_by_image(lidar_np, params, raw_resolution=None, broaden_horizontal=1.5, image_list=[0, 1, 2], dataset_type='v2x-radar'):
    """
    Filter LiDAR points that cannot be projected to any camera image.
    
    A point is kept if it can be projected to at least one camera image (OR operation).
    This function checks all cameras specified by image_list and keeps points visible in any of them.
    
    Parameters
    ----------
    lidar_np : np.ndarray
        LiDAR point cloud, shape [N, 4] (x, y, z, intensity)
    params : dict
        Dictionary containing CAV parameters including camera information
    raw_resolution : tuple or None
        (H, W) raw image resolution. If None, will try to get from actual images
    broaden_horizontal : float
        Multiplier to enlarge the image size to avoid points being projected to the edge
    image_list : list
        List of camera indices to check (e.g., [0, 1, 2, 3])
        Points visible in any specified camera will be kept
    
    Returns
    -------
    filtered_lidar_np : np.ndarray
        Filtered LiDAR point cloud (only points visible in at least one camera)
    """
    lidar_xyz = lidar_np[:, :3]  # [N, 3]
    N = lidar_xyz.shape[0]
    
    # Get image dimensions
    if raw_resolution is not None:
        img_H, img_W = raw_resolution
    else:
        # Try to get from camera0 intrinsic or default values
        # Default to common camera resolution if not specified
        img_H, img_W = 864, 1536  # Default resolution
    
    # Store original dimensions
    orig_H, orig_W = img_H, img_W
    
    # Calculate expansion: enlarge image size symmetrically on both sides
    # For example, if broaden_horizontal=2, we expand from center:
    # Original: [0, W], expanded: [-W*(broaden-1)/2, W + W*(broaden-1)/2]
    expand_W = orig_W * (broaden_horizontal - 1) / 2
    expand_H = orig_H * (broaden_horizontal - 1) / 2
    
    # New boundaries (symmetric expansion)
    u_min, u_max = -expand_W, orig_W + expand_W
    v_min, v_max = -expand_H, orig_H + expand_H
    
    # Convert to homogeneous coordinates
    lidar_xyz_hom = np.concatenate([lidar_xyz, np.ones((N, 1))], axis=1)  # [N, 4]
    
    # Track which points are valid (can be projected to at least one camera)
    valid_mask = np.zeros(N, dtype=bool)
    
    # Check each camera
    for idx in image_list:
        try:
            camera_to_lidar, camera_intrinsic = get_ext_int(params, idx, dataset_type)
        except Exception as e:
            continue
        
        # Compute lidar2cam (inverse of camera_to_lidar)
        try:
            lidar2cam = np.linalg.inv(camera_to_lidar)[:3, :4]  # [3, 4]
        except np.linalg.LinAlgError:
            continue
        
        # Project points: int_matrix @ lidar2cam @ xyz_hom.T
        # camera_intrinsic is [3, 3], lidar2cam is [3, 4], xyz_hom is [N, 4]
        img_pts = (camera_intrinsic @ lidar2cam @ lidar_xyz_hom.T).T  # [N, 3] = (u*z, v*z, z)
        
        # Extract depth and normalize
        depth = img_pts[:, 2]  # [N]
        
        # Filter by depth: only keep points in front of camera (depth > 0.1)
        # Points with negative depth are behind the camera and should be filtered
        valid_depth = depth > 0.1  # Filter by depth (must be positive and > 0.1)
        
        # Normalize by depth to get pixel coordinates
        # Only compute uv for points with valid depth (positive and > 0.1)
        valid_depth_mask = np.isfinite(depth) & (depth > 0.1)
        uv = np.zeros((len(img_pts), 2), dtype=np.float32)
        if valid_depth_mask.any():
            uv[valid_depth_mask] = img_pts[valid_depth_mask, :2] / (depth[valid_depth_mask, None] + 1e-8)
        uv[~valid_depth_mask] = -1e6
        
        # Check if points are within image bounds (with symmetric expansion)
        in_bounds = np.zeros(len(uv), dtype=bool)
        in_bounds[valid_depth_mask] = (uv[valid_depth_mask, 0] >= u_min) & (uv[valid_depth_mask, 0] < u_max) & \
                                     (uv[valid_depth_mask, 1] >= v_min) & (uv[valid_depth_mask, 1] < v_max)
        
        # Combined valid mask for this camera
        camera_valid_mask = valid_depth & in_bounds
        valid_mask = valid_mask | camera_valid_mask  # OR operation: valid if visible in any camera
    
    # Filter points: keep only those that can be projected to at least one camera
    filtered_lidar_np = lidar_np[valid_mask]
    
    return filtered_lidar_np


# ==================== Projection Functions ====================
def project_lidar_to_image(points, camera_to_lidar, camera_intrinsic, image_shape):
    """
    Project LiDAR points to image coordinates
    
    Args:
        points: [N, 3] or [N, 4] point cloud
        camera_to_lidar: [4, 4] transformation from camera to LiDAR
        camera_intrinsic: [3, 3] camera intrinsic matrix
        image_shape: (H, W) image shape
    
    Returns:
        uv: [N, 2] image coordinates
        valid_mask: [N] valid mask
        depth: [N] depth values
    """
    xyz = points[:, :3]
    xyz_hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    
    lidar2cam = np.linalg.inv(camera_to_lidar)[:3, :4]
    img_pts = (camera_intrinsic @ lidar2cam @ xyz_hom.T).T
    
    depth = img_pts[:, 2]
    uv = img_pts[:, :2] / (depth[:, None] + 1e-8)
    
    H, W = image_shape
    valid_depth = depth > 0.1
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    
    return uv, valid_depth & in_bounds, depth


# ==================== GT Box Processing Functions ====================
def create_bbx(extent):
    """Create bounding box with 8 corners"""
    return np.array([
        [extent[0], -extent[1], -extent[2]], [extent[0], extent[1], -extent[2]],
        [-extent[0], extent[1], -extent[2]], [-extent[0], -extent[1], -extent[2]],
        [extent[0], -extent[1], extent[2]], [extent[0], extent[1], extent[2]],
        [-extent[0], extent[1], extent[2]], [-extent[0], -extent[1], extent[2]]
    ])


def get_gt_boxes_in_lidar(params, lidar_pose):
    """Get GT bounding boxes in LiDAR coordinate system"""
    if 'vehicles' not in params or len(params['vehicles']) == 0:
        return None, None
    
    boxes_list, object_ids = [], []
    
    for obj_id, vehicle in params['vehicles'].items():
        location, rotation = vehicle['location'], vehicle['angle']
        center = vehicle.get('center', [0, 0, 0])
        extent = vehicle['extent']
        
        object_pose = [
            location[0] + center[0], location[1] + center[1], location[2] + center[2],
            rotation[0], rotation[1], rotation[2]
        ]
        
        object2lidar = x1_to_x2(object_pose, lidar_pose)
        bbx = create_bbx(extent).T
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]
        bbx_lidar = (object2lidar @ bbx).T[:, :3]
        
        boxes_list.append(bbx_lidar)
        object_ids.append(int(obj_id))
    
    if boxes_list:
        return np.array(boxes_list), object_ids
    else:
        return None, None


def deduplicate_gt_boxes(gt_boxes, sources, distance_threshold=1.0, size_ratio_threshold=0.8):
    """
    Remove duplicate GT boxes based on center distance and size similarity
    
    Args:
        gt_boxes: [N, 8, 3] GT boxes with 8 corners
        sources: [N] list of source CAV IDs for each box
        distance_threshold: Maximum center distance to consider as duplicate (meters)
        size_ratio_threshold: Minimum size ratio to consider as duplicate
    
    Returns:
        deduplicated_boxes: [M, 8, 3] deduplicated boxes
        deduplicated_sources: [M] deduplicated sources
        keep_indices: [M] indices of kept boxes
    """
    if gt_boxes is None or len(gt_boxes) == 0:
        return gt_boxes, sources, []
    
    N = len(gt_boxes)
    
    # Compute centers and sizes for each box
    centers = gt_boxes.mean(axis=1)  # [N, 3]
    sizes = gt_boxes.max(axis=1) - gt_boxes.min(axis=1)  # [N, 3]
    
    keep_mask = np.ones(N, dtype=bool)
    
    for i in range(N):
        if not keep_mask[i]:
            continue
        
        for j in range(i + 1, N):
            if not keep_mask[j]:
                continue
            
            # Compute center distance
            center_dist = np.linalg.norm(centers[i] - centers[j])
            
            # Compute size similarity (ratio of smaller to larger)
            size_i = np.linalg.norm(sizes[i])
            size_j = np.linalg.norm(sizes[j])
            size_ratio = min(size_i, size_j) / (max(size_i, size_j) + 1e-6)
            
            # If boxes are similar (close center and similar size), remove the duplicate
            if center_dist < distance_threshold and size_ratio > size_ratio_threshold:
                # Keep the one from ego CAV if available, otherwise keep the first one
                # For simplicity, we keep the first one (i) and remove the later one (j)
                keep_mask[j] = False
    
    keep_indices = np.where(keep_mask)[0]
    deduplicated_boxes = gt_boxes[keep_indices]
    deduplicated_sources = [sources[i] for i in keep_indices]
    
    return deduplicated_boxes, deduplicated_sources, keep_indices


def project_box3d_simple(box_corners, T):
    """Project 3D bounding box corners using transformation matrix"""
    N = box_corners.shape[0]
    corners_flat = box_corners.reshape(-1, 3)
    corners_hom = np.concatenate([corners_flat, np.ones((corners_flat.shape[0], 1))], axis=1)
    projected = (T @ corners_hom.T).T[:, :3]
    return projected.reshape(N, 8, 3)


def project_boxes_to_image(gt_boxes_corners, camera_to_lidar, camera_intrinsic, H, W):
    """Project 3D bounding boxes to 2D image coordinates"""
    if gt_boxes_corners is None or len(gt_boxes_corners) == 0:
        return None, None, None
    
    N = gt_boxes_corners.shape[0]
    xyz = gt_boxes_corners.reshape(-1, 3)
    xyz_hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    
    lidar2cam = np.linalg.inv(camera_to_lidar)[:3, :4]
    img_pts = (camera_intrinsic @ lidar2cam @ xyz_hom.T).T
    
    depth = img_pts[:, 2]
    uv = img_pts[:, :2] / (depth[:, None] + 1e-8)
    uv = uv.reshape(N, 8, 2).round().astype(np.int32)
    depth = depth.reshape(N, 8)
    
    valid_mask1 = ((uv[:, :, 0] >= 0) & (uv[:, :, 0] < W) & 
                   (uv[:, :, 1] >= 0) & (uv[:, :, 1] < H))
    valid_mask2 = (depth > 0.5) & (depth < 100)
    gt_box2d_mask = valid_mask1.any(axis=1) & valid_mask2.all(axis=1)
    
    uv[:, :, 0] = np.clip(uv[:, :, 0], 0, W-1)
    uv[:, :, 1] = np.clip(uv[:, :, 1], 0, H-1)
    
    gt_box2d_bbox = np.zeros((N, 4), dtype=np.int32)
    for i in range(N):
        x_coords, y_coords = uv[i, :, 0], uv[i, :, 1]
        x_min, x_max = max(0, min(x_coords.min(), W-1)), max(0, min(x_coords.max(), W-1))
        y_min, y_max = max(0, min(y_coords.min(), H-1)), max(0, min(y_coords.max(), H-1))
        gt_box2d_bbox[i] = [x_min, y_min, x_max, y_max]
    
    return uv, gt_box2d_mask, gt_box2d_bbox


# ==================== Visualization Helper Functions ====================
def normalize_depth_for_colormap(depth, percentile_range=(2, 98)):
    """Normalize depth values to 0-255 range"""
    depth_min, depth_max = np.percentile(depth, percentile_range)
    if depth_max > depth_min:
        depth_clipped = np.clip(depth, depth_min, depth_max)
        return ((depth_clipped - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    return np.zeros_like(depth, dtype=np.uint8)


def draw_points_with_depth_coloring(img, uv, depth, color_scale='red', radius=2):
    """
    Draw points with depth-based coloring using different color schemes
    
    Args:
        img: Target image
        uv: [N, 2] image coordinates
        depth: [N] depth values
        color_scale: 'red', 'blue', 'green', 'yellow'
        radius: Point radius
    """
    if len(uv) == 0:
        return
    
    depth_norm = normalize_depth_for_colormap(depth)
    
    for pt, d_norm in zip(uv, depth_norm):
        u, v = int(pt[0]), int(pt[1])
        
        if color_scale == 'red':
            r, g, b = int(50 + (d_norm * 205 / 255)), int(d_norm * 0.4), 0
        elif color_scale == 'blue':
            r, g, b = 0, int(d_norm * 0.4), int(50 + (d_norm * 205 / 255))
        elif color_scale == 'green':
            r, g, b = int(d_norm * 0.4), int(50 + (d_norm * 205 / 255)), 0
        elif color_scale == 'yellow':
            r, g, b = int(50 + (d_norm * 205 / 255)), int(50 + (d_norm * 205 / 255)), 0
        else:
            r, g, b = 255, 255, 255
        
        cv2.circle(img, (u, v), radius, (b, g, r), -1)


def separate_ego_and_other(points_dict, ref_cav_id):
    """Separate ego CAV and other CAV point clouds"""
    ego_points, other_points = None, None
    ref_cav_id_str = str(ref_cav_id)
    
    for cav_id, points in points_dict.items():
        if str(cav_id) == ref_cav_id_str:
            ego_points = points
        elif other_points is None:
            other_points = points
    
    return ego_points, other_points


def get_gt_box_color(cav_id, ref_cav_id, is_bgr=True):
    """Get GT box color for a CAV: ego gets one color, others get unified color"""
    if str(cav_id) == str(ref_cav_id):
        return GT_BOX_COLOR_EGO_BGR if is_bgr else GT_BOX_COLOR_EGO_MPL
    else:
        return GT_BOX_COLOR_OTHER_BGR if is_bgr else GT_BOX_COLOR_OTHER_MPL


# ==================== Main Visualization Functions ====================
def visualize_projection(image_path, all_lidar_points_dict, camera_to_lidar, camera_intrinsic,
                        gt_boxes_corners=None, gt_boxes_source=None, ref_cav_id=None,
                        all_radar_points_dict=None):
    """
    Project LiDAR and Radar points to image and visualize
    
    Returns:
        img_original_with_gt, img_ego_lidar, img_other_lidar, img_ego_radar, img_other_radar
    """
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"Error: Could not load image {image_path}")
        return None, None, None, None, None
    
    H, W = img_original.shape[:2]
    images = {
        'original': img_original.copy(),
        'ego_lidar': img_original.copy(),
        'other_lidar': img_original.copy(),
        'ego_radar': img_original.copy(),
        'other_radar': img_original.copy()
    }
    
    # Separate ego and other CAV point clouds
    ego_lidar, other_lidar = separate_ego_and_other(all_lidar_points_dict, ref_cav_id)
    
    # Draw LiDAR points
    if ego_lidar is not None:
        uv, valid_mask, depth = project_lidar_to_image(ego_lidar, camera_to_lidar, camera_intrinsic, (H, W))
        draw_points_with_depth_coloring(images['ego_lidar'], uv[valid_mask], depth[valid_mask], 
                                       'red', LIDAR_POINT_RADIUS)
    
    if other_lidar is not None:
        uv, valid_mask, depth = project_lidar_to_image(other_lidar, camera_to_lidar, camera_intrinsic, (H, W))
        draw_points_with_depth_coloring(images['other_lidar'], uv[valid_mask], depth[valid_mask], 
                                       'blue', LIDAR_POINT_RADIUS)
    
    # Draw GT boxes
    if gt_boxes_corners is not None:
        gt_box2d, gt_box2d_mask, _ = project_boxes_to_image(gt_boxes_corners, camera_to_lidar, 
                                                             camera_intrinsic, H, W)
        if gt_box2d is not None:
            edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                     [0, 4], [1, 5], [2, 6], [3, 7]]
            
            for box_idx in range(len(gt_box2d)):
                if not gt_box2d_mask[box_idx]:
                    continue
                
                source_cav = gt_boxes_source[box_idx] if gt_boxes_source else ref_cav_id
                box_color = get_gt_box_color(source_cav, ref_cav_id, is_bgr=True)
                box_2d = gt_box2d[box_idx]
                
                # For ego: draw filled polygon, for others: draw outline only
                if str(source_cav) == str(ref_cav_id):
                    # Draw filled polygon for ego
                    box_2d_int = box_2d.astype(np.int32)
                    cv2.fillPoly(images['original'], [box_2d_int], box_color)
                    # Also draw outline
                    for edge in edges:
                        pt1 = tuple(box_2d_int[edge[0]])
                        pt2 = tuple(box_2d_int[edge[1]])
                        if (0 <= pt1[0] < W and 0 <= pt1[1] < H and 
                            0 <= pt2[0] < W and 0 <= pt2[1] < H):
                            cv2.line(images['original'], pt1, pt2, box_color, 2)
                else:
                    # Draw outline only for other CAVs
                    for edge in edges:
                        pt1 = tuple(box_2d[edge[0]].astype(int))
                        pt2 = tuple(box_2d[edge[1]].astype(int))
                        if (0 <= pt1[0] < W and 0 <= pt1[1] < H and 
                            0 <= pt2[0] < W and 0 <= pt2[1] < H):
                            cv2.line(images['original'], pt1, pt2, box_color, 2)
    
    # Draw Radar points
    if all_radar_points_dict is not None:
        ego_radar, other_radar = separate_ego_and_other(all_radar_points_dict, ref_cav_id)
        
        if ego_radar is not None:
            uv, valid_mask, depth = project_lidar_to_image(ego_radar, camera_to_lidar, 
                                                           camera_intrinsic, (H, W))
            draw_points_with_depth_coloring(images['ego_radar'], uv[valid_mask], depth[valid_mask], 
                                           'green', RADAR_POINT_RADIUS)
        
        if other_radar is not None:
            uv, valid_mask, depth = project_lidar_to_image(other_radar, camera_to_lidar, 
                                                           camera_intrinsic, (H, W))
            draw_points_with_depth_coloring(images['other_radar'], uv[valid_mask], depth[valid_mask], 
                                           'yellow', RADAR_POINT_RADIUS)
    
    return (images['original'], images['ego_lidar'], images['other_lidar'], 
            images['ego_radar'], images['other_radar'])


def visualize_pc_on_bev(all_lidar_points_dict, gt_boxes_corners, 
                        gt_range=None, gt_boxes_source=None, ref_cav_id=None, 
                        all_radar_points_dict=None, timestamp=None, display_mode='both'):
    """
    Visualize LiDAR and Radar point clouds in BEV view and return as numpy array
    
    Parameters
    ----------
    display_mode : str
        'both' - display both LiDAR and Radar points (default)
        'lidar' - display only LiDAR points
        'radar' - display only Radar points
    
    Returns:
        frame: [H, W, 3] BGR image array
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    
    # Draw Radar points (if display_mode is 'both' or 'radar')
    if display_mode in ['both', 'radar'] and all_radar_points_dict is not None:
        for cav_id, radar_points in all_radar_points_dict.items():
            # Ego uses RADAR_COLORS, others use magenta
            if str(cav_id) == str(ref_cav_id):
                color = RADAR_COLORS[0]  # First color for ego
            else:
                color = 'magenta'  # Magenta for other CAVs
            # Downsample if too many points
            if len(radar_points) > MAX_POINTS_PER_CAV:
                indices = np.random.choice(len(radar_points), MAX_POINTS_PER_CAV, replace=False)
                radar_points = radar_points[indices]
            ax.scatter(radar_points[:, 0], radar_points[:, 1], s=15.0,
                      c=color, alpha=0.1, marker='x', rasterized=True)
    
    # Draw LiDAR points (if display_mode is 'both' or 'lidar')
    if display_mode in ['both', 'lidar']:
        for cav_id, lidar_points in all_lidar_points_dict.items():
            # Ego uses red, others use blue
            if str(cav_id) == str(ref_cav_id):
                color = 'red'  # Red for ego
            else:
                color = 'blue'  # Blue for other CAVs
            
            # Downsample if too many points
            if len(lidar_points) > MAX_POINTS_PER_CAV:
                indices = np.random.choice(len(lidar_points), MAX_POINTS_PER_CAV, replace=False)
                lidar_points = lidar_points[indices]
            ax.scatter(lidar_points[:, 0], lidar_points[:, 1], s=0.01,
                      c=color, alpha=0.5, marker='o', rasterized=True)
    
    # Draw GT boxes
    if gt_boxes_corners is not None and len(gt_boxes_corners) > 0:
        for box_idx in range(len(gt_boxes_corners)):
            box_corners = gt_boxes_corners[box_idx, :4, :2]
            box_corners_closed = np.vstack([box_corners, box_corners[0]])
            
            source_cav = gt_boxes_source[box_idx] if gt_boxes_source else ref_cav_id
            box_color = get_gt_box_color(source_cav, ref_cav_id, is_bgr=False)
            
            # For boxes from ref_cav_id (ego), use filled polygon
            # For other boxes, use line plot only
            if str(source_cav) == str(ref_cav_id):
                # Draw filled polygon for ego
                ax.fill(box_corners[:, 0], box_corners[:, 1], 
                       color=box_color, alpha=0.6)
            else:
                # Draw edges as lines for other CAVs (outline only)
                ax.plot(box_corners_closed[:, 0], box_corners_closed[:, 1],
                       color=box_color, linewidth=2.0, alpha=0.9)
    
    # Set coordinate range
    if gt_range is not None:
        ax.set_xlim(gt_range[0], gt_range[3])
        ax.set_ylim(gt_range[1], gt_range[4])
    else:
        all_x = np.concatenate([pts[:, 0] for pts in all_lidar_points_dict.values()])
        all_y = np.concatenate([pts[:, 1] for pts in all_lidar_points_dict.values()])
        ax.set_xlim(all_x.min() - 10, all_x.max() + 10)
        ax.set_ylim(all_y.min() - 10, all_y.max() + 10)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    # Only show legend if enabled (disabled by default for performance)
    if SHOW_LEGEND:
        ax.legend()
    
    # Add timestamp to title
    title = 'BEV View with GT Boxes (All CAVs)'
    if timestamp is not None:
        title += f' - Timestamp: {timestamp}'
    ax.set_title(title)
    
    # Convert matplotlib figure to numpy array
    # Use tight_layout=False and bbox_inches='tight' to speed up
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    
    # Add timestamp text overlay
    if timestamp is not None:
        cv2.putText(frame, f'Timestamp: {timestamp}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame


# ==================== Data Loading Functions ====================
def load_lidar_data(scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose,
                   filter_by_image=False, raw_resolution=None, broaden_horizontal=1.5, image_list=[0, 1, 2]):
    """
    Load LiDAR data from all CAVs and transform to ego coordinate system
    
    Each CAV filters points using its own camera parameters (in its own coordinate system),
    then transforms to ego coordinate system.
    
    Parameters
    ----------
    scenario_dir : str
        Path to scenario directory
    cav_ids : list
        List of CAV IDs
    timestamp : str
        Timestamp string
    ref_cav_id : str
        Reference (ego) CAV ID
    ref_lidar_pose : np.ndarray
        Reference LiDAR pose
    filter_by_image : bool
        Whether to filter points that cannot be projected to any camera
    raw_resolution : tuple or None
        (H, W) raw image resolution for filtering
    broaden_horizontal : float
        Multiplier to enlarge image size for filtering
    image_list : list
        List of camera indices to check (e.g., [0, 1, 2, 3])
    """
    all_lidar_points = {}
    
    # Determine dataset type
    dataset_type = 'v2x-r' if 'v2x-r' in scenario_dir.split('/') else 'v2x-radar'
    
    for cav_id in cav_ids:
        lidar_path = os.path.join(scenario_dir, cav_id, f"{timestamp}.pcd")
        if not os.path.exists(lidar_path):
            if VERBOSE:
                print(f"Warning: LiDAR file not found for CAV {cav_id}, skipping...")
            continue
        
        if VERBOSE:
            print(f"\nLoading LiDAR from CAV {cav_id}")
        xyz, intensity = load_lidar_point_cloud(lidar_path)
        if xyz is None:
            continue
        
        if len(xyz) == 0:
            continue
        
        # Additional safety check (pcd_utils.read_lidar already handles nan filtering)
        valid_mask = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity).all(axis=1)
        if not valid_mask.all():
            xyz = xyz[valid_mask]
            intensity = intensity[valid_mask]
            if len(xyz) == 0:
                continue
        
        lidar_points = np.hstack([xyz, intensity]).astype(np.float32)
        
        # Load CAV parameters (needed for filtering and/or transformation)
        cav_yaml_path = os.path.join(scenario_dir, cav_id, f"{timestamp}.yaml")
        cav_params = None
        if os.path.exists(cav_yaml_path):
            cav_params = load_yaml(cav_yaml_path)
        
        # Filter points using each CAV's own camera parameters (in its own coordinate system)
        if filter_by_image and cav_params is not None:
            lidar_points = filter_lidar_points_by_image(
                lidar_points, cav_params, raw_resolution, broaden_horizontal, image_list, dataset_type)
        
        # Transform to ego coordinate system
        if cav_id != ref_cav_id and cav_params is not None:
            cav_lidar_pose = get_lidar_pose(cav_params)
            lidar_points = transform_points_to_ego(lidar_points, cav_lidar_pose, ref_lidar_pose)
        
        all_lidar_points[cav_id] = lidar_points
        if VERBOSE:
            print(f"  Loaded {len(lidar_points)} LiDAR points")
    
    return all_lidar_points


def load_radar_data(scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose):
    """Load Radar data from all CAVs and transform to ego coordinate system"""
    all_radar_points = {}
    
    for cav_id in cav_ids:
        radar_path = os.path.join(scenario_dir, cav_id, f"{timestamp}_radar.pcd")
        if not os.path.exists(radar_path):
            if VERBOSE:
                print(f"Warning: Radar file not found for CAV {cav_id}, skipping...")
            continue
        
        if VERBOSE:
            print(f"\nLoading Radar from CAV {cav_id}")
        xyz, intensity = load_radar_point_cloud(radar_path)
        if xyz is None:
            continue
        
        radar_points = np.hstack([xyz, intensity]).astype(np.float32)
        
        # First transform to each CAV's lidar coordinate system
        cav_yaml_path = os.path.join(scenario_dir, cav_id, f"{timestamp}.yaml")
        if os.path.exists(cav_yaml_path):
            cav_params = load_yaml(cav_yaml_path)
            radar_to_lidar = get_radar_to_lidar(cav_params)
            
            if radar_to_lidar is not None:
                xyz_hom = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
                xyz_lidar = (radar_to_lidar @ xyz_hom.T).T[:, :3]
                radar_points = np.hstack([xyz_lidar, intensity]).astype(np.float32)
            else:
                xyz_lidar = xyz
        
        # Then transform to ego coordinate system
        if cav_id != ref_cav_id:
            if os.path.exists(cav_yaml_path):
                cav_lidar_pose = get_lidar_pose(cav_params)
                radar_points = transform_points_to_ego(radar_points, cav_lidar_pose, ref_lidar_pose)
        
        all_radar_points[cav_id] = radar_points
        if VERBOSE:
            print(f"  Loaded {len(radar_points)} radar points")
    
    return all_radar_points


def load_gt_boxes(scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose):
    """Load GT boxes from all CAVs and transform to ego coordinate system"""
    all_gt_boxes, all_object_ids, all_sources = [], [], []
    
    if VERBOSE:
        print(f"\nLoading GT boxes from CAVs:")
    for cav_id in cav_ids:
        cav_yaml_path = os.path.join(scenario_dir, cav_id, f"{timestamp}.yaml")
        if not os.path.exists(cav_yaml_path):
            continue
        
        cav_params = load_yaml(cav_yaml_path)
        cav_lidar_pose = get_lidar_pose(cav_params)
        cav_gt_boxes, cav_object_ids = get_gt_boxes_in_lidar(cav_params, cav_lidar_pose)
        
        if cav_gt_boxes is None or len(cav_gt_boxes) == 0:
            if VERBOSE:
                print(f"  CAV {cav_id}: 0 GT boxes")
            continue
        
        if VERBOSE:
            print(f"  CAV {cav_id}: {len(cav_gt_boxes)} GT boxes")
        
        # Transform to ego coordinate system
        if cav_id != ref_cav_id:
            T_cav_to_ref = x1_to_x2(cav_lidar_pose, ref_lidar_pose)
            cav_gt_boxes = project_box3d_simple(cav_gt_boxes, T_cav_to_ref)
        
        all_gt_boxes.append(cav_gt_boxes)
        all_object_ids.extend(cav_object_ids)
        all_sources.extend([cav_id] * len(cav_gt_boxes))
    
    if all_gt_boxes:
        unified_boxes = np.vstack(all_gt_boxes)
        if VERBOSE:
            print(f"  Before deduplication: {len(unified_boxes)} GT boxes")
        
        # Deduplicate boxes
        deduplicated_boxes, deduplicated_sources, _ = deduplicate_gt_boxes(
            unified_boxes, all_sources, distance_threshold=1.0, size_ratio_threshold=0.8)
        
        if VERBOSE:
            print(f"  After deduplication: {len(deduplicated_boxes)} GT boxes")
            print(f"\nTotal GT boxes: {len(deduplicated_boxes)}")
        # Return both: (all_boxes, all_sources) for visualization, (deduplicated_boxes, deduplicated_sources) for stats
        return unified_boxes, all_sources, deduplicated_boxes, deduplicated_sources
    
    return None, None, None, None


# ==================== Main Functions ====================
def create_image_grid(original_images, ego_lidar_images, other_lidar_images,
                      ego_radar_images, other_radar_images, timestamp=None):
    """
    Create 5xN image grid and return as numpy array where N is the number of images
    
    Returns:
        grid_img: [H*5, W*N, 3] BGR image array
    """
    num_images = len(original_images)
    if num_images == 0:
        print("Warning: No images to create grid")
        return None
    
    if len(ego_lidar_images) != num_images or len(other_lidar_images) != num_images:
        print(f"Warning: Image count mismatch. Original: {num_images}, ego_lidar: {len(ego_lidar_images)}, other_lidar: {len(other_lidar_images)}")
        return None
    
    H, W = original_images[0].shape[:2]
    grid_img = np.zeros((H * 5, W * num_images, 3), dtype=np.uint8)
    
    # Fill each row
    for col_idx in range(num_images):
        grid_img[0:H, col_idx*W:(col_idx+1)*W] = original_images[col_idx]
        grid_img[H:2*H, col_idx*W:(col_idx+1)*W] = ego_lidar_images[col_idx]
        grid_img[2*H:3*H, col_idx*W:(col_idx+1)*W] = other_lidar_images[col_idx]
        if col_idx < len(ego_radar_images):
            grid_img[3*H:4*H, col_idx*W:(col_idx+1)*W] = ego_radar_images[col_idx]
        if col_idx < len(other_radar_images):
            grid_img[4*H:5*H, col_idx*W:(col_idx+1)*W] = other_radar_images[col_idx]
    
    # Add timestamp text overlay
    if timestamp is not None:
        cv2.putText(grid_img, f'Timestamp: {timestamp}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    return grid_img


def process_frame(scenario_dir, cav_ids, timestamp, ref_cav_id, 
                 filter_by_image=False, raw_resolution=None, broaden_horizontal=1.5, image_list=[0, 1, 2],
                 display_mode='both'):
    """
    Process a single timestamp and return BEV frame
    
    Returns:
        bev_frame: BEV visualization frame (BGR image array) or None
    """
    # Load ego CAV configuration
    ref_yaml_path = os.path.join(scenario_dir, ref_cav_id, f"{timestamp}.yaml")
    if not os.path.exists(ref_yaml_path):
        return None
    
    ref_params = load_yaml(ref_yaml_path)
    ref_lidar_pose = get_lidar_pose(ref_params)
    
    # Load Data
    all_gt_boxes, all_gt_boxes_source, _, _ = load_gt_boxes(
        scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose)
    
    all_lidar_points = load_lidar_data(scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose,
                                       filter_by_image, raw_resolution, broaden_horizontal, image_list)
    if not all_lidar_points:
        return None
        
    all_radar_points = load_radar_data(scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose)
    
    # Generate BEV Frame
    bev_frame = visualize_pc_on_bev(all_lidar_points, all_gt_boxes, GT_RANGE,
                        all_gt_boxes_source, ref_cav_id, all_radar_points, timestamp=timestamp,
                        display_mode=display_mode)
    
    return bev_frame


def process_frame_cam(scenario_dir, cav_ids, timestamp, ref_cav_id, image_list=[0, 1, 2], cam_scale=1.0):
    """
    Process a single timestamp and return CAM (camera grid) frame
    
    Args:
        cam_scale: Scale factor for CAM video resolution (default: 1.0). Use < 1.0 to reduce resolution for faster processing.
    
    Returns:
        cam_frame: Camera grid visualization frame (BGR image array) or None
    """
    # Load ego CAV configuration
    ref_yaml_path = os.path.join(scenario_dir, ref_cav_id, f"{timestamp}.yaml")
    if not os.path.exists(ref_yaml_path):
        return None
    
    ref_params = load_yaml(ref_yaml_path)
    ref_lidar_pose = get_lidar_pose(ref_params)
    
    # Load Data
    all_gt_boxes, all_gt_boxes_source, _, _ = load_gt_boxes(
        scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose)
    
    all_lidar_points = load_lidar_data(scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose)
    if not all_lidar_points:
        return None
        
    all_radar_points = load_radar_data(scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose)
    
    # Determine dataset type
    dataset_type = 'v2x-r' if 'v2x-r' in scenario_dir.split('/') else 'v2x-radar'
    
    # Process each camera
    ref_base_dir = os.path.dirname(ref_yaml_path)
    image_lists = {
        'original': [],
        'ego_lidar': [],
        'other_lidar': [],
        'ego_radar': [],
        'other_radar': []
    }
    
    for cam_id in image_list:
        try:
            camera_to_lidar, camera_intrinsic = get_ext_int(ref_params, cam_id, dataset_type)
        except Exception as e:
            if VERBOSE:
                print(f"Error getting camera parameters for camera{cam_id}: {e}")
            continue
        
        image_path = os.path.join(ref_base_dir, f"{timestamp}_camera{cam_id}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(ref_base_dir, f"{timestamp}_camera{cam_id}.png")
            if not os.path.exists(image_path): 
                if VERBOSE:
                    print(f"Warning: Image not found: {image_path}, skipping...")
                continue
        
        img_original, img_ego_lidar, img_other_lidar, img_ego_radar, img_other_radar = \
            visualize_projection(image_path, all_lidar_points, camera_to_lidar, camera_intrinsic,
                               all_gt_boxes, all_gt_boxes_source, ref_cav_id, all_radar_points)
        
        if img_original is not None:
            image_lists['original'].append(img_original)
            image_lists['ego_lidar'].append(img_ego_lidar)
            image_lists['other_lidar'].append(img_other_lidar)
            image_lists['ego_radar'].append(img_ego_radar if img_ego_radar is not None else img_original.copy())
            image_lists['other_radar'].append(img_other_radar if img_other_radar is not None else img_original.copy())
    
    # Create image grid
    if len(image_lists['original']) == 0:
        return None
    
    cam_frame = create_image_grid(image_lists['original'], image_lists['ego_lidar'], 
                                   image_lists['other_lidar'], image_lists['ego_radar'], 
                                   image_lists['other_radar'], timestamp=timestamp)
    
    # Scale down resolution if needed (for faster processing)
    if cam_scale < 1.0 and cam_frame is not None:
        h, w = cam_frame.shape[:2]
        new_h, new_w = int(h * cam_scale), int(w * cam_scale)
        cam_frame = cv2.resize(cam_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return cam_frame


def process_single_scenario(scenario_path, output_dir, cav_ids, ref_cav_id, fps=10, 
                            video_mode='both', image_list=[0, 1, 2], filter_by_image=False,
                            raw_resolution=None, broaden_horizontal=1.5, display_mode='both',
                            cam_scale=1.0):
    """
    Process a single scenario directory and generate video file(s)
    
    Args:
        scenario_path: Path to scenario directory (Path object or string)
        output_dir: Output directory (Path object or string)
        cav_ids: List of CAV IDs
        ref_cav_id: Reference CAV ID (ego)
        fps: Video frame rate
        video_mode: 'BEV', 'CAM', or 'both' - which video(s) to generate
        image_list: List of camera indices for CAM video generation (e.g., [0, 1, 2, 3])
    """
    scenario_path = Path(scenario_path)
    scenario_name = scenario_path.name  # e.g., "2024-05-13-22-23-14"
    
    # Extract dataset type from parent directory (e.g., "train" or "validate")
    split_folder_name = scenario_path.parent.name
    
    print(f"\n{'='*20} Processing Scenario: {scenario_name} {'='*20}")
    print(f"Video mode: {video_mode}")
    
    # 1. Determine the maximum frame number for this scenario
    ref_path = scenario_path / ref_cav_id
    if not ref_path.exists():
        print(f"Skipping {scenario_name}: Ego folder {ref_cav_id} not found.")
        return
    
    # Get all valid PCD files with numeric names and sort by frame number
    pcd_files = [x for x in ref_path.iterdir() 
                 if x.is_file() and x.suffix == '.pcd' and x.stem.isdigit()]
    
    if len(pcd_files) == 0:
        print(f"Skipping {scenario_name}: No valid PCD files found.")
        return
    
    # Sort by frame number (convert stem to int for proper numeric sorting)
    pcd_files.sort(key=lambda x: int(x.stem))
    frame_numbers = [int(x.stem) for x in pcd_files]
    
    # 2. Prepare video output path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Video filename format: train_2024-04-29-18-07-12_BEV.mp4 or _CAM.mp4
    generate_bev = video_mode in ['BEV', 'both']
    generate_cam = video_mode in ['CAM', 'both']
    
    bev_video_path = output_dir / f"{split_folder_name}_{scenario_name}_BEV.mp4" if generate_bev else None
    cam_video_path = output_dir / f"{split_folder_name}_{scenario_name}_CAM.mp4" if generate_cam else None
    
    video_bev = None
    video_cam = None
    
    # Determine timestamp format based on dataset type
    dataset_type = 'v2x-r' if 'v2x-r' in str(scenario_path).split('/') else 'v2x-radar'
    
    # 3. Process frame by frame (only existing frames)
    # Use tqdm to show progress bar, desc distinguishes different scenarios
    for frame_num in tqdm.tqdm(frame_numbers, desc=f"Scenario {scenario_name}"):
        timestamp = f"{frame_num:05d}" if dataset_type == 'v2x-radar' else f"{frame_num:06d}"
        
        # Generate BEV frame if needed
        bev_frame = None
        if generate_bev:
            bev_frame = process_frame(str(scenario_path), cav_ids, timestamp, ref_cav_id,
                                     filter_by_image, raw_resolution, broaden_horizontal, image_list,
                                     display_mode)
            if bev_frame is not None:
                # Initialize VideoWriter (only executed on first valid frame)
                if video_bev is None:
                    h_bev, w_bev = bev_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_bev = cv2.VideoWriter(str(bev_video_path), fourcc, fps, (w_bev, h_bev))
                
                # Write to video
                if video_bev is not None:
                    video_bev.write(bev_frame)
        
        # Generate CAM frame if needed
        cam_frame = None
        if generate_cam:
            cam_frame = process_frame_cam(str(scenario_path), cav_ids, timestamp, ref_cav_id, image_list, cam_scale)
            if cam_frame is not None:
                # Initialize VideoWriter (only executed on first valid frame)
                if video_cam is None:
                    h_cam, w_cam = cam_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_cam = cv2.VideoWriter(str(cam_video_path), fourcc, fps, (w_cam, h_cam))
                
                # Write to video
                if video_cam is not None:
                    video_cam.write(cam_frame)
    
    # 4. Release resources
    if video_bev: 
        video_bev.release()
    if video_cam:
        video_cam.release()
    
    print(f"Scenario {scenario_name} done.")
    if generate_bev:
        print(f"  - BEV video: {bev_video_path}")
    if generate_cam:
        print(f"  - CAM video: {cam_video_path}")


if __name__ == "__main__":
    
    # NOTE: not ready: train/2024-05-13-22-23-14   ----- 111 117 120 121 125 127 128 129 130
    # NOTE: not ready: train/2024-05-13-16-48-37-B ----- 92 94 97 99 ? still included now
    
    import argparse
    parser = argparse.ArgumentParser(description="Generate BEV video from scenario directory")
    parser.add_argument("--scenario_path", type=str, default='data/v2x-r/train/2024_06_27_10_09_59', help="Path to a single scenario directory (mutually exclusive with --scenario_dir)")
    parser.add_argument("--scenario_dir", type=str, default=None, help="Path to directory containing multiple scenario subdirectories (mutually exclusive with --scenario_path)")
    parser.add_argument("--ref_cav_id", type=str, default=None, help="Reference CAV ID (ego). If not specified, auto-selects the largest numeric CAV ID from cav_ids")
    
    parser.add_argument("--display_mode", type=str, default='both', choices=['both', 'lidar', 'radar'], help="BEV display mode: 'both' for lidar+radar, 'lidar' for lidar only, 'radar' for radar only (default: both)")
    parser.add_argument("--video_mode", type=str, default='BEV', choices=['BEV', 'CAM', 'both'], help="Video generation mode: 'BEV' for BEV video only, 'CAM' for camera grid video only, 'both' for both videos (default: both)")
    parser.add_argument("--image_list", nargs="+", type=int, default=[0], help="List of camera indices for CAM video generation (e.g., --image_list 0 1 2 3)")
    parser.add_argument("--filter_by_image", default=True, help="Filter LiDAR points that cannot be projected to any camera")
    parser.add_argument("--broaden_horizontal", type=float, default=1, help="Multiplier to enlarge image size for filtering (default: 1.5)")
    
    parser.add_argument("--cam_scale", type=float, default=0.5, help="Scale factor for CAM video resolution (default: 0.5). Use < 1.0 to reduce resolution for faster processing.")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate (default: 5)")
    parser.add_argument("--output_dir", type=str, default='docs/output_videos', help="Output directory (default: docs/output_videos)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--max_points", type=int, default=50000, help="Maximum points per CAV for rendering (default: 50000)")
    parser.add_argument("--show_legend", action="store_true", help="Show legend (slower, default: False)")
    
    args = parser.parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set raw_resolution based on dataset type
    assert args.scenario_dir is None or args.scenario_path is None, "Either --scenario_dir or --scenario_path must be specified"
    scenario_path_str = args.scenario_dir or args.scenario_path
    if 'v2x-r' in scenario_path_str: 
        raw_resolution = (600, 800)
        GT_RANGE = [-140.8, -40, -8.5, 140.8, 40, 3.5]  # Default visualization range
    if 'v2x-radar' in scenario_path_str: 
        raw_resolution = (864, 1536)
        GT_RANGE = [-102.4, -102.4, -8.5, 102.4, 102.4, 3.5]  # Default visualization range
    
    
    # Set global flags and output directory
    globals().update({'VERBOSE': args.verbose, 'MAX_POINTS_PER_CAV': args.max_points, 'SHOW_LEGEND': args.show_legend})
    output_dir = Path(project_root) / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    
    # Helper function to process a single scenario
    def process_scenario(scenario_path, is_batch=False):
        try:
            # Auto-select ref_cav_id from scenario directory if not specified
            numeric_cav_ids = [(int(f.name), f.name) for f in scenario_path.iterdir() if f.is_dir() and f.name.lstrip('-').isdigit()]
            numeric_cav_ids_exclude_RSU = [x for x in numeric_cav_ids if x[1] != '-1']
            scenario_cav_ids = [x[1] for x in numeric_cav_ids]
            scenario_ref_cav_id = min(numeric_cav_ids_exclude_RSU, key=lambda x: x[0])[1] if args.ref_cav_id is None else args.ref_cav_id
            if args.ref_cav_id is None: print(f"Auto-selected ref_cav_id: {scenario_ref_cav_id} from {scenario_path}")
            else: print(f"Using specified ref_cav_id: {scenario_ref_cav_id}")
            print(f"Processing scenario: {scenario_path}")
            print(f"Output: {output_dir}, CAV IDs: {scenario_cav_ids}, ref_cav_id: {scenario_ref_cav_id}, fps: {args.fps}")
            process_single_scenario(scenario_path, output_dir, scenario_cav_ids, scenario_ref_cav_id, 
                                   args.fps, args.video_mode, args.image_list, args.filter_by_image,
                                   raw_resolution, args.broaden_horizontal, args.display_mode,
                                   args.cam_scale)
            return True
        except Exception as e:
            print(f"{'!!!' if is_batch else ''}Error processing scenario {scenario_path.name}: {e}")
            if VERBOSE:
                import traceback
                traceback.print_exc()
            return False

    
    if args.scenario_dir:
        # Batch mode
        scenario_dir = Path(project_root) / args.scenario_dir
        scenarios = sorted([p for p in scenario_dir.iterdir() if p.is_dir()], key=lambda x: x.name)
        print(f"Found {len(scenarios)} scenarios in {scenario_dir}")
        success_count = sum(process_scenario(s, is_batch=True) for s in scenarios)
        print(f"\n{'='*60}\nBatch processing completed: {success_count}/{len(scenarios)} scenarios processed successfully")
    
    if args.scenario_path:
        # Single scenario mode
        scenario_path = Path(project_root) / args.scenario_path
        process_scenario(scenario_path)
        print("\nVideo generation completed!")
