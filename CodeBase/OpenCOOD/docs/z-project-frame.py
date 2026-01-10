#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project LiDAR and Radar point clouds to camera images and visualize

This script reads YAML configuration files and corresponding LiDAR/Radar point cloud data,
then projects the point clouds onto three camera images (camera0, camera1, camera2).

Output:
    - lidar_bev_{timestamp}.png: BEV view with all CAVs' LiDAR and Radar point clouds
    - lidar_cameras_{timestamp}.png: 5 rows x 3 columns grid image
        - Row 1: Original images + GT boxes
        - Row 2: Ego CAV LiDAR projection (red depth coloring)
        - Row 3: Other CAV LiDAR projection (blue depth coloring)
        - Row 4: Ego CAV Radar projection (green depth coloring)
        - Row 5: Other CAV Radar projection (yellow depth coloring)
"""
import yaml
import numpy as np
import cv2
import os
import sys
import argparse
import open3d as o3d

# Add project root to path to import opencood modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from opencood.utils import pcd_utils
from opencood.utils.box_utils import corner_to_center
from packages.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

# ==================== Constants ====================
LIDAR_POINT_RADIUS = 2
RADAR_POINT_RADIUS = 10

# Color configuration
LIDAR_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
RADAR_COLORS = ['cyan', 'magenta', 'magenta', 'pink', 'lightblue', 'lightgreen']
GT_BOX_COLORS_BGR = [
    (0, 255, 0),      # Green for ego
    (0, 165, 255),    # Orange
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Cyan
    (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (255, 165, 0),    # Orange Red
]
GT_BOX_COLORS_MPL = ['green', 'orange', 'magenta', 'cyan', 'blue', 'yellow', 'purple', 'red']


# ==================== Utility Functions ====================
def load_yaml(yaml_path):
    """Load YAML file"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


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

def get_radar_to_lidar(params):
    """Get transformation matrix from radar to lidar"""
    if 'radar' in params.keys(): 
        radar_coords = np.array(params["radar"]["cords"]).astype(np.float32)
    else: radar_coords = np.array(params["radar_pose"]).astype(np.float32)
    lidar_pose = get_lidar_pose(params)
    radar_to_lidar = x1_to_x2(radar_coords, lidar_pose).astype(np.float32)
    return radar_to_lidar

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


def get_gt_box_colors(all_cavs, ref_cav_id, color_palette):
    """Assign GT box colors for each CAV"""
    colors = {}
    if ref_cav_id in all_cavs:
        colors[ref_cav_id] = color_palette[0]
    
    color_idx = 1
    for cav_id in all_cavs:
        if cav_id != ref_cav_id:
            colors[cav_id] = color_palette[color_idx % len(color_palette)]
            color_idx += 1
    
    return colors


def scatter_points_with_intensity(ax, points, cav_id, point_type='lidar', idx=0, 
                                  size=0.2, size_intensity=None, marker='.', 
                                  alpha_intensity=0.8, alpha_fixed=0.6, 
                                  cmap=None, color_palette=None):
    """
    Scatter plot points on BEV view with intensity-based coloring, fallback to fixed color if no intensity.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        matplotlib axes object
    points : np.ndarray
        Point cloud data, shape [N, 3] or [N, 4], 4th dimension is intensity (optional)
    cav_id : str
        CAV ID string
    point_type : str
        Point cloud type, 'lidar' or 'radar', used to select default color list and label
    idx : int
        Color index (for fixed color fallback)
    size : float
        Point size (used when fixed color)
    size_intensity : float, optional
        Point size when using intensity coloring, if None then use size
    marker : str
        Marker type ('.' or 'x')
    alpha_intensity : float
        Transparency when using intensity coloring
    alpha_fixed : float
        Transparency when using fixed color
    cmap : matplotlib.colors.Colormap, optional
        Color map, default uses plt.cm.viridis
    color_palette : list, optional
        Fixed color list, default selects LIDAR_COLORS or RADAR_COLORS based on point_type
    
    Returns
    -------
    bool
        Whether successfully plotted (returns False if point cloud is empty or invalid)
    """
    import matplotlib.pyplot as plt
    
    if points is None or len(points) == 0:
        return False
    
    # Set default parameters
    if cmap is None:
        cmap = plt.cm.viridis
    if color_palette is None:
        color_palette = LIDAR_COLORS if point_type == 'lidar' else RADAR_COLORS
    if size_intensity is None:
        size_intensity = size
    
    # Check if intensity information exists (4th dimension)
    has_intensity = points.shape[1] >= 4
    
    if has_intensity:
        intensity = points[:, 3]
        valid_mask = np.isfinite(intensity)
        
        if not valid_mask.any():
            # All intensity values are invalid, fallback to fixed color
            has_intensity = False
        else:
            inten_valid = intensity[valid_mask]
            
            # Use robust percentile range for normalization to avoid extreme value effects
            vmin, vmax = np.percentile(inten_valid, [2, 98])
            if vmax <= vmin:
                vmin, vmax = inten_valid.min(), inten_valid.max()
            
            if vmax <= vmin:
                # All values are the same, fallback to fixed color
                has_intensity = False
            else:
                # Normalize intensity values
                inten_norm = np.clip((inten_valid - vmin) / (vmax - vmin), 0.0, 1.0)
                
                # Create color array
                colors = np.zeros((points.shape[0], 4))
                colors[valid_mask] = cmap(inten_norm)
                colors[~valid_mask] = (0, 0, 0, 0)  # Invalid points are transparent
                
                # Draw scatter plot (using intensity coloring)
                ax.scatter(
                    points[:, 0],
                    points[:, 1],
                    s=size_intensity,
                    c=colors,
                    alpha=alpha_intensity,
                    marker=marker,
                    label=f'CAV {cav_id} {point_type.capitalize()} (intensity)',
                )
                return True
    
    # Fallback to fixed color (no intensity or intensity invalid)
    if not has_intensity:
        color = color_palette[idx % len(color_palette)]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=size,
            c=color,
            alpha=alpha_fixed,
            marker=marker,
            label=f'CAV {cav_id} {point_type.capitalize()}',
        )
        return True
    
    return False


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
            all_cavs = list(all_lidar_points_dict.keys())
            gt_colors = get_gt_box_colors(all_cavs, ref_cav_id, GT_BOX_COLORS_BGR)
            
            edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                     [0, 4], [1, 5], [2, 6], [3, 7]]
            
            for box_idx in range(len(gt_box2d)):
                if not gt_box2d_mask[box_idx]:
                    continue
                
                source_cav = gt_boxes_source[box_idx] if gt_boxes_source else ref_cav_id
                box_color = gt_colors.get(source_cav, (0, 255, 0))
                box_2d = gt_box2d[box_idx]
                
                for edge in edges:
                    pt1 = tuple(box_2d[edge[0]].astype(int))
                    pt2 = tuple(box_2d[edge[1]].astype(int))
                    if (0 <= pt1[0] < W and 0 <= pt1[1] < H and 
                        0 <= pt2[0] < W and 0 <= pt2[1] < H):
                        cv2.line(images['original'], pt1, pt2, box_color, 3)
    
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


def visualize_pc_on_bev(all_lidar_points_dict, all_radar_points_dict, gt_boxes_corners, output_path, show_legend=False, 
                        gt_range=None, gt_boxes_source=None, ref_cav_id=None, 
                        scenario_dir=None, timestamp=None, cav_ids=None, ref_lidar_pose=None,
                        display_mode='both'):
    """
    Visualize LiDAR and Radar point clouds in BEV view
    
    Parameters
    ----------
    display_mode : str
        'both' - display both LiDAR and Radar points (default)
        'lidar' - display only LiDAR points
        'radar' - display only Radar points
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create mapping from CAV ID to radar color (for FOV lines)
    radar_color_map = {}
    if all_radar_points_dict is not None:
        for idx, (cav_id, radar_points) in enumerate(all_radar_points_dict.items()):
            radar_color_map[str(cav_id)] = RADAR_COLORS[idx % len(RADAR_COLORS)]
    
    # Draw Radar points (if display_mode is 'both' or 'radar')
    if display_mode in ['both', 'radar'] and all_radar_points_dict is not None:
        for idx, (cav_id, radar_points) in enumerate(all_radar_points_dict.items()):
            scatter_points_with_intensity(
                ax, radar_points, cav_id, point_type='radar', idx=idx,
                size=15.0, marker='x', color_palette=RADAR_COLORS
            )
    
    # Draw LiDAR points (if display_mode is 'both' or 'lidar')
    if display_mode in ['both', 'lidar']:
        for idx, (cav_id, lidar_points) in enumerate(all_lidar_points_dict.items()):
            # For LiDAR, use larger points (0.5) when intensity coloring, 0.2 when fixed color
            scatter_points_with_intensity(
                ax, lidar_points, cav_id, point_type='lidar', idx=idx,
                size=0.2, size_intensity=0.5, marker='.', 
                color_palette=LIDAR_COLORS
            )
    
    # Draw GT boxes
    if gt_boxes_corners is not None and len(gt_boxes_corners) > 0:
        all_cavs = list(all_lidar_points_dict.keys())
        gt_colors = get_gt_box_colors(all_cavs, ref_cav_id, GT_BOX_COLORS_MPL)
        
        # Track which CAVs have been labeled for legend
        labeled_cavs = set()
        
        for box_idx in range(len(gt_boxes_corners)):
            box_corners = gt_boxes_corners[box_idx, :4, :2]
            box_corners_closed = np.vstack([box_corners, box_corners[0]])
            
            source_cav = gt_boxes_source[box_idx] if gt_boxes_source else ref_cav_id
            box_color = gt_colors.get(source_cav, 'green')
            
            # Add label only for the first box from each CAV
            label = None
            if source_cav not in labeled_cavs:
                label = f'GT Box (Ego CAV {source_cav})' if source_cav == ref_cav_id else f'GT Box (CAV {source_cav})'
                labeled_cavs.add(source_cav)
            
            # For boxes from ref_cav_id (ego), use filled polygon
            # For other boxes, use line plot
            if source_cav == ref_cav_id:
                # Draw filled polygon with border
                ax.fill(box_corners[:, 0], box_corners[:, 1], 
                       color=box_color, alpha=0.6, label=label)
            else:
                # Draw edges as lines for other CAVs
                ax.plot(box_corners_closed[:, 0], box_corners_closed[:, 1],
                       color=box_color, linewidth=2.0, alpha=0.9, label=label)
    
    # Draw Radar HFOV lines for infra (-1) and vehicle (142)
    # Only show FOV lines when radar is displayed
    if display_mode in ['both', 'radar'] and scenario_dir is not None and timestamp is not None and cav_ids is not None and ref_lidar_pose is not None:
        for cav_id in cav_ids:  # Only draw for infra and vehicle
            if cav_id not in cav_ids:
                continue
            
            cav_yaml_path = os.path.join(scenario_dir, cav_id, f"{timestamp}.yaml")
            if not os.path.exists(cav_yaml_path):
                continue
            
            try:
                cav_params = load_yaml(cav_yaml_path)
                
                # Check if it's Infra and set HFOV threshold
                lidar_pose = get_lidar_pose(cav_params)
                t_world = x_to_world(lidar_pose)
                is_Infra = (t_world[0, 3] == 0) and (t_world[1, 3] == 0)
                # hfov_thresh = 56 if is_Infra else 50
                dataset_type = 'v2x-r' if 'v2x-r' in scenario_dir.split('/') else 'v2x-radar'
                if dataset_type == 'v2x-radar': hfov_thresh = 48.77/2 if is_Infra else 68.40/2
                if dataset_type == 'v2x-r': hfov_thresh = 50
                
                # Get CAV position in ego coordinate system
                cav_pos_world = t_world[:3, 3]
                T_world_to_ego = np.linalg.inv(x_to_world(ref_lidar_pose))
                cav_pos_ego = (T_world_to_ego @ np.concatenate([cav_pos_world, [1]]))[:3]
                
                # Get radar to lidar transformation (same as radar point cloud conversion)
                radar_to_lidar = get_radar_to_lidar(cav_params)
                if radar_to_lidar is None:
                    continue
                
                # Build FOV boundary direction vectors in radar coordinate system
                # Infra: x-axis points forward; Vehicle: y-axis points forward

                max_range = 100.0
                if (dataset_type == 'v2x-radar' and is_Infra) or (dataset_type == 'v2x-r'):
                    # Infra: FOV relative to x-axis (forward direction)
                    left_dir_radar = np.array([max_range * np.cos(np.radians(hfov_thresh)), max_range * np.sin(np.radians(hfov_thresh)), 0.0, 0.0])
                    right_dir_radar = np.array([max_range * np.cos(np.radians(-hfov_thresh)), max_range * np.sin(np.radians(-hfov_thresh)), 0.0, 0.0])
                if dataset_type == 'v2x-radar' and (not is_Infra):
                    # Vehicle: FOV relative to y-axis (forward direction)
                    left_dir_radar = np.array([max_range * np.sin(np.radians(hfov_thresh)), max_range * np.cos(np.radians(hfov_thresh)), 0.0, 0.0])
                    right_dir_radar = np.array([max_range * np.sin(np.radians(-hfov_thresh)), max_range * np.cos(np.radians(-hfov_thresh)), 0.0, 0.0])

                # Transform FOV direction vectors: radar -> lidar -> ego (same as radar point cloud)
                left_dir_lidar = (radar_to_lidar @ left_dir_radar)[:3]
                right_dir_lidar = (radar_to_lidar @ right_dir_radar)[:3]
                
                if cav_id != ref_cav_id:
                    # Transform from CAV's lidar coordinate to ego (ref_cav_id) lidar coordinate
                    T_cav_lidar_to_ego = x1_to_x2(lidar_pose, ref_lidar_pose)
                    left_dir_ego = (T_cav_lidar_to_ego @ np.concatenate([left_dir_lidar, [0]]))[:3]
                    right_dir_ego = (T_cav_lidar_to_ego @ np.concatenate([right_dir_lidar, [0]]))[:3]
                else:
                    # If this CAV is ref_cav_id, lidar coordinate is already ego coordinate
                    left_dir_ego = left_dir_lidar
                    right_dir_ego = right_dir_lidar
                
                # Calculate line endpoints (FOV lines start from CAV position)
                left_end = cav_pos_ego + left_dir_ego
                right_end = cav_pos_ego + right_dir_ego
                
                # Draw FOV lines with color matching radar points
                cav_label = 'Infra' if is_Infra else 'Vehicle'
                line_color = radar_color_map.get(str(cav_id), 'gray')
                ax.plot([cav_pos_ego[0], left_end[0]], [cav_pos_ego[1], left_end[1]],
                       color=line_color, linewidth=2.0, linestyle='--', alpha=0.7,
                       label=f'Radar HFOV {cav_label} CAV {cav_id} (±{hfov_thresh}°)')
                ax.plot([cav_pos_ego[0], right_end[0]], [cav_pos_ego[1], right_end[1]],
                       color=line_color, linewidth=2.0, linestyle='--', alpha=0.7)
                
            except Exception as e:
                print(f"Warning: Could not draw Radar HFOV for CAV {cav_id}: {e}")
                continue
    
    # Set coordinate range
    if gt_range is not None:
        ax.set_xlim(gt_range[0], gt_range[3])
        ax.set_ylim(gt_range[1], gt_range[4])
    else:
        all_x_list = [pts[:, 0] for pts in all_lidar_points_dict.values() if len(pts) > 0]
        all_y_list = [pts[:, 1] for pts in all_lidar_points_dict.values() if len(pts) > 0]
        
        if all_x_list:
            all_x = np.concatenate(all_x_list)
            all_y = np.concatenate(all_y_list)
            ax.set_xlim(all_x.min() - 10, all_x.max() + 10)
            ax.set_ylim(all_y.min() - 10, all_y.max() + 10)
        else:
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    if show_legend:
        ax.legend()
    ax.set_title('BEV View with GT Boxes (All CAVs)')
    
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved BEV visualization to {output_path}")


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
    
    Returns
    -------
    all_lidar_points : dict
        Dictionary mapping CAV IDs to LiDAR point clouds (in ego coordinate system)
    """
    all_lidar_points = {}
    
    for cav_id in cav_ids:
        lidar_path = os.path.join(scenario_dir, cav_id, f"{timestamp}.pcd")
        if not os.path.exists(lidar_path):
            print(f"Warning: LiDAR file not found for CAV {cav_id}, skipping...")
            continue
        
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
            type = 'v2x-r' if 'v2x-r' in scenario_dir.split('/') else 'v2x-radar'
            lidar_points = filter_lidar_points_by_image(
                lidar_points, cav_params, raw_resolution, broaden_horizontal, image_list, type)
        
        # Transform to ego coordinate system
        if cav_id != ref_cav_id and cav_params is not None:
            cav_lidar_pose = get_lidar_pose(cav_params)
            lidar_points = transform_points_to_ego(lidar_points, cav_lidar_pose, ref_lidar_pose)
        
        all_lidar_points[cav_id] = lidar_points[:,:3]
        print(f"  Loaded {len(lidar_points)} LiDAR points")
    
    return all_lidar_points


def load_radar_data(scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose):
    """Load Radar data from all CAVs and transform to ego coordinate system"""
    all_radar_points = {}
    
    for cav_id in cav_ids:
        radar_path = os.path.join(scenario_dir, cav_id, f"{timestamp}_radar.pcd")
        if not os.path.exists(radar_path):
            print(f"Warning: Radar file not found for CAV {cav_id}, skipping...")
            continue
        
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
        
        all_radar_points[cav_id] = radar_points[:,:3]
        print(f"  Loaded {len(radar_points)} radar points")
    
    return all_radar_points


def load_gt_boxes(scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose):
    """Load GT boxes from all CAVs and transform to ego coordinate system"""
    all_gt_boxes, all_object_ids, all_sources = [], [], []
    
    print(f"\nLoading GT boxes from CAVs:")
    for cav_id in cav_ids:
        cav_yaml_path = os.path.join(scenario_dir, cav_id, f"{timestamp}.yaml")
        if not os.path.exists(cav_yaml_path):
            continue
        
        cav_params = load_yaml(cav_yaml_path)
        cav_lidar_pose = get_lidar_pose(cav_params)
        cav_gt_boxes, cav_object_ids = get_gt_boxes_in_lidar(cav_params, cav_lidar_pose)
        
        if cav_gt_boxes is None or len(cav_gt_boxes) == 0:
            print(f"  CAV {cav_id}: 0 GT boxes")
            continue
        
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
        print(f"  Before deduplication: {len(unified_boxes)} GT boxes")
        
        # Deduplicate boxes
        deduplicated_boxes, deduplicated_sources, _ = deduplicate_gt_boxes(
            unified_boxes, all_sources, distance_threshold=1.0, size_ratio_threshold=0.8)
        
        print(f"  After deduplication: {len(deduplicated_boxes)} GT boxes")
        print(f"\nTotal GT boxes: {len(deduplicated_boxes)}")
        # Return both: (all_boxes, all_sources) for visualization, (deduplicated_boxes, deduplicated_sources) for stats
        return unified_boxes, all_sources, deduplicated_boxes, deduplicated_sources
    
    return None, None, None, None


def filter_gt_boxes_by_points(gt_boxes, gt_boxes_source, lidar_points_dict, radar_points_dict, 
                              filter_mode='both'):
    """
    Filter GT boxes based on whether they contain LiDAR/Radar points
    
    Parameters
    ----------
    gt_boxes : np.ndarray
        GT boxes with shape [N, 8, 3] (8 corners)
    gt_boxes_source : list
        List of source CAV IDs for each box, length N
    lidar_points_dict : dict
        Dictionary mapping CAV ID to LiDAR points [M, 3] or [M, 4]
    radar_points_dict : dict
        Dictionary mapping CAV ID to Radar points [M, 3] or [M, 4]
    filter_mode : str
        'lidar': keep boxes that contain LiDAR points
        'radar': keep boxes that contain Radar points
        'both': keep boxes that contain LiDAR OR Radar points
    
    Returns
    -------
    filtered_boxes : np.ndarray
        Filtered GT boxes [M, 8, 3], M <= N
    filtered_sources : list
        Filtered source CAV IDs, length M
    empty_mask_num : int
        Number of boxes filtered out
    """
    if gt_boxes is None or len(gt_boxes) == 0:
        return gt_boxes, gt_boxes_source, 0
    
    # Convert boxes from 8 corners to [x, y, z, dx, dy, dz, heading] format
    # corner_to_center expects [N, 8, 3] and returns [N, 7] with order='hwl'
    # We need to convert to [x, y, z, l, w, h, heading] for points_in_boxes_cpu
    boxes_7d = corner_to_center(gt_boxes, order='hwl')  # [N, 7] with order hwl
    # Convert to lwh order: [x, y, z, l, w, h, heading] -> [x, y, z, h, w, l, heading]
    # points_in_boxes_cpu expects [x, y, z, dx, dy, dz, heading]
    boxes_7d_reordered = boxes_7d[:, [0, 1, 2, 5, 4, 3, 6]]  # [x, y, z, l, w, h, heading]
    # boxes_7d_reordered[:, [5, 4, 3]] = boxes_7d_reordered[:, [5, 4, 3]] * 1.5 # scale up the box size to 1.5x
    
    N = len(gt_boxes)
    valid_mask = np.zeros(N, dtype=bool)
    empty_mask_num = 0
    
    # Stack all points from all CAVs
    all_lidar_points = []
    all_radar_points = []
    
    for cav_id in lidar_points_dict.keys():
        lidar_pts = lidar_points_dict[cav_id]
        if lidar_pts is not None and len(lidar_pts) > 0:
            # Extract xyz coordinates
            if lidar_pts.shape[1] >= 3:
                all_lidar_points.append(lidar_pts[:, :3])
    
    for cav_id in radar_points_dict.keys():
        radar_pts = radar_points_dict[cav_id]
        if radar_pts is not None and len(radar_pts) > 0:
            # Extract xyz coordinates
            if radar_pts.shape[1] >= 3:
                all_radar_points.append(radar_pts[:, :3])
    
    # Stack all points
    if len(all_lidar_points) > 0:
        stacked_lidar = np.vstack(all_lidar_points)
    else:
        stacked_lidar = np.empty((0, 3), dtype=np.float32)
    
    if len(all_radar_points) > 0:
        stacked_radar = np.vstack(all_radar_points)
    else:
        stacked_radar = np.empty((0, 3), dtype=np.float32)
    
    # Check each box
    if filter_mode == 'lidar':
        if len(stacked_lidar) > 0:
            point_indices = points_in_boxes_cpu(stacked_lidar, boxes_7d_reordered)
            # point_indices: [N, num_points], each row indicates which points are in that box
            valid_mask = point_indices.sum(axis=1) > 0
        else:
            valid_mask = np.zeros(N, dtype=bool)
    
    elif filter_mode == 'radar':
        if len(stacked_radar) > 0:
            r_point_indices = points_in_boxes_cpu(stacked_radar, boxes_7d_reordered)
            valid_mask = r_point_indices.sum(axis=1) > 0
        else:
            valid_mask = np.zeros(N, dtype=bool)
    
    elif filter_mode == 'both':
        lidar_mask = np.zeros(N, dtype=bool)
        radar_mask = np.zeros(N, dtype=bool)
        
        if len(stacked_lidar) > 0:
            point_indices = points_in_boxes_cpu(stacked_lidar, boxes_7d_reordered)
            lidar_mask = point_indices.sum(axis=1) > 0
        
        if len(stacked_radar) > 0:
            r_point_indices = points_in_boxes_cpu(stacked_radar, boxes_7d_reordered)
            radar_mask = r_point_indices.sum(axis=1) > 0
        
        valid_mask = lidar_mask | radar_mask
    
    else:
        raise ValueError(f"Invalid filter_mode: {filter_mode}. Must be 'lidar', 'radar', or 'both'")
    
    empty_mask_num = (~valid_mask).sum()
    
    # Filter boxes
    filtered_boxes = gt_boxes[valid_mask]
    filtered_sources = [gt_boxes_source[i] for i in range(N) if valid_mask[i]]
    
    print(f"\nGT boxes filtering (mode: {filter_mode}):")
    print(f"  Before filtering: {N} boxes")
    print(f"  After filtering: {len(filtered_boxes)} boxes")
    print(f"  Filtered out: {empty_mask_num} boxes")
    
    return filtered_boxes, filtered_sources, empty_mask_num


# ==================== Main Functions ====================
def create_image_grid(original_images, ego_lidar_images, other_lidar_images,
                      ego_radar_images, other_radar_images, timestamp, output_path):
    """Create 5xN image grid where N is the number of images"""
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
    
    cv2.imwrite(output_path, grid_img)
    size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved combined image to {output_path} ({size:.2f} MB)")
    print(f"  Layout: 5 rows x {num_images} columns")
    print("  Row 1: Original images with GT boxes")
    print("  Row 2: Ego CAV LiDAR (red scale)")
    print("  Row 3: Other CAV LiDAR (blue scale)")
    print("  Row 4: Ego CAV Radar (green scale)")
    print("  Row 5: Other CAV Radar (yellow scale)")
    
    return output_path


def main(scenario_dir, cav_ids, timestamp, ref_cav_id, output_dir=None,
         filter_by_image=False, raw_resolution=None, broaden_horizontal=1.5, image_list=[0, 1, 2],
         display_mode='both', filter_gt_by_points=None, show_legend=False):
    """
    Main function
    
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
    output_dir : str or None
        Output directory path
    filter_by_image : bool
        Whether to filter LiDAR points that cannot be projected to any camera
    raw_resolution : tuple or None
        (H, W) raw image resolution for filtering
    broaden_horizontal : float
        Multiplier to enlarge image size for filtering
    image_list : list
        List of camera indices to check (e.g., [0, 1, 2, 3])
        Points visible in any specified camera will be kept
    filter_gt_by_points : str or None
        Filter GT boxes by points: 'lidar' (LiDAR points only), 
        'radar' (Radar points only), 'both' (either LiDAR or Radar points), 
        or None (no filtering). This filtering is performed after filter_by_image.
    """
    # Set output directory to docs/ if not specified
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    # Extract scenario name from scenario_dir (e.g., "2024-05-13-22-23-14")
    scenario_name = os.path.basename(scenario_dir)
    # Extract dataset type from parent directory (e.g., "train" or "validate")
    dataset_splits = os.path.basename(os.path.dirname(scenario_dir))
    
    # Generate output filenames
    bev_filename = f"BEV_{dataset_splits}_{scenario_name}_{timestamp}.png"
    cam_filename = f"CAM_{dataset_splits}_{scenario_name}_{timestamp}.png"
    bev_output_path = os.path.join(output_dir, bev_filename)
    cam_output_path = os.path.join(output_dir, cam_filename)
    
    # Load ego CAV configuration
    ref_yaml_path = os.path.join(scenario_dir, ref_cav_id, f"{timestamp}.yaml")
    if not os.path.exists(ref_yaml_path):
        print(f"Error: ego YAML file not found: {ref_yaml_path}")
        return
    
    print(f"Loading ego CAV {ref_cav_id} YAML")
    ref_params = load_yaml(ref_yaml_path)
    ref_lidar_pose = get_lidar_pose(ref_params)
    print(f"Ego CAV {ref_cav_id} pose: {ref_lidar_pose}")
    
    # Load GT boxes
    all_gt_boxes, all_gt_boxes_source, unified_gt_boxes, unified_gt_boxes_source = load_gt_boxes(
        scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose)
    
    # Load LiDAR data
    # Each CAV filters points using its own camera parameters, then transforms to ego coordinate system
    all_lidar_points = load_lidar_data(
        scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose,
        filter_by_image=filter_by_image,
        raw_resolution=raw_resolution, broaden_horizontal=broaden_horizontal, image_list=image_list)
    if not all_lidar_points:
        print("Error: No LiDAR point clouds loaded")
        return
    
    # Load Radar data
    all_radar_points = load_radar_data(scenario_dir, cav_ids, timestamp, ref_cav_id, ref_lidar_pose)
    
    # Filter GT boxes based on point cloud data (after filter_by_image)
    if filter_gt_by_points and filter_gt_by_points in ['lidar', 'radar', 'both']:
        unified_gt_boxes, unified_gt_boxes_source, _ = filter_gt_boxes_by_points(
            unified_gt_boxes, unified_gt_boxes_source, 
            all_lidar_points, all_radar_points, 
            filter_mode=filter_gt_by_points)
        # Also filter all_gt_boxes for visualization
        all_gt_boxes, all_gt_boxes_source, _ = filter_gt_boxes_by_points(
            all_gt_boxes, all_gt_boxes_source,
            all_lidar_points, all_radar_points,
            filter_mode=filter_gt_by_points)
    
    # BEV visualization (use all boxes before deduplication to show all CAV detections)
    visualize_pc_on_bev(all_lidar_points, all_radar_points, all_gt_boxes, bev_output_path, show_legend=show_legend,
                        gt_range=GT_RANGE, gt_boxes_source=all_gt_boxes_source, ref_cav_id=ref_cav_id,
                        scenario_dir=scenario_dir, timestamp=timestamp, cav_ids=cav_ids, ref_lidar_pose=ref_lidar_pose,
                        display_mode=display_mode)
    
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
            dataset_type = 'v2x-r' if 'v2x-r' in scenario_dir.split('/') else 'v2x-radar'
            camera_to_lidar, camera_intrinsic = get_ext_int(ref_params, cam_id, dataset_type)
        except Exception as e:
            print(f"Error getting camera parameters for camera{cam_id}: {e}")
            continue
        
        image_path = os.path.join(ref_base_dir, f"{timestamp}_camera{cam_id}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(ref_base_dir, f"{timestamp}_camera{cam_id}.png")
            if not os.path.exists(image_path): 
                print(f"Warning: Image not found: {image_path}, skipping...")
                continue
        
        print(f"\nProcessing camera{cam_id}")
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
    create_image_grid(image_lists['original'], image_lists['ego_lidar'], image_lists['other_lidar'],
                     image_lists['ego_radar'], image_lists['other_radar'], timestamp, cam_output_path)
    
    print("\nDone! Generated files:")
    print(f"  - {bev_output_path}")
    if os.path.exists(cam_output_path):
        size = os.path.getsize(cam_output_path) / (1024 * 1024)
        print(f"  - {cam_output_path} ({size:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project LiDAR and Radar point clouds to camera images and visualize")
    parser.add_argument("--scenario_dir", type=str,  default="data/v2x-radar/v2x-radar-c/validate/2024-05-07-18-42-36", help="Path to scenario directory")
    parser.add_argument("--cav_ids", nargs="+", default=["-1", "142"], help="List of CAV IDs (default: ['142', '-1'])")
    parser.add_argument("--timestamp", type=str, default="00000", help="Timestamp string (default: '00079')")
    parser.add_argument("--ref_cav_id", type=str, default="142", help="Reference (ego) CAV ID (default: '142')")
    parser.add_argument("--output_dir", type=str, default='docs/visualization', help="Output directory (default: docs/)")
    parser.add_argument("--filter_by_image", default=True, help="Filter LiDAR points that cannot be projected to any camera")
    parser.add_argument("--broaden_horizontal", type=float, default=1, help="Multiplier to enlarge image size for filtering (default: 1.5)")
    parser.add_argument("--image_list", nargs="+", type=int, default=[1], help="List of camera indices to check (e.g., --image_list 0 1 2 3). Points visible in any specified camera will be kept.")
    parser.add_argument("--display_mode", type=str, default='both', choices=['both', 'lidar', 'radar'], help="BEV display mode: 'both' for lidar+radar, 'lidar' for lidar only, 'radar' for radar only (default: both)")
    parser.add_argument("--filter_gt_by_points", type=str, default=None, choices=['lidar', 'radar', 'both'], help="Filter GT boxes by points: 'lidar' (keep boxes with LiDAR points), 'radar' (keep boxes with Radar points), 'both' (keep boxes with LiDAR OR Radar points). Applied after filter_by_image.")
    parser.add_argument("--show_legend", default=False, help="Show legend in BEV visualization (default: False)")
    
    args = parser.parse_args()
    
    # Get the project root directory (parent of docs/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scenario_dir = args.scenario_dir
    assert args.ref_cav_id in args.cav_ids, "ego CAV must be in the list of CAVs"
    
    # Convert raw_resolution to tuple if provided
    if 'v2x-r' in args.scenario_dir.split('/'): 
        args.raw_resolution = (600, 800)
        GT_RANGE = [-140.8, -40, -8.5, 140.8, 40, 3.5]  # Default visualization range
    if 'v2x-radar' in args.scenario_dir.split('/'): 
        args.raw_resolution = (864, 1536)
        GT_RANGE = [-102.4, -102.4, -8.5, 102.4, 102.4, 3.5]  # Default visualization range
    
    main(scenario_dir, args.cav_ids, args.timestamp, args.ref_cav_id, 
         output_dir=args.output_dir, filter_by_image=args.filter_by_image,
         raw_resolution=args.raw_resolution, broaden_horizontal=args.broaden_horizontal, image_list=args.image_list,
         display_mode=args.display_mode, filter_gt_by_points=args.filter_gt_by_points, show_legend=args.show_legend)
