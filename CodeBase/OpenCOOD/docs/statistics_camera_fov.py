"""
统计 data/v2x-r/train 目录下所有相机的视场角
"""

import os
import yaml
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json

# 自定义YAML loader以支持numpy类型
class NumpyLoader(yaml.SafeLoader):
    """自定义YAML loader，支持numpy标量类型"""
    pass

def numpy_scalar_constructor(loader, node):
    """处理numpy标量的构造函数"""
    # numpy标量通常是一个序列：[类型, 值]
    if isinstance(node, yaml.SequenceNode):
        sequence = loader.construct_sequence(node)
        if len(sequence) >= 2:
            # 第二个元素通常是实际的值
            value = sequence[1]
            # 尝试转换为Python原生类型
            if isinstance(value, (np.floating, float)):
                return float(value)
            elif isinstance(value, (np.integer, int)):
                return int(value)
            return value
    # 如果不是序列，尝试直接构造标量
    try:
        value = loader.construct_scalar(node)
        if isinstance(value, str):
            # 尝试转换为数字
            if '.' in value:
                return float(value)
            else:
                return int(value)
        return value
    except:
        return None

def numpy_array_constructor(loader, node):
    """处理numpy数组的构造函数"""
    if isinstance(node, yaml.SequenceNode):
        sequence = loader.construct_sequence(node)
        # 尝试构造numpy数组
        try:
            return np.array(sequence)
        except:
            return sequence
    else:
        return loader.construct_scalar(node)

# 注册numpy相关的构造函数
NumpyLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', numpy_scalar_constructor)
NumpyLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.ndarray', numpy_array_constructor)
NumpyLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.core.multiarray._reconstruct', numpy_array_constructor)

def calculate_camera_fov(intrinsic_matrix, image_width, image_height):
    """
    从相机内参矩阵计算相机的视场角（Field of View, FOV）
    
    参数:
    ----------
    intrinsic_matrix : np.ndarray
        相机内参矩阵，形状为 [3, 3]
    image_width : int
        图像宽度（像素）
    image_height : int
        图像高度（像素）
    
    返回:
    ----------
    fov_dict : dict
        包含视场角信息
    """
    # 提取内参
    fx = intrinsic_matrix[0, 0]  # x方向焦距
    fy = intrinsic_matrix[1, 1]  # y方向焦距
    cx = intrinsic_matrix[0, 2]  # x方向主点
    cy = intrinsic_matrix[1, 2]  # y方向主点
    
    # 计算水平视场角 (Horizontal FOV)
    hfov_rad = 2 * np.arctan(image_width / (2 * fx))
    hfov_deg = np.degrees(hfov_rad)
    
    # 计算垂直视场角 (Vertical FOV)
    vfov_rad = 2 * np.arctan(image_height / (2 * fy))
    vfov_deg = np.degrees(vfov_rad)
    
    # 计算对角线视场角
    diagonal = np.sqrt(image_width**2 + image_height**2)
    f_diagonal = np.sqrt(fx**2 + fy**2)
    diagonal_fov_rad = 2 * np.arctan(diagonal / (2 * f_diagonal))
    diagonal_fov_deg = np.degrees(diagonal_fov_rad)
    
    return {
        'hfov_deg': hfov_deg,
        'vfov_deg': vfov_deg,
        'diagonal_fov_deg': diagonal_fov_deg,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }


def find_all_yaml_files(root_dir):
    """递归查找所有yaml文件"""
    yaml_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml'):
                yaml_files.append(os.path.join(root, file))
    return yaml_files


def load_yaml_file(yaml_path):
    """加载yaml文件，支持numpy类型"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            # 先尝试使用safe_load（更安全）
            try:
                data = yaml.safe_load(f)
            except (yaml.constructor.ConstructorError, yaml.YAMLError) as e:
                # 如果safe_load失败，尝试使用FullLoader（可以处理更多类型）
                try:
                    f.seek(0)  # 重置文件指针
                    data = yaml.load(f, Loader=yaml.FullLoader)
                except:
                    # 如果还是失败，使用自定义loader
                    f.seek(0)
                    data = yaml.load(f, Loader=NumpyLoader)
        return data
    except Exception:
        # 静默处理错误，不打印（因为错误已经在tqdm中显示）
        return None


def extract_camera_intrinsics(params):
    """从params中提取所有相机的内参"""
    cameras = {}
    for key in params.keys():
        if key.startswith('camera') and isinstance(params[key], dict):
            if 'intrinsic' in params[key]:
                try:
                    intrinsic = np.array(params[key]['intrinsic'], dtype=np.float32)
                    if intrinsic.shape == (3, 3):
                        cameras[key] = intrinsic
                except Exception as e:
                    print(f"Error extracting intrinsic for {key}: {e}")
    return cameras


def extract_cav_id_from_path(yaml_path, data_dir):
    """
    从yaml文件路径中提取CAV ID
    
    对于v2x-radar: 路径格式为 .../scenario/-1/xxx.yaml 或 .../scenario/142/xxx.yaml
    对于v2x-r: 路径格式为 .../scenario/CAV_ID/xxx.yaml
    """
    # 获取相对于data_dir的路径
    rel_path = os.path.relpath(yaml_path, data_dir)
    path_parts = rel_path.split(os.sep)
    
    # 检查是否是v2x-radar格式（包含-1或142）
    for part in path_parts:
        if part == '-1':
            return '-1'  # Infra
        elif part == '142':
            return '142'  # Vehicle
    
    # 对于v2x-r，CAV ID通常是数字目录名
    # 路径格式: scenario_folder/CAV_ID/xxx.yaml
    if len(path_parts) >= 2:
        # 尝试将倒数第二个部分作为CAV ID
        cav_id_candidate = path_parts[-2]
        # 检查是否是数字（v2x-r的CAV ID通常是数字）
        if cav_id_candidate.isdigit() or cav_id_candidate.startswith('-'):
            return cav_id_candidate
    
    return 'unknown'


def statistics_camera_fov(data_dir, image_width=800, image_height=600):
    """
    统计数据集中所有相机的视场角
    
    参数:
    ----------
    data_dir : str
        数据目录路径
    image_width : int
        图像宽度（像素），默认800
    image_height : int
        图像高度（像素），默认600
    """
    print(f"开始扫描目录: {data_dir}")
    yaml_files = find_all_yaml_files(data_dir)
    print(f"找到 {len(yaml_files)} 个yaml文件")
    
    # 检测数据集类型
    is_v2x_radar = 'v2x-radar' in data_dir
    is_v2x_r = 'v2x-r' in data_dir
    
    if is_v2x_radar:
        print("检测到 v2x-radar 数据集")
        print("将分别统计 -1 (Infra) 和 142 (Vehicle)")
        print(f"使用图像尺寸: {image_width} x {image_height}")
    elif is_v2x_r:
        print("检测到 v2x-r 数据集")
        print("将按CAV ID分别统计")
        print(f"使用图像尺寸: {image_width} x {image_height}")
    
    # 统计信息：按(CAV_ID, camera_id)分组
    all_fovs = defaultdict(lambda: defaultdict(list))  # {cav_id: {camera_id: [fovs]}}
    all_intrinsics = defaultdict(lambda: defaultdict(list))  # {cav_id: {camera_id: [intrinsics]}}
    
    # 处理每个yaml文件
    processed_count = 0
    error_count = 0
    
    for yaml_path in tqdm(yaml_files, desc="处理yaml文件"):
        params = load_yaml_file(yaml_path)
        if params is None:
            error_count += 1
            continue
        
        cameras = extract_camera_intrinsics(params)
        if not cameras:
            continue
        
        # 提取CAV ID
        cav_id = extract_cav_id_from_path(yaml_path, data_dir)
        
        processed_count += 1
        
        # 计算每个相机的FOV
        for cam_id, intrinsic in cameras.items():
            fov = calculate_camera_fov(intrinsic, image_width, image_height)
            all_fovs[cav_id][cam_id].append(fov)
            all_intrinsics[cav_id][cam_id].append(intrinsic)
    
    print(f"\n处理完成:")
    print(f"  成功处理: {processed_count} 个文件")
    print(f"  错误文件: {error_count} 个")
    print(f"  发现CAV数量: {len(all_fovs)} 个")
    
    # 统计结果
    print("\n" + "="*80)
    print("相机视场角统计结果（按CAV ID分组）")
    print("="*80)
    
    results = {}
    
    # 按CAV ID排序（v2x-radar优先显示-1和142）
    def sort_cav_id(cav_id):
        if cav_id == '-1':
            return (0, cav_id)
        elif cav_id == '142':
            return (1, cav_id)
        elif cav_id.isdigit():
            return (2, int(cav_id))
        else:
            return (3, cav_id)
    
    sorted_cav_ids = sorted(all_fovs.keys(), key=sort_cav_id)
    
    for cav_id in sorted_cav_ids:
        cav_fovs = all_fovs[cav_id]
        cav_intrinsics = all_intrinsics[cav_id]
        
        # 显示CAV ID信息
        cav_type = ""
        if cav_id == '-1':
            cav_type = " (Infra)"
        elif cav_id == '142':
            cav_type = " (Vehicle)"
        
        print(f"\n{'='*80}")
        print(f"CAV ID: {cav_id}{cav_type}")
        print(f"{'='*80}")
        
        results[cav_id] = {}
        
        # 按相机ID排序
        for cam_id in sorted(cav_fovs.keys()):
            fovs = cav_fovs[cam_id]
            hfovs = [f['hfov_deg'] for f in fovs]
            vfovs = [f['vfov_deg'] for f in fovs]
            diag_fovs = [f['diagonal_fov_deg'] for f in fovs]
            fxs = [f['fx'] for f in fovs]
            fys = [f['fy'] for f in fovs]
            
            # 检查内参是否一致
            intrinsics = cav_intrinsics[cam_id]
            unique_intrinsics = len(set([str(intr.tobytes()) for intr in intrinsics]))
            
            results[cav_id][cam_id] = {
                'count': len(fovs),
                'hfov': {
                    'mean': np.mean(hfovs),
                    'std': np.std(hfovs),
                    'min': np.min(hfovs),
                    'max': np.max(hfovs)
                },
                'vfov': {
                    'mean': np.mean(vfovs),
                    'std': np.std(vfovs),
                    'min': np.min(vfovs),
                    'max': np.max(vfovs)
                },
                'diagonal_fov': {
                    'mean': np.mean(diag_fovs),
                    'std': np.std(diag_fovs),
                    'min': np.min(diag_fovs),
                    'max': np.max(diag_fovs)
                },
                'fx': {
                    'mean': np.mean(fxs),
                    'std': np.std(fxs),
                    'min': np.min(fxs),
                    'max': np.max(fxs)
                },
                'fy': {
                    'mean': np.mean(fys),
                    'std': np.std(fys),
                    'min': np.min(fys),
                    'max': np.max(fys)
                },
                'unique_intrinsics': unique_intrinsics
            }
            
            print(f"\n  {cam_id}:")
            print(f"    样本数量: {len(fovs)}")
            print(f"    唯一内参数量: {unique_intrinsics}")
            print(f"    水平视场角 (HFOV):")
            print(f"      平均值: {results[cav_id][cam_id]['hfov']['mean']:.2f}°")
            print(f"      标准差: {results[cav_id][cam_id]['hfov']['std']:.4f}°")
            print(f"      范围: [{results[cav_id][cam_id]['hfov']['min']:.2f}°, {results[cav_id][cam_id]['hfov']['max']:.2f}°]")
            print(f"    垂直视场角 (VFOV):")
            print(f"      平均值: {results[cav_id][cam_id]['vfov']['mean']:.2f}°")
            print(f"      标准差: {results[cav_id][cam_id]['vfov']['std']:.4f}°")
            print(f"      范围: [{results[cav_id][cam_id]['vfov']['min']:.2f}°, {results[cav_id][cam_id]['vfov']['max']:.2f}°]")
            print(f"    对角线视场角:")
            print(f"      平均值: {results[cav_id][cam_id]['diagonal_fov']['mean']:.2f}°")
            print(f"    焦距 (fx): {results[cav_id][cam_id]['fx']['mean']:.2f} ± {results[cav_id][cam_id]['fx']['std']:.4f} 像素")
            print(f"    焦距 (fy): {results[cav_id][cam_id]['fy']['mean']:.2f} ± {results[cav_id][cam_id]['fy']['std']:.4f} 像素")
    
    # 保存结果到JSON文件
    output_file = os.path.join(os.path.dirname(data_dir), 'camera_fov_statistics.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型
        json_results = {}
        for cav_id, cav_stats in results.items():
            json_results[cav_id] = {}
            for cam_id, stats in cav_stats.items():
                json_results[cav_id][cam_id] = {}
                for key, value in stats.items():
                    if isinstance(value, dict):
                        json_results[cav_id][cam_id][key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                                              for k, v in value.items()}
                    else:
                        json_results[cav_id][cam_id][key] = int(value) if isinstance(value, (np.integer, int)) else value
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n统计结果已保存到: {output_file}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # 数据目录
    data_dir = "data/v2x-r/validate"
    
    # 如果提供了命令行参数，使用该参数
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    if 'v2x-radar' in data_dir:
        # v2x-radar数据集使用1536x864
        image_width = 1536
        image_height = 864
    elif 'v2x-r' in data_dir:
        # v2x-r数据集使用800x600
        image_width = 800
        image_height = 600
    
    # 如果提供了命令行参数，覆盖默认值
    if len(sys.argv) > 3:
        image_width = int(sys.argv[2])
        image_height = int(sys.argv[3])
    
    print("="*80)
    print("相机视场角统计工具")
    print("="*80)
    print(f"数据目录: {data_dir}")
    print(f"图像尺寸: {image_width} x {image_height}")
    print("="*80)
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 目录不存在: {data_dir}")
        sys.exit(1)
    
    # 执行统计
    results = statistics_camera_fov(data_dir, image_width, image_height)
    
    print("\n" + "="*80)
    print("统计完成!")
    print("="*80)

