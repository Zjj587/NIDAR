"""
投影和点云处理工具
"""
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List


def project_points_to_image(
    points: np.ndarray,
    intrinsic: List[float],
    extrinsic: np.ndarray,
    image_width: int,
    image_height: int,
    ego2world: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将点云投影到图像平面

    Args:
        points: (N, 4), [x, y, z, intensity] in world/ego coordinates
        intrinsic: [fx, fy, cx, cy]
        extrinsic: (4, 4), cam->ego transformation
        image_width: 图像宽度
        image_height: 图像高度
        ego2world: (4, 4), ego->world transformation (optional)

    Returns:
        valid_points: (M, 4), 投影成功的点
        uv_coords: (M, 2), 图像坐标 [u, v]
        point_indices: (M,), 原始点云中的索引
    """
    N = points.shape[0]

    # 点云坐标转换
    pts_xyz = points[:, :3]
    pts_h = np.concatenate([pts_xyz, np.ones((N, 1), dtype=np.float32)], axis=1)

    # 如果提供了ego2world,先转换到ego坐标系
    if ego2world is not None:
        world2ego = np.linalg.inv(ego2world)
        pts_ego = (world2ego @ pts_h.T).T
    else:
        pts_ego = pts_h

    # ego -> cam
    T_cam_ego = np.linalg.inv(extrinsic)  # extrinsic是cam->ego,需要求逆
    pts_cam = (T_cam_ego @ pts_ego.T).T  # (N, 4)

    # Waymo相机坐标系: X轴向前(深度), Y轴向左, Z轴向上
    # 投影到归一化平面: x_norm = -Y/X, y_norm = -Z/X
    Xc = pts_cam[:, 0]  # depth
    Yc = pts_cam[:, 1]
    Zc = pts_cam[:, 2]

    eps = 1e-6
    valid_depth = Xc > eps

    if not np.any(valid_depth):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 2)), np.zeros((0,), dtype=np.int64)

    # 投影到图像平面
    # Waymo intrinsic 格式: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    # 简单投影只用前4个值
    if len(intrinsic) >= 4:
        fx, fy, cx, cy = intrinsic[:4]
    else:
        fx, fy, cx, cy = intrinsic

    x_norm = -Yc[valid_depth] / Xc[valid_depth]
    y_norm = -Zc[valid_depth] / Xc[valid_depth]

    u = fx * x_norm + cx
    v = fy * y_norm + cy

    # 过滤在图像范围内的点
    inside = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)

    if not np.any(inside):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 2)), np.zeros((0,), dtype=np.int64)

    # 提取有效点
    idxs_all = np.nonzero(valid_depth)[0]
    point_indices = idxs_all[inside]
    valid_points = points[point_indices]
    uv_coords = np.stack([u[inside], v[inside]], axis=1)

    return valid_points, uv_coords, point_indices


def assign_intensity_from_image(
    points: np.ndarray,
    reflectance_image: np.ndarray,
    intrinsic: List[float],
    extrinsic: np.ndarray,
    ego2world: np.ndarray = None
) -> np.ndarray:
    """
    从反射率图像中提取强度值并赋给点云

    Args:
        points: (N, 4), [x, y, z, old_intensity]
        reflectance_image: (H, W) 或 (H, W, 1), 反射率图像 [0, 255] uint8
        intrinsic: [fx, fy, cx, cy]
        extrinsic: (4, 4), cam->ego
        ego2world: (4, 4), ego->world (optional)

    Returns:
        points_with_new_intensity: (N, 4), 最后一列更新为新强度
    """
    if reflectance_image.ndim == 3:
        reflectance_image = reflectance_image[:, :, 0]

    h, w = reflectance_image.shape

    # 投影点云到图像
    valid_points, uv_coords, point_indices = project_points_to_image(
        points, intrinsic, extrinsic, w, h, ego2world
    )

    if len(point_indices) == 0:
        return points

    # 提取图像中对应像素的强度值
    u = uv_coords[:, 0].astype(np.int32)
    v = uv_coords[:, 1].astype(np.int32)
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)

    intensities = reflectance_image[v, u].astype(np.float32)

    # 更新点云强度
    points_new = points.copy()
    points_new[point_indices, 3] = intensities

    return points_new


def merge_multi_camera_intensity(
    points: np.ndarray,
    reflectance_images: List[np.ndarray],
    intrinsics: List[List[float]],
    extrinsics: List[np.ndarray],
    ego2world: np.ndarray = None,
    merge_strategy: str = 'max',
    normalize: bool = False,
) -> np.ndarray:
    """
    从多个相机的反射率图像中融合强度值

    Args:
        points: (N, 4)
        reflectance_images: 反射率图像列表 [(H, W), ...]
        intrinsics: 内参列表 [[fx, fy, cx, cy], ...]
        extrinsics: 外参列表 [(4, 4), ...]
        ego2world: (4, 4)
        merge_strategy: 'max', 'mean', 'first'
        normalize: 是否将输出强度归一化到 [0, 1]

    Returns:
        points_merged: (N, 4), 融合后的点云
    """
    N = points.shape[0]
    intensity_matrix = []  # (num_cams, N)
    hit_matrix = []  # (num_cams, N), bool

    for ref_img, intr, extr in zip(reflectance_images, intrinsics, extrinsics):
        # 投影每个相机
        points_cam = assign_intensity_from_image(
            points, ref_img, intr, extr, ego2world
        )

        # 记录哪些点被该相机看到
        _, _, pt_idxs = project_points_to_image(
            points, intr, extr,
            ref_img.shape[1] if ref_img.ndim == 2 else ref_img.shape[1],
            ref_img.shape[0],
            ego2world
        )

        hit = np.zeros(N, dtype=bool)
        hit[pt_idxs] = True

        intensity_matrix.append(points_cam[:, 3])
        hit_matrix.append(hit)

    intensity_matrix = np.stack(intensity_matrix, axis=0)  # (C, N)
    hit_matrix = np.stack(hit_matrix, axis=0)  # (C, N)

    # 记录哪些点被任意相机看到
    any_hit = np.any(hit_matrix, axis=0)  # (N,)

    # 融合策略
    points_merged = points.copy()

    for i in range(N):
        hits = hit_matrix[:, i]
        if not np.any(hits):
            # 没有相机看到，将强度设为 -1 作为标记
            points_merged[i, 3] = -1.0
            continue

        values = intensity_matrix[hits, i]

        if merge_strategy == 'max':
            points_merged[i, 3] = np.max(values)
        elif merge_strategy == 'mean':
            points_merged[i, 3] = np.mean(values)
        elif merge_strategy == 'first':
            points_merged[i, 3] = values[0]
        else:
            raise ValueError(f"不支持的融合策略: {merge_strategy}")

    if normalize:
        # 只对被相机覆盖的点进行归一化
        # 使用被覆盖点的 min-max，而不是全局（包含原始GT值）的 min-max
        covered_mask = any_hit
        if np.sum(covered_mask) > 0:
            covered_intensities = points_merged[covered_mask, 3]
            imin = np.min(covered_intensities)
            imax = np.max(covered_intensities)

            if imax > imin:
                # 归一化被覆盖的点
                points_merged[covered_mask, 3] = (covered_intensities - imin) / (imax - imin)

            # 未被覆盖的点设为 0
            points_merged[~covered_mask, 3] = 0.0

    return points_merged


def save_pointcloud(points: np.ndarray, output_path: Path, format: str = 'npy'):
    """
    保存点云

    Args:
        points: (N, 4), [x, y, z, intensity]
        output_path: 输出路径
        format: 'npy' 或 'ply'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'npy':
        np.save(str(output_path), points)

    elif format == 'ply':
        # 写入ASCII PLY格式
        N = points.shape[0]
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {N}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float intensity\n")
            f.write("end_header\n")

            for i in range(N):
                x, y, z, inten = points[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {inten:.2f}\n")

    else:
        raise ValueError(f"不支持的格式: {format}")


def load_pointcloud(path: Path) -> np.ndarray:
    """
    加载点云

    Args:
        path: 点云文件路径 (.npy 或 .ply)

    Returns:
        points: (N, 4)
    """
    path = Path(path)

    if path.suffix == '.npy':
        points = np.load(str(path))

    elif path.suffix == '.ply':
        with open(path, 'r') as f:
            line = f.readline().strip()
            if line != "ply":
                raise ValueError(f"不是PLY文件: {path}")

            while line != "end_header":
                line = f.readline().strip()

            rows = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rows.append([float(x) for x in parts])

        points = np.asarray(rows, dtype=np.float32)

    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")

    # 确保形状 (N, 4)
    if points.shape[1] < 4:
        pad = np.zeros((points.shape[0], 4 - points.shape[1]), dtype=np.float32)
        points = np.concatenate([points, pad], axis=1)
    elif points.shape[1] > 4:
        points = points[:, :4]

    return points
