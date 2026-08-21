"""
球面投影模块
将点云投影到球面Range Image或前视图
"""

import math
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt


def project_to_range_image(
    points: np.ndarray,
    H: int = 64,
    W: int = 2650,
    inc_bottom_deg: float = -24.9,
    inc_top_deg: float = 2.0,
    azimuth_left: float = np.pi,
    azimuth_right: float = -np.pi,
    max_depth: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将点云投影到固定分辨率的 range / intensity 图 (H, W)。

    使用球面投影:
    - 方位角 azimuth = atan2(y, x)
    - 俯仰角 inclination = atan2(z, sqrt(x^2 + y^2))
    - 每个像素只保留距离最近的点 (z-buffer)

    Args:
        points: (N, 4+) 点云, [x, y, z, intensity, ...]
        H: 图像高度 (垂直方向像素数)
        W: 图像宽度 (水平方向像素数)
        inc_bottom_deg: 垂直视场下边界 (度)
        inc_top_deg: 垂直视场上边界 (度)
        azimuth_left: 水平视场左边界 (弧度)
        azimuth_right: 水平视场右边界 (弧度)
        max_depth: 最大深度 (米)

    Returns:
        range_map: (H, W), 深度图, 无效位置为-1
        intensity_map: (H, W), 强度图, 无效位置为-1
    """
    xyzs = points[:, :3]
    intensities = points[:, 3]
    dists = np.linalg.norm(xyzs, axis=1)

    # 垂直视场边界（弧度）
    inc_bottom = math.radians(inc_bottom_deg)
    inc_top = math.radians(inc_top_deg)

    # 角分辨率（弧度/像素）
    h_res = (azimuth_right - azimuth_left) / float(W)
    v_res = (inc_bottom - inc_top) / float(H)

    range_map = np.full((H, W), -1.0, dtype=np.float32)
    intensity_map = np.full((H, W), -1.0, dtype=np.float32)

    for i, ((x, y, z), intensity, dist) in enumerate(zip(xyzs, intensities, dists)):
        if dist > max_depth:
            continue

        azimuth = np.arctan2(y, x)
        inclination = np.arctan2(z, np.sqrt(x**2 + y**2))

        w_idx = int(np.round((azimuth - azimuth_left) / h_res))
        h_idx = int(np.round((inclination - inc_top) / v_res))

        if w_idx < 0 or w_idx >= W or h_idx < 0 or h_idx >= H:
            continue

        # Z-buffer: 保留距离更近的点
        if range_map[h_idx, w_idx] < 0 or range_map[h_idx, w_idx] > dist:
            range_map[h_idx, w_idx] = dist
            intensity_map[h_idx, w_idx] = intensity

    return range_map, intensity_map


def project_to_range_image_fast(
    points: np.ndarray,
    H: int = 64,
    W: int = 2650,
    inc_bottom_deg: float = -24.9,
    inc_top_deg: float = 2.0,
    azimuth_left: float = np.pi,
    azimuth_right: float = -np.pi,
    max_depth: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    向量化的球面投影（比循环版本快很多）
    """
    xyzs = points[:, :3]
    intensities = points[:, 3]
    dists = np.linalg.norm(xyzs, axis=1)

    # 过滤超出最大距离的点
    valid_mask = dists <= max_depth
    xyzs = xyzs[valid_mask]
    intensities = intensities[valid_mask]
    dists = dists[valid_mask]

    if len(xyzs) == 0:
        return np.full((H, W), -1.0, dtype=np.float32), np.full((H, W), -1.0, dtype=np.float32)

    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]

    # 垂直视场边界（弧度）
    inc_bottom = math.radians(inc_bottom_deg)
    inc_top = math.radians(inc_top_deg)

    # 角分辨率
    h_res = (azimuth_right - azimuth_left) / float(W)
    v_res = (inc_bottom - inc_top) / float(H)

    # 计算方位角和俯仰角
    azimuth = np.arctan2(y, x)
    inclination = np.arctan2(z, np.sqrt(x**2 + y**2))

    # 计算像素索引
    w_idx = np.round((azimuth - azimuth_left) / h_res).astype(np.int32)
    h_idx = np.round((inclination - inc_top) / v_res).astype(np.int32)

    # 过滤越界点
    valid = (w_idx >= 0) & (w_idx < W) & (h_idx >= 0) & (h_idx < H)
    w_idx = w_idx[valid]
    h_idx = h_idx[valid]
    dists = dists[valid]
    intensities = intensities[valid]

    # 初始化
    range_map = np.full((H, W), -1.0, dtype=np.float32)
    intensity_map = np.full((H, W), -1.0, dtype=np.float32)

    # 按距离排序（远到近），这样近的点会覆盖远的点
    order = np.argsort(-dists)
    w_idx = w_idx[order]
    h_idx = h_idx[order]
    dists = dists[order]
    intensities = intensities[order]

    # 直接赋值（近的覆盖远的）
    range_map[h_idx, w_idx] = dists
    intensity_map[h_idx, w_idx] = intensities

    return range_map, intensity_map


def project_to_range_image_waymo(
    points: np.ndarray,
    calib_json_path: str,
    ego2world: np.ndarray = None,
) -> np.ndarray:
    """
    使用Waymo标定文件进行球面投影

    Args:
        points: (N, 4+) 点云 [x, y, z, intensity]
                - 如果提供了 ego2world，则点云在世界坐标系下，需要先转换
                - 如果没有 ego2world，则假定点云已在车辆坐标系下
        calib_json_path: Waymo激光雷达标定JSON文件路径
        ego2world: (4, 4) 车辆到世界的变换矩阵，如果提供则进行逆变换

    Returns:
        range_image: (H, W, 2), [range, intensity]
    """
    with open(calib_json_path, 'r', encoding='utf-8') as f:
        calib = json.load(f)

    # 读取beam inclinations
    if 'beam_inclinations' in calib and len(calib['beam_inclinations']) > 0:
        beam_inclinations = np.array(calib['beam_inclinations'])
    else:
        beam_inc_min = calib.get('beam_inclination_min', -0.31450354886531773)
        beam_inc_max = calib.get('beam_inclination_max', 0.03988560766067384)
        beam_inclinations = np.linspace(beam_inc_min, beam_inc_max, 64)

    beam_inclinations = beam_inclinations[::-1]  # 从上到下
    H = len(beam_inclinations)
    W = 2650  # Waymo默认宽度

    # 读取外参矩阵 (lidar -> vehicle)
    extrinsic = np.reshape(np.array(calib['extrinsic_transform']), [4, 4])
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t

    pts = np.asarray(points)
    if pts.size == 0:
        return np.full((H, W, 2), -1.0, dtype=np.float32)

    xyz = pts[:, :3].copy()

    # 如果提供了 ego2world，先将世界坐标转换到车辆坐标
    if ego2world is not None:
        # world -> vehicle (逆变换)
        world2ego = np.linalg.inv(ego2world)
        R_w2e = world2ego[:3, :3]
        t_w2e = world2ego[:3, 3]
        xyz = xyz @ R_w2e.T + t_w2e.reshape(1, 3)

    # 车辆坐标系 -> 激光雷达坐标系
    xyz_lidar = xyz @ R_inv.T + t_inv.reshape(1, 3)

    x, y, z = xyz_lidar[:, 0], xyz_lidar[:, 1], xyz_lidar[:, 2]
    range_vals = np.linalg.norm(xyz_lidar, axis=1)
    valid = range_vals > 0

    # 计算俯仰角和方位角
    inclination = np.arcsin(np.clip(z / (range_vals + 1e-12), -1.0, 1.0))
    azimuth = np.arctan2(y, x)

    # 匹配最近的beam inclination
    diff = np.abs(inclination.reshape(-1, 1) - beam_inclinations.reshape(1, -1))
    row_idx = np.argmin(diff, axis=1)

    # 方位角归一化到[0, 1]
    azimuth_normalized = (azimuth + np.pi) / (2 * np.pi)
    col_idx = (azimuth_normalized * W).astype(np.int32) % W

    # 初始化range image
    range_image = np.full((H, W, 2), -1.0, dtype=np.float32)

    if pts.shape[1] >= 4:
        intensity = pts[:, 3]
    else:
        intensity = np.zeros_like(range_vals)

    # Z-buffer投影
    for i in range(len(pts)):
        if not valid[i]:
            continue
        r, c = row_idx[i], col_idx[i]
        current_range = range_image[r, c, 0]
        if current_range < 0 or range_vals[i] < current_range:
            range_image[r, c, 0] = range_vals[i]
            range_image[r, c, 1] = intensity[i]

    return range_image


def project_to_range_image_nuscenes(
    points: np.ndarray,
    calib_json_path: str,
) -> np.ndarray:
    """
    使用nuScenes/Velodyne HDL-32E标定文件进行球面投影

    Args:
        points: (N, 4+) 点云 [x, y, z, intensity]
                nuScenes点云已在ego坐标系下，无需额外坐标变换
        calib_json_path: 激光雷达标定JSON文件路径

    Returns:
        range_image: (H, W, 2), [range, intensity]
    """
    with open(calib_json_path, 'r', encoding='utf-8') as f:
        calib = json.load(f)

    # 支持多种字段名：beam_inclinations, beam_inclinations_rad, 或从channel_mapping提取
    beam_inclinations = None
    if 'beam_inclinations' in calib and len(calib['beam_inclinations']) > 0:
        beam_inclinations = np.array(calib['beam_inclinations'])
    elif 'beam_inclinations_rad' in calib and len(calib['beam_inclinations_rad']) > 0:
        beam_inclinations = np.array(calib['beam_inclinations_rad'])
    elif 'channel_mapping' in calib:
        mapping = calib['channel_mapping']
        indices = sorted([int(k) for k in mapping.keys()])
        beam_inclinations = np.array([mapping[str(i)]['angle_rad'] for i in indices])

    # 如果没找到，使用默认线性分布
    if beam_inclinations is None:
        beam_inc_min = calib.get('beam_inclination_min', -0.5352924816)
        beam_inc_max = calib.get('beam_inclination_max', 0.1862266312)
        beam_inclinations = np.linspace(beam_inc_max, beam_inc_min, 32)

    # 核心修复：强制按俯仰角从大到小排序（从上到下）
    # 解决 interleaved 排列导致的图像稀疏和特征破碎问题
    beam_inclinations = np.sort(beam_inclinations)[::-1]

    H = len(beam_inclinations)
    # nuScenes 360° FOV
    # 使用较低的水平分辨率以避免过度横向拉伸（改为 1024，用户要求）
    W = 1024

    pts = np.asarray(points)
    if pts.size == 0:
        return np.full((H, W, 2), -1.0, dtype=np.float32)

    xyz = pts[:, :3].copy()
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    range_vals = np.linalg.norm(xyz, axis=1)
    valid = range_vals > 0

    # 计算俯仰角
    inclination = np.arcsin(np.clip(z / (range_vals + 1e-12), -1.0, 1.0))
    azimuth = np.arctan2(y, x)

    # 匹配最近的beam inclination
    diff = np.abs(inclination.reshape(-1, 1) - beam_inclinations.reshape(1, -1))
    row_idx = np.argmin(diff, axis=1)

    # 方位角归一化到[0, 1]
    azimuth_normalized = (azimuth + np.pi) / (2 * np.pi)
    col_idx = (azimuth_normalized * W).astype(np.int32) % W

    # 初始化
    range_image = np.full((H, W, 2), -1.0, dtype=np.float32)

    intensity = pts[:, 3] if pts.shape[1] >= 4 else np.zeros_like(range_vals)

    # Z-buffer投影
    order = np.argsort(-range_vals)
    v_order = valid[order]

    row_s = row_idx[order][v_order]
    col_s = col_idx[order][v_order]
    range_s = range_vals[order][v_order]
    inte_s = intensity[order][v_order]

    range_image[row_s, col_s, 0] = range_s
    range_image[row_s, col_s, 1] = inte_s

    return range_image


def lidar_to_front_view(
    points: np.ndarray,
    v_res: float = 0.4,
    h_res: float = 0.35,
    v_fov: Tuple[float, float] = (-30.0, 10.0),
    val: str = "reflectance",
    y_fudge: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    将点云投影到前视图（散点图形式）

    Args:
        points: (N, 4+), [x, y, z, intensity/reflectance]
        v_res: 垂直角分辨率（度）
        h_res: 水平角分辨率（度）
        v_fov: 垂直视场角范围 (下边界, 上边界)，下边界通常为负
        val: 像素值类型 "reflectance", "depth", "height"
        y_fudge: Y方向额外边距

    Returns:
        x_img: (N,), 图像X坐标
        y_img: (N,), 图像Y坐标
        width: 图像宽度
        height: 图像高度
    """
    assert len(v_fov) == 2, "v_fov must be tuple of length 2"
    assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"
    assert val in {"depth", "height", "reflectance"}, \
        'val must be one of {"depth", "height", "reflectance"}'

    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    r_lidar = points[:, 3]  # reflectance

    # 水平距离
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]

    # 转换为弧度
    v_res_rad = v_res * (np.pi / 180)
    h_res_rad = h_res * (np.pi / 180)

    # 投影到图像坐标
    x_img = np.arctan2(-y_lidar, x_lidar) / h_res_rad
    y_img = np.arctan2(z_lidar, d_lidar) / v_res_rad

    # 平移坐标使(0,0)为最小值
    x_min = -360.0 / h_res / 2
    x_img -= x_min
    x_max = 360.0 / h_res

    y_min = v_fov[0] / v_res
    y_img -= y_min
    y_max = v_fov_total / v_res

    y_max += y_fudge

    # 选择像素值
    if val == "reflectance":
        pixel_values = r_lidar
    elif val == "height":
        pixel_values = z_lidar
    else:
        pixel_values = -d_lidar

    width = int(np.ceil(x_max))
    height = int(np.ceil(y_max))

    return x_img, y_img, pixel_values, width, height


def save_front_view_image(
    points: np.ndarray,
    save_path: str,
    v_res: float = 0.4,
    h_res: float = 0.35,
    v_fov: Tuple[float, float] = (-30.0, 10.0),
    val: str = "reflectance",
    cmap: str = "jet",
    dpi: int = 100,
    y_fudge: float = 0.0,
):
    """
    保存点云前视图图像
    """
    x_img, y_img, pixel_values, width, height = lidar_to_front_view(
        points, v_res, h_res, v_fov, val, y_fudge
    )

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.scatter(x_img, y_img, s=1, c=pixel_values, linewidths=0, alpha=1, cmap=cmap)
    ax.set_facecolor((0, 0, 0))
    ax.axis('scaled')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.xlim([0, width])
    plt.ylim([0, height])

    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)


def save_range_image_vis(
    range_image: np.ndarray,
    out_path: str,
    cmap_name: str = 'cividis',
    vmax: float = 1.5,
    channel: int = 1,
):
    """
    保存Range Image的可视化图像

    Args:
        range_image: (H, W, 2) 或 (H, W), range image数据
        out_path: 输出路径
        cmap_name: colormap名称
        vmax: 最大值归一化阈值
        channel: 如果是(H,W,2)格式，选择哪个通道 (0=range, 1=intensity)
    """
    import cv2
    cmap = plt.get_cmap(cmap_name)

    if range_image.ndim == 3:
        data = range_image[..., channel]
    else:
        data = range_image

    # 归一化
    norm = np.clip(data / (vmax + 1e-6), 0.0, 1.0)

    # 应用colormap
    rgba = cmap(norm)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)

    # OpenCV 使用 BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)


def remove_empty_columns(
    img: np.ndarray,
    cmap_name: str = 'plasma',
    empty_value: float = 0.0,
    tolerance: int = 5,
) -> Tuple[np.ndarray, List[int]]:
    """
    删除图像中所有像素都是 colormap(empty_value) 颜色的列

    Args:
        img: BGR图像 (OpenCV格式)
        cmap_name: colormap名称
        empty_value: 对应的归一化值 (0.0-1.0)
        tolerance: RGB容差

    Returns:
        filtered_img: 删除空列后的图像
        removed_cols: 被删除的列索引列表
    """
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(empty_value)
    rgba_array = np.array(rgba)
    target_rgb = (rgba_array[:3] * 255).astype(np.uint8)
    target_bgr = target_rgb[::-1]  # OpenCV使用BGR

    H, W = img.shape[:2]
    cols_to_keep = []
    removed_cols = []

    for col_idx in range(W):
        column = img[:, col_idx, :]
        diffs = np.abs(column.astype(np.int32) - target_bgr.astype(np.int32))
        max_diff = np.max(diffs)

        if max_diff > tolerance:
            cols_to_keep.append(col_idx)
        else:
            removed_cols.append(col_idx)

    if len(cols_to_keep) == 0:
        return img, removed_cols

    filtered_img = img[:, cols_to_keep, :]
    return filtered_img, removed_cols
