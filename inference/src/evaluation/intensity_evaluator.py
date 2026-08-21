"""
强度评估器主类
统一处理Waymo和nuScenes数据集的强度评估
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

from .spherical_projection import (
    project_to_range_image_fast,
    project_to_range_image_waymo,
    project_to_range_image_nuscenes,
    save_range_image_vis,
    remove_empty_columns,
)
from .metrics import (
    compute_intensity_metrics,
    compute_all_metrics,
    format_metrics,
)


def load_ply_as_points_with_intensity(
    ply_path: str,
    remove_white: bool = False,
    exclude_coords: set = None,
) -> Tuple[np.ndarray, set]:
    """
    从PLY文件加载点云数据（带强度/反射率）

    Args:
        ply_path: PLY文件路径
        remove_white: 是否移除纯白色(255,255,255)的点
        exclude_coords: 要排除的坐标集合

    Returns:
        points: (N, 4), [x, y, z, intensity]
        excluded_coords: 被移除的点坐标集合
    """
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("plyfile package is required. Install with: pip install plyfile")

    # Robust loading: Read bytes, sanitize header (remove non-ascii in comments), then parse
    import io
    with open(ply_path, 'rb') as f:
        content = f.read()

    # Find end of header
    header_end_marker = b'end_header\n'
    header_end_pos = content.find(header_end_marker)

    if header_end_pos != -1:
        header_bytes = content[:header_end_pos + len(header_end_marker)]
        body_bytes = content[header_end_pos + len(header_end_marker):]

        # Decode header, sanitize non-ascii (e.g. Chinese comments)
        try:
            header_str = header_bytes.decode('utf-8')
        except UnicodeDecodeError:
            header_str = header_bytes.decode('latin-1')

        # Ensure pure ascii for plyfile compatibility by replacing non-ascii
        header_str = header_str.encode('ascii', errors='replace').decode('ascii')

        # Re-assemble
        sanitized_content = header_str.encode('ascii') + body_bytes
        f_obj = io.BytesIO(sanitized_content)
        pd = PlyData.read(f_obj)
    else:
        # No header end found? Fallback to direct read
        f_obj = io.BytesIO(content)
        pd = PlyData.read(f_obj)

    elem = None
    for el in pd.elements:
        el_names = el.data.dtype.names
        if el_names and all(n in el_names for n in ('x', 'y', 'z')):
            elem = el
            break

    if elem is None:
        raise RuntimeError(f'No element with x,y,z fields found in {ply_path}')

    v = elem.data
    names = v.dtype.names

    x = v['x'].astype(np.float32)
    y = v['y'].astype(np.float32)
    z = v['z'].astype(np.float32)

    # 查找强度字段
    inten_field = None
    for candidate in ('intensity', 'reflectance', 'intensity_r', 'i', 'scalar_intensity'):
        if candidate in names:
            inten_field = candidate
            break

    if inten_field is None:
        raise RuntimeError(f'No intensity/reflectance field found in {ply_path}')

    inten = v[inten_field].astype(np.float32)
    excluded_coords_output = set()

    # 移除纯白色点
    if remove_white:
        color_field_names = None
        for trip in (('red', 'green', 'blue'), ('r', 'g', 'b')):
            if all(c in names for c in trip):
                color_field_names = trip
                break

        if color_field_names is not None:
            r = v[color_field_names[0]].astype(np.int32)
            g = v[color_field_names[1]].astype(np.int32)
            b = v[color_field_names[2]].astype(np.int32)
            white_mask = (r == 255) & (g == 255) & (b == 255)
            removed = int(white_mask.sum())

            if removed > 0:
                print(f"Removed {removed} pure-white points from {ply_path}")
                for i in np.where(white_mask)[0]:
                    excluded_coords_output.add((float(x[i]), float(y[i]), float(z[i])))

            keep_mask = ~white_mask
            x = x[keep_mask]
            y = y[keep_mask]
            z = z[keep_mask]
            inten = inten[keep_mask]

    # 移除指定坐标的点
    if exclude_coords is not None and len(exclude_coords) > 0:
        keep_mask = np.ones(len(x), dtype=bool)
        for i in range(len(x)):
            coord = (float(x[i]), float(y[i]), float(z[i]))
            if coord in exclude_coords:
                keep_mask[i] = False

        removed = int((~keep_mask).sum())
        if removed > 0:
            print(f"Removed {removed} points matching exclude coordinates from {ply_path}")

        x = x[keep_mask]
        y = y[keep_mask]
        z = z[keep_mask]
        inten = inten[keep_mask]

    pts = np.column_stack((x, y, z, inten))
    return pts, excluded_coords_output


def load_npy_as_points(npy_path: str) -> np.ndarray:
    """
    从NPY文件加载点云数据

    Args:
        npy_path: NPY文件路径

    Returns:
        points: (N, 4+), [x, y, z, intensity, ...]
    """
    points = np.load(npy_path)
    return points


class IntensityEvaluator:
    """
    强度评估器

    支持三种数据集:
    - Waymo: 部分FOV，需要截取没有成功投影的区域
    - nuScenes: 完整FOV，使用Velodyne HDL-32E参数
    - KITTI-360: 使用Velodyne HDL-64E参数，类似 Waymo 处理
    """

    def __init__(
        self,
        dataset_type: str = 'waymo',
        waymo_calib_json: str = None,
        nuscenes_calib_json: str = None,
        # Waymo 投影参数
        waymo_h_res: float = 0.35,
        waymo_v_res: float = 0.4,
        waymo_v_fov: Tuple[float, float] = (-3, 37),
        # nuScenes 投影参数
        nuscenes_h_res: float = 0.35,
        nuscenes_v_res: float = 0.4,
        nuscenes_v_fov: Tuple[float, float] = (-30, 10),
        # Range image size overrides (H x W). If None, defaults are used per-dataset
        range_H: int = 64,
        default_W: Optional[int] = None,
        # 通用参数
        cmap_name: str = 'cividis',
        intensity_vmax: float = 1.5,
        compute_lpips: bool = True,
        lpips_net: str = 'alex',
        device: str = 'cpu',
    ):
        """
        初始化评估器

        Args:
            dataset_type: 数据集类型 'waymo', 'nuscenes', 或 'kitti360'
            waymo_calib_json: Waymo激光雷达标定JSON文件路径
            nuscenes_calib_json: nuScenes激光雷达标定JSON文件路径
            waymo_h_res: Waymo/KITTI-360 水平角分辨率（度）
            waymo_v_res: Waymo/KITTI-360 垂直角分辨率（度）
            waymo_v_fov: Waymo/KITTI-360 垂直视场角范围
            nuscenes_h_res: nuScenes水平角分辨率
            nuscenes_v_res: nuScenes垂直角分辨率
            nuscenes_v_fov: nuScenes垂直视场角范围
            cmap_name: colormap名称
            intensity_vmax: 强度归一化最大值
            compute_lpips: 是否计算LPIPS
            lpips_net: LPIPS网络类型
            device: 计算设备
        """
        self.dataset_type = dataset_type.lower()
        self.waymo_calib_json = waymo_calib_json
        self.nuscenes_calib_json = nuscenes_calib_json

        # Waymo/KITTI-360 参数（KITTI-360 复用 Waymo 参数）
        self.waymo_h_res = waymo_h_res
        self.waymo_v_res = waymo_v_res
        self.waymo_v_fov = waymo_v_fov

        # nuScenes参数
        self.nuscenes_h_res = nuscenes_h_res
        self.nuscenes_v_res = nuscenes_v_res
        self.nuscenes_v_fov = nuscenes_v_fov
        # Range image overrides
        # range_H: number of vertical pixels (e.g. 64)
        # default_W: horizontal pixels (e.g. 2650 for full-360 setting)
        self.range_H = range_H
        self.default_W = default_W

        # nuScenes 默认水平像素数（改为 1024，用户要求）
        self.nuscenes_default_W = 1024

        # 通用参数
        self.cmap_name = cmap_name
        self.intensity_vmax = intensity_vmax
        self.compute_lpips = compute_lpips
        self.lpips_net = lpips_net
        self.device = device

        # LPIPS模型（延迟加载）
        self._lpips_model = None

    @property
    def lpips_model(self):
        """延迟加载LPIPS模型"""
        if self._lpips_model is None and self.compute_lpips:
            try:
                import lpips
                self._lpips_model = lpips.LPIPS(net=self.lpips_net).eval()
                if self.device != 'cpu':
                    self._lpips_model = self._lpips_model.to(self.device)
            except ImportError:
                warnings.warn("lpips not available")
        return self._lpips_model

    def project_pointcloud(
        self,
        points: np.ndarray,
        ego2world: np.ndarray = None,
    ) -> np.ndarray:
        """
        将点云投影到Range Image

        Args:
            points: (N, 4+), [x, y, z, intensity, ...]
            ego2world: (4, 4) 车辆到世界的变换矩阵（如果点云是世界坐标）

        Returns:
            range_image: (H, W, 2), [range, intensity]
        """
        if self.dataset_type == 'waymo' and self.waymo_calib_json:
            return project_to_range_image_waymo(points, self.waymo_calib_json, ego2world=ego2world)
        elif self.dataset_type == 'nuscenes' and self.nuscenes_calib_json:
            # nuScenes使用精确beam inclination投影（类似Waymo）
            return project_to_range_image_nuscenes(points, self.nuscenes_calib_json)
        else:
            # 使用通用投影（后备方案）- 支持 KITTI-360 和无标定文件的情况
            # 如果有 ego2world，先将世界坐标转换到车辆坐标
            pts = points.copy()
            if ego2world is not None:
                world2ego = np.linalg.inv(ego2world)
                xyz_world = pts[:, :3]
                xyz_vehicle = xyz_world @ world2ego[:3, :3].T + world2ego[:3, 3]
                pts[:, :3] = xyz_vehicle

            if self.dataset_type == 'waymo':
                v_fov = self.waymo_v_fov
                # allow override of default W (full-360 optimized)
                W = self.default_W if self.default_W is not None else 2650
            elif self.dataset_type == 'kitti360':
                # KITTI-360 使用类似 Waymo 的参数
                v_fov = self.waymo_v_fov  # 使用配置的 KITTI-360 FOV
                if self.default_W is not None:
                    W = self.default_W
                else:
                    W = int(360.0 / self.waymo_h_res)  # 根据水平分辨率计算宽度
            else:
                v_fov = self.nuscenes_v_fov
                W = self.nuscenes_default_W

            range_map, intensity_map = project_to_range_image_fast(
                pts,
                H=self.range_H,
                W=W,
                inc_bottom_deg=v_fov[0],
                inc_top_deg=v_fov[1],
            )

            # 合并为(H, W, 2)格式
            return np.stack([range_map, intensity_map], axis=-1)

    def evaluate(
        self,
        gt_points: np.ndarray,
        pred_points: np.ndarray,
        output_dir: str,
        sample_name: str = 'sample',
        remove_invalid_columns: bool = None,
        normalize_gt: bool = True,
        ego2world: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        评估预测点云强度与GT的差异

        Args:
            gt_points: GT点云 (N, 4+), [x, y, z, intensity]
            pred_points: 预测点云 (M, 4+), [x, y, z, intensity]
            output_dir: 输出目录
            sample_name: 样本名称（用于输出文件命名）
            remove_invalid_columns: 是否移除无效列（默认Waymo=True, nuScenes=False）
            normalize_gt: 是否对GT强度进行归一化（如果GT范围不是[0,1]）
            ego2world: (4, 4) 车辆到世界的变换矩阵（如果点云是世界坐标）

        Returns:
            metrics: 评估指标字典
        """
        import cv2

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 默认: Waymo 和 KITTI-360 需要移除无效列，nuScenes不需要
        if remove_invalid_columns is None:
            remove_invalid_columns = (self.dataset_type in ['waymo', 'kitti360'])

        # 复制点云以避免修改原始数据
        gt_points = gt_points.copy()
        pred_points = pred_points.copy()

        # 归一化GT强度（如果需要）
        gt_intensity = gt_points[:, 3]
        pred_intensity = pred_points[:, 3]

        print(f"GT intensity range: {gt_intensity.min():.4f} - {gt_intensity.max():.4f}")
        print(f"Pred intensity range: {pred_intensity.min():.4f} - {pred_intensity.max():.4f}")

        if normalize_gt and gt_intensity.max() > 1.5:
            # GT强度范围很大，使用百分位数归一化以处理异常值
            # 使用 1%-99% 百分位数作为归一化范围，避免极端值影响
            p1 = np.percentile(gt_intensity, 1)
            p99 = np.percentile(gt_intensity, 99)
            print(f"GT percentiles: p1={p1:.4f}, p99={p99:.4f}")

            if p99 > p1:
                # 裁剪并归一化
                gt_clipped = np.clip(gt_intensity, p1, p99)
                gt_points[:, 3] = (gt_clipped - p1) / (p99 - p1)
                print(f"Normalized GT intensity using p1-p99 percentile to [0, 1]")
            else:
                # fallback to min-max
                gt_min = gt_intensity.min()
                gt_max = gt_intensity.max()
                if gt_max > gt_min:
                    gt_points[:, 3] = (gt_intensity - gt_min) / (gt_max - gt_min)
                    print(f"Normalized GT intensity using min-max to [0, 1]")

        # 确保Pred也在[0,1]范围
        if pred_intensity.max() > 1.0:
            pred_min = pred_intensity.min()
            pred_max = pred_intensity.max()
            if pred_max > pred_min:
                pred_points[:, 3] = (pred_intensity - pred_min) / (pred_max - pred_min)
                print(f"Normalized Pred intensity to [0, 1]")

        # 投影到Range Image（传入 ego2world 进行坐标变换）
        print(f"Projecting GT pointcloud ({len(gt_points)} points)...")
        gt_range_image = self.project_pointcloud(gt_points, ego2world=ego2world)

        print(f"Projecting Pred pointcloud ({len(pred_points)} points)...")
        pred_range_image = self.project_pointcloud(pred_points, ego2world=ego2world)

        # 获取有效像素掩码
        # GT有效：有点投影到该像素（强度 >= 0）
        # Pred有效：强度 > 0（排除未被相机覆盖的点，它们的强度被设为0）
        gt_valid_mask = gt_range_image[..., 1] >= 0
        pred_valid_mask = pred_range_image[..., 1] > 0  # 注意：> 0，不是 >= 0

        # 计算两者都有效的像素（用于公平比较）
        both_valid_mask = gt_valid_mask & pred_valid_mask
        print(f"GT valid pixels: {gt_valid_mask.sum()}")
        print(f"Pred valid pixels (covered by camera): {pred_valid_mask.sum()}")
        print(f"Both valid pixels: {both_valid_mask.sum()}")

        # 保存完整的可视化图像
        gt_img_path = output_dir / f'{sample_name}_gt_intensity.png'
        pred_img_path = output_dir / f'{sample_name}_pred_intensity.png'

        save_range_image_vis(gt_range_image, str(gt_img_path),
                            cmap_name=self.cmap_name, vmax=1.0, channel=1)
        save_range_image_vis(pred_range_image, str(pred_img_path),
                            cmap_name=self.cmap_name, vmax=1.0, channel=1)

        # 读取生成的图像
        gt_img = cv2.imread(str(gt_img_path))
        pred_img = cv2.imread(str(pred_img_path))

        if gt_img is None or pred_img is None:
            raise RuntimeError('Failed to read generated images')

        # 截取有效区域用于指标计算
        if remove_invalid_columns:
            # 1. 找出有有效像素的列（both_valid_mask 中有 True 的列）
            valid_cols = np.where(both_valid_mask.any(axis=0))[0]
            if len(valid_cols) == 0:
                raise RuntimeError('No valid overlapping pixels between GT and Pred')

            # 2. 截取有效列
            gt_img_cropped = gt_img[:, valid_cols, :]
            pred_img_cropped = pred_img[:, valid_cols, :]
            both_valid_cropped = both_valid_mask[:, valid_cols]

            print(f'Cropped to {len(valid_cols)} valid columns out of {both_valid_mask.shape[1]}')
        else:
            gt_img_cropped = gt_img
            pred_img_cropped = pred_img
            both_valid_cropped = both_valid_mask
            print(f'Keeping all {both_valid_mask.shape[1]} columns (remove_invalid_columns=False)')

        # 3. 创建用于指标计算的掩码图像（只保留 both_valid 的像素）
        # 将无效像素在两个图像中都设为相同值（避免影响指标）
        gt_img_masked = gt_img_cropped.copy()
        pred_img_masked = pred_img_cropped.copy()

        # 对于不在 both_valid 中的像素，设置为相同的背景色
        invalid_mask_cropped = ~both_valid_cropped
        for c in range(3):
            gt_img_masked[:, :, c][invalid_mask_cropped] = 0
            pred_img_masked[:, :, c][invalid_mask_cropped] = 0

        # 保存 mask 图像 (0: 无效, 255: 有效)
        mask_vis = (both_valid_cropped.astype(np.uint8) * 255)
        mask_path = output_dir / f'{sample_name}_mask.png'
        cv2.imwrite(str(mask_path), mask_vis)
        print(f'Saved validation mask: {mask_path}')

        # 保存对比图（使用截取后的图像）
        comparison = np.vstack([gt_img_cropped, pred_img_cropped])

        # 针对低分辨率（如nuScenes 32线）进行视觉上采样
        H_comp, W_comp = comparison.shape[:2]
        if H_comp < 512: # 对比图由两张图叠加，这里按 256*2 的高度判断
            target_h = 512
            scale = target_h / H_comp
            new_W = int(W_comp * scale)
            comparison_vis = cv2.resize(comparison, (new_W, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            comparison_vis = comparison

        comparison_path = output_dir / f'{sample_name}_comparison.png'
        cv2.imwrite(str(comparison_path), comparison_vis)
        print(f'Saved comparison image: {comparison_path}')

        # 保存只包含有效区域的对比图 (黑底)
        comparison_masked = np.vstack([gt_img_masked, pred_img_masked])

        # 创建白底的 masked 图像
        gt_img_masked_white = gt_img_cropped.copy()
        pred_img_masked_white = pred_img_cropped.copy()
        for c in range(3):
            gt_img_masked_white[:, :, c][invalid_mask_cropped] = 255
            pred_img_masked_white[:, :, c][invalid_mask_cropped] = 255

        comparison_masked_white = np.vstack([gt_img_masked_white, pred_img_masked_white])

        if H_comp < 512:
            comparison_masked_vis = cv2.resize(comparison_masked, (new_W, target_h), interpolation=cv2.INTER_NEAREST)
            comparison_masked_white_vis = cv2.resize(comparison_masked_white, (new_W, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            comparison_masked_vis = comparison_masked
            comparison_masked_white_vis = comparison_masked_white

        comparison_masked_path = output_dir / f'{sample_name}_comparison_masked.png'
        cv2.imwrite(str(comparison_masked_path), comparison_masked_vis)
        print(f'Saved masked comparison image: {comparison_masked_path}')

        comparison_masked_white_path = output_dir / f'{sample_name}_comparison_masked_white.png'
        cv2.imwrite(str(comparison_masked_white_path), comparison_masked_white_vis)
        print(f'Saved white masked comparison image: {comparison_masked_white_path}')

        # 转换为灰度用于指标计算（使用 masked 版本）
        gt_gray = cv2.cvtColor(gt_img_masked, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        pred_gray = cv2.cvtColor(pred_img_masked, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gt_rgb = cv2.cvtColor(gt_img_masked, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pred_rgb = cv2.cvtColor(pred_img_masked, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 只在有效像素上计算指标
        valid_pixel_count = both_valid_cropped.sum()
        print(f'Computing metrics on {valid_pixel_count} valid pixels')

        # 提取有效像素的值
        gt_valid_values = gt_gray[both_valid_cropped]
        pred_valid_values = pred_gray[both_valid_cropped]

        # 计算逐像素指标（只在有效像素上）
        from .metrics import compute_rmse, compute_mae, compute_psnr

        rmse = compute_rmse(gt_valid_values, pred_valid_values)
        mae = compute_mae(gt_valid_values, pred_valid_values)
        medae = float(np.median(np.abs(gt_valid_values - pred_valid_values)))
        psnr = compute_psnr(gt_valid_values, pred_valid_values)

        # SSIM 需要在图像上计算（使用 masked 图像）
        from .metrics import compute_ssim
        ssim = compute_ssim(gt_gray, pred_gray)

        # LPIPS
        lpips_val = float('nan')
        if self.compute_lpips and self.lpips_model is not None:
            from .metrics import compute_lpips
            lpips_val = compute_lpips(gt_rgb, pred_rgb, self.lpips_model)

        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'medae': float(medae),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'lpips': float(lpips_val),
            'valid_pixels': int(valid_pixel_count),
            'total_pixels': int(gt_img_cropped.shape[0] * gt_img_cropped.shape[1]),
            'coverage': float(valid_pixel_count / (gt_img_cropped.shape[0] * gt_img_cropped.shape[1])),
        }

        # 保存指标
        metrics_path = output_dir / f'{sample_name}_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f'Saved metrics: {metrics_path}')

        # 打印指标
        print(format_metrics(metrics))

        return metrics

    def evaluate_from_files(
        self,
        gt_path: str,
        pred_path: str,
        output_dir: str,
        sample_name: str = 'sample',
        remove_white: bool = True,
        remove_invalid_columns: bool = None,
    ) -> Dict[str, float]:
        """
        从文件加载点云并评估

        Args:
            gt_path: GT点云路径 (.ply 或 .npy)
            pred_path: 预测点云路径 (.ply 或 .npy)
            output_dir: 输出目录
            sample_name: 样本名称
            remove_white: 是否移除白色点
            remove_invalid_columns: 是否移除无效列

        Returns:
            metrics: 评估指标字典
        """
        gt_path = Path(gt_path)
        pred_path = Path(pred_path)

        # 加载GT
        print(f"Loading GT: {gt_path}")
        if gt_path.suffix.lower() == '.ply':
            gt_points, excluded_coords = load_ply_as_points_with_intensity(
                str(gt_path), remove_white=remove_white
            )
        else:
            gt_points = load_npy_as_points(str(gt_path))
            excluded_coords = set()

        # 加载Pred
        print(f"Loading Pred: {pred_path}")
        if pred_path.suffix.lower() == '.ply':
            pred_points, _ = load_ply_as_points_with_intensity(
                str(pred_path), remove_white=remove_white, exclude_coords=excluded_coords
            )
        else:
            pred_points = load_npy_as_points(str(pred_path))

        return self.evaluate(
            gt_points, pred_points, output_dir, sample_name, remove_invalid_columns
        )

    def evaluate_batch(
        self,
        gt_pred_pairs: List[Tuple[str, str]],
        output_dir: str,
        remove_white: bool = True,
        remove_invalid_columns: bool = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        批量评估多个样本

        Args:
            gt_pred_pairs: [(gt_path, pred_path), ...] 列表
            output_dir: 输出目录
            remove_white: 是否移除白色点
            remove_invalid_columns: 是否移除无效列

        Returns:
            all_metrics: {sample_name: metrics} 字典
        """
        output_dir = Path(output_dir)
        all_metrics = {}

        for idx, (gt_path, pred_path) in enumerate(gt_pred_pairs):
            sample_name = f'sample_{idx:06d}'
            print(f"\n{'='*60}")
            print(f"Evaluating {sample_name}: {Path(gt_path).name} vs {Path(pred_path).name}")
            print('='*60)

            try:
                metrics = self.evaluate_from_files(
                    gt_path, pred_path,
                    str(output_dir / sample_name),
                    sample_name,
                    remove_white,
                    remove_invalid_columns,
                )
                all_metrics[sample_name] = metrics
            except Exception as e:
                print(f"Error evaluating {sample_name}: {e}")
                all_metrics[sample_name] = {'error': str(e)}

        # 计算平均指标
        avg_metrics = self._compute_average_metrics(all_metrics)

        # 保存汇总
        summary_path = output_dir / 'evaluation_summary.json'
        summary = {
            'individual': all_metrics,
            'average': avg_metrics,
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("Average Metrics")
        print('='*60)
        print(format_metrics(avg_metrics))

        return all_metrics

    def _compute_average_metrics(
        self,
        all_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """计算所有样本的平均指标"""
        metric_keys = ['rmse', 'mae', 'medae', 'psnr', 'ssim', 'lpips']
        avg = {}

        for key in metric_keys:
            values = []
            for sample_metrics in all_metrics.values():
                if 'error' not in sample_metrics and key in sample_metrics:
                    val = sample_metrics[key]
                    if not np.isnan(val):
                        values.append(val)

            if values:
                avg[key] = float(np.mean(values))
            else:
                avg[key] = float('nan')

        avg['num_samples'] = len([m for m in all_metrics.values() if 'error' not in m])
        return avg


def create_evaluator_from_config(config: Dict) -> IntensityEvaluator:
    """
    从配置字典创建评估器

    Args:
        config: 配置字典

    Returns:
        IntensityEvaluator实例
    """
    eval_cfg = config.get('evaluation', {})
    dataset_cfg = config.get('dataset', {})

    dataset_type = dataset_cfg.get('type', 'waymo').lower()
    waymo_calib_json = eval_cfg.get('waymo_calib_json')

    return IntensityEvaluator(
        dataset_type=dataset_type,
        waymo_calib_json=waymo_calib_json,
        waymo_h_res=eval_cfg.get('waymo_h_res', 0.35),
        waymo_v_res=eval_cfg.get('waymo_v_res', 0.4),
        waymo_v_fov=tuple(eval_cfg.get('waymo_v_fov', [-3, 37])),
        nuscenes_h_res=eval_cfg.get('nuscenes_h_res', 0.35),
        nuscenes_v_res=eval_cfg.get('nuscenes_v_res', 0.4),
        nuscenes_v_fov=tuple(eval_cfg.get('nuscenes_v_fov', [-30, 10])),
        cmap_name=eval_cfg.get('cmap_name', 'cividis'),
        intensity_vmax=eval_cfg.get('intensity_vmax', 1.5),
        compute_lpips=eval_cfg.get('compute_lpips', True),
        lpips_net=eval_cfg.get('lpips_net', 'alex'),
        device=eval_cfg.get('device', 'cpu'),
    )
