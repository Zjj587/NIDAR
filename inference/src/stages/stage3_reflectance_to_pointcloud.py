"""
Stage 3: 将反射率图像投影到点云,生成伪强度
"""
import sys
from pathlib import Path
from typing import Dict
import numpy as np

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.waymo_dataset import WaymoDatasetWrapper
from src.projection.projection_utils import (
    assign_intensity_from_image,
    merge_multi_camera_intensity,
    save_pointcloud
)
from src.utils.image_utils import load_image
from src.utils.logger import setup_logger


def process_stage3(config: Dict, logger=None):
    """
    Stage 3 主处理函数

    Args:
        config: 配置字典
        logger: 日志对象

    Returns:
        success_count: 成功处理的点云数量
    """
    if logger is None:
        logger = setup_logger('stage3')

    logger.info("=" * 60)
    logger.info("Stage 3: Reflectance -> Pointcloud Intensity")
    logger.info("=" * 60)

    # 解析配置
    dataset_cfg = config['dataset']
    stage3_cfg = config['stage3']

    work_dir = dataset_cfg['work_dir']
    stage2_dir = work_dir / "stage2_reflectance"
    stage3_dir = work_dir / "stage3_pointcloud"
    stage3_dir.mkdir(parents=True, exist_ok=True)

    # 参数
    merge_strategy = stage3_cfg.get('merge_strategy', 'max')
    save_format = stage3_cfg.get('save_format', 'npy')
    save_ply = stage3_cfg.get('save_ply', False)
    skip_existing = stage3_cfg.get('skip_existing', True)
    normalize_intensity = stage3_cfg.get('normalize_intensity', False)
    save_gt_pointcloud = stage3_cfg.get('save_gt_pointcloud', True)  # 保存GT点云用于评估

    # 初始化数据集
    logger.info(f"加载数据集...")
    dataset = WaymoDatasetWrapper(
        image_dir=dataset_cfg['image_dir'],
        pointcloud_dir=dataset_cfg['pointcloud_dir'],
        calib_dir=dataset_cfg.get('calib_dir'),
        start_frame=dataset_cfg.get('start_frame', 0),
        num_frames=dataset_cfg.get('num_frames', None),
        step=dataset_cfg.get('step', 1),
    )

    # GT点云保存目录
    gt_dir = stage3_dir / "gt_pointcloud"
    if save_gt_pointcloud:
        gt_dir.mkdir(parents=True, exist_ok=True)

    # 处理每一帧
    success_count = 0
    total_count = len(dataset.frames)

    for frame_id in dataset.frames:
        # 输出路径
        output_npy = stage3_dir / f"frame_{frame_id:06d}_pseudo_intensity.npy"
        output_ply = stage3_dir / f"frame_{frame_id:06d}_pseudo_intensity.ply"
        gt_npy = gt_dir / f"frame_{frame_id:06d}_gt_intensity.npy"
        gt_ply = gt_dir / f"frame_{frame_id:06d}_gt_intensity.ply"

        # 检查是否需要保存GT点云（即使伪强度已存在也要保存GT）
        need_save_gt = save_gt_pointcloud and not gt_npy.exists()

        # 跳过已存在（但如果需要保存GT则不跳过）
        if skip_existing and output_npy.exists() and not need_save_gt:
            logger.debug(f"跳过已存在: {output_npy.name}")
            success_count += 1
            continue

        try:
            # 加载点云
            points = dataset.load_pointcloud(frame_id)

            # 加载雷达标定（需要提前加载以便保存变换矩阵）
            lidar_calib = dataset.load_lidar_calibration(frame_id)
            ego2world = lidar_calib['ego2world']

            # 保存GT点云（原始强度）用于后续评估
            # 同时保存 ego2world 变换矩阵，用于评估时将世界坐标转回lidar坐标
            if need_save_gt:
                # 使用 npz 格式保存点云和变换矩阵
                gt_npz = gt_dir / f"frame_{frame_id:06d}_gt_intensity.npz"
                np.savez(str(gt_npz), points=points, ego2world=ego2world)
                # 同时保存为 npy 格式（只有点云，用于兼容性）
                save_pointcloud(points, gt_npy, format='npy')
                if save_ply:
                    save_pointcloud(points, gt_ply, format='ply')
                logger.info(f"保存GT点云: {gt_npz.name} (含变换矩阵)")

            # 如果伪强度已存在，只需要保存GT就可以跳过后续处理
            if skip_existing and output_npy.exists():
                success_count += 1
                continue

            # 收集每个相机的反射率图像和标定
            frame_info = dataset.get_frame_info(frame_id)
            reflectance_images = []
            intrinsics = []
            extrinsics = []

            for cam_id in frame_info['cameras']:
                # 反射率图像路径
                ref_path = stage2_dir / f"frame_{frame_id:06d}_cam{cam_id}_r.png"
                if not ref_path.exists():
                    logger.warning(f"反射率图像不存在: {ref_path.name}")
                    continue

                # 加载反射率图像 (灰度)
                ref_img = load_image(ref_path, color_mode='GRAY')

                # 加载标定
                calib = dataset.load_calibration(frame_id, cam_id)

                reflectance_images.append(ref_img)
                intrinsics.append(calib['intrinsic'])
                extrinsics.append(calib['extrinsic'])

            if len(reflectance_images) == 0:
                logger.error(f"帧 {frame_id:06d} 没有可用的反射率图像")
                continue

            # 融合多相机强度
            logger.debug(f"处理帧 {frame_id:06d}: {len(reflectance_images)} 个相机")
            points_with_intensity = merge_multi_camera_intensity(
                points=points,
                reflectance_images=reflectance_images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                ego2world=ego2world,
                merge_strategy=merge_strategy,
                normalize=normalize_intensity,
            )

            # 保存为NPY格式
            save_pointcloud(points_with_intensity, output_npy, format='npy')

            # 可选: 保存为PLY格式用于可视化
            if save_ply:
                save_pointcloud(points_with_intensity, output_ply, format='ply')

            success_count += 1
            if success_count % 10 == 0:
                logger.info(f"已处理: {success_count}/{total_count}")

        except Exception as e:
            logger.error(f"处理失败 frame_{frame_id:06d}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue

    logger.info(f"Stage 3 完成: {success_count}/{total_count} 成功")
    logger.info(f"输出目录: {stage3_dir}")

    return success_count


if __name__ == '__main__':
    import argparse
    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description='Stage 3: Reflectance -> Pointcloud')
    parser.add_argument('config', type=str, help='配置文件路径 (YAML)')
    parser.add_argument('--frames', type=str, help='处理特定帧,格式: 1,2,3 或 1-10')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置日志
    log_dir = config['dataset']['work_dir'] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('stage3', log_file=log_dir / 'stage3.log')

    # 运行Stage 3
    process_stage3(config, logger)
