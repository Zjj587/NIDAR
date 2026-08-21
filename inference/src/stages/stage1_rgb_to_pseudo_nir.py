"""
Stage 1: RGB 图像转换为伪 NIR 图像
使用 STN 模型进行转换
"""
import sys
from pathlib import Path
from typing import Dict
import numpy as np

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.stn_wrapper import STNWrapper
from src.datasets.waymo_dataset import WaymoDatasetWrapper
from src.utils.image_utils import load_image, save_image, convert_bit_depth
from src.utils.logger import setup_logger


def process_stage1(config: Dict, logger=None):
    """
    Stage 1 主处理函数

    Args:
        config: 配置字典
        logger: 日志对象

    Returns:
        success_count: 成功处理的图像数量
    """
    if logger is None:
        logger = setup_logger('stage1')

    logger.info("=" * 60)
    logger.info("Stage 1: RGB -> Pseudo-NIR")
    logger.info("=" * 60)

    # 解析配置
    dataset_cfg = config['dataset']
    stage1_cfg = config['stage1']

    image_dir = dataset_cfg['image_dir']
    work_dir = dataset_cfg['work_dir']
    stage1_dir = work_dir / "stage1_pseudo_nir"
    stage1_dir.mkdir(parents=True, exist_ok=True)

    # 参数
    checkpoint_path = Path(stage1_cfg['checkpoint'])
    exp_ratio = stage1_cfg.get('exp_ratio', 0.5)
    bit_depth = stage1_cfg.get('bit_depth', 8)
    use_gpu = stage1_cfg.get('use_gpu', True)
    skip_existing = stage1_cfg.get('skip_existing', True)

    # 初始化数据集
    logger.info(f"加载数据集: {image_dir}")
    dataset = WaymoDatasetWrapper(
        image_dir=image_dir,
        pointcloud_dir=dataset_cfg['pointcloud_dir'],
        calib_dir=dataset_cfg.get('calib_dir', image_dir),
        start_frame=dataset_cfg.get('start_frame', 0),
        num_frames=dataset_cfg.get('num_frames', None),
        step=dataset_cfg.get('step', 1),
    )

    # 初始化STN模型
    logger.info(f"加载STN模型: {checkpoint_path}")
    device = 'cuda' if use_gpu else 'cpu'
    stn = STNWrapper(
        checkpoint_path=checkpoint_path,
        image_shape=(384, 1248),  # 根据实际训练尺寸调整
        device=device
    )

    # 处理每一帧的每个相机
    success_count = 0
    total_count = 0

    for frame_id in dataset.frames:
        frame_info = dataset.get_frame_info(frame_id)

        for cam_id in frame_info['cameras']:
            total_count += 1

            # 输出路径
            output_path = stage1_dir / f"frame_{frame_id:06d}_cam{cam_id}_nir.png"

            # 跳过已存在
            if skip_existing and output_path.exists():
                logger.debug(f"跳过已存在: {output_path.name}")
                success_count += 1
                continue

            try:
                # 加载RGB图像
                rgb_img = dataset.load_image(frame_id, cam_id, color_mode='RGB')

                # STN推理
                pseudo_nir = stn.process_image(rgb_img, exp_ratio=exp_ratio)

                # 位深度转换
                if bit_depth == 16:
                    pseudo_nir = convert_bit_depth(pseudo_nir, 8, 16)

                # 保存
                save_image(pseudo_nir, output_path, color_mode='GRAY')

                success_count += 1
                if total_count % 50 == 0:
                    logger.info(f"已处理: {total_count}/{len(dataset.frames) * 5} "
                              f"({success_count} 成功)")

            except Exception as e:
                logger.error(f"处理失败 frame_{frame_id:06d}_cam{cam_id}: {e}")
                continue

    logger.info(f"Stage 1 完成: {success_count}/{total_count} 成功")
    logger.info(f"输出目录: {stage1_dir}")

    return success_count


if __name__ == '__main__':
    import argparse
    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description='Stage 1: RGB -> Pseudo-NIR')
    parser.add_argument('config', type=str, help='配置文件路径 (YAML)')
    parser.add_argument('--frames', type=str, help='处理特定帧,格式: 1,2,3 或 1-10')
    parser.add_argument('--cameras', type=str, help='处理特定相机,格式: 1,2,3')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置日志
    log_dir = config['dataset']['work_dir'] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('stage1', log_file=log_dir / 'stage1.log')

    # 运行Stage 1
    process_stage1(config, logger)
