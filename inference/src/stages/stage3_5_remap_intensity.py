"""
Stage 3.5: 使用分位数映射模型重映射点云强度
将 Stage 3 输出的伪强度通过学习的分位数映射进行重分布，使其更接近真实雷达强度分布
"""
import sys
import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pickle

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

# 默认分位数模型目录
DEFAULT_QUANTILE_MODEL_DIR = Path(
    os.environ.get("NIDAR_QUANTILE_MODEL_DIR", str(project_root / "quantile_model"))
)


def load_quantile_model(model_path: Path) -> dict:
    """加载分位数映射模型

    Args:
        model_path: 模型文件路径 (.pkl)

    Returns:
        模型字典，包含 quantiles 和 ref_quantiles
    """
    if not model_path.exists():
        raise FileNotFoundError(f"分位数模型不存在: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if model.get('method') != 'quantile_mapping':
        raise ValueError(f"不支持的模型类型: {model.get('method')}")

    return model


def apply_quantile_mapping(intensity: np.ndarray, model: dict) -> np.ndarray:
    """应用分位数映射

    Args:
        intensity: 原始强度值 (N,)
        model: 分位数模型字典

    Returns:
        重映射后的强度值 (N,)
    """
    # 保护无效强度点 (0值)
    # 对于Waymo等数据集，未被相机覆盖的点强度为0，应保持为0，不参与映射
    valid_mask = intensity > 1e-6
    if valid_mask.sum() == 0:
        return intensity  # 如果全是0，直接返回

    valid_intensity = intensity[valid_mask]

    # 计算有效点的分位数
    source_quantiles = np.percentile(valid_intensity, model['quantiles'])

    # 使用参考分位数进行映射
    mapped_valid = np.interp(valid_intensity, source_quantiles, model['ref_quantiles'])
    mapped_valid = np.clip(mapped_valid, 0.0, 1.0)

    # 填充回结果数组
    remapped = intensity.copy()
    remapped[valid_mask] = mapped_valid

    return remapped.astype(np.float32)


def remap_pointcloud_intensity(
    points: np.ndarray,
    model: dict,
    intensity_col: int = 3
) -> np.ndarray:
    """重映射点云强度

    Args:
        points: 点云数组 (N, 4+)，假设第4列为强度
        model: 分位数模型
        intensity_col: 强度列索引，默认为3

    Returns:
        重映射后的点云数组
    """
    points_out = points.copy()
    intensity = points[:, intensity_col].astype(np.float32)

    # 应用分位数映射
    remapped = apply_quantile_mapping(intensity, model)
    points_out[:, intensity_col] = remapped

    return points_out


def save_remapped_pointcloud(points: np.ndarray, output_path: Path, format: str = 'npy'):
    """保存重映射后的点云

    Args:
        points: 点云数组 (N, 4+)
        output_path: 输出路径
        format: 保存格式 ('npy' 或 'ply')
    """
    if format == 'npy':
        np.save(str(output_path), points)
    elif format == 'ply':
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            raise RuntimeError("保存PLY格式需要 plyfile: pip install plyfile")

        # 构建PLY数据
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]
        if points.shape[1] > 4:
            # 可能有额外字段如 elongation
            for i in range(4, points.shape[1]):
                dtype.append((f'field_{i}', 'f4'))

        structured = np.zeros(len(points), dtype=dtype)
        structured['x'] = points[:, 0]
        structured['y'] = points[:, 1]
        structured['z'] = points[:, 2]
        structured['intensity'] = points[:, 3]
        for i in range(4, points.shape[1]):
            structured[f'field_{i}'] = points[:, i]

        el = PlyElement.describe(structured, 'vertex')
        PlyData([el], text=False).write(str(output_path))
    else:
        raise ValueError(f"不支持的格式: {format}")


def get_quantile_model_path(dataset_type: str, model_dir: Optional[Path] = None) -> Path:
    """根据数据集类型获取对应的分位数模型路径

    Args:
        dataset_type: 数据集类型 ('waymo', 'nuscenes', 或 'kitti360')
        model_dir: 模型目录，默认为 DEFAULT_QUANTILE_MODEL_DIR

    Returns:
        模型文件路径
    """
    if model_dir is None:
        model_dir = DEFAULT_QUANTILE_MODEL_DIR

    model_dir = Path(model_dir)

    # 根据数据集类型选择模型
    # 命名规则: w = waymo, n = nuscenes, k = kitti360, 1 = 其他/通用
    if dataset_type.lower() == 'waymo':
        model_path = model_dir / "wintensity_mapping_quantile_model.pkl"
    elif dataset_type.lower() == 'nuscenes':
        model_path = model_dir / "nintensity_mapping_quantile_model.pkl"
    elif dataset_type.lower() == 'kitti360':
        model_path = model_dir / "kintensity_mapping_quantile_model.pkl"
    else:
        # 默认使用通用模型
        model_path = model_dir / "1intensity_mapping_quantile_model.pkl"

    return model_path


def process_stage3_5(config: Dict, logger=None, dataset_type: str = 'waymo'):
    """
    Stage 3.5 主处理函数 - 强度重映射

    Args:
        config: 配置字典
        logger: 日志对象
        dataset_type: 数据集类型 ('waymo', 'nuscenes', 或 'kitti360')

    Returns:
        success_count: 成功处理的点云数量
    """
    if logger is None:
        logger = setup_logger('stage3_5')

    logger.info("=" * 60)
    logger.info("Stage 3.5: Intensity Remap (Quantile Mapping)")
    logger.info("=" * 60)

    # 解析配置
    dataset_cfg = config['dataset']
    stage3_5_cfg = config.get('stage3_5', {})

    work_dir = Path(dataset_cfg['work_dir'])
    stage3_dir = work_dir / "stage3_pointcloud"
    stage3_5_dir = work_dir / "stage3_5_remapped"
    stage3_5_dir.mkdir(parents=True, exist_ok=True)

    # 参数
    skip_existing = stage3_5_cfg.get('skip_existing', True)
    save_ply = stage3_5_cfg.get('save_ply', False)
    model_dir = stage3_5_cfg.get('quantile_model_dir', str(DEFAULT_QUANTILE_MODEL_DIR))
    model_path_override = stage3_5_cfg.get('quantile_model_path', None)

    # 确定模型路径
    if model_path_override:
        model_path = Path(model_path_override)
    else:
        model_path = get_quantile_model_path(dataset_type, Path(model_dir))

    logger.info(f"数据集类型: {dataset_type}")
    logger.info(f"分位数模型: {model_path}")

    # 加载模型
    try:
        model = load_quantile_model(model_path)
        logger.info(f"模型加载成功，分位数数量: {model.get('n_quantiles', 'N/A')}")
    except Exception as e:
        logger.error(f"加载分位数模型失败: {e}")
        return 0

    # 查找所有 Stage 3 输出的点云
    # 支持 frame_ 和 sample_ 命名
    pred_files = list(stage3_dir.glob("frame_*_pseudo_intensity.npy"))
    pred_files.extend(stage3_dir.glob("sample_*_pseudo_intensity.npy"))
    pred_files = sorted(pred_files)

    if len(pred_files) == 0:
        logger.error(f"在 {stage3_dir} 中未找到伪强度点云文件")
        return 0

    logger.info(f"找到 {len(pred_files)} 个待处理点云")

    # 处理每个点云
    success_count = 0
    total_count = len(pred_files)

    for pred_path in pred_files:
        # 构建输出路径 (使用 dis_learned_ 前缀以区分)
        stem = pred_path.stem  # e.g., frame_000000_pseudo_intensity
        output_npy = stage3_5_dir / f"dis_learned_{stem}.npy"
        output_ply = stage3_5_dir / f"dis_learned_{stem}.ply"

        # 跳过已存在
        if skip_existing and output_npy.exists():
            logger.debug(f"跳过已存在: {output_npy.name}")
            success_count += 1
            continue

        try:
            # 加载点云
            points = np.load(str(pred_path))

            if points.shape[1] < 4:
                logger.warning(f"点云格式异常，列数不足: {pred_path.name}")
                continue

            # 重映射强度
            remapped_points = remap_pointcloud_intensity(points, model)

            # 保存
            save_remapped_pointcloud(remapped_points, output_npy, format='npy')

            if save_ply:
                save_remapped_pointcloud(remapped_points, output_ply, format='ply')

            success_count += 1

            if success_count % 20 == 0:
                logger.info(f"已处理: {success_count}/{total_count}")

        except Exception as e:
            logger.error(f"处理失败 {pred_path.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue

    logger.info(f"Stage 3.5 完成: {success_count}/{total_count} 成功")
    logger.info(f"输出目录: {stage3_5_dir}")

    return success_count


def process_stage3_5_waymo(config: Dict, logger=None):
    """Waymo 数据集的 Stage 3.5"""
    return process_stage3_5(config, logger, dataset_type='waymo')


def process_stage3_5_nuscenes(config: Dict, logger=None):
    """nuScenes 数据集的 Stage 3.5"""
    return process_stage3_5(config, logger, dataset_type='nuscenes')


if __name__ == '__main__':
    import argparse
    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description='Stage 3.5: Intensity Remap')
    parser.add_argument('config', type=str, help='配置文件路径 (YAML)')
    parser.add_argument('--dataset-type', type=str, default='waymo',
                        choices=['waymo', 'nuscenes'],
                        help='数据集类型')
    parser.add_argument('--model-path', type=str, default=None,
                        help='覆盖配置中的分位数模型路径')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 覆盖模型路径
    if args.model_path:
        if 'stage3_5' not in config:
            config['stage3_5'] = {}
        config['stage3_5']['quantile_model_path'] = args.model_path

    # 设置日志
    log_dir = Path(config['dataset']['work_dir']) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('stage3_5', log_file=log_dir / 'stage3_5.log')

    # 运行 Stage 3.5
    process_stage3_5(config, logger, dataset_type=args.dataset_type)
