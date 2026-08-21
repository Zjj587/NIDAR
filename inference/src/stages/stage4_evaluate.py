"""
Stage 4: 评估生成的伪强度点云
对比GT点云和预测点云的强度差异，生成可视化对比图和评估指标
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import json

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.intensity_evaluator import (
    IntensityEvaluator,
    load_npy_as_points,
    load_ply_as_points_with_intensity,
    create_evaluator_from_config,
)
from src.utils.logger import setup_logger


def process_stage4(config: Dict, logger=None, use_remapped: bool = False):
    """
    Stage 4 主处理函数 - 评估

    Args:
        config: 配置字典
        logger: 日志对象
        use_remapped: 是否使用 Stage 3.5 重映射后的点云进行评估

    Returns:
        success_count: 成功评估的帧数
    """
    if logger is None:
        logger = setup_logger('stage4')

    logger.info("=" * 60)
    logger.info("Stage 4: Intensity Evaluation")
    logger.info("=" * 60)

    # 解析配置
    dataset_cfg = config['dataset']
    eval_cfg = config.get('evaluation', {})

    work_dir = Path(dataset_cfg['work_dir'])
    stage3_dir = work_dir / "stage3_pointcloud"
    stage3_5_dir = work_dir / "stage3_5_remapped"
    gt_dir = stage3_dir / "gt_pointcloud"
    eval_dir = work_dir / "stage4_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 确定预测点云来源目录
    # 如果 use_remapped=True 且 stage3_5 目录存在，则使用重映射后的点云
    if use_remapped and stage3_5_dir.exists():
        pred_source_dir = stage3_5_dir
        pred_pattern = "dis_learned_*_pseudo_intensity.npy"
        logger.info(f"使用重映射点云: {stage3_5_dir}")
    else:
        pred_source_dir = stage3_dir
        pred_pattern = "frame_*_pseudo_intensity.npy"
        if use_remapped:
            logger.warning(f"重映射目录不存在: {stage3_5_dir}，使用原始点云")

    # 评估参数
    dataset_type = dataset_cfg.get('type', 'waymo').lower()
    waymo_calib_json = eval_cfg.get('waymo_calib_json')
    skip_existing = eval_cfg.get('skip_existing', True)
    remove_invalid_columns = eval_cfg.get('remove_invalid_columns', None)

    # 解析标定文件路径（相对路径转绝对路径）
    if waymo_calib_json and not Path(waymo_calib_json).is_absolute():
        # 相对于项目根目录
        project_root = Path(__file__).parent.parent.parent
        waymo_calib_json = str(project_root / waymo_calib_json)

    if waymo_calib_json:
        logger.info(f"使用Waymo标定文件: {waymo_calib_json}")

    # 创建评估器
    logger.info(f"数据集类型: {dataset_type}")
    evaluator = IntensityEvaluator(
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

    # 收集GT-Pred对
    gt_pred_pairs = []

    # 查找所有预测点云
    pred_files = sorted(pred_source_dir.glob(pred_pattern))

    for pred_path in pred_files:
        # 提取帧ID
        name = pred_path.stem  # frame_000000_pseudo_intensity 或 dis_learned_frame_000000_pseudo_intensity
        # 处理可能的 dis_learned_ 前缀
        name_clean = name.replace('dis_learned_', '')
        frame_id = name_clean.replace('frame_', '').replace('sample_', '').replace('_pseudo_intensity', '')

        # 对应的GT路径（优先使用npz格式，包含变换矩阵）
        gt_npz = gt_dir / f"frame_{frame_id}_gt_intensity.npz"
        gt_npy = gt_dir / f"frame_{frame_id}_gt_intensity.npy"

        if gt_npz.exists():
            gt_path = gt_npz
        elif gt_npy.exists():
            gt_path = gt_npy
        else:
            logger.warning(f"GT文件不存在: frame_{frame_id}")
            continue

        gt_pred_pairs.append((str(gt_path), str(pred_path), frame_id))

    if len(gt_pred_pairs) == 0:
        logger.error("没有找到可评估的GT-Pred对")
        logger.info(f"请确保stage3已运行，且配置了save_gt_pointcloud: true")
        return 0

    logger.info(f"找到 {len(gt_pred_pairs)} 对GT-Pred点云")

    # 评估每一帧
    all_metrics = {}
    success_count = 0

    for gt_path, pred_path, frame_id in gt_pred_pairs:
        sample_name = f"frame_{frame_id}"
        output_dir = eval_dir / sample_name

        # 跳过已存在
        metrics_file = output_dir / f"{sample_name}_metrics.json"
        if skip_existing and metrics_file.exists():
            logger.debug(f"跳过已存在: {sample_name}")
            # 加载已有指标
            with open(metrics_file, 'r') as f:
                all_metrics[sample_name] = json.load(f)
            success_count += 1
            continue

        logger.info(f"评估: {sample_name}")

        try:
            # 加载点云
            ego2world = None
            if gt_path.endswith('.npz'):
                # 从NPZ文件加载点云和变换矩阵
                gt_data = np.load(gt_path)
                gt_points = gt_data['points']
                if 'ego2world' in gt_data:
                    ego2world = gt_data['ego2world']
                    logger.debug(f"加载了ego2world变换矩阵")
            else:
                gt_points = load_npy_as_points(gt_path)

            pred_points = load_npy_as_points(pred_path)

            # 确定是否移除无效列
            if remove_invalid_columns is None:
                # 默认: Waymo需要移除无效列，nuScenes不需要
                do_remove = (dataset_type == 'waymo')
            else:
                do_remove = remove_invalid_columns

            # 评估（传入ego2world用于坐标变换）
            metrics = evaluator.evaluate(
                gt_points, pred_points,
                str(output_dir),
                sample_name,
                remove_invalid_columns=do_remove,
                ego2world=ego2world,
            )

            all_metrics[sample_name] = metrics
            success_count += 1

        except Exception as e:
            logger.error(f"评估失败 {sample_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            all_metrics[sample_name] = {'error': str(e)}

    # 计算平均指标
    avg_metrics = _compute_average_metrics(all_metrics)

    # 保存汇总
    summary = {
        'dataset_type': dataset_type,
        'num_samples': len(gt_pred_pairs),
        'num_success': success_count,
        'individual': all_metrics,
        'average': avg_metrics,
    }

    summary_path = eval_dir / 'evaluation_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # 打印汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("评估汇总")
    logger.info("=" * 60)
    logger.info(f"总样本数: {len(gt_pred_pairs)}")
    logger.info(f"成功评估: {success_count}")
    logger.info("")
    logger.info("平均指标:")
    for key in ['rmse', 'mae', 'medae', 'psnr', 'ssim', 'lpips']:
        if key in avg_metrics and not np.isnan(avg_metrics[key]):
            logger.info(f"  {key.upper():8s}: {avg_metrics[key]:.4f}")

    # 打印 PSNR 最高的 Top-5 帧
    logger.info("")
    logger.info("-" * 40)
    logger.info("PSNR Top-5 帧:")
    logger.info("-" * 40)
    _print_top_k_psnr(all_metrics, logger, k=5)

    logger.info("=" * 60)
    logger.info(f"评估结果保存到: {eval_dir}")
    logger.info(f"汇总文件: {summary_path}")

    return success_count


def _print_top_k_psnr(all_metrics: Dict, logger, k: int = 5):
    """打印 PSNR 最高的 k 帧"""
    # 筛选有效的 (sample_name, psnr) 对
    valid_samples = []
    for sample_name, metrics in all_metrics.items():
        if 'error' not in metrics and 'psnr' in metrics:
            psnr_val = metrics['psnr']
            if not np.isnan(psnr_val):
                valid_samples.append((sample_name, psnr_val, metrics))

    if not valid_samples:
        logger.info("  没有有效的 PSNR 数据")
        return

    # 按 PSNR 降序排序
    valid_samples.sort(key=lambda x: x[1], reverse=True)

    # 打印 Top-k
    for rank, (sample_name, psnr_val, metrics) in enumerate(valid_samples[:k], 1):
        ssim_val = metrics.get('ssim', float('nan'))
        mae_val = metrics.get('mae', float('nan'))
        logger.info(f"  #{rank}: {sample_name} | PSNR={psnr_val:.4f} | SSIM={ssim_val:.4f} | MAE={mae_val:.4f}")


def _compute_average_metrics(all_metrics: Dict) -> Dict:
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


if __name__ == '__main__':
    import argparse
    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description='Stage 4: Intensity Evaluation')
    parser.add_argument('config', type=str, help='配置文件路径 (YAML)')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置日志
    work_dir = Path(config['dataset']['work_dir'])
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('stage4', log_file=log_dir / 'stage4.log')

    # 运行Stage 4
    process_stage4(config, logger)
