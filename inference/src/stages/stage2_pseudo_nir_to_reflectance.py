"""
Stage 2: 伪 NIR 图像分解为反射率 (Reflectance)
使用本征图像分解算法
"""
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.waymo_dataset import WaymoDatasetWrapper
from src.utils.logger import setup_logger


def process_stage2(config: Dict, logger=None):
    """
    Stage 2 主处理函数

    Args:
        config: 配置字典
        logger: 日志对象

    Returns:
        success_count: 成功处理的图像数量
    """
    if logger is None:
        logger = setup_logger('stage2')

    logger.info("=" * 60)
    logger.info("Stage 2: Pseudo-NIR -> Reflectance")
    logger.info("=" * 60)

    # 解析配置
    dataset_cfg = config['dataset']
    stage2_cfg = config['stage2']

    work_dir = dataset_cfg['work_dir']
    stage1_dir = work_dir / "stage1_pseudo_nir"
    stage2_dir = work_dir / "stage2_reflectance"
    stage2_dir.mkdir(parents=True, exist_ok=True)

    # 参数
    use_linear = stage2_cfg.get('linear', False)
    skip_existing = stage2_cfg.get('skip_existing', True)
    decompose_script = stage2_cfg.get('decompose_script', None)

    # 查找原始decompose.py脚本
    if decompose_script is None:
        # 尝试从环境中自动发现
        possible_paths = [
            Path(os.path.expandvars("${PROJECT_ROOT}/third_party/Intrinsic-Image-Decomposition/decompose.py")),
            project_root.parent / "Intrinsic-Image-Decomposition" / "decompose.py",
        ]
        for p in possible_paths:
            if p.exists():
                decompose_script = p
                break
    else:
        decompose_script = Path(decompose_script)

    if decompose_script is None or not decompose_script.exists():
        logger.error("找不到decompose.py脚本!")
        logger.error("请在配置文件中设置 stage2.decompose_script 路径")
        logger.error("或将 Intrinsic-Image-Decomposition 项目放在上层目录")
        return 0

    logger.info(f"使用分解脚本: {decompose_script}")

    # 初始化数据集(只用于获取帧列表)
    dataset = WaymoDatasetWrapper(
        image_dir=dataset_cfg['image_dir'],
        pointcloud_dir=dataset_cfg['pointcloud_dir'],
        start_frame=dataset_cfg.get('start_frame', 0),
        num_frames=dataset_cfg.get('num_frames', None),
        step=dataset_cfg.get('step', 1),
    )

    # 处理每个伪NIR图像（只处理当前配置对应的帧）
    success_count = 0
    total_count = 0

    # 构建当前配置要处理的帧ID集合
    current_frame_ids = set(dataset.frames)

    for nir_path in sorted(stage1_dir.glob("frame_*_cam*_nir.png")):
        # 解析文件名获取frame_id
        stem = nir_path.stem  # frame_000158_cam1_nir
        parts = stem.split('_')
        try:
            frame_id = int(parts[1])
        except (IndexError, ValueError):
            continue

        # 只处理当前配置对应的帧
        if frame_id not in current_frame_ids:
            continue

        total_count += 1

        base_name = '_'.join(parts[:-1])  # frame_000158_cam1

        # 输出路径
        output_r = stage2_dir / f"{base_name}_r.png"
        output_s = stage2_dir / f"{base_name}_s.png"

        # 跳过已存在
        if skip_existing and output_r.exists():
            logger.debug(f"跳过已存在: {output_r.name}")
            success_count += 1
            continue

        try:
            # 构建命令
            cmd = [
                sys.executable,
                str(decompose_script),
                str(nir_path),
                '-r', str(output_r),
                '-s', str(output_s),
                '-q',  # quiet mode
            ]

            if use_linear:
                cmd.append('--linear')

            # 执行分解
            # 注意: Intrinsic-Image-Decomposition 算法本身比较耗时，
            # 单张大图在CPU上运行可能超过60秒。
            # 为避免频繁超时, 这里将超时时间放宽到10分钟。
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=6000  # 10分钟超时, 避免正常情况下被杀掉
            )

            if output_r.exists():
                success_count += 1
                if total_count % 50 == 0:
                    logger.info(f"已处理: {total_count} ({success_count} 成功)")
            else:
                logger.warning(f"分解完成但未生成输出: {base_name}")

        except subprocess.TimeoutExpired:
            logger.error(f"超时: {base_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"分解失败 {base_name}: {e}")
            if e.stderr:
                logger.debug(f"stderr: {e.stderr[:200]}")
        except Exception as e:
            logger.error(f"处理失败 {base_name}: {e}")

    logger.info(f"Stage 2 完成: {success_count}/{total_count} 成功")
    logger.info(f"输出目录: {stage2_dir}")

    return success_count


if __name__ == '__main__':
    import argparse
    from src.utils.config import load_config

    parser = argparse.ArgumentParser(description='Stage 2: Pseudo-NIR -> Reflectance')
    parser.add_argument('config', type=str, help='配置文件路径 (YAML)')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置日志
    log_dir = config['dataset']['work_dir'] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('stage2', log_file=log_dir / 'stage2.log')

    # 运行Stage 2
    process_stage2(config, logger)
