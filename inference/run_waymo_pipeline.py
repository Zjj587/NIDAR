#!/usr/bin/env python3
"""
Waymo 伪强度管道主程序
完整运行流程: RGB -> 伪NIR -> 反射率 -> 点云强度 -> 强度重映射 -> 评估
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, validate_config
from src.utils.logger import setup_logger
from src.stages.stage1_rgb_to_pseudo_nir import process_stage1
from src.stages.stage2_pseudo_nir_to_reflectance import process_stage2
from src.stages.stage2_deep_intrinsic import run_stage2_deep
from src.stages.stage3_reflectance_to_pointcloud import process_stage3
from src.stages.stage3_5_remap_intensity import process_stage3_5_waymo
from src.stages.stage4_evaluate import process_stage4


def main():
    parser = argparse.ArgumentParser(
        description='Waymo 伪强度生成管道',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行完整管道 (1-3)
  python run_waymo_pipeline.py configs/my_waymo_demo.yaml

  # 只运行某个阶段
  python run_waymo_pipeline.py configs/my_waymo_demo.yaml --stages 1
  python run_waymo_pipeline.py configs/my_waymo_demo.yaml --stages 2,3

  # 运行论文路线：生成 + R投影 + 重映射 + 评估
  python run_waymo_pipeline.py configs/my_waymo_demo.yaml --stages 1,2,3,3.5,4 --use-deep --keep-remap-with-deep

  # 使用深度内在分解网络（替代传统 decompose.py）
  python run_waymo_pipeline.py configs/my_waymo_demo.yaml --stages 1,2,3,4 --use-deep

  # 只运行重映射和评估（假设已有stage3输出）
  python run_waymo_pipeline.py configs/my_waymo_demo.yaml --stages 3.5,4

  # 跳过某个阶段
  python run_waymo_pipeline.py configs/my_waymo_demo.yaml --skip-stages 1

阶段说明:
  1   : RGB -> Pseudo-NIR
  2   : Pseudo-NIR -> Reflectance (传统方法或Deep网络)
  3   : Reflectance -> Pointcloud Intensity
  3.5 : Intensity Remap (Quantile Mapping)，论文R+remap路线需要
  4   : Evaluation (对比重映射后的强度与GT)
        """
    )

    parser.add_argument(
        'config',
        type=str,
        help='配置文件路径 (YAML格式)'
    )

    parser.add_argument(
        '--stages',
        type=str,
        default='1,2,3',
        help='要运行的阶段,逗号分隔 (默认: 1,2,3, 可选3.5为强度重映射, 4为评估阶段)'
    )

    parser.add_argument(
        '--skip-stages',
        type=str,
        default='',
        help='要跳过的阶段,逗号分隔'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别 (默认: INFO)'
    )

    parser.add_argument(
        '--use-deep',
        action='store_true',
        help='使用深度内在分解网络替代传统方法 (Stage2使用Deep模型；默认保持旧行为跳过Stage3.5)'
    )

    parser.add_argument(
        '--keep-remap-with-deep',
        action='store_true',
        help='Deep模式下仍保留Stage3.5；用于stage2_deep.use_i_hat_as_r=false的R+remap论文路线'
    )

    parser.add_argument(
        '--deep-checkpoint',
        type=str,
        default=None,
        help='Deep模型检查点路径 (覆盖配置文件中的值)'
    )

    args = parser.parse_args()

    # 解析阶段参数 (支持整数和小数如 3.5)
    def parse_stage(s):
        s = s.strip()
        if not s:
            return None
        return float(s) if '.' in s else int(s)

    stages_to_run = set(parse_stage(s) for s in args.stages.split(',') if s.strip())
    stages_to_run.discard(None)

    if args.skip_stages:
        stages_to_skip = set(parse_stage(s) for s in args.skip_stages.split(',') if s.strip())
        stages_to_skip.discard(None)
        stages_to_run -= stages_to_skip

    # 加载配置
    print(f"加载配置: {args.config}")
    config = load_config(args.config)

    # 验证配置
    try:
        validate_config(config)
    except ValueError as e:
        print(f"配置错误: {e}")
        sys.exit(1)

    # 设置日志
    work_dir = config['dataset']['work_dir']
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    import logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logger('pipeline', log_file=log_dir / 'pipeline.log', level=log_level)

    logger.info("=" * 80)
    logger.info("Waymo 伪强度生成管道")
    logger.info("=" * 80)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"工作目录: {work_dir}")
    logger.info(f"运行阶段: {sorted(stages_to_run)}")
    logger.info(f"使用Deep模式: {args.use_deep}")
    logger.info("")

    # 如果使用 deep 模式，默认保持旧行为；显式开关用于R+remap论文路线。
    if args.use_deep and not args.keep_remap_with_deep:
        # Deep 模式默认跳过 stage 3.5
        if 3.5 in stages_to_run:
            logger.info("Deep模式: 自动跳过 Stage 3.5 (使用 --keep-remap-with-deep 可保留)")
            stages_to_run.discard(3.5)
    elif args.use_deep and 3.5 in stages_to_run:
        logger.info("Deep模式: 保留 Stage 3.5，用于R+remap路线")

    # 运行各阶段
    results = {}

    try:
        if 1 in stages_to_run:
            count = process_stage1(config, logger)
            results['stage1'] = count
            logger.info(f"Stage 1 完成: 处理了 {count} 张图像\n")

        if 2 in stages_to_run:
            if args.use_deep:
                # 使用 Deep Intrinsic 网络
                logger.info("Stage 2: 使用深度内在分解网络")
                result = run_stage2_deep(
                    config=config,
                    work_dir=str(work_dir),
                    checkpoint_path=args.deep_checkpoint,
                    skip_existing=config.get('stage2', {}).get('skip_existing', False)
                )
                count = result.get('processed', 0)
            else:
                # 使用传统方法
                count = process_stage2(config, logger)
            results['stage2'] = count
            logger.info(f"Stage 2 完成: 处理了 {count} 张图像\n")

        if 3 in stages_to_run:
            count = process_stage3(config, logger)
            results['stage3'] = count
            logger.info(f"Stage 3 完成: 处理了 {count} 个点云\n")

        if 3.5 in stages_to_run:
            count = process_stage3_5_waymo(config, logger)
            results['stage3_5'] = count
            logger.info(f"Stage 3.5 完成: 重映射了 {count} 个点云\n")

        if 4 in stages_to_run:
            # 判断评估哪个目录：
            # - 如果使用 deep 模式且未保留Stage3.5，直接评估 stage3 的输出
            # - 如果运行了 stage 3.5 或存在重映射目录，则评估重映射后的点云
            if args.use_deep and not args.keep_remap_with_deep:
                use_remapped = False
                logger.info("Stage 4 将评估 Deep 模式生成的点云 (stage3_pointcloud)")
            else:
                use_remapped = 3.5 in stages_to_run or (work_dir / "stage3_5_remapped").exists()
                if use_remapped:
                    logger.info("Stage 4 将评估重映射后的点云 (stage3_5_remapped)")
            count = process_stage4(config, logger, use_remapped=use_remapped)
            results['stage4'] = count
            logger.info(f"Stage 4 完成: 评估了 {count} 帧\n")

    except KeyboardInterrupt:
        logger.warning("\n用户中断,停止处理")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n管道执行失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("管道执行完成!")
    logger.info("=" * 80)
    for stage, count in sorted(results.items()):
        logger.info(f"  {stage}: {count} 项")
    logger.info("")
    logger.info(f"输出目录: {work_dir}")
    logger.info(f"  - Stage 1: {work_dir / 'stage1_pseudo_nir'}")
    logger.info(f"  - Stage 2: {work_dir / 'stage2_reflectance'}")
    logger.info(f"  - Stage 3: {work_dir / 'stage3_pointcloud'}")
    logger.info(f"  - Stage 3.5: {work_dir / 'stage3_5_remapped'}")
    logger.info(f"  - Stage 4: {work_dir / 'stage4_evaluation'}")
    logger.info("")
    logger.info(f"日志文件: {log_dir / 'pipeline.log'}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
