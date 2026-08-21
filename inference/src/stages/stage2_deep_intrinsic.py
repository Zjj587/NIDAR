#!/usr/bin/env python3
"""
Stage 2-Deep: 使用训练好的深度内在分解网络处理 Pseudo-NIR 图像
替代传统的 decompose.py 方法，直接使用 checkpoint 推理

输入: stage1_pseudo_nir 目录中的 Pseudo-NIR 图像
输出: stage2_reflectance 目录中的反射率 (R) 图像

使用方法:
    python -m src.stages.stage2_deep_intrinsic --config configs/waymo_example.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import cv2
import torch
import yaml

# 设置项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deep_intrinsic_wrapper import DeepIntrinsicWrapper


def setup_logging(level: str = "INFO"):
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_output_dir(config: Dict[str, Any], work_dir: Optional[str] = None) -> Path:
    """获取输出目录"""
    if work_dir:
        return Path(work_dir) / "stage2_reflectance"

    base_dir = Path(config.get('work_dir', 'output'))
    return base_dir / "stage2_reflectance"


def get_input_dir(config: Dict[str, Any], work_dir: Optional[str] = None) -> Path:
    """获取输入目录（stage1 输出）"""
    if work_dir:
        return Path(work_dir) / "stage1_pseudo_nir"

    base_dir = Path(config.get('work_dir', 'output'))
    return base_dir / "stage1_pseudo_nir"


def find_pseudo_nir_images(input_dir: Path) -> List[Path]:
    """查找所有 Pseudo-NIR 图像"""
    images = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
        images.extend(input_dir.glob(ext))
        images.extend(input_dir.glob(ext.upper()))

    # 按文件名排序
    images = sorted(images, key=lambda x: x.stem)
    return images


def load_image(image_path: Path) -> np.ndarray:
    """加载图像并转换为灰度浮点格式 [0, 1]"""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")

    # 转换为灰度图（Pseudo-NIR 是单通道）
    if len(img.shape) == 3 and img.shape[2] == 3:
        # 如果是 BGR 彩色图，转为灰度
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        # 如果是 BGRA，先转 BGR 再转灰度
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # 否则已经是灰度图，保持原样

    # 归一化到 [0, 1]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / img.max()

    return img


def save_reflectance(reflectance: np.ndarray, output_path: Path):
    """保存反射率图像"""
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果有三个通道，转换为单通道灰度
    if len(reflectance.shape) == 3 and reflectance.shape[2] == 3:
        reflectance_gray = 0.299 * reflectance[..., 0] + 0.587 * reflectance[..., 1] + 0.114 * reflectance[..., 2]
    elif len(reflectance.shape) == 2:
        reflectance_gray = reflectance
    else:
        reflectance_gray = reflectance.squeeze()

    # 规范化到 [0,1]：支持 uint8 (0-255), uint16 (0-65535), float
    if np.issubdtype(reflectance_gray.dtype, np.integer):
        # 判定位深
        if reflectance_gray.dtype == np.uint8:
            refl_f = reflectance_gray.astype(np.float32) / 255.0
        else:
            # 其他整数类型，尝试按 65535 归一化
            refl_f = reflectance_gray.astype(np.float32) / 65535.0
    else:
        refl_f = reflectance_gray.astype(np.float32)
        # 若最大值大于1，则假设其范围为 0-255 或 0-65535，按相应比例缩放
        if refl_f.max() > 1.0:
            if refl_f.max() <= 255.0:
                refl_f = refl_f / 255.0
            else:
                refl_f = refl_f / 65535.0

    # 裁剪并保存为 uint16 PNG（无损）
    refl_f = np.clip(refl_f, 0.0, 1.0)
    reflectance_uint16 = (refl_f * 65535.0).astype(np.uint16)
    cv2.imwrite(str(output_path), reflectance_uint16)
    logging.debug(f"保存反射率图像: {output_path}")


def process_single_image(
    model: DeepIntrinsicWrapper,
    image_path: Path,
    output_dir: Path,
    save_shading: bool = False,
    save_reconstruction: bool = False,
    use_i_hat_as_r: bool = True
) -> Dict[str, Any]:
    """
    处理单张图像

    Args:
        model: DeepIntrinsicWrapper 模型
        image_path: 输入图像路径
        output_dir: 输出目录
        save_shading: 是否保存着色图
        save_reconstruction: 是否保存重建图

    Returns:
        处理结果信息
    """
    # 加载图像
    pseudo_nir = load_image(image_path)

    # 推理
    R, S, I_hat, I_recon = model.process_image(pseudo_nir)

    # 生成输出文件名
    stem = image_path.stem
    # 移除可能的后缀（如 _pseudo_nir）
    if stem.endswith('_pseudo_nir'):
        stem = stem[:-11]
    elif stem.endswith('_pseudo-nir'):
        stem = stem[:-11]
    elif stem.endswith('_nir'):
        stem = stem[:-4]

    # 保存主要输出：根据配置决定保存 I_hat 还是 R 到 *_r.png
    # (文件名保持匹配 Stage3 期待的命名: frame_XXXXXX_camY_r.png，以便后续流程兼容)
    output_path = output_dir / f"{stem}_r.png"
    if use_i_hat_as_r:
        # 默认保持原有行为：将模型预测的 I_hat 写入 r.png
        save_reflectance(I_hat, output_path)
    else:
        # 可选：将模型预测的 Reflectance 写入 r.png（语义清晰）
        save_reflectance(R, output_path)

    result = {
        'input': str(image_path),
        'output': str(output_path),
        'output_shape': I_hat.shape
    }

    # 可选：保存真实的反射率 R (如果需要调试或对比)
    # 这里我们复用 save_shading 参数位置或者新增参数，但目前先只替换主输出
    # 如果用户需要 R，可以之后再添加 --save-reflectance 选项

    # 可选：保存着色图
    if save_shading:
        shading_path = output_dir / f"{stem}_shading.png"
        S_gray = S if len(S.shape) == 2 else S.mean(axis=2)
        S_uint16 = (np.clip(S_gray, 0, 1) * 65535).astype(np.uint16)
        cv2.imwrite(str(shading_path), S_uint16)
        result['shading'] = str(shading_path)

    # 可选：保存重建图
    if save_reconstruction:
        recon_path = output_dir / f"{stem}_recon.png"
        I_recon_gray = I_recon if len(I_recon.shape) == 2 else I_recon.mean(axis=2)
        I_uint16 = (np.clip(I_recon_gray, 0, 1) * 65535).astype(np.uint16)
        cv2.imwrite(str(recon_path), I_uint16)
        result['reconstruction'] = str(recon_path)

    return result


def run_stage2_deep(
    config: Dict[str, Any],
    work_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    model_config_path: Optional[str] = None,
    device: Optional[str] = None,
    save_shading: bool = False,
    save_reconstruction: bool = False,
    skip_existing: bool = False
) -> Dict[str, Any]:
    """
    运行 Stage 2-Deep 处理流程

    Args:
        config: 配置字典
        work_dir: 工作目录（覆盖配置中的值）
        checkpoint_path: 模型检查点路径（覆盖配置中的值）
        model_config_path: 模型配置文件路径（覆盖配置中的值）
        device: 设备（覆盖配置中的值）
        save_shading: 是否保存着色图
        save_reconstruction: 是否保存重建图
        skip_existing: 是否跳过已处理的图像

    Returns:
        处理结果摘要
    """
    # 获取 stage2_deep 配置
    stage2_deep_config = config.get('stage2_deep', {})

    # 确定参数（命令行参数优先）
    checkpoint = checkpoint_path or stage2_deep_config.get('checkpoint_path')
    model_config = model_config_path or stage2_deep_config.get('model_config_path', 'configs/deep_intrinsic_v3_improved.yaml')
    dev = device or stage2_deep_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    image_shape = stage2_deep_config.get('image_shape', [256, 256])
    model_kwargs = stage2_deep_config.get('model_kwargs', {})
    # 是否将 I_hat 写入 *_r.png （默认 True，保持向后兼容）
    use_i_hat_as_r = stage2_deep_config.get('use_i_hat_as_r', True)

    if not checkpoint:
        raise ValueError("必须指定 checkpoint_path（通过命令行或配置文件）")

    # 检查 checkpoint 是否存在
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint 文件不存在: {checkpoint}")

    # 检查 model_config 是否存在
    if not os.path.exists(model_config):
        # 尝试相对于项目根目录
        model_config_abs = PROJECT_ROOT / model_config
        if model_config_abs.exists():
            model_config = str(model_config_abs)
        else:
            raise FileNotFoundError(f"模型配置文件不存在: {model_config}")

    # 获取输入输出目录
    input_dir = get_input_dir(config, work_dir)
    output_dir = get_output_dir(config, work_dir)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("Stage 2-Deep: 深度内在分解网络推理")
    logging.info("=" * 60)
    logging.info(f"输入目录: {input_dir}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"检查点: {checkpoint}")
    logging.info(f"模型配置: {model_config}")
    logging.info(f"设备: {dev}")
    logging.info(f"图像尺寸: {image_shape}")

    # 查找输入图像
    images = find_pseudo_nir_images(input_dir)

    if not images:
        logging.warning(f"未找到输入图像: {input_dir}")
        return {
            'success': False,
            'error': f"未找到输入图像: {input_dir}",
            'processed': 0
        }

    logging.info(f"找到 {len(images)} 张 Pseudo-NIR 图像")

    # 加载模型配置
    final_model_kwargs = model_kwargs.copy()
    if model_config and os.path.exists(model_config):
        logging.info(f"读取模型配置: {model_config}")
        with open(model_config, 'r') as f:
            cfg = yaml.safe_load(f)

        # 提取模型参数
        model_cfg = cfg.get('model', {}) if isinstance(cfg, dict) else {}

        # 允许的参数列表 (参考 scripts/validate_model_simple.py)
        allowed_keys = [
            'in_channels', 'base_channels', 'use_shading_for_intensity',
            'intensity_hidden_channels', 'intensity_depth', 'intensity_kernel_size',
            'intensity_use_bn', 'intrinsic_net_type', 'intensity_head_type',
            'deeplab_base_channels', 'deeplab_output_stride'
        ]

        for k, v in model_cfg.items():
            if k in allowed_keys and k not in final_model_kwargs:
                final_model_kwargs[k] = v

    # 加载模型
    logging.info("加载深度内在分解模型...")
    model = DeepIntrinsicWrapper(
        checkpoint_path=checkpoint,
        image_shape=tuple(image_shape),
        device=dev,
        model_kwargs=final_model_kwargs
    )
    logging.info("模型加载完成")

    # 处理所有图像
    results = []
    processed = 0
    skipped = 0

    for i, image_path in enumerate(images):
        # 生成输出文件名
        stem = image_path.stem
        if stem.endswith('_pseudo_nir'):
            stem = stem[:-11]
        elif stem.endswith('_pseudo-nir'):
            stem = stem[:-11]

        output_path = output_dir / f"{stem}_r.png"

        # 检查是否跳过
        if skip_existing and output_path.exists():
            logging.debug(f"跳过已存在: {output_path.name}")
            skipped += 1
            continue

        try:
            result = process_single_image(
                model=model,
                image_path=image_path,
                output_dir=output_dir,
                save_shading=save_shading,
                save_reconstruction=save_reconstruction,
                use_i_hat_as_r=use_i_hat_as_r
            )
            results.append(result)
            processed += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(images):
                logging.info(f"进度: {i + 1}/{len(images)} ({100 * (i + 1) / len(images):.1f}%)")

        except Exception as e:
            logging.error(f"处理失败 {image_path.name}: {e}")
            results.append({
                'input': str(image_path),
                'error': str(e)
            })

    # 汇总结果
    summary = {
        'success': True,
        'total': len(images),
        'processed': processed,
        'skipped': skipped,
        'errors': len([r for r in results if 'error' in r]),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'checkpoint': checkpoint,
        'device': dev
    }

    logging.info("=" * 60)
    logging.info("Stage 2-Deep 完成")
    logging.info(f"总计: {summary['total']}, 处理: {summary['processed']}, 跳过: {summary['skipped']}, 错误: {summary['errors']}")
    logging.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 2-Deep: 深度内在分解网络推理")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--work-dir', type=str, default=None, help='工作目录（覆盖配置）')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径（覆盖配置）')
    parser.add_argument('--model-config', type=str, default=None, help='模型配置文件路径（覆盖配置）')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='设备')
    parser.add_argument('--save-shading', action='store_true', help='保存着色图')
    parser.add_argument('--save-reconstruction', action='store_true', help='保存重建图')
    parser.add_argument('--skip-existing', action='store_true', help='跳过已处理的图像')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='日志级别')

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)

    # 加载配置
    config = load_config(args.config)

    # 运行处理
    run_stage2_deep(
        config=config,
        work_dir=args.work_dir,
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        device=args.device,
        save_shading=args.save_shading,
        save_reconstruction=args.save_reconstruction,
        skip_existing=args.skip_existing
    )


if __name__ == '__main__':
    main()
