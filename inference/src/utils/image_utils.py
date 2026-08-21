"""
图像I/O工具函数
"""
import cv2
import numpy as np
from pathlib import Path


def load_image(image_path, color_mode='BGR'):
    """
    加载图像

    Args:
        image_path: 图像路径
        color_mode: 'BGR' 或 'RGB' 或 'GRAY'

    Returns:
        image: numpy数组
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像不存在: {image_path}")

    if color_mode == 'GRAY':
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if color_mode == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        raise IOError(f"无法加载图像: {image_path}")

    return image


def save_image(image, output_path, color_mode='BGR'):
    """
    保存图像

    Args:
        image: numpy数组
        output_path: 输出路径
        color_mode: 'BGR' 或 'RGB' 或 'GRAY'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if color_mode == 'RGB' and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise IOError(f"无法保存图像: {output_path}")


def resize_image(image, scale=1.0, target_size=None):
    """
    调整图像大小

    Args:
        image: numpy数组
        scale: 缩放比例 (如果target_size为None)
        target_size: 目标尺寸 (H, W)

    Returns:
        resized: 调整后的图像
    """
    if target_size is not None:
        h, w = target_size
    else:
        h = int(image.shape[0] * scale)
        w = int(image.shape[1] * scale)

    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized


def adjust_exposure(image, exp_ratio):
    """
    调整图像曝光

    Args:
        image: numpy数组, 值范围 [0, 255] uint8
        exp_ratio: 曝光倍数

    Returns:
        adjusted: 调整后的图像
    """
    if image.dtype == np.uint8:
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image.astype(np.float32)

    adjusted = image_float * exp_ratio
    adjusted = np.clip(adjusted * 255, 0, 255).astype(np.uint8)
    return adjusted


def convert_bit_depth(image, source_depth=8, target_depth=8):
    """
    转换图像位深度

    Args:
        image: numpy数组
        source_depth: 源位深度 (8 或 16)
        target_depth: 目标位深度 (8 或 16)

    Returns:
        converted: 转换后的图像
    """
    if source_depth == target_depth:
        return image

    if source_depth == 8 and target_depth == 16:
        # 8-bit -> 16-bit: 0-255 -> 0-65535
        converted = (image.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
    elif source_depth == 16 and target_depth == 8:
        # 16-bit -> 8-bit: 0-65535 -> 0-255
        converted = (image.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    else:
        raise ValueError(f"不支持的位深度转换: {source_depth} -> {target_depth}")

    return converted
